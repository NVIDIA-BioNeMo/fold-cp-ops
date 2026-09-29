# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""Tests for ``fold_cp_ops.kernels.gemm_sm90`` -- the base SM90 GEMM functor.

The subject here is ``GemmSm90`` **itself**, with no epilogue mixin: the WGMMA mainloop, the tile
and cluster geometry it will accept, and the plain ``D = A @ B^T`` store its default epilogue hooks
produce. ``tests/kernels/test_gemm.py`` covers the host entry and the default epilogue on top of it.

**That the base is runnable at all is a property worth testing, not an accident.** Two hooks
(``epi_setup_postact`` / ``epi_convert_postact``) and one attribute (``rounding_mode``) were missing
from the base upstream -- every mixin happened to supply them, so a bare ``GemmSm90`` died on
``AttributeError`` several frames inside ``epilogue()``. ``test_gemm_sm90_base_class_is_concrete``
below is what stops that from silently coming back, and it is the reason this module can have a
correctness grid and a perf gate at all.
"""

import functools
import math
import time

import pytest
import torch

import cutlass
import cutlass.cute as cute
from cutlass import Float32

from fold_cp_ops._internal.arch import get_max_active_clusters
from fold_cp_ops._internal.compile_time.cute_dsl_utils import torch2cute_dtype_map
from fold_cp_ops._internal.gemm_tvm_ffi_utils import (
    compile_gemm_kernel,
    make_fake_gemm_tensors,
    make_fake_scheduler_args,
    make_scheduler_args,
    perm3d,
)
from fold_cp_ops._internal.rounding import RoundingMode
from fold_cp_ops.kernels.gemm_sm90 import check_fp8_major, GemmSm90
from fold_cp_ops.testing.numerics import (
    assert_bitwise,
    assert_gemm_close,
    assert_gemm_exact,
    integer_operands,
    assert_written,
    max_exact_operand,
)
from fold_cp_ops.testing.kernel_matrix import (
    Axis,
    KernelMatrix,
    Unsupported,
    dtype_facets,
    front_door_raises,
    int_facets,
    matrix_exempt,
)

_SM = torch.cuda.get_device_capability(0)[0] if torch.cuda.is_available() else 0
requires_sm90 = pytest.mark.skipif(_SM != 9, reason=f"GemmSm90 needs sm_90; this GPU is sm_{_SM}0")

_FP8 = (torch.float8_e4m3fn, torch.float8_e5m2)


def _atom_layout_n(tile_M: int, tile_N: int, pingpong: bool) -> int:
    """How many warpgroups ``GemmSm90`` splits the N dimension across, for a given tile.

    A mirror of the branch in ``GemmSm90.__init__``, kept here so the matrix's unsupported region
    and the kernel agree by construction rather than by two people reading the same code. Only
    ``tile_M`` 192 (with a wide ``tile_N``) and 320 split N; everything else puts all of N on one
    warpgroup, and pingpong always does.

    Args:
        tile_M: CTA tile M. Assumed to be one of the accepted values -- an illegal ``tile_M`` is
            rejected before this matters, so no validation happens here.
        tile_N: CTA tile N. Only consulted at ``tile_M == 192``, where the split depends on it.
        pingpong: Whether the ping-pong schedule is in use; it fixes the atom layout at (1, 1, 1).

    Returns:
        1 or 2.
    """
    if pingpong:
        return 1
    if tile_M == 320:
        return 2
    if tile_M == 192:
        return 2 if tile_N > 128 else 1
    return 1


def _tile_n_is_buildable(tile_M: int, tile_N: int, pingpong: bool) -> bool:
    """Whether ``GemmSm90`` accepts this ``tile_N`` -- BOTH the written and the derived check.

    The written check bounds the epilogue subtile; the derived one is what the WGMMA atom can
    actually be built with (``8 <= tile_N // atom_layout_n <= 256``, in steps of 8). Both are
    reproduced because the kernel applies both, and the second is the one the first used to miss.

    Args:
        tile_M: CTA tile M. Assumed accepted -- callers filter illegal ``tile_M`` first, since the
            kernel raises on that before it ever looks at ``tile_N``.
        tile_N: CTA tile N to test.
        pingpong: Whether the ping-pong schedule is in use, which tightens the written bound.

    Returns:
        True if ``GemmSm90.__init__`` will accept the pair.
    """
    if pingpong:
        cap = 256 if tile_M == 64 else (208 if tile_M == 128 else 128)
        if not (tile_N % 16 == 0 and tile_N <= cap):
            return False
    elif tile_M in (192, 320):
        cap = 256 if tile_M == 192 else 160
        if not (tile_N % 32 == 0 and tile_N <= cap):
            return False
    else:
        if not ((tile_N % 16 == 0 and tile_N <= 256) or (tile_N % 32 == 0 and tile_N <= 512)):
            return False
    atom_n = tile_N // _atom_layout_n(tile_M, tile_N, pingpong)
    return 8 <= atom_n <= 256 and atom_n % 8 == 0


# ── the declared test matrix for this kernel ───────────────────────────────────────────────────
# **Add a config HERE, not to one test.** Every test below draws on this; narrowing it requires a
# written `because=`. tests/perf/test_benchmark_perf_gemm_sm90.py imports this very object, so the
# configs that are timed and the ones that are correctness-tested cannot drift apart.
GEMM_SM90 = KernelMatrix(
    kernel="gemm_sm90",
    arg_faults_waived=(
        "this module drives the bare functor: run_base_gemm passes epilogue_args=() and there is "
        "no front door, so the only tensors are A, B and D and there is no optional argument to "
        "malform. The tensor-argument contract belongs to gemm(), and test_gemm.py sweeps it."
    ),
    axes=(
        Axis(
            name="tile_M",
            domain=(
                "one of 64/128/192/256/320 -- the CTA tile M values the WGMMA atom layout can be "
                "built for. Under pingpong only 64/128/192, because the schedule needs exactly two "
                "MMA warpgroups. Anything else must RAISE"
            ),
            values=(32, 64, 128, 192, 256, 320),
            facets={
                "pingpong_legal": lambda v: v in (64, 128, 192),
                "wide": lambda v: v >= 256,
                "power_of_two": lambda v: v > 0 and v & (v - 1) == 0,
                "illegal": lambda v: v not in (64, 128, 192, 256, 320),
            },
        ),
        Axis(
            name="tile_N",
            domain=(
                "divisible by 16 and <= 256, OR divisible by 32 and <= 512; capped at 256 when "
                "tile_M == 192, at 160 when tile_M == 320, and at 256/208/128 under pingpong for "
                "tile_M 64/128/192. Anything else must RAISE"
            ),
            values=(16, 64, 128, 160, 192, 208, 256, 300, 384, 512),
            facets={
                "power_of_two": lambda v: v > 0 and v & (v - 1) == 0,
                "wide": lambda v: v > 256,
                "div16_only": lambda v: v % 16 == 0 and v % 32 != 0,
                "illegal": lambda v: v % 16 != 0,
            },
        ),
        Axis(
            name="pingpong",
            domain="bool; the two-warpgroup alternating schedule. Requires persistent",
            values=(False, True),
            facets={"pingpong": lambda v: v, "cooperative": lambda v: not v},
        ),
        Axis(
            name="persistent",
            domain="bool; one resident wave looping over work tiles vs one CTA per tile",
            values=(False, True),
            facets={"persistent": lambda v: v, "one_shot": lambda v: not v},
        ),
        Axis(
            name="cluster_mn",
            domain=(
                "(cluster_M, cluster_N), each a power of two with product <= 8 (the hardware "
                "cluster limit). cluster_N multicasts A, cluster_M multicasts B"
            ),
            values=((1, 1), (2, 1), (1, 2), (2, 2), (4, 1), (1, 4)),
            facets={
                "single_cta": lambda v: v == (1, 1),
                "multicast_a": lambda v: v[1] > 1,
                "multicast_b": lambda v: v[0] > 1,
                "size_four_plus": lambda v: v[0] * v[1] >= 4,
            },
        ),
        Axis(
            name="ab_dtype",
            domain=(
                "the A/B element type: bfloat16, float16, or 8-bit float (e4m3fn / e5m2). At 16 "
                "bits A and B must have the SAME type; at 8 bits only the same WIDTH, and both "
                "operands must be k-major"
            ),
            values=(torch.bfloat16, torch.float16, torch.float8_e4m3fn, torch.float8_e5m2),
            facets=dtype_facets(
                (torch.bfloat16, torch.float16, torch.float8_e4m3fn, torch.float8_e5m2)
            ),
        ),
        Axis(
            name="a_major",
            domain="'k' or 'm' -- which axis of A is contiguous. fp8 requires 'k'",
            values=("k", "m"),
            facets={"k_major": lambda v: v == "k", "m_major": lambda v: v == "m"},
        ),
        Axis(
            name="b_major",
            domain="'k' or 'n' -- which axis of B is contiguous. fp8 requires 'k'",
            values=("k", "n"),
            facets={"k_major": lambda v: v == "k", "n_major": lambda v: v == "n"},
        ),
        Axis(
            name="M",
            domain=(
                "any positive int -- UNCONSTRAINED. M is the tiled-and-predicated axis, so it need "
                "not be a tile multiple, a power of two, or even. MEASURED: 1, 7, 63, 65, 301 and "
                "1001 all compute correctly at bf16"
            ),
            values=(1, 7, 63, 65, 128, 301, 1000, 1001, 2048, 4096, 8192, 12288, 1000000),
            facets=int_facets(tile=128, big=1000, small=64),
        ),
        Axis(
            name="N",
            domain=(
                "any positive int, as an EXTENT -- MEASURED: N = 1, 3, 7, 201 and 257 all "
                "compute correctly at bf16. The 16-byte floor lands on the row PITCH, not on N: "
                "torch.empty(l, m, 208)[:, :, :201] works and torch.empty(l, m, 201) does not, and "
                "the difference between them is the pitch. A CONTIGUOUS D conflates the two, which "
                "is why the everyday statement 'N must be a multiple of 8' is true and misleading"
            ),
            values=(
                1,
                3,
                7,
                8,
                64,
                128,
                192,
                200,
                201,
                256,
                257,
                384,
                512,
                768,
                1000,
                1024,
                2048,
                4096,
                # Production N_token for the A2A-fused TriMul back einsum. The profiled
                # grid runs it to 12288, where the kernel measures 247 TFLOP/s against
                # 514 at 2048 -- so these are not 'bigger of the same', they are where
                # the efficiency curve actually is.
                8192,
                12288,
            ),
            facets={
                "power_of_two": lambda v: v > 0 and v & (v - 1) == 0,
                "tile_aligned": lambda v: v % 128 == 0,
                "fp8_aligned": lambda v: v % 16 == 0,
                "large": lambda v: v >= 1000,
                "trimul_postact": lambda v: v in (256, 512, 768, 1024),
            },
        ),
        Axis(
            name="K",
            domain=(
                "any positive int, as an EXTENT -- MEASURED: K = 1, 3, 7 and 129 all compute "
                "correctly at bf16 with A/B's pitch padded to the floor. Same pitch-vs-extent "
                "split as N. The mainloop tiles K by 64, so a K below one tile is a single "
                "predicated iteration"
            ),
            values=(1, 3, 7, 8, 16, 64, 128, 129, 192, 256, 384, 1000, 2048, 4096, 8192, 12288),
            facets={
                "power_of_two": lambda v: v > 0 and v & (v - 1) == 0,
                "sub_tile": lambda v: v < 64,
                "tile_aligned": lambda v: v % 64 == 0,
                "large": lambda v: v >= 1000,
                "trimul_feature_dim": lambda v: v in (128, 256, 384),
            },
        ),
        Axis(
            name="L",
            domain="any positive int; the batch extent is a symbolic dim and never recompiles",
            values=(1, 2, 3, 7, 8, 16, 32, 48),
            facets={
                "single_batch": lambda v: v == 1,
                "multi_batch": lambda v: v > 1,
                "odd_batch": lambda v: v % 2 == 1,
                # L = Dloc * B for the A2A-fused TriMul back einsum, where Dloc = D / cp and
                # B is the sequence batch. The profiled production grid
                # (profiling/distributed/trimul/dlc_composite_k_dtensor_cp_le16) runs D in
                # {128, 256, 384} at cp up to 16 with B in {1, 2}, so Dloc reaches 8 and the
                # reachable L values are 8, 16, 32 and 48. 8 was MISSING from this pool -- it is
                # D=128 at cp=16, which the grid covers -- and the gap was invisible because the
                # pool jumped 7 -> 16 and 7 looks like a batch test.
                "trimul_per_device": lambda v: v in (8, 16, 32, 48),
            },
        ),
    ),
    # Ordered to match the order `GemmSm90.__init__` performs the checks: the persistent check
    # comes first, then tile_M, then tile_N.  `parametrize_unsupported` stops at the first matching
    # region, so a combo in two regions must list the one the kernel actually raises from first.
    # The bare contraction, at both fp8 encodings.
    computes=("contraction", "fp8_operands"),
    unsupported=(
        Unsupported(
            where=lambda pingpong, persistent: pingpong and not persistent,
            raises=ValueError,
            match=r"pingpong requires is_persistent=True",
            reason=(
                "the ping-pong schedule alternates mainloop and epilogue between two warpgroups "
                "across SUCCESSIVE work tiles, which only exists in a persistent grid. Upstream "
                "asserted this; an assert is stripped by `python -O`, and a stripped check here "
                "builds a kernel whose two warpgroups wait on tiles the grid will never hand them"
            ),
        ),
        Unsupported(
            where=lambda pingpong, tile_M: not pingpong and tile_M not in (64, 128, 192, 256, 320),
            raises=ValueError,
            match=r"CTA tile shape M must be 64/128/192/256/320",
            reason=(
                "the MMA atom layout is derived from tile_M (tile_M // 64 warpgroups along M, with "
                "192 and 320 special-cased into an N-split); no other value has a layout"
            ),
        ),
        Unsupported(
            where=lambda pingpong, tile_M: pingpong and tile_M not in (64, 128, 192),
            raises=ValueError,
            match=r"CTA tile shape M must be 64/128/192 if pingpong",
            reason=(
                "ping-pong fixes the atom layout at (1, 1, 1) and uses exactly two MMA warpgroups, "
                "so the wide tiles that would need three or an N-split have nowhere to put them"
            ),
        ),
        Unsupported(
            where=lambda pingpong, tile_M, tile_N: (
                tile_M in (64, 128, 192, 256, 320)
                and not (pingpong and tile_M not in (64, 128, 192))
                and not _tile_n_is_buildable(tile_M, tile_N, pingpong)
            ),
            raises=ValueError,
            match=r"CTA tile shape N must be divisible by",
            reason=(
                "tile_N has to survive TWO checks. The written one bounds the epilogue subtile "
                "width and the accumulator's register footprint, and is tighter at tile_M 192/320 "
                "(which spend a warpgroup on an N-split) and under pingpong (which halves the "
                "budget). The DERIVED one is what actually binds: the SM90 WGMMA atom is built "
                "with N = tile_N // atom_layout_n and accepts only 8 <= N <= 256 in steps of 8. "
                "The written check alone admitted tile_M=128 with tile_N=512, which then died "
                "inside cutlass with `OpError: expects the N-mode to satisfy 8 <= N <= 256` -- an "
                "internal explosion where a refusal belongs. This region covers both"
            ),
        ),
    ),
)

# One compiled artifact per configuration, reused across tests in this module. Compiling is ~1 s
# and the grid revisits configurations, so without this the module would spend minutes recompiling
# identical kernels. Keyed on exactly what `compile_gemm_kernel` varies on.
_COMPILED: dict = {}


def _pad_to_floor(extent, dtype):
    """Round an extent up to the row pitch the TMA descriptor requires.

    Args:
        extent: The logical extent.
        dtype: The element type, whose width sets the floor at ``128 // width`` elements.

    Returns:
        The smallest multiple of the floor that is >= ``extent``. Equal to ``extent`` when it
        already conforms, so callers can apply it unconditionally.
    """
    div = 128 // torch2cute_dtype_map[dtype].width
    return (extent + div - 1) // div * div


def _build_operands(M, N, K, L, dtype, a_major, b_major, d_major="n", device="cuda", pitch="tight"):
    """Allocate A, B and D with the requested element type and major modes.

    Args:
        M: Rows of A / D. Unconstrained.
        N: Columns of B / D. Must be a multiple of ``128 // width`` for the TMA alignment floor.
        K: Contraction extent. Same alignment floor.
        L: Batch extent.
        dtype: A torch dtype in ``torch2cute_dtype_map``. fp8 operands are built by casting from
            ``randn() / 4``, which keeps the values inside e4m3's range so the comparison measures
            the kernel rather than saturation.
        a_major: ``"k"`` for a plain ``(L, M, K)`` tensor, ``"m"`` for the transpose of an
            ``(L, K, M)`` one -- i.e. a real non-contiguous view, not a relabelled contiguous one.
        b_major: ``"k"`` or ``"n"``, same construction.
        d_major: ``"n"`` or ``"m"`` for the output.
        device: CUDA device to allocate on.

    Returns:
        ``(A, B, D)``, with D uninitialized -- the kernel overwrites every element it is
        responsible for, so reading D before the launch is meaningless.
    """
    fp8 = dtype in _FP8
    out_dtype = torch.bfloat16 if fp8 else dtype

    def gen(*shape):
        """Allocate one operand, padding its trailing extent up to the pitch floor when asked.

        Args:
            *shape: The logical shape. Only the LAST entry is padded, because that is the axis the
                row pitch is measured along before ``perm3d`` moves the batch.

        Returns:
            A view with exactly ``shape``, whose pitch is ``shape[-1]`` under ``pitch="tight"`` and
            the next multiple of the floor under ``pitch="padded"``.
        """
        want = shape[-1]
        alloc = shape[:-1] + ((_pad_to_floor(want, dtype),) if pitch == "padded" else (want,))
        if fp8:
            t = (torch.randn(*alloc, device=device) / 4).to(dtype)
        else:
            t = torch.randn(*alloc, device=device, dtype=dtype)
        return t[..., :want]

    A = gen(L, M, K) if a_major == "k" else gen(L, K, M).transpose(-1, -2)
    B = gen(L, N, K) if b_major == "k" else gen(L, K, N).transpose(-1, -2)
    if d_major == "n":
        alloc = _pad_to_floor(N, out_dtype) if pitch == "padded" else N
        D = torch.empty(L, M, alloc, device=device, dtype=out_dtype)[:, :, :N]
    else:
        alloc = _pad_to_floor(M, out_dtype) if pitch == "padded" else M
        D = torch.empty(L, N, alloc, device=device, dtype=out_dtype).transpose(-1, -2)[:, :M, :]
    return A, B, D


def run_base_gemm(A, B, D, tile_M, tile_N, cluster_mn, pingpong, persistent):
    """Compile (once per config) and launch a bare ``GemmSm90`` over already-built operands.

    The whole point of this helper is that it uses **no epilogue mixin**: ``epilogue_args`` is the
    empty tuple, so what runs is the base class's own default hooks and the store is a plain
    ``D = A @ B^T``. That is what makes this module's subject ``gemm_sm90.py`` and not the default
    epilogue on top of it.

    Args:
        A: A operand from :func:`_build_operands`. Its device decides where the kernel runs.
        B: B operand, same element type as A at 16 bits and the same width at 8.
        D: Preallocated output, **written in place**.
        tile_M: CTA tile M. Must be a value ``GemmSm90.__init__`` accepts, or it raises.
        tile_N: CTA tile N, likewise.
        cluster_mn: ``(cluster_M, cluster_N)``.
        pingpong: Whether to use the ping-pong schedule. Requires ``persistent``.
        persistent: Whether the grid is persistent.

    Returns:
        None. The result is in ``D``.

    Raises:
        ValueError: Propagated from ``GemmSm90`` for a geometry or operand layout it refuses --
            which is exactly what ``test_gemm_sm90_unsupported_combos_raise`` relies on.
    """
    a_dt, b_dt, d_dt = (torch2cute_dtype_map[t.dtype] for t in (A, B, D))
    A_p, B_p, D_p, _ = perm3d(A, B, D, None)
    majors = tuple("k" if t.stride(1) == 1 else m for t, m in ((A_p, "m"), (B_p, "n")))
    d_major = "n" if D_p.stride(1) == 1 else "m"
    cluster_mnk = (cluster_mn[0], cluster_mn[1], 1)
    key = (a_dt, b_dt, d_dt, majors, d_major, (tile_M, tile_N), cluster_mnk, pingpong, persistent)
    if key not in _COMPILED:
        mA, mB, mD, mC, _, _, _, l = make_fake_gemm_tensors(
            a_dt, b_dt, d_dt, None, majors[0], majors[1], d_major, None
        )
        _COMPILED[key] = compile_gemm_kernel(
            GemmSm90,
            a_dt,
            (tile_M, tile_N),
            cluster_mnk,
            pingpong,
            persistent,
            False,
            (9, 0),
            mA,
            mB,
            mD,
            mC,
            (),
            make_fake_scheduler_args(False, False, l),
        )
    clusters = get_max_active_clusters(cluster_mn[0] * cluster_mn[1]) if persistent else 0
    _COMPILED[key](A_p, B_p, D_p, None, (), make_scheduler_args(clusters, 8, None, None))


def assert_close(D, A, B):
    """Compare the kernel's D against an exact fp64 reference, **element by element**.

    Delegates to :func:`assert_gemm_close`, whose bound is a per-element tensor
    ``gamma_K * sum_k |A[i,k]*B[j,k]| + half_ulp``, not one number for the whole output. The scalar
    gate this replaced divided the max absolute error by the max absolute reference, so an output
    element that cancelled to near zero was judged against the largest element's slack and could be
    wrong by 100% of itself while passing.

    Args:
        D: The kernel's output.
        A: The A operand it was computed from, ``(..., M, K)``.
        B: The B operand, ``(..., N, K)``. The contraction is over the last axis of both.

    Returns:
        None.

    Raises:
        AssertionError: If D holds a non-finite value, if the reference is degenerately all-zero,
            or if any element exceeds its own bound -- reported with the violating count and the
            three worst coordinates, values and ratios.
    """
    # fp8 operands take a wider bound: the analytic model covers the fp32 accumulation and the
    # store, but NOT the e4m3/e5m2 quantisation of the operands themselves, whose mantissa step is
    # ~6%. The factor is stated here rather than hidden in a tolerance constant.
    scale = 8.0 if A.dtype in _FP8 else 1.0
    assert_gemm_close(D, A, B, scale=scale)


# ── geometry: the tile/cluster/schedule configurations the kernel accepts ──────────────────────
_GEOMETRY_CELLS = [
    # (tile_M, tile_N, cluster_mn, pingpong, persistent)
    (128, 128, (1, 1), False, True),  # the default
    (128, 256, (1, 1), False, True),  # widest N at the plain rung
    (64, 128, (1, 1), False, True),  # one MMA warpgroup
    (256, 128, (1, 1), False, True),  # two warpgroups via the >=256 branch
    (192, 128, (1, 1), False, True),  # the 3-warpgroup M-split special case
    (192, 256, (1, 1), False, True),  # 192 with the N-split instead (tile_N > 128)
    (320, 160, (1, 1), False, True),  # the other special case, at its tile_N cap
    (256, 256, (1, 1), False, True),  # the widest single-warpgroup N the atom can build
    (128, 208, (1, 1), False, True),  # div-16-only, not a power of two
    (128, 128, (2, 1), False, True),  # multicast B
    (128, 128, (1, 2), False, True),  # multicast A
    (128, 128, (2, 2), False, True),  # both, cluster size 4
    (128, 128, (4, 1), False, True),  # cluster size 4 along M
    (128, 128, (1, 4), False, True),  # cluster size 4 along N
    (128, 128, (1, 1), False, False),  # non-persistent grid
    (64, 256, (1, 1), True, True),  # pingpong at its widest tile_N
    (128, 208, (1, 1), True, True),  # pingpong at its tile_M=128 cap
    (192, 128, (1, 1), True, True),  # pingpong at tile_M=192
]


@requires_sm90
@GEMM_SM90.parametrize(
    "tile_M",
    "tile_N",
    "cluster_mn",
    "pingpong",
    "persistent",
    cells=_GEOMETRY_CELLS,
    because=(
        "geometry is a LIST of valid configurations, not a product: the cross product of the four "
        "axes is 480 cells of which most are declared unsupported, and the supported remainder "
        "recompiles for every one. These 18 cells cover each distinct code path exactly once -- "
        "every atom-layout branch (1/2/3 warpgroups, the 192 and 320 special cases), both tile_N "
        "branches (div-16 <= 256 and div-32 <= 512), every cluster size the hardware allows, both "
        "grid modes, and all three pingpong tile_M rungs. Shape/dtype are held at one aligned bf16 "
        "cell so a failure names the geometry and nothing else."
    ),
)
def test_gemm_sm90_geometry(tile_M, tile_N, cluster_mn, pingpong, persistent):
    """Every accepted tile/cluster/schedule configuration computes the same product.

    A wrong tile geometry does not usually crash -- it mis-partitions, and the result is wrong in
    the tail rows or the last k-tile. So the shape here is deliberately NOT a multiple of any tile
    dimension: M=301 leaves a partial M tile under every tile_M, and K=192 is three 64-wide k-tiles
    exactly, so the mainloop's steady state and its drain are both exercised.
    """
    torch.manual_seed(0)
    A, B, D = _build_operands(301, 256, 192, 2, torch.bfloat16, "k", "k")
    run_base_gemm(A, B, D, tile_M, tile_N, cluster_mn, pingpong, persistent)
    assert_close(D, A, B)


@requires_sm90
@GEMM_SM90.parametrize(
    "M",
    "N",
    "K",
    "L",
    cells=[
        (1, 8, 8, 1),  # the smallest legal problem: one row, one 16-byte vector
        (7, 64, 16, 1),  # M far below tile_M, K below one k-tile
        (63, 128, 64, 2),  # M just under a tile, K exactly one k-tile
        (65, 192, 128, 3),  # M just over a tile; odd batch
        (301, 200, 192, 2),  # off-grid M and N (200 is 16-byte aligned, not tile-aligned)
        (1001, 384, 256, 1),  # odd M at a production-sized N
        (128, 1000, 1000, 1),  # off-grid N and K, both large
        (4096, 2048, 2048, 1),  # a shape big enough to make the persistent grid loop
        (128, 256, 8, 7),  # minimum K with the largest batch
        # --- the A2A-fused TriMul's own shapes, which is what this kernel is the basis for ---
        # BACK half: the square einsum out[b,i,j,d] = sum_k a[b,i,k,d] b[b,j,k,d], so M == N == K
        # == N_token and L = B * (D/cp). A decoupled K here would measure an O(N^2) thin-K matmul
        # instead of the O(N^3) einsum -- see the K = N_token HARD RULE in CLAUDE.md.
        # L = Dloc * B sweeps the whole reachable family: Dloc = D/cp in {8, 16} for D in
        # {128, 256} at cp in {8, 16}, times B in {1, 2}.
        (1000, 1000, 1000, 8),  # N_token=1000, Dloc=8  (D=128 at cp=16), B=1
        (1000, 1000, 1000, 16),  # N_token=1000, Dloc=16 (D=128 at cp=8),  B=1
        (2048, 2048, 2048, 16),  # N_token=2048, Dloc=8,  B=2 -- the same L from the other factor
        (2048, 2048, 2048, 32),  # N_token=2048, Dloc=16, B=2 (or D=256 at cp=8, B=1)
        # FRONT half: contracts over the feature dim, so K = D in {128, 256, 384} and M = N_token^2.
        # Exempt from the K = N_token rule -- the front IS O(N^2). M is held at 4096 rather than a
        # real N_token^2 because the fp32 reference is what would not fit, not the kernel.
        (4096, 512, 128, 1),  # D=128 -> 2*2*D postact width
        (4096, 768, 384, 1),  # D=384, the widest TriMul feature dim
    ],
    because=(
        "shapes are a list, not a product: the pools cross to over 5000 cells, each a full "
        "recompile. The first 9 span the axes independently -- M below/at/over a tile and odd, K "
        "below/at/over one 64-wide k-tile, N off-grid and large, L single/even/odd -- inside the "
        "16-byte floor that is this kernel's only shape constraint. The last 4 are the A2A-fused "
        "TriMul's own regimes, which no generic cell reaches: a SQUARE M=N=K einsum at a "
        "per-device batch, and a thin-K projection at the feature dims 128/384. The TriMul feature "
        "dims are not a formality -- 128 and 384 were exactly the values missing from the LayerNorm "
        "grid when a defect lived there."
    ),
)
def test_gemm_sm90_shapes(M, N, K, L):
    """The declared shape freedom is real: M and L unconstrained, N/K only 16-byte aligned.

    This is the FIRST PRINCIPLE test. If any of these raises or returns a wrong answer, the kernel
    has acquired a shape constraint it does not admit to -- a regression, not a limitation.

    The last six cells are the TriMul workflow's own shapes. They are here rather than only in the
    perf gate because the two regimes exercise different code: the square back einsum runs a
    ``k_tile_cnt`` in the hundreds with a batch of 16-48, and the thin-K front runs one or two
    k-tiles with a huge M -- the mainloop's steady state and its drain, at opposite extremes.
    """
    torch.manual_seed(0)
    A, B, D = _build_operands(M, N, K, L, torch.bfloat16, "k", "k")
    run_base_gemm(A, B, D, 128, 128, (1, 1), False, True)
    assert_close(D, A, B)


@requires_sm90
@GEMM_SM90.parametrize(
    "tile_M",
    "tile_N",
    "pingpong",
    "cluster_mn",
    cells=[
        (128, 128, False, (1, 1)),
        (128, 256, False, (1, 1)),
        (64, 128, True, (1, 1)),
        (192, 128, False, (1, 1)),
        (256, 128, False, (1, 1)),
        (128, 128, False, (2, 1)),
        (128, 128, False, (1, 2)),
        (128, 128, False, (2, 2)),
    ],
    because=(
        "the point of this test is that the answer is INDEPENDENT of the schedule, so the cells "
        "are the geometries that reorder the summation -- tile width, the N-split at tile_M 192, "
        "the two-warpgroup pingpong alternation, and multicast along each cluster axis. Sweeping "
        "the shape pools too would recompile thousands of configs to re-assert one property."
    ),
)
def test_gemm_sm90_is_bit_exact_for_integer_operands(tile_M, tile_N, pingpong, cluster_mn):
    """**Binary identity.** With integer operands inside fp32's exact window, D is EXACT.

    The strongest gate this kernel admits, and the reason it is available at all is stated in
    ``fold_cp_ops/testing/numerics``: if ``K * P * Q < 2**24`` then every partial sum is an integer
    fp32 represents exactly, so rounding is the identity at every step and the accumulator equals
    an exact integer reference **whatever order the terms were summed in**.

    That order-independence is what makes one ``torch.equal`` meaningful across this whole cell
    list. Tile width, the N-split, pingpong's alternation and cluster multicast all change the
    order in which the K products are accumulated; none of them may change the answer by even one
    ulp. A tolerance-based gate cannot say that -- it would accept a kernel that dropped a k-tile
    whose contribution happened to be small.

    The output is **fp32**, not fp16: the exact product reaches ``K*P*Q``, far past the 2048 an
    fp16 store represents exactly, so an fp16 D would round and the comparison would fail for a
    reason that is not the kernel's. ``assert_gemm_exact`` checks that precondition itself rather
    than trusting this comment.
    """
    K, M, N, L = 512, 256, 256, 2
    P = max_exact_operand(K, torch.float16)
    g = torch.Generator(device="cuda").manual_seed(0)
    A = integer_operands((L, M, K), torch.float16, "cuda", generator=g, bound=P)
    B = integer_operands((L, N, K), torch.float16, "cuda", generator=g, bound=P)
    D = torch.zeros(L, M, N, device="cuda", dtype=torch.float32)
    run_base_gemm(A, B, D, tile_M, tile_N, cluster_mn, pingpong, True)
    assert_gemm_exact(D, A, B, what=f"tile=({tile_M},{tile_N}) cluster={cluster_mn} pp={pingpong}")


@requires_sm90
@GEMM_SM90.parametrize(
    "M",
    "N",
    "K",
    "L",
    cells=[
        (1, 8, 8, 1),
        (7, 64, 64, 1),
        (301, 128, 128, 2),
        (128, 200, 192, 3),
        (1000, 256, 1000, 1),
    ],
    because=(
        "shapes are a list rather than a product for the same reason as test_gemm_sm90_shapes -- "
        "each cell is a recompile. These five put the partial trailing tile in a different place "
        "each time: M below a tile, M odd, M off-grid with a batch, N off-grid, and a large "
        "non-power-of-two K. A dropped or double-counted tail tile changes an integer sum by a "
        "whole integer, which bit-equality sees and a 2% tolerance does not."
    ),
)
def test_gemm_sm90_is_bit_exact_at_partial_tiles(M, N, K, L):
    """**Binary identity at the tail.** Predication is exact arithmetic, not an approximation.

    A masked lane must contribute exactly zero and an unmasked one exactly its product. Both are
    integer statements, so the whole output is bit-exact -- including the ragged edge, which is
    where predication bugs live and where a relative tolerance is loosest because the reference
    element is built from fewer terms.
    """
    P = max_exact_operand(K, torch.float16)
    g = torch.Generator(device="cuda").manual_seed(1)
    A = integer_operands((L, M, K), torch.float16, "cuda", generator=g, bound=P)
    B = integer_operands((L, N, K), torch.float16, "cuda", generator=g, bound=P)
    D = torch.zeros(L, M, N, device="cuda", dtype=torch.float32)
    run_base_gemm(A, B, D, 128, 128, (1, 1), False, True)
    assert_gemm_exact(D, A, B, what=f"M={M} N={N} K={K} L={L}")


@requires_sm90
@GEMM_SM90.parametrize(
    "ab_dtype",
    "a_major",
    "b_major",
    drop={"a_major": ["m"], "b_major": ["n"]},
    because=(
        "the dtype sweep holds both operands k-major because fp8 REQUIRES it -- the mn-major "
        "combinations are covered by test_gemm_sm90_major_modes at 16 bits, and refused outright "
        "at 8 (tests/kernels/test_gemm.py declares that region)."
    ),
)
def test_gemm_sm90_dtypes(ab_dtype, a_major, b_major):
    """All four supported element types compute, including both fp8 encodings.

    fp8 reaching this kernel at all is new here: ``torch2cute_dtype_map`` had no fp8 entries, so
    the host entry died on ``KeyError`` while ``GemmSm90.is_valid_dtypes`` claimed to support it.
    The output is bf16 for the fp8 cases -- an fp8 accumulator output would measure the store's
    saturation rather than the MMA.
    """
    torch.manual_seed(0)
    A, B, D = _build_operands(256, 256, 256, 2, ab_dtype, a_major, b_major)
    run_base_gemm(A, B, D, 128, 128, (1, 1), False, True)
    assert_close(D, A, B)


@requires_sm90
@GEMM_SM90.parametrize(
    "a_major",
    "b_major",
    "ab_dtype",
    only={"ab_dtype": [torch.bfloat16]},
    because=(
        "major modes are a 16-bit-only question: at 8 bits every non-k-major operand is refused "
        "(Hopper has no mn-major fp8 WGMMA atom), and float16 shares bfloat16's atom exactly, so "
        "sweeping it here would recompile four kernels to re-test one code path."
    ),
)
def test_gemm_sm90_major_modes(a_major, b_major, ab_dtype):
    """Each operand may be contiguous along either of its two axes, independently.

    The transposed operands are real non-contiguous views (``randn(L, K, M).transpose(-1, -2)``),
    not relabelled contiguous tensors -- the kernel reads the major mode off the stride, so a
    relabelled tensor would test nothing.
    """
    torch.manual_seed(0)
    A, B, D = _build_operands(256, 256, 256, 2, ab_dtype, a_major, b_major)
    run_base_gemm(A, B, D, 128, 128, (1, 1), False, True)
    assert_close(D, A, B)


@requires_sm90
@GEMM_SM90.parametrize(
    "M",
    "N",
    "K",
    "L",
    cells=[(128, 200, 192, 2)],
    because=(
        "the output-major test varies the OUTPUT layout, not the problem shape, so one cell is "
        "enough. M must be 8-aligned HERE specifically: an m-major D makes M the row pitch, so the "
        "16-byte floor lands on M rather than on N. M=301 -- fine for every n-major cell above -- "
        "is refused with 'Invalid mD.strides[1] ... expected to be divisible by 8'. N stays "
        "off-grid at 200 so the cell is not aligned on every axis at once."
    ),
)
def test_gemm_sm90_m_major_output(M, N, K, L):
    """The output may be m-major, which changes the epilogue's SMEM-to-GMEM store, not the result.

    Note what the layout choice moves: the 16-byte alignment floor applies to whichever extent is
    the *pitch*, so an m-major D constrains M and an n-major D constrains N. That is a real
    interaction between two axes of the matrix, not a second shape constraint -- the floor itself
    is unchanged.
    """
    torch.manual_seed(0)
    A, B, D = _build_operands(M, N, K, L, torch.bfloat16, "k", "k", d_major="m")
    run_base_gemm(A, B, D, 128, 128, (1, 1), False, True)
    assert_close(D, A, B)


@requires_sm90
@GEMM_SM90.parametrize(
    "N",
    "K",
    only={"N": [1, 3, 7, 201, 257], "K": [1, 3, 7, 129]},
    because=(
        "this test's subject is the extents that are NOT multiples of the 16-byte floor, so it "
        "sweeps exactly those. The conforming values are covered by test_gemm_sm90_shapes; running "
        "them here would recompile the same kernels to re-assert the same thing."
    ),
)
def test_gemm_sm90_odd_and_sub_floor_extents(N, K):
    """Odd and sub-floor N and K compute correctly -- the floor is on the PITCH, not the extent.

    This is the FIRST PRINCIPLE stated precisely. ``N = 201`` is odd and ``N = 3`` is six bytes, and
    both are correct when the operand's row pitch is padded to the floor; what the kernel cannot
    address is a *pitch* of 201, which is a TMA descriptor requirement rather than a shape one.
    ``tests/kernels/test_gemm.py`` declares the tight-pitch case as an unsupported region, so the
    two halves of the distinction are both asserted rather than either being assumed.

    Every cell here is a 20-cell product on purpose: N and K reach the kernel through different
    code -- N through the epilogue's store predication, K through the mainloop's trailing k-tile --
    and a single diagonal would leave one of the two untested at each value.
    """
    torch.manual_seed(0)
    A, B, D = _build_operands(65, N, K, 2, torch.bfloat16, "k", "k", pitch="padded")
    run_base_gemm(A, B, D, 128, 128, (1, 1), False, True)
    assert_close(D, A, B)


def _zeros_padded(shape, dtype, device="cuda"):
    """Allocate a zeroed tensor whose ROW PITCH meets the 16-byte floor, then slice to ``shape``.

    The kernel's only shape constraint is on the pitch, never on an extent, so an arbitrary
    trailing extent is legal **provided the allocation is padded**. A contiguous
    ``torch.zeros(1, M, 201)`` in fp32 has a 804-byte pitch and the TMA descriptor cannot be built
    for it; the same 201-wide view over a 204-wide allocation is fine.

    Args:
        shape: The logical shape. Its last entry is the one that may be off the floor.
        dtype: Element type; sets the floor at ``16 // itemsize`` elements.
        device: Where to allocate.

    Returns:
        A zeroed **view** with the requested shape and a padded pitch. Not contiguous when padding
        was applied, which is the point.
    """
    *lead, last = shape
    floor = 16 // torch.empty((), dtype=dtype).element_size()
    padded = ((last + floor - 1) // floor) * floor
    return torch.zeros(*lead, padded, device=device, dtype=dtype)[..., :last]


def _full_padded_for_gate(shape, dtype, device, value):
    """Pitch-padded pre-filled allocator, in the argument order ``assert_written`` calls with.

    Passed as that gate's ``alloc`` hook so the two pre-filled buffers meet the 16-byte pitch
    floor. Without it the gate allocates contiguously, and at an off-grid trailing extent the TMA
    descriptor cannot be built -- which is exactly the ragged shape the gate is most useful at.

    Args:
        shape: Logical output shape.
        dtype: Element type.
        device: Where to allocate.
        value: The pre-fill. Two different values across two runs is what makes an unwritten
            element visible; one value cannot tell "not written" from "written with that value".

    Returns:
        A pitch-padded view filled with ``value``.
    """
    t = _zeros_padded(shape, dtype, device)
    t.fill_(value)
    return t


@requires_sm90
@GEMM_SM90.parametrize(
    "M",
    "K",
    cells=[(7, 8), (65, 64), (128, 128), (301, 256), (1001, 192)],
    because=(
        "an identity operand forces the output's N to equal K, so the M and K pools cannot be "
        "crossed freely here. These five span a sub-tile square, M just over one tile, the tile "
        "itself, and two multi-tile off-grid M -- the partial trailing tile being where a "
        "permuted or dropped element would land. K is drawn from the pool values that are a "
        "multiple of 8, since the identity is allocated contiguously and its row pitch carries "
        "the 16-byte floor."
    ),
)
def test_gemm_sm90_load_and_store_paths_are_bit_exact_transports(M, K):
    """**Binary identity of the data path.** ``A @ I^T`` must return A itself, bit for bit.

    Stronger than the general integer-exact test, and in a different way. There, each output
    element is a sum of K terms, so a permutation of A's columns that happened to preserve the sum
    would pass. Against an identity B every output element traces to **exactly one** input
    element: ``out[i,j] = sum_k A[i,k]*d_jk = A[i,j]``. So this is a read-back of A through the
    whole transport -- TMA load, swizzled SMEM staging, the WGMMA register fragments, the epilogue
    retile and the store -- and any stage that reorders, drops or duplicates an element shows up
    as a wrong VALUE AT A NAMED COORDINATE rather than as a wrong sum.

    Exact for the same reason but more strongly than usual: every product is either 0 or the input
    value, and each sum has exactly one non-zero term, so there is no accumulation at all. That
    lets the operand magnitude go to fp16's full exact-integer range rather than the
    ``sqrt(2**24/K)`` the general test is limited to.

    The mirror direction is asserted too: ``I @ B^T`` is B transposed, which reads back B's own
    load path (a different TMA descriptor, a different SMEM tile, a different MMA operand).
    """
    g = torch.Generator(device="cuda").manual_seed(11)
    eye = torch.eye(K, device="cuda", dtype=torch.float16).unsqueeze(0)

    A = integer_operands((1, M, K), torch.float16, "cuda", generator=g, bound=2048)
    D = _zeros_padded((1, M, K), torch.float32)
    run_base_gemm(A, eye, D, 128, 128, (1, 1), False, True)
    assert_bitwise(D, A.float(), what=f"A @ I^T read-back (M={M} K={K})")

    B = integer_operands((1, M, K), torch.float16, "cuda", generator=g, bound=2048)
    D2 = _zeros_padded((1, K, M), torch.float32)
    run_base_gemm(eye, B, D2, 128, 128, (1, 1), False, True)
    assert_bitwise(
        D2, B.float().transpose(-1, -2).contiguous(), what=f"I @ B^T read-back (M={M} K={K})"
    )


@requires_sm90
@GEMM_SM90.parametrize(
    "M",
    "N",
    "K",
    cells=[(1, 8, 8), (7, 200, 64), (65, 200, 192), (301, 1000, 384)],
    because=(
        "the subject is the STORE predicate at a partial tile, so the cells are shapes whose M "
        "and N both fall off the tile grid. A shape that tiles evenly has no masked store lane "
        "and would exercise nothing."
    ),
)
def test_gemm_sm90_writes_every_output_element(M, N, K):
    """Every element of D is stored -- including the ones in a partial trailing tile.

    Run twice into buffers pre-filled with two DIFFERENT values and require the results to agree.
    An element the kernel never stores keeps its pre-fill, so the two runs disagree exactly there.

    A value comparison against a zeroed buffer cannot do this: wherever the reference is itself
    near zero -- which a ragged edge often is -- an unwritten element passes. That is the gap this
    closes, and it is the reason ``assert_written`` uses two runs rather than a magic poison value
    (no float value is one a GEMM cannot legitimately produce).
    """
    g = torch.Generator(device="cuda").manual_seed(12)
    A = integer_operands((1, M, K), torch.float16, "cuda", generator=g, K=K)
    B = integer_operands((1, N, K), torch.float16, "cuda", generator=g, K=K)

    assert_written(
        lambda D: run_base_gemm(A, B, D, 128, 128, (1, 1), False, True),
        (1, M, N),
        torch.float32,
        "cuda",
        what=f"M={M} N={N} K={K}",
        alloc=_full_padded_for_gate,
    )


# ── the individual stages of __call__ and kernel() ────────────────────────────────────────────
# `__call__` and `kernel()` are each a short sequence of named stages now, and each stage below is
# tested for the ONE property that only it can establish. These are deliberately not end-to-end:
# an assembled GEMM that computes the right answer proves the stages compose, and proves nothing
# about the two whose failure mode is a HANG rather than a wrong number.


def _compiled_instance(
    a_dtype=torch.bfloat16, d_dtype=None, tile=(128, 128), cluster=(1, 1, 1), **kwargs
):
    """Build a ``GemmSm90``, trace it once against fake tensors, and return the instance.

    Tracing is what runs the ``__call__`` stages, so afterwards the instance carries each stage's
    output as an attribute -- ``a_dtype`` from ``bind_operand_types``, ``num_tma_load_bytes`` from
    ``make_mainloop_tma``, ``threads_per_cta`` from ``resolve_launch_shape``. Inspecting those is
    how a stage is tested individually without a device.

    Args:
        a_dtype: torch dtype for A and B.
        d_dtype: torch dtype for D; defaults to ``a_dtype``.
        tile: ``(tile_M, tile_N)``.
        cluster: ``(cluster_M, cluster_N, 1)``.
        **kwargs: Forwarded to ``GemmSm90.__init__`` (``pingpong``, ``is_persistent``, ...).

    Returns:
        The traced ``GemmSm90`` instance.

    Raises:
        TypeError / ValueError: Propagated from the stages, which is what the refusal tests want.
    """
    a_dt = torch2cute_dtype_map[a_dtype]
    d_dt = torch2cute_dtype_map[d_dtype or a_dtype]
    mA, mB, mD, mC, _, _, _, l = make_fake_gemm_tensors(a_dt, a_dt, d_dt, None, "k", "k", "n", None)
    obj = GemmSm90(Float32, a_dt, tile, cluster, **kwargs)
    cute.compile(
        obj,
        mA,
        mB,
        mD,
        mC,
        (),
        make_fake_scheduler_args(False, False, l),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )
    return obj


@requires_sm90
@matrix_exempt("one stage's own contract: reading operand types off the tensors, not a launch")
def test_bind_operand_types_reads_the_operands_and_refuses_what_the_atom_cannot_do():
    """``bind_operand_types`` is where a configuration becomes a kernel for specific operands.

    Nothing downstream can recover from a wrong answer here: ``_setup_attributes`` derives the MMA
    atom and every SMEM layout from these fields, so a mis-read dtype produces a self-consistent
    kernel for the wrong type rather than an error.

    The refusals are ``raise`` and not ``assert`` because ``python -O`` strips asserts, and each
    is checked here at its own message -- a single "it raises something" assertion would pass if
    two of the three checks were deleted.
    """
    g = _compiled_instance(torch.bfloat16)
    assert g.a_dtype is cutlass.BFloat16 and g.b_dtype is cutlass.BFloat16
    assert g.d_dtype is cutlass.BFloat16
    assert g.c_dtype is None, "an absent C must bind to None, not to a default type"
    assert g.two_tensor_B is False
    assert g.a_layout.is_k_major_a() and g.b_layout.is_k_major_b()

    a_dt, b_dt = torch2cute_dtype_map[torch.bfloat16], torch2cute_dtype_map[torch.float16]
    obj = GemmSm90(Float32, a_dt, (128, 128), (1, 1, 1))
    mA = make_fake_gemm_tensors(a_dt, a_dt, a_dt, None, "k", "k", "n", None)[0]
    mB = make_fake_gemm_tensors(b_dt, b_dt, b_dt, None, "k", "k", "n", None)[1]
    with pytest.raises(TypeError, match=r"Type mismatch"):
        obj.bind_operand_types(mA, mB, None, None)


@requires_sm90
@matrix_exempt("one stage's own contract: the pipeline transaction count, not a shape sweep")
def test_make_mainloop_tma_counts_exactly_the_bytes_the_descriptors_move():
    """``num_tma_load_bytes`` must equal one A tile plus one B tile, per stage.

    **This one's failure mode is a hang, not a wrong number**, which is why it earns its own test.
    The AB pipeline's producer commits this byte count and the consumer waits for exactly it: too
    low and the consumer proceeds on a half-written tile, too high and it waits forever for bytes
    nobody will send. An end-to-end correctness test cannot distinguish "right count" from "count
    that happens to be right for this shape".
    """
    for tile_M, tile_N in ((128, 128), (128, 256), (64, 128)):
        g = _compiled_instance(torch.bfloat16, tile=(tile_M, tile_N))
        tile_K = g.cta_tile_shape_mnk[2]
        width = g.a_dtype.width // 8
        expected = (tile_M * tile_K + tile_N * tile_K) * width
        assert g.num_tma_load_bytes == expected, (
            f"tile=({tile_M},{tile_N},{tile_K}): counted {g.num_tma_load_bytes} bytes, "
            f"descriptors move {expected}"
        )


@matrix_exempt("pure host arithmetic on the register file; there is nothing to launch")
def test_resolve_launch_shape_keeps_the_register_budget_feasible():
    """The per-warpgroup register carve must FIT the register file, or the launch wedges.

    An SM has 65536 32-bit registers. ``setmaxregister_increase`` past what the block can afford
    does not fail -- it blocks forever waiting for registers no warp will release. So the
    invariant is arithmetic and absolute:

        mma_threads * num_regs_mma + other_threads * num_regs_load <= 65536

    Checked at the default (no extra warpgroup, where nothing may change) and with one and two
    extra warpgroups, which is the configuration the carve exists for. Granularity 8 is asserted
    too: Hopper allocates registers in blocks of 8, so a budget that is not a multiple of 8 is
    rounded UP by the hardware and can breach a bound that looked satisfied.
    """
    for tile in ((128, 128), (128, 256), (192, 128), (256, 128)):
        base = GemmSm90(Float32, cutlass.BFloat16, tile, (1, 1, 1))
        base.resolve_launch_shape()
        assert base.threads_per_cta == (base.mma_warp_groups + 1) * 128
        default_regs = base.num_regs_mma

        for n_extra in (1, 2):
            g = GemmSm90(Float32, cutlass.BFloat16, tile, (1, 1, 1))
            g._num_extra_warpgroups = lambda n=n_extra: n
            g.resolve_launch_shape()
            assert g.threads_per_cta == (g.mma_warp_groups + 1 + n_extra) * 128
            mma_threads = g.mma_warp_groups * 128
            other = g.threads_per_cta - mma_threads
            used = mma_threads * g.num_regs_mma + other * g.num_regs_load
            assert used <= 65536, (
                f"tile={tile} +{n_extra}wg: carve reserves {used} registers of 65536 -- "
                f"a setmaxnreg above the ceiling blocks forever"
            )
            assert g.num_regs_mma % 8 == 0, "Hopper allocates registers in blocks of 8"
            assert g.num_regs_mma <= default_regs, "an extra warpgroup may only SHRINK the budget"


@matrix_exempt("a structural property of the stage split itself; no kernel is involved")
def test_the_kernel_context_expands_into_exactly_what_each_role_declares():
    """The context-to-parameter expansion cannot drift from the role signatures.

    ``kernel_prologue`` returns a ``_KernelContext``, but the two role methods are ``@cute.jit``
    and so take FLAT parameters -- the DSL flattens arguments into MLIR values and cannot accept a
    Python object. ``_PRODUCER_CTX`` / ``_CONSUMER_CTX`` bridge the two by name.

    That bridge is exactly the kind of thing that rots: adding a field to the context and to one
    role's signature but not to the tuple gives a ``TypeError`` at trace time; adding it to the
    tuple but not the signature gives an unexpected-keyword error. Both are caught by compiling,
    but only for the configurations someone compiles. This checks the names line up by
    construction.
    """
    import inspect

    from fold_cp_ops.kernels.gemm_sm90 import _CONSUMER_CTX, _KernelContext, _PRODUCER_CTX

    slots = set(_KernelContext.__slots__)
    for tup, name in (
        (_PRODUCER_CTX, "producer_warpgroup_role"),
        (_CONSUMER_CTX, "mma_warpgroup_role"),
    ):
        assert set(tup) <= slots, f"{name}: {sorted(set(tup) - slots)} are not context fields"
        # The role methods are @cute.jit-wrapped; unwrap to the function the DSL will call.
        fn = getattr(GemmSm90, name)
        fn = getattr(fn, "__wrapped__", fn)
        params = set(inspect.signature(fn).parameters)
        missing = set(tup) - params
        assert not missing, f"{name} does not accept context field(s) {sorted(missing)}"


# ── the declared unsupported regions ──────────────────────────────────────────────────────────
@GEMM_SM90.parametrize_unsupported("pingpong", "persistent", "tile_M", "tile_N")
def test_gemm_sm90_unsupported_combos_raise(
    pingpong, persistent, tile_M, tile_N, expected_error, expected_match
):
    """Every declared unsupported geometry is refused by the CONSTRUCTOR, with a usable message.

    No GPU and no compile: ``GemmSm90.__init__`` validates the geometry from plain Python ints, so
    each of these costs ~26 us. That is deliberate -- a region whose test is expensive is a region
    people stop running.

    Asserting the raise, rather than xfailing the config, is the whole point: an xfail is satisfied
    by a kernel that quietly accepts the geometry and mis-partitions, which is the failure this
    matrix exists to prevent. ``pytest.raises`` fails that kernel with "DID NOT RAISE".
    """
    with front_door_raises(expected_error, expected_match):
        GemmSm90(
            Float32,
            cutlass.BFloat16,
            (tile_M, tile_N),
            (1, 1, 1),
            pingpong=pingpong,
            is_persistent=persistent,
        )


# ── properties that are not a launch over shapes ───────────────────────────────────────────────
@matrix_exempt("argument validation of a helper function, not a kernel launch over shapes")
@pytest.mark.parametrize(
    "a_major,b_major,should_raise",
    [("k", "k", False), ("m", "k", True), ("k", "n", True), ("m", "n", True)],
)
def test_check_fp8_major(a_major, b_major, should_raise):
    """``check_fp8_major`` refuses a non-k-major 8-bit operand and passes every 16-bit one.

    The same function guards both front doors -- ``GemmSm90.__call__`` (which holds a
    ``LayoutEnum``) and ``kernels.gemm.gemm`` (which holds torch strides) -- so testing it directly
    is testing both. The 16-bit half of the assertion is the one that matters: a check that fired
    on bf16 too would refuse the majors ``test_gemm_sm90_major_modes`` proves work.
    """
    fp8, bf16 = cutlass.Float8E4M3FN, cutlass.BFloat16
    if should_raise:
        with pytest.raises(ValueError, match="must be k-major"):
            check_fp8_major(fp8, a_major, fp8, b_major)
    else:
        check_fp8_major(fp8, a_major, fp8, b_major)
    check_fp8_major(bf16, a_major, bf16, b_major)  # 16-bit: never constrained


@matrix_exempt("a property of the class hierarchy; it has no shape or dtype to parametrize")
def test_gemm_sm90_base_class_is_concrete():
    """``GemmSm90`` defines every hook its own ``epilogue()`` calls, and the attributes it reads.

    Upstream it did not, and the gap was invisible because every mixin filled it: a bare
    ``GemmSm90`` compiled fine and then died inside ``epilogue()`` on ``AttributeError: 'GemmSm90'
    object has no attribute 'epi_setup_postact'``, and after that one was added, on
    ``'rounding_mode'``. Checking for the attributes is worth more than it looks -- it fails at the
    moment the base stops being self-contained, rather than in whichever downstream kernel is next
    to be written without a mixin.

    ``rounding_mode`` is now a CALL-phase template parameter rather than a constructor default, so
    what makes the base concrete is :meth:`GemmSm90.epi_rounding_mode` answering for epilogue
    arguments that carry no such field. That is a stronger guarantee than the old default was: the
    value cannot be missing, and it also cannot be rebound after the kernel is traced off it.
    """
    for hook in (
        "epi_begin",
        "epi_begin_loop",
        "epi_visit_subtile",
        "epi_visit_acc",
        "epi_end",
        "epi_setup_postact",
        "epi_convert_postact",
        "epi_to_underlying_arguments",
        "epi_rounding_mode",
    ):
        assert hasattr(GemmSm90, hook), f"GemmSm90 is missing the epilogue hook {hook!r}"
    gemm = GemmSm90(Float32, cutlass.BFloat16, (128, 128), (1, 1, 1))
    assert not hasattr(gemm, "rounding_mode"), (
        "rounding_mode must NOT exist before __call__: it is a call-phase parameter read off the "
        "epilogue arguments, and a constructor default would be a second, stale source for it"
    )
    assert gemm.epi_rounding_mode(()) is RoundingMode.RN, (
        "the base must answer for epilogue arguments that carry no rounding mode, or a bare "
        "GemmSm90 fails to bind its call parameters"
    )


@matrix_exempt("a property of GemmSm90's constructor; there is no shape or dtype to sweep")
def test_the_base_class_defaults_that_other_modules_rely_on():
    """``rounding_mode`` defaults to ``RoundingMode.RN`` on the base class.

    The assertion is about ``gemm_sm90.py``'s code, so by the repo's naming rule it belongs here --
    it lived in ``tests/_internal/test_rounding.py``, where it read naturally and was wrong twice
    over: a source file is tested in ONE file, and a test that imports a module its own commit does
    not introduce cannot pass at that commit. It was red on three intermediate commits until it
    moved.

    What it pins is a contract between this kernel and ``rounding.py``: ``epilogue()`` reads
    ``self.rounding_mode``, so if nothing supplied one a bare ``GemmSm90`` would die on
    ``AttributeError`` several frames in. The supplier is :meth:`GemmSm90.epi_rounding_mode`, which
    reads the mode off this launch's epilogue arguments and answers ``RN`` when they carry none --
    so both a mixin with a ``rounding_mode`` field and a bare tuple of arguments are covered, by
    one seam rather than by a default that a mixin then overwrites.

    (This test also used to assert that ``use_clc_persistence`` was refused. That flag is gone --
    it could only ever raise, since ``arch`` is the constant 90 -- along with
    ``PersistenceMode.CLC`` and the scheduler branches behind it.)
    """
    from collections import namedtuple

    assert GemmSm90.epi_rounding_mode(()) == RoundingMode.RN
    assert GemmSm90.epi_rounding_mode(None) == RoundingMode.RN
    Args = namedtuple("Args", ["rounding_mode"])
    assert GemmSm90.epi_rounding_mode(Args(RoundingMode.RS)) == RoundingMode.RS, (
        "the seam must PASS THROUGH what the arguments ask for -- an unconditional RN would make "
        "the SM90-only refusal in the post-activation epilogue unreachable, so a Blackwell-only "
        "request would be silently downgraded instead of refused"
    )


@requires_sm90
@matrix_exempt("measures compile WALL TIME against the 5 s bar; the shape is incidental to it")
def test_gemm_sm90_cold_compile_under_five_seconds():
    """A cold compile of the base kernel stays under CLAUDE.md's 5 s bar.

    The bar is a correctness property here, not a nicety: the repo carries a known inherited breach
    on the TriMul path, and the rule that follows is "do not make it worse". A GEMM that drifts over
    five seconds would be a new breach introduced by this bring-back rather than inherited by it.

    The measurement is of the FIRST compile of this configuration in this process. ``_COMPILED``
    would hide a regression behind a cache hit, so this uses a configuration no other test compiles.
    """
    a_dt = torch2cute_dtype_map[torch.bfloat16]
    mA, mB, mD, mC, _, _, _, l = make_fake_gemm_tensors(a_dt, a_dt, a_dt, None, "k", "k", "n", None)
    start = time.perf_counter()
    compile_gemm_kernel(
        GemmSm90,
        a_dt,
        (64, 64),
        (1, 1, 1),
        False,
        True,
        False,
        (9, 0),
        mA,
        mB,
        mD,
        mC,
        (),
        make_fake_scheduler_args(False, False, l),
    )
    elapsed = time.perf_counter() - start
    assert elapsed < 5.0, f"cold compile took {elapsed:.2f}s; CLAUDE.md's hard bar is 5 s"


@matrix_exempt("a property of the declared matrix itself, not of a kernel launch")
def test_every_geometry_cell_is_outside_the_unsupported_regions():
    """The geometry cells and the declared regions do not overlap -- checked here, not assumed.

    ``KernelMatrix.parametrize`` already refuses a cell inside a region at import time, so this
    cannot fail while the module imports. It earns its place by covering the converse: that the
    region predicates agree with ``GemmSm90.__init__`` on the cells we claim are VALID. A region
    predicate that was accidentally too wide would silently shrink the supported grid, and nothing
    else would notice.
    """
    for tile_M, tile_N, _cluster, pingpong, persistent in _GEOMETRY_CELLS:
        bound = dict(tile_M=tile_M, tile_N=tile_N, pingpong=pingpong, persistent=persistent)
        for region in GEMM_SM90.regions():
            args = {k: bound[k] for k in region.axis_names()}
            assert not region.where(**args), (
                f"cell {bound} is claimed valid but matches region {region.reason!r}"
            )
        GemmSm90(
            Float32,
            cutlass.BFloat16,
            (tile_M, tile_N),
            (1, 1, 1),
            pingpong=pingpong,
            is_persistent=persistent,
        )


@matrix_exempt("scores the declared pools; there is no kernel launch to parametrize")
def test_gemm_sm90_matrix_is_diverse():
    """Every facet of every axis is straddled by its pool, above the entropy floor.

    ``tests/testing/test_kernel_matrix.py`` runs this same check across all registered matrices;
    having it here too means a pool narrowed in THIS file fails in this file, where the fix is.
    """
    from fold_cp_ops.testing.kernel_matrix import MIN_FACET_ENTROPY

    weak = [
        s
        for axis in GEMM_SM90.axes
        for s in axis.diversity()
        if s.waived is None and s.entropy < MIN_FACET_ENTROPY
    ]
    assert not weak, "under-covered facets:\n  " + "\n  ".join(str(s) for s in weak)


@matrix_exempt("pure arithmetic on the declared pools; no kernel involved")
def test_the_extent_pools_are_not_all_pitch_aligned():
    """The N and K pools must contain extents that are NOT multiples of the 16-byte floor.

    **This test replaces one that asserted the opposite**, and the correction is the point. The
    earlier version required every pool value to satisfy ``v % 8 == 0`` and called that "the 16-byte
    floor" -- encoding a constraint the kernel does not have. The floor is on the row PITCH; every
    extent is free. Measured: ``torch.empty(l, m, 208)[:, :, :201]`` computes correctly at N=201,
    and ``torch.empty(l, m, 201)`` does not, which localizes the constraint to the pitch and not
    to the 201.

    A pool of only pitch-aligned values cannot tell those two apart, so it would look exactly like
    a pool that had considered odd N and excluded it -- which is the blind spot this whole matrix
    exists to close.
    """
    for name in ("N", "K"):
        pool = GEMM_SM90.axis(name).values
        odd = [v for v in pool if v % 2 == 1]
        assert odd, f"the {name} pool has no odd extent, so odd {name} is untested"
        off_floor = [v for v in pool if v % 8]
        assert off_floor, (
            f"the {name} pool has no extent off the bf16 16-byte floor, so the pitch-vs-extent "
            f"distinction is untested -- keep 1/3/7/201/257 (N) or 1/3/7/129 (K)"
        )
    fp8_only = {v for v in GEMM_SM90.axis("N").values if v % 8 == 0 and v % 16}
    assert fp8_only, (
        "the N pool no longer contains an extent whose PITCH is legal at bf16 but not at fp8 "
        "(8, 200, 1000), so the fp8 tests' narrowing is no longer testing anything."
    )


@matrix_exempt("a table of derived values per configuration; there is no launch to parametrize")
@pytest.mark.parametrize(
    "tile,pingpong,fp8,expect",
    [
        # (tile_M, tile_N)   atom_layout  wgs  block  (load, mma)  epi_warps  sched
        ((64, 128), False, False, ((1, 1, 1), 1, 256, (40, 232), 4, 1)),
        ((128, 128), False, False, ((2, 1, 1), 2, 384, (40, 232), 8, 1)),
        ((128, 256), False, False, ((2, 1, 1), 2, 384, (40, 232), 8, 1)),
        ((192, 128), False, False, ((3, 1, 1), 3, 512, (32, 160), 12, 1)),
        ((192, 256), False, False, ((1, 2, 1), 2, 384, (40, 232), 8, 1)),
        ((256, 128), False, False, ((2, 1, 1), 2, 384, (40, 232), 8, 1)),
        ((320, 128), False, False, ((1, 2, 1), 2, 384, (40, 232), 8, 1)),
        ((64, 128), True, False, ((1, 1, 1), 2, 384, (40, 232), 4, 2)),
        ((128, 192), True, False, ((1, 1, 1), 2, 384, (40, 232), 4, 2)),
        # The heavy-pressure rung: pingpong puts the whole 128x208 tile on ONE warpgroup, so
        # regs_per_thread is exactly 208 and the split flips to (24, 240). Without a row on each
        # side of that threshold the branch is untested.
        ((128, 208), True, False, ((1, 1, 1), 2, 384, (24, 240), 4, 2)),
        # fp8 slow-accum DOUBLES the accumulator footprint, which is the only way a 64x256 pingpong
        # tile reaches the threshold. This row is what proves the derived `fp8_slow_accum` actually
        # reaches the register split rather than merely existing.
        ((64, 256), True, True, ((1, 1, 1), 2, 384, (24, 240), 4, 2)),
        ((128, 128), False, True, ((2, 1, 1), 2, 384, (40, 232), 8, 1)),
    ],
)
def test_the_derived_launch_geometry_matches_the_values_the_constructor_used_to_assign(
    tile, pingpong, fp8, expect
):
    """Every ``cached_property`` reproduces the number ``__init__`` used to write. Table-pinned.

    **This is the perf-neutrality claim made checkable.** Converting 23 constructor assignments into
    derived properties is only safe if each derives the SAME value; a drift in any one of them
    changes the launch geometry -- the block dimension, the register carve, the epilogue barrier's
    arrive count -- and those do not fail loudly. A wrong ``num_epi_warps`` never releases the
    barrier; a wrong ``num_regs_mma`` wedges the launch waiting for registers no warp will free.

    The expected column is the arithmetic the constructor performed, transcribed per configuration
    rather than recomputed: recomputing it here from the same formulas would pass for any formula.

    Args:
        tile: ``(tile_M, tile_N)``.
        pingpong: The two-warpgroup alternating schedule.
        fp8: Whether to build with an 8-bit operand and slow accumulation.
        expect: ``(atom_layout_mnk, mma_warp_groups, threads_per_cta, (num_regs_load,
            num_regs_mma), num_epi_warps, sched_stage)``.
    """
    dtype = torch2cute_dtype_map[torch.float8_e4m3fn if fp8 else torch.bfloat16]
    g = GemmSm90(Float32, dtype, tile, (1, 1, 1), pingpong=pingpong)
    atom, wgs, block, regs, epi_warps, sched = expect
    assert g.atom_layout_mnk == atom
    assert g.mma_warp_groups == wgs
    assert g.threads_per_cta == block, "the block dim is what .launch() is given"
    assert (g.num_regs_load, g.num_regs_mma) == regs
    assert g.num_epi_warps == epi_warps
    assert g.epilogue_barrier.num_threads == epi_warps * 32
    assert g.sched_stage == sched
    assert g.ab_load_warp_id == wgs * 4
    assert g.fp8_slow_accum is fp8


@matrix_exempt("a property of the cluster shape alone; no kernel launch to parametrize")
@pytest.mark.parametrize("cluster", [(1, 1, 1), (2, 1, 1), (1, 2, 1), (2, 2, 1)])
def test_the_multicast_flags_follow_the_cluster_shape_crosswise(cluster):
    """A is multicast along the cluster's N extent and B along its M extent -- crosswise, not direct.

    The crossing is the whole content: A is SHARED by the CTAs that differ in N, so its multicast
    count is ``cluster_N``. Reading them straight would build a multicast descriptor along the wrong
    axis, which delivers each CTA a tile of the wrong operand rows -- a wrong answer, silently, and
    only when the cluster is non-square.
    """
    g = GemmSm90(Float32, cutlass.BFloat16, (128, 128), cluster)
    assert g.num_mcast_ctas_a == cluster[1] and g.num_mcast_ctas_b == cluster[0]
    assert g.is_a_mcast is (cluster[1] > 1) and g.is_b_mcast is (cluster[0] > 1)


@requires_sm90
@matrix_exempt("audits attribute WRITES during one compile; there is no shape or dtype to sweep")
def test_every_attribute_is_a_declared_parameter_or_a_derived_property():
    """No ``GemmSm90`` attribute is loose mutable state: each is a bound parameter or derived from one.

    **This is the invariant the two-phase parameter binding buys, and it is measured rather than
    asserted.** The audit counts every attribute WRITE through one real compile and partitions the
    instance's ``__dict__`` afterwards. Three classes are legitimate and everything else is a bug:

    * a field of ``Params`` (construction) or ``CallParams`` (operands), both frozen once bound;
    * a ``cached_property``, which writes ``__dict__`` directly and cannot desynchronize from what
      it was derived from;
    * the two pack objects themselves, plus ``shared_storage`` -- a Python class assembled per
      launch, not a constant folded into the kernel.

    Counting writes rather than diffing values is deliberate: an equality check misses a rebind to
    the same value, and ``rounding_mode`` and ``threads_per_cta`` were both exactly that before this
    refactor -- rebound during ``__call__`` to a value that compared equal.

    What the earlier version of this test recorded is worth keeping in view, because it is why the
    refactor was not a decoration: of 48 attributes, 10 were REBOUND during ``__call__`` and 15 did
    not exist until it. ``TemplateParams``' "complete at construction" guarantee was structurally
    false for 25 of them. Adding ``CallParams`` is what made the guarantee expressible per phase,
    and converting the 23 derived attributes to properties is what made it total.
    """
    import cutlass.cute as cute

    from fold_cp_ops.kernels.gemm import GemmDefaultSm90

    writes: dict = {}
    tracking = {"on": False}

    class Traced(GemmDefaultSm90):
        """``GemmDefaultSm90`` that counts attribute writes while ``tracking`` is on."""

        def __setattr__(self, k, v):
            """Record the write, then delegate. Bypasses the param guard via ``object``."""
            if tracking["on"]:
                writes[k] = writes.get(k, 0) + 1
            object.__setattr__(self, k, v)

    dt = torch2cute_dtype_map[torch.bfloat16]
    obj = Traced(Float32, dt, (128, 128), (1, 1, 1), pingpong=False, is_persistent=True)
    mA, mB, mD, mC, _, _, _, l = make_fake_gemm_tensors(dt, dt, dt, None, "k", "k", "n", None)
    tracking["on"] = True
    cute.compile(
        obj,
        mA,
        mB,
        mD,
        mC,
        GemmDefaultSm90.EpilogueArguments(),
        make_fake_scheduler_args(False, False, l),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )
    tracking["on"] = False

    declared = set(GemmDefaultSm90.Params.field_names()) | set(
        GemmDefaultSm90.CallParams.field_names()
    )
    derived = {
        name
        for klass in type(obj).__mro__
        for name, member in vars(klass).items()
        if isinstance(member, functools.cached_property)
    }
    # `params` / `call_params` are the packs; `shared_storage` is the per-launch SharedStorage type,
    # a Python class rather than a constant folded into the kernel.
    bookkeeping = {"params", "call_params", "shared_storage"}
    # The post-activation epilogue latches three facts off its own arguments, which the base does
    # not see. They are the remaining loose state and are named here so that adding a FOURTH is a
    # decision rather than an accident.
    epilogue_latched = {"postact_dtype", "postact_layout", "cta_tile_shape_postact_mn"}
    # The FOURTH, and a DIFFERENT kind from the three above -- which is why it gets its own set
    # rather than joining theirs. Those latch facts off the epilogue's ARGUMENTS; `_epi_storage` is
    # a TRACE-TIME SMEM allocation handle, `smem.allocate(self.shared_storage)`, stashed on `self`
    # inside `GemmSm90.kernel()` so the decoupled-A2A producer copy_fn -- reached deep inside the
    # MMA warpgroup's epilogue, which is NOT threaded `storage` -- can reach the ring's SMEM.
    #
    # WHY IT IS INERT for this audit's purpose, which is the only question that matters here. The
    # audit's hazard is state the COMPILE KEY cannot see desynchronizing a cached artifact from the
    # config it was built for. `_epi_storage` cannot: it occurs exactly THREE times in the tree (a
    # comment, this write, one read in `distributed/dual_gated_gemm_a2a.py`), so there is **no
    # host-side read at all** and therefore no phase in which a stale value could be consulted; the
    # write strictly precedes its single reader within the SAME trace, so every compile rewrites it
    # before anything looks; its value derives wholly from `shared_storage`, which this very set
    # already accepts, so two configs sharing a key have the same handle; and it appears in no
    # `compile_key()`, correctly. It is the MIRROR of the recorded `compile_key()` gap -- that was
    # attributes the key could not see, this is one the key does not need to see.
    #
    # WHY A DEFAULT GEMM HAS IT AT ALL, which is the part a reader will not guess. The write serves
    # ONE distant subclass's read, but it lives in the BASE `kernel()`, so every `GemmSm90`
    # descendant acquires it -- including `GemmDefaultSm90`, which never reads it and is the
    # instance this audit compiles. That placement is a deliberate bring-back divergence: `main`
    # sets it in `kernels/dual_gated_gemm_staged.py`'s own `kernel()`, because its A2A class
    # descends from a type that OVERRIDES `kernel()`. Ours descends
    # `DualGatedGemmDistSm90 -> DualGatedGemmSm90 -> GemmGatedMixin, GemmSm90` and overrides
    # `kernel()` NOWHERE, so the base is the only `kernel()` that runs for it. The contract that
    # makes this safe -- that the resolved `kernel()` really does perform the write -- is asserted
    # by `tests/distributed/test_dual_gated_gemm_a2a.py::
    # test_the_kernel_the_a2a_subclass_resolves_is_the_one_that_stashes_epi_storage`.
    #
    # Threading `storage` through the epilogue signature and deleting the attribute is the REAL
    # fix. It is deliberately not done here: it re-plumbs the epilogue store seam, and this is a
    # bring-back where `main` is ground truth for structure as well as behaviour. Scoped work with
    # its own gate, not a side effect of an audit list.
    trace_time_smem_handoff = {"_epi_storage"}

    loose = (
        set(vars(obj))
        - declared
        - derived
        - bookkeeping
        - epilogue_latched
        - trace_time_smem_handoff
    )
    assert not loose, (
        f"these attributes are neither a declared parameter, a derived property, nor known "
        f"bookkeeping: {sorted(loose)}. Every `self.X` a traced method reads is folded into the "
        f"kernel as a constant, so loose state can desynchronize from the kernel compiled off it."
    )

    rebound = {k for k in writes if k in declared}
    assert not rebound, (
        f"declared parameters must be written exactly once, by their bind call: {sorted(rebound)} "
        f"were written again during __call__"
    )
    assert declared <= set(vars(obj)), "a declared parameter did not read back as an attribute"
    assert {"a_dtype", "tile_shape_mn"} <= set(GemmDefaultSm90.Params.field_names())
    assert {"b_dtype", "a_layout", "epi_tile", "ab_stage"} <= set(
        GemmDefaultSm90.CallParams.field_names()
    ), "the operand facts and the stage counts must be CALL parameters, not loose attributes"
    assert not hasattr(obj, "tiled_mma"), (
        "the MMA atom carries MLIR values, so it must be returned by bind_operand_types and passed "
        "to kernel() as an argument -- never stashed on self, where it does not survive the "
        "@cute.jit -> @cute.kernel boundary"
    )


@matrix_exempt("a property of the module's own declaration, not of a kernel launch")
def test_matrix_axis_names_match_the_runner_signature():
    """Axis names line up with what ``run_base_gemm`` and ``_build_operands`` actually take.

    Parametrize passes axis values by NAME into the test function; a renamed axis therefore fails
    loudly at collection. What it does NOT catch is an axis that no longer corresponds to anything
    the runner can vary -- a dead axis still scores for diversity while testing nothing.
    """
    import inspect

    runner = set(inspect.signature(run_base_gemm).parameters)
    builder = set(inspect.signature(_build_operands).parameters)
    declared = {a.name for a in GEMM_SM90.axes}
    unreachable = declared - runner - builder - {"ab_dtype", "persistent"}
    assert not unreachable, (
        f"axes {sorted(unreachable)} are declared but cannot be varied through run_base_gemm or "
        f"_build_operands, so they score for diversity while testing nothing"
    )
    assert math.prod(len(a.values) for a in GEMM_SM90.axes) > 1000, "the declared space collapsed"

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

# Copyright (c) 2025-2026, Tri Dao.
# This file has been modified by NVIDIA CORPORATION & AFFILIATES.

"""The host entry point for the default SM90 GEMM: validate, compile once, dispatch.

`gemm()` is what a caller uses. Everything it does falls into three phases:

1. **Validate at the front door.** Dtypes, the layout requirements
   each of those imposes, and the SM90 capability boundary. All of it raises `ValueError` (or
   `UnsupportedArchError`) with a sentence naming the constraint, *before* anything is traced --
   because the alternative is an MLIR verification failure or a `KeyError` several frames down,
   which tells the caller nothing about which argument to change.
2. **Compile once per configuration.** `_compile_gemm` is `@jit_cache`d on a key built from dtypes,
   majors, tile and cluster shape, and every epilogue *mode* (present/absent/tensor-valued) -- not
   on the shapes. Shapes enter as `cute.sym_int()` symbols through fake tensors, so one compiled
   artifact serves every M/N/K/L.
3. **Dispatch.** The compiled entry is called with the real tensors and the runtime argument
   structs.

The epilogue "modes" in the cache key deserve a note: `alpha` is keyed as 0 (absent, folded away),
1 (a host scalar, baked in), or 2 (a device tensor, read through a pointer). Those are three
different kernels, and collapsing them into one key would hand a caller the wrong one.

This module is SM90-only by construction. Upstream branched on the capability to pick between
`GemmDefaultSm90` / `GemmDefaultSm100` / `GemmDefaultSm120`; `require_sm90()` runs first here, so the
branch has nothing to select between and is gone.
"""

from typing import Optional

from torch import Tensor

import cutlass.cute as cute
from cutlass import Int32, Float32
from cutlass.cute.runtime import make_ptr

from fold_cp_ops._internal.autotune import AxisSpace, TuneAxis, autotune
from fold_cp_ops._internal.cache_utils import jit_cache
from fold_cp_ops._internal.compile_time.compile_utils import make_fake_tensor as fake_tensor
from fold_cp_ops._internal.arch import (
    check_arch_supported,
    get_max_active_clusters,
    require_sm90,
    UnsupportedArchError,
)
from fold_cp_ops._internal.compile_time.cute_dsl_utils import torch2cute_dtype_map
from fold_cp_ops._internal.epi_default import GemmDefaultEpiMixin
from fold_cp_ops._internal.rounding import RoundingMode
from fold_cp_ops._internal.tensor_contract import check_tensor
from fold_cp_ops.kernels.gemm_sm90 import check_fp8_major, GemmSm90
from fold_cp_ops._internal.gemm_tvm_ffi_utils import (
    check_broadcast_alignment,
    describe_operands,
    perm3d,
    make_scheduler_args,
    make_fake_scheduler_args,
    make_fake_gemm_tensors,
    compile_gemm_kernel,
)


def _raise_unsupported_dtype(A, B, D):
    """Name the first operand whose dtype is outside the map, and list what is supported.

    Off `gemm()`'s hot path, so it may build the message. Indexing the map directly would raise
    `KeyError: torch.int8`, which names neither the operand nor the alternatives.

    Args:
        A: The A operand.
        B: The B operand.
        D: The output.

    Returns:
        Never returns.

    Raises:
        ValueError: Always -- one of the three is unsupported, or the caller would not be here.
    """
    for name, t in (("A", A), ("B", B), ("D", D)):
        if t.dtype not in torch2cute_dtype_map:
            raise ValueError(
                f"unsupported dtype for {name}: {t.dtype}. Supported: "
                f"{sorted(str(d) for d in torch2cute_dtype_map)}."
            )
    raise AssertionError("unreachable: _raise_unsupported_dtype called with all dtypes supported")


def _raise_operand_shape_mismatch(A, B, D, C):
    """Name the first extent that disagrees ACROSS the four operands, and say which pair.

    Purpose
        Every other check in this module validates one tensor alone. Nothing compared them to each
        other, so a caller who got one extent wrong reached the TVM-FFI ABI check and was told
        ``Mismatched mB.shape[1] on argument #1 when calling: __call__(mA: Tensor([n0, ...``, which
        names a traced symbol and an FFI slot rather than the argument they passed.

    Semantics
        Checks in the order K, then M, then N, then C, and raises on the FIRST disagreement -- so a
        caller who got two extents wrong is told about the contraction extent first, which is the
        one that makes the other two meaningless. Off the hot path: the caller has already
        established that something disagrees, so this may allocate and build a message.

    Args:
        A: ``(l, m, k)``. Only its trailing two extents are read, so a 2-D A is handled the same way.
        B: ``(l, n, k)`` -- note the transpose; the contraction is over the LAST axis of both.
        D: ``(l, m, n)``, the output.
        C: The optional addend, or None. Must have D's shape exactly; it is added elementwise and is
            never broadcast.

    Returns:
        Never returns.

    Raises:
        ValueError: Always, naming the two tensors that disagree and the axis they disagree on.
        AssertionError: If every extent agrees -- unreachable, and a signal that the guard at the
            call site and this function have drifted apart.
    """
    m, k, n = A.shape[-2], A.shape[-1], B.shape[-2]
    if B.shape[-1] != k:
        raise ValueError(
            f"A and B must contract over the same extent: A is (..., {m}, {k}) and B is "
            f"(..., {n}, {B.shape[-1]}), so K is {k} on one side and {B.shape[-1]} on the other. "
            f"B is stored (l, n, k) -- pass it in that shape rather than transposing an (l, k, n) "
            f"tensor, which would give a stride the TMA descriptor cannot use."
        )
    if D.shape[-2] != m:
        raise ValueError(
            f"D must have A's row count: A is (..., {m}, {k}) and D is (..., {D.shape[-2]}, "
            f"{D.shape[-1]}). D is written in place and nothing resizes it, so it must be "
            f"preallocated at (l, m, n)."
        )
    if D.shape[-1] != n:
        raise ValueError(
            f"D must have B's row count as its column count: B is (..., {n}, {k}) and D is "
            f"(..., {D.shape[-2]}, {D.shape[-1]}). The product is A @ B^T, so D's N comes from B's "
            f"leading matrix axis."
        )
    if C is not None and C.shape != D.shape:
        raise ValueError(
            f"C must be D's shape {tuple(D.shape)}; got {tuple(C.shape)}. C is added elementwise "
            f"and is never broadcast, so a C of any other shape would read past its own end."
        )
    raise AssertionError(
        "unreachable: _raise_operand_shape_mismatch called with every extent agreeing"
    )


class GemmDefaultSm90(GemmDefaultEpiMixin, GemmSm90):
    """The default SM90 GEMM: `GemmSm90`'s mainloop under `GemmDefaultEpiMixin`'s epilogue.

    Everything this class does comes from its two bases, and the ordering is the content: the mixin
    must come first so its `epi_*` hooks override the no-op defaults `GemmSm90` declares. What you
    get is `D = alpha * (A @ B) + beta * C + rowvec + colvec`, with each term dropping out of the
    generated code when the corresponding argument is None.

    Instantiate it the way `GemmSm90.__init__` documents — the mixin adds no constructor arguments,
    only epilogue ones, and those travel in `EpilogueArguments` at call time rather than at
    construction. `gemm()` below is the supported way to reach it; constructing it directly is for
    callers that need to drive `cute.compile` themselves.

    Upstream also defined `GemmDefaultSm100` / `GemmDefaultSm120` alongside this. Both are dropped:
    `fold-cp-ops` is SM90-only, and callers gate on the architecture with `require_sm90()` *before*
    selecting a class, so there is nothing left to select between.
    """


@jit_cache
def _compile_gemm(
    a_dtype,
    b_dtype,
    d_dtype,
    c_dtype,
    a_major,
    b_major,
    d_major,
    c_major,
    tile_shape_mn,
    cluster_shape_mnk,
    pingpong,
    persistent,
    is_dynamic_persistent,
    rowvec_dtype,
    colvec_dtype,
    colvec_ndim,
    alpha_mode,
    beta_mode,
    add_to_output,
    has_batch_idx_permute,
    device_capacity,
    rounding_mode,
    sr_seed_mode,
    run_j_tiles=0,
):
    """Compile one `GemmDefaultSm90` configuration against fake tensors. Cached on every argument.

    Every parameter is part of the `@jit_cache` key, and that is the point: two calls agreeing on
    all of them can share a compiled artifact, and two that differ in any of them must not. Shapes
    are deliberately absent -- they enter as `cute.sym_int()` symbols via `make_fake_gemm_tensors`,
    so one artifact serves all M/N/K/L.

    Args:
        a_dtype: Cutlass element type of A. 16-bit (fp16/bf16) or 8-bit (fp8 e4m3/e5m2).
        b_dtype: Element type of B. Must equal `a_dtype` at 16 bits, and match its *width* at 8.
        d_dtype: Element type of the output D.
        c_dtype: Element type of the optional C addend, or None when there is no C.
        a_major: `"m"` or `"k"` -- which axis of A is contiguous. fp8 requires `"k"`.
        b_major: `"n"` or `"k"`. fp8 requires `"k"`.
        d_major: `"m"` or `"n"`.
        c_major: `"m"`, `"n"`, or None when C is absent.
        tile_shape_mn: `(tile_M, tile_N)`. Validated by `GemmSm90.__init__`, which raises on a
            shape the WGMMA atom cannot be built for.
        cluster_shape_mnk: `(cluster_M, cluster_N, 1)`. Product must be a power of two <= 8.
        pingpong: Whether to use the two-warpgroup ping-pong schedule.
        persistent: Whether the grid is persistent (one wave, looping over tiles).
        is_dynamic_persistent: Whether tiles are handed out by a GMEM atomic rather than statically.
        rowvec_dtype: Element type of the row-vector bias, or None when absent.
        colvec_dtype: Element type of the column-vector bias, or None when absent.
        colvec_ndim: 2 for a `(l, m)` bias, 0 for absent.
        alpha_mode: 0 absent / 1 host scalar (baked in) / 2 device pointer. Three distinct kernels.
        beta_mode: Same encoding for beta.
        add_to_output: Whether the epilogue accumulates into D rather than overwriting it.
        has_batch_idx_permute: Whether the scheduler reorders batches.
        device_capacity: `(major, minor)`, re-checked below rather than trusted.
        rounding_mode: A `RoundingMode`. Baked in as a compile-time constant.
        sr_seed_mode: Same 0/1/2 encoding as `alpha_mode`, for the stochastic-rounding seed.
        run_j_tiles: Scheduler j-tile count; a `Constexpr` field baked at compile time.

    Returns:
        The compiled TVM-FFI entry, callable as
        `fn(A, B, D, C, epi_args, scheduler_args)`.

    Raises:
        UnsupportedArchError: If `device_capacity` is not SM90.
        ValueError: Propagated from `GemmSm90` for a tile geometry or operand layout it refuses.
    """
    # SM90-only: upstream picked GemmDefaultSm100 when device_capacity[0] > 9.  `gemm()` rejects a
    # non-SM90 device before it ever gets here, and this entry is @jit_cache'd on device_capacity,
    # so re-check rather than trust the key.
    check_arch_supported(device_capacity)
    GemmCls = GemmDefaultSm90
    mA, mB, mD, mC, m, n, k, l = make_fake_gemm_tensors(
        a_dtype,
        b_dtype,
        d_dtype,
        c_dtype,
        a_major,
        b_major,
        d_major,
        c_major,
    )

    def fake_scalar(mode, dtype=Float32):
        """Build the compile-time stand-in for one epilogue scalar, per its mode.

        Args:
            mode: 0 absent, 1 host scalar, 2 device pointer. Anything else falls into the pointer
                branch, which would compile a kernel expecting an argument the caller will not pass.
            dtype: The scalar's cutlass type.

        Returns:
            None, a dummy value of ``dtype``, or a null GMEM pointer -- whichever makes the traced
            kernel take the branch this mode selects. The VALUES are irrelevant; only the shape of
            the argument is being declared.
        """
        if mode == 0:
            return None
        elif mode == 1:
            return dtype(1.0 if dtype == Float32 else 0)
        else:
            return make_ptr(dtype, 0, cute.AddressSpace.gmem, assumed_align=4)

    mRowVec = fake_tensor(rowvec_dtype, (l, n), leading_dim=1, divisibility=4)
    if colvec_ndim == 2:
        mColVec = fake_tensor(colvec_dtype, (l, m), leading_dim=1, divisibility=4)
    else:
        mColVec = None

    epi_args = GemmCls.EpilogueArguments(
        alpha=fake_scalar(alpha_mode),
        beta=fake_scalar(beta_mode),
        mRowVecBroadcast=mRowVec,
        mColVecBroadcast=mColVec,
        add_to_output=add_to_output,
        rounding_mode=rounding_mode,
        sr_seed=fake_scalar(sr_seed_mode, dtype=Int32),
    )
    scheduler_args = make_fake_scheduler_args(
        (is_dynamic_persistent and device_capacity[0] == 9),
        has_batch_idx_permute,
        l,
        run_j_tiles=run_j_tiles,
    )
    return compile_gemm_kernel(
        GemmCls,
        a_dtype,
        tile_shape_mn,
        cluster_shape_mnk,
        pingpong,
        persistent,
        is_dynamic_persistent,
        device_capacity,
        mA,
        mB,
        mD,
        mC,
        epi_args,
        scheduler_args,
    )


# ───────────────────────────── autotuning: the declared space ─────────────────────────────
# Reproduced from `main`'s `gemm_config._get_sm90_configs(epilogue=None)`, which is the pool a tuned
# build there sweeps. Reproducing it rather than inventing one is the point: a pool narrower than
# `main`'s is a perf regression that no correctness test can see, and a pool WIDER than `main`'s
# would make a comparison against it meaningless.
#
# ONE difference remains, and it is now scoped rather than absent: `main` also swept `swap_ab`
# (compute B @ A^T and store the result transposed), doubling its pool to 44. It was dropped here
# because it lived in `main`'s `gemm_interface` layer, which this tree does not have. That reasoning
# held for the LAYER and not for the KNOB: the swap is a host-side operand permutation, ~6 lines,
# with no kernel variant behind it (`swap_ab` appears in NO kernel file on main's side). Measured
# consequence of its absence: main's tuner picks `swap_ab=True` in 18 of 18 cells and our tuned
# `gemm_hadamard` runs up to 1.0994x slower for want of the axis.
#
# So `gemm_tuning_space(swap_ab=True)` opts an entry into it, and `gemm_hadamard` does. `gemm()`
# itself still does not: it has no `swap_ab` parameter, so its pool stays at 22 and the axis stays
# off by default. Widening `gemm()` the same way is the obvious next step and is NOT done here --
# it is a separate entry with a separate result cache and its own parity measurement to make.

#: The ``(tile_M, tile_N, pingpong)`` combinations `main` measures. Two lists there, because the
#: legal tile shapes differ between the cooperative and ping-pong schedules -- so the tile axes are
#: JOINT with the schedule axis rather than independent of it.
_MAIN_TILE_SCHEDULES = frozenset(
    [(256, n, False) for n in (128, 160, 192, 208)]
    + [(128, 224, False), (128, 256, False)]
    + [(128, n, True) for n in (128, 160, 192, 208)]
    + [(192, 128, True)]
)

#: The clusters `main` measures. ``(1, 1)`` is deliberately absent -- see the axis domain below.
_MAIN_CLUSTERS = frozenset({(1, 2), (2, 1)})


def gemm_tuning_space(*, swap_ab: bool = False) -> AxisSpace:
    """The declared autotuning axes for `gemm`, reproducing `main`'s SM90 pool.

    Purpose
        Says what is tuned, in the kernel's OWN parameter names, so a reader can match an axis to
        the argument it sets without a translation step.

    Semantics
        Five axes over their full domains -- 144 points -- of which 122 are excluded and 22 survive:
        exactly `main`'s pool modulo the absent ``swap_ab`` knob. Declaring the full grid and
        removing from it, rather than listing 22 points, is what keeps a combination nobody
        considered distinguishable from one considered and rejected. ``cluster (1, 1)`` is the case
        that matters: it IS in the grid and IS excluded, with the measurement behind it written down.

    Args:
        swap_ab: Append the ``swap_ab`` axis, doubling the pool 22 -> 44 exactly as `main`'s does.
            Keyword-only and default False because it is NOT a universal axis of this tree: `gemm()`
            has no such parameter, so a pool carrying it would hand the tuner a knob the entry
            cannot accept. Only an entry that IMPLEMENTS the swap may ask for it -- today that is
            `gemm_hadamard` alone. Passing True from an entry without the parameter surfaces as a
            `TypeError` on the first tuned call, not as a silently ignored axis.

    Returns:
        A fresh `AxisSpace`. Fresh rather than a module singleton so a caller can inspect or subset
        it without mutating what the decorator below is bound to.
    """
    swap_axis = (
        (
            TuneAxis(
                "swap_ab",
                domain="compute the TRANSPOSED problem (B @ A^T) and store transposed; a pure host-side "
                "operand permutation, no kernel variant",
                values=(False, True),
            ),
        )
        if swap_ab
        else ()
    )
    return AxisSpace(
        *swap_axis,
        TuneAxis(
            "tile_M",
            domain="64/128/192/256/320; 64/128/192 under pingpong. GemmSm90.__init__ raises on "
            "the rest",
            values=(128, 192, 256),
        ),
        TuneAxis(
            "tile_N",
            domain="divisible by 16 and <= 256, or by 32 and <= 512, with narrower limits at "
            "tile_M 192/320 and under pingpong",
            values=(128, 160, 192, 208, 224, 256),
        ),
        TuneAxis(
            "pingpong",
            domain="the two-warpgroup alternating schedule; requires persistent=True",
            values=(False, True),
        ),
        TuneAxis(
            "cluster_M",
            domain="power of two, cluster_M * cluster_N <= 8",
            values=(1, 2),
        ),
        TuneAxis(
            "cluster_N",
            domain="power of two, cluster_M * cluster_N <= 8",
            values=(1, 2),
        ),
        exclude=lambda p: (
            (p["tile_M"], p["tile_N"], p["pingpong"]) not in _MAIN_TILE_SCHEDULES
            or (p["cluster_M"], p["cluster_N"]) not in _MAIN_CLUSTERS
        ),
        because=(
            "two independent reasons, both from main. (1) The tile axes are JOINT with the schedule "
            "axis: main enumerated tile_mn_coop_vals and tile_mn_pingpong_vals separately because "
            "the legal shapes differ, so a coop tile under pingpong is either refused by "
            "GemmSm90.__init__ or is a shape main never measured. (2) Clusters (1, 1) and (2, 2) "
            "are out. (2, 2) main never swept on SM90. (1, 1) upstream turned OFF deliberately "
            "(upstream 785c0dc8, 'Only tune cluster 2x1 and 1x2, don't tune 1x1'), and measurement "
            "here supports it: across five shapes the best config including all eleven (1, 1) "
            "variants matches the best without them to within noise -- (1, 1) TIES at thin-K "
            "(0.0145 vs 0.0145 ms) and loses by 1% compute-bound and 11% at the M-tail shape, while "
            "adding it grows every sweep by 50%. Note SM100 keeps (1, 1), so this is SM90-specific."
        ),
    )


#: The pool the tuned path sweeps. Built once at import: `AxisSpace.configs()` is pure, and the
#: decorator needs the points at decoration time.
GEMM_TUNING_SPACE = gemm_tuning_space()


def gemm_config_is_valid(config, request) -> bool:
    """Whether one candidate CAN RUN for this request. Rejection only -- never preference.

    Purpose
        `main`'s `prune_invalid_gemm_configs` filtered on `swap_ab`, gather-A and varlen, none of
        which exist in this tree, plus a device-capacity filter that is vacuous in an SM90-only
        package. What remains is the one constraint this tree's own front door imposes.

    Semantics
        Must be a PURE function of the config and the request: the candidate list has to be
        identical on every rank, or ranks measure different pools and the consensus step compares
        numbers for different kernels. It may reject; it may not reorder or prefer.

    Args:
        config: The candidate, carrying `tile_M`, `tile_N`, `pingpong`, `cluster_M`, `cluster_N`.
        request: The bound call arguments by name, as the tuner assembles them.

    Returns:
        False for a pingpong candidate under a non-persistent request -- `GemmSm90.__init__` raises
        on that pair, and a candidate that raises is not a slow config, it is a crashed sweep.
        True otherwise.
    """
    if config["pingpong"] and not request.get("persistent", True):
        return False
    return True


def default_config() -> dict:
    """The config `gemm` runs when the caller pins no tile — `main`'s `default_config`, SM90 arm.

    Purpose
        Give a caller who has no measured tile the same config `main` uses, so a call that names no
        tile means the same thing in both trees. `main`'s `gemm_interface.gemm_tuned` resolves
        ``config is None`` here; this is the port of that resolution.

    Functionality & semantics
        Returns `gemm`'s knob names, not `main`'s (`tile_M` vs `main`'s `tile_m`), because the
        result is splatted straight into this module's signature. The values are `main`'s verbatim.

        `main` branches on `get_device_capacity(device)[0] != 10` and carries an SM100 arm
        (256/256, no pingpong, dynamic-persistent). **That arm is dropped, not forgotten:** this
        package is SM90-only by decision — `require_sm90` gates every entry here — so the SM100
        branch would be unreachable by construction, and shipping an unreachable branch that reads
        live is worse than not shipping it. The returned values ARE `main`'s non-SM100 arm, which
        is the only one an SM90-only package can take. There is consequently no `device` argument:
        with one reachable arm there is nothing to dispatch on.

    Returns:
        A dict of keyword arguments for `gemm`: ``tile_M``, ``tile_N``, ``cluster_M``, ``cluster_N``,
        ``pingpong``, ``is_dynamic_persistent``. Fresh each call, so a caller may mutate it.
    """
    return dict(
        tile_M=128,
        tile_N=192,
        cluster_M=2,
        cluster_N=1,
        pingpong=True,
        is_dynamic_persistent=False,
    )


@autotune(
    space=GEMM_TUNING_SPACE,
    key=["persistent", "is_dynamic_persistent", "add_to_output", "max_swizzle_size"],
    validity=gemm_config_is_valid,
    gate="do_autotune",
)
def gemm(
    A: Tensor,  # (l, m, k)
    B: Tensor,  # (l, n, k)
    D: Tensor,  # (l, m, n)
    C: Optional[Tensor],  # (l, m, n)
    tile_count_semaphore: Optional[Tensor],  # (1,)
    tile_M: Optional[int] = None,
    tile_N: Optional[int] = None,
    cluster_M: int = 1,
    cluster_N: int = 1,
    pingpong: bool = False,
    persistent: bool = True,
    is_dynamic_persistent: bool = False,
    max_swizzle_size: int = 8,
    rowvec_bias: Optional[Tensor] = None,  # (l, n)
    colvec_bias: Optional[Tensor] = None,  # (l, m)
    alpha: float | Tensor = 1.0,
    beta: float | Tensor = 1.0,
    batch_idx_permute: Optional[Tensor] = None,  # (l,) permutation of batch indices for scheduler
    add_to_output: bool = False,
    rounding_mode: int = RoundingMode.RN,
    sr_seed: int | Tensor = 0,
    run_j_tiles: int = 0,
    *,
    do_autotune: bool = False,
) -> None:
    """Batched GEMM `D = alpha * (A @ B^T) + beta * C + rowvec + colvec`, written in place into D.

    Note the transpose: B is stored `(l, n, k)`, so the contraction is over the LAST axis of both
    operands. That is the layout WGMMA wants and it is not negotiable through this entry -- pass B
    already in that shape rather than transposing a `(l, k, n)` tensor, which would give a non-unit
    stride the TMA descriptor cannot use.

    Args:
        A: `(l, m, k)`. Must be on a CUDA device -- this is the device whose capability is gated
            on.
        B: `(l, n, k)`.
        D: The output, `(l, m, n)`. **Written in place**; it
            is an argument rather than a return value so the caller controls the allocation. Must be
            preallocated with the right shape -- nothing resizes it.
        C: Optional addend with D's shape and layout. Absent means the beta term is folded away
            entirely, not multiplied by zero.
        tile_count_semaphore: A zeroed `int32` tensor of shape `(1,)`. Required when
            `is_dynamic_persistent`, ignored otherwise. Must be **zeroed before every launch**: it is
            an atomic ticket counter, and a stale value makes the grid skip tiles, which is a
            silently truncated output.
        tile_M: CTA tile M. One of 64/128/192/256/320 (64/128/192 with `pingpong`). Required
            unless `do_autotune`, whose sweep supplies it; the None default exists only for that.
        tile_N: CTA tile N. Divisible by 16 and <= 256, or divisible by 32 and <= 512; narrower
            limits at `tile_M` 192/320 and under `pingpong`. `GemmSm90.__init__` raises on the rest.
        cluster_M: Threadblock-cluster extent along M. Power of two; `cluster_M * cluster_N <= 8`.
        cluster_N: Cluster extent along N. Power of two; `cluster_M * cluster_N <= 8`.
        pingpong: Use the two-warpgroup ping-pong schedule, which overlaps one warpgroup's epilogue
            with the other's mainloop. Requires `persistent=True`.
        persistent: Launch one resident wave that loops over work tiles, instead of one CTA per
            tile.
        is_dynamic_persistent: Hand out work tiles through the GMEM atomic in `tile_count_semaphore`
            rather than statically. Helps when tiles are uneven; costs an atomic per tile.
        max_swizzle_size: Tile-scheduler rasterization swizzle width, for L2 locality. Purely a
            performance knob.
        rowvec_bias: Optional `(l, n)` vector added to every row of the output.
        colvec_bias: Optional `(l, m)` vector added to every column.
        alpha: Scale on `A @ B^T`. A Python float is baked into the kernel at compile time; a
            device `Tensor` (1 element, fp32) is read at launch. `1.0` compiles the multiply away.
        beta: Scale on C, with the same three modes. Ignored when C is None.
        batch_idx_permute: Optional `(l,)` int32 permutation changing the order the scheduler visits
            batches. Affects scheduling only, never the result.
        add_to_output: Accumulate into D instead of overwriting it.
        rounding_mode: A `RoundingMode`. Only `RN` is reachable here; `RS` is SM100-only.
        sr_seed: Stochastic-rounding seed, scalar or device tensor. Unused under `RN`.
        run_j_tiles: Scheduler j-tile count, baked in at compile time. 0 disables it.
        do_autotune: Choose `tile_M`, `tile_N`, `cluster_M`, `cluster_N` and `pingpong` by
            MEASUREMENT instead of taking them from the caller. Keyword-only. Left False (the
            default) this function is exactly the fixed-config entry it has always been, and the
            gate costs one dict lookup -- measured free against a 14.5 us kernel -- so a pinned perf
            cell still times a kernel rather than a sweep. Set True and those five arguments must
            NOT be passed: pinning one while the tuner picks the rest would make the config that was
            measured and the config that ran differ, so it is refused rather than merged. The
            winning config is readable afterwards as ``gemm.autotuner.best_config``.

    Returns:
        None. The result is in `D`.

    Raises:
        ValueError: For any front-door violation above -- an unsupported dtype, an operand that is
            neither 16- nor 8-bit, a non-contiguous or misaligned operand, `is_dynamic_persistent`
            without a semaphore, or a missing tile shape without `do_autotune`. Also raised by the
            tuner when a tuned knob is passed alongside `do_autotune=True`, and propagated from
            `GemmSm90` for a tile geometry or fp8 operand layout it refuses.
        UnsupportedArchError: If the device is not SM90, or `rounding_mode=RoundingMode.RS`.
    """
    # The capability boundary comes FIRST, before any argument validation. On a device this package
    # has no kernels for, a complaint about a dtype or a stride is a distraction -- and upstream's
    # arch branch would have died on a NameError for a class that was deleted. Everything below
    # assumes SM90.
    device_capacity = require_sm90(A.device)
    if rounding_mode == RoundingMode.RS:
        # Stochastic rounding is an SM100+ (Blackwell) epilogue; unreachable on SM90.
        raise UnsupportedArchError(
            "Stochastic rounding (RoundingMode.RS) requires SM100+ (Blackwell); "
            "fold-cp-ops is SM90-only."
        )

    # Front-door validation.  Every one of these was an `assert` upstream; they are `raise
    # ValueError` here because `python -O` strips asserts, and a stripped guard does not fail
    # loudly -- it produces a kernel reading the wrong strides, which is a wrong answer.
    # One combined membership test rather than a loop over a tuple of pairs: three dict lookups and
    # no allocation on the success path, which this is on every call.
    if (
        A.dtype not in torch2cute_dtype_map
        or B.dtype not in torch2cute_dtype_map
        or D.dtype not in torch2cute_dtype_map
    ):
        _raise_unsupported_dtype(A, B, D)
    # C and the two broadcast vectors go through the SAME check as the operands. They used to have
    # none at all, so an unsupported dtype on any of them surfaced as a bare `KeyError` from the
    # `torch2cute_dtype_map[...]` lookup that builds the compile key -- naming a torch dtype and no
    # argument. Which of these the kernel TMAs and which it broadcast-loads is not visible in this
    # signature, so it must not decide whether a caller mistake is named.
    check_tensor("C", C)
    check_tensor("rowvec_bias", rowvec_bias, expect_shape=(None, D.shape[-1]))
    check_tensor("colvec_bias", colvec_bias, expect_shape=(None, D.shape[-2]))
    # The four operands' EXTENTS must agree with each other, which none of the checks above tests --
    # each validates one tensor alone. A mismatch therefore reached the TVM-FFI ABI check as
    # `Mismatched mB.shape[1] on argument #1`, naming a traced symbol and an FFI slot rather than
    # the argument the caller passed. Four int comparisons and no allocation on the success path,
    # which this is on every call; the message is built in `_raise_operand_shape_mismatch`, off it.
    # Note D's extent was previously validated only as a SIDE EFFECT of `rowvec_bias` being present
    # (that check compares against D.shape[-1]), so a mis-sized D went unnamed whenever the optional
    # arguments were absent -- a pass contingent on an unrelated argument, which reads as coverage.
    # `.shape` is a PROPERTY that builds a fresh `torch.Size` on every read, so the obvious spelling
    # of this test costs eight of them per call. Hoisting the three into locals is worth 5.4% of this
    # entry's whole host dispatch -- measured, and it is the entire cost of adding this check: the
    # hoisted form lands at 1.452x a bare GemmSm90 launch against 1.448x for deleting the check
    # outright. Two intuitive-looking alternatives are SLOWER and were measured before being
    # discarded: `.size(-1)` instead of `.shape[-1]` (1.547x), and replacing the C tuple compare with
    # per-dimension compares (1.489x). Do not "simplify" this back.
    _sa, _sb, _sd = A.shape, B.shape, D.shape
    if (
        _sa[-1] != _sb[-1]
        or _sa[-2] != _sd[-2]
        or _sb[-2] != _sd[-1]
        or (C is not None and C.shape != _sd)
    ):
        _raise_operand_shape_mismatch(A, B, D, C)
    # A and B must additionally be 16- or 8-bit -- fp32 is IN the map (D and the biases may be
    # fp32) so the membership check lets it through, and GemmSm90.__call__ then refuses it 59 ms
    # later from inside tracing. That check rides along in `describe_operands` below rather than
    # here, because it needs the cutlass dtype that pass is already looking up.

    if tile_M is None and tile_N is None:
        # Naming NO tile means `main`'s default config, exactly as `main`'s `gemm_tuned` resolves
        # `config is None`.  The WHOLE config is taken, not just the tile: `main`'s default carries
        # cluster (2,1) and pingpong, and a caller who got the tile but kept this signature's
        # cluster/pingpong defaults would run a config that exists in NEITHER tree.  Measured: op 7
        # at N_token=2048/D=128 is 1.425x `main` on the (128,128) no-cluster no-pingpong config and
        # 0.989x on this one, so the difference is the whole of that gap, not a detail.
        for _k, _v in default_config().items():
            if _k == "tile_M":
                tile_M = _v
            elif _k == "tile_N":
                tile_N = _v
            elif _k == "cluster_M":
                cluster_M = _v
            elif _k == "cluster_N":
                cluster_N = _v
            elif _k == "pingpong":
                pingpong = _v
            elif _k == "is_dynamic_persistent":
                is_dynamic_persistent = _v
    elif tile_M is None or tile_N is None:
        raise ValueError(
            "tile_M and tile_N must be given together, or both omitted. One alone is refused "
            "because there is nothing to pair it with: omitting BOTH selects `default_config()` "
            "as a unit (tile, cluster AND pingpong), and half a pinned tile would silently mix "
            "one caller value into that config. Pass both, pass neither, or pass "
            "do_autotune=True and let the sweep choose."
        )

    if is_dynamic_persistent and tile_count_semaphore is None:
        raise ValueError(
            "is_dynamic_persistent=True requires tile_count_semaphore: the SM90 dynamic scheduler "
            "hands out work tiles through an atomic counter in GMEM, and there is no hardware "
            "cluster-launch-control fallback on Hopper. Pass a zeroed int32 tensor of shape (1,)."
        )

    A_p, B_p, D_p, C_p = perm3d(A, B, D, C)
    # ONE pass per operand for the majors, the dtypes and every layout check. Asking for them
    # separately indexed torch2cute_dtype_map 13 times per launch and walked the strides twice --
    # invisible on a device-bound shape, ~10% of a launch-bound one, and caught by the perf gate.
    majors, dtypes = describe_operands(A_p, B_p, D_p, C_p)
    a_major, b_major, d_major, c_major = majors
    check_broadcast_alignment("rowvec_bias", rowvec_bias)
    check_broadcast_alignment("colvec_bias", colvec_bias)
    a_dtype, b_dtype, d_dtype, c_dtype = dtypes
    check_fp8_major(a_dtype, a_major, b_dtype, b_major)

    alpha_mode = 2 if isinstance(alpha, Tensor) else (1 if alpha != 1.0 else 0)
    beta_mode = 2 if isinstance(beta, Tensor) else (1 if beta != 1.0 else 0)
    colvec_ndim = colvec_bias.ndim if colvec_bias is not None else 0

    sr_seed_mode = (
        2 if isinstance(sr_seed, Tensor) else (1 if rounding_mode == RoundingMode.RS else 0)
    )
    compiled_fn = _compile_gemm(
        a_dtype,
        b_dtype,
        d_dtype,
        c_dtype,
        a_major,
        b_major,
        d_major,
        c_major,
        (tile_M, tile_N),
        (cluster_M, cluster_N, 1),
        pingpong,
        persistent,
        is_dynamic_persistent,
        torch2cute_dtype_map[rowvec_bias.dtype] if rowvec_bias is not None else None,
        torch2cute_dtype_map[colvec_bias.dtype] if colvec_bias is not None else None,
        colvec_ndim,
        alpha_mode,
        beta_mode,
        add_to_output,
        batch_idx_permute is not None,
        device_capacity,
        rounding_mode,
        sr_seed_mode,
        run_j_tiles,
    )

    from fold_cp_ops._internal.cache_utils import COMPILE_ONLY

    if COMPILE_ONLY:
        return

    def scalar_arg(scalar, mode, dtype=Float32):
        """Convert one epilogue scalar into what the compiled entry expects for its mode.

        The launch-time counterpart of ``fake_scalar``: the same 0/1/2 encoding, but carrying the
        real value. The two must agree, since the compiled prototype was built from the fake one.

        Args:
            scalar: The user's value -- ignored at mode 0, converted at mode 1, and expected to be a
                torch tensor at mode 2 (its ``data_ptr()`` is taken).
            mode: 0 absent, 1 host scalar, 2 device pointer.
            dtype: The cutlass type to convert a host scalar to.

        Returns:
            None, a ``dtype`` value, or a device address.
        """
        if mode == 0:
            return None
        elif mode == 1:
            return dtype(scalar)
        else:
            return scalar.data_ptr()

    max_active_clusters = get_max_active_clusters(cluster_M * cluster_N) if persistent else 0

    epi_args = GemmDefaultEpiMixin.EpilogueArguments(
        alpha=scalar_arg(alpha, alpha_mode),
        beta=scalar_arg(beta, beta_mode),
        mRowVecBroadcast=rowvec_bias,
        mColVecBroadcast=colvec_bias,
        add_to_output=None,
        rounding_mode=None,
        sr_seed=scalar_arg(sr_seed, sr_seed_mode, dtype=Int32),
    )
    scheduler_args = make_scheduler_args(
        max_active_clusters,
        max_swizzle_size,
        tile_count_semaphore,
        batch_idx_permute,
        run_j_tiles=run_j_tiles,
    )
    # SM90 call form (6 args).  Upstream had a 9-arg SM100 branch here; `require_sm90` above makes
    # it unreachable.  Note this short call against a longer-arity entry is exactly what
    # tests/test_jit_cache_reload_abi.py guards on a warm disk cache.
    compiled_fn(A_p, B_p, D_p, C_p, epi_args, scheduler_args)

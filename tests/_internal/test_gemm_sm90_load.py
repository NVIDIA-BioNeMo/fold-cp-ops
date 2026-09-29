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
"""Tests for ``fold_cp_ops._internal.gemm_sm90_load`` -- the operand-load layer, in ISOLATION.

**Why an isolated harness and not another end-to-end GEMM.** The assembled kernel already gates
this code bit-exactly (``tests/kernels/test_gemm_sm90.py`` runs ``A @ I^T`` and requires A back
unchanged). What it cannot do is *localise*: a wrong element there says the transport is broken,
not which stage broke it. A load that stages the wrong k-tile and an epilogue that stores to the
wrong subtile produce the same symptom.

So the harness here replaces the MMA consumer with a **dump**: the producer runs the real
``load_AB`` against real TMA descriptors and a real pipeline, and the consumer copies the staged
SMEM tile straight out to global memory instead of multiplying it. If that copy does not return
the input, the defect is in the load layer and nowhere else.

**The harness is itself evidence the refactor worked.** It is a nine-line subclass that overrides
``mma_warpgroup_role`` and inherits the prologue, the pipelines, the descriptors and the producer
unchanged. Before the split there was no seam smaller than ``kernel()``, so this test could only
have been written by copying the whole kernel -- which is exactly the failure mode the split
exists to remove.

Data movement is *pure transport*, so every assertion here is ``assert_bitwise``: there is no
tolerance to grant, and none is granted.

**The probe runs NON-PERSISTENT, and that is load-bearing rather than incidental.** In a
persistent grid the tile scheduler broadcasts each next work tile through its own pipeline, whose
consumer arrive count is sized to include the MMA warpgroups (``make_sched_pipeline``). A probe
that replaces the MMA role therefore never arrives, and the producer blocks forever on a handshake
nobody completes -- measured, twice, as a 7-minute test timeout with no error. One CTA per work
tile removes the handshake entirely, and since the probe already requires a single work tile there
is nothing to schedule anyway.
"""

import pytest
import torch

import cutlass
import cutlass.cute as cute
from cutlass import Float32, const_expr

from fold_cp_ops._internal.compile_time.cute_dsl_utils import torch2cute_dtype_map
from fold_cp_ops._internal.gemm_sm90_load import GemmSm90LoadMixin
from fold_cp_ops._internal.gemm_tvm_ffi_utils import (
    compile_gemm_kernel,
    make_fake_gemm_tensors,
    make_fake_scheduler_args,
    make_scheduler_args,
    perm3d,
)
from fold_cp_ops._internal.pipeline import make_pipeline_state
from fold_cp_ops.kernels.gemm_sm90 import GemmSm90, NamedBarrierGemm
from fold_cp_ops.testing.numerics import (
    assert_bitwise,
    exact_int_bound,
    integer_operands,
)

_SM = torch.cuda.get_device_capability(0)[0] if torch.cuda.is_available() else 0
requires_sm90 = pytest.mark.skipif(_SM != 9, reason=f"needs sm_90; this GPU is sm_{_SM}0")


class _LoadProbeGemm(GemmSm90):
    """A ``GemmSm90`` whose MMA warpgroups dump the staged A tile instead of multiplying it.

    Everything upstream is the production path: the same prologue, the same AB pipeline, the same
    TMA descriptors, the same ``producer_warpgroup_role`` calling the same ``load_AB``. Only the
    consumer is replaced, so a mismatch in the dumped tile can only have come from the load layer.

    The dump target is the kernel's own ``mD_mnl``. Reusing the output tensor rather than adding a
    parameter keeps the kernel signature -- and therefore the traced graph and the compile path --
    identical to production; the harness differs from the real kernel in exactly one method.

    Input requirements (the harness does not check them; a violation is a wrong answer, not an
    error):

    * exactly ONE work tile, i.e. ``M <= tile_M`` and ``N <= tile_N``. The dump indexes D by k-tile
      alone, so a second work tile would overwrite the first's dump.
    * ``mD`` shaped ``(1, tile_M, tile_K)`` -- ONE k-tile wide. The compiled signature ties
      ``mD.shape[1]`` to ``mB.shape[0]``, so a wider dump target is refused outright ("Mismatched
      mD.shape[1] ... expected to match mB.shape[0]"). Each k-tile therefore overwrites the last,
      and D ends holding the FINAL one -- the pipeline still cycles every stage on the way there,
      which is what the multi-k-tile cells are for.
    * ``mD`` element type fp32 -- the staged operand is converted on the way out, and fp32 holds
      every fp16/bf16 value exactly, so the comparison stays bitwise.
    """

    def make_epilogue_tma(self, mD, mC, epilogue_args):
        """Hand the kernel the RAW output tensor instead of a TMA descriptor view.

        The production seam returns ``tma_tensor_d``, a descriptor's view of D, and the kernel
        partitions against that. A descriptor view supports TMA copies and **not** elementwise
        load/store -- which is exactly what the dump below needs. Returning ``mD`` itself makes
        ``mD_mnl`` inside the kernel an ordinary global tensor.

        Safe here only because this probe also replaces ``mma_warpgroup_role``, so nothing ever
        reaches the epilogue that would use the (now absent) atom.

        Args:
            mD: The output tensor, returned unwrapped.
            mC: Ignored; the probe never loads a C.
            epilogue_args: Ignored.

        Returns:
            ``(None, mD, None, None)`` -- no atoms, the raw tensor in the descriptor's slot.
        """
        return None, mD, None, None

    @cute.jit
    def mma_warpgroup_role(
        self,
        warp_idx,
        tiled_mma,
        mA_mkl,
        mD_mnl,
        mC_mnl,
        tma_atom_d,
        tma_atom_c,
        epilogue_params,
        tile_sched_params,
        TileSchedulerCls: cutlass.Constexpr[cutlass.Constexpr],
        ab_pipeline,
        epi_pipeline,
        epi_smem_tensors,
        has_C: cutlass.Constexpr[bool],
        has_D: cutlass.Constexpr[bool],
        len_k,
        sA,
        sB,
        sC,
        sD,
        storage,
    ):
        """Consume each staged k-tile and copy ``sA`` out to ``mD_mnl`` instead of doing MMA.

        Follows the real consumer's pipeline protocol exactly -- ``consumer_wait`` before reading
        the stage, ``consumer_release`` after -- because getting that wrong would deadlock or read
        a half-written tile, and either would be mistaken for a load defect.

        Args:
            warp_idx: Warp-uniform index; only warps below ``ab_load_warp_id`` participate.
            mD_mnl: The dump target, ``(tile_M, K, 1)`` as the kernel sees it.
            sA: The staged A tile, ``(tile_M, tile_K, ab_stage)``, swizzled.
            ab_pipeline: The real mainloop pipeline.
            len_k: Contraction extent, giving the k-tile count.
            (the rest): Unused here; present because the signature is the production seam.

        Returns:
            None. Its effect is the dump into ``mD_mnl``.
        """
        if warp_idx < self.ab_load_warp_id:
            tidx, _, _ = cute.arch.thread_idx()
            n_threads = self.mma_warp_groups * self.num_threads_per_warp_group
            tile_M = const_expr(self.cta_tile_shape_mnk[0])
            tile_K = const_expr(self.cta_tile_shape_mnk[2])
            per_tile = const_expr(tile_M * tile_K)
            reps = const_expr((per_tile + n_threads - 1) // n_threads)

            k_tile_cnt = self._k_tile_cnt(len_k)
            ab_read_state = make_pipeline_state(
                cutlass.pipeline.PipelineUserType.Consumer, self.ab_stage
            )
            mD = mD_mnl[(None, None, 0)]
            for k_tile in cutlass.range(k_tile_cnt):
                ab_pipeline.consumer_wait(ab_read_state)
                stage = ab_read_state.index
                sA_stage = sA[(None, None, stage)]
                for r in cutlass.range_constexpr(reps):
                    idx = tidx + r * n_threads
                    if idx < per_tile:
                        m = idx // tile_K
                        k = idx % tile_K
                        mD[m, k] = sA_stage[m, k].to(Float32)
                # A NAMED barrier over the MMA warpgroups only. `cute.arch.barrier()` with no
                # arguments is a full-CTA sync, and the producer warpgroup is in the other branch
                # and never reaches it -- that deadlocks, which is how this was first written.
                cute.arch.barrier(
                    barrier_id=int(NamedBarrierGemm.Epilogue), number_of_threads=n_threads
                )
                ab_pipeline.consumer_release(ab_read_state)
                ab_read_state.advance()


_COMPILED = {}


def _run_load_probe(A, B, D, tile_M, tile_N):
    """Compile (once per config) and launch the load probe over already-built operands.

    Args:
        A: The operand whose staging is under test, ``(1, M, K)``, k-major.
        B: A B operand of the right shape; its values are irrelevant -- it exists so the real
            producer runs its real two-operand load, which is what the pipeline's byte count and
            the mbarrier arrivals are sized for.
        D: The fp32 dump target, ``(1, tile_M, K)``, **written in place**.
        tile_M: CTA tile M. Must be >= A's M, so there is exactly one work tile.
        tile_N: CTA tile N. Must be >= B's N, likewise.

    Returns:
        None.
    """
    a_dt, d_dt = torch2cute_dtype_map[A.dtype], torch2cute_dtype_map[D.dtype]
    A_p, B_p, D_p, _ = perm3d(A, B, D, None)
    key = (a_dt, d_dt, tile_M, tile_N)
    if key not in _COMPILED:
        mA, mB, mD, mC, _, _, _, l = make_fake_gemm_tensors(
            a_dt, a_dt, d_dt, None, "k", "k", "n", None
        )
        _COMPILED[key] = compile_gemm_kernel(
            _LoadProbeGemm,
            a_dt,
            (tile_M, tile_N),
            (1, 1, 1),
            False,
            False,  # NOT persistent -- see the note in the module docstring
            False,
            (9, 0),
            mA,
            mB,
            mD,
            mC,
            (),
            make_fake_scheduler_args(False, False, l),
        )
    clusters = 0  # non-persistent: the grid is one CTA per work tile
    _COMPILED[key](A_p, B_p, D_p, None, (), make_scheduler_args(clusters, 8, None, None))


@requires_sm90
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "M,K,tile_M",
    [
        (128, 64, 128),  # exactly one k-tile
        (128, 256, 128),  # several k-tiles -- the pipeline actually cycles
        (128, 512, 128),  # more k-tiles than stages -- stage reuse
        (64, 128, 64),  # a narrower tile
    ],
)
def test_load_AB_stages_A_bit_exactly(dtype, M, K, tile_M):
    """``load_AB`` stages A into SMEM byte for byte -- no reorder, no drop, no duplicate.

    This is the isolated version of the ``A @ I^T`` read-back: the tile is dumped straight out of
    SMEM without an MMA or an epilogue in between, so a mismatch localises to the load layer.

    Bit-exact rather than approximate, and it must be: staging is **pure transport**, and any
    tolerance at all would accept a swizzle applied twice, a stage index off by one, or a k-tile
    fetched from the wrong offset -- all of which move real data to the wrong place while leaving
    the magnitudes plausible.

    Random integer values are used only because they are easy to read in a failure message; the
    gate holds for any bit pattern, since nothing here does arithmetic.

    **Not covered here: a partial M tile** (M < tile_M). The dump writes a full ``tile_M`` rows
    regardless of M, so what lands in rows ``M..tile_M-1`` is whatever TMA left there, and this
    harness has no model of that -- the gap is in the PROBE, not in the kernel. The assembled
    kernel is tested at M=65 and other off-tile M by
    ``tests/kernels/test_gemm_sm90.py::test_gemm_sm90_is_bit_exact_at_partial_tiles``; teaching
    the probe to predicate its dump would close the localisation gap for that case too.
    """
    g = torch.Generator(device="cuda").manual_seed(5)
    # The widest integer the operand dtype represents exactly -- 2048 at fp16, 256 at bf16. The
    # gate holds for any bit pattern (nothing here does arithmetic); readable integers just make a
    # failure message legible.
    bound = exact_int_bound(dtype) // 2
    A = integer_operands((1, M, K), dtype, "cuda", generator=g, bound=bound)
    B = integer_operands((1, 64, K), dtype, "cuda", generator=g, bound=4)
    tile_K = 64  # 16-bit operands: 4 MMA k-steps of 16
    D = torch.zeros(1, tile_M, tile_K, device="cuda", dtype=torch.float32)

    _run_load_probe(A, B, D, tile_M, tile_K)

    # D holds the LAST staged k-tile, so compare against A's last tile_K columns.
    assert_bitwise(
        D[:, :M, :],
        A[:, :, -tile_K:].float(),
        what=f"last staged A k-tile ({dtype}, M={M} K={K} tile_M={tile_M})",
    )


@requires_sm90
@pytest.mark.parametrize("K", [64, 256])
def test_load_AB_reuses_pipeline_stages_without_tearing(K):
    """Repeating the same launch gives the same staged bytes, every time.

    The mainloop cycles a small ring of SMEM stages, so a missing ``consumer_release`` or an
    off-by-one on the stage index shows up as a tile that is *sometimes* the previous k-tile's --
    a race, not a deterministic error. Ten launches compared against the first turn that from a
    flaky end-to-end failure into a direct one.

    Ten is enough to be useful and cheap; it is not a proof of absence, and is not claimed as one.
    """
    g = torch.Generator(device="cuda").manual_seed(6)
    A = integer_operands((1, 128, K), torch.float16, "cuda", generator=g, bound=1024)
    B = integer_operands((1, 64, K), torch.float16, "cuda", generator=g, bound=4)

    first = torch.zeros(1, 128, 64, device="cuda", dtype=torch.float32)
    _run_load_probe(A, B, first, 128, 64)
    for i in range(9):
        again = torch.zeros(1, 128, 64, device="cuda", dtype=torch.float32)
        _run_load_probe(A, B, again, 128, 64)
        assert_bitwise(again, first, what=f"staged A tile, launch {i + 2} of 10 (K={K})")


@pytest.mark.parametrize("name", ["mainloop_remap_mA", "mainloop_remap_mB"])
def test_the_addressing_seams_default_to_the_identity(name):
    """The overridable A/B remap hooks are pass-throughs on the base class.

    They exist so a derivation can redirect where an operand is read from -- the A2A per-peer base
    row is the case they were built for -- and the base must therefore add **nothing**: not a copy,
    not an offset, not a re-layout. A default that did anything at all would be a per-tile cost
    paid by every plain GEMM to support a kernel that is not in this repo yet.

    Checked by identity of the returned object, which is stronger than equality: a hook that
    rebuilt an identical view would still be doing work.
    """
    sentinel = object()
    fn = getattr(GemmSm90LoadMixin, name)
    assert fn(None, sentinel, (0, 0, 0, 0)) is sentinel, (
        f"{name} must return its input unchanged on the base class"
    )

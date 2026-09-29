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

# Based on the cute-dsl example:
# https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/hopper/dense_gemm.py
# Copyright (c) 2025-2026, Tri Dao.
# Copyright (c) 2025, Wentao Guo, Tri Dao.
# This file has been modified by NVIDIA CORPORATION & AFFILIATES.
"""``out = (LayerNorm(x) @ B + out_bias) * gate3`` in ONE kernel that reads ``x`` exactly once.

The SINGLE-projection sibling of `fold_cp_ops.kernels.layernorm_dual_gated_gemm`: one weight, no
weight pair, no doubled pre-activation, no non-linear gate. The trailing ``gate3`` arrives as a
per-element C operand and is applied in the epilogue, so it costs no extra pass over DRAM. This
covers ops 8, 9 and 12 of the TriMul reference numbering (`docs/cp1_kernel_bringback_complete.md`
§2), and it is the workflow's UNIVERSAL FALLBACK -- the path every feature width that is not a
multiple of 32 routes through.

**The fusion is `alg_fold`, the same algebra the dual's fold uses, applied to one projection.**
Nothing here ever forms ``LayerNorm(x)``. The host folds the LayerNorm gain into the weight, the
mainloop multiplies the RAW activation, and the epilogue repairs the difference rank-one::

    Bw = diag(w_ln) . B        c = colsum(Bw)        d = B^T . b_ln          (host, one launch)
    G  = A @ Bw                                                              (the WGMMA)
    out[m, p] = r_m * G[m, p] - s_m * c[p] + d[p]      s_m = r_m * mu_m      (epilogue, fp32)

with ``(mu_m, r_m)`` the per-row mean and reciprocal standard deviation of the RAW ``A``. Those two
are accumulated off the SAME staged ``sA`` tile the WGMMA is already consuming, which is the whole
point: ``A`` is read from global memory exactly once, and the statistics ride along with the
multiply instead of forcing a second sweep.

**Why the fold and not a prologue normalize.** The prologue variant must see a whole row before it
can scale any of it, so it sweeps ``A`` twice and gives up the mainloop's MMA/load overlap. The fold
gives that up nowhere. The price is conditioning, not speed: ``r*(x@Bw)`` and ``s*c`` are two large
nearly-equal quantities on a row whose mean dwarfs its spread, so the answer is computed as a
cancellation. `fold_cp_ops.testing.numerics.alg_fold_error_bound` models exactly that, and it is
why the tests compare against a bound rather than against the other variant bitwise.

**Where the pieces live.** The reduction arithmetic is NOT written here: `make_row_stat_partials`,
`accumulate_row_stats` and `finalize_row_stats` are free functions in
`fold_cp_ops._internal.ln_prologue`, shared with the dual, and this module reaches them from the two
MMA seams `mma_setup_fragments` / `mma_consume_work_tile`. Two sources of truth for the same sums is
the failure that would make a divergence between the two kernels undetectable.

**What is deliberately NOT brought back**, and would be a new value rather than a signature change:
``fusion_variant="prolog_ln"`` (declared, refused at the front door -- the cp=1 heuristic never
selects it); the ``fuse_stats`` / ``fuse_correction`` timing ablations, whose output is not a
LayerNorm by construction; the dedicated-statistics-warpgroup schedule; and the
operand-A-from-registers variant, whose register row-map is hard-coded for one tile and is silently
wrong at every other, so exposing it as a public flag would ship a wrong-answer path an autotuner
could pick. The statistics schedule is fixed at "issue the WGMMA, then reduce", which is the only
one the size heuristic ever chose.
"""

import operator
from functools import partial
from typing import NamedTuple, Optional

import torch
from torch import Tensor

import cutlass
import cutlass.cute as cute
from cutlass import Float32, const_expr
from cutlass.cute.nvgpu import warpgroup

from fold_cp_ops._internal.arch import (
    check_arch_supported,
    get_max_active_clusters,
    require_sm90,
)
from fold_cp_ops._internal.autotune import AutotuneConfig, AxisSpace, TuneAxis, autotune
import fold_cp_ops._internal.copy_utils as copy_utils
import fold_cp_ops._internal.sm90_utils as fold_cp_ops_sm90_utils
from fold_cp_ops._internal.cache_utils import jit_cache
from fold_cp_ops._internal.compile_time.compile_utils import make_fake_tensor as fake_tensor
from fold_cp_ops._internal.compile_time.cute_dsl_utils import torch2cute_dtype_map
from fold_cp_ops._internal.compile_time.ln_prologue_layout import (
    ALLOWED_BLK_K,
    STATS_THREADS_PER_ROW,
    mma_inst_tile_k,
    stats_scratch_bytes,
    stats_tiled_copy,
)
from fold_cp_ops._internal.epi_default import GemmDefaultEpiMixin
from fold_cp_ops._internal.epi_ops import RowVecLoad, Scalar, SmemColVecBroadcast
from fold_cp_ops._internal.gemm_tvm_ffi_utils import (
    compile_gemm_kernel,
    div_for_dtype,
    get_dtypes,
    get_majors,
    make_fake_scheduler_args,
    make_scheduler_args,
    perm3d,
)
from fold_cp_ops._internal.heuristic_arch import (
    TUNED_ARCH,
    heuristic_arch,
    warn_arch_suboptimal_once,
)
from fold_cp_ops._internal.ln_prologue import (
    RS_QUAD_LANES,
    RS_ROW_STRIDE,
    rs_owned_rows,
    accumulate_row_stats,
    finalize_row_stats,
    make_row_stat_partials,
)
from fold_cp_ops._internal.rounding import RoundingMode
from fold_cp_ops._internal.runtime_params import mlir_namedtuple
from fold_cp_ops.kernels.gemm_sm90 import GemmSm90, GemmSm90Params

__all__ = [
    "FUSION_VARIANTS",
    "LayerNormGemmSm90",
    "build_folded_operands",
    "layernorm_gemm",
    "layernorm_gemm_config_is_valid",
    "layernorm_gemm_freeze",
    "layernorm_gemm_heuristic_config",
    "layernorm_gemm_tuning_space",
    "resolve_blk_k",
]

#: The LayerNorm fusions this kernel's front door NAMES. Only `_IMPLEMENTED_VARIANTS` is buildable;
#: the second is declared so that adding it later is a new value rather than a signature change, and
#: so that the refusal is a tested fact rather than an absence. The names describe BEHAVIOUR --
#: ``alg_fold`` folds the LayerNorm into the weight algebraically, ``prolog_ln`` runs it physically
#: in the prologue -- which is the same vocabulary `layernorm_dual_gated_gemm` uses.
FUSION_VARIANTS = ("alg_fold", "prolog_ln")


def resolve_blk_k(gemm_k: int) -> int:
    """The mainloop K tile for a contraction extent: the widest allowed tile that DIVIDES it.

    Purpose
        The single place this kernel's K tiling is decided. It is an arithmetic decision wearing a
        tiling costume -- the tile sets the k-loop trip count, so it decides how the per-row sums and
        the WGMMA accumulator are ASSOCIATED, and two tilings of the same contraction give the same
        value in exact arithmetic and different bit patterns in floating point.

    Semantics
        Tries :data:`ALLOWED_BLK_K` widest-first (64, 32, 16) and returns the first that divides
        ``gemm_k`` exactly, so the k-loop has no partial tile. When NONE divides it -- every feature
        width that is ``8 mod 16``, which this kernel exists to serve -- it returns the narrowest,
        16, and the LAST k-tile is PARTIAL: the TMA out-of-bounds path zero-fills its
        ``[gemm_k, ceil(gemm_k/16)*16)`` columns of BOTH A and the folded weight, so they contribute
        zero to the contraction AND zero to the row statistics. The normalizer stays the TRUE
        ``gemm_k``, which is what makes the padding exact rather than merely out of range.

        **This is deliberately NOT `ln_prologue_layout.auto_blk_k`, and the difference is the whole
        point.** That helper pads the extent to a multiple of 64 first and therefore always runs the
        64-wide tile; it exists for the dual, whose staged ``(K,)`` gain is a shared-memory
        allocation and so must have a padded extent. This kernel stages no gain -- the fold absorbed
        it -- so it does not pad, and reproduces the narrower tile the kernel it ports from picks.
        Using the padding helper here would tile ``gemm_k=136`` three ways where the reference tiles
        it nine, which is a different summation grouping and therefore a different answer in the last
        bits.

    Args:
        gemm_k: The TRUE contraction extent (the feature width). Must be positive and meet the
            16-byte alignment floor -- ``gemm_k % 8 == 0`` for a 16-bit operand. This does not
            re-check that; the front door does, because an unaligned extent is a TMA fault rather
            than a wrong answer and belongs at the boundary.

    Returns:
        One of :data:`ALLOWED_BLK_K`. 16 is returned both when it divides and when nothing does; the
        two cases differ only in whether the final k-tile is whole.
    """
    return next((c for c in ALLOWED_BLK_K if gemm_k % c == 0), ALLOWED_BLK_K[-1])


class _AlreadyOrderedBarrier:
    """A no-op stand-in for a barrier the SCHEDULE already provides.

    Purpose
        Under ping-pong the kernel runs ``pingpong_barrier_sync(wg, "epi")`` between a warpgroup's
        mainloop and its own epilogue, which already orders the statistics WRITE before the READ.
        `finalize_row_stats` still needs an object exposing ``arrive_and_wait()``, and this is it --
        so the shared helper stays barrier-agnostic instead of learning about ping-pong.

    Semantics
        **Adding a real barrier here would be correct and would cost a rendezvous per work tile**, on
        exactly the path ping-pong exists to keep busy. Worse, the obvious candidate is actively
        wrong: `GemmSm90.epilogue_barrier` is sized for one warpgroup under ping-pong -- which is
        also this reduction's thread count -- so reusing it COMPILES, does not hang, and silently
        rendezvouses two warpgroups that are on different work tiles in different stages.

        Every method is a no-op emitting nothing, so the traced kernel is identical to one with no
        call at all.
    """

    def arrive_and_wait(self) -> None:
        """Do nothing, because the ping-pong epilogue barrier already ordered this write.

        Returns:
            None. Emits no instruction: the caller's ``barrier.arrive_and_wait()`` disappears.
        """
        return None


#: The ONLY tile the register-source (`a_in_regs`) reduction is valid at. Not a tuning choice: the
#: map from a thread to the two rows it owns is a closed form of the WGMMA operand-A fragment for a
#: 128-row tile split across two warpgroups, and it enumerates 0..127 over 256 threads and nothing
#: else. The kernel this reproduces derives the same form for its own BLK_M and does not check it.
_A_IN_REGS_TILE_MN = (128, 128)


class LayerNormGemmParams(GemmSm90Params):
    """`GemmSm90Params` plus what the LayerNorm fold fixes at construction.

    Extending the base pack rather than declaring a second one is what keeps the guarantee total:
    ``_bind_params`` binds exactly ONE pack, so a subclass pack that omitted the base's fields would
    leave them unbound and the functor half-configured.

    Attributes:
        fusion_variant: Which LayerNorm fusion to build -- a member of :data:`FUSION_VARIANTS`.
            Compile-time and immutable. ``"prolog_ln"`` is accepted by the pack and refused by
            ``__init__``, which is the split that lets the selector exist before the second variant
            does.
        blk_k: The mainloop K tile, a member of
            :data:`~fold_cp_ops._internal.compile_time.ln_prologue_layout.ALLOWED_BLK_K`. **A
            numerical parameter, not only a performance one** -- see :func:`resolve_blk_k`. It need
            NOT divide ``gemm_k``: the final k-tile is then partial and the TMA zero-fills it.
        gemm_k: The TRUE contraction extent ``K``, which for this kernel also equals ``N`` and the
            output width ``P``. Baked, because the consumer's k-loop is unrolled off it and because
            it is the LayerNorm normalizer. Passing the padded extent instead would divide every row
            mean by too large a count -- a wrong answer, not a fault.
    """

    fusion_variant: str = "alg_fold"
    blk_k: int = 64
    gemm_k: int = 0


class LayerNormGemmSm90(GemmDefaultEpiMixin, GemmSm90):
    """SM90 read-once ``(LayerNorm(x) @ B + out_bias) * gate3``, fused as the algebraic fold.

    Purpose
        The single-projection member of this package's LayerNorm-fused GEMM family, and the cp=1
        TriMul workflow's universal fallback: every feature width the wider kernels' tiling cannot
        take routes here. One weight, one accumulator, one output.

    Semantics
        Derives from `GemmSm90` and overrides exactly the seams a fused mainloop needs, so the
        scheduler loop, the staging pipeline, the epilogue store and the ping-pong alternation are
        all the base's:

        * `mma_setup_fragments` -- builds the statistics partition and the shared scratch ONCE, and
          parks them in `MmaFragments.extra`.
        * `mma_consume_work_tile` -- calls the base's own k-loop with an ``on_ktile`` closure that
          reduces the staged tile and an ``on_drain`` closure that finalizes the row statistics.
        * `epi_get_smem_tensors` -- hands the two `SmemColVecBroadcast` ops the MAINLOOP's scratch
          instead of an epilogue-owned buffer. That one substitution is the whole mechanism by which
          a value computed in the k-loop is read in the epilogue.
        * `epi_visit_subtile` -- the rank-one repair, the output bias, and the trailing gate.
        * `_extra_smem_struct` / `_compute_stages` -- allocate and RESERVE the scratch, in step.

        **The statistics ride the same staged tile as the WGMMA, and the ORDER is load-bearing.**
        The base's ``on_ktile`` hook fires after the k-tile's WGMMA has been ISSUED and before the
        group wait, so the reduction's shared-memory reads overlap a flying WGMMA rather than
        serializing with the MMA warps' own operand fetch. `accumulate_row_stats` only reads, which
        is exactly that hook's contract.

        **The finalize goes through ``on_drain``, not after the k-loop returns.** It is
        accumulator-independent -- register partials in, shared scratch out -- so it is legal in the
        window between the last WGMMA issue and the drain's ``wait_group(0)``, where it executes in
        the MMA's shadow. Placing it after the loop leaves that wait naked.

        **The reduction tiling keeps every output row inside ONE warp.**
        :data:`STATS_THREADS_PER_ROW` lanes cooperate on a row and divide a warp, so the per-row
        K-reduction is a pure intra-warp butterfly with no cross-warp shared-memory combine, and the
        butterfly is deferred to the finalize so it runs once per row rather than once per k-tile.

    Attributes:
        _IMPLEMENTED_VARIANTS: The subset of :data:`FUSION_VARIANTS` this class can build. Naming it
            separately from the pack's accepted set is what lets `layernorm_gemm` present the axis
            before the second variant exists, and refuse it in a sentence the caller can act on.
        _epi_ops: Five declared terms, in the order that defines both the generated
            ``EpilogueParams`` and the shared-memory map -- an op MOVED is an op handed another op's
            buffer. ``sRstd``/``sScale`` are the two columns of the mainloop's scratch;
            ``mColsum``/``mDbias`` are the fold's ``c`` and ``d``; ``mOutBias`` is the bias on the
            GEMM's OUTPUT, which is a different thing from the LayerNorm's ``beta``.
        _extra_param_fields: ``eps`` travels through ``EpilogueParams`` because that struct is the
            only one that crosses into the kernel region. A mainloop reading it off ``self`` would
            capture a host-side traced value that does not survive the launch.
        _KEEP_STATIC_LEN_K: True. This kernel bakes its K extent, so the PRODUCER's k-loop bound
            folds to a literal instead of becoming a top-tested loop with an early exit. See
            `GemmSm90._KEEP_STATIC_LEN_K` for the measurement.
    """

    Params = LayerNormGemmParams

    _IMPLEMENTED_VARIANTS = ("alg_fold",)

    _epi_ops = (
        SmemColVecBroadcast("sRstd", slot=0),
        SmemColVecBroadcast("sScale", slot=1),
        RowVecLoad("mColsum"),
        RowVecLoad("mDbias"),
        RowVecLoad("mOutBias"),
    )

    _extra_param_fields = (("eps", Float32, Float32(1e-6)),)

    _KEEP_STATIC_LEN_K = True

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        """What the epilogue is handed per launch.

        A NamedTuple is POSITIONAL on the wire, so re-ordering these silently hands each field
        another's value.

        Attributes:
            sRstd, sScale: Placeholders for the two `SmemColVecBroadcast` ops. **Always None.** Their
                data is the mainloop's shared scratch, supplied by :meth:`epi_get_smem_tensors`; the
                fields exist only because the composition machinery expects one per declared op.
            mColsum: ``(l, P)`` fp32 ``c = colsum(Bw)``. **Required** -- without it the repair
                subtracts nothing and the output is the raw ``r*(x@Bw)``, which is finite, plausible
                and entirely wrong.
            mDbias: ``(l, P)`` fp32 ``d = B^T @ b_ln``, or None when there is no LayerNorm bias. None
                compiles the term out; a zero tensor instead would cost a load to add nothing.
            mOutBias: ``(l, P)`` fp32/fp16/bf16 bias on the GEMM's OUTPUT, or None. Added to the fp32
                accumulator AFTER the rank-one repair and BEFORE the gate. **Distinct from the
                LayerNorm bias**, which is folded into ``mDbias`` on the host and sits INSIDE the
                identity being repaired.
            eps: The LayerNorm variance floor, added before the reciprocal square root. A runtime
                value: baking it would key the compiled artifact on a caller-supplied scalar.
            rounding_mode: Compile-time ``RoundingMode``. Only ``RN`` is reachable on SM90.
            sr_seed: Stochastic-rounding seed. Unused under ``RN``; kept so the argument shape
                matches every other epilogue in the package.
        """

        sRstd: Optional[cute.Tensor] = None
        sScale: Optional[cute.Tensor] = None
        mColsum: Optional[cute.Tensor] = None
        mDbias: Optional[cute.Tensor] = None
        mOutBias: Optional[cute.Tensor] = None
        eps: Float32 = Float32(1e-6)
        rounding_mode: cutlass.Constexpr[int] = RoundingMode.RN
        sr_seed: None = None

    def __init__(
        self,
        *args,
        fusion_variant: str = "alg_fold",
        blk_k: int = 64,
        gemm_k: int = 0,
        a_in_regs: bool = False,
        rs_ring: bool = False,
        **kwargs,
    ):
        """Bind the fold's three parameters on top of `GemmSm90`'s, refusing what cannot be built.

        Args:
            *args: `GemmSm90.__init__`'s positional arguments -- ``acc_dtype``, ``a_dtype``,
                ``tile_shape_mn``, ``cluster_shape_mnk``.
            fusion_variant: A member of :data:`FUSION_VARIANTS`. Must additionally be in
                :attr:`_IMPLEMENTED_VARIANTS`.
            blk_k: The mainloop K tile; a member of :data:`ALLOWED_BLK_K`. Normally
                :func:`resolve_blk_k`'s answer. A value outside the allowed set would build a WGMMA
                K-block count the atom cannot feed.
            gemm_k: The TRUE contraction extent. Must be positive; it is the LayerNorm normalizer and
                the consumer's unrolled trip count, so a zero here divides every mean by zero.
            **kwargs: `GemmSm90.__init__`'s keyword arguments (``pingpong``, ``is_persistent``, ...).

        Returns:
            None.

        Raises:
            ValueError: On an unknown or unimplemented ``fusion_variant``, a ``blk_k`` outside
                :data:`ALLOWED_BLK_K`, or a non-positive ``gemm_k``. ``raise`` rather than ``assert``
                because ``python -O`` strips asserts, and a stripped check here builds a kernel that
                normalizes with a meaningless count instead of refusing.
        """
        if fusion_variant not in FUSION_VARIANTS:
            raise ValueError(
                f"fusion_variant must be one of {FUSION_VARIANTS}; got {fusion_variant!r}"
            )
        if fusion_variant not in self._IMPLEMENTED_VARIANTS:
            raise ValueError(
                f"fusion_variant={fusion_variant!r} is not brought back on "
                f"{type(self).__name__}: the cp=1 TriMul heuristic never selects it, so only "
                f"{self._IMPLEMENTED_VARIANTS} is implemented"
            )
        if blk_k not in ALLOWED_BLK_K:
            raise ValueError(f"blk_k must be one of {ALLOWED_BLK_K}; got {blk_k}")
        if gemm_k <= 0:
            raise ValueError(
                f"gemm_k must be the TRUE positive contraction extent; got {gemm_k}. It is the "
                "LayerNorm normalizer and the consumer's unrolled trip count"
            )
        # ── the register-source mainloop's region, refused at the FRONT DOOR ──────────────────
        # `a_in_regs` reduces the per-row statistics out of the WGMMA operand-A REGISTER fragment
        # instead of re-reading the staged tile, and the map from a thread to the two rows it owns
        # is a CLOSED FORM of that fragment for a 128-row tile split across two warpgroups:
        #
        #     base = (tidx // 128) * 64 + ((tidx % 128) // 32) * 16 + ((tidx % 32) // 4)
        #
        # which enumerates 0..127 over 256 threads and NOTHING else. At tile_M=256 a thread owns
        # four rows and that form addresses only the first 96; at 64 and 192 the warpgroup split
        # changes underneath it. The kernel this reproduces derives the same form for its own
        # BLK_M=128 and does NOT check it -- its own comment calls the config "silent-WRONG outside
        # (128,128)" and keeps it out of its heuristic, which is a note in a source file rather
        # than a refusal. Here it is a refusal: a silently-wrong config is worse than an absent one,
        # and an autotuner that can reach it will.
        if rs_ring and not a_in_regs:
            raise ValueError(
                "rs_ring double-buffers the operand-A REGISTER fragment, which only exists on the "
                "a_in_regs path; with a_in_regs=False there is nothing to ring and the flag would "
                "silently select the same kernel under a different cache key."
            )
        super().__init__(*args, fusion_variant=fusion_variant, blk_k=blk_k, gemm_k=gemm_k, **kwargs)
        # AFTER the base, so the tile is read from the resolved attribute rather than from a
        # positional slot -- `tile_shape_mn` may arrive positionally or by keyword, and indexing
        # `args` would refuse the wrong thing (or nothing) depending on how the caller spelled it.
        if a_in_regs:
            if tuple(self.tile_shape_mn) != _A_IN_REGS_TILE_MN:
                raise ValueError(
                    f"a_in_regs requires tile_shape_mn == {_A_IN_REGS_TILE_MN}; got "
                    f"{tuple(self.tile_shape_mn)}. The register-source reduction's row map is a "
                    f"closed form of the operand-A fragment for a 128-row tile across two "
                    f"warpgroups; at any other tile it addresses the wrong rows and answers "
                    f"plausibly and WRONGLY."
                )
            if self.pingpong:
                raise ValueError(
                    "a_in_regs is cooperative-only: the row map assumes 256 threads spanning the "
                    "whole tile, and ping-pong gives each warpgroup its own tile and 128 threads."
                )
        self.a_in_regs = a_in_regs
        self.rs_ring = rs_ring

    @property
    def _MMA_INST_TILE_K(self) -> int:
        """WGMMA K-instructions per mainloop k-tile, so ``cta_tile_k`` becomes the declared ``blk_k``.

        The base derives ``cta_tile_k = atom_K * _MMA_INST_TILE_K`` from a class constant of 4,
        giving the plain GEMM's 64. Making it a property of ``blk_k`` is how the K tile becomes a
        declared parameter without touching the atom: the MMA stays ``m64nNk16`` and only the number
        of K-blocks per tile -- and therefore the staged shared-memory box -- changes.

        Returns:
            ``blk_k // 16``: 4, 2 or 1.
        """
        return mma_inst_tile_k(self.blk_k)

    def _k_tile_cnt_const(self) -> int:
        """The consumer's k-loop trip count, as a true Python int so the loop UNROLLS.

        Purpose
            Handed to `GemmSm90.mma` as ``k_tile_cnt_const``. A dynamic ``cutlass.range`` is a
            separate MLIR region, so the per-row partials this mainloop carries ACROSS k-tiles could
            not stay straight-line SSA -- they would be carried through the region and spill.

        Semantics
            **CEIL, not floor.** The producer stages ``ceil_div(gemm_k, blk_k)`` tiles, so the
            consumer must drain them ALL, including the partial last one that a ``gemm_k`` no
            allowed tile divides produces. Draining fewer than the producer stages is a DEADLOCK,
            not a wrong answer. When ``blk_k`` divides ``gemm_k`` the ceiling equals the floor.

        Returns:
            ``ceil(gemm_k / blk_k)``.
        """
        return (self.gemm_k + self.blk_k - 1) // self.blk_k

    def _stats_num_threads(self) -> int:
        """Threads cooperating on the per-row reduction: every MMA warpgroup, or one under ping-pong.

        Cooperatively the two warpgroups share one output tile, so they reduce it jointly and publish
        through the epilogue barrier, which is sized for exactly that group. Under ping-pong they own
        DIFFERENT output tiles concurrently, so each reduces its own with its own 128 threads.

        Returns:
            ``128`` under ping-pong, else ``mma_warp_groups * 128``.
        """
        groups = 1 if self.pingpong else self.mma_warp_groups
        return groups * self.num_threads_per_warp_group

    @property
    def _stats_slots(self) -> int:
        """Statistics buffers in the scratch: one per MMA warpgroup under ping-pong, else one.

        Read by the ALLOCATION (:meth:`_extra_smem_struct`) and by the epilogue's view of it. Those
        two disagreeing does not fail at compile -- it overruns the dynamic shared-memory cap at
        LAUNCH, as ``cudaErrorInvalidValue``, which names shared memory nowhere.

        Returns:
            ``mma_warp_groups`` under ping-pong (2), else 1.
        """
        return self.mma_warp_groups if self.pingpong else 1

    def _stats_scratch_all(self, storage):
        """The whole ``(tile_M, 2, slots)`` statistics block, the warpgroup mode OUTERMOST.

        Purpose
            What the EPILOGUE takes. `SmemColVecBroadcast.begin` slices both the column and the
            warpgroup itself -- from a fresh ``warp_idx()`` read taken inside the epilogue region,
            which is what avoids carrying a warpgroup index across it -- so it needs the block whole.

        Semantics
            The buffer index is the outermost mode so each warpgroup's ``(tile_M, 2)`` slice keeps
            ``tile_M`` at UNIT STRIDE, which is what that op's stride-``(1, 0)`` broadcast requires.

        Args:
            storage: The kernel's shared storage. ``storage.decoupled.s_stats`` must be the block
                :meth:`_extra_smem_struct` reserved; a different layout here is read as if it were
                that one, giving a plausible wrong statistic rather than a fault.

        Returns:
            A ``(tile_M, 2, slots)`` fp32 shared tensor.
        """
        return storage.decoupled.s_stats.get_tensor(
            cute.make_layout((self.tile_shape_mn[0], 2, self._stats_slots))
        )

    def _stats_scratch(self, storage, warp_group_idx):
        """This warpgroup's ``(tile_M, 2)`` slice -- what the MAINLOOP writes.

        The mainloop already holds a warp-uniform ``warp_group_idx`` from its role, so it slices
        directly; the epilogue cannot (see :meth:`_stats_scratch_all`). Both go through the same
        block, which is what stops the two from disagreeing about where a warpgroup's statistics live.

        Args:
            storage: The kernel's shared storage.
            warp_group_idx: This MMA warpgroup's index, warp-uniform. Ignored cooperatively, where
                there is one buffer.

        Returns:
            A ``(tile_M, 2)`` fp32 shared tensor: column 0 receives ``rstd``, column 1 ``rstd*mu``.
        """
        block = self._stats_scratch_all(storage)
        return block[None, None, warp_group_idx if self.pingpong else 0]

    def _stats_barrier(self, warp_group_idx):
        """The barrier the row reduction publishes through, for THIS warpgroup.

        Cooperatively this is the epilogue barrier, which spans exactly the reducing threads and is
        also what the epilogue rendezvouses on -- that shared identity is what orders the NEXT work
        tile's writes after this tile's reads, so `finalize_row_stats` needs no leading barrier.

        **Under ping-pong the epilogue barrier must NOT be reused, and the trap is that its size
        would FIT.** Ping-pong sizes it for one warpgroup, exactly this reduction's count, so reusing
        it compiles, does not hang, and silently rendezvouses two warpgroups that are on different
        work tiles. `_AlreadyOrderedBarrier` says what is true instead: the schedule already ordered
        the write against the read.

        Args:
            warp_group_idx: This MMA warpgroup's index. Warp-uniform; a non-uniform value would split
                a warpgroup across two barriers and hang. Unused in both branches today, taken so the
                signature does not change when a schedule needs it.

        Returns:
            An object exposing ``arrive_and_wait()``.
        """
        if not self.pingpong:
            return self.epilogue_barrier
        return _AlreadyOrderedBarrier()

    def _extra_smem_struct(self):
        """The fold's shared memory: the per-CTA row-statistics scratch, and nothing else.

        Purpose
            Splices one buffer into the kernel's ``SharedStorage`` through the base's single hook, so
            the fusion needs no change to `GemmSm90.__call__`.

        Semantics
            **No staged LayerNorm gain or bias.** The fold absorbed both into the weight on the host,
            so this kernel never reads them; allocating them would be a barrier and two copies feeding
            buffers nothing loads, out of the very budget :meth:`_compute_stages` divides into
            pipeline stages.

            The field lands between the epilogue's own struct and the 1024-byte-aligned ``sA``, which
            is exactly where the kernel this ports from puts it -- there the scratch is an epilogue
            op's allocation and every other declared op allocates nothing, so the two layouts
            coincide byte for byte.

        Returns:
            A ``cute.struct`` type with one ``s_stats`` member, reached in the kernel as
            ``storage.decoupled.s_stats``.
        """
        fields = {
            "s_stats": cute.struct.Align[
                cute.struct.MemRange[Float32, self.tile_shape_mn[0] * 2 * self._stats_slots], 16
            ],
        }

        class LnFoldStorage:
            __annotations__ = fields

        return cute.struct(LnFoldStorage)

    def _compute_stages(self, *args, **kwargs):
        """Reserve the statistics scratch before the base sizes the mainloop pipeline.

        Purpose
            The pipeline depth is whatever fits in the shared memory left over, so a buffer allocated
            out of the same budget must be subtracted BEFORE the base divides.

        Semantics
            An instance method overriding a ``classmethod``, because the reservation depends on the
            tile, which is an instance parameter. ``args[7]`` is the shared-memory capacity, matching
            the base's positional signature.

            **The reservation counts every SLOT and is rounded UP to the buffer alignment, and both
            halves were paid for.** The scratch holds one block per MMA warpgroup under ping-pong, so
            reserving one block's worth there under-budgets by a block; and the field is spliced
            AHEAD of ``sD``/``sA``/``sB``, which are ``Align[..., 1024]``, so inserting ``r`` bytes
            displaces those by up to ``1024 - (r % 1024)`` bytes of fresh padding on top of ``r``.

            **The kernel this ports from reserves ONE slot's worth, unrounded, and that is a defect
            reachable from its own autotuner.** Measured at ``tile=(192, 128)`` with ping-pong:
            allocation 3072 bytes, reservation 1536, and the assembled storage struct comes to
            233472 bytes against a 232448-byte cap -- over by exactly 1024. The symptom is a LAUNCH
            failure, ``cudaErrorInvalidValue``, with no compile error and no mention of shared
            memory; the same pair is in that kernel's own sweep grid, so an autotuner reaches it.
            ``(192, 128)`` cooperatively and ``(128, 128)`` under ping-pong both survive it, the
            latter by landing at exactly 232448 bytes.

            **The fix costs nothing anywhere else, and that was measured rather than hoped.** Across
            the declared sweep -- ``(64|128|192|256, 128)`` and ``(128, 64|208|256)``, both schedules
            -- the two reservations produce the SAME struct size, the same ``ab_stage`` and the same
            ``epi_stage`` at every tile but the one that overflows, because every other reserved
            figure already rounds to the same multiple of the alignment. That single cell is the
            whole divergence from the reference kernel, and it is a cell the reference cannot launch.

        Args:
            *args: The base's positional arguments ``(cta_tile, epi_tile, a, b, d, c, epi_args,
                smem_capacity, occupancy)``; ``args[7]`` is the capacity in bytes.
            **kwargs: Forwarded unchanged.

        Returns:
            ``(ab_stage, epi_stage, epi_c_stage)``.
        """
        args = list(args)
        align = self.buffer_align_bytes
        # The slot count MUST match `_extra_smem_struct`'s -- both read `_stats_slots`, which is what
        # keeps them from disagreeing, and a disagreement here is a LAUNCH failure, not a compile
        # error.
        reserve = stats_scratch_bytes(self.tile_shape_mn[0]) * self._stats_slots
        args[7] = args[7] - (reserve + align - 1) // align * align
        return super()._compute_stages(*args, **kwargs)

    def epi_to_underlying_arguments(self, args, *, loc=None, ip=None):
        """Lower the launch-time epilogue arguments, threading ``eps`` through explicitly.

        The per-op tensors go through the shared helper; ``eps`` is added by hand because it is a
        declared extra param rather than an op's field, and because reading it off ``self`` in the
        kernel region would capture a host-traced value.

        Args:
            args: This kernel's :class:`EpilogueArguments`.
            loc: Optional DSL source location, forwarded by the framework.
            ip: Optional DSL insertion point, forwarded by the framework.

        Returns:
            An ``EpilogueParams`` carrying one entry per declared op plus ``eps``.
        """
        d = self._epi_ops_to_params_dict(args)
        d["eps"] = args.eps
        return self.EpilogueParams(**d)

    def epi_get_smem_tensors(self, params, storage):
        """Hand the two statistics ops the MAINLOOP's scratch instead of an epilogue buffer.

        Purpose
            The one seam that lets a value computed in the k-loop be read in the epilogue. Every
            other declared op either owns a buffer in ``storage.epi`` or loads straight to registers;
            these two read ``storage.decoupled.s_stats``, which the mainloop wrote.

        Semantics
            Calls the base for every op, then substitutes the `SmemColVecBroadcast` entries BY
            POSITION, walking the same ``_epi_ops`` tuple the base iterates. A hard-coded index would
            silently shift the moment an op is appended.

            The FULL ``(tile_M, 2, slots)`` block is passed, not a per-warpgroup slice: the op picks
            its warpgroup itself, inside the epilogue region.

        Args:
            params: The traced ``EpilogueParams``.
            storage: The kernel's shared storage.

        Returns:
            One tensor per non-`Scalar` op, in declaration order -- the base's tuple with the two
            statistics entries replaced.
        """
        tensors = list(super().epi_get_smem_tensors(params, storage))
        stats = self._stats_scratch_all(storage)
        non_scalar = [op for op in self._epi_ops if not isinstance(op, Scalar)]
        for i, op in enumerate(non_scalar):
            if isinstance(op, SmemColVecBroadcast):
                tensors[i] = stats
        return tuple(tensors)

    def _make_rs_tiled_mma(self) -> cute.TiledMma:
        """Rebuild the MMA atom with operand A sourced from REGISTERS instead of shared memory.

        Purpose
            The one construction the register-source mainloop needs that the shared-memory one does
            not. Everything downstream -- the fragment shapes, the s2r copy's destination, the
            reduction's view -- is derived from this atom, so building it wrong is not a local error.

        Semantics
            Identical to the base atom except for ``a_source``. **Forked rather than shared with the
            two-A fusion's namesake**: it reads seven attributes off ``self``, so sharing it would
            mean either passing seven arguments or inheriting class state across a sweep-count
            boundary -- and the latter is the pairing this codebase has already recorded as
            deadlocking while it compiles cleanly.

        Returns:
            The register-source `cute.TiledMma`.

        Raises:
            AssertionError: If N is split across warpgroups (``atom_layout_mnk[1] != 1``). The
                register A-fragment must span the whole ``BLK_K`` for one warpgroup, which an
                N-split atom does not give it.
        """
        assert self.atom_layout_mnk[1] == 1, (
            f"the register-source mainloop needs atom_layout_n == 1; got {self.atom_layout_mnk}. "
            f"The rs A-fragment must span the whole BLK_K for one warpgroup."
        )
        return fold_cp_ops_sm90_utils.sm90_utils_og.make_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_layout.sm90_mma_major_mode(),
            self.b_layout.sm90_mma_major_mode(),
            self.acc_dtype,
            self.atom_layout_mnk,
            tiler_mn=(64, self.tile_shape_mn[1] // self.atom_layout_mnk[1]),
            a_source=warpgroup.OperandSource.RMEM,
        )

    def _make_a_s2r_copy(self, tiled_mma):
        """The shared-memory -> register copy that fills the register-source A fragment.

        Purpose
            The register path reads the staged tile ONCE, through this copy, and then both the
            WGMMA and the statistics reduction consume the registers. That single read is the whole
            point of the path for an MN-major activation, whose shared-memory tile is strided along
            K and therefore expensive to sweep twice.

        Semantics
            The atom is selected by the A operand's MAJOR mode, which decides whether the load must
            transpose: an MN-major A takes the transposing `ldmatrix`, which is conflict-free for
            this layout and delivers K-ordered registers. Passing the plain shared-memory atom
            instead builds a copy whose destination does not match the fragment -- a wrong answer,
            not a fault.

        Args:
            tiled_mma: The REGISTER-source atom from :meth:`_make_rs_tiled_mma`. Passing the
                shared-memory atom silently produces the wrong destination layout.

        Returns:
            A `cute.TiledCopy` whose destination is the atom's A fragment.
        """
        layout_a = (
            cutlass.utils.LayoutEnum.COL_MAJOR
            if tiled_mma.op.a_major_mode == warpgroup.OperandMajorMode.MN
            else cutlass.utils.LayoutEnum.ROW_MAJOR
        )
        return cute.make_tiled_copy_A(
            copy_utils.sm90_get_smem_load_op(layout_a, self.a_dtype), tiled_mma
        )

    @cute.jit
    def _load_a_to_rf(self, a_s2r_copy, a_tidx, sA_stage, tCrA):
        """Fill this k-tile's register A fragment from the staged shared-memory tile.

        Purpose
            The single read the register path is built around.

        Semantics
            Forked rather than shared even though it sequences nothing: its failure mode is a LOUD
            shape mismatch at trace time, so a drifting copy announces itself. That is the opposite
            of `rs_owned_rows` and `rs_accumulate_row_stats`, whose layout arithmetic fails
            silently and is therefore shared.

        Args:
            a_s2r_copy: The tiled copy from :meth:`_make_a_s2r_copy`.
            a_tidx: This thread's GLOBAL index across both warpgroups.
            sA_stage: This pipeline stage's staged A tile. Read only.
            tCrA: The register fragment, filled IN PLACE.

        Returns:
            None; `tCrA` is filled.
        """
        thr_copy = a_s2r_copy.get_slice(a_tidx)
        cute.copy(thr_copy, thr_copy.partition_S(sA_stage), a_s2r_copy.retile(tCrA))

    @cute.jit
    def _finalize_stats_rf(self, stats_smem, a_tidx, row_sum, row_sqsum, n_elems, eps):
        """Complete the register partials into ``(rstd, rstd*mu)`` in the shared statistics scratch.

        Purpose
            The register path's finalize. It is the counterpart of `finalize_row_stats` for a
            reduction whose partials live per-quad-lane rather than per-row-group.

        Semantics
            The four lanes of a quad hold DISJOINT K of the same two rows, so each row's K-sum is
            completed by a `RS_QUAD_LANES`-wide warp reduction and written once, by the owner lane.
            The rows themselves come from the SHARED `rs_owned_rows` -- the map is empirically
            derived and a second copy that drifted would mis-attribute statistics with no error.

            **Publishes ``(rstd, rstd*mu)``, not ``(mu, rstd)``**, because this kernel is the
            algebraic fold: its epilogue repairs the accumulator with ``r*acc - s*c``, so forming
            the product here costs one multiply per ROW instead of one per output ELEMENT.

            **Forked** rather than shared: it ends on ``epilogue_barrier.arrive_and_wait()``, so it
            sequences an ARRIVE and is exactly what the share/fork rule keeps out of shared code.

        Args:
            stats_smem: This CTA's ``(tile_M, 2)`` fp32 scratch. Column 0 receives ``rstd``,
                column 1 ``rstd*mu``; nothing checks that the reader agrees.
            a_tidx: This thread's GLOBAL index, 0..255. A warpgroup-local index maps both
                warpgroups onto rows 0..63 and leaves half the tile unwritten.
            row_sum: This thread's fp32 ``Sum x`` partials, one per owned row.
            row_sqsum: Likewise for ``Sum x^2``.
            n_elems: The TRUE per-row feature width, not the padded one.
            eps: The variance floor.

        Returns:
            None. Every thread of the cooperating group must reach this call -- it ends on a
            collective barrier, and partial participation hangs.
        """
        inv_n = 1.0 / Float32(n_elems)
        base, is_owner = rs_owned_rows(a_tidx)
        for m in cutlass.range_constexpr(cute.size(row_sum)):
            s_sum = cute.arch.warp_reduction(
                row_sum[m], operator.add, threads_in_group=RS_QUAD_LANES
            )
            s_sqsum = cute.arch.warp_reduction(
                row_sqsum[m], operator.add, threads_in_group=RS_QUAD_LANES
            )
            mean = s_sum * inv_n
            var = s_sqsum * inv_n - mean * mean
            rstd = cute.math.rsqrt(var + eps, fastmath=True)
            if is_owner:
                stats_smem[base + m * RS_ROW_STRIDE, 0] = rstd
                stats_smem[base + m * RS_ROW_STRIDE, 1] = rstd * mean
        cute.arch.fence_view_async_shared()
        self.epilogue_barrier.arrive_and_wait()

    @cute.jit
    def _rs_gemm(self, mma_atom, acc, rA_k, rB_k, zero_init):
        """Issue ONE k-tile's register-source WGMMA.

        Semantics
            Operand A comes from the register fragment, B from its shared-memory descriptor. A
            FRESH ``mma_atom.set`` per call is not redundant: without it the accumulate flag's
            definition does not dominate its use and the DSL rejects the region.

            **Forked**: `warpgroup.fence()` and `commit_group()` sequence MMA groups, which is the
            other half of the share/fork rule.

        Args:
            mma_atom: The atom from the register-source tiled MMA.
            acc: The accumulator, updated in place.
            rA_k: This stage's register A fragment.
            rB_k: This stage's shared-memory B descriptor slice.
            zero_init: True on the FIRST k-tile only; it selects overwrite rather than accumulate.

        Returns:
            None.
        """
        warpgroup.fence()
        mma_atom.set(warpgroup.Field.ACCUMULATE, not zero_init)
        for k in cutlass.range_constexpr(cute.size(rA_k.shape[2])):
            cute.gemm(mma_atom, acc, rA_k[None, None, k], rB_k[None, None, k], acc)
            mma_atom.set(warpgroup.Field.ACCUMULATE, True)
        warpgroup.commit_group()

    def mma_setup_fragments(
        self, tiled_mma, sA, sB, warp_group_thread_layout, warp_group_idx, mma_ctx=None
    ):
        """Build the base fragments, then the reduction's partition and this warpgroup's scratch.

        Purpose
            Everything here is per-CTA and tile-INDEPENDENT, so it must happen once, outside the
            work-tile loop. The partition in particular is pure layout algebra that the compiler
            folds away; re-deriving it per tile would emit it again for nothing.

        Semantics
            A plain ``def`` overriding a plain ``def``, as the base requires: ``@cute.jit`` flattens
            arguments into MLIR values and cannot accept or return an `MmaFragments`. The consequence
            is that the DSL preprocessor does not run here, so this body contains no dynamic ``if``
            and no ``cutlass.range_constexpr``.

            Unlike the prologue-normalize fusion this stages NOTHING and therefore ends at NO
            barrier, so it imposes no arrival requirement on its callers.

        Args:
            tiled_mma: The MMA atom, sliced by warpgroup.
            sA: The staged A tile ``(tile_M, blk_k, ab_stage)``; sets ``tCrA``'s layout and is kept
                for the mainloop to index a stage without re-deriving the view.
            sB: The staged B tile.
            warp_group_thread_layout: Maps a warpgroup index to its first thread.
            warp_group_idx: Which MMA warpgroup this is; selects the scratch slot under ping-pong.
            mma_ctx: The `MmaContext`. **Required here** despite the base's default: it is the only
                route to the shared scratch (through ``storage``), to ``eps`` (through
                ``epilogue_params``) and to this thread's index. Passing None raises rather than
                silently building a functor that repairs with uninitialised statistics.

        Returns:
            An `MmaFragments` whose ``extra`` is the ``(thr_copy, sA, stats_smem, tidx, eps)`` tuple
            the mainloop unpacks.

        Raises:
            ValueError: If ``mma_ctx`` is None.
        """
        if mma_ctx is None:
            raise ValueError(
                "LayerNormGemmSm90.mma_setup_fragments requires an MmaContext: the row-statistics "
                "scratch and eps are reachable only through it. A direct caller (a test driving the "
                "seam without a role) must build one."
            )
        frags = super().mma_setup_fragments(
            tiled_mma, sA, sB, warp_group_thread_layout, warp_group_idx, mma_ctx
        )
        tiled_copy = stats_tiled_copy(
            self.a_dtype,
            (self.tile_shape_mn[0], self.cta_tile_shape_mnk[2]),
            self._stats_num_threads(),
            STATS_THREADS_PER_ROW,
        )
        # `tidx` arrives warpgroup-local under ping-pong (the role reduces it), which is exactly what
        # the row ownership inside `finalize_row_stats` assumes, so nothing is adjusted here.
        frags.extra = (
            tiled_copy.get_slice(mma_ctx.tidx),
            sA,
            self._stats_scratch(mma_ctx.storage, warp_group_idx),
            mma_ctx.tidx,
            mma_ctx.epilogue_params.eps,
        )
        return frags

    def mma_consume_work_tile(
        self, frags, ab_pipeline, ab_read_state, len_k, warp_group_idx, mma_carry=None
    ):
        """Drain one work tile in a SINGLE pass, reducing and multiplying the same staged tile.

        Purpose
            The mainloop side of the fusion. The prologue-normalize variant needs two sweeps because
            it must see a whole row before it can scale any of it; this one scales nothing in the
            mainloop, so the statistics ride along with the multiply.

        Semantics
            Delegates to the base's own consumer loop with two closures. ``on_ktile`` fires once per
            k-tile with the pipeline stage, AFTER the WGMMA is issued and BEFORE the group wait, so
            the reduction's reads overlap the flying WGMMA; `accumulate_row_stats` only reads, which
            is that hook's contract. ``on_drain`` fires between the last WGMMA issue and the drain's
            ``wait_group(0)``, where it runs in the MMA's shadow.

            The partials are register tensors updated IN PLACE, so the closures ignore the return
            value and the base's k-loop needs no loop-carried variable for them.

            **The trip count is handed over as a compile-time int**, which unrolls the k-loop -- that
            is what keeps the partials straight-line SSA instead of state carried across a dynamic
            MLIR region.

            **``(rstd, rstd*mu)`` is published rather than ``(mu, rstd)``**, and forming the product
            here is a performance decision, not a formatting one: it costs one multiply per ROW,
            where forming it in the epilogue costs one per output ELEMENT.

            A plain ``def``, not ``@cute.jit``: the base requires it, since the DSL cannot carry an
            `MmaFragments` across the boundary.

        Args:
            frags: The `MmaFragments` from :meth:`mma_setup_fragments`.
            ab_pipeline: The mainloop staging pipeline.
            ab_read_state: The consumer state, threaded across work tiles. Must be what the previous
                call returned; a stale one reads an already-released stage.
            len_k: The contraction extent, converted by ``_k_tile_cnt`` -- the SAME count the
                producer stages, and equal to :meth:`_k_tile_cnt_const` by construction. A mismatch
                is a deadlock, not a wrong answer.
            warp_group_idx: Which MMA warpgroup, forwarded to the base for the ping-pong barriers and
                used to select this warpgroup's publication barrier.
            mma_carry: The previous call's carry; always None here, because the statistics are
                recomputed per work tile from that tile's own rows.

        Returns:
            ``(ab_read_state, mma_carry)``.
        """
        thr_copy, sA, stats_smem, tidx, eps = frags.extra
        row_sum, row_sqsum = make_row_stat_partials(thr_copy, sA[None, None, 0])

        def on_ktile(stage):
            accumulate_row_stats(thr_copy, sA[None, None, stage], row_sum, row_sqsum)

        def on_drain():
            finalize_row_stats(
                stats_smem,
                tidx,
                row_sum,
                row_sqsum,
                self.gemm_k,
                eps,
                STATS_THREADS_PER_ROW,
                self._stats_num_threads(),
                self._stats_barrier(warp_group_idx),
                publish_fold_scale=True,
            )

        ab_read_state = self.mma(
            ab_pipeline,
            ab_read_state,
            frags.mma_fn,
            frags.acc,
            frags.acc_slow,
            self._k_tile_cnt(len_k),
            warp_group_idx,
            on_ktile=on_ktile,
            k_tile_cnt_const=self._k_tile_cnt_const(),
            on_drain=on_drain,
        )
        return ab_read_state, mma_carry

    @cute.jit
    def epi_visit_subtile(
        self,
        params,
        epi_loop_tensors,
        tRS_rD: cute.Tensor,
        tRS_rC: Optional[cute.Tensor] = None,
        epi_gate3: cutlass.Constexpr = False,
    ) -> Optional[cute.Tensor]:
        """Repair one accumulator subtile rank-one, add the output bias, then apply the gate.

        Purpose
            The epilogue half of the algebraic fold, and the only arithmetic this class adds. In
            place and in fp32, on the accumulator holding the RAW ``x @ Bw``.

        Semantics
            Three steps, and the ORDER is the contract, not a preference::

                acc <- r*acc - s*c + d        the fold's identity: this IS LayerNorm(x) @ B
                acc <- acc + out_bias         a bias on the FINISHED projection
                acc <- acc * gate3            the TriMul trailing output gate

            ``c`` and ``d`` belong to the fold and sit INSIDE the identity being repaired, so they go
            first. ``out_bias`` is outside it: adding it before the ``r*`` scaling would bias every
            row by a per-row amount no reference computes. The gate multiplies the finished value,
            which is what makes it a Hadamard product of the projection and the gate rather than of
            the accumulator and the gate.

            Every term is ``const_expr``-guarded on its presence, so a kernel built without a
            LayerNorm bias, without an output bias or without a gate emits no instructions for it --
            the epilogue does not add zero or multiply by one.

            **The statistics are the MAINLOOP's**, so this is valid only after `finalize_row_stats`
            has published them behind its barrier. Nothing here checks that.

        Args:
            params: This kernel's ``EpilogueParams``. Unused -- every value arrives through
                `epi_loop_tensors`, which is what keeps this call site independent of which buffer a
                term was loaded from.
            epi_loop_tensors: This subtile's loaded terms, keyed by op name: ``sRstd`` (``r``, a
                column vector broadcast along N), ``sScale`` (``s = r*mu``), ``mColsum`` (``c``),
                ``mDbias`` (``d`` or None) and ``mOutBias`` (or None).
            tRS_rD: The accumulator fragment holding ``x @ Bw``, **overwritten in place** with the
                finished output.
            tRS_rC: The per-element ``gate3`` fragment for this subtile, or None when the kernel was
                built without a gate.
            epi_gate3: Whether this is a fused output-gate PASS. Always False here -- this kernel has
                one output region, and the gate arrives as a C operand rather than as a second
                region. Accepted because the caller passes it.

        Returns:
            None. This epilogue has no second post-activation output; the store path reads
            `tRS_rD`.
        """
        r = epi_loop_tensors["sRstd"]
        s = epi_loop_tensors["sScale"]
        c = epi_loop_tensors["mColsum"]
        d = epi_loop_tensors["mDbias"]
        ob = epi_loop_tensors["mOutBias"]
        for i in cutlass.range(cute.size(tRS_rD), unroll_full=True):
            tRS_rD[i] = r[i] * tRS_rD[i] - s[i] * c[i]
            if const_expr(d is not None):
                tRS_rD[i] = tRS_rD[i] + d[i]
        if const_expr(ob is not None):
            for i in cutlass.range(cute.size(tRS_rD), unroll_full=True):
                tRS_rD[i] = tRS_rD[i] + ob[i]
        if const_expr(tRS_rC is not None):
            rC = tRS_rC.load().to(tRS_rD.element_type)
            for i in cutlass.range(cute.size(tRS_rD), unroll_full=True):
                tRS_rD[i] = tRS_rD[i] * rC[i]
        return None


@jit_cache
def _compile_layernorm_gemm(
    a_dtype,
    b_dtype,
    d_dtype,
    c_dtype,
    a_major,
    b_major,
    d_major,
    c_major,
    gemm_k,
    blk_k,
    tile_shape_mn,
    cluster_shape_mnk,
    rowvec_dtype,
    has_dbias,
    out_bias_dtype,
    pingpong,
    fusion_variant,
    device_capacity,
):
    """Compile one `LayerNormGemmSm90` configuration against fake tensors.

    Every parameter is part of the ``@jit_cache`` key, and that is the point: two calls agreeing on
    all of them can share an artifact, and two differing in any must not.

    **M and L are the only symbolic extents.** ``gemm_k`` is baked because it is BOTH the contraction
    extent and the output width -- in the workflow this kernel serves, ``n == k == P == D`` is one
    feature dimension -- so baking it multiplies nothing: a caller sees one artifact per feature width
    either way. What it buys is a consumer k-loop whose trip count is a compile-time int, which is
    what lets the reduction's partials stay in registers instead of being carried across a dynamic
    MLIR region.

    Args:
        a_dtype, b_dtype, d_dtype: Cutlass element types of A, the folded weight, and the output.
        c_dtype: Element type of the ``gate3`` C operand, or None when there is no gate. None is a
            different kernel, not a runtime branch.
        a_major, b_major, d_major: ``"m"``/``"k"``/``"n"`` leading-dimension tags. A layout-left
            ``x`` (stride ``(1, M)``) arrives here as ``a_major="m"`` and needs no separate
            transpose pass -- it compiles to its own kernel with an MN-major WGMMA A operand.
        c_major: The gate's tag, or None.
        gemm_k: The TRUE contraction extent ``K == N == P``, baked.
        blk_k: The mainloop K tile, from :func:`resolve_blk_k`. Baked; it decides the accumulation
            grouping.
        tile_shape_mn: ``(tile_M, tile_N)`` CTA output tile.
        cluster_shape_mnk: ``(cluster_M, cluster_N, 1)``. Always ``(1, 1, 1)`` from this module's
            front door.
        rowvec_dtype: Element type of the ``c``/``d`` row vectors; fp32.
        has_dbias: Whether ``d = B^T @ b_ln`` is present. False compiles the term out entirely, so a
            kernel built without it cannot be handed one.
        out_bias_dtype: Element type of the output-bias row vector, or None.
        pingpong: The two-warpgroup alternating schedule.
        fusion_variant: Which fusion; only ``"alg_fold"`` compiles.
        device_capacity: ``(major, minor)``, re-checked below rather than trusted.

    Returns:
        The compiled TVM-FFI entry, callable as ``fn(A, B, D, C, epi_args, scheduler_args, None)``.

    Raises:
        UnsupportedArchError: If ``device_capacity`` is not SM90.
        ValueError: Propagated from the functor for a refused geometry or variant.
    """
    check_arch_supported(device_capacity)
    m, l = cute.sym_int(), cute.sym_int()
    k = gemm_k
    mA = fake_tensor(
        a_dtype,
        (m, k, l),
        leading_dim=1 if a_major == "k" else 0,
        divisibility=div_for_dtype(a_dtype),
    )
    mB = fake_tensor(
        b_dtype,
        (k, k, l),
        leading_dim=1 if b_major == "k" else 0,
        divisibility=div_for_dtype(b_dtype),
    )
    mD = fake_tensor(
        d_dtype,
        (m, k, l),
        leading_dim=1 if d_major == "n" else 0,
        divisibility=div_for_dtype(d_dtype),
    )
    mC = (
        fake_tensor(
            c_dtype,
            (m, k, l),
            leading_dim=1 if c_major == "n" else 0,
            divisibility=div_for_dtype(c_dtype),
        )
        if c_dtype is not None
        else None
    )
    epi_args = LayerNormGemmSm90.EpilogueArguments(
        # The two statistics ops take no host argument: their data is the mainloop's shared scratch,
        # supplied by `epi_get_smem_tensors`. The fields exist because the composition machinery
        # expects one per declared op.
        sRstd=None,
        sScale=None,
        mColsum=fake_tensor(rowvec_dtype, (l, k), leading_dim=1, divisibility=4),
        mDbias=fake_tensor(
            rowvec_dtype if has_dbias else None, (l, k), leading_dim=1, divisibility=4
        ),
        mOutBias=fake_tensor(
            out_bias_dtype,
            (l, k),
            leading_dim=1,
            divisibility=1 if out_bias_dtype is None else div_for_dtype(out_bias_dtype),
        ),
        eps=Float32(0),
    )
    scheduler_args = make_fake_scheduler_args(False, False, l)
    return compile_gemm_kernel(
        partial(LayerNormGemmSm90, fusion_variant=fusion_variant, blk_k=blk_k, gemm_k=gemm_k),
        a_dtype,
        tile_shape_mn,
        cluster_shape_mnk,
        pingpong,
        True,  # persistent
        False,  # is_dynamic_persistent
        device_capacity,
        mA,
        mB,
        mD,
        mC,
        epi_args,
        scheduler_args,
    )


def _launch_layernorm_gemm(
    A: Tensor,
    B: Tensor,
    D: Tensor,
    colsum: Tensor,
    dbias: Optional[Tensor],
    eps: float,
    tile_M: int,
    tile_N: int,
    pingpong: bool,
    fusion_variant: str,
    gate3: Optional[Tensor],
    out_bias: Optional[Tensor],
    max_swizzle_size: int = 8,
) -> None:
    """Permute the batched operands to the majors the kernel wants, compile, and launch.

    Purpose
        The one place a torch tensor becomes a kernel argument. Splitting it from the front door
        keeps the argument checking, the host fold and the configuration selection in one function
        and the ABI in another.

    Semantics
        `perm3d` and `get_majors` derive each operand's leading dimension from its STRIDES, so a
        layout-left activation or a transposed output passes through as a different compiled kernel
        rather than as a copy. Under ``COMPILE_ONLY`` this returns after the compile without
        launching, which is how the autotuner's compile worker warms the cache.

    Args:
        A: ``(l, m, k)`` activation. Its device decides the capability check.
        B: ``(l, P, k)`` folded weight -- ``Bw.mT``, from :func:`build_folded_operands`.
        D: ``(l, m, P)`` output, **written in place**.
        colsum: ``(l, P)`` fp32 ``c``.
        dbias: ``(l, P)`` fp32 ``d``, or None.
        eps: The LayerNorm variance floor.
        tile_M: CTA tile M.
        tile_N: CTA tile N.
        pingpong: The two-warpgroup schedule.
        fusion_variant: Which fusion to build.
        gate3: ``(l, m, P)`` per-element output gate, or None.
        out_bias: ``(l, P)`` output bias, or None.
        max_swizzle_size: Tile-scheduler rasterization swizzle width; scheduling only.

    Returns:
        None. Its effect is `D`.

    Raises:
        AssertionError: If the device is not SM90.
    """
    # The FRONT DOOR's capability gate, and it runs BEFORE anything selects a kernel class -- once
    # dispatch has begun, a non-SM90 device surfaces as an attribute lookup on a name that was
    # never defined, which tells the caller nothing. `require_sm90` raises `UnsupportedArchError`
    # naming the capability it found; it is a raise rather than an assert because `python -O` strips
    # asserts, and a stripped gate here is the internal explosion coming back.
    device_capacity = require_sm90(A.device)
    A_p, B_p, D_p, C_p = perm3d(A, B, D, gate3)
    a_major, b_major, d_major, c_major = get_majors(A_p, B_p, D_p, C_p)
    a_dtype, b_dtype, d_dtype, c_dtype = get_dtypes(A, B, D, gate3)
    gemm_k = A.shape[-1]
    compiled_fn = _compile_layernorm_gemm(
        a_dtype,
        b_dtype,
        d_dtype,
        c_dtype,
        a_major,
        b_major,
        d_major,
        c_major,
        gemm_k,
        resolve_blk_k(gemm_k),
        (tile_M, tile_N),
        (1, 1, 1),
        torch2cute_dtype_map[colsum.dtype],
        dbias is not None,
        torch2cute_dtype_map[out_bias.dtype] if out_bias is not None else None,
        pingpong,
        fusion_variant,
        device_capacity,
    )
    # Local import, NOT hoisted: COMPILE_ONLY is flipped at RUNTIME by the autotuner's compile
    # worker, and a module-level `from ... import COMPILE_ONLY` would bind False once at import and
    # never see that mutation.
    from fold_cp_ops._internal.cache_utils import COMPILE_ONLY

    if COMPILE_ONLY:
        return
    epi_args = LayerNormGemmSm90.EpilogueArguments(
        mColsum=colsum,
        mDbias=dbias,
        mOutBias=out_bias,
        eps=Float32(eps),
        rounding_mode=None,  # Constexpr: baked at compile time, passed as None at launch
    )
    scheduler_args = make_scheduler_args(get_max_active_clusters(1), max_swizzle_size, None)
    compiled_fn(A_p, B_p, D_p, C_p, epi_args, scheduler_args, None)


# ───────────────────────────── the host fold: Bw, c and d in one launch ─────────────────────────
# The B-side half of the algebraic fusion. It rewrites the WEIGHT once, which is why the fusion is
# worth having: the weight is loop-invariant, so this runs once per weight rather than once per token.
#
#   Bw[n, p] = round_to_B_dtype( w_ln[n] * B[n, p] )     (N, P), B's dtype
#   c[p]     = sum_n w_ln[n] * B[n, p]                   (P,) fp32 -- the fold BEFORE rounding
#   d[p]     = sum_n B[n, p] * b_ln[n]                   (P,) fp32, only with a LayerNorm bias
#
# **Deterministic by construction, no atomics.** Each CTA owns TILE_P contiguous columns across all
# rows; its TILE_P*TPC threads form a (TPC, TILE_P) grid where thread (rg, col) walks the strided row
# set {rg, rg+TPC, ...} of column `col`, and the TPC partials are then folded by row-group 0 in fixed
# ascending order through a shared scratch. Both the per-thread sum and the cross-group sum use an
# order independent of scheduling, so c and d are bit-reproducible run to run -- which an atomic_add
# reduction is not.
#
# **`c` reduces the UNROUNDED fold, which is not the column sum of the stored `Bw`.** That asymmetry
# is inherited, not chosen: it means the epilogue's `- s*c` does not subtract exactly what the MMA
# accumulated, and the residual is proportional to raw x rather than to centred x. It is modelled
# exactly by `numerics._alg_fold_preact_error(colsum_rounded=False)`.

#: Columns one CTA owns. A multiple of 32 and the fast (unit-stride) dimension of the row-major
#: ``(N, P)`` tensors, so each warp's reads of ``B[n, :]`` and writes of ``Bw[n, :]`` coalesce.
_FOLD_TILE_P = 64
#: Row-groups cooperating per column, so a CTA is ``_FOLD_TILE_P * _FOLD_TPC == 1024`` threads.
_FOLD_TPC = 16


@cute.kernel
def _fold_weight_kernel(
    mB: cute.Tensor,
    mW: cute.Tensor,
    mBias: cute.Tensor,
    mBw: cute.Tensor,
    mC: cute.Tensor,
    mD: cute.Tensor,
    has_bias: cutlass.Constexpr[bool],
):
    """Fold the gain into the weight and column-reduce, deterministically -- see the block above.

    Args:
        mB: ``(N, P)`` weight in the activation's 16-bit dtype. Read only.
        mW: ``(N,)`` fp32 LayerNorm gain. fp32 is required rather than converted: the fold rounds
            exactly once, and a narrower gain would round twice.
        mBias: ``(N,)`` fp32 LayerNorm bias. Read only when `has_bias`; may be None otherwise.
        mBw: ``(N, P)`` output, the rounded fold, in `mB`'s dtype. Written in place.
        mC: ``(P,)`` fp32 output, ``colsum`` of the UNROUNDED fold.
        mD: ``(P,)`` fp32 output, ``B^T @ bias``. Written only when `has_bias`; may be None.
        has_bias: Compile-time. False prunes both the ``d`` accumulation and its scratch entirely,
            rather than computing a zero.

    Returns:
        None. Its effects are `mBw`, `mC` and, with a bias, `mD`.

    Note:
        ``N`` and ``P`` are read from the tensor shapes and stay DYNAMIC, so one compiled callable
        serves every feature width -- which is what keeps the per-call dispatch off the critical path
        for a launch whose body is tens of microseconds.
    """
    tidx, _, _ = cute.arch.thread_idx()
    ctile, _, _ = cute.arch.block_idx()
    tile_p = const_expr(_FOLD_TILE_P)
    tpc = const_expr(_FOLD_TPC)
    n_rows = mB.shape[0]
    n_cols = mB.shape[1]
    col = tidx % tile_p
    rg = tidx // tile_p
    p = ctile * tile_p + col

    smem = cutlass.utils.SmemAllocator()
    sC = smem.allocate_tensor(
        Float32, cute.make_ordered_layout((tpc, tile_p), order=(1, 0)), byte_alignment=16
    )
    sD = (
        smem.allocate_tensor(
            Float32, cute.make_ordered_layout((tpc, tile_p), order=(1, 0)), byte_alignment=16
        )
        if const_expr(has_bias)
        else None
    )

    if p < n_cols:
        csum = Float32(0.0)
        dsum = Float32(0.0)
        # A COUNTED loop over this row-group's rows n = it*TPC + rg, ascending -- the same fixed
        # order a strided `range(rg, N, TPC)` would visit, so c and d are unchanged, but with the
        # address arithmetic hoisted out of the body. The tail below covers the final partial group.
        n_full = n_rows // tpc
        for it in cutlass.range(n_full):
            n = it * tpc + rg
            b = mB[n, p].to(Float32)
            scaled = mW[n] * b
            mBw[n, p] = scaled.to(mBw.element_type)
            csum = csum + scaled
            if const_expr(has_bias):
                dsum = dsum + b * mBias[n]
        n = n_full * tpc + rg
        if n < n_rows:
            b = mB[n, p].to(Float32)
            scaled = mW[n] * b
            mBw[n, p] = scaled.to(mBw.element_type)
            csum = csum + scaled
            if const_expr(has_bias):
                dsum = dsum + b * mBias[n]
        sC[rg, col] = csum
        if const_expr(has_bias):
            sD[rg, col] = dsum
    cute.arch.barrier()  # publish every partial before row-group 0 folds them

    if rg == 0 and p < n_cols:
        tot_c = Float32(0.0)
        tot_d = Float32(0.0)
        for g in cutlass.range_constexpr(tpc):
            tot_c = tot_c + sC[g, col]
            if const_expr(has_bias):
                tot_d = tot_d + sD[g, col]
        mC[p] = tot_c
        if const_expr(has_bias):
            mD[p] = tot_d


@cute.jit
def _fold_weight_jit(
    mB: cute.Tensor,
    mW: cute.Tensor,
    mBias: cute.Tensor,
    mBw: cute.Tensor,
    mC: cute.Tensor,
    mD: cute.Tensor,
    stream,
):
    """Launch :func:`_fold_weight_kernel` over ``ceil(P / TILE_P)`` CTAs.

    Args:
        mB: ``(N, P)`` weight; its second extent sets the grid.
        mW: ``(N,)`` fp32 gain.
        mBias: ``(N,)`` fp32 bias, or None. Its presence is the kernel's compile-time ``has_bias``.
        mBw: ``(N, P)`` fold output.
        mC: ``(P,)`` column-sum output.
        mD: ``(P,)`` bias-projection output, or None.
        stream: The CUDA stream, taken from the FFI environment at call time.

    Returns:
        None.
    """
    col_tiles = cute.ceil_div(mB.shape[1], _FOLD_TILE_P)
    threads = const_expr(_FOLD_TILE_P * _FOLD_TPC)
    _fold_weight_kernel(mB, mW, mBias, mBw, mC, mD, mBias is not None).launch(
        grid=[col_tiles, 1, 1], block=[threads, 1, 1], stream=stream
    )


@jit_cache
def _compile_fold_weight(dtype, has_bias):
    """Compile the fold for one ``(dtype, has_bias)``, generic over every ``(N, P)``.

    Both extents are symbolic, so this traces once per functional variant rather than once per shape.
    It compiles through TVM-FFI, which takes torch tensors directly and reads the stream from the FFI
    environment -- the DLPack round trip it replaces added a fixed per-call cost that dominated the
    wall time of a launch whose body is tens of microseconds.

    Args:
        dtype: Cutlass element type of the weight -- 16-bit.
        has_bias: Whether a LayerNorm bias is present. Part of the cache key: it is a compile-time
            branch, so the two builds are different kernels.

    Returns:
        The compiled callable ``fn(B, weight, bias, Bw, c, d)``.
    """
    n_sym, p_sym = cute.sym_int(), cute.sym_int()
    div = div_for_dtype(dtype)
    return cute.compile(
        _fold_weight_jit,
        fake_tensor(dtype, (n_sym, p_sym), div),
        fake_tensor(Float32, (n_sym,)),
        fake_tensor(Float32, (n_sym,)) if has_bias else None,
        fake_tensor(dtype, (n_sym, p_sym), div),
        fake_tensor(Float32, (p_sym,)),
        fake_tensor(Float32, (p_sym,)) if has_bias else None,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def build_folded_operands(B: Tensor, weight: Tensor, bias: Optional[Tensor] = None):
    """Fold the LayerNorm gain into the weight and build the epilogue's two row vectors, in ONE launch.

    Purpose
        The host half of the algebraic fusion, exposed because a caller with a static weight should
        hoist it out of the token loop. `layernorm_gemm` calls it per invocation for convenience;
        that is a correctness-preserving default, not the intended steady state.

    Semantics
        Produces ``Bw = round(diag(weight) @ B)``, ``c = colsum`` of the UNROUNDED fold, and
        ``d = B^T @ bias``. ``Bw`` is bitwise identical to doing the multiply and the cast in torch,
        because it is pure elementwise work. ``c`` and ``d`` are REDUCTIONS, so they agree with a
        torch reference only up to summation order -- the kernel fixes its own order so it is
        reproducible run to run, but that order is not torch's. A test must use ``torch.equal`` for
        the first and an fp32 summation-order bound for the other two.

    Args:
        B: ``(N, P)`` weight. Must be CUDA, contiguous and 16-bit (fp16/bf16); the fold rounds into
            this dtype, and a wider one would make the kernel's B operand disagree with what the MMA
            reads.
        weight: ``(N,)`` fp32 LayerNorm gain. Must have exactly `B`'s row count -- a shorter one is
            read past its end and scales the tail of the fold with whatever follows it in memory.
        bias: ``(N,)`` fp32 LayerNorm bias, or None. None returns ``d=None``, which compiles the
            corresponding epilogue term out; passing zeros instead costs a reduction to add nothing.

    Returns:
        ``(Bw, c, d)`` -- ``Bw`` ``(N, P)`` in `B`'s dtype, ``c`` ``(P,)`` fp32, ``d`` ``(P,)`` fp32
        or None.

    Raises:
        ValueError: If `weight` (or `bias`) does not have `B`'s row count or is not fp32. Checked
            here rather than trusted because both are read with an unpredicated broadcast load.
    """
    n, p = B.shape
    if weight.shape != (n,) or weight.dtype != torch.float32:
        raise ValueError(
            f"weight must be a ({n},) float32 tensor matching B's rows; got "
            f"{tuple(weight.shape)}/{weight.dtype}"
        )
    if bias is not None and (bias.shape != (n,) or bias.dtype != torch.float32):
        raise ValueError(
            f"bias must be a ({n},) float32 tensor matching B's rows; got "
            f"{tuple(bias.shape)}/{bias.dtype}"
        )
    Bw = torch.empty((n, p), device=B.device, dtype=B.dtype)
    c = torch.empty((p,), device=B.device, dtype=torch.float32)
    d = torch.empty((p,), device=B.device, dtype=torch.float32) if bias is not None else None
    _compile_fold_weight(torch2cute_dtype_map[B.dtype], bias is not None)(B, weight, bias, Bw, c, d)
    return Bw, c, d


# ───────────────────────────── the autotune-free size heuristic ─────────────────────────────
# Maps (M, D) straight to a configuration by BOUND REGIME -- no timing, no do_bench, no device-memory
# query. The levers were bisected on `heuristic_arch.TUNED_ARCH`; the tile VALIDITY guards below are
# hardware correctness and are arch-INDEPENDENT, which is why they sit in the tree and not in the dict.
#
#   * tile_N = 128 is universally valid: the epilogue predicates a partial N tile, so D needs only
#     the 16-byte alignment the front door already checks -- there is no D % 128 requirement.
#   * COMPUTE / memory-bandwidth bulk (the common case): tile_M = 256. It ties or wins at every swept
#     M once the launch-bound floor is crossed, and is within noise below it.
#   * D == 128 at LARGE M: ping-pong (128, 128) edges the 256-wide M tile by a few percent -- the
#     per-warpgroup epilogue overlap pays only in the small-tile, launch-pressured regime. At D >= 256
#     the 256-wide tile is at least as good everywhere, so it stays the pick there.
#   * tile_M = 256 wastes an M wave when M < 256, so it is gated on M >= tile_M.
#
# CAPACITY is not a regime here: the kernel is fully fused, so its footprint is the output's and there
# is nothing for a memory gate to prevent.
_LN_GEMM_PERF = {
    "H200_SXM5": {
        # Above this feature width the shape is compute / bandwidth bound and takes the bulk tile.
        # The gate is `>` and not `>= 256` deliberately: an off-grid D between 128 and 256 (192, 320)
        # follows the BULK rule, and a `>= 256` gate would drop those to the small-tile default.
        "BULK_MIN_D": 128,
        # D == 128 only: the M at which ping-pong overtakes both streaming tiles. Below it the three
        # configurations are within clock noise of each other.
        "PP_MIN_M_D128": 196608,
    },
}

#: The always-valid fallback: it runs any 16-byte-aligned feature width through predicated partial
#: N tiles, and its M tile cannot waste a wave on any M >= 128.
_DEFAULT_CONFIG_KWARGS = dict(tile_M=128, tile_N=128, pingpong=False)


def layernorm_gemm_heuristic_config(M: int, D: int, device=None) -> AutotuneConfig:
    """Pick the performance configuration for ``(M, D)`` by bound regime -- no timing, no memory query.

    Purpose
        The default path. `layernorm_gemm` runs this instead of a sweep, so a caller pays no tuning
        cost and gets a deterministic, reproducible pick.

    Semantics
        VALIDITY FIRST, then regime. The chosen tile must be able to run the shape -- ``tile_M <= M``
        so no M wave is wasted, and ping-pong only at a CTA-M the schedule admits -- and anything that
        fails falls through to the always-valid ``(128, 128)`` streaming default.

        **Arch-aware in the PERFORMANCE layer only.** The two thresholds were bisected on
        `TUNED_ARCH`; on any other architecture this warns ONCE and uses that set, because a
        mis-tuned threshold is slow and a mis-derived validity guard is a crash.

    Args:
        M: The token count. Only its comparison against the tile matters; it is a symbolic extent in
            the compiled kernel, so this never causes a recompile.
        D: The feature width ``K == N == P``. Assumed already checked for the 16-byte floor by the
            caller; this makes no correctness decision from it.
        device: The device whose architecture selects the threshold set, or None to use `TUNED_ARCH`
            silently. None is what a test passes when it wants the tuned numbers without a warning.

    Returns:
        An `AutotuneConfig` carrying ``tile_M``, ``tile_N`` and ``pingpong``, consumable by
        `layernorm_gemm`'s ``_config=`` fast path.
    """
    arch = heuristic_arch(device) if device is not None else TUNED_ARCH
    if arch != TUNED_ARCH:
        warn_arch_suboptimal_once(arch, "layernorm_gemm")
    perf = _LN_GEMM_PERF.get(arch, _LN_GEMM_PERF[TUNED_ARCH])
    default = AutotuneConfig(**_DEFAULT_CONFIG_KWARGS)
    if D > perf["BULK_MIN_D"]:
        if M >= 256:
            return AutotuneConfig(tile_M=256, tile_N=128, pingpong=False)
        return default
    if D == 128 and M >= perf["PP_MIN_M_D128"]:
        return AutotuneConfig(tile_M=128, tile_N=128, pingpong=True)
    return default


def layernorm_gemm_tuning_space() -> AxisSpace:
    """The declared autotuning axes for `layernorm_gemm`.

    Purpose
        Says what is tuned, in the kernel's own parameter names, so a reader can match an axis to the
        argument it sets without a translation step.

    Semantics
        Three axes over their full domains, with the pairs `GemmSm90.__init__` refuses removed --
        declaring the full grid and excluding from it, rather than listing the survivors, is what
        keeps a combination nobody considered distinguishable from one considered and rejected.

        **Only performance knobs are here.** The fusion variant, the statistics schedule and the
        reduction's lanes-per-row are not swept: the first is a different kernel, and the other two
        are fixed at the values the reference kernel's own tuner never moved off.

    Returns:
        A fresh `AxisSpace`, so a caller can subset it without mutating what the decorator is bound
        to.
    """
    return AxisSpace(
        TuneAxis(
            "tile_M",
            domain="64/128/192/256/320 cooperatively; 64/128/192 under ping-pong",
            values=(64, 128, 192, 256),
        ),
        TuneAxis(
            "tile_N",
            domain="divisible by 16 and <= 256, or by 32 and <= 512, with narrower ping-pong limits",
            values=(64, 128, 256),
        ),
        TuneAxis(
            "pingpong",
            domain="the two-warpgroup alternating schedule; requires a persistent grid",
            values=(False, True),
        ),
        exclude=lambda p: p["pingpong"] and (p["tile_M"] > 192 or p["tile_N"] > 208),
        because=(
            "the tile axes are JOINT with the schedule axis: GemmSm90.__init__ refuses tile_M > 192 "
            "under ping-pong outright, and caps tile_N at 208 for tile_M=128 and 128 for tile_M=192. "
            "A candidate that RAISES is not a slow configuration, it is a crashed sweep, so it is "
            "excluded here rather than caught later"
        ),
    )


#: The pool the tuned path sweeps. Built once at import: `AxisSpace.configs()` is pure and the
#: decorator needs the points at decoration time.
LN_GEMM_TUNING_SPACE = layernorm_gemm_tuning_space()


def layernorm_gemm_config_is_valid(config, request) -> bool:
    """Whether one candidate CAN RUN for this request. Rejection only -- never preference.

    Purpose
        Keeps a sweep from timing a geometry the functor RAISES on. A candidate that raises is not a
        slow configuration, it is a crashed sweep, and it fails inside the tuner where the message
        names a config index rather than a shape.

    Semantics
        Must be a PURE function of the config and the request: the candidate list has to be
        identical on every rank, or ranks measure different pools and compare numbers for different
        kernels.

        **A tile WIDER than the token count is deliberately NOT rejected here.** It runs -- the
        epilogue predicates a partial M tile -- it is merely slower, and "slower" is what the sweep
        exists to discover by measurement. Putting it in this filter would make it a claim that the
        configuration cannot run, which would in turn make the size heuristic's own always-valid
        fallback look invalid at a small token count: that fallback is ``(128, 128)`` at every
        shape, including shapes below 128 rows.

    Args:
        config: The candidate, carrying ``tile_M``, ``tile_N`` and ``pingpong``.
        request: The bound call arguments by name, as the tuner assembles them. Unused -- every
            constraint here is a property of the geometry alone. Taken because the tuner passes it
            positionally.

    Returns:
        False when ping-pong is paired with a CTA-M or a tile_N the schedule refuses at
        construction; True otherwise.
    """
    if not config["pingpong"]:
        return True
    # `GemmSm90.__init__` refuses tile_M above 192 under ping-pong outright, and caps tile_N per
    # tile_M: 256 at 64, 208 at 128, 128 at 192.
    cap = {64: 256, 128: 208, 192: 128}.get(config["tile_M"])
    return cap is not None and config["tile_N"] <= cap and config["tile_N"] % 16 == 0


@autotune(
    space=LN_GEMM_TUNING_SPACE,
    key=["has_bias", "has_gate3", "has_out_bias"],
    validity=layernorm_gemm_config_is_valid,
)
def _layernorm_gemm_tuned(
    x,
    weight,
    B,
    bias=None,
    eps=1e-6,
    out=None,
    gate3=None,
    out_bias=None,
    has_bias=False,
    has_gate3=False,
    has_out_bias=False,
    tile_M=128,
    tile_N=128,
    pingpong=False,
):
    """Sweep the performance knobs for this shape and functional variant, then run the winner.

    The autotuner injects ``tile_M``/``tile_N``/``pingpong``; everything else is forwarded verbatim.
    Each FUNCTIONAL variant -- distinguished by the three ``has_*`` flags plus the shapes and dtype
    the tuner keys on automatically -- tunes and caches independently, because they are different
    kernels rather than different schedules of one.

    Args:
        x, weight, B, bias, eps, out, gate3, out_bias: As `layernorm_gemm`.
        has_bias, has_gate3, has_out_bias: Cache-key components naming the functional variant. Passed
            explicitly rather than derived, because the tuner keys on argument VALUES and a tensor is
            not a key.
        tile_M, tile_N, pingpong: Injected by the tuner.
        do_autotune: The tuner's gate. Always True here; the fixed path is reached by not calling
            this function at all.

    Returns:
        The ``(M, P)`` output tensor.
    """
    return layernorm_gemm(
        x,
        weight,
        B,
        bias=bias,
        eps=eps,
        out=out,
        gate3=gate3,
        out_bias=out_bias,
        tile_M=tile_M,
        tile_N=tile_N,
        pingpong=pingpong,
        select="default",
    )


def layernorm_gemm(
    x: Tensor,
    weight: Tensor,
    B: Tensor,
    bias: Optional[Tensor] = None,
    eps: float = 1e-6,
    out: Optional[Tensor] = None,
    gate3: Optional[Tensor] = None,
    out_bias: Optional[Tensor] = None,
    tile_M: Optional[int] = None,
    tile_N: Optional[int] = None,
    pingpong: bool = False,
    fusion_variant: str = "alg_fold",
    select: str = "heuristic",
    _config: Optional[AutotuneConfig] = None,
) -> Tensor:
    """``out = (LayerNorm(x) @ B + out_bias) * gate3`` in one kernel that reads ``x`` exactly once.

    Purpose
        The front door. It checks the arguments, folds the LayerNorm gain into the weight on the
        host, resolves a configuration, and launches. This is the cp=1 TriMul workflow's universal
        fallback: every feature width the wider fused kernels' tiling cannot take arrives here.

    Semantics
        The LayerNorm is per ROW of `x`, over its `N` features, and it is never formed explicitly --
        see the module docstring for the algebra and for what that costs on an off-centre row.

        **Layouts pass through as strides, not as copies.** `x` may be ``(M, N)`` row-major (the
        default) OR layout-left with stride ``(1, M)``: the second compiles to its own kernel whose
        WGMMA takes an MN-major A operand, which fuses away a transpose pass. `out` likewise may be a
        transposed view. Neither is a runtime branch -- they are separate compiled artifacts.

        **Selection.** ``select="heuristic"`` (the default) picks the tile from ``(M, N)`` with no
        timing. It DEFERS to the fixed path the moment the caller pins any performance knob, so an
        explicit ``tile_M``/``tile_N``/``pingpong`` runs exactly as given rather than being
        overridden. ``select="autotune"`` sweeps; ``select="default"`` runs the arguments as passed.

    Args:
        x: ``(M, N)`` fp16/bf16 activation. `N` must be a multiple of 8 -- the 16-byte TMA floor, and
            the ONLY shape constraint this kernel has. It is NOT restricted to a power of two, a tile
            multiple, or a multiple of 32.
        weight: ``(N,)`` fp32 LayerNorm gain. fp32 is required, not converted: it is folded into the
            weight and rounded exactly once.
        B: ``(N, P)`` fp16/bf16 GEMM weight with ``P == N``, and the same dtype as `x`.
        bias: ``(N,)`` fp32 LayerNorm bias, or None. Folded into the epilogue's ``d`` vector.
        eps: The variance floor, added before the reciprocal square root. A runtime value.
        out: Optional ``(M, P)`` destination, allocated with `x`'s dtype when None. Written in place
            when given; nothing resizes it.
        gate3: Optional ``(M, P)`` per-element output gate with `out`'s layout. Applied as the LAST
            step of the epilogue, so it costs no extra pass over DRAM. Its contiguous `P` extent needs
            the same 16-byte row alignment, since it rides the TMA C load.
        out_bias: Optional ``(P,)`` fp32/fp16/bf16 bias on the GEMM's OUTPUT, added after the
            LayerNorm repair and before the gate. **Distinct from `bias`**, which is the LayerNorm's
            beta on the normalized INPUT.
        tile_M: CTA tile M, or None to let the selection decide. Pinning it switches to the fixed
            path.
        tile_N: CTA tile N, likewise.
        pingpong: The two-warpgroup alternating schedule. Pinning it to True switches to the fixed
            path.
        fusion_variant: Which LayerNorm fusion. Only ``"alg_fold"`` is implemented; see Raises.
        select: ``"heuristic"`` (default), ``"autotune"`` or ``"default"``.
        _config: A frozen configuration from `layernorm_gemm_freeze`, which skips the per-call
            selection. Private: it bypasses the argument checks the public path performs.

    Returns:
        The ``(M, P)`` output, `out` when one was given.

    Raises:
        ValueError: On ``fusion_variant="prolog_ln"`` (declared but not brought back -- the cp=1
            heuristic never selects it, so implementing it would add no reachable behaviour); on an
            `x` that is not 2-D; on a feature width that is not 16-byte aligned; on a `B` whose rows
            do not match `N` or whose columns do not equal them; on a non-fp32 or mis-sized `weight`
            or `bias`; on a `gate3` or `out_bias` of the wrong shape or dtype; or on an unknown
            `select`. Every one is a ``raise`` rather than an ``assert`` because ``python -O`` strips
            asserts, and a stripped check here is a TMA fault or a silently wrong answer three frames
            down instead of a sentence naming the argument.
    """
    if _config is not None:
        return layernorm_gemm(
            x,
            weight,
            B,
            bias=bias,
            eps=eps,
            out=out,
            gate3=gate3,
            out_bias=out_bias,
            fusion_variant=fusion_variant,
            select="default",
            **_config.all_kwargs(),
        )
    if select == "autotune":
        return _layernorm_gemm_tuned(
            x,
            weight,
            B,
            bias=bias,
            eps=eps,
            out=out,
            gate3=gate3,
            out_bias=out_bias,
            has_bias=bias is not None,
            has_gate3=gate3 is not None,
            has_out_bias=out_bias is not None,
        )
    if select == "heuristic":
        # The heuristic owns the PERFORMANCE knobs. If the caller pinned ANY of them, defer to the
        # fixed path so their exact configuration runs untouched -- silently overriding a pinned tile
        # would make an explicit argument a no-op.
        if tile_M is None and tile_N is None and not pingpong:
            return layernorm_gemm(
                x,
                weight,
                B,
                bias=bias,
                eps=eps,
                out=out,
                gate3=gate3,
                out_bias=out_bias,
                fusion_variant=fusion_variant,
                _config=layernorm_gemm_heuristic_config(x.shape[0], x.shape[1], device=x.device),
            )
        return layernorm_gemm(
            x,
            weight,
            B,
            bias=bias,
            eps=eps,
            out=out,
            gate3=gate3,
            out_bias=out_bias,
            tile_M=tile_M,
            tile_N=tile_N,
            pingpong=pingpong,
            fusion_variant=fusion_variant,
            select="default",
        )
    if select != "default":
        raise ValueError(f"select must be 'heuristic', 'autotune' or 'default'; got {select!r}")

    if fusion_variant not in FUSION_VARIANTS:
        raise ValueError(f"fusion_variant must be one of {FUSION_VARIANTS}; got {fusion_variant!r}")
    if fusion_variant not in LayerNormGemmSm90._IMPLEMENTED_VARIANTS:
        raise ValueError(
            f"fusion_variant={fusion_variant!r} is not brought back; the cp=1 heuristic never "
            f"selects it, so only {LayerNormGemmSm90._IMPLEMENTED_VARIANTS} is implemented"
        )
    if x.dim() != 2:
        raise ValueError(f"x must be 2-D (M, N); got {tuple(x.shape)}")
    M, N = x.shape
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"x must be float16 or bfloat16; got {x.dtype}")
    if weight.dim() != 1 or weight.shape[0] != N or weight.dtype != torch.float32:
        raise ValueError(
            f"weight must be a ({N},) float32 tensor -- the gain is folded into B in fp32 and "
            f"rounded once; got {tuple(weight.shape)}/{weight.dtype}"
        )
    if bias is not None and (bias.dim() != 1 or bias.shape[0] != N or bias.dtype != torch.float32):
        raise ValueError(
            f"bias must be a ({N},) float32 tensor; got {tuple(bias.shape)}/{bias.dtype}"
        )
    if B.dim() != 2 or B.shape[0] != N:
        raise ValueError(f"B must be (N, P) with N={N}; got {tuple(B.shape)}")
    P = B.shape[1]
    if P != N:
        raise ValueError(
            f"this kernel contracts and outputs over the SAME feature width, so it requires P == N; "
            f"got P={P}, N={N}"
        )
    if B.dtype != x.dtype:
        raise ValueError(f"B must share x's dtype ({x.dtype}); got {B.dtype}")
    # The ONLY shape constraint: the 16-byte TMA floor. A feature width that is a multiple of 16
    # picks an exact k-tile; one that is 8 mod 16 runs the 16-wide tile with a TMA-zero-filled
    # partial last tile. Either way the row statistics divide by the TRUE N.
    n_per_16b = 16 // x.element_size()
    if N % n_per_16b != 0:
        raise ValueError(
            f"N must be 16-byte aligned (N % {n_per_16b} == 0 for {x.dtype}); got N={N}"
        )
    if gate3 is not None:
        # Membership FIRST, matching `check_tensor`'s documented order, and it is the check that was
        # missing: gate3's extent and its 16-byte row were both validated below while its DTYPE was
        # not, so an unsupported one reached `get_dtypes`' `torch2cute_dtype_map[...]` and surfaced
        # as a bare `KeyError: torch.float64` -- naming a torch dtype, no argument, and not a type a
        # caller can act on. `out_bias` three lines down has always had its dtype checked; this is
        # the sibling that did not. It also makes the code match this entry's own docstring, which
        # already promises a raise "on a `gate3` or `out_bias` of the wrong shape or dtype".
        if gate3.dtype not in torch2cute_dtype_map:
            raise ValueError(
                f"unsupported dtype for gate3: {gate3.dtype}. Supported: "
                f"{sorted(str(d) for d in torch2cute_dtype_map)}."
            )
        if gate3.dim() != 2 or tuple(gate3.shape) != (M, P):
            raise ValueError(f"gate3 must be (M, P)=({M}, {P}); got {tuple(gate3.shape)}")
        g_div = 16 // gate3.element_size()
        if P % g_div != 0:
            raise ValueError(
                f"gate3's contiguous P extent must be a multiple of {g_div} (the 16-byte TMA row "
                f"for {gate3.dtype}); got P={P}"
            )
    if out_bias is not None:
        if out_bias.dim() != 1 or out_bias.shape[0] != P:
            raise ValueError(f"out_bias must be 1-D (P,)=({P},); got {tuple(out_bias.shape)}")
        if out_bias.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError(f"out_bias must be fp32/fp16/bf16; got {out_bias.dtype}")
    if tile_M is None or tile_N is None:
        tile_M, tile_N = _DEFAULT_CONFIG_KWARGS["tile_M"], _DEFAULT_CONFIG_KWARGS["tile_N"]
    if out is None:
        out = torch.empty((M, P), dtype=x.dtype, device=x.device)

    Bw, c, d = build_folded_operands(B, weight, bias)
    _launch_layernorm_gemm(
        x.unsqueeze(0),
        Bw.mT.unsqueeze(0),
        out.unsqueeze(0),
        c.unsqueeze(0),
        d.unsqueeze(0) if d is not None else None,
        eps,
        tile_M,
        tile_N,
        pingpong,
        fusion_variant,
        gate3.unsqueeze(0) if gate3 is not None else None,
        out_bias.unsqueeze(0) if out_bias is not None else None,
    )
    return out


def layernorm_gemm_freeze(x, weight, B, bias=None, eps=1e-6, gate3=None, out_bias=None):
    """Resolve the winning configuration for THIS shape and variant once, and bind a callable to it.

    Purpose
        Removes the per-call selection from a steady-state loop. The sweep runs once here; every
        later call goes straight to the fixed path with the resolved knobs.

    Args:
        x, weight, B, bias, eps, gate3, out_bias: A REPRESENTATIVE call. The sweep is keyed on these
            shapes and on which optional arguments are present, so freezing on a shape the loop will
            not run returns a configuration tuned for something else.

    Returns:
        A callable with `layernorm_gemm`'s signature minus the selection arguments, carrying the pick
        on its ``.config`` attribute.
    """
    layernorm_gemm(
        x, weight, B, bias=bias, eps=eps, gate3=gate3, out_bias=out_bias, select="autotune"
    )
    # `best_config` is the TUNER's, not the wrapper's: `@autotune` attaches `.autotuner` and
    # `.axes` and nothing else, so reading it off the wrapper is an AttributeError on the first
    # call. `alg_fold_freeze` in the dual takes the same path.
    frozen = _layernorm_gemm_tuned.autotuner.best_config

    def frozen_call(x, weight, B, bias=bias, eps=eps, out=None, gate3=gate3, out_bias=out_bias):
        """Run `layernorm_gemm` with the frozen configuration.

        Args:
            x, weight, B, bias, eps, out, gate3, out_bias: As `layernorm_gemm`.

        Returns:
            The ``(M, P)`` output.
        """
        return layernorm_gemm(
            x,
            weight,
            B,
            bias=bias,
            eps=eps,
            out=out,
            gate3=gate3,
            out_bias=out_bias,
            _config=frozen,
        )

    frozen_call.config = frozen
    return frozen_call

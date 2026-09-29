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
# Copyright (c) 2025, Wentao Guo, Tri Dao.
# This file has been modified by NVIDIA CORPORATION & AFFILIATES.
"""The fused LayerNorm + dual-gated GEMM: the TriMul input projection, in one kernel.

    An[m, :]  = (x[m, :] - mu_m) * rstd_m * gain + lnbias      # LayerNorm over the K axis
    out[m, n] = sigmoid(An @ Wg^T + bg)[m, n] * (An @ Wp^T + bp)[m, n]   [* mask[m]]

`DualGatedGemmSm90` already computes everything after the first line, so this module adds the first
line and nothing else. It is one class, one compile entry and one front door.

**``fusion_variant`` is a compile-time selector, not a runtime branch.** Two fusions form the same
2N pre-activation by different means, and which one is built is fixed at construction:

* ``"prolog_ln"`` -- the PHYSICAL fusion, and the only one implemented here. Per-row statistics are
  reduced over the staged A tile, ``A`` is rewritten in shared memory as ``LayerNorm(A)``, and the
  WGMMA that follows is an ordinary GEMM. The epilogue is inherited from `GemmGatedMixin`
  unchanged.
* ``"alg_fold"`` -- the ALGEBRAIC fusion: never materialize the normalized A, and apply
  ``LN(x) @ B = r*(x @ Bw) - s*c + d`` as a rank-one correction in the epilogue instead. Declared
  as a value so the selector does not have to change shape when it lands; constructing one raises.

**Why ``prolog_ln`` is kept even where ``alg_fold`` is faster.** It leaves the NORMALIZED activation
resident in shared memory, which is precisely what an all-to-all-fused front needs in order to
scatter it to peers. It is the variant the distributed path will want, not a fallback.

**Where the all-to-all fusion attaches.** The distributed front replaces the post-activation STORE
-- the gated output is scattered to peers instead of written locally -- and nothing else. That seam
is `TileStore("mPostAct")` plus the epilogue mixin's store hooks, both untouched here. The mask is
declared through `TransposedMaskColVecLoad` rather than the plain `ColVecLoad` for the same reason:
it is byte-identical for the 2-D mask this entry builds, and it is already the form a transposed
distributed front hands it.

**The two weight layouts.** ``chunk_g > 1`` loads ``Wg`` and ``Wp`` directly through a two-tensor
TMA and needs NO host preparation -- zero extra launches per call, which is why it is preferred.
``chunk_g == 1`` needs an element-interleaved ``(2N, K)`` weight, which costs one host-side
interleave. Neither layout folds the LayerNorm gain into the weight: the gain is applied to the
ACTIVATION in shared memory, so the weights travel raw.

.. warning::

   **This kernel keys its compiled artifact on the FEATURE WIDTH -- both K and N. The token count M
   is the only symbolic extent.** Two separate reasons landed there:

   * K was forced. The staged ``(K,)`` gain and bias live in shared memory, and a shared-memory
     allocation must be a compile-time size.
   * N was chosen, for speed. A static N folds the epilogue's out-of-bounds bounds to literals
     where a symbolic one predicates per output element; a static K does the same for the
     mainloop's. Measured on the two-A sibling, kernel-to-kernel against the kernel this
     reproduces: **1.02-1.05x with N symbolic, 0.98-1.00x with it baked** -- the whole gap.

   **The artifact count does not grow.** In the TriMul workflow the activation is ``(tokens, D)``
   and both projections are ``(D, D)``, so ``K == N == D``: the two extents are the SAME feature
   dimension, and a caller sees one artifact per width rather than one per ``(K, N)`` pair. A model
   has one feature width, so the count is bounded by the model and not by the batch.

   This is a deviation from this package's dynamic-shape principle along the feature axis, and it
   is the kernel this reproduces' own design -- its ``K``/``twoN``/``N3``/``dual_n`` are each
   commented ``# STATIC``. The cost lands on a caller who sweeps a feature width at runtime, which
   the workflow does not do.

   The alternative for K -- reading the gain straight from global memory in the normalize, which
   removes K from the key at the cost of an L1 access per element -- is a measurable change to the
   mainloop and is deliberately NOT taken silently.
"""

import operator
from functools import cached_property, partial
from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import torch
import torch.nn.functional as F
from cutlass import Boolean, Float32, Int32, const_expr
from cutlass.cute.nvgpu import warpgroup
from torch import Tensor

import fold_cp_ops._internal.copy_utils as copy_utils
import fold_cp_ops._internal.sm90_utils as fold_cp_ops_sm90_utils
from fold_cp_ops._internal.autotune import AxisSpace, TuneAxis, autotune
from fold_cp_ops._internal.heuristic_arch import (
    TUNED_ARCH,
    heuristic_arch,
    warn_arch_suboptimal_once,
)
from fold_cp_ops._internal.activation import act_fn_map, as_gate_fn, gate_fn_map
from fold_cp_ops._internal.arch import (
    check_arch_supported,
    get_max_active_clusters,
    require_sm90,
)
from fold_cp_ops._internal.cache_utils import jit_cache
from fold_cp_ops._internal.compile_time.compile_utils import make_fake_tensor as fake_tensor
from fold_cp_ops._internal.compile_time.cute_dsl_utils import torch2cute_dtype_map
from fold_cp_ops._internal.compile_time.ln_prologue_layout import (
    ALLOWED_BLK_K,
    STATS_THREADS_PER_ROW,
    auto_blk_k,
    ln_affine_bytes,
    mma_inst_tile_k,
    stats_scratch_bytes,
    stats_tiled_copy,
)
from fold_cp_ops._internal.gemm_tvm_ffi_utils import (
    check_broadcast_alignment,
    compile_gemm_kernel,
    div_for_dtype,
    get_major,
    make_fake_scheduler_args,
    make_scheduler_args,
    perm3d_single,
)
from fold_cp_ops._internal.epi_act import GemmActMixin
from fold_cp_ops._internal.epi_ops import (
    ColVecLoad,
    Gate3RowVecLoad,
    RowVecLoad,
    Scalar,
    SmemColVecBroadcast,
    TileStore,
)
from fold_cp_ops._internal.ln_prologue import (
    RS_QUAD_LANES,
    RS_ROW_STRIDE,
    accumulate_row_stats,
    finalize_row_stats,
    make_row_stat_partials,
    normalize_tile,
    rs_accumulate_row_stats,
    rs_owned_rows,
    stage_ln_affine,
)
from fold_cp_ops._internal.rounding import RoundingMode
from fold_cp_ops._internal.tensor_contract import check_tensor
from fold_cp_ops._internal.pipeline import make_pipeline_state
from fold_cp_ops._internal.runtime_params import ParamsBase, mlir_namedtuple
from fold_cp_ops.kernels.dual_gated_gemm import (
    DualGatedGemmParams,
    DualGatedGemmSm90,
    _as_batched,
    _pitch_align_broadcast,
    append_gate3_weight,
    build_dual_operands,
)

__all__ = [
    "FUSION_VARIANTS",
    "LayerNormDualGatedGemmParams",
    "LayerNormDualGatedGemmAlgFoldSm90",
    "LayerNormDualGatedGemmSm90",
    "LayerNormDualGatedGemmXGateAlgFoldSm90",
    "LayerNormDualGatedGemmXGatePrologLnSm90",
    "layernorm_dual_gated_gemm",
    "layernorm_dual_gated_gemm_ref",
]

#: The compile-time fusion selectors this functor accepts. ``"alg_fold"`` is declared but not built
#: -- see the module docstring. Listing it here rather than inventing it later is what keeps the
#: selector's shape (and the front door's error message) stable across the two bring-backs.
FUSION_VARIANTS = ("prolog_ln", "alg_fold")

#: CTA tile M extents this fusion accepts. A SUBSET of what the base GEMM accepts (which adds 320):
#: the fused reduction tiles ``tile_M`` by its row-group count, and 320 is not a multiple of the 128
#: groups the base's two-warpgroup split for that tile leaves. Refused at the front door rather than
#: three frames down in `stats_tiled_copy`, where it was an `AssertionError` naming a layout.
_SUPPORTED_TILE_M = (64, 128, 192, 256)

#: The ONE CTA tile M the two-A register-source mainloop's row mapping is closed-form for. Its
#: ``base = (t//128)*64 + ((t%128)//32)*16 + ((t%32)//4)`` enumerates rows 0..127 over 256 threads,
#: two per thread, which is the WGMMA operand-A fragment for a 128-row tile across two warpgroups
#: and nothing else.
#:
#: **What goes wrong without this gate is a SILENT WRONG ANSWER, not a fault.** At ``tile_M = 256``
#: the same two warpgroups cover 128 rows each and a thread owns FOUR rows, but the form still
#: addresses two, and only within the first 96 of them: every row past that is normalized with
#: another row's ``mu``/``rstd``, and rows nothing claims keep whatever the scratch held. At 64 and
#: 192 the warpgroup split changes underneath it (``atom_layout_mnk`` is ``(1,1,1)`` and ``(3,1,1)``
#: there, not ``(2,1,1)``), so the ``t//128`` half-select is wrong outright. Every one of those is a
#: finite, plausible output with no shape error, no assertion and no NaN to notice.
#:
#: The kernel this reproduces derives the same form for its own ``BLK_M = 128``, says so in a
#: comment, and does NOT check it -- while taking ``tile_M`` as a launch parameter, so it answers
#: wrongly at 256. Gating here is the inherited-defect rule: do not reproduce a bug. It refuses no
#: configuration -- `_XGateTwoASm90._value_a_in_regs` simply selects the shared-memory mainloop,
#: which is a complete and correct implementation at every tile, merely without the MN-major saving.
_A_IN_REGS_TILE_M = 128

#: Ceiling on the mainloop pipeline depth. It is where ``GemmSm90._compute_stages``'s FLAT
#: 1024-byte allowance for mbarriers and alignment pads is exactly consumed by the mainloop
#: pipeline's own barriers (16 bytes a stage), past which the shared-storage struct is larger than
#: what was budgeted for it. Reachable only at a degenerate tile; see
#: :meth:`LayerNormDualGatedGemmSm90._compute_stages` for the measurement and for why the base's
#: estimate is left alone.
_MAX_AB_STAGE = 64


class _LnPrologueState:
    """The trace-time state the fused mainloop builds once and reuses for every work tile.

    Purpose
        `MmaFragments.extra` is a free slot for exactly this. Carrying the prologue's partition,
        scratch and scalars in a named object -- rather than a tuple -- means a field added later
        cannot be silently mis-positioned at the one call site that unpacks it.

    Semantics
        A plain Python object built inside :meth:`LayerNormDualGatedGemmSm90.mma_setup_fragments`,
        so it costs nothing at runtime. Its fields are a mixture of trace-time descriptors (the
        tiled copy slice, the shared-memory tensors) and MLIR values (``tidx``, ``n_elems``,
        ``eps``), which is why it is never passed across a ``@cute.jit`` boundary: the DSL flattens
        arguments into MLIR values and cannot accept an object. The mainloop unpacks it and passes
        the fields.

        ``__slots__`` makes a typo an ``AttributeError`` at trace time rather than a ``None`` that
        normalizes with garbage statistics.

    Attributes:
        thr_copy: This thread's slice of the statistics tiled copy. The SAME slice drives the
            reduction and the normalize, which is what guarantees a thread rescales the rows it
            helped reduce.
        sA: The staged A tile, ``(tile_M, tile_K, ab_stage)``. Held so the mainloop can index a
            stage without re-deriving the view.
        stats_smem: The ``(tile_M, 2)`` fp32 scratch. WHICH statistic sits in which column is
            the fusion's choice, published by `finalize_row_stats`: ``prolog_ln`` writes
            ``(mu, rstd)`` because its normalize rescales rows; ``alg_fold`` writes
            ``(rstd, rstd*mu)`` because its epilogue wants that product already formed.
        s_weight: The staged ``(gemm_k,)`` fp32 LayerNorm gain.
        s_bias: The staged ``(gemm_k,)`` fp32 LayerNorm bias, or None when the kernel was built
            without one.
        tidx: This thread's index within the MMA warpgroups, used to select its row group.
        n_elems: The per-row element count -- the TRUE feature width, a runtime value.
        eps: The variance floor, a runtime value.
    """

    __slots__ = ("thr_copy", "sA", "stats_smem", "s_weight", "s_bias", "tidx", "n_elems", "eps")

    def __init__(self, **fields):
        """Bind every slot by keyword.

        Args:
            **fields: One entry per name in ``__slots__``. All required and keyword-only, because
                several fields are shared-memory tensors of compatible shape that a positional form
                would let be swapped with no diagnostic.

        Raises:
            TypeError: If a slot is missing.
            AttributeError: If a name is not a declared slot.
        """
        missing = set(self.__slots__) - set(fields)
        if missing:
            raise TypeError(f"_LnPrologueState missing field(s): {sorted(missing)}")
        for k, v in fields.items():
            setattr(self, k, v)


class LayerNormDualGatedGemmParams(DualGatedGemmParams):
    """`DualGatedGemmParams` plus what the LayerNorm prologue fixes at construction.

    Extending the base pack rather than declaring a second one is what keeps the guarantee total:
    ``_bind_params`` binds exactly one pack, so a subclass pack that omitted the base's fields would
    leave them unbound and the functor half-configured.

    Attributes:
        fusion_variant: Which fusion to build -- a member of :data:`FUSION_VARIANTS`. Compile-time
            and immutable: the two variants differ in the mainloop AND the epilogue, so a functor is
            one or the other. ``"alg_fold"`` is accepted by the pack and refused by ``__init__``,
            which is the split that lets the selector exist before the second variant does.
        blk_k: The mainloop K tile, one of
            :data:`~fold_cp_ops._internal.compile_time.ln_prologue_layout.ALLOWED_BLK_K`. **This is
            a numerical parameter, not only a performance one**: it sets the k-loop trip count, so
            it decides how the per-row sums and the WGMMA accumulator are associated. Two functors
            differing only in ``blk_k`` compute the same value in exact arithmetic and different bit
            patterns in floating point. It must divide ``gemm_k`` exactly; a partial last K-tile
            would index the staged gain past its end.
        gemm_k: The PADDED contraction extent -- ``ceil(K / blk_k) * blk_k``. Required, because the
            staged ``(K,)`` gain is a shared-memory allocation and therefore a compile-time size.
            This is the one shape extent this functor bakes; see the module docstring.
        gemm_k_real: The TRUE feature width. Equal to ``gemm_k`` unless the front door zero-extended
            a K that the tile does not divide. It is the LayerNorm normalizer and the number of gain
            elements copied from global, so passing the padded value here would divide every mean by
            too large an N -- a wrong answer, not a fault.

    The output gate's ``n_dual_tiles`` / ``gate3_n3`` are NOT declared here: they come from
    `DualGatedGemmParams`, because the gate is a property of the dual-gated GEMM and not of this
    fusion. This pack adds only what the LayerNorm prologue itself needs.
    """

    fusion_variant: str = "prolog_ln"
    blk_k: int = 64
    gemm_k: int = 0
    gemm_k_real: int = 0


class _AlreadyOrderedBarrier:
    """A no-op stand-in for a barrier the SCHEDULE already provides.

    Purpose
        Under ping-pong the kernel executes ``pingpong_barrier_sync(wg, "epi")`` between a
        warpgroup's mainloop and its own epilogue (`gemm_sm90_mma.py:716`), which already orders the
        statistics write before the read. `finalize_row_stats` still needs an object with
        ``arrive_and_wait()``, and this is it -- so the helper stays barrier-agnostic instead of
        learning about ping-pong.

    Semantics
        **Adding a real barrier here is CORRECT but costs ~36%**, measured: an extra rendezvous per
        work tile, on the path ping-pong exists to keep busy. The upstream says the same in prose --
        "ping-pong defers that visibility to the kernel's ping-pong epi barrier instead" -- and
        matching it is what makes this port's ping-pong/cooperative RATIO match the upstream's
        rather than invert it.

        The shared-memory FENCE that precedes this in `finalize_row_stats` is still required and
        still emitted; only the rendezvous is redundant.

    Input requirements
        Valid **only** where a later collective sync is guaranteed between the write and the read.
        Cooperatively there is none, which is why that path keeps the epilogue barrier.
    """

    def arrive_and_wait(self):
        """Do nothing; the ping-pong epi barrier is the rendezvous.

        Returns:
            None.
        """


class LayerNormDualGatedGemmSm90(DualGatedGemmSm90):
    """`DualGatedGemmSm90` with a LayerNorm prologue in front of its mainloop.

    It subclasses the dual-gated GEMM rather than re-deriving from `GemmSm90`, so the fold, the
    mask, the register permute, the store AND the fused output gate all arrive inherited. The
    prologue is NOT a mixin -- it is four free functions in
    :mod:`fold_cp_ops._internal.ln_prologue` reached from the two MMA seams below, so no third class
    enters the MRO.

    **This class knows nothing about the output gate.** `DualGatedGemmSm90` owns it, because
    ``act3(A @ W3^T + b3)`` needs nothing a LayerNorm provides. The two compose without either
    mentioning the other: a gate work tile runs the same two-pass prologue as a dual tile (it is the
    same ``A``), and the region branch that routes its result lives one level up.

    **Only two seams are overridden**, and that is the design:

    * :meth:`mma_setup_fragments` -- builds the statistics partition and the shared scratch ONCE,
      outside the work-tile loop, and stages the gain/bias there. Everything it needs beyond the
      base's arguments arrives through `MmaContext`.
    * :meth:`mma_consume_work_tile` -- replaces the plain k-loop with the two-pass streaming loop.
      The producer's matching second sweep is :meth:`load_AB`.

    Everything else -- the scheduler, the epilogue, the store, the register permute, the ping-pong
    machinery this kernel refuses -- is inherited untouched.

    Args:
        *args: Forwarded to ``GemmSm90.__init__`` -- ``(acc_dtype, ab_dtype, tile_shape_mn,
            cluster_shape_mnk)`` and its keyword configuration.
        chunk_g: The weight layout, 1 or a multiple of 16. See `DualGatedGemmSm90`.
        fusion_variant: A member of :data:`FUSION_VARIANTS`.
        blk_k: The mainloop K tile. Must divide ``gemm_k``.
        gemm_k: The padded contraction extent. Must be positive.
        gemm_k_real: The true feature width. Must satisfy ``0 < gemm_k_real <= gemm_k``.
        n_dual_tiles: Dual work-tile count, or 0 for no output gate.
        gate3_n3: The output gate's N extent, or 0.
        **kwargs: Forwarded to ``GemmSm90.__init__``.

    Raises:
        NotImplementedError: If ``fusion_variant`` is ``"alg_fold"``.
        ValueError: If ``fusion_variant`` is not a member of :data:`FUSION_VARIANTS`, if ``blk_k``
            is not an allowed tile or does not divide ``gemm_k``, or if ``gemm_k``/``gemm_k_real``
            are inconsistent.
        AssertionError: If ``pingpong`` is requested -- see :meth:`__init__`.
    """

    Params = LayerNormDualGatedGemmParams

    #: Which `FUSION_VARIANTS` this class builds. The selector is declared on the PACK (so it exists
    #: before every variant does) and refused here, which is what let ``"alg_fold"`` be a legal
    #: parameter value while only one fusion existed. A subclass that implements another variant
    #: redeclares this; the refusal above then names the class that does build it rather than
    #: claiming it is unbuilt.
    _IMPLEMENTED_VARIANTS = ("prolog_ln",)

    #: Whether this fusion's row reduction survives the two-warpgroup schedule. False here, True
    #: only on the algebraic fold, and the difference is not a preference: `prolog_ln`'s normalize
    #: requires that the threads which REDUCE a row are the threads which RESCALE it, so its
    #: reduction must span every MMA warpgroup -- and a 256-thread reduction publishing through a
    #: barrier ping-pong sizes for 128 hangs. The fold's reduction feeds an epilogue correction, so
    #: it can be warpgroup-local and needs no cross-warpgroup agreement.
    _SUPPORTS_PINGPONG = False

    #: Whether the LayerNorm gain and bias are staged into shared memory. True here, because the
    #: prologue normalizes with them in the mainloop. False on the algebraic fold, which absorbed
    #: both into the WEIGHT on the host and never reads them -- staging them there is a barrier and
    #: two copies feeding buffers nothing loads. Skipping it changes no arithmetic, and the buffers
    #: are still RESERVED (see :meth:`_extra_smem_struct`) so the pipeline depth, and therefore the
    #: emitted schedule, is identical either way.
    _STAGES_LN_AFFINE = True

    #: The parent's declared ops, EXTENDED by the three the prologue needs. They are appended, so
    #: the op order -- which defines the params struct and the shared-memory map -- is the parent's
    #: unchanged; an op moved is an op handed another op's buffer.
    #:
    #: The three LayerNorm fields are NOT epilogue arithmetic. They are carried here because
    #: ``EpilogueParams`` is the only struct that reaches the kernel region, and a mainloop reading
    #: them off ``self`` instead would capture host-side values that do not survive the launch.
    _extra_param_fields = (
        *DualGatedGemmSm90._extra_param_fields,
        ("mWeight", Optional[cute.Tensor], None),
        ("mBias", Optional[cute.Tensor], None),
        ("eps", Float32, Float32(1e-5)),
    )

    def __init__(
        self,
        *args,
        chunk_g: int = 1,
        fusion_variant: str = "prolog_ln",
        blk_k: int = 64,
        gemm_k: int = 0,
        gemm_k_real: int = 0,
        n_dual_tiles: int = 0,
        gate3_n3: int = 0,
        **kwargs,
    ):
        """Validate the fusion's own parameters, then bind them in the base's single binding call.

        Purpose
            Every condition checked here is a silent-wrong-answer condition if it is not: a
            ``blk_k`` that does not divide ``gemm_k`` reads the gain past its end, a ``gemm_k_real``
            larger than ``gemm_k`` reads it out of range, and ``pingpong`` deadlocks (see below).
            None of them raises on its own inside the kernel.

        Semantics
            The parameters travel INTO ``GemmSm90.__init__`` rather than being bound separately --
            that constructor makes the one and only ``_bind_params`` call, and a second would raise.
            The ping-pong refusal is an assert AFTER that call because it reads the derived
            ``pingpong`` property.

        Args:
            See the class docstring.

        Returns:
            None.

        Raises:
            NotImplementedError: For ``fusion_variant="alg_fold"``, which is declared but not built.
            ValueError: For an unknown variant, an illegal or non-dividing ``blk_k``, a non-positive
                ``gemm_k``, or a ``gemm_k_real`` outside ``(0, gemm_k]``. The output-gate pair is
                checked by `DualGatedGemmSm90`, which owns it.
            AssertionError: If ``pingpong`` is set. The per-row reduction spans EVERY MMA warpgroup
                and publishes through the epilogue barrier, which under ping-pong covers only one --
                so a ping-pong build hangs rather than answering wrongly, and is refused here.
        """
        if fusion_variant not in FUSION_VARIANTS:
            raise ValueError(
                f"unknown fusion_variant {fusion_variant!r}; expected one of {FUSION_VARIANTS}."
            )
        if fusion_variant not in self._IMPLEMENTED_VARIANTS:
            raise NotImplementedError(
                f"fusion_variant={fusion_variant!r} is declared but not built by "
                f"{type(self).__name__}, which implements {self._IMPLEMENTED_VARIANTS}. The "
                f"algebraic rank-one correction is built by LayerNormDualGatedGemmAlgFoldSm90; the "
                f"front door selects it, so reaching this from the front door is a routing bug."
            )
        if blk_k not in ALLOWED_BLK_K:
            raise ValueError(f"blk_k must be one of {ALLOWED_BLK_K}; got {blk_k}.")
        if gemm_k <= 0:
            raise ValueError(
                f"gemm_k must be the positive PADDED contraction extent; got {gemm_k}. It sizes the "
                f"staged (K,) LayerNorm gain, which is a shared-memory allocation and so cannot be "
                f"a runtime value."
            )
        if gemm_k % blk_k != 0:
            raise ValueError(
                f"blk_k (={blk_k}) must divide gemm_k (={gemm_k}) exactly: a partial last K-tile "
                f"would make the in-place normalize index the staged (K,) gain past its end, which "
                f"is a wrong answer rather than a fault."
            )
        if not 0 < gemm_k_real <= gemm_k:
            raise ValueError(
                f"gemm_k_real must satisfy 0 < gemm_k_real <= gemm_k; got {gemm_k_real} with "
                f"gemm_k={gemm_k}."
            )
        super().__init__(
            *args,
            chunk_g=chunk_g,
            fusion_variant=fusion_variant,
            blk_k=blk_k,
            gemm_k=gemm_k,
            gemm_k_real=gemm_k_real,
            n_dual_tiles=n_dual_tiles,
            gate3_n3=gate3_n3,
            **kwargs,
        )
        assert self._SUPPORTS_PINGPONG or not self.pingpong, (
            f"{type(self).__name__} is cooperative-only. Its per-row reduction spans every MMA "
            "warpgroup and publishes through the epilogue barrier, which ping-pong sizes for ONE "
            "-- 256 threads would arrive at a 128-thread barrier and the second half would wait "
            "forever. The algebraic fold reduces WARPGROUP-LOCALLY and does support it."
        )

    # ── compile-time geometry ────────────────────────────────────────────────────────────────

    @property
    def _MMA_INST_TILE_K(self) -> int:
        """WGMMA K-instructions per mainloop K tile, so ``cta_tile_k`` becomes the declared ``blk_k``.

        The base derives ``cta_tile_k = atom_K * _MMA_INST_TILE_K`` with a class constant of 4,
        giving the plain GEMM's 64. Making it a property of ``blk_k`` is how the K tile becomes a
        declared parameter without touching the atom: the MMA stays ``m64nNk16`` and only the number
        of K-blocks per tile -- and therefore the SMEM box -- changes.

        Returns:
            ``blk_k // 16``: 4, 2 or 1.
        """
        return mma_inst_tile_k(self.blk_k)

    def _stats_num_threads(self) -> int:
        """Threads cooperating on the per-row reduction: every MMA warpgroup, or one under pingpong.

        Cooperatively, the threads that reduce a row must be the threads that rescale it -- they
        share one partition -- so the group is every MMA warpgroup and it publishes through the
        epilogue barrier, sized for exactly that group.

        **Under ping-pong the group is ONE warpgroup.** That schedule sizes the epilogue barrier for
        one AND puts the two warpgroups on DIFFERENT work tiles; both facts point the same way. Only
        a fusion declaring `_SUPPORTS_PINGPONG` reaches this branch.

        Returns:
            ``128`` under ping-pong, else ``mma_warp_groups * 128``.
        """
        groups = 1 if self.pingpong else self.mma_warp_groups
        return groups * self.num_threads_per_warp_group

    @property
    def _stats_slots(self) -> int:
        """Statistics buffers in the scratch: one per MMA warpgroup under ping-pong, else one.

        Read by BOTH the allocation (`_extra_smem_struct`) and the reservation (`_compute_stages`).
        Those two disagreeing does not fail at compile -- it overruns the dynamic shared-memory cap
        at LAUNCH, as ``cudaErrorInvalidValue``, which names shared memory nowhere.

        Returns:
            ``2`` under ping-pong, else ``1``.
        """
        return 2 if self.pingpong else 1

    def _stats_scratch_all(self, storage):
        """The whole ``(tile_M, 2, slots)`` statistics block, warpgroup mode OUTERMOST.

        Purpose
            What the EPILOGUE takes. `SmemColVecBroadcast.begin` slices both the column and the
            warpgroup itself, so it needs the block whole.

        Semantics
            The buffer index is the outermost mode so each warpgroup's ``(tile_M, 2)`` slice keeps
            ``tile_M`` at UNIT STRIDE, which is what the op's stride-``(1, 0)`` broadcast requires.

        Args:
            storage: The kernel's shared storage; ``storage.decoupled.s_stats`` must be the block
                `_extra_smem_struct` reserved.

        Returns:
            A ``(tile_M, 2, slots)`` fp32 shared tensor.
        """
        return storage.decoupled.s_stats.get_tensor(
            cute.make_layout((self.tile_shape_mn[0], 2, self._stats_slots))
        )

    def _stats_scratch(self, storage, warp_group_idx):
        """This warpgroup's ``(tile_M, 2)`` slice -- what the MAINLOOP writes.

        Purpose
            The mainloop already holds a warp-uniform ``warp_group_idx`` from its role, so it slices
            directly. The epilogue cannot (see :meth:`_stats_scratch_all`), and both go through the
            same block so the two cannot disagree about where a warpgroup's statistics live.

        Args:
            storage: The kernel's shared storage.
            warp_group_idx: This MMA warpgroup's index, warp-uniform. Ignored cooperatively, where
                there is one buffer.

        Returns:
            A ``(tile_M, 2)`` fp32 shared tensor: column 0 the row mean, column 1 its rstd.
        """
        block = self._stats_scratch_all(storage)
        return block[None, None, 0 if not self.pingpong else warp_group_idx]

    def _stats_barrier(self, warp_group_idx):
        """The barrier the row reduction publishes through, for THIS warpgroup.

        Purpose
            The one place the ping-pong hazard is contained. Cooperatively this is the epilogue
            barrier and nothing changes.

        Semantics
            **Under ping-pong the epilogue barrier must NOT be reused, and the trap is that its size
            would FIT.** Ping-pong sizes it for 4 warps == 128 threads, exactly a warpgroup-local
            reduction's count, so reusing it compiles, does not hang, and silently rendezvouses the
            two warpgroups with EACH OTHER while they are in different stages on different work
            tiles -- a wrong answer, not a stall.

        Args:
            warp_group_idx: This MMA warpgroup's index. Warp-uniform; it selects the barrier id, so
                a non-uniform value splits a warpgroup across two barriers and hangs. Ignored
                cooperatively.

        Returns:
            A barrier object exposing ``arrive_and_wait()``, sized to :meth:`_stats_num_threads`.
        """
        if not self.pingpong:
            return self.epilogue_barrier
        return _AlreadyOrderedBarrier()

    def _extra_smem_struct(self):
        """The prologue's shared memory: the statistics scratch and the staged gain/bias.

        Purpose
            Splices three buffers into the kernel's ``SharedStorage`` through the base's one hook,
            so the prologue needs no change to `GemmSm90.__call__`.

        Semantics
            The bias buffer is allocated whether or not a bias was supplied. That is deliberate: the
            reservation feeds :meth:`_compute_stages`, so allocating conditionally would make the
            mainloop's pipeline DEPTH -- and therefore the emitted schedule -- depend on whether the
            caller passed a bias. Reserving it always keeps the two configurations comparable.

            **The gain/bias pair is allocated only when the variant STAGES it**
            (:data:`_STAGES_LN_AFFINE`). That is not the same relaxation: the flag is a CLASS
            constant, so within one variant the depth still does not depend on any argument. A
            variant that folded the gain into the weight on the host -- `alg_fold` and its `x_gate`
            subclass -- reads neither buffer, and allocating them costs ``2 * gemm_k * 4`` bytes out
            of the budget :meth:`_compute_stages` divides into pipeline stages. Measured at
            ``gemm_k=384`` that is 3072 bytes, which rounds an EPILOGUE stage off (8 -> 7) and
            showed up as up to 4.8% against the kernel this package reproduces, whose Stage-C
            counterpart reserves the statistics scratch alone.

        Returns:
            A ``cute.struct`` type with ``s_stats``, plus ``s_weight`` and ``s_bias`` when the
            variant stages them, spliced in as the storage's extra field. Reached in the kernel as
            ``storage.decoupled.<name>``.
        """
        tile_m = self.tile_shape_mn[0]
        gemm_k = self.gemm_k
        fields = {
            "s_stats": cute.struct.Align[
                cute.struct.MemRange[Float32, tile_m * 2 * self._stats_slots], 16
            ],
        }
        if self._STAGES_LN_AFFINE:
            # `s_stats` stays FIRST so the fold path's scratch keeps the offset the staging path
            # gives it; the two vectors are appended, never inserted.
            fields["s_weight"] = cute.struct.Align[cute.struct.MemRange[Float32, gemm_k], 16]
            fields["s_bias"] = cute.struct.Align[cute.struct.MemRange[Float32, gemm_k], 16]

        class LnPrologueStorage:
            __annotations__ = fields

        return cute.struct(LnPrologueStorage)

    def _compute_stages(self, *args, **kwargs):
        """Reserve the prologue's shared memory before the base sizes the mainloop pipeline.

        Purpose
            The pipeline depth is whatever fits in the shared memory left over. The prologue's
            buffers are allocated out of the same budget, so they must be subtracted BEFORE the
            base divides -- an over-estimate here costs a pipeline stage, an under-estimate
            overruns the dynamic shared-memory cap at launch as ``cudaErrorInvalidValue``.

        Semantics
            An instance method overriding a ``classmethod``, because the reservation depends on
            ``gemm_k``, which is an instance parameter. ``args[7]`` is the shared-memory capacity,
            matching the base's positional signature.

            **The reservation is rounded UP to the buffer alignment**, and that is not defensive
            padding -- it is the actual cost. The prologue's buffers are spliced in AHEAD of ``sA``
            and ``sB``, which are ``Align[..., 1024]``, so inserting ``r`` bytes displaces those two
            by up to ``1024 - (r % 1024)`` bytes of fresh alignment padding on top of ``r``.
            Reserving only ``r`` leaves that padding unbudgeted; the struct then exceeds the dynamic
            shared-memory cap and the LAUNCH fails with ``cudaErrorInvalidValue`` -- not the compile,
            and with no reference to shared memory in the message.

            **The stage count is also capped at 64**, for a reason that is the base's and not this
            kernel's. ``GemmSm90._compute_stages`` budgets a FLAT 1024 bytes for every mbarrier
            array and alignment pad in the shared-storage struct, and the mainloop pipeline's own
            barriers cost ``16`` bytes per stage -- so past 64 stages that allowance is already
            spent and the struct exceeds what was budgeted for it. The unfused kernel survives
            because it lands just under the cap; adding ANY field tips it over, and the symptom is a
            launch that fails with ``cudaErrorInvalidValue`` -- no compile error, and no mention of
            shared memory. Measured at ``tile=(64, 32)``, ``blk_k=16``: 73 stages, a 233472-byte
            struct against a 232448-byte cap.

            64 is not a tuning choice, it is where that arithmetic breaks (``64 * 16 == 1024``). It
            cannot bind on a real shape: a pipeline that deep needs a tile so small that a stage
            costs under 3.5 KB, and the smallest production geometry here (``tile=(128, 256)``,
            ``blk_k=64``) costs 48 KB a stage and asks for four. And the two-pass loop has nothing to
            gain from a deeper ring in any case -- it holds one tile at a time by construction.

            The fix belongs in the base's estimate, and is deliberately NOT made there: correcting
            it would move ``ab_stage`` for every kernel in the package at large stage counts, and
            with it every pinned perf number, for a regime only this degenerate tile reaches.

        Args:
            *args: The base's positional arguments; ``args[7]`` is the capacity in bytes.
            **kwargs: Forwarded unchanged.

        Returns:
            ``(ab_stage, epi_stage, epi_c_stage)``.

        Raises:
            AssertionError: If fewer than two mainloop stages fit. One stage cannot overlap a load
                with a compute, so the two-pass streaming loop would serialize completely; it is
                refused loudly rather than shipped as a mystery slowdown.
        """
        args = list(args)
        align = self.buffer_align_bytes
        # The slot count MUST match `_extra_smem_struct`'s -- see `_stats_slots`. So must the
        # PRESENCE of the gain/bias pair: a variant that folds the gain into the weight on the host
        # allocates neither, and reserving for them anyway spends budget the epilogue ring would
        # otherwise get. Both sites read the same `_STAGES_LN_AFFINE`, which is what keeps them
        # from disagreeing -- and a disagreement here is a LAUNCH failure, not a compile error.
        reserve = stats_scratch_bytes(self.tile_shape_mn[0]) * self._stats_slots
        if self._STAGES_LN_AFFINE:
            reserve += ln_affine_bytes(self.gemm_k)
        args[7] = args[7] - (reserve + align - 1) // align * align
        ab_stage, epi_stage, epi_c_stage = super(LayerNormDualGatedGemmSm90, self)._compute_stages(
            *args, **kwargs
        )
        ab_stage = min(ab_stage, _MAX_AB_STAGE)
        assert ab_stage >= 2, (
            f"the fused LayerNorm prologue needs at least 2 mainloop stages; the tile "
            f"{self.tile_shape_mn} with blk_k={self.blk_k} and a staged (K={self.gemm_k},) "
            f"gain/bias leaves room for {ab_stage}. Reduce tile_N or blk_k."
        )
        return ab_stage, epi_stage, epi_c_stage

    # ── epilogue plumbing ────────────────────────────────────────────────────────────────────

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        """`DualGatedGemmSm90`'s epilogue terms plus the LayerNorm prologue's three inputs.

        The first fourteen fields are the parent's, in the parent's ORDER -- a NamedTuple is
        positional on the wire, so re-ordering them would silently hand each field the value of
        another. Only the last three are new, and they are NOT epilogue arithmetic: they are carried
        here because this struct is the only one that crosses into the kernel region, and a mainloop
        that read them off ``self`` would capture host-side values that do not survive the launch.

        Attributes:
            mPostAct, act_fn, alpha, beta, mRowVecBroadcast, mColVecBroadcast, mMaskColVec, mBiasUp,
                mBiasGate, mPostAct3, act_fn_3, mRowVecBroadcast3, rounding_mode, sr_seed: See
                `DualGatedGemmSm90.EpilogueArguments`.
            mWeight: ``(K,)`` fp32 LayerNorm gain. **fp32 is required, not converted** -- a 16-bit
                gain would lose more precision than the normalize it scales. Required.
            mBias: ``(K,)`` fp32 LayerNorm bias, or None. None compiles the add out entirely.
            eps: The variance floor. A RUNTIME scalar: baking it would key the artifact on it.

        Note:
            The `Constexpr` fields are erased from the ABI, so at LAUNCH they must be passed None.
        """

        mPostAct: cute.Tensor
        act_fn: cutlass.Constexpr[Optional[Callable]] = None
        alpha: Optional[Float32 | cute.Tensor] = None
        beta: Optional[Float32 | cute.Tensor] = None
        mRowVecBroadcast: Optional[cute.Tensor] = None
        mColVecBroadcast: Optional[cute.Tensor] = None
        mMaskColVec: Optional[cute.Tensor] = None
        mBiasUp: Optional[cute.Tensor] = None
        mBiasGate: Optional[cute.Tensor] = None
        mPostAct3: Optional[cute.Tensor] = None
        act_fn_3: cutlass.Constexpr[Optional[Callable]] = None
        mRowVecBroadcast3: Optional[cute.Tensor] = None
        rounding_mode: cutlass.Constexpr[int] = RoundingMode.RN
        sr_seed: Optional[Int32 | cute.Tensor] = None
        mWeight: Optional[cute.Tensor] = None
        mBias: Optional[cute.Tensor] = None
        eps: Float32 = Float32(1e-5)

    def epi_to_underlying_arguments(self, args, *, loc=None, ip=None):
        """Lower the launch arguments into the traced params, adding the prologue's three fields.

        Purpose
            `DualGatedGemmSm90.gated_params_dict` already performs every geometric check, latches
            the post-activation attributes and contributes the output gate's own entry; this extends
            its dict rather than restating any of it, which is what that seam exists for.

        Args:
            args: This launch's :class:`EpilogueArguments`.
            loc: Optional DSL source location.
            ip: Optional DSL insertion point.

        Returns:
            An ``EpilogueParams`` with one entry per declared op, the two activations, and the three
            prologue fields.

        Raises:
            AssertionError: Propagated from `gated_params_dict` for a refused geometry or an
                output-gate presence that disagrees with how the functor was built.
        """
        d = self.gated_params_dict(args)
        d["mWeight"] = args.mWeight
        d["mBias"] = args.mBias
        d["eps"] = args.eps
        return self.EpilogueParams(**d)

    # ── the fused mainloop: two seams, nothing else ──────────────────────────────────────────

    def mma_setup_fragments(
        self, tiled_mma, sA, sB, warp_group_thread_layout, warp_group_idx, mma_ctx=None
    ):
        """Build the base fragments, then the prologue's partition and scratch, and stage the gain.

        Purpose
            Everything here is per-CTA and tile-independent, so it must happen ONCE, outside the
            work-tile loop. Staging the ``(K,)`` gain per work tile instead would re-read it from
            global on every tile of a persistent grid.

        Semantics
            A plain ``def`` overriding a plain ``def``, as the base requires: ``@cute.jit`` flattens
            arguments into MLIR values and cannot accept or return an `MmaFragments`. The
            consequence is that the DSL preprocessor does not run here, so this body contains no
            dynamic ``if`` and no ``cutlass.range_constexpr`` -- the one construct that needs both,
            the strided gain copy, is delegated to the ``@cute.jit`` helper
            :func:`~fold_cp_ops._internal.ln_prologue.stage_ln_affine`.

            The staging ends with a barrier, so **every MMA warpgroup thread must reach this
            method**. That is automatic: the role calls it before any divergence.

        Args:
            tiled_mma: The MMA atom, sliced by warpgroup.
            sA: The staged A tile; sets ``tCrA``'s layout and is kept for the mainloop.
            sB: The staged B tile.
            warp_group_thread_layout: Maps a warpgroup index to its first thread.
            warp_group_idx: Which MMA warpgroup this is.
            mma_ctx: The `MmaContext`. **Required here** despite the base's default: it is the only
                route to the prologue's shared memory (through ``storage``) and to ``eps`` and the
                LayerNorm tensors (through ``epilogue_params``). Passing None raises rather than
                silently building a functor that normalizes with uninitialised statistics.

        Returns:
            An `MmaFragments` whose ``extra`` is the :class:`_LnPrologueState` the mainloop unpacks.

        Raises:
            ValueError: If ``mma_ctx`` is None.
        """
        if mma_ctx is None:
            raise ValueError(
                "LayerNormDualGatedGemmSm90.mma_setup_fragments requires an MmaContext: the "
                "prologue's shared scratch, the LayerNorm gain and eps are reachable only through "
                "it. A direct caller (a test driving the seam without a role) must build one."
            )
        frags = super(LayerNormDualGatedGemmSm90, self).mma_setup_fragments(
            tiled_mma, sA, sB, warp_group_thread_layout, warp_group_idx, mma_ctx
        )
        storage, params, tidx = mma_ctx.storage, mma_ctx.epilogue_params, mma_ctx.tidx
        tile_m = self.cta_tile_shape_mnk[0]
        nthreads = self._stats_num_threads()
        # Under ping-pong the scratch holds one block per warpgroup and this takes THIS one.
        # `tidx` already arrives warpgroup-local (the role does `tidx % num_threads_per_warp_group`),
        # so the row ownership inside `finalize_row_stats` needs no adjustment here.
        stats_smem = self._stats_scratch(storage, warp_group_idx)
        # Both are None on a variant that does not stage the affine pair -- `_extra_smem_struct`
        # does not allocate them there, so reaching for the field is an AttributeError, not a
        # zero-cost read. The consumers that use them are gated on the same class constant.
        s_weight = (
            storage.decoupled.s_weight.get_tensor(cute.make_layout((self.gemm_k,)))
            if const_expr(self._STAGES_LN_AFFINE)
            else None
        )
        s_bias = (
            storage.decoupled.s_bias.get_tensor(cute.make_layout((self.gemm_k,)))
            if const_expr(self._STAGES_LN_AFFINE) and params.mBias is not None
            else None
        )
        if const_expr(self._STAGES_LN_AFFINE):
            stage_ln_affine(
                s_weight,
                s_bias,
                params.mWeight,
                params.mBias,
                self.gemm_k,
                self.gemm_k_real,
                nthreads,
                tidx,
                self._stats_barrier(warp_group_idx),
            )
        tiled_copy = stats_tiled_copy(
            self.a_dtype,
            (tile_m, self.cta_tile_shape_mnk[2]),
            nthreads,
            STATS_THREADS_PER_ROW,
        )
        frags.extra = _LnPrologueState(
            thr_copy=tiled_copy.get_slice(tidx),
            sA=sA,
            stats_smem=stats_smem,
            s_weight=s_weight,
            s_bias=s_bias,
            tidx=tidx,
            n_elems=Int32(self.gemm_k_real),
            eps=params.eps,
        )
        return frags

    def mma_consume_work_tile(
        self, frags, ab_pipeline, ab_read_state, len_k, warp_group_idx, mma_carry=None
    ):
        """Drain one work tile through the two-pass prologue instead of the plain k-loop.

        Purpose
            The seam. Everything around it -- the scheduler loop, the epilogue, the store -- is the
            base's, so this method IS the fusion on the consumer side.

        Semantics
            A plain ``def`` unpacking `MmaFragments.extra` and forwarding its fields, because the
            DSL cannot pass an object across a ``@cute.jit`` boundary. It carries nothing between
            work tiles: the statistics are recomputed per tile from that tile's own rows, so the
            base's ``None`` carry is correct and is returned unchanged.

        Args:
            frags: The `MmaFragments` from :meth:`mma_setup_fragments`.
            ab_pipeline: The mainloop staging pipeline.
            ab_read_state: The consumer state, threaded across work tiles.
            len_k: The contraction extent. Unused: this consumer derives its trip count from the
                compile-time ``gemm_k // blk_k`` instead, which is EQUAL to the producer's
                ``_k_tile_cnt(len_k)`` by construction (``gemm_k`` is ``len_k`` rounded up to a
                whole number of tiles). The two counts must agree -- the producer stages
                ``2 * k_tile_cnt`` tiles for this loop's ``2 * k_tile_cnt`` waits, and a mismatch is
                a deadlock, not a wrong answer.
            warp_group_idx: Which MMA warpgroup. Unused: ping-pong is refused at construction.
            mma_carry: The previous call's carry; always None here.

        Returns:
            ``(ab_read_state, mma_carry)``.
        """
        st = frags.extra
        ab_read_state = self.mma_prolog_ln(
            ab_pipeline,
            ab_read_state,
            frags.mma_fn,
            st.thr_copy,
            st.sA,
            st.stats_smem,
            st.s_weight,
            st.s_bias,
            st.tidx,
            st.n_elems,
            st.eps,
        )
        return ab_read_state, mma_carry

    @cute.jit
    def mma_prolog_ln(
        self,
        ab_pipeline,
        ab_read_state,
        mma_fn,
        thr_copy,
        sA,
        stats_smem,
        s_weight,
        s_bias,
        tidx,
        n_elems,
        eps,
    ):
        """The two-pass streaming mainloop: reduce over A, then normalize A and multiply.

        Purpose
            LayerNorm needs the WHOLE row before it can scale any of it, and the row does not fit in
            shared memory at useful K. This is the resolution: stream A's k-tiles twice through the
            SAME small pipeline -- once to reduce, once to normalize and multiply -- so nothing has
            to stay resident and the kernel scales to any K.

        Semantics
            Pass 1 waits each staged tile, adds its per-row ``Sum x`` / ``Sum x^2`` into registers,
            and RELEASES the tile immediately: nothing is held, so the ring keeps turning at its
            natural depth. The finalize then publishes ``(mu, rstd)``.

            Pass 2 waits the RE-READ tile (A comes from L2 the second time, which is what makes the
            second sweep cheap), rewrites it in place as ``LayerNorm(A)``, fences and barriers so
            every warpgroup sees the rewrite, issues the WGMMA, and only then releases.

            **Pass 2 deliberately does not pipeline the WGMMA.** ``wait_group(0)`` before the
            release is what makes the release safe: the tile it frees is the tile the MMA just read,
            and the normalize of a LATER k-tile writes into that same shared buffer. Releasing while
            the MMA is still in flight lets the producer refill the stage under a running WGMMA --
            a silent wrong answer that appears only under pipeline pressure. This costs the
            mainloop's usual MMA/load overlap, and that cost is the price of the fusion.

            The ONE ``ab_read_state`` crosses both passes, so its ``2 * k_tile_cnt`` waits stay in
            lock-step with the producer's ``2 * k_tile_cnt`` loads.

        Args:
            ab_pipeline: The mainloop staging pipeline.
            ab_read_state: The consumer state; advanced ``2 * k_tile_cnt`` times and returned.
            mma_fn: The WGMMA closure, called with ``(A_idx, B_idx, zero_init)``.
            thr_copy: This thread's slice of the statistics tiled copy.
            sA: The staged A tile, ``(tile_M, tile_K, ab_stage)``. **Written**, by the normalize.
            stats_smem: The ``(tile_M, 2)`` fp32 scratch.
            s_weight: The staged ``(gemm_k,)`` gain.
            s_bias: The staged ``(gemm_k,)`` bias, or None.
            tidx: This thread's index within the MMA warpgroups.
            n_elems: The per-row element count (the TRUE feature width).
            eps: The variance floor.

        Returns:
            The advanced consumer state.
        """
        # A COMPILE-TIME trip count, so both passes unroll. This buys nothing in generality that is
        # not already spent: `gemm_k` is baked anyway (it sizes the staged gain), so deriving the
        # count from it adds no artifact a dynamic loop would have saved -- and it is worth 7-20%.
        # Measured against the unfused sibling's dynamic form on six shapes: the fused kernel with a
        # runtime `cutlass.range` ran 1.07-1.20x slower than the kernel it reproduces, worst at
        # SMALL k-tile counts where the per-iteration overhead is least amortized. Unrolling closed
        # it. Revisit ONLY together with the staged gain -- a dynamic loop over a baked K is the
        # worst of both.
        k_tile_cnt = const_expr(self.gemm_k // self.blk_k)
        row_sum, row_sqsum = make_row_stat_partials(thr_copy, sA[None, None, 0])

        # PASS 1 -- reduce, releasing each tile as soon as its values are in registers.
        peek = Boolean(True)
        if const_expr(0 < k_tile_cnt):
            peek = ab_pipeline.consumer_try_wait(ab_read_state)
        for k_idx in cutlass.range_constexpr(k_tile_cnt):
            ab_pipeline.consumer_wait(ab_read_state, peek)
            stage = ab_read_state.index
            row_sum, row_sqsum = accumulate_row_stats(
                thr_copy, sA[None, None, stage], row_sum, row_sqsum
            )
            ab_pipeline.consumer_release(ab_read_state)
            ab_read_state.advance()
            peek = Boolean(True)
            if const_expr(k_idx + 1 < k_tile_cnt):
                peek = ab_pipeline.consumer_try_wait(ab_read_state)
        finalize_row_stats(
            stats_smem,
            tidx,
            row_sum,
            row_sqsum,
            n_elems,
            eps,
            STATS_THREADS_PER_ROW,
            self._stats_num_threads(),
            self.epilogue_barrier,
            # The persistent-tile stats-race guard the upstream `staged` kernel emits
            # (`_stats_race_guard`, True on its default streaming mode). Restored HERE and
            # not on the alg_fold siblings, whose `stagec` counterpart has no such guard.
            leading_barrier=True,
        )

        # PASS 2 -- normalize in place, then multiply. See the docstring for why the WGMMA is
        # drained before the release rather than pipelined across k-tiles.
        peek = Boolean(True)
        if const_expr(0 < k_tile_cnt):
            peek = ab_pipeline.consumer_try_wait(ab_read_state)
        for k_idx in cutlass.range_constexpr(k_tile_cnt):
            ab_pipeline.consumer_wait(ab_read_state, peek)
            stage = ab_read_state.index
            normalize_tile(
                sA[None, None, stage],
                stats_smem,
                s_weight,
                s_bias,
                thr_copy,
                Int32(k_idx),
                (self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[2]),
                self.a_dtype,
            )
            cute.arch.fence_view_async_shared()
            self.epilogue_barrier.arrive_and_wait()
            mma_fn(A_idx=stage, B_idx=stage, zero_init=Boolean(k_idx == 0))
            cute.nvgpu.warpgroup.wait_group(0)
            ab_pipeline.consumer_release(ab_read_state)
            ab_read_state.advance()
            peek = Boolean(True)
            if const_expr(k_idx + 1 < k_tile_cnt):
                peek = ab_pipeline.consumer_try_wait(ab_read_state)
        return ab_read_state

    @cute.jit
    def load_AB(
        self,
        ab_pipeline,
        ab_producer_state,
        copy_A,
        copy_B,
        k_tile_cnt,
        copy_SFA=None,
        copy_SFB=None,
    ):
        """Stage every k-tile TWICE -- the producer half of the two-pass prologue.

        Purpose
            The consumer sweeps A twice, so the producer must supply it twice. Expressing that as
            two calls to the base loop, rather than as a second loop written here, is what keeps
            the acquire/commit protocol in one place.

        Semantics
            B is loaded on BOTH passes even though pass 1 never reads it. That is not waste that
            could be removed: the pipeline's per-stage transaction count is ``A + B`` bytes, fixed
            at construction, and a pass that committed fewer bytes would leave the consumer waiting
            on a barrier that never completes. It is a HANG, not a slow path, so the second load
            stays.

        Args:
            ab_pipeline: The mainloop pipeline.
            ab_producer_state: The producer state; advanced ``2 * k_tile_cnt`` times and returned.
            copy_A: A's copy closure. Never None here -- the fused prologue reads A.
            copy_B: B's copy closure.
            k_tile_cnt: k-tiles per PASS. Must be the same value the consumer derives, or the two
                deadlock against each other.
            copy_SFA: Blockscale closure; None on SM90 and unused.
            copy_SFB: Likewise.

        Returns:
            The advanced producer state.
        """
        ab_producer_state = super(LayerNormDualGatedGemmSm90, self).load_AB(
            ab_pipeline, ab_producer_state, copy_A, copy_B, k_tile_cnt, copy_SFA, copy_SFB
        )
        return super(LayerNormDualGatedGemmSm90, self).load_AB(
            ab_pipeline, ab_producer_state, copy_A, copy_B, k_tile_cnt, copy_SFA, copy_SFB
        )

    # ── the fused output gate: one region branch, no second mainloop ─────────────────────────


# ───────────────────────────────── the compiled entry ─────────────────────────────────


class LayerNormDualGatedGemmAlgFoldSm90(LayerNormDualGatedGemmSm90):
    """The ALGEBRAIC fold: multiply the raw activation, repair it rank-one in the epilogue.

    Purpose
        The second `FUSION_VARIANTS` member. Where `prolog_ln` normalizes the activation in shared
        memory and multiplies the result, this multiplies the RAW activation by a pre-folded weight
        and corrects the difference in the epilogue::

            LN(x) @ B  =  r*(x @ Bw) - s*c + d
                Bw = diag(w_ln) @ B     c = colsum(Bw)     d = b_ln @ B
                r = rstd,  s = rstd*mu, both accumulated per row IN this mainloop

        `Bw`, `c`, `d` and the interleaved projection bias come from `build_folded_dual_operands`,
        which produces all four in ONE launch because the weight is loop-invariant.

    Semantics
        **It is ONE sweep over A where `prolog_ln` is two, and that is the whole speed argument.**
        The statistics and the WGMMA read the SAME staged tile, so the reduction rides along with
        the multiply instead of forcing a second pass. It also keeps the base's MMA/load overlap,
        which `prolog_ln` must give up: `mma_prolog_ln`'s second pass REWRITES the staged tile, so
        it has to drain the WGMMA before releasing, and that costs the pipelining.

        **Three overrides, no more.** The mainloop consumer, the epilogue's pre-activation combine,
        and the shared-memory hand-off that lets the second read what the first wrote. Everything
        else -- the scheduler loop, the gate, the mask, the register permute, the store, the output
        gate -- is inherited unchanged, which is what
        ``test_the_fusion_overrides_only_the_two_declared_mainloop_seams`` exists to keep true.

        **It is NOT bit-comparable with `prolog_ln`.** Different arithmetic order, different
        rounding points: `prolog_ln` rounds a normalized activation to 16 bits before the MMA and
        this never does, while this accumulates raw `x` and corrects. Cross-variant tests compare
        BOUNDS (`alg_fold_error_bound` vs `fused_ln_gated_error_bound`), never bytes.

    Attributes:
        _IMPLEMENTED_VARIANTS: ``("alg_fold",)`` -- redeclared so the base's refusal, which is keyed
            on this, admits the variant here and continues to refuse it on the parent.
        _epi_ops: The parent's tuple with four ops APPENDED. Appending is the only safe edit: the
            order defines both the generated params struct and the shared-memory map, so an op moved
            is an op handed another op's buffer.
    """

    _IMPLEMENTED_VARIANTS = ("alg_fold",)

    #: The fold's reduction feeds an epilogue correction rather than a normalize, so it does not
    #: need the threads that reduced a row to be the threads that rescale it. That is the whole
    #: reason it can be warpgroup-local, and therefore the whole reason ping-pong is reachable here
    #: and refused on the parent. Measured worth ~4.5% at m >= 65536 and up to 10%.
    _SUPPORTS_PINGPONG = True

    #: The fold absorbed the LayerNorm gain and bias into the WEIGHT on the host, so this kernel
    #: never reads ``mWeight``/``mBias``. Staging them would be a barrier and two copies feeding
    #: buffers nothing loads -- and under ping-pong the barrier would be an active hazard.
    _STAGES_LN_AFFINE = False

    #: This kernel's K extent is baked (see `_compile_layernorm_dual_gated_gemm`), so the producer's
    #: k-loop trip count is knowable at compile time. Keeping it that way is what the kernel this
    #: fusion reproduces does -- its ``len_k = mA_mkl.shape[1]`` carries no cast -- and it is the
    #: difference between a bottom-tested counted loop and a top-tested one with a ``BREAK``. See
    #: `GemmSm90._KEEP_STATIC_LEN_K` for the measurement.
    _KEEP_STATIC_LEN_K = True

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        """The parent's terms plus the four the rank-one correction reads.

        A NamedTuple is positional on the wire and cannot be extended by subclassing, so the
        parent's seventeen fields are RESTATED in the parent's exact order and the four new ones
        appended. Re-ordering any of them silently hands each field another's value.

        Attributes:
            mPostAct … eps: See `LayerNormDualGatedGemmSm90.EpilogueArguments`, unchanged.
            sRstd, sScale: Placeholders for the two `SmemColVecBroadcast` ops. **Always None.** Their
                data is the mainloop's shared-memory scratch, supplied by
                :meth:`LayerNormDualGatedGemmAlgFoldSm90.epi_get_smem_tensors`; the fields exist only
                because the composition machinery expects one per declared op.
            mColsum: ``(l, 2N)`` fp32 ``c = colsum(Bw)``. Required -- without it the correction
                subtracts nothing and the output is the raw ``r*(x@Bw)``, which is finite and wrong.
            mDbias: ``(l, 2N)`` fp32 ``d = b_ln @ B``, or None when there is no LayerNorm bias. None
                compiles the term out; a zero tensor instead would cost a load to add nothing.
            mColsum3, mDbias3: The output gate's own ``(l, n3)`` fp32 ``c3``/``d3``, or None
                without a gate. **Separate tensors indexed from column 0, NOT a wider `mColsum`
                sliced by region** -- `run_epilogue` re-bases the gate arm's n-block by
                ``n_dual_tiles``, so on a gate tile every row-vector op sees a LOCAL coordinate.
                A single wide vector would therefore be read at the DUAL region's columns on every
                gate tile: a finite, plausible, entirely wrong output. This mirrors
                `mRowVecBroadcast3`, which is `b3` for the same reason.
        """

        mPostAct: cute.Tensor
        act_fn: cutlass.Constexpr[Optional[Callable]] = None
        alpha: Optional[Float32 | cute.Tensor] = None
        beta: Optional[Float32 | cute.Tensor] = None
        mRowVecBroadcast: Optional[cute.Tensor] = None
        mColVecBroadcast: Optional[cute.Tensor] = None
        mMaskColVec: Optional[cute.Tensor] = None
        mBiasUp: Optional[cute.Tensor] = None
        mBiasGate: Optional[cute.Tensor] = None
        mPostAct3: Optional[cute.Tensor] = None
        act_fn_3: cutlass.Constexpr[Optional[Callable]] = None
        mRowVecBroadcast3: Optional[cute.Tensor] = None
        rounding_mode: cutlass.Constexpr[int] = RoundingMode.RN
        sr_seed: Optional[Int32 | cute.Tensor] = None
        mWeight: Optional[cute.Tensor] = None
        mBias: Optional[cute.Tensor] = None
        eps: Float32 = Float32(1e-5)
        sRstd: Optional[cute.Tensor] = None
        sScale: Optional[cute.Tensor] = None
        mColsum: Optional[cute.Tensor] = None
        mDbias: Optional[cute.Tensor] = None
        mColsum3: Optional[cute.Tensor] = None
        mDbias3: Optional[cute.Tensor] = None

    #: The parent's ops plus what the correction reads. ``sRstd``/``sScale`` are the two columns of
    #: the mainloop's ``(tile_M, 2)`` scratch -- supplied by :meth:`epi_get_smem_tensors`, not
    #: allocated -- and ``mColsum``/``mDbias`` are the fold's ``c`` and ``d``, ordinary ``(l, 2N)``
    #: fp32 row vectors. Appended, so every inherited op keeps its index.
    #:
    #: The gate's ``c3``/``d3`` are `Gate3RowVecLoad`, not `RowVecLoad`, for the same reason `b3`
    #: is: that op slices at the gate arm's RE-BASED n-block and predicates the trailing partial
    #: tile, which is what an ``(n3,)`` vector indexed from column 0 needs.
    _epi_ops = (
        *LayerNormDualGatedGemmSm90._epi_ops,
        SmemColVecBroadcast("sRstd", slot=0),
        SmemColVecBroadcast("sScale", slot=1),
        RowVecLoad("mColsum"),
        RowVecLoad("mDbias"),
        Gate3RowVecLoad("mColsum3"),
        Gate3RowVecLoad("mDbias3"),
    )

    def epi_get_smem_tensors(self, params, storage):
        """Hand the two statistics ops the MAINLOOP's scratch instead of an epilogue buffer.

        Purpose
            The one seam that lets a value computed in the k-loop be read in the epilogue. Every
            other epi-op owns its buffer in ``storage.epi``; these two read
            ``storage.decoupled.s_stats``, which the mainloop wrote and the epilogue must not
            duplicate.

        Semantics
            Calls the base for every op, then substitutes the two `SmemColVecBroadcast` entries.
            The base already receives the WHOLE storage object and merely narrows to ``storage.epi``
            at the per-op call, so this needs no change to `EpiOp.get_smem_tensor`'s signature and
            leaves every other fusion untouched -- `prolog_ln` does not override this method, so its
            emitted code cannot move.

            Substituting by POSITION, from the same ``_epi_ops`` tuple the base iterates, is what
            keeps the two in step: a hard-coded index would silently shift the moment an op is
            appended above.

        Args:
            params: The traced ``EpilogueParams``.
            storage: The kernel's shared storage. ``storage.decoupled.s_stats`` must be the
                ``(tile_M, 2)`` fp32 scratch this functor's mainloop publishes; a different layout
                here is read as if it were that one, giving a plausible wrong statistic rather than
                a fault.

        Returns:
            One tensor per non-`Scalar` op, in declaration order -- the base's tuple with the two
            statistics entries replaced.
        """
        tensors = list(
            super(LayerNormDualGatedGemmAlgFoldSm90, self).epi_get_smem_tensors(params, storage)
        )
        # The FULL (tile_M, 2, slots) block, NOT a per-warpgroup slice. `SmemColVecBroadcast.begin`
        # picks both the column and the warpgroup itself, from a fresh `warp_idx()` read taken
        # inside the epilogue region -- which is what avoids carrying a warp-group index across it.
        # Slicing here instead would capture that value in the kernel body.
        stats = self._stats_scratch_all(storage)
        non_scalar = [op for op in self._epi_ops if not isinstance(op, Scalar)]
        for i, op in enumerate(non_scalar):
            if isinstance(op, SmemColVecBroadcast):
                tensors[i] = stats
        return tuple(tensors)

    @cute.jit
    def load_AB(
        self,
        ab_pipeline,
        ab_producer_state,
        copy_A,
        copy_B,
        k_tile_cnt,
        copy_SFA=None,
        copy_SFB=None,
    ):
        """Stage every k-tile ONCE, skipping the parent's double-staging.

        Purpose
            The producer must supply exactly what the consumer takes. `prolog_ln` sweeps A twice, so
            its producer stages twice; this fusion sweeps ONCE, so inheriting that producer would
            stage ``2 * k_tile_cnt`` tiles against ``k_tile_cnt`` waits.

            **That mismatch is a HANG, not a wrong answer** -- the producer fills the ring and blocks
            forever on an acquire nothing will release. It is also invisible to every host-side test,
            because the kernel compiles perfectly and then never returns. This override is the whole
            fix, and it is why the one-pass/two-pass difference is stated here rather than inherited.

        Semantics
            Delegates to the GRANDPARENT -- `DualGatedGemmSm90`'s plain single-pass producer -- by
            naming `LayerNormDualGatedGemmSm90` in the ``super()`` call, which skips exactly one level
            of the MRO. The explicit two-argument form is required, not stylistic: the DSL re-derives
            a ``@cute.jit`` function from source, and the zero-argument form then resolves against
            ``type(self)``, so on a subclass it re-enters the same method and recurses until the
            stack ends.

        Args:
            ab_pipeline: The mainloop pipeline.
            ab_producer_state: The producer state; advanced ``k_tile_cnt`` times and returned.
            copy_A: A's copy closure.
            copy_B: B's copy closure.
            k_tile_cnt: k-tiles per pass. Must be the value the consumer derives from
                ``gemm_k // blk_k``, or the two deadlock against each other.
            copy_SFA: Blockscale closure; None on SM90 and unused.
            copy_SFB: Likewise.

        Returns:
            The advanced producer state.
        """
        return super(LayerNormDualGatedGemmSm90, self).load_AB(
            ab_pipeline,
            ab_producer_state,
            copy_A,
            copy_B,
            k_tile_cnt,
            copy_SFA,
            copy_SFB,
        )

    def mma_consume_work_tile(
        self, frags, ab_pipeline, ab_read_state, len_k, warp_group_idx, mma_carry=None
    ):
        """Drain one work tile in a SINGLE pass, reducing and multiplying the same staged tile.

        Purpose
            The mainloop side of the fusion. `prolog_ln` needs two sweeps because it must see a
            whole row before it can scale any of it; this one does not scale anything, so the
            statistics can ride along with the multiply.

        Semantics
            Delegates to the base's own consumer loop, passing an ``on_ktile`` closure. The base
            calls it once per k-tile with the pipeline stage, AFTER the WGMMA is issued and BEFORE
            the group wait, so the reduction's reads overlap the flying WGMMA rather than serializing
            with the MMA warps' operand fetch. `accumulate_row_stats` only reads, which is exactly
            the contract that hook documents.

            The partials are rmem tensors updated IN PLACE, so the closure ignores the return value
            and the base's runtime k-loop needs no loop-carried variable for them.

            **The finalize runs through ``on_drain``, not after `mma` returns, and the placement is
            worth ~4.7% at D=128.** It is accumulator-independent -- register partials in, SMEM
            stats scratch out -- so it is legal in the window between the last WGMMA issue and the
            drain's ``wait_group(0)``, where it executes for free in the MMA's shadow. Calling it
            after `mma` returns leaves that wait naked: measured, the drain's
            ``WARPGROUP.DEPBAR.LE gsb0, 0x0`` carried 240 barrier-stall samples against 0 for the
            kernel this reproduces, whose own comment says "finalize stats into the SMEM scratch
            while the last WGMMAs drain". This port had lost that ordering.

            After the loop the partials become ``(r, s) = (rstd, rstd*mu)`` in the shared
            scratch, where the epilogue's two `SmemColVecBroadcast` ops read them. **Forming the
            product HERE is a performance decision, not a formatting one:** it costs one multiply
            per ROW, where forming it in the epilogue costs one per output ELEMENT -- 512 extra warp
            instructions per 128x128 tile, ~9% of the kernel at K=128. `main` publishes the same
            pair for the same reason.

            A plain ``def``, not ``@cute.jit``: the base requires it, since the DSL flattens
            arguments into MLIR values and cannot carry an `MmaFragments` across the boundary. So
            this body contains no dynamic ``if`` and no ``range_constexpr``.

        Args:
            frags: The `MmaFragments` from :meth:`mma_setup_fragments`; ``extra`` is the
                `_LnPrologueState`.
            ab_pipeline: The mainloop staging pipeline.
            ab_read_state: The consumer state, threaded across work tiles.
            len_k: The contraction extent, converted to the tile count by ``_k_tile_cnt`` -- the
                SAME count the producer stages. A mismatch is a deadlock, not a wrong answer.
            warp_group_idx: Which MMA warpgroup, forwarded to the base.
            mma_carry: The previous call's carry; always None, and returned unchanged, because the
                statistics are recomputed per work tile from that tile's own rows.

        Returns:
            ``(ab_read_state, mma_carry)``.
        """
        st = frags.extra
        row_sum, row_sqsum = make_row_stat_partials(st.thr_copy, st.sA[None, None, 0])

        def on_ktile(stage):
            accumulate_row_stats(st.thr_copy, st.sA[None, None, stage], row_sum, row_sqsum)

        # The trip count goes in TWICE, and the second one is the point. `gemm_k` and `blk_k` are
        # both compile keys, so the count is known statically -- and handing it over unrolls the
        # k-loop, which is what keeps `row_sum`/`row_sqsum` straight-line SSA instead of state
        # carried across a dynamic MLIR region. The upstream's consumer unrolls for the same
        # reason; leaving it dynamic measured ~10% slower cooperatively and worse under ping-pong.
        def on_drain():
            finalize_row_stats(
                st.stats_smem,
                st.tidx,
                row_sum,
                row_sqsum,
                st.n_elems,
                st.eps,
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
            k_tile_cnt_const=self.gemm_k // self.blk_k,
            on_drain=on_drain,
        )
        return ab_read_state, mma_carry

    @cute.jit
    def epi_combine_preact(self, params, epi_loop_tensors, tRS_rD, tRS_rC=None):
        """Correct the raw accumulator rank-one, then add the projection bias.

        Purpose
            The epilogue side of the fusion, and the seam
            `GemmGatedMixin.epi_combine_preact` names for exactly this variant.

        Semantics
            In place and in fp32::

                tRS_rD[i] = r[i]*tRS_rD[i] - r[i]*mu[i]*c[i] + d[i] + bias[i]

            **The correction precedes the bias, and that ordering is not cosmetic.** ``c`` and ``d``
            are properties of the FOLD, so they belong inside the identity being repaired; the
            projection bias is added to the finished pre-activation. Adding the bias first would put
            it inside the ``r*`` scaling and scale it by the row's ``rstd``.

            ``s = r*mu`` is formed here rather than stored, because the shared helper
            `finalize_row_stats` publishes ``mean`` and ``rstd`` -- one multiply per element against
            not forking a helper `prolog_ln` also uses.

            The LayerNorm bias term ``d`` and the projection bias are each ``const_expr``-pruned when
            absent, so a kernel built without them emits neither.

        Args:
            params: This kernel's ``EpilogueParams``. Unused -- every term arrives via
                `epi_loop_tensors`.
            epi_loop_tensors: This subtile's loaded terms by op name: ``sRstd``, ``sScale``,
                ``mColsum``, ``mDbias`` and the inherited ``mRowVecBroadcast``.
            tRS_rD: The accumulator fragment holding the RAW ``x @ Bw``, overwritten in place with
                the full 2N pre-activation.
            tRS_rC: The C fragment. Unused; this variant takes no C.

        Returns:
            None; `tRS_rD` is modified in place.
        """
        self.epi_repair_rank_one(epi_loop_tensors, tRS_rD)
        gated_bias = epi_loop_tensors["mRowVecBroadcast"]
        if const_expr(gated_bias is not None):
            for i in cutlass.range(cute.size(tRS_rD), unroll_full=True):
                tRS_rD[i] = tRS_rD[i] + gated_bias[i]

    @cute.jit
    def epi_repair_rank_one(self, epi_loop_tensors, tRS_rD, c_name="mColsum", d_name="mDbias"):
        """Undo the fold on one accumulator fragment: ``acc <- r*acc - s*c + d``, in place, fp32.

        Purpose
            The fusion's one piece of arithmetic, factored out because BOTH output regions need it
            and they reach the epilogue by different routes. The dual arrives through
            `epi_combine_preact`; the output gate bypasses that entirely and arrives at
            `epi_visit_subtile`'s gate pass, which in the unfolded kernel has nothing to repair.

        Semantics
            **The two regions pass DIFFERENT `c`/`d`, which is why they are arguments.** Each
            region's vector is indexed from its own column 0, because `run_epilogue` re-bases the
            gate arm's n-block by ``n_dual_tiles`` -- so one wide vector shared by both would be
            read at the dual's columns on every gate tile. The statistics are genuinely shared:
            ``r`` and ``s`` are per-ROW, and the row is the same in both regions.

            ``s = r*mu`` is formed here rather than stored, because `finalize_row_stats` publishes
            ``mean`` and ``rstd``. One multiply per element buys not forking a helper `prolog_ln`
            also uses. ``d`` is ``const_expr``-pruned when there is no LayerNorm bias.

        Args:
            epi_loop_tensors: This subtile's loaded terms. Must carry ``sRstd``, ``sScale`` and the
                named colsum; the named dbias may be None. The two statistics are the MAINLOOP's,
                so this is valid only after `finalize_row_stats` has published them behind its
                barrier.
            tRS_rD: The accumulator fragment holding the RAW ``x @ Bw``, overwritten in place.
            c_name: Key of this region's column-sum vector. Must name an op whose slice matches
                `tRS_rD`'s columns -- passing the dual's on a gate tile is not an error anyone
                catches, it is simply a wrong number.
            d_name: Key of this region's LayerNorm-bias vector, which may hold None.

        Returns:
            None; `tRS_rD` is modified in place.
        """
        r = epi_loop_tensors["sRstd"]
        s = epi_loop_tensors["sScale"]
        c = epi_loop_tensors[c_name]
        d = epi_loop_tensors[d_name]
        for i in cutlass.range(cute.size(tRS_rD), unroll_full=True):
            tRS_rD[i] = r[i] * tRS_rD[i] - s[i] * c[i]
            if const_expr(d is not None):
                tRS_rD[i] = tRS_rD[i] + d[i]

    @cute.jit
    def epi_visit_subtile(
        self, params, epi_loop_tensors, tRS_rD, tRS_rC=None, epi_gate3: cutlass.Constexpr = False
    ):
        """Repair the OUTPUT GATE's tile too, then hand both regions to the inherited body.

        Purpose
            The output gate reads the same ``LN(x)`` as the dual, so under this fusion its
            accumulator is equally raw and equally in need of the rank-one repair. The base's gate
            pass adds ``b3`` and applies the activation to what it is given -- correct for a kernel
            that normalized its activation, and silently wrong for one that did not.

        Semantics
            Only the GATE arm is touched, and only before delegating: the dual arm's repair already
            happens inside `epi_combine_preact`, which the inherited body calls, so repairing here
            as well would apply it twice. The branch is `const_expr` on `epi_gate3`, so a kernel
            built without an output gate emits exactly the inherited body.

            **Order: repair, THEN the base adds ``b3``.** ``c`` and ``d`` belong to the fold and so
            sit inside the identity being repaired; ``b3`` is a bias on the finished pre-activation.
            Adding it first would put it inside the ``r*`` scaling, and the gate would be biased by
            a per-row amount no reference computes.

        Args:
            params: This kernel's ``EpilogueParams``.
            epi_loop_tensors: This subtile's loaded terms by op name.
            tRS_rD: The accumulator fragment. Repaired in place on the gate pass before delegating.
            tRS_rC: The C fragment, or None. Forwarded untouched.
            epi_gate3: Whether this is the output-gate pass. `Constexpr`: both arms are traced by
                the caller and this selects which body exists.

        Returns:
            The fp32 post-activation fragment the inherited body returns -- half the input's size
            on the dual pass, the same size on the gate pass.
        """
        if const_expr(epi_gate3):
            self.epi_repair_rank_one(epi_loop_tensors, tRS_rD, "mColsum3", "mDbias3")
        return super(LayerNormDualGatedGemmAlgFoldSm90, self).epi_visit_subtile(
            params, epi_loop_tensors, tRS_rD, tRS_rC, epi_gate3
        )


class _XGateTwoASm90:
    """The two-A (``x_gate``) machinery, shared by both LayerNorm fusions.

    Purpose
        Both fusions compute the SAME two-A shape::

            out[m, n] = act(x_gate @ Wg^T + bg)[m, n] * (LN(x) @ Wp^T + bp)[m, n]

        and differ only in HOW the value arm gets its ``LN(x)`` -- algebraically, by folding the
        gain into ``Wp`` and repairing rank-one in the epilogue, or physically, by rewriting the
        staged ``x`` in shared memory. Everything else -- four TMA descriptors under one
        transaction barrier, the halved pipeline, the gate's own MMA atom, the two-accumulator
        combining epilogue -- is identical, so it lives here once.

    Semantics
        **A mixin, placed FIRST in each concrete class's bases**, so its methods win over the
        fusion base's while every fusion-specific method (`_epi_ops`, `epi_to_underlying_arguments`,
        `_mma_stats_xgate`, `_xgate_combine_subtile`) still resolves to the concrete class. It
        declares no ``__init__``: construction, validation and the ping-pong refusal are the fusion
        base's, reached unchanged.

        **The two hooks a concrete class MUST get right are a matched PAIR, and a mismatch is a
        HANG rather than a wrong answer.** :meth:`_load_AB_xgate` here stages every operand ONCE
        per k-tile; a fusion whose consumer sweeps A twice must override it to stage twice.
        :data:`_A_SWEEPS` records which, so the two are declared side by side rather than inferred
        from two loop bodies in different methods.

        **It is deliberately not a `GemmSm90` subclass.** A mixin that inherited the base would put
        a second copy of it in every concrete MRO, and Python would refuse the linearization; more
        practically, this class is not runnable on its own -- it reads `self.gemm_k`, `self.blk_k`,
        `self.a_dtype`, `self.epilogue_barrier` and `self._stats_scratch`, all of which the fusion
        base supplies.

    Requirements every concrete class inherits from here, all of them upstream's scope:

    - **Cooperative only.** ``pingpong`` is refused. Two warpgroups alternating over four resident
      operand buffers is not a schedule this was measured at.
    - **``chunk_g == 1``.** The block-interleaved weight layout exists to pair an up column with a
      gate column inside one 2N accumulator; there is no 2N accumulator here.
    - **No output gate.** ``x_gate`` and ``W3`` are mutually exclusive -- both want to be the second
      thing this kernel computes.
    - **No D and no C.** The output is the post-activation store; a D operand would be the
      un-combined 2N pre-activation, which does not exist on this path.

    One shape requirement is worth stating because it is easy to trip and its diagnostic names a
    stride rather than a shape: **an MN-major activation needs ``M % 8 == 0`` at 16 bits**, since an
    MN-major ``(M, K)`` view has K-stride ``M``, and the package's one shape constraint is the
    16-byte floor. An off-grid ``M`` is fully supported with the ordinary K-major activation; it is
    only the transposed view that couples the two. Upstream refuses the same case with the same
    ``ValueError``, from the same descriptor check.

    Not ported, and the omission is a decision rather than an oversight: upstream's ``_normalize``
    knob takes the VALUE pre-normalized too, which makes the whole kernel a dual-x GEMM with no
    LayerNorm at all. That belongs on `dual_gated_gemm` -- this package HAS an unfused dual-gated
    kernel, where upstream only had this flag -- so an LN-fused front door needs no such flag.

    Attributes:
        _SUPPORTS_PINGPONG: False. Re-declared because the algebraic fold turns ping-pong ON and a
            subclass inherits that unless something says otherwise; with the mixin first in the
            MRO, this is what that something is.
        _KEEP_STATIC_LEN_K: False, and it is a MEASURED refusal rather than an oversight -- see the
            note on the attribute itself.
        _MIN_XGATE_AB_STAGE: The floor :meth:`_compute_stages` applies after halving. A per-fusion
            constant because a one-sweep mainloop still works at one stage and a two-sweep one does
            not.
        _A_SWEEPS: How many times the producer stages each k-tile, which MUST equal how many times
            the consumer waits for it.
    """

    #: Cooperative only: four resident operand buffers per stage and a full per-tile WGMMA drain
    #: are not a schedule the two-warpgroup alternation was measured at. Re-declared because the
    #: algebraic fold turns ping-pong ON and this mixin sits ahead of it in the MRO.
    _SUPPORTS_PINGPONG = False

    #: The generated ``EpilogueParams`` base. Declared here rather than on each concrete class
    #: because both epilogues are the plain composable kind -- neither extends the dual's pack.
    _epi_param_bases = (ParamsBase,)

    #: **This path does NOT take the algebraic fold's static-trip-count treatment, and that is a
    #: MEASURED refusal rather than an oversight.** With `GemmSm90._KEEP_STATIC_LEN_K` inherited as
    #: True this kernel returns a WRONG ANSWER intermittently: at ``N=512, K=384`` the two-A
    #: correctness test failed **11 of 20** runs against its fp64 reference (exceeding the
    #: elementwise bound by ~2x), where the same 20 runs with the flag False are **0/20**. The
    #: failure is intermittent, so it is a race, not an arithmetic change -- and it is INTRODUCED
    #: by the flag, not exposed by it: the shipped code without it is clean at that shape.
    #:
    #: The trip counts are not the mechanism -- at ``K=384`` the producer's ``ceil_div(len_k, 64)``
    #: and the consumer's ``gemm_k // blk_k`` are both 6 either way. What differs is the LOOP FORM:
    #: a static bound lets the producer's k-loop lower to a counted loop with its mbarrier address
    #: hoisted, and this path -- alone among the fusions -- stages FOUR operands per slot on a
    #: HALVED ``ab_stage``, so its producer/consumer handshake is the one with the least pipeline
    #: slack. The interaction is NOT root-caused; it is recorded here so the next reader does not
    #: "restore consistency" by deleting this line. Removing it reintroduces a silent wrong answer.
    _KEEP_STATIC_LEN_K = False

    #: Mainloop stages this fusion needs AFTER :meth:`_compute_stages` halves the base's count.
    #: Overridden by a two-sweep fusion, whose streaming loop cannot run at one stage.
    _MIN_XGATE_AB_STAGE = 1

    #: Sweeps the producer makes over A per work tile. **Must equal the consumer's**, because the
    #: producer stages ``_A_SWEEPS * k_tile_cnt`` tiles against exactly that many consumer waits --
    #: and a disagreement is a DEADLOCK, invisible to every host-side test because the kernel
    #: compiles perfectly and then never returns.
    _A_SWEEPS = 1

    #: Whether this fusion has a REGISTER-source mainloop for an MN-major value activation. False
    #: here, and True only where it pays: a fusion that reads the staged value tile ONCE has nothing
    #: to gain, because the strided MN-major shared-memory read it would avoid happens once either
    #: way. See :attr:`_value_a_in_regs` for the measurement.
    _SUPPORTS_A_IN_REGS = False

    @property
    def _value_a_in_regs(self) -> bool:
        """Whether THIS launch sources the value WGMMA's A operand from registers.

        Purpose
            The one switch the register-source mainloop hangs off. It is a property rather than a
            parameter because it is a FUNCTION of a bound parameter -- the value activation's major
            -- and a launch cannot choose it independently.

        Semantics
            True only when the fusion has the path, the value activation is MN-major, and the CTA
            tile is 128 rows. The middle condition is the case the TriMul back half actually feeds:
            its ``x`` comes MN-major straight out of a batched GEMM while its ``x_gate`` is K-major
            out of a LayerNorm.

            **The ``tile_M == 128`` condition is a CORRECTNESS bound, not a tuning one, and it is
            this port's one addition to what the kernel it reproduces does.** The register
            mainloop's row mapping is a closed form of the WGMMA operand-A fragment for a 128-row
            tile split across two warpgroups: each thread owns exactly the two rows
            ``{base, base+8}`` with ``base = (t//128)*64 + ((t%128)//32)*16 + ((t%32)//4)``, which
            enumerates 0..127 over 256 threads and NOTHING else. At ``tile_M = 256`` a thread owns
            four rows and that form addresses only the first 96; at 64 and 192 the warpgroup split
            changes underneath it. The kernel this reproduces derives the same form for its own
            ``BLK_M = 128`` and does not check it -- it would answer, plausibly and wrongly, at
            another tile. Selecting the shared-memory mainloop instead is not a refusal: that path
            is a complete implementation at every tile, so the only cost of an unusual tile is the
            MN-major penalty this one exists to avoid.

            **Why the major decides it.** An MN-major ``(tile_M, tile_K)`` tile is strided along K
            in shared memory, so a fusion that reads it TWICE -- once to reduce, once to normalize
            -- pays that stride twice, and measured against the kernel this reproduces the SMEM path
            runs **1.13-1.35x slower across the workflow ladder** (worst at D=256, M=1048576). The
            register path reads it once through the TRANSPOSING ``ldmatrix``, which is
            bank-conflict-free for MN-major shared memory and delivers K-ordered registers, then
            reduces and normalizes in place there. A K-major value keeps the shared-memory path,
            which is byte-identical to the kernel this reproduces.

            **It is NOT bit-comparable with the shared-memory path**, and that is expected rather
            than a defect: the register reduction completes a row across a 4-lane quad where the
            shared-memory one butterflies 2 lanes, so the fp32 sums associate differently and
            ``mu``/``rstd`` differ in their last bits. Both agree with the kernel this reproduces at
            their own major, which is the comparison that means anything.

        Returns:
            True to build the register-source value mainloop, False for the shared-memory one.
        """
        return (
            self._SUPPORTS_A_IN_REGS
            and self.a_layout.sm90_mma_major_mode() == warpgroup.OperandMajorMode.MN
            and self.tile_shape_mn[0] == _A_IN_REGS_TILE_M
        )

    def _make_rs_tiled_mma(self) -> cute.TiledMma:
        """Rebuild the VALUE MMA atom with operand A sourced from REGISTERS.

        Purpose
            `_make_tiled_mma` builds the shared-memory-source atom every other path wants; this is
            the same atom with one field changed, and it is built here rather than by widening that
            helper because the register source is this kernel's concern alone.

        Semantics
            Every other parameter -- dtypes, accumulator, atom layout, tiler, B-major -- is the
            shared-memory atom's, so the two produce IDENTICAL accumulator M/N layouts and the
            epilogue can still combine the value and gate accumulators elementwise. Only the operand
            source flips, which is what makes `partition_fragment_ABC` hand back a register
            A-fragment instead of a partition of ``sA``. The staged ``sA`` layout and the per-stage
            transaction count are unchanged: the tile is still loaded by TMA, it is simply read out
            of shared memory by ``ldmatrix`` instead of by the WGMMA.

            **No N-permutation is applied, and none is needed**: this path asserts
            ``atom_layout_n == 1`` below, and the permutation exists only to make two warpgroups'
            N-halves adjacent in the epilogue tile.

        Returns:
            The register-source ``cute.TiledMma``.

        Raises:
            AssertionError: If N is split across warpgroups (``atom_layout_mnk[1] != 1``). The
                register A-fragment must span the full ``BLK_K`` for one warpgroup, which an N-split
                atom does not give it; the cooperative ``(2, 1, 1)`` layout this path uses does.
        """
        assert self.atom_layout_mnk[1] == 1, (
            f"the register-source value mainloop needs atom_layout_n == 1; got "
            f"{self.atom_layout_mnk}. The rs A-fragment must span the whole BLK_K for one "
            f"warpgroup, which an N-split atom does not give it."
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
            The register mainloop's one piece of plumbing. The TMA still lands the value tile in
            shared memory; this is what moves it into the WGMMA's operand registers.

        Semantics
            The transpose flag is read off the ATOM's A-major rather than off ``self``, so it stays
            correct for any value major and cannot disagree with the atom the fragment was
            partitioned from. For an MN-major A that selects the TRANSPOSING ``ldmatrix``, which
            reads MN-major shared memory bank-conflict-free and delivers K-ordered registers --
            which is the whole reason this path is faster than reading the same tile strided.

        Args:
            tiled_mma: The REGISTER-source value MMA atom, i.e. what :meth:`_make_rs_tiled_mma`
                returned. Passing the shared-memory atom builds a copy whose destination layout does
                not match the fragment, which is a wrong answer rather than a fault.

        Returns:
            A ``cute.TiledCopy`` whose destination is the atom's A fragment.
        """
        layout_a = (
            cutlass.utils.LayoutEnum.COL_MAJOR
            if tiled_mma.op.a_major_mode == warpgroup.OperandMajorMode.MN
            else cutlass.utils.LayoutEnum.ROW_MAJOR
        )
        return cute.make_tiled_copy_A(
            copy_utils.sm90_get_smem_load_op(layout_a, self.a_dtype), tiled_mma
        )

    @cached_property
    def a2_smem_layout_staged(self):
        """The SECOND A operand's staged SMEM layout, ``(tile_M, tile_K, ab_stage)`` with swizzle.

        Purpose
            The gate's ``sA2`` box. Derived rather than assigned during ``__call__`` because that
            method is traced: a ``self.X`` read inside it is folded in at trace time, so the value
            must come from a declared parameter or a property of them -- assigning it mid-trace is
            mutable state that can desynchronize from an already-compiled kernel. This is the same
            rule `GemmSm90.a_smem_layout_staged` follows, and this is its two-A counterpart.

        Semantics
            When the two A operands share a major this IS `a_smem_layout_staged`, aliased rather
            than rebuilt -- the SM90 WGMMA bakes the A-major into the atom, so identical majors
            mean an identical box and the one-major code path. When they differ it takes the A
            output of a second `_make_smem_layouts` call; B and the epilogue layouts are already
            built for the value operand and are not rebuilt here.

            Depends only on bound call parameters (`a2_layout`, `a_layout`, `a_dtype`, `b_dtype`,
            the tile shapes and the stage counts), so it cannot be read before
            :meth:`bind_operand_types` has run -- which is exactly the ordering a `cached_property`
            enforces and an ``__init__`` assignment cannot.

        Returns:
            The staged, swizzled A2 layout.
        """
        if self.a2_layout == self.a_layout:
            return self.a_smem_layout_staged
        # Take the A output only; B and the epilogue layouts were already built for the value.
        return self._make_smem_layouts(
            self.cta_tile_shape_mnk,
            self.epi_tile,
            self.a_dtype,
            self.a2_layout,
            self.b_dtype,
            self.b_layout,
            self.ab_stage,
            self.d_dtype,
            self.d_layout,
            self.epi_stage,
            self.c_dtype,
            self.c_layout,
            self.epi_c_stage,
        )[0]

    def _compute_stages(self, *args, **kwargs):
        """Take the parent's stage count and HALVE it, because a stage holds four operands not two.

        Purpose
            The base sizes a pipeline stage as one A tile plus one B tile. This kernel stages
            ``sA``, ``sA2``, ``sB`` and ``sB2`` per slot -- exactly twice that -- so the count that
            fits is half.

        Semantics
            Calls the parent first, which is what reserves the LayerNorm scratch out of the same
            budget, then halves only the mainloop count. The epilogue counts are untouched: the
            epilogue is unchanged in footprint.

            **Inheriting the parent's count does not fail at compile.** The SMEM struct is simply
            larger than the budget it was sized against, and the launch fails with
            ``cudaErrorInvalidValue``, which names shared memory nowhere. That is the whole reason
            this override exists and the reason it is stated here rather than assumed.

        Args:
            *args: The base's positional arguments (CTA tile, epilogue tile, the four dtypes, the
                epilogue arguments, the SMEM capacity, the occupancy), forwarded unchanged.
            **kwargs: Likewise.

        Returns:
            ``(ab_stage, epi_stage, epi_c_stage)`` with ``ab_stage`` halved and floored at
            :data:`_MIN_XGATE_AB_STAGE`. A tile so large that even one four-operand stage does not
            fit is a SMEM overrun at launch, not a zero-stage pipeline -- the floor keeps the
            failure at the size that caused it.
        """
        ab_stage, epi_stage, epi_c_stage = super()._compute_stages(*args, **kwargs)
        return max(self._MIN_XGATE_AB_STAGE, ab_stage // 2), epi_stage, epi_c_stage

    @cached_property
    def num_tma_load_bytes(self) -> int:
        """Bytes one pipeline stage moves: TWO A tiles and TWO B tiles.

        Purpose
            The producer commits exactly this count and the consumer waits for exactly it, so a
            value that disagrees with what the four descriptors move is a HANG, not a wrong answer.
            The base derives it from one A and one B, which is half of what this kernel loads.

        Semantics
            The two A tiles are accounted SEPARATELY rather than doubled. They cover the same
            ``(tile_M, tile_K)`` extent and therefore the same byte count today, but they are built
            from DIFFERENT staged layouts whenever the gate's major differs from the value's, and
            an assumption that two layouts have equal size is exactly the kind that stops being
            true silently.

        Returns:
            The per-stage transaction count in bytes.

        Note:
            A ``cached_property``, and so is :attr:`a2_smem_layout_staged`, which this reads. The
            ordering that used to have to be respected by hand -- the A2 layout assigned before
            this was materialized -- is now enforced by the property chain: both derive from the
            bound call parameters, so neither can be read before `bind_operand_types` has run.
        """
        b_bytes = cute.size_in_bytes(
            self.b_dtype, cute.slice_(self.b_smem_layout_staged, (None, None, 0))
        )
        a_bytes = cute.size_in_bytes(
            self.a_dtype, cute.slice_(self.a_smem_layout_staged, (None, None, 0))
        )
        a2_bytes = cute.size_in_bytes(
            self.a_dtype, cute.slice_(self.a2_smem_layout_staged, (None, None, 0))
        )
        return 2 * b_bytes + a_bytes + a2_bytes

    @cute.jit
    def epi_convert_postact(
        self,
        tRS_rPostAct,
        sr_seed,
        tidx,
        tile_coord_mnkl,
        num_prev_subtiles,
        epi_idx,
        epi_gate3: cutlass.Constexpr = False,
    ):
        """Narrow the fragment WITHOUT the gated register permute.

        Purpose
            **The one method-resolution hazard this class has, and it is silent.** `GemmGatedMixin`
            sits between this class and `GemmActMixin` in the MRO, and its version appends
            `permute_gated_Cregs_b16` -- correct when a 2N pre-activation was folded to N, because
            output column ``o`` is then held by the lane that owned column ``2o``. Nothing is folded
            here: the accumulator is already N-wide and every column is already owned by the lane
            ``stmatrix`` expects. Running the permute anyway scrambles valid data and raises
            nothing.

        Semantics
            Delegates to `GemmActMixin`'s plain conversion by naming the class explicitly, which is
            the only way to reach past an intermediate override. `epi_gate3` is accepted to match
            the hook's signature and is always False here.

        Args:
            tRS_rPostAct: The fp32 combined fragment for this subtile.
            sr_seed: Stochastic-rounding seed; unused under RN.
            tidx: Calling thread index; unused under RN.
            tile_coord_mnkl: Work-tile coordinate; unused under RN.
            num_prev_subtiles: Subtiles already stored; unused under RN.
            epi_idx: Index of this subtile; unused under RN.
            epi_gate3: Always False on this path; there is no output gate.

        Returns:
            A new register fragment of ``self.postact_dtype``.
        """
        return GemmActMixin.epi_convert_postact(
            self, tRS_rPostAct, sr_seed, tidx, tile_coord_mnkl, num_prev_subtiles, epi_idx
        )

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mD: Optional[cute.Tensor],
        mC: Optional[cute.Tensor],
        epilogue_args: tuple,
        scheduler_args,
        stream,
        mA2: Optional[cute.Tensor] = None,
        mB2: Optional[cute.Tensor] = None,
    ):
        """Host entry for the two-A kernel: bind the operands, build four descriptors, launch.

        Purpose
            The base's ``__call__`` binds one A and one B. This binds the value pair through the
            base -- so every derived parameter (the MMA atom, the tiles, the stage counts, the
            epilogue geometry) comes from exactly the same code -- and then adds the gate pair.

        Semantics
            **The gate operands ride the base's ``mB2``/``mB3`` compile slots.** `compile_gemm_kernel`
            appends whatever it is given after ``stream``, and this signature takes ``mA2`` there --
            so the gate A goes in the ``mB2`` slot and the gate B in the ``mB3`` slot at COMPILE
            time, and the launch passes them in the same positions. That indirection is upstream's
            and is kept: it is what lets a two-A kernel reuse the one-A compile plumbing untouched.

            The gate's tiled MMA is built from :attr:`a2_layout` ONLY when it differs from the
            value's, and :attr:`a2_smem_layout_staged` makes the same choice for the ``sA2`` box;
            otherwise both are ALIASED to the value objects, so the matching-major case emits
            identical code. The two MMAs share every other parameter --
            dtypes, accumulator, atom layout, tiler, N permutation -- so the two accumulators have
            IDENTICAL M/N register layouts and the epilogue can combine them elementwise.

        Args:
            mA: ``(M, K, L)`` value activation ``x``, RAW (the fold's convention: never normalized).
            mB: ``(N, K, L)`` value weight ``Bw = diag(w_ln) @ Wp``, N-wide -- NOT the dual's 2N
                interleave.
            mD: Must be None. A D operand would be the un-combined pre-activation, which this path
                never materializes.
            mC: Must be None. This path takes no residual.
            epilogue_args: This launch's :class:`EpilogueArguments`.
            scheduler_args: Tile-scheduler options.
            stream: The CUDA stream.
            mA2: ``(M, K, L)`` gate activation ``x_gate``, PRE-NORMALIZED. Required. May carry a
                different major than `mA`; may NOT be the same tensor.
            mB2: ``(N, K, L)`` gate weight ``Wg``, RAW -- no gain folded in, because the gate's
                activation was normalized upstream. Required.

        Returns:
            None. The kernel is launched on `stream`.

        Raises:
            AssertionError: If either gate operand is missing, if `mD`/`mC` is given, if ping-pong
                is set, if ``chunk_g != 1``, or if the functor was built with an output gate.
        """
        assert mA2 is not None and mB2 is not None, (
            "the two-A x_gate kernel requires BOTH gate operands: mA2 (x_gate) and mB2 (Wg)"
        )
        assert mD is None, "the x_gate output is the post-activation store; mD must be None"
        assert mC is None, "the x_gate path takes no C residual"
        assert not self.pingpong, (
            "the two-A x_gate path is cooperative only; ping-pong is not a schedule it was measured "
            "at with four resident operand buffers per stage"
        )
        assert self.chunk_g == 1, (
            "the two-A x_gate path requires chunk_g == 1: the block-interleaved layout pairs an up "
            "column with a gate column inside one 2N accumulator, and there is no 2N accumulator here"
        )
        assert not self._has_gate3, "x_gate and the output gate (W3) are mutually exclusive"

        # Bind the VALUE pair through the base, so the MMA atom, the CTA tile, the stage counts and
        # the epilogue geometry are derived by exactly the code the one-A path uses.
        # `mA2` goes in so the gate's A major is BOUND as a call parameter (`a2_layout`) rather
        # than assigned mid-trace: the workflow's back half feeds an MN-major value x (out of a
        # batched GEMM) and a K-major x_gate (out of a LayerNorm), and the SM90 WGMMA bakes the
        # A-major into the atom, so a differing major needs its own atom and its own staged layout.
        tiled_mma = self.bind_operand_types(mA, mB, None, None, None, epilogue_args, mA2=mA2)
        # Each call re-declares only its FIRST argument's strides (mD is None and two_tensor_B is
        # False, so the other slots pass through), which is how the base's one helper covers three
        # operands without a second spelling of the same `cute.assume`.
        mA, _, _, _ = self.assume_aligned_strides(mA, mB, None, None)
        mA2, _, _, _ = self.assume_aligned_strides(mA2, mB, None, None)
        mB2, _, _, _ = self.assume_aligned_strides(mB2, mB, None, None)

        if const_expr(self.a2_layout == self.a_layout):
            # Matching majors: ALIAS the value's atom so this case emits the one-major code. The
            # alias is taken from the SMEM-SOURCE atom, BEFORE the value's may be rebuilt as
            # register-source below -- the gate reads its raw `sA2` from shared memory on every
            # path, and an rs atom for it would partition a fragment nothing fills.
            tiled_mma_gate = tiled_mma
        else:
            tiled_mma_gate = self._make_tiled_mma(self.b_dtype, self.a2_layout, self.b_layout)
        if const_expr(self._value_a_in_regs):
            tiled_mma = self._make_rs_tiled_mma()

        tma_atom_a, tma_tensor_a, tma_atom_b, tma_tensor_b, _, _ = self.make_mainloop_tma(
            mA, mB, epilogue_args=epilogue_args
        )
        a2_smem_layout = cute.slice_(self.a2_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, 0))
        tma_atom_a2, tma_tensor_a2 = self._make_tma_atoms_and_tensors(
            mA2,
            a2_smem_layout,
            (self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[2]),
            self.cluster_shape_mnk[1],
        )
        tma_atom_b2, tma_tensor_b2 = self._make_tma_atoms_and_tensors(
            mB2,
            b_smem_layout,
            (self.cta_tile_shape_mnk[1], self.cta_tile_shape_mnk[2]),
            self.cluster_shape_mnk[0],
        )
        # Materialize the transaction count HERE, in this region: a cached_property that emits MLIR
        # and is first read inside `@cute.kernel` lands its ops in the kernel region while their
        # operands live outside it, and the module fails verification. `a2_smem_layout_staged` is
        # already set, which is what this override reads.
        _ = self.num_tma_load_bytes
        epilogue_params = self.epi_to_underlying_arguments(epilogue_args)

        TileSchedulerCls = self.get_scheduler_class()
        tile_sched_args = self.get_scheduler_arguments(mA, mB, None, scheduler_args, epilogue_args)
        tile_sched_params = TileSchedulerCls.to_underlying_arguments(tile_sched_args)
        grid = TileSchedulerCls.get_grid_shape(
            tile_sched_params, scheduler_args.max_active_clusters
        )

        @cute.struct
        class SharedStorage:
            """The base's layout with the gate's two operand tiles appended.

            Field ORDER is allocation order and the alignments are load-bearing; the two gate tiles
            are appended AFTER the value pair so every offset the base derives is unchanged.
            ``sD``/``sC`` are declared at zero size (there is no D and no C) rather than omitted,
            because `GemmSm90.kernel_prologue` -- which this kernel reuses -- indexes the struct by
            name.
            """

            ab_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.ab_stage * 2]
            epi_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.epi_c_stage * 2]
            sched_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.sched_stage * 2]
            sched_data: cute.struct.MemRange[Int32, self.sched_stage * 4]
            sD: cute.struct.Align[cute.struct.MemRange[Int32, 0], self.buffer_align_bytes]
            sC: cute.struct.Align[cute.struct.MemRange[Int32, 0], self.buffer_align_bytes]
            epi: self.epi_get_smem_struct(epilogue_params)
            decoupled: self._extra_smem_struct()
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sA2: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a2_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sB2: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage
        self.resolve_launch_shape()

        self.kernel_xgate(
            tiled_mma,
            tiled_mma_gate,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_a2,
            tma_tensor_a2,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_b2,
            tma_tensor_b2,
            epilogue_params,
            self.cluster_layout_mnk,
            self.a_smem_layout_staged,
            self.a2_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
            tile_sched_params,
            TileSchedulerCls,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            stream=stream,
            min_blocks_per_mp=1,
        )
        return

    @cute.kernel
    def kernel_xgate(
        self,
        tiled_mma,
        tiled_mma_gate,
        tma_atom_a,
        mA_mkl,
        tma_atom_a2,
        mA2_mkl,
        tma_atom_b,
        mB_nkl,
        tma_atom_b2,
        mB2_nkl,
        epilogue_params,
        cluster_layout_mnk,
        a_smem_layout,
        a2_smem_layout,
        b_smem_layout,
        epi_smem_layout,
        tile_sched_params,
        TileSchedulerCls: cutlass.Constexpr[Callable],
    ):
        """Device entry: the base's prologue, then the two-A producer and the two-A consumer.

        Purpose
            The device-side counterpart of :meth:`__call__`. It exists as a separate entry rather
            than as an override of `GemmSm90.kernel` because both of its roles take four operands
            where the base's take two, and a signature that carried both shapes would be a runtime
            branch in the one place a runtime branch is unaffordable.

        Semantics
            **The prologue is the base's, unchanged.** `GemmSm90.kernel_prologue` allocates the
            shared storage, builds and publishes every mbarrier and rendezvouses the cluster --
            reading only ``sA``/``sB`` out of the struct, so the two gate tiles are simply fetched
            after it returns. Reusing it is what keeps the mbarrier init ordering, which is a HANG
            when it is wrong, from being restated here.

        Args:
            tiled_mma: The value MMA atom.
            tiled_mma_gate: The gate MMA atom -- the same object as `tiled_mma` when the two A
                majors agree.
            tma_atom_a, mA_mkl: Value A's descriptor and its descriptor-side view.
            tma_atom_a2, mA2_mkl: Gate A's, likewise.
            tma_atom_b, mB_nkl: Value B's.
            tma_atom_b2, mB2_nkl: Gate B's.
            epilogue_params: The lowered epilogue params.
            cluster_layout_mnk: The cluster layout, for the multicast masks.
            a_smem_layout: Value A's staged SMEM layout.
            a2_smem_layout: Gate A's -- the same object as `a_smem_layout` when the majors agree.
            b_smem_layout: B's staged SMEM layout, shared by both B operands.
            epi_smem_layout: The epilogue's staged SMEM layout.
            tile_sched_params: Scheduler params.
            TileSchedulerCls: The scheduler class.

        Returns:
            None. Its effect is the post-activation store.
        """
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        self.prefetch_tma_descriptors(warp_idx, tma_atom_a, tma_atom_a2, tma_atom_b, tma_atom_b2)
        ctx = self.kernel_prologue(
            warp_idx,
            tiled_mma,
            mA_mkl,
            None,
            None,
            epilogue_params,
            cluster_layout_mnk,
            a_smem_layout,
            b_smem_layout,
            epi_smem_layout,
            None,
            tile_sched_params,
            TileSchedulerCls,
        )
        sA2 = ctx.storage.sA2.get_tensor(a2_smem_layout.outer, swizzle=a2_smem_layout.inner)
        sB2 = ctx.storage.sB2.get_tensor(b_smem_layout.outer, swizzle=b_smem_layout.inner)
        self._producer_xgate(
            warp_idx,
            cluster_layout_mnk,
            mA_mkl,
            mA2_mkl,
            mB_nkl,
            mB2_nkl,
            tma_atom_a,
            tma_atom_a2,
            tma_atom_b,
            tma_atom_b2,
            ctx.ab_pipeline,
            ctx.len_k,
            ctx.sA,
            sA2,
            ctx.sB,
            sB2,
            ctx.TileSchedulerCls,
        )
        self._consumer_xgate(
            warp_idx,
            tiled_mma,
            tiled_mma_gate,
            ctx.sA,
            sA2,
            ctx.sB,
            sB2,
            ctx.storage,
            epilogue_params,
            ctx.epi_smem_tensors,
            ctx.ab_pipeline,
            ctx.TileSchedulerCls,
        )

    @cute.jit
    def _producer_xgate(
        self,
        warp_idx,
        cluster_layout_mnk,
        mA_mkl,
        mA2_mkl,
        mB_nkl,
        mB2_nkl,
        tma_atom_a,
        tma_atom_a2,
        tma_atom_b,
        tma_atom_b2,
        ab_pipeline,
        len_k,
        sA,
        sA2,
        sB,
        sB2,
        TileSchedulerCls: cutlass.Constexpr[Callable],
    ):
        """The AB-load warpgroup, addressing FOUR operands per work tile instead of two.

        Purpose
            The two-A counterpart of `GemmSm90Load.producer_warpgroup_role`. It is a separate method
            rather than an override because the base's builds two copy closures and hands them to a
            two-argument :meth:`load_AB`; every difference here is in that pair of facts.

        Semantics
            Identical in structure to the base's role -- the same register decrease, the same
            multicast masks, the same persistent scheduler loop, the same addressing seams
            (`select_batch`, `mainloop_remap_mA`, `_gA_local_tile`, `_k_tile_cnt`) -- with the value
            operands going through those seams and the gate operands tiled directly beside them.
            The gate side takes no remap hook: those seams exist for the distributed front's
            per-peer row and column bases, and the gate's A is a plain local tensor on every path
            that reaches this kernel.

            ``k_tile_cnt`` comes from the SAME `_k_tile_cnt` the consumer uses. The producer staging
            a different number of tiles than the consumer drains is a DEADLOCK, not a wrong answer.

        Args:
            warp_idx: Warp-uniform index; the role guard reads it.
            cluster_layout_mnk: Cluster layout, for the multicast masks.
            mA_mkl, mA2_mkl: Value and gate A, as their descriptors see them.
            mB_nkl, mB2_nkl: Value and gate B, likewise.
            tma_atom_a, tma_atom_a2, tma_atom_b, tma_atom_b2: The four descriptors.
            ab_pipeline: The mainloop pipeline. Its transaction count must cover all four loads --
                see :attr:`num_tma_load_bytes`.
            len_k: The contraction extent, converted by `_k_tile_cnt`.
            sA, sA2, sB, sB2: The four staged destinations.
            TileSchedulerCls: The bound scheduler factory.

        Returns:
            None. Its effect is the SMEM staging the consumer drains.
        """
        if warp_idx >= self.ab_load_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_load)
            if (
                warp_idx >= self.ab_load_warp_id
                and warp_idx < self.ab_load_warp_id + self.num_ab_load_warps
            ):
                cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
                block_in_cluster_coord_mnk = cluster_layout_mnk.get_flat_coord(cta_rank_in_cluster)
                a_mcast_mask = cute.make_layout_image_mask(
                    cluster_layout_mnk, block_in_cluster_coord_mnk, mode=1
                )
                b_mcast_mask = cute.make_layout_image_mask(
                    cluster_layout_mnk, block_in_cluster_coord_mnk, mode=0
                )
                a_mcast_mask = a_mcast_mask if self.is_a_mcast else 0
                b_mcast_mask = b_mcast_mask if self.is_b_mcast else 0
                a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_mnk, (0, None, 0)).shape)
                b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_mnk, (None, 0, 0)).shape)
                is_scheduler_warp = self.num_ab_load_warps == 1 or warp_idx == self.ab_load_warp_id
                if const_expr(cute.size(cluster_layout_mnk) > 1):
                    is_scheduler_warp = is_scheduler_warp and cute.arch.block_idx_in_cluster() == 0
                tile_scheduler = TileSchedulerCls()
                work_tile = tile_scheduler.initial_work_tile_info()
                ab_producer_state = make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.ab_stage
                )
                while work_tile.is_valid_tile:
                    tile_coord_mnkl = work_tile.tile_idx
                    batch_idx = tile_coord_mnkl[3]
                    mA_mk = self.mainloop_remap_mA(
                        self.select_batch(mA_mkl, batch_idx), tile_coord_mnkl
                    )
                    gA_mk = self._gA_local_tile(mA_mk, tile_coord_mnkl)
                    gA2_mk = self._gA_local_tile(
                        self.select_batch(mA2_mkl, batch_idx), tile_coord_mnkl
                    )
                    mB_nk = self.mainloop_remap_mB(
                        self.select_batch(mB_nkl, batch_idx), tile_coord_mnkl
                    )
                    gB_nk = self._gB_local_tile(mB_nk, tile_coord_mnkl)
                    gB2_nk = self._gB_local_tile(
                        self.select_batch(mB2_nkl, batch_idx), tile_coord_mnkl
                    )
                    copy_A, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_a,
                        cta_coord=block_in_cluster_coord_mnk[1],
                        cta_layout=a_cta_layout,
                        src_tensor=gA_mk,
                        dst_tensor=sA,
                        mcast_mask=a_mcast_mask,
                    )
                    copy_A2, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_a2,
                        cta_coord=block_in_cluster_coord_mnk[1],
                        cta_layout=a_cta_layout,
                        src_tensor=gA2_mk,
                        dst_tensor=sA2,
                        mcast_mask=a_mcast_mask,
                    )
                    copy_B, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_b,
                        cta_coord=block_in_cluster_coord_mnk[0],
                        cta_layout=b_cta_layout,
                        src_tensor=gB_nk,
                        dst_tensor=sB,
                        mcast_mask=b_mcast_mask,
                    )
                    copy_B2, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_b2,
                        cta_coord=block_in_cluster_coord_mnk[0],
                        cta_layout=b_cta_layout,
                        src_tensor=gB2_nk,
                        dst_tensor=sB2,
                        mcast_mask=b_mcast_mask,
                    )
                    ab_producer_state = self._load_AB_xgate(
                        ab_pipeline,
                        ab_producer_state,
                        copy_A,
                        copy_A2,
                        copy_B,
                        copy_B2,
                        self._k_tile_cnt(len_k),
                    )
                    tile_scheduler.advance_to_next_work(is_scheduler_warp=is_scheduler_warp)
                    work_tile = tile_scheduler.get_current_work()
                ab_pipeline.producer_tail(ab_producer_state)
                if is_scheduler_warp:
                    tile_scheduler.producer_tail()

    @cute.jit
    def _load_AB_xgate(
        self, ab_pipeline, ab_producer_state, copy_A, copy_A2, copy_B, copy_B2, k_tile_cnt
    ):
        """Stage all FOUR operands of one k-tile into ONE pipeline slot.

        Purpose
            The two-A counterpart of `GemmSm90Load.load_AB`. The base issues two copies per slot;
            this issues four.

        Semantics
            All four copies arrive on the SAME stage barrier, whose transaction count
            (:attr:`num_tma_load_bytes`) covers all four. That is what makes a stage atomic: the
            consumer either sees every operand of a k-tile or waits. Issuing the four under
            separate barriers would let a WGMMA read a half-filled stage -- a wrong answer with no
            diagnostic.

        Args:
            ab_pipeline: The mainloop pipeline.
            ab_producer_state: The producer state; advanced ``k_tile_cnt`` times and returned.
            copy_A: Value A's copy closure.
            copy_A2: Gate A's.
            copy_B: Value B's.
            copy_B2: Gate B's.
            k_tile_cnt: k-tiles for this work tile. Must equal what the consumer drains, or the two
                deadlock against each other.

        Returns:
            The advanced producer state.
        """
        peek_ab_empty_status = Boolean(True)
        if 0 < k_tile_cnt:
            peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)
        for k_tile in cutlass.range(k_tile_cnt, unroll=1):
            ab_pipeline.producer_acquire(ab_producer_state, peek_ab_empty_status)
            tma_bar_ptr = ab_pipeline.producer_get_barrier(ab_producer_state)
            smem_idx = ab_producer_state.index
            copy_A(k_tile, smem_idx, tma_bar_ptr=tma_bar_ptr)
            copy_A2(k_tile, smem_idx, tma_bar_ptr=tma_bar_ptr)
            copy_B(k_tile, smem_idx, tma_bar_ptr=tma_bar_ptr)
            copy_B2(k_tile, smem_idx, tma_bar_ptr=tma_bar_ptr)
            ab_pipeline.producer_commit(ab_producer_state)
            ab_producer_state.advance()
            peek_ab_empty_status = Boolean(True)
            if k_tile + 1 < k_tile_cnt:
                peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)
        return ab_producer_state

    def _stage_ln_affine(self, storage, params, tidx, warp_group_idx):
        """Stage the ``(K,)`` LayerNorm gain and bias into shared memory, once per CTA.

        Purpose
            The physical fusion normalizes with the gain in the mainloop, so the vector must be
            resident before the first k-tile. Doing it here -- outside the work-tile loop -- turns
            ``k_tiles * threads`` global reads into ``K`` of them; doing it per work tile would
            re-read it from global on every tile of a persistent grid.

        Semantics
            **A plain ``def`` called from a ``@cute.jit`` body, and that is load-bearing rather
            than stylistic.** Inside a ``@cute.jit`` function the DSL preprocessor turns the
            enclosing dynamic ``if`` into an MLIR region and tries to flatten every object the body
            names into IR values -- so writing ``storage.decoupled.s_weight`` there makes it
            attempt to convert the whole ``SharedStorage`` and fail with *"unable to convert
            <SharedStorage> to Numeric"*, which names neither shared memory nor this vector.
            Reading the field in a plain method keeps the object on the Python side, exactly as
            :meth:`_stats_scratch` already does for the statistics scratch. The preprocessor does
            not run here either, so this body carries no dynamic ``if``; the emitting work is
            delegated to the ``@cute.jit`` helper `stage_ln_affine`.

            **Both buffers are absent, not merely unused, on the algebraic fold**
            (:data:`~LayerNormDualGatedGemmSm90._STAGES_LN_AFFINE` False): that fusion absorbed the
            gain into the WEIGHT on the host, so `_extra_smem_struct` never allocates them and
            reaching for the field is an `AttributeError`. The guard therefore covers the field
            ACCESS, not just the copy.

        Args:
            storage: The kernel's shared storage. On a staging variant ``storage.decoupled`` must
                carry ``s_weight`` and ``s_bias``, each ``gemm_k`` fp32 slots.
            params: The traced ``EpilogueParams``. ``mWeight`` must be the ``(K,)`` fp32 gain and
                ``mBias`` the ``(K,)`` fp32 bias or None; a 16-bit gain here would be read as fp32.
            tidx: This thread's index within the cooperating group, in
                ``[0, _stats_num_threads())``. A value outside that range writes another thread's
                slots.
            warp_group_idx: This MMA warpgroup's index, warp-uniform. Selects the publishing
                barrier only.

        Returns:
            ``(s_weight, s_bias)`` -- the staged fp32 shared tensors, both ``None`` on a variant
            that does not stage them, and ``s_bias`` ``None`` when the kernel was built without a
            LayerNorm bias. Valid on return: `stage_ln_affine` ends with a fence and a barrier.

        Note:
            The trailing barrier is COLLECTIVE, so every thread of the cooperating group must
            reach this call. That is automatic in the one caller -- the MMA role branches before
            it, and all of its warpgroups arrive.
        """
        if not self._STAGES_LN_AFFINE:
            return None, None
        s_weight = storage.decoupled.s_weight.get_tensor(cute.make_layout((self.gemm_k,)))
        s_bias = (
            storage.decoupled.s_bias.get_tensor(cute.make_layout((self.gemm_k,)))
            if params.mBias is not None
            else None
        )
        stage_ln_affine(
            s_weight,
            s_bias,
            params.mWeight,
            params.mBias,
            self.gemm_k,
            self.gemm_k_real,
            self._stats_num_threads(),
            tidx,
            self._stats_barrier(warp_group_idx),
        )
        return s_weight, s_bias

    @cute.jit
    def _consumer_xgate(
        self,
        warp_idx,
        tiled_mma,
        tiled_mma_gate,
        sA,
        sA2,
        sB,
        sB2,
        storage,
        epilogue_params,
        epi_smem_tensors,
        ab_pipeline,
        TileSchedulerCls: cutlass.Constexpr[Callable],
    ):
        """The MMA warpgroups: per work tile run both WGMMAs, then the combining epilogue.

        Purpose
            The two-A counterpart of `GemmSm90Mma.mma_warpgroup_role`. It is a separate method for
            two reasons, both structural: the mainloop cannot use the base's 1-deep software
            pipeline (see :meth:`_mma_stats_xgate`), and the epilogue needs TWO accumulators where
            `run_epilogue` passes one.

            **This method REPLACES the base consumer, so `mma_setup_fragments` is NOT on this
            path.** It builds its own fragments (`partition_fragment_ABC` for both accumulators,
            `stats_tiled_copy` for the reduction), stages the LayerNorm gain/bias itself, and calls
            :meth:`_mma_stats_xgate` directly. Anything conditioned inside `mma_setup_fragments`
            therefore has no effect here, however the inheritance chain reads. Two separate
            investigations of a wrong-answer defect on this path spent time on that method before
            checking the call site: **inheritance tells you which implementation WOULD run; it does
            not tell you that anything calls it.** That is also why the gain/bias staging below is
            duplicated from `mma_setup_fragments` rather than inherited from it.

        Semantics
            Everything built here is per-CTA and tile-independent, so it is built ONCE outside the
            work-tile loop: both accumulator/operand fragment sets, the statistics tiled copy, and
            this warpgroup's slice of the shared statistics scratch.

            **The gate fragments are partitioned off the GATE MMA**, so the gate A fragment matches
            the gate A major. Both accumulators still have identical M/N layouts -- the two atoms
            differ only in A-major, and in whether the VALUE one sources A from registers -- which
            is what lets the epilogue read them at the same index.

            **On the register-source path ``tCrA`` is a fragment nothing has filled yet**, because
            `partition_fragment_ABC` reads the source off the atom and partitions by SHAPE when it
            is RMEM. The shared-memory -> register copy that fills it is built here, once, and
            handed to the mainloop; it is None on every other path and the mainloop prunes it.

            Cooperative only, so ``warp_group_idx`` selects a slice of the M range and the
            statistics scratch has one slot; there is no ping-pong barrier dance here.

            **The ``(K,)`` LayerNorm gain and bias are staged here, once, and only by the variant
            that reads them** (:data:`~LayerNormDualGatedGemmSm90._STAGES_LN_AFFINE`). The
            algebraic fold absorbed them into the weight on the host, so `_extra_smem_struct` does
            not even allocate the buffers there -- reaching for the field would be an
            `AttributeError`, not a zero-cost read, which is why both the buffers and the staging
            call sit behind the same `const_expr`. Staging inside the work-tile loop instead would
            re-read the vector from global on every tile of a persistent grid, and its trailing
            barrier is collective, so every MMA warpgroup thread must reach this point -- which it
            does, the role having already branched.

        Args:
            warp_idx: Warp-uniform index; the role guard reads it.
            tiled_mma: The value MMA atom.
            tiled_mma_gate: The gate MMA atom.
            sA, sA2, sB, sB2: The four staged operand tiles.
            storage: The kernel's shared storage, for the statistics scratch.
            epilogue_params: The lowered epilogue params. ``eps`` is read from it, and so are
                the ``(K,)`` LayerNorm gain and bias on the variant that stages them.
            epi_smem_tensors: The epilogue ops' SMEM tensors.
            ab_pipeline: The mainloop pipeline.
            TileSchedulerCls: The bound scheduler factory.

        Returns:
            None. Its effect is the post-activation store.
        """
        if warp_idx < self.ab_load_warp_id:
            cute.arch.setmaxregister_increase(self.num_regs_mma)
            is_tma_warp = Boolean(warp_idx == 0)
            tidx, _, _ = cute.arch.thread_idx()
            warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)
            warp_group_thread_layout = cute.make_layout(
                self.mma_warp_groups, stride=self.num_threads_per_warp_group
            )
            thr_mma = tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx))
            acc, tCrA, tCrB = fold_cp_ops_sm90_utils.partition_fragment_ABC(
                thr_mma, self.cta_tile_shape_mnk, sA, sB
            )
            thr_mma_gate = tiled_mma_gate.get_slice(warp_group_thread_layout(warp_group_idx))
            acc_gate, tCrA2, tCrB2 = fold_cp_ops_sm90_utils.partition_fragment_ABC(
                thr_mma_gate, self.cta_tile_shape_mnk, sA2, sB2
            )

            tiled_copy = stats_tiled_copy(
                self.a_dtype,
                (self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[2]),
                self._stats_num_threads(),
                STATS_THREADS_PER_ROW,
            )
            thr_copy = tiled_copy.get_slice(tidx)
            stats_smem = self._stats_scratch(storage, warp_group_idx)
            s_weight, s_bias = self._stage_ln_affine(storage, epilogue_params, tidx, warp_group_idx)
            a_s2r_copy = None
            if const_expr(self._value_a_in_regs):
                a_s2r_copy = self._make_a_s2r_copy(tiled_mma)

            ab_read_state = make_pipeline_state(pipeline.PipelineUserType.Consumer, self.ab_stage)
            epi_store_pipeline = self.make_epi_store_pipeline()
            tile_scheduler = TileSchedulerCls()
            work_tile = tile_scheduler.initial_work_tile_info()
            while work_tile.is_valid_tile:
                tile_coord_mnkl = work_tile.tile_idx
                ab_read_state = self._mma_stats_xgate(
                    ab_pipeline,
                    ab_read_state,
                    tiled_mma,
                    tiled_mma_gate,
                    acc,
                    acc_gate,
                    tCrA,
                    tCrB,
                    tCrA2,
                    tCrB2,
                    thr_copy,
                    sA,
                    stats_smem,
                    s_weight,
                    s_bias,
                    a_s2r_copy,
                    tidx,
                    warp_group_idx,
                    epilogue_params.eps,
                )
                self._epilogue_xgate(
                    epilogue_params,
                    epi_smem_tensors,
                    epi_store_pipeline,
                    tiled_mma,
                    acc,
                    acc_gate,
                    tile_coord_mnkl,
                    tile_scheduler,
                    tidx,
                    is_tma_warp,
                )
                tile_scheduler.advance_to_next_work()
                work_tile = tile_scheduler.get_current_work()
            if is_tma_warp:
                epi_store_pipeline.producer_tail()

    @cute.jit
    def _epilogue_xgate(
        self,
        params,
        epi_smem_tensors,
        epi_store_pipeline,
        tiled_mma,
        acc,
        acc_gate,
        tile_coord_mnkl,
        tile_scheduler,
        tidx,
        is_tma_warp,
    ):
        """Combine the two accumulators subtile by subtile and store the N-wide result.

        Purpose
            The store leg. The shared
            :meth:`~fold_cp_ops._internal.gemm_sm90_epilogue.GemmSm90Epilogue.epilogue` passes ONE
            accumulator to its per-subtile hooks, so a two-accumulator combine cannot be expressed
            through them and this mirrors its post-activation path instead.

        Semantics
            **Both accumulators are threaded as LOCALS.** Stashing the gate subtile on ``self`` --
            the obvious alternative, and what makes the generic hook look reachable -- puts a value
            defined inside the persistent ``while`` onto an object the loop carries, and the DSL
            then has to thread ``self`` through the region as an SSA value. Upstream hit exactly
            this and resolved it the same way.

            Both accumulators are retiled against the SAME destination fragment layout, which is
            sound because the two MMA atoms differ only in A-major and therefore produce identical
            accumulator M/N layouts.

            There is no D and no C, so this is the post-activation branch of the shared epilogue
            with the D staging, the C load and the C pipeline removed rather than
            ``const_expr``-gated off -- there is nothing on this path that could ever turn them on.

        Args:
            params: The traced epilogue params.
            epi_smem_tensors: The epilogue ops' SMEM tensors.
            epi_store_pipeline: The store pipeline.
            tiled_mma: The value MMA atom, for the store partitioning.
            acc: The value accumulator, holding the RAW ``x @ Bw``.
            acc_gate: The gate accumulator, holding ``x_gate @ Wg``.
            tile_coord_mnkl: This work tile's coordinate.
            tile_scheduler: The scheduler, for the subtile counter the store buffer rotation uses.
            tidx: This thread's index.
            is_tma_warp: Whether this warp issues the TMA store.

        Returns:
            None. Its effect is the store into ``params.mPostAct``.
        """
        tiled_copy_r2s, tRS_rD, _ = self.epilog_smem_store_and_partition(
            tiled_mma, self.d_layout, cutlass.BFloat16, None, tidx
        )
        tRS_rAcc = self.epi_retile_acc(acc, tRS_rD, tiled_copy_r2s)
        tRS_rAccGate = self.epi_retile_acc(acc_gate, tRS_rD, tiled_copy_r2s)
        tRS_rGate = cute.make_rmem_tensor(tRS_rD.layout, self.acc_dtype)

        postact_ctx = self.epi_setup_postact(
            params, epi_smem_tensors, tiled_copy_r2s, None, tile_coord_mnkl, tidx
        )
        epi_tile_shape = cute.zipped_divide(
            cute.make_layout(self.cta_tile_shape_mnk[:2]), self.epi_tile
        ).shape[1]
        epi_tile_layout = cute.make_ordered_layout(epi_tile_shape, order=(1, 0))
        epi_tile_num = cute.size(epi_tile_shape)
        num_prev_subtiles = tile_scheduler.num_tiles_executed * epi_tile_num

        epi_tensors = self.epi_begin(
            params,
            epi_smem_tensors,
            self.epi_tile,
            None,
            tiled_copy_r2s,
            tile_coord_mnkl,
            self.epilogue_barrier,
            tidx,
            tile_scheduler.num_tiles_executed,
        )

        for epi_idx in cutlass.range_constexpr(epi_tile_num):
            gmem_coord = epi_tile_layout.get_hier_coord(epi_idx)
            cute.autovec_copy(tRS_rAcc[None, None, None, epi_idx], tRS_rD)
            cute.autovec_copy(tRS_rAccGate[None, None, None, epi_idx], tRS_rGate)
            epi_loop_tensors = self.epi_begin_loop(params, epi_tensors, gmem_coord)
            tRS_rPostAct = self._xgate_combine_subtile(params, epi_loop_tensors, tRS_rD, tRS_rGate)
            tRS_rPostAct_out = self.epi_convert_postact(
                tRS_rPostAct,
                epi_loop_tensors["sr_seed"],
                tidx,
                tile_coord_mnkl,
                num_prev_subtiles,
                epi_idx,
            )
            if is_tma_warp:
                epi_store_pipeline.producer_acquire()
            self.epilogue_barrier.arrive_and_wait()
            epi_buffer = (num_prev_subtiles + epi_idx) % self.epi_stage
            tiled_copy_postact_r2s, tRS_sPostAct, copy_postact = postact_ctx
            cute.copy(
                tiled_copy_postact_r2s,
                tiled_copy_postact_r2s.retile(tRS_rPostAct_out),
                tRS_sPostAct[None, None, None, epi_buffer],
            )
            cute.arch.fence_view_async_shared()
            self.epilogue_barrier.arrive_and_wait()
            if is_tma_warp:
                copy_postact(src_idx=epi_buffer, dst_idx=gmem_coord)
                epi_store_pipeline.producer_commit()

        self.epi_end(
            params, epi_tensors, self.epi_tile, None, tiled_copy_r2s, tile_coord_mnkl, tidx
        )


class LayerNormDualGatedGemmXGateAlgFoldSm90(_XGateTwoASm90, LayerNormDualGatedGemmAlgFoldSm90):
    """The ALGEBRAIC fold with a SECOND activation: the gate reads its own pre-normalized input.

    Purpose
        The TriMul BACK half's output gate, on the fold. Both one-A fusions form ONE 2N-wide
        pre-activation from ONE activation and split it in the epilogue; here the gate consumes a
        DIFFERENT matrix::

            out[m, n] = sigmoid(x_gate @ Wg^T + bg)[m, n] * (LN(x) @ Wp^T + bp)[m, n]

        ``x_gate`` arrives already normalized -- it is the SAME ``LN(x)`` the caller computed once
        upstream and now feeds to two consumers -- so the gate side takes it RAW.

    Semantics
        `_XGateTwoASm90` carries the two-A shape: four operands per pipeline slot, the gate's own
        MMA atom when its major differs, the halved stage count, the two-accumulator combining
        epilogue. What this class adds is the VALUE arm's fusion, and only that.

        **Only the VALUE side carries statistics.** ``x_gate`` is pre-normalized, so the gate weight
        travels raw, its accumulator gets no rank-one repair and it contributes nothing to the row
        reduction. Only ``x`` is reduced, and only the value accumulator is repaired. Mixing that
        up produces a finite, plausible, entirely wrong gate -- there is no shape or dtype error to
        catch it, which is why the two sides are named apart everywhere below.

        **One sweep over A.** The statistics ride along with the multiply, so the producer's
        inherited single-pass :meth:`~_XGateTwoASm90._load_AB_xgate` is already right and there is
        nothing to override -- which is what :data:`~_XGateTwoASm90._A_SWEEPS` records.

    Attributes:
        _epi_ops: Declared afresh, NOT extended from the parent. The op tuple defines the params
            struct and the shared-memory map, and this epilogue is N-wide with a second projection
            bias where the parent's is 2N-wide with an interleaved one -- there is no prefix of the
            parent's tuple that is correct here.
    """

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        """The two-A epilogue's terms. Declared afresh, not extended -- see the class docstring.

        Attributes:
            mPostAct: ``(l, M, N)`` 16-bit output, N-wide. The gate does not halve it: a gated
                column here is one output column, not two folded into one. May be m-major for the
                transposed store.
            act_fn: The gate activation, a `Constexpr` callable of ONE argument (sigmoid). Distinct
                from the dual's two-argument gate, which takes ``(gate, up)``.
            sRstd, sScale: Placeholders for the two `SmemColVecBroadcast` ops. **Always None.**
                Their data is the mainloop's shared scratch, supplied by
                :meth:`~LayerNormDualGatedGemmAlgFoldSm90.epi_get_smem_tensors`; the fields exist
                only because the composition machinery expects one per declared op.
            mColsum: ``(l, N)`` fp32 ``c = colsum(Bw)`` for the VALUE weight. Required. Without it
                the repair subtracts nothing and the value arm is the raw ``r*(x@Bw)`` -- finite
                and wrong.
            mDbias: ``(l, N)`` fp32 ``d = b_ln @ Wp``, or None when there is no LayerNorm bias.
            mBiasUp: ``(l, N)`` fp32 value-projection bias ``bp``, or None. A FULL-width vector,
                not a half of an interleaved one: the two projections have separate accumulators
                here, so each takes its own bias directly.
            mBiasGate: ``(l, N)`` fp32 gate-projection bias ``bg``, or None.
            mMaskColVec: ``(l, M)`` fp32 per-row multiplicative mask, or None. Applied in fp32
                after the combine and before the narrowing store.
            eps: The LayerNorm variance floor. A RUNTIME scalar; baking it would key the artifact
                on it.
            rounding_mode: `Constexpr`. Must be `RoundingMode.RN`.
            sr_seed: Stochastic-rounding seed; unused under RN.

        Note:
            The `Constexpr` fields are erased from the ABI, so at LAUNCH they must be passed None.
        """

        mPostAct: cute.Tensor
        act_fn: cutlass.Constexpr[Optional[Callable]] = None
        sRstd: Optional[cute.Tensor] = None
        sScale: Optional[cute.Tensor] = None
        mColsum: Optional[cute.Tensor] = None
        mDbias: Optional[cute.Tensor] = None
        mBiasUp: Optional[cute.Tensor] = None
        mBiasGate: Optional[cute.Tensor] = None
        mMaskColVec: Optional[cute.Tensor] = None
        eps: Float32 = Float32(1e-5)
        rounding_mode: cutlass.Constexpr[int] = RoundingMode.RN
        sr_seed: Optional[Int32 | cute.Tensor] = None

    #: The N-wide two-A epilogue's terms, in the order the params struct and the SMEM map take
    #: them. `ColVecLoad` rather than the dual's `TransposedMaskColVecLoad`: the two are identical
    #: for the rank-2 ``(l, M)`` mask this path builds -- the transposed op delegates to this one --
    #: and naming the simpler one keeps the declaration honest about what is actually read.
    _epi_ops = (
        Scalar("sr_seed", dtype=Int32),
        SmemColVecBroadcast("sRstd", slot=0),
        SmemColVecBroadcast("sScale", slot=1),
        RowVecLoad("mColsum"),
        RowVecLoad("mDbias"),
        RowVecLoad("mBiasUp"),
        RowVecLoad("mBiasGate"),
        ColVecLoad("mMaskColVec"),
        TileStore("mPostAct"),
    )
    #: ``act_fn`` is the gate activation and ``eps`` the LayerNorm floor. Neither is a loaded
    #: tensor, so neither can be an `_epi_ops` entry; the composition mixin splices declared extra
    #: fields into the generated ``EpilogueParams``.
    _extra_param_fields = (
        ("act_fn", cutlass.Constexpr, None),
        ("eps", Float32, Float32(1e-5)),
    )

    def epi_to_underlying_arguments(self, args, *, loc=None, ip=None):
        """Lower the launch arguments into the traced params: the ops, the gate activation, `eps`.

        Purpose
            The parent's version routes through `DualGatedGemmSm90.gated_params_dict`, which checks
            a 2N geometry, halves the post-activation tile and contributes the output gate's entry.
            None of the three applies here, so this goes to `GemmActMixin`'s full-width version
            instead and adds ``eps``.

        Semantics
            `GemmActMixin._latch_postact_attributes` sets ``cta_tile_shape_postact_mn`` to the FULL
            CTA ``(M, N)``, which is what an unhalved store wants, and validates the three
            post-activation requirements (16-bit, a known major mode, round-to-nearest).

        Args:
            args: This launch's :class:`EpilogueArguments`.
            loc: Optional DSL source location.
            ip: Optional DSL insertion point.

        Returns:
            An ``EpilogueParams`` with one entry per declared op, plus ``act_fn`` and ``eps``.

        Raises:
            AssertionError: Propagated from `GemmActMixin._latch_postact_attributes` for a
                post-activation that is not 16-bit, is neither n- nor m-major, or asks for
                stochastic rounding.
        """
        self._latch_postact_attributes(args)
        d = self._epi_ops_to_params_dict(args)
        d["act_fn"] = args.act_fn
        d["eps"] = args.eps
        return self.EpilogueParams(**d)

    @cute.jit
    def _mma_stats_xgate(
        self,
        ab_pipeline,
        ab_read_state,
        tiled_mma,
        tiled_mma_gate,
        acc,
        acc_gate,
        tCrA,
        tCrB,
        tCrA2,
        tCrB2,
        thr_copy,
        sA,
        stats_smem,
        s_weight,
        s_bias,
        a_s2r_copy,
        tidx,
        warp_group_idx,
        eps,
    ):
        """One work tile: two WGMMAs and the value side's row reduction, per k-tile.

        Purpose
            The mainloop. Per staged k-tile it issues the value WGMMA (``x @ Bw``), the gate WGMMA
            (``x_gate @ Wg``) and the value side's ``Sum x`` / ``Sum x^2`` reduction, then releases
            the stage.

        Semantics
            **It fully drains both WGMMA groups before releasing, and that is a correctness
            requirement rather than a scheduling choice.** The base's :meth:`mma` keeps a 1-deep
            software pipeline -- ``wait_group(1)`` releases the stage the PREVIOUS k-tile read while
            this one is in flight. Two ``commit_group``s per k-tile break that invariant: after the
            gate's commit, one outstanding group is this tile's value WGMMA, not the previous
            tile's, so a 1-deep wait would free a stage a running WGMMA is still reading. The cost
            is the intra-tile overlap only; the producer's multi-stage pipeline still overlaps the
            NEXT tile's loads with this tile's compute.

            **The reduction reads only ``sA``.** The gate's activation arrived normalized, so it
            contributes no statistics; reducing it too would produce a plausible wrong ``rstd``.
            The read is issued AFTER both WGMMAs, so it overlaps them rather than serializing with
            the MMA warps' own operand fetch.

            The k-loop is ``range_constexpr``-unrolled over the compile-time tile count, which is
            what keeps the partials straight-line SSA instead of state carried across a dynamic
            MLIR region -- the same reason the fold's cooperative mainloop unrolls, measured at
            ~10% there.

            After the loop the partials become ``(r, s) = (rstd, rstd*mu)`` in the shared scratch,
            already multiplied, so the epilogue's repair costs one multiply per element rather than
            two.

        Args:
            ab_pipeline: The mainloop pipeline.
            ab_read_state: The consumer state, threaded across work tiles and returned.
            tiled_mma: The value MMA atom.
            tiled_mma_gate: The gate MMA atom.
            acc: The value accumulator, overwritten by the first k-tile and accumulated after.
            acc_gate: The gate accumulator, likewise.
            tCrA, tCrB: The value operand fragments.
            tCrA2, tCrB2: The gate operand fragments.
            thr_copy: This thread's slice of the statistics tiled copy. Must have been built for
                ``sA``'s tile shape, or the reduction reads the wrong elements.
            sA: The staged VALUE activation -- the only operand that carries statistics.
            stats_smem: This warpgroup's ``(tile_M, 2)`` fp32 scratch.
            s_weight: The staged ``(gemm_k,)`` LayerNorm gain. **Always None here and never read**
                -- this fusion absorbed the gain into the WEIGHT on the host. It is in the
                signature because the shared consumer calls both variants' mainloops through one
                call site, and a signature that differed per variant would move the decision from
                a declaration to a branch.
            s_bias: The staged ``(gemm_k,)`` LayerNorm bias. Always None here, for the same reason.
            a_s2r_copy: The shared-memory -> register A copy. Always None here and never read: this
                fusion reads the staged value tile ONCE, so it has nothing to gain from moving it
                into registers first. Same reason as the two above.
            tidx: This thread's index within the CTA.
            warp_group_idx: Which MMA warpgroup; selects the publishing barrier.
            eps: The LayerNorm variance floor.

        Returns:
            The advanced consumer state.
        """
        k_tile_cnt_const = const_expr(self.gemm_k // self.blk_k)
        mma_fn = partial(fold_cp_ops_sm90_utils.gemm_w_idx, tiled_mma, acc, tCrA, tCrB)
        gate_fn = partial(fold_cp_ops_sm90_utils.gemm_w_idx, tiled_mma_gate, acc_gate, tCrA2, tCrB2)
        row_sum, row_sqsum = make_row_stat_partials(thr_copy, sA[None, None, 0])

        peek_ab_full_status = Boolean(True)
        if const_expr(k_tile_cnt_const > 0):
            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)
        zero_init = Boolean(True)
        for k_tile in cutlass.range_constexpr(k_tile_cnt_const):
            ab_pipeline.consumer_wait(ab_read_state, peek_ab_full_status)
            mma_fn(A_idx=ab_read_state.index, B_idx=ab_read_state.index, zero_init=zero_init)
            gate_fn(A_idx=ab_read_state.index, B_idx=ab_read_state.index, zero_init=zero_init)
            accumulate_row_stats(thr_copy, sA[None, None, ab_read_state.index], row_sum, row_sqsum)
            zero_init = Boolean(False)
            warpgroup.wait_group(0)  # drain BOTH groups before this stage is released
            # `wait_group(0)` above drains the two WGMMAs' reads of this stage; it does NOT drain
            # the plain `LDS` that `accumulate_row_stats` just issued over the same tile. Without
            # this fence the release below is signalled with those loads still in flight, the
            # producer refills the stage, and they return the NEXT k-tile's activations -- a silent
            # wrong answer. Measured at M=32768: N=288 40/40, N=320 29/40, N=448 40/40 corrupt
            # without it, 0/40 at all three with it. See `accumulate_row_stats` for the mechanism,
            # for why one fence per warp is sufficient, and for why this sits here rather than in
            # that function (placing it there segfaults the compiler on the fold paths).
            cute.arch.fence_view_async_shared()
            ab_pipeline.consumer_release(ab_read_state)
            ab_read_state.advance()
            peek_ab_full_status = Boolean(True)
            if const_expr(k_tile + 1 < k_tile_cnt_const):
                peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)

        finalize_row_stats(
            stats_smem,
            tidx,
            row_sum,
            row_sqsum,
            Int32(self.gemm_k_real),
            eps,
            STATS_THREADS_PER_ROW,
            self._stats_num_threads(),
            self._stats_barrier(warp_group_idx),
            publish_fold_scale=True,
        )
        return ab_read_state

    @cute.jit
    def _xgate_combine_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rGate):
        """Repair the value, activate the gate, multiply. One epilogue subtile, in fp32.

        Purpose
            The kernel's arithmetic, and the only place the two accumulators meet::

                value[i] = r[i]*acc[i] - s[i]*c[i] (+ d[i]) + bp[i]
                gate[i]  = act(acc_gate[i] (+ bg[i]))
                out[i]   = gate[i] * value[i]              (* mask[i])

        Semantics
            **Only the VALUE arm is repaired.** ``r`` and ``s`` undo the fold on ``x @ Bw``; the
            gate's activation was normalized before it ever reached this kernel, so its accumulator
            is already what it should be. Applying the repair to both is the single most plausible
            way to get this wrong, and nothing downstream would notice.

            The order within the value arm is the fold's: the rank-one correction belongs INSIDE
            the identity being repaired, the projection bias is added to the finished
            pre-activation. Adding the bias first would scale it by the row's ``rstd``.

            **The mask indexes DIRECTLY.** The dual path indexes ``tDrMask[2*i]`` because its
            post-activation is half its pre-activation's width; this path is full-N, so the
            post-activation shares the accumulator's index space. The mask is a column vector, so
            it is constant along N either way -- which is why the wrong indexing would still
            produce a finite, wrong answer rather than an out-of-range read.

            Every optional term is ``const_expr``-pruned, so a kernel built without a LayerNorm
            bias, a projection bias or a mask emits none of them.

        Args:
            params: This kernel's ``EpilogueParams``; ``act_fn`` (the gate activation) is read here.
            epi_loop_tensors: This subtile's loaded terms by op name -- ``sRstd``, ``sScale``,
                ``mColsum``, ``mDbias``, ``mBiasUp``, ``mBiasGate``, ``mMaskColVec``.
            tRS_rD: The VALUE accumulator subtile, repaired IN PLACE.
            tRS_rGate: The GATE accumulator subtile, read but not modified.

        Returns:
            A fresh fp32 fragment holding the combined result, the same size as `tRS_rD`.
        """
        self.epi_repair_rank_one(epi_loop_tensors, tRS_rD)
        bp = epi_loop_tensors["mBiasUp"]
        if const_expr(bp is not None):
            for i in cutlass.range(cute.size(tRS_rD), unroll_full=True):
                tRS_rD[i] = tRS_rD[i] + bp[i]
        bg = epi_loop_tensors["mBiasGate"]
        tRS_rPostAct = cute.make_rmem_tensor(tRS_rD.layout.shape, self.acc_dtype)
        for i in cutlass.range(cute.size(tRS_rPostAct), unroll_full=True):
            g_i = tRS_rGate[i] + bg[i] if const_expr(bg is not None) else tRS_rGate[i]
            tRS_rPostAct[i] = params.act_fn(g_i) * tRS_rD[i]
        tDrMask = epi_loop_tensors["mMaskColVec"]
        if const_expr(tDrMask is not None):
            for i in cutlass.range(cute.size(tRS_rPostAct), unroll_full=True):
                tRS_rPostAct[i] = tRS_rPostAct[i] * tDrMask[i]
        return tRS_rPostAct


class LayerNormDualGatedGemmXGatePrologLnSm90(_XGateTwoASm90, LayerNormDualGatedGemmSm90):
    """The PHYSICAL fusion with a SECOND activation: the value is normalized in shared memory.

    Purpose
        The TriMul BACK half's output gate at ``fusion_variant="prolog_ln"``, and the branch the
        cp=1 heuristic returns for most production feature widths. It computes exactly what
        `LayerNormDualGatedGemmXGateAlgFoldSm90` computes::

            out[m, n] = sigmoid(x_gate @ Wg^T + bg)[m, n] * (LN(x) @ Wp^T + bp)[m, n]

        and gets the value arm's ``LN(x)`` the other way: the staged ``x`` is rewritten in shared
        memory, so the value WGMMA is an ordinary GEMM and the epilogue has nothing to repair.

    Semantics
        **Two staged A operands is what makes this variant reachable at all.** The one-A
        ``prolog_ln`` rewrites THE staged activation as ``LN(x)``, which is the one thing a gate
        reading its own pre-normalized input must not be handed -- and that was the stated reason
        the combination was refused. With two, ``sA`` is rewritten and ``sA2`` stays raw, so the
        objection dissolves rather than being worked around. Upstream implements the same
        derivation from the same base.

        **Two sweeps over A, so the producer stages every k-tile twice.** LayerNorm needs the whole
        row before it can scale any of it: pass 1 reduces ``Sum x`` / ``Sum x^2`` over ``sA`` and
        releases each stage immediately, the finalize publishes ``(mu, rstd)``, and pass 2 re-reads
        each tile, normalizes it IN PLACE, and issues both WGMMAs. The gate's ``sA2`` is loaded on
        both passes and read on neither the first nor normalized on the second -- the per-stage
        transaction count is fixed at construction, so a pass that committed fewer bytes would
        leave the consumer waiting on a barrier that never completes.

        **Only the VALUE side carries statistics, and only ``sA`` is rewritten.** ``x_gate`` was
        normalized before it reached this kernel, so reducing it too would publish a plausible
        wrong ``rstd``, and normalizing it would normalize twice. Neither has a shape or dtype that
        could catch it.

        **The epilogue is a plain combine.** The value accumulator already holds ``LN(x) @ Wp``, so
        there are no statistics colvecs and no ``colsum``/``dbias`` rowvecs -- the four ops the
        algebraic fold's epilogue needs are exactly the four this one does not declare.

    Attributes:
        _epi_ops: Declared afresh. Compared with the fold's sibling this is the same tuple MINUS
            the rank-one repair's four inputs; declaring it as a subset rather than deriving one
            from the other keeps the op ORDER -- which defines the params struct and the
            shared-memory map -- readable in one place.
        _MIN_XGATE_AB_STAGE: 2. One stage cannot overlap a load with a compute, so the two-pass
            streaming loop would serialize completely.
        _A_SWEEPS: 2, matched by :meth:`_load_AB_xgate`.
    """

    #: The two-pass streaming mainloop needs a real ring: at one stage the producer's next load
    #: cannot start until the consumer has released the only slot, so both passes serialize. The
    #: fold's sibling floors at 1 because one sweep still overlaps across work tiles.
    _MIN_XGATE_AB_STAGE = 2

    #: Pass 1 reduces, pass 2 normalizes and multiplies -- so the producer stages every k-tile
    #: twice, which :meth:`_load_AB_xgate` implements and this records.
    _A_SWEEPS = 2

    #: This is the fusion the register-source mainloop exists for, and the reason is exactly the
    #: two sweeps above: an MN-major staged tile is strided along K, and reading it twice out of
    #: shared memory costs 1.13-1.35x against the kernel this reproduces across the workflow ladder.
    #: The algebraic fold reads it ONCE, so it has nothing to recover and leaves this False.
    _SUPPORTS_A_IN_REGS = True

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        """The physical two-A epilogue's terms, plus the LayerNorm prologue's own three inputs.

        A NamedTuple is positional on the wire, so the order below IS the ABI; re-ordering any two
        fields silently hands each the other's value.

        Attributes:
            mPostAct: ``(l, M, N)`` 16-bit output, N-wide. The gate does not halve it: a gated
                column here is one output column, not two folded into one. May be m-major for the
                transposed store.
            act_fn: The gate activation, a `Constexpr` callable of ONE argument (sigmoid). Distinct
                from the dual's two-argument gate, which takes ``(gate, up)``.
            mBiasUp: ``(l, N)`` fp32 value-projection bias ``bp``, or None. A FULL-width vector,
                not a half of an interleaved one: the two projections have separate accumulators
                here, so each takes its own bias directly.
            mBiasGate: ``(l, N)`` fp32 gate-projection bias ``bg``, or None.
            mMaskColVec: ``(l, M)`` fp32 per-row multiplicative mask, or None. Applied in fp32
                after the combine and before the narrowing store.
            mWeight: ``(K,)`` fp32 LayerNorm gain. **fp32 is required, not converted** -- a 16-bit
                gain would lose more precision than the normalize it scales. Required, and read by
                the MAINLOOP rather than the epilogue: this struct is simply the only one that
                crosses into the kernel region.
            mBias: ``(K,)`` fp32 LayerNorm bias, or None. None compiles the add out entirely.
            eps: The LayerNorm variance floor. A RUNTIME scalar; baking it would key the artifact
                on it.
            rounding_mode: `Constexpr`. Must be `RoundingMode.RN`.
            sr_seed: Stochastic-rounding seed; unused under RN.

        Note:
            The `Constexpr` fields are erased from the ABI, so at LAUNCH they must be passed None.
        """

        mPostAct: cute.Tensor
        act_fn: cutlass.Constexpr[Optional[Callable]] = None
        mBiasUp: Optional[cute.Tensor] = None
        mBiasGate: Optional[cute.Tensor] = None
        mMaskColVec: Optional[cute.Tensor] = None
        mWeight: Optional[cute.Tensor] = None
        mBias: Optional[cute.Tensor] = None
        eps: Float32 = Float32(1e-5)
        rounding_mode: cutlass.Constexpr[int] = RoundingMode.RN
        sr_seed: Optional[Int32 | cute.Tensor] = None

    #: The N-wide two-A epilogue's terms with NO statistics and NO rank-one inputs: the value WGMMA
    #: already produced ``LN(x) @ Wp`` from a normalized ``sA``, so there is nothing to repair.
    #: `ColVecLoad` rather than the dual's `TransposedMaskColVecLoad` for the same reason the fold's
    #: sibling names it: the two are identical for the rank-2 ``(l, M)`` mask this path builds.
    _epi_ops = (
        Scalar("sr_seed", dtype=Int32),
        RowVecLoad("mBiasUp"),
        RowVecLoad("mBiasGate"),
        ColVecLoad("mMaskColVec"),
        TileStore("mPostAct"),
    )
    #: The gate activation, the LayerNorm gain/bias and the variance floor. None of the four is a
    #: loaded epilogue term, so none can be an `_epi_ops` entry; the composition mixin splices
    #: declared extra fields into the generated ``EpilogueParams``. The gain and bias are carried
    #: here because ``EpilogueParams`` is the only struct that reaches the kernel region, and a
    #: mainloop reading them off ``self`` instead would capture host-side values that do not
    #: survive the launch.
    _extra_param_fields = (
        ("act_fn", cutlass.Constexpr, None),
        ("mWeight", Optional[cute.Tensor], None),
        ("mBias", Optional[cute.Tensor], None),
        ("eps", Float32, Float32(1e-5)),
    )

    def epi_to_underlying_arguments(self, args, *, loc=None, ip=None):
        """Lower the launch arguments: the ops, the gate activation, the gain/bias pair and `eps`.

        Purpose
            The fusion base's version routes through `DualGatedGemmSm90.gated_params_dict`, which
            checks a 2N geometry, halves the post-activation tile and contributes the output gate's
            entry. None of the three applies here, so this goes to `GemmActMixin`'s full-width
            version instead and adds the four extra fields.

        Semantics
            `GemmActMixin._latch_postact_attributes` sets ``cta_tile_shape_postact_mn`` to the FULL
            CTA ``(M, N)``, which is what an unhalved store wants, and validates the three
            post-activation requirements (16-bit, a known major mode, round-to-nearest).

            ``mWeight``/``mBias`` are threaded through this struct although the MAINLOOP is what
            reads them; see :attr:`_extra_param_fields` for why that is the only route.

        Args:
            args: This launch's :class:`EpilogueArguments`.
            loc: Optional DSL source location.
            ip: Optional DSL insertion point.

        Returns:
            An ``EpilogueParams`` with one entry per declared op, plus ``act_fn``, ``mWeight``,
            ``mBias`` and ``eps``.

        Raises:
            AssertionError: Propagated from `GemmActMixin._latch_postact_attributes` for a
                post-activation that is not 16-bit, is neither n- nor m-major, or asks for
                stochastic rounding.
        """
        self._latch_postact_attributes(args)
        d = self._epi_ops_to_params_dict(args)
        d["act_fn"] = args.act_fn
        d["mWeight"] = args.mWeight
        d["mBias"] = args.mBias
        d["eps"] = args.eps
        return self.EpilogueParams(**d)

    @cute.jit
    def _load_AB_xgate(
        self, ab_pipeline, ab_producer_state, copy_A, copy_A2, copy_B, copy_B2, k_tile_cnt
    ):
        """Stage all four operands TWICE per k-tile -- the producer half of the two-pass prologue.

        Purpose
            The consumer sweeps A twice, so the producer must supply it twice. Expressing that as
            two calls to the mixin's single-pass loop, rather than as a second loop written here,
            is what keeps the acquire/commit protocol in one place -- and is the same shape the
            one-A `LayerNormDualGatedGemmSm90.load_AB` uses against its own base.

        Semantics
            The GATE's two operands are staged on both passes even though pass 1 reads neither.
            That is not waste that could be removed: the pipeline's per-stage transaction count is
            all four operands' bytes, fixed at construction, and a pass that committed fewer would
            leave the consumer waiting on a barrier that never completes. It is a HANG, not a slow
            path.

            **The explicit two-argument ``super()`` is required, not stylistic**: the DSL re-derives
            a ``@cute.jit`` function from source, and the zero-argument form then resolves against
            ``type(self)``, so on a subclass it re-enters this method and recurses until the stack
            ends.

        Args:
            ab_pipeline: The mainloop pipeline.
            ab_producer_state: The producer state; advanced ``2 * k_tile_cnt`` times and returned.
            copy_A: Value A's copy closure.
            copy_A2: Gate A's.
            copy_B: Value B's.
            copy_B2: Gate B's.
            k_tile_cnt: k-tiles per PASS. Must be the value the consumer derives from
                ``gemm_k // blk_k``, or the two deadlock against each other.

        Returns:
            The advanced producer state.
        """
        ab_producer_state = super(LayerNormDualGatedGemmXGatePrologLnSm90, self)._load_AB_xgate(
            ab_pipeline, ab_producer_state, copy_A, copy_A2, copy_B, copy_B2, k_tile_cnt
        )
        return super(LayerNormDualGatedGemmXGatePrologLnSm90, self)._load_AB_xgate(
            ab_pipeline, ab_producer_state, copy_A, copy_A2, copy_B, copy_B2, k_tile_cnt
        )

    @cute.jit
    def _mma_stats_xgate(
        self,
        ab_pipeline,
        ab_read_state,
        tiled_mma,
        tiled_mma_gate,
        acc,
        acc_gate,
        tCrA,
        tCrB,
        tCrA2,
        tCrB2,
        thr_copy,
        sA,
        stats_smem,
        s_weight,
        s_bias,
        a_s2r_copy,
        tidx,
        warp_group_idx,
        eps,
    ):
        """One work tile: reduce over ``sA``, normalize it in place, then run both WGMMAs.

        Purpose
            The mainloop, and the whole of what this fusion is. Pass 1 reduces the value
            activation's rows; pass 2 rewrites the same tiles as ``LayerNorm(x)`` and issues the
            value WGMMA ``LN(x) @ Wp`` beside the gate WGMMA ``x_gate @ Wg``.

            **For an MN-major value activation it delegates to the REGISTER-source form instead**,
            which does the same two passes out of the WGMMA's operand registers rather than out of
            shared memory. The branch is `const_expr` on :attr:`~_XGateTwoASm90._value_a_in_regs`,
            so exactly one of the two bodies is traced.

        Semantics
            **Pass 1 releases each stage as soon as its values are in registers**, so nothing is
            held and the ring keeps turning at its natural depth. `finalize_row_stats` then
            publishes ``(mu, rstd)`` -- NOT the fold's ``(rstd, rstd*mu)``, because what reads them
            here is `normalize_tile`, which rescales rows.

            **Pass 2 fences and barriers between the normalize and the WGMMAs**: the rewrite is a
            shared-memory store issued by the threads that own those elements, and every MMA
            warpgroup must see all of it before its WGMMA reads the tile.

            **It fully drains both WGMMA groups before releasing, and that is a correctness
            requirement rather than a scheduling choice.** The base's :meth:`mma` keeps a 1-deep
            software pipeline -- ``wait_group(1)`` releases the stage the PREVIOUS k-tile read while
            this one is in flight. Two ``commit_group``s per k-tile break that invariant: after the
            gate's commit, one outstanding group is this tile's value WGMMA, not the previous
            tile's, so a 1-deep wait would free a stage a running WGMMA is still reading. Draining
            is also what makes the release safe against the NORMALIZE: the tile pass 2 frees is the
            tile the MMA just read, and a later k-tile's normalize writes into that same buffer.

            **The reduction reads only ``sA``.** The gate's activation arrived normalized, so it
            contributes no statistics and is never rewritten; reducing or normalizing it would
            produce a plausible wrong answer with nothing to catch it.

            Both k-loops are ``range_constexpr``-unrolled over the compile-time tile count, which
            is what keeps the partials straight-line SSA instead of state carried across a dynamic
            MLIR region -- measured at ~10% on the fold's cooperative mainloop, and worth 7-20% on
            the one-A physical fusion.

            The ONE ``ab_read_state`` crosses both passes, so its ``2 * k_tile_cnt`` waits stay in
            lock-step with the producer's ``2 * k_tile_cnt`` loads.

        Args:
            ab_pipeline: The mainloop pipeline.
            ab_read_state: The consumer state, threaded across work tiles and returned.
            tiled_mma: The value MMA atom.
            tiled_mma_gate: The gate MMA atom.
            acc: The value accumulator, overwritten by the first k-tile and accumulated after.
            acc_gate: The gate accumulator, likewise.
            tCrA, tCrB: The value operand fragments.
            tCrA2, tCrB2: The gate operand fragments.
            thr_copy: This thread's slice of the statistics tiled copy. The SAME slice drives the
                reduction and the normalize, which is what guarantees a thread rescales the rows it
                helped reduce; a different partition here is a wrong answer, not a fault.
            sA: The staged VALUE activation. Read in pass 1 and **written** in pass 2.
            stats_smem: This warpgroup's ``(tile_M, 2)`` fp32 scratch.
            s_weight: The staged ``(gemm_k,)`` fp32 LayerNorm gain. Required -- this fusion applies
                it to the ACTIVATION, so it is never None here.
            s_bias: The staged ``(gemm_k,)`` fp32 LayerNorm bias, or None. None prunes the add.
            a_s2r_copy: The shared-memory -> register A copy, or None. Non-None exactly when
                :attr:`~_XGateTwoASm90._value_a_in_regs`, which is what selects the body below.
            tidx: This thread's index within the CTA.
            warp_group_idx: Which MMA warpgroup; selects the publishing barrier.
            eps: The LayerNorm variance floor.

        Returns:
            The advanced consumer state.
        """
        if const_expr(self._value_a_in_regs):
            return self._mma_stats_xgate_rs(
                ab_pipeline,
                ab_read_state,
                tiled_mma,
                tiled_mma_gate,
                acc,
                acc_gate,
                tCrA,
                tCrB,
                tCrA2,
                tCrB2,
                a_s2r_copy,
                sA,
                stats_smem,
                s_weight,
                s_bias,
                tidx,
                eps,
            )
        k_tile_cnt = const_expr(self.gemm_k // self.blk_k)
        mma_fn = partial(fold_cp_ops_sm90_utils.gemm_w_idx, tiled_mma, acc, tCrA, tCrB)
        gate_fn = partial(fold_cp_ops_sm90_utils.gemm_w_idx, tiled_mma_gate, acc_gate, tCrA2, tCrB2)
        row_sum, row_sqsum = make_row_stat_partials(thr_copy, sA[None, None, 0])

        # PASS 1 -- reduce, releasing each tile as soon as its values are in registers.
        peek = Boolean(True)
        if const_expr(0 < k_tile_cnt):
            peek = ab_pipeline.consumer_try_wait(ab_read_state)
        for k_idx in cutlass.range_constexpr(k_tile_cnt):
            ab_pipeline.consumer_wait(ab_read_state, peek)
            stage = ab_read_state.index
            row_sum, row_sqsum = accumulate_row_stats(
                thr_copy, sA[None, None, stage], row_sum, row_sqsum
            )
            # **There is deliberately NO `fence_view_async_shared` here, and the reason is a
            # MEASUREMENT rather than an argument.** The staged tile is read by plain `LDS` and
            # released with nothing draining them, which is the hazard `accumulate_row_stats`
            # documents and which the one-sweep sibling carries a fence for -- so the fence looks
            # owed here too. It is not: at ``M=32768``, 40 pairs a shape, this loop is **0/40
            # unfenced** at N=288, 320 and 448, the three shapes where that sibling is 40/40,
            # 29/40 and 40/40 corrupt. The probe was verified sensitive on the same run by deleting
            # the sibling's fence, which reproduced 40/40 at N=320 and N=448.
            #
            # The likely reason it is clean is structural: this pass issues NO WGMMA, so there is
            # no `wait_group` for the compiler to sink the reduction past, and the producer must
            # re-stage every tile for pass 2 before it can overtake pass 1. Whatever the mechanism,
            # an unobserved hazard is not a defect: adding a fence would diverge from the kernel
            # this reproduces on prophylaxis, which this package refuses on principle. If it is
            # ever observed here, the fence goes back at exactly this line.
            ab_pipeline.consumer_release(ab_read_state)
            ab_read_state.advance()
            peek = Boolean(True)
            if const_expr(k_idx + 1 < k_tile_cnt):
                peek = ab_pipeline.consumer_try_wait(ab_read_state)
        finalize_row_stats(
            stats_smem,
            tidx,
            row_sum,
            row_sqsum,
            Int32(self.gemm_k_real),
            eps,
            STATS_THREADS_PER_ROW,
            self._stats_num_threads(),
            self._stats_barrier(warp_group_idx),
            # The persistent-tile stats-race guard the upstream `staged` kernel emits
            # (`_stats_race_guard`, True on its default streaming mode). Restored HERE and
            # not on the alg_fold siblings, whose `stagec` counterpart has no such guard.
            leading_barrier=True,
        )

        # PASS 2 -- normalize in place, then multiply. See the docstring for why both WGMMA groups
        # are drained before the release rather than pipelined across k-tiles.
        peek = Boolean(True)
        if const_expr(0 < k_tile_cnt):
            peek = ab_pipeline.consumer_try_wait(ab_read_state)
        for k_idx in cutlass.range_constexpr(k_tile_cnt):
            ab_pipeline.consumer_wait(ab_read_state, peek)
            stage = ab_read_state.index
            normalize_tile(
                sA[None, None, stage],
                stats_smem,
                s_weight,
                s_bias,
                thr_copy,
                Int32(k_idx),
                (self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[2]),
                self.a_dtype,
            )
            cute.arch.fence_view_async_shared()
            self.epilogue_barrier.arrive_and_wait()
            mma_fn(A_idx=stage, B_idx=stage, zero_init=Boolean(k_idx == 0))
            gate_fn(A_idx=stage, B_idx=stage, zero_init=Boolean(k_idx == 0))
            warpgroup.wait_group(0)  # drain BOTH groups before this stage is released
            ab_pipeline.consumer_release(ab_read_state)
            ab_read_state.advance()
            peek = Boolean(True)
            if const_expr(k_idx + 1 < k_tile_cnt):
                peek = ab_pipeline.consumer_try_wait(ab_read_state)
        return ab_read_state

    # ── the register-source value mainloop: MN-major activations only ────────────────────────
    # Everything below is reached ONLY when `_value_a_in_regs` is true, i.e. when the value
    # activation is MN-major. The register layout closed forms are the WGMMA operand-A fragment's
    # for the cooperative ``(2, 1, 1)`` atom at ``tile_M = 128``; they do not depend on the K-tile
    # count, so the same forms serve every ``blk_k`` and only ``cute.size(tCrA)`` changes -- which
    # the ``range_constexpr`` loops absorb. They are the kernel this reproduces', verbatim.

    @cute.jit
    def _load_a_to_rf(self, a_s2r_copy, a_tidx, sA_stage, tCrA):
        """``ldmatrix`` one staged value tile into the register-source operand-A fragment.

        Purpose
            The register path's load. One ``cute.copy`` fills the whole ``(tile_M, blk_k)`` tile;
            the retile maps the copy's per-thread destination onto the WGMMA fragment.

        Semantics
            For an MN-major ``sA_stage`` the atom is the TRANSPOSING ``ldmatrix``, so the strided
            shared-memory read is bank-conflict-free and the registers come out K-ordered -- which
            is what the reduction and the normalize below index.

        Args:
            a_s2r_copy: The tiled copy from :meth:`~_XGateTwoASm90._make_a_s2r_copy`. Must have been
                built from the SAME atom ``tCrA`` was partitioned from, or the destination layout
                does not match and the fragment is filled with the wrong elements.
            a_tidx: This thread's index within the MMA warpgroups.
            sA_stage: One staged ``(tile_M, blk_k)`` value tile. Read only.
            tCrA: The register A fragment, OVERWRITTEN.

        Returns:
            None; `tCrA` is filled.
        """
        thr_copy = a_s2r_copy.get_slice(a_tidx)
        cute.copy(thr_copy, thr_copy.partition_S(sA_stage), a_s2r_copy.retile(tCrA))

    @cute.jit
    def _accumulate_stats_rf(self, tCrA, row_sum, row_sqsum):
        """Add this k-tile's per-row ``Sum x`` / ``Sum x^2`` from the REGISTER fragment.

        Purpose
            The register path's pass 1. It replaces `accumulate_row_stats`'s second shared-memory
            read with a reduction over registers that are already loaded for the WGMMA.

        Semantics
            The WGMMA operand-A value mode is ``(k_lo, M, k_hi)``, so this thread owns TWO M-rows --
            ``base`` and ``base + 8`` -- and a disjoint share of each row's K. The reduction profile
            keeps ONLY that M sub-mode and sums everything else, leaving a partial per owned row;
            the four quad lanes that split a row's K are combined later, in
            :meth:`_finalize_stats_local_rf`, so the cross-lane step runs once rather than per
            k-tile.

            ``Sum x^2`` is accumulated directly rather than as a centred sum, because the mean is
            not known until every tile has been seen -- the same trade `accumulate_row_stats` makes,
            and bounded for the same reason: the activation is 16-bit, so ``x^2`` is exact in fp32
            and only the summation rounds.

        Args:
            tCrA: The register A fragment for this k-tile, already filled. Read only.
            row_sum: This thread's two fp32 ``Sum x`` partials, updated in place AND returned.
            row_sqsum: Likewise for ``Sum x^2``.

        Returns:
            ``(row_sum, row_sqsum)`` -- the same fragments, returned so the caller's k-loop can
            rebind them.
        """
        # SHARED, like `rs_owned_rows`: the reduction profile encodes the same empirically-derived
        # fragment layout, so a drifting copy would sum across ROWS and still look plausible.
        return rs_accumulate_row_stats(tCrA, row_sum, row_sqsum)

    @cute.jit
    def _finalize_stats_local_rf(self, stats_smem, a_tidx, row_sum, row_sqsum, n_elems, eps):
        """Complete the register partials into ``(mu, rstd)`` in the shared statistics scratch.

        Purpose
            The register path's counterpart of `finalize_row_stats`. It cannot use that helper: the
            cooperating group here is the 4-lane QUAD that splits a row's K in the WGMMA operand
            layout, not the ``STATS_THREADS_PER_ROW`` lanes of the statistics tiled copy, and the
            owned rows are at a different stride.

        Semantics
            A 4-lane ``shfl.bfly`` completes each of this thread's two rows, then ``mu = Sum/N``,
            ``var = Sum2/N - mu*mu``, ``rstd = rsqrt(var + eps)`` with the reciprocal hoisted and
            ``rsqrt`` in fast-math form -- the same arithmetic as the shared-memory finalize, in a
            different association. Only the quad's lane 0 writes; the other three computed the same
            value redundantly, which is cheaper than a predicated broadcast.

            ``base = (t//128)*64 + ((t%128)//32)*16 + ((t%32)//4)`` and row ``m`` at ``base + m*8``
            are the WGMMA operand-A fragment's own row mapping. **They must agree with
            :meth:`_normalize_tile_rf`'s**, because the whole point is that a thread rescales the
            rows it reduced; a mismatch normalizes every row by another row's statistics, which is
            finite and entirely wrong.

            Ends with a fence and the epilogue barrier so the writes are visible to the normalize.
            There is deliberately NO LEADING barrier, for the reason `finalize_row_stats` records:
            the epilogue's own rendezvous already orders the previous tile's reads before this
            tile's writes, and adding one costs ~4%.

        Args:
            stats_smem: The ``(tile_M, 2)`` fp32 scratch -- column 0 the mean, column 1 the rstd.
            a_tidx: This thread's index within the MMA warpgroups. It selects both the owned rows
                and the owner lane, so a value not reduced modulo the group size makes two
                warpgroups claim the same rows.
            row_sum: This thread's ``Sum x`` partials.
            row_sqsum: Its ``Sum x^2`` partials.
            n_elems: The per-row element count -- the TRUE feature width, not the padded one.
            eps: The variance floor.

        Returns:
            None. Its effect is `stats_smem`, visible to the whole group on return.
        """
        inv_n = 1.0 / Float32(n_elems)
        # The row map is SHARED, not spelled here: it is empirically derived, and a second copy
        # that drifted would mis-attribute statistics silently. See `rs_owned_rows`.
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
                stats_smem[base + m * RS_ROW_STRIDE, 0] = mean
                stats_smem[base + m * RS_ROW_STRIDE, 1] = rstd
        cute.arch.fence_view_async_shared()
        self.epilogue_barrier.arrive_and_wait()

    @cute.jit
    def _normalize_tile_rf(self, tCrA, a_tidx, stats_smem, s_weight, s_bias, g_ktile):
        """Rewrite the register A fragment IN REGISTERS as ``LayerNorm(x)``.

        Purpose
            The register path's pass 2, and where it beats the shared-memory form: there is no
            write-back to shared memory and no second strided read, so the whole normalize costs
            register arithmetic plus the gain lookup.

        Semantics
            ``tCrA[i] = ((tCrA[i] - mu_m) * rstd_m * gain[gk] (+ bias[gk])).to(a_dtype)``, where the
            per-slot ``(m, k_local)`` coordinate is the closed form of the WGMMA operand-A register
            layout::

                base    = (t//128)*64 + ((t%128)//32)*16 + ((t%32)//4)
                slot i: m       = base + ((i % 4) // 2) * 8
                        k_local = (i // 4) * 8 + (t % 4) * 2 + (i % 2)

            The gain is indexed by the GLOBAL column ``g_ktile * blk_k + k_local``, not the local
            one -- which is why `g_ktile` is an argument rather than derived, exactly as in
            `normalize_tile`.

            **Because the fragment is normalized in place, this k-tile's WGMMA must be issued
            before the next tile overwrites it**, which the caller's loop guarantees by issuing
            immediately after.

        Args:
            tCrA: The register A fragment holding this k-tile's raw activation. OVERWRITTEN, in the
                activation's dtype -- the same 16-bit narrowing the shared-memory path performs, so
                the two differ only in how the statistics were associated.
            a_tidx: This thread's index within the MMA warpgroups. Must be the value
                :meth:`_finalize_stats_local_rf` used, or a thread rescales rows it did not reduce.
            stats_smem: The ``(tile_M, 2)`` scratch that finalize published, behind its barrier.
            s_weight: The staged ``(gemm_k,)`` fp32 gain. Must cover ``(g_ktile + 1) * blk_k``
                columns; the padded extent is what `stage_ln_affine` zero-fills.
            s_bias: The staged ``(gemm_k,)`` fp32 bias, or None. None prunes the add entirely.
            g_ktile: The GLOBAL k-tile index this fragment holds, a runtime ``Int32``.

        Returns:
            None; `tCrA` is modified in place.
        """
        blk_k = const_expr(self.cta_tile_shape_mnk[2])
        base = (a_tidx // 128) * 64 + ((a_tidx % 128) // 32) * 16 + ((a_tidx % 32) // 4)
        lane2 = (a_tidx % 4) * 2
        x = tCrA.load().to(Float32)
        for i in cutlass.range_constexpr(cute.size(tCrA)):
            m = base + ((i % 4) // 2) * 8
            k = g_ktile * blk_k + (i // 4) * 8 + lane2 + (i % 2)
            val = (x[i] - stats_smem[m, 0]) * stats_smem[m, 1] * s_weight[k]
            if const_expr(s_bias is not None):
                val = val + s_bias[k]
            tCrA[i] = val.to(self.a_dtype)

    @cute.jit
    def _rs_gemm(self, mma_atom, acc, rA_k, rB_k, zero_init):
        """Issue one k-tile's WGMMA with operand A from REGISTERS and B from shared memory.

        Purpose
            `fold_cp_ops_sm90_utils.gemm_w_idx` indexes both operands by pipeline stage, which an
            already-in-registers A has no index for. This is its register-source counterpart.

        Semantics
            Loops the MMA_K instructions of one k-tile; the atom knows its A source, so nothing
            here selects it. **A fresh ``mma_atom.set`` per call is required, not defensive**: the
            accumulate flag is an atom field, and reusing a set from a previous trace point makes
            the DSL report an "operand does not dominate" SSA error rather than a wrong answer.

        Args:
            mma_atom: The register-source MMA atom.
            acc: The accumulator, updated in place.
            rA_k: The register A fragment for this k-tile, normalized.
            rB_k: B's shared-memory descriptor fragment for this k-tile, already sliced to the
                stage -- unlike A, B is still read from shared memory by the WGMMA.
            zero_init: True on the first k-tile, which overwrites `acc` instead of accumulating.

        Returns:
            None. The WGMMA group is committed; the caller must wait on it before releasing the
            stage `rB_k` points into.
        """
        warpgroup.fence()
        mma_atom.set(warpgroup.Field.ACCUMULATE, not zero_init)
        for k in cutlass.range_constexpr(cute.size(rA_k.shape[2])):
            cute.gemm(mma_atom, acc, rA_k[None, None, k], rB_k[None, None, k], acc)
            mma_atom.set(warpgroup.Field.ACCUMULATE, True)
        warpgroup.commit_group()

    @cute.jit
    def _mma_stats_xgate_rs(
        self,
        ab_pipeline,
        ab_read_state,
        tiled_mma,
        tiled_mma_gate,
        acc,
        acc_gate,
        tCrA,
        tCrB,
        tCrA2,
        tCrB2,
        a_s2r_copy,
        sA,
        stats_smem,
        s_weight,
        s_bias,
        tidx,
        eps,
    ):
        """The two-pass mainloop with the VALUE operand in registers. MN-major activations only.

        Purpose
            Structurally :meth:`_mma_stats_xgate`, with the value tile moved into the WGMMA's
            operand registers once per pass instead of being read strided out of shared memory
            twice and rewritten there. That is what removes the MN-major penalty; see
            :attr:`~_XGateTwoASm90._value_a_in_regs` for the measurement.

        Semantics
            Pass 1 ``ldmatrix``-loads each staged tile into ``tCrA``, reduces over the registers and
            releases the stage; the finalize then publishes ``(mu, rstd)``. Pass 2 re-loads the same
            tiles, normalizes ``tCrA`` in place, issues the value WGMMA with A from registers and
            the gate WGMMA with A from shared memory, drains both groups and releases.

            **The GATE side is untouched.** ``x_gate`` arrives normalized and K-major, so its WGMMA
            keeps the ordinary shared-memory source and contributes no statistics -- which is why
            ``tiled_mma_gate`` is the SMEM-source atom even when the two majors agree.

            **Pass 2 needs no fence or barrier between the normalize and the WGMMA**, unlike the
            shared-memory form: the normalize writes REGISTERS this thread's own WGMMA then reads,
            so there is no cross-thread visibility to establish. That is one CTA-wide barrier per
            k-tile saved on top of the load.

            Both WGMMA groups are still fully drained before the release, for the reason
            :meth:`_mma_stats_xgate` gives: two ``commit_group``s per k-tile break the base's 1-deep
            software pipeline, and B is still read from the stage being freed.

        Args:
            ab_pipeline: The mainloop pipeline.
            ab_read_state: The consumer state, threaded across work tiles and returned.
            tiled_mma: The REGISTER-source value MMA atom.
            tiled_mma_gate: The shared-memory-source gate MMA atom.
            acc: The value accumulator.
            acc_gate: The gate accumulator.
            tCrA: The register A fragment, filled here rather than partitioned from ``sA``.
            tCrB: The value B fragment, still shared-memory-sourced and indexed by stage.
            tCrA2, tCrB2: The gate operand fragments, both shared-memory-sourced.
            a_s2r_copy: The shared-memory -> register A copy. Required here; None would be a
                routing bug, since this body is reached only when it was built.
            sA: The staged VALUE activation. Read; NOT written -- the normalize happens in
                registers, so unlike the shared-memory form this leaves the tile intact.
            stats_smem: This warpgroup's ``(tile_M, 2)`` fp32 scratch.
            s_weight: The staged ``(gemm_k,)`` fp32 gain. Required.
            s_bias: The staged ``(gemm_k,)`` fp32 bias, or None.
            tidx: This thread's index within the MMA warpgroups.
            eps: The LayerNorm variance floor.

        Returns:
            The advanced consumer state, after ``2 * k_tile_cnt`` waits.
        """
        k_tile_cnt = const_expr(self.gemm_k // self.blk_k)
        gate_fn = partial(fold_cp_ops_sm90_utils.gemm_w_idx, tiled_mma_gate, acc_gate, tCrA2, tCrB2)
        mma_atom = cute.make_mma_atom(tiled_mma.op)
        # Two M-rows per thread, {base, base+8} -- the WGMMA operand-A value mode's M sub-mode. The
        # 4-lane quad that splits each row's K is completed in `_finalize_stats_local_rf`.
        row_sum = cute.make_rmem_tensor(cute.make_layout(2), Float32)
        row_sqsum = cute.make_rmem_tensor(cute.make_layout(2), Float32)
        for m in cutlass.range_constexpr(2):
            row_sum[m] = Float32(0.0)
            row_sqsum[m] = Float32(0.0)

        # PASS 1 -- s2r-load, reduce over the registers, release.
        peek = Boolean(True)
        if const_expr(0 < k_tile_cnt):
            peek = ab_pipeline.consumer_try_wait(ab_read_state)
        for k_idx in cutlass.range_constexpr(k_tile_cnt):
            ab_pipeline.consumer_wait(ab_read_state, peek)
            stage = ab_read_state.index
            self._load_a_to_rf(a_s2r_copy, tidx, sA[None, None, stage], tCrA)
            row_sum, row_sqsum = self._accumulate_stats_rf(tCrA, row_sum, row_sqsum)
            ab_pipeline.consumer_release(ab_read_state)
            ab_read_state.advance()
            peek = Boolean(True)
            if const_expr(k_idx + 1 < k_tile_cnt):
                peek = ab_pipeline.consumer_try_wait(ab_read_state)
        self._finalize_stats_local_rf(
            stats_smem, tidx, row_sum, row_sqsum, Int32(self.gemm_k_real), eps
        )

        # PASS 2 -- s2r-load, normalize in registers, then both WGMMAs.
        peek = Boolean(True)
        if const_expr(0 < k_tile_cnt):
            peek = ab_pipeline.consumer_try_wait(ab_read_state)
        for k_idx in cutlass.range_constexpr(k_tile_cnt):
            ab_pipeline.consumer_wait(ab_read_state, peek)
            stage = ab_read_state.index
            self._load_a_to_rf(a_s2r_copy, tidx, sA[None, None, stage], tCrA)
            self._normalize_tile_rf(tCrA, tidx, stats_smem, s_weight, s_bias, Int32(k_idx))
            self._rs_gemm(mma_atom, acc, tCrA, tCrB[None, None, None, stage], Boolean(k_idx == 0))
            gate_fn(A_idx=stage, B_idx=stage, zero_init=Boolean(k_idx == 0))
            warpgroup.wait_group(0)  # drain BOTH groups before this stage is released
            ab_pipeline.consumer_release(ab_read_state)
            ab_read_state.advance()
            peek = Boolean(True)
            if const_expr(k_idx + 1 < k_tile_cnt):
                peek = ab_pipeline.consumer_try_wait(ab_read_state)
        return ab_read_state

    @cute.jit
    def _xgate_combine_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rGate):
        """Bias the value, activate the gate, multiply. One epilogue subtile, in fp32.

        Purpose
            The kernel's arithmetic, and the only place the two accumulators meet::

                value[i] = acc[i] + bp[i]
                gate[i]  = act(acc_gate[i] (+ bg[i]))
                out[i]   = gate[i] * value[i]              (* mask[i])

        Semantics
            **There is no rank-one repair, and its absence is the whole difference from the
            algebraic fold's sibling.** The value WGMMA multiplied a NORMALIZED ``sA`` by a RAW
            ``Wp``, so ``acc`` already IS ``LN(x) @ Wp``; applying the fold's ``r*acc - s*c + d``
            here would repair an identity nothing broke, and the result would be finite and wrong.

            **The mask indexes DIRECTLY.** The dual path indexes ``tDrMask[2*i]`` because its
            post-activation is half its pre-activation's width; this path is full-N, so the
            post-activation shares the accumulator's index space. The mask is a column vector, so
            it is constant along N either way -- which is why the wrong indexing would still
            produce a finite, wrong answer rather than an out-of-range read.

            Every optional term is ``const_expr``-pruned, so a kernel built without a projection
            bias or a mask emits none of them.

        Args:
            params: This kernel's ``EpilogueParams``; ``act_fn`` (the gate activation) is read here.
            epi_loop_tensors: This subtile's loaded terms by op name -- ``mBiasUp``, ``mBiasGate``,
                ``mMaskColVec``.
            tRS_rD: The VALUE accumulator subtile, holding ``LN(x) @ Wp``. Biased IN PLACE.
            tRS_rGate: The GATE accumulator subtile, read but not modified.

        Returns:
            A fresh fp32 fragment holding the combined result, the same size as `tRS_rD`.
        """
        bp = epi_loop_tensors["mBiasUp"]
        if const_expr(bp is not None):
            for i in cutlass.range(cute.size(tRS_rD), unroll_full=True):
                tRS_rD[i] = tRS_rD[i] + bp[i]
        bg = epi_loop_tensors["mBiasGate"]
        tRS_rPostAct = cute.make_rmem_tensor(tRS_rD.layout.shape, self.acc_dtype)
        for i in cutlass.range(cute.size(tRS_rPostAct), unroll_full=True):
            g_i = tRS_rGate[i] + bg[i] if const_expr(bg is not None) else tRS_rGate[i]
            tRS_rPostAct[i] = params.act_fn(g_i) * tRS_rD[i]
        tDrMask = epi_loop_tensors["mMaskColVec"]
        if const_expr(tDrMask is not None):
            for i in cutlass.range(cute.size(tRS_rPostAct), unroll_full=True):
                tRS_rPostAct[i] = tRS_rPostAct[i] * tDrMask[i]
        return tRS_rPostAct


def _functor_for(fusion_variant: str):
    """The functor class that BUILDS this variant.

    Purpose
        One mapping, consulted by both the compile path and the launch path, so the class that
        compiles a variant and the class whose ``EpilogueArguments`` are filled for it cannot
        diverge. They must agree: `alg_fold` declares four extra epilogue ops, so a mismatch is a
        missing or surplus field at construction, not a silently wrong result -- but only because
        the two are derived from here rather than written twice.

    Args:
        fusion_variant: A member of :data:`FUSION_VARIANTS`.

    Returns:
        The class implementing it.

    Raises:
        ValueError: If no class declares the variant in ``_IMPLEMENTED_VARIANTS``. That is a routing
            bug rather than a user error -- the front door validates the name before reaching here.
    """
    for cls in (LayerNormDualGatedGemmSm90, LayerNormDualGatedGemmAlgFoldSm90):
        if fusion_variant in cls._IMPLEMENTED_VARIANTS:
            return cls
    raise ValueError(f"no functor implements fusion_variant={fusion_variant!r}")


def _xgate_functor_for(fusion_variant: str):
    """The TWO-A functor class that builds this variant.

    Purpose
        The `x_gate` counterpart of :func:`_functor_for`, and separate from it for the reason the
        two-A classes are separate: they share a fusion base with the one-A functors but not an
        epilogue, so a single mapping would have to return two classes per variant and every caller
        would have to know which it wanted.

    Semantics
        The two-A classes declare no ``_IMPLEMENTED_VARIANTS`` of their own -- each inherits its
        fusion base's -- so the same membership test that routes the one-A path routes this one,
        and a variant that gains a two-A implementation is picked up by both.

    Args:
        fusion_variant: A member of :data:`FUSION_VARIANTS`.

    Returns:
        The two-A class implementing it.

    Raises:
        ValueError: If no two-A class declares the variant. A routing bug rather than a user
            error -- the front door validates the name first.
    """
    for cls in (
        LayerNormDualGatedGemmXGatePrologLnSm90,
        LayerNormDualGatedGemmXGateAlgFoldSm90,
    ):
        if fusion_variant in cls._IMPLEMENTED_VARIANTS:
            return cls
    raise ValueError(f"no two-A functor implements fusion_variant={fusion_variant!r}")


@jit_cache
def _compile_layernorm_dual_gated_gemm(
    a_dtype,
    b_dtype,
    postact_dtype,
    ln_dtype,
    ln_bias_dtype,
    a_major,
    b_major,
    postact_major,
    tile_shape_mn,
    cluster_shape_mnk,
    persistent,
    is_dynamic_persistent,
    activation,
    gate3_activation,
    rowvec_dtype,
    maskvec_dtype,
    half_bias_dtype,
    gate3_bias_dtype,
    chunk_g,
    two_tensor_B,
    n_b,
    n_out,
    n_rowvec,
    fusion_variant,
    blk_k,
    gemm_k,
    gemm_k_real,
    n_dual_tiles,
    gate3_n3,
    device_capacity,
    pingpong,
):
    """Compile one `LayerNormDualGatedGemmSm90` configuration against fake tensors.

    Every parameter is part of the ``@jit_cache`` key. **M is the only symbolic extent**: one
    artifact serves every token count, and none serves a second feature width.

    **Both feature extents are baked, and that is a perf decision with a measured price tag.** K was
    already a key -- ``gemm_k`` sizes the staged gain, a shared-memory allocation -- and now sizes
    the operand extents too. N joins it because in the workflow this kernel exists for, ``n == k ==
    D``: both are the SAME feature dimension, so a caller sees one artifact per feature width either
    way and baking N multiplies nothing. What it buys is that the epilogue's
    out-of-bounds bounds fold to literals instead of predicating per element -- measured on the
    sibling two-A path at 1.02-1.05x the upstream with N symbolic and 0.98-1.00x with it baked. The
    upstream bakes every feature extent for the same reason (its ``K``/``twoN``/``N3``/``dual_n``
    are all commented ``# STATIC``), so this is parity rather than a new idea.

    The cost lands on a caller who sweeps N at a FIXED K, which the workflow never does.

    The PRESENCE of each optional term is a key component while its value is not: an absent bias is
    compiled out rather than added as zero, so a kernel built without one cannot be handed one.

    Args:
        a_dtype: Cutlass element type of the activation. 16-bit.
        b_dtype: Element type of the weights. Must equal `a_dtype`.
        postact_dtype: Element type of the gated output. 16-bit.
        ln_dtype: Element type of the LayerNorm gain. Float32; required.
        ln_bias_dtype: Element type of the LayerNorm bias, or None. None compiles the add out
            entirely, so a kernel built without a LayerNorm bias cannot be handed one.
        a_major: ``"m"`` or ``"k"``.
        b_major: ``"n"`` or ``"k"``.
        postact_major: ``"m"`` (the transposed store) or ``"n"``.
        tile_shape_mn: ``(tile_M, tile_N)``, where tile_N spans the 2N PRE-activation.
        cluster_shape_mnk: ``(cluster_M, cluster_N, 1)``.
        persistent: Whether the grid is persistent.
        is_dynamic_persistent: Whether tiles are handed out by a GMEM atomic.
        activation: The dual gate's name, a key of ``gate_fn_map``.
        gate3_activation: The output gate's name, a key of ``act_fn_map``, or None.
        rowvec_dtype: Element type of the interleaved ``(2N,)`` bias, or None.
        maskvec_dtype: Element type of the ``(M,)`` post-gate mask, or None.
        half_bias_dtype: Element type of the two ``(N,)`` half-width biases, or None. Mutually
            exclusive with `rowvec_dtype`.
        gate3_bias_dtype: Element type of the ``(N3,)`` output-gate bias, or None.
        chunk_g: The weight layout.
        two_tensor_B: Whether ``Wg`` arrives as a separate tensor. Only valid with ``chunk_g > 1``.
        n_b: The B operand's row count, BAKED. Not derivable from N here: the output gate APPENDS
            W3's rows and pads the result to a work-tile edge, so this is the padded ``B_all``
            width, and re-deriving the padding in this function would duplicate
            `append_gate3_weight`'s arithmetic where it could silently drift. Pass ``B.shape[-2]``.
        n_out: The post-activation's width, BAKED. ``N`` on the dual path (the gate halves the 2N
            pre-activation); equal to `n_b` under the block interleave, where B is ``Wp`` alone.
        n_rowvec: The interleaved ``[bg, bp]`` bias width, BAKED, or 0 when no such bias exists.
            Spans the DUAL pre-activation, which is B's own width in exactly one configuration --
            the element interleave with no output gate -- and differs from it in both others.
        fusion_variant: A member of :data:`FUSION_VARIANTS`.
        blk_k: The mainloop K tile.
        gemm_k: The padded contraction extent.
        gemm_k_real: The true feature width.
        n_dual_tiles: Dual work-tile count, or 0 for no output gate.
        gate3_n3: The output gate's N extent, or 0.
        device_capacity: ``(major, minor)``, re-checked rather than trusted.
        pingpong: Two MMA warpgroups alternating mainloop and epilogue. Only ``alg_fold`` accepts
            it (`LayerNormDualGatedGemmSm90._SUPPORTS_PINGPONG`); the functor asserts, so an invalid
            combination is refused at construction rather than compiled.

    Returns:
        The compiled TVM-FFI entry, callable as
        ``fn(A, B, None, None, epi_args, scheduler_args, [B2])``.

    Raises:
        UnsupportedArchError: If `device_capacity` is not SM90.
        NotImplementedError: For ``fusion_variant="alg_fold"``.
        ValueError: Propagated from the functor for a refused geometry.
    """
    check_arch_supported(device_capacity)
    m, l = cute.sym_int(), cute.sym_int()
    # M is the ONLY symbolic extent. K is the TriMul feature dimension and was already a key
    # (`gemm_k` sizes the staged gain); using it for the operand extents too lets the mainloop's
    # bounds fold rather than predicate, exactly as N's do.
    k, n2 = gemm_k_real, n_b
    div_a, div_b = div_for_dtype(a_dtype), div_for_dtype(b_dtype)
    div_p = div_for_dtype(postact_dtype)
    mA = fake_tensor(a_dtype, (m, k, l), leading_dim=1 if a_major == "k" else 0, divisibility=div_a)
    mB = fake_tensor(
        b_dtype, (n2, k, l), leading_dim=1 if b_major == "k" else 0, divisibility=div_b
    )
    mB2 = (
        fake_tensor(b_dtype, (n2, k, l), leading_dim=1 if b_major == "k" else 0, divisibility=div_b)
        if two_tensor_B
        else None
    )
    # `n_b`, `n_out` and `n_rowvec` arrive as three SEPARATE baked extents rather than one, for
    # the reason that used to need three separate symbols: they coincide in exactly one
    # configuration (the element interleave with no output gate) and differ in both others -- with
    # an output gate B carries the appended W3 rows, so a shared value would assert
    # `2N == 2N + n3_pad`; under the block interleave B is `Wp`, only N wide, so it would assert
    # `2N == N`. Both were measured, not hypothesized. Baking them changes which KIND of thing
    # states the relation, not the relation.
    mPostAct = fake_tensor(
        postact_dtype,
        (m, n_out, l),
        leading_dim=1 if postact_major == "n" else 0,
        divisibility=div_p,
    )
    mPostAct3 = (
        fake_tensor(postact_dtype, (m, gate3_n3, l), leading_dim=1, divisibility=div_p)
        if n_dual_tiles > 0
        else None
    )
    functor_cls = _functor_for(fusion_variant)
    epi_extra = (
        {}
        if fusion_variant != "alg_fold"
        else {
            # The two statistics ops take no host argument: their data is the mainloop's SMEM
            # scratch, supplied by the functor's `epi_get_smem_tensors`. The fields exist because
            # the composition machinery expects one per declared op.
            "sRstd": None,
            "sScale": None,
            # `c` and `d` span the DUAL pre-activation -- the same 2n the interleaved [bg, bp]
            # spans -- so they take `n_rowvec`, which is B's own symbol in the one configuration
            # where those coincide and its own symbol otherwise. With an output gate they do NOT
            # coincide: B carries the appended W3 rows, and the gate reads `mColsum3`/`mDbias3`
            # rather than a region of these, because the gate arm's n-block is re-based.
            #
            # `c` is unconditional -- the fold always emits a column sum -- but `d = b_ln @ B`
            # exists only when a LayerNorm bias does, and the fold returns None otherwise. Keying it
            # on `ln_bias_dtype` (the SAME key `mBias` uses) is what makes the artifact's signature
            # agree with the launch: an unconditional fake compiles a kernel demanding a tensor the
            # front door will pass as None, which the FFI refuses with a type error naming only an
            # epilogue-argument INDEX.
            "mColsum": fake_tensor(Float32, (l, n_rowvec), leading_dim=1, divisibility=4),
            "mDbias": fake_tensor(
                None if ln_bias_dtype is None else Float32,
                (l, n_rowvec),
                leading_dim=1,
                divisibility=4,
            ),
            # The gate's own pair, keyed on the gate existing (`n_dual_tiles`) exactly as
            # `mRowVecBroadcast3` is, and on its OWN symbol: `n3` is unrelated to B's width, and
            # sharing `n2` would assert `n3 == 2n + n3_pad`.
            "mColsum3": fake_tensor(
                Float32 if n_dual_tiles > 0 else None,
                (l, gate3_n3),
                leading_dim=1,
                divisibility=4,
            ),
            "mDbias3": fake_tensor(
                Float32 if (n_dual_tiles > 0 and ln_bias_dtype is not None) else None,
                (l, gate3_n3),
                leading_dim=1,
                divisibility=4,
            ),
        }
    )
    epi_args = functor_cls.EpilogueArguments(
        **epi_extra,
        mPostAct=mPostAct,
        act_fn=gate_fn_map[activation],
        mRowVecBroadcast=fake_tensor(rowvec_dtype, (l, n_rowvec), leading_dim=1, divisibility=4),
        mMaskColVec=fake_tensor(maskvec_dtype, (l, m), leading_dim=1, divisibility=4),
        mBiasUp=fake_tensor(half_bias_dtype, (l, n_out), leading_dim=1, divisibility=4),
        mBiasGate=fake_tensor(half_bias_dtype, (l, n_out), leading_dim=1, divisibility=4),
        mPostAct3=mPostAct3,
        act_fn_3=None if gate3_activation is None else act_fn_map[gate3_activation],
        mRowVecBroadcast3=fake_tensor(
            gate3_bias_dtype, (l, gate3_n3), leading_dim=1, divisibility=4
        ),
        rounding_mode=RoundingMode.RN,
        mWeight=fake_tensor(ln_dtype, (gemm_k_real,), leading_dim=0, divisibility=1),
        mBias=fake_tensor(ln_bias_dtype, (gemm_k_real,), leading_dim=0, divisibility=1),
        eps=Float32(1e-5),
    )
    scheduler_args = make_fake_scheduler_args(
        (is_dynamic_persistent and device_capacity[0] == 9), False, l
    )
    return compile_gemm_kernel(
        partial(
            functor_cls,
            chunk_g=chunk_g,
            fusion_variant=fusion_variant,
            blk_k=blk_k,
            gemm_k=gemm_k,
            gemm_k_real=gemm_k_real,
            n_dual_tiles=n_dual_tiles,
            gate3_n3=gate3_n3,
        ),
        a_dtype,
        tile_shape_mn,
        cluster_shape_mnk,
        pingpong,
        persistent,
        is_dynamic_persistent,
        device_capacity,
        mA,
        mB,
        None,  # no D: the gated output IS the post-activation store
        None,  # no C
        epi_args,
        scheduler_args,
        mB2=mB2,
    )


@jit_cache
def _compile_layernorm_dual_gated_gemm_xgate(
    a_dtype,
    b_dtype,
    postact_dtype,
    a_major,
    a2_major,
    b_major,
    postact_major,
    n,
    tile_shape_mn,
    cluster_shape_mnk,
    persistent,
    is_dynamic_persistent,
    activation,
    rowvec_dtype,
    maskvec_dtype,
    bias_dtype,
    has_dbias,
    blk_k,
    gemm_k,
    gemm_k_real,
    device_capacity,
    fusion_variant="alg_fold",
    ln_dtype=None,
    ln_bias_dtype=None,
):
    """Compile one two-A (`x_gate`) configuration against fake tensors.

    Every parameter is part of the ``@jit_cache`` key. **M is the only symbolic extent**: one
    artifact serves every token count, and none serves a second feature width.

    K is baked for the reason the module docstring records. **N is baked because it costs nothing
    at the shapes this runs and buys the perf parity the port exists for.** In the TriMul workflow
    both projections are ``(D, D)`` against a ``(tokens, D)`` activation, so ``n == k == D`` -- N is
    a FUNCTION of a key that is ALREADY baked, and adding it multiplies the artifact count by
    exactly one. The upstream bakes it too, and that difference was the entire measured perf gap:
    kernel-to-kernel this path ran 1.02-1.05x the upstream with N symbolic and 0.98-1.00x with it
    baked, because a static N folds the epilogue's out-of-bounds bounds to literals where a symbolic
    one predicates per element.

    The cost lands only on a caller who sweeps N at a FIXED K, which the workflow never does.

    **The gate operands ride the ``mB2``/``mB3`` compile slots.** `compile_gemm_kernel` appends
    whatever it is handed after ``stream``, and
    :meth:`LayerNormDualGatedGemmXGateAlgFoldSm90.__call__` takes ``mA2`` in the first of those
    positions -- so the gate A goes in the ``mB2`` slot and the gate B in the ``mB3`` slot. The
    launch passes them in the same order. That is upstream's indirection and it is kept: it is what
    lets a two-A kernel reuse the one-A compile plumbing without changing it.

    **All four N-wide operands take the ONE baked extent.** The value weight, the gate weight, the
    output and the four row vectors are all exactly N here -- unlike the dual path, where the
    interleaved bias and the weight genuinely differ. Using one value asserts an equality that
    actually holds, which is what makes a mis-sized operand a trace-time error.

    Args:
        a_dtype: Cutlass element type of both activations. 16-bit.
        b_dtype: Element type of both weights. Must equal `a_dtype`.
        postact_dtype: Element type of the gated output. 16-bit.
        a_major: ``"m"`` or ``"k"`` for the VALUE activation.
        a2_major: ``"m"`` or ``"k"`` for the GATE activation. May differ from `a_major` -- the
            workflow's back half feeds an MN-major value and a K-major gate -- and it keys the
            artifact because the SM90 WGMMA bakes the A-major into its atom.
        b_major: ``"n"`` or ``"k"``, shared by both weights.
        postact_major: ``"m"`` (the transposed store) or ``"n"``.
        n: The output width, BAKED as a static extent rather than taken as a symbol. See the
            note above for why that is free at the shapes the workflow runs.
        tile_shape_mn: ``(tile_M, tile_N)``, where tile_N spans the FINAL width N -- not 2N. There
            is no 2N pre-activation on this path.
        cluster_shape_mnk: ``(cluster_M, cluster_N, 1)``.
        persistent: Whether the grid is persistent.
        is_dynamic_persistent: Whether tiles are handed out by a GMEM atomic.
        activation: The gate's name, a key of ``act_fn_map``. A ONE-argument activation (sigmoid),
            unlike the dual's two-argument gate.
        rowvec_dtype: Element type of the ``(N,)`` value column sum ``c``. Required on `alg_fold`,
            which always emits one; ignored on `prolog_ln`, which has nothing to repair.
        maskvec_dtype: Element type of the ``(M,)`` mask, or None.
        bias_dtype: Element type of the two ``(N,)`` projection biases, or None. The two are
            compiled in or out TOGETHER: a gate bias without a value bias is not a configuration the
            front door builds, and giving each its own key would double the artifact count to
            express it.
        has_dbias: Whether ``d = b_ln @ Wp`` exists, i.e. whether a LayerNorm bias was given. Keyed
            separately from `bias_dtype` because the two are independent: an LN bias is a property
            of the normalization, a projection bias of the projections. On `prolog_ln` it is
            `ln_bias_dtype` that carries the same fact, because the bias reaches the kernel as a
            ``(K,)`` operand rather than as a folded ``d``.
        blk_k: The mainloop K tile.
        gemm_k: The padded contraction extent.
        gemm_k_real: The true feature width, which divides the row statistics.
        device_capacity: ``(major, minor)``, re-checked rather than trusted.
        fusion_variant: Which two-A class to build -- a member of :data:`FUSION_VARIANTS`. A KEY,
            because the two declare different epilogue ops and therefore different artifacts.
        ln_dtype: Element type of the ``(K,)`` LayerNorm gain. Required on `prolog_ln` (fp32);
            None on `alg_fold`, which absorbed the gain into the weight on the host.
        ln_bias_dtype: Element type of the ``(K,)`` LayerNorm bias, or None. `prolog_ln` only.

    Returns:
        The compiled TVM-FFI entry, callable as
        ``fn(A, B, None, None, epi_args, scheduler_args, A2, B2)``.

    Raises:
        UnsupportedArchError: If `device_capacity` is not SM90.
        AssertionError: Propagated from the functor for a refused geometry.
    """
    check_arch_supported(device_capacity)
    m, l = cute.sym_int(), cute.sym_int()
    k = gemm_k_real  # the feature dimension: a key already, and now the operand extent too
    div_a, div_b = div_for_dtype(a_dtype), div_for_dtype(b_dtype)
    div_p = div_for_dtype(postact_dtype)
    mA = fake_tensor(a_dtype, (m, k, l), leading_dim=1 if a_major == "k" else 0, divisibility=div_a)
    mA2 = fake_tensor(
        a_dtype, (m, k, l), leading_dim=1 if a2_major == "k" else 0, divisibility=div_a
    )
    mB = fake_tensor(b_dtype, (n, k, l), leading_dim=1 if b_major == "k" else 0, divisibility=div_b)
    mB2 = fake_tensor(
        b_dtype, (n, k, l), leading_dim=1 if b_major == "k" else 0, divisibility=div_b
    )
    mPostAct = fake_tensor(
        postact_dtype, (m, n, l), leading_dim=1 if postact_major == "n" else 0, divisibility=div_p
    )
    functor_cls = _xgate_functor_for(fusion_variant)
    # The two variants' epilogues differ by exactly the rank-one repair's four inputs against the
    # physical fusion's ``(K,)`` gain/bias pair, so the split is the whole variant difference on the
    # host side. Deriving BOTH the class and its extra fields from `fusion_variant` is what stops
    # the class that compiles a variant and the arguments filled for it from disagreeing -- a
    # mismatch is a missing or surplus field at construction, not a silently wrong result.
    epi_extra = (
        {
            # The two statistics ops take no host argument: their data is the mainloop's SMEM
            # scratch, supplied by the functor's inherited `epi_get_smem_tensors`.
            "sRstd": None,
            "sScale": None,
            "mColsum": fake_tensor(rowvec_dtype, (l, n), leading_dim=1, divisibility=4),
            "mDbias": fake_tensor(
                Float32 if has_dbias else None, (l, n), leading_dim=1, divisibility=4
            ),
        }
        if fusion_variant == "alg_fold"
        else {
            "mWeight": fake_tensor(ln_dtype, (gemm_k_real,), leading_dim=0, divisibility=1),
            "mBias": fake_tensor(ln_bias_dtype, (gemm_k_real,), leading_dim=0, divisibility=1),
        }
    )
    epi_args = functor_cls.EpilogueArguments(
        **epi_extra,
        mPostAct=mPostAct,
        act_fn=act_fn_map[activation],
        mBiasUp=fake_tensor(bias_dtype, (l, n), leading_dim=1, divisibility=4),
        mBiasGate=fake_tensor(bias_dtype, (l, n), leading_dim=1, divisibility=4),
        mMaskColVec=fake_tensor(maskvec_dtype, (l, m), leading_dim=1, divisibility=4),
        eps=Float32(1e-5),
        rounding_mode=RoundingMode.RN,
    )
    scheduler_args = make_fake_scheduler_args(
        (is_dynamic_persistent and device_capacity[0] == 9), False, l
    )
    return compile_gemm_kernel(
        partial(
            functor_cls,
            chunk_g=1,
            fusion_variant=fusion_variant,
            blk_k=blk_k,
            gemm_k=gemm_k,
            gemm_k_real=gemm_k_real,
        ),
        a_dtype,
        tile_shape_mn,
        cluster_shape_mnk,
        False,  # pingpong: cooperative only
        persistent,
        is_dynamic_persistent,
        device_capacity,
        mA,
        mB,
        None,  # no D: the combined output IS the post-activation store
        None,  # no C
        epi_args,
        scheduler_args,
        mB2=mA2,  # the mB2 compile slot -> the kernel's mA2 positional (gate A = x_gate)
        mB3=mB2,  # the mB3 compile slot -> the kernel's mB2 positional (gate B = Wg)
    )


def _launch_xgate(
    x,
    x_gate,
    norm_weight,
    Wg,
    Wp,
    PostAct,
    tile_M,
    tile_N,
    cluster_M,
    cluster_N,
    *,
    norm_bias,
    bg,
    bp,
    mask_p,
    eps,
    gate3_activation,
    blk_k,
    gemm_k,
    K,
    persistent,
    is_dynamic_persistent,
    tile_count_semaphore,
    max_swizzle_size,
    device_capacity,
    fusion_variant="alg_fold",
):
    """Build the two-A path's operands and launch it. The `x_gate` arm of the front door.

    Purpose
        Extracted from `layernorm_dual_gated_gemm` because the two paths share their VALIDATION and
        almost none of their operand preparation: one interleaves two weights into a 2N tensor, the
        other folds one weight and passes the other raw. Inlining the second arm would put two
        unrelated operand builds either side of one ``if`` in an already long function.

    Semantics
        **On `alg_fold`, only the VALUE weight is folded, and it goes through the GATE's fold
        helper.** The LayerNorm gain belongs to ``LN(x)``, which only the value projection consumes;
        ``Wg`` sees a pre-normalized activation and travels raw. `build_folded_dual_operands` is the
        wrong tool here -- it interleaves a PAIR -- so this uses `build_folded_gate3_operands`,
        which folds a single weight and emits its ``c``/``d``. That is also the helper upstream uses
        for exactly this operand, so the convention (``colsum`` reduces the ROUNDED fold) matches by
        construction rather than by coincidence.

        **On `prolog_ln` there is no host-side fold at all** -- and that is the visible half of the
        variant difference. Both weights travel RAW and the ``(k,)`` gain and bias go to the kernel
        as operands, because the normalize happens in shared memory. So this arm costs ONE FEWER
        DEVICE LAUNCH per call than the fold's, which pre-computes ``Bw``/``c``/``d`` in its own
        kernel; the fold buys that back in the mainloop by sweeping A once instead of twice.

        The projection biases widen to fp32 for the same reason they do on the dual path, and each
        is passed at its FULL ``(n,)`` width -- the two projections have separate accumulators here,
        so neither is half of an interleaved vector. That part is shared, which is why it sits
        outside the branch.

    Args:
        x: ``(m, k)`` value activation, RAW. Validated by the caller.
        x_gate: ``(m, k)`` gate activation, PRE-NORMALIZED and a different tensor from `x`.
        norm_weight: ``(k,)`` fp32 LayerNorm gain. Folded into the value weight here on `alg_fold`;
            passed straight to the kernel on `prolog_ln`, which applies it to the ACTIVATION.
        Wg: ``(n, k)`` gate weight. Passed RAW; only made contiguous, because the TMA descriptor
            needs the k extent contiguous.
        Wp: ``(n, k)`` value weight. Folded on `alg_fold`, raw (and made contiguous) on `prolog_ln`.
        PostAct: ``(m, n)`` output, written in place. Its STRIDES select the transposed store.
        tile_M: CTA tile M.
        tile_N: CTA tile N over the OUTPUT width ``n`` -- not ``2n``. The caller has checked
            ``n % tile_N == 0``.
        cluster_M: Cluster extent along M.
        cluster_N: Cluster extent along N.
        norm_bias: ``(k,)`` fp32 LayerNorm bias, or None. Its presence is a compile key either way:
            on `alg_fold` it decides whether ``d`` exists, on `prolog_ln` whether the normalize's
            add is emitted at all.
        bg: ``(n,)`` gate projection bias, or None.
        bp: ``(n,)`` value projection bias, or None.
        mask_p: The pitch-aligned ``(1, m)`` mask, or None. Already prepared by the caller so both
            arms align it the same way.
        eps: The LayerNorm variance floor.
        gate3_activation: The gate's activation name, a key of ``act_fn_map``. A ONE-argument
            function; the dual's two-argument `activation` has no meaning on this path.
        blk_k: The mainloop K tile.
        gemm_k: The padded contraction extent.
        K: The true feature width.
        persistent: Whether the grid is persistent.
        is_dynamic_persistent: Whether tiles come from a GMEM atomic.
        tile_count_semaphore: The dynamic scheduler's counter, or None.
        max_swizzle_size: Rasterization swizzle width.
        device_capacity: ``(major, minor)``.
        fusion_variant: A member of :data:`FUSION_VARIANTS`. Selects the functor, the operand build
            and the epilogue argument set together -- all three from this one value, so they cannot
            disagree.

    Returns:
        None. The result is in `PostAct`.
    """
    is_fold = fusion_variant == "alg_fold"
    # `alg_fold` folds the gain into the value weight and emits `c`/`d` in the SAME launch;
    # `prolog_ln` does neither, and its value weight only has to be made contiguous for the TMA.
    folded = build_folded_gate3_operands(Wp, norm_weight, norm_bias=norm_bias) if is_fold else None
    B_value = folded.B3 if is_fold else Wp.contiguous()
    colsum_p = _pitch_align_broadcast(folded.colsum.reshape(1, -1)) if is_fold else None
    dbias_p = (
        _pitch_align_broadcast(None if folded.dbias is None else folded.dbias.reshape(1, -1))
        if is_fold
        else None
    )
    bp_p = _pitch_align_broadcast(None if bp is None else bp.float().reshape(1, -1).contiguous())
    bg_p = _pitch_align_broadcast(None if bg is None else bg.float().reshape(1, -1).contiguous())
    check_broadcast_alignment("colsum", colsum_p)
    check_broadcast_alignment("dbias", dbias_p)
    check_broadcast_alignment("bp", bp_p)
    check_broadcast_alignment("bg", bg_p)

    A_p = perm3d_single(_as_batched(x, "x"))
    A2_p = perm3d_single(_as_batched(x_gate, "x_gate"))
    B_p = perm3d_single(_as_batched(B_value, "the folded Wp" if is_fold else "Wp"))
    B2_p = perm3d_single(_as_batched(Wg.contiguous(), "Wg"))
    PostAct_p = perm3d_single(_as_batched(PostAct, "PostAct"))

    compiled_fn = _compile_layernorm_dual_gated_gemm_xgate(
        torch2cute_dtype_map[x.dtype],
        torch2cute_dtype_map[B_value.dtype],
        torch2cute_dtype_map[PostAct.dtype],
        get_major(A_p, "m", "k"),
        get_major(A2_p, "m", "k"),
        get_major(B_p, "n", "k"),
        get_major(PostAct_p, "m", "n"),
        Wg.shape[-2],
        (tile_M, tile_N),
        (cluster_M, cluster_N, 1),
        persistent,
        is_dynamic_persistent,
        gate3_activation,
        None if colsum_p is None else torch2cute_dtype_map[colsum_p.dtype],
        None if mask_p is None else torch2cute_dtype_map[mask_p.dtype],
        None if bp_p is None else torch2cute_dtype_map[bp_p.dtype],
        dbias_p is not None,
        blk_k,
        gemm_k,
        K,
        device_capacity,
        fusion_variant=fusion_variant,
        # `prolog_ln` reads the (K,) pair as OPERANDS, so their dtypes -- and the bias's PRESENCE --
        # key the artifact. On `alg_fold` both are None: the fold consumed them on the host.
        ln_dtype=None if is_fold else torch2cute_dtype_map[norm_weight.dtype],
        ln_bias_dtype=(
            None if (is_fold or norm_bias is None) else torch2cute_dtype_map[norm_bias.dtype]
        ),
    )

    from fold_cp_ops._internal.cache_utils import COMPILE_ONLY

    if COMPILE_ONLY:
        return

    max_active_clusters = get_max_active_clusters(cluster_M * cluster_N) if persistent else 0
    # The rank-one repair's four inputs against the physical fusion's (K,) pair -- the same split
    # the compile made, from the same value, which is what keeps the two in step.
    epi_extra = (
        {"sRstd": None, "sScale": None, "mColsum": colsum_p, "mDbias": dbias_p}
        if is_fold
        else {"mWeight": norm_weight, "mBias": norm_bias}
    )
    epi_args = _xgate_functor_for(fusion_variant).EpilogueArguments(
        **epi_extra,
        mPostAct=PostAct_p,
        act_fn=None,  # Constexpr: baked at compile, passed as None at launch
        mBiasUp=bp_p,
        mBiasGate=bg_p,
        mMaskColVec=mask_p,
        eps=Float32(eps),
        rounding_mode=None,  # Constexpr: baked at compile
    )
    scheduler_args = make_scheduler_args(
        max_active_clusters, max_swizzle_size, tile_count_semaphore
    )
    compiled_fn(A_p, B_p, None, None, epi_args, scheduler_args, A2_p, B2_p)


# ─────────────────────────── the algebraic fold's precompute ───────────────────────────
# One launch that turns (Wg, Wp, w_ln, b_ln, bg, bp) into everything the `alg_fold` fusion's
# mainloop and epilogue read. See `build_folded_dual_operands` for the identity it serves.

#: Output columns one CTA owns. A multiple of 32 so the coalesced store below fills whole sectors.
_FOLD_TILE_P = 64
#: Row-groups cooperating on each column. ``_FOLD_TILE_P * _FOLD_TPC`` is the block size (1024, the
#: SM90 maximum), and TPC is also the width of the K-tile staged in shared memory per iteration.
_FOLD_TPC = 16


@cute.kernel
def _fold_precompute_kernel(
    mWg: cute.Tensor,  # (n, k) gate weight
    mWp: cute.Tensor,  # (n, k) up weight
    mW: cute.Tensor,  # (k,) fp32 LayerNorm gain
    mNormBias: cute.Tensor,  # (k,) fp32 LayerNorm bias, or None
    mB: cute.Tensor,  # (2n, k) folded weight, written
    mColsum: cute.Tensor,  # (2n,) fp32, written
    mDbias: cute.Tensor,  # (2n,) fp32, written iff mNormBias is not None
    mBg: cute.Tensor,  # (n,) gate projection bias, or None
    mBp: cute.Tensor,  # (n,) up projection bias, or None
    mGatedBias: cute.Tensor,  # (2n,) fp32 interleaved [bg, bp], written iff not None
    chunk_g: cutlass.Constexpr[int],
    p_tile_aligned: cutlass.Constexpr[bool],
    colsum_rounded: cutlass.Constexpr[bool],
):
    """Interleave, fold and column-reduce the two projection weights in ONE launch.

    Purpose
        Everything the algebraic fold needs on the B side, produced together because it is all one
        pass over the same weights. Emitting the interleaved projection bias here as well is free --
        the pass has to read `mBg`/`mBp` anyway -- which is why there is no separate bias interleave
        for this path the way there is for the unfused one.

    Semantics
        Computes, for output column ``p`` and contraction row ``kk``::

            fold[kk]   = w_ln[kk] * W_side[i, kk]             # fp32, computed once
            B[p, kk]   = round_to_weight_dtype( fold[kk] )    # rounded ONCE, at the store
            colsum[p]  = sum_kk fold[kk]                      # fp32, over the UNROUNDED fold
            dbias[p]   = sum_kk b_ln[kk] * W_side[i, kk]      # fp32, over the unrounded weight
            gated_bias[p] = the side's projection bias, widened to fp32

        where ``(i, side)`` is the interleave mapping for `chunk_g` (below).

        **`colsum` reduces the fold BEFORE it is rounded, and that is a deliberate inheritance
        rather than the better choice.** Write ``dBw = B - fold`` for the rounding the store
        introduces. The epilogue computes ``r*acc - s*colsum``, where the MMA's ``acc`` is built from
        the ROUNDED ``B``, so the residual against exact arithmetic is::

            colsum over the unrounded fold  ->  r * sum_kk dBw[kk] * x[kk]
            colsum over the rounded B       ->  r * sum_kk dBw[kk] * (x[kk] - mu)

        The second is smaller by roughly ``|mu|/sigma`` on an offset row, and identical on a centred
        one -- so reducing the ROUNDED fold would be strictly more accurate. It is not done here
        because the upstream reduces the unrounded fold, and byte-identity with it is a hard
        requirement of this bring-back. If that requirement is ever lifted, this is a one-line change
        with a measurable accuracy gain and no cost; ``alg_fold_error_bound`` models the convention
        actually in force.

        **The interleave is done on the fly, not materialized.** Column ``p`` reads ``mWg[i, kk]`` or
        ``mWp[i, kk]`` directly, so there is no ``(2n, k)`` staging buffer and no second launch. For
        ``chunk_g == 1`` the mapping is the element interleave (``i = p//2``, even columns gate); for
        ``chunk_g == G`` it is the block interleave (within each ``2G``-wide block the first ``G``
        columns are up and the last ``G`` are gate), matching the chunked epilogue's half-split.

        **The store is staged through shared memory because the natural one is not coalesced.** The
        reduction binds thread to COLUMN, so a warp holds consecutive ``p`` at the same ``kk`` and
        would write addresses striding by ``k`` -- one element per sector. Instead each K-tile is
        written to `sB` and re-read transposed, so a warp writes a contiguous run of ``kk`` for one
        row of `mB`.

        **The reduction order is fixed, not merely deterministic-per-run.** Row-groups accumulate
        ascending ``kk``, then row-group 0 sums the ``TPC`` partials in ascending group order. Two
        runs on two machines with two grid shapes therefore produce identical `colsum` bits, which is
        what lets the fold be compared bitwise rather than within a tolerance.

        **Both extents are relaxed to the 16-byte floor.** The K loop is a ``ceil_div`` whose trip
        count every thread shares -- so the in-loop barriers stay uniform -- with the partial last
        tile handled by per-element predicates: an out-of-range row contributes zero rather than
        reading out of bounds. The column axis is covered by a ``ceil_div`` grid plus a store guard
        selected at COMPILE time by `p_tile_aligned`, so the aligned case emits the plain unguarded
        store and pays nothing for the general one.

    Args:
        mWg: ``(n, k)`` gate weight, k-major. Contiguity in K is assumed, not checked -- a strided
            source is read as if contiguous, giving a wrong fold with no fault.
        mWp: ``(n, k)`` up weight. Must match `mWg` in shape, dtype and layout.
        mW: ``(k,)`` fp32 LayerNorm gain. fp32 is required: the fold multiplies in fp32 and rounds
            once, and a 16-bit gain would round twice, breaking the bitwise comparison against the
            two-step reference.
        mNormBias: ``(k,)`` fp32 LayerNorm bias, or None to compile the `dbias` term out entirely.
            None and a zero tensor are NOT equivalent -- the zero tensor still runs a reduction whose
            fp32 rounding the None path never incurs.
        mB: ``(2n, k)`` destination for the folded weight, in `mWg`'s dtype. This is the GEMM's B
            operand layout, so nothing transposes it afterwards.
        mColsum: ``(2n,)`` fp32 destination.
        mDbias: ``(2n,)`` fp32 destination, or None. Must be None exactly when `mNormBias` is.
        mBg: ``(n,)`` gate projection bias, any dtype convertible to fp32, or None.
        mBp: ``(n,)`` up projection bias, or None. Either may be None independently of the other;
            the missing side's columns are filled with zero, which is the identity for a bias.
        mGatedBias: ``(2n,)`` fp32 destination, or None to compile the bias emission out. Must be
            None exactly when both `mBg` and `mBp` are.
        chunk_g: The interleave block width, a compile-time constant. 1 selects the element
            interleave; any other value ``G`` selects the block interleave and requires ``n % G == 0``,
            unchecked here -- a non-dividing ``G`` maps two columns to one row and silently drops the
            other.
        p_tile_aligned: Whether ``2n`` is a multiple of ``_FOLD_TILE_P``, a compile-time constant.
            Passing True when it is false writes past `mB`'s last row.

    Returns:
        None; `mB`, `mColsum`, `mDbias` and `mGatedBias` are written in place.
    """
    has_norm_bias = const_expr(mNormBias is not None)
    has_gated_bias = const_expr(mGatedBias is not None)
    tidx, _, _ = cute.arch.thread_idx()
    ctile, _, _ = cute.arch.block_idx()
    TILE_P = const_expr(_FOLD_TILE_P)
    TPC = const_expr(_FOLD_TPC)
    K = mWg.shape[1]
    P = mB.shape[0]
    col = tidx % TILE_P
    rg = tidx // TILE_P
    p = ctile * TILE_P + col

    # Output column p -> (projection row i, which weight).
    if const_expr(chunk_g == 1):
        i = p // 2
        is_gate = p % 2 == 0
    else:
        G = const_expr(chunk_g)
        blk = p // (2 * G)
        off = p % (2 * G)
        is_gate = off >= G
        i = blk * G + (off - G if is_gate else off)
    # On a partial last column-tile the threads with p >= 2n have i past the weight's last row.
    # Clamp so their (discarded) read stays in bounds; their stores are all guarded on p < P.
    if const_expr(not p_tile_aligned):
        i = i if p < P else 0

    smem = cutlass.utils.SmemAllocator()
    sB = smem.allocate_tensor(
        Float32, cute.make_ordered_layout((TILE_P, TPC), order=(1, 0)), byte_alignment=16
    )
    sC = smem.allocate_tensor(
        Float32, cute.make_ordered_layout((TPC, TILE_P), order=(1, 0)), byte_alignment=16
    )
    sD = (
        smem.allocate_tensor(
            Float32, cute.make_ordered_layout((TPC, TILE_P), order=(1, 0)), byte_alignment=16
        )
        if const_expr(has_norm_bias)
        else None
    )

    s_p = tidx // TPC  # store phase: which of the TILE_P columns this lane writes back
    s_n = tidx % TPC  # store phase: which k within the TPC-wide tile
    csum = Float32(0.0)
    dsum = Float32(0.0)
    for it in cutlass.range(cute.ceil_div(K, TPC)):
        kk = it * TPC + rg  # ascending across `it` -> a fixed reduction order
        in_k = kk < K
        kk_rd = kk if in_k else 0
        raw = mWg[i, kk_rd].to(Float32) if is_gate else mWp[i, kk_rd].to(Float32)
        scaled = (mW[kk_rd] * raw) if in_k else Float32(0.0)
        # The ONE line the `colsum_rounded` contract turns on. False accumulates the fp32 fold as
        # computed; True accumulates it as it will be STORED, so `colsum` sums exactly the numbers
        # the mainloop's MMA will later read. See the front door for why that is the more accurate
        # convention and why it is nonetheless not the default.
        if const_expr(colsum_rounded):
            scaled = scaled.to(mB.element_type).to(Float32)
        csum = csum + scaled
        if const_expr(has_norm_bias):
            # Zero the out-of-range BIAS rather than the product, so the accumulate stays the plain
            # `dsum + raw*bias` an FMA can fuse. Wrapping the product in a select instead blocks the
            # fusion and changes the bits.
            bias_k = mNormBias[kk_rd] if in_k else Float32(0.0)
            dsum = dsum + raw * bias_k
        sB[col, rg] = scaled
        cute.arch.barrier()
        kk_store = it * TPC + s_n
        p_store = ctile * TILE_P + s_p
        if const_expr(p_tile_aligned):
            if kk_store < K:
                mB[p_store, kk_store] = sB[s_p, s_n].to(mB.element_type)
        else:
            if kk_store < K and p_store < P:
                mB[p_store, kk_store] = sB[s_p, s_n].to(mB.element_type)
        cute.arch.barrier()

    sC[rg, col] = csum
    if const_expr(has_norm_bias):
        sD[rg, col] = dsum
    cute.arch.barrier()

    # Row-group 0 folds the TPC partials in fixed ascending order, and emits the projection bias.
    if rg == 0 and p < P:
        tot_c = Float32(0.0)
        tot_d = Float32(0.0)
        for g in cutlass.range_constexpr(TPC):
            tot_c = tot_c + sC[g, col]
            if const_expr(has_norm_bias):
                tot_d = tot_d + sD[g, col]
        mColsum[p] = tot_c
        if const_expr(has_norm_bias):
            mDbias[p] = tot_d
        if const_expr(has_gated_bias):
            if is_gate:
                mGatedBias[p] = mBg[i].to(Float32) if const_expr(mBg is not None) else Float32(0.0)
            else:
                mGatedBias[p] = mBp[i].to(Float32) if const_expr(mBp is not None) else Float32(0.0)


@cute.jit
def _fold_precompute_jit(
    mWg: cute.Tensor,
    mWp: cute.Tensor,
    mW: cute.Tensor,
    mNormBias: cute.Tensor,
    mB: cute.Tensor,
    mColsum: cute.Tensor,
    mDbias: cute.Tensor,
    mBg: cute.Tensor,
    mBp: cute.Tensor,
    mGatedBias: cute.Tensor,
    chunk_g: cutlass.Constexpr[int],
    p_tile_aligned: cutlass.Constexpr[bool],
    colsum_rounded: cutlass.Constexpr[bool],
    stream,
):
    """Launch `_fold_precompute_kernel` over ``ceil(2n / TILE_P)`` CTAs of ``TILE_P * TPC`` threads.

    Purpose
        The launch half, separated so the kernel body compiles once against symbolic extents.

    Semantics
        One CTA per column tile; the grid is one-dimensional because the K axis is a loop inside the
        kernel, not a grid dimension -- the reduction over K has to stay within one CTA for the
        fixed summation order that makes `colsum` bitwise reproducible.

    Args:
        mWg, mWp, mW, mNormBias, mB, mColsum, mDbias, mBg, mBp, mGatedBias, chunk_g,
            p_tile_aligned: As `_fold_precompute_kernel`.
        stream: The CUDA stream. Under ``--enable-tvm-ffi`` this is the environment stream, so the
            compiled callable takes no stream argument.

    Returns:
        None.
    """
    _fold_precompute_kernel(
        mWg,
        mWp,
        mW,
        mNormBias,
        mB,
        mColsum,
        mDbias,
        mBg,
        mBp,
        mGatedBias,
        chunk_g,
        p_tile_aligned,
        colsum_rounded,
    ).launch(
        grid=[cute.ceil_div(mB.shape[0], _FOLD_TILE_P), 1, 1],
        block=[_FOLD_TILE_P * _FOLD_TPC, 1, 1],
        stream=stream,
    )


@jit_cache
def _compile_fold_precompute(
    dtype, has_norm_bias, bg_dtype, bp_dtype, chunk_g, p_tile_aligned, colsum_rounded
):
    """Compile (and cache) the fold precompute for one compile-time signature.

    Purpose
        Keeps the fold off the per-call critical path: every extent is symbolic, so one compile
        serves every shape and the cache hits from the second call onward.

    Semantics
        `chunk_g` and `p_tile_aligned` are part of the key because they are `Constexpr` parameters --
        each value bakes a different store guard and a different column mapping. They are passed as
        ARGUMENTS rather than published to module globals before tracing, which is how the upstream
        did it; the global form works only while compiles are serial and is silently wrong the
        moment they are not.

    Args:
        dtype: The weights' cutlass dtype; also the folded output's.
        has_norm_bias: Whether the LayerNorm bias, and hence the `dbias` reduction, exists.
        bg_dtype: The gate projection bias's cutlass dtype, or None if absent.
        bp_dtype: The up projection bias's, or None. May differ from `bg_dtype` in presence but the
            caller keeps them the same dtype where both exist.
        chunk_g: The interleave block width. Must be 1 or a multiple of 16; other values compile but
            the caller rejects them.
        p_tile_aligned: Whether ``2n % _FOLD_TILE_P == 0``.

    Returns:
        The compiled callable, taking ``(Wg, Wp, w, norm_bias, B, colsum, dbias, bg, bp,
        gated_bias)`` torch tensors directly.
    """
    n_sym, k_sym, n2_sym = (cute.sym_int() for _ in range(3))
    has_gated_bias = bg_dtype is not None or bp_dtype is not None
    return cute.compile(
        _fold_precompute_jit,
        fake_tensor(dtype, (n_sym, k_sym)),
        fake_tensor(dtype, (n_sym, k_sym)),
        fake_tensor(Float32, (k_sym,)),
        fake_tensor(Float32, (k_sym,)) if has_norm_bias else None,
        fake_tensor(dtype, (n2_sym, k_sym)),
        fake_tensor(Float32, (n2_sym,)),
        fake_tensor(Float32, (n2_sym,)) if has_norm_bias else None,
        fake_tensor(bg_dtype, (n_sym,)) if bg_dtype is not None else None,
        fake_tensor(bp_dtype, (n_sym,)) if bp_dtype is not None else None,
        fake_tensor(Float32, (n2_sym,)) if has_gated_bias else None,
        chunk_g,
        p_tile_aligned,
        colsum_rounded,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


class FoldedDualOperands(NamedTuple):
    """What the algebraic fold reads on the B side, all produced by one launch.

    Attributes:
        B: ``(2N, K)`` folded interleaved weight, ``round(w_ln[k] * W[i, k])``, in the weights'
            dtype. This is the GEMM's B operand as-is.
        colsum: ``(2N,)`` fp32 column sums of the fold BEFORE it was rounded into `B` -- so not,
            quite, the column sums of `B`. `_fold_precompute_kernel` explains why that convention is
            inherited rather than chosen, and what it costs.
        dbias: ``(2N,)`` fp32 ``b_ln @ W^T``, or None when there is no LayerNorm bias. A separate
            additive term that never meets the fold at all.
        gated_bias: ``(2N,)`` fp32 interleaved projection bias, or None when neither `bg` nor `bp`
            was given. A side that was absent contributes zeros.
    """

    B: Tensor
    colsum: Tensor
    dbias: Optional[Tensor]
    gated_bias: Optional[Tensor]


def _pad_gate3_vec_to_tile(t, tile_N: int):
    """Extend a gate row vector's STORAGE to a whole work tile, keeping its extent unchanged.

    Purpose
        Closes a memory-safety fault, and the fault is not this package's to fix at its source. The
        epilogue's broadcast load in ``_internal/epi_ops.py`` reads the vector UNCONDITIONALLY and
        predicates afterwards::

            val = tDgV[i]                                   # the load always happens
            tDrV[i] = val if tDcV[i][src_dim] < limit else zero

        So on a work tile whose trailing part is out of range the lanes past ``limit`` still ISSUE
        the global read; only the resulting value is discarded. For the dual region that is
        harmless -- `append_gate3_weight` requires ``2n % tile_N == 0``, so there is no partial
        tile. The OUTPUT GATE has no such guarantee: its ``n3`` need not reach a tile boundary, and
        the weight's rows are padded to one while its row vectors are not. The lanes covering
        ``[n3, tile_N)`` then read off the end of a live allocation.

        MEASURED with the PyTorch caching allocator disabled, `gemm_layernorm_gemm` at ``B=1``: ``D=128`` 5634
        errors, ``D=64`` 7650, ``D=32`` 7970, and ``D=256`` **0** -- the one width where
        ``n3 == tile_N`` and no padding exists. A 4-byte fp32 read exactly one element past a
        ``4*n3``-byte allocation, threads from the halfway point of the group upward.

    Semantics
        Allocates ``ceil(n3 / tile_N) * tile_N`` zeros, copies the vector into the leading columns
        and returns a VIEW of extent ``n3``. The extent is deliberately unchanged: ``gate3_n3`` is
        baked into the compiled signature, so widening it would recompile and change the kernel.
        What changes is only that the bytes the epilogue reads past ``n3`` are now inside the
        allocation and are ZERO -- which the predicate discards exactly as before, so the OUTPUT IS
        BIT-IDENTICAL. Same shape as `_pitch_align_broadcast`, which solves the pitch problem the
        same way for the same reason.

        **This is containment at the operand, not a repair of the load.** `Gate3RowVecLoad` is
        deliberately left on the unconditional load -- its vectors do not fill the tile at the
        widths the workflow runs, so the warp-uniform full-tile arm that repaired the parent
        `VecLoad` would be refused here and would cost rather than save. This pad is what makes the
        gate arm safe, and it is load-bearing rather than belt-and-braces.

    Args:
        t: A rank-2 ``(1, n3)`` gate row vector, or None (returned unchanged).
        tile_N: The CTA tile the gate arm is read across. Must be the tile the launch uses, or the
            padding stops short of what the epilogue reads.

    Returns:
        `t` itself when ``n3`` already fills a whole tile, else an ``(1, n3)`` view of a
        tile-padded zero-filled buffer. The caller must keep the result alive for the launch: it
        owns the only reference to that buffer.
    """
    if t is None or t.shape[-1] % tile_N == 0:
        return t
    rows, n3 = t.shape
    padded = t.new_zeros((rows, (n3 + tile_N - 1) // tile_N * tile_N))
    padded[:, :n3] = t
    return padded[:, :n3]


def build_folded_dual_operands(
    Wg: Tensor,
    Wp: Tensor,
    norm_weight: Tensor,
    *,
    norm_bias: Optional[Tensor] = None,
    bg: Optional[Tensor] = None,
    bp: Optional[Tensor] = None,
    chunk_g: int = 1,
    colsum_rounded: bool = False,
) -> FoldedDualOperands:
    """Fold the LayerNorm gain into the interleaved projection weights, in ONE device launch.

    Purpose
        The B-side half of the algebraic fusion. Where the prologue variant normalizes the
        ACTIVATION and multiplies, this rewrites the WEIGHT once so the mainloop can multiply the
        raw activation and repair the difference rank-one in the epilogue::

            LN(x) @ B  =  r*(x @ Bw) - s*colsum + dbias,    r = rstd,  s = rstd*mu

        Because the weight is loop-invariant, this runs once per weight rather than once per token,
        which is the entire reason the fusion is worth having.

    Semantics
        One launch produces all four outputs (see `FoldedDualOperands`). The interleave is done
        inside the kernel, so no ``(2N, K)`` weight buffer is materialized and there is no separate
        bias interleave -- the projection bias is a byproduct of a pass that had to read it anyway.

        `B` and `gated_bias` are BITWISE identical to interleaving, folding and interleaving the bias
        in separate steps -- both are pure elementwise work, so there is nothing for a fusion to
        change. `colsum` and `dbias` are REDUCTIONS, so they agree with a separate-step version only
        up to summation order; the kernel fixes its order (ascending, then row-group 0 folding the
        partials ascending) so its own output is reproducible run to run and machine to machine, but
        that order is not torch's. A test comparing against a torch reference must use `torch.equal`
        for the first pair and an fp32 summation-order bound for the second; using `torch.equal`
        throughout fails for a reason that has nothing to do with the kernel.

        Which fold each reduction runs over -- and why `colsum`'s convention is inherited rather than
        chosen -- is in `_fold_precompute_kernel`.

    Args:
        Wg: ``(N, K)`` gate weight. Must be CUDA, contiguous, 16-bit, and share `Wp`'s dtype -- the
            two halves land in one tensor, so one dtype is the only representable choice.
        Wp: ``(N, K)`` up weight. Must have `Wg`'s exact shape.
        norm_weight: ``(K,)`` fp32 LayerNorm gain. fp32 is required rather than converted: the fold
            rounds exactly once, and a narrower gain would round twice.
        norm_bias: Optional ``(K,)`` fp32 LayerNorm bias. Its absence is compiled in and yields
            ``dbias=None``; passing zeros instead costs a reduction and is not bitwise equivalent.
        bg: Optional ``(N,)`` gate projection bias, any float dtype -- it is widened to fp32 here.
        bp: Optional ``(N,)`` up projection bias. Unlike the unfused path, either may be given
            alone; the other side's columns become zero.
        chunk_g: 1 for the element interleave, or a multiple of 16 dividing ``N`` for the block
            interleave. Any other value raises rather than silently mapping two columns to one row.
        colsum_rounded: Which fold `colsum` reduces. **False (default) reproduces the upstream's DUAL
            path and is what byte-identity requires; True reproduces its GATE3 path and is the more
            accurate convention.** The upstream uses both -- unrounded in its two CUDA precompute
            kernels, rounded in its gate3 torch helper, whose comment calls it "cast-matched" -- so a
            port that picks one convention for everything cannot match it. Does not affect `B`,
            `dbias` or `gated_bias`; only `colsum`.

            **Why True is more accurate.** The MMA reads the STORED fold, so with ``c = Σ Bw`` the
            epilogue computes ``Σ_k (x[k] − mu)·r·Bw[k]`` exactly -- the identity, evaluated for the
            weight actually in memory. The whole residual is then ``Bw ≠ diag(w_ln)·B``, an ordinary
            weight perturbation. With ``c = Σ f`` the correction is computed for one weight and
            applied to an accumulator built from another, leaving ``r·Σ dBw·x`` instead of
            ``r·Σ dBw·(x − mu)`` -- larger by ~``|mu|/sigma`` and corresponding to no weight at all.
            So the "correct" colsum gives the worse answer: the MMA's error is already committed, and
            matching it cancels rather than compounds.

            **When it matters, as a threshold rather than a verdict.** The gap is governed by the
            fold input's per-row ``|mu|/sigma`` and by NOTHING else -- bf16 and fp16 give the same
            ratio curve (their absolute errors differ 8x, matching the mantissa gap), and so do
            ``K=128,N=256`` and ``K=512,N=512``. Measured, as unrounded/rounded median error:

            ==========  ======  ======  ======  ======  ======  ======  ======
            |mu|/sigma  0.06    0.21    0.67    2.05    6.76    20.7    205
            ratio       1.00x   1.05x   1.32x   2.29x   6.35x   18.9x   122x
            ==========  ======  ======  ======  ======  ======  ======  ======

            The asymmetry is the point: **the rounded convention is FLAT** -- 1.021e-3 at 0.06 and
            1.022e-3 at 20.7, a 350x change in offset moving its error by 0.1% -- while the unrounded
            one grows linearly. Above ``|mu|/sigma`` ~70 both degrade, from the fp32
            ``E[x^2]-mu^2`` cancellation rather than from the convention, and both go NaN by ~665.

            So the threshold transfers to any model without re-measuring: below ~0.1 the two are
            indistinguishable (sign varies, it is noise); at ~0.2 the gap is 5%; from ~2 it is 2x and
            unbounded above. **a production protein-structure model sits at 0.035 median / 0.10 max** over 2.6M real pair-rep rows,
            i.e. squarely in the indistinguishable band -- but that is one model, and this kernel does
            not know which it will be pointed at.

            **Cost, measured:** 1.1% of THIS kernel at device-bound shapes (4096², 8192²) and inside
            noise at the workflow's (they sit on the ~30 us host-dispatch floor). Against a front
            projection where this kernel is 0.05-1.3% of the GEMM, ~0.01% end to end. The extra
            ``.to(dtype).to(f32)`` is the entire difference.

            Flat where the alternative degrades, ~0.01% to buy, and no need to know the deployment's
            activation statistics: that makes True the right DEFAULT for a library, and the default
            is False here only because byte-identity with the upstream is currently a hard
            requirement. Flipping it is scheduled, not hypothetical -- see the plan's step 9.

    Returns:
        A `FoldedDualOperands`. `B` is freshly allocated ``(2N, K)``; the caller owns it.

    Raises:
        ValueError: If any tensor breaks its dtype, device, shape or alignment contract, naming the
            argument; or if `chunk_g` is neither 1 nor a multiple of 16 dividing ``N``.
    """
    check_tensor("Wg", Wg, expect_width=2, expect_shape=(None, None), align_elems=8)
    check_tensor("Wp", Wp, expect_dtype=Wg.dtype, expect_shape=tuple(Wg.shape))
    N, K = Wg.shape
    check_tensor("norm_weight", norm_weight, expect_dtype=torch.float32, expect_shape=(K,))
    check_tensor("norm_bias", norm_bias, expect_dtype=torch.float32, expect_shape=(K,))
    check_tensor("bg", bg, expect_shape=(N,))
    check_tensor("bp", bp, expect_shape=(N,))
    if chunk_g != 1 and (chunk_g % 16 != 0 or N % chunk_g != 0):
        raise ValueError(
            f"chunk_g must be 1 or a multiple of 16 dividing N (={N}), got {chunk_g}; any other "
            "value maps two output columns to one weight row and drops the other silently"
        )
    two_n = 2 * N
    B = torch.empty((two_n, K), device=Wg.device, dtype=Wg.dtype)
    colsum = torch.empty((two_n,), device=Wg.device, dtype=torch.float32)
    dbias = (
        torch.empty((two_n,), device=Wg.device, dtype=torch.float32)
        if norm_bias is not None
        else None
    )
    gated_bias = (
        torch.empty((two_n,), device=Wg.device, dtype=torch.float32)
        if (bg is not None or bp is not None)
        else None
    )
    _compile_fold_precompute(
        torch2cute_dtype_map[Wg.dtype],
        norm_bias is not None,
        torch2cute_dtype_map[bg.dtype] if bg is not None else None,
        torch2cute_dtype_map[bp.dtype] if bp is not None else None,
        chunk_g,
        two_n % _FOLD_TILE_P == 0,
        colsum_rounded,
    )(Wg, Wp, norm_weight, norm_bias, B, colsum, dbias, bg, bp, gated_bias)
    return FoldedDualOperands(B, colsum, dbias, gated_bias)


def build_folded_dual_operands_ref(
    Wg: Tensor,
    Wp: Tensor,
    norm_weight: Tensor,
    *,
    norm_bias: Optional[Tensor] = None,
    bg: Optional[Tensor] = None,
    bp: Optional[Tensor] = None,
    chunk_g: int = 1,
    colsum_rounded: bool = False,
) -> FoldedDualOperands:
    """Torch reference for `build_folded_dual_operands`, in separate steps.

    Purpose
        The thing the fused kernel must match BITWISE. Written as the obvious sequence -- interleave,
        then fold, then reduce -- so that a disagreement points at the kernel and not at a second
        clever implementation.

    Semantics
        Reproduces the two reduction choices exactly: `colsum` sums the fold AFTER rounding it to the
        weight dtype, `dbias` sums over the unrounded weight in fp32. Getting either backwards
        produces a result that is close but not equal, which is the failure this reference exists to
        catch.

        The reductions use ``.sum(dim=1)``, whose order torch does not specify. That is sufficient
        here only because the summands are few and the comparison has been observed to hold; if it
        ever becomes flaky, the fix is a fixed-order reference, not a tolerance.

    Args:
        Wg, Wp, norm_weight, norm_bias, bg, bp, chunk_g: As `build_folded_dual_operands`. Validated
            only as far as the arithmetic needs; this is a test helper, and the front door is where
            a caller's mistake is diagnosed.

    Returns:
        A `FoldedDualOperands` with the same shapes and dtypes the fused path produces.
    """
    N, K = Wg.shape
    if chunk_g == 1:
        order = torch.stack([torch.arange(N), torch.arange(N)], dim=-1).reshape(2 * N)  # p -> row i
        is_gate = torch.arange(2 * N) % 2 == 0
    else:
        G = chunk_g
        p = torch.arange(2 * N)
        blk, off = p // (2 * G), p % (2 * G)
        is_gate = off >= G
        order = blk * G + torch.where(is_gate, off - G, off)
    src = torch.where(
        is_gate.to(Wg.device).unsqueeze(1), Wg[order.to(Wg.device)], Wp[order.to(Wg.device)]
    )
    exact_fold = norm_weight.unsqueeze(0).float() * src.float()
    folded = exact_fold.to(Wg.dtype)
    colsum = (folded.float() if colsum_rounded else exact_fold).sum(dim=1)
    dbias = (
        (norm_bias.unsqueeze(0).float() * src.float()).sum(dim=1) if norm_bias is not None else None
    )
    gated_bias = None
    if bg is not None or bp is not None:
        zeros = torch.zeros(N, device=Wg.device, dtype=torch.float32)
        gside = zeros if bg is None else bg.float()
        pside = zeros if bp is None else bp.float()
        gated_bias = torch.where(
            is_gate.to(Wg.device), gside[order.to(Wg.device)], pside[order.to(Wg.device)]
        )
    return FoldedDualOperands(folded, colsum, dbias, gated_bias)


class FoldedGate3Operands(NamedTuple):
    """The algebraic fold of the TriMul output-gate projection, the dual fold's native sibling.

    Attributes:
        B3: ``(n3, k)`` folded gate weight, ``round(w_ln[k] * W3[j, k])``, in `W3`'s dtype. NATIVE,
            not interleaved -- there is one gate weight, so there is nothing to pair it with.
        colsum: ``(n3,)`` fp32 column sums. Reduces the ROUNDED fold, which is the OPPOSITE
            convention from `FoldedDualOperands.colsum` -- see `build_folded_gate3_operands` for
            why that asymmetry is reproduced rather than resolved.
        dbias: ``(n3,)`` fp32 ``b_ln @ W3^T``, or None when there is no LayerNorm bias. Formed from
            the RAW `W3`, never from the fold, exactly as the dual's is.
    """

    B3: Tensor
    colsum: Tensor
    dbias: Optional[Tensor]


def build_folded_gate3_operands(
    W3: Tensor,
    norm_weight: Tensor,
    *,
    norm_bias: Optional[Tensor] = None,
    colsum_rounded: bool = True,
) -> FoldedGate3Operands:
    """Fold the LayerNorm gain into the output-gate weight, so the gate shares the dual's stats.

    Purpose
        The gate3 half of the algebraic fusion. The output gate consumes the SAME ``LN(x)`` as the
        dual projection, so it can be recovered by the SAME rank-one correction from the SAME
        per-row ``(r, s)`` the mainloop already computed::

            LN(x) @ W3^T  =  r*(x @ B3^T) - s*colsum + dbias,    r = rstd,  s = rstd*mu

        That is the whole reason the fused output gate is worth having: its weight's rows are
        appended to the dual's, so a gate tile is an ORDINARY work tile of one GEMM rather than a
        second launch reading ``x`` a second time.

    Semantics
        **Torch, not a kernel, and deliberately so.** The dual's fold is a kernel because it must
        interleave two weights and would otherwise cost a separate launch; there is nothing to
        interleave here. Keeping it in torch also keeps it bit-comparable with the upstream, which
        implements this same function the same way -- fusing it into `_fold_precompute_kernel`
        would change the reduction order and break that comparison before it has been made.

        **`colsum_rounded` DEFAULTS TO True here and to False for the dual, and the asymmetry is
        inherited, not chosen.** The upstream reduces the rounded fold for the gate and the
        unrounded fold for the dual, in the same file. Reproducing both is what byte-identity
        requires; the rounded convention is the better one (flat in ``|mu|/sigma`` where the
        unrounded grows linearly), so the eventual unification goes toward this default, not away
        from it. See `build_folded_dual_operands` for the measurements.

    Args:
        W3: ``(n3, k)`` output-gate weight, 16-bit, CUDA, row-major with `k` contiguous and
            16-byte aligned. A non-contiguous tensor is read as if it were contiguous, which
            silently folds the wrong elements.
        norm_weight: ``(k,)`` fp32 LayerNorm gain. fp32 EXACTLY -- a 16-bit gain would be rounded
            twice, once here and once into `B3`.
        norm_bias: ``(k,)`` fp32 LayerNorm bias, or None. Its presence decides whether `dbias`
            exists, and that is a COMPILE key downstream: a kernel built without the term cannot
            be handed one.
        colsum_rounded: Whether `colsum` reduces the fold AFTER rounding it into `B3`'s dtype
            (True, the upstream's gate convention) or before (False, the upstream's dual
            convention). Only affects accuracy, never shape or speed.

    Returns:
        A `FoldedGate3Operands`. `B3` is freshly allocated; the caller owns it.

    Raises:
        ValueError: If any tensor breaks its dtype, device, shape or alignment contract, naming
            the argument.
    """
    check_tensor("W3", W3, expect_width=2, expect_shape=(None, None), align_elems=8)
    k = W3.shape[-1]
    check_tensor("norm_weight", norm_weight, expect_dtype=torch.float32, expect_shape=(k,))
    check_tensor("norm_bias", norm_bias, expect_dtype=torch.float32, expect_shape=(k,))
    W3f = W3.float()
    folded = norm_weight.reshape(1, k) * W3f
    B3 = folded.to(W3.dtype).contiguous()
    colsum = (B3.float() if colsum_rounded else folded).sum(dim=-1).contiguous()
    dbias = None if norm_bias is None else (norm_bias.reshape(1, k) * W3f).sum(dim=-1).contiguous()
    return FoldedGate3Operands(B3, colsum, dbias)


def resolve_k_tiling(k: int, blk_k: Optional[int] = None) -> Tuple[int, int]:
    """Choose the mainloop K tile and the padded contraction extent, the way the upstream does.

    Purpose
        The one place this policy lives. It decides how the fp32 accumulation is GROUPED, so it is
        an arithmetic decision wearing a tiling costume: two tilings of the same contraction give
        answers that differ in the last bits, and this project's contract is byte-identity with its
        upstream.

    Semantics
        **Pad to a multiple of 64 FIRST, then ask for the tile.** The upstream always runs
        ``BLK_K = 64`` and pads ``ceil(k/64)*64``, letting the TMA zero-fill the partial last
        k-tile; the padded columns contribute 0 to the contraction AND to the row statistics, and
        the normalizer divides by the TRUE ``k``, so the padding is exact rather than merely
        out of range.

        The order is the whole content of this function. `auto_blk_k` REFUSES a ``k`` that no
        allowed tile divides, so asking it about the raw ``k`` forces the caller to catch that
        refusal -- and a fallback to the narrowest tile silently re-tiles the mainloop. At ``k=136``
        that was nine 16-wide k-tiles against the upstream's three 64-wide ones, and it cost one
        element in 65536 by one bf16 ulp: a byte-identity failure, and slower besides.

    Args:
        k: The TRUE contraction extent. Must be positive and meet the 16-byte alignment floor
            (``k % 8 == 0`` for a 16-bit operand); this does not re-check it, the front door does.
        blk_k: A caller-pinned tile, or None to apply the policy. When pinned it is used AS GIVEN
            and the extent is padded to a multiple of IT -- an explicit tile is a deliberate choice
            about accumulation grouping, so it is honoured rather than overridden. It must be a
            member of `ALLOWED_BLK_K`; the functor checks.

    Returns:
        ``(blk_k, gemm_k)``: the tile, and the PADDED extent it divides exactly.
    """
    if blk_k is None:
        gemm_k = (k + 63) // 64 * 64
        return auto_blk_k(gemm_k), gemm_k
    return blk_k, (k + blk_k - 1) // blk_k * blk_k


# ───────────────────────────────── the front door ─────────────────────────────────


def layernorm_dual_gated_gemm(
    x: Tensor,  # (m, k)
    norm_weight: Tensor,  # (k,) fp32
    Wg: Tensor,  # (n, k)
    Wp: Tensor,  # (n, k)
    PostAct: Tensor,  # (m, n), or (n, m).T for the transposed store
    tile_M: int,
    tile_N: int,
    cluster_M: int = 1,
    cluster_N: int = 1,
    *,
    norm_bias: Optional[Tensor] = None,  # (k,) fp32
    bg: Optional[Tensor] = None,  # (n,)
    bp: Optional[Tensor] = None,  # (n,)
    mask: Optional[Tensor] = None,  # (m,)
    eps: float = 1e-5,
    W3: Optional[Tensor] = None,  # (n3, k)
    b3: Optional[Tensor] = None,  # (n3,)
    PostAct3: Optional[Tensor] = None,  # (m, n3)
    x_gate: Optional[Tensor] = None,  # (m, k), pre-normalized
    fusion_variant: str = "prolog_ln",
    chunk_g: int = 1,
    blk_k: Optional[int] = None,
    activation: str = "glu",
    gate3_activation: str = "sigmoid",
    persistent: bool = True,
    is_dynamic_persistent: bool = False,
    tile_count_semaphore: Optional[Tensor] = None,  # (1,)
    max_swizzle_size: int = 8,
    pingpong: bool = False,
) -> None:
    """Fused LayerNorm + dual-gated GEMM, written into `PostAct` (and `PostAct3` when gated).

        out[m, n] = gate_fn(LN(x) @ Wg^T + bg, LN(x) @ Wp^T + bp)[m, n] * mask[m]
        out3[m, n] = gate3_fn(LN(x) @ W3^T + b3)[m, n]                   # only when W3 is given

    Note the transposes: every weight is stored ``(n, k)``, so the contraction is over the LAST axis
    of each operand. That is the layout WGMMA wants; pass the weights in that shape rather than
    transposing a ``(k, n)`` tensor, whose non-unit stride the TMA descriptor cannot use.

    Args:
        x: ``(m, k)`` activation, **not** normalized -- this entry fuses the LayerNorm. Must be on a
            CUDA SM90 device, 16-bit (fp16/bf16), contiguous, and its ``k`` extent must satisfy the
            package's 16-byte alignment floor (``k % 8 == 0`` at 16 bits). Not modified.
        norm_weight: ``(k,)`` fp32 LayerNorm gain. **fp32 is required, not converted.**
        Wg: ``(n, k)`` gate-projection weight, same dtype and device as `x`.
        Wp: ``(n, k)`` up-projection weight, same shape, dtype and device as `Wg`.
        PostAct: The output, **written in place**. Its LOGICAL shape is always ``(m, n)``; the
            transposed store is selected by its STRIDES, so an m-major output is
            ``torch.empty(n, m).T``. Passing a raw ``(n, m)`` tensor is a shape mismatch reported
            against a symbol name, not a transposed store.
        tile_M: CTA tile M. One of the geometries ``GemmSm90.__init__`` accepts.
        pingpong: Two MMA warpgroups alternating mainloop and epilogue, so one warpgroup's
            epilogue overlaps the other's MMA. **``fusion_variant="alg_fold"`` only** -- requesting
            it on `prolog_ln` is an `AssertionError` at construction rather than a hang. Requires
            ``persistent=True`` and ``tile_M`` in {64, 128, 192}, both enforced by the base, and
            caps ``tile_N`` well below the cooperative limit. Measured worth ~4.5% at ``m >= 65536``
            and up to 10%; it LOSES at small ``m``, so it is a knob for a tuner, not a better
            default.
        tile_N: CTA tile N over the **2N pre-activation**, so the output tile is ``tile_N // 2``.
            Must be a multiple of 16, and of ``2*chunk_g`` when ``chunk_g > 1``.
        cluster_M: Cluster extent along M. Power of two; ``cluster_M * cluster_N <= 8``.
        cluster_N: Cluster extent along N. Must be 1 when ``chunk_g > 1``.
        norm_bias: Optional ``(k,)`` fp32 LayerNorm bias. Its ABSENCE is compiled in.
        bg: Optional ``(n,)`` gate bias, added to the pre-activation before the gate.
        bp: Optional ``(n,)`` up bias. Must be present exactly when `bg` is.
        mask: Optional ``(m,)`` per-row multiplier applied to the fp32 output AFTER the gate. It
            cannot be folded into `bg`/`bp`: the gate is nonlinear.
        eps: The LayerNorm variance floor. A runtime value; it does not key the compiled artifact.
        W3: Optional ``(n3, k)`` output-gate weight. Its rows are APPENDED to the dual weight, so
            the gate costs no second mainloop. Requires ``chunk_g == 1`` and ``2*n % tile_N == 0``
            -- the region boundary must fall on a work-tile edge.
        b3: Optional ``(n3,)`` output-gate bias, added before the gate's activation.
        PostAct3: The output gate's result, ``(m, n3)`` row-major, **written in place**. Required
            exactly when `W3` is given.
        x_gate: Optional ``(m, k)`` SEPARATE gate activation, **already normalized**. When given,
            the gate projection consumes it RAW -- no LayerNorm, no rank-one correction, no
            contribution to the row statistics -- while the value projection keeps ``LN(x)``::

                out[m, n] = gate3_fn(x_gate @ Wg^T + bg)[m, n] * (LN(x) @ Wp^T + bp)[m, n]

            This is the TriMul back half's output gate, where ``x_gate`` is the same ``LN(x)`` the
            caller already computed for another consumer. Requirements, each refused explicitly:
            ``chunk_g == 1``, ``pingpong=False``, no `W3`, and ``x_gate`` a DIFFERENT tensor from
            `x` with the same shape and dtype.

            **Both `FUSION_VARIANTS` have a two-A form**, so `fusion_variant` selects here exactly
            as it does without `x_gate`: `alg_fold` multiplies the raw ``x`` by a pre-folded ``Wp``
            and repairs the value arm rank-one in the epilogue, `prolog_ln` rewrites the staged
            ``x`` as ``LN(x)`` and multiplies a raw ``Wp``. Only ``sA`` is rewritten -- ``sA2``
            stays raw, which is what makes the physical fusion expressible here at all and is why
            an earlier front door refused the pair.

            **``tile_N`` means something different on this path**: it tiles the output width ``n``
            directly, not ``2n``, because there is no 2N pre-activation to halve. It adds NO
            divisibility constraint -- a partial last N tile is predicated, measured to write no
            padding, and even ``tile_N > n`` is correct (merely wasteful) -- but a caller moving a
            working dual call across without halving `tile_N` gets half-empty tiles rather than an
            error.

            The gate's activation is `gate3_activation` (a one-argument function, default sigmoid),
            NOT `activation` -- the dual's `activation` takes ``(gate, up)`` and there is no such
            pair here.
        fusion_variant: ``"prolog_ln"``. See :data:`FUSION_VARIANTS`.
        chunk_g: Weight layout. 1 interleaves the two weights on the host every call; a multiple of
            16 loads them directly and is preferred wherever ``n`` and ``k`` allow it.
        blk_k: The mainloop K tile, one of :data:`ALLOWED_BLK_K`. **A numerical choice, not only a
            performance one** -- it decides how the k-loop associates the sums, so two values give
            two different bit patterns. None (the default) picks
            :func:`~fold_cp_ops._internal.compile_time.ln_prologue_layout.auto_blk_k`, the widest
            tile that divides ``k``.
        activation: The dual gate's name; a key of ``gate_fn_map``.
        gate3_activation: The output gate's name; a key of ``act_fn_map``. Ignored without `W3`.
        persistent: Launch one resident wave that loops over work tiles.
        is_dynamic_persistent: Hand out tiles through a GMEM atomic instead of statically.
        tile_count_semaphore: A **zeroed** ``int32`` ``(1,)`` tensor. Required when
            `is_dynamic_persistent`; a stale value makes the grid skip tiles, silently truncating
            the output.
        max_swizzle_size: Scheduler rasterization swizzle width. A performance knob only.

    **There is no ``stats_mode`` here. That is a decision, not an omission.** (There IS an
    autotuner now, but only for ``alg_fold`` and only over tile and schedule -- see
    `alg_fold_heuristic_config` and `layernorm_dual_gated_gemm_alg_fold` at the foot of this
    module. The paragraph below is about ``stats_mode``, whose pool this package declines to one
    point.)
    The upstream's counterpart of this kernel (its ``dual_gated_gemm_staged``, which despite the name
    IS the LayerNorm-fused one -- it inherits the staged LN-fusion mixin, and the upstream has no
    non-LN dual-gated kernel at all) exposes ``stats_mode ∈ {"streaming", "cluster"}`` and autotunes
    over it. Its entire candidate pool is those two entries at a fixed ``tile_M=128``, ``tile_N=auto``.

    ``"cluster"`` computes the per-row statistics across a ``K/64``-wide thread-block cluster and
    exchanges the partials through distributed shared memory, instead of each CTA recomputing them.
    This package declines it: every CTA of an N-cluster holds the same activation rows, so
    recomputing the statistics beats exchanging them. Declining it leaves ONE candidate, so there is
    nothing left for an autotuner to choose between -- which is why none is wired.

    **`chunk_g` is not the missing half of that.** It is a DUAL-GATED concern -- how ``Wg`` and
    ``Wp`` are laid out so the epilogue can pair gate with up -- and a LayerNorm in front does not
    change the answer, so its selection belongs one layer down and not here. It is selected there:
    `dual_gated_gemm` carries ``@autotune`` over ``DUAL_GATED_TUNING_SPACE`` (``chunk_g ∈ {1, 16}``)
    with `dual_gated_config_is_valid` as its prune. The upstream agrees by placement -- its own
    ``chunk_g`` heuristic lives on its LN-OFF entry, and its staged LN-fused entry resolves
    ``stats_mode`` only, leaving ``chunk_g`` at 1.

    Should the fused path ever want that selection, INHERIT it: `DUAL_GATED_TUNING_SPACE` and
    `dual_gated_config_is_valid` are module-level pure functions of ``(config, request)`` and are
    meant to be reused. A second copy here is the drift the one-declaration rule exists to stop. The
    one snag is that `dual_gated_config_is_valid` keys on ARGUMENT NAMES and reads ``request["A"]``,
    while this front door calls its activation ``x`` -- so reuse needs an adapter or a rename, not a
    copy-paste.

    None of this is reachable from the TriMul workflow in any case: its front carries gate3 and its
    back carries ``x_gate``, and each requires ``chunk_g == 1`` on its own.

    Two consequences worth knowing before anyone restores it:

    * **Behaviour diverges from the upstream default at K=128, and only there.** Its heuristic picks
      ``cluster`` when ``normalize and not gate3 and not x_gate and K == 128 and K % 64 == 0 and
      N % K == 0``. So an upstream call with default arguments at K=128 takes a path this kernel
      cannot, and the two are NOT bit-identical -- measured, 1 element in 65536 differs by up to
      7.6e-6. Parity claims against the upstream are therefore parity with its STREAMING path.
      Every other K matches its default exactly, and so does K=128 whenever gate3, ``x_gate`` or
      ``_normalize=False`` is in play -- which covers every call the TriMul workflow makes.
    * **On H100 the decline is faster, not slower.** Paired measurement of the upstream against
      itself, ``M = N_token²/16`` for N_token ∈ {2048…12288}, bf16:

      ===========  ==================  =====================
      shape        cluster / streaming verdict
      ===========  ==================  =====================
      K=128 N=128  1.24 - 1.31x        cluster LOSES 24-31%
      K=128 N=256  0.94 - 0.95x        cluster wins 5-6%
      K=256 N=256  1.45 - 1.59x        (heuristic says streaming; it is right)
      ===========  ==================  =====================

      Its gate admits ``N % K == 0``, so ``N == K`` opens it -- but the gate's own comment says the
      read-once geometry is "valid for N=2K", and at ``N == K`` it selects a large regression on this
      arch. The heuristic is bisected on H200 and warns on anything else, so this says nothing about
      its own tuned hardware; it does mean that on H100, not reproducing the upstream default is the
      better of the two behaviours at N=K, and that the 5-6% at N=2K is unreachable from the TriMul
      front anyway because that path carries gate3.

    **Four upstream knobs are absent BY DECISION, and are named here rather than left to be
    discovered missing.** A reader comparing signatures should find each one accounted for:

    * ``fuse_ln=False`` -- computes plain ``x @ B`` with no rank-one correction. A WIRING SMOKE
      test, not a production behaviour: it produces a number the LayerNorm identity does not
      define. Declined. What it was for is covered here by ``fusion_variant`` plus the unfused
      `dual_gated_gemm`, both of which compute something a caller can name.
    * ``stats_sched="post"`` -- where the statistics reduction sits relative to the WGMMA issue.
      Declined as a KNOB and kept as the FIXED placement: the reduction runs after the MMA is
      issued and before the group wait, so its shared-memory reads overlap the flying WGMMA. That
      is the upstream's default and its measured best; exposing the alternatives would offer two
      slower orderings and one correct one.
    * ``stats_tpr=2`` -- lanes cooperating on one row's reduction. Fixed at
      `STATS_THREADS_PER_ROW`, which is 2. It must divide a warp (the reduction is an intra-warp
      butterfly) and divide ``tile_M``; a knob would expose two constraints that only interact
      correctly at one value anyone has measured.
    * a FUSED ``precompute_bw3_cd3`` -- the output gate's fold operands are built with torch here,
      exactly as upstream, whose own docstring calls the fused version a follow-up. Not declined,
      just not done; it is a launch, not a semantic.

    Returns:
        None. The results are in `PostAct` and, when gated, `PostAct3`.

    Raises:
        UnsupportedArchError: If the device is not SM90.
        NotImplementedError: For ``fusion_variant="alg_fold"``, and for ``W3`` with ``chunk_g > 1``
            -- that combination needs a THIRD B operand in the kernel signature, which this
            package's ``GemmSm90.__call__`` does not carry yet.
        ValueError: For any front-door violation -- an unsupported dtype, a shape disagreement, a
            bias supplied on one projection only, a `norm_weight` that is not fp32, an output-gate
            request whose region boundary is not tile-aligned, `is_dynamic_persistent` without a
            semaphore, a `chunk_g` that is neither 1 nor a multiple of 16, an unknown `activation`,
            or a ``k`` that is not 16-byte aligned.
    """
    device_capacity = require_sm90(x.device)
    if activation not in gate_fn_map:
        as_gate_fn(activation)  # raises with the available names
    # Check ORDER is part of the contract: the declared unsupported regions in
    # tests/kernels/test_layernorm_dual_gated_gemm.py are matched first-wins, so a combo violating
    # two rules must report the one listed first there.
    #
    # EVERY tensor argument goes through `check_tensor`. Which of them the kernel feeds to a TMA and
    # which it reads with a broadcast load is plumbing this signature does not expose, so it must not
    # decide whether a caller mistake is named or left to fail as a compile-key `KeyError`.
    m, k = x.shape[-2], x.shape[-1]
    n = Wg.shape[-2]
    align = 16 // x.element_size()
    # DTYPES first, EXTENTS after the configuration checks below -- the order the declared
    # unsupported regions mirror. See the unfused sibling for why the split exists.
    for _nm, _t in (
        ("x", x),
        ("Wg", Wg),
        ("Wp", Wp),
        ("PostAct", PostAct),
        ("W3", W3),
        ("PostAct3", PostAct3),
    ):
        check_tensor(_nm, _t, expect_width=2)
    # The LayerNorm affine pair is fp32 EXACTLY, not merely supported: it is applied in fp32 inside
    # the normalize, so a 16-bit gain would lose more precision than the normalize it scales.
    check_tensor("norm_weight", norm_weight, expect_dtype=torch.float32)
    check_tensor("norm_bias", norm_bias, expect_dtype=torch.float32)
    K = k  # the name the rest of this front door uses for the contraction extent
    if pingpong and not _functor_for(fusion_variant)._SUPPORTS_PINGPONG:
        raise ValueError(
            f"pingpong=True is not supported by fusion_variant={fusion_variant!r}. Its per-row "
            f"reduction spans every MMA warpgroup and publishes through the epilogue barrier, "
            f"which ping-pong sizes for ONE -- 256 threads would arrive at a 128-thread barrier "
            f"and the second half would wait forever, so this is a HANG rather than a wrong "
            f"answer. Use fusion_variant='alg_fold', whose reduction feeds an epilogue correction "
            f"and is therefore warpgroup-local."
        )
    if tile_M not in _SUPPORTED_TILE_M or (tile_M == 192 and tile_N > 128):
        raise ValueError(
            f"tile_M must be one of {_SUPPORTED_TILE_M} (and 192 only with tile_N <= 128); got "
            f"tile_M={tile_M}, tile_N={tile_N}. The fused per-row reduction tiles the CTA's M "
            f"extent by its ROW-GROUP count -- the MMA warpgroups' threads over "
            f"{STATS_THREADS_PER_ROW} lanes per row -- and tile_M must be a multiple of it, or the "
            f"last group owns fewer rows than the reduction iterates over. Both excluded cases are "
            f"where the base splits the tile across TWO warpgroups, leaving 128 groups: 320 always, "
            f"and 192 once tile_N passes 128 (at or below it the base uses three warpgroups, giving "
            f"192 groups, and 192 tiles). Neither is a property of the tile -- at 4 lanes per row "
            f"the groups would be 64 and both would tile."
        )
    if chunk_g != 1 and chunk_g % 16 != 0:
        raise ValueError(
            f"chunk_g must be 1 (element interleave) or a multiple of 16 (block interleave); got "
            f"{chunk_g}. 16 is the stmatrix atom N-width."
        )
    if (bg is None) != (bp is None):
        raise ValueError(
            "bg and bp must both be given or both omitted: a bias on one projection only would "
            "bias half the pre-activation, which is not a meaningful operation."
        )
    if is_dynamic_persistent and tile_count_semaphore is None:
        raise ValueError(
            "is_dynamic_persistent=True requires tile_count_semaphore: the SM90 dynamic scheduler "
            "hands out work tiles through an atomic counter in GMEM. Pass a zeroed int32 (1,)."
        )
    if chunk_g > 1 and cluster_N != 1:
        raise ValueError(
            f"chunk_g>1 loads Wg and Wp through a two-tensor TMA that does not multicast B, so it "
            f"requires cluster_N=1; got cluster_N={cluster_N}."
        )
    if (W3 is None) != (PostAct3 is None):
        raise ValueError(
            f"W3 and PostAct3 describe one output and must be given together; got "
            f"W3={'a tensor' if W3 is not None else None}, "
            f"PostAct3={'a tensor' if PostAct3 is not None else None}."
        )
    if x_gate is not None:
        # Each of these is a wrong ANSWER rather than a wrong shape if it slips through, so each is
        # named here rather than left to the kernel's own assertions three frames down.
        if W3 is not None:
            raise ValueError(
                "x_gate and W3 are mutually exclusive: both ask this kernel to compute a SECOND "
                "thing, and the two-A path has no accumulator left for an output gate. Run the "
                "output gate as its own call."
            )
        if chunk_g != 1:
            raise ValueError(
                f"x_gate requires chunk_g=1; got {chunk_g}. The block-interleaved layout exists to "
                f"pair an up column with a gate column inside one 2N accumulator, and the two-A "
                f"path has two N-wide accumulators instead."
            )
        if pingpong:
            raise ValueError(
                "x_gate requires pingpong=False. A pipeline stage holds FOUR operand buffers here, "
                "and two warpgroups alternating over them is not a schedule this was measured at."
            )
        if x_gate.data_ptr() == x.data_ptr():
            raise ValueError(
                "x_gate must be a DIFFERENT tensor from x. Passing the same one asks for a gate "
                "over the RAW activation while the value side normalizes -- expressible, but not "
                "what any caller means, and indistinguishable in the output from a wiring mistake."
            )
        check_tensor(
            "x_gate",
            x_gate,
            expect_width=2,
            expect_dtype=x.dtype,
            expect_shape=(m, k),
            align_elems=align,
        )

    N = Wg.shape[-2]
    n2 = 2 * N
    # The two-tensor B path exists to ELIDE A LAUNCH, not to express the block interleave: with
    # `chunk_g > 1` the unfused kernel would otherwise need a host pass to interleave `Wg` and `Wp`
    # into one operand, so it takes them as two and interleaves while loading.
    #
    # `alg_fold` has no such launch to elide -- its fold kernel must run regardless, and it emits
    # the weight ALREADY in chunked column order as one tensor. So for this fusion `chunk_g` is a
    # pure layout choice and B stays single. Deriving `two_tensor_B` from `chunk_g` alone gave the
    # fold's one 2n-row B to a kernel compiled for two n-row ones, and the postact width was
    # checked against the wrong symbol.
    two_tensor_B = chunk_g > 1 and fusion_variant != "alg_fold"
    n_dual_tiles, gate3_n3 = 0, 0
    if W3 is not None:
        if two_tensor_B:
            raise NotImplementedError(
                "the fused output gate (W3) currently requires chunk_g=1. With chunk_g>1 the dual "
                "weights arrive as two separate TMA tensors, so W3 cannot be appended to them and "
                "needs a THIRD B operand -- a slot GemmSm90.__call__ does not carry yet. Use "
                "chunk_g=1, or run the output gate as a separate dual_gated_gemm call."
            )
        if n2 % tile_N != 0:
            raise ValueError(
                f"the fused output gate appends W3's rows after the dual weight, so the region "
                f"boundary must fall on a work-tile edge: 2*n (={n2}) must be a multiple of tile_N "
                f"(={tile_N}); got remainder {n2 % tile_N}."
            )
        if W3.shape[-1] != K:
            raise ValueError(f"W3 must be (n3, k={K}); got {tuple(W3.shape)}.")
        n_dual_tiles = n2 // tile_N
        gate3_n3 = W3.shape[-2]

    for _nm, _t in (("bg", bg), ("bp", bp), ("b3", b3), ("mask", mask)):
        check_tensor(_nm, _t)
    # EXTENTS, after the configuration checks -- the shallowest refusal, reported last.
    check_tensor("x", x, expect_shape=(m, k), align_elems=align)
    check_tensor("Wg", Wg, expect_shape=(n, k))
    check_tensor("Wp", Wp, expect_shape=(n, k))
    check_tensor("PostAct", PostAct, expect_shape=(m, n))
    check_tensor("W3", W3, expect_shape=(None, k))
    if W3 is not None:
        check_tensor("PostAct3", PostAct3, expect_shape=(m, W3.shape[-2]))
        check_tensor("b3", b3, expect_shape=(W3.shape[-2],))
    check_tensor("norm_weight", norm_weight, expect_shape=(k,))
    check_tensor("norm_bias", norm_bias, expect_shape=(k,))
    check_tensor("bg", bg, expect_shape=(n,))
    check_tensor("bp", bp, expect_shape=(n,))
    check_tensor("mask", mask, expect_shape=(m,))

    blk_k, gemm_k = resolve_k_tiling(K, blk_k)

    mask_p = _pitch_align_broadcast(None if mask is None else mask.reshape(1, -1))
    check_broadcast_alignment("mask", mask_p)

    # The two-A arm forks HERE, after every shared check and the shared mask preparation, because
    # from here down the two paths share nothing: one interleaves a weight PAIR into a 2N operand,
    # the other passes two N-wide weights into a second TMA stream. `fusion_variant` travels with
    # it -- BOTH fusions have a two-A form, and which one runs is that argument, not the arm.
    if x_gate is not None:
        return _launch_xgate(
            x,
            x_gate,
            norm_weight,
            Wg,
            Wp,
            PostAct,
            tile_M,
            tile_N,
            cluster_M,
            cluster_N,
            norm_bias=norm_bias,
            bg=bg,
            bp=bp,
            mask_p=mask_p,
            eps=eps,
            gate3_activation=gate3_activation,
            blk_k=blk_k,
            gemm_k=gemm_k,
            K=K,
            persistent=persistent,
            is_dynamic_persistent=is_dynamic_persistent,
            tile_count_semaphore=tile_count_semaphore,
            max_swizzle_size=max_swizzle_size,
            device_capacity=device_capacity,
            fusion_variant=fusion_variant,
        )

    # Shared with the unfused sibling: the weight interleave, the output gate's append, the fp32
    # widening of the biases and the M*N bias-path crossover are all layout work, not LayerNorm
    # work, and one copy is what keeps the two front doors from drifting.
    if fusion_variant == "alg_fold":
        # The fold absorbs the LayerNorm gain into the weight and emits `c`/`d` alongside it, all in
        # ONE launch -- so this replaces the interleave rather than following it, and the projection
        # bias arrives as a free byproduct instead of a second host op.
        folded = build_folded_dual_operands(
            Wg, Wp, norm_weight, norm_bias=norm_bias, bg=bg, bp=bp, chunk_g=chunk_g
        )
        B, B2, rowvec_bias, half_up, half_gate = folded.B, None, folded.gated_bias, None, None
        colsum, dbias = folded.colsum, folded.dbias
        colsum3 = dbias3 = None
        if W3 is not None:
            # The output gate consumes the SAME LN(x) as the dual, so it is recovered by the SAME
            # rank-one correction from the SAME per-row (r, s). Folding its weight the same way and
            # APPENDING the rows is what makes a gate tile an ordinary work tile of one GEMM -- the
            # alternative is a second launch that reads `x` again.
            #
            # The WEIGHT is appended, indexed globally by the mainloop's TMA; the correction vectors
            # are NOT, because the epilogue re-bases the gate arm's n-block. They are their own
            # zero-indexed operands, exactly like `b3`.
            gate3 = build_folded_gate3_operands(W3, norm_weight, norm_bias=norm_bias)
            B = append_gate3_weight(B, gate3.B3, tile_N)
            colsum3 = _pad_gate3_vec_to_tile(
                _pitch_align_broadcast(gate3.colsum.reshape(1, -1)), tile_N
            )
            dbias3 = _pad_gate3_vec_to_tile(
                _pitch_align_broadcast(None if gate3.dbias is None else gate3.dbias.reshape(1, -1)),
                tile_N,
            )
        colsum_p = _pitch_align_broadcast(colsum.reshape(1, -1))
        dbias_p = _pitch_align_broadcast(None if dbias is None else dbias.reshape(1, -1))
        if rowvec_bias is not None:
            rowvec_bias = _pitch_align_broadcast(rowvec_bias.reshape(1, -1))
    else:
        colsum_p = dbias_p = None
        B, B2, rowvec_bias, half_up, half_gate = build_dual_operands(
            Wg, Wp, bg, bp, chunk_g=chunk_g, tile_N=tile_N, W3=W3
        )

    A_p = perm3d_single(_as_batched(x, "x"))
    B_p = perm3d_single(_as_batched(B, "Wp" if two_tensor_B else "the interleaved weight"))
    PostAct_p = perm3d_single(_as_batched(PostAct, "PostAct"))
    B2_p = perm3d_single(_as_batched(B2, "Wg")) if two_tensor_B else None
    PostAct3_p = perm3d_single(_as_batched(PostAct3, "PostAct3")) if PostAct3 is not None else None
    a_major = get_major(A_p, "m", "k")
    b_major = get_major(B_p, "n", "k")
    postact_major = get_major(PostAct_p, "m", "n")

    # fp32 for the same reason as the projection biases (see `build_dual_operands`), and it is the
    # dtype the kernel this reproduces REQUIRES outright (`assert b3.dtype == torch.float32`).
    # Accepting a 16-bit b3 and passing it through would put `Gate3RowVecLoad` on the same narrow
    # load the half-bias pair was measured 1.23x slower on.
    # Padded to a whole gate tile for the same reason as the fold's correction vectors -- the
    # epilogue's broadcast load reads past `n3` and discards afterwards. See `_pad_gate3_vec_to_tile`.
    b3_p = _pad_gate3_vec_to_tile(
        _pitch_align_broadcast(None if b3 is None else b3.float().reshape(1, -1).contiguous()),
        tile_N,
    )
    check_broadcast_alignment("rowvec_bias", rowvec_bias)
    check_broadcast_alignment("bg", half_gate)
    check_broadcast_alignment("bp", half_up)
    check_broadcast_alignment("b3", b3_p)

    # The LayerNorm vectors go to the kernel at their TRUE length `K`, never zero-extended to the
    # padded `gemm_k`. `stage_ln_affine` reads `mWeight[idx]` only for `idx < k_real` and writes the
    # `[k_real, gemm_k)` tail of its SHARED copy as zero itself, so a padded GMEM vector is not
    # merely unnecessary -- it is longer than the compiled signature's static extent (`gemm_k_real`)
    # and the call is refused. That fires exactly when `K % blk_k != 0`, e.g. K=136.
    norm_weight_p, norm_bias_p = norm_weight, norm_bias

    _rowvec_src = colsum_p if colsum_p is not None else rowvec_bias
    compiled_fn = _compile_layernorm_dual_gated_gemm(
        torch2cute_dtype_map[x.dtype],
        torch2cute_dtype_map[B.dtype],
        torch2cute_dtype_map[PostAct.dtype],
        torch2cute_dtype_map[norm_weight.dtype],
        None if norm_bias is None else torch2cute_dtype_map[norm_bias.dtype],
        a_major,
        b_major,
        postact_major,
        (tile_M, tile_N),
        (cluster_M, cluster_N, 1),
        persistent,
        is_dynamic_persistent,
        activation,
        None if W3 is None else gate3_activation,
        None if rowvec_bias is None else torch2cute_dtype_map[rowvec_bias.dtype],
        None if mask is None else torch2cute_dtype_map[mask.dtype],
        None if half_up is None else torch2cute_dtype_map[half_up.dtype],
        None if b3_p is None else torch2cute_dtype_map[b3_p.dtype],
        chunk_g,
        two_tensor_B,
        # The three N-wide extents, BAKED. Read off the tensors that will actually be passed rather
        # than re-derived from N: the output gate appends W3's rows to B and pads to a work-tile
        # edge, so `B.shape[-2]` is the only spelling that cannot drift from `append_gate3_weight`.
        B.shape[-2],
        PostAct.shape[-1],
        # The DUAL pre-activation's width, BAKED. Taken from whichever tensor actually carries it:
        # `mColsum` is UNCONDITIONAL on the fold path (the fold always emits a column sum) while the
        # interleaved bias is absent whenever the caller passed no projection bias, so deriving this
        # from the bias alone yields 0 there and `(?, 0)` is not a legal layout -- an MLIR parse
        # error naming a shape mode, three frames into tracing. `2 * n` is the fallback rather than
        # 0 for the same reason: the value is unused when neither tensor exists, and a positive one
        # cannot become an invalid extent if a later path does read it.
        (2 * n if _rowvec_src is None else _rowvec_src.shape[-1]),
        fusion_variant,
        blk_k,
        gemm_k,
        K,
        n_dual_tiles,
        gate3_n3,
        device_capacity,
        pingpong,
    )

    from fold_cp_ops._internal.cache_utils import COMPILE_ONLY

    if COMPILE_ONLY:
        return

    max_active_clusters = get_max_active_clusters(cluster_M * cluster_N) if persistent else 0
    epi_args = _functor_for(fusion_variant).EpilogueArguments(
        **(
            {}
            if fusion_variant != "alg_fold"
            else {
                "sRstd": None,
                "sScale": None,
                "mColsum": colsum_p,
                "mDbias": dbias_p,
                "mColsum3": colsum3,
                "mDbias3": dbias3,
            }
        ),
        mPostAct=PostAct_p,
        act_fn=None,  # Constexpr: baked at compile, passed as None at launch
        mRowVecBroadcast=rowvec_bias,
        mMaskColVec=mask_p,
        mBiasUp=half_up,
        mBiasGate=half_gate,
        mPostAct3=PostAct3_p,
        act_fn_3=None,  # Constexpr: baked at compile
        mRowVecBroadcast3=b3_p,
        rounding_mode=None,  # Constexpr: baked at compile
        mWeight=norm_weight_p,
        mBias=norm_bias_p,
        eps=Float32(eps),
    )
    scheduler_args = make_scheduler_args(
        max_active_clusters, max_swizzle_size, tile_count_semaphore
    )
    if two_tensor_B:
        compiled_fn(A_p, B_p, None, None, epi_args, scheduler_args, B2_p)
    else:
        compiled_fn(A_p, B_p, None, None, epi_args, scheduler_args)


def layernorm_dual_gated_gemm_ref(
    x: Tensor,
    norm_weight: Tensor,
    Wg: Tensor,
    Wp: Tensor,
    norm_bias: Optional[Tensor] = None,
    bg: Optional[Tensor] = None,
    bp: Optional[Tensor] = None,
    mask: Optional[Tensor] = None,
    eps: float = 1e-5,
    W3: Optional[Tensor] = None,
    b3: Optional[Tensor] = None,
    x_gate: Optional[Tensor] = None,
    activation: str = "glu",
    transpose_out: bool = False,
):
    """Reference implementation in fp32 torch, for tests and for reading the contract.

    Purpose
        States the operation once, with no tiling, no layout and no fused epilogue, so a kernel
        disagreement localizes to the kernel.

    Semantics
        Everything is computed in fp32 regardless of the inputs' dtype, and returned in fp32 --
        narrowing is the caller's decision, because the kernel's own narrowing is a separate step a
        test may want to model differently. The mask is applied AFTER the gate, in fp32, which is
        where the kernel applies it. The LayerNorm is ``torch.nn.functional.layer_norm``, so the
        reference does NOT model the fusion's one extra rounding (the kernel writes the normalized
        activation back in 16 bits before the MMA); a test comparing them must budget for it.

    Args:
        x: ``(m, k)`` activation, un-normalized.
        norm_weight: ``(k,)`` LayerNorm gain.
        Wg: ``(n, k)`` gate weight.
        Wp: ``(n, k)`` up weight.
        norm_bias: Optional ``(k,)`` LayerNorm bias.
        bg: Optional ``(n,)`` gate bias.
        bp: Optional ``(n,)`` up bias.
        mask: Optional ``(m,)`` post-gate multiplier.
        eps: The LayerNorm variance floor.
        W3: Optional ``(n3, k)`` output-gate weight. When given, a SECOND result is returned.
        b3: Optional ``(n3,)`` output-gate bias.
        activation: Gate name; a key of ``gate_fn_map``. Only ``"glu"`` is modelled here, matching
            what the kernel ships.
        transpose_out: Return the dual result as ``(n, m)`` instead of ``(m, n)``. The output gate's
            result is ALWAYS ``(m, n3)``, matching the kernel.

    Returns:
        The fp32 dual result, or ``(dual, gate3)`` when `W3` is given.

    Raises:
        ValueError: If `activation` is not modelled here.
    """
    if activation != "glu":
        raise ValueError(f"the reference models only 'glu'; got {activation!r}")
    k = x.shape[-1]
    xn = F.layer_norm(
        x.float(),
        (k,),
        norm_weight.float(),
        norm_bias.float() if norm_bias is not None else None,
        eps,
    )
    # `x_gate` is ALREADY normalized, so it enters the gate projection raw -- no LayerNorm, and
    # none of `xn`'s statistics. Only the value projection consumes `LN(x)`.
    gate = (xn if x_gate is None else x_gate.float()) @ Wg.float().T
    up = xn @ Wp.float().T
    if bg is not None:
        gate = gate + bg.float()
    if bp is not None:
        up = up + bp.float()
    out = torch.sigmoid(gate) * up
    if mask is not None:
        out = out * mask.float().reshape(-1, 1)
    out = out.T.contiguous() if transpose_out else out
    if W3 is None:
        return out
    assert x_gate is None, "x_gate and W3 are mutually exclusive; the kernel refuses the pair"
    g3 = xn @ W3.float().T
    if b3 is not None:
        g3 = g3 + b3.float()
    return out, torch.sigmoid(g3)


# ───────────────────────── config selection: heuristic, sweep, freeze ─────────────────────────
# Config selection for the algebraic-fold fusion: a heuristic, a measured sweep, and a freeze.
#
# `layernorm_dual_gated_gemm` takes its tile and its schedule as ARGUMENTS and computes; choosing
# them is a separate concern and lives here. Three ways to choose, in increasing cost and confidence:
#
# * `alg_fold_heuristic_config` -- a pure function of the shapes. No timing, no device work.
# * ``do_autotune=True`` on `layernorm_dual_gated_gemm_alg_fold` -- measures the pool for this shape.
# * `alg_fold_freeze` -- measures ONCE, then returns a callable bound to the winner.
#
# **Only ``alg_fold`` has anything to choose.** The other fusion, ``prolog_ln``, refuses ping-pong at
# the front door and has no second tile worth sweeping, so its space is empty and no entry point here
# mentions it. That asymmetry is why this module is variant-specific rather than a wrapper over the
# front door in general.
#
# **Mapping to the upstream, which spells the same three choices differently.** It carries a
# ``select ∈ {"heuristic", "default", "autotune"}`` string plus a ``_config`` argument on the front
# door itself. This package had already chosen a different idiom for the two kernels that tune
# (`gemm`, `dual_gated_gemm`): a declared `AxisSpace`, an ``@autotune`` decorator, and a boolean gate
# naming the parameter that turns it on. Reproducing ``select`` here would leave the repo with two
# autotuning idioms and a reader with no way to know which one a given kernel uses. So:
# ``select="default"`` is the plain call, ``select="heuristic"`` is the plain call with this module's
# heuristic splatted in, ``select="autotune"`` is ``do_autotune=True``, and ``_config`` is
# ``_config``. The BEHAVIOUR is the upstream's; only the spelling is this package's.
#
# **One coupling deliberately NOT reproduced.** Upstream, the distributed front's config resolver
# calls this heuristic, so a change to a single-device tile silently moves a distributed one. That is
# a footgun with a name, and importing it along with the formula would be importing the bug.


#: The CTA tile M every entry here fixes. Not an axis: the upstream measured 256 at 6-14x slower on
#: this epilogue and 192 as invalid for it, so a pool spanning tile_M would sweep configurations
#: already measured and rejected. It stays a front-door argument for a caller who knows better.
_TILE_M = 128

#: Arch-keyed PERF thresholds -- the only numbers here that are not hardware-correctness gates.
#:
#: ``PINGPONG_MAX_K`` is the K below which the two-warpgroup schedule beats the wide cooperative
#: tile. It was bisected on `TUNED_ARCH` UPSTREAM and is **ported, not validated here**: this
#: package's development hardware is an H100, and inventing an ``H100_SXM5`` entry from a number
#: nobody bisected would be worse than a stated fallback, because it would look measured. A
#: non-tuned arch warns once through `warn_arch_suboptimal_once` and uses this set.
_ALG_FOLD_PERF = {
    "H200_SXM5": {"PINGPONG_MAX_K": 512},
}


def alg_fold_heuristic_config(
    m: int,
    k: int,
    n: int,
    *,
    has_gate3: bool = False,
    has_mask: bool = False,
    transpose_out: bool = False,
    has_xgate: bool = False,
    device: Any = None,
) -> Dict[str, Any]:
    """Pick ``{}`` | ``{"pingpong": True}`` | ``{"tile_N": 256}`` from the shapes alone.

    Purpose
        The no-timing choice, for a caller that cannot afford a sweep. Splatted into
        `layernorm_dual_gated_gemm_alg_fold`, it reproduces the upstream's ``select="heuristic"``.

    Semantics
        An empty dict means the kernel's own defaults, which is why every uncertain path returns
        one: the default config is the always-valid one.

        **Only the PLAIN dual is tile-sensitive.** Every functional variant -- the output gate, a
        mask, a transposed store, the separate gate input -- measured tile-insensitive upstream AND
        forbids ping-pong on its own account, so they take the default. That makes this heuristic
        INERT for the TriMul workflow, whose front carries gate3 and a transposed store and whose
        back carries ``x_gate``: both return ``{}``. The arch-dependent number below is therefore
        unreachable from the workflow and only a direct plain-dual call can meet it.

        Then two branches. Below ``PINGPONG_MAX_K`` the GEMM is small enough that overlapping one
        warpgroup's epilogue with the other's mainloop wins. At or above it the wide tile wins
        instead -- but only if ``2N`` is a multiple of 256, since the kernel tiles the
        pre-activation and will refuse a tile that does not divide it. That second test is a
        VALIDITY gate, not a perf one, so it is not arch-keyed.

    Args:
        m: Token extent. Accepted for signature stability and not currently read -- the upstream's
            formula does not branch on it either, and dropping the parameter would make a future
            M-dependent threshold an API change.
        k: Contraction extent (the model's hidden width). The arch-tuned branch tests this.
        n: Output feature extent, i.e. HALF the pre-activation width. ``2 * n`` is what gets tiled.
        has_gate3: Whether the fused output gate is built.
        has_mask: Whether a per-row mask is applied.
        transpose_out: Whether the store is m-major.
        has_xgate: Whether the gate consumes a separate raw input. **Always False today** -- that
            variant is not built yet. It is a parameter rather than an omission so the predicate is
            complete when it lands; silently defaulting a future flag into the "plain" branch would
            hand `x_gate` a ping-pong config it cannot run.
        device: The device whose arch selects the threshold set. **Pass ``x.device``.** None means
            `TUNED_ARCH` with NO warning, which is right for an offline audit that wants the tuned
            arch's answer and wrong for anything timing a kernel -- a harness that omits it records
            one arch's pick beside another arch's measurement, and the two agree on the tuned arch,
            so the mistake is invisible exactly where it is harmless.

    Returns:
        Keyword arguments for `layernorm_dual_gated_gemm_alg_fold`. Possibly empty.
    """
    arch = heuristic_arch(device) if device is not None else TUNED_ARCH
    if arch != TUNED_ARCH:
        warn_arch_suboptimal_once(arch, "layernorm_dual_gated_gemm")
    perf = _ALG_FOLD_PERF.get(arch, _ALG_FOLD_PERF[TUNED_ARCH])
    if has_gate3 or has_mask or transpose_out or has_xgate:
        return {}
    if k < perf["PINGPONG_MAX_K"]:
        return {"pingpong": True}  # tile_M=128 is a valid ping-pong CTA M
    if (2 * n) % 256 == 0:
        return {"tile_N": 256}
    return {}


def alg_fold_tuning_space() -> AxisSpace:
    """The declared axes the measured sweep spans: the CTA tile N and the schedule.

    Purpose
        Makes "what is tuned" readable off the kernel rather than recoverable by scraping a list of
        config tuples.

    Semantics
        Two axes, named for the front-door parameters they set, so a config's keys ARE the call's
        keyword arguments. ``tile_N`` tiles the 2N-wide PRE-activation, not the output; ``pingpong``
        selects the two-warpgroup schedule. The product is six points, of which
        `alg_fold_config_is_valid` typically admits a handful.

        ``tile_M`` is not here (see `_TILE_M`), and neither is ``blk_k``: it decides how the k-loop
        associates its sums, so two values give two different bit patterns, and tuning it would make
        a numerical result depend on a timing measurement.

    Returns:
        A fresh `AxisSpace`, so a caller may subset or inspect it without touching the decorated
        pool.
    """
    return AxisSpace(
        TuneAxis(
            "tile_N",
            domain="the CTA tile over the 2N-wide pre-activation; must divide 2N and meet the "
            "SM90 gated STSM-quad floor (a multiple of 16)",
            values=(64, 128, 256),
        ),
        TuneAxis(
            "pingpong",
            domain="whether two MMA warpgroups alternate mainloop and epilogue; requires a "
            "tile_M of 64/128/192 and caps tile_N",
            values=(False, True),
        ),
    )


#: The pool the tuned path sweeps: `alg_fold_tuning_space`'s product, pruned per request.
ALG_FOLD_TUNING_SPACE = alg_fold_tuning_space()


def alg_fold_config_is_valid(config, request) -> bool:
    """Whether one candidate CAN RUN for this request -- a hardware gate, not a preference.

    Purpose
        Keeps the sweep from timing configs the front door would refuse. Without it every such
        candidate raises inside the timed region and the sweep reports "config failed" for a
        reason that was knowable from the shapes.

    Semantics
        Pure in ``(config, request)``, as `ConfigSpace` requires: under a collective the surviving
        pool must be identical on every rank, or ranks compare timings for different kernels.

        Three gates, all read off the kernel's own front-door checks. ``tile_N`` must divide the 2N
        pre-activation, or the last tile is partial and the entry refuses it. Ping-pong caps
        ``tile_N`` at 208 for ``tile_M=128`` -- the two-warpgroup schedule's register budget -- so
        256 is invalid with it, which is not obvious and was found by a harvest run failing. And a
        tile taller than the token extent is pure rounding waste.

        The pool can never empty: ``tile_N=64`` cooperative survives whenever 2N is a multiple of
        64, and the fallback below covers the rest. An empty pool would leave the kernel with
        nothing to run at all.

    Args:
        config: The candidate. Reads ``config["tile_N"]`` and ``config["pingpong"]``.
        request: The bound call arguments by name. Reads ``x`` for M and ``Wg`` for N.

    Returns:
        True if this config can run for these operands.
    """
    tile_N, pingpong = config["tile_N"], config["pingpong"]
    m = request["x"].shape[-2]
    two_n = 2 * request["Wg"].shape[-2]
    if two_n % tile_N != 0:
        return False
    if pingpong and tile_N > 208:
        return False
    return _TILE_M <= m


@autotune(
    space=ALG_FOLD_TUNING_SPACE,
    key=["chunk_g", "blk_k", "activation", "gate3_activation"],
    validity=alg_fold_config_is_valid,
    gate="do_autotune",
)
def layernorm_dual_gated_gemm_alg_fold(
    x: Tensor,
    norm_weight: Tensor,
    Wg: Tensor,
    Wp: Tensor,
    PostAct: Tensor,
    *,
    tile_N: int = 128,
    pingpong: bool = False,
    norm_bias: Optional[Tensor] = None,
    bg: Optional[Tensor] = None,
    bp: Optional[Tensor] = None,
    mask: Optional[Tensor] = None,
    eps: float = 1e-5,
    W3: Optional[Tensor] = None,
    b3: Optional[Tensor] = None,
    PostAct3: Optional[Tensor] = None,
    chunk_g: int = 1,
    blk_k: Optional[int] = None,
    activation: str = "glu",
    gate3_activation: str = "sigmoid",
    do_autotune: bool = False,
) -> None:
    """The algebraic fold with its tile and schedule chosen for you, by measurement if asked.

    Purpose
        One entry point that is both the fixed-config call and the tuned one, so a caller who knows
        its config and a caller who wants one measured reach the same kernel through the same name.

    Semantics
        With ``do_autotune`` falsy this is a thin forward to `layernorm_dual_gated_gemm` with
        ``fusion_variant="alg_fold"``: one dict lookup of overhead, then the kernel. With it truthy
        the decorator sweeps `ALG_FOLD_TUNING_SPACE`, pruned by `alg_fold_config_is_valid`, times
        each survivor, caches the winner per shape, and calls the kernel with it.

        **``tile_N`` and ``pingpong`` are the tuned knobs, so passing them WITH ``do_autotune=True``
        is refused** -- overriding one knob would make the measured winner and the executed kernel
        different kernels. Pass them freely on the fixed path; that is how the heuristic's output is
        applied, and how `alg_fold_freeze` re-applies a winner.

        ``tile_M`` is fixed at 128 and is not exposed; a caller needing another CTA M wants
        `layernorm_dual_gated_gemm` directly.

    Args:
        x, norm_weight, Wg, Wp, PostAct: As `layernorm_dual_gated_gemm`. ``PostAct`` is written in
            place; its STRIDES select a transposed store.
        tile_N: CTA tile over the 2N-wide pre-activation. Must divide 2N. A tuned knob.
        pingpong: The two-warpgroup schedule. Caps ``tile_N`` at 208 here. A tuned knob.
        norm_bias, bg, bp, mask, eps, W3, b3, PostAct3: The optional fusion terms, forwarded
            unchanged. Their PRESENCE is part of the tuning key automatically, because each is a
            tensor and absence removes it from the request.
        chunk_g, blk_k, activation, gate3_activation: Forwarded, and named in the tuning key --
            each changes what is computed or how the k-loop associates, so a config measured under
            one value may not be the winner under another.
        do_autotune: Turn the sweep on. Keyword-only, so the check is one dict lookup and a
            positionally-passed flag cannot silently start a sweep.

    Returns:
        None. The results are in `PostAct` and, when gated, `PostAct3`.

    Raises:
        ValueError: From the decorator if a tuned knob is passed alongside ``do_autotune=True``;
            from the front door for a refused geometry.
    """
    return layernorm_dual_gated_gemm(
        x,
        norm_weight,
        Wg,
        Wp,
        PostAct,
        _TILE_M,
        tile_N,
        norm_bias=norm_bias,
        bg=bg,
        bp=bp,
        mask=mask,
        eps=eps,
        W3=W3,
        b3=b3,
        PostAct3=PostAct3,
        fusion_variant="alg_fold",
        chunk_g=chunk_g,
        blk_k=blk_k,
        activation=activation,
        gate3_activation=gate3_activation,
        pingpong=pingpong,
    )


def alg_fold_freeze(x: Tensor, norm_weight: Tensor, Wg: Tensor, Wp: Tensor, PostAct: Tensor, **kw):
    """Sweep ONCE for this shape, then return a callable bound to the winner.

    Purpose
        Autotuning is a development-time activity. Re-measuring in every fresh process is slow and
        non-deterministic, and a deployment wants the answer, not the search. This is how the answer
        gets carried: measure in one process, read ``.config``, and pin it thereafter.

    Semantics
        Runs one tuning pass through `layernorm_dual_gated_gemm_alg_fold`, then binds the winning
        config onto a plain call. The returned callable takes the SAME arguments as the tuned entry
        minus its knobs -- it already has those -- so a caller can keep passing fresh tensors of the
        same shape without re-entering the tuner. ``.config`` exposes the pick, which is what makes
        a frozen result reviewable rather than folklore.

        **The tuning pass RUNS the kernel**, repeatedly, and writes ``PostAct``. Pass the real
        operands; the contents afterwards are some candidate's output, so treat them as garbage.

    Args:
        x, norm_weight, Wg, Wp, PostAct: The operands to tune against. Their shapes and dtypes ARE
            the tuning key, so a frozen callable is only valid for that shape.
        **kw: Forwarded to the tuned entry. Must NOT include ``tile_N`` or ``pingpong`` -- those are
            what is being chosen -- nor ``do_autotune``, which is set here.

    Returns:
        A callable ``(x, norm_weight, Wg, Wp, PostAct, **kw) -> None`` with a ``.config`` attribute
        holding the winning `AutotuneConfig`.

    Raises:
        ValueError: If a tuned knob is passed in ``kw``.
        RuntimeError: From the tuner if every candidate failed, or if launched distributed with no
            initialized process group.
    """
    layernorm_dual_gated_gemm_alg_fold(x, norm_weight, Wg, Wp, PostAct, do_autotune=True, **kw)
    won = layernorm_dual_gated_gemm_alg_fold.autotuner.best_config

    def frozen(x, norm_weight, Wg, Wp, PostAct, **call_kw):
        """Run the kernel with the frozen config; `call_kw` overrides the freezing call's."""
        return layernorm_dual_gated_gemm_alg_fold(
            x, norm_weight, Wg, Wp, PostAct, **{**kw, **call_kw, **won.all_kwargs()}
        )

    frozen.config = won
    return frozen

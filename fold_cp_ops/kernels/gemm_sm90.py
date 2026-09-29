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
# This file has been modified by NVIDIA CORPORATION & AFFILIATES.

from typing import Tuple, Type, Callable, Optional
from functools import cached_property, partial
import math


import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.cute.nvgpu import cpasync, warpgroup
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass import Int32, Float32, Float16, const_expr
from cutlass.utils import LayoutEnum


from fold_cp_ops._internal.tile_scheduler import (
    TileSchedulerOptions,
    TileSchedulerArguments,
    TileScheduler,
    PersistenceMode,
)

# return PipelineStateWAdvance instead of PipelineState
import fold_cp_ops._internal.sm90_utils as fold_cp_ops_sm90_utils
from fold_cp_ops._internal.compile_time.template_params import (
    TemplateParams,
    TemplateParamsMixin,
)
from fold_cp_ops._internal.rounding import RoundingMode
from fold_cp_ops._internal.gemm_sm90_load import GemmSm90LoadMixin
from fold_cp_ops._internal.gemm_sm90_mma import GemmSm90MmaMixin, NamedBarrierGemm  # noqa: F401
from fold_cp_ops._internal.gemm_sm90_epilogue import GemmSm90EpilogueMixin


"""
A high-performance batched dense GEMM (C = A * B) example for the NVIDIA Hopper architecture
using CUTE DSL.
- Matrix A is MxKxL, L is batch dimension, A can be row-major("K") or column-major("M")
- Matrix B is NxKxL, L is batch dimension, B can be row-major("N") or column-major("K")
- Matrix C is MxNxL, L is batch dimension, C can be row-major("N") or column-major("M")

This GEMM kernel supports the following features:
    - Utilizes Tensor Memory Access (TMA) for efficient memory operations
    - Utilizes Hopper's WGMMA for matrix multiply-accumulate (MMA) operations
    - Implements TMA multicast with cluster to reduce L2 memory traffic
    - Supports multi-stage pipeline to overlap computation and memory access

This GEMM works as follows:
1. Load A and B matrices from global memory (GMEM) to shared memory (SMEM) using TMA operations.
2. Perform matrix multiply-accumulate (MMA) operations using WGMMA instruction.
3. Store results from registers (RMEM) to shared memory (SMEM), then to global memory (GMEM) with TMA operations.

Hopper WGMMA instructions operate as follows:
- Read matrix A from SMEM
- Read matrix B from SMEM
- Perform MMA operation and store the result in Accumulator(register)

Constraints:
* Supported input data types: fp16, fp8 (e4m3fn, e5m2)
* For fp16 types, A and B must have the same data type
* For fp8 types, A and B can have different types (e4m3fn or e5m2) but both must be 8-bit
* Fp8 types only support k-major layout
* Only fp32 accumulation is supported in this example
* CTA tile shape M must be 64/128
* CTA tile shape N must be 64/128/256
* CTA tile shape K must be 64
* Cluster shape M/N must be positive and power of 2, total cluster size <= 4
* The contiguous dimension of A/B/C tensors must be at least 16 bytes aligned,
  i.e, number of elements is a multiple of 8, 16 for Float16, and Float8, respectively.
"""


def check_fp8_major(a_dtype, a_major: str, b_dtype, b_major: str) -> None:
    """Refuse an 8-bit operand that is not k-major, naming the constraint.

    Hopper's 8-bit WGMMA atom exists only in the k-major form. Without this check the mistake
    surfaces as an MLIR verification failure -- "only support f16/bf16 with mn-major", then "failed
    to verify operands for the provided atom" -- raised from inside `partition_fragment_ABC`, which
    tells the caller nothing about which operand to transpose.

    Shared by the two front doors that can catch it: `GemmSm90.__call__` (which knows the operands'
    `LayoutEnum`) and `kernels/gemm.gemm` (which knows their torch strides). One function so the
    sentence is written once and cannot drift between them.

    Args:
        a_dtype: A's cutlass element type. Only `.width` is consulted, so any 8-bit float type is
            covered without enumerating them.
        a_major: `"k"` if A's k axis is contiguous, `"m"` otherwise. Callers holding a `LayoutEnum`
            pass `"k" if layout.is_k_major_a() else "m"`.
        b_dtype: B's cutlass element type.
        b_major: `"k"` if B's k axis is contiguous, `"n"` otherwise.

    Returns:
        None. Returning normally means both operands are acceptable.

    Raises:
        ValueError: If either operand is 8-bit and not k-major. 16-bit operands are unconstrained
            here and always pass.
    """
    # Two direct branches rather than a loop over a tuple of tuples: this runs on every call of the
    # host entry, and at 16 bits `.width == 8` is False immediately, so the success path allocates
    # nothing.  The message is built only when it is raised.
    if a_dtype is not None and a_dtype.width == 8 and a_major != "k":
        _raise_fp8_major("A", a_dtype, "m")
    if b_dtype is not None and b_dtype.width == 8 and b_major != "k":
        _raise_fp8_major("B", b_dtype, "n")


def _raise_fp8_major(name, dtype, other):
    """Build and raise the fp8-layout message. Off the hot path, so it may allocate.

    Args:
        name: ``"A"`` or ``"B"``.
        dtype: The 8-bit element type, named in the message.
        other: The major mode that is NOT allowed -- ``"m"`` for A, ``"n"`` for B.

    Returns:
        Never returns.

    Raises:
        ValueError: Always.
    """
    raise ValueError(
        f"unsupported layout for {dtype}: {name} must be k-major (row-major, "
        f"{name}.stride(-1) == 1). Hopper's 8-bit WGMMA atom has no mn-major form, so an "
        f"{other}-major fp8 {name} cannot be staged for the MMA. Transpose {name}, or use "
        f"a 16-bit dtype."
    )


#: Which :class:`_KernelContext` fields each warpgroup role takes. The roles are ``@cute.jit``, so
#: their parameters are flattened into MLIR values and cannot be a single Python object -- these
#: tuples are what expand the context back into the flat keyword arguments the DSL requires, in one
#: place, so adding a field to the context is a one-line change rather than four.
_PRODUCER_CTX = ("TileSchedulerCls", "ab_pipeline", "len_k", "sA", "sB", "storage")
_CONSUMER_CTX = (
    "TileSchedulerCls",
    "ab_pipeline",
    "epi_pipeline",
    "epi_smem_tensors",
    "has_C",
    "has_D",
    "len_k",
    "sA",
    "sB",
    "sC",
    "sD",
    "storage",
)


class _KernelContext:
    """Everything :meth:`GemmSm90.kernel_prologue` builds, handed to both warpgroup roles.

    **This is a trace-time bundle, not a runtime struct.** Every field holds a value produced while
    the kernel is being traced -- an SMEM tensor view, a pipeline object, a Python ``partial`` --
    so constructing it emits no instructions and costs no registers. It exists so the two role
    methods take one argument instead of fourteen, which is what makes them overridable: a subclass
    replacing ``producer_warpgroup_role`` does not have to track a positional signature that grows
    every time the prologue learns to build one more thing.

    ``__slots__`` is not an optimisation here -- it is the check that a typo in a field name fails
    at trace time rather than silently reading ``None`` and producing a kernel that stages nothing.

    Attributes:
        storage: The allocated ``SharedStorage`` instance.
        ab_pipeline: The mainloop A/B staging pipeline.
        epi_pipeline: The C-load pipeline, or None when there is no C.
        sched_pipeline: The work-tile broadcast pipeline, or None when not persistent.
        sched_data: SMEM tensor the scheduler broadcasts through, or None when not persistent.
        sA: Staged A tile in SMEM, swizzled.
        sB: Staged B tile in SMEM, swizzled.
        sD: Epilogue output staging tile, or None when there is no D.
        sC: C staging tile, or None when there is no C.
        epi_smem_tensors: Whatever extra SMEM tensors the epilogue mixin asked for.
        len_k: The contraction extent, as an ``Int32``.
        TileSchedulerCls: The scheduler class already bound to its params, data and pipeline, so
            each role constructs an identically-configured instance by calling it with no arguments.
        has_D: Whether an output tensor was supplied. ``const_expr``, so it prunes branches.
        has_C: Whether an addend was supplied. Likewise.
    """

    __slots__ = (
        "storage",
        "ab_pipeline",
        "epi_pipeline",
        "sched_pipeline",
        "sched_data",
        "sA",
        "sB",
        "sD",
        "sC",
        "epi_smem_tensors",
        "len_k",
        "TileSchedulerCls",
        "has_D",
        "has_C",
    )

    def __init__(self, **fields):
        """Bind every slot by keyword.

        Args:
            **fields: One entry per name in ``__slots__``. All are required and keyword-only --
                a positional form would let two same-typed SMEM tensors be swapped silently, which
                is a wrong-address bug with no diagnostic.

        Raises:
            AttributeError: If a name is not a declared slot.
            TypeError: If any slot is missing.
        """
        missing = set(self.__slots__) - set(fields)
        if missing:
            raise TypeError(f"_KernelContext missing field(s): {sorted(missing)}")
        for k, v in fields.items():
            setattr(self, k, v)


class GemmSm90Params(TemplateParams):
    """What a ``GemmSm90`` is configured with at CONSTRUCTION -- the seven genuine inputs.

    Purpose
        These are the knobs a caller (or an autotuner) picks. Everything else the constructor used
        to assign -- the atom layout, the warpgroup counts, the register split, the multicast flags
        -- is DERIVED from these, and is exposed as a ``cached_property`` rather than a parameter so
        it cannot drift from what it was derived from.

    Semantics
        Frozen and complete at construction, like every :class:`TemplateParams` pack. Together with
        :class:`GemmSm90CallParams` these fields are the functor's entire compile-time surface, so
        ``compile_key()`` is a complete cache key rather than a hand-maintained approximation of one.

    Attributes:
        acc_dtype: Accumulator element type. ``Float32`` for every supported operand type; a
            narrower accumulator is not an SM90 WGMMA option and is not checked here.
        a_dtype: The A operand's element type, declared at construction. It is re-read off the
            tensor in ``bind_operand_types`` and asserted equal -- a mismatch would mean the
            register/SMEM sizing was done for one type and the MMA issued for another.
        tile_shape_mn: ``(tile_M, tile_N)``. Which pairs are legal depends on ``pingpong`` and is
            checked in ``__init__``; an illegal pair raises there rather than exploding inside
            cutlass when the WGMMA atom is built.
        cluster_shape_mnk: ``(cluster_M, cluster_N, 1)``. Each of M/N must be a power of two and the
            product at most 4 -- the hardware cluster limit.
        pingpong: Two MMA warpgroups alternating mainloop and epilogue. Requires ``is_persistent``.
        is_persistent: Whether the grid is persistent (one CTA per SM walking a tile stream).
        fp8_fast_accum: Skip the slow-accumulation fixup for 8-bit operands. Ignored at 16 bits;
            ``fp8_slow_accum`` derives the effective behaviour.
    """

    acc_dtype: Type[cutlass.Numeric]
    a_dtype: Type[cutlass.Numeric]
    tile_shape_mn: Tuple[int, int]
    cluster_shape_mnk: Tuple[int, int, int]
    pingpong: bool = False
    is_persistent: bool = True
    fp8_fast_accum: bool = False


class GemmSm90CallParams(TemplateParams):
    """What a ``GemmSm90`` learns when it is handed real operands -- still compile-time constants.

    Purpose
        ``__call__`` is ``@cute.jit``, so every ``self.X`` read while tracing it is folded into the
        kernel. The operand dtypes and major modes are therefore just as much a template parameter
        as the tile shape; they are simply not knowable until the call. Declaring them makes that
        explicit and makes them immutable once bound.

    Semantics
        Bound exactly once, at the top of ``__call__``, by ``bind_operand_types``. Every field is a
        scalar, a numeric TYPE, or an enum -- never a tensor, layout or atom. Derived MLIR objects
        (the SMEM layouts, the cluster layout) are ``cached_property`` over these; the MMA atom is
        not on ``self`` at all, because it is already an explicit kernel argument.

        The last five fields are stage/tile counts rather than raw operand facts. They live here
        because they depend on ``epilogue_args``, which is a call ARGUMENT: no property of ``self``
        can see it, so they are computed once in ``bind_operand_types`` and bound alongside.

    Attributes:
        b_dtype: B's element type. Must equal ``a_dtype`` at 16 bits and match its width at 8.
        d_dtype: D's element type, or None when the kernel produces only a post-activation.
        c_dtype: C's element type, or None when there is no addend. None compiles the beta term out
            entirely, so a kernel built without one cannot be handed one later.
        a_layout: A's major mode. Decides the WGMMA atom's A operand mode and A's SMEM swizzle.
        a2_layout: The SECOND A operand's major mode, or None when there is only one A. Declared
            here rather than stashed on the functor because ``__call__`` is traced: a ``self.X`` read
            there is folded in at trace time, so an operand fact must be a bound parameter or it is
            mutable state that can desynchronize from an already-compiled kernel. Mirrors
            `two_tensor_B`, which is the same idea for a second B. The SM90 WGMMA bakes the A-major
            into the atom, so a second A whose major differs from the first needs its own atom and
            its own staged SMEM layout -- which is the whole reason this is a separate fact and not
            an assumption that it equals `a_layout`.
        b_layout: B's major mode. fp8 is k-major only -- ``check_fp8_major`` refuses the rest.
        d_layout: D's major mode, or None with no D.
        c_layout: C's major mode, or None with no C.
        two_tensor_B: Whether the gate weight arrives as a second B tensor (the block-interleaved
            gated load) rather than folded into B. Requires ``chunk_g > 1`` and ``cluster_M == 1``.
        rounding_mode: How the epilogue downconverts to the output type. Read off the epilogue
            arguments, defaulting to ``RN``; SM90 supports only ``RN`` for the post-activation store.
        cta_tile_k: The mainloop K tile, in elements -- the WGMMA atom's K extent times
            ``_MMA_INST_TILE_K``. Not a free choice: it is read back off the built atom, so it
            cannot disagree with what the MMA actually consumes (64 at 16 bits, 128 at fp8).
        epi_tile: The epilogue subtile ``(M, N)``, after any ``maybe_override_epi_tile`` hook.
        ab_stage: Mainloop pipeline depth, sized to fill SMEM after the epilogue's reservation.
        epi_stage: Epilogue pipeline depth for D.
        epi_c_stage: Epilogue pipeline depth for C; 0 when there is no C.
    """

    b_dtype: Type[cutlass.Numeric]
    d_dtype: Optional[Type[cutlass.Numeric]]
    c_dtype: Optional[Type[cutlass.Numeric]]
    a_layout: LayoutEnum
    a2_layout: Optional[LayoutEnum]
    b_layout: LayoutEnum
    d_layout: Optional[LayoutEnum]
    c_layout: Optional[LayoutEnum]
    two_tensor_B: bool
    rounding_mode: RoundingMode
    cta_tile_k: int
    epi_tile: Tuple[int, int]
    ab_stage: int
    epi_stage: int
    epi_c_stage: int


class GemmSm90(TemplateParamsMixin, GemmSm90EpilogueMixin, GemmSm90MmaMixin, GemmSm90LoadMixin):
    """
    This class implements batched matrix multiplication (C = A x B) with support for various data types
    and architectural features specific to Hopper GPUs with persistent tile scheduling and warp specialization.

    :param acc_dtype: Data type for accumulation during computation
    :type acc_dtype: type[cutlass.Numeric]
    :param tile_shape_mn: Shape of the CTA tile (M,N)
    :type tile_shape_mn: Tuple[int, int, int]
    :param cluster_shape_mnk: Cluster dimensions (M,N,K) for parallel processing
    :type cluster_shape_mnk: Tuple[int, int, int]

    :note: Data type requirements:
        - For 16-bit types: A and B must have the same data type
        - For 8-bit types: A and B can have different types (Float8E4M3FN/Float8E5M2) as long as both are 8-bit
        - Float8 types only support k-major layout

    :note: Supported data types:
        - Float16
        - BFloat16
        - Float8E4M3FN/Float8E5M2

    :note: Supported accumulation types:
        - Float32 (for all floating point inputs)

    :note: Constraints:
        - Cluster shape M/N must be positive and power of 2, total cluster size <= 4

    Example:
        >>> gemm = GemmSm90(
        ...     acc_dtype=Float32,
        ...     tile_shape_mn=(128, 256),
        ...     cluster_shape_mnk=(1, 1, 1)
        ... )
        >>> gemm(a_tensor, b_tensor, c_tensor, stream)
    """

    arch = 90

    #: Construction-phase template parameters -- the seven knobs a caller or autotuner picks.
    Params = GemmSm90Params
    #: Call-phase template parameters -- the operand facts, bound once at the top of ``__call__``.
    CallParams = GemmSm90CallParams

    #: Post-construction ``const_expr`` gates -- the compile-time reads this class makes that are
    #: NOT declared fields of the two packs above. Pinned against an AST scan of this class body by
    #: ``tests/_internal/compile_time/test_template_params.py``; adding a gate without adding it
    #: here fails that test. ``_a2a_cluster_*`` live here rather than in ``Params`` because the A2A
    #: SUBCLASS sets them and this parent only reads them.
    COMPILE_GATED_ATTRS = (
        "_a2a_cluster_drain",
        "_a2a_cluster_n",
        "_a2a_drain_tail",
        "_a2a_enabled",
        "_a2a_ib_wide",
        "_decoupled_active",
        "_gate3_full_width",
        "_skip_warpgroup_reg_realloc",
        "chunk_g",
    )

    # Per-work-tile epilogue-subtile stride multiplier.  1 for the standard single-epilogue-per-
    # work-tile path.  Subclasses that drive MULTIPLE epilogue invocations per work-tile (e.g. the
    # dual-gated cluster n_per_cta>1 multi-accumulator loop) set this to the invocation count so the
    # epilogue subtile counter (num_prev_subtiles) advances continuously across the per-work-tile
    # invocations (each given a distinct subtile_offset).
    _epi_subtile_mult = 1

    #: How many WGMMA K-instructions one mainloop K tile covers. With the atom's own K extent this
    #: gives ``cta_tile_k``; read off the built atom rather than assumed, so the loader stages
    #: exactly what the MMA consumes.
    _MMA_INST_TILE_K = 4

    #: Whether to PRESERVE the contraction extent's compile-time value when the operand's K extent
    #: is already static, instead of casting it to a runtime ``Int32``.
    #:
    #: **This is a codegen switch, not a numerics one.** ``len_k`` feeds ``_k_tile_cnt``, which is
    #: the PRODUCER's k-loop trip count. Cast to ``Int32`` the bound is a runtime value, so
    #: `cutlass.range` emits a top-tested loop with an early exit -- SASS ``ISETP`` + ``BREAK``
    #: inside a ``BSSY``/``BSYNC`` convergence region -- and the whole mainloop is built around a
    #: bound the compiler cannot use.
    #:
    #: **What that costs is NOT concentrated on the barrier instructions**, and the measurement is
    #: recorded here because the obvious reading is wrong. PC-sampling the convergence class finds
    #: only +0.18% of the kernel on it, against a 1.5% effect -- they do not serialize. The cost is
    #: structural: the dynamic bound makes the producer carry an EXTRA LOOP LEVEL, which is where
    #: the ``S2R``/``S2UR SR_CgaCtaId`` sits (**it is not in the innermost k-loop in either
    #: variant** -- do not grep the innermost loop for it and conclude the fix never landed), and
    #: the innermost loop then carries **three** ``LEA`` address computations where the upstream
    #: carries **one**. Left alone on a static extent the bound folds, that loop level disappears,
    #: and the producer's loop nest matches the upstream's row for row.
    #:
    #: Measured on the algebraic fold: producer loops 4 -> 3, innermost ``LEA`` 3 -> 1, SASS
    #: ``BSSY``/``BSYNC``/``BREAK`` 8/8/1 -> 7/7/0 (upstream: 7/7/0), and the D sweep at M=524288
    #: from up to +4.8% to within +/-0.3% at every D.
    #:
    #: **Default False, so every existing caller is byte-identical.** A kernel whose K extent is
    #: genuinely dynamic (varlen) is unaffected either way: ``shape[1]`` is already a runtime value
    #: there, so setting this True is a no-op rather than a correctness change. It is opt-in only
    #: because flipping it globally would move the codegen of every kernel in the package at once.
    #:
    #: **NOT IN THE ``jit_cache`` KEY -- toggle it at runtime only via
    #: `fold_cp_ops.testing.codegen_flags.codegen_flag`, never by plain assignment.** The key is
    #: ``(fn.__qualname__, *args, **kwargs)`` and this is a class attribute, not an argument, so two
    #: values hash the same. MEASURED: with a fresh cache dir, compiling one shape at True wrote 4
    #: ``.o`` files and re-running it at False wrote **4** -- the second setting silently ran the
    #: first's compiled code. A test that assigns this directly reports "no difference" without ever
    #: having compiled the second variant. Editing this default in the source is safe (the source
    #: fingerprint busts the disk cache); mutating it in a live process is not.
    _KEEP_STATIC_LEN_K = False

    #: CTAs per SM the SMEM stage heuristic budgets for. A class constant, not an instance
    #: attribute: nothing sets it per functor, and a per-instance copy would only be a place for it
    #: to be changed without the stage counts being recomputed.
    occupancy = 1
    #: Threads in a warpgroup. Hardware, not configuration.
    num_threads_per_warp_group = 128
    #: Warps driving the mainloop TMA loads. One warp issues both descriptors; this was a variable
    #: only to give the removed gather-A path its four cp.async warps.
    num_ab_load_warps = 1
    #: Alignment for the A/B/epilogue SMEM tiles, so their swizzled layouts sit on a swizzle
    #: boundary. Lowering it silently mis-swizzles rather than failing.
    buffer_align_bytes = 1024

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        a_dtype: Type[cutlass.Numeric],
        tile_shape_mn: Tuple[int, int],
        cluster_shape_mnk: Tuple[int, int, int],
        pingpong: bool = False,
        is_persistent: bool = True,
        fp8_fast_accum: bool = False,
        **extra_params,
    ):
        """Bind the construction-phase parameters and refuse a tile/cluster geometry SM90 cannot build.

        Purpose
            Establishes the kernel's configuration. Nothing operand-dependent happens here -- the
            dtypes of B/C/D, the major modes, the SMEM layouts and the stage counts are all bound in
            ``__call__``, from the tensors.

        Semantics
            Every argument becomes a field of :class:`GemmSm90Params`, immutable from here on.
            Everything the constructor used to *derive* -- ``atom_layout_mnk``, ``mma_warp_groups``,
            the register split, the multicast flags -- is now a ``cached_property``, so it is
            computed on first read and cannot be reassigned out from under an already-traced kernel.

        Args:
            acc_dtype: Accumulator type; ``Float32``.
            a_dtype: A's element type. 16-bit (fp16/bf16) or 8-bit (fp8 e4m3/e5m2). Used here to
                size the register budget, and asserted against the actual operand at call time.
            tile_shape_mn: ``(tile_M, tile_N)``. Legal pairs depend on ``pingpong``; see Raises.
            cluster_shape_mnk: ``(cluster_M, cluster_N, 1)``; each a power of two, product <= 4.
            pingpong: Two-warpgroup alternating schedule. Requires ``is_persistent=True``.
            is_persistent: Persistent grid.
            fp8_fast_accum: Skip the fp8 slow-accumulation fixup. No effect at 16 bits.
            **extra_params: Additional fields declared by a SUBCLASS's ``Params`` pack, forwarded
                into the single :meth:`_bind_params` call. This is how a subclass adds a parameter
                (``chunk_g``) without a second binding phase -- a second ``_bind_params`` raises,
                and rightly: the pack is complete at construction or it is not a pack. An unknown
                name is a ``TypeError`` from the pack constructor, naming it.

        Returns:
            None.

        Raises:
            ValueError: On a tile shape the WGMMA atom cannot be built for, or on
                ``pingpong and not is_persistent``. These are ``raise`` rather than ``assert``
                because ``python -O`` strips asserts, and a stripped check here does not fail
                loudly -- it builds a kernel whose scheduler and multicast configuration disagree.
            TypeError: From :meth:`_bind_params` if an argument is a runtime value.
        """
        self._bind_params(
            acc_dtype=acc_dtype,
            a_dtype=a_dtype,
            tile_shape_mn=tuple(tile_shape_mn),
            cluster_shape_mnk=tuple(cluster_shape_mnk),
            pingpong=pingpong,
            is_persistent=is_persistent,
            fp8_fast_accum=fp8_fast_accum,
            **extra_params,
        )
        if self.pingpong and not self.is_persistent:
            raise ValueError(
                "pingpong requires is_persistent=True: the two warpgroups alternate mainloop and "
                "epilogue across successive work tiles, which only exists in a persistent grid."
            )

        tile_M, tile_N = self.tile_shape_mn
        # check the cta tile shape
        if not self.pingpong:
            if tile_M not in [64, 128, 192, 256, 320]:
                raise ValueError("CTA tile shape M must be 64/128/192/256/320")
            if tile_M in [192, 320]:  # special case
                tile_N_max = 256 if tile_M == 192 else 160
                if not (tile_N % 32 == 0 and tile_N <= tile_N_max):
                    raise ValueError(
                        f"If tile_m == {tile_M}, CTA tile shape N must be divisible by 32 and <= {tile_N_max}"
                    )
            else:
                if not (
                    (tile_N % 16 == 0 and tile_N <= 256) or (tile_N % 32 == 0 and tile_N <= 512)
                ):
                    raise ValueError(
                        "CTA tile shape N must be divisible by 16 and <= 256, or divisible by 32 and <= 512"
                    )
        else:
            if tile_M not in [64, 128, 192]:
                raise ValueError("CTA tile shape M must be 64/128/192 if pingpong")
            tile_N_max = 256 if tile_M == 64 else (208 if tile_M == 128 else 128)
            if not (tile_N % 16 == 0 and tile_N <= tile_N_max):
                raise ValueError(f"CTA tile shape N must be divisible by 16 and <= {tile_N_max}")

        # The DERIVED constraint, and the one that actually binds: `_make_tiled_mma` builds the
        # WGMMA atom with `tiler_mn=(64, tile_N // atom_layout_n)`, and the SM90 f16/bf16 atom
        # accepts only 8 <= N <= 256 in steps of 8.  The per-branch checks above are a coarser
        # filter written in terms of tile_N alone, and one of their disjuncts -- "divisible by 32
        # and <= 512" -- admits tile_N values no atom_layout_n == 1 configuration can build:
        # tile_M=128, tile_N=512 passed validation and then died inside cutlass with
        # `OpError: expects the N-mode to satisfy 8 <= N <= 256`, an internal explosion rather than
        # a refusal.  Checking the derived value cannot drift from what the atom is built with.
        atom_n = tile_N // self.atom_layout_mnk[1]
        if not (8 <= atom_n <= 256 and atom_n % 8 == 0):
            raise ValueError(
                f"CTA tile shape N must be divisible by 8 after the N-split and leave at most 256 "
                f"per warpgroup: tile_N={tile_N} split across {self.atom_layout_mnk[1]} warpgroup(s) "
                f"gives {atom_n}, and the SM90 WGMMA atom accepts only 8 <= N <= 256 in steps of 8. "
                f"Use tile_N <= {256 * self.atom_layout_mnk[1]}."
            )
        if self.pingpong:
            assert self.mma_warp_groups == 2
        assert self.mma_warp_groups in [1, 2, 3]

        # The one genuinely mutable piece of launch state: the SharedStorage struct type, which is
        # assembled in __call__ from the (by then bound) stage counts. It is a Python class, not a
        # constant folded into the kernel, so it is not a template parameter.
        self.shared_storage = None

    # ── derived from the CONSTRUCTION parameters ──────────────────────────────────────────────
    # Each of these was an `__init__` assignment. As `cached_property` they cannot desynchronize
    # from what they are derived from, and `cached_property` writes ``__dict__`` directly, so the
    # parameter guard in `TemplateParamsMixin.__setattr__` does not block the memoization.

    @cached_property
    def fp8_slow_accum(self) -> bool:
        """Whether the mainloop keeps a second, slow accumulator for 8-bit operands.

        Returns:
            True only for an 8-bit ``a_dtype`` with ``fp8_fast_accum=False``. Doubles the
            accumulator register footprint, which is why the register split consults it.
        """
        return not self.fp8_fast_accum and self.a_dtype.width == 8

    @cached_property
    def atom_layout_mnk(self) -> Tuple[int, int, int]:
        """How the CTA tile is split across MMA warpgroups, as ``(m, n, 1)``.

        Semantics
            Derived purely from ``tile_shape_mn`` and ``pingpong``. tile_M 320 and the wide tile_M
            192 split along N because their M is not an even multiple of the 64-row atom; pingpong
            is always ``(1, 1, 1)`` because its two warpgroups alternate rather than divide.

        Returns:
            The atom layout. ``m`` in {1, 2, 3} and ``n`` in {1, 2}, asserted here because every
            downstream sizing (warpgroup count, register split, epilogue tile) assumes it.
        """
        tile_M, tile_N = self.tile_shape_mn
        if self.pingpong:
            return (1, 1, 1)
        if tile_M == 320:  # tile_M / 64 is not even so we have to split along N
            atom_layout_m, atom_layout_n = 1, 2
        elif tile_M == 192:
            atom_layout_m, atom_layout_n = (3, 1) if tile_N <= 128 else (1, 2)
        else:
            atom_layout_m = tile_M // 64 if tile_M < 256 else 2
            atom_layout_n = 1
        assert atom_layout_m in [1, 2, 3] and atom_layout_n in [1, 2]
        return (atom_layout_m, atom_layout_n, 1)

    @cached_property
    def num_mcast_ctas_a(self) -> int:
        """CTAs A is multicast to -- the cluster's N extent, since A is shared along N."""
        return self.cluster_shape_mnk[1]

    @cached_property
    def num_mcast_ctas_b(self) -> int:
        """CTAs B is multicast to -- the cluster's M extent, since B is shared along M."""
        return self.cluster_shape_mnk[0]

    @cached_property
    def is_a_mcast(self) -> bool:
        """Whether A's TMA descriptor is a multicast one. Decides the load atom, not just a count."""
        return self.num_mcast_ctas_a > 1

    @cached_property
    def is_b_mcast(self) -> bool:
        """Whether B's TMA descriptor is a multicast one."""
        return self.num_mcast_ctas_b > 1

    @cached_property
    def mma_warp_groups(self) -> int:
        """Warpgroups issuing WGMMA: the atom layout's size, doubled under pingpong."""
        return math.prod(self.atom_layout_mnk) * (2 if self.pingpong else 1)

    @cached_property
    def num_epi_warps(self) -> int:
        """Warps that run the epilogue. Under pingpong only one warpgroup is in the epilogue at a
        time, so this is 4 rather than ``mma_warp_groups * 4``; the epilogue named barrier is sized
        from it, and an over-count never releases."""
        return (1 if self.pingpong else self.mma_warp_groups) * 4

    @cached_property
    def epilogue_barrier(self) -> pipeline.NamedBarrier:
        """The named barrier the epilogue warps rendezvous on, sized to :attr:`num_epi_warps`."""
        return pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierGemm.Epilogue),
            num_threads=self.num_epi_warps * cute.arch.WARP_SIZE,
        )

    @cached_property
    def ab_load_warp_id(self) -> int:
        """Index of the first mainloop-load warp -- immediately after the MMA warpgroups."""
        return self.mma_warp_groups * 4

    @cached_property
    def sched_stage(self) -> int:
        """Depth of the tile-scheduler broadcast pipeline. 2 under pingpong, where one warpgroup
        consumes a tile while the other is still on the previous one."""
        return 2 if self.pingpong else 1

    @cached_property
    def threads_per_cta(self) -> int:
        """Block dimension: MMA warpgroups + one AB-load warpgroup + any extra warpgroups.

        Semantics
            Reads :meth:`_num_extra_warpgroups`, whose answer can depend on a flag a subclass flips
            AFTER construction (``configure_a2a_*`` is the case this exists for). As a lazy property
            the first read happens in ``resolve_launch_shape``, i.e. at launch, so the value cannot
            be computed too early -- which is what an ``__init__`` assignment got wrong and had to
            be patched up by recomputing.

        Returns:
            Thread count for ``.launch(block=...)``.
        """
        return (
            self.mma_warp_groups + 1 + self._num_extra_warpgroups()
        ) * self.num_threads_per_warp_group

    @cached_property
    def _base_regs_split(self) -> Tuple[int, int]:
        """The ``(load, mma)`` per-thread register budgets before any extra-warpgroup carve.

        Returns:
            ``(32, 160)`` for three MMA warpgroups; otherwise ``(24, 240)`` under heavy accumulator
            pressure (>= 208 registers per thread, doubled when ``fp8_slow_accum``) and ``(40, 232)``
            otherwise.
        """
        regs_per_thread = math.prod(self.tile_shape_mn) // (
            math.prod(self.atom_layout_mnk) * self.num_threads_per_warp_group
        )
        if self.fp8_slow_accum:
            regs_per_thread *= 2
        if self.mma_warp_groups == 3:
            return 32, 160
        return (24, 240) if regs_per_thread >= 208 else (40, 232)

    @cached_property
    def num_regs_load(self) -> int:
        """Per-thread register budget the load warpgroup shrinks to via ``setmaxnreg.dec``."""
        return self._base_regs_split[0]

    @cached_property
    def num_regs_mma(self) -> int:
        """Per-thread register budget the MMA warpgroups grow to -- clamped to stay FEASIBLE.

        Semantics
            **The clamp is a deadlock guard, not a tuning knob.** An SM has 65536 32-bit registers.
            With an extra warpgroup the block is larger, so the per-thread ceiling falls and the
            default MMA budget can land above it. A ``setmaxnreg.inc`` past the ceiling does not
            fail -- it WEDGES the launch, waiting for registers no warp will release. So the budget
            is shrunk to fit, rounded down to Hopper's 8-register allocation granularity.

            With no extra warpgroup (the default) the clamp does not apply and the value is exactly
            :attr:`_base_regs_split`'s -- which is what keeps the default kernel byte-identical.

        Returns:
            The MMA warpgroups' register budget.
        """
        r_mma = self._base_regs_split[1]
        n_extra_wg = self._num_extra_warpgroups()
        if n_extra_wg > 0:
            mma_threads = self.mma_warp_groups * self.num_threads_per_warp_group
            other_threads = self.threads_per_cta - mma_threads
            regs_budget = 65536  # 32-bit registers per SM (SM90/SM100 H100/H200/B200)
            r_mma_max = (regs_budget - other_threads * self.num_regs_load) // mma_threads
            r_mma_max = (r_mma_max // 8) * 8  # round down to the 8-reg allocation granularity
            r_mma = min(r_mma, r_mma_max)
        return r_mma

    @cached_property
    def cluster_layout_mnk(self) -> cute.Layout:
        """The cluster's layout. An MLIR object, hence a derived property and never a parameter."""
        return cute.make_layout(self.cluster_shape_mnk)

    # ── derived from the CALL parameters ──────────────────────────────────────────────────────

    @cached_property
    def cta_tile_shape_mnk(self) -> Tuple[int, int, int]:
        """The full CTA tile ``(M, N, K)``.

        The K extent is not a free choice: it is ``cta_tile_k``, read back off the WGMMA atom in
        :meth:`bind_operand_types`, so the loader stages exactly what the MMA consumes. Available
        only after the call parameters are bound -- before that there is no operand and therefore
        no atom.
        """
        return (*self.tile_shape_mn, self.cta_tile_k)

    @cached_property
    def _smem_layouts_staged(self):
        """The four staged SMEM layouts, built once from the bound parameters.

        Returns:
            ``(a, b, epi_d, epi_c)``. ``epi_c`` is None when there is no C. These are MLIR objects,
            so they are derived rather than declared -- a parameter pack would reject them.
        """
        return self._make_smem_layouts(
            self.cta_tile_shape_mnk,
            self.epi_tile,
            self.a_dtype,
            self.a_layout,
            self.b_dtype,
            self.b_layout,
            self.ab_stage,
            self.d_dtype,
            self.d_layout,
            self.epi_stage,
            self.c_dtype,
            self.c_layout,
            self.epi_c_stage,
        )

    @cached_property
    def a_smem_layout_staged(self):
        """A's staged SMEM layout, ``(tile_M, tile_K, ab_stage)`` with its swizzle."""
        return self._smem_layouts_staged[0]

    @cached_property
    def b_smem_layout_staged(self):
        """B's staged SMEM layout, ``(tile_N, tile_K, ab_stage)`` with its swizzle."""
        return self._smem_layouts_staged[1]

    @cached_property
    def epi_smem_layout_staged(self):
        """D's staged epilogue SMEM layout, ``(epi_M, epi_N, epi_stage)``."""
        return self._smem_layouts_staged[2]

    @cached_property
    def epi_c_smem_layout_staged(self):
        """C's staged epilogue SMEM layout, or None when there is no C."""
        return self._smem_layouts_staged[3]

    @cached_property
    def num_tma_load_bytes(self) -> int:
        """Bytes one AB pipeline stage moves -- the producer's transaction count.

        The producer commits exactly this many bytes and the consumer waits for exactly them, so a
        value that disagrees with what the descriptors actually move is a HANG, not a wrong answer.
        Derived from the same staged layouts the descriptors are built from, which is what makes
        the two unable to disagree.
        """
        return cute.size_in_bytes(
            self.b_dtype, cute.slice_(self.b_smem_layout_staged, (None, None, 0))
        ) + cute.size_in_bytes(
            self.a_dtype, cute.slice_(self.a_smem_layout_staged, (None, None, 0))
        )

    @classmethod
    def epi_rounding_mode(cls, epilogue_args) -> RoundingMode:
        """The downconversion rounding mode this launch's epilogue arguments ask for.

        Purpose
            ``rounding_mode`` is a call-phase template parameter, so it must be known BEFORE the
            epilogue params are built. This is the one seam that reads it off the arguments.

        Args:
            epilogue_args: The epilogue ``NamedTuple``. Read via ``getattr`` so a mixin whose
                arguments have no such field is not forced to declare one.

        Returns:
            The requested mode, or ``RoundingMode.RN`` when the arguments do not carry one. RN is
            the correct default rather than a placeholder: SM90 has no stochastic-rounding epilogue.
        """
        return getattr(epilogue_args, "rounding_mode", RoundingMode.RN)

    def bind_operand_types(self, mA, mB, mD, mC, mB2=None, epilogue_args=None, mA2=None):
        """Read every operand fact off the tensors, bind the call parameters, return the MMA atom.

        Purpose
            This is the boundary where ``GemmSm90`` stops being a configuration and becomes a kernel
            for specific operands. Everything downstream -- the SMEM layouts, the TMA boxes, the
            epilogue tile -- is derived from what is bound here, so a wrong answer here is not
            recoverable later.

        Semantics
            Computes the call-phase parameters in dependency order using PURE helpers, then binds
            them in a single :meth:`_bind_call_params` call, after which they are immutable. The
            order matters and is not arbitrary: the WGMMA atom is needed for ``cta_tile_k``, the CTA
            tile is needed for ``epi_tile``, and ``epi_tile`` is needed for the stage counts (which
            also need ``epilogue_args``, a call ARGUMENT no property of ``self`` can see -- that is
            why the stage counts are bound here rather than derived).

            **The MMA atom is returned, not stashed.** It carries MLIR values, so ``self`` is the
            wrong place for it; it was already an explicit ``kernel()`` argument, and now that is
            its only home.

        Args:
            mA: The A operand. Its element type must equal the declared ``a_dtype`` -- the register
                budget and SMEM sizing were done for that type, so a mismatch would size for one
                type and issue the MMA for another.
            mB: The B operand. Must have the SAME dtype as A at 16 bits and the same WIDTH at 8 --
                the SM90 atom has one accumulator per operand-type pair, and a mismatch is refused
                here rather than producing a wrong-typed MMA.
            mD: The output, or None. None means a post-activation-only epilogue.
            mC: The addend, or None.
            mB2: Optional second B operand for the two-tensor block-interleaved gated load. When
                given, ``chunk_g`` must be > 1 and ``cluster_M`` must be 1 -- B is never multicast
                on that path, and a cluster_M > 1 would build a multicast descriptor for a box
                that is not the full tile.
            epilogue_args: This launch's epilogue arguments. Consulted for the rounding mode and
                for the epilogue's per-stage SMEM footprint; ``None`` is treated as an epilogue that
                needs no extra SMEM and rounds to nearest.

        Returns:
            The ``cute.TiledMma`` built for these operands. Pass it to ``kernel()``.

        Raises:
            TypeError: On an A dtype that does not match the declared one, a 16-bit A/B dtype
                mismatch, an A/B width mismatch, or an operand that is neither 16- nor 8-bit. These
                are ``raise`` and not ``assert`` because ``python -O`` strips asserts, and a
                stripped check here builds a kernel whose MMA atom disagrees with its operands.
            ValueError: From :func:`check_fp8_major` for an mn-major fp8 operand, which Hopper has
                no atom for.
            RuntimeError: If called twice on one functor -- one functor is traced once, so a second
                bind means an instance is being reused for operands that need their own kernel.
            AssertionError: For the two-tensor preconditions above.
        """
        a_dtype = mA.element_type
        b_dtype = mB.element_type
        d_dtype = mD.element_type if mD is not None else None
        c_dtype = mC.element_type if mC is not None else None
        a_layout = LayoutEnum.from_tensor(mA)
        b_layout = LayoutEnum.from_tensor(mB)
        d_layout = LayoutEnum.from_tensor(mD) if mD is not None else None
        c_layout = LayoutEnum.from_tensor(mC) if mC is not None else None

        # `a_dtype` is declared at construction because the register split needs it there. This is
        # the assertion that keeps the declaration honest -- it used to be an overwrite, which
        # cannot fail and so hid a mismatch.
        if const_expr(a_dtype is not self.a_dtype):
            raise TypeError(
                f"A operand is {a_dtype} but this functor was constructed for {self.a_dtype}. The "
                f"register budget and SMEM stage counts were sized for the declared type; build a "
                f"second functor rather than passing a different one."
            )

        # Strategy-1: two-tensor block-interleaved gated B load. mB=Wp (up, (N,K,L)),
        # mB2=Wg (gate, (N,K,L)); the kernel synthesizes the 2N preact SMEM tile by interleaving
        # G-wide chunks [up_G | gate_G | ...].  Gated on mB2 is not None; the entire non-two-tensor
        # path is byte-for-byte unchanged.
        # The SECOND A operand's major, bound for the same reason `two_tensor_B` is: it is an
        # operand fact, and `__call__` is traced, so it must be a bound parameter rather than an
        # attribute assigned mid-trace. None when there is only one A, which is every caller but
        # the two-A x_gate path.
        a2_layout = const_expr(LayoutEnum.from_tensor(mA2) if mA2 is not None else None)
        two_tensor_B = const_expr(mB2 is not None)
        if const_expr(two_tensor_B):
            assert getattr(self, "chunk_g", 1) != 1, "two-tensor B requires chunk_g>1"
            assert self.cluster_shape_mnk[0] == 1, (
                "two-tensor (chunked) B requires cluster_M=1 (no B-multicast)"
            )
            assert mB2.element_type == mB.element_type

        if const_expr(a_dtype.width == 16 and a_dtype != b_dtype):
            raise TypeError(f"Type mismatch: {a_dtype} != {b_dtype}")
        if const_expr(a_dtype.width != b_dtype.width):
            raise TypeError(f"Type width mismatch: {a_dtype.width} != {b_dtype.width}")
        if const_expr(a_dtype.width != 16 and a_dtype.width != 8):
            raise TypeError("a_dtype should be float16 or float8")
        check_fp8_major(
            a_dtype,
            "k" if a_layout.is_k_major_a() else "m",
            b_dtype,
            "k" if b_layout.is_k_major_b() else "n",
        )

        tiled_mma = self._make_tiled_mma(b_dtype, a_layout, b_layout)
        cta_tile_k = cute.size(tiled_mma.shape_mnk, mode=[2]) * self._MMA_INST_TILE_K
        cta_tile_shape_mnk = (*self.tile_shape_mn, cta_tile_k)

        epi_tile = self._sm90_compute_tile_shape_or_override(
            cta_tile_shape_mnk, self.atom_layout_mnk, d_dtype
        )
        # Hook: gated kernels with a contiguous up/gate chunk size G>1 force epi_tile_n = max(32, 2G)
        # so each epilogue N-subtile holds a whole [up_G | gate_G] block (register-local gating).
        epi_tile = self.maybe_override_epi_tile(epi_tile)

        ab_stage, epi_stage, epi_c_stage = self._compute_stages(
            cta_tile_shape_mnk,
            epi_tile,
            a_dtype,
            b_dtype,
            d_dtype,
            c_dtype,
            epilogue_args,
            cutlass.utils.get_smem_capacity_in_bytes(f"sm_{self.arch}"),  # smem_capacity
            self.occupancy,
        )

        self._bind_call_params(
            b_dtype=b_dtype,
            d_dtype=d_dtype,
            c_dtype=c_dtype,
            a_layout=a_layout,
            a2_layout=a2_layout,
            b_layout=b_layout,
            d_layout=d_layout,
            c_layout=c_layout,
            two_tensor_B=two_tensor_B,
            rounding_mode=self.epi_rounding_mode(epilogue_args),
            cta_tile_k=cta_tile_k,
            epi_tile=epi_tile,
            ab_stage=ab_stage,
            epi_stage=epi_stage,
            epi_c_stage=epi_c_stage,
        )
        return tiled_mma

    def assume_aligned_strides(self, mA, mB, mD, mB2=None):
        """Re-declare every dynamic stride as 128-bit aligned, so TMA can use its widest access.

        The 16-byte floor is a promise the host entry (``gemm()``) checks on real tensors, so
        asserting it to the compiler here is sound. **A caller that bypasses the host entry and
        breaks the promise gets a misaligned access, not a diagnostic** -- there is no runtime
        check on this path, by design, because one would cost a branch per launch.

        Static strides are left alone: they are already known and ``cute.assume`` on a constant
        adds an op for nothing.

        Args:
            mA: A operand. Returned re-declared.
            mB: B operand. Only re-declared on the two-tensor path; on the ordinary path the
                descriptor is built from ``_remap_B_operand_layout(mB)`` and the assumption would
                be redundant, so it is returned unchanged.
            mD: Output, or None. Returned re-declared (None passes through).
            mB2: Second B operand, or None. Re-declared on the two-tensor path.

        Returns:
            ``(mA, mB, mD, mB2)`` -- views with the same iterators and shapes and re-declared
            strides. Views, not copies: nothing is moved.
        """

        def new_stride(t: cute.Tensor):
            """Assume every dynamic stride is 128-bit aligned, leaving static ones untouched.

            Args:
                t: The tensor whose strides to re-declare.

            Returns:
                A tuple of strides with ``cute.assume(..., divby=...)`` on the dynamic ones.
            """
            return tuple(
                cute.assume(s, divby=128 // t.element_type.width) if not cute.is_static(s) else s
                for s in t.stride
            )

        mA, mD = [
            cute.make_tensor(t.iterator, cute.make_layout(t.shape, stride=new_stride(t)))
            if t is not None
            else None
            for t in (mA, mD)
        ]
        if const_expr(self.two_tensor_B):
            mB = cute.make_tensor(mB.iterator, cute.make_layout(mB.shape, stride=new_stride(mB)))
            mB2 = cute.make_tensor(
                mB2.iterator, cute.make_layout(mB2.shape, stride=new_stride(mB2))
            )
        return mA, mB, mD, mB2

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mD: Optional[cute.Tensor],
        mC: Optional[cute.Tensor],
        epilogue_args: tuple,
        scheduler_args: TileSchedulerOptions,
        stream: cuda.CUstream,
        mB2: Optional[cute.Tensor] = None,
    ):
        """Execute the GEMM operation in steps:
        - Setup static attributes
        - Setup TMA load/store atoms and tensors
        - Compute grid size
        - Define shared storage for kernel
        - Launch the kernel synchronously

        :param mA: Input tensor A
        :type mA: cute.Tensor
        :param mB: Input tensor B
        :type mB: cute.Tensor
        :param mD: Output tensor D
        :type mD: cute.Tensor
        :param stream: CUDA stream for asynchronous execution
        :type stream: cuda.CUstream
        """

        tiled_mma = self.bind_operand_types(mA, mB, mD, mC, mB2, epilogue_args)
        mA, mB, mD, mB2 = self.assume_aligned_strides(mA, mB, mD, mB2)
        (
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_b2,
            tma_tensor_b2,
        ) = self.make_mainloop_tma(mA, mB, mB2, epilogue_args)
        tma_atom_d, tma_tensor_d, tma_atom_c, tma_tensor_c = self.make_epilogue_tma(
            mD, mC, epilogue_args
        )
        # Force the one derived value the DEVICE kernel is the first to read. A `cached_property`
        # that emits MLIR must be materialized HERE, in `__call__`'s region: evaluated first from
        # inside `@cute.kernel`, its ops land in the kernel region while their operands live outside
        # it, and the module fails verification with "using value defined outside the region".
        _ = self.num_tma_load_bytes
        epilogue_params = self.epi_to_underlying_arguments(epilogue_args)

        TileSchedulerCls = self.get_scheduler_class()
        tile_sched_args = self.get_scheduler_arguments(mA, mB, mD, scheduler_args, epilogue_args)
        tile_sched_params = TileSchedulerCls.to_underlying_arguments(tile_sched_args)
        grid = TileSchedulerCls.get_grid_shape(
            tile_sched_params, scheduler_args.max_active_clusters
        )
        # Overridable persistent-grid hook (default identity -> byte-identical). A subclass may reduce the
        # z-extent (dim 2, the per-CTA count) here; the base returns grid unchanged.
        grid = self._persistent_grid_adjust(grid)

        epi_smem_size = cute.cosize(self.epi_smem_layout_staged) if mD is not None else 0
        epi_c_smem_size = cute.cosize(self.epi_c_smem_layout_staged) if mC is not None else 0

        @cute.struct
        class SharedStorage:
            """The kernel's shared-memory layout, assembled per configuration.

            Field ORDER is the allocation order, and the alignment attributes are load-bearing: the
            mbarrier arrays must come first at their natural alignment, and the A/B/epilogue tiles
            are 1024-byte aligned so their swizzled layouts sit on a swizzle boundary. Reordering
            the fields changes the offsets every partitioning below was derived against.
            """

            ab_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.ab_stage * 2]
            epi_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.epi_c_stage * 2]
            sched_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.sched_stage * 2]
            sched_data: cute.struct.MemRange[Int32, self.sched_stage * 4]
            sD: cute.struct.Align[
                cute.struct.MemRange[
                    self.d_dtype if self.d_dtype is not None else Int32, epi_smem_size
                ],
                self.buffer_align_bytes,
            ]
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype if self.c_dtype is not None else Int32, epi_c_smem_size
                ],
                self.buffer_align_bytes,
            ]
            epi: self.epi_get_smem_struct(epilogue_params)
            # Optional extra SMEM for a subclass (e.g. the decoupled peer-store ring's
            # mbarriers + metadata). Default hook -> a 0-size MemRange (adds nothing -> the
            # struct + SMEM footprint are byte-identical).
            decoupled: self._extra_smem_struct()
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage
        self.resolve_launch_shape()

        # Launch the kernel synchronously
        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_d,
            tma_tensor_d,
            tma_atom_c,
            tma_tensor_c,
            epilogue_params,
            self.cluster_layout_mnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
            self.epi_c_smem_layout_staged,
            tile_sched_params,
            TileSchedulerCls,
            tma_atom_b2,
            tma_tensor_b2,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            stream=stream,
            min_blocks_per_mp=1,
        )
        return

    def resolve_launch_shape(self):
        """Force the launch geometry to be derived NOW, then let a subclass veto the register dance.

        Purpose
            :attr:`threads_per_cta` and :attr:`num_regs_mma` both consult
            :meth:`_num_extra_warpgroups`, whose answer can change AFTER construction (a subclass's
            ``configure_a2a_*`` call is the case this exists for). Reading them here pins them at
            launch, which is the first moment the answer is final; the derivation itself lives in
            those properties, where it cannot drift from what the kernel is launched with.

        Semantics
            With no extra warpgroup (the default) this changes nothing: the properties evaluate to
            exactly the constructor's values and the hook is not reached, so the emitted code is
            identical.

        Returns:
            None. Materializes the two cached properties and calls the subclass hook.

        Note:
            ``_extra_wg_reg_adjust`` is a subclass hook that may suppress the asymmetric realloc
            entirely -- needed where the MMA ``.inc`` target is unreachable from the kernel's actual
            REGCOUNT baseline, which blocks forever. Default is a no-op.
        """
        n_extra_wg = self._num_extra_warpgroups()
        # Materialize both before the hook: `_extra_wg_reg_adjust` may set a flag the kernel reads,
        # and the budgets it is reasoning about must already be the final ones.
        _ = (self.threads_per_cta, self.num_regs_mma)
        if const_expr(n_extra_wg > 0):
            self._extra_wg_reg_adjust(n_extra_wg)

    #  GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: Optional[cute.CopyAtom],
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_d: Optional[cute.CopyAtom],
        mD_mnl: Optional[cute.Tensor],
        tma_atom_c: Optional[cute.CopyAtom],
        mC_mnl: Optional[cute.Tensor],
        epilogue_params,
        cluster_layout_mnk: cute.Layout,
        a_smem_layout: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
        epi_smem_layout: cute.ComposedLayout,
        epi_c_smem_layout: cute.ComposedLayout,
        tile_sched_params,
        TileSchedulerCls: cutlass.Constexpr[Callable],
        tma_atom_b2: Optional[cute.CopyAtom] = None,
        mB2_nkl: Optional[cute.Tensor] = None,
    ):
        """
        GPU device kernel performing the batched GEMM computation.

        :param tma_atom_a: TMA copy atom for A tensor
        :type tma_atom_a: cute.CopyAtom
        :param mA_mkl: Input tensor A
        :type mA_mkl: cute.Tensor
        :param tma_atom_b: TMA copy atom for B tensor
        :type tma_atom_b: cute.CopyAtom
        :param mB_nkl: Input tensor B
        :type mB_nkl: cute.Tensor
        :param tma_atom_d: TMA copy atom for D tensor
        :type tma_atom_d: cute.CopyAtom
        :param mD_mnl: Output tensor D
        :type mD_mnl: cute.Tensor
        :param tiled_mma: Tiled MMA object
        :type tiled_mma: cute.TiledMma
        :param cluster_layout_mnk: CTA layout
        :type cluster_layout_mnk: cute.Layout
        :param a_smem_layout: Shared memory layout for A
        :type a_smem_layout: cute.ComposedLayout
        :param b_smem_layout: Shared memory layout for B
        :type b_smem_layout: cute.ComposedLayout
        :param epi_smem_layout: Shared memory layout for epilogue
        :type epi_smem_layout: cute.ComposedLayout
        """

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        self.prefetch_tma_descriptors(
            warp_idx, tma_atom_a, tma_atom_b, tma_atom_b2, tma_atom_d, tma_atom_c
        )
        ctx = self.kernel_prologue(
            warp_idx,
            tiled_mma,
            mA_mkl,
            mD_mnl,
            mC_mnl,
            epilogue_params,
            cluster_layout_mnk,
            a_smem_layout,
            b_smem_layout,
            epi_smem_layout,
            epi_c_smem_layout,
            tile_sched_params,
            TileSchedulerCls,
        )
        self.producer_warpgroup_role(
            warp_idx,
            cluster_layout_mnk,
            mA_mkl,
            mB_nkl,
            mB2_nkl,
            tma_atom_a,
            tma_atom_b,
            tma_atom_b2,
            epilogue_params,
            tile_sched_params,
            **{f: getattr(ctx, f) for f in _PRODUCER_CTX},
        )
        self.mma_warpgroup_role(
            warp_idx,
            tiled_mma,
            mA_mkl,
            mD_mnl,
            mC_mnl,
            tma_atom_d,
            tma_atom_c,
            epilogue_params,
            tile_sched_params,
            **{f: getattr(ctx, f) for f in _CONSUMER_CTX},
        )
        self.kernel_exit_sync()

    @cute.jit
    def prefetch_tma_descriptors(self, warp_idx, *tma_atoms):
        """Warm the TMA descriptor cache from the single load warp, before any copy is issued.

        A descriptor read on its first use stalls the load warp for hundreds of cycles at the top
        of the mainloop, where nothing else is in flight to hide it. Prefetching costs one
        instruction per atom and is issued once per CTA.

        Args:
            warp_idx: This thread's warp-uniform index. Only ``ab_load_warp_id`` prefetches; the
                others skip, since the cache is per-SM and duplicate prefetches are pure overhead.
            *tma_atoms: The atoms to warm, ``None`` entries skipped. ``const_expr``-pruned, so an
                absent optional operand emits nothing at all.

        Returns:
            None. The prefetch is a side effect.
        """
        if warp_idx == self.ab_load_warp_id:
            for tma_atom in tma_atoms:
                if const_expr(tma_atom is not None):
                    cpasync.prefetch_descriptor(tma_atom)

    def kernel_prologue(
        self,
        warp_idx,
        tiled_mma,
        mA_mkl,
        mD_mnl,
        mC_mnl,
        epilogue_params,
        cluster_layout_mnk,
        a_smem_layout,
        b_smem_layout,
        epi_smem_layout,
        epi_c_smem_layout,
        tile_sched_params,
        TileSchedulerCls,
    ):
        """Allocate SMEM, build the pipelines, initialise every mbarrier, and sync the cluster.

        **Everything here must happen before any warp diverges into a role**, and the ordering is
        the correctness content: the mbarriers are constructed, then ``pipeline_init_arrive``
        publishes them, then ``pipeline_init_wait`` blocks until every CTA in the cluster has done
        the same. A warp that reached a barrier before its owner initialised it would wait on
        garbage -- which is a hang, not a wrong answer. The subclass hook ``_init_extra_smem`` is
        called inside that same window for exactly this reason.

        Args:
            warp_idx: Warp-uniform index, forwarded to ``_init_extra_smem``.
            tiled_mma: The MMA atom, which sizes the AB pipeline's consumer arrive count.
            mA_mkl: A operand, read only for its K extent.
            mD_mnl: Output tensor, or None. Presence decides whether ``sD`` exists.
            mC_mnl: Addend, or None. Presence decides whether ``sC`` and the epi pipeline exist.
            epilogue_params: Underlying epilogue params, for the mixin's own SMEM tensors.
            cluster_layout_mnk: Cluster layout, sizing the multicast and the cluster barriers.
            a_smem_layout: Staged SMEM layout for A -- outer layout plus swizzle.
            b_smem_layout: Staged SMEM layout for B.
            epi_smem_layout: Staged epilogue SMEM layout, used only when ``mD_mnl`` is present.
            epi_c_smem_layout: Staged C SMEM layout, used only when ``mC_mnl`` is present.
            tile_sched_params: Scheduler params, bound into the returned factory.
            TileSchedulerCls: The scheduler class; returned partially applied so both roles
                construct their own instance from identical arguments.

        Returns:
            A :class:`_KernelContext` holding the storage, the three pipelines, the SMEM tensors
            and the bound scheduler factory. Every field is a trace-time value, so the context is
            an ordinary Python object with no runtime cost.
        """
        has_D = const_expr(mD_mnl is not None)
        has_C = const_expr(mC_mnl is not None)

        # Alloc and init AB full/empty + ACC full mbar (pipeline)
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        # Stash the per-CTA storage so the decoupled-A2A producer copy_fn (reached deep inside the
        # MMA warpgroup's epilogue, which is NOT threaded `storage`) can reach the decoupled ring's
        # SMEM (full/empty mbarriers + meta + pcount). Trace-time attribute binding within THIS
        # kernel trace; the subclass reads it only on the decoupled path. Harmless for the default
        # path (a plain Python attr set, no device code), and never read when _a2a_decoupled is off.
        #
        # PLACEMENT DIVERGES FROM `main` ON PURPOSE. `main` sets this inside
        # `DualGatedGemmStagedSm90.kernel` in `kernels/dual_gated_gemm_staged.py` (verified: that
        # file is where `main` holds `_epi_storage`, and `gemm_sm90.py` on `main` has no occurrence
        # of it at all), because its A2A class descends from that type, which overrides `kernel()`.
        # Ours descends `DualGatedGemmDistSm90 -> DualGatedGemmSm90 -> GemmSm90` and overrides
        # `kernel()` nowhere, so the base is what runs for it. Without the write,
        # `dual_gated_gemm_a2a.py` raises `AttributeError: no attribute '_epi_storage'` on the
        # ib_drain_hybrid path.
        #
        # NOTE THE METHOD: this is `kernel_prologue`, NOT `kernel`. The distinction IS the
        # robustness argument, and an earlier revision of this comment got it wrong -- it said
        # "THIS is the `kernel()` that actually runs", which reads as "any `kernel()` override loses
        # the write". That is FALSE, and the false reading was reported as a live fragility before
        # the code was re-read. `kernel()` (:1191-1284) merely DELEGATES here at :1244, and every
        # `kernel()` override in this tree delegates too -- `layernorm_dual_gated_gemm.py:2250` is
        # the in-tree instance, overriding `kernel()` and staying correct precisely because it calls
        # `self.kernel_prologue(...)`. So an override is not the hazard; an override that SKIPS the
        # prologue is, as is a prologue that stops writing. Both are asserted by
        # `tests/distributed/test_dual_gated_gemm_a2a.py::
        # test_the_prologue_this_class_resolves_still_stashes_epi_storage`, because a comment is not
        # a check -- this one was wrong for a while and nothing noticed.
        self._epi_storage = storage

        ab_pipeline = self.make_ab_pipeline(
            tiled_mma=tiled_mma,
            cluster_layout_vmnk=cute.make_layout((1, *cluster_layout_mnk.shape)),
            ab_pipeline_mbar_ptr=storage.ab_pipeline_array_ptr.data_ptr(),
        )
        epi_pipeline = None
        if const_expr(has_C):
            epi_pipeline = self.make_epi_pipeline(
                c_smem_layout=cute.slice_(epi_c_smem_layout, (None, None, 0)),
                epi_pipeline_mbar_ptr=storage.epi_pipeline_array_ptr.data_ptr(),
            )
        sched_pipeline = None
        sched_data = None
        if const_expr(self.is_persistent):
            sched_pipeline = self.make_sched_pipeline(
                cluster_layout_mnk,
                sched_pipeline_mbar_ptr=storage.sched_pipeline_array_ptr.data_ptr(),
            )
            sched_data = storage.sched_data.get_tensor((4, self.sched_stage))

        # Optional subclass SMEM mbarrier init (e.g. the decoupled ring's full/empty barriers),
        # in the SAME pre-split window as the pipeline mbarrier inits so the cluster-wide
        # pipeline_init_wait below orders them before any warp uses them. Default no-op.
        self._init_extra_smem(storage, warp_idx)

        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mnk[:-1], is_relaxed=True)

        # Generate smem tensor A/B
        sA = storage.sA.get_tensor(a_smem_layout.outer, swizzle=a_smem_layout.inner)
        sB = storage.sB.get_tensor(b_smem_layout.outer, swizzle=b_smem_layout.inner)
        sD = None
        if const_expr(has_D):
            sD = storage.sD.get_tensor(epi_smem_layout.outer, swizzle=epi_smem_layout.inner)
        sC = None
        if const_expr(has_C):
            sC = storage.sC.get_tensor(epi_c_smem_layout.outer, swizzle=epi_c_smem_layout.inner)
        epi_smem_tensors = self.epi_get_smem_tensors(epilogue_params, storage)

        # See `_KEEP_STATIC_LEN_K`: the cast is what makes the producer's k-loop bound a RUNTIME
        # value, and with it the loop form and the hoisting of everything inside that loop.
        len_k = mA_mkl.shape[1] if const_expr(self._KEEP_STATIC_LEN_K) else Int32(mA_mkl.shape[1])

        TileSchedulerCls = partial(
            TileSchedulerCls.create, tile_sched_params, sched_data, sched_pipeline
        )

        # Cluster wait for barrier init
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mnk[:-1])

        return _KernelContext(
            storage=storage,
            ab_pipeline=ab_pipeline,
            epi_pipeline=epi_pipeline,
            sched_pipeline=sched_pipeline,
            sched_data=sched_data,
            sA=sA,
            sB=sB,
            sD=sD,
            sC=sC,
            epi_smem_tensors=epi_smem_tensors,
            len_k=len_k,
            TileSchedulerCls=TileSchedulerCls,
            has_D=has_D,
            has_C=has_C,
        )

    def kernel_exit_sync(self):
        """Cluster-exit barrier, for a subclass whose peers write into this CTA's SMEM.

        Default: nothing at all -- the whole body is ``const_expr``-gated off, so a plain GEMM's
        PTX is unchanged.

        The case it exists for: the A2A cluster drain has rank 0's consumer warp cross-CTA-arrive
        into every sibling's SMEM. Without this barrier a sibling can RETIRE and free that SMEM
        while rank 0's slower drain is still running, so the arrive lands on freed memory --
        reliably a CUDA_ERROR_LAUNCH_FAILED at cluster_n >= 4, and passing by luck at 2. Deferring
        every CTA's retirement until all siblings arrive here is perf-neutral, because the drain is
        already the critical path: the producers spin rather than retire and leave the SM idle.

        Returns:
            None.
        """
        # FIX-1 : cluster-exit barrier. rank-0's cluster_drain consumer warp cross-CTA-arrives
        # empty[] into EVERY sibling CTA's SMEM (_cluster_drain_loop*: mbarrier_arrive peer_cta_rank_in_
        # cluster=r). With no cluster-exit sync a sibling producer CTA can RETIRE (free its SMEM) while
        # rank-0's slow blocking-IB drain still runs -> the DSMEM arrive hits freed SMEM -> CUDA_ERROR_
        # LAUNCH_FAILED (reliably at cluster_n>=4 multi-band; cn2 passes by wall-clock luck). cluster_
        # arrive_relaxed + cluster_wait defer every CTA's retirement until all siblings reach here (they
        # spin, resident), so every empty-arrive lands on a live sibling. Perf-neutral (the drain is
        # already the critical path -> producers spin instead of retire-then-SM-idle). const_expr-gated
        # -> non-cluster / cluster_n==1 paths are byte-identical.
        if const_expr(
            getattr(self, "_a2a_cluster_drain", False)
            and int(getattr(self, "_a2a_cluster_n", 1)) > 1
        ):
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()

    def tail_drain_role(self, warp_idx, storage, epilogue_params, tile_sched_params):
        """No-op tail-drain hook (overridden by the A2A claim-drain subclass).

        Default: pass — so the symbol always resolves and the const_expr-gated call above is a no-op
        on the plain parent (the branch is elided at trace anyway)."""
        pass

    def get_scheduler_class(self):
        """Return the tile-scheduler class to instantiate.

        Returns:
            :class:`TileScheduler`. Overridden by a subclass that needs a different traversal --
            ``TriangularTileScheduler`` for the TriMul half-matrix, or an A2A run-aware scheduler.
            Whatever is returned must expose ``to_underlying_arguments``, ``get_grid_shape`` and
            ``create``, since :meth:`__call__` calls all three.
        """
        return TileScheduler

    def get_scheduler_arguments(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mD: Optional[cute.Tensor],
        scheduler_args,
        epilogue_args,
    ):
        """Build the scheduler's arguments from the operands and the launch options.

        Derives the tile grid (``ceil_div`` of each problem extent by the CTA tile) and picks the
        persistence mode from the launch options: NONE when not persistent, DYNAMIC when a tile
        counter semaphore was supplied, STATIC otherwise. There is no cluster-launch-control mode:
        CLC is Blackwell hardware and this package is SM90-only.

        Args:
            mA: The A operand, read for its M extent only.
            mB: The B operand, read for its N extent (doubled under two-tensor B, whose ``shape[0]``
                is the post-activation width) and, when ``mD`` is None, its batch extent.
            mD: The output, read for its batch extent. May be None for a post-activation-only
                epilogue, in which case ``mB.shape[2]`` supplies the batch count instead.
            scheduler_args: A :class:`TileSchedulerOptions`. ``tile_count_semaphore`` selects
                DYNAMIC persistence; ``raster_order``, ``max_swizzle_size``, ``batch_idx_permute``
                and ``run_j_tiles`` pass straight through.
            epilogue_args: The epilogue arguments. Unused here; present because subclasses derive
                their grid from epilogue state (the A2A per-peer tiling reads its PE map from it).

        Returns:
            A :class:`TileSchedulerArguments` ready for ``to_underlying_arguments``.
        """
        if const_expr(not self.is_persistent):
            persistence_mode = PersistenceMode.NONE
        elif const_expr(scheduler_args.tile_count_semaphore is not None):
            persistence_mode = PersistenceMode.DYNAMIC
        else:
            persistence_mode = PersistenceMode.STATIC
        num_problems = mD.shape[2] if mD is not None else mB.shape[2]
        # Two-tensor B: mB.shape[0] is the postact width N, but the logical preact N is 2N.
        n_problem = (
            mB.shape[0] * 2 if const_expr(getattr(self, "two_tensor_B", False)) else mB.shape[0]
        )
        problem_shape_ntile_mnl = (
            cute.ceil_div(mA.shape[0], self.cta_tile_shape_mnk[0]),
            cute.ceil_div(n_problem, self.cta_tile_shape_mnk[1]),
            num_problems,
        )
        return TileSchedulerArguments(
            problem_shape_ntile_mnl=problem_shape_ntile_mnl,
            raster_order=scheduler_args.raster_order,
            group_size=scheduler_args.max_swizzle_size,
            cluster_shape_mnk=self.cluster_shape_mnk,
            tile_count_semaphore=scheduler_args.tile_count_semaphore,
            batch_idx_permute=scheduler_args.batch_idx_permute,
            persistence_mode=persistence_mode,
            run_j_tiles=const_expr(getattr(scheduler_args, "run_j_tiles", 0)),
        )

    def _extra_smem_struct(self):
        """Optional extra SMEM struct field for a subclass (default: 0-size dummy).

        Returned type is spliced into ``SharedStorage`` as the ``decoupled`` field. The
        default 0-size ``MemRange`` adds no SMEM (byte-identical footprint). A subclass that
        needs CTA-shared SMEM across warpgroups (e.g. the decoupled peer-store ring's
        full/empty mbarriers + per-slot metadata) overrides this with a real ``cute.struct``,
        flag-gated so the default stays empty."""
        return cute.struct.MemRange[cutlass.Int32, 0]

    def _init_extra_smem(self, storage, warp_idx) -> None:
        """Init the subclass extra SMEM (mbarriers) in the pre-warp-split window (default no-op).

        Called once, just before ``pipeline_init_arrive``, so the cluster-wide
        ``pipeline_init_wait`` orders the init before any warp (producer or the extra consumer
        warpgroup) uses it. A subclass inits its mbarriers here (one thread, then fence)."""
        pass

    def _num_extra_warpgroups(self) -> int:
        """Number of EXTRA (non-MMA, non-AB-load) warpgroups to allocate in the CTA.

        Default 0 → the block is ``(mma_warp_groups + 1) * 128`` threads exactly as
        before (byte-identical). A subclass overrides this (flag-gated) to add a
        dedicated consumer warpgroup, e.g. for the decoupled peer-store pipeline. Pure
        extract-method hook: the default return reproduces today's behavior."""
        return 0

    def _drain_on_producer_wg(self) -> bool:
        """Route the extra consumer-role work onto the PRODUCER warpgroup's spare warps
        (``ab_load_warp_id + num_ab_load_warps .. (mma_warp_groups+1)*4``, e.g. warps 9-11) instead of
        a dedicated extra warpgroup — used when ``_num_extra_warpgroups()`` returns 0 but a subclass
        still wants a consumer role WITHOUT growing the block (keeps a smaller thread count so the
        reqntid-derived register cap stays high enough for a wide-tile WGMMA). Default ``False`` -> the
        consumer role only runs when a dedicated extra warpgroup exists (byte-identical)."""
        return False

    def _persistent_grid_adjust(self, grid):
        """Hook to adjust the persistent launch grid after ``get_grid_shape`` (default IDENTITY).

        Called from ``__call__`` right after the grid is computed, before ``.launch(grid=grid)``. The
        default returns ``grid`` UNCHANGED → byte-identical (the call inlines to identity in the trace).
        A subclass overrides this (flag-gated, const_expr-elided when off) to, e.g., reduce the z-extent
        (dim 2, the per-CTA count) for a concentration lever — the persistent scheduler + any strided
        consumer both read the RUNTIME ``grid_dim()[2]`` and terminate on the baked problem size, so a
        reduced grid stays coverage-preserving (each launched CTA does proportionally more work)."""
        return grid

    def _extra_wg_reg_adjust(self, n_extra_wg: int) -> None:
        """Hook to adjust the warpgroup register split after the extra-WG carve (default no-op).

        Called from ``__call__`` right after the ``n_extra_wg > 0`` carve that shrinks
        ``num_regs_mma`` to fit the steady-state budget. The default does NOTHING, so every
        extra-WG user keeps the carve's value -> byte-identical. A subclass overrides this to
        handle a *realloc-reachability* hazard the steady-state carve does not cover: the
        Hopper ``setmaxnreg.inc`` (MMA warps grow to ``num_regs_mma``) blocks until the CTA
        register pool has enough free registers, and the pool is fed only by the load/consumer
        warps' ``setmaxnreg.dec``. When the extra consumer WG is present but does NO work (idle,
        low register pressure), ptxas keeps the asymmetric realloc and the MMA ``.inc`` target
        is unreachable from the kernel's actual (low) REGCOUNT baseline -> ``.inc`` deadlocks.
        The override fixes it by setting ``self._skip_warpgroup_reg_realloc`` so the paired
        ``setmaxregister_decrease``/``setmaxregister_increase`` (above + in the MMA block) are
        elided -> ptxas uses one uniform allocation -> no pool dance -> no unreachable ``.inc``."""
        return

    def consumer_warpgroup_role(self, warp_idx, storage, epilogue_params, tile_sched_params):
        """Hook for the EXTRA consumer warpgroup's per-CTA work (default: no-op).

        Called once near the top of the kernel for every warp whose ``warp_idx`` lies in
        the extra-warpgroup range ``[(mma_warp_groups+1)*4, threads_per_cta//32)`` — i.e.
        the warps allocated by :meth:`_num_extra_warpgroups`. The default does nothing, so
        with ``_num_extra_warpgroups()==0`` (the parent default) this hook is never reached
        and the kernel is byte-identical. A subclass that adds a consumer warpgroup overrides
        this to run the consumer loop (e.g. drain a local-GMEM ring to a peer). It must NOT
        touch the MMA warpgroup's epilogue barrier / pipelines (the extra warpgroup does not
        participate in them)."""
        pass

    def make_sched_pipeline(
        self, cluster_layout_mnk: cute.Layout, sched_pipeline_mbar_ptr: cute.Pointer
    ):
        """Construct the pipeline the tile scheduler broadcasts work tiles over.

        One CTA computes the next work tile and pushes it into its peers' SMEM; this pipeline is
        what makes that push safe. The consumer arrive count must equal the number of warps that
        actually wait on it, or the barrier never completes (too high) or is released early (too
        low) -- both are hangs or corrupted tile indices, not diagnostics. Under pingpong only ONE
        of the two MMA warpgroups waits per round, hence the ``1`` rather than ``mma_warp_groups``.

        Args:
            cluster_layout_mnk: The cluster layout. Its size multiplies the arrive count, since
                every CTA in the cluster contributes.
            sched_pipeline_mbar_ptr: SMEM pointer to the barrier array, which must have room for
                ``sched_stage * 2`` ``Int64`` mbarriers and be initialised before first use.

        Returns:
            A :class:`PipelineAsync` with ``sched_stage`` stages, created with ``defer_sync=True``
            -- the caller is responsible for the cluster-wide init barrier.
        """
        sched_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        cluster_size = cute.size(cluster_layout_mnk)
        # Each warp contributes 1 to the arrive count.
        consumer_arrive_cnt = (
            (1 if self.pingpong else self.mma_warp_groups) * 4 + self.num_ab_load_warps
        ) * cluster_size
        sched_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, consumer_arrive_cnt
        )
        return pipeline.PipelineAsync.create(
            barrier_storage=sched_pipeline_mbar_ptr,
            num_stages=self.sched_stage,
            producer_group=sched_pipeline_producer_group,
            consumer_group=sched_pipeline_consumer_group,
            # If there's cluster, the consumers must arrive at the mbar of CTA 0 in the cluster.
            consumer_mask=None if const_expr(cluster_size == 1) else 0,
            defer_sync=True,
        )

    @classmethod
    def _compute_stages(
        cls,
        cta_tile_shape_mnk: Tuple[int, int, int],
        epi_tile: Tuple[int, int],
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        d_dtype: Optional[Type[cutlass.Numeric]],
        c_dtype: Optional[Type[cutlass.Numeric]],
        epilogue_args: "GemmSm90.EpilogueArguments",
        smem_capacity: int,
        occupancy: int,
    ) -> Tuple[int, int]:
        """Computes the number of stages for A/B/C operands based on heuristics.

        :param cta_tile_shape_mnk: The shape (M, N, K) of the CTA tile.
        :type cta_tile_shape_mnk: Tuple[int, int, int]
        :param a_dtype: Data type of operand A.
        :type a_dtype: type[cutlass.Numeric]
        :param b_dtype: Data type of operand B.
        :type b_dtype: type[cutlass.Numeric]
        :param smem_capacity: Total available shared memory capacity in bytes.
        :type smem_capacity: int
        :param occupancy: Target number of CTAs per SM (occupancy).
        :type occupancy: int

        :return: A tuple containing the computed number of stages for:
                 (A/B operand stages, epilogue stages)
        :rtype: Tuple[int, int]
        """

        epi_stage = 4 if epi_tile[1] <= 16 else 2
        d_bytes_per_stage = cute.size(epi_tile) * d_dtype.width // 8 if d_dtype is not None else 0
        epi_bytes_per_stage = d_bytes_per_stage + cls.epi_smem_bytes_per_stage(
            epilogue_args, cta_tile_shape_mnk, epi_tile
        )
        epi_bytes = epi_bytes_per_stage * epi_stage
        epi_c_stage = 0 if c_dtype is None else (4 if epi_tile[1] <= 16 else 2)
        if c_dtype is not None:
            epi_bytes += cute.size(epi_tile) * c_dtype.width // 8 * epi_c_stage

        a_shape = cute.slice_(cta_tile_shape_mnk, (None, 0, None))
        b_shape = cute.slice_(cta_tile_shape_mnk, (0, None, None))
        ab_bytes_per_stage = (
            cute.size(a_shape) * a_dtype.width // 8 + cute.size(b_shape) * b_dtype.width // 8
        )
        mbar_helpers_bytes = 1024

        remaining_bytes = smem_capacity // occupancy - mbar_helpers_bytes - epi_bytes
        ab_stage = remaining_bytes // ab_bytes_per_stage

        # Refine epilogue stages:
        # Calculate remaining smem after allocating for A/B stages and reserved bytes
        # Add remaining unused smem to epilogue
        if epi_bytes_per_stage > 0:
            epi_stage += (remaining_bytes - ab_bytes_per_stage * ab_stage) // epi_bytes_per_stage
        return ab_stage, epi_stage, epi_c_stage

    @staticmethod
    def _sm90_compute_tile_shape_or_override(
        cta_tile_shape_mnk: Tuple[int, int, int],
        atom_layout_mnk: Tuple[int, int, int],
        element_type: Optional[Type[cutlass.Numeric]] = None,
        epi_tile_override: Tuple[int, int] | None = None,
    ) -> Tuple[int, int]:
        """Compute the epilogue tile shape or use override if provided.

        :param cta_tile_shape_mnk: CTA tile shape (M,N,K)
        :type cta_tile_shape_mnk: Tuple[int, int, int]
        :param element_type: Data type of elements
        :type element_type: type[cutlass.Numeric]
        :param is_cooperative: Whether to use cooperative approach
        :type is_cooperative: bool
        :param epi_tile_override: Optional override for epilogue tile shape
        :type epi_tile_override: Tuple[int, int] or None

        :return: Computed epilogue tile shape
        :rtype: Tuple[int, int]
        """
        if epi_tile_override is not None:
            return epi_tile_override
        if cta_tile_shape_mnk[0] % 128 == 0 and atom_layout_mnk[0] > 1:
            tile_m = math.gcd(128, cute.size(cta_tile_shape_mnk, mode=[0]))
            tile_n = math.gcd(32, cute.size(cta_tile_shape_mnk, mode=[1]))
        elif cta_tile_shape_mnk[0] % 192 == 0 and atom_layout_mnk[0] > 1:
            tile_m = math.gcd(192, cute.size(cta_tile_shape_mnk, mode=[0]))
            tile_n = math.gcd(32, cute.size(cta_tile_shape_mnk, mode=[1]))
        else:
            # In the case of tile shape 128 x N but atom_layout 1 x 2, we need to set
            # epi_tile_m = 64. If epi_tile_m = 128, the epilogue would iterate along the
            # M dimension first, then move to the N dimension. But the accumulator in registers
            # iterate along the N dimension first, then move to the M dimension.
            # We could change the epilogue to accommodate this,
            # but it's easier to just set epi_tile_m = 64.
            n_perf = 64 if element_type is not None and element_type.width == 8 else 32
            tile_m = math.gcd(64, cute.size(cta_tile_shape_mnk, mode=[0]))
            tile_n = math.gcd(n_perf, cute.size(cta_tile_shape_mnk, mode=[1]))
        return (tile_m, tile_n)

    @staticmethod
    def _make_smem_layouts(
        cta_tile_shape_mnk: Tuple[int, int, int],
        epi_tile: Tuple[int, int],
        a_dtype: Type[cutlass.Numeric],
        a_layout: LayoutEnum,
        b_dtype: Type[cutlass.Numeric],
        b_layout: LayoutEnum,
        ab_stage: int,
        d_dtype: Optional[Type[cutlass.Numeric]],
        d_layout: LayoutEnum,
        epi_stage: int,
        c_dtype: Optional[Type[cutlass.Numeric]],
        c_layout: Optional[LayoutEnum],
        epi_c_stage: int,
    ) -> Tuple[
        cute.ComposedLayout, cute.ComposedLayout, cute.ComposedLayout, Optional[cute.ComposedLayout]
    ]:
        """Create shared memory layouts for A, B, and C tensors.

        :param cta_tile_shape_mnk: CTA tile shape (M,N,K)
        :type cta_tile_shape_mnk: Tuple[int, int, int]
        :param epi_tile: Epilogue tile shape
        :type epi_tile: Tuple[int, int]
        :param a_dtype: Data type for matrix A
        :type a_dtype: type[cutlass.Numeric]
        :param a_layout: Layout enum for matrix A
        :type a_layout: LayoutEnum
        :param b_dtype: Data type for matrix B
        :type b_dtype: type[cutlass.Numeric]
        :param b_layout: Layout enum for matrix B
        :type b_layout: LayoutEnum
        :param ab_stage: Number of stages for A/B tensors
        :type ab_stage: int
        :param d_dtype: Data type for output matrix D
        :type d_dtype: type[cutlass.Numeric]
        :param d_layout: Layout enum for the output matrix C
        :type d_layout: LayoutEnum
        :param epi_stage: Number of epilogue stages
        :type epi_stage: int

        :return: Tuple of shared memory layouts for A, B, and C
        :rtype: Tuple[cute.ComposedLayout, cute.ComposedLayout, cute.ComposedLayout]
        """
        a_smem_shape = cute.slice_(cta_tile_shape_mnk, (None, 0, None))

        a_is_k_major = a_layout.sm90_mma_major_mode() == warpgroup.OperandMajorMode.K
        b_is_k_major = b_layout.sm90_mma_major_mode() == warpgroup.OperandMajorMode.K
        a_major_mode_size = cta_tile_shape_mnk[2 if a_is_k_major else 0]
        a_smem_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(a_layout, a_dtype, a_major_mode_size),
            a_dtype,
        )
        a_smem_layout_staged = cute.tile_to_shape(
            a_smem_layout_atom,
            cute.append(a_smem_shape, ab_stage),
            order=(0, 1, 2) if a_is_k_major else (1, 0, 2),
        )

        b_smem_shape = cute.slice_(cta_tile_shape_mnk, (0, None, None))

        b_major_mode_size = cta_tile_shape_mnk[2 if b_is_k_major else 1]
        b_smem_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(b_layout, b_dtype, b_major_mode_size),
            b_dtype,
        )
        b_smem_layout_staged = cute.tile_to_shape(
            b_smem_layout_atom,
            cute.append(b_smem_shape, ab_stage),
            order=(0, 1, 2) if b_is_k_major else (1, 0, 2),
        )

        epi_smem_layout_staged = None
        if d_dtype is not None:
            epi_smem_layout_staged = fold_cp_ops_sm90_utils.make_smem_layout_epi(
                d_dtype, d_layout, epi_tile, epi_stage
            )

        epi_c_smem_layout_staged = None
        if c_dtype is not None:
            assert c_layout is not None
            epi_c_smem_layout_staged = fold_cp_ops_sm90_utils.make_smem_layout_epi(
                c_dtype, c_layout, epi_tile, epi_c_stage
            )

        return (
            a_smem_layout_staged,
            b_smem_layout_staged,
            epi_smem_layout_staged,
            epi_c_smem_layout_staged,
        )

    @staticmethod
    def is_valid_dtypes(
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        acc_dtype: Type[cutlass.Numeric],
        d_dtype: Optional[Type[cutlass.Numeric]],
        a_major: str,
        b_major: str,
    ) -> bool:
        """
        Check if the dtypes are valid

        :param a_dtype: The data type of tensor A
        :type a_dtype: Type[cutlass.Numeric]
        :param b_dtype: The data type of tensor B
        :type b_dtype: Type[cutlass.Numeric]
        :param acc_dtype: The data type of the accumulator
        :type acc_dtype: Type[cutlass.Numeric]
        :param d_dtype: The data type of the output tensor
        :type d_dtype: Type[cutlass.Numeric]
        :param a_major: major mode of tensor A
        :type a_major: str
        :param b_major: major mode of tensor B
        :type b_major: str

        :return: True if the dtypes are valid, False otherwise
        :rtype: bool
        """
        is_valid = True
        if a_dtype not in {Float16, cutlass.BFloat16, cutlass.Float8E4M3FN, cutlass.Float8E5M2}:
            is_valid = False
        # tested b_dtype
        if b_dtype not in {Float16, cutlass.BFloat16, cutlass.Float8E4M3FN, cutlass.Float8E5M2}:
            is_valid = False
        if acc_dtype not in {Float32, Float16}:
            is_valid = False
        # tested d_dtype
        if d_dtype not in {
            None,
            Float32,
            Float16,
            cutlass.BFloat16,
            cutlass.Float8E4M3FN,
            cutlass.Float8E5M2,
        }:
            is_valid = False
        # make sure a_dtype == b_dtype for Float16
        if a_dtype.width == 16 and a_dtype != b_dtype:
            is_valid = False
        # make sure a_dtype.width == b_dtype.width (i.e, Float8E4M3FN or Float8E5M2)
        if a_dtype.width != b_dtype.width:
            is_valid = False

        # for Float8 types, this implementation only supports k-major layout
        if (a_dtype.width == 8 and a_major != "k") or (b_dtype.width == 8 and b_major != "k"):
            is_valid = False
        return is_valid

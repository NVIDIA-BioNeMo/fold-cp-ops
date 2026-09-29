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
# Copyright (c) 2025, Wentao Guo, Tri Dao.
# This file has been modified by NVIDIA CORPORATION & AFFILIATES.

"""The SM90 GEMM whose epilogue MULTIPLIES a full per-element operand into the accumulator.

`gemm_hadamard()` computes ``D = (alpha * (A @ B^T) + bias) ⊙ C`` in one launch. ``C`` is a full
``(l, m, n)`` tensor with D's shape -- not a broadcast vector -- and ⊙ is the elementwise
(Hadamard) product. In the TriMul workflow this is reference ops 9 and 12 fused: the output
projection ``p_out = trin @ p_out_w^T + p_out_b`` immediately multiplied by the trailing output
gate ``gate3``.

**The kernel is the stock GEMM with one operator changed.** `GemmHadamardEpiMixin` differs from
`GemmDefaultEpiMixin` in exactly one expression -- ``rD += beta * C`` becomes ``rD *= C`` -- and
reuses the SAME per-element C path: the same TMA load into the epilogue's C pipeline, the same
register tiling, the same predication. Nothing in the mainloop, the scheduler or the store is
touched, and no new epilogue op is declared. That is why this module is ~1 function of arithmetic
and not a second GEMM.

**Every term is `const_expr`-gated, so the trace collapses.** With no bias and no C, the emitted
epilogue is ``load -> alpha -> store``, which is what `GemmDefaultSm90` emits for the same
configuration -- byte-identical, not merely equivalent. `tests/kernels/test_gemm_hadamard.py`
asserts that at the CUBIN level rather than assuming it, because the claim is about which
instructions the compiler chose to emit and no source reading can settle that.

CUTLASS expresses the same fusion as the EVT tree
``Sm90EVT<Sm90Compute<cutlass::multiplies>, Sm90AccFetch, Sm90AuxLoad>``
(``sm90_visitor_{load,compute}_tma_warpspecialized.hpp``); there is no named single-op
"HadamardMultiply", which is why this is a mixin here rather than a configuration flag.

Three deliberate differences from the module this was ported from (``main:kernels/gemm_elem.py``):

1. **The contraction convention is this tree's.** B is ``(l, n, k)`` and the product is
   ``A @ B^T``, matching `fold_cp_ops.kernels.gemm.gemm`. The upstream took B as ``(l, k, n)`` at
   its public entry and transposed it one layer down; a caller who transliterates the transposes
   instead of re-deriving them from the einsum gets a shape error at best.
2. **There is no allocating wrapper and no `swap_ab` knob.** Upstream had both, layered on a
   `gemm_interface` front door this tree does not have. `gemm_hadamard()` writes in place into a
   preallocated D, exactly as `gemm()` does. A caller that wants the swap transposes the problem
   itself and feeds the ``(n,)`` bias as `colvec_bias` -- which is why BOTH broadcast vectors are
   exposed here even though the fused TriMul call site only uses the row one.
3. **`add_to_output`, `rounding_mode`, `sr_seed` and `run_j_tiles` are not exposed.** Upstream
   pinned them (False / RN / None / 0) and so does this module. They are epilogue knobs of the
   general GEMM, and every one of them is a distinct compiled kernel; adding a knob nothing here
   reaches would widen the compile-key space without widening what is tested.
"""

from typing import Optional

from torch import Tensor

import cutlass
import cutlass.cute as cute
from cutlass import Float32, const_expr
from cutlass.cute.runtime import make_ptr

from fold_cp_ops._internal.arch import (
    check_arch_supported,
    get_max_active_clusters,
    require_sm90,
)
from fold_cp_ops._internal.autotune import autotune
from fold_cp_ops._internal.cache_utils import jit_cache
from fold_cp_ops._internal.compile_time.compile_utils import make_fake_tensor as fake_tensor
from fold_cp_ops._internal.compile_time.cute_dsl_utils import torch2cute_dtype_map
from fold_cp_ops._internal.epi_default import GemmDefaultEpiMixin
from fold_cp_ops._internal.gemm_tvm_ffi_utils import (
    check_broadcast_alignment,
    compile_gemm_kernel,
    describe_operands,
    make_fake_gemm_tensors,
    make_fake_scheduler_args,
    make_scheduler_args,
    perm3d,
)
from fold_cp_ops._internal.rounding import RoundingMode
from fold_cp_ops._internal.tensor_contract import check_tensor
import fold_cp_ops._internal.utils as utils
from fold_cp_ops.kernels.gemm import (
    _raise_operand_shape_mismatch,
    gemm_config_is_valid,
    gemm_tuning_space,
)
from fold_cp_ops.kernels.gemm_sm90 import check_fp8_major, GemmSm90


class GemmHadamardEpiMixin(GemmDefaultEpiMixin):
    """The default GEMM epilogue with the C operand MULTIPLIED in rather than added.

    Purpose
        Fuse ``⊙ C`` -- a full per-element gate -- into the GEMM epilogue, so the gate is applied
        while the accumulator is still in registers instead of costing a separate read-modify-write
        pass over an ``(l, m, n)`` tensor.

    Functionality and semantics
        Computes ``D = (alpha * acc + rowvec + colvec) ⊙ C`` per subtile, in that order, and the
        order is load-bearing rather than incidental:

        * ``alpha`` scales the accumulator only, as in the parent.
        * The biases are added **on the register tensor's VIEW** (``tRS_rD[i] += vec[i]``), which
          means the accumulator is stored back before the adds and reloaded after them. That looks
          like a wasted round trip and is not: it is exactly the sequence the parent epilogue emits
          for its own bias adds, so the additions round bit-for-bit the way `GemmDefaultSm90`'s do.
          Rewriting it as a register-vector add would be a *different function* at the last ulp.
        * ``C`` multiplies last, after every additive term. A gate applied before the bias would
          scale the bias too, which is not what ops 9/12 compute.

        Everything is `const_expr`-gated on PRESENCE, not on a runtime flag: with both biases and C
        absent the whole block disappears and the emitted trace is the parent's unbiased one.
        Presence is therefore part of the compile cache key -- see `_compile_gemm_hadamard`.

        ``beta`` is inherited from the parent's ``_epi_ops`` and is never read here. It stays
        declared so this class's generated ``EpilogueParams`` keeps the parent's field layout; the
        host always passes ``beta=None``, so no instruction is emitted for it.

    Input requirements
        These are the requirements this mixin adds on top of `GemmDefaultEpiMixin`'s:

        * ``tRS_rC`` -- when present, must be the per-element C fragment for THIS subtile, with the
          same epilogue partition as ``tRS_rD``. It is multiplied elementwise with no broadcast, so
          a fragment from a differently-partitioned tensor silently permutes the gate across the
          tile rather than failing.
        * ``mRowVecBroadcast`` / ``mColVecBroadcast`` -- at most one of the two is what the fused
          TriMul call passes, but both are supported and both add before the multiply. Their
          register tensors must share ``tRS_rD``'s partition, which is what `RowVecLoad` /
          `ColVecLoad` guarantee; nothing checks it here.
        * ``params.alpha`` -- absent, a baked `Float32`, or a GMEM pointer. The CHOICE is
          compile-time; a launch that passes a pointer to a kernel compiled without one is an ABI
          mismatch, not a value change.

    Raises:
        Nothing at this level. Every argument fault is refused by `gemm_hadamard()` at the host
        entry, before tracing; a violation that reaches here is a wrong answer rather than an error.
    """

    @cute.jit
    def epi_visit_subtile(
        self,
        params,
        epi_loop_tensors,
        tRS_rD: cute.Tensor,
        tRS_rC: Optional[cute.Tensor] = None,
        epi_gate3: cutlass.Constexpr = False,
    ) -> Optional[cute.Tensor]:
        """Combine one epilogue subtile: scale by alpha, add the biases, then multiply by C.

        Args:
            params: This kernel's ``EpilogueParams``. Read for ``alpha`` only; ``beta`` is
                inherited but unused, and reading it would reintroduce the additive combine.
            epi_loop_tensors: The per-subtile loaded values keyed by ``_epi_ops`` name. Entries for
                absent terms are None, which is what the `const_expr` guards test.
            tRS_rD: The accumulator fragment for this subtile. **Updated in place** and also the
                source of the value being combined -- it is both input and output, so a caller that
                needs the pre-epilogue accumulator must have copied it already.
            tRS_rC: The per-element C (gate) fragment for this subtile, or None when the kernel was
                compiled without a C operand. Must carry ``tRS_rD``'s partition; see the class
                docstring.
            epi_gate3: Whether this is a fused TriMul output-gate invocation. Unused -- this kernel
                takes its gate as the C OPERAND, which is the point of the module.

        Returns:
            None. Like the parent, this epilogue produces no second (postact) output; a gated
            subclass would return the pre-activation fragment instead.
        """
        tDrRowVec = epi_loop_tensors["mRowVecBroadcast"]
        tDrColVec = epi_loop_tensors["mColVecBroadcast"]
        rD = tRS_rD.load()
        # Optional alpha scaling of the accumulator (free; matches Linear with a scale).
        if const_expr(hasattr(params, "alpha") and params.alpha is not None):
            alpha = utils.load_scalar_or_pointer(params.alpha)
            rD *= alpha
        # Optional bias add (per output column N, broadcast over rows M) BEFORE the Hadamard ⊙C.
        # The bias rowvec/colvec register tensors share tRS_rD's epilogue partition, so they index
        # the tensor VIEW (tRS_rD[i]), matching the parent default epilogue's adds bit-for-bit;
        # store the alpha-scaled acc first, add the bias on the view, then reload for the multiply.
        # This whole block is const_expr-gated on bias presence, so a None bias leaves the unbiased
        # trace (single load -> alpha -> ⊙C -> store) byte-identical.  Under a swap_ab-style
        # transposed call the (N,) bias arrives as a (M,) col vec (ColVecLoad) instead of a (N,) row
        # vec (RowVecLoad), which is why both are handled here rather than only the row form.
        if const_expr(tDrRowVec is not None or tDrColVec is not None):
            tRS_rD.store(rD)
            if const_expr(tDrRowVec is not None):
                for i in cutlass.range(cute.size(tDrRowVec), unroll_full=True):
                    tRS_rD[i] += tDrRowVec[i]
            if const_expr(tDrColVec is not None):
                for i in cutlass.range(cute.size(tDrColVec), unroll_full=True):
                    tRS_rD[i] += tDrColVec[i]
            rD = tRS_rD.load()
        # Hadamard: multiply by the per-element C tile (the gate).  Same C-load path the default
        # epilogue uses for beta*C -- only the operator differs.
        if const_expr(tRS_rC is not None):
            rD *= tRS_rC.load().to(tRS_rD.element_type)
        tRS_rD.store(rD)
        return None


class GemmHadamardSm90(GemmHadamardEpiMixin, GemmSm90):
    """`GemmSm90`'s mainloop under the Hadamard epilogue: ``D = (alpha * A @ B^T + bias) ⊙ C``.

    Everything this class does comes from its two bases, and the ORDER is the content: the mixin
    must come first so its ``epi_*`` hooks override the no-op defaults `GemmSm90` declares. Reversed,
    the class compiles and silently computes the plain GEMM.

    Construct it the way `GemmSm90.__init__` documents -- the mixin adds no constructor arguments,
    only epilogue ones, and those travel in ``EpilogueArguments`` at call time. `gemm_hadamard()`
    below is the supported way to reach it; constructing it directly is for callers driving
    `cute.compile` themselves, which is what the cubin-identity test does.

    Upstream selected a ``GemmElemSm100`` when the device capability exceeded 9. `fold-cp-ops` is
    SM90-only, so there is nothing to select between and the branch is gone; the capability is
    checked at the host entry instead.
    """


@jit_cache
def _compile_gemm_hadamard(
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
    alpha_mode,
    rowvec_dtype,
    colvec_dtype,
    colvec_ndim,
    has_batch_idx_permute,
    device_capacity,
):
    """Compile one `GemmHadamardSm90` configuration against fake tensors. Cached on every argument.

    Every parameter is part of the `@jit_cache` key, and that is the point: two calls agreeing on
    all of them share a compiled artifact, and two differing in any of them must not. Shapes are
    deliberately absent -- they enter as `cute.sym_int()` symbols through `make_fake_gemm_tensors`,
    so one artifact serves every M/N/K/L and a new shape never forces a recompile.

    Args:
        a_dtype: Cutlass element type of A. 16-bit (fp16/bf16) or 8-bit (fp8 e4m3/e5m2); anything
            else has no SM90 WGMMA form and is refused at the host entry.
        b_dtype: Element type of B. Must match `a_dtype`'s WIDTH.
        d_dtype: Element type of the output D. May be fp32.
        c_dtype: Element type of the per-element C operand, or None to compile the ⊙C away
            entirely. **None is not "multiply by one"** -- the TMA load, the C pipeline and the
            multiply are all absent from the emitted kernel, which is what makes that trace
            byte-identical to the stock GEMM's.
        a_major: ``"m"`` or ``"k"`` -- which axis of A is contiguous. fp8 requires ``"k"``. Must
            match the REAL tensor's major; a mismatch compiles a kernel reading A transposed.
        b_major: ``"n"`` or ``"k"``. fp8 requires ``"k"``.
        d_major: ``"m"`` or ``"n"``.
        c_major: ``"m"``, ``"n"``, or None when C is absent.
        tile_shape_mn: ``(tile_M, tile_N)``. Validated by `GemmSm90.__init__`, which raises on a
            shape the WGMMA atom cannot be built for.
        cluster_shape_mnk: ``(cluster_M, cluster_N, 1)``. Product must be a power of two <= 8.
        pingpong: Whether to use the two-warpgroup ping-pong schedule. Requires `persistent`.
        persistent: Whether the grid is persistent (one wave, looping over tiles).
        is_dynamic_persistent: Whether tiles are handed out by a GMEM atomic rather than statically.
        alpha_mode: 0 absent (folded away) / 1 host scalar (baked in) / 2 device pointer. Three
            distinct kernels; collapsing them into one key would hand a caller the wrong one.
        rowvec_dtype: Element type of the ``(l, n)`` row bias, or None when absent.
        colvec_dtype: Element type of the ``(l, m)`` column bias, or None when absent.
        colvec_ndim: 2 for an ``(l, m)`` column bias, 0 for absent. **Only those two.** A 1-D
            ``(m,)`` column bias is refused at the host entry rather than accepted here: this
            tree's `ColVecLoad` slices its batch as ``param[batch_idx, None]``, so a flat vector
            would index along M and silently broadcast the wrong elements.
        has_batch_idx_permute: Whether the scheduler reorders batches.
        device_capacity: ``(major, minor)``, re-checked below rather than trusted.

    Returns:
        The compiled TVM-FFI entry, callable as ``fn(A, B, D, C, epi_args, scheduler_args)``.

    Raises:
        UnsupportedArchError: If `device_capacity` is not SM90.
        ValueError: Propagated from `GemmSm90` for a tile geometry or operand layout it refuses.
    """
    # SM90-only: upstream picked GemmElemSm100 when device_capacity[0] > 9.  `gemm_hadamard()`
    # rejects a non-SM90 device before it ever gets here, and this entry is @jit_cache'd on
    # device_capacity, so re-check rather than trust the key.
    check_arch_supported(device_capacity)
    GemmCls = GemmHadamardSm90
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

    # Fake bias operand(s) so the biased variant compiles under a distinct cache key (rowvec_dtype /
    # colvec_dtype / colvec_ndim are part of the @jit_cache key). A None dtype gives a None tensor,
    # which is what the epilogue's const_expr guard prunes on.
    mRowVec = fake_tensor(rowvec_dtype, (l, n), leading_dim=1, divisibility=4)
    if colvec_ndim == 2:
        mColVec = fake_tensor(colvec_dtype, (l, m), leading_dim=1, divisibility=4)
    else:
        mColVec = None

    epi_args = GemmCls.EpilogueArguments(
        alpha=fake_scalar(alpha_mode),
        beta=None,
        mRowVecBroadcast=mRowVec,
        mColVecBroadcast=mColVec,
        add_to_output=False,
        rounding_mode=RoundingMode.RN,
        sr_seed=None,
    )
    scheduler_args = make_fake_scheduler_args(
        (is_dynamic_persistent and device_capacity[0] == 9),
        has_batch_idx_permute,
        l,
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


#: The pool the tuned path sweeps. A FRESH `AxisSpace` rather than `gemm.GEMM_TUNING_SPACE`: the
#: axes are the same because the mainloop and the scheduler are the same, but the two entries are
#: separate tuning subjects with separate result caches, and sharing one object would make an
#: inspection or a subset taken on one visible to the other.
#:
#: This is the SUBSTITUTION the bring-back required. Upstream imported `default_config` and
#: `prune_invalid_gemm_configs` from a `kernels/gemm_interface.py` -- a second GEMM front door,
#: 822 lines, for a GEMM this tree already ships. `gemm_tuning_space()` / `gemm_config_is_valid()`
#: are this tree's statements of the same two facts (what to sweep, what cannot run), so they are
#: used directly instead.
#:
#: ``swap_ab=True`` is passed because this entry IMPLEMENTS the swap (see the body). It doubles the
#: pool 22 -> 44, which is exactly `main`'s SM90 pool for the same kernel, and it is what closes the
#: measured tuned-path gap against it.
GEMM_HADAMARD_TUNING_SPACE = gemm_tuning_space(swap_ab=True)


@autotune(
    space=GEMM_HADAMARD_TUNING_SPACE,
    key=["persistent", "is_dynamic_persistent", "max_swizzle_size"],
    validity=gemm_config_is_valid,
    gate="do_autotune",
)
def gemm_hadamard(
    A: Tensor,  # (l, m, k)
    B: Tensor,  # (l, n, k)
    D: Tensor,  # (l, m, n)
    C: Tensor,  # (l, m, n) -- REQUIRED: the per-element gate
    tile_count_semaphore: Optional[Tensor],  # (1,)
    tile_M: Optional[int] = None,
    tile_N: Optional[int] = None,
    cluster_M: int = 1,
    cluster_N: int = 1,
    pingpong: bool = False,
    swap_ab: bool = False,
    persistent: bool = True,
    is_dynamic_persistent: bool = False,
    max_swizzle_size: int = 8,
    rowvec_bias: Optional[Tensor] = None,  # (l, n)
    colvec_bias: Optional[Tensor] = None,  # (l, m)
    alpha: float | Tensor = 1.0,
    batch_idx_permute: Optional[Tensor] = None,  # (l,) permutation of batch indices for scheduler
    *,
    do_autotune: bool = False,
) -> None:
    """Batched GEMM with a fused per-element gate: ``D = (alpha * (A @ B^T) + bias) ⊙ C``.

    Note the transpose: B is stored ``(l, n, k)``, so the contraction is over the LAST axis of both
    operands. That is the layout WGMMA wants and it is not negotiable through this entry -- pass B
    already in that shape rather than transposing a ``(l, k, n)`` tensor, which would give a
    non-unit stride the TMA descriptor cannot use.

    ``C`` is a FULL per-element tensor with D's shape, not a broadcast vector, and it is
    **required**: a kernel with no C is the plain `gemm()`, and accepting ``C=None`` here would let
    a caller who forgot the gate get a silently ungated result. It is multiplied in LAST, after
    alpha and after both biases.

    Args:
        A: ``(l, m, k)``. Must be on a CUDA device -- this is the device whose capability is gated
            on. 16- or 8-bit; fp32 has no Hopper MMA form and is refused by name.
        B: ``(l, n, k)``. Same width as A.
        D: The output, ``(l, m, n)``. **Written in place**; it is an argument rather than a return
            value so the caller controls the allocation. Must be preallocated with the right shape
            -- nothing resizes it.
        C: The per-element gate, ``(l, m, n)``. Must have D's shape; broadcasting is NOT supported,
            and a shape mismatch is refused here rather than surfacing as a TMA descriptor built
            against the wrong extents. May be a different dtype from D (it is converted to fp32
            before the multiply), and may be m- or n-major independently of D.
        tile_count_semaphore: A zeroed ``int32`` tensor of shape ``(1,)``. Required when
            `is_dynamic_persistent`, ignored otherwise. Must be **zeroed before every launch**: it
            is an atomic ticket counter, and a stale value makes the grid skip tiles, which is a
            silently truncated output.
        tile_M: CTA tile M. One of 64/128/192/256/320 (64/128/192 with `pingpong`). Required unless
            `do_autotune`, whose sweep supplies it; the None default exists only for that.
        tile_N: CTA tile N. Divisible by 16 and <= 256, or by 32 and <= 512; narrower limits at
            `tile_M` 192/320 and under `pingpong`. `GemmSm90.__init__` raises on the rest.
        cluster_M: Threadblock-cluster extent along M. Power of two; ``cluster_M * cluster_N <= 8``.
        cluster_N: Cluster extent along N. Power of two; ``cluster_M * cluster_N <= 8``.
        pingpong: Use the two-warpgroup ping-pong schedule, which overlaps one warpgroup's epilogue
            with the other's mainloop. Requires ``persistent=True``.
        swap_ab: Compute the TRANSPOSED problem -- run the mainloop as ``B @ A^T`` and store into a
            transposed view of `D` -- instead of ``A @ B^T``. **The result in `D` is identical
            either way**; this is a pure performance knob, because exchanging the operands exchanges
            which one feeds the WGMMA A-fragment and transposes the tile scheduler's rasterization
            over the output. It is a HOST-SIDE permutation only: no kernel variant, no second
            compiled body, no new epilogue op. `tile_M` then applies to the output's N extent and
            `tile_N` to its M extent, which is why the tuner sweeps the tiles and this jointly.
            The two broadcast vectors exchange roles with the axes (`rowvec_bias` becomes the column
            vector of the transposed output and vice versa), so a caller who passes one need not
            think about it -- but a caller who pins ``swap_ab=True`` by hand must still pass the
            vectors in their UNSWAPPED spelling, i.e. `rowvec_bias` stays the ``(l, n)`` one.
        persistent: Launch one resident wave that loops over work tiles, instead of one CTA per
            tile.
        is_dynamic_persistent: Hand out work tiles through the GMEM atomic in
            `tile_count_semaphore` rather than statically. Helps when tiles are uneven; costs an
            atomic per tile.
        max_swizzle_size: Tile-scheduler rasterization swizzle width, for L2 locality. Purely a
            performance knob; it never changes the result.
        rowvec_bias: Optional ``(l, n)`` vector added to every row BEFORE the ⊙C. Its row pitch (N)
            must be a multiple of 4 elements -- the broadcast load is 4-element vectorized.
        colvec_bias: Optional ``(l, m)`` vector added to every column, likewise before the ⊙C.
            **Must be 2-D**: a flat ``(m,)`` vector is refused, because `ColVecLoad` slices its
            batch as ``param[batch_idx, None]`` and a 1-D tensor would index along M instead. Its
            row pitch is M, so passing one imposes a 4-element alignment on an otherwise free
            extent. This is the argument a caller doing a swap_ab-style transposed call feeds the
            ``(n,)`` bias to.
        alpha: Scale on ``A @ B^T``, applied before the biases and before the gate. A Python float
            is baked into the kernel at compile time; a device `Tensor` (1 element, fp32) is read
            at launch. ``1.0`` compiles the multiply away entirely.
        batch_idx_permute: Optional ``(l,)`` int32 permutation changing the order the scheduler
            visits batches. Affects scheduling only, never the result.
        do_autotune: Choose `tile_M`, `tile_N`, `cluster_M`, `cluster_N`, `pingpong` and `swap_ab`
            by MEASUREMENT instead of taking them from the caller. Keyword-only. Left False (the
            default) this function is exactly the fixed-config entry it reads as, and the gate costs
            one dict lookup. Set True and those six arguments must NOT be passed: pinning one while
            the tuner picks the rest would make the config that was measured and the config that ran
            differ, so it is refused rather than merged. The winning config is readable afterwards
            as ``gemm_hadamard.autotuner.best_config``.

    Returns:
        None. The result is in `D`.

    Raises:
        ValueError: For any front-door violation above -- an unsupported dtype on any of the six
            tensors, ``C=None`` or a C whose shape is not D's, an operand that is neither 16- nor
            8-bit, a non-contiguous or misaligned operand, a bias whose pitch is not 4-element
            aligned, `is_dynamic_persistent` without a semaphore, or a missing tile shape without
            `do_autotune`. Also raised by the tuner when a tuned knob is passed alongside
            ``do_autotune=True``, and propagated from `GemmSm90` for a tile geometry or fp8 operand
            layout it refuses.
        UnsupportedArchError: If the device is not SM90.
    """
    # The capability boundary comes FIRST, before any argument validation. On a device this package
    # has no kernels for, a complaint about a dtype or a stride is a distraction.
    device_capacity = require_sm90(A.device)

    # Front-door validation.  Every one of these was an `assert` upstream; they are `raise
    # ValueError` here because `python -O` strips asserts, and a stripped guard does not fail
    # loudly -- it produces a kernel reading the wrong strides, which is a wrong answer.
    if C is None:
        raise ValueError(
            "gemm_hadamard requires the per-element C (gate) tensor: it computes "
            "(alpha * A @ B^T + bias) ⊙ C, and with no C there is nothing to multiply by. A GEMM "
            "without the gate is fold_cp_ops.kernels.gemm.gemm -- call that instead of passing "
            "None here, so an omitted gate cannot read as an ungated result."
        )
    # C, D and the two broadcast vectors go through the SAME dtype check as the operands. Which of
    # these the kernel TMAs and which it broadcast-loads is not visible in this signature, so it
    # must not decide whether a caller mistake is named: without these, an unsupported dtype on any
    # of them surfaces as a bare `KeyError` from the `torch2cute_dtype_map[...]` lookup that builds
    # the compile key, naming a torch dtype and no argument.
    # A and B go through the membership check FIRST. Leaving them to `describe_operands` -- whose
    # first act is `torch2cute_dtype_map[t.dtype]` -- meant an unsupported operand dtype surfaced as
    # a bare `KeyError: torch.float64`, naming a torch dtype and no argument, and `KeyError` is not
    # an API-level error a caller can act on. The WIDTH check (16- or 8-bit) still rides along in
    # `describe_operands`, which needs the cutlass dtype that pass is already looking up.
    check_tensor("A", A)
    check_tensor("B", B)
    check_tensor("D", D)
    check_tensor("C", C)
    check_tensor("rowvec_bias", rowvec_bias, expect_shape=(None, D.shape[-1]))
    check_tensor("colvec_bias", colvec_bias, expect_shape=(None, D.shape[-2]))
    # The four operands' EXTENTS must agree with each other, which none of the checks above tests.
    # Shared with `gemm` rather than re-derived: the two entries take the same operands in the same
    # layout, so a second copy of this arithmetic would be a second place for it to drift.
    # `.shape` is a PROPERTY that builds a fresh `torch.Size` on every read, so the obvious spelling
    # of this test costs eight of them per call. Hoisting the three into locals is worth 5.4% of this
    # entry's whole host dispatch -- measured, and it is the entire cost of adding this check: the
    # hoisted form lands at 1.452x a bare GemmSm90 launch against 1.448x for deleting the check
    # outright. Two intuitive-looking alternatives are SLOWER and were measured before being
    # discarded: `.size(-1)` instead of `.shape[-1]` (1.547x), and replacing the C tuple compare with
    # per-dimension compares (1.489x). Do not "simplify" this back.
    _sa, _sb, _sd = A.shape, B.shape, D.shape
    if _sa[-1] != _sb[-1] or _sa[-2] != _sd[-2] or _sb[-2] != _sd[-1] or C.shape != _sd:
        _raise_operand_shape_mismatch(A, B, D, C)

    if tile_M is None or tile_N is None:
        raise ValueError(
            "tile_M and tile_N are required. They default to None only so that `do_autotune=True` "
            "can supply them -- the tuner refuses a caller who also pins one, because the measured "
            "winner and the executed kernel would then differ. Either pass a tile shape, or pass "
            "do_autotune=True and let the sweep choose."
        )

    if is_dynamic_persistent and tile_count_semaphore is None:
        raise ValueError(
            "is_dynamic_persistent=True requires tile_count_semaphore: the SM90 dynamic scheduler "
            "hands out work tiles through an atomic counter in GMEM, and there is no hardware "
            "cluster-launch-control fallback on Hopper. Pass a zeroed int32 tensor of shape (1,)."
        )

    # ── swap_ab ──────────────────────────────────────────────────────────────────────────────
    # Run the mainloop on the TRANSPOSED problem. `D[l,m,n] = sum_k A[l,m,k] * B[l,n,k]` and
    # `D^T[l,n,m] = sum_k B[l,n,k] * A[l,m,k]` are the same einsum with the free axes renamed, so
    # exchanging the operands and handing the kernel a transposed VIEW of D and C computes exactly
    # the same numbers into exactly the same memory. Nothing below needs to know: `describe_operands`
    # reads each tensor's own major-ness, so the transposed views produce their own descriptors and
    # their own compile key, and the kernel body is untouched.
    #
    # Placed HERE, after every front-door check and before `perm3d`, on purpose. Validating first
    # means a caller's shape error still names the tensor THEY passed ("B must be ...") rather than
    # the one the swap moved into that slot; swapping before `perm3d` means every descriptor, major
    # and compile-key term downstream describes what actually runs.
    #
    # Both operands are already ``(l, ., k)`` in this tree's convention, so the exchange is a plain
    # swap -- `main` needs `A if not swap else B` on a `(l, k, n)` B and gets the same effect. The
    # broadcast vectors follow the axes they broadcast along: an ``(l, n)`` row vector of D is an
    # ``(l, n)`` COLUMN vector of D^T, so the two simply trade places.
    if swap_ab:
        A, B = B, A
        D, C = D.mT, C.mT
        rowvec_bias, colvec_bias = colvec_bias, rowvec_bias

    A_p, B_p, D_p, C_p = perm3d(A, B, D, C)
    majors, dtypes = describe_operands(A_p, B_p, D_p, C_p)
    a_major, b_major, d_major, c_major = majors
    check_broadcast_alignment("rowvec_bias", rowvec_bias)
    check_broadcast_alignment("colvec_bias", colvec_bias)
    a_dtype, b_dtype, d_dtype, c_dtype = dtypes
    check_fp8_major(a_dtype, a_major, b_dtype, b_major)

    alpha_mode = 2 if isinstance(alpha, Tensor) else (1 if alpha != 1.0 else 0)
    colvec_ndim = colvec_bias.ndim if colvec_bias is not None else 0

    compiled_fn = _compile_gemm_hadamard(
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
        alpha_mode,
        torch2cute_dtype_map[rowvec_bias.dtype] if rowvec_bias is not None else None,
        torch2cute_dtype_map[colvec_bias.dtype] if colvec_bias is not None else None,
        colvec_ndim,
        batch_idx_permute is not None,
        device_capacity,
    )

    from fold_cp_ops._internal.cache_utils import COMPILE_ONLY

    if COMPILE_ONLY:
        return

    def scalar_arg(scalar, mode, dtype=Float32):
        """Convert one epilogue scalar into what the compiled entry expects for its mode.

        The launch-time counterpart of ``fake_scalar``: the same 0/1/2 encoding, but carrying the
        real value. The two must agree, since the compiled prototype was built from the fake one.

        Args:
            scalar: The user's value -- ignored at mode 0, converted at mode 1, and expected to be
                a torch tensor at mode 2 (its ``data_ptr()`` is taken).
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

    epi_args = GemmHadamardEpiMixin.EpilogueArguments(
        alpha=scalar_arg(alpha, alpha_mode),
        beta=None,
        mRowVecBroadcast=rowvec_bias,
        mColVecBroadcast=colvec_bias,
        add_to_output=None,
        rounding_mode=None,
        sr_seed=None,
    )
    scheduler_args = make_scheduler_args(
        max_active_clusters,
        max_swizzle_size,
        tile_count_semaphore,
        batch_idx_permute,
    )
    # SM90 call form (6 args).  Upstream had a 9-arg SM100 branch here plus a trailing varlen
    # argument; `require_sm90` above makes the first unreachable, and this tree's GEMM has no varlen
    # path at all, so both are dropped rather than left calling a path this package does not build.
    compiled_fn(A_p, B_p, D_p, C_p, epi_args, scheduler_args)

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

# Copyright (c) 2025, Tri Dao.
# This file has been modified by NVIDIA CORPORATION & AFFILIATES.
"""Composable epilogue operations (EpiOps) for GEMM kernels.

Each EpiOp encapsulates a single tensor kind's behavior across the epilogue lifecycle:
smem allocation, begin (one-time per-tile setup), begin_loop (per-subtile extraction),
end (cleanup).

The ops are composed via ComposableEpiMixin which iterates over a static _epi_ops tuple
to generate epi_smem_bytes_per_stage, epi_get_smem_struct, epi_get_smem_tensors,
epi_begin, and epi_begin_loop automatically.
"""

import math
import operator
from functools import partial

import cutlass
import cutlass.cute as cute
from cutlass import Float32, const_expr

from fold_cp_ops._internal.epi_utils import assume_stride_divisibility, setup_epi_tensor
from fold_cp_ops._internal.sm90_utils import partition_for_epilogue
import fold_cp_ops._internal.utils as utils
import fold_cp_ops._internal.compile_time.layout_utils as layout_utils


def _vec_fills_every_tile(mVec, tile_dim) -> bool:
    """Can the broadcast vector be PROVEN, at compile time, to fill every tile it is read over?

    Purpose
        `_vec_limit` is ``min(extent - coord_idx*tile_dim, tile_dim)`` and `coord_idx` is a runtime
        tile index, so ``limit >= tile_dim`` is a RUNTIME test even when the extent is static. The
        consequence is not a mispredicted branch, it is CODEGEN: both arms are emitted. Measured on
        the x_gate `alg_fold` kernel -- 2136 -> 4056 instructions and STACK 0 -> 184 bytes/thread of
        spill, which on a kernel running at ~87% of DRAM speed-of-light cost 1.24-1.68x.

        And on that path the predicated arm is DEAD: the vectors are ``(N,)`` with N == tile_N, so
        ``limit == tile_dim`` at every tile. It was emitted, never taken, and paid for.

    Semantics
        Proves the property from two STATIC facts and nothing else: if the extent is a compile-time
        int and a multiple of the tile, then for every valid ``coord_idx`` the remaining extent is
        at least one whole tile, so ``limit == tile_dim`` always and the predicate is a tautology.

        Returns False whenever either value is dynamic -- a symbolic extent is exactly the case the
        runtime guard exists for, and the guard is left in place there. So this NARROWS the emitted
        code without narrowing the containment: no shape loses its predicate.

    Args:
        mVec: The broadcast vector. Only ``shape[0]`` is read; a dynamic extent is not an error.
        tile_dim: The tile's extent along the broadcast SOURCE dimension.

    Returns:
        True only when both are static ints and the extent is a positive multiple of the tile.
    """
    extent = mVec.shape[0]
    if not isinstance(extent, int) or not isinstance(tile_dim, int):
        return False
    return tile_dim > 0 and extent > 0 and extent % tile_dim == 0


class EpiContext:
    """Shared context passed to EpiOp.begin methods. Bundles common arguments."""

    __slots__ = (
        "epi_tile",
        "tiled_copy_t2r",
        "tiled_copy_r2s",
        "tile_coord_mnkl",
        "epilogue_barrier",
        "tidx",
        "tile_idx",
        "partition_for_epilogue_fn",
        "num_epi_threads",
        "batch_idx",
        "tile_M",
        "tile_N",
    )

    def __init__(
        self,
        gemm,
        epi_tile,
        tiled_copy_t2r,
        tiled_copy_r2s,
        tile_coord_mnkl,
        epilogue_barrier,
        tidx,
        tile_idx=None,
    ):
        """Bundle everything an epilogue op needs about the current work tile.

        Args:
            gemm: The kernel functor, for the tile shapes and any subclass-specific attributes an
                op consults (``chunk_g``, gather flags).
            epi_tile: The epilogue subtile shape.
            tiled_copy_t2r: Tensor-memory-to-register copy, or None on SM90.
            tiled_copy_r2s: Register-to-shared copy for the epilogue.
            tile_coord_mnkl: This work tile's coordinate.
            epilogue_barrier: The epilogue's named barrier.
            tidx: Thread index within the CTA.
            tile_idx: Optional linear work-tile index.
        """
        self.epi_tile = epi_tile
        self.tiled_copy_t2r = tiled_copy_t2r
        self.tiled_copy_r2s = tiled_copy_r2s
        self.tile_coord_mnkl = tile_coord_mnkl
        self.epilogue_barrier = epilogue_barrier
        self.tidx = tidx
        # Persistent tile counter (num tiles executed by this CTA so far), available to ops
        # that need per-tile state. Currently unused by VecLoad (which loads gmem->regs).
        self.tile_idx = tile_idx
        self.tile_M = gemm.cta_tile_shape_mnk[0]
        self.tile_N = gemm.cta_tile_shape_mnk[1]
        self.batch_idx = tile_coord_mnkl[3]
        self.num_epi_threads = gemm.num_epi_warps * cute.arch.WARP_SIZE
        self.partition_for_epilogue_fn = partial(
            partition_for_epilogue,
            epi_tile=epi_tile,
            tiled_copy=tiled_copy_t2r if tiled_copy_t2r is not None else tiled_copy_r2s,
            tidx=tidx,
            reference_src=tiled_copy_t2r is None,
        )


def _get_lane_warp_layouts(tiled_copy, reference_src=True):
    """Derive lane and warp layouts along M and N from the epilogue tiled_copy.

    Follows the CUTLASS Sm90RowReduction / Sm90ColReduction pattern.
    Uses layout_src_tv_tiled (SM90, reference_src=True) or
    layout_dst_tv_tiled (SM100, reference_src=False), matching the C++ impl's
    get_layoutS_TV / get_layoutD_TV selection.

    Returns (lane_layout_MN, warp_layout_MN) where each is a 2D layout (M, N):
      lane_layout_MN[0] = lane_M: (lanes_in_M):(lane_stride_M) — e.g. 8:4
      lane_layout_MN[1] = lane_N: (lanes_in_N):(lane_stride_N) — e.g. 4:1
      warp_layout_MN[0] = warp_M: (warps_in_M):(warp_stride_M) — e.g. 4:1
      warp_layout_MN[1] = warp_N: (warps_in_N):(warp_stride_N) — e.g. 1:0

    For RowVecReduce (reduce along M): shuffle across lane_M, smem reduce across warp_M.
    For ColVecReduce (reduce along N): shuffle across lane_N, direct write (warps_in_N == 1).
    """
    # right_inverse of the TV layout gives tile_element_idx -> tv_idx.
    # SM90: use src (register) layout; SM100: use dst (smem) layout.
    layout_tv = tiled_copy.layout_src_tv_tiled if reference_src else tiled_copy.layout_dst_tv_tiled
    ref_layout = cute.right_inverse(layout_tv)
    tile_M_size, tile_N_size = cute.size(tiled_copy.tiler_mn[0]), cute.size(tiled_copy.tiler_mn[1])
    ref_layout_MN = cute.composition(
        ref_layout, cute.make_layout((tile_M_size, tile_N_size))
    )  # (tile_M, tile_N) -> tv_idx

    num_warps = cute.size(tiled_copy) // cute.arch.WARP_SIZE

    # tv2lane: tv_idx -> lane_idx  (lane = tv_idx % 32)
    tv2lane = cute.make_layout((cute.arch.WARP_SIZE, num_warps, 1), stride=(1, 0, 0))
    ref2lane = cute.composition(tv2lane, ref_layout_MN)  # (tile_M, tile_N) -> lane_idx
    # select mode [0] = M part, [1] = N part; filter removes stride-0
    lane_M = cute.filter(cute.select(ref2lane, [0]))  # lane_m -> lane_idx
    lane_N = cute.filter(cute.select(ref2lane, [1]))  # lane_n -> lane_idx
    lane_layout_MN = layout_utils.concat_layout(lane_M, lane_N)  # (lane_M, lane_N) -> lane_idx

    # tv2warp: tv_idx -> warp_idx  (warp = tv_idx / 32)
    tv2warp = cute.make_layout((cute.arch.WARP_SIZE, num_warps, 1), stride=(0, 1, 0))
    ref2warp = cute.composition(tv2warp, ref_layout_MN)  # (tile_M, tile_N) -> warp_idx
    warp_M = cute.filter(cute.select(ref2warp, [0]))  # warp_m -> warp_idx
    warp_N = cute.filter(cute.select(ref2warp, [1]))  # warp_n -> warp_idx
    warp_layout_MN = layout_utils.concat_layout(warp_M, warp_N)  # (warp_M, warp_N) -> warp_idx

    return lane_layout_MN, warp_layout_MN


class EpiOp:
    """Base class for composable epilogue operations."""

    def __init__(self, name):
        """Name this op, which is the key it is addressed by everywhere else.

        Args:
            name: The op's name. It must match the ``EpilogueArguments`` field it reads, the
                generated ``EpilogueParams`` field it writes, and the key ``epi_visit_subtile``
                looks it up under -- one string, three roles, and a mismatch in any of them is a
                silently absent term rather than an error.
        """
        self.name = name

    # --- Host-side: args → params ---
    def param_fields(self):
        """Return [(field_name, type, default), ...] for auto-generating EpilogueParams.
        Must match the keys returned by to_params()."""
        return []

    def to_params(self, gemm, args):
        """Convert this op's arg field(s) to param dict entries.
        Returns dict of {param_name: value}. Like EVT's to_underlying_arguments."""
        return {}

    # --- Host-side: smem allocation ---
    def smem_bytes(self, arg_tensor, cta_tile_shape_mnk, epi_tile):
        """Bytes of smem needed per stage. arg_tensor is the EpilogueArguments field."""
        return 0

    def smem_struct_field(self, gemm, params):
        """Return (field_name, field_type) for @cute.struct, or None if no smem needed.
        params is the full EpilogueParams object."""
        return None

    def get_smem_tensor(self, gemm, params, storage_epi):
        """Extract smem tensor from storage.epi. Returns tensor or None.
        params is the full EpilogueParams object."""
        return None

    def tma_atoms(self, gemm, params):
        """Return list of TMA atoms for this op."""
        return []

    # --- Device-side: kernel execution ---
    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        """One-time per-tile setup. Returns state for begin_loop."""
        return None

    def begin_loop(self, gemm, state, epi_coord):
        """Per-subtile extraction. Returns value for epi_visit_subtile."""
        return state

    def needs_async_fence(self):
        """Whether this op issues async copies that need a fence."""
        return False

    def end(
        self,
        gemm,
        param,
        state,
        epi_tile,
        tiled_copy_t2r,
        tiled_copy_r2s,
        tile_coord_mnkl,
        tidx,
    ):
        """Cleanup after all subtiles (reductions, direct writes)."""
        pass


class Scalar(EpiOp):
    """Loads a scalar value or device pointer once per tile. No smem."""

    def __init__(self, name, dtype=None):
        """Declare a scalar epilogue term.

        Args:
            name: The op's name; see :meth:`EpiOp.__init__`.
            dtype: The type to read the scalar as when it arrives as a device pointer, or None to
                default to fp32. Must match the type the pointer was created with -- nothing checks
                it, and a mismatch reinterprets the bits.
        """
        super().__init__(name)
        self.dtype = dtype

    def param_fields(self):
        """One untyped params field, defaulting to None so an absent scalar costs nothing.

        Returns:
            ``[(name, object, None)]``.
        """
        return [(self.name, object, None)]

    def to_params(self, gemm, args):
        """Copy this scalar straight from the arguments into the params dict.

        Args:
            gemm: The kernel functor. Unused here.
            args: The ``EpilogueArguments``.

        Returns:
            ``{name: value}``, where the value may be a scalar, a pointer, or None.
        """
        return {self.name: getattr(args, self.name)}

    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        """Resolve the scalar once per work tile: dereference a pointer, or pass a constant through.

        Args:
            gemm: The kernel functor. Unused here.
            param: The scalar, a device pointer, or None.
            smem_tensor: Unused -- a scalar needs no SMEM.
            ctx: The :class:`EpiContext`. Unused here.

        Returns:
            The resolved value, or None when the term is absent -- which is what makes the whole
            term vanish from the generated epilogue.
        """
        result = None
        if const_expr(param is not None):
            result = (
                utils.load_scalar_or_pointer(param, dtype=self.dtype)
                if const_expr(self.dtype is not None)
                else utils.load_scalar_or_pointer(param)
            )
        return result


class VecLoad(EpiOp):
    """Base class for broadcast vector loads (row or col).

    The per-tile broadcast vector is loaded directly from GMEM into registers in `begin`
    (each epilogue thread loads exactly the elements its tile partition needs, via the
    epilogue TV partition over a stride-0 broadcast GMEM view). This deliberately avoids
    staging through a single-buffered `s_<name>` SMEM tile: a cooperative persistent CTA
    that processes >1 output tile would otherwise hit a WAR race where a later tile's
    cp.async store into `s_<name>` clobbers an earlier tile's still-being-consumed buffer.
    The per-tile epilogue_barrier only orders load-before-consume for the CURRENT tile,
    and a plain named barrier does NOT serialize cp.async writes across the persistent
    loop, so epilogue threads drifting >1 tile apart corrupt the shared buffer
    (empirically: double-buffering only papers over the race until the per-CTA tile count
    exceeds the buffer count). Loading into per-thread registers removes the shared
    resource entirely, eliminating the cross-tile hazard. The vec is tiny and stays
    L2-resident, so the redundant broadcast loads are cheap.

    Subclasses set `dim` to 0 (M/col) or 1 (N/row) and override `_get_gmem_vec`

    """

    dim = None  # 0 for col (M), 1 for row (N)

    def param_fields(self):
        """One untyped params field, defaulting to None.

        Returns:
            ``[(name, object, None)]``.
        """
        return [(self.name, object, None)]

    def to_params(self, gemm, args):
        """Copy the vector into the params, telling the compiler its strides are 32-bit aligned.

        Args:
            gemm: The kernel functor. Unused here.
            args: The ``EpilogueArguments``.

        Returns:
            ``{name: tensor}`` with the divisibility assumption applied, which is what lets the
            broadcast load vectorize. The assumption is a PROMISE -- ``gemm()`` checks it at the
            front door, and a caller bypassing that gets a misaligned access.
        """
        return {self.name: assume_stride_divisibility(getattr(args, self.name))}

    def _tile_size(self, cta_tile_shape_mnk):
        """The CTA-tile extent along this vector's broadcast dimension.

        Args:
            cta_tile_shape_mnk: The CTA tile.

        Returns:
            ``cta_tile_shape_mnk[self.dim]`` -- N for a row vector, M for a column vector.
        """
        return cta_tile_shape_mnk[self.dim]

    def _broadcast_stride(self):
        # Row: stride (0,1) — broadcast along M. Col: stride (1,0) — broadcast along N.
        """The stride pair that broadcasts this vector across the other dimension.

        Returns:
            ``(0, 1)`` for a row vector (constant down M) and ``(1, 0)`` for a column vector
            (constant across N). The zero is the broadcast, so swapping them silently transposes
            which axis the bias varies along.
        """
        return (0, 1) if self.dim == 1 else (1, 0)

    def _tile_dim(self, ctx):
        """The runtime tile extent along this vector's dimension.

        Args:
            ctx: The :class:`EpiContext`.

        Returns:
            ``ctx.tile_N`` for a row vector, ``ctx.tile_M`` for a column vector.
        """
        return ctx.tile_N if self.dim == 1 else ctx.tile_M

    def _coord_idx(self):
        """Which entry of the work-tile coordinate indexes this vector.

        Returns:
            1 for a row vector (the N coordinate), 0 for a column vector (the M coordinate).
        """
        return 1 if self.dim == 1 else 0

    # No SMEM and no async fence: the vec is loaded gmem->registers directly in begin().
    def smem_bytes(self, arg_tensor, cta_tile_shape_mnk, epi_tile):
        """SMEM this op needs per stage: none.

        Broadcast vectors are read GMEM -> registers directly; there is no tile to stage.

        Args:
            arg_tensor: The vector, or None.
            cta_tile_shape_mnk: The CTA tile.
            epi_tile: The epilogue subtile.

        Returns:
            0.
        """
        return 0

    def smem_struct_field(self, gemm, params):
        """No SMEM struct field.

        Args:
            gemm: The kernel functor.
            params: The traced epilogue params.

        Returns:
            None.
        """
        return None

    def get_smem_tensor(self, gemm, params, storage_epi):
        """No SMEM tensor.

        Args:
            gemm: The kernel functor.
            params: The traced epilogue params.
            storage_epi: The epilogue's SMEM struct.

        Returns:
            None.
        """
        return None

    def needs_async_fence(self):
        """Whether this op issues ``cp.async`` and so needs a group fence before the tile is used.

        Returns:
            False. A subclass that stages through ``cp.async`` overrides this, and
            ``ComposableEpiMixin.epi_begin`` then emits ONE commit/wait/barrier for all such ops.
        """
        return False

    def _get_gmem_vec(self, param, ctx):
        """Select this batch's slice of the vector. Override to change the batch mapping."""
        return param[ctx.batch_idx, None]

    @cute.jit
    def _vec_limit(self, mVec, coord_idx, tile_dim, ctx):
        """Out-of-bounds guard along the broadcast-source dimension."""
        return min(mVec.shape[0] - coord_idx * tile_dim, tile_dim)

    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        """Load the broadcast vec gmem->registers, returning a register tensor whose layout
        matches the epilogue subtile partition (broadcast dim has stride 0)."""
        tDrV = None
        if const_expr(param is not None):
            mVec = self._get_gmem_vec(param, ctx)
            tile_dim = self._tile_dim(ctx)
            coord_idx = ctx.tile_coord_mnkl[self._coord_idx()]
            gVec = cute.local_tile(mVec, (tile_dim,), (coord_idx,))
            # Broadcast GMEM view over the (tile_M, tile_N) tile, then partition for the
            # epilogue thread-value layout (same partition that consumes the data).
            gBroadcast = cute.make_tensor(
                gVec.iterator,
                cute.make_layout((ctx.tile_M, ctx.tile_N), stride=self._broadcast_stride()),
            )
            tDgV = ctx.partition_for_epilogue_fn(gBroadcast)
            # Identity (coordinate) tensor over the full tile to predicate the source dim.
            tDcV = ctx.partition_for_epilogue_fn(
                cute.make_identity_tensor((ctx.tile_M, ctx.tile_N))
            )
            if const_expr(ctx.tiled_copy_t2r is not None):
                tDgV = ctx.tiled_copy_r2s.retile(tDgV)
                tDcV = ctx.tiled_copy_r2s.retile(tDcV)
            limit = self._vec_limit(mVec, coord_idx, tile_dim, ctx)
            src_dim = self.dim  # 0 (col, source = M) or 1 (row, source = N)
            tDrV = cute.make_rmem_tensor(tDgV.layout, tDgV.element_type)
            # tDgV (broadcast, stride 0 along the non-source dim) and tDrV share a layout;
            # tDcV is the matching identity partition. Iterate the flat partition and
            # predicate by the source-dim coordinate.
            zero = tDgV.element_type(0.0)
            # PING-PONG TAKES THE PREDICATED FORM, and the reason is measured, not structural.
            # `_vec_fills_every_tile` proves the predicate is a tautology, so dropping it is always
            # CORRECT -- it removes 32 `FSEL` + 31 `ISETP.GE.AND` that can never change a value.
            # Cooperatively that is a win (0.92-0.97x of the kernel this reproduces, D in
            # {128,256,384,512}); under ping-pong the SAME removal COSTS 4.9-5.6%. Three-arm
            # interleaved, one H100, M=262144, 9 reps: at D=512 main 0.67197, ours 0.70970 (1.0561),
            # ours with the predicate restored 0.67232 (1.0005). The third arm is this tree with only
            # this method's body replaced, so the gap is this branch, not the surrounding kernel.
            #
            # What ships is NOT that arm: predicating the LOAD spends `SEL`/`ISETP` where an
            # unconditional load spends `FSEL` (census 2312 vs 2352). Re-measured on what ships, same
            # box and interleave: D=128 1.0084, D=256 1.0169, D=384 0.9912, D=512 0.9921 -- against
            # 0.9914 / 1.0482 / 0.9878 / 1.0512 before. Worst 1.7% at a per-arm spread of 1.0-1.5%.
            # The residual 1-2% at D=128/256 is the PRICE OF THE CONTAINMENT, not of the schedule, and
            # is left standing. The unpaid option if it ever matters: CLAMP the index and keep the load
            # unconditional, as `TransposedMaskColVecLoad` does -- contained AND main-shaped, but
            # unmeasured, and an unmeasured third variant is how this branch grew its second blind spot.
            #
            # Why the fast arm costs anything here is NOT established; only that it does, reproducibly.
            # Cooperative is byte-identical across this change (digests checked, not assumed), and
            # `pingpong` is already a compile key, so no configuration can take another's artifact.
            #
            # SCOPE: the gate is on the SCHEDULE, so every `VecLoad` user reaching ping-pong takes it.
            # Deliberate -- nothing about the mechanism is fold-specific, and it moves each of them
            # toward the kernel this package reproduces, which has no fast arm at all.
            #
            # It goes to the PER-ELEMENT PREDICATE, never the runtime-branch arm below: that arm emits
            # BOTH bodies (`limit >= tile_dim` is a runtime test) and was measured spilling 184
            # bytes/thread, and ping-pong has LESS register headroom than cooperative. It also keeps the
            # LOAD predicated, which main's form does not -- main issues the global access
            # unconditionally and selects only the VALUE, the out-of-bounds read closed here (933
            # memcheck errors on the plain `gemm` rowvec_bias path). Taking main's bytes back re-opens it.
            if const_expr(gemm.pingpong):
                for i in cutlass.range(cute.size(tDgV), unroll_full=True):
                    if tDcV[i][src_dim] < limit:
                        tDrV[i] = tDgV[i]
                    else:
                        tDrV[i] = zero
            elif const_expr(_vec_fills_every_tile(mVec, tile_dim)):
                for i in cutlass.range(cute.size(tDgV), unroll_full=True):
                    tDrV[i] = tDgV[i]
            else:
                # NESTED, not `elif`: the arms above are COMPILE-TIME conditions, so Python -- not
                # the DSL -- evaluates that chain. An `elif` on the dynamic `limit` would be forced
                # to a Python bool at trace time and raise "Unable to convert dynamic Boolean value
                # to bool at compile time". Nesting keeps it a DSL branch.
                if limit >= tile_dim:
                    for i in cutlass.range(cute.size(tDgV), unroll_full=True):
                        tDrV[i] = tDgV[i]
                else:
                    for i in cutlass.range(cute.size(tDgV), unroll_full=True):
                        if tDcV[i][src_dim] < limit:
                            tDrV[i] = tDgV[i]
                        else:
                            tDrV[i] = zero
        return tDrV

    @cute.jit
    def begin_loop(self, gemm, state, epi_coord):
        """Convert this epi tile's slice of the broadcast fragment to ``acc_dtype``.

        WHY THE ELEMENTWISE LOOP (do NOT "simplify" this back to ``.store(.load())``).
        ``state`` carries the BROADCAST layout from :meth:`begin` — stride 0 along the non-source
        dim — so ``tDrV_cur`` is a **non-injective** memref: at ``epi_tile[1] == 16`` its size is 8
        while its cosize is 2 (``(((2,2,2),1),1,1):(((0,1,0),0),0,0)``). libNVVM cannot lower a VECTOR
        op on such a memref: it rejects the module with an unlocated ``Error: unsupported operation``
        and no instruction, no source location — the failure recorded as the "cp=16 masked NVVM
        compile failure" (``docs/migration_failing_tests.md`` §2.4 / §6.1). It is neither a mask bug
        nor a distributed one: the plain pre-glu ``mColVecBroadcast`` on the non-distributed
        ``DualGatedGemmStagedSm90`` fails identically, and every ``ColVecLoad``-family user inherits
        this method.

        Trigger = ``epi_tile[1] == 16`` (⟺ ``tile_N % 32 == 16`` ⟺ ``postact_epi_n == 8``) AND a
        ColVec-family param non-None AND >=2 distinct storage offsets of the fragment staying live.
        The last clause is why it looks sporadic: the degenerate vector op is ALWAYS emitted, but with
        one live offset LLVM promotes the alloca away before libNVVM ever sees it.

        Making only the DESTINATION compact does not help — ``tDrV_cur.load()`` is itself a vector op
        on the non-injective SOURCE. So no vector op may touch it at all: gather elementwise into a
        compact fragment, then re-view it at the original shape. ``cute.make_layout(shape)`` is the
        compact column-major layout, so flat slot ``i`` holds exactly what ``tDrV_cur[i]`` returned —
        consumers index flat (``gemm_act.py:60-68``, ``gemm_default_epi.py``) and are unchanged.
        """
        tDrV_cvt = None
        if const_expr(state is not None):
            tDrV_cur = cute.group_modes(state, 3, cute.rank(state))[None, None, None, epi_coord]
            flat = cute.make_rmem_tensor(cute.make_layout(cute.size(tDrV_cur)), gemm.acc_dtype)
            for i in cutlass.range_constexpr(cute.size(tDrV_cur)):
                flat[i] = tDrV_cur[i].to(gemm.acc_dtype)
            tDrV_cvt = cute.make_tensor(flat.iterator, cute.make_layout(tDrV_cur.shape))
        return tDrV_cvt


class RowVecLoad(VecLoad):
    """Loads a row vector (N,) from GMEM, broadcasts along M with stride (0,1)."""

    dim = 1


class Gate3RowVecLoad(RowVecLoad):
    """Loads the gate3 bias rowvec for the dual-gated TriMul output-gate.  Two designs share this op,
    keyed on gemm._gate3_full_width:
      REGION-AWARE (staged, full_width=True): the gate3 acc is the FULL BLK_N-wide dual accumulator (a
        first-class gate3 work-tile against B_all's W3 rows), so this slices the bias at the FULL BLK_N
        granularity by the kernel-supplied LOCAL gate3 n-block (tile_coord_mnkl[1]); the bias source is
        a (n3_pad,) b3 rowvec indexed from col 0.  The limit predicates the trailing partial slice
        (N3 < BLK_N): out-of-range cols read 0, and their acc (zero-W3-padding) is not stored.
      NATIVE half-width (stagec/legacy, full_width=False): the gate3 acc is BLK_N//2-wide, so this
        slices the (N3,) b3 rowvec at BLK_N//2 granularity by the gate3 sub-tile n-block."""

    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        """Load this work tile's slice of the broadcast vector into registers.

        Args:
            gemm: The kernel functor, consulted for the gather/chunk attributes that widen the
                slice in the fused variants.
            param: The vector, or None.
            smem_tensor: Unused.
            ctx: The :class:`EpiContext`, supplying the tile coordinate and the ragged offsets.

        Returns:
            The register fragment, or None when the term is absent. Out-of-range lanes are
            predicated rather than clamped, so a partial trailing tile reads no garbage.
        """
        tDrV = None
        if const_expr(param is not None):
            mVec = self._get_gmem_vec(param, ctx)  # b3 rowvec (region-aware n3_pad / legacy N3)
            full_width = const_expr(getattr(gemm, "_gate3_full_width", False))
            tile_dim = const_expr(ctx.tile_N if full_width else ctx.tile_N // 2)
            coord_idx = ctx.tile_coord_mnkl[1]  # gate3 n-block (local for both designs)
            gVec = cute.local_tile(mVec, (tile_dim,), (coord_idx,))  # b3 slice
            # Broadcast the BLK_N-wide b3 slice over the (tile_M, BLK_N) tile (stride (0,1)), then
            # partition with the epilogue thread-value layout.
            gBroadcast = cute.make_tensor(
                gVec.iterator, cute.make_layout((ctx.tile_M, tile_dim), stride=(0, 1))
            )
            tDgV = ctx.partition_for_epilogue_fn(gBroadcast)
            tDcV = ctx.partition_for_epilogue_fn(cute.make_identity_tensor((ctx.tile_M, tile_dim)))
            if const_expr(ctx.tiled_copy_t2r is not None):
                tDgV = ctx.tiled_copy_r2s.retile(tDgV)
                tDcV = ctx.tiled_copy_r2s.retile(tDcV)
            limit = self._vec_limit(mVec, coord_idx, tile_dim, ctx)
            tDrV = cute.make_rmem_tensor(tDgV.layout, tDgV.element_type)
            zero = tDgV.element_type(0.0)
            for i in cutlass.range(cute.size(tDgV), unroll_full=True):
                val = tDgV[i]
                tDrV[i] = val if tDcV[i][1] < limit else zero
        return tDrV


class ColVecLoad(VecLoad):
    """Loads a col vector (M,) from GMEM, broadcasts along N with stride (1,0)."""

    dim = 0

    @cute.jit
    def _get_gmem_vec(self, param, ctx):
        """Select this batch's slice of the column vector.

        Args:
            param: The full ``(l, m)`` vector. Mode 0 must be the batch axis; passing a flat 1-D
                vector indexes along ``m`` instead and silently reads the wrong elements.
            ctx: The :class:`EpiContext`, read for ``batch_idx`` only.

        Returns:
            A 1-D view -- no copy -- of this batch's vector.
        """
        return param[ctx.batch_idx, None]

    @cute.jit
    def _vec_limit(self, mVec, coord_idx, tile_dim, ctx):
        """How many elements of the vector this tile may read.

        Args:
            mVec: This batch's vector view; its ``shape[0]`` is the true extent.
            coord_idx: This tile's index along the vector's dimension.
            tile_dim: The tile extent along that dimension. Must be positive.
            ctx: The :class:`EpiContext`. Unused here -- present so a subclass that does need
                per-tile state overrides with the same signature.

        Returns:
            The element count, capped at ``tile_dim``. Used to build the load predicate, so an
            over-large value reads past the vector -- silently, with no fault on a mapped page.
        """
        return min(mVec.shape[0] - coord_idx * tile_dim, tile_dim)


class SmemColVecBroadcast(EpiOp):
    """Broadcasts one column of a per-CTA SMEM scratch along N, for a mainloop-computed statistic.

    Purpose
        The algebraic-fold fusion computes its per-row statistics in the MAINLOOP and consumes them
        in the EPILOGUE. Every other broadcast op here sources from GMEM, which cannot serve that:
        the values do not exist until the k-loop has run, and they are per-CTA, not per-tensor.

    Semantics
        Reads ``smem_tensor[:, slot]`` -- one column of the ``(tile_M, 2)`` fp32 scratch the
        mainloop wrote -- and broadcasts it along N with stride ``(1, 0)``, partitioned by the same
        epilogue thread-value layout that consumes it. Returns a register tensor, exactly as
        `VecLoad` does, so `epi_visit_subtile` sees the same shape of value from either.

        **It allocates nothing.** `smem_struct_field` returns None and `get_smem_tensor` returns
        None, because the scratch belongs to the mainloop, not the epilogue. The functor supplies it
        by overriding ``epi_get_smem_tensors``, which already receives the WHOLE storage object and
        merely narrows to ``storage.epi`` for the ops that own their buffers. That override is the
        entire mechanism -- no shared signature changes, and a fusion that does not override it is
        unaffected.

        **No source predication, deliberately.** `VecLoad` predicates its load because it reads
        GMEM past the vector's end on a partial tile. Here the scratch is always ``tile_M`` rows and
        is CTA-local, so an out-of-range row reads a stale fp32 rather than unmapped memory, and the
        value only reaches an output element the store predicate drops. Adding a limit would cost a
        comparison per element to change nothing.

    Attributes:
        slot: Which column of the scratch to read. Must index a column the mainloop actually
            writes; a wrong slot silently broadcasts the other statistic, which stays finite and
            plausibly-scaled, so nothing downstream faults.
    """

    def __init__(self, name, slot):
        """Bind the op's name and its column of the scratch.

        Args:
            name: The op's key, in all three roles `EpiOp` describes.
            slot: Column index into the ``(tile_M, 2)`` scratch. Not range-checked here -- the
                scratch's width is a functor detail this op does not see.
        """
        super().__init__(name)
        self.slot = slot

    def param_fields(self):
        """One untyped params field, defaulting to None, to match every other op's shape.

        Returns:
            ``[(name, object, None)]``. The field is never read: this op's data is device-side SMEM,
            not a host argument. It exists so the generated ``EpilogueParams`` has an entry under
            this op's name, which the composition machinery assumes for every declared op.
        """
        return [(self.name, object, None)]

    def smem_bytes(self, arg_tensor, cta_tile_shape_mnk, epi_tile):
        """Zero -- the scratch is the mainloop's allocation, not the epilogue's.

        Args:
            arg_tensor: Unused.
            cta_tile_shape_mnk: Unused.
            epi_tile: Unused.

        Returns:
            0. Reserving bytes here would double-allocate the scratch and, because the reservation
            feeds the pipeline-depth computation, would change the emitted schedule.
        """
        return 0

    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        """Broadcast this op's scratch column into a register tensor over the epilogue tile.

        Args:
            gemm: The kernel functor. Unused.
            param: The params field. Unused -- see `param_fields`.
            smem_tensor: The ``(tile_M, 2)`` fp32 scratch, supplied by the functor's
                ``epi_get_smem_tensors`` override. None compiles the op out entirely.
            ctx: The :class:`EpiContext`, read for the tile extents and the epilogue partition.

        Returns:
            A register tensor whose layout matches the epilogue subtile partition, with the
            broadcast axis at stride 0; or None when `smem_tensor` is None.
        """
        if const_expr(smem_tensor is None):
            return None
        # Pick THIS warpgroup's buffer with a FRESH warp_idx read rather than a value captured in
        # the kernel body: the upstream does the same and says why -- it avoids carrying an SSA
        # warp-group index across the epilogue region.
        wg = (
            cute.arch.make_warp_uniform(cute.arch.warp_idx() // 4)
            if const_expr(gemm.pingpong)
            else 0
        )
        col_iter = smem_tensor[None, self.slot, wg].iterator
        tDsV = ctx.partition_for_epilogue_fn(
            cute.make_tensor(col_iter, cute.make_layout((ctx.tile_M, ctx.tile_N), stride=(1, 0)))
        )
        if const_expr(ctx.tiled_copy_t2r is not None):
            tDsV = ctx.tiled_copy_r2s.retile(tDsV)
        tDsV_sub = cute.group_modes(tDsV, 3, cute.rank(tDsV))[None, None, None, 0]
        return [tDsV, cute.make_rmem_tensor(tDsV_sub.layout, gemm.acc_dtype)]

    def begin_loop(self, gemm, state, epi_coord):
        """Load this row-tile's slice on the FIRST N subtile; reuse the cached registers after.

        Purpose
            A column vector is constant along N, so it changes only when the subtile's M changes.
            Reloading per subtile is correct and pure waste; never reloading is cheap and WRONG once
            there is more than one M subtile. This does neither.

        Semantics
            **Both halves of that were paid for.** Returning the whole-tile fragment unsliced --
            what this did originally -- handed every subtile the FIRST one's rows, which at
            ``tile_M=256`` produced a finite, plausible output wrong by ~20%. Then slicing on every
            subtile with an elementwise gather fixed the answer and cost ~10% of the kernel. The
            upstream does what is written here: slice by ``epi_coord``, but only reload when
            ``epi_n == 0``, and copy with a vectorized `autovec_copy` over the zero-FILTERED view
            rather than element by element.

            ``filter_zeros`` is what makes the vector copy legal: the broadcast layout has stride 0
            along N, so the fragment is non-injective and a vector op on it is rejected by libNVVM
            (see `VecLoad.begin_loop` for that failure). Filtering leaves a compact view.

        Args:
            gemm: The kernel functor; read for ``acc_dtype``.
            state: The ``[tDsV, tDrV_cvt]`` pair `begin` returned, or None.
            epi_coord: This subtile's ``(epi_m, epi_n)`` coordinate. **Load-bearing on both axes**:
                the M part selects the rows, the N part decides whether a reload is needed.

        Returns:
            The cached ``acc_dtype`` fragment for this subtile, or None when the term is absent.
        """
        if const_expr(state is None):
            return None
        tDsV, tDrV_cvt = state[0], state[1]
        if epi_coord[1] == 0:
            tDsV_cur = cute.group_modes(tDsV, 3, cute.rank(tDsV))[None, None, None, epi_coord]
            tDrV = cute.make_rmem_tensor(tDsV_cur.layout, tDsV_cur.element_type)
            cute.autovec_copy(cute.filter_zeros(tDsV_cur), cute.filter_zeros(tDrV))
            tDrV_cvt.store(tDrV.load().to(gemm.acc_dtype))
        return tDrV_cvt


class TransposedMaskColVecLoad(ColVecLoad):
    """ZERO-COPY pair-mask ColVec for the transpose_in (composite_k / route2_ni) front.

    Reads the user's NATIVE ``(l=1, Xg=N_i_loc, Yg=N_j_loc)`` mask DIRECTLY with in-kernel transpose-walk
    index math — NO host transpose / zero-pad / fp32-cast (the ColVecLoad's begin_loop casts native->fp32).
    For the NON-transpose front (``gemm._transpose_in`` False) it is BYTE-IDENTICAL to the stock ColVecLoad
    (the ``const_expr`` branch picks the base path), so the plain (1,M) mask read is unchanged.

    transpose_in walk: the M-tile ``tile_m`` unravels to ``(tile_y=j_loc, tile_x=i_loc-block)`` with tile_x
    INNER (``n_x = ceil(Xg/BLK_M)`` x-tiles per Yg, RUNTIME from Xg). Row ``m`` of the tile is native
    ``i = tile_x*BLK_M + m`` at fixed ``j = tile_y`` -> the mask value is ``mask[b, i, tile_y]``, a strided
    (stride Yg) column read of the native tensor. Pad rows (``i >= Xg``, the partial last x-tile) are
    CLAMPED to ``i = Xg-1`` so the gmem ADDRESS stays in-bounds; their VALUE is irrelevant because the pad
    output is 0 (acc=0 -> glu(0)=0) — the exact invariant that makes nomask straddle correct. ``Xg, Yg`` come
    from the RUNTIME mask shape (mark_layout_dynamic) so the dynamic-N (compile-once-many-N) path is safe;
    ``BLK_M`` is the static ``ctx.tile_M``.

    BATCH. ``tile_m // n_x`` is the flat ``(b*Yg + y)``, so with a batch plane it is SPLIT: ``b`` selects
    the mask's mode-0 and ``y`` is the Yg walk index. Gated on the store's ``_a2a_b_plane`` so a
    non-batch build keeps ``tile_y = tile_m // n_x`` and the l-plane ``ctx.batch_idx`` verbatim --
    ``ctx.batch_idx`` is 0 there (the front GEMM's L extent is 1), which is exactly the plane the split
    arm computes at B==1, so the two agree on value and differ only in emitted arithmetic.
    """

    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        # Branch on the mask param's STATIC RANK (robust, no reliance on a gemm attr): the transpose_in
        # front passes a 3-D native (l, Xg, Yg) mask -> the zero-copy read below; the plain front passes a
        # 2-D (l, M) mask (or None for nomask) -> the stock ColVecLoad, byte-identical. cute.rank is wrapped
        # in const_expr FIRST (mirrors dual_gated_gemm_staged.py:1635) so it folds to a static int.
        """Load a transposed, masked column vector, branching on the mask parameter's static RANK.

        The rank is what distinguishes the masked variant from the plain one: a rank-3 parameter
        carries the mask, anything else is the unmasked case and delegates to the base. Branching on
        the rank rather than on a functor attribute keeps the two variants from depending on the
        kernel remembering to set a flag.

        Args:
            gemm: The kernel functor.
            param: The vector, or None (which delegates straight to the base).
            smem_tensor: Unused.
            ctx: The :class:`EpiContext`.

        Returns:
            The register fragment, or None.
        """
        if const_expr(param is None):
            return super().begin(gemm, param, smem_tensor, ctx)  # nomask
        _rank = const_expr(cute.rank(param))
        if const_expr(_rank != 3):
            return super().begin(gemm, param, smem_tensor, ctx)  # plain (1,M): stock ColVecLoad
        if const_expr(bool(getattr(gemm, "_pad_inner", False))):
            return self._begin_pad_inner(gemm, param, ctx)
        tDrV = None
        if const_expr(param is not None):
            mMask = param  # native (l, Xg, Yg); mark_layout_dynamic -> Xg,Yg runtime
            blk_m = ctx.tile_M
            Xg = mMask.shape[1]
            Yg = mMask.shape[2]
            n_x = cute.ceil_div(Xg, blk_m)  # runtime x-tiles per Yg (partial last tile allowed)
            tile_m = ctx.tile_coord_mnkl[0]
            tile_x = tile_m % n_x
            rem = tile_m // n_x  # flat (b*Yg + y) over the transposed walk's outer axes
            if const_expr(bool(getattr(gemm, "_a2a_b_plane", False))):
                b_idx = rem // Yg  # batch plane; the mask's mode-0 is B, not a degenerate l
                tile_y = rem - b_idx * Yg  # Yg walk index within the plane
            else:
                b_idx = ctx.batch_idx  # 0 (front L extent is 1) -- byte-identical to before
                tile_y = rem
            gCol = mMask[b_idx, None, tile_y]  # (Xg,) stride-Yg column at j=tile_y
            # Register LAYOUT from the epilogue TV partition (matches tRS_rD). The gmem STRIDE is irrelevant
            # to the partition SHAPE (we never READ this broadcast view — values come from the clamped
            # gather below); it only carries the mask element_type into make_rmem_tensor.
            gLayout = cute.make_tensor(
                gCol.iterator,
                cute.make_layout((ctx.tile_M, ctx.tile_N), stride=self._broadcast_stride()),
            )
            tDgV = ctx.partition_for_epilogue_fn(gLayout)
            tDcV = ctx.partition_for_epilogue_fn(
                cute.make_identity_tensor((ctx.tile_M, ctx.tile_N))
            )
            if const_expr(ctx.tiled_copy_t2r is not None):
                tDgV = ctx.tiled_copy_r2s.retile(tDgV)
                tDcV = ctx.tiled_copy_r2s.retile(tDcV)
            tDrV = cute.make_rmem_tensor(tDgV.layout, tDgV.element_type)
            x0 = tile_x * blk_m
            last = Xg - 1
            zero = tDgV.element_type(0.0)
            for i in cutlass.range(cute.size(tDgV), unroll_full=True):
                native_i = x0 + tDcV[i][0]  # M-row within tile -> native i
                in_range = native_i < Xg
                # CLAMP the ADDRESS in-bounds (no OOB load), then ZERO the pad-row VALUE — byte-identical
                # to the old host mask_pad[Xg:]=0 (pad output is 0 regardless; this holds even if a front
                # value-bias were ever added, so it does not rely on the glu(0)==0 tail invariant).
                src_i = native_i if in_range else last
                val = gCol[src_i]  # scalar strided (stride Yg) gmem load, always in-bounds
                tDrV[i] = val if in_range else zero
        return tDrV

    @cute.jit
    def _begin_pad_inner(self, gemm, param, ctx):
        """P2 pad_inner pair-mask read — the MIRROR of the transpose_in walk above.

        pad_inner keeps the NATIVE (x, y) token order and pads the native-INNER axis ``Yg`` to a whole
        ``BLK_M`` count, so the M-tile ``tile_m`` unravels ``(tile_x, tile_y)`` with tile_y INNER
        (``n_y = ceil(Yg/BLK_M)`` y-tiles per Xg, RUNTIME from the mask shape). Row ``m`` of the tile is
        native ``y = tile_y*BLK_M + m`` at fixed ``x = tile_x`` -> the mask value is ``mask[b, tile_x, y]``,
        a CONTIGUOUS (stride-1) row read (the transposed walk's read is stride-``Yg``). Pad rows
        (``y >= Yg``) CLAMP the ADDRESS to ``Yg-1`` so no OOB load is issued, and their VALUE is zeroed —
        the pad output is 0 regardless (acc=0 -> glu(0)=0), so this never relies on that tail invariant.

        In the DECLINE regime the host passes ``(l, 1, M)``: ``n_y = ceil(M/BLK_M)``, ``tile_y == tile_m``,
        ``tile_x == 0`` and ``native y == tile_m*BLK_M + m`` — exactly the flat (1, M) ColVecLoad read."""
        mMask = param  # native (l, Xg, Yg); mark_layout_dynamic -> Xg,Yg runtime
        blk_m = ctx.tile_M
        Yg = mMask.shape[2]
        n_y = cute.ceil_div(Yg, blk_m)  # runtime y-tiles per Xg (partial last tile allowed)
        tile_m = ctx.tile_coord_mnkl[0]
        tile_y = tile_m % n_y
        tile_x = tile_m // n_y
        gRow = mMask[ctx.batch_idx, tile_x, None]  # (Yg,) stride-1 row at x=tile_x
        # Register LAYOUT from the epilogue TV partition (matches tRS_rD); the gmem STRIDE of this
        # broadcast view is irrelevant (values come from the clamped gather below) — it only carries the
        # mask element_type into make_rmem_tensor. Same contract as the transposed branch.
        gLayout = cute.make_tensor(
            gRow.iterator,
            cute.make_layout((ctx.tile_M, ctx.tile_N), stride=self._broadcast_stride()),
        )
        tDgV = ctx.partition_for_epilogue_fn(gLayout)
        tDcV = ctx.partition_for_epilogue_fn(cute.make_identity_tensor((ctx.tile_M, ctx.tile_N)))
        if const_expr(ctx.tiled_copy_t2r is not None):
            tDgV = ctx.tiled_copy_r2s.retile(tDgV)
            tDcV = ctx.tiled_copy_r2s.retile(tDcV)
        tDrV = cute.make_rmem_tensor(tDgV.layout, tDgV.element_type)
        y0 = tile_y * blk_m
        last = Yg - 1
        zero = tDgV.element_type(0.0)
        for i in cutlass.range(cute.size(tDgV), unroll_full=True):
            native_y = y0 + tDcV[i][0]  # M-row within tile -> native y
            in_range = native_y < Yg
            src_y = native_y if in_range else last
            val = gRow[src_y]  # scalar contiguous gmem load, always in-bounds
            tDrV[i] = val if in_range else zero
        return tDrV


class ChunkedHalfBiasLoad(RowVecLoad):
    """Loads a half-width (N,) projection bias (bp OR bg) and scatters it into the (2N,)
    block-interleaved [up_G | gate_G] PREACT register layout, so the chunked dual-gated epilogue can
    add bp to the up-half registers and bg to the gate-half registers WITHOUT a host-side (2N,)
    interleaved-bias precompute (chunk_g>1 only; the (2N,) RowVecLoad still serves chunk_g==1).

    The CTA preact N-tile is BLK_N cols = nblk blocks of [up_G | gate_G]; the matching postact tile is
    BLK_N//2 cols.  For preact col n the postact col is (n//2G)*G + (n%2G clamped into [0,G)) -- up-reg
    j and its gate partner j+H share that postact col, so this single (N,) source feeds whichever
    register half the epilogue consumes.  The produced register tensor has the SAME T2R partition as
    the accumulator tRS_rD (VecLoad contract): tDrV[i] aligns with tRS_rD[i], reproducing the old
    (2N,) rowvec add bit-for-bit (up regs += bp[pc], gate regs += bg[pc]).
    """

    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        """Load one half of a block-interleaved (chunked) bias for this tile.

        Used by the dual-gated layouts, where the bias covers a 2N-wide pre-activation laid out as
        alternating ``G``-wide up/gate blocks. Only ``chunk_g > 1`` takes this path; otherwise it is
        the plain row-vector load.

        Args:
            gemm: The kernel functor, read for ``chunk_g``.
            param: The bias vector, or None.
            smem_tensor: Unused.
            ctx: The :class:`EpiContext`.

        Returns:
            The register fragment, or None.
        """
        tDrV = None
        G = const_expr(getattr(gemm, "chunk_g", 1))
        if const_expr(param is not None and G > 1):
            mVec = self._get_gmem_vec(param, ctx)  # (N,) postact-width rowvec (bp or bg)
            post_tile_dim = const_expr(ctx.tile_N // 2)  # BLK_N//2 postact cols per CTA
            nblk = const_expr(post_tile_dim // G)  # [up_G|gate_G] blocks per CTA tile
            coord_idx = ctx.tile_coord_mnkl[1]
            gVec = cute.local_tile(
                mVec, (post_tile_dim,), (coord_idx,)
            )  # (post_tile_dim,) postact slice
            # VECTORIZED broadcast: the preact N is shaped (G, 2, nblk) where the up/gate "2" axis has
            # stride 0 -> up-reg j and gate-reg j+H both read the SAME postact col, contiguous (stride 1)
            # within a G-block and stride G across blocks.  So preact col n reads postact col
            # (n//2G)*G + n%G -- the chunked column map -- via a normal coalesced load (M broadcast
            # stride 0), NOT a per-register scalar gather.
            gB = cute.make_tensor(
                gVec.iterator,
                cute.make_layout((ctx.tile_M, (G, 2, nblk)), stride=(0, (1, 0, G))),
            )
            tDgV = ctx.partition_for_epilogue_fn(gB)
            tDcV = ctx.partition_for_epilogue_fn(
                cute.make_identity_tensor((ctx.tile_M, ctx.tile_N))
            )
            if const_expr(ctx.tiled_copy_t2r is not None):
                tDgV = ctx.tiled_copy_r2s.retile(tDgV)
                tDcV = ctx.tiled_copy_r2s.retile(tDcV)
            limit = min(mVec.shape[0] - coord_idx * post_tile_dim, post_tile_dim)
            twoG = const_expr(2 * G)
            tDrV = cute.make_rmem_tensor(tDgV.layout, gVec.element_type)
            zero = tDgV.element_type(0.0)
            # The same compile-time proof as `VecLoad.begin`; see `_vec_fills_every_tile`. This site
            # is reachable at ``chunk_g > 1``, which is in `ALG_FOLD_TUNING_SPACE` -- one autotuner
            # pick away -- so leaving it on the runtime test would put the same doubled body and the
            # same spill one tuning decision from production.
            if const_expr(_vec_fills_every_tile(mVec, post_tile_dim)):
                for i in cutlass.range(cute.size(tDgV), unroll_full=True):
                    tDrV[i] = tDgV[i]
            else:
                # NESTED, not `elif` -- see the same construct in `VecLoad.begin`.
                if limit >= post_tile_dim:
                    for i in cutlass.range(cute.size(tDgV), unroll_full=True):
                        tDrV[i] = tDgV[i]
                else:
                    for i in cutlass.range(cute.size(tDgV), unroll_full=True):
                        n_pre = tDcV[i][1]  # preact col within the CTA tile (0 .. BLK_N)
                        pc = (n_pre // twoG) * G + n_pre % G  # postact col in tile (gB's mapping)
                        if pc < limit:
                            tDrV[i] = tDgV[i]
                        else:
                            tDrV[i] = zero
        return tDrV


class TileStore(EpiOp):
    """Tile-sized output tensor stored via TMA (e.g. postact).

    Args:
        name: field name in EpilogueArguments/Params (e.g. "mPostAct")
        epi_tile_fn: optional (gemm, epi_tile) -> epi_tile for half-tile (GemmGated)
    """

    def __init__(self, name, epi_tile_fn=None):
        """Declare a second output tensor written by the epilogue.

        Args:
            name: The op's name; see :meth:`EpiOp.__init__`.
            epi_tile_fn: Optional callable remapping the epilogue subtile for THIS output, for a
                store whose tile differs from D's (a half-width post-activation, say). None keeps
                D's tile.
        """
        super().__init__(name)
        self.epi_tile_fn = epi_tile_fn

    def _tma_atom_key(self):
        """Params key holding this store's TMA atom.

        Returns:
            The key string. Derived from the name so several ``TileStore`` ops coexist without
            colliding.
        """
        return f"tma_atom_{self.name}"

    def _smem_layout_key(self):
        """Params key holding this store's staged SMEM layout.

        Returns:
            The key string.
        """
        return f"epi_{self.name}_smem_layout_staged"

    def _epi_tile_key(self):
        """Params key holding this store's epilogue tile.

        Returns:
            The key string.
        """
        return f"epi_tile_{self.name}"

    def param_fields(self):
        # Optional (default None): an absent (None) output tensor (e.g. the optional gate3 mPostAct3
        # on the W3=None path) lowers to all-None params; the smem/tma/store hooks below treat a None
        # tensor/smem_layout as "absent" so the trace is unchanged when the tensor is not provided.
        """The four params fields a tile store needs, all defaulting to None.

        Defaulting matters: an absent output tensor -- the optional gate3 post-activation, for
        instance -- must leave every one of them None so the whole store is compiled out.

        Returns:
            ``[(tma_atom_key, ...), (name, ...), (smem_layout_key, ...), (epi_tile_key, ...)]``.
        """
        return [
            (self._tma_atom_key(), object, None),
            (self.name, object, None),
            (self._smem_layout_key(), object, None),
            (self._epi_tile_key(), object, None),
        ]

    def to_params(self, gemm, args):
        """Build the TMA atom, SMEM layout and epilogue tile for this output, or all None.

        Args:
            gemm: The kernel functor, whose ``_make_tma_epi_atoms_and_tensors`` builds the atom.
            args: The ``EpilogueArguments``.

        Returns:
            A dict for the four params fields. Every entry is None when the tensor is absent, which
            is the state the rest of the machinery checks for.
        """
        tensor = getattr(args, self.name)
        if tensor is None:
            # Absent output (optional TileStore, e.g. gate3 mPostAct3 when W3 is None): all-None.
            return {
                self._tma_atom_key(): None,
                self.name: None,
                self._smem_layout_key(): None,
                self._epi_tile_key(): None,
            }
        epi_tile = self.epi_tile_fn(gemm, gemm.epi_tile) if self.epi_tile_fn else None
        tma_atom, tma_tensor, smem_layout, epi_tile_out = setup_epi_tensor(
            gemm, tensor, epi_tile=epi_tile
        )
        return {
            self._tma_atom_key(): tma_atom,
            self.name: tma_tensor,
            self._smem_layout_key(): smem_layout,
            self._epi_tile_key(): epi_tile_out,
        }

    def smem_bytes(self, arg_tensor, cta_tile_shape_mnk, epi_tile):
        """SMEM bytes this store needs per stage.

        Args:
            arg_tensor: The output tensor, or None.
            cta_tile_shape_mnk: The CTA tile.
            epi_tile: The epilogue subtile, remapped by ``epi_tile_fn`` when one was given.

        Returns:
            Bytes per stage, or 0 when the output is absent. An underestimate is a SMEM overrun at
            launch, not a tight fit.
        """
        if arg_tensor is None:
            return 0
        if self.epi_tile_fn is not None:
            epi_tile = self.epi_tile_fn(None, epi_tile)
        return cute.size(cute.shape(epi_tile)) * (arg_tensor.element_type.width // 8)

    def smem_struct_field(self, gemm, params):
        """The SMEM struct field for this store's staging buffer.

        Args:
            gemm: The kernel functor.
            params: The traced epilogue params.

        Returns:
            ``(field_name, field_type)``. When the output is absent the field is a **zero-length**
            range rather than nothing at all, so the struct's field set does not change shape
            between configurations.
        """
        smem_layout_key = self._smem_layout_key()
        # Absent (key missing) OR present-but-None (optional output not provided) -> 0-size dummy.
        if getattr(params, smem_layout_key, None) is None:
            return (f"s_{self.name}", cute.struct.MemRange[Float32, 0])
        return (
            f"s_{self.name}",
            cute.struct.Align[
                cute.struct.MemRange[
                    gemm.postact_dtype,
                    cute.cosize(getattr(params, smem_layout_key)),
                ],
                gemm.buffer_align_bytes,
            ],
        )

    def get_smem_tensor(self, gemm, params, storage_epi):
        """Slice this store's staging tensor out of the epilogue SMEM struct.

        Args:
            gemm: The kernel functor.
            params: The traced epilogue params.
            storage_epi: The epilogue's SMEM struct.

        Returns:
            The staging tensor, or None when the output is absent.
        """
        smem_layout_key = self._smem_layout_key()
        smem_layout = getattr(params, smem_layout_key, None)
        if smem_layout is None:
            return None
        return getattr(storage_epi, f"s_{self.name}").get_tensor(
            smem_layout.outer,
            swizzle=smem_layout.inner,
        )

    def tma_atoms(self, gemm, params):
        """This store's TMA atom, for the kernel's descriptor prefetch.

        Args:
            gemm: The kernel functor.
            params: The traced epilogue params.

        Returns:
            A one-element list, or empty when the output is absent.
        """
        tma_key = self._tma_atom_key()
        atom = getattr(params, tma_key, None)
        return [atom] if atom is not None else []


@cute.jit
def vec_multiply(gemm, tRS_rD, tDrColVec, tDrRowVec):
    """Multiply tRS_rD by colvec and/or rowvec in-place. Uses packed f32x2 on SM100+."""
    if const_expr(tDrColVec is not None):
        if const_expr(gemm.arch < 100):
            for i in cutlass.range(cute.size(tDrColVec), unroll_full=True):
                tRS_rD[i] *= tDrColVec[i]
        else:
            for i in cutlass.range(cute.size(tRS_rD) // 2, unroll_full=True):
                tRS_rD[2 * i], tRS_rD[2 * i + 1] = cute.arch.mul_packed_f32x2(
                    (tRS_rD[2 * i], tRS_rD[2 * i + 1]),
                    (tDrColVec[2 * i], tDrColVec[2 * i + 1]),
                )
    if const_expr(tDrRowVec is not None):
        if const_expr(gemm.arch < 100):
            for i in cutlass.range(cute.size(tDrRowVec), unroll_full=True):
                tRS_rD[i] *= tDrRowVec[i]
        else:
            for i in cutlass.range(cute.size(tRS_rD) // 2, unroll_full=True):
                tRS_rD[2 * i], tRS_rD[2 * i + 1] = cute.arch.mul_packed_f32x2(
                    (tRS_rD[2 * i], tRS_rD[2 * i + 1]),
                    (tDrRowVec[2 * i], tDrRowVec[2 * i + 1]),
                )


@cute.jit
def colvec_reduce_accumulate(gemm, tDrReduce, tRS_rInput, transform_fn=None, rScale=None):
    """Accumulate transform_fn(input) or input * rScale into a ColVecReduce buffer.

    If transform_fn is provided, accumulates transform_fn(input[i]).
    If rScale is provided, accumulates input[i] * rScale[i] (uses mul/fma for SM100).
    If neither, accumulates input directly (identity).
    """
    if const_expr(tDrReduce is not None):
        if const_expr(transform_fn is None):
            transform_fn = lambda x: x
        if const_expr(gemm.arch < 100):
            for i in cutlass.range(cute.size(tDrReduce), unroll_full=True):
                val = transform_fn(tRS_rInput[i])
                tDrReduce[i] += val * rScale[i] if const_expr(rScale is not None) else val
        else:
            tDrReduce_mn = layout_utils.convert_layout_zero_stride(tDrReduce, tDrReduce.layout)
            tRS_rInput_mn = layout_utils.convert_layout_zero_stride(tRS_rInput, tDrReduce.layout)
            if const_expr(rScale is not None):
                rScale_mn = layout_utils.convert_layout_zero_stride(rScale, tDrReduce.layout)
            for m in cutlass.range(cute.size(tDrReduce_mn, mode=[0]), unroll_full=True):
                inp = lambda n: (tRS_rInput_mn[m, 2 * n], tRS_rInput_mn[m, 2 * n + 1])
                val0 = transform_fn(inp(0))
                if const_expr(rScale is not None):
                    row_sum = cute.arch.mul_packed_f32x2(val0, (rScale_mn[m, 0], rScale_mn[m, 1]))
                else:
                    row_sum = val0
                for n in cutlass.range(1, cute.size(tDrReduce_mn, mode=[1]) // 2, unroll_full=True):
                    val = transform_fn(inp(n))
                    if const_expr(rScale is not None):
                        row_sum = cute.arch.fma_packed_f32x2(
                            val, (rScale_mn[m, 2 * n], rScale_mn[m, 2 * n + 1]), row_sum
                        )
                    else:
                        row_sum = cute.arch.add_packed_f32x2(val, row_sum)
                tDrReduce_mn[m, 0] += row_sum[0] + row_sum[1]


class ColVecReduce(EpiOp):
    """Column vector reduction: accumulates across N subtiles in registers,
    then warp-reduces and writes to gmem in epi_end.

    No smem. The accumulation itself happens in epi_visit_subtile (user code).
    This op handles the register allocation (begin), per-subtile slicing (begin_loop),
    and final warp reduction + gmem write (end).
    """

    def param_fields(self):
        """One untyped params field, defaulting to None.

        Returns:
            ``[(name, object, None)]``.
        """
        return [(self.name, object, None)]

    def to_params(self, gemm, args):
        """Copy the reduction output into the params with the stride assumption applied.

        Args:
            gemm: The kernel functor. Unused here.
            args: The ``EpilogueArguments``.

        Returns:
            ``{name: tensor}``.
        """
        return {self.name: assume_stride_divisibility(getattr(args, self.name))}

    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        """Allocate the per-tile column accumulator this op reduces into.

        Args:
            gemm: The kernel functor.
            param: The output tensor, or None.
            smem_tensor: Unused.
            ctx: The :class:`EpiContext`.

        Returns:
            The accumulator fragment, laid out with stride ``(1, 0)`` so every N position of a row
            aliases one element -- which is what makes the reduction across N free. None when the
            term is absent.
        """
        tDrReduce = None
        if const_expr(param is not None):
            colvec_mma_layout = cute.make_layout((ctx.tile_M, ctx.tile_N), stride=(1, 0))
            tDrReduce_layout = ctx.partition_for_epilogue_fn(
                cute.make_rmem_tensor(colvec_mma_layout, Float32)
            ).layout
            tDrReduce = cute.make_rmem_tensor(tDrReduce_layout, Float32)
            cute.filter_zeros(tDrReduce).fill(0.0)
        return tDrReduce

    @cute.jit
    def begin_loop(self, gemm, state, epi_coord):
        """Narrow the accumulator to the current epilogue subtile.

        Args:
            gemm: The kernel functor. Unused here.
            state: Whatever :meth:`begin` returned.
            epi_coord: This subtile's coordinate.

        Returns:
            The subtile's slice of the accumulator, or None.
        """
        result = None
        if const_expr(state is not None):
            result = cute.group_modes(state, 3, cute.rank(state))[None, None, None, epi_coord]
        return result

    @cute.jit
    def end(
        self,
        gemm,
        param,
        state,
        epi_tile,
        tiled_copy_t2r,
        tiled_copy_r2s,
        tile_coord_mnkl,
        tidx,
    ):
        """Intra-warp shuffle reduction across N lanes, then direct gmem write."""
        if const_expr(param is not None):
            tDrReduce = state
            tiled_copy = tiled_copy_t2r if tiled_copy_t2r is not None else tiled_copy_r2s
            reference_src = tiled_copy_t2r is None

            # ── Derive lane layout from tiled_copy ──
            lane_layout_MN, warp_layout_MN = _get_lane_warp_layouts(tiled_copy, reference_src)
            # For ColVecReduce: reduce across N lanes (lanes_in_N threads share same M row)
            lanes_in_N = cute.size(lane_layout_MN, mode=[1])
            # Typically lanes_in_N is 4 for Sm90
            assert lanes_in_N == 1 << int(math.log2(lanes_in_N)), (
                "lanes_in_N must be a power of 2 for butterfly reduction"
            )

            # ── Intra-warp shuffle reduction across N lanes ──
            if const_expr(lanes_in_N > 1):
                assert lane_layout_MN.stride[1] == 1
                tDrReduce_flt = cute.filter_zeros(tDrReduce)
                for i in cutlass.range(cute.size(tDrReduce_flt), unroll_full=True):
                    tDrReduce_flt[i] = cute.arch.warp_reduction(
                        tDrReduce_flt[i], operator.add, threads_in_group=lanes_in_N
                    )

            warp_N = warp_layout_MN[1]
            assert cute.size(warp_N) == 1, (
                "ColVecReduce assumes all reduction cols are within the same warp"
            )

            # ── Direct gmem write (no inter-warp reduction needed: warps_in_N == 1) ──
            partition_for_epilogue_fn = partial(
                partition_for_epilogue,
                epi_tile=epi_tile,
                tiled_copy=tiled_copy,
                tidx=tidx,
                reference_src=tiled_copy_t2r is None,
            )
            tile_M, tile_N = gemm.cta_tile_shape_mnk[:2]
            batch_idx = tile_coord_mnkl[3]
            limit_n = param.shape[2]
            if tile_coord_mnkl[1] < limit_n:
                mColVec = param[batch_idx, None, tile_coord_mnkl[1]]
                gColVec = cute.local_tile(mColVec, (tile_M,), (tile_coord_mnkl[0],))
                limit_m = min(mColVec.shape[0] - tile_coord_mnkl[0] * tile_M, tile_M)
                tDcD = partition_for_epilogue_fn(cute.make_identity_tensor((tile_M, tile_N)))
                tDrReduce_m = layout_utils.convert_layout_zero_stride(tDrReduce, tDrReduce.layout)[
                    None, 0
                ]
                tDcD_m = layout_utils.convert_layout_zero_stride(tDcD, tDrReduce.layout)[None, 0]
                if tDcD_m[0][1] == 0:
                    for m in cutlass.range(cute.size(tDcD_m, mode=[0])):
                        row_idx = tDcD_m[m][0]
                        if row_idx < limit_m:
                            gColVec[row_idx] = tDrReduce_m[m]

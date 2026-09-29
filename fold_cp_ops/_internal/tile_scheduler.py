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

from typing import NamedTuple, Tuple, Optional
from dataclasses import dataclass
from enum import IntEnum

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32, Boolean, const_expr

import fold_cp_ops._internal.utils as utils
from fold_cp_ops._internal.fast_math import FastDivmod
from fold_cp_ops._internal.pipeline import PipelineStateWAdvance
from fold_cp_ops._internal.runtime_params import mlir_namedtuple


class RasterOrderOption(IntEnum):
    """How the caller asks for the CTA rasterization order.

    ``Heuristic`` is resolved to a concrete :class:`RasterOrder` at trace time by
    :func:`get_raster_order_from_option`, so nothing downstream ever sees it.
    """

    AlongM = 0
    AlongN = 1
    Heuristic = 2  # Pick AlongM if tiles_n > tiles_m, else AlongN


class RasterOrder(IntEnum):
    """The resolved rasterization order: which of M/N the linear work index varies fastest along.

    Purely an L2-locality choice -- every order visits the same tiles and produces the same result.
    """

    AlongM = 0
    AlongN = 1


class PersistenceMode(IntEnum):
    """How work tiles are handed out.

    ``NONE`` launches one CTA per tile. ``STATIC`` launches one resident wave that walks a
    precomputed range. ``DYNAMIC`` hands out tiles through a GMEM atomic, which balances uneven
    tiles at the cost of one atomic each.

    There is deliberately no cluster-launch-control mode: CLC is Blackwell hardware, this
    package is SM90-only, and a mode that can never be selected is a branch every reader has
    to rule out.
    """

    NONE = 0
    STATIC = 1
    DYNAMIC = 2


@cute.jit
def get_raster_order_from_option(
    raster_order_option: RasterOrderOption, problem_shape_ncluster_mn: cute.Shape, group_size: Int32
) -> RasterOrder:
    """Resolve a :class:`RasterOrderOption` into a concrete :class:`RasterOrder`.

    Args:
        raster_order_option: What the caller asked for. ``Heuristic`` picks the order that makes the
            SHORT dimension the fast-varying one, so a group of consecutive work indices covers a
            compact rectangle rather than a long strip.
        problem_shape_ncluster_mn: The problem in clusters, ``(m, n)``.
        group_size: The swizzle group width, which bounds how many clusters the heuristic considers
            adjacent.

    Returns:
        ``RasterOrder.AlongM`` or ``RasterOrder.AlongN``.
    """
    raster_order = (
        RasterOrder.AlongM
        if raster_order_option == RasterOrderOption.AlongM
        else RasterOrder.AlongN
    )
    if raster_order_option == RasterOrderOption.Heuristic:
        problem_blocks_m = cute.round_up(problem_shape_ncluster_mn[0], group_size)
        problem_blocks_n = cute.round_up(problem_shape_ncluster_mn[1], group_size)
        raster_order = (
            RasterOrder.AlongM if problem_blocks_n > problem_blocks_m else RasterOrder.AlongN
        )
    return raster_order


# Grouping arguments together that should be passed to __call__
@mlir_namedtuple
class TileSchedulerOptions(NamedTuple):
    """Scheduler knobs as passed to ``__call__``.

    Attributes:
        max_active_clusters: The persistent grid size in clusters; 0 for a non-persistent launch.
        raster_order: Compile-time rasterization choice.
        max_swizzle_size: Swizzle group width for L2 locality.
        tile_count_semaphore: GMEM atomic counter for ``DYNAMIC``, else None. **Must be zeroed
            before every launch** -- it is a ticket dispenser, and a stale value makes the grid skip
            tiles.
        batch_idx_permute: Optional reordering of the batch visit order; scheduling only.

    Note:
        The ``Constexpr`` fields are baked in at compile time and their ABI slots erased, so at
        launch they must be passed as None -- see ``gemm_tvm_ffi_utils.make_scheduler_args``.
    """

    max_active_clusters: Int32
    raster_order: cutlass.Constexpr[RasterOrderOption] = RasterOrderOption.Heuristic
    max_swizzle_size: Int32 = Int32(8)
    tile_count_semaphore: Optional[cute.Pointer] = None
    batch_idx_permute: Optional[cute.Tensor] = None
    # Contiguous-j-run order (STATIC scheduler only): when > 0, each persistent CTA produces a
    # CONTIGUOUS RUN of `run_j_tiles` j-clusters for a fixed i-band within one plane (instead of
    # the default L2-reuse serpentine swizzle), so a downstream decoupled store can accumulate a
    # contiguous recv-row chunk. 0 (default) = today's order, byte-identical. Requires
    # run_j_tiles | ncluster_n (asserted in Params.create). Compile-time constant.
    run_j_tiles: cutlass.Constexpr[int] = 0
    # DYNAMIC-N contiguous-j-run (run_j_dynamic=True): the run-length is the RUNTIME ncluster_n
    # (== the full band, so runs_per_band==1) rather than a baked constexpr — for a dynamic-shape
    # store that owns one full (tile_m, N) band per persistent CTA at a token count N not known at
    # compile time. Mutually exclusive with run_j_tiles>0 (which is the static baked run). 0/False
    # (default) leaves BOTH off -> the default serpentine order, byte-identical.
    run_j_dynamic: cutlass.Constexpr[bool] = False
    # Contiguous-I-run order (the TRANSPOSE of run_j_tiles; STATIC): when > 0, each persistent CTA
    # produces a CONTIGUOUS RUN of `run_i_tiles` i-clusters (M-tiles) for a fixed J-BAND (N-cluster)
    # within one plane. The FRONT wide-put decoupled drain needs this axis (it coalesces along TOKEN =
    # GEMM-M, at a FIXED feature = GEMM-N), where run_j (fixed-M, run-N) is the back's axis. BOUNDED
    # (not the full M-band): the front's band axis is FEATURE (ncluster_n small ~16) so a full-M-band-
    # per-CTA (a "run_i_dynamic") would leave only ncluster_n CTAs active — an occupancy collapse; the
    # bounded run (R M-tiles/CTA, R chosen host-side = min(knee, ncluster_m·ncluster_n/grid_z)) keeps
    # ncluster_n·ceil(ncluster_m/R) runs → full occupancy AND an R-long consecutive-M run. Safe for
    # ARBITRARY run_i_tiles (no run_i_tiles | ncluster_m constraint) via the clamped-padding decode
    # (mirror of the run_j b275675 fix): the last run of a band pads i past ncluster_m → clamp to the
    # last valid i-cluster (idempotent re-store) + gate is_valid on run_global ALONE (monotonic). 0
    # (default) = today's order, byte-identical. Mutually exclusive with run_j_tiles/run_j_dynamic.
    run_i_tiles: cutlass.Constexpr[int] = 0


@dataclass
class TileSchedulerArguments:
    """Everything the scheduler needs to derive its ``Params``, before any runtime math.

    Attributes:
        problem_shape_ntile_mnl: The problem in TILES, ``(m, n, l)``.
        raster_order: The requested rasterization order.
        group_size: Swizzle group width.
        cluster_shape_mnk: The cluster shape. Its K extent must be 1; the scheduler asserts it.
        tile_count_semaphore: GMEM counter for ``DYNAMIC``, else None.
        batch_idx_permute: Optional batch reordering.
        persistence_mode: Which :class:`PersistenceMode` to build for.
    """

    problem_shape_ntile_mnl: cute.Shape
    raster_order: cutlass.Constexpr[RasterOrderOption]
    group_size: Int32
    cluster_shape_mnk: cutlass.Constexpr[cute.Shape]
    tile_count_semaphore: Optional[cute.Pointer] = None
    batch_idx_permute: Optional[cute.Tensor] = None
    persistence_mode: cutlass.Constexpr[PersistenceMode] = PersistenceMode.NONE
    run_j_tiles: cutlass.Constexpr[int] = 0
    run_j_dynamic: cutlass.Constexpr[bool] = False
    run_i_tiles: cutlass.Constexpr[int] = 0
    # DYNAMIC-N contiguous-I-run (run_i_dynamic=True): the run-length is a RUNTIME Int32 (run_i_run_len)
    # rather than the baked constexpr run_i_tiles — for the wide-put front A2A under dynamic-shape compile
    # (ncluster_m is a runtime value, so the occupancy-safe / route2-divisor R must be derived at runtime).
    # Mutually exclusive with run_i_tiles>0 (the static baked run). Off (default) -> run_i_run_len unused.
    run_i_dynamic: cutlass.Constexpr[bool] = False
    run_i_run_len: Optional[Int32] = None


class TileScheduler:
    """Hands out work tiles to a persistent grid, in an L2-friendly order.

    A persistent GEMM launches one resident wave and loops. This class is what turns "which
    iteration am I on" into a ``(pid_m, pid_n, batch)`` coordinate, and it is where the swizzled
    rasterization lives: consecutive work indices are grouped so that the tiles a wave works on at
    the same time share operand rows and columns in L2.

    The state is per-CTA and crosses the host -> kernel boundary, so the two MLIR protocol methods
    at the bottom carry it. Subclasses override the delinearization (and sometimes the grid shape)
    while reusing everything else.
    """

    @dataclass
    class Params:
        """The scheduler's traced state: shapes in clusters, plus precomputed fast divisors.

        The ``FastDivmod`` fields are the point: delinearizing a work index needs several integer
        divisions per tile, and a magic-number reciprocal computed once on the host turns each into
        a multiply-high.
        """

        problem_shape_ncluster_mnl: cute.Shape
        raster_order: RasterOrder
        num_clusters_per_problem_fdd: FastDivmod
        num_groups_regular: Int32
        group_size_fdd: FastDivmod
        group_size_tail_fdd: FastDivmod
        num_clusters_in_group_fdd: FastDivmod
        tile_count_semaphore: Optional[cute.Pointer]
        batch_idx_permute: Optional[cute.Tensor]
        cluster_shape_mn: cutlass.Constexpr[cute.Shape]
        persistence_mode: cutlass.Constexpr[PersistenceMode]
        # Contiguous-j-run mode (run_j_tiles>0 STATIC, or run_j_dynamic DYNAMIC-N). All host/trace-
        # computed; default-off path leaves both off (the run-decode branch is const_expr-elided).
        run_j_tiles: cutlass.Constexpr[int]
        runs_per_band_fdd: Optional[
            FastDivmod
        ]  # ceil(ncluster_n / run_j_tiles); ==1 when run_j_dynamic
        ncluster_m_fdd: Optional[FastDivmod]  # band -> (plane, i-band): m = band % ncluster_m
        ncluster_n_run: Int32  # for the j < ncluster_n validity guard
        total_runs: Int32  # L * ncluster_m * runs_per_band
        # DYNAMIC-N run (run_j_dynamic): run-length == the RUNTIME ncluster_n (runs_per_band==1), so the
        # decode's c//R, c%R become divmod(c, run_len_fdd). run_len_fdd is None on the static run_j path.
        run_j_dynamic: cutlass.Constexpr[bool]
        run_len_fdd: Optional[FastDivmod]
        # Contiguous-I-run mode (run_i_tiles>0 STATIC, the TRANSPOSE of run_j_tiles). Reuses
        # runs_per_band_fdd (= ceil(ncluster_m / run_i_tiles)) + total_runs (= L·ncluster_n·runs_per_band),
        # but decodes band -> (plane, J-BAND=cid_n) via ncluster_n_fdd and clamps padding i via
        # ncluster_m_run. None/0 on the run_j path (the run_i decode branch is const_expr-elided).
        run_i_tiles: cutlass.Constexpr[int]
        ncluster_n_fdd: Optional[FastDivmod]  # band -> (plane, j-band): cid_n = band % ncluster_n
        ncluster_m_run: Int32  # for the i-clamp: cid_m = min(i, ncluster_m_run - 1)
        # DYNAMIC-N run_i (run_i_dynamic): the run-length R is a RUNTIME Int32 -> the decode's c//R, c%R
        # become divmod(c, run_i_len_fdd) and q*R uses run_i_len. None/0 on the static run_i path (elided).
        run_i_dynamic: cutlass.Constexpr[bool]
        run_i_len_fdd: Optional[FastDivmod]
        run_i_len: Int32

        @staticmethod
        @cute.jit
        def create(args: TileSchedulerArguments, *, loc=None, ip=None) -> "TileScheduler.Params":
            """Derive the dense scheduler's traced params, precomputing every fast divisor.

            Args:
                args: The scheduler arguments. ``cluster_shape_mnk[2]`` must be 1 -- clusters do not
                    extend along K, and the assertion catches a caller that assumed they might.
                loc: Optional DSL source location.
                ip: Optional DSL insertion point.

            Returns:
                The ``Params``.

            Raises:
                AssertionError: If the cluster's K extent is not 1.
            """
            assert args.cluster_shape_mnk[2] == 1
            cluster_shape_mn = const_expr(cute.select(args.cluster_shape_mnk, mode=[0, 1]))
            problem_shape_ntile_mn = cute.select(args.problem_shape_ntile_mnl, mode=[0, 1])
            problem_shape_ncluster_mn = cute.ceil_div(problem_shape_ntile_mn, cluster_shape_mn)
            problem_shape_ncluster_mnl = problem_shape_ncluster_mn + (
                args.problem_shape_ntile_mnl[2],
            )
            num_clusters_per_problem = cute.size(problem_shape_ncluster_mn)
            raster_order = get_raster_order_from_option(
                args.raster_order, problem_shape_ncluster_mn, args.group_size
            )
            ncluster_fast = (
                problem_shape_ncluster_mn[0]
                if raster_order == RasterOrder.AlongM
                else problem_shape_ncluster_mn[1]
            )
            ncluster_slow = (
                problem_shape_ncluster_mn[1]
                if raster_order == RasterOrder.AlongM
                else problem_shape_ncluster_mn[0]
            )
            group_size = min(args.group_size, ncluster_fast)
            group_size_tail = ncluster_fast % group_size
            num_groups_regular = ncluster_fast // group_size
            num_clusters_in_group = group_size * ncluster_slow
            if const_expr(args.persistence_mode == PersistenceMode.DYNAMIC):
                assert args.tile_count_semaphore is not None
            # ---- contiguous-j-run mode setup (compile-time gated) ----
            # args.run_j_tiles is ConstNone-erased at the tvm-ffi call boundary (see
            # gemm_tvm_ffi_utils.make_scheduler_args) — the const value is baked from the FAKE args
            # (0). The bitcode route, however, traces Params.create with the REAL args (None), so
            # normalize None -> 0 (the off value) to keep `> 0` valid on every compile path.
            run_j_tiles = const_expr(args.run_j_tiles if args.run_j_tiles is not None else 0)
            run_j_dynamic = const_expr(
                args.run_j_dynamic if getattr(args, "run_j_dynamic", None) is not None else False
            )
            run_i_tiles = const_expr(
                args.run_i_tiles if getattr(args, "run_i_tiles", None) is not None else 0
            )
            run_i_dynamic = const_expr(
                args.run_i_dynamic if getattr(args, "run_i_dynamic", None) is not None else False
            )
            runs_per_band_fdd = None
            ncluster_m_fdd = None
            ncluster_n_run = Int32(0)
            total_runs = Int32(0)
            run_len_fdd = None
            ncluster_n_fdd = None
            ncluster_m_run = Int32(0)
            run_i_len_fdd = None
            run_i_len = Int32(0)
            if const_expr(run_j_tiles > 0 or run_j_dynamic):
                assert args.persistence_mode == PersistenceMode.STATIC, (
                    "run_j contiguous-j-run order requires the STATIC persistent scheduler"
                )
                ncluster_m = problem_shape_ncluster_mn[0]
                ncluster_n = problem_shape_ncluster_mn[1]
                L = args.problem_shape_ntile_mnl[2]
                if const_expr(run_j_dynamic):
                    # DYNAMIC-N: run-length == the RUNTIME ncluster_n (the full band) -> runs_per_band
                    # == 1 (const, never a ceil_div(runtime,const)); the decode's c//R, c%R become
                    # divmod(c, run_len_fdd = FastDivmod(ncluster_n)). run_j_tiles stays 0 on this path.
                    runs_per_band = 1
                    run_len_fdd = FastDivmod(ncluster_n)
                else:
                    runs_per_band = cute.ceil_div(ncluster_n, run_j_tiles)
                runs_per_band_fdd = FastDivmod(runs_per_band)
                ncluster_m_fdd = FastDivmod(ncluster_m)
                ncluster_n_run = Int32(ncluster_n)
                total_runs = Int32(L * ncluster_m * runs_per_band)
            # ---- contiguous-I-run mode setup (the TRANSPOSE; STATIC baked R or DYNAMIC-N runtime R) ----
            if const_expr(run_i_tiles > 0 or run_i_dynamic):
                assert args.persistence_mode == PersistenceMode.STATIC, (
                    "run_i contiguous-i-run order requires the STATIC persistent scheduler"
                )
                assert not (run_j_tiles > 0 or run_j_dynamic), (
                    "run_i_tiles is mutually exclusive with run_j_tiles / run_j_dynamic"
                )
                ncluster_m = problem_shape_ncluster_mn[0]
                ncluster_n = problem_shape_ncluster_mn[1]
                L = args.problem_shape_ntile_mnl[2]
                # runs per J-BAND, walking i=M: ceil(ncluster_m / R). Arbitrary R is SAFE (no R | ncluster_m
                # constraint) — the last run pads i past ncluster_m and the decode CLAMPS it (idempotent
                # re-store) + gates is_valid on run_global (monotonic). route2 wide-put snaps R | ncm host/
                # runtime-side so the clamp is a no-op (no dup-absorb -> no b_j deadlock).
                if const_expr(run_i_dynamic):
                    # DYNAMIC-N: R is a RUNTIME Int32 (run_i_run_len). SCALAR ceil_div (both runtime; NOT
                    # cute.ceil_div -> that builds a cute.tile op). run_i_len_fdd = FastDivmod(R) drives the
                    # decode c//R, c%R; run_i_len carries R for q*R + the drain my_tiles count-parity.
                    R_i = Int32(args.run_i_run_len)
                    runs_per_band = (ncluster_m + R_i - Int32(1)) // R_i
                    run_i_len_fdd = FastDivmod(R_i)
                    run_i_len = R_i
                else:
                    runs_per_band = cute.ceil_div(ncluster_m, run_i_tiles)
                    run_i_len = Int32(run_i_tiles)
                runs_per_band_fdd = FastDivmod(runs_per_band)
                ncluster_n_fdd = FastDivmod(ncluster_n)  # band -> (plane, j-band=cid_n)
                ncluster_m_run = Int32(ncluster_m)  # for the i-clamp
                total_runs = Int32(L * ncluster_n * runs_per_band)
            return TileScheduler.Params(
                problem_shape_ncluster_mnl,
                raster_order,
                FastDivmod(num_clusters_per_problem),
                num_groups_regular,
                FastDivmod(group_size),
                # Don't divide by 0
                FastDivmod(group_size_tail if group_size_tail > 0 else 1),
                FastDivmod(num_clusters_in_group),
                args.tile_count_semaphore
                if const_expr(args.persistence_mode == PersistenceMode.DYNAMIC)
                else None,
                args.batch_idx_permute,
                cluster_shape_mn,
                args.persistence_mode,
                run_j_tiles,
                runs_per_band_fdd,
                ncluster_m_fdd,
                ncluster_n_run,
                total_runs,
                run_j_dynamic,
                run_len_fdd,
                run_i_tiles,
                ncluster_n_fdd,
                ncluster_m_run,
                run_i_dynamic,
                run_i_len_fdd,
                run_i_len,
            )

    def __init__(
        self,
        current_work_idx: Int32,
        num_tiles_executed: Int32,
        current_batch_idx: Int32,
        num_work_idx_before_cur_batch: Int32,
        sched_smem: Optional[cute.Tensor],
        scheduler_pipeline: Optional[cutlass.pipeline.PipelineAsync],
        pipeline_state: PipelineStateWAdvance,
        params: Params,
        *,
        loc=None,
        ip=None,
    ):
        """Bind this CTA's scheduler state.

        Args:
            current_work_idx: The linear work index this CTA starts at.
            num_tiles_executed: How many tiles it has completed; 0 at launch.
            current_batch_idx: The batch it is currently in.
            num_work_idx_before_cur_batch: Work indices consumed by earlier batches, so the
                per-batch offset need not be recomputed each tile.
            sched_smem: SMEM slice for the cluster-wide work broadcast, or None when each CTA
                schedules for itself.
            scheduler_pipeline: Pipeline guarding that broadcast, or None.
            is_scheduler_warp: Whether this warp is the one computing and broadcasting work tiles.
                Exactly one warp per cluster may pass True.
            params: The traced ``Params``.
            pipeline_state: The broadcast pipeline's state, or None.
            loc: Optional DSL source location, stashed for the rebuild.
            ip: Optional DSL insertion point.
        """
        self._current_work_idx = current_work_idx
        self.num_tiles_executed = num_tiles_executed
        self._current_batch_idx = current_batch_idx
        self._num_work_idx_before_cur_batch = num_work_idx_before_cur_batch
        self._sched_smem = sched_smem
        self._scheduler_pipeline = scheduler_pipeline
        self._pipeline_state = pipeline_state
        self.params = params
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(args: TileSchedulerArguments, *, loc=None, ip=None) -> Params:
        """Derive the traced ``Params`` from the caller's arguments.

        Args:
            args: The scheduler arguments.
            loc: Optional DSL source location.
            ip: Optional DSL insertion point.

        Returns:
            The ``Params``.
        """
        return TileScheduler.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    @cute.jit
    def _cluster_idx_to_work_idx_batch(
        params: Params, cluster_idx: Tuple[Int32, Int32, Int32], *, loc=None, ip=None
    ) -> Tuple[Int32, Optional[Int32]]:
        """Map this CTA's cluster coordinate to its starting work index and batch.

        Args:
            params: The traced ``Params``, whose persistence mode decides the mapping.
            cluster_idx: This CTA's ``(x, y, z)`` cluster coordinate.
            loc: Optional DSL source location.
            ip: Optional DSL insertion point.

        Returns:
            ``(current_work_idx, batch_idx)``. Under a non-persistent grid the coordinate IS the
            tile, so the batch comes straight from z; under a persistent one the batch is derived
            later during delinearization and None is returned for it.
        """
        if const_expr(params.persistence_mode == PersistenceMode.NONE):
            current_work_idx = Int32(cluster_idx[0])
            batch_idx = Int32(cluster_idx[2])
            return current_work_idx, batch_idx
        else:
            current_work_idx = Int32(cluster_idx[2])
            batch_idx = None
            return current_work_idx, batch_idx

    @staticmethod
    @cute.jit
    def create(
        params: Params,
        sched_smem: Optional[cute.Tensor] = None,
        scheduler_pipeline: Optional[cutlass.pipeline.PipelineAsync] = None,
        is_scheduler_warp: bool | Boolean = False,
        *,
        loc=None,
        ip=None,
    ) -> "TileScheduler":
        """is_scheduler_warp should only be true for one warp in the whole cluster"""
        current_work_idx, _ = TileScheduler._cluster_idx_to_work_idx_batch(
            params, cute.arch.cluster_idx(), loc=loc, ip=ip
        )
        stages = 0
        if const_expr(params.persistence_mode in [PersistenceMode.STATIC, PersistenceMode.DYNAMIC]):
            assert sched_smem is not None
            assert scheduler_pipeline is not None
            stages = const_expr(cute.size(sched_smem, mode=[1]))
        return TileScheduler(
            current_work_idx,
            Int32(0),  # num_tiles_executed
            Int32(0),  # current_batch_idx
            Int32(0),  # num_work_idx_before_cur_batch
            sched_smem,
            scheduler_pipeline,
            PipelineStateWAdvance(stages, Int32(0), Int32(0), Int32(0)),
            params,
            loc=loc,
            ip=ip,
        )

    # called by host
    @staticmethod
    def get_grid_shape(
        params: Params,
        max_active_clusters: Int32,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        """The launch grid, in CTAs, for this scheduler configuration.

        Args:
            params: The traced ``Params``.
            max_active_clusters: How many clusters the device holds resident. Used only by the
                persistent modes; a non-persistent grid is sized from the problem instead.
            loc: Optional DSL source location.
            ip: Optional DSL insertion point.

        Returns:
            ``(grid_x, grid_y, grid_z)`` in CTAs.
        """
        if const_expr(params.persistence_mode == PersistenceMode.NONE):
            return (
                params.cluster_shape_mn[0] * cute.size(params.problem_shape_ncluster_mnl[:2]),
                params.cluster_shape_mn[1],
                params.problem_shape_ncluster_mnl[2],
            )
        else:
            num_ctas_in_problem = cute.size(
                params.problem_shape_ncluster_mnl, loc=loc, ip=ip
            ) * cute.size(params.cluster_shape_mn)
            num_ctas_per_cluster = cute.size(params.cluster_shape_mn, loc=loc, ip=ip)
            # Total ctas that can run in one wave
            num_ctas_per_wave = max_active_clusters * num_ctas_per_cluster
            num_persistent_ctas = cutlass.min(num_ctas_in_problem, num_ctas_per_wave)
            num_persistent_clusters = num_persistent_ctas // num_ctas_per_cluster
            return (*params.cluster_shape_mn, num_persistent_clusters)

    @cute.jit
    def _swizzle_cta(
        self, cluster_id_in_problem: Int32, *, loc=None, ip=None
    ) -> Tuple[Int32, Int32]:
        # CTA Swizzle to promote L2 data reuse
        """Map a linear cluster index to a swizzled ``(cid_m, cid_n)``, for L2 reuse.

        Consecutive work indices are grouped ``group_size`` at a time along the slow axis, so a wave
        covers a compact rectangle rather than a full row. The tail group is narrower than the rest,
        which is why there are two fast divisors rather than one.

        Args:
            cluster_id_in_problem: Linear cluster index within this batch.
            loc: Optional DSL source location.
            ip: Optional DSL insertion point.

        Returns:
            ``(cid_m, cid_n)`` in clusters.
        """
        params = self.params
        group_id, id_in_group = divmod(cluster_id_in_problem, params.num_clusters_in_group_fdd)
        cid_fast_in_group, cid_slow = Int32(0), Int32(0)
        if group_id < params.num_groups_regular:
            cid_slow, cid_fast_in_group = divmod(id_in_group, params.group_size_fdd)
        else:  # tail part
            cid_slow, cid_fast_in_group = divmod(id_in_group, params.group_size_tail_fdd)
        if group_id % 2 == 1:  # serpentine order
            ncluster_slow = (
                params.problem_shape_ncluster_mnl[1]
                if params.raster_order == RasterOrder.AlongM
                else params.problem_shape_ncluster_mnl[0]
            )
            cid_slow = ncluster_slow - 1 - cid_slow
        cid_fast = group_id * params.group_size_fdd.divisor + cid_fast_in_group
        cid_m, cid_n = cid_fast, cid_slow
        if params.raster_order == RasterOrder.AlongN:
            cid_m, cid_n = cid_slow, cid_fast
        return cid_m, cid_n

    @cute.jit
    def _cluster_id_to_cta_id(
        self, cid_m: Int32, cid_n: Int32, *, block_zero_only: bool = False, loc=None, ip=None
    ) -> Tuple[Int32, Int32]:
        """Convert a cluster coordinate to this CTA's tile coordinate within it.

        Args:
            cid_m: Cluster index along M.
            cid_n: Cluster index along N.
            block_zero_only: Report the coordinate of the cluster's CTA 0 rather than of this CTA.
                Used when one warp computes a tile on behalf of the whole cluster.
            loc: Optional DSL source location.
            ip: Optional DSL insertion point.

        Returns:
            ``(pid_m, pid_n)`` in tiles.
        """
        if const_expr(block_zero_only):
            bidx_in_cluster = (Int32(0), Int32(0))
        else:
            # Get the pid from cluster id
            bidx_in_cluster = cute.arch.block_in_cluster_idx()
        pid_m = cid_m * self.params.cluster_shape_mn[0] + bidx_in_cluster[0]
        pid_n = cid_n * self.params.cluster_shape_mn[1] + bidx_in_cluster[1]
        return pid_m, pid_n

    @cute.jit
    def _delinearize_work_idx(
        self,
        work_idx: Int32,
        bidz: Optional[Int32] = None,
        is_valid: Optional[Boolean] = None,
        *,
        block_zero_only: bool = False,
        loc=None,
        ip=None,
    ) -> cutlass.utils.WorkTileInfo:
        """Turn a linear work index into a ``WorkTileInfo``: batch, tile coordinate, validity.

        The scheduler's core. Divides out the batch, swizzles what remains into a cluster
        coordinate, and offsets to this CTA's tile.

        Args:
            work_idx: The linear work index.
            bidz: Batch index when the caller already knows it (the non-persistent grid), else None
                to derive it.
            is_valid: Validity when the caller already knows it, else None to derive it. A
                persistent grid runs past the end of the work, so most tiles' validity is a
                comparison rather than a constant.
            block_zero_only: Report the cluster's CTA-0 tile rather than this CTA's.
            loc: Optional DSL source location.
            ip: Optional DSL insertion point.

        Returns:
            A ``WorkTileInfo``. An invalid one must be skipped, not clamped -- its coordinate is
            outside the problem.
        """
        params = self.params
        # ---- contiguous-j-run order (STATIC only, run_j_tiles>0): each persistent CTA fills a
        # CONTIGUOUS RUN of run_j_tiles j-clusters for a fixed i-band within one plane. The CTA's
        # persistent work_idx is z + c*gz (z=grid-z cluster id, gz=#persistent clusters, c=tile
        # counter); we re-decode (z, c) into a band-padded run assignment that is a bijection over
        # [0, L*ncluster_m*ncluster_n) when run_j_tiles | ncluster_n (asserted). Default-off branch
        # below is byte-identical to today (this branch is const_expr-elided when run_j_tiles==0 AND
        # run_j_dynamic is False). run_j_dynamic takes the SAME decode with a runtime run-length.
        if const_expr(
            params.run_j_tiles > 0
            or params.run_j_dynamic
            or params.run_i_tiles > 0
            or params.run_i_dynamic
        ):
            return self._delinearize_work_idx_run(
                work_idx, bidz, is_valid, block_zero_only=block_zero_only, loc=loc, ip=ip
            )
        if const_expr(is_valid is None):
            if const_expr(params.persistence_mode == PersistenceMode.NONE):
                is_valid = self.num_tiles_executed == 0
            else:
                is_valid = work_idx < cute.size(params.problem_shape_ncluster_mnl)
        pid_m, pid_n, batch_idx = Int32(0), Int32(0), Int32(0)
        if is_valid:
            if const_expr(params.persistence_mode == PersistenceMode.NONE):
                cluster_id_in_problem = work_idx
                _, _, bidz_ = cute.arch.block_idx()
            else:
                bidz_, cluster_id_in_problem = divmod(work_idx, params.num_clusters_per_problem_fdd)
            if const_expr(bidz is not None):
                bidz_ = bidz
            cid_m, cid_n = self._swizzle_cta(cluster_id_in_problem, loc=loc, ip=ip)
            pid_m, pid_n = self._cluster_id_to_cta_id(
                cid_m, cid_n, block_zero_only=block_zero_only, loc=loc, ip=ip
            )
            batch_idx = (
                bidz_
                if const_expr(params.batch_idx_permute is None)
                else params.batch_idx_permute[bidz_]
            )
        tile_coord_mnkl = (pid_m, pid_n, None, batch_idx)
        return cutlass.utils.WorkTileInfo(tile_coord_mnkl, is_valid)

    @cute.jit
    def _delinearize_work_idx_run(
        self,
        work_idx: Int32,
        bidz: Optional[Int32] = None,
        is_valid: Optional[Boolean] = None,
        *,
        block_zero_only: bool = False,
        loc=None,
        ip=None,
    ) -> cutlass.utils.WorkTileInfo:
        """Contiguous-j-run decode (STATIC). work_idx = z + c*gz; emit (plane, i-band, j) so that a
        CTA's consecutive tiles walk j contiguously (run length = params.run_j_tiles) within one
        (plane, i-band). Bijection over all tiles when run_j_tiles | ncluster_n."""
        params = self.params
        gz = Int32(cute.arch.grid_dim()[2])  # num_persistent_clusters
        z = work_idx % gz
        c = work_idx // gz
        if const_expr(params.run_i_tiles > 0 or params.run_i_dynamic):
            # CONTIGUOUS-I-RUN (the TRANSPOSE of run_j): a CTA's consecutive tiles walk i=M contiguously
            # (run length = R) for a FIXED J-BAND (cid_n) within one plane -> consecutive-M at fixed-N =
            # the front wide-put's coalescing axis. Mutually exclusive with run_j_*. Arbitrary R is SAFE
            # via the b275675 clamp mirrored onto the i-axis (below). STATIC bakes R (const_expr); DYNAMIC-N
            # sources R from run_i_len_fdd/run_i_len (runtime) — bit-identical mapping, only the run-length
            # is runtime (route2 wide-put snaps R|ncm so the clamp is a no-op -> no dup-absorb -> no hang).
            if const_expr(params.run_i_dynamic):
                run_local, offset = divmod(
                    c, params.run_i_len_fdd
                )  # c//R, c%R via FastDivmod(runtime R)
                run_global = z + run_local * gz
                band, q = divmod(run_global, params.runs_per_band_fdd)
                i = q * params.run_i_len + offset
            else:
                R = const_expr(params.run_i_tiles)
                run_local = c // R
                offset = c % R
                run_global = z + run_local * gz
                # band = run_global // runs_per_band ; q = run within the band. runs_per_band =
                # ceil(ncluster_m / run_i_tiles) (runs to cover the M-column of one (plane, j-band)).
                band, q = divmod(run_global, params.runs_per_band_fdd)
                i = q * R + offset
            # plane (bidz) and J-band (cid_n) from band: p = band // ncluster_n, n = band % ncluster_n.
            bidz_, cid_n = divmod(band, params.ncluster_n_fdd)
            # CLAMP padding i (>= ncluster_m) to the last valid i-cluster (idempotent RE-STORE) — the
            # run_j b275675 fix on the i-axis. Padding appears only when run_i_tiles ∤ ncluster_m (the
            # last run of a band pads i past ncluster_m); clamping keeps is_valid gated on run_global
            # ALONE (monotonic) so the persistent loop's stop-on-first-invalid is exact and NO tile is
            # dropped, for ARBITRARY run_i_tiles. Dividing run_i_tiles -> i < ncluster_m -> clamp no-op.
            cid_m = cutlass.min(i, params.ncluster_m_run - 1)
            if const_expr(is_valid is None):
                is_valid = run_global < params.total_runs
        else:
            if const_expr(params.run_j_dynamic):
                # DYNAMIC-N run: run-length == the RUNTIME ncluster_n, runs_per_band == 1 -> band ==
                # run_global, j == offset. The ONLY delta vs the static branch is sourcing the run-length
                # from a FastDivmod (runtime) instead of the const R -- bit-identical mapping (gridwalk
                # spike: full==dyn + bijection). The static branch is LITERALLY unchanged (const-elided).
                run_local, offset = divmod(c, params.run_len_fdd)
                run_global = z + run_local * gz
                band = run_global
                j = offset
            else:
                R = const_expr(params.run_j_tiles)
                run_local = c // R
                offset = c % R
                run_global = z + run_local * gz
                # band = run_global // runs_per_band ; q = run_global % runs_per_band
                band, q = divmod(run_global, params.runs_per_band_fdd)
                j = q * R + offset
            # plane (bidz) and i-band (cid_m) from band: p = band // ncluster_m, m = band % ncluster_m
            bidz_, cid_m = divmod(band, params.ncluster_m_fdd)
            # CLAMP padding j (>= ncluster_n) to the last valid j-cluster. Padding appears ONLY when
            # run_j_tiles does not divide ncluster_n (the last run of a band pads j past ncluster_n); those
            # padding CTAs then idempotently RE-STORE an existing tile (the benign double-store pe_aligned's
            # ceil-spill already relies on) rather than the decode gating them invalid -- gating made is_valid
            # NON-monotonic in the tile counter (j oscillates), so the persistent loop stopped on the first
            # pad-invalid and SILENTLY dropped every later tile. With the clamp, is_valid gates on run_global
            # ALONE (monotonic) so the stop-on-first-invalid is exact and NO tile is dropped, for ARBITRARY
            # run_j_tiles. For dividing run_j_tiles (coalesce full-band; the cluster-drain bounded band,
            # ncluster_n = cp1*run_j_tiles) j < ncluster_n ALWAYS -> the clamp is a no-op -> byte-identical.
            cid_n = cutlass.min(j, params.ncluster_n_run - 1)
            if const_expr(is_valid is None):
                # Gate on run_global ALONE. run_global is monotonic in work_idx (run_local = c//R is
                # non-decreasing), so the persistent loop's stop-on-first-invalid is exact. (Gating on
                # work_idx<total would cut CTAs off early -- a CTA's last run can sit above total/gz.)
                is_valid = run_global < params.total_runs
        if const_expr(bidz is not None):
            bidz_ = bidz
        pid_m, pid_n, batch_idx = Int32(0), Int32(0), Int32(0)
        if is_valid:
            pid_m, pid_n = self._cluster_id_to_cta_id(
                cid_m, cid_n, block_zero_only=block_zero_only, loc=loc, ip=ip
            )
            batch_idx = (
                bidz_
                if const_expr(params.batch_idx_permute is None)
                else params.batch_idx_permute[bidz_]
            )
        tile_coord_mnkl = (pid_m, pid_n, None, batch_idx)
        return cutlass.utils.WorkTileInfo(tile_coord_mnkl, is_valid)

    @cute.jit
    def get_current_work(self, *, loc=None, ip=None) -> cutlass.utils.WorkTileInfo:
        """The work tile this CTA should process now.

        Returns:
            A ``WorkTileInfo``. Under ``NONE`` this is the CTA's single tile; under the persistent
            modes it is the one the last ``advance`` landed on, read from SMEM when a scheduler warp
            broadcasts on the cluster's behalf.
        """
        params = self.params
        pid_m, pid_n, batch_idx, is_valid = Int32(0), Int32(0), Int32(0), Boolean(False)
        if const_expr(params.persistence_mode == PersistenceMode.NONE):
            pass
        # elif const_expr(params.persistence_mode == PersistenceMode.STATIC):
        #     return self._delinearize_work_idx(loc=loc, ip=ip)
        else:
            self._scheduler_pipeline.consumer_wait(self._pipeline_state)
            pid_m, pid_n, batch_idx, is_valid_i32 = [
                self._sched_smem[i, self._pipeline_state.index] for i in range(4)
            ]
            # Need this fence since the STAS from the producer is using the async proxy.
            # Without this, we get race condition / deadlock.
            if const_expr(cute.size(params.cluster_shape_mn) > 1):
                cute.arch.fence_view_async_shared()
            cute.arch.sync_warp()
            with cute.arch.elect_one():
                self._scheduler_pipeline.consumer_release(self._pipeline_state)
            self._pipeline_state.advance()
            is_valid = Boolean(is_valid_i32)
        tile_coord_mnkl = (pid_m, pid_n, None, batch_idx)
        return cutlass.utils.WorkTileInfo(tile_coord_mnkl, Boolean(is_valid))

    # @cute.jit
    def initial_work_tile_info(self, *, loc=None, ip=None) -> cutlass.utils.WorkTileInfo:
        """The first work tile, before any advance.

        Returns:
            The ``WorkTileInfo`` for this CTA's starting work index.
        """
        return self._delinearize_work_idx(self._current_work_idx, loc=loc, ip=ip)
        # if is_scheduler_warp:
        # work_tile_info = self._delinearize_work_idx(block_zero_only=True, loc=loc, ip=ip)
        # self.write_work_tile_to_smem(work_tile_info, loc=loc, ip=ip)
        # self.write_work_tile_to_smem(self._delinearize_work_idx(block_zero_only=True, loc=loc, ip=ip), loc=loc, ip=ip)

    @cute.jit
    def _fetch_next_work_idx(self, *, loc=None, ip=None) -> Int32 | Tuple[Int32, Int32, Boolean]:
        """should only be called by the scheduler warp"""
        params = self.params
        num_persistent_clusters = Int32(cute.arch.grid_dim()[2])
        if const_expr(params.persistence_mode == PersistenceMode.STATIC):
            return self._current_work_idx + num_persistent_clusters
        elif const_expr(params.persistence_mode == PersistenceMode.DYNAMIC):
            next_work_linear_idx = Int32(0)
            if cute.arch.lane_idx() == 0:
                # If varlen_m, problem_shape_ncluster_mnl[0] is None, so we use atomic_add
                # instead of atomic_inc, and at the end of the kernel must reset the semaphore to 0.
                #                 # cute.printf("before atomicadd, tidx = {}, bidz = {}, idx = {}", cute.arch.thread_idx()[0], cute.arch.block_idx()[2], current_work_idx)
                if const_expr(params.problem_shape_ncluster_mnl[0] is not None):
                    next_work_linear_idx = num_persistent_clusters + utils.atomic_inc_i32(
                        cute.size(params.problem_shape_ncluster_mnl) - 1,
                        params.tile_count_semaphore,
                    )
                else:  # varlen_m
                    next_work_linear_idx = num_persistent_clusters + utils.atomic_add_i32(
                        1, params.tile_count_semaphore
                    )
                # cute.printf("after atomicadd, tidx = {}, bidz = {}, idx = {}", cute.arch.thread_idx()[0], cute.arch.block_idx()[2], current_work_idx)
            return cute.arch.shuffle_sync(next_work_linear_idx, 0)
        else:
            return Int32(0)

    @cute.jit
    def write_work_tile_to_smem(
        self, work_tile_info: cutlass.utils.WorkTileInfo, *, loc=None, ip=None
    ):
        """Broadcast a work tile to every CTA in the cluster through SMEM.

        Only the scheduler warp calls this. The four values (``pid_m``, ``pid_n``, batch, valid) go
        out in ONE vectorized ``st.async``, so the peers' barrier sees a single arrival rather than
        four -- which is why ``store_shared_remote_x4`` exists.

        Args:
            work_tile_info: The tile to publish.
            loc: Optional DSL source location.
            ip: Optional DSL insertion point.

        Returns:
            None.
        """
        params = self.params
        if const_expr(self._sched_smem is not None):
            # producer phase is always consumer_phase ^ 1
            pipeline_state_producer = PipelineStateWAdvance(
                self._pipeline_state.stages,
                self._pipeline_state.count,
                self._pipeline_state.index,
                self._pipeline_state.phase ^ 1,
            )
            self._scheduler_pipeline.producer_acquire(pipeline_state_producer)
            sched_data = [
                work_tile_info.tile_idx[0],
                work_tile_info.tile_idx[1],
                work_tile_info.tile_idx[3],
                Int32(work_tile_info.is_valid_tile),
            ]
            lane_idx = cute.arch.lane_idx()
            if lane_idx < cute.size(params.cluster_shape_mn):
                # cute.printf("Producer pid_m = {}, pid_n = {}, batch_idx = {}, is_valid = {}, after empty wait, idx = {}", sched_data[0], sched_data[1], sched_data[2], sched_data[3], self._current_work_idx)
                pipeline_idx = self._pipeline_state.index
                if const_expr(cute.size(params.cluster_shape_mn) == 1):
                    for i in cutlass.range_constexpr(4):
                        self._sched_smem[i, pipeline_idx] = sched_data[i]
                    self._scheduler_pipeline.producer_commit(self._pipeline_state)
                else:
                    peer_cta_rank_in_cluster = lane_idx
                    # Here we assume that the block idx in cluster is linearized such that
                    # x is the fastest moving direction.
                    bidx_in_cluster = peer_cta_rank_in_cluster % params.cluster_shape_mn[0]
                    bidy_in_cluster = peer_cta_rank_in_cluster // params.cluster_shape_mn[0]
                    mbar_ptr = self._scheduler_pipeline.producer_get_barrier(self._pipeline_state)
                    cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr, 16, peer_cta_rank_in_cluster)
                    utils.store_shared_remote_x4(
                        sched_data[0] + bidx_in_cluster,
                        sched_data[1] + bidy_in_cluster,
                        sched_data[2],
                        sched_data[3],
                        smem_ptr=self._sched_smem[None, pipeline_idx].iterator,
                        mbar_ptr=mbar_ptr,
                        peer_cta_rank_in_cluster=peer_cta_rank_in_cluster,
                    )

    @cute.jit
    def advance_to_next_work(
        self,
        is_scheduler_warp: bool | Boolean = False,
        *,
        advance_count: int = 1,
        loc=None,
        ip=None,
    ):
        """is_scheduler_warp should only be true for one warp in the whole cluster.
        Moreover, we assume that only block zero in the cluster is calling this function.
        If calling with is_scheduler_warp = True, advance_count must be 1.
        """
        params = self.params
        self.num_tiles_executed += Int32(advance_count)
        if const_expr(self._pipeline_state is not None and advance_count > 1):
            self._pipeline_state.advance_iters(advance_count - 1)
        if const_expr(params.persistence_mode in [PersistenceMode.STATIC, PersistenceMode.DYNAMIC]):
            # We assume here that advance_count is 1 for scheduler_warp
            if is_scheduler_warp:
                self._current_work_idx = self._fetch_next_work_idx(loc=loc, ip=ip)
                work_tile_info = self._delinearize_work_idx(
                    self._current_work_idx, block_zero_only=True, loc=loc, ip=ip
                )
                self.write_work_tile_to_smem(work_tile_info, loc=loc, ip=ip)

    def producer_tail(self):
        """Drain the broadcast pipeline at the end of the grid, so no consumer waits forever.

        Returns:
            None. A no-op when this CTA has no scheduler pipeline.
        """
        if const_expr(self._scheduler_pipeline is not None):
            pipeline_state_producer = PipelineStateWAdvance(
                self._pipeline_state.stages,
                self._pipeline_state.count,
                self._pipeline_state.index,
                self._pipeline_state.phase ^ 1,
            )
            self._scheduler_pipeline.producer_tail(pipeline_state_producer)

    def __extract_mlir_values__(self):
        """Flatten the scheduler's per-CTA state into MLIR values.

        Returns:
            The concatenated values, in a fixed order, with the per-object counts recorded in
            ``self._values_pos`` for the rebuild.
        """
        values, self._values_pos = [], []
        for obj in [
            self._current_work_idx,
            self.num_tiles_executed,
            self._current_batch_idx,
            self._num_work_idx_before_cur_batch,
            self._sched_smem,
            self._scheduler_pipeline,
            self._pipeline_state,
            self.params,
        ]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        """Rebuild the scheduler inside the kernel from the flattened values.

        Args:
            values: The flat sequence produced by ``__extract_mlir_values__``.

        Returns:
            A new scheduler of the same class, carrying the stashed source location.
        """
        obj_list = []
        for obj, n_items in zip(
            [
                self._current_work_idx,
                self.num_tiles_executed,
                self._current_batch_idx,
                self._num_work_idx_before_cur_batch,
                self._sched_smem,
                self._scheduler_pipeline,
                self._pipeline_state,
                self.params,
            ],
            self._values_pos,
        ):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return self.__class__(*(tuple(obj_list)), loc=self._loc)


@cute.jit
def triangular_idx_to_coord(idx: Int32) -> Tuple[Int32, Int32]:
    """
    Convert a triangular index to 2D coordinates.
    This is used to convert the linear index to 2D coordinates for triangular matrices.
    """
    row = utils.ceil((utils.sqrt(2 * idx + 2.25) - 0.5)) - 1
    col = idx - (row * (row + 1)) // 2
    return row, col


class TriangularTileScheduler(TileScheduler):
    """We assume the tile size per cluster is square (e.g., 128 x 256 per CTA, with cluster 2 x 1)"""

    @dataclass
    class Params:
        """``TriangularTileScheduler``'s traced state.

        Differs from the base's by carrying ``group_size_inv_f32``: the triangular swizzle inverts a
        quadratic rather than dividing, so it needs a float reciprocal alongside the fast divisors.
        """

        problem_shape_ncluster_mnl: cute.Shape
        num_clusters_per_problem_fdd: FastDivmod
        group_size_inv_f32: Float32
        num_groups_regular: Int32
        group_size_fdd: FastDivmod
        group_size_tail_fdd: FastDivmod
        group_size_mul_group_size_fdd: FastDivmod
        group_size_tail_mul_group_size_fdd: FastDivmod
        tile_count_semaphore: Optional[cute.Pointer]
        cluster_shape_mn: cutlass.Constexpr[cute.Shape]
        persistence_mode: cutlass.Constexpr[PersistenceMode]

        @staticmethod
        @cute.jit
        def create(
            args: TileSchedulerArguments, *, loc=None, ip=None
        ) -> "TriangularTileScheduler.Params":
            """Derive the triangular scheduler's traced params.

            Args:
                args: The scheduler arguments; cluster K extent must be 1.
                loc: Optional DSL source location.
                ip: Optional DSL insertion point.

            Returns:
                The ``Params``, including the float group-size reciprocal the quadratic inversion
                needs.

            Raises:
                AssertionError: If the cluster's K extent is not 1.
            """
            assert args.cluster_shape_mnk[2] == 1
            cluster_shape_mn = const_expr(cute.select(args.cluster_shape_mnk, mode=[0, 1]))
            problem_shape_ntile_mn = cute.select(args.problem_shape_ntile_mnl, mode=[0, 1])
            problem_shape_ncluster_mn = cute.ceil_div(problem_shape_ntile_mn, cluster_shape_mn)
            problem_shape_ncluster_mnl = problem_shape_ncluster_mn + (
                args.problem_shape_ntile_mnl[2],
            )
            cluster_m = problem_shape_ncluster_mn[0]
            # Assume that each cluster is responsible for a square tile
            num_clusters_per_problem = cluster_m * (cluster_m + 1) // 2
            group_size = min(args.group_size, cluster_m)
            group_size_tail = cluster_m % group_size
            num_groups_regular = cluster_m // group_size
            if const_expr(args.persistence_mode == PersistenceMode.DYNAMIC):
                assert args.tile_count_semaphore is not None
            return TriangularTileScheduler.Params(
                problem_shape_ncluster_mnl,
                FastDivmod(num_clusters_per_problem),
                Float32(1.0 / group_size),
                num_groups_regular,
                FastDivmod(group_size),
                # Don't divide by 0
                FastDivmod(group_size_tail if group_size_tail > 0 else 1),
                FastDivmod(group_size * group_size),
                FastDivmod((group_size_tail if group_size_tail > 0 else 1) * group_size),
                args.tile_count_semaphore
                if const_expr(args.persistence_mode == PersistenceMode.DYNAMIC)
                else None,
                cluster_shape_mn,
                args.persistence_mode,
            )

    @staticmethod
    def to_underlying_arguments(args: TileSchedulerArguments, *, loc=None, ip=None) -> Params:
        """Derive the traced ``Params`` from the caller's arguments.

        Args:
            args: The scheduler arguments.
            loc: Optional DSL source location.
            ip: Optional DSL insertion point.

        Returns:
            The ``Params``.
        """
        return TriangularTileScheduler.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    @cute.jit
    def create(
        params: Params,
        sched_smem: Optional[cute.Tensor] = None,
        scheduler_pipeline: Optional[cutlass.pipeline.PipelineAsync] = None,
        is_scheduler_warp: bool | Boolean = False,
        *,
        loc=None,
        ip=None,
    ) -> "TriangularTileScheduler":
        """Construct the scheduler inside the kernel, positioned at this CTA's first work index.

        Args:
            params: The traced ``Params``.
            sched_smem: SMEM slice for the cluster broadcast, or None.
            scheduler_pipeline: Pipeline guarding that broadcast, or None.
            is_scheduler_warp: Whether this warp computes work tiles for the cluster.
            loc: Optional DSL source location.
            ip: Optional DSL insertion point.

        Returns:
            The scheduler.
        """
        current_work_idx, _ = TileScheduler._cluster_idx_to_work_idx_batch(
            params, cute.arch.cluster_idx(), loc=loc, ip=ip
        )
        stages = 0
        if const_expr(params.persistence_mode in [PersistenceMode.STATIC, PersistenceMode.DYNAMIC]):
            assert sched_smem is not None
            assert scheduler_pipeline is not None
            stages = const_expr(cute.size(sched_smem, mode=[1]))
        return TriangularTileScheduler(
            current_work_idx,
            Int32(0),  # num_tiles_executed
            Int32(0),  # current_batch_idx
            Int32(0),  # num_work_idx_before_cur_batch
            sched_smem,
            scheduler_pipeline,
            PipelineStateWAdvance(stages, Int32(0), Int32(0), Int32(0)),
            params,
            loc=loc,
            ip=ip,
        )

    # called by host
    @staticmethod
    def get_grid_shape(
        params: Params,
        max_active_clusters: Int32,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        """The launch grid for the triangular schedule.

        Args:
            params: The traced ``Params``.
            max_active_clusters: Resident cluster count, for the persistent modes.
            loc: Optional DSL source location.
            ip: Optional DSL insertion point.

        Returns:
            ``(grid_x, grid_y, grid_z)`` in CTAs.
        """
        clusters = (params.num_clusters_per_problem_fdd.divisor, 1)
        num_ctas_mnl = tuple(x * y for x, y in zip(clusters, params.cluster_shape_mn)) + (
            params.problem_shape_ncluster_mnl[2],
        )
        if const_expr(params.persistence_mode == PersistenceMode.NONE):
            return num_ctas_mnl
        else:
            num_ctas_in_problem = cute.size(num_ctas_mnl, loc=loc, ip=ip)
            num_ctas_per_cluster = cute.size(params.cluster_shape_mn, loc=loc, ip=ip)
            # Total ctas that can run in one wave
            num_ctas_per_wave = max_active_clusters * num_ctas_per_cluster
            num_persistent_ctas = cutlass.min(num_ctas_in_problem, num_ctas_per_wave)
            num_persistent_clusters = num_persistent_ctas // num_ctas_per_cluster
            return (*params.cluster_shape_mn, num_persistent_clusters)

    @cute.jit
    def _swizzle_cta(
        self, cluster_id_in_problem: Int32, *, loc=None, ip=None
    ) -> Tuple[Int32, Int32]:
        # CTA Swizzle to promote L2 data reuse
        """Swizzle for a TRIANGULAR problem, where only ``i <= j`` tiles exist.

        The group a linear index falls in is found by inverting a quadratic -- hence the ``sqrt``
        and the float reciprocal -- rather than by dividing, because group sizes grow along the
        diagonal.

        Args:
            cluster_id_in_problem: Linear cluster index within this batch.
            loc: Optional DSL source location.
            ip: Optional DSL insertion point.

        Returns:
            ``(cid_m, cid_n)`` in clusters, always in the lower triangle.
        """
        params = self.params
        group_size = params.group_size_fdd.divisor
        group_id = (
            utils.ceil(
                (utils.sqrt(2 * cluster_id_in_problem + 2.25) - 0.5) * params.group_size_inv_f32
            )
            - 1
        )
        cid_m_start = group_id * group_size
        id_in_group = cluster_id_in_problem - (cid_m_start * (cid_m_start + 1)) // 2
        group_size_actual = (
            group_size
            if group_id < params.num_groups_regular
            else params.group_size_tail_fdd.divisor
        )
        group_col, group_remainder = Int32(0), Int32(0)
        if group_id < params.num_groups_regular:
            group_col, group_remainder = divmod(id_in_group, params.group_size_mul_group_size_fdd)
        else:  # tail part
            group_col, group_remainder = divmod(
                id_in_group, params.group_size_tail_mul_group_size_fdd
            )
        cid_m_in_group, cid_n_in_group = Int32(0), Int32(0)
        if id_in_group >= group_size_actual * group_size * group_id:  # triangular tail
            cid_m_in_group, cid_n_in_group = triangular_idx_to_coord(group_remainder)
        else:
            if group_id < params.num_groups_regular:
                cid_n_in_group, cid_m_in_group = divmod(group_remainder, params.group_size_fdd)
            else:
                cid_n_in_group, cid_m_in_group = divmod(group_remainder, params.group_size_tail_fdd)
        cid_m = cid_m_start + cid_m_in_group
        cid_n = group_col * group_size + cid_n_in_group
        return cid_m, cid_n

    @cute.jit
    def _delinearize_work_idx(
        self,
        work_idx: Int32,
        bidz: Optional[Int32] = None,
        is_valid: Optional[Boolean] = None,
        *,
        block_zero_only: bool = False,
        loc=None,
        ip=None,
    ) -> cutlass.utils.WorkTileInfo:
        """Delinearize a work index over the triangular tile set.

        Args:
            work_idx: The linear work index.
            bidz: Batch index if already known, else None.
            is_valid: Validity if already known, else None.
            block_zero_only: Report the cluster's CTA-0 tile.
            loc: Optional DSL source location.
            ip: Optional DSL insertion point.

        Returns:
            A ``WorkTileInfo``.
        """
        params = self.params
        if const_expr(is_valid is None):
            if const_expr(params.persistence_mode == PersistenceMode.NONE):
                is_valid = self.num_tiles_executed == 0
            else:
                is_valid = (
                    work_idx
                    < params.num_clusters_per_problem_fdd.divisor
                    * params.problem_shape_ncluster_mnl[2]
                )
        pid_m, pid_n, batch_idx = Int32(0), Int32(0), Int32(0)
        if is_valid:
            if const_expr(params.persistence_mode == PersistenceMode.NONE):
                cluster_id_in_problem = work_idx
                _, _, bidz_ = cute.arch.block_idx()
            else:
                bidz_, cluster_id_in_problem = divmod(work_idx, params.num_clusters_per_problem_fdd)
                cluster_id_in_problem = Int32(cluster_id_in_problem)  # divmod returns IntValue
            if const_expr(bidz is not None):
                bidz_ = bidz
            cid_m, cid_n = self._swizzle_cta(cluster_id_in_problem, loc=loc, ip=ip)
            pid_m, pid_n = self._cluster_id_to_cta_id(
                cid_m, cid_n, block_zero_only=block_zero_only, loc=loc, ip=ip
            )
            batch_idx = bidz_
        tile_coord_mnkl = (pid_m, pid_n, None, batch_idx)
        # tidx, _, _ = cute.arch.thread_idx()
        # if tidx == 0:
        #     cute.printf("bidx = {}, bidy = {}, group_id = {}, id_in_group = {}, group_size_actual = {}, group_col = {}, group_remainder = {}, cid_n_in_group = {}, cid_m_in_group = {}, cid_m = {}, cid_n = {}, is_valid = {}",
        #                 bidx, bidy, group_id, id_in_group, group_size_actual, group_col, group_remainder, cid_n_in_group, cid_m_in_group, cid_m, cid_n, is_valid)
        return cutlass.utils.WorkTileInfo(tile_coord_mnkl, is_valid)

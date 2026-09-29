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

# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.
# This file has been modified by NVIDIA CORPORATION & AFFILIATES.

"""Shared base for row-reduction kernel functors.

Compile-time configuration is declared once, as :class:`ReductionParams`, and bound in ``__init__``
via :class:`~fold_cp_ops._internal.compile_time.template_params.TemplateParamsMixin`. That module
documents why the compile-time/runtime split is enforced rather than trusted; the short version is
that a runtime value stashed on ``self`` does not survive the ``@cute.jit`` -> ``@cute.kernel``
boundary and does not raise when it fails.

What that buys here concretely: ``cluster_n`` used to be established only by ``_set_cluster_n()``,
called from the subclass's ``__call__``, so a freshly constructed instance did not have the
attribute and ``_get_tiled_copy()`` raised ``AttributeError`` on it — an ordering requirement
documented nowhere. It is now a declared parameter, present from construction and immutable after.
"""

from typing import Optional, Tuple, Type

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Float32, const_expr

from fold_cp_ops._internal.compile_time.copy_descriptors import tiled_copy_2d
from fold_cp_ops._internal.compile_time.template_params import TemplateParams, TemplateParamsMixin


class ReductionParams(TemplateParams):
    """Compile-time parameters every row-reduction kernel needs.

    Declaring a field here is the statement that ``self.<field>`` is a compile-time constant,
    validated as such at construction and immutable afterwards. Subclass to add kernel-specific
    parameters (see ``kernels/layernorm.py``'s ``LayerNormParams``).

    Attributes:
        dtype: Element type of the reduced tensor. A ``cutlass`` numeric *type* (a value of
            ``torch2cute_dtype_map``, not a ``torch.dtype``); its ``.width`` drives the vector
            width and the cluster ladder.
        N: Feature (reduced) extent. Must be positive. Need not be a power of two nor a tile
            multiple — an off-grid N engages the copy predicate rather than being rejected.
        stage: Independent reduction slots, >= 1 (2 for LayerNorm's mean-then-variance). Sizes the
            reduction buffer's third mode and the mbarrier array; under-sizing under-allocates
            shared memory rather than raising.
        cluster_n: Thread-block cluster width, >= 1 and a power of two. ``> 1`` requires SM90 and
            selects the distributed-shared-memory reduction. Computed by the subclass **before**
            binding, so it is present from construction.
        reduction_dtype: Accumulator type for the staging buffer. Float32 unless a subclass has a
            reason; a narrower type silently loses precision across the block reduction.
    """

    dtype: Type[cutlass.Numeric]
    N: int
    stage: int
    cluster_n: int = 1
    reduction_dtype: Type[cutlass.Numeric] = Float32


class ReductionBase(TemplateParamsMixin):
    """Row-reduction functor base: frozen compile-time params plus tile/SMEM helpers.

    Subclasses declare their parameter pack via ``Params`` and call ``_bind_params(...)`` once in
    ``__init__``. They must override :meth:`_threads_per_row`, which is a pure function of the
    bound parameters — deliberately a method rather than a parameter, since it derives from ``N``
    and storing it would be a second source of truth.
    """

    Params = ReductionParams

    def __init__(self, dtype: Type[cutlass.Numeric], N: int, stage: int, reduction_dtype=Float32):
        """Bind the base parameter pack. Subclasses with extra fields bind them instead.

        Args:
            dtype: See :class:`ReductionParams`.
            N: See :class:`ReductionParams`.
            stage: See :class:`ReductionParams`.
            reduction_dtype: See :class:`ReductionParams`.

        Raises:
            TypeError: If any argument is a runtime value rather than a compile-time constant.
        """
        self._bind_params(dtype=dtype, N=N, stage=stage, reduction_dtype=reduction_dtype)

    def _threads_per_row(self) -> int:
        """Lanes cooperating on one row. **Subclasses must override.**

        Returns:
            A power of two — ``reduce.row_reduce`` finishes with a butterfly shuffle, which
            requires it — that also divides :meth:`_num_threads`.

        Raises:
            NotImplementedError: Always, in the base. No default: a silent one would hand a
                subclass a plausible-but-untuned geometry with no signal that it was never set.
        """
        raise NotImplementedError()

    def _num_threads(self) -> int:
        """Threads per block. Wider blocks past N=16384 keep elements-per-thread bounded.

        Returns:
            128 for ``N <= 16384``, else 256. Must be a multiple of the warp size and of
            :meth:`_threads_per_row`; both are asserted in :meth:`_get_tiled_copy`.
        """
        return 128 if self.N <= 16384 else 256

    def _get_tiled_copy(self, vecsize: int = 1) -> Tuple[cute.TiledCopy, tuple, int]:
        """Build the tiled copy and CTA tile shape for this configuration.

        Args:
            vecsize: Elements each thread moves per instruction. Must divide ``N``; otherwise the
                tile cannot cover the row and trailing elements would be dropped silently. Callers
                derive it as ``gcd(N, max_vec_elems(dtype))``.

        Returns:
            ``(tiled_copy, tiler_mn, threads_per_row)``. ``tiler_mn`` is
            ``(rows_per_cta, cols_per_cta)``; ``cols_per_cta`` may exceed ``N``, and the excess is
            masked by the caller's copy predicate.

        Raises:
            AssertionError: If ``vecsize`` does not divide ``N``, or the block size is not a
                multiple of the warp size.
        """
        assert self.N % vecsize == 0, f"Input N {self.N} is not divisible by vector size {vecsize}"
        threads_per_row = self._threads_per_row()
        num_threads = self._num_threads()
        assert num_threads % cute.arch.WARP_SIZE == 0
        num_blocks_N = cute.ceil_div(self.N // vecsize, threads_per_row * self.cluster_n)
        tiler_mn = (num_threads // threads_per_row, vecsize * num_blocks_N * threads_per_row)
        tiled_copy = tiled_copy_2d(self.dtype, threads_per_row, num_threads, vecsize)
        return tiled_copy, tiler_mn, threads_per_row

    def _get_reduction_buffer_layout(self, tv_layout: cute.Layout, cluster_n: int) -> cute.Layout:
        """Shape the shared-memory staging buffer the block/cluster reduction writes through.

        Args:
            tv_layout: The tiled copy's thread-value layout; its thread mode determines how many
                warps exist and how many cooperate on one row.
            cluster_n: Cluster width, passed explicitly so the layout is a pure function of its
                arguments rather than of instance state.

        Returns:
            A layout of shape ``(rows, (warps_per_row, cluster_n), stage)``, ordered so the
            warps-per-row mode is fastest — the order the reduction indexes it in.
        """
        num_warps = cute.size(tv_layout, mode=[0]) // cute.arch.WARP_SIZE
        warps_per_row = (
            num_warps
            if cute.rank(tv_layout.shape[0]) == 1
            else max(tv_layout.shape[0][0] // cute.arch.WARP_SIZE, 1)
        )
        return cute.make_ordered_layout(
            (num_warps // warps_per_row, (warps_per_row, cluster_n), self.stage),
            order=(1, 0, 2),
        )

    def _allocate_reduction_buffer_and_mbar(
        self, smem: cutlass.utils.SmemAllocator, tv_layout: cute.Layout, is_persistent: bool = False
    ) -> Tuple[cute.Tensor, Optional[cute.Pointer]]:
        """Carve the reduction staging buffer, and an mbarrier array when clustered.

        Args:
            smem: The kernel's shared-memory allocator. Allocation order is layout-significant —
                call this at the same point in every kernel that shares a tile shape.
            tv_layout: Thread-value layout of the tiled copy.
            is_persistent: If True, allocate a second mbarrier set for a persistent kernel's empty
                phase. The caller must then initialize them.

        Returns:
            ``(reduction_buffer, mbar_ptr)``. ``mbar_ptr`` is None when ``cluster_n == 1`` — the
            single-CTA path uses a plain block barrier and never touches an mbarrier.
        """
        reduction_buffer = smem.allocate_tensor(
            self.reduction_dtype,
            self._get_reduction_buffer_layout(tv_layout, self.cluster_n),
            byte_alignment=8,
        )
        if const_expr(self.cluster_n > 1):
            mbar_ptr = smem.allocate_array(
                Int64, num_elems=self.stage if not is_persistent else self.stage * 2
            )
        else:
            mbar_ptr = None
        return reduction_buffer, mbar_ptr

    @cute.jit
    def _initialize_cluster(
        self,
        tidx: Int32,
        mbar_ptr: cute.Pointer,
        num_warps: int,
        is_persistent: bool = False,
    ) -> None:
        """Initialize the cluster's mbarriers and arrive. Entirely pruned when unclustered.

        Args:
            tidx: This thread's index within the block — the real thread index, not a lane id;
                only threads below ``stage`` initialize a barrier.
            mbar_ptr: Base of the mbarrier array. Unused when ``cluster_n == 1``.
            num_warps: Warps per block; sets the empty barrier's arrival count under
                ``is_persistent``.
            is_persistent: Must match what was passed to the allocation, or the second barrier set
                is initialized outside its allocation.

        Returns:
            None. Emits barrier init, init fence and a relaxed cluster arrive as side effects.
        """
        if const_expr(self.cluster_n > 1):
            if tidx < self.stage:  # Initialize full barrier
                cute.arch.mbarrier_init(mbar_ptr + tidx, 1)
                if const_expr(is_persistent):  # Initialize empty barrier
                    cute.arch.mbarrier_init(
                        mbar_ptr + self.stage + tidx, num_warps * self.cluster_n
                    )
            cute.arch.mbarrier_init_fence()
            # Cluster arrive after barrier init
            cute.arch.cluster_arrive_relaxed()

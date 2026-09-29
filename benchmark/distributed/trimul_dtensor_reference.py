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

"""Reference context-parallel TriMul -- the FAIR distributed baseline for every speedup.

This module is the complete, self-contained reference implementation of the triangular
multiplicative update under context parallelism: the single-device layer, the process-group comm
schedules, the DTensor parameter wrappers, and the 1-D and 2-D CP forward paths. It has no
dependency outside `torch` (+ `numpy`) and this package's own `LayoutMap`, so a benchmark cell can
import it without pulling in a model framework.

**Why it exists rather than a `.redistribute()` one-liner.** A baseline that reshards a 1-D
row-shard through a feature-axis all-to-all measures the reshard, not the layer, and flatters the
fused path for the wrong reason. The two CP paths below instead each match the input's sharding
NATIVELY, so the comparison is layer-against-layer:

* **1-D** (one token axis sharded -- mesh ``(dp, cp)``, placements ``(Shard(0), Shard(1))``, local
  ``(B, N/cp, N, D)`` row slab): `TriangularMultiplicationOutgoing1D` /
  `TriangularMultiplicationIncoming1D`. The outgoing direction contracts over a *local* index, so
  only the second operand travels -- a ring rotation. The incoming direction contracts over the
  *sharded* index, so every rank holds a partial sum and the comm is a tiled reduce-scatter. **No
  all-to-all in either.**
* **2-D** (both token axes sharded -- mesh ``(dp, cp0, cp1)``, placements ``(Shard(0), Shard(1),
  Shard(2))``, local ``(B, N/cp0, N/cp1, D)`` block): `TriangularMultiplicationOutgoing2D` /
  `TriangularMultiplicationIncoming2D` over a `Ring2DComm`, which walks the 2-D grid so each rank
  sees every operand block it needs exactly once. Requires a SQUARE grid (``cp0 == cp1``).

The two are separate native pipelines, selected by how the input is actually sharded
(`trimul_dtensor_baseline.build_trimul_dtensor_baseline` does the dispatch). One is never morphed into the
other.

Contents, in file order:

* `get_group_rank_from_axial_shift`, `update_exhaustive_strides` -- rank/stride arithmetic over a
  `LayoutMap`.
* `One2OneComm`, `TransposeComm`, `ternary_parity`, `Ring2DComm` -- the batched-P2P comm objects
  the 2-D path schedules its operand exchange with.
* `_ring_p2p_send_recv` -- the single-hop ring exchange the 1-D outgoing path rotates with.
* `LayerNormParamsReplicated`, `LinearParamsReplicated`, `sigmoid_gate` -- DTensor ops that keep
  the *parameters* replicated while the activations stay sharded, each with an explicit autograd
  `Function` so the sharding of the backward is stated rather than inferred.
* `TriangularMultiplication{,Outgoing,Incoming}` -- the single-device layer. **Clean-room**: see
  its section comment. It is also the parameter container the CP wrappers below are built from.
* `TriangularMultiplication{,Outgoing,Incoming}1D` and `...2D` -- the CP forward paths.

Naming: the direction-suffixed classes are `...Outgoing1D` / `...Incoming2D` and so on, with the
undecorated `TriangularMultiplication{Outgoing,Incoming}` reserved for the single-device layer.
The suffix is load-bearing -- 1-D and 2-D are different algorithms over different meshes, not one
algorithm at two sizes -- so a call site that drops it is a bug, not a shorthand.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard, distribute_tensor

from fold_cp_ops.distributed.layout_map import LayoutMap

# ============================================================================================== #
# Rank + stride arithmetic over a `LayoutMap`. Two standalone helpers, no comm, no CUDA --
# they answer "which rank sits `delta` steps along axis `a`" and "what strides preserve this
# axis ORDER at a new shape", which is what the 2-D ring needs to name its neighbours.
# ============================================================================================== #


def get_group_rank_from_axial_shift(coord: tuple[int, ...], axis: int, delta: int, layout_group: LayoutMap) -> int:
    """
    Get the rank of a process after shifting its coordinates along an axis.

    Parameters
    ----------
    coord : tuple of ints
        The current coordinates of the process in the group layout.
    axis : int
        The axis along which to shift the coordinates.
    delta : int
        The amount to shift the coordinates by (can be positive or negative).
    layout_group : LayoutMap
        The layout mapping of the process group.

    Returns
    -------
    int
        The rank of the process after shifting its coordinates.

    Raises
    ------
    ValueError
        If the coordinates are incompatible with the layout shape or if the axis is out of range.
    """
    if len(coord) != len(layout_group.shape):
        raise ValueError(f"Incompatible coord {coord} and layout_group shape {layout_group.shape}")
    if axis >= len(coord):
        raise ValueError(f"Axis {axis} is out of range for coord {coord}")
    coord_shifted = list(coord)
    coord_shifted[axis] = (coord_shifted[axis] + delta) % layout_group.shape[axis]
    return layout_group(coord_shifted)


def update_exhaustive_strides(
    shape_original: Sequence[int], strides_original: Sequence[int], shape_new: Sequence[int]
) -> Sequence[int]:
    """
    Update strides to maintain the same memory layout pattern when shape changes.

    This function computes new strides that preserve the same axis ordering and memory
    layout pattern as the original exhaustive layout, but with a new shape. The resulting
    strides will create an exhaustive layout with the same dimension ordering as the
    original layout.

    An exhaustive layout is one where the mapping from multidimensional indices to flat
    indices is surjective, meaning every valid flat index corresponds to at least one
    multidimensional index. Meanwhile, a non-unique layout is not practically useful
    for our application so we further require the input shape and strides to form an
    unique layout, which implies the output layout is also unique. Overall, both the
    input and output layouts are bijective

    Parameters
    ----------
    shape_original : Sequence[int]
        The original shape of the tensor layout.
    strides_original : Sequence[int]
        The original strides of the tensor layout. Must form an exhaustive layout
        with shape_original.
    shape_new : Sequence[int]
        The new shape for which to compute corresponding strides. Must have the
        same number of dimensions as shape_original.

    Returns
    -------
    Sequence[int]
        New strides that maintain the same memory layout pattern as the original
        but are compatible with the new shape. The resulting strides will form
        an exhaustive layout with shape_new.

    Raises
    ------
    ValueError
        If the original layout (shape_original, strides_original) is not exhaustive.

    Examples
    --------
    >>> # Original layout: right-aligned (row-major) for shape (2, 3, 4)
    >>> shape_orig = (2, 3, 4)
    >>> strides_orig = (12, 4, 1)  # exhaustive right-aligned strides
    >>> shape_new = (3, 5, 2)
    >>> new_strides = update_exhaustive_strides(shape_orig, strides_orig, shape_new)
    >>> # Result: (10, 2, 1) - maintains right-aligned pattern

    Notes
    -----
    The algorithm works by:
    1. Creating a LayoutMap from the original shape and strides
    2. Verifying the original layout is exhaustive
    3. Reordering the new shape according to the original layout's stride ordering
    4. Computing exhaustive strides for the reordered new shape
    5. Reordering the computed strides back to match the original dimension order

    This is useful when reshaping tensors while preserving their memory access patterns,
    particularly in distributed computing scenarios where maintaining consistent
    memory layouts across different tensor shapes is important.
    """
    layout_original = LayoutMap(tuple(strides_original), tuple(shape_original))
    if not layout_original.is_exhaustive:
        raise ValueError(f"Input layout with shape {shape_original} and strides {strides_original} is not exhaustive")
    shape_new_ascending = np.array(shape_new)[layout_original._argsort_ascend_strides]
    argsort_output = np.argsort(layout_original._argsort_ascend_strides)
    strides_new_ascending = np.concatenate(([1], shape_new_ascending[:-1])).cumprod()
    strides_new = strides_new_ascending[argsort_output]
    return tuple(strides_new.tolist())


# ============================================================================================== #
# Comm objects for the 2-D CP path. `One2OneComm` batches a rank's send/recv pair into ONE
# `batch_isend_irecv` and hands back a handle to wait on; `TransposeComm` specializes it to the
# grid transpose; `Ring2DComm` composes both into the schedule that walks a square cp grid.
# `ternary_parity` breaks the send/recv ordering tie that would otherwise deadlock a symmetric
# pair. Nothing here is TriMul-specific -- any pairwise operand exchange over a 2-D mesh can
# use it.
# ============================================================================================== #


class One2OneComm:
    def __init__(self, group: dist.ProcessGroup, rank_send_to: int, rank_recv_from: int, parity: Optional[bool] = None):
        """
        Initializes a One2OneComm instance for point-to-point communication.

        Arguments:
            group (dist.ProcessGroup): The process group that provides the communication.
            rank_send_to (int): The rank within the group to send data to.
            rank_recv_from (int): The rank within the group to receive data from.
            parity (bool): If True, issue [isend, irecv]; otherwise issue [irecv, isend]
                in batch_isend_irecv. If None, parity is `rank % 2`, where `rank` is the
                calling rank's index in the WORLD group. If self.is_self_comm is True, i.e.,
                `rank_send_to == rank and rank_recv_from == rank`, this argument has no effect.
                The motivation of setting the parity is to avoid potential deadlocks in NCCL backend
                when doing batch_isend_irecv.

        Note: rank_send_to and rank_recv_from must be ranks within the input process group.

        Raises:
            ValueError: If rank_send_to or rank_recv_from is not a valid rank within the group.
        """
        self.group = group

        self.rank = dist.get_rank(self.group)
        self.world_size = dist.get_world_size(self.group)

        if rank_send_to >= self.world_size:
            raise ValueError(f"rank_send_to >= world_size {self.world_size}")
        if rank_recv_from >= self.world_size:
            raise ValueError(f"rank_recv_from >= world_size {self.world_size}")
        # make all comm functions no-ops if self-send and self-recv
        is_self_send = rank_send_to == self.rank
        is_self_recv = rank_recv_from == self.rank
        if is_self_send != is_self_recv:
            raise ValueError(
                "Asymmetric send/recv tends to cause NCCL backend deadlocking "
                f"and it's not supported: is_self_send: {is_self_send}, "
                f"is_self_recv: {is_self_recv}"
            )
        self.is_self_comm = is_self_send
        self._rank_in_group_send_to = rank_send_to
        self._rank_in_group_recv_from = rank_recv_from

        self.parity = parity

        if not self.is_self_comm:
            # convert to global rank
            self.rank_send_to = dist.get_global_rank(self.group, rank_send_to)
            self.rank_recv_from = dist.get_global_rank(self.group, rank_recv_from)

            if self.parity is None:
                self.parity = self.rank % 2

            self._queue_send_recv = []
            self._work_to_finish = None

    def __deepcopy__(self, memo):
        """
        Create a deep copy of the One2OneComm instance.

        This method enables the One2OneComm object to be deep copied using the copy.deepcopy() function.
        It creates a new One2OneComm instance with the same communication parameters as the original.

        Args:
            memo (dict): Dictionary used by deepcopy to avoid circular references.

        Returns:
            One2OneComm: A new One2OneComm instance with identical configuration to the original.
        """
        return One2OneComm(self.group, self._rank_in_group_send_to, self._rank_in_group_recv_from, self.parity)

    def _prep_batch_isend_irecv(
        self,
        to_send: torch.Tensor,
        to_recv: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Prepare tensors and communication operations for batch send/receive.

        This private method sets up the tensor operations and queues the communication
        operations for later dispatch. It handles both self-communication (where send
        and receive are on the same rank) and inter-rank communication.

        Args:
            to_send (torch.Tensor): The tensor to be sent to the target rank.
            to_recv (Optional[torch.Tensor], optional): The tensor buffer to receive data into.
                If None, a new tensor with the same shape and properties as `to_send` will be created.

        Returns:
            torch.Tensor: The tensor that will contain the received data. For self-communication,
                this is either a clone of `to_send` or `to_recv` with data copied from `to_send`.
                For inter-rank communication, this is the buffer where received data will be stored.

        Note:
            - For self-communication (`is_self_comm=True`), the data is immediately copied
              and no communication operations are queued.
            - For inter-rank communication, P2P operations are queued based on parity to
              avoid potential deadlocks in NCCL backend.
            - The order of send/receive operations depends on the parity flag to ensure
              consistent ordering across ranks.
        """
        if self.is_self_comm:
            # the copy semantics remain even if self.is_self_comm
            if to_recv is None:
                ans = to_send.detach().clone()
            else:
                ans = to_recv
                ans.copy_(to_send)
            return ans

        ans = torch.empty_like(to_send) if to_recv is None else to_recv

        if self.parity:
            # TODO: verify if the order of P2POp calls matter
            # and consolidate the two branches' P2POp calls if not
            send_op = dist.P2POp(
                dist.isend,
                to_send,
                self.rank_send_to,
                group=self.group,
            )
            recv_op = dist.P2POp(
                dist.irecv,
                ans,
                self.rank_recv_from,
                group=self.group,
            )
            self._queue_send_recv.append(send_op)
            self._queue_send_recv.append(recv_op)
        else:
            recv_op = dist.P2POp(
                dist.irecv,
                ans,
                self.rank_recv_from,
                group=self.group,
            )
            send_op = dist.P2POp(
                dist.isend,
                to_send,
                self.rank_send_to,
                group=self.group,
            )
            self._queue_send_recv.append(recv_op)
            self._queue_send_recv.append(send_op)
        return ans

    def _dispatch(self):
        """
        Dispatch all queued communication operations.

        This private method initiates all point-to-point communication operations that have been
        queued by previous calls to `_prep_batch_isend_irecv`. The operations are dispatched
        asynchronously using `dist.batch_isend_irecv`.

        Raises:
            RuntimeError: If there are already unfinished communications in the queue when trying
                to dispatch new operations. This prevents overlapping communication operations
                which could lead to undefined behavior.

        Note:
            - For self-communication (`is_self_comm=True`), this method does nothing as no
              actual network communication is required.
            - After dispatching, the work handles are stored in `_work_to_finish` for later
              synchronization via `wait_until_finished()`.
            - This method should only be called after communication operations have been
              queued via `_prep_batch_isend_irecv()`.
        """
        if self.is_self_comm:
            return
        if self._work_to_finish is not None:
            raise RuntimeError("There is unfinished communication in queue. Cannot dispatch new communication")
        self._work_to_finish = dist.batch_isend_irecv(self._queue_send_recv)

    def wait_until_finished(self):
        """
        Wait for all dispatched communication operations to complete.

        This method blocks until all previously dispatched communication operations have
        finished. It ensures data consistency by synchronizing all pending send/receive
        operations before proceeding.

        Raises:
            RuntimeError: If called when there are no unfinished communications in the queue.
                This typically happens when `wait_until_finished()` is called without a
                preceding `_dispatch()` call.

        Note:
            - For self-communication (`is_self_comm=True`), this method returns immediately
              as no actual network communication needs to be synchronized.
            - After completion, the internal communication queue and work handles are reset,
              allowing new communication operations to be queued.
            - This method must be called after `_dispatch()` to ensure communication
              operations have completed before accessing the received data.
        """
        if self.is_self_comm:
            return
        if self._work_to_finish is None:
            raise RuntimeError("Cannot wait without unfinished communication in queue")
        for work in self._work_to_finish:
            work.wait()
        self._work_to_finish = None
        self._queue_send_recv = []

    def enqueue_to_dispatch(
        self,
        to_send: torch.Tensor,
        to_recv: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Enqueue a communication operation and immediately dispatch it.

        This method combines the functionality of `_prep_batch_isend_irecv()` and `_dispatch()`
        in a single call. It prepares the tensors for communication, queues the operations,
        and immediately dispatches them for execution.

        Args:
            to_send (torch.Tensor): The tensor to be sent to the target rank.
            to_recv (Optional[torch.Tensor], optional): The tensor buffer to receive data into.
                If None, a new tensor with the same shape and properties as `to_send` will be created.

        Returns:
            torch.Tensor: The tensor that will contain the received data. For self-communication,
                this contains the copied data immediately. For inter-rank communication, this is
                the buffer where data will be received once the communication completes.

        Note:
            - For self-communication (`is_self_comm=True`), the data is immediately available
              in the returned tensor.
            - For inter-rank communication, you must call `wait_until_finished()` before
              accessing the data in the returned tensor to ensure the communication has completed.
            - This is a convenience method that internally calls `_prep_batch_isend_irecv()`
              followed by `_dispatch()`.

        Example:
            ```python
            comm = One2OneComm(group, send_rank, recv_rank)
            recv_tensor = comm.enqueue_to_dispatch(send_tensor)
            comm.wait_until_finished()  # Wait for completion before using recv_tensor
            ```
        """
        recv = self._prep_batch_isend_irecv(to_send, to_recv)
        if self.is_self_comm:
            return recv
        self._dispatch()
        return recv


class TransposeComm(One2OneComm):
    def __init__(self, process_group: dist.ProcessGroup, group_layout: LayoutMap):
        if group_layout.shape is None:
            raise ValueError("group_layout must have a shape")

        self.world_size = dist.get_world_size(process_group)
        if self.world_size != group_layout.numel:
            raise ValueError("Inconsistent world_size with the num elements of group_layout")

        if len(group_layout.shape) != 2:
            raise ValueError(f"{self.__class__} only supports 2D group layout")

        if group_layout.shape[0] != group_layout.shape[1]:
            raise ValueError(f"group_layout.shape {group_layout.shape} is not square")

        self.group_layout = group_layout

        self.global_rank = dist.get_rank()
        self.group_rank = dist.get_rank(process_group)
        self.rank_coords: tuple[int, int] = self.group_layout.unravel(self.group_rank)

        transpose_group_rank = self.group_layout(self.rank_coords[::-1])
        self.transpose_rank = dist.get_global_rank(process_group, transpose_group_rank)

        self.parity_transpose = self.rank_coords[0] < self.rank_coords[1]

        # Call One2OneComm's __init__ instead of creating a separate comm instance
        super().__init__(process_group, transpose_group_rank, transpose_group_rank, parity=self.parity_transpose)

    def __deepcopy__(self, memo):
        return TransposeComm(self.group, self.group_layout)


def ternary_parity(my_rank: int, send_rank: int, recv_rank: int) -> bool:
    """
    Determines parity for communication ordering based on rank relationships.

    Used to establish consistent communication ordering between three ranks to avoid deadlocks.
    Returns True if the current rank is less than both the send and receive ranks.

    Args:
        my_rank: Current process rank
        send_rank: Rank to send data to
        recv_rank: Rank to receive data from

    Returns:
        bool: True if current rank is less than both send and receive ranks, False otherwise
    """
    return my_rank < min(send_rank, recv_rank)


class Ring2DComm:
    """
    Implements communication primitives for distributed operations on a 2D grid of devices.

    This class provides general-purpose ring communication patterns for operations like
    TriangularMultiplication and OuterProductMean across a 2D grid of devices. Unlike
    Ring2DCommTriAttn which is specialized for triangular attention, this class provides
    more general ring communication patterns.

    The communication patterns implemented include:
    1. Transpose communication for matrix operations
    2. Row-wise ring communication (left shifts)
    3. Column-wise ring communication (up shifts)

    Parameters
    ----------
    group_2d : dist.ProcessGroup
        The process group representing the 2D grid of devices. This should include
        all processes in the distributed computation.
    group_col : dist.ProcessGroup
        A subprocess group that provides communication between ranks in the same column.
    group_layout : LayoutMap
        A mapping from the 2D grid index to the flattened index of the devices on the 2D grid.
        Must represent a square grid (same dimensions in both axes).

    Notes
    -----
    The class implements various communication patterns needed for distributed matrix
    operations, including initial communication (with different shift patterns based on
    coordinates) and subsequent iterations (with fixed shifts).

    Communication ordering is carefully managed to prevent deadlocks by using
    ternary_parity to determine consistent send/receive ordering across different ranks.
    """

    def __init__(
        self,
        group_2d: dist.ProcessGroup,
        group_col: dist.ProcessGroup,
        group_layout: LayoutMap,
    ):
        """
        Ring comm over a 2d grid of devices with comm happening along both axes
        Arguments:
            group_2d: Group torch process group that provides communication
                across the full cross-device
            group_col: Subprocess group that provides communication
                between ranks in the same column
            group_layout: mapping from the 2d grid index to the flatten index
            of the devices on the 2d grid
        """
        # TODO: consolidate the ring 2d comm groups with other modules e,g. triangle attn
        self.group_2d = group_2d
        self.group_col = group_col
        self.group_layout = group_layout
        ranks_group_2d = set(dist.get_process_group_ranks(self.group_2d))
        ranks_group_col = set(dist.get_process_group_ranks(self.group_col))

        if not ranks_group_col.issubset(ranks_group_2d):
            raise ValueError("The col ranks are not a subset of ranks_group_2d")

        self.size_2d = dist.get_world_size(self.group_2d)

        if self.size_2d != self.group_layout.numel:
            raise ValueError(
                f"size of group_2d {self.size_2d} differs from the number of elements in group_layout {self.group_layout.numel}"
            )

        if self.group_layout.shape[0] != self.group_layout.shape[1]:
            raise ValueError(f"group_layout.shape {self.group_layout.shape} is not square")

        self.rank_2d = dist.get_rank(self.group_2d)
        self.coord_2d = self.group_layout.unravel(self.rank_2d)

        # all the send/recv ranks must be global in order to use isend/irecv
        # only need transpose at the beginning of the batched GEMM for b or a
        self.comm_2d_trans = TransposeComm(self.group_2d, self.group_layout)

        # always do left shift per row
        # for iteration 0, i'th row left shift by i column
        self.send_rank_row_init = get_group_rank_from_axial_shift(
            self.coord_2d, 1, -self.coord_2d[0], self.group_layout
        )
        self.recv_rank_row_init = get_group_rank_from_axial_shift(self.coord_2d, 1, self.coord_2d[0], self.group_layout)

        self.comm_row_init = One2OneComm(
            self.group_2d,
            self.send_rank_row_init,
            self.recv_rank_row_init,
            parity=ternary_parity(self.rank_2d, self.send_rank_row_init, self.recv_rank_row_init),
        )
        # for other iterations left shift by 1 col
        self.send_rank_row = get_group_rank_from_axial_shift(self.coord_2d, 1, -1, self.group_layout)
        self.recv_rank_row = get_group_rank_from_axial_shift(self.coord_2d, 1, 1, self.group_layout)

        self.comm_row = One2OneComm(
            self.group_2d,
            self.send_rank_row,
            self.recv_rank_row,
            parity=ternary_parity(self.rank_2d, self.send_rank_row, self.recv_rank_row),
        )

        # always do up shift per col
        # for iteration 0, j'th col up shift by j row
        self.send_rank_col_init = get_group_rank_from_axial_shift(
            self.coord_2d, 0, -self.coord_2d[1], self.group_layout
        )
        self.recv_rank_col_init = get_group_rank_from_axial_shift(self.coord_2d, 0, self.coord_2d[1], self.group_layout)
        self.comm_col_init = One2OneComm(
            self.group_2d,
            self.send_rank_col_init,
            self.recv_rank_col_init,
            parity=ternary_parity(self.rank_2d, self.send_rank_col_init, self.recv_rank_col_init),
        )
        # for other iterations, up shift by 1 row
        self.send_rank_col = get_group_rank_from_axial_shift(self.coord_2d, 0, -1, self.group_layout)
        self.recv_rank_col = get_group_rank_from_axial_shift(self.coord_2d, 0, 1, self.group_layout)
        self.comm_col = One2OneComm(
            self.group_2d,
            self.send_rank_col,
            self.recv_rank_col,
            parity=ternary_parity(self.rank_2d, self.send_rank_col, self.recv_rank_col),
        )

        # fused communication for transposition and initial row/col shift in backward
        coords_transpose = self.coord_2d[::-1]
        self.send_rank_transpose_row_init = get_group_rank_from_axial_shift(
            coords_transpose, 1, -coords_transpose[0], self.group_layout
        )  # shifting the transposed rank
        recv_rank_transpose_row_init = get_group_rank_from_axial_shift(
            self.coord_2d, 1, self.coord_2d[0], self.group_layout
        )  # counter-shifting
        self.recv_rank_transpose_row_init = self.group_layout(
            self.group_layout.unravel(recv_rank_transpose_row_init)[::-1]
        )  # counter-transposition
        self.comm_transpose_row_init = One2OneComm(
            self.group_2d,
            self.send_rank_transpose_row_init,
            self.recv_rank_transpose_row_init,
            parity=ternary_parity(self.rank_2d, self.send_rank_transpose_row_init, self.recv_rank_transpose_row_init),
        )

        self.send_rank_transpose_col_init = get_group_rank_from_axial_shift(
            coords_transpose, 0, -coords_transpose[1], self.group_layout
        )  # shifting the transposed rank
        recv_rank_transpose_col_init = get_group_rank_from_axial_shift(
            self.coord_2d, 0, self.coord_2d[1], self.group_layout
        )  # counter-shifting
        self.recv_rank_transpose_col_init = self.group_layout(
            self.group_layout.unravel(recv_rank_transpose_col_init)[::-1]
        )  # counter-transposition
        self.comm_transpose_col_init = One2OneComm(
            self.group_2d,
            self.send_rank_transpose_col_init,
            self.recv_rank_transpose_col_init,
            parity=ternary_parity(self.rank_2d, self.send_rank_transpose_col_init, self.recv_rank_transpose_col_init),
        )


# ============================================================================================== #
# The single-hop ring exchange the 1-D outgoing path rotates its second operand with: send to
# `rank+1`, receive from `rank-1`, both posted before either is waited on.
# ============================================================================================== #


def _ring_p2p_send_recv(
    sends: list[torch.Tensor],
    recvs: list[torch.Tensor],
    send_to: int,
    recv_from: int,
    group: dist.ProcessGroup,
    parity: bool,
) -> list:
    """Issue async P2P send+recv for one or more tensor pairs.

    All send/recv operations are batched into a single ``batch_isend_irecv``
    call to avoid gloo backend issues with concurrent P2P dispatches from
    the same ranks.

    Returns an empty list for self-communication (cp_size=1).  The caller
    must call ``w.wait()`` on each handle before reading recvs.
    """
    assert len(sends) == len(recvs), f"sends/recvs length mismatch: {len(sends)} vs {len(recvs)}"
    rank = dist.get_rank(group)
    if send_to == rank and recv_from == rank:
        for s, r in zip(sends, recvs):
            r.copy_(s)
        return []

    send_to_global = dist.get_global_rank(group, send_to)
    recv_from_global = dist.get_global_rank(group, recv_from)

    ops = []
    for s, r in zip(sends, recvs):
        if parity:
            ops.append(dist.P2POp(dist.isend, s, send_to_global, group=group))
            ops.append(dist.P2POp(dist.irecv, r, recv_from_global, group=group))
        else:
            ops.append(dist.P2POp(dist.irecv, r, recv_from_global, group=group))
            ops.append(dist.P2POp(dist.isend, s, send_to_global, group=group))
    return dist.batch_isend_irecv(ops)


# ============================================================================================== #
# LayerNorm over a SHARDED activation with REPLICATED parameters. The normalization axis is the
# feature axis, which no CP mesh shards, so the forward is purely local -- what the explicit
# autograd `Function` buys is a backward whose parameter gradient is reduced across the mesh
# exactly once, instead of inferred per-op by DTensor's sharding propagation.
# ============================================================================================== #


class _LayerNormParamsReplicatedImpl(torch.autograd.Function):
    """
    A custom implementation of LayerNorm with replicated parameters for distributed training.

    This class provides a forward and backward implementation of LayerNorm, ensuring compatibility
    with distributed tensor placements and device meshes. It supports replicated and sharded
    placements for input tensors and replicated placements for weight and bias tensors.

    NOTE: by default, avg reduce over the Replicate placements of the weight and bias gradients
    is performed. This is to ensure identical parameter updates across all ranks and avoid
    gradual divergence during training. This can be disabled by setting
    avg_over_replicate_param_grad to False.

    Methods:
        forward(ctx, x, normalized_shape, weight, bias, eps):
            Computes the forward pass of LayerNorm.

        backward(ctx, grad_output):
            Computes the backward pass of LayerNorm.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        x: DTensor,
        normalized_shape: list[int],
        weight: Optional[DTensor],
        bias: Optional[DTensor],
        eps: float,
        cast_params_dtype_to_x: Optional[bool] = False,
        avg_over_replicate_param_grad: bool = True,
    ) -> DTensor:
        """
        Forward pass of LayerNorm with replicated parameters.

        Args:
            ctx: Context for saving tensors for backward computation.
            x (DTensor): Input tensor.
            normalized_shape (list[int]): Shape of the input tensor to normalize.
            weight (Optional[DTensor]): Weight tensor for affine transformation.
            bias (Optional[DTensor]): Bias tensor for affine transformation.
            eps (float): A small value added for numerical stability.
            cast_params_dtype_to_x (Optional[bool]): whether to cast the dtype of
                the weights and bias to the dtype of the input tensor
            avg_over_replicate_param_grad (bool): Whether to perform avg reduce over the
                Replicate placements of the weight and bias gradients. For example,
                if the input DTensor x.placements = (Shard(0), Replicate()), this layer's
                parameters' gradients.placements = (Partial("sum"), Replicate()) if
                self._avg_over_replicate_param_grad is False; otherwise, it will be
                (Partial("sum"), Partial("avg")). The motivation is to ensure identical
                parameter updates across all ranks and avoid gradual divergence during
                training.

        Returns:
            DTensor: The normalized output tensor.
        """
        if not isinstance(x, DTensor):
            dtensor_instance = x
            raise TypeError(
                ", ".join(
                    [
                        f"DTensor instance '{dtensor_instance}' should have type {DTensor}",
                        f"but instead has type {type(dtensor_instance)}.",
                    ]
                )
            )
        device_mesh = x.device_mesh
        ndim_device_mesh = device_mesh.ndim
        all_replicate_placements = tuple([Replicate()] * ndim_device_mesh)
        if weight is not None:
            if not isinstance(weight, DTensor):
                dtensor_instance = weight
                raise TypeError(
                    ", ".join(
                        [
                            f"DTensor instance '{dtensor_instance}' should have type {DTensor}",
                            f"but instead has type {type(dtensor_instance)}.",
                        ]
                    )
                )
            if weight.device_mesh != device_mesh:
                raise ValueError("weight and x must be on the same device mesh")
            if weight.placements != all_replicate_placements:
                raise ValueError("weight must be replicated on all device mesh dimensions")
        if bias is not None:
            if not isinstance(bias, DTensor):
                dtensor_instance = bias
                raise TypeError(
                    ", ".join(
                        [
                            f"DTensor instance '{dtensor_instance}' should have type {DTensor}",
                            f"but instead has type {type(dtensor_instance)}.",
                        ]
                    )
                )
            if bias.device_mesh != device_mesh:
                raise ValueError("bias and x must be on the same device mesh")
            if bias.placements != all_replicate_placements:
                raise ValueError("bias must be replicated on all device mesh dimensions")
        if weight is not None or bias is not None:
            if avg_over_replicate_param_grad:
                placements_grad_params = [Partial("avg")] * ndim_device_mesh
            else:
                placements_grad_params = list(weight.placements) if weight is not None else None
        else:
            placements_grad_params = None
        n_dim_norm = len(normalized_shape)
        for i_dim_device_mesh, p in enumerate(x.placements):
            if isinstance(p, Partial):
                # partial reduction along any input dimension requires complicated backward pass
                raise ValueError("Partial reduction along any input dimension is not supported")
            if isinstance(p, Shard):
                if p.dim >= x.ndim - n_dim_norm:
                    # the normalized dimensions must not be sharded by the device mesh
                    raise ValueError("LayerNorm's normalizing dimensions must not be sharded by the device mesh")
                if x.shape[p.dim] % device_mesh.shape[i_dim_device_mesh] != 0:
                    raise ValueError(
                        f"Uneven sharding tensor dimension {p.dim} of size {x.shape[p.dim]} "
                        f"along device mesh dimension {i_dim_device_mesh} of size "
                        f"{device_mesh.shape[i_dim_device_mesh]} is not supported"
                    )
                # the only supported placement for the input is Shard, which corresponding
                # to the backward's grad partial sum. Otherwise, we can only support Replicate
                # placements for other device mesh dimensions. Also, by using the Partial("sum")
                # placement on the params, the all_reduce is postponed for the params' gradients
                # until needed
                if weight is not None or bias is not None:
                    placements_grad_params[i_dim_device_mesh] = Partial("sum")
            elif not isinstance(p, Replicate):
                raise ValueError(f"Unsupported x's placements along {i_dim_device_mesh} axis of the device mesh: {p}")
        ctx.device_mesh = device_mesh
        # will use x.placements for the x.grad in the backward pass, i.e., this function
        # enforces consistent placements for the input and its gradient
        ctx.placements_x = x.placements
        ctx.placements_grad_params = placements_grad_params

        # Save weight and bias shapes and strides for backward pass
        if weight is not None:
            ctx.weight_shape = weight.shape
            ctx.weight_stride = weight.stride()
        if bias is not None:
            ctx.bias_shape = bias.shape
            ctx.bias_stride = bias.stride()

        weight_needs_grad = weight is not None and weight.requires_grad
        bias_needs_grad = bias is not None and bias.requires_grad
        # IMPORTANT: no modification on *_local for the rest of the code
        x_local = x.to_local()
        if weight is None:
            weight_local = None
        else:
            weight_local = weight.to_local()
            if cast_params_dtype_to_x:
                weight_local = weight_local.to(x.dtype)
        if bias is None:
            bias_local = None
        else:
            bias_local = bias.to_local()
            if cast_params_dtype_to_x:
                bias_local = bias_local.to(x.dtype)
        # For unknown reasons, using ctx.needs_input_grad in the forward pass can occasionally
        # cause NCCL hanging. ctx.need_input_grad should not be accessed during the forward pass
        # according to this discussion on pytorch forum:
        # https://discuss.pytorch.org/t/is-there-a-diffrence-between-ctx-needs-input-grad-behaviour-vs-input-tensor-requires-grad/195063/2
        if x.requires_grad or weight_needs_grad or bias_needs_grad:
            ctx.eps = eps
            ctx.normalized_shape = normalized_shape

            if not x.requires_grad:
                weight = None

            ctx.save_for_backward(x_local, weight_local)

        output_local = F.layer_norm(x_local, normalized_shape, weight_local, bias_local, eps)
        # LayerNorm does not change input's shape
        output = DTensor.from_local(
            output_local,
            device_mesh=device_mesh,
            placements=x.placements,
            shape=x.shape,
            stride=x.stride(),
        )
        return output

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(
        ctx, grad_output: DTensor
    ) -> tuple[Optional[DTensor], None, Optional[DTensor], Optional[DTensor], None, None]:
        """
        Backward pass of LayerNorm with replicated parameters.

        Args:
            ctx: Context containing saved tensors and attributes from the forward pass.
            grad_output (DTensor): Gradient of the output tensor.

        Returns:
            tuple: Gradients for input, weight, bias, and other parameters.
        """
        x_local, weight_local = ctx.saved_tensors
        eps = ctx.eps
        normalized_shape = ctx.normalized_shape

        # IMPORTANT: no modification on *_local for the rest of the code
        grad_output_local = grad_output.to_local()

        ids_dim_norm = tuple(-(i + 1) for i in range(len(normalized_shape)))
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[2]:
            mean_local = x_local.mean(dim=ids_dim_norm, keepdim=True)
            var_local = x_local.var(dim=ids_dim_norm, unbiased=False, keepdim=True)
            x_norm_local = (x_local - mean_local) / torch.sqrt(var_local + eps)

        if ctx.needs_input_grad[0]:
            if weight_local is not None:
                dy_local = grad_output_local * weight_local.view(
                    *([1] * (grad_output_local.ndim - len(normalized_shape))), *weight_local.shape
                )
            else:
                dy_local = grad_output_local

            dy_mean_local = dy_local.mean(dim=ids_dim_norm, keepdim=True)
            dy_x_norm_mean_local = (dy_local * x_norm_local).mean(dim=ids_dim_norm, keepdim=True)
            grad_input_local = (dy_local - dy_mean_local - x_norm_local * dy_x_norm_mean_local) / torch.sqrt(
                var_local + eps
            )
            # LayerNorm does not change input's shape in both forward and backward passes
            grad_input = DTensor.from_local(
                grad_input_local,
                device_mesh=ctx.device_mesh,
                placements=ctx.placements_x,
                shape=grad_output.shape,
                stride=grad_output.stride(),
            )
        else:
            grad_input = None

        reduce_dims = list(range(grad_output_local.ndim - len(normalized_shape)))
        if ctx.needs_input_grad[2]:
            grad_weight_local = (grad_output_local * x_norm_local).sum(dim=reduce_dims)
            # all-replicate weight implies identical shape and stride across all ranks
            grad_weight = DTensor.from_local(
                grad_weight_local,
                device_mesh=ctx.device_mesh,
                placements=ctx.placements_grad_params,
                shape=ctx.weight_shape,
                stride=ctx.weight_stride,
            )
        else:
            grad_weight = None

        if ctx.needs_input_grad[3]:
            grad_bias_local = grad_output_local.sum(dim=reduce_dims)
            # all-replicate weight implies identical shape and stride across all ranks
            grad_bias = DTensor.from_local(
                grad_bias_local,
                device_mesh=ctx.device_mesh,
                placements=ctx.placements_grad_params,
                shape=ctx.bias_shape,
                stride=ctx.bias_stride,
            )
        else:
            grad_bias = None

        return grad_input, None, grad_weight, grad_bias, None, None, None


class LayerNormParamsReplicated(nn.Module):
    """
    A LayerNorm module with replicated parameters for distributed training.

    This module wraps around `_LayerNormParamsReplicatedImpl` to provide a user-friendly interface
    for LayerNorm operations using the DTensor API. It supports distributed training with replicated
    and sharded placements for input tensors and replicated placements for weight and bias tensors.

    NOTE: by default, avg reduce over the Replicate placements of the weight and bias gradients
    is performed. This is to ensure identical parameter updates across all ranks and avoid
    gradual divergence during training. This can be disabled by setting
    avg_over_replicate_param_grad to False.

    Args:
        layer_local (nn.LayerNorm): An already-initialized nn.LayerNorm instance.
        device_mesh (DeviceMesh): The device mesh for distributed training.
        avg_over_replicate_param_grad (bool): Whether to perform avg reduce over the
            Replicate placements of the weight and bias gradients. For example,
            if the input DTensor x.placements = (Shard(0), Replicate()), this layer's
            parameters' gradients.placements = (Partial("sum"), Replicate()) if
            self._avg_over_replicate_param_grad is False; otherwise, it will be
            (Partial("sum"), Partial("avg")). The motivation is to ensure identical
            parameter updates across all ranks and avoid gradual divergence during
            training.
    """

    def __init__(
        self, layer_local: nn.LayerNorm, device_mesh: DeviceMesh, avg_over_replicate_param_grad: bool = True
    ) -> None:
        if not isinstance(layer_local, nn.LayerNorm):
            raise TypeError("layer_local is not an instance of nn.LayerNorm")
        if layer_local.weight is not None and layer_local.weight.device.type != device_mesh.device_type:
            raise ValueError(
                f"layer_local.weight and device_mesh are not on the same device type: "
                f"{layer_local.weight.device.type} != {device_mesh.device_type}"
            )
        if layer_local.bias is not None and layer_local.bias.device.type != device_mesh.device_type:
            raise ValueError(
                f"layer_local.bias and device_mesh are not on the same device type: "
                f"{layer_local.bias.device.type} != {device_mesh.device_type}"
            )

        super().__init__()
        self.normalized_shape = layer_local.normalized_shape
        self.eps = layer_local.eps
        self.device_mesh = device_mesh
        self.elementwise_affine = layer_local.elementwise_affine
        self._avg_over_replicate_param_grad = avg_over_replicate_param_grad

        all_replicate_placements = [Replicate()] * device_mesh.ndim

        if layer_local.weight is None:
            self.register_parameter("weight", None)
        else:
            self.weight = nn.Parameter(
                distribute_tensor(layer_local.weight.data, device_mesh, all_replicate_placements)
            )
        if layer_local.bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(distribute_tensor(layer_local.bias.data, device_mesh, all_replicate_placements))

    def forward(self, x: DTensor) -> DTensor:
        """
        Forward pass of LayerNormParamsReplicated.

        Args:
            x (DTensor): Input tensor.

        Returns:
            DTensor: The normalized output tensor.
        """
        return _LayerNormParamsReplicatedImpl.apply(
            x,
            self.normalized_shape,
            self.weight,
            self.bias,
            self.eps,
            True,
            self._avg_over_replicate_param_grad,
        )


# ============================================================================================== #
# Linear projection over a SHARDED activation with REPLICATED weight -- the feature-axis twin of
# the LayerNorm above, and the same reason for an explicit autograd `Function`.
# ============================================================================================== #


class _LinearParamsReplicatedImpl(torch.autograd.Function):
    """
    Custom autograd Function implementation for distributed linear operation with replicated parameters.

    The main purpose of this implementation is to avoid the unnecessary overhead seen the the
    equivalent distribute_module-wrapped linear layer, where the output tensors have nonsensical Replicate
    placements along device mesh dimensions that are not intended

    This implementation handles the forward and backward passes for a distributed linear layer where
    parameters (weight and bias) are replicated across the device mesh. The input tensor can have
    various placement strategies.

    NOTE: by default, avg reduce over the Replicate placements of the weight and bias gradients
    is performed. This is to ensure identical parameter updates across all ranks and avoid
    gradual divergence during training. This can be disabled by setting
    avg_over_replicate_param_grad to False.

    Assumptions and requirements:
        (see the respective docstring for forward and backward)
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        x: DTensor,
        weight: DTensor,
        bias: Optional[DTensor],
        cast_params_dtype_to_x: bool = False,
        avg_over_replicate_param_grad: bool = True,
    ) -> DTensor:
        """
        Forward pass for the distributed linear operation.

        Assumptions and requirements:
        1. Parameters (weight and bias) must be replicated on all device mesh dimensions
        2. Input tensor and parameters must be on the same device mesh
        3. Feature/hidden dimension of the input must not be sharded across the device mesh
        4. Partial reduction along any input dimension is not supported
        5. Input and outputs must be on the same device mesh with the same placements

        Args:
            ctx: Context object to store information for backward pass
            x: Input tensor with arbitrary placement strategy
            weight: Weight tensor (must be replicated across all device mesh dimensions)
            bias: Optional bias tensor (must be replicated if provided)
            cast_params_dtype_to_x: whether to cast the dtype of the weight and bias
                to the dtype of the input tensor
            avg_over_replicate_param_grad: whether to perform avg reduce over the
                Replicate placements of the weight and bias gradients. For example,
                if the input DTensor x.placements = (Shard(0), Replicate()), this layer's
                parameters' gradients.placements = (Partial("sum"), Replicate()) if
                self._avg_over_replicate_param_grad is False; otherwise, it will be
                (Partial("sum"), Partial("avg")). The motivation is to ensure identical
                parameter updates across all ranks and avoid gradual divergence during
                training.

        Returns:
            Output tensor with same placement strategy as input

        Raises:
            ValueError: If any of the placement requirements are violated
        """
        device_mesh = x.device_mesh
        if weight.device_mesh != device_mesh:
            raise ValueError("weight and x must be on the same device mesh")
        if bias is not None and bias.device_mesh != device_mesh:
            raise ValueError("bias and x must be on the same device mesh")
        ndim_device_mesh = device_mesh.ndim
        all_replicate_placements = tuple([Replicate()] * ndim_device_mesh)
        if weight.placements != all_replicate_placements:
            raise ValueError("weight must be replicated on all device mesh dimensions")
        if bias is not None and bias.placements != all_replicate_placements:
            raise ValueError("bias must be replicated on all device mesh dimensions")
        if avg_over_replicate_param_grad:
            placements_grad_params = [Partial("avg")] * ndim_device_mesh
        else:
            # all-replicate placements
            placements_grad_params = list(weight.placements)
        for i_dim_device_mesh, p in enumerate(x.placements):
            if isinstance(p, Partial):
                # partial reduction along any input dimension requires complicated backward pass
                raise ValueError("Partial reduction along any input dimension is not supported")
            if isinstance(p, Shard):
                if p.dim == x.ndim - 1:
                    # the feature or hidden dimension must not be a part of the device mesh
                    raise ValueError("feature or hidden dimension must not be a part of the device mesh")
                if x.shape[p.dim] % device_mesh.shape[i_dim_device_mesh] != 0:
                    raise ValueError(
                        f"Uneven sharding tensor dimension {p.dim} of size {x.shape[p.dim]} "
                        f"along device mesh dimension {i_dim_device_mesh} of size "
                        f"{device_mesh.shape[i_dim_device_mesh]} is not supported"
                    )
                # the only supported placement for the input is Shard, which corresponding
                # to the backward's grad partial sum. Otherwise, we can only support Replicate
                # placements for other device mesh dimensions. Also, by using the Partial("sum")
                # placement on the params, the all_reduce is postponed for the params' gradients
                # until needed
                placements_grad_params[i_dim_device_mesh] = Partial("sum")
            elif not isinstance(p, Replicate):
                raise ValueError(f"Unsupported x's placements along {i_dim_device_mesh} axis of the device mesh: {p}")
        ctx.device_mesh = device_mesh
        # will use x.placements for the x.grad in the backward pass, i.e., this function
        # enforces consistent placements for the input and its gradient
        ctx.placements_x = x.placements
        ctx.placements_grad_params = placements_grad_params
        ctx.shape_input = x.shape
        ctx.stride_input = x.stride()
        ctx.weight_shape = weight.shape
        ctx.weight_stride = weight.stride()
        ctx.dtype_input = x.dtype
        ctx.dtype_weight = weight.dtype
        if bias is not None:
            ctx.bias_shape = bias.shape
            ctx.bias_stride = bias.stride()
            ctx.dtype_bias = bias.dtype
        else:
            ctx.dtype_bias = None
        x_local = x.to_local()
        weight_local = weight.to_local()
        bias_local = None if bias is None else bias.to_local()

        # Save original-precision locals for backward *before* any dtype cast.
        # Native autocast saves fp32 weights and lets the backward autocast
        # context handle further casts.  Saving the bf16-cast version would
        # bake in bf16 rounding on CPU (where custom_bwd does NOT restore
        # autocast), silently lowering gradient precision.
        if x.requires_grad or weight.requires_grad or (bias is not None and bias.requires_grad):
            ctx.save_for_backward(
                x_local.detach().clone() if weight.requires_grad else None,
                weight_local.detach().clone() if x.requires_grad else None,
            )

        if cast_params_dtype_to_x:
            weight_local = weight_local.to(x.dtype)
            if bias_local is not None:
                bias_local = bias_local.to(x.dtype)
        # Extract the local shard to perform the linear operation.
        # This enforces local matrix multiplication without any communication given that:
        # 1. the linear operation is performed locally on each rank along the hidden dimension,
        #    which is agnostic to the device mesh dimensions
        # 2. the weight and bias are replicated on all device mesh dimensions
        # 3. the output has the same placements as the input
        output_local = torch.nn.functional.linear(x_local, weight_local, bias_local)
        # linear only change the last dimension of the input so we need to
        # modify the output shape and strides accordingly
        shape_output = tuple(x.shape[:-1]) + (output_local.shape[-1],)
        strides_output = update_exhaustive_strides(x.shape, x.stride(), shape_output)
        output = DTensor.from_local(output_local, device_mesh, x.placements, shape=shape_output, stride=strides_output)
        return output

    @staticmethod
    def _all_reduce_grad_gteqfp32(
        grad: torch.Tensor,
        device_mesh: DeviceMesh,
        placements: list,
        target_dtype: torch.dtype,
    ) -> torch.Tensor:
        """All-reduce a parameter gradient in at least fp32 across mesh dims.

        For each mesh dimension with a ``Partial`` placement, performs an
        all-reduce in at least float32 to avoid bf16/fp16 accumulation errors.
        Only the parameter-sized gradient is promoted — not the large
        activation tensors.  If the gradient is already >=fp32, it is reduced
        in its native dtype.
        """
        needs_reduce = any(isinstance(p, Partial) and device_mesh.size(dim) > 1 for dim, p in enumerate(placements))
        if not needs_reduce:
            return grad.to(target_dtype)

        reduce_dtype = torch.promote_types(grad.dtype, torch.float32)
        grad = grad.to(reduce_dtype).contiguous()
        for mesh_dim, p in enumerate(placements):
            if not isinstance(p, Partial) or device_mesh.size(mesh_dim) <= 1:
                continue
            group = device_mesh.get_group(mesh_dim)
            if p.reduce_op == "sum":
                dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=group)
            elif p.reduce_op == "avg":
                dist.all_reduce(grad, op=dist.ReduceOp.AVG, group=group)
            else:
                raise ValueError(f"Unsupported reduce_op {p.reduce_op!r} in _all_reduce_grad_gteqfp32")
        return grad.to(target_dtype)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(
        ctx, grad_output: DTensor
    ) -> tuple[Optional[DTensor], Optional[DTensor], Optional[DTensor], None, None]:
        """Backward pass for the distributed linear operation.

        Local einsum stays in the compute dtype (bf16 MMA accumulates in
        fp32 internally on CUDA, so local results are accurate).  The
        precision hazard is cross-rank reduction: implicit ``Partial("SUM")``
        would reduce in bf16, accumulating errors.  We manually all-reduce
        ``dw``/``db`` in fp32 via ``_all_reduce_grad_gteqfp32`` and return
        them with ``Replicate`` placements.

        On CPU (unit tests), ``custom_bwd`` does not restore autocast, so
        we explicitly cast operands to the compute dtype.
        """
        if grad_output.device_mesh != ctx.device_mesh:
            raise ValueError(
                "_LinearParamsReplicatedImpl: different device mesh between grad_output and the forward input"
            )
        x_local, weight_local = ctx.saved_tensors

        if grad_output.placements != ctx.placements_x:
            # DTensor's backward may spuriously all_gather to Replicate();
            # redistribute back to the input's placements.
            grad_output = grad_output.redistribute(ctx.device_mesh, ctx.placements_x)

        grad_output_local = grad_output.to_local()
        all_replicate = tuple([Replicate()] * ctx.device_mesh.ndim)

        # Compute dtype (e.g. bf16 under mixed precision).  We cast operands
        # to this dtype explicitly for CPU compatibility — on CUDA, custom_bwd
        # restores autocast which handles this automatically.
        go_dtype = grad_output_local.dtype

        if ctx.needs_input_grad[1]:
            dw_local = torch.einsum("...i,...o->io", grad_output_local, x_local.to(go_dtype))
            dw_local = _LinearParamsReplicatedImpl._all_reduce_grad_gteqfp32(
                dw_local, ctx.device_mesh, ctx.placements_grad_params, ctx.dtype_weight
            )
            dw = DTensor.from_local(
                dw_local, ctx.device_mesh, all_replicate, shape=ctx.weight_shape, stride=ctx.weight_stride
            )
        else:
            dw = None

        if ctx.needs_input_grad[2]:
            dims = list(range(grad_output_local.ndim - 1))
            if ctx.dtype_bias is None:
                raise RuntimeError("bias gradient requested but bias dtype metadata is missing")
            db_local = grad_output_local.sum(dim=dims)
            db_local = _LinearParamsReplicatedImpl._all_reduce_grad_gteqfp32(
                db_local, ctx.device_mesh, ctx.placements_grad_params, ctx.dtype_bias
            )
            db = DTensor.from_local(
                db_local, ctx.device_mesh, all_replicate, shape=ctx.bias_shape, stride=ctx.bias_stride
            )
        else:
            db = None

        if ctx.needs_input_grad[0]:
            if weight_local is None:
                raise RuntimeError("input gradient requested but saved weight tensor is missing")
            grad_input_local = torch.einsum("...i,io->...o", grad_output_local, weight_local.to(go_dtype))
            grad_input = DTensor.from_local(
                grad_input_local, ctx.device_mesh, ctx.placements_x, shape=ctx.shape_input, stride=ctx.stride_input
            )
        else:
            grad_input = None

        return grad_input, dw, db, None, None


class LinearParamsReplicated(nn.Module):
    """
    Distributed linear layer with parameters replicated across all device mesh dimensions.

    This is almost equivalent to
    ```python
    layer = torch.distributed.tensor.distribute_module(layer_local, device_mesh)
    ```
    with the exception that the torch.distributed.tensor.distribute_module version will incur
    significant overhead due to the unnecessary replication of the output tensor along certain
    device mesh dimensions.

    This class avoids such unnecessary overhead by using the custom _LinearParamsReplicatedImpl
    autograd function for forward and backward pass computation instead of relying on the distributed
    module's forward implementation.

    NOTE: by default, avg reduce over the Replicate placements of the weight and bias gradients
    is performed. This is to ensure identical parameter updates across all ranks and avoid
    gradual divergence during training. This can be disabled by setting
    avg_over_replicate_param_grad to False.

    Key requirements:
        1. Parameters (weight and bias) will replicated on all device mesh dimensions
        2. Input tensor and parameters must be on the same device mesh
        3. Feature/hidden dimension of the input must not be sharded across the device mesh
        4. Partial reduction along any input dimension is not supported
        5. Input and outputs must be on the same device mesh with the same placements
        6. Gradients of the input have the same placements on the same device mesh as the input
        7. Gradients of the weight and bias have Partial("sum") placements along the input's Shard placements'
           dimension so that the all-reduce will be performed along those device-grid dimensions

    """

    def __init__(self, layer_local: nn.Linear, device_mesh: DeviceMesh, avg_over_replicate_param_grad: bool = True):
        """
        Initialize the distributed linear layer.

        Args:
            layer_local: nn.Linear to be distributed
            device_mesh: Device mesh for distributed computation
            avg_over_replicate_param_grad: whether to perform avg reduce over the
                Replicate placements of the weight and bias gradients. For example,
                if the input DTensor x.placements = (Shard(0), Replicate()), this layer's
                parameters' gradients.placements = (Partial("sum"), Replicate()) if
                self._avg_over_replicate_param_grad is False; otherwise, it will be
                (Partial("sum"), Partial("avg")). The motivation is to ensure identical
                parameter updates across all ranks and avoid gradual divergence during
                training.
        """
        if not isinstance(layer_local, nn.Linear):
            raise ValueError("layer_local is not an instance of nn.Linear")
        if layer_local.weight.device.type != device_mesh.device_type:
            raise ValueError(
                f"layer_local.weight and device_mesh are not on the same device type: "
                f"{layer_local.weight.device.type} != {device_mesh.device_type}"
            )
        if layer_local.bias is not None and layer_local.bias.device.type != device_mesh.device_type:
            raise ValueError(
                f"layer_local.bias and device_mesh are not on the same device type: "
                f"{layer_local.bias.device.type} != {device_mesh.device_type}"
            )
        super().__init__()
        all_replicate_placements = [Replicate()] * device_mesh.ndim
        self.weight = nn.Parameter(
            distribute_tensor(layer_local.weight.data, device_mesh, all_replicate_placements),
            requires_grad=layer_local.weight.requires_grad,
        )
        if layer_local.bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(
                distribute_tensor(layer_local.bias.data, device_mesh, all_replicate_placements),
                requires_grad=layer_local.bias.requires_grad,
            )
        self._avg_over_replicate_param_grad = avg_over_replicate_param_grad

    def forward(self, input: DTensor) -> DTensor:
        """
        Forward pass for the distributed linear layer.

        Uses the custom _LinearParamsReplicatedImpl autograd function to perform the computation
        efficiently while preserving correct autograd behavior for distributed tensors.

        Args:
            input: Input DTensor with appropriate placement strategy

        Returns:
            Output DTensor with same placement strategy as input
        """
        return _LinearParamsReplicatedImpl.apply(
            input,
            self.weight,
            self.bias,
            True,  # cast_params_dtype_to_x: under bf16-mixed autocast, upstream
            # ops produce bf16 activations while weights stay fp32.  custom_fwd
            # disables autocast inside the function, so F.linear would get
            # mismatched dtypes.  Casting weight to input dtype matches what
            # native autocast does for F.linear.  No-op when dtypes already match.
            self._avg_over_replicate_param_grad,
        )


# ============================================================================================== #
# `x * sigmoid(g)` on two identically-sharded DTensors, fused into one local op so the gate is
# never materialized as a separate distributed tensor.
# ============================================================================================== #


class _SigmoidGateImpl(torch.autograd.Function):
    """Distributed implementation of sigmoid gating using DTensors.

    This autograd function implements a distributed sigmoid gating operation that applies
    a sigmoid-activated gate to an input tensor. The operation is performed element-wise
    across distributed tensors while maintaining proper gradient computation.

    The sigmoid gate computes:
        output = x * sigmoid(g)

    Key features:
    - Distributed computation across device meshes with various sharding strategies
    - Memory-efficient implementation that operates on local tensor chunks
    - Supports gradient computation through custom backward pass
    - Validates tensor compatibility (device mesh, placements, shapes)

    Notes
    -----
    Input tensors must be DTensors with:
    - Identical device mesh and placements
    - Compatible shapes (x and g must have the same shape)
    - No Partial placements (not currently supported)
    """

    @staticmethod
    def forward(ctx, x: DTensor, g: DTensor) -> DTensor:
        """Forward pass of distributed sigmoid gating.

        Parameters
        ----------
        ctx : torch.autograd.function.BackwardCFrame
            Context object for saving information needed in backward pass.
        x : DTensor
            Input tensor to be gated. Can have any shape and sharding strategy.
        g : DTensor
            Gate tensor with pre-sigmoid values. Must have identical shape,
            device mesh, and placements as x.

        Returns
        -------
        DTensor
            Output tensor with shape identical to input tensors.
            Contains the result of x * sigmoid(g).

        Raises
        ------
        TypeError
            If inputs are not DTensors.
        ValueError
            If tensors have incompatible device meshes, placements, or if
            Partial placements are used (not supported).
        """
        if not isinstance(x, DTensor):
            raise TypeError(f"Input 'x' must be of type DTensor. Got type {type(x)}.")
        if not isinstance(g, DTensor):
            raise TypeError(f"Input 'g' must be of type DTensor. Got type {type(g)}.")

        device_mesh_input = x.device_mesh
        if g.device_mesh != device_mesh_input:
            raise ValueError(
                f"Input tensors 'x' and 'g' must have identical device mesh. "
                f"Got device meshes {device_mesh_input} and {g.device_mesh}."
            )

        placements_input = x.placements
        for i_dim_device_mesh, placement in enumerate(placements_input):
            if isinstance(placement, Partial):
                raise ValueError("Partial placements are not supported")
            if isinstance(placement, Shard):
                if x.shape[placement.dim] % device_mesh_input.shape[i_dim_device_mesh] != 0:
                    raise ValueError(
                        f"Uneven sharding tensor dimension {placement.dim} of size {x.shape[placement.dim]} "
                        f"along device mesh dimension {i_dim_device_mesh} of size "
                        f"{device_mesh_input.shape[i_dim_device_mesh]} is not supported"
                    )

        if g.placements != placements_input:
            raise ValueError(
                f"Input tensors 'x' and 'g' must have identical placements. "
                f"Got placements {placements_input} and {g.placements}."
            )

        input_shape = x.shape
        if input_shape != g.shape:
            raise ValueError(
                f"Input tensors 'x' and 'g' must have identical shapes. Got shapes {input_shape} and {g.shape}."
            )

        g_local = g.to_local().sigmoid()
        x_gated_local = x.to_local() * g_local

        ctx.save_for_backward(x_gated_local, g_local)
        ctx.device_mesh_input = device_mesh_input
        ctx.placements_input = placements_input
        ctx.input_shape = input_shape

        out = DTensor.from_local(
            x_gated_local,
            device_mesh=device_mesh_input,
            placements=placements_input,
            shape=x.shape,
            stride=x.stride(),
        )
        return out

    @staticmethod
    def backward(ctx, grad_output: DTensor) -> tuple[DTensor, DTensor]:
        """Backward pass of distributed sigmoid gating.

        Computes gradients with respect to both input tensor x and gate tensor g.

        The gradients are:
        - dx = grad_output * sigmoid(g)
        - dg = grad_output * x * sigmoid(g) * (1 - sigmoid(g))

        Parameters
        ----------
        ctx : torch.autograd.function.BackwardCFrame
            Context object containing saved tensors and metadata from forward pass.
        grad_output : DTensor
            Gradient of the loss with respect to the output tensor.
            Must have identical device mesh and placements as the input tensors.

        Returns
        -------
        tuple[DTensor, DTensor]
            Gradients with respect to x and g respectively.
            Both have the same shape and distribution as their corresponding inputs.

        Raises
        ------
        TypeError
            If grad_output is not a DTensor.
        ValueError
            If grad_output has incompatible device mesh or placements compared
            to the input tensors from the forward pass.
        """
        if not isinstance(grad_output, DTensor):
            raise TypeError(f"Input 'grad_output' must be of type DTensor. Got type {type(grad_output)}.")

        if grad_output.device_mesh != ctx.device_mesh_input:
            raise ValueError(
                f"Input 'grad_output' must have the same device mesh as the input tensor. "
                f"Got device meshes {grad_output.device_mesh} and {ctx.device_mesh_input}."
            )

        if grad_output.placements != ctx.placements_input:
            raise ValueError(
                f"Input 'grad_output' must have the same placements as the input tensor. "
                f"Got placements {grad_output.placements} and {ctx.placements_input}."
            )

        if grad_output.shape != ctx.input_shape:
            raise ValueError(
                f"Input 'grad_output' must have the same shape as the input tensor. "
                f"Got shapes {grad_output.shape} and {ctx.input_shape}."
            )

        x_gated_local, g_local = ctx.saved_tensors
        grad_output_local = grad_output.to_local()

        dx_local = grad_output_local * g_local
        dx = DTensor.from_local(
            dx_local,
            device_mesh=ctx.device_mesh_input,
            placements=ctx.placements_input,
            shape=grad_output.shape,
            stride=grad_output.stride(),
        )

        dg_local = grad_output_local * x_gated_local
        dg_local *= 1 - g_local
        dg = DTensor.from_local(
            dg_local,
            device_mesh=ctx.device_mesh_input,
            placements=ctx.placements_input,
            shape=grad_output.shape,
            stride=grad_output.stride(),
        )

        return dx, dg


def sigmoid_gate(x: DTensor, g: DTensor) -> DTensor:
    """Apply sigmoid gating to a distributed tensor.

    This function performs element-wise sigmoid gating: x * sigmoid(g), where both
    input and gate tensors are distributed across multiple devices. The operation
    is performed efficiently using local tensor operations while maintaining
    gradient computation capabilities.

    Parameters
    ----------
    x : DTensor
        Input tensor to be gated. Can have any shape and sharding strategy.
    g : DTensor
        Gate tensor with pre-sigmoid values. Must have identical shape,
        device mesh, and placements as x.

    Returns
    -------
    DTensor
        Gated output tensor with shape identical to input tensors.
        Contains the result of x * sigmoid(g).

    Examples
    --------
    >>> # Assume we have distributed tensors x and g with shape (B, N, D)
    >>> output = sigmoid_gate(x, g)
    >>> # output = x * torch.sigmoid(g), computed in distributed fashion

    Notes
    -----
    - Both input tensors must be DTensors with compatible device meshes and placements
    - Partial placements are not currently supported
    - The function is differentiable and supports gradient computation
    - The operation is performed on local tensor chunks for efficiency
    """
    return _SigmoidGateImpl.apply(x, g)


# ============================================================================================== #
# Single-device TriangularMultiplication -- the parameter container, and the reference forward.
#
# CLEAN-ROOM. Written from this repository's OWN statement of the layer -- `fold_cp_ops.workflows
# .trimul_autotune.trimul_ref`, the fp32 oracle every fused combo in this package is checked
# against -- and from the published algebra of the triangular multiplicative update
# (`z_ij <- g_ij * LayerNorm(sum_k a_ik * b_jk)`, with the two directions differing only in which
# token index the contraction runs over). Nothing here is transcribed from a third-party
# implementation.
#
# TWO deliberate absences, both consequences of writing this from the algebra rather than porting
# a training-time layer:
#
#   1. **No custom weight initializers.** A training implementation of this layer zero-initializes
#      the gate projections and the output projection so the residual branch starts as a no-op.
#      That choice is meaningless here and actively harmful: every one of the eight parameters is
#      OVERWRITTEN by `trimul_dtensor_baseline._seed_reference_layer` before the module is wrapped, and
#      a caller who forgets to seed is better served by a non-degenerate default than by a layer
#      that silently returns zeros. So `nn.Linear`/`nn.LayerNorm`'s own defaults stand.
#   2. **No `use_kernels` escape hatch.** A production layer routes to a fused external kernel
#      under a flag. This module IS the denominator such a kernel is measured against, so the flag
#      would only offer a way to measure the fused path against itself.
# ============================================================================================== #


class TriangularMultiplication(nn.Module):
    """Single-device triangular multiplicative update over a pair representation.

    Purpose
    -------
    The reference single-device TriMul layer, and the parameter container the distributed CP
    wrappers in this module are built from. It is the denominator of this repository's speedup
    claims: `TriangularMultiplication{Outgoing,Incoming}1D` and `...2D` do not call this class's
    `forward` -- they read its six submodules' `.weight`/`.bias` and re-implement the same algebra
    with the contraction sharded -- so this `forward` is what defines what those wrappers are
    supposed to compute.

    Semantics
    ---------
    With ``LN_in``/``LN_out`` the two LayerNorms and ``sigma`` the logistic sigmoid::

        xn   = LN_in(x)                          # (B, N, N, D)
        ab   = p_in(xn) * sigma(g_in(xn))        # (B, N, N, 2D)
        ab   = ab * mask[..., None]              # the mask reaches the DUAL only
        a, b = ab.chunk(2, dim=-1)               # (B, N, N, D) each
        tri  = contract(a, b)                    # subclass-specific; see `EINSUM`
        out  = p_out(LN_out(tri)) * sigma(g_out(xn))

    Three points are where an independent implementation most easily diverges, so they are stated
    rather than left to the code:

    * the mask multiplies ``ab`` **before** the chunk, so it reaches the contraction operands and
      never the output gate;
    * the output gate consumes ``xn`` -- the *normalized input* at the output ``(i, j)`` position
      -- not the contraction result;
    * the contraction is evaluated at ``promote_types(dtype, float32)``. Under bf16 the sum over
      ``k`` runs over ``N`` terms, and accumulating that in bf16 loses roughly a decimal digit at
      ``N = 2048``. Everything else runs at the module's own dtype.

    Input requirements
    ------------------
    dim : int
        Feature width ``D``. Must be ``>= 1``. It fixes every parameter shape, so a mismatch
        against the seeded weights raises from `Tensor.copy_`, not from here.
    eps : float
        Variance floor shared by BOTH LayerNorms. Must match the value the comparison reference was
        given -- `trimul_ref`'s default is ``1e-5`` and so is this one; a silent mismatch shows up
        as a small, shape-independent relative error that is easily mistaken for bf16 noise.
    device, dtype : optional
        Forwarded to the submodules, so the layer can be built directly on the target device
        instead of built on CPU and moved.

    Raises
    ------
    ValueError
        If ``dim < 1``.

    Notes
    -----
    Instantiate the direction-specific subclass, never this base: `EINSUM` is `None` here and
    `forward` raises.
    """

    #: ``torch.einsum`` subscript for the token contraction, set by each direction subclass.
    EINSUM: str | None = None
    #: Direction label carried for error messages and for `direction`-keyed callers.
    DIRECTION: str | None = None

    def __init__(self, dim: int = 128, *, eps: float = 1e-5, device=None, dtype=None) -> None:
        """Build the six parameter-holding submodules at feature width ``dim``.

        Purpose
        -------
        Allocate `norm_in`, `p_in`, `g_in`, `norm_out`, `p_out`, `g_out` with the shapes the rest
        of the repository expects, and nothing else.

        Semantics
        ---------
        The projections are ``bias=False``: this repository's fused TriMul contract carries no
        projection bias, and a bias here would be a parameter no caller seeds and no comparison
        reference applies. The LayerNorms keep their affine gain and bias. That is exactly eight
        parameters -- ``norm_in.{weight,bias}``, ``p_in.weight``, ``g_in.weight``,
        ``norm_out.{weight,bias}``, ``p_out.weight``, ``g_out.weight`` -- which is the set
        `trimul_dtensor_baseline._seed_reference_layer` writes, so seeding leaves nothing at its
        default.

        No custom initialization is applied; see this section's module comment for why.

        Input requirements
        ------------------
        dim : int
            ``>= 1``. ``p_in``/``g_in`` widen ``D -> 2D`` and ``p_out``/``g_out`` are ``D -> D``,
            so ``dim`` also fixes the chunk point of the dual.
        eps : float
            ``> 0``. Stored on `self.eps` and used by both LayerNorms.
        device, dtype
            Standard PyTorch factory kwargs, forwarded verbatim.

        Raises
        ------
        ValueError
            If ``dim < 1``.
        """
        super().__init__()
        if dim < 1:
            raise ValueError(f"dim must be >= 1; got {dim}")
        f = {"device": device, "dtype": dtype}
        self.dim = int(dim)
        self.eps = float(eps)
        self.norm_in = nn.LayerNorm(dim, eps=eps, **f)
        self.p_in = nn.Linear(dim, 2 * dim, bias=False, **f)
        self.g_in = nn.Linear(dim, 2 * dim, bias=False, **f)
        self.norm_out = nn.LayerNorm(dim, eps=eps, **f)
        self.p_out = nn.Linear(dim, dim, bias=False, **f)
        self.g_out = nn.Linear(dim, dim, bias=False, **f)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Run the triangular multiplicative update on a full, unsharded pair tensor.

        Purpose
        -------
        The single-device semantics the CP wrappers are checked against.

        Semantics
        ---------
        The seven steps listed in the class docstring, in that order. The contraction is promoted
        to ``promote_types(x.dtype, float32)`` and the result is fed to `norm_out` at that
        precision -- there is no downcast between the contraction and the second LayerNorm, which
        is what keeps a bf16 forward from losing the small-magnitude rows the normalization is
        about to rescale. The returned dtype is therefore the promoted one when ``x`` is bf16/fp16,
        matching what the distributed wrappers hand back.

        Input requirements
        ------------------
        x : Tensor
            ``(B, N, N, D)`` RAW pair representation -- NOT pre-normalized; step 1 is `norm_in`.
            ``D`` must equal ``self.dim``; a mismatch raises from the LayerNorm. Any float dtype.
        mask : Tensor or None
            ``(B, N, N)`` pair mask, broadcast over the feature axis. Zeros exclude a pair from the
            contraction operands. ``None`` means "all pairs valid" and skips the multiply entirely
            rather than building a ones tensor -- these are numerically identical, but the ones
            tensor costs an ``O(N^2)`` allocation and a full pass over ``ab``.

        Returns
        -------
        Tensor
            ``(B, N, N, D)`` at ``promote_types(x.dtype, float32)``.

        Raises
        ------
        NotImplementedError
            If called on this base class, which declares no contraction.
        """
        if self.EINSUM is None:
            raise NotImplementedError(
                "TriangularMultiplication is abstract: instantiate "
                "TriangularMultiplicationOutgoing or TriangularMultiplicationIncoming, "
                "which declare the token contraction."
            )
        xn = self.norm_in(x)
        ab = self.p_in(xn) * torch.sigmoid(self.g_in(xn))
        if mask is not None:
            ab = ab * mask.unsqueeze(-1)
        a, b = torch.chunk(ab.to(torch.promote_types(ab.dtype, torch.float32)), 2, dim=-1)
        tri = torch.einsum(self.EINSUM, a, b)
        return self.p_out(self.norm_out(tri)) * torch.sigmoid(self.g_out(xn))


class TriangularMultiplicationOutgoing(TriangularMultiplication):
    """Triangular multiplicative update contracting over the SECOND token index of both operands.

    Purpose
    -------
    The "outgoing" direction: ``out[b,i,j,d] = sum_k a[b,i,k,d] * b[b,j,k,d]``. Row ``i`` of the
    output is built from row ``i`` of ``a`` against every row of ``b``, so a 1-D row-shard of the
    pair tensor makes ``a`` local and ``b`` the operand that has to travel -- which is why the CP
    wrapper for this direction is a ring rotation of ``b``.

    Semantics
    ---------
    Identical to the base class except for `EINSUM`. Everything else -- normalization, gating,
    masking, the fp32 contraction promotion -- is inherited unchanged.

    Input requirements
    ------------------
    Exactly the base class's; see `TriangularMultiplication.__init__` and `.forward`.

    Returns / Raises
    ----------------
    As the base class.
    """

    EINSUM = "bikd,bjkd->bijd"
    DIRECTION = "outgoing"


class TriangularMultiplicationIncoming(TriangularMultiplication):
    """Triangular multiplicative update contracting over the FIRST token index of both operands.

    Purpose
    -------
    The "incoming" direction: ``out[b,i,j,d] = sum_k a[b,k,i,d] * b[b,k,j,d]``. Here the
    contraction index is the axis a 1-D row-shard splits, so every rank holds a partial sum over
    its own ``k`` slice and the CP wrapper for this direction is a reduce-scatter rather than a
    ring.

    Semantics
    ---------
    Identical to the base class except for `EINSUM`.

    Input requirements
    ------------------
    Exactly the base class's; see `TriangularMultiplication.__init__` and `.forward`.

    Returns / Raises
    ----------------
    As the base class.
    """

    EINSUM = "bkid,bkjd->bijd"
    DIRECTION = "incoming"


# ============================================================================================== #
# 1-D context parallelism: mesh `(dp, cp)`, local `(B, N/cp, N, D)` row slab.
#
# The two directions need DIFFERENT comm because they contract different indices. Outgoing
# contracts `k`, which is LOCAL to a row slab, so each rank can finish its own output rows once
# it has seen every rank's `b` -- a ring rotation of `b`, `cp-1` hops, peak `O(N^2/cp)`.
# Incoming contracts the SHARDED index, so every rank produces a partial sum over the WHOLE
# output and the comm is a reduce-scatter -- tiled, so peak stays `O(N^2/cp)` instead of the
# `O(N^2)` a single un-tiled reduce-scatter would need.
# ============================================================================================== #


class _Direction(Enum):
    """Shared by BOTH the 1-D and 2-D distributed impls below (identical upstream definitions in
    their separate source files; unified here since they are now one module -- see module
    docstring)."""

    Outgoing = auto()
    Incoming = auto()


def _ring_rotate_b_assemble_cols(
    a_local: torch.Tensor,
    b_shard: torch.Tensor,
    cp_group: dist.ProcessGroup,
    cp_size: int,
    cp_rank: int,
) -> torch.Tensor:
    """Ring matmul: rotate b, assemble column blocks in output.

    Computes ``a_local @ b_full`` where ``b_full`` is the concatenation of
    ``b_shard`` across all CP ranks along dim -1.

    Shapes::
        a_local : [B, c_h, M, K]
        b_shard : [B, c_h, K, N/cp]
        output  : [B, c_h, M, N]  (column blocks assembled by source rank)
    """
    n_local = b_shard.shape[-1]
    out = a_local.new_zeros(*a_local.shape[:-1], n_local * cp_size)
    buf = [b_shard.contiguous(), torch.empty_like(b_shard)]
    i_ready, i_recv = 0, 1

    send_to = (cp_rank - 1) % cp_size
    recv_from = (cp_rank + 1) % cp_size
    parity = cp_rank % 2 == 0

    for step in range(cp_size):
        if step < cp_size - 1:
            works = _ring_p2p_send_recv([buf[i_ready]], [buf[i_recv]], send_to, recv_from, cp_group, parity)

        source_rank = (cp_rank + step) % cp_size
        col_start = source_rank * n_local
        out[..., col_start : col_start + n_local] = torch.matmul(a_local, buf[i_ready])

        if step < cp_size - 1:
            for w in works:
                w.wait()
            i_ready ^= 1
            i_recv ^= 1

    return out


def _tiled_reduce_scatter_incoming(
    a_perm_local: torch.Tensor,
    b_perm_local: torch.Tensor,
    cp_group: dist.ProcessGroup,
    cp_size: int,
    cp_rank: int,
    num_tiles: int | None = None,
) -> torch.Tensor:
    """Incoming matmul via reduce-scatter, tiled along the j axis.

    Incoming's contraction axis ``k`` is the *sharded* (N/cp) axis, so each rank
    holds its k-chunk's contribution to *every* output ``(i, j)``.  The correct
    cross-rank reduction sums these per-k-chunk partials across the cp axis and
    scatters along the output **row axis ``i``** so each rank keeps its native
    ``N/cp`` i-slab — exactly the ``(Shard(0), Shard(1))`` row-slab the pipeline
    requires, with ``j`` full locally.  No transpose, no all-gather, no
    2D-sharded intermediate.

    This replaces the previous 2-tensor operand-rotation ring
    (``_ring_rotate_ab_accumulate_incoming``).  Both forms do identical FLOPs
    (``c_h * N^3 / cp`` per rank).  The reduce-scatter form's durable advantage
    over the ring (independent of the tile count) is **per-collective, not
    per-count**: each ``reduce_scatter`` moves ~half the bytes (it reduces one
    computed output instead of rotating two raw operands), there is **no
    send/recv double-buffer**, and each rank's partial is computed **once**
    instead of recomputed on every ring step.  See
    ``trimul-reduce-scatter-feasibility.md`` §1.

    **Tile count ``num_tiles`` (T) is a collective-count vs peak-memory knob, NOT
    a free "1 collective" win.**  The full per-rank partial ``[i=N, j=N]`` is
    O(N^2) if materialized whole.  We split the output ``j`` (full-N) axis into T
    tiles and issue one ``reduce_scatter`` per tile:

    - ``T = 1``  → a single collective, but the reduce-scatter input is the full
      ``[cp_size, N/cp, N]`` partial → **O(N^2) peak**.
    - ``T = cp_size`` (default) → ``cp_size`` collectives, each over an
      ``[cp_size, N/cp, N/cp]`` input → **O(N^2/cp) peak** (ring-parity peak).

    So at ring-parity peak the collective *count* is ~cp_size — comparable to the
    ring's cp_size steps.  The win is the per-collective structure above, not a
    reduction in collective count.  ``num_tiles`` defaults to ``None`` →
    ``cp_size``; callers may override to sweep the knee where per-collective
    savings beat tile-loop launch overhead (heavier on PCIe / small cp).

    Each tile's reduce-scatter input is a list of ``cp_size`` chunks ``[B, c_h,
    N/cp, j_tile]`` where chunk ``r`` is this rank's k-contribution to rank
    ``r``'s output i-slab; ``dist.reduce_scatter`` sums chunk ``r`` across ranks
    (the k-reduction) and lands it on rank ``r``.  On the **gloo** backend
    (which lacks reduce_scatter on some torch builds) the per-tile path falls
    back to ``all_reduce`` of the full-i partial then slices this rank's i-slab,
    mirroring ``triangular_attention_1d.py``; the per-tile transient is the same
    O(N^2 / num_tiles) on both backends.

    Note: the contraction axis must be divisible by cp — guaranteed by the
    even-sharding guard on ``N`` in the caller (``N % cp_size == 0``).

    Shapes::
        a_perm_local : [B, c_h, N, N/cp]  (i=full N, k=N/cp sharded)
        b_perm_local : [B, c_h, N/cp, N]  (k=N/cp sharded, j=full N)
        output       : [B, c_h, N/cp, N]  (i=local slab, j=full)

    Returns the reduce-scattered local i-slab in the same dtype as the matmul
    output (``a_perm_local``/``b_perm_local`` dtype, already >= fp32 in the
    forward path).
    """
    n_full = a_perm_local.shape[-2]  # i = full N
    n_local = n_full // cp_size  # N/cp

    if cp_size == 1:
        # Single rank: the local partial IS the output, no collective.
        return torch.matmul(a_perm_local, b_perm_local)

    if num_tiles is None:
        num_tiles = cp_size
    assert num_tiles >= 1, f"num_tiles must be >= 1, got {num_tiles}"
    # Tile the j (full-N) axis.  Bound num_tiles by N so each tile is non-empty;
    # use even tiling when it divides, else fall back to torch.tensor_split which
    # handles the remainder (still bounds peak to ~O(N^2 * ceil/N) per tile).
    num_tiles = min(num_tiles, n_full)

    out = a_perm_local.new_empty(*a_perm_local.shape[:2], n_local, n_full)

    # Gloo lacks reduce_scatter on some torch builds; mirror the sibling
    # fallback in triangular_attention_1d.py (all_reduce the full-i partial,
    # then slice this rank's i-slab).  The all_reduce path's per-tile transient
    # is the same O(cp_size * N/cp * j_tile) = O(N^2 / num_tiles) as the
    # reduce_scatter path (the full-i partial [B, c_h, N, j_tile] equals the
    # cp_size stacked i-slab chunks), so the peak budget is preserved on both
    # backends.
    use_gloo_fallback = dist.get_backend(cp_group) == "gloo"

    # Per-tile column ranges over the full-N j axis. int32 suffices: these are
    # token-index boundaries bounded by n_full (realistic N is O(10^4), far
    # below the int32 max ~2.1e9), and `.tolist()` converts them to Python ints
    # used only as slice bounds, so the tensor dtype never reaches a kernel.
    j_bounds = torch.linspace(0, n_full, steps=num_tiles + 1).round().to(torch.int32).tolist()
    for t in range(num_tiles):
        j0, j1 = j_bounds[t], j_bounds[t + 1]
        if j1 <= j0:
            continue
        b_tile = b_perm_local[..., j0:j1]  # [B, c_h, N/cp, j_tile]
        if use_gloo_fallback:
            # Full-i partial for this j-tile [B, c_h, N, j_tile] = this rank's
            # k-contribution to every output row; all_reduce sums across ranks
            # (the k-reduction), then slice this rank's i-slab.  .contiguous()
            # binds the buffer we reduce into and slice from to the same tensor.
            full_i = torch.matmul(a_perm_local, b_tile).contiguous()
            dist.all_reduce(full_i, op=dist.ReduceOp.SUM, group=cp_group)
            out[..., j0:j1] = full_i[..., cp_rank * n_local : (cp_rank + 1) * n_local, :]
        else:
            # Per-rank i-slab partials for this j-tile: chunk r = this rank's
            # k-contribution to rank r's output i-slab, [B, c_h, N/cp, j_tile].
            # dist.reduce_scatter (list form) sums chunk r across ranks (the
            # k-reduction) and returns rank r's chunk.  The list form is used
            # (rather than reduce_scatter_tensor) because it is backend-portable —
            # reduce_scatter_tensor enforces input.shape[0] == worldSize *
            # output.shape[0], which does not hold when the scattered cp axis is a
            # leading stack dim distinct from the batch dim.  All cp_size chunks
            # for one tile are alive at once: O(cp_size * N/cp * j_tile) =
            # O(N^2 / num_tiles) transient per tile.
            chunks = [
                torch.matmul(a_perm_local[..., r * n_local : (r + 1) * n_local, :], b_tile).contiguous()
                for r in range(cp_size)
            ]
            rs_out = torch.empty(chunks[0].shape, dtype=chunks[0].dtype, device=chunks[0].device)
            dist.reduce_scatter(rs_out, chunks, op=dist.ReduceOp.SUM, group=cp_group)
            out[..., j0:j1] = rs_out
    return out


def _ring_rotate_b_extract_cols_accumulate(
    a_local: torch.Tensor,
    b_shard: torch.Tensor,
    cp_group: dist.ProcessGroup,
    cp_size: int,
    cp_rank: int,
) -> torch.Tensor:
    """Ring matmul: rotate b, extract matching columns from a, accumulate.

    Used for outgoing backward dA where we need:
        dA[i_local, k] = sum_j dX[i_local, j] * B[k, j]
    rewritten as ``dX @ B^T``, split over sharded j-ranges of B.

    Shapes::
        a_local : [B, c_h, M, N]     (e.g. dX: M=N/cp, last dim=N full)
        b_shard : [B, c_h, K, N/cp]  (e.g. B_perm: K=N, last dim=N/cp)
        output  : [B, c_h, M, K]     (accumulated)
    """
    n_local = b_shard.shape[-1]
    out = a_local.new_zeros(*a_local.shape[:3], b_shard.shape[-2])
    buf = [b_shard.contiguous(), torch.empty_like(b_shard)]
    i_ready, i_recv = 0, 1

    send_to = (cp_rank - 1) % cp_size
    recv_from = (cp_rank + 1) % cp_size
    parity = cp_rank % 2 == 0

    for step in range(cp_size):
        if step < cp_size - 1:
            works = _ring_p2p_send_recv([buf[i_ready]], [buf[i_recv]], send_to, recv_from, cp_group, parity)

        source_rank = (cp_rank + step) % cp_size
        j_start = source_rank * n_local
        out = out + torch.matmul(
            a_local[..., j_start : j_start + n_local],
            buf[i_ready].transpose(-1, -2),
        )

        if step < cp_size - 1:
            for w in works:
                w.wait()
            i_ready ^= 1
            i_recv ^= 1

    return out


class _TriangularMultiplication1DImpl(torch.autograd.Function):
    """Distributed triangle multiplication BMM via 1D ring communication.

    Handles the gated projection, masking, and distributed matmul.
    The surrounding layer norms and output gating are computed by the
    nn.Module wrapper so that PyTorch autograd handles their gradients.

    Inputs:
        x : DTensor  [B, N, N, 2*c_hidden], placements (Shard(0), Shard(1))
            Output of p_in linear projection (pre-gating).
        mask : DTensor  [B, N, N], placements (Shard(0), Shard(1))
        g : DTensor  [B, N, N, 2*c_hidden], placements (Shard(0), Shard(1))
            Pre-sigmoid gate tensor (output of g_in).
        cp_group : dist.ProcessGroup
        direction : _Direction
        incoming_rs_tiles : int | None
            Incoming-only reduce-scatter tile count (T). ``None`` → cp_size
            (O(N^2/cp) peak). See _tiled_reduce_scatter_incoming for the
            collective-count vs peak-memory tradeoff. Ignored for outgoing.

    Output:
        DTensor [B, N, N, c_hidden], placements (Shard(0), Shard(1))

    Communication (forward):
        Outgoing: cp_size ring steps, 1 tensor per step.
        Incoming: T reduce-scatters (T = incoming_rs_tiles, default cp_size).
    Communication (backward):
        2 * cp_size ring steps for both directions.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        x: DTensor,
        mask: DTensor,
        g: DTensor,
        cp_group: dist.ProcessGroup,
        direction: _Direction,
        incoming_rs_tiles: int | None = None,
    ) -> DTensor:
        _expected = (Shard(0), Shard(1))
        for name, t in [("x", x), ("mask", mask), ("g", g)]:
            assert isinstance(t, DTensor), f"{name} must be a DTensor, got {type(t)}"
            assert t.device_mesh is x.device_mesh, f"{name} device_mesh differs from x"
            plc = tuple(t.placements)
            assert plc == _expected, f"{name} placements must be {_expected}, got {plc}"

        assert x.shape[-1] % 2 == 0, f"x last dim must be even, got {x.shape[-1]}"
        assert x.shape == g.shape, f"x and g must have same shape, got {x.shape} vs {g.shape}"
        assert mask.shape == x.shape[:3], f"mask shape {mask.shape} must match x shape[:3] {x.shape[:3]}"

        cp_size = dist.get_world_size(cp_group)
        n_full = x.shape[1]
        assert n_full % cp_size == 0, f"N ({n_full}) must be evenly divisible by cp_size ({cp_size})"
        assert (
            x.shape[1] == x.shape[2]
        ), f"Pair tensor N dims must be square, got dim1={x.shape[1]} vs dim2={x.shape[2]}"

        device_mesh = x.device_mesh
        placements = x.placements
        cp_rank = dist.get_rank(cp_group)

        # Apply sigmoid gating and mask locally.
        # Gating and masking are fused into this autograd.Function (rather than
        # handled externally by PyTorch autograd as in the OpenFold reference)
        # to avoid materializing intermediate DTensors for the gated output,
        # saving one O(N^2/cp) DTensor allocation.
        mask_local = mask.to_local().unsqueeze(-1)  # [B, N/cp, N, 1]
        # Capture the original input dtype so saved-for-backward tensors can be
        # downcast back to it (bf16 under AMP) to halve the pair-tensor memory
        # footprint at the autograd boundary. Forward arithmetic still runs in
        # safe_dtype (>= fp32) for numerical accuracy. NOTE: the 2-D path deliberately
        # does NOT do this -- it takes precision from bf16 autocast instead. That
        # asymmetry is by design; neither side is an omission.
        input_dtype = x.to_local().dtype
        safe_dtype = torch.promote_types(input_dtype, torch.float32)
        sig_g_local = g.to_local().sigmoid().to(dtype=safe_dtype)
        x_local = x.to_local().to(dtype=safe_dtype) * mask_local
        x_local = x_local * sig_g_local

        # Split into a, b projections
        c_hidden = x_local.shape[-1] // 2
        a_local = x_local[..., :c_hidden]
        b_local = x_local[..., c_hidden:]

        if direction == _Direction.Outgoing:
            # Serial einsum: "bikd,bjkd->bijd"
            # a_perm [B, c_h, N/cp, N] (i=local, k=full)
            # b_perm [B, c_h, N, N/cp] (k=full, j=local) -> transposed for matmul
            a_perm = a_local.permute(0, 3, 1, 2).contiguous()
            b_perm = b_local.permute(0, 3, 2, 1).contiguous()

            x_perm = _ring_rotate_b_assemble_cols(a_perm, b_perm, cp_group, cp_size, cp_rank)
            out_local = x_perm.permute(0, 2, 3, 1).contiguous()
        else:
            # Serial einsum: "bkid,bkjd->bijd"
            # a_perm [B, c_h, N, N/cp] (k=N/cp sharded, i=full N)
            # b_perm [B, c_h, N/cp, N] (k=N/cp sharded, j=full N)
            a_perm = a_local.permute(0, 3, 2, 1).contiguous()
            b_perm = b_local.permute(0, 3, 1, 2).contiguous()

            x_perm = _tiled_reduce_scatter_incoming(a_perm, b_perm, cp_group, cp_size, cp_rank, incoming_rs_tiles)
            out_local = x_perm.permute(0, 2, 3, 1).contiguous()

        # out_local carries safe_dtype = promote_types(input_dtype, fp32), exactly
        # mirroring the single-device layer above, whose einsum output (also
        # safe_dtype) is fed straight into norm_out with NO downcast --
        # `TriangularMultiplication.forward`, and stated in its docstring because
        # it is the easiest step to get wrong. Returning safe_dtype here is
        # therefore FAITHFUL to the single-device semantics: it preserves the fp32
        # precision kept at the einsum->norm_out handoff (and preserves fp64 paths
        # via promote_types), and is a no-op under bf16-mixed autocast in
        # production (the matmul autocasts to bf16 anyway, as does the
        # single-device einsum). A `.to(input_dtype)` downcast here would DIVERGE
        # by clamping the fp32 intermediate to bf16 before norm_out. No silent fp32
        # *upcast* occurs: there is no hardcoded `.float()`, and the fp64 path is
        # preserved via promote_types.

        if x.requires_grad:
            # Save a, b (masked+gated halves), mask, and post-sigmoid gate.
            # x_local = cat(a_local, b_local) is reconstructed in backward to
            # avoid saving redundant data at O(N^2/cp) scale.
            #
            # Memory optimisation: downcast a/b/sig_g to input_dtype (bf16 under
            # AMP) before saving — pair-tensor saves are halved (4B fp32 -> 2B
            # bf16). Backward re-promotes to fp32 at entry. The fp32->bf16->fp32
            # round-trip introduces ~eps_bf16 (~7.8e-3) relative error per saved
            # element; backward matmuls amplify this by sqrt(N_local). A bf16
            # backward-parity test must derive its tolerance from that bound
            # rather than from the forward bar.
            ctx.save_for_backward(
                a_local.to(dtype=input_dtype),
                b_local.to(dtype=input_dtype),
                mask_local,
                sig_g_local.to(dtype=input_dtype),
            )
            ctx.cp_group = cp_group
            ctx.direction = direction
            ctx.placements = placements
            ctx.device_mesh = device_mesh
            ctx.shape_x = x.shape
            ctx.stride_x = x.stride()
            ctx.shape_g = g.shape
            ctx.stride_g = g.stride()
            ctx.cp_size = cp_size
            ctx.cp_rank = cp_rank

        c_hidden_out = out_local.shape[-1]
        shape_out = x.shape[:-1] + (c_hidden_out,)
        stride_out = update_exhaustive_strides(x.shape, x.stride(), shape_out)

        return DTensor.from_local(
            out_local,
            device_mesh=device_mesh,
            placements=placements,
            shape=shape_out,
            stride=stride_out,
        )

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output: DTensor):
        a_local, b_local, mask_local, sig_g_local = ctx.saved_tensors
        cp_group = ctx.cp_group
        direction = ctx.direction
        cp_size = ctx.cp_size
        cp_rank = ctx.cp_rank

        # Re-promote saved tensors (downcast to input_dtype at save time to
        # halve the saved-for-backward memory footprint) back to safe_dtype
        # for gradient arithmetic. promote_types preserves fp64 paths.
        compute_dtype = torch.promote_types(a_local.dtype, torch.float32)
        a_local = a_local.to(dtype=compute_dtype)
        b_local = b_local.to(dtype=compute_dtype)
        sig_g_local = sig_g_local.to(dtype=compute_dtype)

        grad_out_local = grad_output.to_local().to(dtype=compute_dtype)

        if direction == _Direction.Outgoing:
            # Forward was: X = A_perm @ B_perm_assembled
            a_perm = a_local.permute(0, 3, 1, 2).contiguous()
            b_perm = b_local.permute(0, 3, 2, 1).contiguous()
            grad_perm = grad_out_local.permute(0, 3, 1, 2).contiguous()

            n_local = a_local.shape[1]

            # dA: ring-rotate B, extract matching j-cols from grad
            grad_a_perm = _ring_rotate_b_extract_cols_accumulate(grad_perm, b_perm, cp_group, cp_size, cp_rank)
            grad_a_local = grad_a_perm.permute(0, 2, 3, 1).contiguous()

            # dB: ring-rotate both A and grad
            j_start = cp_rank * n_local

            send_to = (cp_rank - 1) % cp_size
            recv_from = (cp_rank + 1) % cp_size
            parity = cp_rank % 2 == 0

            grad_b_perm = a_perm.new_zeros(*a_perm.shape[:2], a_perm.shape[3], n_local)
            buf_a = [a_perm.contiguous(), torch.empty_like(a_perm)]
            buf_g = [grad_perm.contiguous(), torch.empty_like(grad_perm)]
            i_ready, i_recv = 0, 1

            for step in range(cp_size):
                if step < cp_size - 1:
                    works = _ring_p2p_send_recv(
                        [buf_a[i_ready], buf_g[i_ready]],
                        [buf_a[i_recv], buf_g[i_recv]],
                        send_to,
                        recv_from,
                        cp_group,
                        parity,
                    )

                grad_j_chunk = buf_g[i_ready][..., j_start : j_start + n_local]
                grad_b_perm = grad_b_perm + torch.matmul(buf_a[i_ready].transpose(-1, -2), grad_j_chunk)

                if step < cp_size - 1:
                    for w in works:
                        w.wait()
                    i_ready ^= 1
                    i_recv ^= 1

            grad_b_local = grad_b_perm.permute(0, 3, 2, 1).contiguous()

        else:
            # Incoming
            a_perm = a_local.permute(0, 3, 2, 1).contiguous()
            b_perm = b_local.permute(0, 3, 1, 2).contiguous()
            grad_perm = grad_out_local.permute(0, 3, 1, 2).contiguous()

            n_local = a_local.shape[1]

            # dA: ring-rotate dX^T, assemble col blocks
            dX_t = grad_perm.transpose(-1, -2).contiguous()
            grad_a_perm_t = _ring_rotate_b_assemble_cols(
                b_perm,
                dX_t,
                cp_group,
                cp_size,
                cp_rank,
            )
            grad_a_perm = grad_a_perm_t.transpose(-1, -2).contiguous()
            grad_a_local = grad_a_perm.permute(0, 3, 2, 1).contiguous()

            # dB: ring-rotate dX
            send_to = (cp_rank - 1) % cp_size
            recv_from = (cp_rank + 1) % cp_size
            parity = cp_rank % 2 == 0

            a_perm_t = a_perm.transpose(-1, -2).contiguous()
            grad_b_perm = torch.zeros_like(b_perm)
            buf = [grad_perm.contiguous(), torch.empty_like(grad_perm)]
            i_ready, i_recv_b = 0, 1

            for step in range(cp_size):
                if step < cp_size - 1:
                    works = _ring_p2p_send_recv([buf[i_ready]], [buf[i_recv_b]], send_to, recv_from, cp_group, parity)

                source_rank = (cp_rank + step) % cp_size
                n2_start = source_rank * n_local
                grad_b_perm = grad_b_perm + torch.matmul(
                    a_perm_t[..., n2_start : n2_start + n_local],
                    buf[i_ready],
                )

                if step < cp_size - 1:
                    for w in works:
                        w.wait()
                    i_ready ^= 1
                    i_recv_b ^= 1

            grad_b_local = grad_b_perm.permute(0, 2, 3, 1).contiguous()

        # Chain rule through mask and sigmoid gating:
        # forward: x_gated = x_raw * mask * sig(g)
        # d_x_raw = d_x_gated * mask * sig(g)
        # d_g = d_x_gated * x_raw * mask * sig(g) * (1 - sig(g))
        #      = d_x_gated * x_gated * (1 - sig(g))
        # where x_gated = cat(a_local, b_local)
        grad_ab_local = torch.cat([grad_a_local, grad_b_local], dim=-1)

        # d_x_raw (through mask * sig(g))
        grad_x_local = grad_ab_local * mask_local * sig_g_local

        # d_g (through sigmoid gate); reconstruct x_gated from saved halves
        x_gated_local = torch.cat([a_local, b_local], dim=-1)
        grad_g_local = grad_ab_local * x_gated_local * (1 - sig_g_local)

        grad_x = DTensor.from_local(
            grad_x_local,
            device_mesh=ctx.device_mesh,
            placements=ctx.placements,
            shape=ctx.shape_x,
            stride=ctx.stride_x,
        )
        grad_g = DTensor.from_local(
            grad_g_local,
            device_mesh=ctx.device_mesh,
            placements=ctx.placements,
            shape=ctx.shape_g,
            stride=ctx.stride_g,
        )

        # Grads for (x, mask, g, cp_group, direction, incoming_rs_tiles).
        return grad_x, None, grad_g, None, None, None


class TriangularMultiplication1D(nn.Module):
    """Distributed triangle multiplication for 1D CP (2D mesh ``(dp, cp)``).

    Wraps a serial ``TriangularMultiplicationOutgoing`` or
    ``TriangularMultiplicationIncoming`` and replaces the local matmul
    with a 1D ring-based distributed BMM over the cp axis.

    Parameters
    ----------
    direction : _Direction
        Whether this is outgoing or incoming multiplication.
    layer : TriangularMultiplicationOutgoing | TriangularMultiplicationIncoming
        The serial module whose weights we wrap.
    device_mesh : DeviceMesh
        The 2D device mesh ``(dp, cp)``.
    cp_group : dist.ProcessGroup
        The CP process group for ring communication.
    incoming_rs_tiles : int | None
        Incoming-only reduce-scatter tile count (T). ``None`` (default) →
        cp_size, which holds peak at O(N^2/cp). See
        _tiled_reduce_scatter_incoming for the collective-count vs peak-memory
        tradeoff. Ignored for the outgoing direction (it uses the ring).
    """

    def __init__(
        self,
        direction: _Direction,
        layer: TriangularMultiplicationOutgoing | TriangularMultiplicationIncoming,
        device_mesh: DeviceMesh,
        cp_group: dist.ProcessGroup,
        incoming_rs_tiles: int | None = None,
    ) -> None:
        super().__init__()
        self.device_mesh = device_mesh
        self.cp_group = cp_group
        self._direction = direction
        self._incoming_rs_tiles = incoming_rs_tiles

        self.norm_in = LayerNormParamsReplicated(layer.norm_in, device_mesh)
        self.norm_out = LayerNormParamsReplicated(layer.norm_out, device_mesh)
        self.p_in = LinearParamsReplicated(layer.p_in, device_mesh)
        self.g_in = LinearParamsReplicated(layer.g_in, device_mesh)
        self.p_out = LinearParamsReplicated(layer.p_out, device_mesh)
        self.g_out = LinearParamsReplicated(layer.g_out, device_mesh)

    def forward(self, z: DTensor, mask: DTensor) -> DTensor:
        """Forward pass.

        Parameters
        ----------
        z : DTensor  [B, N, N, C_z], placements (Shard(0), Shard(1))
        mask : DTensor  [B, N, N], placements (Shard(0), Shard(1))
        """
        z_norm = self.norm_in(z)
        g_out = self.g_out(z_norm)

        # Projection (pre-gating)
        g = self.g_in(z_norm)
        x = self.p_in(z_norm)

        # Distributed triangle multiplication (mask and sigmoid gating applied inside)
        x = _TriangularMultiplication1DImpl.apply(x, mask, g, self.cp_group, self._direction, self._incoming_rs_tiles)

        # Output gating
        x = self.p_out(self.norm_out(x))
        x = sigmoid_gate(x, g_out)

        return x


class TriangularMultiplicationOutgoing1D(TriangularMultiplication1D):
    """Distributed outgoing triangle multiplication for 1D CP."""

    def __init__(
        self,
        layer: TriangularMultiplicationOutgoing,
        device_mesh: DeviceMesh,
        cp_group: dist.ProcessGroup,
    ) -> None:
        super().__init__(_Direction.Outgoing, layer, device_mesh, cp_group)


class TriangularMultiplicationIncoming1D(TriangularMultiplication1D):
    """Distributed incoming triangle multiplication for 1D CP."""

    def __init__(
        self,
        layer: TriangularMultiplicationIncoming,
        device_mesh: DeviceMesh,
        cp_group: dist.ProcessGroup,
    ) -> None:
        super().__init__(_Direction.Incoming, layer, device_mesh, cp_group)


# ============================================================================================== #
# 2-D context parallelism: mesh `(dp, cp0, cp1)`, local `(B, N/cp0, N/cp1, D)` block.
#
# Both token axes are sharded, so neither operand is local and the schedule has to walk the grid:
# `Ring2DComm` rotates along one axis while transposing along the other, so every rank meets each
# operand block it needs exactly once. Requires a SQUARE grid (`cp0 == cp1`) -- the transpose step
# pairs coordinate `(i, j)` with `(j, i)`, which is not a rank on a non-square grid.
#
# The `2D` suffix is load-bearing: the undecorated names belong to the SINGLE-DEVICE layer above,
# and `_Direction` is the one enum shared with the 1-D section.
# ============================================================================================== #


class _XposeArgs(Enum):
    lhs = auto()
    rhs = auto()


def _distributed_bmm(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    comm: Ring2DComm,
    permute_lhs: tuple[int, ...] | None = None,
    permute_rhs: tuple[int, ...] | None = None,
    permute_out: tuple[int, ...] | None = None,
    xpose_args: _XposeArgs | None = None,
) -> torch.Tensor:
    """Perform distributed batch matrix multiplication using ring communication.

    This function implements a memory-efficient distributed batch matrix
    multiply operation across a 2D process grid using ring communication patterns.
    It computes the matrix multiplication of two tensors while minimizing memory usage
    through double buffering and overlapping computation with communication.

    The algorithm works by:
    1. Optionally permuting input tensors to desired layouts
    2. Setting up communication buffers based on transpose requirements
    3. Using ring communication to rotate tensor chunks across processes
    4. Performing overlapped computation and communication with double buffering
    5. Accumulating partial results to compute the final distributed bmm

    Communication Patterns
    ----------------------
    The function uses Ring2DComm to implement sophisticated communication patterns
    across a 2D process grid. Below are ASCII diagrams illustrating the key phases:

    **Phase 1: Initial 2D Grid Setup**

    For a 3x3 process grid, each process (i,j) initially owns tensor chunks:
    ```
    ┌─────┬─────┬─────┐
    │(0,0)│(0,1)│(0,2)│  ← Row 0
    ├─────┼─────┼─────┤
    │(1,0)│(1,1)│(1,2)│  ← Row 1
    ├─────┼─────┼─────┤
    │(2,0)│(2,1)│(2,2)│  ← Row 2
    └─────┴─────┴─────┘
      ↑     ↑     ↑
     Col 0 Col 1 Col 2
    ```

    **Phase 2: Transpose Communication (if xpose_args specified)**

    e.g., when xpose_args=_XposeArgs.rhs, RHS tensor is transposed across the 2D grid:
    ```
    Original RHS Ownership    After Transpose Communication
    ┌─────┬─────┬─────┐      ┌─────┬─────┬─────┐
    │ R00 │ R01 │ R02 │      │ R00 │ R10 │ R20 │
    ├─────┼─────┼─────┤  →   ├─────┼─────┼─────┤
    │ R10 │ R11 │ R12 │      │ R01 │ R11 │ R21 │
    ├─────┼─────┼─────┤      ├─────┼─────┼─────┤
    │ R20 │ R21 │ R22 │      │ R02 │ R12 │ R22 │
    └─────┴─────┴─────┘      └─────┴─────┴─────┘
    ```
    When xpose_args=_XposeArgs.lhs, LHS tensor is similarly transposed across the 2D grid

    **Phase 3: Initial Ring Setup**

    Row initialization (comm_row_init): Each row i shifts left by i positions
    ```
    Before Row Init              After Row Init
    ┌─────┬─────┬─────┐         ┌─────┬─────┬─────┐
    │ L00 │ L01 │ L02 │ ←shift 0│ L00 │ L01 │ L02 │
    ├─────┼─────┼─────┤         ├─────┼─────┼─────┤
    │ L10 │ L11 │ L12 │ ←shift 1│ L11 │ L12 │ L10 │
    ├─────┼─────┼─────┤         ├─────┼─────┼─────┤
    │ L20 │ L21 │ L22 │ ←shift 2│ L22 │ L20 │ L21 │
    └─────┴─────┴─────┘         └─────┴─────┴─────┘
    ```

    Column initialization (comm_col_init): Each column j shifts up by j positions
    ```
    Before Col Init              After Col Init
    ┌─────┬─────┬─────┐         ┌─────┬─────┬─────┐
    │ R00 │ R01 │ R02 │         │ R00 │ R11 │ R22 │
    ├─────┼─────┼─────┤  shift  ├─────┼─────┼─────┤
    │ R10 │ R11 │ R12 │   ↑     │ R10 │ R21 │ R02 │
    ├─────┼─────┼─────┤   0,1,2 ├─────┼─────┼─────┤
    │ R20 │ R21 │ R22 │         │ R20 │ R01 │ R12 │
    └─────┴─────┴─────┘         └─────┴─────┴─────┘
    ```

    **Phase 4: Ring Computation Loop**

    For each iteration k in range(grid_size):
    1. Compute partial matmul: out += matmul(lhs_chunk, rhs_chunk)
    2. Ring shift both tensors for next iteration

    Ring communication pattern (each step shifts by 1):
    ```
    Step 0 → Step 1 → Step 2 (back to original)

    LHS Row Shifts (left by 1):
    ┌─────┬─────┬─────┐    ┌─────┬─────┬─────┐    ┌─────┬─────┬─────┐
    │ L00 │ L01 │ L02 │ →  │ L01 │ L02 │ L00 │ →  │ L02 │ L00 │ L01 │
    ├─────┼─────┼─────┤    ├─────┼─────┼─────┤    ├─────┼─────┼─────┤
    │ L11 │ L12 │ L10 │ →  │ L12 │ L10 │ L11 │ →  │ L10 │ L11 │ L12 │
    ├─────┼─────┼─────┤    ├─────┼─────┼─────┤    ├─────┼─────┼─────┤
    │ L22 │ L20 │ L21 │ →  │ L20 │ L21 │ L22 │ →  │ L21 │ L22 │ L20 │
    └─────┴─────┴─────┘    └─────┴─────┴─────┘    └─────┴─────┴─────┘

    RHS Column Shifts (up by 1):
    ┌─────┬─────┬─────┐    ┌─────┬─────┬─────┐    ┌─────┬─────┬─────┐
    │ R00 │ R11 │ R22 │    │ R10 │ R21 │ R02 │    │ R20 │ R01 │ R12 │
    ├─────┼─────┼─────┤    ├─────┼─────┼─────┤    ├─────┼─────┼─────┤
    │ R10 │ R21 │ R02 │ →  │ R20 │ R01 │ R12 │ →  │ R00 │ R11 │ R22 │
    ├─────┼─────┼─────┤    ├─────┼─────┼─────┤    ├─────┼─────┼─────┤
    │ R20 │ R01 │ R12 │    │ R00 │ R11 │ R22 │    │ R10 │ R21 │ R02 │
    └─────┴─────┴─────┘    └─────┴─────┴─────┘    └─────┴─────┴─────┘
    ```

    **Double Buffering Strategy**

    The algorithm uses double buffering to overlap communication with computation:
    ```
    Time →  │ Compute │ Compute │ Compute │
            │ Buffer0 │ Buffer1 │ Buffer0 │
            │    ↓    │    ↓    │    ↓    │
    Comm →  │   Send  │   Send  │   Send  │
            │  Buffer1│ Buffer0 │ Buffer1 │
            │   Recv  │   Recv  │   Recv  │
            │ Buffer1 │ Buffer0 │ Buffer1 │
    ```

    This ensures that while one buffer is being used for computation, the other
    buffer is being prepared through communication for the next iteration.


    Parameters
    ----------
    lhs : torch.Tensor
        Left-hand side tensor for matrix multiplication.
        Typically has shape (B, ...) where B is batch dimension.
    rhs : torch.Tensor
        Right-hand side tensor for matrix multiplication.
        Must be compatible with lhs for matrix multiplication after permutations.
    comm : Ring2DComm
        Ring communication object configured for 2D process grid communication.
        Provides row and column communication groups for distributed computation.
    permute_lhs : tuple[int, ...] | None, optional
        Permutation indices to apply to lhs tensor before computation. Typically
        the permutation with group the batch-like axes into leading axes and reshape
        the last two axes into "N" and "K" dimensions (in the NMK notation)
        If None, no permutation is applied. Default is None.
    permute_rhs : tuple[int, ...] | None, optional
        Permutation indices to apply to rhs tensor before computation. Typically
        the permutation with group the batch-like axes into leading axes and reshape
        the last two axes into "K" and "M" dimensions (in the NMK notation)
        If None, no permutation is applied. Default is None.
    permute_out : tuple[int, ...] | None, optional
        Permutation indices to apply to output tensor after computation. Typically
        the permutation reverts the resulting permutation of the output matrix
        due to the permutation of the input lhs' and rhs' axes.
        If None, no permutation is applied. Default is None.
    xpose_args : _XposeArgs | None, optional
        Specifies which tensor requires transpose communication:
        - _XposeArgs.lhs: Transpose communication for left-hand side tensor
        - _XposeArgs.rhs: Transpose communication for right-hand side tensor
        - None: No transpose communication required
        Default is None.

    Returns
    -------
    torch.Tensor
        Result of the distributed batch matrix multiplication.
        Shape depends on input shapes and permutation arguments.

    Examples
    --------
    Typical usage in triangle multiplication:

    >>> # For outgoing triangle multiplication
    >>> result = _distributed_bmm(
    ...     lhs=tensor_a,
    ...     rhs=tensor_b,
    ...     comm=ring_comm,
    ...     permute_lhs=(0, 3, 1, 2),  # (B, n, k, D) -> (B, D, n, k)
    ...     permute_rhs=(0, 3, 2, 1),  # (B, m, k, D) -> (B, D, k, m)
    ...     permute_out=(0, 2, 3, 1),  # (B, D, n, m) -> (B, n, m, D)
    ...     xpose_args=_XposeArgs.rhs
    ... )
    """
    if permute_lhs is not None:
        lhs = lhs.permute(permute_lhs)
    # this enforces lhs and rhs to be a clone so that the in-place modification
    # does not affect the input tensor
    lhs = lhs.clone(memory_format=torch.contiguous_format)
    if permute_rhs is not None:
        rhs = rhs.permute(permute_rhs)
    rhs = rhs.clone(memory_format=torch.contiguous_format)

    if xpose_args == _XposeArgs.lhs:
        lhs_recv = comm.comm_2d_trans.enqueue_to_dispatch(lhs)
        rhs_recv = rhs
        rhs = torch.empty_like(rhs_recv)
    elif xpose_args == _XposeArgs.rhs:
        rhs_recv = comm.comm_2d_trans.enqueue_to_dispatch(rhs)
        lhs_recv = lhs
        lhs = torch.empty_like(lhs_recv)
    elif xpose_args is None:
        lhs_recv = lhs
        lhs = torch.empty_like(lhs_recv)
        rhs_recv = rhs
        rhs = torch.empty_like(rhs_recv)
    else:
        raise ValueError(f"Invalid xpose_args: {xpose_args}")

    # post the comm_2d_trans.wait_until_finished() (or no wait if xpose_args is not None),
    # *_recv are the correct tensors to operate on
    i_ready = 0
    i_recv = i_ready ^ 1
    lhs_buffer = [lhs_recv, lhs]
    rhs_buffer = [rhs_recv, rhs]

    if xpose_args is not None:
        comm.comm_2d_trans.wait_until_finished()

    lhs_buffer[i_recv] = comm.comm_row_init.enqueue_to_dispatch(lhs_buffer[i_ready], lhs_buffer[i_recv])
    rhs_buffer[i_recv] = comm.comm_col_init.enqueue_to_dispatch(rhs_buffer[i_ready], rhs_buffer[i_recv])

    i_ready ^= 1
    i_recv ^= 1

    out = torch.zeros_like(lhs_buffer[i_ready])

    comm.comm_row_init.wait_until_finished()
    comm.comm_col_init.wait_until_finished()

    # Double buffering computation
    for k_step in range(comm.group_layout.shape[1]):
        lhs_ready = lhs_buffer[i_ready]
        rhs_ready = rhs_buffer[i_ready]
        if k_step < comm.group_layout.shape[1] - 1:
            lhs_buffer[i_recv] = comm.comm_row.enqueue_to_dispatch(lhs_ready, lhs_buffer[i_recv])
            rhs_buffer[i_recv] = comm.comm_col.enqueue_to_dispatch(rhs_ready, rhs_buffer[i_recv])
        out = out + torch.matmul(lhs_ready, rhs_ready)
        if k_step < comm.group_layout.shape[1] - 1:
            comm.comm_row.wait_until_finished()
            comm.comm_col.wait_until_finished()
            i_ready = i_ready ^ 1
            i_recv = i_recv ^ 1

    if permute_out is not None:
        out = out.permute(permute_out)
    return out


class _TriangularMultiplicationImpl(torch.autograd.Function):
    """Distributed implementation of triangle multiplication using ring communication.

    This autograd function implements a memory-efficient distributed triangle multiplication
    operation across a 2D process grid. The computation is parallelized using ring
    communication patterns to reduce memory usage and communication overhead.

    The triangle multiplication computes:

    for Outgoing:
        o = torch.einsum("bnkd,bmkd->bnmd", a * mask, b * mask)

    for Incoming:
        o = torch.einsum("bknd,bkmd->bnmd", a * mask, b * mask)

    Key features:
    - Distributed across a 2D grid with sharding on token dimensions (dim 1 and 2)
    - Uses ring communication to rotate data chunks during computation
    - Memory-efficient implementation that avoids materializing full tensors
    - Supports gradient computation through custom backward pass

    Notes
    -----
    Input tensors must be DTensors with:
    - Shape: (B, N_token1, N_token2, c_hidden) for tensors a and b
    - Shape: (B, N_token1, N_token2, 1) for mask tensor
    - Sharding on dimensions 1 and 2 (Shard(1) and Shard(2) placements)
    - Identical device mesh and placements across all inputs

    The algorithm uses a ring-based communication pattern where:
    - Tensor b is transposed and rotated by row
    - Tensor a is rotated by column
    - Each process computes partial matrix products and accumulates results
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, x: DTensor, mask: DTensor, g: DTensor, comm: Ring2DComm, direction: _Direction) -> DTensor:
        """Forward pass of distributed triangle multiplication computation.

        Parameters
        ----------
        ctx : torch.autograd.function.BackwardCFrame
            Context object for saving information needed in backward pass.
        x : DTensor
            Input tensor with shape (B, N_token1, N_token2, c_hidden * 2).
            Must be sharded on dimensions 1 and 2.
        mask : DTensor
            Mask tensor with shape (B, N_token1, N_token2) indicating valid positions.
            Must be sharded on dimensions 1 and 2.
        g : DTensor
            pre-sigmoid gate tensor with shape (B, N_token1, N_token2, c_hidden * 2) indicating valid positions.
            Must be sharded on dimensions 1 and 2.
        comm : Ring2DComm
            Ring communication object configured for the distributed computation.
        direction : _Direction
            Direction of the triangle multiplication, Outgoing or Incoming.

        Returns
        -------
        DTensor
            Output tensor with shape (B, N_token1, N_token2, c_hidden).
            Contains the distributed triangle multiplication result.
        """
        # Check if inputs are of type DTensor
        if not isinstance(x, DTensor):
            raise TypeError(f"Input 'x' must be of type DTensor. Got type {type(x)}.")
        if not isinstance(mask, DTensor):
            raise TypeError(f"Input 'mask' must be of type DTensor. Got type {type(mask)}.")
        if not isinstance(g, DTensor):
            raise TypeError(f"Input 'g' must be of type DTensor. Got type {type(g)}.")

        # Check if inputs have identical device mesh
        device_mesh_input = x.device_mesh
        if device_mesh_input != mask.device_mesh:
            raise ValueError(
                f"Input tensors 'x' and 'mask' must have identical device mesh. "
                f"Got device meshes {device_mesh_input} and {mask.device_mesh}."
            )
        if device_mesh_input != g.device_mesh:
            raise ValueError(
                f"Input tensors 'x' and 'g' must have identical device mesh. "
                f"Got device meshes {device_mesh_input} and {g.device_mesh}."
            )

        # Check if inputs have identical placements
        placements_input = x.placements
        if placements_input != mask.placements:
            raise ValueError(
                f"Input tensors 'x' and 'mask' must have identical placements. "
                f"Got placements {placements_input} and {mask.placements}."
            )
        if placements_input != g.placements:
            raise ValueError(
                f"Input tensors 'x' and 'g' must have identical placements. "
                f"Got placements {placements_input} and {g.placements}."
            )
        if placements_input != (Shard(0), Shard(1), Shard(2)):
            # For debugging, we requires the placements to be (Shard(0), Shard(1), Shard(2))
            # TODO: remove this to only use the previous check
            raise ValueError(
                f"Input tensor 'x's placements are not (Shard(0), Shard(1), Shard(2)). "
                f"Got placements {placements_input}."
            )

        # Check input shapes
        if x.shape[-1] % 2 != 0:
            raise ValueError(f"Input tensor 'x' must have an even number of hidden dimension size. Got {x.shape[-1]}")

        if x.ndim != 4:
            raise ValueError(f"Input tensor 'x' must have 4 dimensions. Got {x.ndim} dimensions.")

        if mask.ndim != 3:
            raise ValueError(f"Input tensor 'mask' must have 3 dimensions. Got {mask.ndim} dimensions.")

        if mask.shape != x.shape[:3]:
            raise ValueError(
                f"Input tensor 'mask' must have the same shape as the first 3 dimensions of 'x'. "
                f"Got mask shape: {mask.shape} vs x shape[:3]: {x.shape[:3]}"
            )
        if g.shape != x.shape:
            raise ValueError(
                f"Input tensor 'g' must have the same shape as 'x'. Got g shape: {g.shape} vs x shape: {x.shape}"
            )

        # Perform consistency check between the ring_comm and the device_mesh_input
        i_tensor_dim_to_i_grid_axis = [-1] * x.ndim
        for i_grid_axis, placement in enumerate(placements_input):
            if isinstance(placement, Shard):
                i_tensor_dim_to_i_grid_axis[placement.dim] = i_grid_axis
        if i_tensor_dim_to_i_grid_axis[1] == -1 or i_tensor_dim_to_i_grid_axis[2] == -1:
            raise ValueError(f"Input tensors' dimensions 1 and 2 must be sharded. Got placements {placements_input}.")

        # Check ring_comm consistency
        if comm.group_col != device_mesh_input.get_group(i_tensor_dim_to_i_grid_axis[1]):
            raise ValueError(
                "Input ring_comm's group_col process group is not the same as the group sharding the input tensors' axis 1"
            )

        coord_device_mesh_input = device_mesh_input.get_coordinate()
        if coord_device_mesh_input is None:
            raise ValueError(f"ring_comm.coord_2d {comm.coord_2d} is not on device_mesh_input {device_mesh_input}.")
        if comm.coord_2d != (
            coord_device_mesh_input[i_tensor_dim_to_i_grid_axis[1]],
            coord_device_mesh_input[i_tensor_dim_to_i_grid_axis[2]],
        ):
            raise ValueError(
                f"Input ring_comm's coord_2d {comm.coord_2d} does not match the "
                f"device mesh's rank coordinates {coord_device_mesh_input} for the sharded dimensions."
            )

        ctx.mark_non_differentiable(mask)

        # Apply mask and prepare for computation.
        #
        # Cast mask to ``x``'s dtype before any arithmetic.  ``mask`` is a
        # boolean-equivalent gating tensor whose precision is meaningless, yet
        # the production data pipeline materialises it as FP32 (the pair pad
        # mask is cast to the pair representation's dtype, and that dtype is
        # itself ``promote_types(s_init.dtype, FP32)`` in the trunk, so the
        # mask arrives FP32 even in a bf16-mixed run).  Under ``bf16-mixed``
        # autocast ``x``/``g`` are
        # BF16 (Linear outputs), so the out-of-place ``BF16 * FP32`` below
        # would type-promote ``x_local`` to FP32 and cascade through the
        # autograd function:
        #
        # * forward output becomes FP32 (the ``_distributed_bmm`` accumulator
        #   is seeded by ``zeros_like(lhs)`` so the FP32 ``a_local`` poisons
        #   the result even though ``matmul`` itself autocasts to BF16);
        # * the saved ``a_local``/``b_local``/``x_masked_gated_local`` are FP32
        #   so ``dg_local = dab_local * x_masked_gated_local`` in backward
        #   computes a FP32 gradient at ``g`` — diverging from the serial
        #   reference whose final trimul output is BF16 under autocast and
        #   whose input ``g`` is BF16.
        #
        # ``custom_fwd(device_type="cuda")`` with the default ``cast_inputs=None``
        # *inherits* the calling autocast state rather than disabling it, but
        # autocast handles only registered ops (matmul/einsum/linear); the
        # element-wise ``*`` here falls back to torch type promotion and must
        # be handled explicitly.
        #
        # NOTE (bf16 autocast): the absence of an explicit fp32 upcast below is
        # DELIBERATE -- unlike the 1-D path, this one takes its matmul precision
        # from the caller's bf16 autocast. Benchmark and nsys paths supply it.
        mask_local = mask.to_local().to(dtype=x.dtype).unsqueeze(-1)
        g_local = g.to_local().sigmoid()
        x_local = x.to_local() * mask_local
        x_local *= g_local

        # the _distributed_bmm will permute a_local and b_local and make
        # the resulting tensors contiguous so we don't need to clone them here
        a_local, b_local = torch.chunk(x_local, 2, dim=-1)

        # Store tensors for backward pass
        if x.requires_grad:
            # here x_local is masked and gated
            ctx.save_for_backward(a_local, b_local, mask_local, x_local, g_local)
            ctx.comm = comm
            ctx.shape_x_input = x.shape
            ctx.stride_x_input = x.stride()
            ctx.shape_g_input = g.shape
            ctx.stride_g_input = g.stride()
            ctx.placements_input = placements_input
            ctx.device_mesh_input = device_mesh_input
            ctx.direction = direction

        if direction == _Direction.Outgoing:
            permute_lhs = (0, 3, 1, 2)  # from (B, n, k, D) to (B, D, n, k)
            permute_rhs = (0, 3, 2, 1)  # from (B, m, k, D) to (B, D, k, m)
            permute_out = (0, 2, 3, 1)  # from (B, D, n, m) to (B, n, m, D)
            xpose_args = _XposeArgs.rhs
        elif direction == _Direction.Incoming:
            permute_lhs = (0, 3, 2, 1)  # from (B, k, n, D) to (B, D, n, k)
            permute_rhs = (0, 3, 1, 2)  # from (B, k, m, D) to (B, D, k, m)
            permute_out = (0, 2, 3, 1)  # from (B, D, n, m) to (B, n, m, D)
            xpose_args = _XposeArgs.lhs
        else:
            raise ValueError(f"Invalid direction: {direction}")

        out_local = _distributed_bmm(
            a_local,
            b_local,
            comm,
            permute_lhs=permute_lhs,
            permute_rhs=permute_rhs,
            permute_out=permute_out,
            xpose_args=xpose_args,
        ).contiguous()

        shape_output = x.shape[:-1] + (out_local.shape[-1],)
        stride_output = update_exhaustive_strides(x.shape, x.stride(), shape_output)
        # Convert back to DTensor
        out = DTensor.from_local(
            out_local,
            device_mesh=device_mesh_input,
            placements=placements_input,
            shape=shape_output,
            stride=stride_output,
        )
        return out

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, d_loss_d_out: DTensor) -> Tuple[DTensor, None, DTensor, None, None, None]:
        """Backward pass of distributed triangle multiplication computation."""
        if not isinstance(d_loss_d_out, DTensor):
            raise TypeError(f"Input 'd_loss_d_out' must be of type DTensor. Got type {type(d_loss_d_out)}.")

        if d_loss_d_out.device_mesh != ctx.device_mesh_input:
            raise ValueError(
                f"Input 'd_loss_d_out' must have the same device mesh as the input tensors. "
                f"Got device meshes {d_loss_d_out.device_mesh} and {ctx.device_mesh_input}."
            )

        if d_loss_d_out.placements != ctx.placements_input:
            raise ValueError(
                f"Input 'd_loss_d_out' must have the same placements as the input tensors. "
                f"Got placements {d_loss_d_out.placements} and {ctx.placements_input}."
            )

        a, b, mask_local, x_masked_gated_local, g_local = ctx.saved_tensors
        comm = ctx.comm
        direction = ctx.direction

        # cast d_loss_d_out to the same dtype as a (saved tensor) to avoid type promotion to FP32
        # Note: torch.amp.custom_bwd disables autocast, so operations run in the input dtype.
        # If the upstream adjoint (d_loss_d_out) arrives as FP32 (e.g. from loss scaling or downstream FP32 layers),
        # mixed-precision ops with saved BF16 tensors would promote to FP32, causing potential communication
        # buffer mismatches and NCCL hangs. Explicit casting ensures consistent precision.
        d_loss_d_out_local = d_loss_d_out.to_local().to(dtype=a.dtype)

        if direction == _Direction.Outgoing:
            lhs_da = d_loss_d_out_local
            rhs_da = b
            permute_lhs_da = (0, 3, 1, 2)  # from (B, n, m, D) to (B, D, n, m)
            permute_rhs_da = (0, 3, 1, 2)  # from (B, m, k, D) to (B, D, m, k)
            permute_out_da = (0, 2, 3, 1)  # from (B, D, n, k) to (B, n, k, D)
            xpose_args_da = None

            lhs_db = d_loss_d_out_local
            rhs_db = a
            permute_lhs_db = (0, 3, 2, 1)  # from (B, n, m, D) to (B, D, m, n)
            permute_rhs_db = (0, 3, 1, 2)  # from (B, n, k, D) to (B, D, n, k)
            permute_out_db = (0, 2, 3, 1)  # from (B, D, m, k) to (B, m, k, D)
            xpose_args_db = _XposeArgs.lhs

        elif direction == _Direction.Incoming:
            lhs_da = b
            rhs_da = d_loss_d_out_local
            permute_lhs_da = (0, 3, 1, 2)  # from (B, k, m, D) to (B, D, k, m)
            permute_rhs_da = (0, 3, 2, 1)  # from (B, n, m, D) to (B, D, m, n)
            permute_out_da = (0, 2, 3, 1)  # from (B, D, k, n) to (B, k, n, D)
            xpose_args_da = _XposeArgs.rhs

            lhs_db = a
            rhs_db = d_loss_d_out_local
            permute_lhs_db = (0, 3, 1, 2)  # from (B, k, n, D) to (B, D, k, n)
            permute_rhs_db = (0, 3, 1, 2)  # from (B, n, m, D) to (B, D, n, m)
            permute_out_db = (0, 2, 3, 1)  # from (B, D, k, m) to (B, k, m, D)
            xpose_args_db = None
        else:
            raise ValueError(f"Invalid direction: {direction}")

        d_loss_d_a_local = _distributed_bmm(
            lhs_da,
            rhs_da,
            comm,
            permute_lhs=permute_lhs_da,
            permute_rhs=permute_rhs_da,
            permute_out=permute_out_da,
            xpose_args=xpose_args_da,
        ).contiguous()

        # Phase 2: d_loss_d_b
        d_loss_d_b_local = _distributed_bmm(
            lhs_db,
            rhs_db,
            comm,
            permute_lhs=permute_lhs_db,
            permute_rhs=permute_rhs_db,
            permute_out=permute_out_db,
            xpose_args=xpose_args_db,
        ).contiguous()

        # concatenate and apply mask to gradients
        dab_local = torch.cat([d_loss_d_a_local, d_loss_d_b_local], dim=-1)

        x_masked_gated_local *= 1 - g_local
        dg_local = dab_local * x_masked_gated_local

        dg = DTensor.from_local(
            dg_local,
            device_mesh=ctx.device_mesh_input,
            placements=ctx.placements_input,
            shape=ctx.shape_g_input,
            stride=ctx.stride_g_input,
        )

        dx_local = dab_local
        dx_local *= mask_local
        dx_local *= g_local

        # Convert gradients back to DTensors
        dx = DTensor.from_local(
            dx_local,
            device_mesh=ctx.device_mesh_input,
            placements=ctx.placements_input,
            shape=ctx.shape_x_input,
            stride=ctx.stride_x_input,
        )

        return dx, None, dg, None, None


class TriangularMultiplication2D(nn.Module):
    """Distributed triangle multiplication layer.

    This layer implements a distributed version of the triangle multiplication operation,
    which is used in attention mechanisms for protein structure prediction and other applications
    requiring pairwise feature interactions.

    The layer performs the following operations:
    1. Layer normalization of input pairwise features
    2. Linear projections to create two representation streams (a and b)
    3. Distributed triangle multiplication computation using ring communication
    4. Output gating and final linear projection

    Parameters
    ----------
    layer : TriangularMultiplicationOutgoing | TriangularMultiplicationIncoming
        The serial triangle multiplication layer to convert to distributed version.
        Used to initialize projection weights and normalization parameters.
    device_mesh : DeviceMesh
        The device mesh for distributed computation across multiple GPUs.
    comm : Ring2DComm
        Ring communication object for efficient distributed triangle multiplication computation.
    """

    def __init__(
        self,
        direction: _Direction,
        layer: TriangularMultiplicationOutgoing | TriangularMultiplicationIncoming,
        device_mesh: DeviceMesh,
        comm: Ring2DComm,
    ) -> None:
        """Initialize the distributed triangle multiplication layer."""
        super().__init__()
        self.device_mesh = device_mesh
        self.ring_comm = comm

        self.norm_in = LayerNormParamsReplicated(layer.norm_in, self.device_mesh)
        self.p_in = LinearParamsReplicated(layer.p_in, self.device_mesh)
        self.g_in = LinearParamsReplicated(layer.g_in, self.device_mesh)

        self.norm_out = LayerNormParamsReplicated(layer.norm_out, self.device_mesh)
        self.p_out = LinearParamsReplicated(layer.p_out, self.device_mesh)
        self.g_out = LinearParamsReplicated(layer.g_out, self.device_mesh)

        if direction == _Direction.Outgoing:
            if not isinstance(layer, TriangularMultiplicationOutgoing):
                raise ValueError(f"Invalid layer type for direction {direction}: {type(layer)}")
        elif direction == _Direction.Incoming:
            if not isinstance(layer, TriangularMultiplicationIncoming):
                raise ValueError(f"Invalid layer type for direction {direction}: {type(layer)}")
        else:
            raise ValueError(f"Invalid direction {direction}")
        self._direction = direction

    def forward(self, x: DTensor, mask: DTensor) -> DTensor:
        """Forward pass of the distributed triangle multiplication layer.

        Parameters
        ----------
        x : DTensor
            Input pairwise tensor with shape (B, N, N, D).
            Must be sharded on dimensions 1 and 2.
        mask : DTensor
            Mask tensor with shape (B, N, N) indicating valid positions.
            Must be sharded on dimensions 1 and 2.

        Returns
        -------
        DTensor
            Output pairwise tensor with shape (B, N, N, D).
        """
        # Stabilize pair embedding tensor with layer norm
        x = self.norm_in(x)
        x_in = x
        g_out = self.g_out(x_in)

        # Decompress: D -> 2D
        g = self.g_in(x)
        x = self.p_in(x)

        # Distributed triangular multiplication (mask is applied inside the implementation)
        x = _TriangularMultiplicationImpl.apply(x, mask, g, self.ring_comm, self._direction)

        # Output gating
        x = self.p_out(self.norm_out(x))
        x = sigmoid_gate(x, g_out)

        return x


class TriangularMultiplicationOutgoing2D(TriangularMultiplication2D):
    """Distributed triangle multiplication outgoing layer."""

    def __init__(
        self,
        layer: TriangularMultiplicationOutgoing,
        device_mesh: DeviceMesh,
        comm: Ring2DComm,
    ) -> None:
        """Initialize the distributed triangle multiplication outgoing layer.

        Parameters
        ----------
        layer : TriangularMultiplicationOutgoing
            The serial triangle multiplication outgoing layer to convert to distributed version.
        device_mesh : DeviceMesh
            The device mesh for distributed computation across multiple GPUs.
        comm : Ring2DComm
            Ring communication object for efficient distributed triangle multiplication computation.
        """
        super().__init__(_Direction.Outgoing, layer, device_mesh, comm)


class TriangularMultiplicationIncoming2D(TriangularMultiplication2D):
    """Distributed triangle multiplication incoming layer."""

    def __init__(
        self,
        layer: TriangularMultiplicationIncoming,
        device_mesh: DeviceMesh,
        comm: Ring2DComm,
    ) -> None:
        """Initialize the distributed triangle multiplication incoming layer.

        Parameters
        ----------
        layer : TriangularMultiplicationIncoming
            The serial triangle multiplication incoming layer to convert to distributed version.
        device_mesh : DeviceMesh
            The device mesh for distributed computation across multiple GPUs.
        comm : Ring2DComm
            Ring communication object for efficient distributed triangle multiplication computation.
        """
        super().__init__(_Direction.Incoming, layer, device_mesh, comm)

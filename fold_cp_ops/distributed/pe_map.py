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

"""Generic (DeviceMesh, placements) -> NVSHMEM PE-map addressing layer (T1.1).

The all-to-all reshards that localize the TriMul einsum
(``(S0,S1,S2) -> (S0,S3,S3)`` and back, design doc §2) are each an **all-to-all
within the context-parallel (cp) group**: on-device, the producer/consumer warp
iterates a *flat communication-peer index* ``r in [0, cp)`` and must translate it
to the **global NVSHMEM PE** to put-to / get-from. This module produces that
``flat_cp_peer -> global_PE`` map from a torch ``DeviceMesh`` + a DTensor
``placements`` spec, with **no hardcoded PE ids** and supporting a **flattened
1-D cp = cp0*cp1*...** (so ``cp in 2..8`` works on 1-D *or* N-D cp grids, §2 end).

Scope / non-goals:
  * This is the *addressing* layer only — it computes PEs. It is **primitive-
    agnostic**: it does not care whether T1.2 uses the upstream's ``put_signal_nbi`` or
    NVSHMEM 3.7's native TMA put/get (independent of the T0.6 verdict).
  * It is **complementary to** (not a replacement for) the per-tile *data*
    coordinate layout (the upstream's ``cp_mesh_layout`` / ``map_symmem_trunk_a2a``),
    which decides *where in the dst buffer* the data lands. PE translation and
    data placement are separate concerns; this module owns only the former.

Invariant: NVSHMEM is initialized over the **world** group
(``DistributedManager.init_nvshmem``, which VERIFIES it at step 5), so
``nvshmem.core.my_pe() == global rank`` and ``DeviceMesh.mesh`` (the rank tensor)
*is* the PE tensor. The cp-group PE list is therefore a slice of that rank tensor
along the cp axes — computable purely host-side (no GPU) for unit testing.

This is the ONLY PE-addressing layer. There used to be a second thing called a
"PE map" on the manager (``nvshmem_pe_map``: process-group name -> that group's
member ranks as a tensor) and a cross-check between them. They were never two
answers to one question — this is an ORDERED table indexed by a flat peer
``r``, that was an unordered roster keyed by group name — so the only comparison
that even typechecked between them was set-equality, which discards the ordering
that is this table's entire content. Nothing consumed the roster, so it was
removed rather than renamed; the shared name is what made it look like a
redundant copy of this module.

On-device representation (what T1.2 consumes):
  * ``cp_pe_table`` — an ``int32`` CUDA tensor of shape ``(cp,)`` mapping the flat
    cp-peer index to the global PE. Pass to a ``@cute.kernel`` via
    ``cute.runtime.from_dlpack(pe_map.cp_pe_table)``; inside the kernel
    ``cp_pe_table[r]`` is a device read returning the ``Int32`` dst PE for peer
    ``r`` (exactly the upstream's ``ring_pe_map[idx]`` pattern, generalized).
  * ``cp_layout_shape_stride`` — the plain-Python ``((cp,), (1,))`` shape/stride
    of that table; the kernel builds ``cute.make_layout(*...)`` from it inside
    its trace context (``cute.make_layout`` cannot run in bare host Python).
  * scalars ``cp`` (peer count) and ``my_cp_rank`` (this rank's own flat-cp index
    -> ``r == my_cp_rank`` is the self put/get).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch

from fold_cp_ops.distributed.layout_map import LayoutRightMap

# Placement objects are duck-typed (``.is_shard()`` / ``.is_replicate()`` /
# ``.dim``) so this module does not hard-import torch.distributed.tensor at module
# load (keeps it importable in minimal envs); the DeviceMesh is duck-typed too.


def _is_shard(placement) -> bool:
    """Whether ``placement`` is a DTensor ``Shard`` (duck-typed)."""
    is_shard = getattr(placement, "is_shard", None)
    if callable(is_shard):
        return bool(is_shard())
    # Fallback: a Shard has an int ``.dim``; Replicate/Partial do not.
    return isinstance(getattr(placement, "dim", None), int)


def _placement_shard_dim(placement) -> Optional[int]:
    """Return the sharded tensor dim for a ``Shard`` placement, else ``None``."""
    return getattr(placement, "dim", None) if _is_shard(placement) else None


def _mesh_rank_tensor(device_mesh) -> np.ndarray:
    """Return the mesh's rank tensor as a numpy ndarray of shape == mesh shape.

    ``DeviceMesh.mesh`` is the (n-D) tensor of global ranks laid out over the
    mesh dims. Under the nvshmem-over-world invariant, rank == PE.
    """
    mesh = device_mesh.mesh
    if isinstance(mesh, torch.Tensor):
        return mesh.detach().cpu().numpy()
    return np.asarray(mesh)


@dataclass
class PeMap:
    """``flat_cp_peer -> global_PE`` map for the cp all-to-all (see module docstring).

    Construct via :meth:`from_mesh_placements`. Attributes:

    Attributes
    ----------
    cp : int
        Number of communication peers (the flattened cp size = product of the
        sharded mesh-axis sizes). The on-device peer loop runs ``r in [0, cp)``.
    my_cp_rank : int
        This rank's own flat-cp index in ``[0, cp)`` (``r == my_cp_rank`` is the
        self put/get).
    cp_pe_table : torch.Tensor
        ``int32`` tensor of shape ``(cp,)`` on ``device``; ``cp_pe_table[r]`` is
        the global PE for flat peer ``r``. Device-resident when ``device`` is
        CUDA (ready for ``from_dlpack``); host (CPU) in the offline unit-test path.
    cp_axes : tuple[int, ...]
        The mesh-dim indices that are sharded (the cp axes), in mesh-dim order.
    cp_axis_sizes : tuple[int, ...]
        Sizes of ``cp_axes`` (``prod == cp``); the flattening is row-major over
        these (``LayoutRightMap``), matching the design's flattened ``cp0*cp1``.
    cp_shard_tensor_dims : tuple[int, ...]
        The DTensor tensor-dim each cp axis shards (e.g. ``(1, 2)`` for tokens),
        in ``cp_axes`` order — recorded so T1.2/T1.3 can map peer index to the
        token sub-block it owns.
    device : torch.device
        Device of ``cp_pe_table``.
    """

    cp: int
    my_cp_rank: int
    cp_pe_table: torch.Tensor
    cp_axes: tuple[int, ...]
    cp_axis_sizes: tuple[int, ...]
    cp_shard_tensor_dims: tuple[int, ...]
    device: torch.device

    @property
    def cp_layout_shape_stride(self) -> tuple[tuple[int], tuple[int]]:
        """``(shape, stride)`` of the trivial 1-D PE-table layout ``(cp,):(1,)``.

        Plain Python (no DSL context needed on the host). A ``@cute.kernel`` /
        ``@cute.jit`` builds the actual layout *inside its trace context* via
        ``cute.make_layout(*pe_map.cp_layout_shape_stride)`` — ``cute.make_layout``
        cannot run in bare host Python (it needs an MLIR Context). In practice
        T1.2 just uses ``cute.runtime.from_dlpack(pe_map.cp_pe_table)`` directly,
        which already carries this ``(cp,)`` layout.
        """
        return ((self.cp,), (1,))

    def __repr__(self) -> str:
        return (
            f"PeMap(cp={self.cp}, my_cp_rank={self.my_cp_rank}, "
            f"cp_axes={self.cp_axes}, cp_axis_sizes={self.cp_axis_sizes}, "
            f"cp_shard_tensor_dims={self.cp_shard_tensor_dims}, "
            f"cp_pe_table={self.cp_pe_table.tolist()}, device={self.device})"
        )

    @staticmethod
    def from_mesh_placements(
        device_mesh,
        placements: Sequence,
        *,
        my_rank: Optional[int] = None,
        device: Optional[torch.device] = None,
        distributed_manager=None,
    ) -> "PeMap":
        """Build a :class:`PeMap` from a ``DeviceMesh`` + DTensor ``placements``.

        Placement-generic: the cp (communication) axes are exactly the mesh dims
        whose placement ``is_shard()``; ``Replicate``/dp axes carry no comm. The
        cp axes are flattened row-major (``LayoutRightMap``) into the 1-D peer
        index, so a 1-D cp grid (``cp=(4,)``) and an N-D grid (``cp=(2,2)``)
        produce the same flat ``[0, cp)`` addressing.

        The cp-group PE list for *this* rank is the slice of ``DeviceMesh.mesh``
        (the rank==PE tensor) obtained by fixing the non-cp (dp) coords at this
        rank's position and walking the cp axes in flattened order.

        Parameters
        ----------
        device_mesh : torch.distributed.device_mesh.DeviceMesh
            The mesh (e.g. dims ``(dp, cp0, cp1)``). ``.mesh`` is the rank tensor.
        placements : Sequence
            One placement per mesh dim (``Shard(tensor_dim)`` or ``Replicate()``).
            Must match ``device_mesh.ndim``.
        my_rank : int, optional
            This process's global rank. If ``None``, taken from
            ``distributed_manager.rank`` if given, else
            ``torch.distributed.get_rank()`` if initialized, else 0 (single-rank
            / offline). The mesh coordinate is then located by searching the rank
            tensor (works offline without a live mesh ``get_coordinate``).
        device : torch.device, optional
            Device for ``cp_pe_table``. Defaults to
            ``distributed_manager.device`` if given, else CUDA-current if
            available, else CPU.
        distributed_manager : DistributedManager, optional
            A live manager, used ONLY as a source of defaults for ``my_rank`` and
            ``device``. It is never consulted about membership: the mesh rank
            tensor is authoritative, and the manager's ``DeviceMesh`` is where
            that tensor came from, so there is no second opinion to be had.
            Never required (the mesh-tensor computation is self-contained).

        Returns
        -------
        PeMap

        Raises
        ------
        ValueError
            If ``len(placements) != device_mesh.ndim``, if no axis is sharded, if
            ``my_rank`` does not appear exactly once in the mesh rank tensor, or
            if this rank is absent from its own cp group.
        """
        rank_tensor = _mesh_rank_tensor(device_mesh)
        ndim = rank_tensor.ndim
        if len(placements) != ndim:
            raise ValueError(
                f"placements length {len(placements)} != device_mesh ndim {ndim} "
                f"(mesh shape {rank_tensor.shape})"
            )

        # 1) Identify cp (sharded) axes vs non-cp (replicate/dp) axes — generic.
        cp_axes = tuple(i for i, p in enumerate(placements) if _is_shard(p))
        if len(cp_axes) == 0:
            raise ValueError(
                f"No sharded mesh axis found in placements {placements}; the cp "
                "all-to-all needs at least one Shard axis (the token sharding)."
            )
        cp_axis_sizes = tuple(int(rank_tensor.shape[a]) for a in cp_axes)
        cp_shard_tensor_dims = tuple(int(_placement_shard_dim(placements[a])) for a in cp_axes)
        cp = int(np.prod(cp_axis_sizes))
        non_cp_axes = tuple(i for i in range(ndim) if i not in cp_axes)

        # 2) Resolve this rank + its mesh coordinate (offline-safe: search the
        #    rank tensor rather than relying on a live mesh.get_coordinate()).
        if my_rank is None:
            if distributed_manager is not None:
                my_rank = int(distributed_manager.rank)
            elif torch.distributed.is_available() and torch.distributed.is_initialized():
                my_rank = int(torch.distributed.get_rank())
            else:
                my_rank = 0
        coord_hits = np.argwhere(rank_tensor == my_rank)
        if coord_hits.shape[0] != 1:
            raise ValueError(
                f"rank {my_rank} appears {coord_hits.shape[0]} times in the mesh rank "
                f"tensor (expected exactly 1); mesh shape {rank_tensor.shape}"
            )
        my_coord = tuple(int(c) for c in coord_hits[0])

        # 3) Flatten the cp axes row-major (LayoutRightMap) -> the 1-D peer index.
        #    cp_flat_layout maps a flattened index r in [0, cp) to a cp-axis
        #    multi-coordinate, which we splice into my_coord (non-cp coords fixed)
        #    to index the rank tensor -> the global PE for peer r.
        cp_flat_layout = LayoutRightMap(cp_axis_sizes)
        cp_pes = np.empty((cp,), dtype=np.int64)
        my_cp_rank = -1
        for r in range(cp):
            cp_coord = cp_flat_layout.unravel(r)  # tuple over cp axes
            full_coord = list(my_coord)
            for axis, c in zip(cp_axes, cp_coord):
                full_coord[axis] = c
            pe = int(rank_tensor[tuple(full_coord)])
            cp_pes[r] = pe
            if pe == my_rank:
                my_cp_rank = r
        if my_cp_rank < 0:
            raise ValueError(f"this rank {my_rank} not found in its own cp group {cp_pes.tolist()}")

        # 5) Materialize the device-resident int32 table.
        if device is None:
            if (
                distributed_manager is not None
                and getattr(distributed_manager, "device", None) is not None
            ):
                device = distributed_manager.device
            elif torch.cuda.is_available():
                device = torch.device("cuda", torch.cuda.current_device())
            else:
                device = torch.device("cpu")
        device = torch.device(device)
        cp_pe_table = torch.tensor(cp_pes, dtype=torch.int32, device=device)

        return PeMap(
            cp=cp,
            my_cp_rank=my_cp_rank,
            cp_pe_table=cp_pe_table,
            cp_axes=cp_axes,
            cp_axis_sizes=cp_axis_sizes,
            cp_shard_tensor_dims=cp_shard_tensor_dims,
            device=device,
        )

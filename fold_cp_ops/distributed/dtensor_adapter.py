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

"""DTensor adapter for ``trimul_autotuned`` (T1.3).

Thin shim so the local single-GPU :func:`fold_cp_ops.trimul_autotune.trimul_autotuned`
op accepts a **DTensor** input: unwrap → run local compute → re-wrap. The
inter-rank communication (the front/back all-to-all reshards that localize the
TriMul einsum) is a **clearly-marked seam** — :func:`_reshard_trimul` — that
T1.2 (the standalone NVSHMEM all-to-all primitive on the upstream CP comm + PeMap)
plugs into. This module owns ONLY the DTensor plumbing + the seam contract; it
builds no comm itself.

Sharding contract (design doc §2, mirrors the upstream CP project's
``distributed/model/layers/triangular_mult.py``):

* ``x`` is a DTensor ``(B, N, N, D)`` with placements ``(Shard(0), Shard(1),
  Shard(2))`` on a ``(dp, cp0, cp1)`` mesh — both token axes sharded over the
  2-D ``(cp0, cp1)`` cp grid; feature ``D`` is **not** sharded (so LN +
  DualGatedGEMM are local). Validation is **placement-generic** (duck-typed
  ``.is_shard()`` like :class:`fold_cp_ops.distributed.PeMap`), not a hard
  ``== (Shard(0),Shard(1),Shard(2))`` compare, so a flattened-cp 1-D mesh
  (``cp=(world,)``) or an N-D cp grid both pass.
* The output is a DTensor with the **same placements** as ``x``.

Why a seam and not the comm here
--------------------------------
``trimul_autotuned`` is a monolithic local op (front GLU+LN → GEMM1 einsum →
back DualGatedGEMM). Distributing it requires resharding the GEMM1 operands so
the einsum (which contracts over the full token axis ``K=N``, sharded across cp)
becomes local: front A2A ``(S0,S1,S2)->(S0,S3,S3)`` → local einsum → back A2A
``(S0,S3,S3)->(S0,S1,S2)`` (design doc §2). That reshard is the Wave-1/2 work
(T1.2 builds the primitive; T2.x fuses it into the kernel epilogues). Here it is
the single function :func:`_reshard_trimul`, whose **default implementation uses
torch DTensor ``redistribute``** (the same all-to-all the T2.0 baseline uses) so
this adapter is correct + torchrun-testable **today**; T1.2 swaps the seam body
for the in-kernel NVSHMEM A2A on the upstream primitive + :class:`PeMap`, keeping
the signature byte-identical.
"""

from __future__ import annotations

from typing import Sequence


from fold_cp_ops.distributed.pe_map import _is_shard, _placement_shard_dim


# --------------------------------------------------------------------------- #
# DTensor duck-typing helpers (no hard torch.distributed.tensor import at load).
# --------------------------------------------------------------------------- #


def _is_dtensor(x) -> bool:
    """True if ``x`` is a ``torch.distributed.tensor.DTensor`` (duck-typed)."""
    # DTensor exposes ``.to_local()`` + ``._spec`` (mesh + placements). Avoid a
    # hard import so this module loads in minimal (non-distributed) envs.
    return hasattr(x, "to_local") and hasattr(x, "_spec") and hasattr(x, "device_mesh")


def _dtensor_mesh_placements(x):
    """Return ``(device_mesh, placements)`` for a DTensor ``x``.

    Prefers the public ``.device_mesh`` / ``.placements``; falls back to
    ``._spec.mesh`` / ``._spec.placements`` (the task's named accessors).
    """
    mesh = getattr(x, "device_mesh", None)
    placements = getattr(x, "placements", None)
    spec = getattr(x, "_spec", None)
    if mesh is None and spec is not None:
        mesh = spec.mesh
    if placements is None and spec is not None:
        placements = spec.placements
    return mesh, tuple(placements)


def _rowmaj_stride(shape: Sequence[int]) -> tuple[int, ...]:
    """Row-major (contiguous) stride for ``shape`` — for ``DTensor.from_local``."""
    s = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        s[i] = s[i + 1] * shape[i + 1]
    return tuple(s)


# --------------------------------------------------------------------------- #
# Sharding validation (placement-generic, §2 TriMul pattern).
# --------------------------------------------------------------------------- #


def validate_trimul_sharding(placements: Sequence, ndim_mesh: int) -> tuple[int, ...]:
    """Validate ``placements`` is the expected TriMul token-sharding; return cp axes.

    Placement-generic: the TriMul pattern is "both token axes (tensor dims 1 and
    2) are sharded over the cp mesh axes; the batch (dim 0) and feature (dim 3)
    are NOT sharded" (design doc §2). We accept any mesh where the sharded axes
    cover tensor dims ⊆ {0, 1, 2} and at least one token dim (1 or 2) is sharded,
    and NO axis shards the feature dim 3 (D must stay local for LN/DualGatedGEMM).
    A ``Shard(0)`` on the batch axis is allowed (matches the upstream CP
    ``(Shard(0),Shard(1),Shard(2))`` contract) but carries no einsum comm.

    Parameters
    ----------
    placements : Sequence
        One placement per mesh dim (duck-typed Shard/Replicate).
    ndim_mesh : int
        Number of mesh dims (== ``len(placements)`` expected).

    Returns
    -------
    tuple[int, ...]
        The mesh-dim indices that shard a **token** axis (tensor dim 1 or 2) —
        the cp (communication) axes that the reshard must all-to-all over.

    Raises
    ------
    ValueError
        If ``len(placements) != ndim_mesh``, the feature dim (3) is sharded, or
        no token axis (1 or 2) is sharded.
    """
    if len(placements) != ndim_mesh:
        raise ValueError(f"placements length {len(placements)} != mesh ndim {ndim_mesh}")
    sharded_tensor_dims = []
    cp_mesh_axes = []
    for mesh_axis, p in enumerate(placements):
        if _is_shard(p):
            tdim = _placement_shard_dim(p)
            sharded_tensor_dims.append(tdim)
            if tdim in (1, 2):
                cp_mesh_axes.append(mesh_axis)
    if 3 in sharded_tensor_dims:
        raise ValueError(
            "TriMul requires the feature dim (3) to be UN-sharded (LN + "
            f"DualGatedGEMM reduce over full D), but placements {placements} "
            "shard tensor dim 3."
        )
    if not cp_mesh_axes:
        raise ValueError(
            "TriMul requires at least one token axis (tensor dim 1 or 2) to be "
            f"sharded over the cp mesh, but placements {placements} shard none. "
            "Expected the (Shard(0),Shard(1),Shard(2)) pattern (design doc §2)."
        )
    return tuple(cp_mesh_axes)


# The GEMM1 einsum couples BOTH token axes: outgoing tri[b,i,j,d] = Σ_k a[b,i,k,d]·
# b[b,j,k,d] (incoming: Σ_k a[b,k,i,d]·b[b,k,j,d]). Output element (i,j) needs a's
# full row/col and b's full row/col over the contracted axis k AND the other token
# index ranges over the WHOLE grid — so if EITHER token axis (tensor dim 1 or 2) is
# really split across cp, the einsum is NOT local: the operands must be resharded
# (front/back A2A, the T1.2 seam). The monolithic `trimul_autotuned` can be
# dispatched directly ONLY when NEITHER token axis is cp-split (the local shard is
# the full square (N,N) grid) — which is also why it asserts N == N2.
_TOKEN_TENSOR_DIMS = (1, 2)


def _token_split_factor(device_mesh, placements: Sequence) -> int:
    """Product of cp mesh-axis sizes that split EITHER token axis (1 or 2).

    ``1`` means no real token sharding (both token axes whole on every rank) →
    the einsum is local → dispatch the real kernel. ``> 1`` means at least one
    token axis is split across cp → the reshard seam is required.
    """
    mesh_shape = tuple(device_mesh.mesh.shape)
    factor = 1
    for mesh_axis, p in enumerate(placements):
        if _is_shard(p) and _placement_shard_dim(p) in _TOKEN_TENSOR_DIMS:
            factor *= int(mesh_shape[mesh_axis])
    return factor


def _effective_placements(device_mesh, placements: Sequence) -> list:
    """Replace each Shard on a SIZE-1 mesh axis with Replicate (it splits nothing).

    The cp comm axes are only the mesh dims that REALLY split a tensor dim (axis
    size > 1). A ``Shard`` on a singleton mesh axis (e.g. dp=1 or cp1=1 in the
    canonical ``(dp=1,cp0=cp,cp1=1)`` mesh) is a no-op; leaving it in confuses
    the PeMap / ReshardLayout (which would treat tensor dim 0 as a cp axis).
    Returns a new placements list with those normalized to ``Replicate``; real
    (size>1) Shards are kept as-is.
    """
    from torch.distributed.tensor import Replicate

    mesh_shape = tuple(device_mesh.mesh.shape)
    out = []
    for mesh_axis, p in enumerate(placements):
        if _is_shard(p) and int(mesh_shape[mesh_axis]) == 1:
            out.append(Replicate())
        else:
            out.append(p)
    return out


# --------------------------------------------------------------------------- #
# THE COMM SEAM — the REAL A2A path (T2.0c RealKernelTriMul) plugs in here.
# --------------------------------------------------------------------------- #


def _ensure_nvshmem(distributed_manager=None) -> None:
    """Ensure NVSHMEM is initialized for the cp all-to-all (idempotent, collective).

    The real A2A path's ``CpAllToAll`` allocates symmetric (NVSHMEM-backed)
    buffers, so NVSHMEM must be initialized first. ``DistributedManager.init_nvshmem``
    is idempotent (no-op if already up) and sets ``NVSHMEM_DISABLE_NVLS=1`` /
    ``NVSHMEM_DISABLE_NCCL=1`` / ``NVSHMEM_SYMMETRIC_SIZE`` via ``setdefault``. It
    is a collective — every rank must call it together; SPMD lockstep dispatch +
    the per-test barrier guarantee that. Uses the passed manager's class if given,
    else imports the fold_cp_ops one.
    """
    DM = type(distributed_manager) if distributed_manager is not None else None
    if DM is None or not hasattr(DM, "init_nvshmem"):
        from fold_cp_ops.distributed.distributed_manager import DistributedManager as DM
    DM.init_nvshmem()


# Cache RealKernelTriMul instances: its ctor compiles the CpAllToAll kernels +
# allocates symmetric buffers ONCE, so it must be reused across calls of the
# same (B, N, D, cp, dtype, pe_map) — never reconstructed per dispatch.
_REAL_KERNEL_CACHE: dict = {}


def _is_rows_sharded_canonical(device_mesh, placements: Sequence) -> bool:
    """True for the §2 canonical layout RealKernelTriMul / ReshardLayout support.

    RealKernelTriMul's ReshardLayout packs the local shard ``(B, N_loc, N, D)``
    assuming token **rows** (tensor dim 1) are the cp-split axis and the other
    token axis (dim 2) is whole. That is the design's canonical mesh
    ``(dp=1, cp0=cp, cp1=1)``. We use the real A2A path iff the ONLY token axis
    cp-split is dim 1; any other token sharding (dim 2, or a 2-D cp grid) falls
    back to the all-gather reference (those are correctness-only meshes).
    """
    mesh_shape = tuple(device_mesh.mesh.shape)
    dim1_factor = 1
    dim2_factor = 1
    for mesh_axis, p in enumerate(placements):
        if _is_shard(p):
            d = _placement_shard_dim(p)
            if d == 1:
                dim1_factor *= int(mesh_shape[mesh_axis])
            elif d == 2:
                dim2_factor *= int(mesh_shape[mesh_axis])
    return dim1_factor > 1 and dim2_factor == 1


# NOTE: the five functions `_import_real_kernel_trimul`, `_get_real_kernel_trimul`,
# `_reshard_trimul`, `_reshard_trimul_allgather_fallback` and `trimul_autotuned_dtensor` were
# dropped here (plan §7 Q3).  They drove the SUPERSEDED unfused T2.0c glue path
# (`real_kernel_trimul.py` + `a2a.py`), which this package does not ship.
#
# `trimul_autotuned_dtensor` was a PUBLIC entry point: it is removed BY DESIGN, not merely
# absent, and `FusedTriMulCP` / `fused_trimul_dtensor` replace it.  Callers of the old name get
# an AttributeError naming it, which is the intended signal.

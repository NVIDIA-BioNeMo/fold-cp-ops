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

"""Host-side cp-per-peer S2G TMA atom construction for the back-A2A GEMM-epilogue
peer store (T2.2 ASK-2).

The fused back-A2A GEMM (``GemmA2ASm90``) stores its GEMM1 result token-shard
S3 -> token-shard S1 by writing each CTA's output tile directly to a PEER's
symmetric recv buffer via a TMA S2G store (the upstream ``put_signal_nbi_tma_peer``
decomposed: a peer-pinned S2G atom + ``cute.copy(s2g_atom, sD_subtile,
peer_gmem_subtile)`` + ``signal_op``). The descriptor inside an S2G TMA atom
bakes the GMEM tensor's base pointer at atom-build time, so a SEPARATE atom is
needed per destination peer (its descriptor pinned to that peer's symmetric heap
base). This module builds those ``cp`` peer atoms host-side; the kernel threads
them as args (region-local) and a ``const_expr``-unrolled per-CTA select picks
atom ``r`` for the CTA whose M-block targets peer ``r``.

DIVISION OF LABOUR (T2.2): this module owns the HOST-SIDE atom/tensor build
(TMA-descriptor expertise — the two silent-corruption gotchas below). The kernel
owns the per-peer select + the ``ReshardLayout.back`` ``tile_m``/``tile_n``/peer
addressing + the ``flat_divide`` + ``tma_partition`` + ``cute.copy`` (region-local
so the partition does not trip "value defined outside the region").

THE TWO DESCRIPTOR GOTCHAS (why this build is correct — see
``cute-dsl:debug`` "TMA Bulk-Tensor Descriptor Mismaps"):

1. **SMEM-box vs peer-GMEM-box stride-1 leading-dim AGREEMENT.** The
   ``cuTensorMap`` follows column-major convention (dim 0 fastest); the TMA
   hardware requires the SMEM box and the GMEM box to agree on which dim is
   stride-1. A mismatch is *silently* wrong (shuffled bytes, NO error). We avoid
   it by construction: we reuse the BASE GEMM's ``epi_smem_layout_staged`` for
   the SMEM box (the exact layout the epilogue already writes ``sD`` with) and
   build the GMEM box from the peer view of ``mRecv`` — which is
   ``get_peer_tensor(mRecv)`` = ``cute.make_tensor(peer_ptr, mRecv.layout)``,
   preserving ``mRecv``'s layout verbatim (only the base ptr changes). So the
   per-peer descriptor is byte-identical to the proven LOCAL D-store descriptor
   (``GemmSm90._make_tma_epi_atoms_and_tensors``) except the base address ->
   leading dims agree for free. (Do NOT hand-roll a fresh ``(tm,tn)`` SMEM box;
   that is what desyncs the leading dim.)
2. **SMEM tile alignment (Align[..., 128] floor).** The SMEM region backing the
   TMA box must be >=128 B aligned (1024 B for the SW128 swizzle). We inherit
   this too: ``sD`` is allocated ``cute.struct.Align[..., buffer_align_bytes]``
   with ``buffer_align_bytes == 1024`` (the base GEMM's value), and reusing
   ``epi_smem_layout_staged`` carries the base's swizzle — the TMA descriptor's
   swizzle mode must match the SMEM tile's swizzle, which is why reusing the base
   layout (vs a fresh one) is mandatory, not just convenient.

T0.5: ``mRecv`` must be a CONCRETE-stride symmetric tensor (from_dlpack static,
no ``mark_layout_dynamic``) — the bitcode-route GEMM compiles per-shape anyway.
T0.7 / T2.2 ASK-1: this op set LINKS clean in the bitcode route (``cute.copy``
on a TMA atom is pure-cutlass ``cp.async.bulk.tensor`` PTX; ``get_peer_tensor``
is ``nvshmem_ptr`` pointer math; only ``signal_op`` is an nvshmem device symbol,
already proven by the shipped ``a2a.py``). It does NOT pull the header-only
``give_smem``/``ask_smem`` (the native-TMA-put wall).
"""

from __future__ import annotations

from typing import List, Tuple

import cutlass.cute as cute
from cutlass import Int32
from cutlass.cute.nvgpu import cpasync

try:
    # alignment-PRESERVING get_peer_tensor (VENDORED into fold_cp_ops from the upstream CP project —
    # fold_cp_ops/distributed/nvshmem_utils.py): the upstream
    # nvshmem.core.device.cute.mem.get_peer_tensor strips assumed_align (falls
    # back to dtype-sized align) -> breaks the 128-bit-atom IR check for the TMA
    # bulk copy. The vendored variant inherits the input tensor's alignment.
    from fold_cp_ops.distributed.nvshmem_utils import get_peer_tensor as _get_peer_tensor_aligned

    HAS_PEER = True
except ImportError:  # pragma: no cover - non-nvshmem host
    HAS_PEER = False


def build_peer_store_atoms(
    mRecv: cute.Tensor,
    cp: int,
    pe_table: Tuple[int, ...],
    epi_smem_layout_staged: cute.ComposedLayout,
    epi_tile: Tuple[int, int],
) -> Tuple[List[cute.CopyAtom], List[cute.Tensor]]:
    """Build the ``cp`` peer-pinned S2G TMA atoms + their (flat-tiled) peer tensors.

    Call HOST-SIDE in the GEMM op's ``@cute.jit __call__`` (the returned lists are
    compile-time-static Python lists of real ``cute`` objects; thread them into
    the ``@cute.kernel`` as args so the kernel's ``tma_partition`` is region-local).

    Parameters
    ----------
    mRecv : cute.Tensor
        This rank's LOCAL symmetric recv buffer (the GEMM output ``mD``), shape
        ``(M, N[, L])`` — the destination of the back-A2A. The SAME tensor passed
        to the base local D-store atom (so the peer atoms are byte-identical
        except the base ptr). Must be CONCRETE-stride (from_dlpack static; T0.5).
        Pass an ``Int16``-recast view if the kernel puts as int16 (bf16 has no
        integer TMA path) — recast on the caller side so the box dtype matches the
        SMEM tile dtype.
    cp : int
        Flat cp peer count (number of atoms to build).
    pe_table : tuple of int
        ``cp``-length flat-peer -> global-PE map (``PeMap.cp_pe_table.tolist()``).
        ``pe_table[r]`` is the global PE whose symmetric heap atom ``r``'s
        descriptor is pinned to.
    epi_smem_layout_staged : cute.ComposedLayout
        The BASE GEMM's ``self.epi_smem_layout_staged`` — REUSED verbatim for the
        SMEM box so the swizzle + leading-dim + Align inherit correct (gotchas
        1 & 3). Do not synthesize a fresh SMEM layout.
    epi_tile : tuple[int, int]
        The epilogue tile ``(epi_m, epi_n)`` — the TMA box shape, same as the base
        D-store.

    Returns
    -------
    (atoms, peer_tensors) : (list[cute.CopyAtom], list[cute.Tensor])
        ``atoms[r]`` is the S2G TMA copy atom whose descriptor base is pinned to
        peer ``pe_table[r]``. ``peer_tensors[r]`` is that peer's ``mRecv`` view as
        returned by ``make_tiled_tma_atom`` (the kernel ``flat_divide``s it by
        ``epi_tile`` then ``tma_partition``s — see module docstring). Both lists
        are length ``cp``.

    Notes
    -----
    The kernel store (the caller's region-local code) is, per peer ``r``::

        gD = cute.flat_divide(peer_tensors[r], epi_tile)   # (epi_m,epi_n,nt_m,nt_n[,L])
        s_r, g_r = cute.nvgpu.cpasync.tma_partition(
            atoms[r], 0, cute.make_layout(1),
            cute.group_modes(sD, 0, cute.rank(sD) - 1),    # SMEM box -> rank-1
            cute.group_modes(gD, 0, 2),                    # fold the (epi_m,epi_n) box
        )
        cute.copy(atoms[r], s_r[(None, src_idx)],
                  g_r[(None, tile_m, tile_n[, 0])])         # box=None + grid coords

    Use ``cute.flat_divide`` (NOT ``zipped_divide`` — it nests the box and
    ``tma_partition`` rejects "same size in first rank"); ``group_modes(gD,0,2)``
    folds exactly the ``(epi_m,epi_n)`` box leaving the grid+L modes; the
    ``cute.copy`` GMEM coord is ``box=None`` + one coord per surviving grid mode
    (for a 3D ``(M,N,L=1)`` GEMM output that is ``(None, tile_m, tile_n, 0)``).
    """
    if not HAS_PEER:
        raise RuntimeError(
            "build_peer_store_atoms requires nvshmem4py + the vendored nvshmem utils "
            "(the upstream CP project's nvshmem ``get_peer_tensor``)."
        )
    if len(pe_table) != cp:
        raise ValueError(f"pe_table {pe_table} length must equal cp={cp}.")

    # SMEM box + GMEM box layouts — IDENTICAL to the base local D-store atom
    # (GemmSm90._make_tma_epi_atoms_and_tensors), so the descriptor is byte-for-byte
    # the local one except the peer base ptr (gotchas 1 & 3 inherited).
    epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
    d_cta_v_layout = cute.composition(cute.make_identity_layout(mRecv.shape), epi_tile)
    op = cpasync.CopyBulkTensorTileS2GOp()

    atoms: List[cute.CopyAtom] = []
    peer_tensors: List[cute.Tensor] = []
    for r in range(cp):
        # get_peer_tensor = make_tensor(peer_ptr, mRecv.layout): swaps ONLY the
        # base ptr, preserves the layout -> peer box leading-dim == local box.
        peer_view = _get_peer_tensor_aligned(mRecv, Int32(pe_table[r]))
        atom_r, tensor_r = cpasync.make_tiled_tma_atom(
            op, peer_view, epi_smem_layout, d_cta_v_layout
        )
        atoms.append(atom_r)
        peer_tensors.append(tensor_r)
    return atoms, peer_tensors

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

"""Reshard DATA-layout for the TriMul cp all-to-all (T2.0c).

The companion to :class:`fold_cp_ops.distributed.PeMap`: PeMap says **which PE** a flat
cp-peer maps to; this module says **which tile goes to which peer and where it
lands in the recv buffer** — the per-tile data mapping for the
``(S0,S1,S2) <-> (S0,S3,S3)`` reshards (design doc §2). It is the reference baseline's
``cp_mesh_layout`` / ``map_symmem_trunk_a2a`` *idea*, ported restrictively (the
shape/packing math only — no forked code, no TMA-descriptor coupling).

It drives :class:`fold_cp_ops.distributed.CpAllToAll`, whose contract is::

    recv[s]  <-  peer s's send[my_cp_rank]        # send/recv: (cp*rows_per_peer, cols)

So the packing must arrange, for the FRONT A2A ``(token-shard S1) -> (feature-
shard S3)``:

* local operand ``a``: ``(B, N_loc, N, D)`` — token axis (tensor dim ``tdim``)
  split ``N_loc = N/cp`` across cp; feature ``D`` **full**.
* ``send[r]`` = my whole local-token block restricted to **D-slice r**
  (``a[..., r*Dloc:(r+1)*Dloc]``), flattened to ``rows_per_peer = B*N_loc*N`` rows
  of ``cols = Dloc = D//cp``. → peer ``r`` keeps my tokens for its D-slice.
* after the exchange, ``recv[s]`` = peer ``s``'s local-token block for **my**
  D-slice. Concatenating ``recv[0..cp)`` along the split token axis reforms the
  **full** token grid for my D-slice: ``(B, N, N, Dloc)`` → the einsum is local.

The BACK A2A ``(S3 feature) -> (S1 token)`` is the exact inverse on ``tri``
``(B, N, N, Dloc)`` → ``(B, N_loc, N, D)``.

Generic over the cp-split token tensor-dim (from ``pe_map.cp_shard_tensor_dims``)
and over the flattened cp size; validated at cp=2 (token dim 1). All packing is
plain torch reshape/permute on the LOCAL shard (no comm here) — the comm is
:class:`CpAllToAll`.
"""

from __future__ import annotations

import torch

from fold_cp_ops.distributed.pe_map import PeMap


class ReshardLayout:
    """Per-tile data mapping for the front/back TriMul cp reshards.

    Parameters
    ----------
    pe_map : PeMap
        The cp addressing (gives ``cp``, ``my_cp_rank``, and the cp-split token
        tensor-dim via ``cp_shard_tensor_dims``).
    B, N, D : int
        Global batch, (square) token extent, and feature width. ``D % cp == 0``
        and ``N % cp == 0`` are required (the feature/token splits must be even).
    feat_width : int, optional
        Feature width of the operand being resharded (``D`` for ``a``/``b`` and
        for ``tri`` in TriMul; the chunk GLU already split ``2D -> D``). Defaults
        to ``D``.

    Notes
    -----
    The cp-split token axis is taken as the FIRST entry of
    ``pe_map.cp_shard_tensor_dims`` (TriMul shards token dims 1/2; for cp=2 on a
    1-D cp grid that is dim 1). ``N_loc = N // cp`` along that axis.
    """

    def __init__(self, pe_map: PeMap, B: int, N: int, D: int, feat_width: int = None):
        self.pe_map = pe_map
        self.cp = pe_map.cp
        self.my_cp_rank = pe_map.my_cp_rank
        self.B = B
        self.N = N
        self.D = D
        self.feat = feat_width if feat_width is not None else D
        if self.D % self.cp != 0:
            raise ValueError(f"D={D} must be divisible by cp={self.cp} (feature split).")
        if self.feat % self.cp != 0:
            raise ValueError(f"feat_width={self.feat} must be divisible by cp={self.cp}.")
        # NATIVE 1-D vs 2-D token sharding: cp axes = the sharded mesh dims (cp_axis_sizes,
        # mesh-dim order). 1-D: cp_axis_sizes==(cp,) -> token dim 1 alone split (N_i_loc=N/cp),
        # dim 2 (j) FULL. 2-D: cp_axis_sizes==(cp0,cp1) -> dim 1 split into cp0 (N_i_loc=N/cp0)
        # AND dim 2 split into cp1 (N_j_loc=N/cp1). The peer flatten is row-major over the cp
        # axes (slot = i_block*cp1 + j_block), matching PeMap / the back store's unravel.
        cp_axis_sizes = tuple(int(s) for s in pe_map.cp_axis_sizes)
        if len(cp_axis_sizes) > 2:
            raise ValueError(
                f"TriMul reshard supports 1-D/2-D token sharding; got {cp_axis_sizes}."
            )
        self.cp0 = cp_axis_sizes[0]
        self.cp1 = cp_axis_sizes[1] if len(cp_axis_sizes) > 1 else 1
        self.n_cp_axes = len(cp_axis_sizes)
        if self.N % self.cp0 != 0 or self.N % self.cp1 != 0:
            raise ValueError(f"N={N} must be divisible by both cp axes {cp_axis_sizes}.")
        self.N_i_loc = self.N // self.cp0  # token dim 1 (i) local extent
        self.N_j_loc = self.N // self.cp1  # token dim 2 (j) local extent (== N when 1-D)
        # cp-split token tensor-dim (TriMul: 1 or 2). Default to the first sharded
        # token dim; for cp=2 1-D grid this is dim 1.
        cp_dims = pe_map.cp_shard_tensor_dims
        self.token_dim = cp_dims[0] if cp_dims else 1
        if self.token_dim not in (1, 2):
            raise ValueError(f"cp-split token dim {self.token_dim} must be 1 or 2 for TriMul.")
        self.N_loc = self.N_i_loc  # back-compat alias (the i-axis peer-block extent)
        self.feat_loc = self.feat // self.cp
        # CpAllToAll send/recv geometry. rows_per_peer = my local 2-D token block.
        self.rows_per_peer = self.B * self.N_i_loc * self.N_j_loc  # tokens in my local block
        self.cols = self.feat_loc  # D-slice width

    # ---- FRONT: token-shard (B,N_loc,N,feat) -> CpAllToAll send -------------
    def front_pack(self, a_local: torch.Tensor, send: torch.Tensor) -> None:
        """Pack local ``(B, N_loc, N, feat)`` into ``send`` ``(cp*rows_per_peer, cols)``.

        ``send[r*rows_per_peer:(r+1)*rows_per_peer]`` = my local-token block's
        D-slice ``r``, flattened over ``(B, N_loc, N)`` row-major.
        """
        self._check_local(a_local)
        # (B, N_loc, N, feat) -> (B*N_loc*N, feat) row-major over tokens.
        flat = a_local.reshape(self.rows_per_peer, self.feat)
        # split feat into cp D-slices -> (rows, cp, feat_loc) -> per-peer rows.
        # send[r] gets column-slice r. Lay out as (cp, rows, feat_loc) then flatten.
        sliced = flat.reshape(self.rows_per_peer, self.cp, self.feat_loc)  # (rows, cp, Dloc)
        # -> (cp, rows, Dloc) so peer r's chunk is contiguous rows.
        send.copy_(sliced.permute(1, 0, 2).reshape(self.cp * self.rows_per_peer, self.feat_loc))

    def front_unpack(self, recv: torch.Tensor) -> torch.Tensor:
        """Unpack ``recv`` ``(cp*rows_per_peer, cols)`` -> feature-shard ``(B, N, N, feat_loc)``.

        ``recv[s]`` = peer ``s``'s local-token block for my D-slice; concatenating
        along the split token axis reforms the full token grid.
        """
        # (cp, rows_per_peer, feat_loc): slot s = peer s's (B,N_loc,N) block.
        chunks = recv.reshape(self.cp, self.B, self.N_loc, self.N, self.feat_loc)
        # concat along the cp-split token axis (token_dim within (B,N,N,feat)).
        # chunks is (cp, B, N_loc, N, feat_loc); the split axis is token_dim-1 in the
        # (B,N_loc,N) frame -> for token_dim==1: cat over axis giving (B, cp*N_loc=N, N, feat_loc).
        if self.token_dim == 1:
            out = chunks.permute(1, 0, 2, 3, 4).reshape(self.B, self.N, self.N, self.feat_loc)
        else:  # token_dim == 2: peer blocks are (B, N, N_loc) -> cat along dim 2
            # here local block was (B, N, N_loc, feat); flat reshape used (B,N_loc,N)
            # frame, so for token_dim==2 reinterpret: chunks (cp,B,N,N_loc,feat_loc).
            chunks = recv.reshape(self.cp, self.B, self.N, self.N_loc, self.feat_loc)
            out = chunks.permute(1, 2, 0, 3, 4).reshape(self.B, self.N, self.N, self.feat_loc)
        return out.contiguous()

    def front_unpack_dmajor(self, recv: torch.Tensor) -> torch.Tensor:
        """Unpack ``recv`` directly to ``(feat_loc, B*N*N)`` D-major (GEMM1-native).

        FUSES :meth:`front_unpack`'s token-grid reassembly with the
        ``(B,N,N,Dloc) -> (Dloc, B*N*N)`` transpose the WGMMA GEMM1 needs — ONE
        permute+contiguous instead of two. Token order is ``(b, i, j)`` row-major
        (``i`` = the cp-gathered full-N axis), matching ``_gemm1``'s ``(D, M)``
        contract. (A transpose is still paid — the A2A delivers token-major
        ``(...,Dloc)`` while the GEMM wants D-major; this just removes the
        redundant second pass. Eliminating it entirely is the Wave-2 fusion: the
        comm store lands in GEMM-native layout.)
        """
        if self.token_dim == 1:
            chunks = recv.reshape(self.cp, self.B, self.N_loc, self.N, self.feat_loc)
            # -> (feat_loc, B, cp, N_loc, N) -> (feat_loc, B*N*N) with i=(cp,N_loc)=N.
            return (
                chunks.permute(4, 1, 0, 2, 3)
                .reshape(self.feat_loc, self.B * self.N * self.N)
                .contiguous()
            )
        chunks = recv.reshape(self.cp, self.B, self.N, self.N_loc, self.feat_loc)
        return (
            chunks.permute(4, 1, 2, 0, 3)
            .reshape(self.feat_loc, self.B * self.N * self.N)
            .contiguous()
        )

    # ---- BACK: feature-shard (B,N,N,feat_loc) -> CpAllToAll send ------------
    def back_pack(self, tri_feat: torch.Tensor, send: torch.Tensor) -> None:
        """Pack feature-shard ``tri`` ``(B, N, N, feat_loc)`` into ``send`` for the back A2A.

        Inverse of :meth:`front_unpack`: ``send[r]`` = the token sub-block destined
        for peer ``r`` (peer ``r`` owns token rows ``r*N_loc:(r+1)*N_loc``), for my
        D-slice. After the exchange + :meth:`back_unpack`, each rank reassembles
        the full feature ``D`` for its token block.
        """
        # tri_feat (B, N, N, feat_loc); split the full token axis into cp peer blocks.
        if self.token_dim == 1:
            blocks = tri_feat.reshape(self.B, self.cp, self.N_loc, self.N, self.feat_loc)
            # -> (cp, B, N_loc, N, feat_loc) -> rows per peer
            send.copy_(
                blocks.permute(1, 0, 2, 3, 4).reshape(self.cp * self.rows_per_peer, self.feat_loc)
            )
        else:
            blocks = tri_feat.reshape(self.B, self.N, self.cp, self.N_loc, self.feat_loc)
            send.copy_(
                blocks.permute(2, 0, 1, 3, 4).reshape(self.cp * self.rows_per_peer, self.feat_loc)
            )

    def back_pack_from_gemm1(self, tri_g1: torch.Tensor, send: torch.Tensor) -> None:
        """Pack ``_gemm1``'s NATIVE ``(feat_loc*B, N, N)`` output into ``send`` directly.

        FUSES the back-A2A pack with the ``(feat_loc*B,N,N) -> (B,N,N,feat_loc)``
        transpose that :meth:`back_pack` would otherwise need — the output-side
        analogue of :meth:`front_unpack_dmajor`. ``_gemm1`` returns
        ``(feat_loc*B, N, N)`` with ``L = d*B + b`` (D-slice as the leading batch
        axis), i.e. a free ``(feat_loc, B, N, N)`` view. We split the cp token axis
        and emit ``send[r]`` = peer ``r``'s token block for my D-slice, rows ordered
        ``(b, i_local, j)`` row-major to match :meth:`front_pack`'s convention — so
        the einsum's result feeds the back A2A with NO separate transpose pass
        (the T2.0c wall; the layout-seam fusion).
        """
        g = tri_g1.reshape(self.feat_loc, self.B, self.N, self.N)  # (Dloc, B, i=N, j=N)
        if self.token_dim == 1:
            # cp-split the i axis (N) -> (Dloc, B, cp, N_loc, N), peer r = i-block r.
            blocks = g.reshape(self.feat_loc, self.B, self.cp, self.N_loc, self.N)
            # send[r] rows = (b, i_local, j) row-major, cols = Dloc -> (cp, B, N_loc, N, Dloc).
            send.copy_(
                blocks.permute(2, 1, 3, 4, 0).reshape(self.cp * self.rows_per_peer, self.feat_loc)
            )
        else:
            # cp-split the j axis (N) -> (Dloc, B, N, cp, N_loc), peer r = j-block r.
            blocks = g.reshape(self.feat_loc, self.B, self.N, self.cp, self.N_loc)
            send.copy_(
                blocks.permute(3, 1, 2, 4, 0).reshape(self.cp * self.rows_per_peer, self.feat_loc)
            )

    def back_unpack(self, recv: torch.Tensor) -> torch.Tensor:
        """Unpack back-A2A ``recv`` -> token-shard (token axis ``token_dim``).

        ``recv[s]`` = peer ``s``'s D-slice for my token block; concatenating along
        the feature axis reforms the full feature ``D``. Returns
        ``(B, N_loc, N, feat)`` for ``token_dim==1`` or ``(B, N, N_loc, feat)`` for
        ``token_dim==2`` (the same token-shard frame as the front's local operand).
        """
        if self.token_dim == 1:
            # (cp, B, N_loc, N, feat_loc): slot s = peer s's D-slice for my token block.
            chunks = recv.reshape(self.cp, self.B, self.N_loc, self.N, self.feat_loc)
            out = chunks.permute(1, 2, 3, 0, 4).reshape(self.B, self.N_loc, self.N, self.feat)
        else:  # token_dim == 2: my token block is (B, N, N_loc)
            chunks = recv.reshape(self.cp, self.B, self.N, self.N_loc, self.feat_loc)
            out = chunks.permute(1, 2, 3, 0, 4).reshape(self.B, self.N, self.N_loc, self.feat)
        return out.contiguous()

    # ---- BACK design-E: GEMM-native (token-contiguous, d-OUTER) recv ----------
    # The fused back store (GemmA2ASm90 design E) writes the einsum's NATIVE output
    # tile directly into a peer's symmetric recv via a plain TMA S2G box (NO
    # transpose at the store). The recv is laid out so (a) the store box is
    # shape-matched to the GEMM's token-major SMEM epilogue tile (j=GEMM-N is the
    # stride-1 dim of BOTH -> leading dims agree -> the TMA atom builds, unlike the
    # d-major/d-inner transpose store which TMA rejects), and (b) the consumer
    # (LN+DualGatedGEMM back-half) reads it as the validated MN-major LayoutLeft
    # input (M=tokens contiguous, K=D strided) through its EXISTING input TMA load
    # with NO separate transpose pass (verified: layernorm/dual_gated kernels
    # auto-detect a_major="m" from stride (1,M); the feature-axis LN reduces over
    # SMEM sA, layout-agnostic). So the transpose the d-major store could not
    # express moves to the consumer's load, which the kernel already performs.
    #
    # Layout: ``(cp, Dloc, B, N_loc, N)`` C-contiguous, axes
    # ``[slot=source-feature-slice, d, b, i_local, j]``. After the back A2A, slot
    # ``s`` holds peer ``s``'s ``Dloc`` feature slice for MY token (i) block;
    # concatenating the cp slots' ``Dloc`` reforms the full feature ``D = cp*Dloc``.

    @property
    def back_recv_gemm_native_shape(self) -> tuple[int, int, int, int, int]:
        """Shape ``(cp, Dloc, B, N_i_loc, N_j_loc)`` of the design-E GEMM-native back recv.

        Allocate the symmetric back recv buffer with THIS shape (C-contiguous). NATIVE
        1-D (cp1==1 -> N_j_loc==N) AND 2-D. The fused einsum stores a CTA ``(i, j)`` tile
        for plane ``L = d*B + b`` to peer ``(i//N_i_loc)*cp1 + (j//N_j_loc)``'s recv at
        ``[my_cp_rank, d, b, i_local, j_local]``.
        """
        return (self.cp, self.feat_loc, self.B, self.N_i_loc, self.N_j_loc)

    # ---- FRONT design-E 2-D: reassemble the slot-major front recv into the einsum-native
    # full (Dloc, B, N, N) token grid. The fused staged front store writes MY Dloc feature
    # slice of a (and b) over EACH peer's LOCAL 2-D token block into recv col
    # ``slot*rows_per_peer + (b, i_local, j_local)`` (slot = i_block*cp1 + j_block). In 1-D
    # (cp1==1) slot indexes i-blocks and j is full, so slot*N_i_loc + i_local = i_global and
    # ``a = recv[:Dloc].reshape(Dloc, B, N, N)`` is a ZERO-COPY view (the wave-1 path). In 2-D
    # the col order (i_block, j_block, b, i_local, j_local) does NOT linearize to (b, i_global,
    # j_global) row-major, so a host permute reassembles it (the design-doc §6 2-D host-unpack).
    def front_unpack_dmajor_2d(self, recv_half: torch.Tensor, N_j_pad: int = None) -> torch.Tensor:
        """Front recv half ``(Dloc, M_full=B*N*N)`` slot-major -> einsum-native ``(Dloc, B, N, N)``.

        ``recv_half`` is ``a = recv[:Dloc]`` (or ``b = recv[Dloc:]``), token col index
        ``slot*rows_per_peer + (b*N_i_loc + i_local)*N_j_loc + j_local`` with
        ``slot = i_block*cp1 + j_block``. Returns the full token grid ``(Dloc, B, N, N)``
        with ``(b, i_global, j_global)`` row-major (``i_global = i_block*N_i_loc + i_local``,
        ``j_global = j_block*N_j_loc + j_local``) — the layout ``_BackFusedStore._operands``
        reshapes as ``(Dloc, B, N, N)``. For 1-D (cp1==1) this is the zero-copy reshape; for
        2-D it is one permute + ``.contiguous()`` (host-side data movement, NO kernel change).

        ``N_j_pad`` (P2 ``pad_inner``): the recv's PADDED innermost token extent, ``N_j_loc`` (or None)
        when unpadded. The pad makes ``rows_per_peer = B*N_i_loc*N_j_pad`` — which factorises through
        ``N_i_loc`` EXACTLY, so the col law becomes ``(slot*N_i_loc + i_local)*N_j_pad + j_local`` and
        the global-i axis is STILL one mode. The 1-D branch therefore stays a **zero-copy VIEW**: a
        reshape onto the padded pitch plus a slice of the pad tail. (Padding ``rows_per_peer`` itself
        would give strides ``(rpp_pad, N_j_loc)`` that never collapse — that variant would force an
        O(Dloc*N^2) copy here, which is exactly why P2 pads the INNERMOST extent instead.) The 2-D
        branch already copies, so the pad only adds a slice before the permute and costs nothing.
        """
        Dloc = self.feat_loc
        njp = self.N_j_loc if N_j_pad is None else int(N_j_pad)
        if self.cp1 == 1 and self.B == 1:
            # 1-D at B == 1: the col law is `slot*rpp + (b*N_i_loc + i_local)*N_j_loc + j_local`,
            # i.e. mode order (slot, b, i_local, j). With B == 1 the `b` mode is degenerate, so that
            # IS (b, i_global, j) and the reshape below is a zero-copy VIEW.
            #
            # It is NOT (b, i_global, j) at B > 1 -- `slot` sits OUTSIDE `b` in the recv and outside
            # is where `b` has to be. Reshaping anyway silently transposed those two modes, which is
            # why every B == 1 test passed while a B > 1 forward returned another plane's values in
            # the right shape. The general branch below handles both cp1 > 1 and B > 1.
            if njp == self.N_j_loc:
                return recv_half.reshape(Dloc, self.B, self.N, self.N)
            # PADDED: (Dloc, B, N, N_j_pad) strided VIEW, then drop the zero pad tail. No copy.
            return recv_half.reshape(Dloc, self.B, self.N, njp)[:, :, :, : self.N]
        if self.cp1 == 1:
            # 1-D, B > 1: transpose the recv's (slot, b) modes into (b, slot). One permute +
            # contiguous, the same host-side cost the 2-D branch below already pays, and it is
            # unreachable at B == 1 -- so the zero-copy 1-D path is byte-identical to before.
            t = recv_half.reshape(Dloc, self.cp0, self.B, self.N_i_loc, njp)
            if njp != self.N_j_loc:
                t = t[:, :, :, :, : self.N_j_loc]
            t = t.permute(0, 2, 1, 3, 4)  # (Dloc, B, cp0, N_i_loc, N_j_loc)
            return t.reshape(Dloc, self.B, self.N, self.N).contiguous()
        # 2-D: (Dloc, slot=cp0*cp1, b, i_local, j_local) with slot = i_block*cp1 + j_block.
        t = recv_half.reshape(Dloc, self.cp0, self.cp1, self.B, self.N_i_loc, njp)
        if njp != self.N_j_loc:
            t = t[:, :, :, :, :, : self.N_j_loc]
        # -> (Dloc, B, i_block, i_local, j_block, j_local) -> (Dloc, B, N, N) row-major (b,i,j).
        t = t.permute(0, 3, 1, 4, 2, 5)  # (Dloc, B, cp0, N_i_loc, cp1, N_j_loc)
        return t.reshape(Dloc, self.B, self.N, self.N).contiguous()

    def back_unpack_gemm_native(self, recv: torch.Tensor) -> torch.Tensor:
        """Unpack the design-E GEMM-native back recv -> consumer LayoutLeft input.

        ``recv`` is ``(cp, Dloc, B, N_loc, N)`` (:meth:`back_recv_gemm_native_shape`).
        Returns the ``(M, D)`` value the LN+DualGatedGEMM back-half consumes, where
        ``M = B*N_loc*N`` (token, row-major ``(b, i_local, j)``) and ``D = cp*Dloc``
        (feature, order ``(cp, Dloc)`` -- the same feature order as :meth:`back_unpack`),
        in **MN-major LayoutLeft** (stride ``(1, M)``: token contiguous, feature
        strided) -- exactly the validated ``a_major="m"`` input. No transpose pass:
        this is a pure ``reshape + .t()`` (a view), the transpose is the kernel's
        native d-strided load.
        """
        D = self.cp * self.feat_loc
        M = self.B * self.N_i_loc * self.N_j_loc  # my local 2-D token block (1-D: N_j_loc==N)
        # (cp, Dloc, B, N_i_loc, N_j_loc) -> (cp*Dloc=D, B*N_i_loc*N_j_loc=M) = (K=feature,
        # M=token) row-major, feature order (cp, Dloc). .t() -> (M, D) stride (1, M) LayoutLeft.
        return recv.reshape(D, M).t()

    # ---- helpers -----------------------------------------------------------
    def _check_local(self, a_local: torch.Tensor) -> None:
        if self.token_dim == 1:
            exp = (self.B, self.N_loc, self.N, self.feat)
        else:
            exp = (self.B, self.N, self.N_loc, self.feat)
        if tuple(a_local.shape) != exp:
            raise ValueError(f"local operand {tuple(a_local.shape)} != expected {exp}")

    @property
    def send_recv_shape(self) -> tuple[int, int]:
        """Shape ``(cp*rows_per_peer, cols)`` of the CpAllToAll send/recv buffers."""
        return (self.cp * self.rows_per_peer, self.cols)

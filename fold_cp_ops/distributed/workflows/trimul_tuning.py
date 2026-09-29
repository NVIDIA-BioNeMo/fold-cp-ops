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



"""Typed, per-kernel tuning knobs for the A2A-fused TriMul -- the escape hatch that replaces
``**fused_kwargs``.

Purpose
    ``TriangularMultiplication`` currently ends in ``**fused_kwargs`` and splats it, unvalidated,
    into a constructor with twenty keyword-only parameters. That surface is flat: nothing in it
    says which knob configures which kernel, a typo survives construction and raises on the first
    forward, and seven names collide with arguments ``trimul_a2a`` binds itself. These dataclasses
    are the same knobs with the structure put back.

Functionality & semantics
    Nested by the STAGE each knob configures, mirroring the four-stage chain ``forward`` already
    narrates: prolog LayerNorm, the fused front DualGatedGEMM + A2A store, the fused back einsum +
    A2A store, and the out-gate. ``to_engine_kwargs`` flattens back to the engine's own flat
    keyword names, so this module is a VOCABULARY over the existing signature rather than a second
    configuration system -- the engine is untouched.

    **Every field defaults to ``None``, and that is load-bearing rather than tidy.**
    ``TriMulTuning().to_engine_kwargs()`` is ``{}``, so a default-tuned module calls the engine with
    LITERALLY the arguments it is called with today -- not with the same values re-spelled. That
    turns argument identity from something to re-measure into a property of the type, which is what
    the R0 oracle (``trimul_engine_args.json``) is diffed against.

    **Knobs deliberately absent, and why absence is the design:**

    * ``back_store`` -- its only non-default value (``"design_e"``) is UNREACHABLE through this
      module: ``forward`` always passes ``dynamic=True`` and the engine refuses ``dynamic`` on
      anything but ``pe_aligned``. Its sole observable effect today is a misleading error, so it is
      not a tuning knob.
    * ``hybrid_ib`` -- auto-detected from the P2P topology. Setting it ``False`` on a job with IB
      peers is a documented CUDA illegal address, not a slower run.
    * ``route2_ni`` / ``composite_k`` -- DERIVED from ``(direction, cp1)``; they select a receive
      LAYOUT, not a speed.
    * ``dynamic``, ``has_mask``, ``eps`` -- not tuning at all. ``dynamic`` has one supported value;
      the other two are properties of the call and of the model.
    * ``front_pingpong`` -- the staged front kernel is cooperative-only and refuses it.

Input requirements
    Every field is optional. A value that IS given is passed through unvalidated to the engine,
    which owns the validity rules (a tile that does not divide the peer feature slice, a cluster
    width that disagrees with the drain, ...). This layer's job is to make the knob's SUBJECT
    legible and to refuse a name that does not exist; it is not a second validator.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Optional


@dataclass(frozen=True)
class FrontTuning:
    """Knobs for STAGE 2 -- the fused front DualGatedGEMM and its A2A peer store.

    Args:
        tile_mn: CTA ``(tile_M, tile_N)`` for the front GEMM. ``tile_N`` tiles the FEATURE axis, so
            the postact half must lie inside one peer's ``D_loc`` slice; the engine clamps a tile
            that does not. ``None`` uses the resolved heuristic.
        pad_inner: Pad the front receive buffer's innermost token extent so every rank's
            destination base is 128-byte clean. ``None`` reads ``CPO_FRONT_PAD_INNER``. Default-off
            on a per-shape neutrality bar, not on caution: it is a win where the cp tier is deep or
            N is large, and a loss where the tier is shallow and N small.
        pad_eager: Compile the padded executor at construction instead of on the first forward that
            needs it. Trades a longer, expected startup for no mid-run stall; worth it only on the
            hybrid-IB front, whose second compile is seconds. Inert unless ``pad_inner`` is on.
    """

    tile_mn: Optional[tuple[int, int]] = None
    pad_inner: Optional[bool] = None
    pad_eager: Optional[bool] = None

    #: field name -> the engine's own keyword. The mapping lives here rather than in a central
    #: table so a field cannot be added without deciding where it lands.
    _ENGINE_KEYS = {"tile_mn": "front_tile_mn", "pad_inner": "front_pad_inner",
                    "pad_eager": "front_pad_eager"}


@dataclass(frozen=True)
class BackTuning:
    """Knobs for STAGE 3 -- the fused back einsum and its A2A store.

    Args:
        tile_mn: CTA ``(tile_M, tile_N)`` for the back GEMM. Unlike the front's, ``tile_N`` here
            tiles the token-j axis and is therefore independent of the feature width.
        pingpong: Two-warpgroup schedule for the back GEMM. ``None`` uses the resolved default.
        cluster_n: N-cluster width. It must match the drain's concentration -- the store derives
            that from ``cluster_shape_mnk[1]`` and the engine asserts they agree -- so this sets
            BOTH. 1-D only; a 2-D shard is force-clamped to 1.
    """

    tile_mn: Optional[tuple[int, int]] = None
    pingpong: Optional[bool] = None
    cluster_n: Optional[int] = None

    _ENGINE_KEYS = {"tile_mn": "back_tile_mn", "pingpong": "back_pingpong",
                    "cluster_n": "back_cluster_n"}


@dataclass(frozen=True)
class OutGateTuning:
    """Knobs for STAGE 4 -- the gated LayerNorm + dual GEMM that consumes the einsum result.

    Args:
        consumer: Which back-half kernel runs: ``"stagec"`` (the cooperative default),
            ``"staged_a_in_regs"`` (transposes via ldmatrix into the WGMMA register file, which
            sidesteps strided shared-memory reads at large N), or ``"torch"`` (the pure-torch
            reference). All three compute the same math.
    """

    consumer: Optional[str] = None

    _ENGINE_KEYS = {"consumer": "consumer"}


@dataclass(frozen=True)
class TransportTuning:
    """Knobs for the two A2A completion drains.

    Args:
        device_signal_nvlink: Use the post-quiet DEVICE-signal drain instead of the barrier. SELF-
            GATING: it engages only when every peer is NVLink-reachable and falls back silently on
            a job with IB peers, because a device-issued IBGDA signal has no completion path -- so
            unlike the equivalent env flag this cannot be set into a hang.
    """

    device_signal_nvlink: Optional[bool] = None

    _ENGINE_KEYS = {"device_signal_nvlink": "use_device_signal_nvlink"}


@dataclass(frozen=True)
class TriMulTuning:
    """The whole tuning surface, nested by the stage each knob configures.

    Args:
        front / back / out_gate / transport: The per-stage groups above. Each defaults to its own
            all-``None`` instance, so ``TriMulTuning()`` is fully inert.
        autotune: Run the distributed autotuner instead of the size heuristic. ``None`` reads
            ``CPO_DIST_AUTOTUNE``. Explicit tile overrides still win over whatever it elects.
        persistent: Persistent-CTA scheduling. Both fused store designs require it; it is a knob
            rather than a buried literal so that requirement is visible.

    Returns:
        From :meth:`to_engine_kwargs`, a dict of the engine's own keyword names. **Empty when
        nothing was set**, which is the property the whole design rests on.
    """

    front: FrontTuning = field(default_factory=FrontTuning)
    back: BackTuning = field(default_factory=BackTuning)
    out_gate: OutGateTuning = field(default_factory=OutGateTuning)
    transport: TransportTuning = field(default_factory=TransportTuning)
    autotune: Optional[bool] = None
    persistent: Optional[bool] = None

    _ENGINE_KEYS = {"autotune": "autotune_config", "persistent": "is_persistent"}

    def to_engine_kwargs(self) -> dict[str, Any]:
        """Flatten to ``TriMulAutotuned.__init__``'s own keyword names, omitting every unset field.

        Semantics
            A field left ``None`` is OMITTED, not passed as ``None``. The difference matters: the
            engine treats ``None`` as "decide for me" for some knobs and as a real value for
            others, and omitting keeps the call byte-identical to one that never mentioned them.

        Returns:
            ``{}`` when nothing was set anywhere in the tree.
        """
        out: dict[str, Any] = {}
        for group in (self.front, self.back, self.out_gate, self.transport):
            for f in fields(group):
                v = getattr(group, f.name)
                if v is not None:
                    out[type(group)._ENGINE_KEYS[f.name]] = v
        for name, engine_key in TriMulTuning._ENGINE_KEYS.items():
            v = getattr(self, name)
            if v is not None:
                out[engine_key] = v
        return out


#: Engine keywords this module deliberately does NOT expose, with the reason. Pinned by a test, so
#: an exclusion cannot rot into an omission nobody decided.
NOT_EXPOSED = {
    "back_store": "only non-default value is unreachable: forward always passes dynamic=True",
    "hybrid_ib": "auto-detected from the P2P topology; False with IB peers is a CUDA fault",
    "route2_ni": "derived from (direction, cp1) -- a receive layout, not a speed",
    "composite_k": "derived from (direction, cp1) -- a receive layout, not a speed",
    "dynamic": "one supported value",
    "has_mask": "a property of the call, not a tuning choice",
    "eps": "a model hyperparameter, promoted to a named constructor argument",
    "front_pingpong": "the staged front kernel is cooperative-only and refuses it",
    "device_mesh": "topology, taken from the constructor",
    "placements": "topology, taken from the constructor",
}

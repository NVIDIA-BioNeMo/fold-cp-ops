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

"""The plug-in seam (HARNESS_DESIGN §1). A benchmarkable kernel/workflow is a small dataclass of callables
so a kernel author writes one in ~30 lines. The driver never inspects the opaque `build` handle."""
from dataclasses import dataclass, field
from typing import Callable, List, Optional


@dataclass(frozen=True)
class Ctx:
    """Frozen context the driver passes to EVERY target callback. One object -> adding a field never breaks a
    target signature. `da` MUST expose is_distributed + all_reduce_max(float)->float (the consensus primitive).
    `cfg` is the current per-cell autotune config (whatever `target.configs(ctx)` yielded; {} for single)."""
    dm: object = None          # DistributedManager (or fake)
    pm: object = None          # placements / pe-map (or fake)
    da: object = None          # dist adapter: .is_distributed, .all_reduce_max(x)
    N: int = 0                 # token extent (the shape)
    cp0: int = 1               # i-axis cp (nodes on a 2-D mesh)
    cp1: int = 1               # j-axis cp (1 => 1-D token shard)
    Dloc: int = 1
    B: int = 1
    rd: int = 2
    device: object = "cpu"
    rank: int = 0
    cfg: dict = field(default_factory=dict)

    def with_cfg(self, cfg):
        # frozen -> return a copy with the per-config cfg swapped in (dataclasses.replace without the import cost)
        return Ctx(dm=self.dm, pm=self.pm, da=self.da, N=self.N, cp0=self.cp0, cp1=self.cp1, Dloc=self.Dloc,
                   B=self.B, rd=self.rd, device=self.device, rank=self.rank, cfg=dict(cfg or {}))


def _always_ok(ctx):  # default supports(): runnable at every shape
    return None


@dataclass
class BenchTarget:
    """One benchmarkable kernel/workflow. `build` MAY raise (isolated + consensus'd by the driver); `run` is the
    TIMED hot call ONLY (no alloc/sync/copy). `supports` MUST be pure shape math IDENTICAL on every rank (the
    determinism is load-bearing for collective symmetry). `role` in {"target","baseline"} (a baseline is just a
    target flagged so). `configs(ctx)->list` yields the per-cell autotune grid (None/[] => a single {} config)."""
    name: str
    build: Callable[[Ctx], object]
    run: Callable[[object], None]
    role: str = "target"
    supports: Callable[[Ctx], Optional[str]] = _always_ok
    output: Optional[Callable[[object], object]] = None
    ref: Optional[Callable[[Ctx], object]] = None
    teardown: Optional[Callable[[object], None]] = None
    configs: Optional[Callable[[Ctx], List[dict]]] = None
    # OPTIONAL comm-BW hook: comm_bytes(ctx) -> per-rank off-diagonal A2A byte volume for THIS cell (bf16).
    # When set AND the cell timed ok, the driver emits gbps_total/gbps_ib/gbps_nvl = comm_bytes / (t·1e6).
    # None (default) => time-only; the fused/back targets leave it None (comm overlaps compute, not separable).
    comm_bytes: Optional[Callable[[Ctx], int]] = None
    # OPTIONAL extra-fields hook: cell_meta(ctx) -> dict merged INTO the ok cell (e.g. {"gemm_cfg": {...}} —
    # the autotuned/heuristic kernel config a Table-B column reads). Deterministic shape math; None => {}.
    cell_meta: Optional[Callable[[Ctx], dict]] = None

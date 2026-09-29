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

"""The Phase-D (strong) + Phase-E (weak) scaling CELL LIST — ONE source of truth.

Both launchers (``run_scaling_grid.sh`` local/direct-ssh, ``run_scaling.sbatch`` slurm) and the aggregator
(``aggregate_scaling.py``) read the grid from HERE, so the shell loop, the batch script and the published
table can never drift apart. GPU-free and dependency-free (stdlib only) — it can be listed/diffed/reviewed
without a cluster.

  python -m benchmark.distributed.harness.scaling_cells            # TSV (the shell driver's input)
  python -m benchmark.distributed.harness.scaling_cells --json     # JSON (the aggregator's input)
  python -m benchmark.distributed.harness.scaling_cells --max-cp 8 # only what fits one node

A CELL = ONE ISOLATED LAUNCH = ``(cp0, cp1, N, direction)`` at D=256 — a fresh ``torchrun``/``srun`` with a
fresh process group + fresh nvshmem init, per the CLAUDE.md multi-cell-isolation HARD RULE. ``mask`` is
deliberately NOT a cell dimension: it is the harness's per-cell CONFIG axis (``trimul_e2e._configs`` ->
``[{has_mask:False},{has_mask:True}]``), so ONE launch times both mask states inside its own process — 2
fused builds/process, each released by the target's ``teardown``.

PHASE OVERLAP IS DEDUPED. Phase D (§6: N in {2048,4096}, cp in {1,2,4,8,16} 1-D and {1,(2,2),(2,4),(4,4),
(2,8)} 2-D) and Phase E (§7 W2: N = 2048*sqrt(cp) snapped to %8) share several cells — e.g. 1-D cp=4 N=4096
and 2-D (2,2) N=4096 are BOTH a Phase-D cell and the matching Phase-E cell, and the cp=1 N=2048 cell is the
t(1) denominator for all four tables. Each unique cell is emitted ONCE, carrying the set of phases it
serves in ``phases``; the aggregator selects rows per table. That is ~6 launches (and ~6x cold compiles)
saved with zero loss of coverage.

WHY N = 2048*sqrt(cp) FOR PHASE E (plan §0.1, decision W2): TriMul is the SQUARE einsum
``out[b,i,j,d] = sum_k a[b,i,k,d]*b[b,j,k,d]`` with i=j=k=N_token, so a global N_i != N_j is not a TriMul at
all and the literal "shard = (2048*cp0, 2048*cp1)" request is unrunnable off the square meshes. W2 instead
holds the PER-DEVICE SHARD constant at N^2/cp = 2048^2 elements, which every requested mesh can express.
Consequence, stated on the results page: per-device compute is O(N^3/cp) while the shard is O(N^2/cp), so a
constant shard makes per-device FLOPs grow proportional to N proportional to sqrt(cp) — the ideal weak-scaling
law is ``t ~ sqrt(cp)``, NOT flat, and the efficiency is ``E(cp) = t(1)*sqrt(cp)/t(cp)``.
"""
import argparse
import json

D_FEAT = 256              # feature dim for the whole scaling study (plan §6/§7)
B_BATCH = 1
GPUS_PER_NODE = 8         # H100 node (8 on every cluster this has run on)

# Strong scaling — strong scaling: FIXED global problem, growing cp.
_D_N = (2048, 4096)
_D_1D = (1, 2, 4, 8, 16)                                  # cp0 (cp1 == 1)
_D_2D = ((1, 1), (2, 2), (2, 4), (4, 4), (2, 8))          # (cp0, cp1)

# Weak scaling — weak scaling, constant per-device shard: constant per-device shard N^2/cp = 2048^2 => N = 2048*sqrt(cp),
# snapped to the 16-byte floor (N % 8 == 0). Snapped values are exact, not re-derived, so the published
# table and the launcher agree byte-for-byte: sqrt(2)*2048 = 2896.3 -> 2896; sqrt(8)*2048 = 5792.6 -> 5792.
_E_1D = ((1, 1, 2048), (2, 1, 2896), (4, 1, 4096), (8, 1, 5792))
_E_2D = ((1, 1, 2048), (2, 2, 4096), (2, 4, 5792), (4, 4, 8192), (2, 8, 8192))

DIRECTIONS = ("outgoing", "incoming")
_SUF = {"outgoing": "out", "incoming": "in"}


def _shape_ok(N, cp0, cp1):
    """Mirror of ``trimul_e2e._supports``'s pure-shape gate, so a cell that the harness would SKIP never
    gets a launch. Kept as an assertion rather than a filter: a grid entry that fails this is an authoring
    bug in the tables above, not a runtime condition to silently drop."""
    cp = cp0 * cp1
    if N % cp0 or N % cp1 or N % 8:
        return f"N={N} not shardable by ({cp0},{cp1}) or N%8!=0"
    if cp1 > 1 and ((N // cp0) % 8 or (N // cp1) % 8):
        return f"N={N} per-axis local extent not %8 for 2-D ({cp0},{cp1})"
    if D_FEAT % cp:
        return f"D={D_FEAT} % cp={cp} != 0"
    if D_FEAT // cp < 8:
        return f"Dloc={D_FEAT // cp} < 8 (front feature-scatter floor)"
    return None


def _mesh_spec(cp0, cp1):
    """``CPO_DIST_MESH`` for the harness CLI: 1-D => 'cp=<cp0>', 2-D => 'cp=<cp0>*<cp1>' (parsed by
    harness/__main__.py:63)."""
    return f"cp={cp0}" if cp1 == 1 else f"cp={cp0}*{cp1}"


def _cell(cp0, cp1, N, direction, phases):
    cp = cp0 * cp1
    nnodes = max(1, -(-cp // GPUS_PER_NODE))               # ceil-div: cp<=8 -> 1 node, cp=16 -> 2
    ntpn = cp if cp <= GPUS_PER_NODE else GPUS_PER_NODE
    suf = _SUF[direction]
    single = (cp == 1)
    return {
        "tag": f"cp{cp0}x{cp1}_N{N}_{suf}",
        "phases": "".join(sorted(phases)),                 # 'D', 'E', or 'DE'
        "sharding": "1d" if cp1 == 1 else "2d",
        "cp0": cp0, "cp1": cp1, "cp": cp, "N": N, "D": D_FEAT, "B": B_BATCH,
        "direction": direction,
        "N_i_loc": N // cp0, "N_j_loc": N // cp1,
        "shard_elems": B_BATCH * (N // cp0) * (N // cp1) * D_FEAT,
        "mesh": _mesh_spec(cp0, cp1),
        "nnodes": nnodes, "ntasks_per_node": ntpn,
        # node_sz drives the driver's IB/NVLink derived-BW split (driver._node_sz). Its default infers
        # node_sz from cp1, which is only right when the 2-D mesh is (nodes, within-node); pass the TRUE
        # intra-NVLink-domain width explicitly so a (4,4)-on-2-nodes cell does not mis-attribute its
        # off-diagonal volume between IB and NVLink.
        "node_sz": min(cp, GPUS_PER_NODE),
        "single_device": single,
        # cp>=2: the shipped DTensor API vs the reference-CP native ring.
        # cp=1: BOTH the production fallback path (FusedTriMulCP on a plain tensor -> _is_single_device ->
        # trimul_autotuned, no mesh/PeMap/nvshmem) AND the raw trimul_autotuned reference, in ONE process.
        # The production path is what the tables quote for t(1) -- it is the same entry point the cp>=2
        # rows use, so the speedup column stays apples-to-apples -- while the raw reference costs almost
        # nothing extra and prices the fallback's dispatch overhead directly. No baseline at cp=1: the
        # reference-CP ring needs a live mesh.
        "targets": (f"trimul_fusedcp_{suf},trimul_single_{suf}" if single else f"trimul_fusedcp_{suf}"),
        "baselines": "" if single else f"trimul_dtensor_baseline_ring_reducescatter_{suf}",
    }


def cells(max_cp=None, phases="DE", directions=DIRECTIONS, min_cp=None):
    """The deduped cell list, cheapest-first (cp ascending, then N ascending).

    Cheapest-first is the smoke order the CLAUDE.md bench discipline wants: the first cell to run is the
    smallest, so a broken grid fails in ~a minute on 1 GPU instead of after burning the 2-node cells.
    """
    want = set(phases.upper())
    seen = {}   # (cp0,cp1,N,direction) -> phases set
    def _add(cp0, cp1, N, direction, ph):
        if ph not in want:
            return
        seen.setdefault((cp0, cp1, N, direction), set()).add(ph)
    for direction in directions:
        for N in _D_N:                                              # Strong scaling
            for cp0 in _D_1D:
                _add(cp0, 1, N, direction, "D")
            for cp0, cp1 in _D_2D:
                _add(cp0, cp1, N, direction, "D")
        for cp0, cp1, N in _E_1D + _E_2D:                           # Weak scaling
            _add(cp0, cp1, N, direction, "E")
    out = []
    for (cp0, cp1, N, direction), ph in seen.items():
        cp = cp0 * cp1
        # min/max cp bracket the grid onto an ALLOCATION: --max-cp 8 is the single-node half, --min-cp 16
        # the two-node remainder. Splitting this way means the 2-node request is held only for the cells
        # that genuinely need two nodes (shared-cluster citizenship), and the two halves still land in ONE
        # results dir so the table is continuous.
        if max_cp is not None and cp > int(max_cp):
            continue
        if min_cp is not None and cp < int(min_cp):
            continue
        bad = _shape_ok(N, cp0, cp1)
        assert bad is None, f"grid authoring bug: cell ({cp0},{cp1}) N={N} is unrunnable: {bad}"
        out.append(_cell(cp0, cp1, N, direction, ph))
    out.sort(key=lambda c: (c["cp"], c["N"], c["cp1"], c["direction"]))
    return out


_TSV_COLS = ("tag", "phases", "sharding", "cp0", "cp1", "cp", "N", "D", "direction", "mesh", "nnodes",
             "ntasks_per_node", "node_sz", "single_device", "targets", "baselines")


def main():
    ap = argparse.ArgumentParser("scaling_cells")
    ap.add_argument("--json", action="store_true", help="emit JSON (default: TSV for the shell driver)")
    ap.add_argument("--max-cp", type=int, default=None, help="drop cells above this cp (e.g. 8 = 1 node)")
    ap.add_argument("--min-cp", type=int, default=None, help="drop cells below this cp (e.g. 16 = 2-node only)")
    ap.add_argument("--phases", default="DE", help="'D' strong only, 'E' weak only, 'DE' both (default)")
    ap.add_argument("--directions", default="outgoing,incoming")
    ap.add_argument("--header", action="store_true", help="TSV: emit a leading column-name row")
    a = ap.parse_args()
    cs = cells(max_cp=a.max_cp, min_cp=a.min_cp, phases=a.phases,
               directions=tuple(d.strip() for d in a.directions.split(",") if d.strip()))
    if a.json:
        print(json.dumps(cs, indent=2))
        return 0
    if a.header:
        print("\t".join(_TSV_COLS))
    for c in cs:
        print("\t".join(str(int(c[k]) if isinstance(c[k], bool) else c[k]) for k in _TSV_COLS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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

"""Turn the Phase-D/E per-cell harness JSON into the two published scaling tables.

  python -m benchmark.distributed.harness.aggregate_scaling --results <SC_OUTBASE> [--md out.md]

Reads every ``<SC_OUTBASE>/<tag>/bench_N<N>.json`` the driver flushed (one dir per isolated cell) and joins
it against the expected grid from ``scaling_cells.py``, so a cell that never ran shows up as an explicit
MISSING row rather than silently vanishing from the table. Stdlib only; runs anywhere, no GPU.

TWO DIFFERENT IDEAL LAWS -- this is the whole point of keeping the tables apart:

  * Phase D, STRONG scaling (fixed global problem, growing cp): ideal ``t(cp) = t(1)/cp``, so
    ``speedup = t(1)/t(cp)`` and ``efficiency = t(1)/(cp*t(cp))``.
  * Phase E, WEAK scaling under definition W2 (constant per-device shard ``N^2/cp = 2048^2``): the ideal is
    ``t(cp) = t(1)*sqrt(cp)``, **NOT flat**. TriMul compute is O(N^3) while the shard is O(N^2), so holding
    the shard constant grows per-device FLOPs as ``N ~ sqrt(cp)``. The efficiency is therefore
    ``E(cp) = t(1)*sqrt(cp)/t(cp)``. A naive ``t(1)/t(cp)`` column would read as a catastrophic failure
    (it would show ~1/sqrt(cp)) and is deliberately NOT emitted as a headline; ``naive_ratio`` is carried
    only in the CSV, clearly named, so nobody re-derives it by accident.

The ``shard_elems`` column is the EVIDENCE that the W2 definition holds: it must be constant (2048^2*D) down
a Phase-E column. It is read back from the measured cell's ``cell_meta``, not recomputed from the grid, so a
mismatch between what was intended and what actually ran is visible rather than assumed.
"""
import argparse
import json
import math
import os

from benchmark.distributed.harness import scaling_cells as SC

_MASK = {False: "mask-off", True: "mask-on"}


def load_many(dirs):
    """Merge several result dirs, LATER dirs winning on a key collision.

    Needed because a subset of cells can legitimately have to be re-measured under a different LAUNCH
    condition (e.g. the cp=1 cells re-run with the JIT cache disabled to dodge the cross-cell cache
    collision). Merging at read time — rather than copying cell dirs together on disk — keeps each run's
    own `env.json` and manifest intact next to the cells it actually produced, so the provenance of every
    number stays traceable to the job that measured it. The condition itself is reported in the doc's
    method section; it is a launch-env setting, not a property of the tree, and `scaling_env.py` already
    captures every `CPO_*` var into each dir's `env.json`."""
    rec = {}
    for d in dirs:
        rec.update(_load(d))
    return rec


def _load(results_dir):
    """(cp0,cp1,N,direction_suffix,target_base,has_mask) -> the measured config record.

    One record per (target, config) -- the harness writes each cell's configs[] with the per-config
    ``cfg`` (``{has_mask: ...}``), ``time_ms``, and the merged ``cell_meta`` columns.
    """
    out = {}
    if not os.path.isdir(results_dir):
        return out
    for tag in sorted(os.listdir(results_dir)):
        d = os.path.join(results_dir, tag)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not (fn.startswith("bench_N") and fn.endswith(".json")):
                continue
            try:
                with open(os.path.join(d, fn)) as f:
                    blob = json.load(f)
            except Exception:
                continue
            for cell in blob.get("cells", []):
                name = str(cell.get("target", ""))
                base, _, suf = name.rpartition("_")
                for sub in cell.get("configs", []) or []:
                    cfg = sub.get("cfg") or {}
                    # the reference CP target declares no config grid (configs=None -> a single {}); it ALWAYS runs mask-ON
                    # (it executes x*mask.unsqueeze(-1) unconditionally), so an empty cfg means mask-ON.
                    hm = bool(cfg.get("has_mask", True)) if cfg else True
                    key = (int(cell.get("cp0", 1)), int(cell.get("cp1", 1)), int(blob.get("N", 0)),
                           suf, base, hm)
                    out[key] = {**sub, "tag": tag}
    return out


def _get(rec, cp0, cp1, N, suf, hm, bases):
    for b in bases:
        r = rec.get((cp0, cp1, N, suf, b, hm))
        if r is not None:
            return r
    return None


def _fused(rec, cp0, cp1, N, suf, hm):
    """The fold_cp_ops number for a cell. ``trimul_fusedcp`` FIRST at every cp INCLUDING cp=1: since the Phase-A
    fallback landed, cp=1 runs the same public entry point (``FusedTriMulCP`` -> ``_is_single_device`` ->
    ``trimul_autotuned``) as the cp>=2 rows, so quoting it keeps the speedup column apples-to-apples.
    ``trimul_single`` (raw ``trimul_autotuned``) is the fallback and the cp=1 cross-check."""
    return _get(rec, cp0, cp1, N, suf, hm, ("trimul_fusedcp", "trimul_single", "trimul_fused"))


def _ms(r):
    return None if not r else r.get("time_ms")


def _fmt(v, nd=3):
    return "--" if v is None else f"{v:.{nd}f}"


def _status(r):
    if r is None:
        return "MISSING"
    return str(r.get("status", "?"))


def _ladder(cells, sharding, N=None, phase="D"):
    """The cp ladder for one table. cp=1 is stored with sharding '1d' (cp1==1) but is the denominator row of
    BOTH the 1-D and the 2-D table, so it is admitted to either."""
    out = []
    for c in cells:
        if phase not in c["phases"]:
            continue
        if c["cp"] != 1 and c["sharding"] != sharding:
            continue
        if N is not None and c["N"] != N:
            continue
        out.append(c)
    seen, uniq = set(), []
    for c in sorted(out, key=lambda c: (c["cp"], c["N"])):
        k = (c["cp0"], c["cp1"], c["N"])
        if k not in seen:
            seen.add(k)
            uniq.append(c)
    return uniq


def table_d(rec, cells, sharding, N, suf, hm):
    """Strong scaling: fixed global N, growing cp. Ideal t(cp) = t(1)/cp."""
    rows, t1 = [], None
    for c in _ladder(cells, sharding, N=N, phase="D"):
        r = _fused(rec, c["cp0"], c["cp1"], c["N"], suf, hm)
        t = _ms(r)
        if c["cp"] == 1:
            t1 = t
        b = _ms(_get(rec, c["cp0"], c["cp1"], c["N"], suf, hm, ("trimul_dtensor_baseline_ring_reducescatter",)))
        rows.append({
            "mesh": f"{c['cp0']}x{c['cp1']}" if c["cp1"] > 1 else str(c["cp0"]),
            "cp": c["cp"], "N_local": f"{c['N_i_loc']}x{c['N_j_loc']}", "ms": t,
            "speedup": (t1 / t) if (t and t1) else None,
            "eff": (t1 / (c["cp"] * t)) if (t and t1) else None,
            "dtensor_baseline_ring_reducescatter_ms": b, "vs_dtensor_baseline_ring_reducescatter": (b / t) if (t and b) else None,
            "cfg": _cfgstr(r), "peak_mb": (r or {}).get("peak_torch_mb"),
            "dev_mb": (r or {}).get("dev_used_mb"), "status": _status(r),
            # cp=1 only: the RAW trimul_autotuned reference measured in the same process, so the
            # fallback's pure dispatch overhead is visible instead of assumed.
            "single_ms": _ms(_get(rec, c["cp0"], c["cp1"], c["N"], suf, hm, ("trimul_single",))),
        })
    return rows


def table_e(rec, cells, sharding, suf, hm):
    """Weak scaling (W2): constant per-device shard. Ideal t(cp) = t(1)*sqrt(cp) -- NOT flat."""
    rows, t1 = [], None
    for c in _ladder(cells, sharding, N=None, phase="E"):
        r = _fused(rec, c["cp0"], c["cp1"], c["N"], suf, hm)
        t = _ms(r)
        if c["cp"] == 1:
            t1 = t
        b = _ms(_get(rec, c["cp0"], c["cp1"], c["N"], suf, hm, ("trimul_dtensor_baseline_ring_reducescatter",)))
        ideal = (t1 * math.sqrt(c["cp"])) if t1 else None
        rows.append({
            "mesh": f"{c['cp0']}x{c['cp1']}" if c["cp1"] > 1 else str(c["cp0"]),
            "cp": c["cp"], "N": c["N"], "N_local": f"{c['N_i_loc']}x{c['N_j_loc']}",
            # measured shard, read back from the cell (not the grid) so intent-vs-reality is visible
            "shard_elems": (r or {}).get("shard_elems", c["shard_elems"]),
            "ms": t, "ideal_ms": ideal,
            "eff": (ideal / t) if (t and ideal) else None,
            "naive_ratio": (t1 / t) if (t and t1) else None,   # CSV only -- never the headline (see module doc)
            "peak_mb": (r or {}).get("peak_torch_mb"), "dev_mb": (r or {}).get("dev_used_mb"),
            "dtensor_baseline_ring_reducescatter_ms": b, "cfg": _cfgstr(r), "status": _status(r),
        })
    return rows


def _cfgstr(r):
    if not r:
        return "--"
    bits = [str(r.get(k)) for k in ("combo", "incoming_variant") if r.get(k)]
    return "/".join(bits) if bits else str(r.get("path", "--"))[:28]


def _md_d(rows, title):
    L = [f"**{title}**", "",
         "| mesh | cp | N_local | fold_cp_ops ms | speedup vs cp=1 | par. eff | dtensor_baseline_ring_reducescatter ms | vs dtensor_baseline_ring_reducescatter | peak MB | config |",
         "|---|--:|---|--:|--:|--:|--:|--:|--:|---|"]
    for r in rows:
        L.append(f"| {r['mesh']} | {r['cp']} | {r['N_local']} | {_fmt(r['ms'])} | "
                 f"{_fmt(r['speedup'], 2)} | {_fmt(r['eff'], 2)} | {_fmt(r['dtensor_baseline_ring_reducescatter_ms'])} | "
                 f"{_fmt(r['vs_dtensor_baseline_ring_reducescatter'], 2)}x | {_fmt(r['peak_mb'], 0)} | {r['cfg']} |")
    # Make the denominator auditable: t(1) is the PRODUCTION cp=1 fallback, and the raw trimul_autotuned
    # measured beside it in the same process shows what that fallback's dispatch actually costs.
    c1 = next((r for r in rows if r["cp"] == 1 and r["ms"] and r.get("single_ms")), None)
    if c1:
        L.append("")
        L.append(f"> t(1) = {c1['ms']:.3f} ms via the production cp=1 fallback (`FusedTriMulCP` → "
                 f"`_is_single_device` → `trimul_autotuned`); raw `trimul_autotuned` measured in the same "
                 f"process = {c1['single_ms']:.3f} ms "
                 f"(fallback dispatch overhead {100 * (c1['ms'] / c1['single_ms'] - 1):+.1f}%).")
    return "\n".join(L) + "\n"


def _md_e(rows, title):
    L = [f"**{title}**  (ideal law `t = t(1)·√cp`, NOT flat — see §W2)", "",
         "| mesh | cp | N | N_local | shard elems | fold_cp_ops ms | ideal t(1)·√cp | E(cp) | peak MB | dtensor_baseline_ring_reducescatter ms | config |",
         "|---|--:|--:|---|--:|--:|--:|--:|--:|--:|---|"]
    for r in rows:
        L.append(f"| {r['mesh']} | {r['cp']} | {r['N']} | {r['N_local']} | {r['shard_elems']} | "
                 f"{_fmt(r['ms'])} | {_fmt(r['ideal_ms'])} | {_fmt(r['eff'], 2)} | "
                 f"{_fmt(r['peak_mb'], 0)} | {_fmt(r['dtensor_baseline_ring_reducescatter_ms'])} | {r['cfg']} |")
    return "\n".join(L) + "\n"


def _env_block(results_dir):
    """Render `env.json` (written by scaling_env.py on an allocated node, inside the job's own container,
    BEFORE the grid) into the doc's provenance table.

    Plan §7 + CLAUDE.md require the tables to document cluster/node, GPU model+count, driver/CUDA,
    NCCL/NVSHMEM, torch+cutlass-dsl, image, transport, profile, clock state and the exact commit. Rendering
    that from the captured artifact — rather than typing it — is what makes it auditable. If the artifact is
    absent the block says so LOUDLY instead of quietly omitting rows: an env block that silently lost half
    its fields is worse than one that admits it.
    """
    p = os.path.join(results_dir, "env.json")
    if not os.path.isfile(p):
        return ("> **ENV CAPTURE MISSING** — `env.json` was not written for this run, so the hardware /\n"
                "> software provenance below is UNVERIFIED. Re-run `scaling_env.py` on the same allocation\n"
                "> before publishing these tables.\n")
    try:
        with open(p) as f:
            e = json.load(f)
    except Exception as exc:
        return f"> **ENV CAPTURE UNREADABLE** ({exc!r}) — do not publish without re-capturing.\n"
    sl = e.get("slurm") or {}
    rows = [
        ("cluster", e.get("cluster")),
        ("node / partition", f"{e.get('hostname')} / {sl.get('SLURM_JOB_PARTITION')}"),
        ("allocation", f"job {sl.get('SLURM_JOB_ID')}, {sl.get('SLURM_NNODES')} node(s), "
                       f"{sl.get('SLURM_NTASKS_PER_NODE')} task(s)/node, nodes {sl.get('SLURM_JOB_NODELIST')}"),
        ("GPU", f"{e.get('gpu_count')} x {e.get('gpu_name')}"),
        ("compute capability", e.get("device_capability")),
        ("driver", e.get("driver")),
        ("CUDA (torch)", e.get("torch_cuda")),
        ("NCCL (torch)", e.get("torch_nccl")),
        ("nvshmem", e.get("pkg::nvidia-nvshmem-cu13") or e.get("pkg::nvidia-nvshmem-cu12")),
        ("torch", e.get("torch")),
        ("cutlass-dsl", e.get("pkg::nvidia-cutlass-dsl") or e.get("pkg::cutlass-dsl")),
        ("container image", e.get("container_image")),
        # Verbatim, never interpreted: venue B reports GpuFreq=control_disabled, so this must read as the
        # fact it is rather than as a claimed clock lock.
        ("clock state (verbatim)", e.get("clocks")),
        ("commit under test", f"`{e.get('commit')}`"
                              + (f" _(source: {e['commit_source']})_" if e.get("commit_source") else "")
                              + ("  **(DIRTY WORKING TREE)**" if e.get("git_dirty") else "")),
    ]
    out = ["| item | value |", "|---|---|"]
    out += [f"| {k} | {v} |" for k, v in rows]
    return "\n".join(out) + "\n"


def _transport_note(cells):
    """State the transport per cp band. Derived from the CELL LIST (run_scaling_cell.sh keys the profile off
    nnodes), not from a captured env var — the env snapshot is taken once before the grid, whereas the
    profile is set per cell."""
    single = sorted({c["cp"] for c in cells if c["nnodes"] == 1})
    multi = sorted({c["cp"] for c in cells if c["nnodes"] > 1})
    L = [f"* **cp {single}** — single node: intra-node NVLink/P2P (transport auto-detected; "
         f"`CPO_NVSHMEM_PROFILE` deliberately UNSET so the fabric class is chosen from topology)."]
    if multi:
        L.append(f"* **cp {multi}** — cross-node: `CPO_NVSHMEM_PROFILE=ib-ibgda`, IBGDA over IB with "
                 f"NVSHMEM's own topology-aware **multi-rail** NIC selection. **No NIC-PE map is set** "
                 f"(a single-rail round-robin map is a measured ~7x regression on this fabric class).")
    return "\n".join(L) + "\n"


def _notes(path):
    """Splice an AUTHORED prose section (findings / limitations) into the generated page.

    The tables stay machine-generated from the per-cell JSON — hand-transcribing forty rows into prose is
    exactly the transcription risk this whole pipeline exists to avoid — while the interpretation lives in
    a reviewable file under version control. So the page is fully regenerable and the narrative is not."""
    if not path:
        return ""
    try:
        with open(path) as f:
            return f.read().rstrip() + "\n"
    except Exception as exc:
        return f"> **NOTES FILE UNREADABLE** ({exc!r}) — the findings section is missing from this page.\n"


def build_doc(rec, cells, chunks, results_dir, notes_path=None):
    """Assemble the full results page from the artifacts (env.json + the per-cell JSON) so every number on
    the page is traceable to a file on disk."""
    L = ["# Context-parallel TriMul — strong- and weak-scaling", "",
         "Generated by `benchmark/distributed/harness/aggregate_scaling.py` from the per-cell harness JSON "
         f"under `{results_dir}`. Every row traces to `<tag>/bench_N<N>.json`; the grid definition is "
         "`benchmark/distributed/harness/scaling_cells.py`.", "",
         "## Environment", "", _env_block(results_dir), "",
         "### Transport", "", _transport_note(cells), "",
         "## Method", "",
         "* **Isolation** — one cell = one fresh `torchrun`/`srun` (fresh process group + fresh nvshmem "
         "init). A crashed, hung or OOM cell is contained and the grid continues.",
         "* **Timing** — `bench_utils.benchmark_single(mode=\"event\", reduce=\"max\")`: CUDA-event device "
         "time, barrier-bracketed, `all_reduce(MAX)` to the slowest PE. Fixed iters keep ranks in lockstep "
         "(the adaptive `mode=\"device\"` would desync in-kernel collectives).",
         "* **Mask** — a per-cell config axis, so mask-off and mask-on are timed in the same process. The "
         "reference CP baseline always applies its mask, so the **mask-on** column is the fair head-to-head.",
         "* **`K = N_token`** — operands are the square `(B, N_i_loc, N_j_loc, D)` token shards, so the back "
         "einsum contracts over `k = N_token` by construction; no decoupled K anywhere.",
         "* **`t(1)`** — measured by the same driver and timer as every other cell, via the production cp=1 "
         "path (`FusedTriMulCP` -> `_is_single_device` -> `trimul_autotuned`), i.e. the same public entry "
         "point the cp>=2 rows use.", "",
         "## Ideal laws — the two phases differ, and that matters", "",
         "* **Strong (Phase D)** — fixed global problem, growing cp. Ideal `t(cp) = t(1)/cp`; "
         "`speedup = t(1)/t(cp)`, `efficiency = t(1)/(cp*t(cp))`.",
         "* **Weak (Phase E, definition W2)** — constant per-device shard `N^2/cp = 2048^2`, i.e. "
         "`N = 2048*sqrt(cp)` snapped to `N%8==0`. TriMul compute is `O(N^3)` while the shard is `O(N^2)`, "
         "so holding the shard constant makes per-device FLOPs grow `~ N ~ sqrt(cp)`. **The ideal weak law "
         "is therefore `t = t(1)*sqrt(cp)`, NOT flat**, and the efficiency is `E(cp) = t(1)*sqrt(cp)/t(cp)`. "
         "A naive `t(1)/t(cp)` would read as catastrophic failure (it tends to `1/sqrt(cp)`) and is "
         "deliberately not a headline column. The `shard elems` column is the evidence the invariant holds.",
         ""]
    return "\n".join(L) + "\n" + _notes(notes_path) + "\n## Tables\n\n" + "\n".join(chunks)


def main():
    ap = argparse.ArgumentParser("aggregate_scaling")
    ap.add_argument("--results", required=True, help="SC_OUTBASE (the dir of per-cell result dirs)")
    ap.add_argument("--merge", default="", help="COMMA-SEP extra result dirs merged in (later wins); use "
                                                "for cells re-measured under a different launch condition")
    ap.add_argument("--md", default=None, help="write JUST the markdown tables here (default: stdout)")
    ap.add_argument("--doc", default=None,
                    help="write the FULL results page (env + method + ideal laws + tables), "
                         "e.g. docs/cp_trimul_strong_weak_scaling.md")
    ap.add_argument("--notes", default=None,
                    help="markdown file with the AUTHORED findings/limitations, spliced before the tables")
    ap.add_argument("--json", default=None, help="also dump the joined rows as JSON")
    a = ap.parse_args()

    rec = load_many([a.results] + [d.strip() for d in a.merge.split(",") if d.strip()])
    cells = SC.cells()
    chunks, blob = [], {"phase_d": {}, "phase_e": {}}
    for direction, suf in (("outgoing", "out"), ("incoming", "in")):
        for hm in (False, True):
            for sharding in ("1d", "2d"):
                for N in SC._D_N:
                    rows = table_d(rec, cells, sharding, N, suf, hm)
                    key = f"D/{sharding}/N{N}/{direction}/{_MASK[hm]}"
                    blob["phase_d"][key] = rows
                    chunks.append(_md_d(rows, f"Phase D — strong scaling · {sharding.upper()} · "
                                              f"N={N} · {direction} · {_MASK[hm]} · D={SC.D_FEAT}"))
                rows = table_e(rec, cells, sharding, suf, hm)
                key = f"E/{sharding}/{direction}/{_MASK[hm]}"
                blob["phase_e"][key] = rows
                chunks.append(_md_e(rows, f"Phase E — weak scaling (W2) · {sharding.upper()} · "
                                          f"{direction} · {_MASK[hm]} · D={SC.D_FEAT}"))
    md = "\n".join(chunks)
    if a.doc:
        with open(a.doc, "w") as f:
            f.write(build_doc(rec, cells, chunks, a.results, notes_path=a.notes))
        print(f"[aggregate] wrote {a.doc} (full results page)")
    if a.md:
        with open(a.md, "w") as f:
            f.write(md)
        print(f"[aggregate] wrote {a.md}  ({len(rec)} measured (target,config) records)")
    elif not a.doc:
        print(md)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(blob, f, indent=2)
    missing = sum(1 for rs in list(blob["phase_d"].values()) + list(blob["phase_e"].values())
                  for r in rs if r["status"] == "MISSING")
    print(f"[aggregate] measured records={len(rec)}  MISSING table rows={missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

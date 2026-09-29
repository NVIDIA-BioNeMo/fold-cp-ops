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

"""cutkeep_reduce.py -- reduce the CUT-VS-KEEP sweep JSONs to the cut/keep decision matrix.

Reads the per-N JSONs that back_a2a_store_bench.py writes (via --out-dir), laid out by
cutkeep_autotune.sbatch under:
    <OUTBASE>/mesh<cp0>x<cp1>_D<D>/roundpark_fp_D<D>_N<N>.json
Point --out-dir at <OUTBASE> (or any parent) -- it walks recursively for roundpark_fp_*.json.

Per-N JSON schema (back_a2a_store_bench.main writer):
  {"N", "cp0", "cp1", "D", "Dloc", "B", "cells": [ <cell>, ... ]}   # one cell per variant
Each AUTOTUNE cell (run_autotune_cell):
  {"variant", "autotune": true, "configs": [...], "n_valid", "n_correct", "status", "wall_s",
   "winner": {"cluster_n","completion","tile_m","tile_n","pingpong","time_ms","path","use_3wg"}
            | {"correct_configs": [...]}   # --check-only (no timing)
            | null}                         # every config ERROR/OOM/wrong
Also tolerates the legacy tile-SWEEP cell ({"sweep": true, "winner": {...}}) and the plain single cell
({"time_ms": ..., "status": ...}) -- so it reduces mixed out-dirs.

Outputs:
  1. Per (cp, N, Dloc): each variant's AUTOTUNED-BEST wall-time (the cell winner) + winning config, and the
     ratio to the per-cell best. ERROR/OOM/skip/missing -> "N/A" (never crashes).
  2. The CUT LIST: a variant NEVER within 5% (<=1.05x) of the per-cell best in ANY cell -> CUT; a variant
     that wins/ties >=1 cell -> KEEP. Per-variant: #cells-won, best (min) ratio achieved, verdict.
  3. Fused-vs-2kernel: per cell the best-fused / 2kernel ratio (<1 == a fused drain BEATS the 2-kernel).
Optional --csv writes the long-format (cp,N,Dloc,variant,best_ms,ratio,cfg,status) table.
"""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict

VARIANTS = ["coalesce", "differential", "strided_putwarp", "cluster", "cluster_multislot",
            "cluster_roundpark", "2kernel"]
FUSED = [v for v in VARIANTS if v != "2kernel"]  # the 6 fused drains (vs the 2-kernel baseline)
TIE = 1.05  # within 5% of the per-cell best == a "win/tie"


def _winner_time_cfg(cell):
    """(best_time_ms | None, cfg_str, status). The autotune/sweep winner IS the best valid config; a plain
    cell carries its own time_ms. None time -> the cell had no valid+timed config (ERROR/OOM/wrong/check)."""
    st = cell.get("status", "?")
    if cell.get("autotune") or cell.get("sweep"):
        w = cell.get("winner")
        if isinstance(w, dict) and w.get("time_ms") is not None:
            cfg = f"{w.get('tile_m')}x{w.get('tile_n')}" + ("pp" if w.get("pingpong") else "")
            if w.get("cluster_n") is not None:
                cfg += f"/cn{w.get('cluster_n')}"
            if w.get("completion"):
                cfg += f"/{w.get('completion')}"
            return float(w["time_ms"]), cfg, st
        # winner None, or {"correct_configs": ...} (--check-only, no timing)
        return None, "N/A", (st if st not in ("autotune_ok", "sweep_ok") else "no_valid_cfg")
    if cell.get("time_ms") is not None:  # plain single cell
        cfg = f"{cell.get('tile_m')}x{cell.get('tile_n')}"
        if cell.get("cluster_n") is not None:
            cfg += f"/cn{cell.get('cluster_n')}"
        if cell.get("completion"):
            cfg += f"/{cell.get('completion')}"
        return float(cell["time_ms"]), cfg, st
    return None, "N/A", st


def load(out_dir):
    """Walk out_dir for roundpark_fp_*.json. Return {(cp0,cp1,N,Dloc): {variant: (time|None, cfg, status)}}
    plus the file list. Legacy cnX cells fold into 'cluster' (keep the fastest). Never raises on bad files."""
    cells = defaultdict(dict)
    files = sorted(glob.glob(os.path.join(out_dir, "**", "roundpark_fp_*.json"), recursive=True))
    for fp in files:
        try:
            with open(fp) as f:
                d = json.load(f)
        except (OSError, ValueError) as e:
            print(f"[warn] unreadable, skipped: {fp} ({type(e).__name__}: {e})")
            continue
        cp0, cp1, N, Dloc = d.get("cp0"), d.get("cp1"), d.get("N"), d.get("Dloc")
        if None in (cp0, cp1, N, Dloc):
            print(f"[warn] missing cp0/cp1/N/Dloc, skipped: {fp}")
            continue
        key = (int(cp0), int(cp1), int(N), int(Dloc))
        for cell in d.get("cells", []):
            v = cell.get("variant")
            if v is None:
                continue
            vname = "cluster" if str(v).startswith("cn") else str(v)  # legacy cnX -> cluster
            if vname not in VARIANTS:
                continue  # unknown variant name -> ignore (keeps the matrix to the 7)
            t, cfg, st = _winner_time_cfg(cell)
            prev = cells[key].get(vname)
            # keep the FASTEST when a variant appears twice (legacy cnX collapse); a timed result beats N/A
            if prev is None or (t is not None and (prev[0] is None or t < prev[0])):
                cells[key][vname] = (t, cfg, st)
            elif prev[0] is None and t is None:
                cells[key][vname] = prev  # both N/A -> keep first
    return cells, files


def _fmt(x, nd=4):
    return "N/A" if x is None else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser(description="reduce the cut-vs-keep sweep JSONs to the cut/keep matrix")
    ap.add_argument("--out-dir", required=True, help="the sweep OUTBASE (walked recursively for "
                    "roundpark_fp_*.json); e.g. .../results/cutkeep")
    ap.add_argument("--csv", default=None, help="optional long-format CSV out path (cp,N,Dloc,variant,...)")
    ap.add_argument("--tie", type=float, default=TIE, help=f"win/tie threshold (default {TIE} = within 5%%).")
    args = ap.parse_args()

    cells, files = load(args.out_dir)
    print(f"[cutkeep-reduce] out_dir={args.out_dir}")
    print(f"[cutkeep-reduce] {len(files)} JSON file(s) -> {len(cells)} (cp,N,Dloc) cell(s)\n")
    if not cells:
        print("no cells found (sweep not started / wrong --out-dir). Nothing to reduce.")
        return 0

    # ---- 1+2: per-cell best + ratio matrix (long format) ----
    wins = {v: 0 for v in VARIANTS}         # #cells where variant is within TIE of the per-cell best
    best_ratio = {v: None for v in VARIANTS}  # min ratio-to-best a variant ever achieved (lower = better)
    present = {v: 0 for v in VARIANTS}      # #cells where variant had a valid time
    rows = []                               # long-format rows for the CSV
    fused_beats = []                        # (cell, fused_variant, ratio<1) where a fused drain beats 2kernel

    hdr = f"{'cp':>5} {'N':>6} {'Dl':>3} | " + " ".join(f"{v[:11]:>13}" for v in VARIANTS) + " | best"
    print("PER-CELL AUTOTUNED-BEST wall-time (ms) + [ratio-to-cell-best]:")
    print(hdr)
    print("-" * len(hdr))
    for key in sorted(cells):
        cp0, cp1, N, Dloc = key
        row = cells[key]
        times = {v: row.get(v, (None, "N/A", "missing"))[0] for v in VARIANTS}
        valid = {v: t for v, t in times.items() if t is not None}
        cell_best = min(valid.values()) if valid else None
        best_v = min(valid, key=valid.get) if valid else "-"
        line = f"{cp0}x{cp1:>2} {N:>6} {Dloc:>3} | "
        for v in VARIANTS:
            t = times[v]
            if t is None:
                line += f"{'N/A':>13} "
                continue
            present[v] += 1
            r = t / cell_best if cell_best else None
            if r is not None:
                if best_ratio[v] is None or r < best_ratio[v]:
                    best_ratio[v] = r
                if r <= args.tie:
                    wins[v] += 1
            line += f"{t:7.3f}[{r:4.2f}] "
            rows.append({"cp": f"{cp0}x{cp1}", "N": N, "Dloc": Dloc, "variant": v, "best_ms": t,
                         "ratio_to_best": round(r, 4) if r else None, "cfg": row.get(v, (None, "N/A"))[1],
                         "status": row.get(v, (None, "N/A", "missing"))[2]})
        line += f"| {best_v} ({_fmt(cell_best, 3)})"
        print(line)
        # ---- 4: fused-vs-2kernel for this cell ----
        k2 = times.get("2kernel")
        fused_valid = {v: times[v] for v in FUSED if times[v] is not None}
        if k2 is not None and fused_valid:
            fb_v = min(fused_valid, key=fused_valid.get)
            fb_ratio = fused_valid[fb_v] / k2
            if fb_ratio < 1.0:
                fused_beats.append((f"{cp0}x{cp1}", N, Dloc, fb_v, fb_ratio))

    # ---- 3: CUT LIST ----
    print("\nCUT LIST (win/tie = within {:.0%} of the per-cell best in >=1 cell):".format(args.tie - 1))
    print(f"{'variant':>17} | {'cells_present':>13} | {'cells_won':>9} | {'best_ratio':>10} | verdict")
    print("-" * 72)
    cut, keep = [], []
    for v in VARIANTS:
        won, br, pr = wins[v], best_ratio[v], present[v]
        verdict = "KEEP" if won > 0 else ("CUT" if pr > 0 else "NO-DATA")
        (keep if won > 0 else (cut if pr > 0 else keep)).append(v)
        print(f"{v:>17} | {pr:>13} | {won:>9} | {_fmt(br, 3):>10} | {verdict}")
    print(f"\n  KEEP: {[v for v in VARIANTS if wins[v] > 0]}")
    print(f"  CUT : {[v for v in VARIANTS if wins[v] == 0 and present[v] > 0]}")
    nodata = [v for v in VARIANTS if present[v] == 0]
    if nodata:
        print(f"  NO-DATA (never a valid+timed cell -> cannot classify): {nodata}")

    # ---- 4: fused-vs-2kernel headline ----
    print("\nFUSED-vs-2KERNEL (best fused / 2kernel; <1 == a fused drain BEATS the 2-kernel baseline):")
    if not fused_beats:
        print("  no cell where any fused drain beats 2kernel (2-kernel wins everywhere it has data).")
    else:
        for (cp, N, Dloc, v, r) in fused_beats:
            print(f"  cp={cp} N={N} Dloc={Dloc}: {v} = {r:.3f}x 2kernel  (fused WINS by {(1 - r) * 100:.1f}%)")

    if args.csv and rows:
        with open(args.csv, "w", newline="") as f:
            wtr = csv.DictWriter(f, fieldnames=["cp", "N", "Dloc", "variant", "best_ms", "ratio_to_best",
                                                "cfg", "status"])
            wtr.writeheader()
            wtr.writerows(rows)
        print(f"\n[cutkeep-reduce] wrote long-format CSV ({len(rows)} rows) -> {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

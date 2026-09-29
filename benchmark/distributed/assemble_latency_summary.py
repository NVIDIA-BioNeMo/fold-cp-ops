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


"""Assemble a G7 TriMul grid's console output into the archive's 13-column latency summary.csv.

VENUE-NEUTRAL. Nothing here knows about a cluster: it reads the benchmark driver's stdout and writes
a CSV whose `venue` column is whatever `--venue` says. Proven on two venues -- the archive grid and a
re-run on a different cluster (`--venue <other>`) -- with no change. (It was once named for the first
caller rather than for the tool, and a name that implies a tool is single-venue deters the reuse it
was already capable of.)

Why this is a separate assembler
--------------------------------
`assemble_all.py` and `assemble_2d.py` exist and are the right tools for what they do -- they join
the nsys-export sqlite (GPU/NIC metric rows) against the console `[LAT]` lines and emit a
metrics-oriented CSV.  That is a DIFFERENT schema from the archived
`profiling/distributed/trimul/dlc_composite_k_dtensor_cp_le16/summary.csv`, which is a flat
latency table:

    venue,mesh,cp,N,D,direction,fused_ms,dtensor_baseline_ring_reducescatter_ms,speedup,baseline,api,mask,notes

The whole point of the reproduction is a line-comparable diff against that archive, so this script
emits that header byte-for-byte and nothing else.  Inventing a new schema would destroy the only
comparison that matters.

What it reads
-------------
The driver's stdout, one console log per cell:

    [LAT] D=256 cp=2 N=2048 dir=outgoing fused_ms=10.0699 dtensor_baseline_ring_reducescatter_ms=107.4341 speedup=10.669
    [LAT] D=256 cp=2x2 N=2048 dir=outgoing fused_ms=8.3163 dtensor_ms=nan speedup=nan
    [NA]  D=256 cp=8 N=8192 dir=outgoing reason=<why this N was rejected>
    [done] D=256 cp=2 a2a_fused_dtensor_api=True mask=True ... baseline=dtensor_baseline_ring_reducescatter ...

`cp=` carries the MESH, not just the width: `2` is 1-D cp2; `2x2` is the square 2-D mesh at cp=4.
A `baseline=none` cell prints `dtensor_ms=nan speedup=nan` (the driver labels the absent baseline
column `dtensor`, and `bmed` is nan because `base_fn is None`).  Those become EMPTY cells in the
CSV, matching the archive -- they are correct results, not missing data: the reference CP baseline OOMs at
large N, and it auto-skips a non-square 2-D mesh.

The non-silence rule
--------------------
Acceptance §8.1: "a cell that did not run is never a silent omission."  So this script does NOT
emit only what it parsed.  It enumerates all 40 expected rows, fills in what the logs contain, and
emits every unfilled row with `notes` naming why it is empty.  It then prints the
enumerated/parsed/missing accounting and, under --strict, exits non-zero if anything is missing --
so a truncated grid cannot be read as a complete one.
"""

import argparse
import csv
import pathlib
import re
import sys

HEADER = [
    "venue",
    "mesh",
    "cp",
    "N",
    "D",
    "direction",
    "fused_ms",
    "dtensor_baseline_ring_reducescatter_ms",
    "speedup",
    "baseline",
    "api",
    "mask",
    "notes",
]

# The archive's 40 rows, in the archive's order, each with the archive's `notes` string.  Ordered so
# a `diff` against the donor CSV lines up row-for-row; keyed (mesh, cp, N, direction).
GRID = [
    ("1D", 2, 2048, "outgoing", "FusedTriMulCP-vs-dtensor_baseline_ring_reducescatter mask-on; GPU-metrics; both-dir"),
    ("1D", 2, 2048, "incoming", "FusedTriMulCP-vs-dtensor_baseline_ring_reducescatter mask-on; GPU-metrics; both-dir"),
    (
        "1D",
        2,
        4096,
        "outgoing",
        "fused-only (baseline OOM at cp2 large-N); per-direction split (2-module footprint)",
    ),
    ("1D", 2, 4096, "incoming", "fused-only (baseline OOM at cp2 large-N); per-direction split"),
    ("1D", 4, 2048, "outgoing", "FusedTriMulCP-vs-dtensor_baseline_ring_reducescatter mask-on; GPU-metrics"),
    ("1D", 4, 2048, "incoming", "FusedTriMulCP-vs-dtensor_baseline_ring_reducescatter mask-on; GPU-metrics"),
    ("1D", 4, 4096, "outgoing", "fused-only (baseline OOM at cp4 large-N)"),
    ("1D", 4, 4096, "incoming", "fused-only (baseline OOM at cp4 large-N)"),
    ("1D", 8, 2048, "outgoing", "FusedTriMulCP-vs-dtensor_baseline_ring_reducescatter mask-on; GPU-metrics"),
    ("1D", 8, 2048, "incoming", "FusedTriMulCP-vs-dtensor_baseline_ring_reducescatter mask-on; GPU-metrics"),
    ("1D", 8, 4096, "outgoing", "FusedTriMulCP-vs-dtensor_baseline_ring_reducescatter mask-on; GPU-metrics"),
    ("1D", 8, 4096, "incoming", "FusedTriMulCP-vs-dtensor_baseline_ring_reducescatter mask-on; GPU-metrics"),
    ("1D", 8, 4608, "outgoing", "FusedTriMulCP-vs-dtensor_baseline_ring_reducescatter mask-on; GPU-metrics"),
    ("1D", 8, 4608, "incoming", "FusedTriMulCP-vs-dtensor_baseline_ring_reducescatter mask-on; GPU-metrics"),
    ("1D", 8, 8192, "outgoing", "fused-only (baseline OOM at cp8 N8192); per-direction split"),
    ("1D", 8, 8192, "incoming", "fused-only (baseline OOM at cp8 N8192); per-direction split"),
    ("1D", 16, 2048, "outgoing", "FusedTriMulCP-vs-dtensor_baseline_ring_reducescatter mask-on; 2-node; NIC-metrics"),
    ("1D", 16, 2048, "incoming", "FusedTriMulCP-vs-dtensor_baseline_ring_reducescatter mask-on; 2-node; NIC-metrics"),
    ("1D", 16, 4096, "outgoing", "FusedTriMulCP-vs-dtensor_baseline_ring_reducescatter mask-on; 2-node; NIC-metrics"),
    ("1D", 16, 4096, "incoming", "FusedTriMulCP-vs-dtensor_baseline_ring_reducescatter mask-on; 2-node; NIC-metrics"),
    ("1D", 16, 5120, "outgoing", "FusedTriMulCP-vs-dtensor_baseline_ring_reducescatter mask-on; 2-node; NIC-metrics"),
    ("1D", 16, 5120, "incoming", "FusedTriMulCP-vs-dtensor_baseline_ring_reducescatter mask-on; 2-node; NIC-metrics"),
    ("1D", 16, 12288, "outgoing", "fused-only large-N; per-direction split; 2-node; NIC-metrics"),
    ("1D", 16, 12288, "incoming", "fused-only large-N; per-direction split; 2-node; NIC-metrics"),
    (
        "2D-2x2",
        4,
        2048,
        "outgoing",
        "FusedTriMulCP-vs-dtensor_baseline_ring_reducescatter mask-on; square 2-D; per-direction split",
    ),
    (
        "2D-2x2",
        4,
        2048,
        "incoming",
        "FusedTriMulCP-vs-dtensor_baseline_ring_reducescatter mask-on; square 2-D; per-direction split",
    ),
    (
        "2D-2x2",
        4,
        4096,
        "outgoing",
        "FusedTriMulCP-vs-dtensor_baseline_ring_reducescatter mask-on; square 2-D; per-direction split",
    ),
    (
        "2D-2x2",
        4,
        4096,
        "incoming",
        "FusedTriMulCP-vs-dtensor_baseline_ring_reducescatter mask-on; square 2-D; per-direction split",
    ),
    ("2D-2x4", 8, 2048, "outgoing", "fused-only (dtensor_baseline_ring_reducescatter non-square 2-D auto-skip); 2-node"),
    ("2D-2x4", 8, 2048, "incoming", "fused-only (dtensor_baseline_ring_reducescatter non-square 2-D auto-skip); 2-node"),
    ("2D-2x4", 8, 4096, "outgoing", "fused-only (dtensor_baseline_ring_reducescatter non-square 2-D auto-skip); 2-node"),
    ("2D-2x4", 8, 4096, "incoming", "fused-only (dtensor_baseline_ring_reducescatter non-square 2-D auto-skip); 2-node"),
    (
        "2D-4x4",
        16,
        2048,
        "outgoing",
        "FusedTriMulCP-vs-dtensor_baseline_ring_reducescatter mask-on; square 2-D; 2-node; NIC-metrics",
    ),
    (
        "2D-4x4",
        16,
        2048,
        "incoming",
        "FusedTriMulCP-vs-dtensor_baseline_ring_reducescatter mask-on; square 2-D; 2-node; NIC-metrics",
    ),
    (
        "2D-4x4",
        16,
        4096,
        "outgoing",
        "FusedTriMulCP-vs-dtensor_baseline_ring_reducescatter mask-on; square 2-D; 2-node; NIC-metrics",
    ),
    (
        "2D-4x4",
        16,
        4096,
        "incoming",
        "FusedTriMulCP-vs-dtensor_baseline_ring_reducescatter mask-on; square 2-D; 2-node; NIC-metrics",
    ),
    (
        "2D-2x8",
        16,
        2048,
        "outgoing",
        "fused-only (dtensor_baseline_ring_reducescatter non-square 2-D auto-skip); 2-node; NIC-metrics",
    ),
    (
        "2D-2x8",
        16,
        2048,
        "incoming",
        "fused-only (dtensor_baseline_ring_reducescatter non-square 2-D auto-skip); 2-node; NIC-metrics",
    ),
    (
        "2D-2x8",
        16,
        4096,
        "outgoing",
        "fused-only (dtensor_baseline_ring_reducescatter non-square 2-D auto-skip); 2-node; NIC-metrics",
    ),
    (
        "2D-2x8",
        16,
        4096,
        "incoming",
        "fused-only (dtensor_baseline_ring_reducescatter non-square 2-D auto-skip); 2-node; NIC-metrics",
    ),
]

_LAT = re.compile(
    r"\[LAT\]\s+D=(?P<D>\d+)\s+cp=(?P<cp>[\dx]+)\s+N=(?P<N>\d+)\s+dir=(?P<dir>\w+)\s+"
    r"fused_ms=(?P<fused>[\w.+-]+)\s+(?P<blabel>\w+)_ms=(?P<base>[\w.+-]+)\s+speedup=(?P<sp>[\w.+-]+)"
)
_NA = re.compile(
    r"\[NA\]\s+D=(?P<D>\d+)\s+cp=(?P<cp>[\dx]+)\s+N=(?P<N>\d+)\s+dir=(?P<dir>\w+)\s+"
    r"reason=(?P<reason>.*)"
)
_DONE = re.compile(
    r"\[done\]\s+D=(?P<D>\d+)\s+cp=(?P<cp>[\dx]+)\s+a2a_fused_dtensor_api=(?P<api>\w+)\s+"
    r"mask=(?P<mask>\w+).*?baseline=(?P<baseline>\w+)"
)


def mesh_and_cp(cp_label):
    """`cp=` in the driver's output is the MESH label, not the width.

    `"8"` -> ("1D", 8);  `"2x4"` -> ("2D-2x4", 8).  Collapsing these would merge the 1-D cp16 cell
    with the 2-D 4x4 cell -- both are cp=16 -- and silently overwrite one with the other."""
    if "x" in cp_label:
        a, b = (int(v) for v in cp_label.split("x"))
        return f"2D-{a}x{b}", a * b
    return "1D", int(cp_label)


def _num(tok):
    """Driver prints `nan` for an absent baseline; the archive writes an EMPTY cell for it."""
    return "" if tok.lower() in ("nan", "-nan", "inf", "-inf") else tok


def parse_dir(console_dir):
    """Return (lat, na, meta) keyed (mesh, cp, N, direction) / (mesh, cp)."""
    lat, na, meta = {}, {}, {}
    logs = sorted(p for p in pathlib.Path(console_dir).rglob("*.log"))
    for p in logs:
        text = p.read_text(errors="replace")
        for m in _DONE.finditer(text):
            mesh, cp = mesh_and_cp(m.group("cp"))
            meta[(mesh, cp)] = {
                "api": "a2a-fused-dtensor" if m.group("api") == "True" else "a2a-fused-raw",
                "mask": "on" if m.group("mask") == "True" else "off",
                "baseline": m.group("baseline"),
                "log": p.name,
            }
        for m in _LAT.finditer(text):
            mesh, cp = mesh_and_cp(m.group("cp"))
            base = _num(m.group("base"))
            lat[(mesh, cp, int(m.group("N")), m.group("dir"))] = {
                "D": m.group("D"),
                "fused_ms": _num(m.group("fused")),
                "dtensor_baseline_ring_reducescatter_ms": base,
                "speedup": _num(m.group("sp")),
                # A cell whose baseline column is empty ran fused-only, whatever it was invoked as.
                "baseline": m.group("blabel") if base else "none",
                "log": p.name,
            }
        for m in _NA.finditer(text):
            mesh, cp = mesh_and_cp(m.group("cp"))
            na[(mesh, cp, int(m.group("N")), m.group("dir"))] = m.group("reason").strip()
    return lat, na, meta, logs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--console-dir", required=True, help="dir of console_*.log from the grid run")
    ap.add_argument("--out", required=True, help="path to write summary.csv")
    # REQUIRED, deliberately -- this used to carry a default venue name. A default venue is a silent
    # mislabel waiting to happen: run the grid somewhere else, forget the flag, and the CSV claims a
    # venue it never ran on. `compare_summary.py` keys its cross-venue guard off THIS column, so a
    # mislabel does not merely annotate wrongly -- it switches the comparator into same-venue mode and
    # diffs absolute milliseconds across two different fabrics, which is the exact category error the
    # guard exists to prevent, reported with full confidence. A results file must carry its own true
    # configuration; there is no safe default for where something ran.
    ap.add_argument(
        "--venue",
        required=True,
        help="MUST match where it ran (a short label for the cluster). Keys the cross-venue "
        "guard in compare_summary.py; a wrong value silently enables an invalid "
        "absolute-ms comparison (reproduce_bench.md §2).",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="exit 1 unless every enumerated row was filled from a [LAT] line",
    )
    a = ap.parse_args()

    lat, na, meta, logs = parse_dir(a.console_dir)
    rows, filled, missing = [], [], []
    for mesh, cp, N, direction, notes in GRID:
        key = (mesh, cp, N, direction)
        hit = lat.get(key)
        m = meta.get((mesh, cp), {})
        if hit:
            filled.append(key)
            rows.append(
                {
                    "venue": a.venue,
                    "mesh": mesh,
                    "cp": cp,
                    "N": N,
                    "D": hit["D"],
                    "direction": direction,
                    "fused_ms": hit["fused_ms"],
                    "dtensor_baseline_ring_reducescatter_ms": hit["dtensor_baseline_ring_reducescatter_ms"],
                    "speedup": hit["speedup"],
                    "baseline": hit["baseline"],
                    "api": m.get("api", "a2a-fused-dtensor"),
                    "mask": m.get("mask", "on"),
                    "notes": notes,
                }
            )
            continue
        missing.append(key)
        why = na.get(key) or (
            "no [LAT] line in any console log -- cell did not run, crashed, or was never launched"
        )
        rows.append(
            {
                "venue": a.venue,
                "mesh": mesh,
                "cp": cp,
                "N": N,
                "D": 256,
                "direction": direction,
                "fused_ms": "",
                "dtensor_baseline_ring_reducescatter_ms": "",
                "speedup": "",
                "baseline": "",
                "api": "",
                "mask": "",
                "notes": f"DID NOT RUN: {why}",
            }
        )

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=HEADER, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    # ---- The non-silence accounting.  Printed ALWAYS, not just on failure: a grid that quietly
    # produced 31 of 40 rows and a clean exit is the exact failure this is here to make impossible.
    print(f"console logs read : {len(logs)}")
    print(f"rows enumerated   : {len(GRID)}")
    print(f"rows filled       : {len(filled)}")
    print(f"rows missing      : {len(missing)}")
    print(f"wrote             : {out}  ({len(rows)} data rows + header)")
    if missing:
        print("\nCELLS WITH NO [LAT] LINE -- each is written to the CSV with a DID NOT RUN note:")
        for mesh, cp, N, direction in missing:
            print(f"    {mesh:8s} cp={cp:<3d} N={N:<6d} {direction}")
    if a.strict and missing:
        print(
            f"\n*** STRICT: {len(missing)} of {len(GRID)} enumerated rows were never measured ***"
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

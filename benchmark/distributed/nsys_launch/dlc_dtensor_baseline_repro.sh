#!/usr/bin/env bash
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

# Reproduce the archived DTensor-baseline latencies in
# profiling/distributed/trimul/dlc_composite_k_dtensor_cp_le16/summary.csv.
#
# WHY THIS EXISTS. The baseline is the denominator of every speedup this repo publishes, and it was
# renamed twice in one day (twice in one day, ending at dtensor_baseline_ring_reducescatter). A rename is only
# safe if the number it labels does not move, and the only way to know that is to re-measure. This
# script is that check, kept so the next renamer can run it instead of trusting a diff.
#
# It RESOLVES ITS BASELINE FROM summary.csv, not from a table in this header. The header used to
# carry the archive's numbers, and when the archive at that path was replaced they silently became
# the PREVIOUS archive's -- 0.07-0.76% off, close enough to read as agreement.
#
# LAST RUN 2026-08-28, one H100 SXM5 node, one isolated srun per cell, 8/8 cells. Against the archive
# then in place: fused worst 1.44% (median -1.03%, signs 2+/6-), baseline worst 1.37% (median -1.18%,
# 8/8 NEGATIVE), paired speedup median -0.10%. A uniform baseline sign is a venue shift, and it
# cancels in the ratio -- which is what the archive's claims rest on. These cells are cp<=8, single
# node, so they are NVLink-only and unaffected by the IBGDA CPU-fallback noted in PROVENANCE.md.
#
# THREE THINGS THAT COST ME FOUR FAILED ATTEMPTS, all avoidable:
#   1. The driver reads os.environ["WORLD_SIZE"] -- it needs TORCHRUN, not bare `srun --ntasks=N`.
#   2. Run the tree you think you are running. A worktree checked out from an uncommitted branch
#      silently runs the previous commit.
#   3. `--gpus-per-task=1` with >1 task per node hands each task ONE gpu; omit it (see CLAUDE.md).
#
# Usage:  JOBID=<slurm jobid> NODE=<node> bash dlc_dtensor_baseline_repro.sh
set -u
JOBID=${JOBID:?set JOBID to a live allocation}
NODE=${NODE:?set NODE to one allocated node}
W=${W:-$(git rev-parse --show-toplevel)}
PY=${PY:-python}
OUT=${OUT:-$PWD/repro_out}; mkdir -p "$OUT"
BASE=dtensor_baseline_ring_reducescatter
cd "$W"; PORT=${PORT:-29750}; REQ=0; OK=0
for spec in "2 2048" "4 2048" "8 2048" "8 4096"; do
  set -- $spec; CP=$1; N=$2
  for DIR in outgoing incoming; do
    REQ=$((REQ+1)); PORT=$((PORT+1)); LOG="$OUT/cp${CP}_N${N}_${DIR}.log"
    timeout 1500 srun --jobid="$JOBID" --overlap -N1 --ntasks=1 -w "$NODE" \
      --export=ALL,PYTHONPATH=$W,CPO_CACHE_ENABLED=1,PYTHONUNBUFFERED=1 \
      "$PY" -m torch.distributed.run --nproc_per_node=$CP --master_port=$PORT \
      benchmark/distributed/nsys_trimul_dtensor.py --D 256 --cp $CP --Ns $N \
      --baseline $BASE --direction $DIR --a2a-fused-dtensor-api --mask --dynamic --warmup 8 --iters 4 \
      > "$LOG" 2>&1 < /dev/null
    RC=$?; LAT=$(grep -oE "\[LAT\].*" "$LOG" | tail -1)
    if [ -n "$LAT" ]; then OK=$((OK+1)); echo "CELL cp=$CP N=$N $DIR rc=$RC  $LAT"
    else echo "CELL cp=$CP N=$N $DIR rc=$RC  NO-LAT $(grep -oiE 'OutOfMemory|CUDA error|Traceback' "$LOG" | head -1)"; fi
  done
done
# An explicit denominator: a truncated sweep must never read as a clean one.
echo "measured=$OK requested=$REQ"; [ "$OK" = "$REQ" ] || echo "INCOMPLETE"

# Resolve every cell against the archive. A reproduction compares to the file, never to prose.
"$PY" - "$OUT" <<'PYCMP'
import csv, glob, os, re, statistics, sys
arch = {(r["cp"], r["N"], r["direction"]): r for r in csv.DictReader(
    open("profiling/distributed/trimul/dlc_composite_k_dtensor_cp_le16/summary.csv")) if r["mesh"] == "1D"}
rows = []
for log in sorted(glob.glob(os.path.join(sys.argv[1], "cp*_N*_*.log"))):
    m = re.search(r"\[LAT\] D=\d+ cp=(\S+) N=(\d+) dir=(\w+) fused_ms=([\d.]+) \w+_ms=([\d.]+)",
                  open(log, errors="replace").read())
    if not m:
        continue
    cp, N, d, f, b = m.group(1), m.group(2), m.group(3), float(m.group(4)), float(m.group(5))
    a = arch.get((cp, N, d))
    if a is None:
        print(f"  cp={cp} N={N} {d}: NOT IN ARCHIVE"); continue
    fa, ba = float(a["fused_ms"]), float(a["dtensor_baseline_ring_reducescatter_ms"])
    rows.append((100 * (f / fa - 1), 100 * (b / ba - 1)))
    print(f"  cp={cp:>2} N={N} {d:<9} fused {fa:8.4f}->{f:<8.4f}{rows[-1][0]:+6.2f}%   "
          f"base {ba:9.4f}->{b:<9.4f}{rows[-1][1]:+6.2f}%")
for i, lbl in ((0, "fused"), (1, "baseline")):
    if not rows:
        break
    d = [r[i] for r in rows]
    print(f"  {lbl:<9} worst {max(abs(x) for x in d):5.2f}%  median {statistics.median(d):+5.2f}%  "
          f"signs +{sum(x > 0 for x in d)}/-{sum(x < 0 for x in d)}")
print(f"  compared {len(rows)} cells against {len(arch)} archive 1-D rows")
PYCMP

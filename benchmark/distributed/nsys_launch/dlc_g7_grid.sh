#!/bin/bash
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


# Drive the FULL TriMul nsys grid: ONE ISOLATED `srun` PER CELL.
#
# Per-cell isolation is a HARD RULE, not a style preference. A single hard fault in ANY cell -- a
# CUDA "unspecified launch failure", an OOM, a hang, or an nvshmem re-init failure on the 2nd+ cell
# -- corrupts the shared CUDA/NCCL context and cascades, silently truncating every remaining cell.
# So each cell is its own `srun` step under its own `timeout`; a dead cell is logged with its rc and
# the loop advances.
#
# NON-SILENCE (acceptance §8.1): a cell that did not run is never a silent omission. Every cell
# appends exactly one line to status.tsv -- tag, rc, and whether a [LAT] line and a .nsys-rep
# appeared -- and the script asserts `launched == enumerated` at the end.
#
# Usage (from a login node, with a RUNNING allocation):
#   JID=<jobid> bash benchmark/distributed/nsys_launch/dlc_g7_grid.sh [tag ...]
# With no tag arguments it runs the SMOKES FIRST, then the 16 grid cells. Naming tags runs only
# those (used to re-run a single cell isolated, which acceptance §8.4 requires before calling any
# out-of-tolerance cell a regression).
set -u
JID=${JID:?set JID to the RUNNING salloc job id}
# Scratch base. Overridable: hardcoding an absolute base violates the repo's path-portability rule
# (the same tree lives under a different base on a different machine) and is the ONE thing in this
# script that is genuinely site-specific rather than merely venue-flavoured.
MD=${MD:?set MD to a writable scratch base for reports and logs}
# Overridable, and PASSED THROUGH to the cell script below -- two checkouts coexist on this filesystem
# and measuring the wrong one is an invalid result that produces no error.
REPO=${REPO:-$MD/mycodes/fold-cp-ops}
OUT=${OUT:-$MD/trimul_dlc/g7_mirror}
REPS=$OUT/reps
CONS=$OUT/console
STATUS=$OUT/status.tsv
CELL_TIMEOUT=${CELL_TIMEOUT:-1800}
mkdir -p "$REPS" "$CONS"

# ---- The grid. Fields: TAG | NODES | NPROC | MESHARG(CP=n or MESH2D=a,b) | NS | DIRECTION | BASELINE
# Reconstructed from the archive's own .nsys-rep tags + summary.csv. The two SMOKES lead: a harness
# self-test that costs ~5 min has repeatedly saved multi-hour allocations on this workflow.
#
# NOTE ON THE COUNT: reproduce_bench.md prose says "20 cells", but its own §5 table and the archive
# both enumerate 18 distinct tags -> 23 .nsys-rep files (13 one-node tags x1 + 5 two-node tags x2).
# 18/23 is what the archive actually contains, so that is what is reproduced here.
CELLS=(
  "g7_smoke_cp2_N2048|1|2|CP=2|2048|both|dtensor_baseline_ring_reducescatter"
  "g7_smoke_cp8|1|8|CP=8|2048|both|dtensor_baseline_ring_reducescatter"
  "g7_cp2_dtensor_baseline_ring_reducescatter|1|2|CP=2|2048|both|dtensor_baseline_ring_reducescatter"
  "g7_cp2_large_none_out|1|2|CP=2|4096|outgoing|none"
  "g7_cp2_large_none_in|1|2|CP=2|4096|incoming|none"
  "g7_cp4_dtensor_baseline_ring_reducescatter|1|4|CP=4|2048|both|dtensor_baseline_ring_reducescatter"
  "g7_cp4_large_none|1|4|CP=4|4096|both|none"
  "g7_cp8_dtensor_baseline_ring_reducescatter|1|8|CP=8|2048,4096,4608|both|dtensor_baseline_ring_reducescatter"
  "g7_cp8_large_none_out|1|8|CP=8|8192|outgoing|none"
  "g7_cp8_large_none_in|1|8|CP=8|8192|incoming|none"
  "g7_2x2_dtensor_baseline_ring_reducescatter_out|1|4|MESH2D=2,2|2048,4096|outgoing|dtensor_baseline_ring_reducescatter"
  "g7_2x2_dtensor_baseline_ring_reducescatter_in|1|4|MESH2D=2,2|2048,4096|incoming|dtensor_baseline_ring_reducescatter"
  "g7_2x4_none|1|8|MESH2D=2,4|2048,4096|both|none"
  "g7_cp16_dtensor_baseline_ring_reducescatter|2|8|CP=16|2048,4096,5120|both|dtensor_baseline_ring_reducescatter"
  "g7_cp16_large_none_out|2|8|CP=16|12288|outgoing|none"
  "g7_cp16_large_none_in|2|8|CP=16|12288|incoming|none"
  "g7_4x4_dtensor_baseline_ring_reducescatter|2|8|MESH2D=4,4|2048,4096|both|dtensor_baseline_ring_reducescatter"
  "g7_2x8_none|2|8|MESH2D=2,8|2048,4096|both|none"
)

WANT=("$@")
selected() { [ ${#WANT[@]} -eq 0 ] && return 0; for w in "${WANT[@]}"; do [ "$w" = "$1" ] && return 0; done; return 1; }

[ -f "$STATUS" ] || printf 'tag\trc\tnodes\tnproc\tlat_lines\treps\tseconds\n' > "$STATUS"
enumerated=0; launched=0
echo "[grid] JID=$JID OUT=$OUT cells=${#CELLS[@]} cell_timeout=${CELL_TIMEOUT}s"
# Record WHICH TREE ran, in the run's own log. A results directory that cannot name the checkout and
# revision that produced it is not reproducible evidence, and with two coexisting checkouts the
# question "which one was this?" cannot be answered after the fact.
echo "[grid] REPO=$REPO rev=$(cat "$REPO/REVISION" 2>/dev/null || git -C "$REPO" rev-parse HEAD 2>/dev/null || echo UNKNOWN)"

for spec in "${CELLS[@]}"; do
  IFS='|' read -r TAG NODES NPROC MESHSPEC NS DIRECTION BASELINE <<< "$spec"
  selected "$TAG" || continue
  enumerated=$((enumerated + 1))
  MESHENV=(); case "$MESHSPEC" in
    CP=*)     MESHENV=(CP="${MESHSPEC#CP=}") ;;
    MESH2D=*) MESHENV=(MESH2D="${MESHSPEC#MESH2D=}") ;;
  esac
  LOG="$CONS/console_${TAG}.log"
  echo "[grid] >>> $TAG  nodes=$NODES nproc=$NPROC $MESHSPEC Ns=$NS dir=$DIRECTION base=$BASELINE"
  t0=$SECONDS
  # `timeout` on EVERY command (HARD RULE). The cell script already bounds nsys at 1700s; this outer
  # bound catches an srun that never reaches it (a wedged node, a bootstrap hang).
  #
  # --overlap, NEVER --exclusive. On a STEP inside an existing allocation `--exclusive` does NOT mean
  # "the whole node" -- it means "exclusive STEP resources", sized by the TASK COUNT. With
  # --ntasks-per-node=1 Slurm hands the step ONE task's worth of CPU: MEASURED 2 threads on a
  # 224-thread DGX-H100 (SLURM_CPUS_ON_NODE=2, nproc=2), vs 224 under --overlap. Nothing errors.
  # nsys is CPU-hungry (sampling + event processing + NVTX + metrics, multiplied by every rank under
  # one profiler), so a starved core count silently inflates the measurement: this cell grid measured
  # cp8 N2048 at 45.76 ms on 2 CPUs vs 4.63 ms on 224 (~10x), and the inflation SCALED with rank count
  # (cp2 ~2-4x, cp8 9.6x) because more ranks contend for the same cores.
  #
  # It is a trap because it does not look like a CPU problem: the MIN still matches the reference, the
  # MEDIAN inflates, sigma explodes (0.07 -> 60.58 ms), and the comm-heavy fused path is hit harder
  # than the plain baseline -- i.e. an exact impostor of distributed straggler / host-arrival skew.
  # Rule the core count out FIRST; it is one command:
  #   echo "cpus=$SLURM_CPUS_ON_NODE bind=${SLURM_CPU_BIND_LIST:-none} nproc=$(nproc)"
  # Any .nsys-rep can also be audited after the fact: TARGET_INFO_SYSTEM_ENV name='DeviceEnvironment'
  # embeds the full env including SLURM_CPU_BIND / SLURM_CPUS_ON_NODE.
  timeout "$CELL_TIMEOUT" srun --jobid="$JID" -N"$NODES" --ntasks-per-node=1 --overlap \
    env D=256 "${MESHENV[@]}" NS="$NS" NPROC="$NPROC" DIRECTION="$DIRECTION" \
        BASELINE="$BASELINE" TAG="$TAG" OUTDIR="$REPS" REPO="$REPO" \
    bash "$REPO/benchmark/distributed/nsys_launch/dlc_trimul_cell.sh" > "$LOG" 2>&1
  rc=$?
  dt=$((SECONDS - t0))
  launched=$((launched + 1))
  # Evidence, not vibes: count the [LAT] lines and the .nsys-rep files this tag actually produced.
  # UNANCHORED on purpose. nsys writes its progress bar with carriage returns and no trailing
  # newline, so the driver's next stdout write lands on the SAME physical line -- a real cell's
  # `[LAT]` then reads as `[1/1] [45%...]...[LAT] D=256 ...` and an anchored `^\[LAT\]` counts ZERO.
  # That misreports a cell that measured fine as one that produced nothing. (No `|| echo 0` either:
  # grep exits 1 on no-match, so `|| echo 0` appended a SECOND 0 and wrote "0\n0" into the TSV.)
  nlat=$(grep -o '\[LAT\] D=' "$LOG" 2>/dev/null | wc -l)
  nrep=$(ls -1 "$REPS/${TAG}_node"*.nsys-rep 2>/dev/null | wc -l)
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$TAG" "$rc" "$NODES" "$NPROC" "$nlat" "$nrep" "$dt" >> "$STATUS"
  echo "[grid] <<< $TAG rc=$rc lat=$nlat reps=$nrep ${dt}s"
  # A SMOKE that produced no [LAT] line means the harness is broken; burning the rest of the
  # allocation on the grid would waste hours to rediscover that.
  case "$TAG" in g7_smoke_*) if [ "$rc" -ne 0 ] || [ "$nlat" -eq 0 ]; then
      echo "[grid] *** SMOKE $TAG FAILED (rc=$rc lat=$nlat) -- ABORTING before the grid ***"; exit 2; fi ;;
  esac
done

echo "[grid] enumerated=$enumerated launched=$launched"
[ "$enumerated" -eq "$launched" ] || { echo "[grid] *** launched != enumerated -- a cell was SKIPPED ***"; exit 3; }
echo "[grid] status: $STATUS"; column -t "$STATUS" 2>/dev/null || cat "$STATUS"

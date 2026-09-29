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

# SINGLE-KERNEL env A/B for the A2A-fused DualGatedGEMM (front route2_ni store).
#
# WHY THIS AND NOT THE e2e GRID. The e2e cell (dlc_g7_grid.sh g7_4x4_...) runs LayerNorm, the back
# O(N^3) einsum, the DTensor baseline and nsys, ~85 s per cell, to exercise a kernel that is ONE call.
# `_route2_run(h)` is that call and nothing else. The harness target `front_route2` already builds it
# with the production configure_fn, so this is a launcher, not a new benchmark -- the timing still
# goes through bench_utils via the harness `_timeit`.
#
# WHAT IT HOLDS FIXED. Same allocation, same nodes, same repo tree, same target, same N/D, same
# config index. ENV is the ONLY variable. Anything else that differs invalidates the comparison.
#
# THE CELL. cp=16 on a 2-D 4x4 mesh, incoming: that is the ONLY configuration measured slow. cp<=8,
# 1-D cp16, and the outgoing direction all reproduce; do not "simplify" to one of those.
#
# Usage:  JID=<jobid> [NS=2048] [ROUNDS=5] bash dlc_front_route2_envab.sh
set -u
JID=${JID:?set JID to a RUNNING allocation}
MD=${MD:?set MD to a writable scratch base for reports and logs}
REPO=${REPO:-$MD/mycodes/fold-cp-ops}
NS=${NS:-2048}
ROUNDS=${ROUNDS:-5}
REPS=${REPS:-2}
# Both arms are REQUIRED inputs -- an A/B whose arms default to hardcoded env names silently
# compares whatever those names happen to point at today, or nothing at all.
OLD=${OLD:?set OLD=<conda prefix> (the reference arm)}
NEW=${NEW:?set NEW=<conda prefix> (the arm under test)}
OUT=${OUT:-$MD/trimul_dlc/front_envab}; mkdir -p "$OUT"
N0=$(scontrol show hostnames "$(squeue -h -j "$JID" -o %N)" | head -1)
cd "$REPO"
PORT=${PORT:-32100}
# Arms INTERLEAVED per rep: a box that drifts mid-sweep cannot masquerade as an env effect.
for rep in $(seq 1 "$REPS"); do
  for arm in old new; do
    E=$OLD; [ "$arm" = new ] && E=$NEW
    [ -x "$E/bin/python" ] || { echo "SKIP $arm: no python at $E"; continue; }
    PORT=$((PORT + 1)); LOG="$OUT/${arm}_r${rep}.log"
    # torchrun, NOT bare `srun --ntasks-per-node=8`: the harness reads WORLD_SIZE and a bare srun
    # does not set it (KeyError: 'WORLD_SIZE'). One agent per node, node-rank from SLURM_PROCID.
    # LD_LIBRARY_PATH must point at THIS env's nvshmem wheel so the host lib matches its device bitcode.
    timeout 900 srun --jobid="$JID" --overlap -N2 --ntasks-per-node=1 \
      --export=ALL,PYTHONPATH=$REPO,CPO_CACHE_ENABLED=1,PYTHONUNBUFFERED=1 \
      bash -c "export LD_LIBRARY_PATH=$E/lib/python3.12/site-packages/nvidia/nvshmem/lib:\${LD_LIBRARY_PATH:-}; \
        $E/bin/python -m torch.distributed.run --nnodes=2 --nproc-per-node=8 \
          --node-rank=\$SLURM_PROCID --master-addr=$N0 --master-port=$PORT \
          -m benchmark.distributed.harness --targets front_route2 --N-list $NS --D 256 \
          --rounds $ROUNDS --max-configs 1 --out-dir $OUT/${arm}_r${rep}" \
      > "$LOG" 2>&1
    rc=$?
    echo "rep$rep $arm rc=$rc $(grep -hoE '\"ms\": *[0-9.]+|median[^,]*' "$OUT/${arm}_r${rep}"/*.json 2>/dev/null | head -2 | tr '\n' ' ')"
  done
done
echo FRONTABDONE

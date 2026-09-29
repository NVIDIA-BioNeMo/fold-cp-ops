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

#SBATCH -J trimul-scaling
#SBATCH --ntasks-per-node=8
# Phase-D/E strong+weak scaling on a SLURM venue. ONE sbatch holds the allocation;
# run_scaling_grid.sh then fires ONE `srun` PER CELL inside it, so per-cell isolation (fresh process group
# + fresh nvshmem init per cell) is preserved while paying the queue wait only once.
#
#   src=benchmark/distributed/harness
#   source $src/cluster_env.sh; cluster_env exclusive
#   sbatch -A $ACCOUNT -p $PARTITION -t $WALLTIME -N 2 --exclusive --mem=0 \
#     -o $OUTBASE/%x-%j.out -e $OUTBASE/%x-%j.err --comment="$REAPER_COMMENT" \
#     --export=ALL,CLUSTER=exclusive,SC_PHASES=DE $src/run_scaling.sbatch
#
# -N 2 covers the whole grid (cp=16 needs 2 nodes). For a 1-node allocation pass SC_MAX_CP=8, which drops
# the 16 two-node cells and runs the other 30 -- never mix the two into one published table (a cross-venue
# or cross-allocation delta can be a clock/config difference masquerading as a scaling effect).
set -x
SELF="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
source "$SELF/cluster_env.sh" 2>/dev/null || source "${REPO:-.}/benchmark/distributed/harness/cluster_env.sh"
cluster_env "${CLUSTER:?set CLUSTER=shared|exclusive}" || { echo "cluster_env failed"; exit 2; }
SELF="$REPO/benchmark/distributed/harness"   # SLURM spools only the .sbatch; the siblings live in $REPO

# SC_REPO = the path the HARNESS sees (inside the pyxis container). SC_HOST_REPO = the path the grid
# driver sees (it runs on the HOST, outside the container). Conflating them made the driver `cd` into a
# container-only path that does not exist on the host — caught by the container smoke test.
export SC_REPO="$WORKDIR"
export SC_HOST_REPO="$REPO"
export SC_WORKDIR="$WORKDIR"
export SC_OUTBASE="${SC_OUTBASE:-$OUTBASE/scaling_$SLURM_JOB_ID}"
export SC_RUNNER=srun
export SC_PY="${SC_PY:-python}"
export SC_PHASES="${SC_PHASES:-DE}"
export SC_ROUNDS="${SC_ROUNDS:-15}"
export SC_WARMUP="${SC_WARMUP:-3}"
export SC_CACHE="${SC_CACHE:-1}"
export SC_CELL_TIMEOUT="${SC_CELL_TIMEOUT:-1200}"
export SC_CONTAINER_ARGS="--container-image=$IMAGE --container-mounts=${SC_MOUNT:-$MOUNT} --container-workdir=$WORKDIR"
export SC_LDLIB_EXPORT="export LD_LIBRARY_PATH=$LDLIB:\$LD_LIBRARY_PATH"
export SC_EXTRA_ENV="${SC_EXTRA_ENV:-}"
# MASTER_ADDR for the srun-per-rank env:// bootstrap (the harness derives the rest from SLURM).
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)"
mkdir -p "$SC_OUTBASE"

echo "[scaling] cluster=$CLUSTER nodes=$SLURM_NNODES image=$IMAGE workdir=$WORKDIR out=$SC_OUTBASE"
echo "[scaling] rev=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || cat "$REPO/REVISION" 2>/dev/null || echo '?')"

# Capture the environment provenance FIRST, so every number in the table is traceable even if the grid is
# later truncated (plan §7: the results doc must carry cluster/GPU/driver/NCCL/NVSHMEM/torch/cutlass/image/
# transport/clock-lock/commit).
timeout 600 srun --overlap -N1 --ntasks-per-node=1 $SC_CONTAINER_ARGS bash -lc "
  cd $WORKDIR; export PYTHONPATH=$WORKDIR; $SC_LDLIB_EXPORT
  $SC_PY -u -m benchmark.distributed.harness.scaling_env --cluster $CLUSTER --out $SC_OUTBASE/env.json
" || echo "[scaling] WARNING: env capture failed (grid continues; fill the env block by hand)"

bash "$SELF/run_scaling_grid.sh"
echo "[scaling] DONE -- aggregate with: python -m benchmark.distributed.harness.aggregate_scaling --results $SC_OUTBASE"

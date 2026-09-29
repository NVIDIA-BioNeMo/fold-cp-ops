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

# Submit the Phase-D/E strong+weak scaling grid. One arm per venue; NEVER mix venues inside one published
# table (a cross-venue delta can be a clock or config difference masquerading as a scaling effect).
#
# Two arms, and the split is structural rather than site-flavoured: `batch` SUBMITS a job, `native`
# runs inside an allocation somebody already holds.
#
#   bash submit_scaling.sh batch                 # sbatch, 2 nodes, whole grid (cp<=16)
#   bash submit_scaling.sh batch 1               # sbatch, 1 node, SC_MAX_CP=8 (the 30 single-node cells)
#   bash submit_scaling.sh native <JOBID>        # run INSIDE an already-held allocation (native env)
#   bash submit_scaling.sh smoke <JOBID>         # ONE cheap cell first (CLAUDE.md local->smoke->grid)
set -u
SELF="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
VENUE="${1:?usage: submit_scaling.sh <batch|native|smoke> [nodes|jobid]}"
ARG="${2:-}"

case "$VENUE" in
  batch)
    source "$SELF/cluster_env.sh"
    cluster_env "${CLUSTER:?set CLUSTER to a cluster_env.sh arm}" || exit 2
    NODES="${ARG:-2}"
    # Site overrides WITHOUT editing cluster_env.sh -- concurrent Phase-B/C sweeps may also use that
    # file, so changing its partition or walltime could disrupt them. Everything site-specific for
    # D/E is injected here instead.
    #   SC_PARTITION / SC_WALLTIME / SC_MOUNT / SC_OUTBASE / SC_EXTRA_ENV
    PART="${SC_PARTITION:-$PARTITION}"
    WALL="${SC_WALLTIME:-${WALLTIME:-04:00:00}}"
    OUTBASE="${SC_OUTBASE:-$OUTBASE}"
    # A 1-node allocation runs the cp<=8 half; the 2-node one runs ONLY the cp=16 remainder, so the
    # second node is held just for the cells that genuinely need it (shared-cluster citizenship).
    EXPORTS="ALL,CLUSTER=$CLUSTER,SC_PHASES=${SC_PHASES:-DE}"
    if [ "$NODES" = 1 ]; then EXPORTS="$EXPORTS,SC_MAX_CP=8"; else EXPORTS="$EXPORTS,SC_MIN_CP=${SC_MIN_CP:-16}"; fi
    for v in SC_OUTBASE SC_MOUNT SC_ROUNDS SC_WARMUP SC_CELL_TIMEOUT SC_RESUME SC_ONLY SC_EXTRA_ENV; do
      eval "val=\${$v:-}"; [ -n "$val" ] && EXPORTS="$EXPORTS,$v=$val"
    done
    # A GPU request is MANDATORY on a GPU partition: some sites' `interactive` partition rejects a job that omits one
    # ("Cannot find GPU specification, you may not submit a job not requesting GPUs in a non-CPU
    # partition") even when --exclusive already implies the whole node. So always name the GPUs; keep
    # --exclusive as well (except where whole-node + --exclusive auto-cancels) so the node is
    # isolated and the perf numbers are not sharing SMs with a co-tenant.
    GPUARG="${SC_GPU_ARG:---gpus-per-node=8}"
    EXTRA="--exclusive --mem=0 $GPUARG"
    # Some schedulers auto-cancel a whole-node job that ALSO passes --exclusive; set SC_NO_EXCLUSIVE=1
    # there. Kept as a knob rather than a site name so the quirk travels with whoever hits it.
    [ -n "${SC_NO_EXCLUSIVE:-}" ] && EXTRA="$GPUARG"
    mkdir -p "$OUTBASE"
    set -x
    sbatch -A "$ACCOUNT" -p "$PART" -t "$WALL" -N "$NODES" $EXTRA \
      -o "$OUTBASE/%x-%j.out" -e "$OUTBASE/%x-%j.err" \
      ${REAPER_COMMENT:+--comment="$REAPER_COMMENT"} \
      --export="$EXPORTS" "$SELF/run_scaling.sbatch"
    ;;

  native|smoke)
    # The native arm runs in a bare environment (no container) inside an allocation the caller already
    # holds, so we drive it with `srun --overlap --jobid=<JID>` per cell rather than submitting a job.
    #
    # DLC_* / SC_RANK_PREAMBLE below give each rank its GPU-LOCAL high-speed HCA. It is deliberately a
    # LAUNCH-ENVIRONMENT knob, not repo Python (CLAUDE.md #45): the fabric CLASS (IBGDA for cross-node) is
    # chosen by run_scaling_cell.sh from nnodes, while the NIC selection itself stays NVSHMEM's own
    # topology-aware multi-rail default unless the operator overrides it here.
    JOBID="${ARG:?pass the held salloc JOBID}"
    : "${DLC_REPO:?set DLC_REPO to the synced checkout on the cluster}"
    : "${DLC_ENV:?set DLC_ENV to the staged conda env on the cluster}"
    # Site helper that maps a local GPU index to its topology-local NIC. Supplied, never guessed: the
    # mapping is a property of the machine, so a default here would be wrong everywhere but one place.
    DGX_HELPER="${DLC_DGX_ENV_HELPER:?set DLC_DGX_ENV_HELPER to your per-rank GPU/NIC env helper}"
    export SC_REPO="$DLC_REPO"
    export SC_OUTBASE="${SC_OUTBASE:-$DLC_REPO/../results/scaling_$JOBID}"
    export SC_RUNNER=srun
    export SC_PY="$DLC_ENV/bin/python"
    export SC_SRUN_EXTRA="--jobid=$JOBID"
    export SC_RANK_PREAMBLE="eval \"\$(bash $DGX_HELPER --gpu \${SLURM_LOCALID:-0})\""
    export SC_ROUNDS="${SC_ROUNDS:-15}"
    export SC_WARMUP="${SC_WARMUP:-3}"
    export SC_CELL_TIMEOUT="${SC_CELL_TIMEOUT:-1200}"
    export SC_PHASES="${SC_PHASES:-DE}"
    export MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "$(squeue -j "$JOBID" -h -o %N)" | head -1)}"
    mkdir -p "$SC_OUTBASE"
    ( cd "$SC_REPO" && PYTHONPATH="$SC_REPO" "$SC_PY" -m benchmark.distributed.harness.scaling_env \
        --cluster "${CLUSTER:-native}" --out "$SC_OUTBASE/env.json" ) || echo "[submit] env capture failed (continuing)"
    if [ "$VENUE" = smoke ]; then
      # ONE cheap cell end-to-end before spending the grid: 1-D cp=2, N=2048, outgoing.
      export SC_ONLY='cp2x1_N2048_out'
      echo "[submit] SMOKE: single cell cp2x1_N2048_out"
    fi
    bash "$SELF/run_scaling_grid.sh"
    ;;

  *) echo "unknown venue '$VENUE'" >&2; exit 2 ;;
esac

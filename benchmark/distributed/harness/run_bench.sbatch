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

#SBATCH -J bench-harness
#SBATCH --ntasks-per-node=8
# ONE cross-cluster sbatch (HARNESS_DESIGN §5). Sources cluster_env.sh (keyed by $CLUSTER), builds the srun
# line from the emitted vars, runs `python -m benchmark.distributed.harness`. The ALLOCATION flags (-A/-p/-t/
# whole-node vs --gpus-per-node/-o/-e/--comment) are supplied at submit (sbatch CLI overrides #SBATCH):
#   src=benchmark/distributed/harness; source $src/cluster_env.sh; cluster_env exclusive
#   sbatch -A $ACCOUNT -p $PARTITION -t $WALLTIME -N 2 --exclusive --mem=0 \
#     -o $OUTBASE/%x-%j.out -e $OUTBASE/%x-%j.err --comment="$REAPER_COMMENT" \
#     --export=ALL,CLUSTER=exclusive,RP_MESH='cp=2*8',RP_TARGETS=...,RP_BASELINES=...,RP_NLIST=... $src/run_bench.sbatch
# (Some sites: drop --exclusive --mem=0, add --gpus-per-node=8.)
set -x
SELF="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
source "$SELF/cluster_env.sh" 2>/dev/null || source "${REPO:-.}/benchmark/distributed/harness/cluster_env.sh"
cluster_env "${CLUSTER:?set CLUSTER=shared|exclusive}" || { echo "cluster_env failed"; exit 2; }
# Re-resolve SELF to the REPO harness dir: SLURM copies ONLY the .sbatch to its spool dir, so the initial
# BASH_SOURCE-based SELF has no sibling gpu_single_flight.sh. $REPO (shared /lustre) does. (cluster_env set it.)
SELF="$REPO/benchmark/distributed/harness"

RP_MESH="${RP_MESH:-cp=2*8}"
NLIST="${RP_NLIST:-2048,3008,4096,5952,8192,10048,13056,16384,24576}"
TARGETS="${RP_TARGETS-cluster,cluster_multislot,gemm}"   # cluster_roundpark REMOVED at the kernel source (epic-close P2); naming it can only raise   # single-dash: empty RP_TARGETS='' => no back targets (baselines-only)
BASELINES="${RP_BASELINES:-2kernel,nccl_a2a}"
DLOCS="${RP_DLOC:-1 8}"; TMO="${RP_TMO:-9000}"; ROUNDS="${RP_ROUNDS:-5}"; MAXCFG="${RP_MAX_CONFIGS:-0}"; CACHE="${RP_CACHE:-1}"
WIDE_W="${RP_FRONT_WIDE_W:-}"   # front wide-put token-subtiles/put (empty => 256B drain, byte-identical no-op)
WIDE_NBI="${RP_FRONT_WIDE_NBI:-}"   # front wide-put B2 (non-blocking per-row + trailing qp-drain); empty => B1 blocking
WIDE_FENCE="${RP_FRONT_WIDE_FENCE:-}"   # #9 sys-fence toggle (empty/1 => ON default, 0 => OFF); paired fenced-vs-unfenced bench
NO_RUN_I="${RP_FRONT_NO_RUN_I:-}"   # DEBUG: disable run_i raster injection (A/B + route2 deadlock isolation)
RUN_I_R="${RP_FRONT_RUN_I_R:-}"     # DEBUG: force run_i R (pick R|ncm to drop the sub-band clamp)
# --targets is OPTIONAL: an empty TARGETS (baselines-only sweep) must DROP the flag entirely, else argparse
# sees `--targets --baselines` and errors "expected one argument". Non-empty -> pass it.
TARG_ARG=""; [ -n "$TARGETS" ] && TARG_ARG="--targets $TARGETS"
# Optional NCCL diagnostics (RP_NCCL_DEBUG=1): dumps the channels/chunksize/GDR NCCL actually picked so a
# comm-BW baseline can be VERIFIED, not assumed. Off by default (INFO floods the .err).
NCCL_DBG_EXPORT=""; [ -n "${RP_NCCL_DEBUG:-}" ] && NCCL_DBG_EXPORT="export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=NET,TUNING"
# NCCL bandwidth-max bundle (probe-topo/nccl_pairwise_sendrecv) — OPT-IN via RP_NCCL_TUNE=1. MEASURED to
# REGRESS our small-message regime (~+10% a2a at 64-628 KiB per-peer): the bundle (4 channels/peer, 512-KiB
# chunk) is proven at 512 MiB; at our sizes NCCL's own small-message heuristics win, so DEFAULT-OFF. Enable
# only for a LARGE-message sweep where the probe's regime applies.
NCCL_TUNE_EXPORT=""; [ -n "${RP_NCCL_TUNE:-}" ] && NCCL_TUNE_EXPORT="export NCCL_NCHANNELS_PER_NET_PEER=4 NCCL_P2P_NET_CHUNKSIZE=524288 NCCL_IB_PCI_RELAXED_ORDERING=2 NCCL_CROSS_NIC=0"
_M=${RP_MESH#cp=}; CP0=${_M%%\**}; CP1=${_M##*\*}; CP=$((CP0*CP1))
NTPN=$(( CP <= 8 ? CP : 8 ))   # tasks/GPUs per node: single-node cp{2,4,8}=>CP, 2-node cp16=>8 (partial-node)
mkdir -p "$OUTBASE"
echo "[run_bench] cluster=$CLUSTER mesh=$RP_MESH cp=$CP targets=$TARGETS baselines=$BASELINES repo=$REPO"

# ONE srun PER (DLOC) CELL -- fresh process group / nvshmem init per cell (HARD RULE: per-cell isolation;
# mirrors the sibling sbatch). Each srun is wrapped in the single-flight lock (defense-in-depth
# GPU-oversubscription guard; SLURM --exclusive already isolates cross-job, this also covers the non-SLURM
# direct-ssh launch path -- see gpu_single_flight.sh header). A cell's crash/timeout kills only its own
# srun step; the loop continues to the next DLOC.
for DLOC in $DLOCS; do
  D=$((DLOC*CP))
  echo "===== BENCH cluster=$CLUSTER mesh=$RP_MESH cp=$CP Dloc=$DLOC D=$D $(date +%H:%M:%S) ====="
  timeout $((TMO + 180)) bash "$SELF/gpu_single_flight.sh" srun --mpi=pmix --ntasks-per-node=$NTPN \
    --container-image="$IMAGE" --container-mounts="$MOUNT" --container-workdir="$WORKDIR" \
    bash -lc "
      cd $WORKDIR
      [ \$SLURM_PROCID = 0 ] && echo REV: \$(git -C $WORKDIR rev-parse --short HEAD 2>/dev/null) mesh=$RP_MESH cp=$CP Dloc=$DLOC D=$D NNODES=\$SLURM_NNODES
      export PYTHONPATH=$WORKDIR CPO_CACHE_ENABLED=$CACHE NVSHMEM_DISABLE_NVLS=1 NVSHMEM_SYMMETRIC_SIZE=40G CPO_DIST_MESH='$RP_MESH'
      export CPO_FRONT_WIDE_W='$WIDE_W'   # front wide-put knob (empty => unset-equiv => 256B drain default)
      export CPO_FRONT_WIDE_NBI='$WIDE_NBI'   # front wide-put B2 pipeline knob (empty => B1 blocking default)
      export CPO_FRONT_WIDE_FENCE='$WIDE_FENCE'   # #9 sys-fence toggle (empty/1 => ON default, 0 => OFF)
      export CPO_FRONT_NO_RUN_I='$NO_RUN_I'   # DEBUG: disable run_i injection (empty => run_i auto per shape)
      export CPO_FRONT_RUN_I_R='$RUN_I_R'   # DEBUG: force run_i R (empty => derived)
      $NCCL_TUNE_EXPORT
      $NCCL_DBG_EXPORT
      export LD_LIBRARY_PATH=$LDLIB:\$LD_LIBRARY_PATH
      timeout $TMO python -u -m benchmark.distributed.harness \
        $TARG_ARG --baselines $BASELINES \
        --N-list $NLIST --D $D --B 1 --rd 2 --rounds $ROUNDS --warmup 3 --max-configs $MAXCFG \
        --out-dir $OUTBASE/mesh${CP0}x${CP1}_D$D
    "
  echo "===== cell Dloc=$DLOC rc=$? $(date +%H:%M:%S) ====="
done
echo "[run_bench] cluster=$CLUSTER GRID DONE"

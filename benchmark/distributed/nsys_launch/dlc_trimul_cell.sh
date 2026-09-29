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


# GENERALIZED per-(cp,mesh) nsys cell for the fair mask-on grid. ONE nsys/node wrapping
# torchrun --nproc_per_node=$NPROC; GPU + NIC(IB) metrics; cudaProfilerApi capture (excludes compile +
# warmup). Profiles the SHIPPED DTensor API TriangularMultiplication (--a2a-fused-dtensor-api default)
# mask-ON (--mask
# default) vs the reference CP baseline (--baseline dtensor_baseline_ring_reducescatter), BOTH directions, DYNAMIC multi-N.
#
# ONE isolated launch PER (cp,mesh) cell (per-cell isolation HARD RULE). Launch idiom:
#   1-D cp<=8 (single node): srun -N1 --ntasks-per-node=1 ... bash dlc_trimul_cell.sh   (NPROC=cp)
#   cp16 / 2-D-16 (2 nodes): srun -N2 --ntasks-per-node=1 ... bash dlc_trimul_cell.sh   (NPROC=8)
# The outer srun spawns one task per node; each task runs THIS script -> one torchrun (NPROC procs).
#
# Per-cell knobs (env; explicit literals — no cluster-specific NIC value baked in, HARD RULE):
#   D        feature dim              (default 256)
#   CP       1-D world size           (set for 1-D; leave MESH2D empty)
#   MESH2D   2-D 'cp0,cp1'            (set for 2-D; leave CP empty)
#   NS       comma N-list            (default 2048,4096; keep bounded — symheap wall at N>=8192)
#   NPROC    procs per node          (REQUIRED: =cp for 1-D single-node, =8 for a 2-node cell)
#   DIRECTION  both|outgoing|incoming (default both)
#   BASELINE   dtensor_baseline_ring_reducescatter|dtensor|none    (default dtensor_baseline_ring_reducescatter — the perf-grid baseline)
#   TAG      output basename tag     (default derived from D/cp/Ns)
#   OUTDIR   nsys-rep dir            (default $MD/trimul_dlc/nsysprof)
set -u
MD=${MD:?set MD to a writable scratch base for reports and logs}
# NEVER hardcode a conda env NAME. Envs get renamed, retired, or are SYMLINKS whose target is a
# different project's env -- a hardcoded default then silently points at nothing (or at the wrong
# tree) and the failure surfaces far from here. Derive it from the interpreter that is actually
# active, and refuse loudly when there is none.
ENV=${ENV:-${CONDA_PREFIX:-$(python3 -c 'import sys; print(sys.prefix)' 2>/dev/null)}}
[ -x "$ENV/bin/python" ] || { echo "ENV=$ENV has no bin/python; pass ENV=<prefix> or activate the env" >&2; exit 2; }
# REPO is overridable so a SECOND checkout can be measured without disturbing the primary one. Two
# trees legitimately coexist here (the primary mirror, and a clean re-clone staged to decide whether a
# result is contaminated by the primary's working tree); hardcoding the path silently measures the
# wrong one, which is an invalid result with no error anywhere.
REPO=${REPO:-$MD/mycodes/fold-cp-ops}
NSYS=$ENV/nsight-compute-2026.1.1/host/target-linux-x64/nsys
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib/python3.12/site-packages/nvidia/nvshmem/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
# This tree's knobs only. The script once ALSO exported the donor tree's differently-prefixed
# equivalents, so one cell script could drive either tree under a byte-identical protocol; that
# comparison is finished and the second prefix is now a knob nothing reads.
export CPO_CACHE_ENABLED=0
export NVSHMEM_IB_ENABLE_IBGDA=1 NVSHMEM_IBGDA_NIC_HANDLER=auto
export NVSHMEM_SYMMETRIC_SIZE=8589934592 NVSHMEM_DISABLE_NVLS=1
# NIC / rail / control-interface selection is a LAUNCH-ENVIRONMENT concern, never a repo constant
# (CLAUDE.md's no-baked-NIC-policy rule): a device list tuned for one fabric is a 7x footgun on
# another, and the names are not portable. Unset leaves NVSHMEM's own topology-aware multi-rail
# selection in charge, which is the intended default. Export NVSHMEM_HCA_LIST / NCCL_IB_HCA /
# NCCL_SOCKET_IFNAME / NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME yourself if your fabric needs a pin, and
# `--export=ALL` carries them through srun unchanged.
export CPO_NVSHMEM_PROFILE=ib-ibgda

D=${D:-256}
NS=${NS:-2048,4096}
DIRECTION=${DIRECTION:-both}
BASELINE=${BASELINE:-dtensor_baseline_ring_reducescatter}
NPROC=${NPROC:?set NPROC (=cp for 1-D single-node, =8 for a 2-node cell)}
NNODES=${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}}
NR=${SLURM_NODEID:-0}
MADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST:-$SLURM_NODELIST}" | head -1)
# Master port UNIQUE per cell: derive from the per-cell TAG (identical across a cell's 2 nodes, distinct
# across cells) so back-to-back cells sharing a master node never collide on a TIME_WAIT'd port (EADDRINUSE).
# Honors an explicit MPORT env override; falls back to the TAG cksum, then the jobid.
#
# MODULUS 12500, NOT the upstream's 20000, and that is a FIX for an inherited defect rather than a
# style change. `20000 + cksum % 20000` spans 20000-40000, which STRADDLES the Linux ephemeral range
# (32768-60999): a tag hashing above 32768 can be handed a port the kernel has already given to an
# unrelated outbound socket, and torchrun's rendezvous then dies with
#   DistNetworkError: ... code: -98, name: EADDRINUSE, message: address already in use
# MEASURED 2026-08-20: `g7_4x4_dtensor_baseline_ring_reducescatter` hashes to 36302 and failed exactly this way on one node.
# Rank 0 died instantly, rank 1 blocked in the rendezvous, and the cell burned 1504 s before its
# timeout -- presenting as a HANG, not as a port problem, which is what makes it expensive. 12500
# keeps every derived port in 20000-32500, entirely below the ephemeral floor.
MPORT=${MPORT:-$((20000 + $(printf '%s' "${TAG:-${SLURM_JOB_ID:-1665230}}" | cksum | cut -d' ' -f1) % 12500))}
OUTDIR=${OUTDIR:-$MD/trimul_dlc/nsysprof}; mkdir -p "$OUTDIR"

# 1-D (--cp CP) vs 2-D (--mesh2d cp0,cp1) mesh arg.
if [ -n "${MESH2D:-}" ]; then
  MESHARG="--mesh2d ${MESH2D}"
  CPTAG="${MESH2D//,/x}"
else
  MESHARG="--cp ${CP:?set CP (1-D) or MESH2D (2-D)}"
  CPTAG="${CP}"
fi
TAG=${TAG:-dlc_maskon_D${D}_cp${CPTAG}_N${NS//,/-}}

cd "$REPO"
# WATCHDOG bound on nsys, NOT a measurement parameter -- raising it cannot change any measured value,
# it only decides how long a wedged cell is allowed to hang before the harness reclaims the node.
# Kept at the donor's 1700 by DEFAULT so an unset environment reproduces the archive run exactly;
# overridable because the heaviest 2-node cell (cp16, 3 N-values x both directions x reference CP baseline)
# overran 1700 s here and was killed BEFORE it could write its .nsys-rep -- losing the cell entirely.
# Anything that changes a NUMBER (--warmup/--iters/--dynamic/--mask) stays hardcoded, deliberately.
NSYS_TIMEOUT=${NSYS_TIMEOUT:-1700}
echo "[dlc] node=$NR/$NNODES host=$(hostname -s) MASTER=$MADDR:$MPORT D=$D cp=$CPTAG Ns=$NS dir=$DIRECTION base=$BASELINE NPROC=$NPROC nsys_timeout=$NSYS_TIMEOUT"
timeout "$NSYS_TIMEOUT" "$NSYS" profile --trace=cuda,nvtx --gpu-metrics-devices=all --nic-metrics=true --sample=none \
  --force-overwrite=true --capture-range=cudaProfilerApi --capture-range-end=stop \
  -o "$OUTDIR/${TAG}_node${NR}" \
  torchrun --nnodes=$NNODES --node_rank=$NR --nproc_per_node=$NPROC --master_addr=$MADDR --master_port=$MPORT \
    benchmark/distributed/nsys_trimul_dtensor.py --D $D $MESHARG --Ns $NS --baseline $BASELINE \
    --direction $DIRECTION --a2a-fused-dtensor-api --mask --dynamic --warmup 8 --iters 4 2>&1 \
  | grep -iE 'libibmad|GPU [0-9]:|General Metrics|LAT|NA |fused_ms|dtensor_baseline_ring_reducescatter_ms|speedup|done|a2a_fused_dtensor_api|Error|Illegal|Insufficient|OutOfMemory|Traceback' | tail -50
# `$?` after a pipeline is the LAST stage's status -- here `tail -50`, which succeeds essentially
# always. So this line printed `rc=0` unconditionally, whatever nsys and torchrun did, and any
# caller whose pass criterion is "each cell reported rc=0" would accept a cell that wrote no
# `.nsys-rep` at all. Capture stage 0 (the nsys/torchrun stage) and EXIT with it, so a failure is
# visible both in the log line and to the launcher's own `$?`.
#
# `PIPESTATUS` is clobbered by the very next command, so this assignment must stay immediately
# after the pipeline -- do not insert anything between them.
rc=${PIPESTATUS[0]}
echo "[dlc] node=$NR rc=$rc"
exit "$rc"

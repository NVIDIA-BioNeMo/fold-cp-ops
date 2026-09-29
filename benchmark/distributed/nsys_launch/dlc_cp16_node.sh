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

# cp16 scale-confirm of the merged front-A2A run_i (both variants), native env, on the
# salloc'd nodes. ONE nsys/node wrapping torchrun --nproc_per_node=8; GPU + NIC(IB) metrics; dtensor_baseline_ring_reducescatter
# baseline; BOTH directions; DYNAMIC + multi-N (2048 anchor + 4096 non-anchor => tests runtime-R
# no-non-anchor-regression at cp16's 8-peer scatter). cudaProfilerApi capture (excludes compile+warmup).
set -u
MD=${MD:?set MD to a writable scratch base for reports and logs}
# NEVER hardcode a conda env NAME. Envs get renamed, retired, or are SYMLINKS whose target is a
# different project's env -- a hardcoded default then silently points at nothing (or at the wrong
# tree) and the failure surfaces far from here. Derive it from the interpreter that is actually
# active, and refuse loudly when there is none.
ENV=${ENV:-${CONDA_PREFIX:-$(python3 -c 'import sys; print(sys.prefix)' 2>/dev/null)}}
[ -x "$ENV/bin/python" ] || { echo "ENV=$ENV has no bin/python; pass ENV=<prefix> or activate the env" >&2; exit 2; }
REPO=$MD/mycodes/fold-cp-ops
NSYS=$ENV/nsight-compute-2026.1.1/host/target-linux-x64/nsys
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib/python3.12/site-packages/nvidia/nvshmem/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
export CPO_CACHE_ENABLED=0 NVSHMEM_IB_ENABLE_IBGDA=1 NVSHMEM_IBGDA_NIC_HANDLER=auto
export NVSHMEM_SYMMETRIC_SIZE=8589934592 NVSHMEM_DISABLE_NVLS=1
# NIC / rail / control-interface selection is a LAUNCH-ENVIRONMENT concern, never a repo constant
# (CLAUDE.md's no-baked-NIC-policy rule): a device list tuned for one fabric is a 7x footgun on
# another, and the names are not portable. Unset leaves NVSHMEM's own topology-aware multi-rail
# selection in charge, which is the intended default. Export NVSHMEM_HCA_LIST / NCCL_IB_HCA /
# NCCL_SOCKET_IFNAME / NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME yourself if your fabric needs a pin, and
# `--export=ALL` carries them through srun unchanged.
export CPO_NVSHMEM_PROFILE=ib-ibgda
NR=${SLURM_NODEID:?}
MADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST:-$SLURM_NODELIST}" | head -1)
MPORT=$((20000 + ${SLURM_JOB_ID:-1} % 20000))
OUT=$MD/trimul_dlc/nsysprof; mkdir -p "$OUT"
TAG=${TAG:-dlc_cp16_N2048-4096_D256}
NS=${NS:-2048,4096}
cd "$REPO"
echo "[dlc] node=$NR host=$(hostname -s) MASTER=$MADDR:$MPORT Ns=$NS HCA_pinned"
timeout 1700 "$NSYS" profile --trace=cuda,nvtx --gpu-metrics-devices=all --nic-metrics=true --sample=none \
  --force-overwrite=true --capture-range=cudaProfilerApi --capture-range-end=stop \
  -o "$OUT/${TAG}_node${NR}" \
  torchrun --nnodes=2 --node_rank=$NR --nproc_per_node=8 --master_addr=$MADDR --master_port=$MPORT \
    benchmark/distributed/nsys_trimul_dtensor.py --D 256 --cp 16 --Ns $NS --baseline dtensor_baseline_ring_reducescatter \
    --direction both --route2_ni --dynamic --warmup 8 --iters 4 2>&1 \
  | grep -iE 'libibmad|GPU [0-9]:|General Metrics|LAT|NA |fused_ms|derived_R|Error|Illegal|Insufficient|OutOfMemory|Traceback' | tail -40
echo "[dlc] node=$NR rc=$?"

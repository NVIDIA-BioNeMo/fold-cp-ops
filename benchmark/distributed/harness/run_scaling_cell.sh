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

# ONE Phase-D/E scaling CELL = ONE isolated launch (CLAUDE.md multi-cell-isolation HARD RULE): a fresh
# process group + fresh nvshmem init per cell, so a crash / OOM / hang in one cell cannot truncate the grid.
# Driven by run_scaling_grid.sh, which loops the cell list emitted by scaling_cells.py. NEVER hand the
# harness more than one N from here -- a multi-N `--N-list` in ONE process is exactly the forbidden shape.
#
# Cell fields arrive as env (the grid driver sets them from the scaling_cells.py TSV):
#   SC_TAG SC_MESH SC_N SC_D SC_CP SC_NNODES SC_NTPN SC_NODE_SZ SC_TARGETS SC_BASELINES
# Runner selection (SC_RUNNER):
#   single   -- cp=1: plain python, ONE GPU, NO process group, NO nvshmem (plan §6)
#   torchrun -- non-slurm single node (e.g. a dev cluster, direct ssh)
#   srun     -- any slurm venue, 1 OR 2 nodes. srun-per-rank: each srun task IS one rank and the harness
#               derives RANK/WORLD_SIZE/LOCAL_RANK from SLURM
#               (DistributedManager._derive_dist_env_from_slurm), so LOCAL_RANK == SLURM_LOCALID and each
#               rank lands on its OWN GPU. This is the shape run_bench.sbatch already proves on
#               a pyxis site; it works unchanged on a native-conda (container-less) setup.
#
# NOTE on the per-GPU HCA pin: do NOT route a native-conda venue through the launcher skill's
# a per-rank torchrun wrapper here. Such a wrapper launches `torchrun --nproc-per-node=1` per srun task, which
# makes LOCAL_RANK==0 on EVERY rank -- with the harness's `torch.cuda.set_device(LOCAL_RANK)` all ranks
# would pile onto GPU 0. Use SC_RANK_PREAMBLE instead (below) to get the same HCA pin on the srun-per-rank
# path, where LOCAL_RANK is correct.
#   SC_RANK_PREAMBLE -- shell evaluated INSIDE each rank before the harness starts, e.g.
#     SC_RANK_PREAMBLE='eval "$(bash /path/gpu_nic_env.sh --gpu $SLURM_LOCALID)"'
#   It is a LAUNCH-ENVIRONMENT knob (operator-supplied), never a NIC/HCA value baked into repo Python
#   (CLAUDE.md #45). Left empty => NVSHMEM's own topology-aware MULTI-RAIL default, which is what the rule
#   prefers and what measured best (a single-rail NIC-PE map was a 7x footgun there).
set -u
REPO="${SC_REPO:?set SC_REPO to the repo checkout under test}"
OUTBASE="${SC_OUTBASE:?set SC_OUTBASE to the results dir}"
RUNNER="${SC_RUNNER:?set SC_RUNNER=single|torchrun|srun}"
ROUNDS="${SC_ROUNDS:-15}"; WARMUP="${SC_WARMUP:-3}"; CACHE="${SC_CACHE:-1}"
TMO="${SC_CELL_TIMEOUT:-1200}"
SYMSZ="${SC_SYMMETRIC_SIZE:-40G}"
BUILD_BUDGET="${SC_BUILD_BUDGET_S:-420}"   # cold JIT at N=8192 needs headroom over the 300 s default
RUN_BUDGET="${SC_RUN_BUDGET_S:-180}"       # bounds a 2-node drain deadlock into a consensus'd skip

OUT="$OUTBASE/$SC_TAG"
mkdir -p "$OUT"

# UNIQUE per-cell MASTER_PORT. Every cell inside ONE allocation otherwise inherits the same
# SLURM_JOB_ID-derived port (distributed_manager.py:703); back-to-back cells then collide with the previous
# cell's socket still in TIME_WAIT -> EADDRINUSE and a silently lost cell. Derived from the cell TAG so it
# is unique per cell, stable across retries, and reproducible from the results dir name (repo commit
# 4503ab7 landed exactly this fix for the G7 grid).
_CK=$(printf '%s' "$SC_TAG" | cksum | cut -d' ' -f1)
# SC_PORT_OFFSET (default 0): the cksum port is deterministic PER TAG, which is what makes a
# cell reproducible -- and what makes a RETRY of the same tag inside one long-lived allocation
# land on the port its own dead predecessor still holds. Shifting by an offset keeps the tag
# (the output key) stable while making retries distinct. 0 reproduces upstream exactly.
# Derived port must stay BELOW the ephemeral range, or the cksum can land on a port the kernel
# hands out to a transient socket. Measured: ip_local_port_range = 32768-60999,
# while the upstream formula 24000 + (_CK % 20000) spans 24000-44000 -- so 32768-44000, over
# half its range, is contestable. A 2-node cell opens many NCCL/nvshmem/TCPStore sockets, and a
# collision presents as rank 0 never listening: the cell burns its whole timeout with only the
# [cell] banner in its log. 20000-31999 is entirely below the floor.
# SC_PORT_OFFSET (default 0) additionally distinguishes RETRIES of one tag inside a long-lived
# allocation, where the deterministic port would otherwise be held by the retry's own corpse.
# NO PORT NUMBER IS BAKED HERE. Port choice is a LAUNCH-ENVIRONMENT concern,
# the same class CLAUDE.md forbids for NIC/HCA/rail values: a band that is free on the host it was
# written against collides elsewhere. `main` derives 24000+(cksum%20000) and this file previously
# derived a narrower band; BOTH are literals, and main's spans the ephemeral range on 36% of job ids.
# `pick_rendezvous_port` reads the floor from /proc at runtime and PROVES the port binds -- which
# also fixes the deterministic-retry trap, where a re-run lands on the port its own dead
# predecessor still holds in TIME_WAIT. An operator-supplied MASTER_PORT always wins.
#
# LIMITATION, stated because it is easy to over-read this: the probe runs HERE (the launcher
# host) while the bind happens on the compute node srun places rank 0 on, so it cannot prove
# freeness there. What DOES transfer is the floor: the returned port is below the ephemeral
# range on any host sharing this kernel setting, which is the part that removes 36% of
# derivations from the collision window. Treat the probe as narrowing, not as a guarantee.
MASTER_PORT="${MASTER_PORT:-$(${SC_PY:-python} -c "import sys; sys.path.insert(0, '$REPO'); from fold_cp_ops._internal.port_selection import pick_rendezvous_port as p; print(p($_CK))" 2>/dev/null)}"
# Fail LOUD rather than launching with an empty port: an unset MASTER_PORT makes every rank
# fall through to a different default and the cell hangs in rendezvous with no error. This
# guard exists because the first version referenced an undefined $PY and silently produced
# an empty string -- 4 cells died rc=1 before the cause was visible.
: "${MASTER_PORT:?could not determine MASTER_PORT (SC_PY=${SC_PY:-python} REPO=$REPO)}"

# --- transport: fabric CLASS only, never a cluster-specific NIC/HCA/rail VALUE (CLAUDE.md #45) ----------
# Cross-node cells need IBGDA; single-node cells leave the profile UNSET so the auto-detect picks the
# P2P/NVLink class. In both cases NVSHMEM's own topology-aware multi-rail NIC selection is left alone.
PROFILE_EXPORT=""
if [ "$SC_NNODES" -gt 1 ]; then
  PROFILE_EXPORT="export CPO_NVSHMEM_PROFILE=${SC_NVSHMEM_PROFILE:-ib-ibgda};"
fi

BASE_ARG=""
[ -n "${SC_BASELINES:-}" ] && BASE_ARG="--baselines $SC_BASELINES"

# cp=1 needs NO process group and NO nvshmem — but it STILL needs the container's torch + fold_cp_ops, so it
# runs through the SAME venue runner as every other cell and differs ONLY by this flag.
# MEASURED BUG: an earlier version gave cp=1 its own bare-HOST `single` runner. On a pyxis
# venue that shell `cd`s into the CONTAINER-only path and dies — all four cp=1 cells failed rc=1 in under
# a second. cp=1 is the DENOMINATOR of every speedup and efficiency number in BOTH tables, so that would
# have silently voided the whole study while every other cell looked healthy.
SINGLE_FLAG=""
[ "${SC_SINGLE:-0}" = 1 ] && SINGLE_FLAG="--single-device"

# IDENTICAL across runners -- only the launcher differs -- so a cp=1 number and a cp=16 number come from
# the same driver, the same bench_utils event-window timer, and the same JSON schema.
HARNESS_ARGS="$SINGLE_FLAG --targets $SC_TARGETS $BASE_ARG --N-list $SC_N --D $SC_D --B 1 --rounds $ROUNDS --warmup $WARMUP --out-dir $OUT"

COMMON_ENV="export PYTHONPATH=$REPO CPO_CACHE_ENABLED=$CACHE NVSHMEM_DISABLE_NVLS=1 NVSHMEM_SYMMETRIC_SIZE=$SYMSZ;
export CPO_DIST_MESH='$SC_MESH' CPO_HARNESS_NODE_SZ=$SC_NODE_SZ;
export CPO_HARNESS_BUILD_BUDGET_S=$BUILD_BUDGET CPO_HARNESS_RUN_BUDGET_S=$RUN_BUDGET;
export MASTER_PORT=$MASTER_PORT; $PROFILE_EXPORT ${SC_EXTRA_ENV:-}"

echo "[cell] tag=$SC_TAG mesh=$SC_MESH N=$SC_N D=$SC_D cp=$SC_CP nodes=$SC_NNODES ntpn=$SC_NTPN runner=$RUNNER port=$MASTER_PORT start=$(date -u +%H:%M:%S)"

case "$RUNNER" in
  single|local)
    # Container-LESS venue only (the local box / a native-conda node): a plain host shell. On a pyxis
    # venue cp=1 must NOT come here — it has no container, hence no torch/fold_cp_ops (see SINGLE_FLAG above);
    # the grid driver keeps cp=1 on the `srun` runner there. GPU pinned via CUDA_VISIBLE_DEVICES.
    timeout "$TMO" bash -lc "
      cd ${SC_HOST_REPO:-$REPO}
      $COMMON_ENV
      ${SC_RANK_PREAMBLE:-:}
      ${SC_PY:-python} -u -m benchmark.distributed.harness $HARNESS_ARGS
    "
    ;;
  torchrun)
    timeout "$TMO" bash -lc "
      cd $REPO
      $COMMON_ENV
      ${SC_RANK_PREAMBLE:-:}
      ${SC_PY:-python} -u -m torch.distributed.run --nnodes=1 --nproc-per-node=$SC_NTPN \
        --master-port=$MASTER_PORT --module benchmark.distributed.harness $HARNESS_ARGS
    "
    ;;
  srun)
    # --overlap so a cell can run inside an already-held allocation without waiting for a new step slot.
    timeout "$TMO" srun ${SC_SRUN_EXTRA:-} --overlap -N "$SC_NNODES" --ntasks-per-node="$SC_NTPN" \
      ${SC_CONTAINER_ARGS:-} bash -lc "
        cd ${SC_WORKDIR:-$REPO}
        [ \"\${SLURM_PROCID:-0}\" = 0 ] && echo \"REV: \$(git -C ${SC_WORKDIR:-$REPO} rev-parse --short HEAD 2>/dev/null) tag=$SC_TAG nodes=\$SLURM_NNODES\"
        $COMMON_ENV
        ${SC_RANK_PREAMBLE:-:}
        ${SC_PY:-python} -u -m benchmark.distributed.harness $HARNESS_ARGS
      "
    ;;
  *) echo "[cell] unknown SC_RUNNER='$RUNNER' (single|local|torchrun|srun)" >&2; exit 2 ;;
esac
RC=$?
echo "[cell] tag=$SC_TAG rc=$RC end=$(date -u +%H:%M:%S)"
exit $RC

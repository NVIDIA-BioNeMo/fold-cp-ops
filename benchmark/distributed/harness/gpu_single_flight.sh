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

# gpu_single_flight.sh -- NODE-LOCAL single-flight lock for a per-cell GPU launch (defense-in-depth
# GPU-oversubscription guard). Wraps ONE cell's launch command (the whole torchrun / srun / python -m
# harness invocation that internally spawns its N ranks) so two DIFFERENT cells landing on the SAME node
# can never run concurrently -- the second blocks until the first releases the lock (process exit).
#
# WHY: SLURM --exclusive already keeps two DIFFERENT job allocations off the same node, but that guarantee
# does NOT hold for a non-SLURM direct-ssh launch path (e.g. a dev cluster: ssh + local torchrun, no
# scheduler) -- there two independently-submitted launches CAN land on the same node's GPUs at once. This
# wrapper makes the "one job per GPU" HARD RULE mechanical instead of manual for any such launcher.
#
# Usage: bash gpu_single_flight.sh <cmd> [args...]     # wraps ONE cell's launch, not per-rank
#   e.g. bash gpu_single_flight.sh timeout 1200 python -u -m benchmark.distributed.harness --N-list ...
# The lock wraps the OUTER command (torchrun/srun/python -m harness) -- the N ranks it spawns internally
# share that single lock acquisition; do NOT wrap per-rank (that would self-deadlock a multi-rank launch).
#
# Env:
#   CPO_GPU_LOCK      lockfile path (default /tmp/fold_cp_ops_gpu_launch.lock; node-local -- different nodes
#                       never serialize against each other, only cells racing for the SAME node do)
#   CPO_GPU_LOCK_WAIT bounded flock wait in seconds (default 1800 = 30min; HARD RULE: never unbounded)
set -u
LOCK="${CPO_GPU_LOCK:-/tmp/fold_cp_ops_gpu_launch.lock}"
WAIT="${CPO_GPU_LOCK_WAIT:-1800}"
[ "$#" -ge 1 ] || { echo "usage: gpu_single_flight.sh <cmd> [args...]" >&2; exit 64; }

exec {fd}>"$LOCK"
flock -w "$WAIT" "$fd" || { echo "[gpu_single_flight] lock timeout (${WAIT}s) on $LOCK -- a prior cell is still running" >&2; exit 75; }
exec "$@"

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

# cluster_env.sh (HARNESS_DESIGN §5) -- per-site table. Sourced by run_bench.sbatch (+ any submit wrapper).
# `cluster_env <name|$CLUSTER>` emits: PARTITION ACCOUNT IMAGE REPO OUTBASE MOUNT WORKDIR WHOLE_NODE(0/1)
# LDLIB REAPER_COMMENT WALLTIME. Adding a site = adding a `case` arm (never edit the sbatch).
#
# NO SITE VALUE IS HARDCODED HERE, deliberately. Account names, filesystem roots, image paths and
# scheduler policy blobs are properties of whoever is running this, not of the code, and a table that
# ships one operator's values is wrong for every other operator while looking authoritative. Each arm
# therefore reads its values from the environment and REFUSES (`:?`) rather than defaulting, so a
# missing value fails at once and names itself instead of silently submitting to the wrong place.
#
# Set before sourcing:
#   CL_ACCOUNT   scheduler account to charge
#   CL_ROOT      shared filesystem root holding the repo, images and results
#   CL_IMAGE     container image path, when the site runs containers (empty = native)
#   CL_MOUNT     container mount spec, when the site runs containers
# Optional: CL_PARTITION, CL_WALLTIME, CL_LDLIB, CL_REAPER_COMMENT, CL_HOSTNAME_PATTERN,
#           REPO_OVERRIDE, OUTBASE_OVERRIDE.
cluster_env() {
  local c="${1:-${CLUSTER:-}}"
  # Autodetect is opt-in: the pattern that maps a hostname to a site arm is site-specific, so it is
  # supplied rather than guessed. With no pattern the caller must name the site explicitly.
  [ -z "$c" ] && [ -n "${CL_HOSTNAME_PATTERN:-}" ] && \
    c="$(hostname 2>/dev/null | grep -oE "$CL_HOSTNAME_PATTERN" | head -1)"
  case "$c" in
    # A shared-partition site: jobs share a node, so WHOLE_NODE=0.
    shared)
      PARTITION="${CL_PARTITION:-interactive}"; WHOLE_NODE=0 ;;
    # An exclusive-partition site: one job per node.
    exclusive)
      PARTITION="${CL_PARTITION:-batch}"; WHOLE_NODE=1 ;;
    *)
      echo "[cluster_env] unknown CLUSTER='$c' (known: shared, exclusive)" >&2; return 2 ;;
  esac
  ACCOUNT="${CL_ACCOUNT:?set CL_ACCOUNT to your scheduler account}"
  MD="${CL_ROOT:?set CL_ROOT to your shared filesystem root}"
  WALLTIME="${CL_WALLTIME:-01:00:00}"
  IMAGE="${CL_IMAGE:-}"
  REPO="${REPO_OVERRIDE:-$MD/fold-cp-ops}"
  OUTBASE="${OUTBASE_OVERRIDE:-$MD/results/bench_$c}"
  # Containerised sites bind the repo in and work from the mount point; native sites work in place.
  if [ -n "$IMAGE" ]; then
    MOUNT="${CL_MOUNT:-$REPO:/workspace/fold-cp-ops}"; WORKDIR="/workspace/fold-cp-ops"
  else
    MOUNT="${CL_MOUNT:-}"; WORKDIR="$REPO"
  fi
  # Where the NVSHMEM host lib lives inside the runtime. Derived from the active interpreter rather
  # than hardcoded, because a container and a native env put site-packages in different places.
  LDLIB="${CL_LDLIB:-$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])' 2>/dev/null)/nvidia/nvshmem/lib}"
  # Scheduler policy annotation, if the site has one. Format is the site's, so it is passed through.
  REAPER_COMMENT="${CL_REAPER_COMMENT:-}"
  export PARTITION ACCOUNT WHOLE_NODE WALLTIME MD IMAGE REPO OUTBASE MOUNT WORKDIR LDLIB REAPER_COMMENT
  return 0
}

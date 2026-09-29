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

# The Phase-D/E scaling GRID DRIVER: loop the cell list from scaling_cells.py, firing ONE isolated,
# `timeout`-bounded launch per cell via run_scaling_cell.sh, and CONTINUE past any cell that crashes,
# hangs, or OOMs (CLAUDE.md multi-cell-isolation HARD RULE -- one fault must never truncate the grid).
#
# The driver itself holds no GPU state: every cell is a separate process tree, so a wedged cell is killed
# by its own `timeout` and the loop advances. Each cell's outcome is appended to a MANIFEST (tsv) next to
# the results, so a partially-completed grid is still a readable, resumable artifact.
#
# Usage (all knobs are env; nothing cluster-specific is baked in):
#   SC_REPO=/path/to/fold_cp_ops SC_OUTBASE=/path/results/scaling SC_RUNNER=srun \
#   SC_PY=/path/conda/bin/python SC_MAX_CP=8 bash run_scaling_grid.sh
#
# Common knobs:
#   SC_MAX_CP     -- only cells with cp <= this (8 = one node; omit for the whole grid incl. cp=16)
#   SC_PHASES     -- 'D' strong only, 'E' weak only, 'DE' both (default)
#   SC_ONLY       -- run only cells whose tag matches this grep -E pattern (the SMOKE knob)
#   SC_RESUME     -- 1 = skip cells whose result JSON already exists (crash-resume)
#   SC_CELL_TIMEOUT, SC_ROUNDS, SC_WARMUP, SC_CACHE, SC_RUNNER, SC_RANK_PREAMBLE, SC_SRUN_EXTRA,
#   SC_CONTAINER_ARGS, SC_WORKDIR, SC_NVSHMEM_PROFILE  -- passed through to run_scaling_cell.sh
set -u
SELF="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
# TWO repo paths, and conflating them is a real bug (caught by the container smoke test): under pyxis only the
# per-cell `srun` runs INSIDE the container, while THIS driver runs on the HOST. SC_REPO is the path the
# HARNESS sees (the container mount, e.g. /workspace/fold-cp-ops); SC_HOST_REPO is the path THIS script sees (the
# lustre path). They are identical on a container-less venue (native conda, local torchrun), so
# the default keeps those unchanged.
REPO="${SC_REPO:?set SC_REPO}"
HOST_REPO="${SC_HOST_REPO:-$REPO}"
OUTBASE="${SC_OUTBASE:?set SC_OUTBASE}"
PY="${SC_PY:-python}"
# Host-side interpreter for the cell enumeration ONLY. scaling_cells.py is deliberately stdlib-only, so a
# bare system python3 serves it — the host has no conda/container env and often no `python` at all.
HOST_PY="${SC_HOST_PY:-python3}"
PHASES="${SC_PHASES:-DE}"
mkdir -p "$OUTBASE"
MANIFEST="$OUTBASE/manifest.tsv"
[ -f "$MANIFEST" ] || printf 'tag\tphases\tsharding\tcp\tN\tdirection\trc\tsecs\tutc\n' > "$MANIFEST"

MAXCP_ARG=""
[ -n "${SC_MAX_CP:-}" ] && MAXCP_ARG="--max-cp $SC_MAX_CP"
[ -n "${SC_MIN_CP:-}" ] && MAXCP_ARG="$MAXCP_ARG --min-cp $SC_MIN_CP"

CELLS="$OUTBASE/cells.tsv"
# cd into $REPO before enumerating. `python -m` puts the INVOKING CWD at sys.path[0], AHEAD of PYTHONPATH —
# so running this driver from another checkout silently resolves `benchmark` to THAT tree and either fails
# to find scaling_cells or, worse, enumerates a stale grid. Pin the cwd so the grid always comes from the
# checkout under test (same reason run_scaling_cell.sh cd's before every launch).
( cd "$HOST_REPO" && PYTHONPATH="$HOST_REPO" "$HOST_PY" -m benchmark.distributed.harness.scaling_cells \
    --phases "$PHASES" $MAXCP_ARG ) > "$CELLS" || {
  echo "[grid] FAILED to enumerate cells from $HOST_REPO (host path) with $HOST_PY" >&2; exit 2; }
NCELLS=$(wc -l < "$CELLS")
# Revision: a cluster checkout is often an rsync'd/untarred tree, not a git clone -- fall back to the
# REVISION file the sync writes so the log always names what is under test.
REV=$(git -C "$HOST_REPO" rev-parse --short HEAD 2>/dev/null || cat "$HOST_REPO/REVISION" 2>/dev/null || echo '?')
echo "[grid] host_repo=$HOST_REPO harness_repo=$REPO rev=$REV cells=$NCELLS phases=$PHASES max_cp=${SC_MAX_CP:-none} min_cp=${SC_MIN_CP:-none} runner=${SC_RUNNER:-} out=$OUTBASE"

I=0
# READ THE CELL LIST ON FD 3, NOT STDIN. `srun` (and ssh, and anything else that forwards a tty) READS
# STDIN — inside a `while read ... done < file` loop it swallows the REST of the file, so the loop exits
# after the FIRST cell and the job reports COMPLETED / exit 0 having silently run 1 of 30 cells.
# MEASURED: a grid run reached cp1x1_N2048_in, then ended at 2:19 with ExitCode 0:0. It only surfaced once
# the cp=1 fix routed cell 1 through `srun`; before that cell 1 used a bare `bash -lc` that died too fast
# to consume anything. Belt and braces: the loop reads fd 3, AND every cell launch gets </dev/null so no
# child can ever reach the list.
while IFS=$'\t' read -r tag phases sharding cp0 cp1 cp N D direction mesh nnodes ntpn node_sz single targets baselines <&3; do
  I=$((I+1))
  if [ -n "${SC_ONLY:-}" ] && ! printf '%s' "$tag" | grep -qE "$SC_ONLY"; then continue; fi
  if [ "${SC_RESUME:-0}" = 1 ] && [ -f "$OUTBASE/$tag/bench_N$N.json" ]; then
    echo "[grid] ($I/$NCELLS) $tag -- RESUME skip (result exists)"; continue
  fi
  # cp=1 has no peers, so the HARNESS skips dist+nvshmem (--single-device, set via SC_SINGLE). But it
  # still needs the container's torch+fold_cp_ops, so on a container venue it MUST stay on that venue's runner.
  # Forcing it to the bare-host `single` runner is what broke that run: the host shell cd'd into the
  # container-only path and all four cp=1 cells died rc=1 in <1 s. Only a container-LESS venue (the local
  # box / native conda) uses the host runner, and there we also pin one GPU (shared-node citizenship).
  RUNNER="${SC_RUNNER:?set SC_RUNNER}"
  EXTRA_VIS=""
  SINGLE=0
  if [ "$single" = 1 ]; then
    SINGLE=1
    case "$RUNNER" in
      srun) : ;;                                                  # keep the container
      *) RUNNER=single; EXTRA_VIS="${SC_SINGLE_GPU:-0}" ;;
    esac
  fi
  T0=$(date +%s)
  echo "[grid] ($I/$NCELLS) ==== $tag phases=$phases cp=$cp N=$N dir=$direction ===="
  env SC_REPO="$REPO" SC_HOST_REPO="$HOST_REPO" SC_OUTBASE="$OUTBASE" SC_RUNNER="$RUNNER" SC_PY="$PY" \
      SC_SINGLE="$SINGLE" \
      SC_TAG="$tag" SC_MESH="$mesh" SC_N="$N" SC_D="$D" SC_CP="$cp" \
      SC_NNODES="$nnodes" SC_NTPN="$ntpn" SC_NODE_SZ="$node_sz" \
      SC_TARGETS="$targets" SC_BASELINES="$baselines" \
      ${EXTRA_VIS:+CUDA_VISIBLE_DEVICES=$EXTRA_VIS} \
      bash "$SELF/run_scaling_cell.sh" < /dev/null
  RC=$?
  SECS=$(( $(date +%s) - T0 ))
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$tag" "$phases" "$sharding" "$cp" "$N" "$direction" \
    "$RC" "$SECS" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$MANIFEST"
  # NEVER abort the grid on a cell failure: a crash/hang/OOM is DATA (the harness records oom/timeout in
  # its own JSON where it can; a hard process death is captured here as a non-zero rc) -- the remaining
  # cells are still worth their GPU time.
  [ "$RC" -ne 0 ] && echo "[grid] cell $tag FAILED rc=$RC after ${SECS}s -- continuing"
done 3< "$CELLS"

# Loud completeness check: a silently-truncated grid is the failure this whole launcher exists to prevent,
# so compare cells LAUNCHED against cells ENUMERATED rather than trusting the loop ran to the end.
_RAN=$(( $(wc -l < "$MANIFEST") - 1 ))
_WANT=$NCELLS
[ -n "${SC_ONLY:-}" ] && _WANT="(filtered by SC_ONLY=$SC_ONLY)"
echo "[grid] GRID DONE -- launched $_RAN cell(s); enumerated $_WANT -- manifest: $MANIFEST"
if [ -z "${SC_ONLY:-}" ] && [ "${SC_RESUME:-0}" != 1 ] && [ "$_RAN" -lt "$NCELLS" ]; then
  echo "[grid] *** TRUNCATED: only $_RAN of $NCELLS cells launched. The grid did NOT complete. ***" >&2
fi
# rc counts PROCESS deaths only (crash / timeout kill). A cell can exit 0 and still have measured nothing:
# the harness catches a build OOM / error / in-cell timeout, records it as that cell's status in its JSON,
# and exits cleanly. So rc=0 does NOT mean "measured" -- aggregate_scaling.py's per-row `status` column is
# the authority on what actually produced a number.
awk -F'\t' 'NR>1{n++; if($7!=0) f++} END{printf "[grid] cells launched=%d process-failures=%d\n", n, f+0}' "$MANIFEST"
echo "[grid] NOTE: rc=0 != measured. Run aggregate_scaling.py --results $OUTBASE for the per-cell status."

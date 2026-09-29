#!/usr/bin/env python3
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

"""Generic, pitfall-guarded NCU profiler for CuTe-DSL kernels: single-device, NVLink-only
distributed, and hybrid NVLink+InfiniBand.

PROJECT-AGNOSTIC BY CONSTRUCTION. No kernel name, conda env, repo path, metric set or launch shape
is baked in. You supply the launch target after ``--`` (a ``torchrun``/``srun``/``python`` line) that
fires the kernel inside an NVTX range; this builds the CORRECT ncu invocation for the chosen mode,
runs it under a bounded timeout, and parses the result into a Speed-of-Light table.

WHY A SCRIPT AND NOT A RECIPE. Every guard below encodes a failure that has actually been paid for.
An ncu command for an in-kernel-NVSHMEM target is not a variation on the single-device one: the
replay mode, the process filter, the output sink and the metric validity all change, and getting any
of them wrong produces a CLEAN-LOOKING run that measured nothing or measured the wrong kernel.

Guards, each raising or warning with the fix named:

  * ``-k regex:kernel``-style broad match -> REFUSE. It also matches NCCL / torch / nvshmem-init
    kernels, so ncu replays the wrong one and burns the pass budget.
  * app-replay on an nvshmem target -> REFUSE. App replay re-runs the WHOLE program per metric pass,
    re-bootstrapping NVSHMEM/NCCL each time, and dies after ~5 passes.
  * a not-replay-safe kernel under kernel-replay -> REFUSE, with the make-it-replay-safe guidance.
  * MULTI-PASS kernel-replay on an nvshmem target -> WARN. Kernel replay saves and restores all
    accessible device memory between passes; against a large symmetric heap that fails with
    ``ContextSaveFailed`` / ``error code (9)``. ``--chunk-metrics`` is the fix.
  * kernel-replay with peers that block in a collective -> WARN. Rank 0 is re-fired once per pass
    (minutes), while every peer waits in the next collective and the default ~10 min NCCL watchdog
    kills them mid-profile. The TARGET must raise its process-group timeout.
  * local-DRAM / L2 metrics on an nvshmem target -> WARN (peer puts bypass them; they read ~0-2%).
  * NVLink byte counters on a HYBRID fabric -> WARN (cross-node traffic is on IB and invisible to
    ``nvlrx__``/``nvltx__``).

Usage::

  # single device
  python scripts/ncu_profiling.py --mode single --kernel-regex GemmSm90 \
    -- python my_launch.py --N 2048

  # NVLink-only, one node
  python scripts/ncu_profiling.py --world 8 --fabric nvlink --replay-safe yes \
    --kernel-regex MyKernelSm90 --nvtx-range profiled \
    -- torchrun --nproc-per-node=8 my_driver.py

  # hybrid NVLink+IB, two nodes, one metric per pass (avoids ContextSaveFailed)
  python scripts/ncu_profiling.py --world 16 --fabric hybrid --replay-safe yes --chunk-metrics \
    --kernel-regex MyKernelSm90 --nvtx-range profiled \
    -- srun -N2 --ntasks-per-node=1 torchrun --nnodes=2 --nproc-per-node=8 my_driver.py

``ncu`` is resolved from ``--ncu``, then ``$NCU_PROFILING_NCU``, then the active interpreter's
prefix, then ``$PATH`` -- never a hardcoded environment.
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
from dataclasses import dataclass, field

# --- metric sets ---------------------------------------------------------------------------------
#: Speed-of-light. Cheap, and the duration is what every derived bandwidth number needs.
SOL_METRICS = [
    "gpu__time_duration.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "smsp__issue_active.avg.pct_of_peak_sustained_active",
]
#: Stall character -- the decisive signal for a comm-fused kernel. `lg_throttle` high means the put
#: pipe is flowing and saturated; `long_scoreboard`/`membar`/`barrier` high means it is WAITING.
STALL_METRICS = [
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
]
#: NOTE `..._lsu_per_issue_active` is NOT a valid metric (returns NaN); use mio_throttle.
STALL_METRICS_EXTRA = [
    "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_membar_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",
]
#: Launch/occupancy attributes. FREE: `--page raw` emits every `launch__*` column regardless of
#: `--metrics`, so these cost no extra pass. Registers-per-thread is how a codegen difference between
#: two toolchains first becomes visible.
LAUNCH_ATTRS = [
    "launch__registers_per_thread",
    "launch__shared_mem_per_block",
    "launch__grid_size",
    "launch__block_size",
]

#: LOCAL memory hierarchy. Meaningless for peer puts: they traverse the fabric, bypassing local
#: DRAM/L2, and read ~0-2% even at saturation.
LOCAL_MEM_METRIC_PREFIXES = (
    "dram__", "gpu__compute_memory_throughput", "lts__", "l1tex__", "gpu__dram_throughput",
)
#: NVLink fabric byte counters. Real, but blind to InfiniBand traffic.
NVLINK_METRIC_PREFIXES = ("nvlrx__", "nvltx__")

#: A `-k` regex matching these also matches NCCL / torch / nvshmem-init kernels.
BROAD_KERNEL_PATTERNS = ("kernel", "cutlass", "main", ".*", "")


@dataclass
class NcuPlan:
    """A resolved ncu invocation plus the guard decisions, inspectable without running anything."""

    mode: str                      # "single" | "nvshmem"
    fabric: str                    # "none" | "nvlink" | "hybrid"
    replay_mode: str               # "kernel" | "application"
    ncu_args: list[str]            # the flags, WITHOUT the metric selection when chunked
    target_cmd: list[str]
    metrics: list[str]
    metric_chunks: list[list[str]]  # one ncu invocation per chunk; single chunk when not chunking
    warnings: list[str] = field(default_factory=list)
    csv_to_stdout: bool = False
    report_path: str | None = None


# --- guards --------------------------------------------------------------------------------------
def _is_broad_kernel_regex(regex: str) -> bool:
    """True when the regex would also select NCCL/torch/nvshmem-init kernels.

    A CuTe-DSL class-name substring (``GemmSm90``) is fine; a bare ``kernel``/``cutlass``/``.*`` is
    not, because CuTe-DSL bakes the class name into every kernel symbol and so does everyone else.
    """
    r = regex.strip()
    r = r.removeprefix("regex:")
    return r.lower() in BROAD_KERNEL_PATTERNS or r in (".*", "^.*$", ".+")


def _matching(metrics: list[str], prefixes: tuple[str, ...]) -> list[str]:
    return [m for m in metrics if any(m.startswith(p) for p in prefixes)]


def resolve_metrics(args) -> list[str]:
    """The metric list: explicit ``--metrics`` wins, else SoL + stalls (+ extras on request)."""
    if args.metrics:
        return [m.strip() for m in args.metrics.split(",") if m.strip()]
    m = list(SOL_METRICS) + list(STALL_METRICS)
    if args.extra_stalls:
        m += STALL_METRICS_EXTRA
    return m


def build_plan(args) -> NcuPlan:
    """Resolve flags into an :class:`NcuPlan`, applying every guard. Pure: no side effects, no GPU.

    :param args: an argparse namespace (or any object with the same attributes).
    :raises ValueError: on a refused configuration; the message names the failure AND the fix.
    """
    warnings: list[str] = []
    mode = args.mode
    if mode == "auto":
        mode = "nvshmem" if (args.world and args.world > 1) or args.nvshmem else "single"

    fabric = getattr(args, "fabric", "auto")
    if fabric == "auto":
        # >8 ranks cannot fit one 8-GPU node, so it must cross nodes. At <=8 assume NVLink; say
        # --fabric hybrid explicitly when a small job really does span nodes.
        fabric = "none" if mode == "single" else ("hybrid" if (args.world or 0) > 8 else "nvlink")

    if not args.allow_broad_kernel and _is_broad_kernel_regex(args.kernel_regex):
        raise ValueError(
            f"refusing broad -k regex {args.kernel_regex!r}: it also matches NCCL/torch/"
            "nvshmem-init kernels, so ncu replays the WRONG kernel and burns the pass budget. Pass "
            "the kernel CLASS-name substring instead. Override with --allow-broad-kernel."
        )

    metrics = resolve_metrics(args)

    if mode == "nvshmem":
        if args.replay_mode == "application" or args.force_app_replay:
            if not args.force_app_replay:
                raise ValueError(
                    "--replay-mode application on an nvshmem target re-runs the WHOLE program per "
                    "metric pass, re-bootstrapping NVSHMEM/NCCL each pass, and dies after ~5. Use "
                    "--replay-mode kernel (the default; needs a replay-safe kernel), or force it "
                    "with --force-app-replay AND a single-pass metric set."
                )
            replay = "application"
            warnings.append(
                "FORCED app-replay on an nvshmem target: re-bootstraps NVSHMEM every metric pass "
                "and dies after ~5. Only viable with a single-counter-domain metric set."
            )
            if len(metrics) > 1 and not args.chunk_metrics:
                warnings.append(
                    f"app-replay with {len(metrics)} metrics forces multiple passes -> expect the "
                    "bootstrap death. Use --chunk-metrics, or reduce to one metric."
                )
        else:
            if args.replay_safe == "no":
                raise ValueError(
                    "kernel-replay refused: --replay-safe no means re-firing the kernel corrupts or "
                    "deadlocks (a signal_wait expecting a fresh signal each pass; SIGNAL_ADD "
                    "counters needing a per-launch host reset). Either make a replay-safe variant "
                    "(pre-seed the wait condition, use monotonic '>=' counters reset ONCE by the "
                    "host) and pass --replay-safe yes, or profile a SINGLE metric under "
                    "--force-app-replay."
                )
            if args.replay_safe == "unknown":
                warnings.append(
                    "--replay-safe unknown: assuming replay-safe. If the run HANGS, it is not -- "
                    "re-run with --replay-safe no for the guidance."
                )
            replay = "kernel"

        # Multi-pass kernel replay saves/restores all accessible device memory between passes. A
        # large symmetric heap makes that fail -- MEASURED as `ContextSaveFailed` + `error code (9)`
        # -- while the SAME target profiles fine with one metric (one pass, no save/restore cycle).
        if replay == "kernel" and len(metrics) > 1 and not args.chunk_metrics:
            warnings.append(
                f"{len(metrics)} metrics under kernel-replay will force several passes, and kernel "
                "replay saves/restores ALL accessible device memory between them. Against a large "
                "symmetric heap that fails with 'ContextSaveFailed' / 'error code (9)'. Use "
                "--chunk-metrics to run one single-pass invocation per metric."
            )
        # Rank 0 is re-fired once per pass while every peer sits in its next collective.
        warnings.append(
            "kernel-replay re-fires rank 0 once per pass (minutes) while peers block in the next "
            "collective: the default ~10 min NCCL watchdog will kill them mid-profile. The TARGET "
            "must raise its process-group timeout, and peers must stay live for the puts to land."
        )

        bad = _matching(metrics, LOCAL_MEM_METRIC_PREFIXES)
        if bad:
            warnings.append(
                f"local memory-hierarchy metrics on an nvshmem target ({', '.join(bad)}): peer puts "
                "BYPASS local DRAM/L2 and these read ~0-2% even at saturation. Judge comm by "
                "gpu__time_duration.sum + known bytes, and by the stall character."
            )
        nvl = _matching(metrics, NVLINK_METRIC_PREFIXES)
        if nvl and fabric == "hybrid":
            warnings.append(
                f"NVLink byte counters ({', '.join(nvl)}) on a HYBRID fabric: cross-node traffic "
                "goes over InfiniBand and is INVISIBLE to nvlrx__/nvltx__, so these undercount. "
                "Read the NIC counters for the IB leg."
            )
    else:
        replay = args.replay_mode if args.replay_mode != "auto" else "kernel"

    # --- assemble (metric selection is appended per chunk by iter_invocations) ---
    ncu: list[str] = ["--replay-mode", replay]
    ncu += ["-k", args.kernel_regex if args.kernel_regex.startswith("regex:")
            else f"regex:{args.kernel_regex}"]
    ncu += ["-c", str(args.launch_count)]
    if args.launch_skip:
        ncu += ["-s", str(args.launch_skip)]
    if mode == "nvshmem":
        # Every rank runs under the profiler, but only the one inside --nvtx-range is measured; the
        # others must keep running so the peer puts have somewhere to land.
        ncu += ["--target-processes", "all"]
    if args.nvtx_range:
        ncu += ["--nvtx", "--nvtx-include", f"{args.nvtx_range}/"]

    csv_to_stdout = False
    report_path = args.output
    if report_path:
        ncu += ["-o", report_path, "-f"]
    else:
        # csv-to-stdout, not -o: a report-file write can hang on an nvshmem target.
        ncu += ["--csv", "--page", "raw"]
        csv_to_stdout = True

    if args.chunk_metrics and not args.set:
        chunks = [[m] for m in metrics]
    else:
        chunks = [metrics]

    return NcuPlan(mode=mode, fabric=fabric, replay_mode=replay, ncu_args=ncu,
                   target_cmd=list(args.target), metrics=metrics, metric_chunks=chunks,
                   warnings=warnings, csv_to_stdout=csv_to_stdout, report_path=report_path)


def iter_invocations(plan: NcuPlan, ncu_bin: str, set_name: str | None = None):
    """Yield one full argv per metric chunk. One chunk unless ``--chunk-metrics``."""
    for chunk in plan.metric_chunks:
        sel = ["--set", set_name] if set_name else ["--metrics", ",".join(chunk)]
        yield [ncu_bin] + plan.ncu_args + sel + plan.target_cmd


# --- CSV parsing ---------------------------------------------------------------------------------
def parse_ncu_csv(text: str) -> list[dict]:
    """Parse ncu CSV into one dict per profiled kernel instance.

    Handles BOTH shapes, which is a real pitfall -- a parser written for one silently returns []
    on the other:

    * WIDE (``--csv --page raw`` to stdout): a header of column names, a UNITS row, then one data
      row per instance. ``Kernel Name`` is a column; there is no ``Metric Name`` column. With
      ``--nvtx-include`` ncu also prepends NVTX-domain columns.
    * LONG (``--import <rep> --csv --page details``): one row per (instance, metric), with
      ``Metric Name`` / ``Metric Value`` / ``Metric Unit``.

    Tolerates the ``==PROF==`` banners and transport noise ncu emits before the CSV. Returns [] when
    nothing parses -- callers must treat that as failure, never as an empty-but-successful profile.
    """
    csv_lines = [ln for ln in text.splitlines() if ln.lstrip().startswith('"')]
    if not csv_lines:
        return []
    rows = list(csv.reader(csv_lines))
    if not rows:
        return []
    header = rows[0]
    if "Metric Name" in header and "Metric Value" in header:
        return _parse_long(rows, header)
    return _parse_wide(rows, header)


def _parse_wide(rows: list[list[str]], header: list[str]) -> list[dict]:
    """WIDE shape: header, units row, data rows."""
    idx = {h: i for i, h in enumerate(header)}
    kname_i, id_i = idx.get("Kernel Name"), idx.get("ID")
    data_rows = rows[1:]
    units_row = None
    dur_i = idx.get("gpu__time_duration.sum")
    # The row after the header is units when its duration cell is non-numeric ('ns', not a number).
    if data_rows and dur_i is not None and dur_i < len(data_rows[0]) \
            and _to_float(data_rows[0][dur_i]) is None:
        units_row, data_rows = data_rows[0], data_rows[1:]
    descriptor = {"ID", "Process ID", "Process Name", "Host Name", "Kernel Name", "Context",
                  "Stream", "Block Size", "Grid Size", "Device", "CC"}
    metric_cols = [(h, i) for i, h in enumerate(header)
                   if h not in descriptor and not h.startswith(("thread Domain:", "Id:Domain:"))]
    out = []
    for r in data_rows:
        if not r or all(c == "" for c in r):
            continue
        inst = {"_kernel": r[kname_i] if kname_i is not None and kname_i < len(r) else "?",
                "_id": r[id_i] if id_i is not None and id_i < len(r) else "", "_units": {}}
        for h, i in metric_cols:
            if i < len(r) and r[i] != "":
                inst[h] = r[i]
                if units_row is not None and i < len(units_row) and units_row[i]:
                    inst["_units"][h] = units_row[i]
        out.append(inst)
    return out


def _parse_long(rows: list[list[str]], header: list[str]) -> list[dict]:
    """LONG shape: pivot ``Metric Name`` -> ``Metric Value`` per (ID, Kernel Name)."""
    dr = csv.DictReader(io.StringIO("\n".join(",".join(_q(c) for c in r) for r in rows)))
    instances: dict = {}
    order: list = []
    for r in dr:
        mn = r.get("Metric Name")
        if not mn:
            continue
        key = (r.get("ID", ""), r.get("Kernel Name", ""))
        if key not in instances:
            instances[key] = {"_kernel": r.get("Kernel Name", ""), "_id": r.get("ID", ""),
                              "_units": {}}
            order.append(key)
        instances[key][mn] = r.get("Metric Value", "")
        if r.get("Metric Unit"):
            instances[key]["_units"][mn] = r["Metric Unit"]
    return [instances[k] for k in order]


def merge_instances(batches: list[list[dict]]) -> list[dict]:
    """Merge per-chunk results into one instance list, keyed by kernel name.

    Chunked collection runs the target once per metric, so each batch carries the same kernel with a
    different metric. Merging by ``_kernel`` reassembles one row -- without this, ``--chunk-metrics``
    would report N separate single-metric instances and the table would be unreadable.
    """
    merged: dict = {}
    order: list = []
    for batch in batches:
        for inst in batch:
            k = inst.get("_kernel", "?")
            if k not in merged:
                merged[k] = {"_kernel": k, "_id": inst.get("_id", ""), "_units": {}}
                order.append(k)
            for mk, mv in inst.items():
                if mk == "_units":
                    merged[k]["_units"].update(mv)
                elif not mk.startswith("_"):
                    merged[k][mk] = mv
    return [merged[k] for k in order]


def _q(s: str) -> str:
    return '"' + s.replace('"', '""') + '"'


def _to_float(v):
    if v is None:
        return None
    try:
        return float(str(v).replace(",", "").strip())
    except ValueError:
        return None


def format_sol_table(instances: list[dict], bytes_moved: int | None = None,
                     only_metrics: list[str] | None = None) -> str:
    """Render a compact table. With ``bytes_moved`` and a duration, derive achieved GB/s.

    ``only_metrics`` filters the columns: the WIDE ``--page raw`` CSV carries ~280
    ``device__attribute_*`` / ``launch__*`` columns, and an unfiltered table is unreadable. The
    launch attributes are always shown when present -- they cost nothing and a registers-per-thread
    difference is often the first visible sign of a codegen change.
    """
    if not instances:
        return "(no kernel instances parsed -- check the -k regex matched and a kernel was profiled)"
    show = None
    if only_metrics:
        show = list(dict.fromkeys(["gpu__time_duration.sum"] + list(only_metrics) + LAUNCH_ATTRS))
    out = []
    for inst in instances:
        kname = inst.get("_kernel", "?")
        m = re.search(r"_([A-Za-z][A-Za-z0-9]*)_object", kname) or \
            re.search(r"___main___([A-Za-z0-9_]+?)(?:_object|_at_|$)", kname)
        out.append(f"kernel: {m.group(1) if m else kname}")
        dur_ns = _to_float(inst.get("gpu__time_duration.sum"))
        for k in (show if show is not None else [k for k in inst if not k.startswith("_")]):
            if k in inst:
                out.append(f"  {k:<70s} {inst[k]!s:>14s} {inst.get('_units', {}).get(k, '')}")
        if bytes_moved and dur_ns:
            out.append(f"  {'-> achieved GB/s (bytes/duration)':<70s} "
                       f"{bytes_moved / (dur_ns * 1e-9) / 1e9:>14.1f} GB/s")
        out.append("")
    return "\n".join(out)


# --- driver --------------------------------------------------------------------------------------
def resolve_ncu(explicit: str | None = None) -> str:
    """Locate ``ncu``: explicit, then ``$NCU_PROFILING_NCU``, then the ACTIVE interpreter's prefix,
    then ``$PATH``. Never a hardcoded environment -- env names get renamed, retired, or are symlinks
    into a different project's prefix.
    """
    if explicit:
        return explicit
    if os.environ.get("NCU_PROFILING_NCU"):
        return os.environ["NCU_PROFILING_NCU"]
    prefix = os.environ.get("CONDA_PREFIX") or sys.prefix
    for cand in (os.path.join(os.path.dirname(sys.executable), "ncu"),
                 os.path.join(prefix, "bin", "ncu")):
        if os.path.exists(cand):
            return cand
    # Some distributions ship it only under the versioned nsight-compute tree.
    import glob
    for pat in (os.path.join(prefix, "nsight-compute-*", "ncu"),
                os.path.join(prefix, "nsight-compute-*", "host", "target-linux-x64", "ncu")):
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[-1]
    found = shutil.which("ncu")
    if found:
        return found
    raise FileNotFoundError("ncu not found. Pass --ncu <path>, set NCU_PROFILING_NCU, or add it to "
                            "PATH. It is NOT assumed to live in any particular conda env.")


def _run_one(argv: list[str], timeout: int) -> tuple[int, str, str]:
    """Run one ncu invocation in its own process group, SIGKILLing the GROUP on timeout.

    ``subprocess.run(timeout=)`` kills only the direct child, which would leave orphaned ncu and rank
    processes holding GPU memory -- the next run then fails to allocate for reasons that look
    unrelated.
    """
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            start_new_session=True)
    try:
        out, err = proc.communicate(timeout=timeout)
        return proc.returncode, out, err
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            out, err = proc.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            out, err = "", ""
        return 124, out, err


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Generic pitfall-guarded NCU profiler for CuTe-DSL kernels "
                    "(single-device, NVLink-only, and hybrid NVLink+IB).",
        epilog="The launch target follows '--'. It MUST wrap the profiled launch in an NVTX range; "
               "for a distributed target, only the profiled rank uses that range and the other "
               "ranks stay live peers.",
    )
    ap.add_argument("--mode", choices=["auto", "single", "nvshmem"], default="auto")
    ap.add_argument("--fabric", choices=["auto", "nvlink", "hybrid"], default="auto",
                    help="auto: hybrid when --world>8 (cannot fit one node), else nvlink.")
    ap.add_argument("--world", type=int, default=0, help="rank count; >1 implies nvshmem mode.")
    ap.add_argument("--nvshmem", action="store_true", help="force nvshmem mode.")
    ap.add_argument("--kernel-regex", required=True, help="kernel CLASS-name substring for -k.")
    ap.add_argument("--nvtx-range", default="profiled", help="NVTX range wrapping the launch.")
    ap.add_argument("--replay-mode", choices=["auto", "kernel", "application"], default="auto")
    ap.add_argument("--replay-safe", choices=["yes", "no", "unknown"], default="unknown",
                    help="is re-firing the kernel safe? 'no' refuses kernel-replay with the fix.")
    ap.add_argument("--force-app-replay", action="store_true")
    ap.add_argument("--allow-broad-kernel", action="store_true")
    ap.add_argument("--chunk-metrics", action="store_true",
                    help="one ncu invocation per metric (each single-pass). The fix for "
                         "ContextSaveFailed on a large symmetric heap; costs one target run each.")
    ap.add_argument("--metrics", default=None, help="comma-separated; overrides the default set.")
    ap.add_argument("--set", default=None, help="ncu --set <section> instead of a metric list.")
    ap.add_argument("--extra-stalls", action="store_true")
    ap.add_argument("--launch-count", "-c", type=int, default=1)
    ap.add_argument("--launch-skip", "-s", type=int, default=0)
    ap.add_argument("--bytes", type=int, default=None, help="bytes moved; derives achieved GB/s.")
    ap.add_argument("--output", "-o", default=None, help="write .ncu-rep instead of csv-to-stdout.")
    ap.add_argument("--ncu", default=None)
    ap.add_argument("--timeout", type=int, default=900, help="seconds, per invocation.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--print-cmd", action="store_true")
    ap.add_argument("target", nargs=argparse.REMAINDER)
    args = ap.parse_args(argv)

    if args.target and args.target[0] == "--":
        args.target = args.target[1:]
    if not args.target:
        ap.error("missing launch target after '--'")

    try:
        plan = build_plan(args)
    except ValueError as e:
        print(f"[ncu_profiling] REFUSED: {e}", file=sys.stderr)
        return 2

    ncu_bin = resolve_ncu(args.ncu)
    for w in plan.warnings:
        print(f"[ncu_profiling] WARNING: {w}", file=sys.stderr, flush=True)

    invocations = list(iter_invocations(plan, ncu_bin, args.set))
    if args.print_cmd or args.dry_run:
        print(f"\n=== resolved ===\nmode={plan.mode} fabric={plan.fabric} "
              f"replay-mode={plan.replay_mode} chunks={len(plan.metric_chunks)}", flush=True)
        for c in invocations:
            print("=== ncu command ===\n" + " ".join(shlex.quote(x) for x in c), flush=True)
    if args.dry_run:
        return 0

    batches, rc_worst = [], 0
    for i, argv_i in enumerate(invocations, 1):
        rc, out, err = _run_one(argv_i, args.timeout)
        if rc == 124:
            print(f"[ncu_profiling] TIMEOUT after {args.timeout}s on chunk {i}/{len(invocations)}. "
                  "Under app-replay this is the bootstrap death; under kernel-replay suspect a "
                  "not-replay-safe kernel, or peers killed by their collective watchdog.",
                  file=sys.stderr)
        if rc != 0:
            rc_worst = rc_worst or rc
            print(f"[ncu_profiling] chunk {i} exited {rc}. stderr tail:", file=sys.stderr)
            print("\n".join(err.splitlines()[-15:]), file=sys.stderr)
            if "ContextSaveFailed" in err or "error code (9)" in err:
                print("[ncu_profiling] ContextSaveFailed: kernel-replay could not save/restore this "
                      "context's device memory between passes. Re-run with --chunk-metrics (one "
                      "single-pass invocation per metric).", file=sys.stderr)
        if plan.csv_to_stdout:
            batches.append(parse_ncu_csv(out))
        else:
            print(out)

    if plan.csv_to_stdout:
        instances = merge_instances(batches)
        print("\n=== SoL table ===")
        print(format_sol_table(instances, bytes_moved=args.bytes,
                               only_metrics=None if args.set else plan.metrics))
        if not instances:
            print("[ncu_profiling] no instances parsed -- the profile measured NOTHING. Check the "
                  "-k regex matched and that the NVTX range actually wrapped a launch.",
                  file=sys.stderr)
            return rc_worst or 3
    else:
        print(f"[ncu_profiling] report(s) written to {plan.report_path}. Read with:\n"
              f"  {ncu_bin} --import {plan.report_path} --csv --page raw")
    return rc_worst


if __name__ == "__main__":
    raise SystemExit(main())

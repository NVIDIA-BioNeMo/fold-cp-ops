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

"""Tests for ``scripts/ncu_profiling.py`` -- the generic pitfall-guarded NCU profiler.

Two tiers:

* PURE-LOGIC (no GPU, always run): every guard, both CSV shapes, chunk merging, the table filter and
  the ncu resolution order. These ENCODE the protocol, so a future profiling run either works or is
  refused with the fix named -- rather than producing a clean-looking run that measured nothing.
* INTEGRATION (needs a GPU + ncu): end-to-end against a real kernel.

The guard cases are deliberately PURE-LOGIC. A guard whose only proof is a GPU run is a guard nobody
checks; each of these fires from ``build_plan`` alone, so the whole protocol is verifiable on a
laptop in milliseconds.
"""
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import ncu_profiling as N

try:
    import torch
    _CUDA = torch.cuda.is_available()
except (ImportError, RuntimeError):
    _CUDA = False
PY = sys.executable
try:
    _NCU = N.resolve_ncu()
    _HAS_NCU = os.path.exists(_NCU)
except FileNotFoundError:
    _NCU, _HAS_NCU = "", False


class _Args:
    """argparse-Namespace stand-in, so the guards are testable without the CLI."""

    def __init__(self, **kw):
        d = {"mode": "auto", "fabric": "auto", "world": 0, "nvshmem": False,
             "kernel_regex": "MyKernelSm90", "nvtx_range": "profiled", "replay_mode": "auto",
             "replay_safe": "unknown", "force_app_replay": False, "allow_broad_kernel": False,
             "chunk_metrics": False, "metrics": None, "set": None, "extra_stalls": False,
             "launch_count": 1, "launch_skip": 0, "output": None,
             "target": ["torchrun", "x.py"]}
        d.update(kw)
        for k, v in d.items():
            setattr(self, k, v)


# --- guards ---------------------------------------------------------------------------------------
def test_broad_kernel_regex_refused():
    """A bare ``kernel``/``cutlass``/``.*`` also matches NCCL/torch/nvshmem-init kernels, so ncu
    profiles the wrong one. Refuse unless explicitly overridden."""
    for bad in ("kernel", "cutlass", ".*", "regex:kernel", "main"):
        with pytest.raises(ValueError, match="broad -k regex"):
            N.build_plan(_Args(mode="nvshmem", world=2, kernel_regex=bad))
    assert "regex:MyKernelSm90" in N.build_plan(
        _Args(mode="nvshmem", world=2, replay_safe="yes")).ncu_args
    assert N.build_plan(_Args(mode="nvshmem", world=2, kernel_regex="kernel", replay_safe="yes",
                              allow_broad_kernel=True)).replay_mode == "kernel"


def test_app_replay_on_nvshmem_refused_and_forceable():
    with pytest.raises(ValueError, match="re-bootstrapping NVSHMEM"):
        N.build_plan(_Args(mode="nvshmem", world=2, replay_mode="application", replay_safe="yes"))
    p = N.build_plan(_Args(mode="nvshmem", world=2, replay_mode="application",
                           force_app_replay=True, replay_safe="yes"))
    assert p.replay_mode == "application"
    assert any("re-bootstraps NVSHMEM" in w for w in p.warnings)
    assert any("multiple passes" in w for w in p.warnings)


def test_not_replay_safe_refused():
    with pytest.raises(ValueError, match="corrupts or\ndeadlocks|kernel-replay refused"):
        N.build_plan(_Args(mode="nvshmem", world=2, replay_safe="no"))


def test_replay_safe_unknown_warns_but_proceeds():
    p = N.build_plan(_Args(mode="nvshmem", world=2, replay_safe="unknown"))
    assert p.replay_mode == "kernel" and any("unknown" in w for w in p.warnings)


def test_multipass_kernel_replay_warns_about_context_save():
    """MEASURED: a multi-metric kernel-replay against a large symmetric heap dies with
    ``ContextSaveFailed``, while the same target profiles fine one metric at a time."""
    p = N.build_plan(_Args(mode="nvshmem", world=16, replay_safe="yes"))
    assert any("ContextSaveFailed" in w and "--chunk-metrics" in w for w in p.warnings)
    # chunking silences it, because each invocation is then single-pass
    p2 = N.build_plan(_Args(mode="nvshmem", world=16, replay_safe="yes", chunk_metrics=True))
    assert not any("ContextSaveFailed" in w for w in p2.warnings)


def test_peer_watchdog_warning_always_on_for_nvshmem():
    """Rank 0 is re-fired per pass while peers block in their next collective; the default watchdog
    kills them mid-profile. Warn unconditionally -- it is a property of the target, not of flags."""
    p = N.build_plan(_Args(mode="nvshmem", world=2, replay_safe="yes"))
    assert any("watchdog" in w and "process-group timeout" in w for w in p.warnings)
    assert not any("watchdog" in w for w in N.build_plan(_Args(mode="single")).warnings)


def test_local_dram_metric_warns_only_on_nvshmem():
    for bad in ("dram__bytes.sum", "lts__t_sectors.sum", "l1tex__data_bank_conflicts_pipe_lsu.sum",
                "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"):
        p = N.build_plan(_Args(mode="nvshmem", world=2, replay_safe="yes",
                               metrics=f"gpu__time_duration.sum,{bad}"))
        assert any("BYPASS" in w for w in p.warnings), bad
    p = N.build_plan(_Args(mode="single", metrics="gpu__time_duration.sum,dram__bytes.sum"))
    assert not any("BYPASS" in w for w in p.warnings)


def test_nvlink_counters_warn_only_on_hybrid_fabric():
    """NVLink byte counters cannot see the InfiniBand leg, so they undercount on a hybrid job."""
    p = N.build_plan(_Args(mode="nvshmem", world=16, fabric="hybrid", replay_safe="yes",
                           metrics="nvltx__bytes.sum"))
    assert any("INVISIBLE to nvlrx__" in w for w in p.warnings)
    p2 = N.build_plan(_Args(mode="nvshmem", world=8, fabric="nvlink", replay_safe="yes",
                            metrics="nvltx__bytes.sum"))
    assert not any("INVISIBLE" in w for w in p2.warnings)


# --- plan shape -----------------------------------------------------------------------------------
def test_mode_and_fabric_auto_detection():
    assert N.build_plan(_Args(world=2, replay_safe="yes")).mode == "nvshmem"
    assert N.build_plan(_Args(world=1)).mode == "single"
    assert N.build_plan(_Args(nvshmem=True, replay_safe="yes")).mode == "nvshmem"
    assert N.build_plan(_Args(mode="single")).fabric == "none"
    # >8 ranks cannot fit one 8-GPU node -> must cross nodes
    assert N.build_plan(_Args(world=8, replay_safe="yes")).fabric == "nvlink"
    assert N.build_plan(_Args(world=16, replay_safe="yes")).fabric == "hybrid"


def test_nvshmem_plan_carries_the_whole_recipe():
    p = N.build_plan(_Args(mode="nvshmem", world=2, replay_safe="yes", nvtx_range="a2a_profiled"))
    a = p.ncu_args
    assert a[a.index("--replay-mode") + 1] == "kernel"
    assert a[a.index("--target-processes") + 1] == "all"      # peers must stay live
    assert "--nvtx" in a and "a2a_profiled/" in a             # single-rank isolation
    assert "--csv" in a and "raw" in a and p.csv_to_stdout    # safer than -o under nvshmem


def test_single_device_plan_has_no_multirank_flags():
    p = N.build_plan(_Args(mode="single"))
    assert p.mode == "single" and "--target-processes" not in p.ncu_args


def test_chunking_produces_one_single_pass_invocation_per_metric():
    p = N.build_plan(_Args(mode="nvshmem", world=16, replay_safe="yes", chunk_metrics=True,
                           metrics="gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed"))
    assert p.metric_chunks == [["gpu__time_duration.sum"],
                               ["sm__throughput.avg.pct_of_peak_sustained_elapsed"]]
    cmds = list(N.iter_invocations(p, "/x/ncu"))
    assert len(cmds) == 2
    assert cmds[0][cmds[0].index("--metrics") + 1] == "gpu__time_duration.sum"
    assert cmds[0][-2:] == ["torchrun", "x.py"]               # target is appended last


def test_output_file_switches_off_csv_stdout():
    p = N.build_plan(_Args(mode="single", output="/tmp/r.ncu-rep"))
    assert not p.csv_to_stdout and "-o" in p.ncu_args and "-f" in p.ncu_args


# --- CSV parsing ----------------------------------------------------------------------------------
# Built column-by-column, not hand-typed: a units row one field out of alignment shifts every
# metric and the parser silently returns a row with no duration.
_WIDE_HEADER = ["ID", "Process ID", "Process Name", "Host Name",
                "thread Domain:Push/Pop_Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg",
                "Kernel Name", "Context", "Stream", "Block Size", "Grid Size", "Device", "CC",
                "device__attribute_warp_size", "launch__registers_per_thread",
                "gpu__time_duration.sum", "sm__throughput.avg.pct_of_peak_sustained_elapsed"]
_WIDE_UNITS = [""] * 14 + ["ns", "%"]
_WIDE_DATA = ["0", "39", "python", "host", "dom",
              "kernel_cutlass_kernel___main___MyKernelSm90_object_at_0x", "1", "7", "256", "64",
              "NVIDIA H100", "9.0", "32", "79", "223008", "0.82"]
assert len(_WIDE_UNITS) == len(_WIDE_HEADER) == len(_WIDE_DATA)


def _csv_row(cells):
    return ",".join('"' + c.replace('"', '""') + '"' for c in cells)


WIDE_CSV = (
    "==PROF== Connected to process 39\n"
    "ibrc.cpp:1753: error status: 7 Device enumeration failed.\n"   # benign noise before the CSV
    + _csv_row(_WIDE_HEADER) + "\n" + _csv_row(_WIDE_UNITS) + "\n" + _csv_row(_WIDE_DATA) + "\n"
    + "==PROF== Disconnected\n"
)
LONG_CSV = (
    '==PROF== Connected\n'
    '"ID","Process ID","Kernel Name","Metric Name","Metric Unit","Metric Value"\n'
    '"0","39","kernel_cutlass_kernel_MyKernelSm90_object_at_0x","gpu__time_duration.sum","ns","6272"\n'
    '"0","39","kernel_cutlass_kernel_MyKernelSm90_object_at_0x",'
    '"sm__throughput.avg.pct_of_peak_sustained_elapsed","%","18.2"\n'
)


def test_parse_wide_csv():
    inst = N.parse_ncu_csv(WIDE_CSV)
    assert len(inst) == 1
    r = inst[0]
    assert r["gpu__time_duration.sum"] == "223008"
    assert r["launch__registers_per_thread"] == "79"
    assert r["_units"]["gpu__time_duration.sum"] == "ns"
    assert "MyKernelSm90" in r["_kernel"]


def test_parse_long_csv():
    inst = N.parse_ncu_csv(LONG_CSV)
    assert len(inst) == 1 and inst[0]["gpu__time_duration.sum"] == "6272"


def test_parse_noise_and_empty_return_nothing():
    """An empty parse must be distinguishable from a successful one -- the caller treats [] as
    failure, because a profile that measured nothing otherwise reads as a clean run."""
    assert N.parse_ncu_csv("==PROF== nothing\nibrc error\n") == []
    assert N.parse_ncu_csv("") == []


def test_merge_instances_reassembles_chunked_metrics():
    """Chunked collection profiles the same kernel once per metric; merging by kernel name puts them
    back on one row. Without it, --chunk-metrics would print N single-metric instances."""
    a = [{"_kernel": "k_MyKernelSm90_object", "_id": "0", "_units": {"gpu__time_duration.sum": "ns"},
          "gpu__time_duration.sum": "100"}]
    b = [{"_kernel": "k_MyKernelSm90_object", "_id": "0", "_units": {}, "sm__throughput.avg": "5"}]
    m = N.merge_instances([a, b])
    assert len(m) == 1
    assert m[0]["gpu__time_duration.sum"] == "100" and m[0]["sm__throughput.avg"] == "5"
    assert m[0]["_units"]["gpu__time_duration.sum"] == "ns"


# --- table ----------------------------------------------------------------------------------------
def test_table_filters_columns_and_keeps_launch_attrs():
    inst = N.parse_ncu_csv(WIDE_CSV)
    t = N.format_sol_table(inst, bytes_moved=16777216,
                           only_metrics=["sm__throughput.avg.pct_of_peak_sustained_elapsed"])
    assert "gpu__time_duration.sum" in t                 # duration always shown
    assert "launch__registers_per_thread" in t           # free, and reveals codegen differences
    assert "device__attribute_warp_size" not in t        # the ~280 raw columns filtered out
    assert "achieved GB/s" in t and "75." in t           # 16777216 B / 223008 ns ~= 75.2 GB/s


def test_table_empty_is_explicit():
    assert "no kernel instances" in N.format_sol_table([])


# --- ncu resolution -------------------------------------------------------------------------------
def test_resolve_ncu_prefers_explicit_then_env(monkeypatch, tmp_path):
    """Resolution order must never reach a hardcoded conda env name."""
    assert N.resolve_ncu("/explicit/ncu") == "/explicit/ncu"
    fake = tmp_path / "ncu"
    fake.write_text("")
    monkeypatch.setenv("NCU_PROFILING_NCU", str(fake))
    assert N.resolve_ncu() == str(fake)


def test_resolve_ncu_raises_with_actionable_message(monkeypatch):
    monkeypatch.delenv("NCU_PROFILING_NCU", raising=False)
    monkeypatch.setenv("CONDA_PREFIX", "/nonexistent-prefix")
    monkeypatch.setattr(N.shutil, "which", lambda _: None)
    monkeypatch.setattr(N.os.path, "exists", lambda _: False)
    monkeypatch.setattr(N, "sys", type("S", (), {"prefix": "/nonexistent-prefix",
                                                 "executable": "/nonexistent-prefix/bin/python"}))
    with pytest.raises(FileNotFoundError, match="NOT assumed to live in any particular conda env"):
        N.resolve_ncu()


# --- CLI ------------------------------------------------------------------------------------------
def test_cli_dry_run_emits_commands():
    r = subprocess.run([PY, os.path.join(REPO, "scripts", "ncu_profiling.py"),
                        "--kernel-regex", "MyKernelSm90", "--world", "16", "--replay-safe", "yes",
                        "--chunk-metrics", "--dry-run", "--ncu", "/x/ncu",
                        "--", "torchrun", "d.py"],
                       capture_output=True, text=True, timeout=120, check=False)
    assert r.returncode == 0
    assert "mode=nvshmem fabric=hybrid" in r.stdout
    assert r.stdout.count("=== ncu command ===") == 7      # 7 default metrics -> 7 chunks
    assert "ContextSaveFailed" not in r.stderr              # chunking silences that warning


def test_cli_refuses_broad_regex_with_exit_2():
    r = subprocess.run([PY, os.path.join(REPO, "scripts", "ncu_profiling.py"),
                        "--kernel-regex", "kernel", "--dry-run", "--", "python", "d.py"],
                       capture_output=True, text=True, timeout=120, check=False)
    assert r.returncode == 2 and "REFUSED" in r.stderr


def test_cli_requires_a_target():
    r = subprocess.run([PY, os.path.join(REPO, "scripts", "ncu_profiling.py"),
                        "--kernel-regex", "MyKernelSm90"],
                       capture_output=True, text=True, timeout=120, check=False)
    assert r.returncode != 0 and "missing launch target" in r.stderr


# --- integration ----------------------------------------------------------------------------------
@pytest.mark.skipif(not (_CUDA and _HAS_NCU), reason="needs a CUDA device + ncu")
def test_integration_single_device_end_to_end(tmp_path):
    """Profile a trivial CuTe-DSL-free CUDA kernel end to end and assert a table parses. Uses torch's
    own kernel so the test depends on no project kernel and stays project-agnostic."""
    target = tmp_path / "t.py"
    target.write_text(
        "import torch\n"
        "a = torch.randn(4096, 4096, device='cuda')\n"
        "torch.cuda.nvtx.range_push('profiled')\n"
        "(a @ a).sum().item()\n"
        "torch.cuda.nvtx.range_pop()\n"
    )
    r = subprocess.run(
        [PY, os.path.join(REPO, "scripts", "ncu_profiling.py"), "--mode", "single",
         "--kernel-regex", "gemm", "--allow-broad-kernel", "--nvtx-range", "profiled",
         "--metrics", "gpu__time_duration.sum", "--ncu", _NCU, "--timeout", "600",
         "--", PY, str(target)],
        capture_output=True, text=True, timeout=900, check=False)
    out = r.stdout + r.stderr
    assert "=== SoL table ===" in out, out[-2000:]
    assert "gpu__time_duration.sum" in out

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

"""Capture the hardware/software PROVENANCE for the Phase-D/E scaling tables.

  python -m benchmark.distributed.harness.scaling_env --cluster <site> --out <SC_OUTBASE>/env.json

Both scaling tables must document (plan §7 + CLAUDE.md): cluster + node type, GPU model/count,
driver/CUDA, NCCL/NVSHMEM versions, torch + cutlass-dsl versions, container image, transport (NVLink vs
IBGDA multi-rail), ``CPO_NVSHMEM_PROFILE``, clock-lock state, and the exact commit under test. Collecting
that by hand after the fact is how a table ends up with an unverifiable env block, so this runs INSIDE the
job's own container, on an allocated node, BEFORE the grid — and writes a JSON artifact every published
number can be traced back to.

Best-effort by construction: every probe is individually try/except'd and records ``null`` on failure. A
missing NCCL version must never abort a benchmark job that is holding a multi-node allocation.
"""
import argparse
import json
import os
import platform
import subprocess


def _sh(cmd, timeout=30):
    """Run a probe; return its output, or None if the command FAILED.

    The returncode check is load-bearing, not defensive tidiness. An earlier version fell back to stderr
    unconditionally, so on a non-git tree `git rev-parse HEAD` returned the string
    "fatal: not a git repository ..." — which is TRUTHY, so the REVISION fallback never fired and the
    results page would have published a git ERROR MESSAGE as its commit hash (MEASURED on the
    smoke). The same bug made `git status --porcelain` report a non-git tree as DIRTY. A failed command
    has no value; it must read as None so the caller's fallback can run.
    """
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        if p.returncode != 0:
            return None
        # stderr only as a fallback for tools that succeed but report on stderr (rc==0).
        return (p.stdout or p.stderr or "").strip() or None
    except Exception:
        return None


def _pkg(name):
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def collect(cluster=None):
    env = {}
    env["cluster"] = cluster or os.environ.get("CLUSTER")
    env["hostname"] = platform.node()
    env["python"] = platform.python_version()
    # Commit provenance. A cluster checkout is often an RSYNC'd tree, not a git clone -- `git rev-parse`
    # returns nothing there, which would silently publish a table with a null revision. Fall back to the
    # REVISION file the sync writes, and record WHICH source the value came from so the doc never implies a
    # verified git HEAD when it only had a text file.
    env["commit"] = _sh("git rev-parse HEAD")
    env["commit_short"] = _sh("git rev-parse --short HEAD")
    env["commit_source"] = "git" if env["commit"] else None
    if not env["commit"]:
        for cand in ("REVISION", os.path.join(os.environ.get("SC_REPO", ""), "REVISION")):
            try:
                with open(cand) as f:
                    rev = f.read().strip()
                if rev:
                    env["commit"] = rev
                    env["commit_short"] = rev[:9]
                    env["commit_source"] = f"REVISION file ({cand}) -- NOT a live git HEAD"
                    break
            except Exception:
                continue
    env["git_dirty"] = bool(_sh("git status --porcelain"))
    # --- SLURM / allocation shape -----------------------------------------------------------------
    env["slurm"] = {k: os.environ.get(k) for k in
                    ("SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_NNODES", "SLURM_NTASKS_PER_NODE",
                     "SLURM_JOB_PARTITION", "SLURM_JOB_ACCOUNT")}
    # --- GPU / driver ------------------------------------------------------------------------------
    env["gpu_name"] = _sh("nvidia-smi --query-gpu=name --format=csv,noheader | head -1")
    env["gpu_count"] = _sh("nvidia-smi --list-gpus | wc -l")
    env["driver"] = _sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1")
    # clock-lock state: an UNLOCKED clock is the single most common source of an unreproducible bench
    # median, so record it verbatim rather than asserting it.
    env["clocks"] = _sh("nvidia-smi --query-gpu=clocks.applications.graphics,clocks.max.graphics,"
                        "persistence_mode,power.limit --format=csv,noheader | head -1")
    env["nvlink"] = _sh("nvidia-smi topo -m | head -20")
    # --- CUDA / torch / cutlass --------------------------------------------------------------------
    env["nvcc"] = _sh("nvcc --version | tail -2 | head -1")
    try:
        import torch
        env["torch"] = torch.__version__
        env["torch_cuda"] = torch.version.cuda
        env["torch_nccl"] = ".".join(str(x) for x in torch.cuda.nccl.version())
        env["device_capability"] = list(torch.cuda.get_device_capability(0))
    except Exception:
        env.setdefault("torch", None)
    for p in ("nvidia-cutlass-dsl", "cutlass-dsl", "nvidia-nvshmem-cu13", "nvidia-nvshmem-cu12",
              "fold_cp_ops-kernels", "cuequivariance-ops-torch-cu13"):
        env[f"pkg::{p}"] = _pkg(p)
    env["nvshmem_info"] = _sh("nvshmem-info -a 2>/dev/null | head -5")
    # --- transport / launch env (what actually selected the fabric) --------------------------------
    env["transport_env"] = {k: v for k, v in os.environ.items()
                            if k.startswith(("NVSHMEM_", "NCCL_", "CPO_", "UCX_")) }
    env["container_image"] = os.environ.get("SC_CONTAINER_IMAGE") or os.environ.get("IMAGE")
    return env


def main():
    ap = argparse.ArgumentParser("scaling_env")
    ap.add_argument("--cluster", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    env = collect(a.cluster)
    blob = json.dumps(env, indent=2, default=str)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            f.write(blob)
        print(f"[scaling_env] wrote {a.out}")
    else:
        print(blob)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

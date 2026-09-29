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

"""Build-time dependency gate for the fold-cp-ops:26.07 image.

Runs as the last layer of docker/Dockerfile. Its job is to make a broken image fail at BUILD
time rather than silently uncover test cells on the cluster — which is exactly how 17 cells in
tests/test_trimul_autotune.py and tests/test_trimul_cp1_fallback.py went unmeasured on venue B
(`ModuleNotFoundError: cuequivariance_ops_torch`, reported as errors, easy to read as "no
coverage needed" rather than "coverage missing").

Three checks, and they fail for different reasons:

1. PRESENCE — every module and shared object the distributed + cp=1 paths need imports here.
   Three outcomes per dependency, and the middle one is the point:
     OK        imported here.
     DEFERRED  installed and its own shared objects resolve, but a full import needs libcuda.so.1
               — the CUDA DRIVER stub, which the container runtime injects only under `--gpus` and
               which is therefore absent during `docker build` by construction. Not a defect, and
               not silently passed either.
     FAILED    anything else -> non-zero exit, build stops.

2. PIN DRIFT — every ``==`` pin in ``pyproject.toml`` is still the version actually installed.
   The Dockerfile no longer carries version numbers of its own; it installs from that file, so
   this is what makes "the image matches the pin file" a checked claim rather than a comment.
   It is NOT a tautology despite pip having enforced the pins: this runs LAST, and every layer
   in between — the cuEquivariance install in particular — can resolve a different
   ``nvidia-nvshmem-cu13`` in behind it, silently, because pip is free to satisfy a later
   requirement by moving an earlier package.

3. TORCH PRESERVATION — the base image's torch is still the base image's torch.
   This is the check the design actually needs, and it guards the direction pin-drift cannot
   see. The image's contract is: take the container's torch (26.07 ships 2.13.0a0+…nv26.07,
   CUDA 13.3) and OVERRIDE its dependencies with our pinned cutlass-dsl / nvshmem — never the
   reverse. But torch declares its own exact nvshmem pin on PyPI (2.11 through 2.13 all say
   ``nvidia-nvshmem-cu13==3.4.5``, against our ``==3.7.0``), so a resolver handed both is free
   to resolve the contradiction by MOVING TORCH instead: a clean ``pip install`` of this
   project's dependency set into an empty env lands on torch **2.9.1** with the CUDA-12 stack.
   Pin drift would report that image as perfect — every pin exactly right — while the torch
   underneath had been swapped for one nobody tested, on the wrong CUDA major.
   NGC's torch declares no ``nvidia-*`` requirements, which is why the override succeeds here
   with no conflict to resolve. That is a property of this base, not of the design, so it is
   checked rather than relied upon.

The pin values are also readable via ``--print-pins``, which the Dockerfile uses to run its
explicit override pass without spelling a version anywhere but ``pyproject.toml``.
"""

import ctypes
import importlib
import importlib.metadata
import importlib.util
import os
import re
import sys
import tomllib

# The pin file the image was built from, copied in by docker/Dockerfile. Absent => the build was
# not produced by that Dockerfile, which is itself worth failing on rather than skipping past.
PYPROJECT = os.environ.get("CPO_IMAGE_GATE_PYPROJECT", "/opt/fold-cp-ops-deps/pyproject.toml")

# The base image's torch version, recorded by docker/Dockerfile BEFORE anything is installed.
# Absent => skip the preservation check with a printed note rather than fail: this script is also
# runnable by hand in an env that never had a "before", and a check that cannot run must say so
# instead of inventing a verdict.
TORCH_BEFORE = os.environ.get("CPO_IMAGE_GATE_TORCH_BEFORE", "/opt/torch_before.txt")

# (module, human label). Driver-dependent ones are allowed to come back DEFERRED.
MODULES = [
    ("cutlass", "CuTe-DSL"),
    ("cutlass.cute", "CuTe-DSL (cute)"),
    ("nvshmem.core", "nvshmem4py"),
    ("tvm_ffi", "apache-tvm-ffi"),
    ("pytest", "pytest"),
    ("cuequivariance_torch", "cuEquivariance (torch)"),
    ("cuequivariance_ops_torch", "cuEquivariance (ops-torch)"),
    ("cuequivariance_ops", "cuEquivariance (ops)"),
]

# (soname, label, substring that must appear in /proc/self/maps, or None to skip the check).
# The nvshmem check is not decoration: the base image ships host lib 3.6.5 while the kernels link
# 3.7.0 device bitcode, and a mismatch fails at NVSHMEM CULibrary init far from its cause.
SHARED_OBJECTS = [
    (
        "libnvshmem_host.so.3",
        "NVSHMEM host lib",
        "dist-packages/nvidia/nvshmem/lib/libnvshmem_host",
    ),
    ("libcue_ops.so", "cuEquivariance ops lib", None),
]


def pinned_versions(path: str) -> dict[str, str]:
    """Read the exact-version pins out of a pyproject's ``project.dependencies``.

    Purpose: give the drift check its expected values without writing any version number into
    this file or the Dockerfile — the pin file is the only place a version is spelled.

    Semantics: parses ``project.dependencies`` and keeps ONLY requirements of the form
    ``name==version`` (an optional ``[extra]`` between the two is stripped, so
    ``nvidia-cutlass-dsl[cu13]==4.4.2`` yields ``nvidia-cutlass-dsl -> 4.4.2``). Every other
    requirement — a bare name like ``torch``, a range like ``apache-tvm-ffi>=0.1.6,<0.2`` — is
    skipped on purpose: only an ``==`` pin makes a single installed version a checkable claim,
    and asserting anything about the others would be inventing a requirement the project does
    not make. Environment markers are not evaluated; a marked ``==`` pin would be checked
    unconditionally, which is wrong but cannot happen today because this project declares none.

    Args:
        path: absolute path to a pyproject.toml. Must exist and must contain
            ``[project] dependencies`` — a missing file means the image was not built by
            docker/Dockerfile, and a missing table means the pin file changed shape. Both
            raise rather than returning an empty dict, because an empty dict would make the
            drift check pass vacuously, which is the one outcome that must not be reachable
            by accident.

    Returns:
        dict mapping distribution name -> exact pinned version string.

    Raises:
        OSError: the pin file is absent.
        KeyError: it has no ``[project] dependencies``.
    """
    with open(path, "rb") as fh:
        deps = tomllib.load(fh)["project"]["dependencies"]
    pat = re.compile(r"^\s*([A-Za-z0-9._-]+)\s*(?:\[[^\]]*\])?\s*==\s*([A-Za-z0-9.*+!_-]+)\s*$")
    return {m.group(1): m.group(2) for m in (pat.match(d) for d in deps) if m}


PINS = pinned_versions(PYPROJECT)

# `--print-pins` emits one `name==version` per line and exits. docker/Dockerfile pipes this into its
# explicit override pass, so the override forces exactly the pinned versions without a version
# number appearing anywhere outside pyproject.toml. It must run BEFORE any import check, because at
# that point in the build the packages it names are the ones not yet installed.
if "--print-pins" in sys.argv:
    print("\n".join(f"{n}=={v}" for n, v in PINS.items()))
    sys.exit(0)

failed, deferred = [], []

# TORCH PRESERVATION. Compared as a string against what the base image had: pip is allowed to
# install anything it likes, and is NOT allowed to have moved torch to make our pins fit.
try:
    before = open(TORCH_BEFORE).read().strip()
except OSError:
    print(f"  SKIPPED   torch preservation        (no {TORCH_BEFORE}; not a Dockerfile build)")
else:
    after = importlib.metadata.version("torch")
    if after != before:
        failed.append(
            f"torch: the base image had {before}, the image now has {after}. pip MOVED torch to "
            "satisfy the pinned deps -- the override is meant to go the other way (take the "
            "container's torch, override ITS dependencies). A clean resolve of this dependency set "
            "lands on torch 2.9.1 with the CUDA-12 stack; that is what this check exists to refuse."
        )
    else:
        print(f"  OK        torch preserved              {after}")

for name, want in PINS.items():
    try:
        got = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        failed.append(f"{name}: pyproject pins =={want}, but it is NOT installed")
        continue
    if got != want:
        failed.append(f"{name}: pyproject pins =={want}, image has {got}")
    else:
        print(f"  OK        {name:<28} =={want}")

for mod, label in MODULES:
    if importlib.util.find_spec(mod.split(".")[0]) is None:
        failed.append(f"{label} ({mod}): not installed")
        continue
    try:
        importlib.import_module(mod)
        print(f"  OK        {label:<28} ({mod})")
    except ImportError as exc:
        if "libcuda.so.1" in str(exc):
            deferred.append(f"{label:<28} ({mod})")
        else:
            failed.append(f"{label} ({mod}): {exc}")

for soname, label, want in SHARED_OBJECTS:
    try:
        ctypes.CDLL(soname)
    except OSError as exc:
        msg = str(exc)
        # A CUDA library legitimately links libcuda.so.1. If the loader got far enough to complain
        # about the DRIVER, our own library was found — the failure is the absent build-time driver,
        # not a missing or mis-pathed wheel. Distinguish those two, because only one is our problem.
        if "libcuda.so.1" in msg and soname not in msg.split(":")[0]:
            deferred.append(f"{label:<28} ({soname})")
        else:
            failed.append(f"{label} ({soname}): {msg}")
        continue
    if want is not None and want not in open("/proc/self/maps").read():
        failed.append(f"{label} ({soname}): resolved, but NOT to the expected wheel ({want})")
    else:
        print(f"  OK        {label:<28} ({soname})")

for item in deferred:
    print(f"  DEFERRED  {item}   <- needs the GPU driver; verified at runtime")

if failed:
    sys.exit("IMAGE GATE FAILED:\n  " + "\n  ".join(failed))

print(
    "IMAGE GATE PASSED — pyproject pins hold; cute-dsl, nvshmem, nvshmem4py, cuEquivariance present"
)

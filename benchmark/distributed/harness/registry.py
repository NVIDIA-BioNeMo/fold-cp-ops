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

"""Registry (HARNESS_DESIGN §4). A target-MODULE registers a NAMED factory that returns its BenchTargets.
--targets/--baselines resolve names against this registry; the core imports zero kernels. Adding a kernel =
adding benchmark/distributed/harness/targets/<name>.py that calls register("<name>", factory)."""
from typing import Callable, List

REGISTRY: "dict[str, Callable[[], list]]" = {}


def register(name: str, factory: "Callable[[], list]"):
    """Register a factory (0-arg -> list[BenchTarget]) under `name`. Idempotent (last wins)."""
    REGISTRY[name] = factory


def resolve(names: List[str], *, role=None) -> list:
    """Flatten the requested module names' BenchTargets. If `role` given ("target"/"baseline"), force it on the
    returned targets (so --baselines x forces role='baseline' even if the module declared 'target'). Raises
    KeyError with the known set on an unknown name (loud, not silent)."""
    out = []
    for nm in names:
        nm = nm.strip()
        if not nm:
            continue
        if nm not in REGISTRY:
            raise KeyError(f"unknown target/baseline {nm!r}; registered: {sorted(REGISTRY)}")
        for t in REGISTRY[nm]():
            if role is not None:
                t.role = role
            out.append(t)
    return out

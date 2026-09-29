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

"""fold_cp_ops.distributed — multi-GPU / distributed kernels and infrastructure.

HARD STRUCTURAL RULE: ALL distributed / multi-GPU / in-kernel-NVSHMEM code lives under this
package (and ``benchmark/distributed/``, ``tests/distributed/``), kept separate from the local
single-GPU compute code.

**This module executes on ANY ``import fold_cp_ops.distributed.X``**, so it must re-export only
what is genuinely shipped API. Importing the kernel modules here would drag the whole subtree --
and every one of its heavy CuTe-DSL and NVSHMEM imports -- into a process that asked for one
helper. The bring-back adds names here one at a time, as each module lands and is tested; an empty
re-export list is correct while the subtree is being rebuilt, and is not an oversight.
"""

from fold_cp_ops.distributed.collective_symmetry import (
    BarrierTimeout,
    CollectiveError,
    CollectiveFailure,
    CollectiveGate,
    PeerFailure,
)
from fold_cp_ops.distributed.distributed_manager import DistributedManager
from fold_cp_ops.distributed.layout_map import LayoutLeftMap, LayoutMap, LayoutRightMap
from fold_cp_ops.distributed.pe_map import PeMap
from fold_cp_ops.distributed.trimul_weights import (
    W_KEYS,
    W_PROJ_BIAS_KEYS,
    weights_from_trimul_module,
)

__all__ = [
    "BarrierTimeout",
    "CollectiveError",
    "CollectiveFailure",
    "CollectiveGate",
    "DistributedManager",
    "LayoutLeftMap",
    "LayoutMap",
    "LayoutRightMap",
    "PeMap",
    "PeerFailure",
    "W_KEYS",
    "W_PROJ_BIAS_KEYS",
    "weights_from_trimul_module",
]

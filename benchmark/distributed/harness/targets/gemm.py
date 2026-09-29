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

"""A SECOND real target (HARNESS_DESIGN §7.3) — proves a different kernel plugs into the SAME driver with
zero core changes. `gemm` = the local K=N batched einsum (torch.bmm, the 2-kernel's GEMM leg) with a
DIFFERENT shape contract (supports asserts K==N == the back-TriMul HARD RULE). `nccl_a2a` = the pure NCCL
all_to_all reshard baseline. Both are ~20 lines each — that is the agnosticism payoff."""
import benchmark.distributed.back_a2a_store_bench as rp
from benchmark.distributed.harness.registry import register
from benchmark.distributed.harness.target import BenchTarget


def _gemm_supports(ctx):
    # K == N is the back-TriMul HARD RULE (a decoupled K would measure an O(N²) thin matmul). Also 16-B.
    ok, reason = rp._shape_check(ctx.N, ctx.cp0, ctx.cp1)
    return None if ok else reason


def _gemm_build(ctx):
    import torch
    L = ctx.Dloc * ctx.B
    N = ctx.N // ctx.cp0                              # per-rank M/K/N ~ N_loc (K==N_token by construction)
    dev = ctx.device
    A = torch.randn(L, N, N, device=dev, dtype=torch.bfloat16) / (N ** 0.5)
    Bt = torch.randn(L, N, N, device=dev, dtype=torch.bfloat16) / (N ** 0.5)
    return {"A": A, "Bt": Bt}


def _gemm_run(h):
    import torch
    torch.bmm(h["A"], h["Bt"].transpose(-1, -2))     # the TIMED O(N³) K=N einsum


def _nccl_build(ctx):
    import torch
    L = ctx.Dloc * ctx.B
    ni, nj, cp = ctx.N // ctx.cp0, ctx.N // ctx.cp1, ctx.cp0 * ctx.cp1
    send = torch.randn(cp, L * ni * nj, device=ctx.device, dtype=torch.bfloat16)
    recv = torch.empty_like(send)
    return {"send": send, "recv": recv}


def _nccl_run(h):
    import torch.distributed as dist
    dist.all_to_all_single(h["recv"], h["send"])     # the TIMED NCCL A2A reshard


def _gemm_factory():
    return [BenchTarget("gemm", build=_gemm_build, run=_gemm_run, role="target", supports=_gemm_supports)]


def _nccl_factory():
    return [BenchTarget("nccl_a2a", build=_nccl_build, run=_nccl_run, role="baseline", supports=_gemm_supports)]


register("gemm", _gemm_factory)
register("nccl_a2a", _nccl_factory)

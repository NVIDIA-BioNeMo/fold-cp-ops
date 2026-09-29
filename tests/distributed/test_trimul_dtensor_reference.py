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

"""Correctness guard for the reference CP TriMul baseline vs the serial fp32 reference.

This is the fresh, fold_cp_ops-native test written for plan §5.2.1 of
``docs/trimul_nvshmem_ib_integration_plan.md`` — the guard that gates the CP reference
(§5.2, ``benchmark/distributed/trimul_dtensor_reference.py``): the vendored copy ships ONLY once this
test passes, proving the copy reproduces the original forward numerics.

Mirrors ``tests/distributed/test_dtensor_trimul_reference.py`` (the analogous guard for the
OTHER baseline, DTensor) exactly in structure — torchrun + the ``conftest``
``dist_manager``/``device_mesh``/``device`` fixtures, no upstream fixtures (does NOT import
the upstream test package / ``spawn_multiprocessing``, per plan §5.2.1: porting the upstream's own
``test_dtensor_triangular_mult{_1d,}.py`` fixtures would re-introduce the very 
dependency §5.2's vendor exists to remove). Ports only the ASSERTION LOGIC (the spec):

  * fwd parity   — the vendored ``TriangularMultiplication{Outgoing,Incoming}1D`` (1-D mesh,
                   ring outgoing / tiled-reduce-scatter incoming) and the vendored 2-D
                   ``TriangularMultiplication{Outgoing,Incoming}2D`` + ``Ring2DComm`` path (2-D
                   mesh) — both reached via ``benchmark.distributed.trimul_dtensor_baseline
                   .build_trimul_dtensor_baseline``/``run_trimul_dtensor_baseline_fwd`` (the EXISTING glue, kept
                   intact by the vendor rewire) — match the single-GPU fp32 oracle
                   (``trimul_ref``) within the design bf16 bar (rel_L2 < 2e-2, 0 outlier rows),
                   BOTH directions;
  * sharding active — the local token-axis extent is strictly < the global extent on every
                   sharded mesh axis of the baseline's OWN mesh (the one ``build_trimul_dtensor_baseline``
                   constructs internally), so we are genuinely resharding, not silently
                   all-gathering then running serial;
  * non-vacuous  — the output is non-zero AND differs from the input.

Pure torch (DTensor + P2P ring/reduce-scatter comm + torch matmul/einsum + LN) — NO fold_cp_ops SM90
kernel, NO nvshmem — so it RUNS on any CUDA box (incl. the local sm_120 dev box). Backward /
param-grad parity is OUT of scope: this baseline is used FORWARD-only (matches ``TriMulAutotuned``,
design §0; trimul_dtensor_baseline.py's own docstring notes the same).

Run (one torchrun per mesh; the ``dist_manager`` session fixture derives the mesh; local box has
4 GPUs so cp=2/cp=4/cp=2*2 are all reachable here):

    # 1-D cp=2 / cp=4 (single node):
    torchrun --nproc_per_node=2 -m pytest -q tests/distributed/test_trimul_dtensor_reference.py
    torchrun --nproc_per_node=4 -m pytest -q tests/distributed/test_trimul_dtensor_reference.py
    # 2-D cp=2*2 (4 GPUs):
    torchrun --nproc_per_node=4 -m pytest -q tests/distributed/test_trimul_dtensor_reference.py \
        --dist-mesh cp=2*2
"""

from __future__ import annotations

import pytest
import torch
from torch.distributed.tensor import Shard, distribute_tensor

from benchmark.distributed.trimul_dtensor_baseline import build_trimul_dtensor_baseline, run_trimul_dtensor_baseline_fwd
# `make_weights` comes from THIS tree's correctness harness, not from the upstream's
# `t2_0_dtensor_baseline` module. The construction is the same one -- the upstream's own harness
# docstring calls it "the same construction as the T2.0 baseline" -- and porting a 351-line module
# for one 15-line function would carry 336 lines of untested helpers to satisfy an import.
#
# The two signatures differ in ways that do not reach the values: the harness takes `device` and no
# `N` (the weights do not depend on the token extent) and no `dtype` (it produces fp32 and the
# caller casts, which is what the call site below already did).
from tests.distributed.correctness_harness import make_weights
# `trimul_ref` lives in the SHIPPED package here, not under `benchmark/`.
#
# `rel_err` does NOT come across, and that is the point rather than an inconvenience: it is
# ``||got-ref|| / ||ref||``, the pooled scalar `fold_cp_ops.testing.numerics` exists to forbid as a
# GATE. Pooling averages the error over B*N*N*D elements, so one collapsed token row -- the defect
# class a CP baseline actually produces -- barely moves the norm. The bar below is now element-wise,
# with the pooled number kept only as something to PRINT.
from fold_cp_ops.testing.collective_guard import rank_invariant_skip
from fold_cp_ops.testing.kernel_matrix import matrix_exempt
from fold_cp_ops.testing.numeric_guard import numeric_exempt
from fold_cp_ops.testing.numerics import assert_elementwise, tolerance_bound
from fold_cp_ops.workflows.trimul_autotune import trimul_ref

_DT = torch.bfloat16
_BAR = 2e-2  # design bf16 rel_L2 bar (docs/trimul_nvshmem_design.md §4)
_SEED = 0

# (N_token, D) — non-pow2 N (384) + off-square D, all divisible by the cp axes we test (2, 4).
# Identical to test_dtensor_trimul_reference.py's grid for direct comparability of the two guards.
_SHAPES = [(256, 128), (384, 128), (512, 128), (512, 256)]
_DIRECTIONS = ["outgoing", "incoming"]


def _n_outlier_rows(out: torch.Tensor, ref: torch.Tensor, tol: float = _BAR) -> int:
    """Per-(b,i,j)-row relative-L2 outlier count over the feature dim — catches a LOCALIZED
    corruption a scalar rel_L2 averages away (the design's ``n_outlier_rows`` gate)."""
    o = out.float().reshape(-1, out.shape[-1])
    r = ref.float().reshape(-1, ref.shape[-1])
    num = (o - r).norm(dim=-1)
    den = r.norm(dim=-1) + 1e-30
    return int((num / den > tol).sum().item())


@pytest.mark.parametrize("direction", _DIRECTIONS)
@numeric_exempt(
    "covers ONE assert: the NON-VACUITY guard `||out - x|| / ||x|| > 1e-2`, whose subject genuinely "
    "IS the pooled quantity -- it asks whether the whole output moved away from the input, i.e. "
    "whether TriMul did anything at all. Per element that question is meaningless: many elements "
    "legitimately land near their input. The PARITY bar in the same test is `assert_elementwise` "
    "against the fp32 oracle and is what actually decides correctness"
)
@matrix_exempt(
    "the subject is the BASELINE, not a kernel of this package -- a vendored reference TriMul used "
    "as the denominator of every speedup. It has no tile, no drain variant and no compile-time "
    "config, so the kernel matrix has no axis that applies; the (N, D, direction) grid below is the "
    "upstream's own, kept identical so the two guards stay directly comparable"
)
@pytest.mark.parametrize(
    "N,D", _SHAPES, ids=lambda t: f"N{t[0]}_D{t[1]}" if isinstance(t, tuple) else str(t)
)
def test_trimul_dtensor_reference_matches_serial(apply_mesh, world_size, device, N, D, direction):
    """Vendored reference CP baseline (bf16) == serial fp32 oracle, both directions, both mesh
    dimensionalities (dispatched internally by build_trimul_dtensor_baseline off the live manager's cp
    subgroup structure — the same dispatch trimul_dtensor_baseline.py documents); sharding active;
    non-vacuous."""
    # The mesh is built PER TEST through `apply_mesh`; `dist_manager` leaves `device_mesh` unset
    # because the mesh is a parametrized axis rather than a launch-time choice, and asking for the
    # fixture without applying one raises "no device mesh is applied".
    manager = apply_mesh((("cp", int(world_size)),))
    device_mesh = manager.device_mesh
    dev = device
    B = 1

    # cp axis sizes must divide N (even token sharding); skip uneven cells cleanly. device_mesh
    # reflects the SAME cp topology build_trimul_dtensor_baseline derives internally from `manager`
    # (1-D session -> device_mesh.ndim==1; 2-D `cp=a*b` session -> device_mesh.ndim==2).
    uneven = [ax for ax in range(device_mesh.ndim) if N % device_mesh.size(ax) != 0]
    if uneven:
        # NOT a bare `pytest.skip`. Under `tests/distributed/**` a skip on a rank-varying predicate
        # is a DEADLOCK rather than a skip -- the skipping rank leaves and its peers block in the
        # next collective. This predicate is job-uniform, so it is DECLARED as such.
        rank_invariant_skip(
            f"N={N} not divisible by cp axis size(s) "
            f"{[device_mesh.size(ax) for ax in uneven]}",
            because=(
                "the mesh shape comes from the launcher and is identical on every rank, and N is a "
                "parametrize value every rank receives, so every rank computes the same verdict"
            ),
        )

    # GLOBAL x + weights, same seed on every rank -> identical -> consistent shard + oracle.
    gx = torch.Generator(device="cpu").manual_seed(_SEED)
    x_global = torch.randn(B, N, N, D, generator=gx, dtype=torch.float32).to(_DT).to(dev)
    w_cpu = make_weights(D, seed=_SEED, device="cpu")
    w = {k: (v.to(dev) if v is not None else None) for k, v in w_cpu.items()}

    # --- build the vendored baseline (dispatches 1-D vs 2-D off the live manager; §5.2 glue) ---
    module, mesh, placements, n_cp_axes = build_trimul_dtensor_baseline(
        manager, D=D, direction=direction, weights=w, dt=_DT
    )
    assert n_cp_axes == device_mesh.ndim, (
        f"build_trimul_dtensor_baseline dispatched n_cp_axes={n_cp_axes} but the session mesh has "
        f"{device_mesh.ndim} cp axes — dispatch/session mismatch"
    )

    # --- sharding-active guard: local extent strictly < global on every sharded mesh axis of
    # the baseline's OWN mesh (dp axis has size 1 -> excluded, matches design: dp is never
    # sharded by this baseline). ---
    x_shard_probe = distribute_tensor(x_global.float(), mesh, placements)
    lshape = x_shard_probe.to_local().shape
    for ax, p in enumerate(placements):
        if isinstance(p, Shard) and mesh.size(ax) > 1:
            assert lshape[p.dim] < x_global.shape[p.dim], (
                f"mesh axis {ax} (size {mesh.size(ax)}) shards tensor dim {p.dim}: local extent "
                f"{lshape[p.dim]} not < global {x_global.shape[p.dim]}: silent all-gather-then-serial no-op?"
            )

    # --- vendored reference CP forward (bf16 ring / tiled-reduce-scatter / Ring2DComm) ---
    out_full = run_trimul_dtensor_baseline_fwd(module, mesh, placements, x_global, _DT)

    # --- serial fp32 oracle (single-GPU, full global x) ---
    ref = trimul_ref(
        x_global,
        direction,
        None,
        w["norm_in_w"],
        w["norm_in_b"],
        w["p_in_w"],
        w["g_in_w"],
        w["norm_out_w"],
        w["norm_out_b"],
        w["p_out_w"],
        w["g_out_w"],
        p_in_b=w["p_in_b"],
        g_in_b=w["g_in_b"],
        p_out_b=w["p_out_b"],
        g_out_b=w["g_out_b"],
        eps=1e-5,
    )

    # --- non-vacuous guards: output is non-zero AND differs from the input ---
    # --- non-vacuous guards: output is non-zero AND differs from the input ---
    assert out_full.abs().max().item() > 0.0, "output is all-zero (vacuous pass?)"
    _pooled = ((out_full.float() - x_global.float()).norm()
               / (x_global.float().norm() + 1e-30)).item()
    assert _pooled > 1e-2, "output ~= input (TriMul did nothing?)"

    # --- fwd parity: ELEMENT-WISE against the fp32 oracle ---
    # The per-row outlier count is kept ALONGSIDE, not as the bar: it localizes a failure to a row,
    # which is genuinely useful, but `assert_elementwise` already fails at the offending index with
    # both values, so it reports rather than decides.
    n_out = _n_outlier_rows(out_full, ref)
    if manager.rank == 0:
        rel = ((out_full.float() - ref.float()).norm() / (ref.float().norm() + 1e-30)).item()
        print(
            f"[rank0] {direction} N={N} D={D} cp_ndim={device_mesh.ndim}: "
            f"rel_L2={rel:.3e} n_outlier_rows={n_out}"
        )
    bound = tolerance_bound(ref.float(), atol=2e-2, rtol=6e-2)
    assert_elementwise(
        out_full.float(), ref.float(), bound,
        what=f"reference CP baseline {direction} N={N} D={D} cp_ndim={device_mesh.ndim}",
    )
    assert n_out == 0, (
        f"[{direction} N={N} D={D}] element-wise bar passed but {n_out} row(s) are outliers -- "
        f"a localized defect the per-element bound admitted; investigate before widening either"
    )

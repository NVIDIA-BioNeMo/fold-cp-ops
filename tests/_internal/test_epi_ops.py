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


"""Tests for ``fold_cp_ops._internal.epi_ops`` -- the declarative epilogue terms.

Host-side. Each op is a small object whose job is to answer questions about itself (what params do I
need, how much SMEM, which axis do I broadcast along); those answers are what
``ComposableEpiMixin`` assembles, and they are checkable without a kernel.
"""

import os
import shutil
import subprocess
import sys

import pytest
import torch

from fold_cp_ops.testing.numerics import assert_elementwise, tolerance_bound
from fold_cp_ops._internal.epi_ops import (
    ColVecLoad,
    ColVecReduce,
    EpiOp,
    RowVecLoad,
    Scalar,
    SmemColVecBroadcast,
    TileStore,
    VecLoad,
)

_SM = torch.cuda.get_device_capability(0)[0] if torch.cuda.is_available() else 0
requires_sm90 = pytest.mark.skipif(_SM != 9, reason=f"needs sm_90; this GPU is sm_{_SM}0")


def test_the_name_is_the_key_in_all_three_roles():
    """An op's name addresses its argument, its param, and its per-subtile value.

    One string, three lookups -- a mismatch in any of them is a silently absent term, so the name is
    stored once rather than derived per role.
    """
    op = Scalar("alpha")
    assert op.name == "alpha"
    assert [n for n, _, _ in op.param_fields()] == ["alpha"]


def test_every_param_field_defaults_to_none():
    """An absent term must leave its params None, which is the state the machinery compiles out."""
    ops = (
        Scalar("a"),
        RowVecLoad("rv"),
        ColVecLoad("cv"),
        ColVecReduce("red"),
        SmemColVecBroadcast("s", slot=0),
    )
    for op in ops:
        for _name, _typ, default in op.param_fields():
            assert default is None, f"{op.name} declares a non-None default"


def test_row_and_column_vectors_broadcast_along_opposite_axes():
    """The zero stride IS the broadcast, so swapping the pair transposes which axis varies."""
    assert RowVecLoad("rv")._broadcast_stride() == (0, 1)
    assert ColVecLoad("cv")._broadcast_stride() == (1, 0)
    assert RowVecLoad("rv")._coord_idx() == 1
    assert ColVecLoad("cv")._coord_idx() == 0


def test_vector_tile_size_comes_from_the_matching_tile_dimension():
    """A row vector is sized by N and a column vector by M -- the same CTA tile, different entry."""
    tile = (128, 256, 64)
    assert RowVecLoad("rv")._tile_size(tile) == 256
    assert ColVecLoad("cv")._tile_size(tile) == 128


def test_broadcast_vectors_need_no_smem():
    """They are read GMEM -> registers directly, which is why the default epilogue's SMEM is empty."""
    for op in (RowVecLoad("rv"), ColVecLoad("cv")):
        assert op.smem_bytes(object(), (128, 256, 64), (64, 64)) == 0
        assert op.smem_struct_field(None, None) is None
        assert op.get_smem_tensor(None, None, None) is None
        assert op.needs_async_fence() is False


def test_tile_store_declares_four_params_all_defaulting_to_none():
    """An absent second output must leave atom, tensor, layout and tile all None together.

    Any one of them surviving as non-None would make the machinery build half a store.
    """
    fields = TileStore("post").param_fields()
    assert len(fields) == 4
    assert all(default is None for _n, _t, default in fields)
    names = {n for n, _t, _d in fields}
    assert "post" in names, "the tensor field must be named after the op"


def test_tile_store_keys_are_derived_from_the_name_so_two_stores_do_not_collide():
    """Several ``TileStore`` ops coexist in one params struct; their keys must stay distinct."""
    a, b = TileStore("postact"), TileStore("gate3")
    assert a._tma_atom_key() != b._tma_atom_key()
    assert a._smem_layout_key() != b._smem_layout_key()
    assert a._epi_tile_key() != b._epi_tile_key()
    assert a.name in a._tma_atom_key()


def test_tile_store_costs_nothing_when_its_tensor_is_absent():
    """An optional output that was not passed must not reserve SMEM for a stage."""
    assert TileStore("post").smem_bytes(None, (128, 256, 64), (64, 64)) == 0
    assert TileStore("post").tma_atoms(None, object()) == []


def test_subclass_hierarchy_is_what_the_shared_behaviour_hangs_off():
    """The variants specialize a vector load rather than reimplementing one."""
    assert issubclass(RowVecLoad, VecLoad) and issubclass(ColVecLoad, VecLoad)
    assert issubclass(VecLoad, EpiOp) and issubclass(Scalar, EpiOp)
    assert issubclass(TileStore, EpiOp) and issubclass(ColVecReduce, EpiOp)


def test_scalar_carries_an_optional_read_type():
    """A pointer-valued scalar is dereferenced as this type; a mismatch reinterprets the bits."""
    import cutlass

    assert Scalar("alpha").dtype is None
    assert Scalar("seed", dtype=cutlass.Int32).dtype is cutlass.Int32


def test_scalar_passes_its_argument_through_unchanged():
    """No stride assumption, no conversion -- the value or pointer reaches the params as it arrived."""
    from types import SimpleNamespace

    assert Scalar("alpha").to_params(None, SimpleNamespace(alpha=2.5)) == {"alpha": 2.5}
    assert Scalar("alpha").to_params(None, SimpleNamespace(alpha=None)) == {"alpha": None}
    with pytest.raises(AttributeError):
        Scalar("missing").to_params(None, SimpleNamespace())


def test_the_smem_colvec_allocates_nothing_because_the_scratch_is_the_mainloops():
    """It reads a buffer the MAINLOOP owns, so reserving bytes here would double-allocate it.

    That matters beyond the wasted SMEM: the reservation feeds the pipeline-depth computation, so an
    extra allocation would change the emitted schedule of a kernel that merely reads a statistic.
    """
    op = SmemColVecBroadcast("sRstd", slot=0)
    assert op.smem_bytes(object(), (128, 256, 64), (64, 64)) == 0
    assert op.smem_struct_field(None, None) is None
    assert op.get_smem_tensor(None, None, None) is None, (
        "it must NOT claim a tensor from storage.epi -- the functor supplies the decoupled scratch "
        "by overriding epi_get_smem_tensors"
    )
    assert op.needs_async_fence() is False


def test_the_smem_colvec_slot_selects_the_statistic():
    """Two ops over one scratch differ ONLY by slot, and the slot is carried, not inferred.

    A wrong slot broadcasts the other statistic -- finite, plausibly scaled, and wrong -- so the
    value is worth pinning rather than trusting to construction order.
    """
    assert SmemColVecBroadcast("sMean", slot=0).slot == 0
    assert SmemColVecBroadcast("sRstd", slot=1).slot == 1
    assert SmemColVecBroadcast("sRstd", slot=1).name == "sRstd"


def test_the_smem_colvec_still_declares_a_params_field():
    """Its data is device-side, but the composition machinery expects an entry per declared op."""
    fields = SmemColVecBroadcast("sRstd", slot=1).param_fields()
    assert [f[0] for f in fields] == ["sRstd"]


# ── the broadcast load's bounds ────────────────────────────────────────────────────────────────

#: One `gemm` whose row bias is 136 wide against a 128 tile, so the second N-tile has 120 lanes
#: past the vector's end. Kept as a string because it must run in a FRESH interpreter under
#: compute-sanitizer.
#:
#: **The allocation ORDER is load-bearing and is not cosmetic.** The bias is allocated BEFORE the
#: large output, so the bytes after it are unmapped. Allocate it after, and the over-read lands
#: inside the output's mapping and memcheck reports nothing -- measured both ways at this exact
#: shape: 1029 errors with this order, 0 with the bias last. Reordering these three lines silently
#: turns the test vacuous, which is worse than deleting it.
_OFFGRID_GEMM = (
    "import torch;"
    "from fold_cp_ops.kernels.gemm import gemm;"
    "r=lambda *s,d=torch.bfloat16: torch.randn(*s,device='cuda',dtype=d);"
    "A=r(1,512,128);B=r(1,136,128);"
    "rv=r(1,136,d=torch.float32);"
    "D=torch.empty(1,512,136,device='cuda',dtype=torch.bfloat16);"
    "gemm(A,B,D,None,None,tile_M=128,tile_N=128,rowvec_bias=rv,colvec_bias=None);"
    "torch.cuda.synchronize();print('RAN')"
)


@requires_sm90
@pytest.mark.skipif(shutil.which("compute-sanitizer") is None, reason="needs compute-sanitizer")
def test_a_broadcast_vector_shorter_than_its_tile_is_not_read_past():
    """`VecLoad` must not read past a broadcast vector whose extent does not fill the work tile.

    Purpose
        The op builds a ``tile_M x tile_N`` broadcast view of a ``(n,)`` or ``(m,)`` vector, so every
        lane of a partial trailing tile has a source coordinate past the vector's end. Predicating
        only the VALUE leaves those lanes issuing the load. Measured at exactly this shape:
        **1029 invalid 4-byte reads before the fix, 0 after** -- "5 bytes after the nearest
        allocation ... of size 544", which is the ``(1, 136)`` fp32 row bias.

    Semantics
        Asserted through ``compute-sanitizer memcheck``, because there is no in-process observable:
        the over-read returns garbage that the predicate discards, so the ANSWER is right either
        way and only the memory tool can tell. ``PYTORCH_NO_CUDA_MEMORY_CACHING=1`` is not optional
        -- with the caching allocator on, a 544-byte vector is served out of a live 2 MiB segment
        and the over-read lands INSIDE it, so memcheck reports nothing and the bug looks absent.
        That is why every earlier subset came back clean.

    Raises:
        AssertionError: If the child fails to run, or if memcheck reports any error at all. The
            count is not thresholded: one out-of-bounds read is the whole finding.
    """
    env = {**os.environ, "CPO_CACHE_ENABLED": "0", "PYTORCH_NO_CUDA_MEMORY_CACHING": "1"}
    proc = subprocess.run(
        ["compute-sanitizer", "--tool", "memcheck", sys.executable, "-c", _OFFGRID_GEMM],
        capture_output=True,
        text=True,
        timeout=1200,
        env=env,
    )
    summary = [ln for ln in proc.stdout.splitlines() if "ERROR SUMMARY" in ln]
    assert summary, f"no memcheck summary; is this really compute-sanitizer?\n{proc.stdout[-2000:]}"
    assert "ERROR SUMMARY: 0 errors" in summary[0], (
        f"{summary[0].strip()} -- a broadcast vector was read past its end. The load in "
        f"`VecLoad.begin` must stay inside the vector, not merely discard the value afterwards.\n"
        f"{proc.stdout[:4000]}"
    )


@requires_sm90
def test_an_off_grid_broadcast_bias_still_lands_in_the_right_places():
    """Predicating the load must not change WHICH output element each bias term reaches.

    A bounds fix that shifted the broadcast by one would leave memcheck happy and the answer wrong,
    so the arithmetic is pinned beside the safety: every row gets its row bias and every column its
    column bias, at extents that do not divide the tile on either axis.
    """
    from fold_cp_ops.kernels.gemm import gemm

    torch.manual_seed(0)
    m, n, k = 200, 136, 128
    A = torch.randn(1, m, k, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(1, n, k, device="cuda", dtype=torch.bfloat16)
    rv = torch.randn(1, n, device="cuda", dtype=torch.float32)
    cv = torch.randn(1, m, device="cuda", dtype=torch.float32)
    D = torch.empty(1, m, n, device="cuda", dtype=torch.bfloat16)
    gemm(A, B, D, None, None, tile_M=128, tile_N=128, rowvec_bias=rv, colvec_bias=cv)
    torch.cuda.synchronize()
    ref = (A.float() @ B.float().transpose(-1, -2)) + rv[:, None, :] + cv[:, :, None]
    assert_elementwise(D.float(), ref, tolerance_bound(ref, 2e-2, 2e-2), what="epilogue output")

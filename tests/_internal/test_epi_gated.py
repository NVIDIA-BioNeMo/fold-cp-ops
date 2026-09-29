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

"""Tests for ``fold_cp_ops._internal.epi_gated`` -- the dual-gated fold.

**What is tested here and what is tested elsewhere.** The fold's NUMERICS run on a GPU inside a
whole kernel, and `tests/kernels/test_dual_gated_gemm.py` gates them there -- including the
strongest available check, that the two weight layouts produce bit-identical results. Repeating that
here would duplicate coverage rather than add it.

What this file gates is the part a numeric test cannot see: the **geometry** the fold implies, and
the **seam structure** that keeps the LayerNorm fusions from forking it. Both are decided at trace
time from Python values, so both are testable without a GPU -- and both are exactly the things that
break silently. A wrong epilogue tile mis-partitions a store; a re-forked `epi_gate_preact` compiles
and runs and is only discovered when someone fixes the fold in one copy.

**The seam contract is asserted structurally, and that limit is deliberate.** No LayerNorm fusion
exists in this tree yet, so there is no real override to run. Asserting that `epi_visit_subtile`
DELEGATES to the two seams, and that nothing overrides the half that must be shared, is what can be
checked today; the behavioural half becomes checkable when the first fusion lands, and belongs in
that fusion's test file.
"""

import ast
import inspect
import textwrap
from pathlib import Path

import pytest

import fold_cp_ops._internal.epi_gated as epi_gated_mod
from fold_cp_ops._internal.epi_act import GemmActMixin
from fold_cp_ops._internal.epi_gated import GemmGatedMixin, gated_epi_tile_fn


class _Stub(GemmGatedMixin):
    """A bare `GemmGatedMixin` carrying only the attributes the geometry methods read.

    Sidesteps `GemmSm90.__init__` entirely: the methods under test read `chunk_g` and
    `tile_shape_mn` and nothing else, so constructing a whole kernel would add failure modes that
    have nothing to do with what is being asserted.

    `tile_shape_mn`, not `cta_tile_shape_mnk`: the override runs BEFORE the call parameters are
    bound (the epilogue tile is one of them), and `cta_tile_shape_mnk` is derived from those, so it
    does not exist yet at the hook's real call site. Same tile_N either way.

    Args:
        chunk_g: The weight layout to simulate.
        tile_n: The CTA tile's N extent over the 2N pre-activation.
    """

    def __init__(self, chunk_g=1, tile_n=128):
        self.chunk_g = chunk_g
        self.tile_shape_mn = (128, tile_n)


# ------------------------------------------------------------------ the halved store tile


def test_the_store_tile_is_half_the_pre_activation_tile():
    """The fold's defining geometric consequence: the output tile is half as wide."""
    assert gated_epi_tile_fn(None, (64, 32)) == (64, 16)
    assert gated_epi_tile_fn(None, (128, 64)) == (128, 32)


def test_the_store_tile_keeps_the_m_extent():
    """Only N is folded. Halving M as well would silently store a quarter of each tile."""
    for m in (32, 64, 128):
        assert gated_epi_tile_fn(None, (m, 64))[0] == m


# ------------------------------------------------------------------ the epilogue subtile


def test_the_element_interleave_layout_leaves_the_epilogue_tile_alone():
    """Adjacent-register pairing works at any subtile width, so nothing needs forcing."""
    assert _Stub(chunk_g=1).maybe_override_epi_tile((128, 64)) == (128, 64)
    assert _Stub(chunk_g=1).maybe_override_epi_tile((128, 16)) == (128, 16)


def test_the_block_interleave_layout_forces_one_block_per_subtile():
    """``epi_tile_n == 2*chunk_g`` is what makes the up/gate half-split a fact, not a hope.

    The fold pairs up register ``j`` with gate register ``j+H``. That identity holds only if one
    whole ``[up_G | gate_G]`` block sits inside one epilogue subtile, which is precisely what this
    override guarantees. Without it the pairing crosses a subtile boundary and combines columns
    from different output positions -- a wrong answer with no diagnostic.
    """
    assert _Stub(chunk_g=16, tile_n=128).maybe_override_epi_tile((128, 64)) == (128, 32)
    assert _Stub(chunk_g=32, tile_n=128).maybe_override_epi_tile((128, 64)) == (128, 64)


@pytest.mark.parametrize("bad", [2, 8, 15, 17, 24])
def test_a_chunk_narrower_than_the_store_atom_is_refused(bad):
    """8 is the tempting value -- half an stmatrix atom -- and it is exactly the broken one."""
    with pytest.raises(AssertionError, match=r"multiple of 16"):
        _Stub(chunk_g=bad).maybe_override_epi_tile((128, 64))


def test_a_tile_that_splits_a_block_across_subtiles_is_refused():
    """``tile_N`` must be a multiple of ``2*chunk_g``, or a block straddles two subtiles."""
    with pytest.raises(AssertionError, match=r"multiple of 2\*chunk_g"):
        _Stub(chunk_g=16, tile_n=48).maybe_override_epi_tile((128, 48))


# ------------------------------------------------------------------ the seam structure


def _method_source(cls, name):
    """The dedented source of one method, for structural assertions."""
    return inspect.getsource(getattr(cls, name))


def test_epi_visit_subtile_delegates_to_both_seams_rather_than_inlining_them():
    """The delegation IS the extension point; inlining either seam removes it.

    A LayerNorm fusion overrides `epi_combine_preact` and inherits everything else. If
    `epi_visit_subtile` ever formed the pre-activation itself, that override would stop being
    reachable -- and nothing would fail, because the inlined version computes the same thing for
    the un-fused kernel.
    """
    src = _method_source(GemmGatedMixin, "epi_visit_subtile")
    assert "self.epi_combine_preact(" in src
    assert "self.epi_gate_preact(" in src
    assert "self.epi_apply_postact_mask(" in src


def test_the_three_steps_happen_in_the_order_the_arithmetic_requires():
    """combine -> gate -> mask. Each swap is a different function, not a rounding difference.

    Masking before the gate computes ``sigmoid(mask*g)``; biasing after it scales the bias by the
    gate. Both produce plausible numbers, which is why the order is pinned rather than trusted.
    """
    src = _method_source(GemmGatedMixin, "epi_visit_subtile")
    order = [
        src.index("self.epi_combine_preact("),
        src.index("self.epi_gate_preact("),
        src.index("self.epi_apply_postact_mask("),
    ]
    assert order == sorted(order), "the fold's three steps are out of order"


def test_nothing_in_the_package_overrides_the_shared_fold():
    """`epi_gate_preact` is the piece both upstream fusions duplicated; it must stay shared.

    An AST scan over the package rather than a runtime check, because the failure this guards
    against is a FUTURE bring-back re-forking the fold -- at which point there would be two copies
    that agree until someone fixes one of them.
    """
    root = Path(epi_gated_mod.__file__).parent.parent
    offenders = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
            if cls.name == "GemmGatedMixin":
                continue
            for fn in [n for n in cls.body if isinstance(n, ast.FunctionDef)]:
                if fn.name == "epi_gate_preact":
                    offenders.append(f"{path.name}:{cls.name}")
    assert not offenders, (
        "the shared fold is overridden by "
        + ", ".join(offenders)
        + ". Override epi_combine_preact instead -- that is the seam; duplicating the fold is the "
        "regression these seams exist to prevent."
    )


def test_the_mask_indexing_differs_between_the_two_layouts():
    """The fold breaks the 1:1 index correspondence the base mixin's mask assumes.

    In the element-interleave layout output ``i`` comes from pre-activation pair ``(2i, 2i+1)``, so
    the mask must be read at ``2*i``; in the block-interleave layout it comes from up column ``i``
    directly. A single indexing rule cannot serve both, which is why this method is overridden at
    all -- and this test is what says so.
    """
    base = _method_source(GemmActMixin, "epi_apply_postact_mask")
    gated = _method_source(GemmGatedMixin, "epi_apply_postact_mask")
    assert "tDrMask[i]" in base and "tDrMask[2 * i]" not in base
    assert "tDrMask[2 * i]" in gated and "self.chunk_g" in gated


def test_the_register_permute_is_skipped_for_the_block_interleave_layout():
    """Running it there would scramble correctly-owned data, with no error.

    The permute exists to fix ownership after the element-interleave fold compresses a column pair.
    In the block-interleave layout the surviving column was never moved, so it is already owned by
    the lane ``stmatrix`` expects.
    """
    src = _method_source(GemmGatedMixin, "epi_convert_postact")
    assert "self.chunk_g == 1" in src, "the permute must be gated on the element-interleave layout"
    assert "permute_gated_Cregs_b16(" in src


def test_the_gate_is_inherited_from_the_post_activation_mixin():
    """The store, its mask plumbing and its dtype conversion are shared, not restated."""
    assert issubclass(GemmGatedMixin, GemmActMixin)
    # Inherited unchanged -- if these were overridden the store would have two implementations.
    assert GemmGatedMixin.epi_setup_postact is GemmActMixin.epi_setup_postact
    assert GemmGatedMixin._latch_postact_attributes is GemmActMixin._latch_postact_attributes


def _method_body(cls, name):
    """One method's source with its DOCSTRING removed.

    Purpose
        The assertions below include negative ones -- "this must NOT appear here" -- and the
        docstrings deliberately discuss the very names being excluded. A raw source search matches
        the prose, so the check would fail on correct code and pass on code that merely stopped
        explaining itself.

    Args:
        cls: The class holding the method.
        name: The method's name.

    Returns:
        The method body as source, docstring stripped.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(getattr(cls, name)))).body[0]
    body = tree.body[1:] if ast.get_docstring(tree) is not None else tree.body
    return "\n".join(ast.unparse(n) for n in body)


def test_the_params_dict_is_an_extract_method_the_lowering_calls():
    """`epi_to_underlying_arguments` delegates its whole body, so a subclass can extend the dict.

    The fused-LayerNorm epilogue adds ONE field (``eps``) to an otherwise identical params struct.
    Without this seam it would have to restate the geometry assertions, the post-activation latch
    and the halved store tile to do so -- and a restated invariant is one that drifts. The extract
    is checked structurally because the default behaviour is unchanged either way, so no numeric
    test can tell whether it happened.
    """
    lowering = _method_body(GemmGatedMixin, "epi_to_underlying_arguments")
    assert "self.gated_params_dict(args)" in lowering
    helper = _method_body(GemmGatedMixin, "gated_params_dict")
    for moved in (
        "_latch_postact_attributes",
        "cta_tile_shape_postact_mn",
        "_epi_ops_to_params_dict",
    ):
        assert moved in helper, f"{moved} belongs to the extracted helper"
        assert moved not in lowering, f"{moved} must not be duplicated back into the lowering"

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


"""Tests for ``fold_cp_ops._internal.epi_composable`` -- generating epilogue hooks from a declaration.

All host-side: what this module does is build classes and dicts at class-creation time, so it can be
exercised without a GPU or an MLIR context.
"""

import pytest

from fold_cp_ops._internal.epi_composable import (
    ComposableEpiMixin,
    _compute_smem_map,
    _make_epi_params,
)
from fold_cp_ops._internal.epi_ops import ColVecLoad, RowVecLoad, Scalar


def test_smem_map_skips_scalars_and_numbers_the_rest_in_order():
    """Scalars need no SMEM, so they must not consume an index.

    The map indexes into the tuple ``epi_get_smem_tensors`` returns, which itself skips scalars --
    counting them here would hand every later op its neighbour's buffer.
    """
    ops = (Scalar("alpha"), RowVecLoad("rv"), Scalar("beta"), ColVecLoad("cv"))
    assert _compute_smem_map(ops) == {"rv": 0, "cv": 1}


def test_generated_params_put_required_fields_before_defaulted_ones():
    """A dataclass cannot declare a required field after a defaulted one, so the order is forced."""
    params_cls = _make_epi_params((Scalar("alpha"), RowVecLoad("rv")), (), ())
    names = list(params_cls.__dataclass_fields__)
    assert names == ["alpha", "rv"]
    params_cls()  # every field defaults, so this must construct with no arguments


def test_subclass_declaration_generates_params_and_the_smem_map():
    """Declaring ``_epi_ops`` is the single statement that produces the whole plumbing."""

    class Mixin(ComposableEpiMixin):
        _epi_ops = (Scalar("alpha"), RowVecLoad("rv"))

    assert Mixin._epi_smem_map == {"rv": 0}
    assert set(Mixin.EpilogueParams.__dataclass_fields__) == {"alpha", "rv"}


def test_a_subclass_with_its_own_params_keeps_them():
    """An explicitly declared ``EpilogueParams`` is never overwritten by the generated one."""

    class Explicit(ComposableEpiMixin):
        _epi_ops = (Scalar("alpha"),)

        class EpilogueParams:
            pass

    assert Explicit.EpilogueParams.__name__ == "EpilogueParams"
    assert not hasattr(Explicit.EpilogueParams, "__dataclass_fields__")


def test_params_are_regenerated_for_a_subclass_that_changes_its_ops():
    """Inheriting a parent's params would mismatch the fields the child's ops write.

    The check is on ``cls.__dict__`` and not ``hasattr``, which is what makes this work: the child
    inherits a params class from the parent, and must still get its own.
    """

    class Parent(ComposableEpiMixin):
        _epi_ops = (Scalar("alpha"),)

    class Child(Parent):
        _epi_ops = (Scalar("alpha"), RowVecLoad("rv"))

    assert set(Child.EpilogueParams.__dataclass_fields__) == {"alpha", "rv"}
    assert Child.EpilogueParams is not Parent.EpilogueParams


def test_a_subclass_declaring_no_ops_inherits_the_empty_defaults():
    """No declaration, no generation -- the base's empty class attributes stand."""

    class Bare(ComposableEpiMixin):
        pass

    assert Bare._epi_ops == ()
    assert Bare._epi_smem_map == {}
    assert not Bare._epi_has_async_ops


def test_async_fence_flag_reflects_whether_any_op_needs_one():
    """``epi_begin`` emits ONE commit/wait/barrier for all cp.async ops, gated on this flag."""

    class NoAsync(ComposableEpiMixin):
        _epi_ops = (Scalar("alpha"), RowVecLoad("rv"))

    assert NoAsync._epi_has_async_ops is False


def test_extra_param_fields_are_merged_into_the_generated_params():
    """Non-op params (an activation function, say) join the generated struct rather than a second one."""

    class WithExtra(ComposableEpiMixin):
        _epi_ops = (Scalar("alpha"),)
        _extra_param_fields = (("act_fn", object, None),)

    assert set(WithExtra.EpilogueParams.__dataclass_fields__) == {"alpha", "act_fn"}


def test_params_to_dict_merges_every_op(monkeypatch):
    """``_epi_ops_to_params_dict`` is the host-side seam subclasses build their params from."""
    from types import SimpleNamespace

    class Mixin(ComposableEpiMixin):
        _epi_ops = (Scalar("alpha"), Scalar("beta"))

    d = Mixin()._epi_ops_to_params_dict(SimpleNamespace(alpha=1.5, beta=None))
    assert d == {"alpha": 1.5, "beta": None}
    with pytest.raises(AttributeError):
        Mixin()._epi_ops_to_params_dict(SimpleNamespace(alpha=1.5))

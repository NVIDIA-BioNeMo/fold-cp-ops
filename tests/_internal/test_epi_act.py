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

"""Tests for ``fold_cp_ops._internal.epi_act`` -- the post-activation (second output) epilogue.

What this file gates is the mixin's **contract**: which post-activation tensors it accepts, which it
refuses and with what message, and what it latches onto the functor for the store leg to read. Those
are decided at trace time from Python values, so they need no GPU -- and they are the checks that,
if missing, turn into a mis-partitioned store or a silently wrong dtype rather than an error.

The mixin's numerics run inside a whole kernel and are gated there:
`tests/kernels/test_dual_gated_gemm.py` exercises this store on every supported shape through the
gated subclass. Duplicating that here would add runtime, not coverage.

**Why the refusals matter more than they look.** Each one guards a case that otherwise fails several
frames deep, in a message naming an MLIR symbol rather than the argument the caller passed. An fp32
post-activation, for instance, has no ``stmatrix`` form at all -- without the assert it surfaces from
inside atom construction.
"""

import pytest

from fold_cp_ops._internal.epi_act import GemmActMixin
from fold_cp_ops._internal.epi_default import GemmDefaultEpiMixin
from fold_cp_ops._internal.epi_ops import ColVecLoad, TileStore
from fold_cp_ops._internal.rounding import RoundingMode


class _FakeLayout:
    """Stands in for ``cutlass.utils.LayoutEnum``, which needs a real tensor to construct."""

    def __init__(self, major):
        self._major = major

    def is_n_major_c(self):
        return self._major == "n"

    def is_m_major_c(self):
        return self._major == "m"


class _FakeTensor:
    """A post-activation tensor stand-in carrying only what the contract inspects."""

    def __init__(self, width=16, major="n"):
        self.element_type = type("T", (), {"width": width})
        self._major = major


class _FakeArgs:
    """The subset of `EpilogueArguments` that :meth:`_latch_postact_attributes` reads."""

    def __init__(self, width=16, major="n", rounding_mode=RoundingMode.RN):
        self.mPostAct = _FakeTensor(width, major)
        self.rounding_mode = rounding_mode


class _Stub(GemmActMixin):
    """A bare mixin instance with the one attribute the latch reads off the kernel."""

    def __init__(self, tile=(128, 256, 64)):
        self.cta_tile_shape_mnk = tile


@pytest.fixture
def latch(monkeypatch):
    """Call `_latch_postact_attributes` with `LayoutEnum.from_tensor` stubbed out.

    The real one needs a live cute tensor. Stubbing it is what lets the CONTRACT -- which is about
    majorness, not about how majorness is discovered -- be tested without a GPU.

    Returns:
        A callable ``(stub, args) -> None`` that runs the latch.
    """
    import cutlass

    monkeypatch.setattr(
        cutlass.utils.LayoutEnum, "from_tensor", staticmethod(lambda t: _FakeLayout(t._major))
    )
    return lambda stub, args: GemmActMixin._latch_postact_attributes(stub, args)


def test_the_declared_terms_extend_the_default_epilogue_rather_than_replacing_it():
    """The default terms must survive: a post-activation GEMM still takes alpha, beta and biases."""
    names = [op.name for op in GemmActMixin._epi_ops]
    for op in GemmDefaultEpiMixin._epi_ops:
        assert op.name in names, f"{op.name} was dropped from the post-activation epilogue"
    assert names[-1] == "mPostAct", "the tile store must be declared last, after every load"


def test_the_mask_is_a_column_vector_and_the_output_is_a_tile_store():
    """Two different kinds of term; declaring either as the other silently changes the plumbing."""
    by_name = {op.name: op for op in GemmActMixin._epi_ops}
    assert isinstance(by_name["mMaskColVec"], ColVecLoad)
    assert isinstance(by_name["mPostAct"], TileStore)


def test_the_activation_is_a_constexpr_parameter_not_a_loaded_tensor():
    """`act_fn` is folded into the kernel, so it is a param field rather than an ``_epi_ops`` entry.

    If it were an op the composition would try to build SMEM and a TMA atom for a Python callable.
    """
    assert ("act_fn", __import__("cutlass").Constexpr, None) in GemmActMixin._extra_param_fields
    assert "act_fn" not in [op.name for op in GemmActMixin._epi_ops]


def test_the_latch_records_what_the_store_leg_reads(latch):
    """The store needs the dtype, the majorness and the CTA tile; all three come off the tensor."""
    stub = _Stub(tile=(128, 256, 64))
    latch(stub, _FakeArgs())
    assert stub.postact_dtype.width == 16
    assert stub.postact_layout.is_n_major_c()
    assert stub.cta_tile_shape_postact_mn == (128, 256), (
        "the un-gated post-activation tile is the FULL CTA tile; only a gate halves it"
    )


def test_an_m_major_post_activation_is_accepted(latch):
    """The transposed store is a supported layout, not a workaround -- it feeds a batched GEMM."""
    stub = _Stub()
    latch(stub, _FakeArgs(major="m"))
    assert stub.postact_layout.is_m_major_c()


def test_a_thirty_two_bit_post_activation_is_refused(latch):
    """There is no 32-bit ``stmatrix``; without this the failure comes from inside atom building."""
    with pytest.raises(AssertionError, match=r"must be 16-bit"):
        latch(_Stub(), _FakeArgs(width=32))


def test_a_layout_that_is_neither_major_is_refused(latch):
    """A tensor major in some third mode would partition wrongly rather than fail."""
    with pytest.raises(AssertionError, match=r"n-major .* or m-major"):
        latch(_Stub(), _FakeArgs(major="k"))


def test_stochastic_rounding_is_refused_here_rather_than_ignored(latch):
    """SM90 has no stochastic-rounding epilogue, so accepting the flag would silently give RN.

    Refused inside the mixin, not only at the front door, so a caller building
    `EpilogueArguments` directly cannot slip past it.
    """
    with pytest.raises(AssertionError, match=r"RoundingMode.RN only"):
        latch(_Stub(), _FakeArgs(rounding_mode=RoundingMode.RS))


def test_the_conversion_has_no_rounding_mode_branch():
    """With RS refused, a mode branch in the converter would be unreachable code in a hot loop."""
    import inspect

    src = inspect.getsource(GemmActMixin.epi_convert_postact)
    assert "RoundingMode.RS" not in src and "convert_f32_to_bf16_sr" not in src


def test_the_mask_multiplies_and_never_adds():
    """`mMaskColVec` is multiplicative; `mColVecBroadcast` is additive. Confusing them is silent.

    Both are column vectors broadcast along N, so nothing about their shapes distinguishes them --
    only which operator the epilogue applies, which is what this pins.
    """
    import inspect

    src = inspect.getsource(GemmActMixin.epi_apply_postact_mask)
    assert "* tDrMask[i]" in src
    assert "+ tDrMask" not in src

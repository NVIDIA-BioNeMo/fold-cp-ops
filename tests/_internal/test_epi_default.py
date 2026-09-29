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


"""Tests for ``fold_cp_ops._internal.epi_default`` -- the default GEMM epilogue mixin.

Host-side: the mixin's declaration is what generates its params and SMEM plumbing, so the
declaration is what can be checked without a kernel. The arithmetic it performs is covered by
``tests/kernels/test_gemm.py``, which compares the assembled result against a torch reference.
"""

from fold_cp_ops._internal.epi_composable import ComposableEpiMixin
from fold_cp_ops._internal.epi_default import GemmDefaultEpiMixin
from fold_cp_ops._internal.epi_ops import ColVecLoad, RowVecLoad, Scalar
from fold_cp_ops._internal.rounding import RoundingMode


def test_it_declares_exactly_the_five_terms_the_public_api_exposes():
    """alpha, beta, the stochastic-rounding seed, and the two broadcast vectors.

    The declaration is the single source of the params struct and the SMEM map, so an op added here
    without a matching ``EpilogueArguments`` field produces a params field nothing ever fills.
    """
    ops = GemmDefaultEpiMixin._epi_ops
    assert [type(op) for op in ops] == [Scalar, Scalar, Scalar, RowVecLoad, ColVecLoad]
    assert [op.name for op in ops] == [
        "alpha",
        "beta",
        "sr_seed",
        "mRowVecBroadcast",
        "mColVecBroadcast",
    ]


def test_every_declared_op_has_a_matching_argument_field():
    """The op name addresses its argument, its param and its per-subtile value -- one string."""
    fields = set(GemmDefaultEpiMixin.EpilogueArguments._fields)
    for op in GemmDefaultEpiMixin._epi_ops:
        assert op.name in fields, f"{op.name} has no EpilogueArguments field"


def test_arguments_default_to_absent_so_every_term_compiles_out():
    """A term is absent from the kernel, not multiplied by one -- which is why None is the default."""
    args = GemmDefaultEpiMixin.EpilogueArguments()
    assert args.alpha is None and args.beta is None
    assert args.mRowVecBroadcast is None and args.mColVecBroadcast is None
    assert args.add_to_output is False
    assert args.rounding_mode == RoundingMode.RN


def test_the_broadcast_vectors_need_no_smem():
    """They are read GMEM -> registers, which is why the default epilogue's SMEM struct is empty."""
    assert (
        GemmDefaultEpiMixin.epi_smem_bytes_per_stage(
            GemmDefaultEpiMixin.EpilogueArguments(), (128, 256, 64), (64, 64)
        )
        == 0
    )


def test_params_are_generated_from_the_declaration():
    """Declaring ``_epi_ops`` is the one statement that produces the params struct."""
    assert issubclass(GemmDefaultEpiMixin, ComposableEpiMixin)
    assert set(GemmDefaultEpiMixin.EpilogueParams.__dataclass_fields__) == {
        "alpha",
        "beta",
        "sr_seed",
        "mRowVecBroadcast",
        "mColVecBroadcast",
    }


def test_the_mixin_does_not_import_a_kernel():
    """It lives in ``_internal`` and must not depend on ``kernels`` -- the layering is one-way.

    ``GemmDefaultSm90`` composes the two in ``kernels/gemm.py``; putting that composition here would
    make an epilogue building block import a kernel.
    """
    import inspect

    src = inspect.getsource(__import__("fold_cp_ops._internal.epi_default", fromlist=["x"]))
    assert "fold_cp_ops.kernels" not in src

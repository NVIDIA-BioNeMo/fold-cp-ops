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

"""Tests for ``fold_cp_ops._internal.activation`` -- the fused epilogue activations.

These functions run in registers inside a kernel, so they cannot be called from the host; the
device probe below is the smallest thing that evaluates one elementwise and hands the result back.
It is a bare ``out[m, t] = fn(x[m, t])`` with one thread per element -- no tiling, no predication,
no shared memory -- so a failure here is a failure of the arithmetic and of nothing else.

**Two kinds of gate, deliberately.** Against torch the comparison must carry a tolerance, because
``tanh.approx.f32`` guarantees only a ``2**-11`` ABSOLUTE error while torch's tanh is correctly
rounded -- a tolerance test can only say "close enough". Against *itself* the comparison is exact:
``glu(x, y)`` must be bit-identical to ``sigmoid(x) * y`` because that is literally how it is
defined, and a `torch.equal` there catches a mis-paired argument or a reassociated product that a
tolerance would absorb. Use the exact gate wherever the relationship is definitional.

**The tolerances are derived, not tuned.** ``_TANH_ABS`` is the PTX ISA's documented bound for the
instruction, and the sigmoid bound is half of it because ``sigmoid = 0.5 + 0.5*tanh(0.5*x)`` scales
the error by 0.5. Nothing here was widened to make a run pass -- if one of these fails, the emitted
instruction changed, and the bound is the thing that should be re-derived rather than the test.
"""

import ast
from pathlib import Path
from typing import Optional

import cuda.bindings.driver as cuda
import pytest
import torch

import cutlass.cute as cute
from cutlass import Float32, const_expr

import fold_cp_ops._internal.activation as activation_mod
from fold_cp_ops.testing.numerics import assert_elementwise, tolerance_bound
from fold_cp_ops._internal.activation import (
    act_fn_map,
    as_gate_fn,
    gate_fn_map,
    glu,
    sigmoid,
    tanh,
)
from fold_cp_ops._internal.compile_time.compile_utils import make_fake_tensor as fake_tensor

#: Threads per block, and therefore the probe's row width. 256 is one full block of the shape the
#: epilogue itself runs at; nothing here depends on the value except that it be a multiple of the
#: 32-lane warp, since the activations are per-lane and must not straddle a partial warp.
_ROW = 256

#: Maximum absolute error the PTX ISA guarantees for ``tanh.approx.f32``. This is the bound the
#: hardware promises, not a tolerance chosen to make a run pass.
_TANH_ABS = 2.0**-11

#: The bound `sigmoid` inherits: ``0.5 + 0.5*tanh(0.5*x)`` scales tanh's error by 0.5.
_SIGMOID_ABS = _TANH_ABS / 2


class _Apply:
    """Device probe: apply one activation elementwise to a ``(M, _ROW)`` fp32 tensor.

    One thread per element and a grid of exactly ``M`` blocks, so there is no predication and no
    partial warp -- the activation is the only thing under test.

    Args:
        fn: The activation, baked in as a compile-time constant exactly as a real epilogue bakes
            ``EpilogueArguments.act_fn``. Must take either one or two `Float32` arguments,
            matching `binary`.
        binary: Whether `fn` is a gate (two pre-activations in, one out) rather than an
            elementwise activation. Must agree with `fn`'s arity; a mismatch is a `TypeError` at
            trace time, not a wrong answer.
    """

    def __init__(self, fn, binary: bool = False):
        self.fn = fn
        self.binary = binary

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mY: Optional[cute.Tensor],
        mOut: cute.Tensor,
        stream: cuda.CUstream,
    ):
        """Launch one block per row, one thread per element.

        Args:
            mX: ``(M, _ROW)`` fp32 input. M is symbolic; ``_ROW`` is baked in as the block width.
            mY: ``(M, _ROW)`` fp32 second input for a gate, or None for a unary activation. Must
                be non-None exactly when ``self.binary``.
            mOut: ``(M, _ROW)`` fp32 output, fully overwritten.
            stream: The launch stream.
        """
        self.kernel(mX, mY, mOut).launch(
            grid=[mX.shape[0], 1, 1], block=[_ROW, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(self, mX: cute.Tensor, mY: Optional[cute.Tensor], mOut: cute.Tensor):
        """One element per thread: ``mOut[block, thread] = fn(mX[...]) `` (or ``fn(mX, mY)``)."""
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        if const_expr(self.binary):
            mOut[bidx, tidx] = self.fn(mX[bidx, tidx], mY[bidx, tidx])
        else:
            mOut[bidx, tidx] = self.fn(mX[bidx, tidx])


def _run(fn, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Compile and run the probe over `x` (and `y` for a gate), returning a fresh fp32 tensor.

    Args:
        fn: The activation to bake in.
        x: ``(M, _ROW)`` fp32 CUDA tensor. Must be contiguous -- the probe indexes it directly, so
            a non-contiguous view would be read as if it were contiguous.
        y: Second operand for a gate, same shape/dtype/device as `x`, or None.

    Returns:
        A new ``(M, _ROW)`` fp32 CUDA tensor holding the elementwise result.
    """
    assert x.is_contiguous() and x.dtype == torch.float32 and x.shape[1] == _ROW
    out = torch.empty_like(x)
    m = cute.sym_int()
    fake = fake_tensor(Float32, (m, _ROW))
    compiled = cute.compile(
        _Apply(fn, binary=y is not None),
        fake,
        fake if y is not None else None,
        fake,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )
    compiled(x, y, out)
    return out


@pytest.fixture(scope="module")
def sweep() -> torch.Tensor:
    """A ``(M, _ROW)`` fp32 sweep covering the ranges each activation has to behave over.

    Spans the saturating tails (where ``tanh`` clamps to +-1 and a naive ``exp(-x)`` sigmoid would
    overflow), the linear region near zero, and exact zero. Returned on CUDA, contiguous.
    """
    torch.manual_seed(0)
    rows = [
        torch.linspace(-40.0, 40.0, _ROW),
        torch.linspace(-1e-3, 1e-3, _ROW),
        torch.randn(_ROW) * 4.0,
        torch.zeros(_ROW),
    ]
    return torch.stack(rows).cuda().contiguous()


def test_tanh_matches_torch_within_the_documented_hardware_bound():
    """Anything outside ``2**-11`` absolute means a different instruction is being emitted.

    Deliberately an ABSOLUTE bound with ``rtol=0``: the guarantee is on the absolute error, and a
    relative tolerance would be far too strict near zero and far too slack in the tails.
    """
    x = torch.stack([torch.linspace(-40.0, 40.0, _ROW), torch.randn(_ROW)]).cuda().contiguous()
    ref = torch.tanh(x)
    assert_elementwise(_run(tanh, x), ref, tolerance_bound(ref, _TANH_ABS, 0.0), what="tanh")


def test_tanh_saturates_exactly_in_the_tails():
    """Beyond about |x| = 9 the exact fp32 result IS +-1, so the approximation must be exact there."""
    x = torch.stack([torch.full((_ROW,), 20.0), torch.full((_ROW,), -20.0)]).cuda().contiguous()
    assert torch.equal(_run(tanh, x), torch.sign(x))


def test_sigmoid_matches_torch_including_where_exp_would_overflow(sweep):
    """The tanh form has no overflow arm; ``1/(1+exp(-x))`` would lose x below about -88.

    The sweep's first row reaches -40, where an exp-based sigmoid is already denormal, so this
    covers the range the tanh form exists to make safe.
    """
    ref = torch.sigmoid(sweep)
    assert_elementwise(
        _run(sigmoid, sweep), ref, tolerance_bound(ref, _SIGMOID_ABS, 0.0), what="sigmoid"
    )


def test_sigmoid_is_exactly_one_half_at_zero():
    """``0.5 + 0.5*tanh(0)`` is exact, and a gate that is off-by-an-ULP at 0 signals a reassociation."""
    z = torch.zeros(1, _ROW, device="cuda")
    assert torch.equal(_run(sigmoid, z), torch.full_like(z, 0.5))


def test_glu_is_bitwise_its_own_definition(sweep):
    """``glu(x, y) == sigmoid(x) * y`` exactly -- the definitional gate, with no tolerance to hide in.

    Computing both sides on the device removes torch's sigmoid from the comparison, so this fails
    on a swapped argument or a reassociated product that the tolerance test above would absorb.
    """
    y = torch.randn_like(sweep)
    assert torch.equal(_run(glu, sweep, y), _run(sigmoid, sweep) * y)


def test_glu_is_not_symmetric_in_its_arguments(sweep):
    """Guards the one mistake the type system cannot: gate and up are not interchangeable."""
    y = torch.randn_like(sweep) + 2.0
    assert not torch.equal(_run(glu, sweep, y), _run(glu, y, sweep))


def test_the_maps_carry_only_what_has_a_test():
    """The maps are deliberately minimal; an entry with no test here is the thing to prevent."""
    assert set(gate_fn_map) == {"glu"}
    assert set(act_fn_map) == {None, "sigmoid"}


def test_the_none_activation_is_an_entry_not_an_absence():
    """Callers do ``act_fn_map[name]`` without special-casing the un-activated path."""
    assert act_fn_map[None] is None


def test_as_gate_fn_names_the_alternatives_rather_than_raising_a_bare_keyerror():
    """A bare KeyError names the missing key; the useful half of the message is what exists."""
    assert as_gate_fn("glu") is gate_fn_map["glu"]
    with pytest.raises(ValueError, match=r"unknown gate activation 'swiglu'.*available.*glu"):
        as_gate_fn("swiglu")


def test_no_blackwell_packed_arm_was_carried_back():
    """The packed ``(Float32, Float32)`` datapath is SM100-only and unreachable here.

    An AST scan rather than a call, because the claim is about what the module *contains*: these
    functions cannot be invoked from the host at all (they emit MLIR and need a Context and an
    InsertionPoint), so there is no way to observe a tuple being rejected from outside a kernel.
    What this does prevent is the arm being reintroduced as dead code during a later bring-back.

    Scanning identifiers rather than raw text is the point: the module docstring *names* the
    removed symbols in order to explain the removal, so a substring search over the file matches
    its own documentation and can never pass.
    """
    tree = ast.parse(Path(activation_mod.__file__).read_text())
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
    }
    banned = {n for n in names if "packed_f32x2" in n or n == "F32_or_F32x2"}
    assert not banned, f"SM100-only packed arm reintroduced: {sorted(banned)}"

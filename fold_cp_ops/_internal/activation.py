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

# Copyright (c) 2025, Tri Dao.
# This file has been modified by NVIDIA CORPORATION & AFFILIATES.
"""Fused epilogue activations, as fp32 register-to-register functions.

Every function here runs **inside a GEMM epilogue**, on accumulator values already resident in
registers, and returns a value in registers -- nothing here touches memory. They are passed to a
kernel as a `cutlass.Constexpr` callable (`EpilogueArguments.act_fn`), so the choice of activation
is folded into the compiled kernel and is part of its cache key; there is no runtime dispatch and
no branch in the emitted code.

**Scope is deliberately minimal.** This module carries only the activations the shipped kernels
actually invoke, because an activation with no caller is an untested code path that still has to be
read. Adding one means adding the function, its entry in the map below, and a case in
``tests/_internal/test_activation.py`` that pins its numerics against a torch reference.

**fp32 only, scalar only.** Accumulators are fp32 (`acc_dtype`), so these take and return
`Float32`. Upstream carried a second arm on every function accepting a packed `(Float32, Float32)`
pair for the Blackwell 2x-fp32 datapath (`mul_packed_f32x2` and friends); this package is SM90-only
by decision, that arm was unreachable, and it is not carried back. Its absence is checked by an AST
scan in the tests rather than by a call, because these functions emit MLIR and cannot be invoked
from the host at all -- so there is no host-side observation of a tuple being refused.

**Why `tanh` and not a divide.** `sigmoid` is computed as `0.5 + 0.5*tanh(0.5*x)` rather than
`1/(1+exp(-x))`. `tanh.approx.f32` is a single MUFU instruction, so sigmoid costs FMUL + MUFU.TANH
+ FFMA with no reciprocal and no transcendental exp.

**Know the accuracy before trusting it.** `tanh.approx.f32` is an *approximation*, not a correctly
rounded function: the PTX ISA guarantees a maximum ABSOLUTE error of ``2**-11`` (4.9e-4) over the
whole input range -- roughly 2000x looser than an ULP of fp32, and a bound on the absolute, not the
relative, error. Composing it gives `sigmoid` a ``2**-12`` absolute bound (the 0.5 scaling halves
it) and `glu` a bound of ``2**-12 * |y|``. Those are comfortably inside a bf16 round of the same
value (bf16's ULP near 1.0 is ``2**-8``) but only about 4x inside an fp16 one (``2**-10``), so an
fp16 post-activation carries most of its error from this approximation rather than from the store.
Any test comparing against torch must use the ``2**-11`` bound; an ULP-scale tolerance will fail.
"""

from typing import Callable, Dict, Optional

from cutlass import Float32
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm


@dsl_user_op
def tanh(a: float | Float32, *, loc=None, ip=None) -> Float32:
    """Hyperbolic tangent via the SM90 ``tanh.approx.f32`` hardware instruction.

    Emits one MUFU.TANH. Used to build :func:`sigmoid`; exposed separately because it is the piece
    whose accuracy the composite inherits, so it is the piece a test pins.

    Args:
        a: The input, an fp32 register value or a Python float (promoted to `Float32`). Must NOT be
            a tuple -- there is no packed-pair arm on SM90 and one would fail inside `Float32()`
            with a type error rather than computing a pair.

    Returns:
        `Float32` -- ``tanh(a)`` to `tanh.approx.f32` accuracy: a maximum ABSOLUTE error of
        ``2**-11``, NOT correctly rounded and NOT an ULP-scale guarantee (see the module
        docstring). Saturates to exactly +-1.0 for |a| beyond about 9, which is the exact fp32
        result there anyway.
    """
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "tanh.approx.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def sigmoid(x: Float32, *, loc=None, ip=None) -> Float32:
    """Logistic sigmoid ``1 / (1 + exp(-x))``, computed as ``0.5 + 0.5*tanh(0.5*x)``.

    The two forms are algebraically identical; the tanh form is used because it is one MUFU
    instruction instead of an exp plus a reciprocal (see the module docstring).

    Args:
        x: fp32 register value. Any finite fp32 is in range -- the tanh form has no overflow arm,
            unlike ``exp(-x)`` which overflows for x below about -88. Must not be a tuple.

    Returns:
        `Float32` in [0, 1]. Exactly 0.5 at x == 0.0.
    """
    return 0.5 + 0.5 * tanh(0.5 * x)


@dsl_user_op
def glu(x: Float32, y: Float32, *, loc=None, ip=None) -> Float32:
    """Gated Linear Unit with a sigmoid gate: ``glu(x, y) = sigmoid(x) * y``.

    This is the gate the TriMul input projection uses: `x` is the gate projection's pre-activation
    and `y` the up projection's, and the result is one element of the halved output. The caller is
    responsible for pairing the right two accumulator registers -- which two they are depends on the
    weight layout (`chunk_g`), and getting it wrong produces a plausible-looking wrong answer rather
    than an error. See `GemmGatedMixin.epi_visit_subtile` for the two pairings.

    Args:
        x: fp32 gate pre-activation (the value the sigmoid is applied to).
        y: fp32 up pre-activation (the value that is scaled). **Not symmetric** -- swapping the
            arguments computes a different function and nothing detects it.

    Returns:
        `Float32` -- ``sigmoid(x) * y``.
    """
    return sigmoid(x) * y


#: Elementwise activations, selected by name at the front door and baked in as a `Constexpr`.
#: `None` maps to `None` (no activation, the post-activation store receives the raw accumulator),
#: which is a meaningful entry rather than an absence: it lets a caller pass ``act_fn_map[name]``
#: without special-casing the un-activated path.
act_fn_map: Dict[Optional[str], Optional[Callable]] = {
    None: None,
    "sigmoid": sigmoid,
}

#: Gating activations -- two pre-activations in, one out. Distinct from :data:`act_fn_map` because
#: a gate HALVES the accumulator width (2N pre-activation -> N post-activation), which changes the
#: epilogue tiling, not just the arithmetic; the two maps therefore select different kernels rather
#: than different functions in one kernel. There is no `None` entry: a gated GEMM without a gate is
#: not a gated GEMM, and the missing key raises at the front door.
gate_fn_map: Dict[str, Callable] = {
    "glu": glu,
}


def as_gate_fn(name: str) -> Callable:
    """Look up a gating activation by name, raising a message that lists what is available.

    Args:
        name: A key of :data:`gate_fn_map`. Case-sensitive.

    Returns:
        The gating callable, suitable for `EpilogueArguments.act_fn`.

    Raises:
        ValueError: If `name` is not a known gate. A bare `KeyError` from the dict would name the
            missing key but not the alternatives, and the set of gates carried here is deliberately
            small enough that the alternatives are the useful half of the message.
    """
    if name not in gate_fn_map:
        raise ValueError(
            f"unknown gate activation {name!r}; available: {sorted(gate_fn_map)}. "
            f"Add it to fold_cp_ops._internal.activation with a test before using it."
        )
    return gate_fn_map[name]

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

"""The contract every TENSOR ARGUMENT of a front door is held to, in one place.

**Why this is one helper and not a check per kernel.** A caller passes named tensors -- ``x``,
``Wg``, ``bg``, ``mask``, ``norm_weight`` -- and has no idea which of them the kernel will feed to a
TMA descriptor and which it will read with an epilogue broadcast load. That split is real inside the
kernel (an operand's dtype selects the MMA atom; a broadcast vector's dtype is only a load width),
and it is exactly why the checks used to diverge: an operand had a hard hardware constraint so
someone wrote a message for it, while a broadcast vector had none so nothing was written and the
only thing that failed was the ``torch2cute_dtype_map[...]`` lookup building the compile key.

Measured, that produced four different behaviours for the same caller mistake: ``ValueError`` for an
operand, bare ``KeyError`` for ``mask`` and for ``gemm``'s two bias vectors, ``AssertionError`` for
``layernorm``'s weight -- and ``AssertionError`` is stripped entirely by ``python -O``, so under
``-O`` that one is not a check at all. A ``KeyError: torch.float64`` is also precisely the failure
class the kernel-matrix rule exists to forbid: ``raises=`` must be one of ``API_LEVEL_ERRORS`` so a
declared region "cannot be satisfied by a DSL ICE, a CUDA fault, or an ``AttributeError`` three
frames down".

So the internal split is not allowed to reach the contract. Every named tensor argument goes through
`check_tensor`, and the only thing that varies per argument is WHICH rule applies -- operands also
carry a width, the LayerNorm gain carries an exact dtype, a broadcast vector carries neither.
"""

from typing import Optional, Sequence

import torch
from torch import Tensor

from fold_cp_ops._internal.compile_time.cute_dsl_utils import torch2cute_dtype_map

__all__ = ["check_tensor", "supported_dtypes"]


def supported_dtypes() -> str:
    """The dtypes any tensor argument may carry, rendered for an error message.

    Purpose
        Every refusal message names the supported set, so a caller does not have to find
        `torch2cute_dtype_map` to learn what to pass. Built here rather than inlined so the two
        never drift.

    Returns:
        A sorted, comma-free ``str`` of the map's keys, e.g. ``"['torch.bfloat16', ...]"``.
    """
    return str(sorted(str(d) for d in torch2cute_dtype_map))


def check_tensor(
    name: str,
    t: Optional[Tensor],
    *,
    expect_dtype: Optional[torch.dtype] = None,
    expect_width: Optional[int] = None,
    expect_shape: Optional[Sequence[Optional[int]]] = None,
    align_elems: Optional[int] = None,
) -> None:
    """Hold one named tensor argument to its dtype and shape contract, or raise naming it.

    Purpose
        The single front-door check. Called once per tensor argument, BEFORE any compile key is
        built, so an unsupported input is refused with a sentence the caller can act on rather than
        surfacing as a `KeyError` from a dict lookup or a shape mismatch that names an argument
        index inside a generated tuple.

    Semantics
        Checks in a fixed order -- membership, exact dtype, width, rank, extents, alignment -- and
        raises on the FIRST violation. The order is part of the contract: a test that declares a
        region for a combination violating two rules must be able to predict which message it gets.

        ``None`` is accepted and returns immediately, so an optional argument needs no guard at the
        call site. That is deliberate: the PRESENCE rules (``bg`` and ``bp`` together, ``W3`` with
        ``PostAct3``) relate two arguments and belong to the front door, not here.

    Args:
        name: The caller-visible argument name, e.g. ``"mask"``. It appears verbatim in every
            message, so it must be the name in the front door's signature and not an internal one --
            a message naming ``mRowVecBroadcast`` sends the reader into the kernel.
        t: The tensor, or None (accepted, returns immediately).
        expect_dtype: An EXACT required dtype, or None to accept any supported one. Used where the
            kernel reads the value at a fixed precision -- the LayerNorm gain is applied in fp32,
            and a 16-bit one would lose more precision than the normalize it scales.
        expect_width: Required element size in BYTES, or None. Used for operands: the SM90 WGMMA
            atom and the stmatrix store leg both have 16-bit forms only, so a wider or narrower
            operand has no kernel at all -- as distinct from a broadcast vector, whose dtype is only
            a load width and which therefore takes neither of these.
        expect_shape: Expected extents, with ``None`` for any extent that is free. Its LENGTH is the
            required rank. A free extent still constrains rank, which is the common mistake this
            catches: a ``(1, n)`` row vector passed where ``(n,)`` was meant indexes fine and
            broadcasts to the wrong axis.
        align_elems: Required divisor of the trailing extent, in ELEMENTS, or None. This is the
            package's one permitted shape constraint -- the 16-byte TMA floor -- expressed in
            elements because the floor is ``16 // itemsize``.

    Returns:
        None.

    Raises:
        ValueError: On the first violated rule, naming `name`, what was required and what arrived.
            Always `ValueError` and never `AssertionError`: ``python -O`` strips ``assert``, and a
            check that vanishes under an optimization flag is not a check.
    """
    if t is None:
        return
    if t.dtype not in torch2cute_dtype_map:
        raise ValueError(
            f"unsupported dtype for {name}: {t.dtype}. Supported: {supported_dtypes()}."
        )
    if expect_dtype is not None and t.dtype is not expect_dtype:
        raise ValueError(
            f"{name} must be {expect_dtype}; got {t.dtype}. The kernel reads it at that precision, "
            f"so a different one is a silent loss rather than a conversion."
        )
    if expect_width is not None and t.element_size() != expect_width:
        raise ValueError(
            f"{name} must be {expect_width * 8}-bit; got {t.dtype}. The SM90 WGMMA atom and the "
            f"stmatrix store leg have {expect_width * 8}-bit forms only, so there is no kernel for "
            f"this operand width."
        )
    if expect_shape is not None:
        shape = t.shape
        if len(shape) != len(expect_shape):
            raise ValueError(
                f"{name} must be {len(expect_shape)}-D {_render(expect_shape)}; got shape "
                f"{tuple(shape)}."
            )
        # One C-level tuple compare in front of the per-axis loop. It is a pure FAST PATH: it can
        # only succeed when every expected extent is given and every one matches, which is the
        # common case, and it changes neither which rule fires nor the message -- the loop below
        # still runs on every mismatch and on every `expect_shape` carrying a free `None` extent.
        # Worth it because this is the priciest check in the function and the front door makes ~13
        # of them per call: measured 371 -> 150 ns fully specified (-60%), 482 -> 258 ns with
        # `align_elems` (-46%), against 10 ns lost on the `None`-axis case whose compare must fail.
        if shape != expect_shape:
            for axis, (want, got) in enumerate(zip(expect_shape, shape)):
                if want is not None and want != got:
                    raise ValueError(
                        f"{name} must be {_render(expect_shape)}; got {tuple(shape)}, which "
                        f"disagrees on axis {axis} ({got} vs {want})."
                    )
    if align_elems is not None and t.shape[-1] % align_elems:
        raise ValueError(
            f"{name}'s trailing extent {t.shape[-1]} violates the 16-byte alignment floor: it must "
            f"be a multiple of {align_elems} elements at {t.element_size()} bytes each. That floor "
            f"is this package's ONLY shape constraint; every other extent is free."
        )


def _render(shape: Sequence[Optional[int]]) -> str:
    """Render an expected shape with ``*`` for the free extents, for an error message.

    Args:
        shape: The expected extents, ``None`` meaning free.

    Returns:
        e.g. ``"(*, 128)"`` for ``(None, 128)``.
    """
    return "(" + ", ".join("*" if s is None else str(s) for s in shape) + ")"

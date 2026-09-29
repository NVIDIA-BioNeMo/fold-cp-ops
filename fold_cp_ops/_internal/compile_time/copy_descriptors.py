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

# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.
# This file has been modified by NVIDIA CORPORATION & AFFILIATES.

"""Compile-time copy builders: descriptors for how threads map onto data.

Both helpers here are pure layout/type algebra -- measured at 1 and 25 emitted ops respectively,
**none** of which produces a runtime value (see ``compile_time/__init__.py``). They construct the
``CopyAtom`` / ``TiledCopy`` descriptors that the runtime ``copy_utils.copy`` then consumes; the
descriptors themselves are erased by the compiler, leaving only their effect in the addressing
arithmetic of the emitted copies.

Neither carries a DSL decorator, and that is correct: they are neither device functions
(``@cute.kernel``) nor op-emitting user helpers needing ``loc``/``ip`` plumbing (``@dsl_user_op``).
They do, however, emit MLIR, so a caller outside ``@cute.jit`` must supply a Context, a Module and
an InsertionPoint.
"""

from typing import Type

import cutlass
import cutlass.cute as cute
from cutlass import const_expr
from cutlass.cute.nvgpu import cpasync
from cutlass.cutlass_dsl import dsl_user_op


#: Access widths the SIMT copy atom accepts, in bits. Read off the DSL's own rejection message
#: ("expected copy_bits to be one of (0, 8, 16, 32, 64, 128, 256)"), not guessed -- a width outside
#: this set is a TypeError raised deep inside the atom builder, with no reference to the caller.
LEGAL_COPY_BITS = (8, 16, 32, 64, 128, 256)

#: Widest single vectorized access the hardware issues. Note the DSL permits a 256-bit atom, so
#: this is a POLICY ceiling, not a hardware one: 128 bits is one LDG.E.128/STG.E.128.
MAX_COPY_BITS = 128


def max_vec_elems(dtype: Type[cutlass.Numeric]) -> int:
    """Elements of ``dtype`` that fit in one maximally-wide vectorized access.

    The single definition of a cap that is otherwise re-derived at every call site as
    ``128 // dtype.width`` -- measured across both upstream branches, seven call sites compute it
    independently (`reduction_base`, `layernorm_lt`, `gemm_ln_gemm`, `layernorm_gemm_stagec`,
    `staged_ln_fusion`, `topk`, `cross_entropy`). A site that forgets it does NOT get an error:
    it gets a tiled copy whose value layout claims more elements than its atom moves.

    Args:
        dtype: Element type. Its ``.width`` must divide ``MAX_COPY_BITS`` -- true for every entry
            of ``torch2cute_dtype_map`` (16- and 32-bit types). A hypothetical 48-bit type would
            floor-divide to a width that under-fills the access rather than raising.

    Returns:
        ``MAX_COPY_BITS // dtype.width`` -- e.g. 8 for bf16/fp16, 4 for fp32. Callers typically
        take ``gcd(N, max_vec_elems(dtype))`` so the vector width also divides the feature extent.
    """
    return MAX_COPY_BITS // dtype.width


@dsl_user_op
def get_copy_atom(
    dtype: Type[cutlass.Numeric], num_copy_elems: int, is_async: bool = False, *, loc=None, ip=None
) -> cute.CopyAtom:
    """Build a copy atom sized to move ``num_copy_elems`` elements per thread.

    Args:
        dtype: Element type being copied; its width sets the per-element bit count.
        num_copy_elems: Elements each thread moves in one instruction. The resulting access
            width is clamped to 128 bits -- the widest single vectorized access the hardware
            issues -- so a larger request silently saturates rather than over-widening.
        is_async: If True use the ``cp.async`` GMEM->SMEM op, else the universal copy op.
        loc: DSL source location, injected by ``@dsl_user_op`` from the CALLER's frame when not
            passed explicitly. Forwarded to ``make_copy_atom`` so a diagnostic about this atom
            names the kernel line that asked for it rather than the line below. Metadata only --
            it cannot affect the emitted program (verified: IR with locations stripped is
            byte-identical either way).
        ip: DSL insertion point; forwarded for the same reason.

    Returns:
        A ``cute.CopyAtom`` configured for ``min(128, num_copy_elems * dtype.width)`` bits.
    """
    num_copy_bits = const_expr(min(128, num_copy_elems * dtype.width))
    copy_op = cpasync.CopyG2SOp() if is_async else cute.nvgpu.CopyUniversalOp()
    return cute.make_copy_atom(copy_op, dtype, num_bits_per_copy=num_copy_bits, loc=loc, ip=ip)


def tiled_copy_2d(
    dtype: Type[cutlass.Numeric],
    threads_per_row: int,
    num_threads: int,
    num_copy_elems: int = 1,
    is_async: bool = False,
) -> cute.TiledCopy:
    """Build a 2-D row-major tiled copy: ``threads_per_row`` lanes cooperate on one row.

    The thread layout is ordered ``(1, 0)`` so consecutive threads walk the contiguous
    (second) mode, which is what makes the resulting GMEM access coalesced.

    Args:
        dtype: Element type being copied.
        threads_per_row: Lanes cooperating on a single row. Must divide ``num_threads``.
        num_threads: Total threads in the block participating in the copy.
        num_copy_elems: Elements each thread moves per instruction (the vector width). Must give
            a total access width in ``LEGAL_COPY_BITS``; **not clamped** -- see the assert below
            for why. Use ``max_vec_elems(dtype)`` to derive a safe ceiling at the call site.
        is_async: If True use the ``cp.async`` GMEM->SMEM op, else the universal copy op.

    Returns:
        A ``cute.TiledCopy`` over a ``(num_threads // threads_per_row, threads_per_row)``
        thread layout with a ``(1, num_copy_elems)`` value layout.

    Raises:
        AssertionError: If the access width is not in ``LEGAL_COPY_BITS``, or if ``num_threads``
            is not a multiple of ``threads_per_row``.
    """
    num_copy_bits = num_copy_elems * dtype.width
    assert num_copy_bits in LEGAL_COPY_BITS, (
        f"num_copy_elems={num_copy_elems} x {dtype} ({dtype.width}b) = {num_copy_bits} bits, which "
        f"is not one of {LEGAL_COPY_BITS}. This is NOT clamped on purpose: the atom width and the "
        f"value layout below are one coupled quantity, and silently narrowing the atom would leave "
        f"a tiled copy whose layout claims more elements than it moves (measured: "
        f"make_tiled_copy_tv accepts that mismatch without error). Cap the vector width at the "
        f"CALLER instead, e.g. math.gcd(N, max_vec_elems(dtype))."
    )
    copy_op = cpasync.CopyG2SOp() if is_async else cute.nvgpu.CopyUniversalOp()
    copy_atom = cute.make_copy_atom(copy_op, dtype, num_bits_per_copy=num_copy_bits)
    assert num_threads % threads_per_row == 0
    thr_layout = cute.make_ordered_layout(
        (num_threads // threads_per_row, threads_per_row),
        order=(1, 0),
    )
    val_layout = cute.make_layout((1, num_copy_elems))
    return cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)

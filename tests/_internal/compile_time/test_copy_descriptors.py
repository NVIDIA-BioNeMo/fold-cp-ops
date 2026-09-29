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

"""Unit tests for ``_internal/compile_time/copy_descriptors.py``.

Both helpers are pure layout/type algebra, so they are tested host-side -- no GPU. They do emit
MLIR, so every test here requests ``mlir_ctx`` (Context + Module + InsertionPoint); see this
directory's ``conftest.py`` for why all three are load-bearing and what happens without them.

The last test drives ``tiled_copy_2d`` the way production reaches it -- through
``ReductionBase._get_tiled_copy`` during a real ``cute.compile`` -- rather than by direct call, so
the divisibility contract is pinned on both the direct and the production path.
"""

import pytest
import torch

import cutlass
import cutlass.cute as cute

from fold_cp_ops._internal.compile_time.copy_descriptors import (
    LEGAL_COPY_BITS,
    MAX_COPY_BITS,
    get_copy_atom,
    max_vec_elems,
    tiled_copy_2d,
)

from tests._internal._rowsum_kernel import RowSum

requires_sm90 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="the production path drives tiled_copy_2d through a real SM90 compile",
)


@pytest.mark.parametrize(
    "dtype,num_copy_elems,expected_elems",
    [
        (cutlass.BFloat16, 1, 1),
        (cutlass.BFloat16, 8, 8),  # 8 x 16b = 128b, exactly at the ceiling
        (cutlass.BFloat16, 16, 8),  # would be 256b -> clamped back to 8 elements
        (cutlass.BFloat16, 64, 8),  # far over -> still 8
        (cutlass.Float32, 4, 4),  # 4 x 32b = 128b
        (cutlass.Float32, 32, 4),  # over -> clamped to 4 elements, NOT 8
        (cutlass.Float16, 8, 8),
    ],
)
def test_get_copy_atom_clamps_at_128_bits(mlir_ctx, dtype, num_copy_elems, expected_elems):
    """The access width saturates at 128 bits, and the clamp is dtype-aware (elements, not bits).

    This is the property a caller cannot see: asking for 64 bf16 elements does not yield a
    1024-bit access, it silently yields a 128-bit one. Pinning the result in ELEMENTS also
    catches a regression that clamped fp32 to 8 elements (256 bits) instead of 4.
    """
    atom = get_copy_atom(dtype, num_copy_elems)
    assert cute.size(atom.layout_dst_tv) == expected_elems
    assert cute.size(atom.layout_src_tv) == expected_elems
    assert atom.value_type == dtype


def test_get_copy_atom_async_and_sync_agree_on_width(mlir_ctx):
    """``is_async`` picks cp.async vs the universal op; it must not change the access width."""
    sync = get_copy_atom(cutlass.BFloat16, 8, is_async=False)
    async_ = get_copy_atom(cutlass.BFloat16, 8, is_async=True)
    assert cute.size(sync.layout_dst_tv) == cute.size(async_.layout_dst_tv) == 8


def test_get_copy_atom_blames_the_caller_not_itself(mlir_ctx):
    """The emitted atom's source location points at THIS file, not at ``copy_descriptors.py``.

    ``@dsl_user_op`` synthesizes a Location from the caller's Python frame, but it only reaches the
    IR if the body forwards ``loc``/``ip`` to ``make_copy_atom``. Without forwarding, the inner
    builder's own decorator wins and records ITS caller -- i.e. ``copy_descriptors.py``'s own body -- so
    a diagnostic blames the helper rather than the kernel line that asked for the atom. That state
    is invisible at the call site and reads as if locations were threaded, so it is pinned here.

    Locations are pure metadata (IR with them stripped is identical either way), so this asserts
    diagnostic quality, not behaviour.
    """
    from cutlass._mlir import ir

    module = ir.Module.create()
    with ir.InsertionPoint(module.body):
        get_copy_atom(cutlass.BFloat16, 8)
    asm = module.operation.get_asm(enable_debug_info=True)
    assert "test_copy_descriptors.py" in asm, (
        "get_copy_atom is not forwarding loc/ip -- the atom's location points into copy_descriptors.py "
        "instead of the caller. Re-add `loc=loc, ip=ip` to the make_copy_atom call."
    )


@pytest.mark.parametrize(
    "threads_per_row,num_threads,num_copy_elems",
    [(8, 128, 8), (16, 128, 8), (32, 128, 8), (32, 256, 4), (64, 256, 8), (128, 256, 1)],
)
def test_tiled_copy_2d_layout(mlir_ctx, threads_per_row, num_threads, num_copy_elems):
    """The tiled copy spans exactly ``num_threads`` and its value mode is ``num_copy_elems`` wide.

    Also pins the thread-mode shape as ``(threads_per_row, num_threads // threads_per_row)``: the
    ``order=(1, 0)`` layout is what makes consecutive threads walk contiguous memory, so an
    accidental transpose here would silently uncoalesce every load in every reduction kernel.
    """
    tc = tiled_copy_2d(cutlass.BFloat16, threads_per_row, num_threads, num_copy_elems)
    assert tc.size == num_threads
    thread_shape, value_shape = tc.layout_tv_tiled.shape
    assert thread_shape == (threads_per_row, num_threads // threads_per_row)
    assert value_shape == num_copy_elems


@pytest.mark.parametrize("threads_per_row,num_threads", [(33, 128), (7, 128), (48, 128)])
def test_tiled_copy_2d_rejects_non_dividing_threads_per_row_on_host(
    mlir_ctx, threads_per_row, num_threads
):
    """A ``threads_per_row`` that does not divide ``num_threads`` raises rather than rounding.

    Rounding would leave a partial row of threads reading outside their own row -- a wrong-answer
    bug with no exception. The assert is the only thing standing between the caller and that.
    """
    with pytest.raises(AssertionError):
        tiled_copy_2d(cutlass.BFloat16, threads_per_row, num_threads, 8)


@requires_sm90
def test_tiled_copy_2d_rejects_non_dividing_threads_per_row():
    """``num_threads`` not divisible by ``threads_per_row`` must raise during tracing, not round.

    Rounding would leave a partial row of threads reading outside their own row -- a wrong-answer
    bug with no exception raised. Driven through ``ReductionBase._get_tiled_copy`` (a subclass
    with a deliberately bad ladder) rather than by calling ``tiled_copy_2d`` directly, because
    the direct host-side call is the heap-corrupting path documented in this module's docstring.
    """

    class BadLadder(RowSum):
        """RowSum whose ``_threads_per_row`` (48) does not divide ``_num_threads`` (128)."""

        def _threads_per_row(self):
            return 48

    from fold_cp_ops._internal.compile_time.cute_dsl_utils import torch2cute_dtype_map
    from fold_cp_ops._internal.compile_time.compile_utils import make_fake_tensor
    from cutlass import Float32

    with pytest.raises(AssertionError):
        cute.compile(
            BadLadder(torch2cute_dtype_map[torch.bfloat16], 1024),
            make_fake_tensor(torch2cute_dtype_map[torch.bfloat16], (cute.sym_int(), 1024), 8),
            make_fake_tensor(Float32, (cute.sym_int(),)),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )


# ── max_vec_elems / the coupled-width invariant ────────────────────────────────────────────────
# Pure Python -- no MLIR, hence no `mlir_ctx`.
@pytest.mark.parametrize(
    "dtype,expected", [(cutlass.Float16, 8), (cutlass.BFloat16, 8), (cutlass.Float32, 4)]
)
def test_max_vec_elems(dtype, expected):
    """One maximally-wide access holds ``MAX_COPY_BITS // dtype.width`` elements.

    Pinned per dtype rather than re-deriving the formula in the test, so a change to
    ``MAX_COPY_BITS`` has to be made deliberately in both places instead of silently agreeing
    with itself.
    """
    assert max_vec_elems(dtype) == expected
    assert max_vec_elems(dtype) * dtype.width == MAX_COPY_BITS


@pytest.mark.parametrize("dtype", [cutlass.Float16, cutlass.BFloat16, cutlass.Float32])
def test_max_vec_elems_lands_on_a_legal_width(dtype):
    """The cap always yields an access width the atom builder accepts.

    A cap that produced an illegal width would turn every call site's ``gcd(N, max_vec_elems(...))``
    idiom into a latent ``TypeError`` from deep inside the DSL.
    """
    assert max_vec_elems(dtype) * dtype.width in LEGAL_COPY_BITS


def test_tiled_copy_2d_rejects_an_illegal_access_width(mlir_ctx):
    """A non-power-of-two vector width raises HERE, not as a TypeError deep in the atom builder.

    ``num_copy_elems=3`` at bf16 is 48 bits, which is not in ``LEGAL_COPY_BITS``. Without the
    assert the failure surfaces from inside ``make_copy_atom`` with no reference to the caller or
    to the vector width that caused it.
    """
    with pytest.raises(AssertionError, match="LEGAL_COPY_BITS|not one of"):
        tiled_copy_2d(cutlass.BFloat16, 32, 128, 3)


def test_tiled_copy_2d_does_not_clamp_above_the_policy_cap(mlir_ctx):
    """256-bit accesses are LEGAL and must pass through unclamped -- the atom must not narrow.

    This is the invariant the assert exists to protect. ``get_copy_atom`` clamps at
    ``MAX_COPY_BITS`` (128) because its atom stands alone; ``tiled_copy_2d`` must NOT, because its
    atom is paired with a value layout that would then disagree. Measured: ``make_tiled_copy_tv``
    accepts such a mismatch silently, so nothing downstream would catch it.
    """
    tc = tiled_copy_2d(cutlass.BFloat16, 32, 128, 16)  # 16 x 16b = 256 bits
    _, value_shape = tc.layout_tv_tiled.shape
    assert value_shape == 16, "the value layout must keep all 16 elements, not a clamped 8"

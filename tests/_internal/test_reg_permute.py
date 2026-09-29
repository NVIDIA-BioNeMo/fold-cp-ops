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

"""Tests for ``fold_cp_ops._internal.reg_permute`` -- the gated post-activation register permute.

The permute is a **pure relabelling**: it moves 16-bit values between lanes and register slots and
changes none of them. That makes it testable far more strongly than the GEMM that uses it -- fill
the fragment with values that are unique per ``(lane, slot)``, run it, and check the result against
an independent Python model of the same permutation, **bitwise**. Nothing here needs a GEMM, an
accumulator layout, or a tolerance.

**Why a model and not a golden dump.** A recorded output tensor would pin the permutation just as
tightly, but it would say nothing about *what* the permutation is; the next reader could not tell a
deliberate change from a regression. `_model_permute` is the specification written a second time,
independently (from the PTX semantics of `shfl.sync.idx` and `prmt`, not from the DSL source), so
the two agreeing is evidence rather than a tautology.

**The probe uses 4 warps on purpose.** The permute is warp-collective and must not mix data across
warps; the model applies to each 32-lane group independently, so a cross-warp leak fails here.
"""

import ast
from pathlib import Path

import cuda.bindings.driver as cuda
import numpy as np
import pytest
import torch

import cutlass
import cutlass.cute as cute

import fold_cp_ops._internal.reg_permute as reg_permute_mod
from fold_cp_ops._internal.compile_time.compile_utils import make_fake_tensor as fake_tensor
from fold_cp_ops._internal.reg_permute import permute_gated_Cregs_b16

#: Threads in the probe block: 4 warps, so warp-locality is observable.
_THREADS = 128

#: 16-bit elements per thread. Must be a multiple of 4 (the permute asserts it) and is otherwise
#: free; 8 gives two independent register-pair iterations, which catches an index that is right for
#: the first pair and wrong for the rest.
_FRAG = 8


class _PermuteProbe:
    """Device probe: load a per-thread b16 fragment, permute it in place, store it back.

    Deliberately does nothing else -- no MMA, no shared memory, no epilogue. The fragment is built
    in registers, permuted, and written out, so the only thing between input and output is the
    function under test.

    Args:
        frag: Elements per thread. Must be a multiple of 4, or the permute's own assert fires at
            trace time (which one test relies on).
    """

    def __init__(self, frag: int = _FRAG):
        self.frag = frag

    @cute.jit
    def __call__(self, mX: cute.Tensor, mOut: cute.Tensor, stream: cuda.CUstream):
        """Launch a single block of `_THREADS` threads.

        Args:
            mX: ``(_THREADS, frag)`` fp16 input, one row per thread. Read only.
            mOut: ``(_THREADS, frag)`` fp16 output, fully overwritten.
            stream: The launch stream.
        """
        self.kernel(mX, mOut).launch(grid=[1, 1, 1], block=[_THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mX: cute.Tensor, mOut: cute.Tensor):
        """Copy GMEM row -> registers, permute, copy registers -> GMEM row."""
        tidx, _, _ = cute.arch.thread_idx()
        frag = cute.make_rmem_tensor((self.frag,), cutlass.Float16)
        for i in cutlass.range_constexpr(self.frag):
            frag[i] = mX[tidx, i]
        permute_gated_Cregs_b16(frag)
        for i in cutlass.range_constexpr(self.frag):
            mOut[tidx, i] = frag[i]


def _run(x: torch.Tensor, frag: int = _FRAG) -> torch.Tensor:
    """Compile and run the probe over `x`.

    Args:
        x: ``(_THREADS, frag)`` fp16 CUDA tensor, contiguous.
        frag: Elements per thread; must equal ``x.shape[1]``.

    Returns:
        A new tensor of the same shape/dtype holding the permuted fragments.
    """
    out = torch.empty_like(x)
    fake = fake_tensor(cutlass.Float16, (_THREADS, frag))
    compiled = cute.compile(
        _PermuteProbe(frag),
        fake,
        fake,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )
    compiled(x, out)
    return out


def _prmt(a: int, b: int, selector: int) -> int:
    """PTX ``prmt.b32`` in plain Python: assemble 4 bytes chosen from the 8 bytes of ``{b, a}``.

    Args:
        a: Low source word -- supplies source bytes 0..3 (byte 0 is its least significant).
        b: High source word -- supplies source bytes 4..7.
        selector: Four 4-bit indices, least significant nibble selecting result byte 0. Only the
            low 3 bits of each nibble are used here; the sign-replicate mode (nibble bit 3) is not
            exercised by the selectors this permute uses, so it is not modelled.

    Returns:
        The assembled 32-bit result as a Python int.
    """
    src = (int(b) << 32) | int(a)
    out = 0
    for j in range(4):
        idx = (selector >> (4 * j)) & 0x7
        out |= ((src >> (8 * idx)) & 0xFF) << (8 * j)
    return out


def _model_permute(bits: np.ndarray) -> np.ndarray:
    """Independent model of :func:`permute_gated_Cregs_b16`, over raw 16-bit patterns.

    Written from the PTX semantics rather than from the DSL source: ``shfl.sync.idx`` with
    ``mask_and_clamp = (32-4) << 8 | 31`` reads lane ``(lane & ~3) + offset`` (a width-4 indexed
    shuffle inside each quad), and ``prmt`` assembles bytes as modelled by :func:`_prmt`.

    Args:
        bits: ``(threads, frag)`` array of uint16 bit patterns. ``threads`` must be a multiple of
            32 (whole warps) and ``frag`` a multiple of 4, matching the kernel's requirements.

    Returns:
        A new array of the same shape holding the permuted bit patterns.
    """
    threads, frag = bits.shape
    assert threads % 32 == 0 and frag % 4 == 0
    # Pack adjacent 16-bit values into 32-bit registers, little-endian: element 2j is the low half.
    u32 = bits.astype(np.uint32).reshape(threads, frag // 2, 2)
    words = (u32[:, :, 0] | (u32[:, :, 1] << 16)).astype(np.uint64)
    out = np.zeros_like(words)

    upper_map = [0, 3, 1, 2]
    for lane in range(threads):
        q = lane % 4
        quad_base = (lane // 4) * 4
        lane_03 = q in (0, 3)
        sel_u, sel_l = (0x5410, 0x7632) if lane_03 else (0x1054, 0x3276)
        upper_idx, lower_idx = upper_map[q], upper_map[q] ^ 1
        for i in range(frag // 4):
            # Each source lane contributes the same a0/b0 choice its OWN quad index selects.
            def a0_of(src_lane: int) -> int:
                hi = src_lane % 4 in (0, 3)
                return int(words[src_lane, 2 * i + (0 if hi else 1)])

            def b0_of(src_lane: int) -> int:
                hi = src_lane % 4 in (0, 3)
                return int(words[src_lane, 2 * i + (1 if hi else 0)])

            a1 = a0_of(quad_base + upper_idx)
            b1 = b0_of(quad_base + lower_idx)
            out[lane, 2 * i + 0] = _prmt(a1, b1, sel_u)
            out[lane, 2 * i + 1] = _prmt(a1, b1, sel_l)

    lo = (out & 0xFFFF).astype(np.uint16)
    hi = ((out >> 16) & 0xFFFF).astype(np.uint16)
    return np.stack([lo, hi], axis=-1).reshape(threads, frag)


@pytest.fixture(scope="module")
def unique_frag() -> torch.Tensor:
    """A ``(_THREADS, _FRAG)`` fp16 tensor whose every element is distinct and exact.

    Values are small integers, which fp16 represents exactly up to 2048, so the bit pattern of each
    element is a unique label -- any lane or slot mix-up shows up as a specific wrong label rather
    than as a near-miss. ``_THREADS * _FRAG = 1024``, comfortably inside the exact range.
    """
    vals = torch.arange(_THREADS * _FRAG, dtype=torch.float16).reshape(_THREADS, _FRAG)
    return vals.cuda().contiguous()


def _bits(t: torch.Tensor) -> np.ndarray:
    """View a fp16 CUDA tensor's raw 16-bit patterns as a host uint16 array."""
    return t.cpu().view(torch.int16).numpy().astype(np.uint16)


def test_permute_matches_an_independent_model_bitwise(unique_frag):
    """The strongest available gate: exact agreement with the permutation written a second time."""
    got = _bits(_run(unique_frag))
    assert np.array_equal(got, _model_permute(_bits(unique_frag)))


def test_permute_only_relabels_and_never_invents_a_value(unique_frag):
    """It is a permutation: the multiset of values per warp is unchanged.

    Independent of the model -- this holds for *any* correct relabelling, so it still fails if both
    the kernel and the model were wrong in the same way, which the bitwise test alone cannot.
    """
    before, after = _bits(unique_frag), _bits(_run(unique_frag))
    for w in range(_THREADS // 32):
        lo, hi = w * 32, (w + 1) * 32
        assert sorted(before[lo:hi].ravel()) == sorted(after[lo:hi].ravel())


def test_permute_does_not_move_values_across_warps(unique_frag):
    """Warp-collective means warp-local; a cross-warp shuffle would be a silent corruption.

    Every value starts in exactly one warp, so if the per-warp value sets are preserved (previous
    test) *and* each warp's set equals its original set, nothing crossed a warp boundary.
    """
    before, after = _bits(unique_frag), _bits(_run(unique_frag))
    for w in range(_THREADS // 32):
        lo, hi = w * 32, (w + 1) * 32
        assert set(after[lo:hi].ravel().tolist()) == set(before[lo:hi].ravel().tolist())


def test_permute_is_not_the_identity(unique_frag):
    """A no-op would pass every conservation check above; this is what makes them meaningful."""
    assert not np.array_equal(_bits(_run(unique_frag)), _bits(unique_frag))


def test_permute_is_not_self_inverse(unique_frag):
    """Applying it twice does NOT restore the fragment -- recorded because it is tempting to assume.

    The lane rotation is the 3-cycle ``(1 3 2)`` on quad indices, not a transposition, so the
    permutation has no reason to be an involution and measurably is not. A caller that "undoes" a
    permute by calling it again gets scrambled data with no error.
    """
    assert not np.array_equal(_bits(_run(_run(unique_frag))), _bits(unique_frag))


def test_a_fragment_size_that_is_not_a_multiple_of_four_is_refused():
    """The assert must fire at COMPILE time -- a fragment read past its end has no runtime signal."""
    x = torch.zeros(_THREADS, 6, dtype=torch.float16, device="cuda")
    with pytest.raises(Exception, match="multiple of 4"):
        _run(x, frag=6)


def test_the_permute_lives_in_a_runtime_module_not_the_compile_time_tree():
    """It emits ``shfl``/``prmt``, so placing it under ``compile_time/`` would be a category error.

    ``compile_time/`` is defined by everything in it constant-folding away; this does not. The
    check is on the module path because that is the invariant a future move would break.
    """
    parts = Path(reg_permute_mod.__file__).parts
    assert "compile_time" not in parts and parts[-2] == "_internal"


def test_the_permute_is_traced_not_a_plain_helper():
    """It contains a runtime loop and lane-dependent control flow, so it must be ``@cute.jit``.

    A plain ``def`` would not get the DSL preprocessor and the lane-dependent ``if`` would be
    evaluated as a Python bool at trace time -- one branch baked in for every lane.
    """
    tree = ast.parse(Path(reg_permute_mod.__file__).read_text())
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "permute_gated_Cregs_b16"
    )
    assert any("cute.jit" in ast.unparse(d) for d in fn.decorator_list)

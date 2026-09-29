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
"""Warp-collective permutes that re-own register fragments between two incompatible layouts.

A GEMM epilogue moves values between fragment layouts that disagree about **which lane holds which
column**. Where the disagreement is a pure relabelling within a 4-lane quad, it is cheaper to fix
in registers -- a `shfl.sync` plus a `prmt` per register pair -- than to round-trip through shared
memory. That is what this module does.

This is a **runtime** module: every function here emits real instructions (`shfl.sync`, `prmt`), so
it does not belong under `_internal/compile_time/`, whose contents constant-fold away entirely. It
is also why it cannot simply be called `layout_utils` -- that basename is taken by the compile-time
layout algebra, and no two source files in this package may share a basename (see CLAUDE.md).

**Everything here is warp-collective.** `shuffle_sync` exchanges values between lanes, so every lane
of the warp must reach the call with the same trip count. A divergent call reads from an inactive
lane and yields undefined data with no error raised -- the failure is a wrong output tile, not an
exception.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Int32


@cute.jit
def permute_gated_Cregs_b16(t: cute.Tensor) -> None:
    """Re-own a gated 16-bit post-activation fragment so the SM90 STSM store atom can consume it.

    Purpose
        The element-interleave gated epilogue reads pre-activation column pair ``(2o, 2o+1)`` and
        writes post-activation column ``o``. Compressing two columns into one halves the fragment
        but does *not* move it between lanes, so column ``o`` ends up held by the lane that owned
        pre-activation column ``2o``. The SM90 `stmatrix` (STSM) atom expects the standard b16 quad
        ownership instead. This rotates values across each 4-lane quad and re-packs the halves so
        the fragment matches what STSM will store.

    Semantics
        In-place and warp-collective. Values are exchanged between the 4 lanes of each quad with a
        width-4 butterfly `shfl.sync`, then the two 16-bit halves of each 32-bit register are
        re-assembled with two `prmt`s (one selector per half). The tensor is `recast` to `Int32`
        and processed a register pair at a time, so the loop runs ``size/2`` iterations over the
        b16 view. The permutation depends on `lane_idx() % 4`, which is why it is warp-collective
        rather than a per-thread shuffle of a local array.

        This is the ONLY step that makes the element-interleave layout storable; it is a no-op
        conceptually for the block-interleave (`chunk_g > 1`) layout, where post-activation column
        ``o`` comes from up-column ``o`` and is already owned by the right quad. Callers must skip
        it there -- applying it anyway is silently wrong, not an error.

    Args:
        t: The post-activation register fragment, mutated in place. Requirements, none of which are
            checked beyond the two asserts below:

            - **16-bit element type** (bf16/fp16). Asserted; a 32-bit fragment would need the
              `b32` permute instead, and the `recast` to `Int32` would pair the wrong values.
            - **``cute.size(t.shape) % 4 == 0``.** Asserted. The permute works on register pairs
              within a quad, so a size that is not a multiple of 4 would read past the fragment.
            - **In registers (rmem), not SMEM or GMEM.** Not checkable here. A non-register tensor
              would generate loads and stores per element and lose the entire point.
            - **Already gated.** Pass the post-activation fragment, not the 2N pre-activation. The
              shapes differ by 2x, so passing the pre-activation trips the size assert only when
              its size is not a multiple of 4 -- otherwise it silently scrambles live data.
            - **Every lane of the warp must call it**, with the same fragment size. Divergence
              yields undefined values from inactive lanes with no diagnostic.

    Returns:
        None. `t` is modified in place; the caller's handle stays valid.

    Raises:
        AssertionError: If the element type is not 16-bit, or the fragment size is not a multiple
            of 4. Both are compile-time (trace-time) checks, so they fire at compile, not at launch.
    """
    assert t.element_type.width == 16
    assert cute.size(t.shape) % 4 == 0, "Tensor size must be a multiple of 4 for b16 permutation"
    t_u32 = cute.recast_tensor(t, Int32)

    quad_idx = cute.arch.lane_idx() % 4
    lane_03 = quad_idx == 0 or quad_idx == 3
    selector_upper = Int32(0x5410) if lane_03 else Int32(0x1054)
    selector_lower = Int32(0x7632) if lane_03 else Int32(0x3276)
    # upper_map = [0, 3, 1, 2]
    # lower_map = [1, 2, 0, 3]
    # upper_idx = upper_map[quad_idx]
    # indexing isn't supported so we have to do arithmetic
    upper_idx = quad_idx // 2 if quad_idx % 2 == 0 else 3 - quad_idx // 2
    lower_idx = upper_idx ^ 1

    # 1 -> 0b11111, 2 -> 0b11110, 4 -> 0b11100, 8 -> 0b11000, 16 -> 0b10000, 32 -> 0b00000
    width = 4
    mask = cute.arch.WARP_SIZE - width
    clamp = cute.arch.WARP_SIZE - 1
    mask_and_clamp = mask << 8 | clamp

    for i in cutlass.range(cute.size(t_u32.shape) // 2, unroll_full=True):
        upper, lower = t_u32[i * 2 + 0], t_u32[i * 2 + 1]
        upper0 = upper if lane_03 else lower
        lower0 = lower if lane_03 else upper
        upper0 = cute.arch.shuffle_sync(upper0, offset=upper_idx, mask_and_clamp=mask_and_clamp)
        lower0 = cute.arch.shuffle_sync(lower0, offset=lower_idx, mask_and_clamp=mask_and_clamp)
        t_u32[i * 2 + 0] = cute.arch.prmt(upper0, lower0, selector_upper)
        t_u32[i * 2 + 1] = cute.arch.prmt(upper0, lower0, selector_lower)

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

import cutlass
import cutlass.cute as cute
from cutlass.base_dsl.typing import Integer
from cutlass.cutlass_dsl import dsl_user_op


class FastDivmod(cute.FastDivmodDivisor):
    """A ``FastDivmodDivisor`` that also carries the divisor it was built from.

    Integer division is expensive on GPU, so the tile scheduler precomputes a magic-number
    reciprocal once and reuses it per tile. The stock ``FastDivmodDivisor`` keeps only that
    reciprocal -- but the scheduler also needs the divisor *itself* (to bound a loop, to compute a
    remainder's complement), and re-deriving it from the magic number is not possible.

    Keeping both means both must cross the host -> kernel boundary, which is why the two MLIR
    protocol methods below are overridden: the base class marshals one value, this marshals that one
    plus however many the divisor needs.
    """

    @dsl_user_op
    def __init__(
        self,
        divisor: Integer,
        is_power_of_2: bool = None,
        *,
        loc=None,
        ip=None,
    ):
        """Precompute the magic-number reciprocal, and keep the divisor alongside it.

        Args:
            divisor: The divisor. Must be positive and nonzero -- the magic-number derivation has no
                guard, and a zero produces a silently wrong quotient rather than a fault. May be a
                runtime ``Int32``, in which case the reciprocal is computed on device.
            is_power_of_2: Assert that the divisor is a power of two, which lets the DSL emit a
                shift instead of the multiply-high sequence. **Unchecked**: passing True for a
                non-power-of-two gives wrong results everywhere the divmod is used.
            loc: Optional DSL source location.
            ip: Optional DSL insertion point.
        """
        super().__init__(divisor, is_power_of_2=is_power_of_2, loc=loc, ip=ip)
        self.divisor = divisor

    def __extract_mlir_values__(self):
        """Extract MLIR values for Host->Device transfer."""
        return [self._divisor] + cutlass.extract_mlir_values(self.divisor)

    def __new_from_mlir_values__(self, values):
        """Reconstruct FastDivmodDivisor from MLIR values."""
        new_obj = object.__new__(FastDivmod)
        new_obj._divisor = values[0]
        new_obj.divisor = cutlass.new_from_mlir_values(self.divisor, values[1:])
        return new_obj

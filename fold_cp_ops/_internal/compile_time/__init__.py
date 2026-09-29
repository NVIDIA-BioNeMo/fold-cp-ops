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

"""Compile-time helpers: metaprograms whose entire output is erased before codegen.

Everything here runs during cute.compile tracing and emits **only** layout/type algebra --
!cute.layout, !cute.shape, !cute.tiled_copy and friends -- which the compiler
constant-folds away. Nothing in this package survives into the binary as an instruction; what
survives is its *effect*, baked into the addressing arithmetic of the runtime ops it configures.

The split is measured, not asserted. Emitting each helper's ops into a module and counting how many
produce a builtin/LLVM value (i32, f32, !llvm.*, inline asm) rather than a !cute.*
type:

    helper                        ops   value-producing   ->
    compile_utils.make_fake_tensor  0          0            compile-time (pure Python)
    copy_descriptors.get_copy_atom        1          0            compile-time
    copy_descriptors.tiled_copy_2d       25          0            compile-time
    layout_utils.expand            26          0            compile-time
    ---------------------------------------------------------------------------
    copy_utils.predicate_k        162          8            RUNTIME -> stays in _internal/
    reduce.row_reduce              19         19            RUNTIME -> stays in _internal/
    utils.set_block_rank            6          3            RUNTIME -> stays in _internal/

Practical consequence for callers and tests: because these emit MLIR, they need a Context, a Module
and an InsertionPoint. @cute.jit supplies all three, so production never notices. A test calling
them directly must supply them -- see tests/_internal/compile_time/conftest.py.

Intentionally NOT re-exporting anything: every module here is imported by its full path, so a
re-export hub would undo the point of the split.
"""

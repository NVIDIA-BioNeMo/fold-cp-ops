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

"""Shared fixtures for the ``_internal`` unit tests.

Only one fixture, and it exists for a specific reason: some ``_internal`` helpers are *metaprograms*
that emit MLIR, and those need an environment to emit into. Measured classification of every
retained helper (3 repeated calls each, to catch the corrupt-on-second-call regime):

    helper                      bare    +Context   +Context+Module+IP
    ReductionBase._num_threads   ok        ok            ok           <- pure Python
    torch2cute_dtype_map         ok        ok            ok           <- pure Python
    make_fake_tensor             ok        ok            ok           <- pure Python
    get_copy_atom              raise       ok            ok           <- needs a Context
    expand                     raise     raise           ok           <- needs Module + IP
    tiled_copy_2d              raise     ABORT           ok           <- needs Module + IP

So ``test_reduction_base.py``, ``test_cute_dsl_utils.py`` and ``test_compile_utils.py`` correctly
request nothing; ``test_copy_descriptors.py`` and ``test_layout_utils.py`` request ``mlir_ctx``.

The ABORT row is the one worth remembering: with a Context but no InsertionPoint, ``tiled_copy_2d``
does not raise -- it corrupts the heap and dies on the SECOND call with no Python traceback
(``malloc(): unaligned tcache chunk detected``). ``get_copy_atom`` emits a single op, so under the
same broken environment it appeared to pass; it was passing by luck, not by safety. A fixture that
supplies all three makes the distinction moot.
"""

import pytest

from cutlass._mlir import ir


@pytest.fixture
def mlir_ctx():
    """An MLIR Context, Module and InsertionPoint -- what op-emitting DSL builders require.

    All three are load-bearing. A ``cute.*`` builder interns its layout types in the **Context**,
    the operations it creates are owned by the **Module**, and the **InsertionPoint** says where
    they go. Supplying only the first leaves the ops with nowhere to land, which manifests as heap
    corruption rather than an exception (see this module's docstring).

    Production never needs this: ``@cute.jit`` establishes all three for the duration of a trace,
    which is why ``tiled_copy_2d`` carries no decorator and is still safe at its real call site.

    **Function-scoped deliberately.** A module- or session-scoped context would stay open while the
    GPU tests in the same file run, and an ambient context breaks ``cute.compile``. Per-function
    scope means only the tests that ask for it ever have one active.

    Yields:
        None. Used purely for its enter/exit side effect; the builders read the ambient context.
    """
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            yield

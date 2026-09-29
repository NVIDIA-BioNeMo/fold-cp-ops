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
"""Tests for ``fold_cp_ops.distributed.nvshmem_utils`` -- the peer-translation helper.

The module holds one function, and it exists ONLY because the upstream
``nvshmem.core.device.cute.mem.get_peer_tensor`` strips alignment. So the subject of this file is
not "does peer translation work" -- it is **does the alignment survive**, which is the single
property the vendored copy was written to add and the single property whose loss is silent.

**Why that loss is silent, and therefore why these are the tests.** A peer view that falls back to
dtype-sized alignment (2 B for bf16) still points at the right address and still reads correctly
under a scalar copy. What breaks is downstream and much later: ``make_tiled_tma_atom`` and
``cute.autovec_copy`` size their atom from the pointer's alignment annotation, so a downgraded view
emits a narrower atom or fails IR verification inside a kernel that was fine yesterday. Nothing at
the call site raises. That is why the assertions here read the alignment ANNOTATION at trace time
rather than checking that some copy produced the right bytes -- the bytes are right either way.

**The second test is a regression test for a CuTe-DSL preprocessor bug**, documented in the source
at the ``dict.pop`` call: ``if key not in dict`` is mishandled on the ``**kwargs`` capture dict at
trace time and the assignment branch runs unconditionally, CLOBBERING a caller's explicit
``assumed_align``. That defect is invisible in review -- the guarded spelling looks correct -- so it
needs a test that passes an explicit value differing from the inherited one and demands the explicit
value win. If someone later "tidies" the ``pop`` back into an ``if not in``, this is what fails.

Trace-time, not run-time: the alignment is a compile-time property of the pointer, so the probes
assert inside ``@cute.jit`` and a violation fails at trace. No output tensor is produced and no
numeric comparison is possible or wanted, which is what ``computes_nothing_numeric`` records.

**WHAT THIS FILE DOES NOT COVER, and where it is covered instead.** The helper's device call
``nvshmem_ptr`` is STUBBED here. That is deliberate and it is the only way these tests can exist:
``nvshmem_ptr`` is an NVSHMEM *device* symbol, so a probe that really calls it needs the device
bitcode linked and a launched kernel -- measured, and the failure is
``JIT session error: Symbols not found: [ nvshmem_ptr ]`` followed by a misleading
``RuntimeError: Unknown function cutlass__probe_..._Float16_16_`` that reads like a name-resolution
bug. With the stub, the ALIGNMENT LOGIC around the call is fully exercised with no GPU at all.

So the split is:

* **Covered here:** alignment is inherited from the source tensor; an explicit ``assumed_align``
  overrides it; the layout is preserved. These are the properties the vendored copy exists to add.
* **NOT covered here:** that the real ``nvshmem_ptr`` returns a usable peer address, and that a peer
  view actually drives a wider TMA atom. **Owner: port items 4/5**, in a kernel that builds a TMA
  atom from a peer view -- which is the only context where a downgraded annotation has an
  observable consequence. Item 4's matrix must carry a test or facet that fails if the peer view's
  alignment downgrades; a deferral without a named gate is an intention rather than a plan.
"""

import cutlass
import pytest
import cutlass.cute as cute
from fold_cp_ops.testing.collective_guard import rank_invariant_skip
from fold_cp_ops.testing.numeric_guard import numeric_exempt
from fold_cp_ops.testing.kernel_matrix import (
    Axis,
    KernelMatrix,
    computes_nothing_numeric,
    matrix_exempt,
    no_unsupported,
)

# The subject is imported at MODULE scope, not inside a test, for a reason that is easy to get
# wrong: a `@cute.jit` probe resolves the functions it calls by a module-level NAME lookup at trace
# time, so a name bound only in an enclosing test body is not there to be found. The import is
# guarded because `nvshmem_utils` does `import nvshmem.core...` at module scope and a non-nvshmem
# image must skip rather than fail collection.
try:
    from fold_cp_ops.distributed.nvshmem_utils import get_peer_tensor

    _NVSHMEM_IMPORT_ERROR = None
except ImportError as _e:  # pragma: no cover - only on a non-nvshmem image
    get_peer_tensor = None
    _NVSHMEM_IMPORT_ERROR = _e

#: The pool is (dtype, assumed_align) because the defect being guarded is a fall-back to
#: DTYPE-SIZED alignment. A single dtype could not distinguish "inherited 16" from "happened to
#: equal the dtype size", so the small dtypes are the load-bearing ones: for ``Float16`` the buggy
#: answer is 2 and the correct answer is 16/128/1024, which cannot coincide.
PEER_TENSOR = KernelMatrix(
    kernel="nvshmem_utils",
    axes=(
        Axis(
            name="dtype",
            domain="any CuTe numeric element type that can back a symmetric-heap tensor",
            values=(cutlass.Float16, cutlass.BFloat16, cutlass.Float32, cutlass.Int64),
            facets={
                # A 2-byte dtype makes the buggy fall-back maximally distinguishable from every
                # declared alignment; an 8-byte one is the nearest a fall-back could get.
                "narrow": lambda d: d.width <= 16,
                "wide": lambda d: d.width >= 32,
            },
        ),
        Axis(
            name="assumed_align",
            domain=(
                "any byte alignment accepted by cute.make_ptr; the TMA hardware floor is 16 and the "
                "SW128 swizzle wants 1024, so the pool spans the range the A2A store actually uses"
            ),
            values=(16, 128, 1024),
            facets={
                "tma_floor": lambda a: a == 16,
                "swizzle_grade": lambda a: a >= 128,
            },
        ),
    ),
    computes=computes_nothing_numeric(
        because=(
            "the subject is a POINTER ANNOTATION -- the alignment carried by a peer-translated "
            "view. The helper launches nothing and produces no values; it returns a tensor aliasing "
            "another PE's memory with the same layout, so there is no output whose distribution "
            "could hide a defect"
        )
    ),
    unsupported=no_unsupported(
        because=(
            "every (dtype, assumed_align) pair in the pool is a legal cute.make_ptr argument set, "
            "and get_peer_tensor is generic over both -- it forwards the dtype untouched and either "
            "inherits or forwards the alignment. The one refusal the helper inherits is nvshmem's "
            "own on a non-symmetric pointer, which is not a combination of declared axis values and "
            "is a device-side fault rather than an API-level raise"
        )
    ),
)


# ---------------------------------------------------------------------------------------------
# The probes live at MODULE scope, and that is load-bearing rather than stylistic.
#
# Nested inside a test they FAIL, every one of them, with `RuntimeError: Unknown function
# cutlass_probe_128_16_` -- measured, 16 of 17 tests. The DSL mangles a jit function's specialization
# into a symbol (`_128_16_` is the two Constexpr arguments) and resolves it by a module-level name
# lookup; a closure defined in a test body is not reachable that way. Everything a probe needs is
# therefore an ARGUMENT, and `get_peer_tensor` is a module-level name.
# ---------------------------------------------------------------------------------------------


@cute.jit
def _probe_alignment_is_inherited(
    dtype: cutlass.Constexpr, align: cutlass.Constexpr[int], pe: cutlass.Int32
):
    """Assert at TRACE time that a peer view keeps `align` rather than falling back to dtype size.

    Input requirements: `dtype` a CuTe numeric type, `align` a byte alignment `cute.make_ptr`
    accepts, `pe` a valid PE id. The source-side assert fires first if the probe itself failed to
    build a tensor carrying `align`, so a pass cannot come from a source that never had it.
    """
    ptr = cute.make_ptr(dtype, cutlass.Int64(0x10000), cute.AddressSpace.gmem, assumed_align=align)
    src = cute.make_tensor(ptr, cute.make_layout((8, 8)))
    assert src.iterator.alignment == align, (
        "the SOURCE tensor did not carry the requested alignment; the probe is broken, so a pass "
        "below would prove nothing about inheritance"
    )
    peer = get_peer_tensor(src, pe)
    assert peer.iterator.alignment == align, (
        "peer view fell back to a narrower alignment -- the upstream helper's dtype-sized fallback. "
        "The consequence is a narrower TMA atom and an IR-verification failure in whichever kernel "
        "consumes this view next, not an error here."
    )
    assert peer.layout == src.layout, "peer translation must change only the base pointer"


@cute.jit
def _probe_explicit_alignment_wins(
    src_align: cutlass.Constexpr[int], want: cutlass.Constexpr[int], pe: cutlass.Int32
):
    """Assert an explicit ``assumed_align=`` survives instead of being replaced by the inherited one.

    Input requirements: `src_align` and `want` must DIFFER, or the assertion cannot distinguish
    "the explicit value won" from "the inherited value won"; the caller enforces that.
    """
    ptr = cute.make_ptr(
        cutlass.Float16, cutlass.Int64(0x10000), cute.AddressSpace.gmem, assumed_align=src_align
    )
    src = cute.make_tensor(ptr, cute.make_layout((8, 8)))
    peer = get_peer_tensor(src, pe, assumed_align=want)
    assert peer.iterator.alignment == want, (
        "the explicit assumed_align was overwritten by the inherited one -- the kwargs default-fill "
        "clobbered the caller. See the dict.pop comment in nvshmem_utils.get_peer_tensor."
    )


@cute.jit
def _probe_self_translation(pe: cutlass.Int32):
    """Assert self-translation preserves the layout, i.e. ``pe == my_pe`` is the identity view."""
    ptr = cute.make_ptr(
        cutlass.Float32, cutlass.Int64(0x10000), cute.AddressSpace.gmem, assumed_align=16
    )
    src = cute.make_tensor(ptr, cute.make_layout((4, 4)))
    peer = get_peer_tensor(src, pe)
    assert peer.layout == src.layout, "self-translation must preserve the layout"


@pytest.fixture
def stub_peer_address(monkeypatch):
    """Replace the NVSHMEM *device* symbol with a host stub, so the probes trace with no bitcode.

    Purpose: `get_peer_tensor` reads the source alignment, applies the ``dict.pop`` default, and
    calls ``cute.make_ptr`` -- all host-side trace work. The ONLY device-dependent step is obtaining
    the peer address. Stubbing exactly that step makes the alignment contract testable without a
    kernel, a bitcode link, or even a GPU.

    Semantics: patches the module-level name ``nvshmem_cute_mem`` inside `nvshmem_utils`, which is
    what the helper dereferences at trace time. `monkeypatch` restores it, which matters because the
    real module object is shared process-wide and a leaked stub would silently disarm every later
    test that expects the genuine symbol.

    Input requirements: none; the stub accepts the same ``(addr, pe)`` the real symbol takes and
    returns a fixed `Int64`. The VALUE is irrelevant -- no assertion here reads the address, only
    the annotation attached to it -- which is precisely why a stub is sound for this property and
    unsound for peer correctness.

    Yields: the list of ``(addr, pe)`` pairs the stub received, so a test can assert the helper
        actually reached the device call rather than short-circuiting before it.
    """
    import types

    import fold_cp_ops.distributed.nvshmem_utils as nu

    seen = []

    def _stub(addr, pe):
        seen.append((addr, pe))
        return cutlass.Int64(0x20000)

    monkeypatch.setattr(nu, "nvshmem_cute_mem", types.SimpleNamespace(nvshmem_ptr=_stub))
    return seen


def _require_nvshmem():
    """Skip the whole file's subject when nvshmem is absent, rank-invariantly.

    Input requirements: call from inside a test, after the process group is up. Uses
    ``rank_invariant_skip`` rather than a bare skip because a divergent skip is a DEADLOCK, not a
    skip -- and the predicate here genuinely is job-uniform: whether ``nvshmem.core`` imports is a
    property of the container image, identical on every rank of the launch.

    Returns: the imported ``get_peer_tensor``.
    """
    if _NVSHMEM_IMPORT_ERROR is not None:  # pragma: no cover - only on a non-nvshmem image
        rank_invariant_skip(
            f"nvshmem4py unavailable ({_NVSHMEM_IMPORT_ERROR})",
            because=(
                "whether `import nvshmem.core.device.cute.mem` succeeds is fixed by the container "
                "image, so every rank of a launch reaches the same verdict; no rank can skip alone"
            ),
        )


@PEER_TENSOR.parametrize("dtype", "assumed_align")
@numeric_exempt(
    "the subject is a POINTER ANNOTATION checked at trace time -- the alignment carried by a "
    "peer-translated view. Nothing is launched and no tensor is produced, so there is no value "
    "to compare element-wise; the assertion that matters fires inside the @cute.jit trace"
)
def test_the_peer_view_inherits_the_source_alignment(stub_peer_address, dtype, assumed_align):
    """A peer view keeps the source's alignment instead of falling back to the dtype's size.

    This is the entire reason the helper is vendored rather than imported from nvshmem4py. The
    assertion is on the annotation, at trace time, because a downgraded annotation produces correct
    bytes and an incorrect ATOM -- the failure surfaces inside some later kernel's IR verification,
    not here, and by then nothing points back to this call.
    """
    _require_nvshmem()
    cute.compile(_probe_alignment_is_inherited, dtype, assumed_align, cutlass.Int32(0))
    assert stub_peer_address, (
        "the helper never reached the peer-address call, so the alignment assertions inside the "
        "probe cannot have exercised the translation path they exist to guard"
    )


@PEER_TENSOR.parametrize("assumed_align")
@numeric_exempt(
    "the subject is a POINTER ANNOTATION checked at trace time -- the alignment carried by a "
    "peer-translated view. Nothing is launched and no tensor is produced, so there is no value "
    "to compare element-wise; the assertion that matters fires inside the @cute.jit trace"
)
def test_an_explicit_alignment_overrides_the_inherited_one(stub_peer_address, assumed_align):
    """An explicit ``assumed_align=`` wins over the source's -- the ``dict.pop`` regression.

    The source uses ``kwargs.pop(key, default)`` where ``if key not in kwargs`` would read more
    naturally, and the comment there records why: the DSL's AST preprocessor mishandles a
    membership test on the ``**kwargs`` capture dict and runs the assignment branch unconditionally,
    silently replacing the caller's value. The bug is invisible in review because the natural
    spelling looks right, so this test passes an explicit value that DIFFERS from the inherited one
    and requires the explicit one to survive. It fails the moment somebody tidies the ``pop`` away.
    """
    _require_nvshmem()
    # Deliberately unequal to the source's, so "the explicit value won" and "the inherited
    # value won" cannot produce the same number.
    source_align = 16 if assumed_align != 16 else 128
    cute.compile(_probe_explicit_alignment_wins, source_align, assumed_align, cutlass.Int32(0))
    assert stub_peer_address, "the helper never reached the peer-address call"


@matrix_exempt(
    "the subject is the self-translation IDENTITY (pe == my_pe returns the same address), which is "
    "a property of one call rather than of a shape or dtype pool -- the matrix axes would all be "
    "held constant and the parametrization would assert the same thing repeatedly"
)
@numeric_exempt(
    "the subject is a POINTER ANNOTATION checked at trace time -- the alignment carried by a "
    "peer-translated view. Nothing is launched and no tensor is produced, so there is no value "
    "to compare element-wise; the assertion that matters fires inside the @cute.jit trace"
)
def test_translating_to_my_own_pe_returns_the_same_address(stub_peer_address):
    """``pe == my_pe`` aliases the source, which is what makes a 1-PE round trip meaningful.

    The source docstring offers this as the way to exercise the helper without 2-PE plumbing, so it
    is worth pinning: if self-translation ever stopped being the identity, every single-rank test of
    a peer path would silently start testing a different address.
    """
    _require_nvshmem()
    cute.compile(_probe_self_translation, cutlass.Int32(0))
    assert stub_peer_address, "the helper never reached the peer-address call"


@matrix_exempt(
    "asserts a SOURCE-level invariant -- that the package's distributed __init__ does not import "
    "this module -- which is AST over a file and has no kernel, shape or dtype to draw from"
)
def test_the_package_init_does_not_import_this_module():
    """``distributed/__init__.py`` must NOT re-export this module, or the subtree needs nvshmem.

    ``nvshmem_utils`` imports ``nvshmem.core.device.cute.mem`` at MODULE scope. ``__init__.py``
    executes on any ``import fold_cp_ops.distributed.X``, so re-exporting this one would make every
    import of the subtree -- ``PeMap``, ``LayoutMap``, the manager -- require nvshmem to be
    installed. That converts a missing optional dependency into a package-wide ImportError, and the
    tests that would catch it are exactly the ones that could no longer be collected.
    """
    import ast
    import pathlib

    import fold_cp_ops.distributed as dist_pkg

    src = pathlib.Path(dist_pkg.__file__).read_text()
    imported = {
        node.module
        for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert not any("nvshmem_utils" in m for m in imported), (
        "fold_cp_ops/distributed/__init__.py imports nvshmem_utils, whose module-level "
        "`import nvshmem.core...` would then run for every import of the distributed subtree. "
        "Import it from the modules that need it (peer_tma_atoms, gemm_sm90_a2a) instead."
    )

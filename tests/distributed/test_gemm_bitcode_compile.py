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
"""Tests for ``fold_cp_ops.distributed.gemm_bitcode_compile`` -- the bitcode-route compile harness.

**SINGLE-GPU by design.** A plain ``D = A @ B^T`` compiled with the nvshmem device bitcode LINKED
but NOT registered (``register=False``), which exercises the compile path and ``make_runner``'s
fold/unfold semantics without a multi-GPU nvshmem context. No torchrun, no process group. It lives
under ``tests/distributed/`` because that is where its source lives, not because it needs ranks.

Three properties, and the third is a pair.

**1. A static compile computes the right answer.** The bitcode link is the part that can silently
go wrong: a kernel that fails to link an nvshmem symbol does not misbehave, it fails to resolve --
so the check is that the whole route produces a correct GEMM, end to end.

**2. A dynamic compile serves an UNSEEN shape on the same executor.** One compile, many token
counts. If dynamic layout marking regressed to baking the shape, this is the only test that would
notice: the first shape would still be correct.

**3. fold and unfold are a PAIR, and neither is meaningful alone.** ``make_runner(fold=False)``
rebuilds its views per call, so a NEW input tensor IS reflected -- that is the deployment-correct
behaviour. ``make_runner(fold=True)`` captures the view ONCE, so a new tensor is NOT reflected;
fold is a benchmarking isolation tool. **Read separately, each test is satisfiable by a broken
runner**: one asserting "reflects A2" passes if the runner recomputes everything from scratch every
call, and one asserting "reflects A1" passes if the runner is simply stuck. Together they pin that
fold and unfold DISAGREE in exactly the direction the design intends, over the same machinery
differing only in that flag -- and if fold were ever ignored entirely, both would compute A2 and the
pair would catch it while neither test alone would.

**What was restructured from the kernel this ports from, and why**

* Five assertions used ``((D - ref).norm() / ref.norm()) < 5e-2`` -- an L2-NORM RATIO, a pooled
  scalar. Corrupting one element of a large output leaves such a metric comfortably inside its bar
  while an element-wise form reports the offending index. Replaced by ``assert_gemm_close``, which
  derives a PER-ELEMENT analytic bound for random operands rather than inheriting a scalar.
* ``test_fold_ignores_new_tensor`` was ``@pytest.mark.xfail(strict=True)`` around an assertion that
  fold DOES reflect the new tensor. That form cannot distinguish "fold correctly reused A1" from
  "the run crashed", "D stayed zero", or "the reference is wrong" -- each makes the assertion fail
  and the xfail pass, green, because ``strict`` catches only an unexpected PASS. It is now a
  positive assertion that fold produces the OLD tensor's result, which fails for all four reasons.
"""

import contextlib

import pytest
import torch

import cutlass.torch as cutlass_torch
from cutlass import Float32, Int32
from cutlass.cute.runtime import from_dlpack

from fold_cp_ops._internal.arch import get_device_capacity, get_max_active_clusters
from fold_cp_ops._internal.compile_time.cute_dsl_utils import torch2cute_dtype_map
from fold_cp_ops._internal.gemm_tvm_ffi_utils import make_scheduler_args
from fold_cp_ops.kernels.gemm import GemmDefaultSm90
from fold_cp_ops.testing.kernel_matrix import (
    Axis,
    KernelMatrix,
    computes_nothing_numeric,
    matrix_exempt,
    no_unsupported,
)
from fold_cp_ops.testing.numeric_guard import numeric_exempt
from fold_cp_ops.testing.numerics import assert_gemm_close

try:
    import nvshmem.core  # noqa: F401

    from fold_cp_ops.distributed.gemm_bitcode_compile import (
        compile_gemm_with_bitcode,
        make_runner,
    )

    _HAS_NVSHMEM = True
except Exception:  # pragma: no cover - a non-nvshmem image
    _HAS_NVSHMEM = False

_SM90 = False
try:
    _SM90 = torch.cuda.is_available() and get_device_capacity()[0] == 9
except Exception:  # pragma: no cover
    pass

#: A module-level mark rather than a `pytest.skip` call. Both predicates are properties of the
#: IMAGE (does nvshmem4py import) and the BOX (is the device SM90), and this cluster's nodes are
#: homogeneous -- so every rank of a launch reaches the same verdict and no rank can skip alone.
#: `collective_guard` refuses bare `pytest.skip()` here but does NOT see marks, so a `skipif` on a
#: PER-NODE property would be rank-divergent in exactly the way the guard exists to prevent while
#: passing it silently. The reasoning is therefore written down rather than enforced.
pytestmark = pytest.mark.skipif(
    not (_HAS_NVSHMEM and _SM90), reason="needs an SM90 GPU + nvshmem4py (bitcode link)"
)

N, K, L = 256, 256, 1
TILE = (128, 256)

BITCODE = KernelMatrix(
    kernel="gemm_bitcode_compile",
    axes=(
        Axis(
            name="m_tokens",
            domain=(
                "any token count the tile can cover; the harness itself imposes no floor. The pool "
                "spans a tile multiple and two off-grid values because the compile route bakes "
                "shape into the descriptor for a STATIC compile and must not for a dynamic one -- "
                "a pool of tile multiples alone could not tell a dynamic compile from a static one "
                "that happened to be re-invoked at the same size"
            ),
            values=(256, 520, 641),
            facets={
                "tile_multiple": lambda m: m % TILE[0] == 0,
                "off_grid": lambda m: m % TILE[0] != 0,
                "small": lambda m: m <= 256,
                "large": lambda m: m >= 512,
            },
        ),
    ),
    computes=computes_nothing_numeric(
        because=(
            "the subject is the COMPILE ROUTE -- bitcode linking, dynamic-shape service and runner "
            "fold semantics -- not an arithmetic kernel. The GEMM it compiles is a plain "
            "GemmDefaultSm90 whose numerics are the single-device kernel's own concern and are "
            "gated by tests/kernels/test_gemm.py; here the answer is checked only to prove the "
            "route produced a WORKING kernel rather than a linked-but-wrong one"
        )
    ),
    unsupported=no_unsupported(
        because=(
            "every declared token count is a legal shape for this tile, so no combination of "
            "declared axis values is refused. The module's one reachable refusal is a missing "
            "device bitcode (_device_bitcode raises RuntimeError when nvshmem is absent), which is "
            "an ENVIRONMENT condition rather than a combination of pool values and is covered "
            "directly by test_a_missing_bitcode_is_refused_by_name"
        )
    ),
)


def _operands(M, *, seed):
    """Random bf16 operands in the layout the bitcode route expects: ``(M, K, L)`` L-minor.

    Input requirements: `M` any token count; `seed` distinct per call site when a test needs two
    tensors that must be distinguishable -- the fold/unfold pair depends on A1 and A2 differing, so
    a shared seed would make both of its assertions pass regardless of which tensor was used.
    """
    torch.manual_seed(seed)
    A = (torch.randn(L, M, K, device="cuda", dtype=torch.bfloat16) / K**0.5).permute(1, 2, 0)
    B = (torch.randn(L, N, K, device="cuda", dtype=torch.bfloat16) / K**0.5).permute(1, 2, 0)
    D = torch.zeros(L, M, N, device="cuda", dtype=torch.bfloat16).permute(1, 2, 0)
    return A, B, D


def _ctx():
    """The per-launch epilogue, scheduler and stream bundle.

    Carries no varlen slot: this tree dropped varlen in the single-device bring-back, so
    ``GemmSm90.__call__`` is the kernel-this-ports-from's signature minus that one argument.
    """
    epi = GemmDefaultSm90.EpilogueArguments(
        alpha=None,
        beta=None,
        mRowVecBroadcast=None,
        mColVecBroadcast=None,
        add_to_output=False,
        rounding_mode=None,
        sr_seed=None,
    )
    sched = make_scheduler_args(get_max_active_clusters(1), Int32(8), None, None)
    return epi, sched, cutlass_torch.current_stream()


def _args(A, B, D, ctx, *, dynamic):
    """Positional arguments for the compile/launch, matching ``GemmSm90.__call__`` exactly.

    Input requirements: `dynamic` selects whether the layouts are marked dynamic. A STATIC compile
    bakes the shape, so re-invoking it at a different M is undefined -- only the dynamic form may be
    served an unseen shape, which is what `test_a_dynamic_compile_serves_an_unseen_shape` relies on.
    """
    epi, sched, stream = ctx
    md = (lambda c: c.mark_layout_dynamic()) if dynamic else (lambda c: c)
    return (
        md(from_dlpack(A, assumed_align=16)),
        md(from_dlpack(B, assumed_align=16)),
        md(from_dlpack(D, assumed_align=16)),
        None,
        epi,
        sched,
        stream,
        None,
    )


def _gemm():
    """A plain `GemmDefaultSm90` -- the simplest kernel that exercises the whole bitcode route."""
    return GemmDefaultSm90(
        Float32, torch2cute_dtype_map[torch.bfloat16], TILE, (1, 1, 1), is_persistent=True
    )


@BITCODE.parametrize("m_tokens")
def test_a_static_compile_produces_a_working_kernel(m_tokens):
    """The bitcode-linked route compiles and the resulting kernel computes ``A @ B^T``.

    The link is the part that fails silently-ish: a missing nvshmem symbol does not produce wrong
    numbers, it fails to resolve. Checking the ANSWER is what proves the route produced a kernel
    rather than merely a compile that returned an object.
    """
    A, B, D = _operands(m_tokens, seed=3)
    compiled = compile_gemm_with_bitcode(
        _gemm(), *_args(A, B, D, _ctx(), dynamic=False), register=False
    )
    compiled(*_args(A, B, D, _ctx(), dynamic=False))
    torch.cuda.synchronize()
    assert_gemm_close(D[:, :, 0], A[:, :, 0], B[:, :, 0], what=f"static compile M={m_tokens}")
    compiled.free()


@matrix_exempt(
    "the subject is that ONE compile serves TWO different shapes, so the test needs a PAIR of "
    "token counts rather than one drawn from the pool; parametrizing would fix the first and leave "
    "the second hardcoded, which states the property less clearly than naming both"
)
def test_a_dynamic_compile_serves_an_unseen_shape():
    """One dynamic compile answers a second, never-compiled token count correctly.

    If dynamic layout marking regressed to baking the shape, the FIRST shape would still be right
    and only this second one would fail -- so a test that compiled and ran a single shape could not
    tell a dynamic compile from a static one.
    """
    A, B, D = _operands(512, seed=3)
    compiled = compile_gemm_with_bitcode(
        _gemm(), *_args(A, B, D, _ctx(), dynamic=True), register=False
    )
    compiled(*_args(A, B, D, _ctx(), dynamic=True))
    torch.cuda.synchronize()
    assert_gemm_close(D[:, :, 0], A[:, :, 0], B[:, :, 0], what="dynamic compile, compiled shape")

    A2, B2, D2 = _operands(640, seed=7)  # never compiled for
    compiled(*_args(A2, B2, D2, _ctx(), dynamic=True))
    torch.cuda.synchronize()
    assert_gemm_close(D2[:, :, 0], A2[:, :, 0], B2[:, :, 0], what="dynamic compile, UNSEEN shape")
    compiled.free()


@matrix_exempt(
    "one half of the fold/unfold PAIR; the property is a DISAGREEMENT between two runner modes over "
    "fixed operands, so varying the token count would repeat the same comparison at more sizes "
    "without making the disagreement any sharper"
)
def test_an_unfolded_runner_reflects_a_new_tensor():
    """``fold=False`` rebuilds its views per call, so a swapped-in tensor IS reflected.

    **Half of a pair — see :func:`test_a_folded_runner_reuses_the_captured_tensor` and the module
    docstring.** Alone this is satisfiable by a runner that recomputes everything from scratch every
    call; it means something only beside the folded case, which must NOT reflect the swap.
    """
    A1, B, D = _operands(512, seed=4)
    A2, _, _ = _operands(512, seed=5)  # different data, new allocation, same shape
    holder = [A1]
    ctx = _ctx()

    def build():
        return _args(holder[0], B, D, ctx, dynamic=False)

    compiled = compile_gemm_with_bitcode(_gemm(), *build(), register=False)
    run = make_runner(compiled, build, fold=False)
    holder[0] = A2  # swap BEFORE the launch
    run()
    torch.cuda.synchronize()
    assert_gemm_close(D[:, :, 0], A2[:, :, 0], B[:, :, 0], what="unfolded runner reflects A2")
    compiled.free()


@matrix_exempt("the other half of the fold/unfold PAIR; see the sibling test's reason")
def test_a_folded_runner_reuses_the_captured_tensor():
    """``fold=True`` captured the view ONCE, so the result is the OLD tensor's, not the new one's.

    **Half of a pair — see :func:`test_an_unfolded_runner_reflects_a_new_tensor`.** Alone this is
    satisfiable by a runner that is simply stuck; it means something only beside the unfolded case,
    which must reflect the swap. Together they pin that the two modes DISAGREE in the direction the
    design intends, and if fold were ever ignored entirely both would compute A2 and the pair would
    catch what neither test alone could.

    Asserted POSITIVELY rather than as a strict xfail: an xfail around "fold reflects A2" passes
    whenever that assertion fails for ANY reason -- a crash, an all-zero D, a wrong reference -- and
    ``strict`` catches only an unexpected pass. This form fails for all of those.
    """
    A1, B, D = _operands(512, seed=4)
    A2, _, _ = _operands(512, seed=5)
    holder = [A1]
    ctx = _ctx()

    def build():
        return _args(holder[0], B, D, ctx, dynamic=False)

    compiled = compile_gemm_with_bitcode(_gemm(), *build(), register=False)
    run = make_runner(compiled, build, fold=True)  # captures A1's view ONCE
    holder[0] = A2  # a folded runner cannot see this
    run()
    torch.cuda.synchronize()
    assert_gemm_close(
        D[:, :, 0], A1[:, :, 0], B[:, :, 0], what="folded runner reuses the CAPTURED A1"
    )
    compiled.free()


@matrix_exempt(
    "asserts a lifecycle property of one object -- that releasing twice is safe -- which does not "
    "vary with any declared shape"
)
@numeric_exempt("asserts idempotence of a release, not a computed value")
def test_free_is_idempotent():
    """``free()`` twice is safe, which is what any eventual refcounted shutdown must preserve.

    The second call must not raise even though the registration is already gone. This pins a
    property that holds today and would be easy to lose while adding the registry the plan wants --
    which is precisely why it is worth writing down before that work starts.
    """
    A, B, D = _operands(256, seed=9)
    compiled = compile_gemm_with_bitcode(
        _gemm(), *_args(A, B, D, _ctx(), dynamic=False), register=False
    )
    compiled.free()
    compiled.free()  # must not raise


@matrix_exempt(
    "asserts an API-level REFUSAL of an option combination, which is not a combination of declared "
    "axis values and cannot be reached by varying a token count"
)
@numeric_exempt("asserts a raise, not a computed value")
def test_nvshmem_plus_tvm_ffi_is_REFUSED_on_both_surfaces():
    """The one combination that must never compile, refused at BOTH entry points.

    It is expressible TODAY and silently broken, which is why this is a raise rather than a comment:
    ``extra_options`` is a free-form string, so ``" --enable-tvm-ffi"`` reached ``cute.compile`` with
    nothing to stop it. What follows is a ``.to()`` that returns ``self`` -- no materialised
    CUlibrary, so ``library_init`` has nothing to register -- plus an artifact tagged ``dump_object``
    that no loader can read back.

    Both surfaces are checked because ~50 call sites still reach `compile_nvshmem` directly; a guard
    only on the new consolidated entry would leave every one of them unprotected.
    """
    from fold_cp_ops.distributed.gemm_bitcode_compile import compile_kernel

    with pytest.raises(NotImplementedError, match="tvm-ffi"):
        compile_kernel(_gemm(), options="--enable-tvm-ffi", link_nvshmem=True)
    with pytest.raises(NotImplementedError, match="tvm-ffi"):
        compile_gemm_with_bitcode(_gemm(), extra_options=" --enable-tvm-ffi")


@matrix_exempt(
    "asserts the holder's forwarding contract, which is a property of the wrapper rather than of "
    "any declared axis value"
)
@numeric_exempt("asserts attribute forwarding, not a computed value")
def test_the_uniform_holder_forwards_attributes_and_frees_honestly():
    """`CompiledKernel` must forward attributes, or `@jit_cache` stops exporting SILENTLY.

    `cache_utils.jit_cache` calls ``compiled_fn.export_to_c(...)`` (``cache_utils.py:311``) on
    exactly what the single-device compile returned, and that block catches ``Exception`` and merely
    prints -- so a holder that did not forward would disable artifact writing with no error anywhere.

    ``free()`` is a no-op when nothing is registered, which is the honest answer on the tvm-ffi path
    rather than an error, and a genuine typo must still raise ``AttributeError`` instead of returning
    None.
    """
    from fold_cp_ops._internal.artifact_cache import CompiledKernel

    class _E:
        def __call__(self, *a):
            return "ran"

        def export_to_c(self, *a, **k):
            return "exported"

    k = CompiledKernel(executor=_E())
    assert k() == "ran"
    assert k.export_to_c("path") == "exported"
    assert k.free() is None
    with pytest.raises(AttributeError):
        k.definitely_not_an_attribute


@matrix_exempt(
    "asserts the artifact WITNESS on one compile -- a property of the store rather than a "
    "combination of declared axis values, and reached at any single token count"
)
@numeric_exempt("asserts artifact metadata, not a computed value")
def test_the_ir_sha_witness_is_recorded_stable_and_discriminating():
    """The witness must be present, deterministic, and actually tell two programs apart.

    All three, because any one alone is satisfiable by a broken witness. A constant would be present
    and stable; an absent one would be trivially "stable". Only discrimination makes it capable of
    falsifying a composed-key collision, which is the single thing it exists for.

    Why the witness is obtainable at all: the DSL computes ``module_hash`` unconditionally at
    ``dsl.py:1715`` -- even under the ``no_cache=True`` that ``cute.compile`` always sets
    (``compiler.py:1089``) -- and then DISCARDS it, so ``hasattr(compiled, "module_hash")`` is False.
    But ``compiled.ir_module`` IS exposed (``jit_executor.py:798``), so the same serialisation is
    reproducible for free on the miss path.
    """
    from fold_cp_ops._internal.artifact_cache import ir_sha

    A, B, D = _operands(256, seed=7)
    first = compile_gemm_with_bitcode(
        _gemm(), *_args(A, B, D, _ctx(), dynamic=False), register=False
    )
    again = compile_gemm_with_bitcode(
        _gemm(), *_args(A, B, D, _ctx(), dynamic=False), register=False
    )

    s1, s2 = ir_sha(first.compiled), ir_sha(again.compiled)
    assert s1 is not None, "no witness recorded -- the miss path must always produce one"
    assert s1 == s2, "the witness is not deterministic, so it can never falsify a collision"

    # ...and a DIFFERENT program must move it, or the witness discriminates nothing. A STATIC
    # compile bakes the shape into the IR, so a different token count IS a different program here --
    # which is exactly what makes this pair a valid discrimination test rather than a tautology.
    A2, B2, D2 = _operands(520, seed=7)
    other = compile_gemm_with_bitcode(
        _gemm(), *_args(A2, B2, D2, _ctx(), dynamic=False), register=False
    )
    assert ir_sha(other.compiled) != s1, "two different programs share one ir_sha"
    for c in (first, again, other):
        c.free()


@matrix_exempt(
    "asserts an ENVIRONMENT refusal -- a missing device bitcode -- which is not a combination of "
    "declared axis values and cannot be reached by varying a token count"
)
@numeric_exempt("asserts a raise, not a computed value")
def test_a_missing_bitcode_is_refused_by_name(monkeypatch):
    """With nvshmem unavailable, the route raises rather than compiling something unlinkable.

    The alternative is worse than a raise: a compile that omits the bitcode succeeds and the kernel
    fails later at an unresolved device symbol, several frames from the cause. The message must
    therefore name the bitcode, which is what the ``match`` pins.
    """
    import fold_cp_ops.distributed.gemm_bitcode_compile as m

    monkeypatch.setattr(m, "HAS_NVSHMEM", False)
    with pytest.raises(RuntimeError, match="bitcode"):
        m._device_bitcode()


@matrix_exempt(
    "asserts the ORDER of two releases on one object -- a lifecycle property that does not vary "
    "with any declared shape, dtype or mesh"
)
@numeric_exempt("asserts release ORDER, not a computed tensor")
def test_free_releases_nvshmem_BEFORE_the_cuda_library():
    """The two deletes must be ordered, or they are a double free.

    `library_init` stores the CUDA library's RAW handle in nvshmem's table. Two things release that
    library: `library_finalize`, and the DSL's GC when the last reference to `compiled`/`module`
    drops. If the DSL unloads FIRST, nvshmem's table holds a dangling handle and the next
    registration or finalize dereferences it -- SIGSEGV, seen as rc=139 at teardown.

    **Asserted by OBSERVING THE FIELDS FROM INSIDE `library_finalize`, not by watching a `__del__`.**
    The first version built an object with a `__del__`, called `gc.collect()`, and asserted the
    unload was recorded after the finalize. That is not deterministic: whether `__del__` has run by
    the time `gc.collect()` returns depends on what else holds a reference, and at world 16 it
    FAILED on a subset of ranks while passing on the rest -- a rank-divergent verdict, which is the
    one thing a distributed test must never produce. It showed up as `ss.ssF` against `ss.ss.` in the
    per-rank progress strings.

    Checking the fields at finalize time tests the same property with no GC in it: if `free()`
    cleared them first, the DSL could have unloaded first, and the fields would already be None here.
    """
    import fold_cp_ops.distributed.gemm_bitcode_compile as gbc

    seen = {}
    obj = gbc.CompiledGemmBitcode(
        executor=lambda *a: None, nvshmem_kernel_obj=object(), options="", device_id=0,
    )
    obj.compiled = object()
    obj.module = object()

    class _Fake:
        @staticmethod
        def library_finalize(_):
            # the retention MUST still be in place at this instant
            seen["held_at_finalize"] = (obj.compiled is not None, obj.module is not None)

    real = gbc.nvshmem.core
    gbc.nvshmem.core = _Fake()
    try:
        obj.free()
    finally:
        gbc.nvshmem.core = real

    assert seen.get("held_at_finalize") == (True, True), (
        f"the retention was released BEFORE nvshmem finalized, so the DSL could unload the CUDA "
        f"library while nvshmem still held its raw handle: {seen}"
    )
    assert obj.compiled is None and obj.module is None, (
        "free() finalized but kept the retention, so the unload is left to an arbitrary later GC -- "
        "which is the ordering bug this fixes, not a safe conservative choice"
    )


def _fake_registered(gbc, *, obj=None):
    """Build a wrapper and put it through `_note_registration`, returning ``(obj, seq)``.

    Args:
        gbc: the `gemm_bitcode_compile` module under test, passed in so a caller cannot accidentally
            note a registration into a DIFFERENT import of it.
        obj: an existing wrapper to register, or None to build a fresh one with stand-in
            ``compiled``/``module`` objects. The stand-ins need only be identifiable by ``id()``;
            nothing calls them.

    Returns:
        ``(wrapper, seq)`` -- the wrapper, and the sequence number `_note_registration` stamped on
        it, which is the key into `_REGISTERED_LIBS`.
    """
    if obj is None:
        obj = gbc.CompiledGemmBitcode(
            executor=lambda *a: None, nvshmem_kernel_obj=object(), options="", device_id=0,
        )
        obj.compiled = object()
        obj.module = object()
    gbc._note_registration(obj)
    return obj, obj.reg_seq


@contextlib.contextmanager
def _isolated_registries(gbc):
    """Run a body against EMPTY registration registries, restoring whatever was there.

    Purpose
        The two sweep tests below assert exactly how many registrations were finalized and in which
        order. Those are properties of the whole process's registries, not of one test's objects, so
        any registration another test left behind changes the answer.

    Semantics
        Saves and CLEARS `_REGISTERED`, `_REGISTERED_LIBS` and `_ORPHANED` on entry; restores all
        three on exit, including after a failure. `_LEAK_WARNED` is pinned True for the duration so a
        deliberately-orphaned fixture does not emit the first-orphan warning as noise.

        Snapshot-and-restore rather than assert-empty-on-entry: a leak from an unrelated test would
        otherwise fail THIS test, and under a random or xdist order it would fail a different one
        each run -- a flake that names the wrong subject. This was not hypothetical; two tests above
        deliberately orphan a registration, and the sweep (correctly) picked them up.
    """
    saved = (dict(gbc._REGISTERED), dict(gbc._REGISTERED_LIBS), dict(gbc._ORPHANED), gbc._LEAK_WARNED)
    gbc._REGISTERED.clear()
    gbc._REGISTERED_LIBS.clear()
    gbc._ORPHANED.clear()
    gbc._LEAK_WARNED = True
    try:
        yield
    finally:
        gbc._REGISTERED.clear()
        gbc._REGISTERED.update(saved[0])
        gbc._REGISTERED_LIBS.clear()
        gbc._REGISTERED_LIBS.update(saved[1])
        gbc._ORPHANED.clear()
        gbc._ORPHANED.update(saved[2])
        gbc._LEAK_WARNED = saved[3]


@matrix_exempt(
    "asserts an object-lifetime property of the registration bookkeeping -- it does not vary "
    "with any declared shape, dtype or mesh, and the SHAPE that reproduces the fault is a "
    "2-NODE JOB, which no matrix axis can express"
)
@numeric_exempt("asserts a reference is held, not a computed tensor")
def test_a_registration_keeps_its_cuda_library_alive_after_its_WRAPPER_is_dropped():
    """The retention must outlive the wrapper, because a caller that forgets `free()` drops it.

    `CompiledGemmBitcode.compiled`/`.module` keep the CUDA library loaded only while the WRAPPER is
    alive, and `_REGISTERED` holds the wrapper WEAKLY on purpose -- so a caller that drops a wrapper
    without `free()` lets the DSL `cuLibraryUnload` a library nvshmem's table still holds the raw
    handle for. `_finalize_nvshmem` then dereferences it.

    Measured before this: `world 8 on ONE node exits 0, the SAME 8 ranks over TWO nodes exit 139`,
    identically on this tree and on base, with the faulthandler stack ending in
    `release_symmetric_mempools`'s `gc.collect()`.

    The assertion is on the LIBRARY objects, not on the wrapper -- retaining the wrapper is what
    would defeat the leak detector, and retaining only the library is what does not.
    """
    import gc

    import fold_cp_ops.distributed.gemm_bitcode_compile as gbc

    obj, seq = _fake_registered(gbc)
    lib_ids = (id(obj.compiled), id(obj.module))
    try:
        del obj
        gc.collect()
        held = gbc._REGISTERED_LIBS.get(seq)
        assert held is not None, (
            "the strong library reference went with the wrapper, so the DSL may now unload a "
            "library nvshmem still holds the raw handle for"
        )
        assert (id(held[0]), id(held[1])) == lib_ids
    finally:
        gbc._REGISTERED_LIBS.pop(seq, None)
        gbc._REGISTERED.pop(seq, None)
        gbc._ORPHANED.pop(seq, None)  # this test ORPHANS one; _ORPHANED is a third registry


@matrix_exempt(
    "the control for the test above: asserts the leak COUNT still moves, a bookkeeping property "
    "with no shape, dtype or mesh dependence"
)
@numeric_exempt("asserts a leak count, not a computed tensor")
def test_retaining_the_library_does_NOT_blind_the_leak_detector():
    """Detection and retention are not exclusive -- which is the objection that kept `_REGISTERED` weak.

    The reason `_REGISTERED` holds a weakref is that a strong reference to the WRAPPER "would keep
    every kernel alive for the process lifetime, which is the leak this exists to detect". True, and
    it does not apply to `_REGISTERED_LIBS`: `compiled`/`module` are distinct objects from the
    wrapper that owns them, so the weakref still dies exactly when the caller drops its wrapper.

    This is the control for the test above. Without it, a fix that quietly retained the wrapper
    would pass that one and silently zero the leak count for the rest of the process.
    """
    import gc

    import fold_cp_ops.distributed.gemm_bitcode_compile as gbc

    before = gbc.leaked_registration_count()
    obj, seq = _fake_registered(gbc)
    keep_alive, _ = _fake_registered(gbc)  # a SECOND registration, so the prune below has work
    try:
        del obj
        gc.collect()
        gbc._note_registration(keep_alive)  # the prune runs on the next registration, not on a timer
        assert gbc.leaked_registration_count() > before, (
            "the dropped wrapper was not counted as leaked -- something is holding it alive, which "
            "is the failure mode retaining the LIBRARY was chosen to avoid"
        )
    finally:
        for k in (seq, keep_alive.reg_seq):
            gbc._REGISTERED_LIBS.pop(k, None)
            gbc._REGISTERED.pop(k, None)
            gbc._ORPHANED.pop(k, None)  # this test ORPHANS one; _ORPHANED is a third registry


@matrix_exempt(
    "asserts the ORDER of two releases across two registries on one object -- a lifecycle "
    "property that does not vary with any declared shape, dtype or mesh"
)
@numeric_exempt("asserts release ORDER, not a computed tensor")
def test_free_drops_the_process_wide_library_reference_too():
    """A freed registration must not leave its library loaded for the process lifetime.

    `free()` finalizes and clears the wrapper's own fields; if it left `_REGISTERED_LIBS` holding
    them, the retention would outlive the registration it exists to protect and every freed kernel
    would leak a loaded CUDA library. The `pop` is keyed on `reg_seq` rather than on identity
    because the allocator reuses addresses.
    """
    import fold_cp_ops.distributed.gemm_bitcode_compile as gbc

    obj, seq = _fake_registered(gbc)

    class _Fake:
        @staticmethod
        def library_finalize(_):
            assert gbc._REGISTERED_LIBS.get(seq) is not None, (
                "the process-wide reference was dropped BEFORE nvshmem finalized -- same ordering "
                "bug as clearing the fields early, one level up"
            )

    real = gbc.nvshmem.core
    gbc.nvshmem.core = _Fake()
    try:
        obj.free()
    finally:
        gbc.nvshmem.core = real
        gbc._REGISTERED.pop(seq, None)
    assert seq not in gbc._REGISTERED_LIBS, "free() left the library retained after finalizing"


@matrix_exempt(
    "reads the SOURCE of an exit hook; there is no kernel, no shape and no mesh in it"
)
@numeric_exempt("asserts call ORDER in source, not a computed tensor")
def test_the_finalize_sweep_runs_in_cleanup_and_ABOVE_the_pool_release():
    """WHERE the sweep runs is the whole fix, and running it from atexit is the fault itself.

    `library_finalize` reaches `nvshmemx_culibrary_finalize` -> `cuLibraryGetGlobal`, and from an
    interpreter-exit hook that dereferences a NULL rwlock inside libcuda -- CUDA's own state is
    already gone. Measured backtrace, rank 0 under gdb on an 8-GPU box::

        #0  ___pthread_rwlock_rdlock (rwlock=0x0)
        #4  cuLibraryGetGlobal ()                    libcuda
        #5  nvshmemx_culibrary_finalize ()           libnvshmem_host

    The A/B that pins it holds everything else fixed -- same sweep, same four registrations, wrappers
    still alive -- and moves only the timing: from atexit, 8/8 ranks exit 139; called inline during
    normal execution, 8/8 exit 0 and the sweep reports finalizing 4.

    So the sweep belongs in `cleanup()`, after its barrier and before the group is destroyed, and
    must NOT be in the exit hook. Asserted on the source because a unit test cannot run an atexit
    hook; weak, and chosen knowingly, because the alternative is no guard at all on the one edit
    that reintroduces a segfault.
    """
    import inspect

    from fold_cp_ops.distributed.distributed_manager import DistributedManager

    cleanup_src = inspect.getsource(DistributedManager.cleanup)
    assert "finalize_registered_libraries" in cleanup_src, (
        "the sweep left cleanup(): that is the only point where a finalize is both collective-safe "
        "and reached with CUDA still alive"
    )
    # And in the exit hook the sweep must sit ABOVE the pool release, which is the ordering the
    # measurement pins: `release_symmetric_mempools()` invalidates the CUlibrary handles nvshmem
    # holds, and `nvshmemx_culibrary_finalize` dereferences one as its first statement.
    exit_src = inspect.getsource(DistributedManager._drain_and_finalize_at_exit)
    code = exit_src.split('"""')[-1]  # past the docstring, which also names both calls
    # MATCH THE CALL, NOT THE NAME. The first version matched the bare names and failed at
    # `2459 < 1053`, because the comment explaining the ordering mentions
    # `release_symmetric_mempools()` in prose ABOVE the statement it is about. The qualified form
    # appears only at the call site.
    sweep = code.index("finalize_registered_libraries()")
    pools = code.index("DistributedManager.release_symmetric_mempools()")
    assert sweep < pools, (
        "the exit-hook backstop sweep moved BELOW release_symmetric_mempools(). Measured: before "
        "the pool release all 4 handles return CUDA_SUCCESS; after it, handle[1] segfaults inside "
        "cuLibraryGetGlobal. Finalize before the pools, never after"
    )


@matrix_exempt(
    "asserts a sweep empties two registries in sequence order -- bookkeeping, with no shape, dtype "
    "or mesh in it"
)
@numeric_exempt("asserts registries are emptied, not a computed tensor")
def test_the_exit_sweep_finalizes_in_REGISTRATION_ORDER_and_empties_both_registries():
    """`library_finalize` is COLLECTIVE, so the sweep's ORDER is part of its correctness.

    Every rank of an SPMD job registers the same programs in the same sequence, so iterating
    `sorted(_REGISTERED)` makes every rank issue the same collectives in the same order. Dict order
    would happen to agree today and stop agreeing the moment anything pops a key; `id()` order would
    not agree at all.

    Also asserts BOTH registries end empty. Leaving `_REGISTERED_LIBS` populated would hold every
    CUDA library loaded past the finalize for no one's benefit; leaving `_REGISTERED` populated
    would let a second sweep re-finalize an object nvshmem has already released.
    """
    import fold_cp_ops.distributed.gemm_bitcode_compile as gbc

    with _isolated_registries(gbc):
        order = []

        class _Fake:
            @staticmethod
            def library_finalize(obj):
                order.append(obj)

        objs, expect = [], []
        for _ in range(4):
            obj, _seq = _fake_registered(gbc)
            objs.append(obj)
            # CAPTURE THE EXPECTED OBJECTS NOW. The sweep NULLS `nvshmem_kernel_obj` on every wrapper it
            # touches (so a later free() cannot double-finalize), so reading them back afterwards
            # compares the recorded order against [None, None, None, None] -- which is what the first
            # version of this test did, and it failed for that reason and not for the property it names.
            expect.append(obj.nvshmem_kernel_obj)

        real = gbc.nvshmem.core
        gbc.nvshmem.core = _Fake()
        try:
            n = gbc.finalize_registered_libraries()
        finally:
            gbc.nvshmem.core = real

        assert n == 4, f"the sweep finalized {n} of 4 registrations"
        assert order == expect, (
            "the sweep did not finalize in REGISTRATION order; library_finalize is collective, so a "
            "per-rank order is a desync, not a cosmetic difference"
        )
        assert not gbc._REGISTERED and not gbc._REGISTERED_LIBS and not gbc._ORPHANED, (
            "the sweep left a registry populated: _REGISTERED or _ORPHANED lets a second sweep "
            "double-finalize, _REGISTERED_LIBS holds every CUDA library loaded past the point it "
            "could be unloaded"
        )


@matrix_exempt(
    "asserts a process-global warning fires once on a bookkeeping transition -- no shape, dtype "
    "or mesh participates"
)
@numeric_exempt("asserts a warning is raised, not a computed tensor")
def test_the_FIRST_orphaned_registration_warns_and_only_the_first():
    """The signal is that an orphan EXISTS, so it fires at one and never again.

    This replaces a bound on the LIVE count (`MAX_REGISTERED_MODULES`, default 64), whose trigger
    was monotone in the wrong quantity: 65 registrations whose owners all call `free()` are healthy,
    and ONE orphan is the armed double delete. The healthy value of the orphan count is 0, so the
    warning can fire on the first occurrence instead of on the 65th -- and must then stay quiet,
    because `_note_registration` runs on every compile and a per-call warning buries the one that
    carried the information.
    """
    import gc
    import warnings

    import fold_cp_ops.distributed.gemm_bitcode_compile as gbc

    saved_warned, saved_leaked = gbc._LEAK_WARNED, list(gbc._LEAKED)
    gbc._LEAK_WARNED = False
    seqs = []
    try:
        doomed, seq0 = _fake_registered(gbc)
        seqs.append(seq0)
        del doomed
        gc.collect()
        with warnings.catch_warnings(record=True) as first:
            warnings.simplefilter("always")
            keep, seq1 = _fake_registered(gbc)  # the prune runs on the NEXT registration
            seqs.append(seq1)
        assert [w for w in first if issubclass(w.category, RuntimeWarning)], (
            "an orphaned registration did not warn. Nothing else reports it: the owner is gone, so "
            "no free() will ever come, and the caller has no other way to learn they forgot one"
        )

        doomed2, seq2 = _fake_registered(gbc)
        seqs.append(seq2)
        del doomed2
        gc.collect()
        with warnings.catch_warnings(record=True) as second:
            warnings.simplefilter("always")
            _keep2, seq3 = _fake_registered(gbc)
            seqs.append(seq3)
        assert not [w for w in second if issubclass(w.category, RuntimeWarning)], (
            "the second orphan warned again. _note_registration runs on every compile, so a "
            "repeating warning drowns the first one, which is the only one that says anything new"
        )
        assert gbc.leaked_registration_count() == len(saved_leaked) + 2, (
            "the running total did not count both orphans -- warning ONCE must not mean counting "
            "once, or the count stops answering 'how much was forgotten'"
        )
        del keep
    finally:
        gbc._LEAK_WARNED = saved_warned
        gbc._LEAKED[:] = saved_leaked
        for k in seqs:
            gbc._REGISTERED_LIBS.pop(k, None)
            gbc._REGISTERED.pop(k, None)
            gbc._ORPHANED.pop(k, None)


@matrix_exempt(
    "asserts the exit sweep covers an orphan -- a lifetime property whose reproducing SHAPE is a "
    "2-NODE JOB, which no matrix axis can express"
)
@numeric_exempt("asserts which registrations were finalized, not a computed tensor")
def test_the_exit_sweep_finalizes_ORPHANS_TOO_and_in_the_same_sequence_order():
    """An orphan is the one registration that can never be `free()`d, so the sweep must cover it.

    The prune has to drop an orphan out of `_REGISTERED`, or the next scan re-finds the same dead
    weakref and the leak count doubles. But the tuple it drops holds the `NvshmemKernelObject` --
    the ONLY handle that can finalize that registration. Discarding it left the orphan in nvshmem's
    table for the life of the process with nothing able to retire it, which is exactly the state
    `finalize_registered_libraries` documents as the 2-node fault: "a registration whose owner was
    dropped without free() stays in that table for the life of the process, and on a 2-node fabric
    finalizing with those still present SEGFAULTS". `_ORPHANED` keeps the handle reachable.

    ORDER IS ASSERTED ACROSS THE SPLIT, not within either map. Which map a seq sits in is decided by
    GC timing and so differs per rank; the union does not, because SPMD ranks register the same
    modules in the same order. A sweep that iterated one map would be both incomplete and, across
    ranks, differently incomplete -- and `library_finalize` is collective, so that is a desync.

    THE FAULT THIS STANDS IN FOR IS MEASURED, on hardware this test does not have: 2 nodes x 1 rank,
    three real registrations, two orphaned -- the old sweep exits 139 (3/3), the union sweep exits 0
    (3/3), identical pre-exit state. Over NVLink BOTH arms are clean, so no single-node run can
    reproduce it and this GPU-free unit is what guards the invariant in CI. Reproducer
    `w8plan/probe/orphan_ab.py`; memory project-orphan-finalize-fixes-2node-segv.
    """
    import gc

    import fold_cp_ops.distributed.gemm_bitcode_compile as gbc

    order = []

    class _Fake:
        @staticmethod
        def library_finalize(obj):
            order.append(obj)

    keepalive, expect = [], []
    with _isolated_registries(gbc):
        # Alternate live / orphaned so a sweep over EITHER map alone gives the wrong answer: over
        # _REGISTERED it finalizes 2 of 4, over _ORPHANED the other 2, and neither is in seq order.
        for i in range(4):
            obj, _seq = _fake_registered(gbc)
            # CAPTURE THE HANDLE NOW: the sweep nulls `nvshmem_kernel_obj` on every LIVE wrapper it
            # touches, so reading it back afterwards compares against None.
            expect.append(obj.nvshmem_kernel_obj)
            if i % 2 == 0:
                del obj
            else:
                keepalive.append(obj)
        gc.collect()
        gbc._prune_collected()
        assert len(gbc._ORPHANED) == 2 and len(gbc._REGISTERED) == 2, (
            f"the fixture did not produce the split it needs: {len(gbc._ORPHANED)} orphaned, "
            f"{len(gbc._REGISTERED)} live"
        )

        real = gbc.nvshmem.core
        gbc.nvshmem.core = _Fake()
        try:
            n = gbc.finalize_registered_libraries()
        finally:
            gbc.nvshmem.core = real

        assert n == 4, (
            f"the sweep finalized {n} of 4. Anything under 4 means an orphan was skipped and is "
            "still in nvshmem's table with its handle already discarded"
        )
        assert order == expect, (
            "the sweep did not finalize in REGISTRATION order across the live/orphaned split; "
            "library_finalize is collective, so a per-rank order is a desync, not cosmetic"
        )
        assert not gbc._ORPHANED, "the sweep left an orphan behind for a second sweep to re-finalize"


# ── in-process compile reuse (`reuse=True`) ────────────────────────────────────────────────────
@matrix_exempt(
    "asserts that the reuse key separates two functor CONFIGURATIONS, which is a property of the "
    "key and not of any declared token/feature extent"
)
@numeric_exempt("asserts key inequality, not a computed value")
def test_reuse_key_separates_two_configurations_of_one_functor():
    """Same functor class, same operands, ONE knob different -> different key.

    This is the collision the 37-gate compile-key gap was: two functors emitting different code
    returning the same key. The knob flipped here (``_a2a_enabled``) is the master gate for the
    whole A2A store, so a collision on it would serve a plain local-store kernel to a caller that
    asked for a peer store -- silently wrong output on every rank but one.
    """
    import cutlass

    import fold_cp_ops.distributed.gemm_bitcode_compile as gbc
    from fold_cp_ops.distributed.gemm_sm90_a2a import GemmSm90A2A

    def key(enabled):
        op = GemmSm90A2A(
            cutlass.Float32, cutlass.BFloat16, (128, 128), (1, 1, 1),
            pingpong=False, is_persistent=True,
        )
        op.__dict__["_a2a_enabled"] = enabled
        return gbc._reuse_key(op, (), " --link-libraries=x", True, "k", 0)

    assert key(False) != key(True), "two A2A configurations collided on one reuse key"


@matrix_exempt("asserts a cache lifecycle property, not a shape behaviour")
@numeric_exempt("asserts holder counting and eviction, not a computed value")
def test_free_on_one_holder_does_not_release_a_shared_compile():
    """The D2 hazard, directly: a shared compile survives every ``free()`` but the last.

    With reuse on, two engines hold ONE `CompiledGemmBitcode`. If the first engine's ``free()``
    finalized it, the second engine's kernel would launch against a registration nvshmem has let go
    of and a CUDA library the DSL is free to unload -- a fault, not an exception, several frames
    from the ``free()`` that caused it.

    Driven through the wrapper's own bookkeeping rather than a real compile so it needs no GPU and
    no nvshmem: the property under test is the holder count and the eviction, both of which live
    entirely in `CompiledGemmBitcode.free`.
    """
    import fold_cp_ops.distributed.gemm_bitcode_compile as gbc

    gbc.reuse_cache_clear()
    w = gbc.CompiledGemmBitcode(
        executor=lambda *a: None, nvshmem_kernel_obj=None, options="", device_id=0
    )
    w.reuse_key = ("probe",)
    gbc._REUSE[w.reuse_key] = w
    w.holders += 1  # a second holder took a hit

    w.free()
    assert w.holders == 1, "the first free() must only decrement"
    assert gbc._REUSE.get(("probe",)) is w, "the entry must stay while a holder remains"

    w.free()
    assert w.holders == 0
    assert ("probe",) not in gbc._REUSE, (
        "the last free() must EVICT, or a later lookup is handed a wrapper whose library the DSL "
        "is now free to unload"
    )
    w.free()  # still idempotent past zero


@matrix_exempt(
    "asserts that ONE configuration compiled twice takes one compile -- a property of the cache, "
    "which does not vary with a declared shape"
)
@numeric_exempt("asserts a hit count and object identity, not a computed value")
def test_the_same_config_compiled_twice_hits_the_in_process_cache():
    """A second `reuse=True` compile of one configuration must HIT, and give back the same object.

    Both halves matter and they fail for different reasons. A hit that returned an equal-but-new
    wrapper would mean the compile was skipped and the registration was not shared -- so ``free()``
    on one would leave the other holding a finalized handle, which is a fault rather than an
    exception. Object identity is what makes the refcount the whole story.

    Asserted on the counter and on ``is``, never on wall clock: this box is not the one the perf
    pins were harvested on, and a timing assertion here would be a perf gate in disguise.

    ``register=False`` so this needs no nvshmem group -- the reuse lookup happens BEFORE the
    registration step, so the property under test is unaffected by it.
    """
    import fold_cp_ops.distributed.gemm_bitcode_compile as gbc

    A, B, D = _operands(256, seed=11)
    gbc.reuse_cache_clear()
    first = compile_gemm_with_bitcode(
        _gemm(), *_args(A, B, D, _ctx(), dynamic=False), register=False, reuse=True
    )
    assert gbc.REUSE_STATS == {"hits": 0, "misses": 1}, (
        f"the first compile should be one miss, got {gbc.REUSE_STATS}"
    )
    second = compile_gemm_with_bitcode(
        _gemm(), *_args(A, B, D, _ctx(), dynamic=False), register=False, reuse=True
    )
    assert gbc.REUSE_STATS == {"hits": 1, "misses": 1}, (
        f"the second identical compile did not hit: {gbc.REUSE_STATS}"
    )
    assert second is first, "a hit returned a different object, so the two do not share a holder count"
    assert first.holders == 2, f"holder count is {first.holders}, expected 2"
    first.free()
    assert first.holders == 1 and gbc._REUSE, "the first free() must not release a shared compile"
    second.free()
    assert not gbc._REUSE, "the last free() must evict"


@matrix_exempt("asserts that a DIFFERENT operand layout misses; a property of the key")
@numeric_exempt("asserts a miss count, not a computed value")
def test_a_different_operand_layout_MISSES_rather_than_reusing():
    """Same functor configuration, DIFFERENT operand marking -> a miss, not a silent wrong kernel.

    A static compile bakes the shape and a dynamic one does not, so serving one where the other was
    asked for is a wrong kernel with no diagnostic -- the static form re-invoked at another M is
    undefined, which `test_a_dynamic_compile_serves_an_unseen_shape` depends on being a real
    distinction. This is the negative half of the cell above: the cache must be able to MISS.
    """
    import fold_cp_ops.distributed.gemm_bitcode_compile as gbc

    A, B, D = _operands(256, seed=12)
    gbc.reuse_cache_clear()
    static = compile_gemm_with_bitcode(
        _gemm(), *_args(A, B, D, _ctx(), dynamic=False), register=False, reuse=True
    )
    dynamic = compile_gemm_with_bitcode(
        _gemm(), *_args(A, B, D, _ctx(), dynamic=True), register=False, reuse=True
    )
    assert gbc.REUSE_STATS == {"hits": 0, "misses": 2}, (
        f"a dynamic-marked operand set reused the static compile: {gbc.REUSE_STATS}"
    )
    assert dynamic is not static
    static.free()
    dynamic.free()

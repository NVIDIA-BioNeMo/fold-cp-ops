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

"""Bitcode-route GEMM compile wrapper (T2.0d) — the gating piece of Wave-2.

The deep Wave-2 fusion fuses the in-kernel NVSHMEM all-to-all *put* into a GEMM
epilogue (back A2A → GEMM1 store; front A2A → DualGatedGEMM store). Such a kernel
issues an nvshmem device op (the CP fork's ``put_signal_nbi_tma_peer`` SMEM→peer-GMEM, or
``put_nbi_warp``) from inside the GEMM epilogue, so it MUST be compiled with the
**nvshmem device bitcode linked** (``--link-libraries={find_device_bitcode_library()}``)
and registered via ``library_init`` — the T0.3/T0.7 route ``a2a.py`` uses for the
standalone primitive. The stock GEMM compile path can't: ``compile_gemm_kernel``
hardcodes ``options="--enable-tvm-ffi"``, and tvm-ffi-ON nvshmem kernels die at
launch (design doc §3 / T0.3 — nvshmem4py↔tvm-ffi IMA).

THE TWO-PART RECIPE (T2.0d — see ``benchmark/distributed/T2_0d_FINDINGS.md``).
Naively swapping ``compile_gemm_kernel``'s options to ``--link-libraries`` + a
real stream SIGFPEs in the host launch at ``cute.fast_divmod.create_divisor``
when the GEMM is compiled with **fake / dynamic-shape** tensors: the SM90 tile
scheduler builds fast-divmod divisors from the problem shape, and a dynamic
``(?,?,?)`` shape is not bound on the host via the non-tvm-ffi
``run_compiled_program`` path → ``num_clusters_per_problem`` reads 0 →
divide-by-zero. The fix (validated; STEP-1 rel_L2 ~1e-3):

1. **STATIC tensor shapes** — ``from_dlpack(t, assumed_align=16)`` over REAL
   tensors WITHOUT ``.mark_layout_dynamic()``. Concrete (M,N,K) are baked into
   the IR, so the shape-derived divisors are compile-time constants. Cost: one
   compile per shape — fine; the fused kernel needs concrete strides anyway
   (T0.5 row-0-collapse guard), so per-shape compile is already the model.
2. **scheduler args built with ``make_scheduler_args``** (concrete
   ``max_active_clusters`` from ``get_max_active_clusters`` on the persistent
   path, else 0) — the SAME run-time
   arg construction the stock GEMM uses, passed positionally to ``cute.compile``.
   (With static shapes the ``TileSchedulerOptions`` NamedTuple marshals fine
   through the non-tvm-ffi path.)

So the canonical route is: construct the GEMM op exactly as ``gemm_interface``
does, build ``epi_args`` / ``make_scheduler_args``, pass
STATIC-shape ``from_dlpack`` tensors + a REAL stream to ``cute.compile`` with
``--link-libraries`` (tvm-ffi OFF), then ``.to(dev)`` + ``library_init``.
:func:`compile_nvshmem` is exactly that ``.to(dev)`` + ``library_init``
boilerplate around the ``cute.compile`` — the part every caller (the T2.2
``GemmA2ASm90`` shim, the plain-GEMM convenience) would otherwise duplicate.
:func:`compile_plain_sm90_gemm_with_bitcode` is the worked plain ``D = A @ B``
example (and the template for the fused path — swap ``GemmCls``).
"""

import warnings
import weakref
from dataclasses import dataclass
from typing import Any, Callable, Optional

import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass import Int32, Float32
from cutlass.cute.runtime import from_dlpack

from fold_cp_ops._internal import artifact_cache as _ac
from fold_cp_ops._internal.arch import get_max_active_clusters
from fold_cp_ops.kernels.gemm import GemmDefaultSm90
from fold_cp_ops._internal.gemm_tvm_ffi_utils import make_scheduler_args

try:
    import nvshmem.core

    HAS_NVSHMEM = True
except ImportError:  # pragma: no cover - non-nvshmem host
    HAS_NVSHMEM = False


# ---------------------------------------------------------------------------
# In-process compile reuse.
# ---------------------------------------------------------------------------
#: Live compiles, keyed by `_reuse_key`. ONLY reached when a caller passes ``reuse=True``.
#:
#: WHY THIS EXISTS. Distributed kernels were never cached at all: the disk artifact cache is entered
#: only under ``if persist is not None and persist.enabled``, ``persist`` defaults to None and no
#: shipped caller sets one, and ``@jit_cache`` has zero users under ``distributed/``. So `cute.compile`
#: ran unconditionally, and a SECOND engine in one process paid a full multi-second compile for a
#: kernel already sitting in memory.
#:
#: WHY NOT ``@jit_cache``. Its disk half calls ``compiled.export_to_c`` and reloads with
#: ``cute.runtime.load_module``, which produces a bare callable. A distributed compile is a
#: `CompiledGemmBitcode` -- an executor PLUS the registered ``NvshmemKernelObject`` and the library
#: retention that keeps nvshmem's raw handle valid. Reloading one through that path would hand the
#: caller something that runs and never registers, i.e. an in-kernel nvshmem call against unwritten
#: device state, which faults rather than raises. Cross-process reuse for these belongs to
#: `artifact_cache` (the ``persist=`` path), which knows about ``library_init``.
_REUSE: dict = {}

#: Hit/miss counters. Read by tests, because a cache that silently never hits satisfies every
#: positive assertion made about it -- the same reason `artifact_cache.STATS` exists.
REUSE_STATS = {"hits": 0, "misses": 0}

#: The ONE operand-key implementation, imported rather than re-typed.
#:
#: It lives in `artifact_cache` because the DISK key needs it too (`flat_key_components`), and two
#: copies of a key function is the failure this repo has already paid for once: a key that drifts
#: does not error, it silently stops matching, and a cache that never hits looks exactly like a
#: cache that is merely cold. `_arg_key` is kept as a module-private ALIAS so the tests that name it
#: here keep naming the thing this module actually uses.
_arg_key = _ac.arg_key


def _reuse_key(op: Any, compile_args, options: str, register: bool, prefix: str, dev_id: int):
    """The full identity of a distributed compile, as a hashable tuple.

    Purpose
        One place that answers "have we already compiled exactly this". Assembled from the three
        independent surfaces, because leaving any one out serves a WRONG artifact rather than
        missing one.

    Functionality & semantics
        Components, and why each is present:

        * the functor's TYPE (module + qualname) -- two different kernels must never collide, and
          `jit_cache` keys on ``__qualname__`` alone, which is a known hazard recorded in CLAUDE.md;
        * ``op.compile_key()`` -- the configuration, including the post-construction
          ``configure_a2a*`` surface (see `COMPILE_GATED_ATTRS`). Sorted, so dict order cannot
          change the key;
        * `_arg_key` over every ``compile_args`` element -- the operands;
        * ``options`` -- the link libraries and any extra compile flags; measured as absent from
          both compile paths' keys under artifact-cache schema 1, which was audit row A3;
        * ``register`` and ``prefix`` -- ``prefix`` is baked into the image bytes, and a
          ``register=False`` wrapper is a different object with no ``NvshmemKernelObject``;
        * ``dev_id`` -- an artifact is ``.to(dev)``-bound.

        The PE is deliberately ABSENT. This cache never leaves the process, so every entry was
        produced by this rank; ``artifact_key`` adds ``(world_size, pe)`` for the DISK path, where a
        wrong-rank artifact reaching ``library_init`` is a CUDA_ERROR_ILLEGAL_ADDRESS inside nvshmem
        ``init.cu:2183`` that poisons the context.

    Args:
        op: the constructed, already-configured functor. Must expose ``compile_key()``; a functor
            without it (not a `TemplateParamsMixin`) raises ``AttributeError`` here rather than
            being keyed on its type alone, which would collide every configuration of it.
        compile_args: the full positional ``__call__`` args, including the trailing stream.
        options: the assembled compile options string.
        register: whether ``library_init`` will be called.
        prefix: the artifact symbol prefix.
        dev_id: the CUDA device ordinal the compile is bound to.

    Returns:
        A hashable tuple suitable as a dict key.

    Raises:
        AttributeError: if ``op`` has no ``compile_key()``. Deliberate: silently keying such a
            functor on its type would give every configuration of it the same entry.
    """
    cfg = tuple(sorted((k, _arg_key(v)) for k, v in op.compile_key().items()))
    return (
        type(op).__module__,
        type(op).__qualname__,
        cfg,
        tuple(_arg_key(a) for a in compile_args),
        options,
        bool(register),
        prefix,
        int(dev_id),
    )


def reuse_cache_clear() -> None:
    """Drop every in-process compile-reuse entry WITHOUT freeing anything.

    Purpose
        For tests that need a cold second measurement. Deliberately NOT a release: the wrappers
        stay live and their libraries stay loaded, so nothing nvshmem holds a handle to is unloaded
        behind its back.

    Returns:
        None. Counters are reset with the table, so a test can assert on a hit count without
        having to subtract a previous test's.
    """
    _REUSE.clear()
    REUSE_STATS["hits"] = 0
    REUSE_STATS["misses"] = 0


# ---------------------------------------------------------------------------
# Compiled-executor wrapper.
# ---------------------------------------------------------------------------
@dataclass
class CompiledGemmBitcode:
    """A bitcode-linked, tvm-ffi-OFF compiled GEMM.

    ``executor`` is the raw ``JitExecutor`` (``compiled.to(dev)``); call it (or
    this wrapper) with the GEMM op's full positional ``__call__`` args INCLUDING
    the trailing stream — for SM90: ``(mA, mB, mD, mC, epi_args, scheduler_args,
    stream[, mB2])``.

    Holds the ``NvshmemKernelObject`` that ``library_init`` registered; keep it
    alive for the kernel's lifetime (``free()`` before ``nvshmem.finalize``).
    """

    executor: Callable[..., Any]
    nvshmem_kernel_obj: Any
    options: str
    device_id: int
    #: The `cute.compile` result, RETAINED ONLY TO KEEP ITS CUDA LIBRARY LOADED. Never called.
    #:
    #: When a compiled function is garbage collected the DSL unloads its CUDA libraries immediately
    #: (``cuLibraryUnload``). NVSHMEM cannot know that: ``library_init`` stored the RAW handle in its
    #: own registration table, so after the unload that table holds a DANGLING handle, and the next
    #: NVSHMEM registration -- or ``finalize`` -- dereferences it and SEGFAULTS. Reproduced
    #: independently by the NVSHMEM team; the shape is
    #:
    #:     compiled = cute.compile(host_fn)
    #:     register_with_nvshmem(int(compiled.library))   # NVSHMEM stores the handle
    #:     del compiled                                   # GC unloads the library HERE
    #:     nvshmem_finalize()                             # segfault: stale handle
    #:
    #: Before this field, `compile_nvshmem` bound the result to a LOCAL and returned only
    #: `executor` and `nvshmem_kernel_obj`, so the library unloaded at every `return` while NVSHMEM
    #: still held its handle. The upstream has the same shape (`a2a.py:283-294` binds `compiled`,
    #: returns `exec_fn, nv_obj, stream`), so this is an INHERITED defect, not one introduced here.
    #:
    #: WHY A REFERENCE AND NOT ``library_finalize``-EACH. Finalizing each registration promptly is
    #: the other way to keep the table valid, and it is NOT AVAILABLE to us: it would force a
    #: recompile on every invocation, which defeats the whole compile-once / dispatch-many design
    #: that cute-DSL + nvshmem4py exist to provide. Holding the library alive is the only
    #: application-side option; the general fix needs a DSL notification point so the DSL can tell
    #: NVSHMEM before it unloads, which neither side can express today.
    #:
    #: SCOPE, measured: this closes the stale-handle pathology. It does NOT fix the
    #: CUDA_ERROR_ILLEGAL_ADDRESS seen at `library_init` on 2-node cp=16 at N=8192/D>=384 -- that
    #: cell fails identically with and without this field, and its module count is 1, so it is a
    #: per-module mechanism rather than accumulation. Do not read this as that fix.
    #:
    #: Do NOT "clean up" this field because nothing reads it -- that is precisely its function.
    compiled: Any = None
    #: The `ExternalBinaryModule` a CACHE HIT was loaded from, retained for EXACTLY the reason
    #: `compiled` is retained on the fresh-compile path: it owns the CUDA library that `library_init`
    #: registered, and dropping it unloads a library NVSHMEM still holds a raw handle to. The two
    #: fields are never both set -- a hit has no `compiled`, a miss has no `module` -- and they are
    #: separate rather than one `Any` so a reader can tell which path produced this object without
    #: guessing from a type. None on the fresh-compile path.
    module: Any = None
    #: This registration's sequence number in `_REGISTERED_LIBS`, stamped by `_note_registration`
    #: and read by `free()` to drop exactly this object's strong library reference. None until
    #: registered, and None forever for a wrapper built with ``register=False`` -- which has no
    #: entry to drop, so `free()`'s `pop(None, ...)` is correct rather than merely harmless.
    #: NOT an identity: the allocator reuses addresses, so a search by `id()` could drop a
    #: DIFFERENT registration's library and leave this one's dangling.
    reg_seq: "Optional[int]" = None
    #: The keys this compile was looked up under, or None when `persist` was not supplied.
    #:
    #: EXPOSED SO TESTS DO NOT RE-DERIVE THEM. Recomputing the key in a test means duplicating
    #: the program key's inputs -- op_factory, compile_args, options, world_size, extra -- and a
    #: duplicate drifts: the test then looks up an artifact that was never written and reports a
    #: miss, which reads as a cache bug rather than as test rot. Reading it off the object makes
    #: that failure impossible by construction.
    program_key: Optional[str] = None
    artifact_key: Optional[str] = None
    #: Where the artifact for `artifact_key` lives, whether or not it was read from there.
    artifact_path: Any = None
    #: True when this object came from a disk artifact rather than a compile. Read by tests (a cache
    #: that silently never hits satisfies every positive assertion made about it) and by the
    #: autotuner, which must agree a hit across ranks before trusting a timing.
    from_cache: bool = False
    #: This compile's entry in `_REUSE`, or None when it was not shared. `free()` uses it to evict
    #: the entry at the moment the last holder lets go, so the table can never hand out a wrapper
    #: whose library has been unloaded.
    reuse_key: Any = None
    #: How many callers hold this object. 1 for an unshared compile; incremented on every `_REUSE`
    #: hit and decremented by `free()`, which does its real work only at zero.
    #:
    #: **Refcounting rather than "never free a shared compile", because `free()` has a CONTRACT.**
    #: It finalizes nvshmem's registration and then drops the library retention, in that order, and
    #: callers rely on it running before `nvshmem.finalize`. Making it a silent no-op for shared
    #: objects would leave every shared registration live at shutdown, where the sweep finalizes it
    #: instead -- a different order, on the path that has already produced an rc=139.
    #:
    #: **Rank-uniform under SPMD, which is what makes the collective safe.** `library_finalize` is
    #: COLLECTIVE. The count is driven by how many times this process asked for this compile, and
    #: every rank runs the same program, so the count reaches zero at the same call on every rank.
    #: A caller that acquires a shared compile on SOME ranks only would desynchronise it -- and
    #: would already be a divergent-collective bug for the same reason.
    holders: int = 1

    def __call__(self, *run_args: Any) -> Any:
        return self.executor(*run_args)

    def free(self) -> None:
        """Release this registration, nvshmem side FIRST and the CUDA library SECOND.

        Purpose
            Undo a `compile_nvshmem` at a point every rank reaches. `library_finalize` is COLLECTIVE,
            so this must never be reached from a GC path -- see the module note below.

        Semantics
            **The ORDER is the whole point, and getting it wrong is a double free.** `library_init`
            stored the CUDA library's RAW handle in nvshmem's registration table. Two things can
            release that library: `library_finalize` (nvshmem lets go) and the DSL's garbage
            collector (`cuLibraryUnload`, when the last reference to `compiled`/`module` drops). If
            the DSL unloads FIRST, nvshmem's table holds a dangling handle and the next registration
            or finalize dereferences it -- SIGSEGV.

            The `compiled` and `module` fields exist to stop that by keeping the library alive. This
            method used to finalize WITHOUT clearing them, which left the unload to happen at some
            later, arbitrary GC -- so the retention did not order the two deletes, it only delayed
            one of them indefinitely. Clearing them here, AFTER the finalize, makes the unload
            happen no earlier than nvshmem letting go, which is the only ordering that is safe.

            Recorded as an OPEN QUESTION at `tests/distributed/test_artifact_cache.py`'s `spec`
            fixture -- "whether re-loading that same path in the same process before or after that
            unload is safe has NOT been established" -- and observed as `rc=139` at teardown of the
            world-16 IB round. This is the answer to it: establish the order rather than avoid the
            situation.

        Reuse
            When this compile is SHARED (`reuse_key` is set, `holders` > 1) the call decrements the
            holder count and returns, doing nothing else. Only the last holder finalizes -- and it
            also evicts `_REUSE[reuse_key]` FIRST, so no later lookup can be handed a wrapper whose
            library is about to be unloaded. Eviction before the finalize rather than after is the
            ordering that matters: the finalize can raise, and a table entry surviving a failed
            finalize is worse than one dropped after a successful one.

        Returns:
            None. Idempotent -- a second call finds `nvshmem_kernel_obj` already None and does
            nothing, which matters because a test may free a wrapper a fixture also frees.
        """
        if self.holders > 1:
            self.holders -= 1
            return
        self.holders = 0
        if self.reuse_key is not None:
            _REUSE.pop(self.reuse_key, None)
            self.reuse_key = None
        try:
            if self.nvshmem_kernel_obj is not None:
                nvshmem.core.library_finalize(self.nvshmem_kernel_obj)
                self.nvshmem_kernel_obj = None
        except Exception:
            pass
        finally:
            # AFTER the finalize, never before: dropping these is what lets the DSL unload the CUDA
            # library, and nvshmem must have let go of its raw handle first. BOTH references go --
            # this object's fields AND the process-wide one `_note_registration` took, or the
            # library stays loaded for a registration that no longer exists.
            #
            # AND THE SWEEP'S WORKLIST, which is a THIRD reference and was the one left behind.
            # `_REGISTERED[seq]` is a `(weakref, obj)` tuple holding its OWN reference to the same
            # `NvshmemKernelObject` this method just finalized -- captured at registration, and NOT
            # reached by clearing `self.nvshmem_kernel_obj`. `finalize_registered_libraries` then
            # iterates `sorted(_REGISTERED | _ORPHANED)` at shutdown and calls `library_finalize`
            # on that stored `obj`, i.e. a SECOND finalize of a handle whose library the DSL has
            # already unloaded (this method dropped `compiled`/`module` precisely to allow that).
            # `nvshmemx_culibrary_finalize` dereferences the handle as its first statement -> segv.
            #
            # MEASURED, 4 ranks x 24 instances, one variable: instances HELD -> rc=0; instances
            # FREED -> rc=1 with SIGSEGV after the work finished. The docstring's idempotence claim
            # is about calling `free()` twice -- true, and irrelevant here, because the sweep does
            # not call `free()`; it reaches past the wrapper to its own copy.
            #
            # Note the DIRECTION versus 8b0adb6: that change made the sweep reach orphans it was
            # MISSING. This makes it skip entries it must no longer touch. Both are the same
            # question -- who still owns this registration -- answered from opposite ends.
            _REGISTERED.pop(self.reg_seq, None)
            _REGISTERED_LIBS.pop(self.reg_seq, None)
            self.compiled = None
            self.module = None


def _device_bitcode() -> str:
    """Locate the nvshmem device bitcode this route must link.

    Returns:
        Absolute path to ``libnvshmem_device.bc``, from
        ``nvshmem.core.find_device_bitcode_library()``.

    Raises:
        RuntimeError: when nvshmem4py is not importable. Raising here is deliberate and is the
            cheaper failure: a compile that silently omits the device bitcode SUCCEEDS, and the
            kernel then dies at an unresolved device symbol several frames from the cause. The
            message names the device bitcode, not merely the missing package, so the reader learns
            what the link step wanted -- the enclosing function's name used to supply that word by
            accident, and a rename removed it while the test that pins it kept passing.
    """
    if not HAS_NVSHMEM:
        raise RuntimeError(
            "compile_nvshmem needs the nvshmem device bitcode (libnvshmem_device.bc), which is "
            "located via nvshmem4py -- not importable in this env. Install nvshmem4py, or pass "
            "bitcode=<path> explicitly."
        )
    return nvshmem.core.find_device_bitcode_library()


# ---------------------------------------------------------------------------
# Registration bookkeeping.
# ---------------------------------------------------------------------------
#
# THERE IS NO REGISTRATION-COUNT CEILING. 64 modules register cleanly on NVLink (8xH20, cp2) AND on
# 2-node IBGDA (2x8 H100, cp16), which is the run that matters because `nvshmemx_culibrary_init`
# allocates per-module transport state only on the IB path.
#
# WHAT ACTUALLY FAULTS IS A DOUBLE DELETE between the CuTe-DSL's garbage collector and nvshmem4py's
# `library_finalize` -- stated in full at `CompiledGemmBitcode.free`. `library_init` stores the CUDA
# library's RAW handle in nvshmem's table, and TWO independent owners can release that library:
# `cuLibraryUnload`, when the DSL collects the last reference to `compiled`/`module`, and
# `library_finalize`, when nvshmem lets go. Neither notifies the other, so whichever runs second
# dereferences a freed handle -> SIGSEGV.
#
# So registrations are COUNTED, but the count is a SYMPTOM READOUT, never a limit to stay under: a
# wrapper dropped without `free()` IS the double-delete precondition, and this bookkeeping is the
# only place a human sees one. ONE orphan is enough; a hundred disciplined ones are fine.
#
# NOTHING HERE EVER CALLS `library_finalize`. Two independent reasons, and the second was MEASURED
# here after the first version got it wrong:
#
#   1. Finalizing a module whose kernel may still be launched is the stale-handle fault
#      `CompiledGemmBitcode.compiled` exists to prevent -- arrived at deliberately instead of by
#      accident. So an LRU that evicts to stay under a bound is not available.
#   2. `library_finalize` IS COLLECTIVE. An earlier version of this function opportunistically
#      finalized registrations whose wrapper had been garbage collected without `free()` -- reasoning
#      that those are provably unreachable, so finalizing them is safe. It is not: GC timing differs
#      per rank, so rank 0 finalizes at a moment rank 1 does not, the ranks enter different
#      collectives, and the job DEADLOCKS. Measured at WORLD_SIZE=2: both ranks wedged inside
#      `nvshmem/core/init_fini.py:453 library_finalize`, reached from this function, and the wedge
#      watchdog had to kill them.
#
# The rule that follows is general and worth stating plainly: A COLLECTIVE MUST NEVER BE ISSUED FROM
# A GC-DRIVEN OR OTHERWISE NON-DETERMINISTIC CODE PATH. Only an explicit, caller-ordered `free()` --
# which every rank reaches at the same point in its own program -- may finalize.
#
# So this counts, and reports. A leak a human can see beats a deadlock they cannot.

#: THERE IS NO COUNT BOUND HERE, and there used to be (`MAX_REGISTERED_MODULES`, env
#: `CPO_JIT_ARTIFACT_MAX_MODULES`, default 64). It was removed because its trigger was monotone in
#: the wrong quantity: 65 registrations whose owners all call `free()` are healthy, and ONE orphan
#: is the armed double delete. A threshold on the healthy quantity fires where nothing is wrong and
#: stays silent where something is. The warning below is keyed on the orphan count instead, whose
#: healthy value is 0, so it fires on the FIRST occurrence rather than on the 65th.

#: pe -> (weakref to the wrapper, NvshmemKernelObject). A weakref because holding the wrapper here
#: would keep every kernel alive for the process lifetime, which is the leak this exists to detect.
_REGISTERED: "dict[int, tuple]" = {}
_REG_SEQ = 0

#: seq -> (compiled, module): a STRONG reference to the CUDA-library owners of every LIVE
#: registration. This is `CompiledGemmBitcode.compiled`/`.module` retention, extended from the
#: WRAPPER's lifetime to the REGISTRATION's.
#:
#: WHY IT HAD TO BE EXTENDED, measured. Those fields keep the library loaded only while the wrapper
#: is alive. `_REGISTERED` above holds the wrapper WEAKLY on purpose, so a caller that drops a
#: wrapper without `free()` unloads its CUDA library (`cuLibraryUnload`) while nvshmem's table still
#: holds the raw handle `library_init` stored -- and `_finalize_nvshmem` at exit then dereferences
#: it. Observed as SIGSEGV in `release_symmetric_mempools`'s `gc.collect()` on a 2-node job:
#: `world 8 on ONE node exits 0, the SAME 8 ranks over TWO nodes exit 139`, on this tree and on
#: base alike.
#:
#: WHY THIS DOES NOT DEFEAT THE LEAK DETECTOR, which is the objection that kept `_REGISTERED` weak.
#: It retains the LIBRARY, not the wrapper. `compiled`/`module` are distinct objects from the
#: `CompiledGemmBitcode` that owns them, so the weakref above still dies exactly when the caller
#: drops its wrapper and `_LEAKED` still counts it. Detection and retention were never exclusive;
#: only holding the WRAPPER would have made them so.
#:
#: The cost is memory: a leaked registration's CUDA library stays loaded for the process lifetime.
#: That is what the retention design always intended, and it is the correct direction to err --
#: a loaded library nobody calls costs bytes, a dangling handle costs the process.
_REGISTERED_LIBS: "dict[int, tuple]" = {}
#: Registrations whose owner was garbage collected without `free()`, CUMULATIVE for the process.
#: Exposed through `leaked_registration_count()` so a test can assert the leak exists rather than
#: inferring it, which is the only way a caller learns they forgot `free()`. Monotone: the exit
#: sweep empties `_ORPHANED` but never this, so the count still answers "did anyone forget".
_LEAKED: "list[int]" = []

#: seq -> NvshmemKernelObject, for orphans ONLY: registrations whose owner was collected without
#: `free()`. THIS EXISTS SO THE EXIT SWEEP CAN STILL FINALIZE THEM, and it closes a real gap.
#:
#: `_prune_collected` must drop an orphan out of `_REGISTERED` -- otherwise the next scan re-finds
#: the same dead weakref and `_LEAKED` double-counts. But the tuple it drops holds the
#: `NvshmemKernelObject`, which is the ONLY handle that can `library_finalize` that registration.
#: Discarding it left the orphan in nvshmem's table for the life of the process with nothing able to
#: retire it -- exactly the state `finalize_registered_libraries` documents as the 2-node fault.
#: Moving the handle here keeps the prune idempotent AND keeps the orphan reachable.
#:
#: RANK UNIFORMITY, which is the property that makes finalizing these legal at all. Whether a given
#: seq sits in `_REGISTERED` or in `_ORPHANED` is decided by GC timing and therefore DIFFERS per
#: rank. Their UNION does not: under SPMD every rank registers the same modules in the same order,
#: so the union is the same set of seqs everywhere. The sweep iterates that union in `sorted` seq
#: order, so every rank issues the same collectives in the same sequence. Iterating either map
#: ALONE would not be rank-uniform, which is why the sweep must never be "simplified" to one.
_ORPHANED: "dict[int, object]" = {}

#: True once the first orphan has been warned about. ONCE per process, on the 0 -> non-zero
#: transition: the hazard is that an orphan EXISTS, not how many, and `_note_registration` runs on
#: every compile -- so a per-call warning would bury the first and only informative one.
_LEAK_WARNED = False


def _prune_collected() -> list:
    """Move registrations whose wrapper was garbage collected out of `_REGISTERED`, into `_ORPHANED`.

    Purpose
        One implementation of the prune, because it has two callers (`_note_registration` and
        `registered_module_count`) that must agree. They previously carried separate copies, and a
        copy that drifts is how a handle gets discarded on one path and kept on the other.

    Semantics
        Not on a timer: the caller decides when. An entry is orphaned iff its weakref is dead, which
        means the owning `CompiledGemmBitcode` was dropped without `free()` (a `free()`d wrapper
        removes itself from `_REGISTERED_LIBS` and nulls its own handle). The `NvshmemKernelObject`
        is CARRIED OVER to `_ORPHANED` rather than dropped, so `finalize_registered_libraries` can
        still retire it. Idempotent: an orphan is moved exactly once, so `_LEAKED` cannot
        double-count it however often this runs.

        Issues NO collective. Finalizing here would be one, and GC timing differs per rank.

    Returns:
        The seqs orphaned by THIS call, newest scan only. Empty is the healthy case.
    """
    orphaned = [k for k, (ref, _) in _REGISTERED.items() if ref() is None]
    for k in orphaned:
        _, obj = _REGISTERED.pop(k, (None, None))
        if obj is not None:
            _ORPHANED[k] = obj
    if orphaned:
        _LEAKED.extend(orphaned)
    return orphaned


def _note_registration(wrapper: "CompiledGemmBitcode") -> None:
    """Record one `library_init`, and warn ONCE if anything has been orphaned. NO collective, ever.

    Args:
        wrapper: the object that owns the registration. Held by WEAK reference only, so this
            bookkeeping never extends a kernel's lifetime.

    Returns:
        None. Purely observational: it records, prunes dead weakrefs into `_ORPHANED`, and warns on
        the first orphan. There is no count bound -- see the note above `_REGISTERED`.

    Note:
        It does NOT finalize anything, including registrations whose wrapper is already gone. Those
        ARE a leak and they ARE provably unreachable -- but `library_finalize` is COLLECTIVE, GC
        timing differs per rank, and finalizing on whichever rank happened to collect first
        DEADLOCKS the job. Measured; see the block comment above. Reporting a leak beats hanging on
        one. `finalize_registered_libraries` retires them later, from a rank-uniform exit hook.
    """
    global _REG_SEQ, _LEAK_WARNED
    _REG_SEQ += 1
    _REGISTERED[_REG_SEQ] = (weakref.ref(wrapper), wrapper.nvshmem_kernel_obj)
    # STRONG, and deliberately not through the wrapper: see `_REGISTERED_LIBS`. The seq is stamped
    # on the wrapper so `free()` can drop exactly its own entry rather than searching by identity --
    # the allocator reuses addresses, so an identity search is unsound.
    _REGISTERED_LIBS[_REG_SEQ] = (wrapper.compiled, wrapper.module)
    wrapper.reg_seq = _REG_SEQ
    if _prune_collected() and not _LEAK_WARNED:
        _LEAK_WARNED = True
        warnings.warn(
            f"{len(_LEAKED)} nvshmem-registered CUDA library/libraries were dropped without free(). "
            "library_init handed nvshmem a RAW handle to each; with the owner gone, the CuTe-DSL's "
            "collector is free to cuLibraryUnload the library while nvshmem still holds that "
            "handle, and whichever of the two deletes runs second faults. The exit sweep retires "
            "orphans, so this is a warning rather than a crash -- but call free() yourself, at a "
            "point EVERY RANK reaches: library_finalize is collective, so nothing can be finalized "
            "for you from a garbage-collection path without deadlocking the job. Warned once per "
            "process; leaked_registration_count() has the running total.",
            RuntimeWarning,
            stacklevel=3,
        )


def registered_module_count() -> int:
    """How many nvshmem registrations this process is currently holding live.

    Returns:
        The count of registrations whose OWNER IS STILL ALIVE, after moving collected ones into
        `_ORPHANED`. Exists so a test can assert the bookkeeping rather than infer it, and so a
        human debugging a registration pathology has a number to look at. Not a quantity to keep
        under any limit -- `leaked_registration_count()` is the one whose healthy value is 0.
    """
    _prune_collected()
    return len(_REGISTERED)


def leaked_registration_count() -> int:
    """Registrations whose owner was dropped without `free()`, and which therefore stay registered.

    Returns:
        The CUMULATIVE count for this process; the exit sweep empties `_ORPHANED` but never this, so
        the answer to "did anyone forget `free()`" survives the sweep. These are not cleaned up
        WHERE THEY ARE FOUND -- `library_finalize` is collective and cannot be called from a
        garbage-collection path without deadlocking a distributed job (measured) -- but they ARE
        retired by `finalize_registered_libraries` from the exit hook. A non-zero value means some
        caller should be calling `free()` itself, at a point every rank reaches.
    """
    return len(_LEAKED)


def finalize_registered_libraries() -> int:
    """``library_finalize`` every registration still live, in a rank-uniform order. BEFORE nvshmem finalize.

    Purpose
        Leave nvshmem's registration table EMPTY before ``nvshmem_finalize`` walks it. A registration
        whose owner was dropped without `free()` stays in that table for the life of the process, and
        on a 2-node fabric finalizing with those still present SEGFAULTS.

        Measured, with no pytest in it (`w8plan/mlirkey/segv_ab.py`, 4 iterations, 16 ranks):

            compile with persist=, LOAD the artifact back, free() each wrapper   2 nodes  exit 0
            same, 1 node                                                                  exit 0
            same but keep the wrappers -- registrations never finalized          2 nodes  exit 139

        Fresh compiles left unfinalized were clean at 2 nodes; it is the CACHE-HIT path, where the
        module is loaded back from a file, that faults. `CPO_JIT_ARTIFACT_ENABLED=0` removes the
        fault while leaving every compile and registration in place, which is what identified it.

    Why this is not the thing `_note_registration` refuses to do
        That refusal is about a GARBAGE-COLLECTION path: `library_finalize` is COLLECTIVE, GC timing
        differs per rank, and finalizing on whichever rank collected first deadlocks the job. An
        exit hook is the opposite -- one deterministic point every rank reaches, which already
        issues a barrier and a collective pool free before this runs. Same collective, opposite
        timing discipline; this is the discipline `release_symmetric_mempools` states for the pools.

    Semantics
        COLLECTIVE. Iterates `sorted(_REGISTERED | _ORPHANED)` -- registration SEQUENCE, not dict
        order and not `id()` -- so every rank of an SPMD job finalizes the same registrations in the
        same order. Order matters for the same reason the count does: these are collectives.

        THE UNION IS NOT AN OPTIMIZATION, it is the rank-uniformity requirement. Which map a seq
        sits in is decided by GC timing and differs per rank; the union does not, because SPMD ranks
        register the same modules in the same order. Iterating `_REGISTERED` alone -- which is what
        this did -- silently skipped every orphan, leaving it in nvshmem's table for the life of the
        process with its handle already discarded. That is the state the docstring above names as
        the 2-node fault, reached by the one path that could not call `free()`.

        AND IT IS A REAL FAULT, not a tidiness argument -- MEASURED. Three real registrations per
        rank, two orphaned, 2 nodes x 1 rank so every collective crosses IB, arms interleaved:

            sweep over _REGISTERED only          rc=139  rc=139  rc=139
            sweep over _REGISTERED | _ORPHANED   rc=0    rc=0    rc=0
            control, every wrapper free()d       rc=0

        Identical pre-exit state in both arms (`live=1 orphaned=2 leaked=2 libs_held=3`); the sweep
        is the only variable. AND ONE NODE CANNOT SEE IT: the same script over NVLink is rc=0 for
        BOTH arms, 2/2, so a single-node run is not a cheaper version of this test but a different
        one. Reproducer `w8plan/probe/orphan_ab.py` (`free` / `orphan_old` / `orphan_new`).

        The MECHANISM is inferred, not measured. `_REGISTERED` membership depends on GC timing, so a
        rank that had collected a wrapper issued FEWER `library_finalize` collectives than one that
        had not, from a hook every rank reaches in lockstep. A stale table left for
        `nvshmem_finalize` to walk is a competing explanation for the same rc=139 and has NOT been
        ruled out. The union removes both, since it does not depend on when anything was collected.

        Failures are swallowed per registration. A finalize that raises must not stop the ones after
        it, because each one it skips is another entry left for `nvshmem_finalize` to walk.

        Drops the strong library references afterwards: once nvshmem has let go, the DSL may unload.
        Idempotent -- a second call finds both registries empty.

    Returns:
        How many registrations were finalized. Zero is the healthy state for a caller that freed
        everything; a large number means some caller should be calling `free()` itself, and
        `leaked_registration_count()` says who.
    """
    n = 0
    for seq in sorted(set(_REGISTERED) | set(_ORPHANED)):
        ref, obj = _REGISTERED.get(seq, (None, None))
        if obj is None:
            obj = _ORPHANED.get(seq)  # an orphan has no live wrapper, only the handle
        if obj is not None:
            try:
                nvshmem.core.library_finalize(obj)
                n += 1
            except Exception:  # noqa: BLE001 - one bad finalize must not strand the rest
                pass
        # AND CLEAR THE OWNER'S FIELDS, which is the half that actually moves the fault. Finalizing
        # alone was measured NOT to fix it: with the wrapper still alive, its `module` keeps the
        # CUDA library loaded to INTERPRETER SHUTDOWN, so the unload lands after nvshmem is gone and
        # after CUDA teardown has begun. `free()` is clean precisely because it drops the module too
        # -- the unload then happens here, while the process is still healthy.
        #
        # Reproduced with no pytest in it (`w8plan/mlirkey/segv_ab.py`, 4 iterations, from_cache
        # True): keep the wrappers and exit 139 at ONE node and at two, on base and on this branch;
        # `free()` each and exit 0 at both. The earlier reading that node count was the
        # discriminator came from the pytest session, where most tests DO free and the 1-node case
        # never crossed the threshold.
        w = ref() if ref is not None else None
        if w is not None:
            w.compiled = None
            w.module = None
            w.nvshmem_kernel_obj = None  # so a later free() does not double-finalize
    _REGISTERED.clear()
    _ORPHANED.clear()
    # LAST, and after every finalize above: dropping these lets the DSL unload, which must not
    # happen until nvshmem has let go of the matching raw handle. Clearing an orphan's entry
    # without having finalized it is precisely the double delete this sweep exists to prevent.
    _REGISTERED_LIBS.clear()
    return n


def agree_on_program_key(prog: str) -> None:
    """Require EVERY rank to have computed the same ``program_key``, or fail on every rank together.

    Args:
        prog: this rank's key from :func:`artifact_cache.resolve_program_key`. Rank-free by
            construction (M5: it is the PE-ordered combination of every rank's MLIR hash), so any
            difference means the ranks are about to compile or load DIFFERENT programs into one
            collective.

    Returns:
        None. A no-op -- deliberately and self-policingly -- when no process group is up: the
        single-device path has no peers to disagree with.

    Raises:
        CollectiveFailure: on a rank whose key differs from rank 0's.
        PeerFailure: on the ranks whose key matched, so none of them proceeds alone.

    Note:
        **A unilateral raise is as bad as a unilateral skip.** A rank that detects a mismatch and
        raises by itself leaves its peers blocked in the next collective until a watchdog kills the
        job, with a traceback naming whatever they happened to be in. So the decision is REDUCED
        before anyone raises -- that is what ``CollectiveGate.guard`` provides, and it is the whole
        reason this is a collective rather than an assert.

        **What it catches, all reachable**: a partial rsync or node-local staging (different
        `source_fingerprint`), a heterogeneous allocation (`arch`), a different container image per
        node (`nvshmem_version`), an unstable `compile_key()`, and -- the one that motivated it -- a
        stale or node-local autotune FREEZE, which makes two ranks elect different configs and so
        build genuinely different functors while every per-rank numeric check stays green.

        **Cost is irrelevant here and must not be optimised away**: one broadcast and one all-reduce
        against a compile measured at 227-699 s.
    """
    try:
        import torch.distributed as dist
    except ImportError:  # pragma: no cover - torch is a hard dep of every distributed path
        return
    if not dist.is_available() or not dist.is_initialized():
        return  # the ONLY exemption, and it is self-policing: no group means no peers

    from fold_cp_ops.distributed.collective_symmetry import CollectiveGate

    gate = CollectiveGate()
    ref = [prog]
    dist.broadcast_object_list(ref, src=0)
    with gate.guard("artifact-program-key"):
        if ref[0] != prog:
            # Attribution is on the FAILURE path only, so a healthy run pays nothing for it. Without
            # it the message is "the keys differ" across N ranks, which names nothing to fix.
            raise RuntimeError(
                f"program_key MISMATCH on rank {gate.rank}: this rank computed {prog[:16]}..., "
                f"rank 0 computed {ref[0][:16]}.... The ranks are about to compile or load DIFFERENT "
                "programs into one collective. Usual causes, in order of likelihood: a stale or "
                "node-local autotune freeze (ranks elected different configs), a partial rsync or "
                "node-local staging (different source fingerprint), a different container image on "
                "one node (different nvshmem version), or a heterogeneous allocation (different arch)."
            )


def _rank_scope() -> Optional["_ac.RankScope"]:
    """The (world_size, pe) this process's nvshmem artifacts are valid for, or None.

    Returns:
        A ``RankScope`` when nvshmem is initialised and reports a positive PE count, else ``None``
        (a single-process use of this route, where there is no rank to scope by).

    Note:
        Artifacts on this route are PER-RANK -- measured: 16 ranks produce 16 distinct shas, and
        handing rank N rank N+1's file faults inside ``nvshmem init.cu:2183`` rather than returning a
        wrong answer. Scoping is therefore applied whenever a rank EXISTS, not only when `register`
        is on: over-scoping costs extra cache misses, which are slow; under-scoping costs a fault
        that also poisons the CUDA context, which is unrecoverable. Never raises -- an uninitialised
        nvshmem is an ordinary state here, not an error.
    """
    if not HAS_NVSHMEM:
        return None
    try:
        n = int(nvshmem.core.n_pes())
        pe = int(nvshmem.core.my_pe())
    except Exception:
        return None
    if n <= 0:
        return None
    try:
        return _ac.RankScope(world_size=n, pe=pe)
    except ValueError:
        return None


def _nvshmem_version() -> str:
    """A version string for the nvshmem runtime, for the artifact key.

    Returns:
        The reported version, or ``"unknown"``. It belongs in the key because the device bitcode is
        LINKED INTO the artifact, so an artifact built against one nvshmem is not a valid input to
        another. ``nvshmemx_culibrary_init`` independently compares the version baked into a module
        against the running host library and refuses a mismatch, so this is belt-and-braces -- but
        the key catches it as a MISS while the runtime catches it as an ERROR, and a miss is cheaper.
    """
    if not HAS_NVSHMEM:
        return "none"
    for attr in ("__version__", "version"):
        v = getattr(nvshmem, attr, None) or getattr(getattr(nvshmem, "core", None), attr, None)
        if v is not None:
            return str(v() if callable(v) else v)
    return "unknown"


def compile_kernel(
    op: Any,
    *compile_args: Any,
    options: str = "",
    link_nvshmem: bool = False,
    register: Optional[bool] = None,
    bitcode: Optional[str] = None,
    persist: Optional["_ac.PersistSpec"] = None,
    prefix: str = "k",
    op_factory: Optional[Any] = None,
    reuse: bool = False,
) -> "_ac.CompiledKernel":
    """ONE compile surface for every kernel in this repo. ``link_nvshmem`` and ``persist`` are the knobs.

    Args:
        op: any constructed cute op.
        *compile_args: its full positional ``__call__`` args, including the trailing stream.
        options: extra compile options. ``--link-libraries`` is added by ``link_nvshmem``; do not
            pass it here.
        link_nvshmem: link the nvshmem device bitcode so in-kernel nvshmem symbols resolve.
        register: hand the module to ``library_init``. Defaults to ``link_nvshmem`` -- a kernel that
            links the bitcode almost always issues a device op -- and may be set False for one that
            links but does not.
        bitcode: override the device bitcode path.
        persist: turn on cross-process artifact reuse. ORTHOGONAL to ``link_nvshmem`` at the surface,
            coupled underneath (see Note).
        prefix: the artifact's symbol prefix.
        reuse: share this compile with any later call in THIS PROCESS whose functor configuration,
            operand layouts, options, ``register``, ``prefix`` and device all match. Forwarded to
            :func:`compile_nvshmem`; ignored on the ``link_nvshmem=False`` branch, which has no
            registration to share and goes through ``artifact_cache.compile_persisted``.
        op_factory: zero-arg callable returning a FRESH, UNTRACED functor equivalent to ``op``.
            **Required whenever ``persist`` is enabled** and ignored otherwise: the program key is
            the hash of the emitted MLIR, and taking it means tracing -- which this tree's
            `template_params` guard refuses to do twice on one instance, so the key cannot come
            from the object about to be compiled. Omitting it with ``persist`` enabled raises
            ``NotImplementedError`` from :func:`artifact_cache.resolve_program_key` rather than
            falling back: the composed key it would have fallen back to was deleted (M5).

    Returns:
        A :class:`artifact_cache.CompiledKernel` -- the SAME type for both paths, so a caller never
        branches on ``link_nvshmem`` to learn what it got.

    Raises:
        NotImplementedError: for ``link_nvshmem=True`` together with tvm-ffi. **This combination is
            expressible today and silently broken**, which is why the guard is a raise rather than a
            comment: in-kernel nvshmem needs a materialised ``CUlibrary`` to register device state
            into, only the EAGER base ``.to()`` produces one, and the TVM-FFI subclass replaces
            ``.to()`` with ``return self``. The result is a ``library_init`` with nothing to register
            and an artifact tagged ``dump_object`` that no loader can read back. The day nvshmem4py
            supports tvm-ffi this is one ``raise`` to delete.
        RuntimeError: when ``register`` and the executor exposes no ``jit_module.cuda_library``.

    Note:
        **Orthogonal at the surface, coupled in the implementation.** Every combination of
        ``link_nvshmem`` x ``persist`` is expressible and meaningful, which is what makes them
        orthogonal to a caller. But ``link_nvshmem`` decides which persist BACKEND applies (the
        backend follows tvm-ffi) and whether the key must carry ``(world_size, pe)`` (a wrong-rank
        artifact faults inside nvshmem's init). That coupling is exactly what a caller should not
        have to know -- it is the argument FOR consolidating -- but a maintainer who reads
        "orthogonal" unqualified will eventually try to handle the two independently.
    """
    if link_nvshmem and "--enable-tvm-ffi" in (options or ""):
        raise NotImplementedError(
            "link_nvshmem=True with --enable-tvm-ffi is not supported. In-kernel nvshmem needs a "
            "materialised CUlibrary to register device state into; only the EAGER base .to() "
            "produces one, and the TVM-FFI subclass overrides .to() to `return self`. The compile "
            "would appear to succeed and then fail at launch against unwritten device state, and "
            "any artifact it wrote could not be read back. If nvshmem4py has since gained TVM-FFI "
            "support, delete this raise and re-run tests/distributed/test_gemm_bitcode_compile.py."
        )
    if register is None:
        register = link_nvshmem
    if not link_nvshmem:
        if register:
            raise NotImplementedError(
                "register=True requires link_nvshmem=True: there is no nvshmem device state to "
                "register without the device bitcode linked."
            )
        compiled = _ac.compile_persisted(
            op, *compile_args, options=options, persist=persist, prefix=prefix,
            op_factory=op_factory,
        )
        return _ac.CompiledKernel(
            executor=compiled, from_cache=getattr(compiled, "_cpo_from_cache", False),
            options=options, _compiled=compiled,
        )
    wrapped = compile_nvshmem(
        op, *compile_args, bitcode=bitcode, extra_options=options,
        register=register, persist=persist, prefix=prefix, op_factory=op_factory, reuse=reuse,
    )
    return _ac.CompiledKernel(
        executor=wrapped.executor,
        from_cache=wrapped.from_cache,
        options=wrapped.options,
        device_id=wrapped.device_id,
        nvshmem_kernel_obj=wrapped.nvshmem_kernel_obj,
        program_key=wrapped.program_key,
        artifact_key=wrapped.artifact_key,
        artifact_path=wrapped.artifact_path,
        _compiled=wrapped.compiled,
        _module=wrapped.module,
    )


def compile_nvshmem(
    op: Any,
    *compile_args: Any,
    bitcode: Optional[str] = None,
    extra_options: str = "",
    register: bool = True,
    persist: Optional["_ac.PersistSpec"] = None,
    prefix: str = "k",
    op_factory=None,
    reuse: bool = False,
) -> CompiledGemmBitcode:
    """Compile ANY constructed cute op with the nvshmem device bitcode linked and registered.

    The canonical ``cute.compile(..., --link-libraries) → .to(dev) → library_init``
    boilerplate (tvm-ffi OFF) — the part every bitcode-route caller duplicates.

    NOTHING HERE IS GEMM-SPECIFIC, and the name this function used to carry
    (``compile_gemm_with_bitcode``) was not harmless: the author of a *signal*
    kernel reasonably read it as not applying to them and re-typed this body inline
    at ``trimul_autotuned.py``. ``op`` is any constructed op with a ``__call__``
    the DSL can trace — a GemmSm90-family op (``GemmDefaultSm90(...)``, a
    ``GemmA2ASm90(...)`` after ``configure_a2a``), a device-signal kernel, anything.
    ``compile_args`` are that op's full positional ``__call__`` args INCLUDING the
    trailing stream — for SM90 GEMMs: ``(mA, mB, mD, mC, epi_args, scheduler_args,
    stream[, mB2])``.

    CRITICAL (T2.0d), FOR A GEMM: the tensors in ``compile_args`` must be **STATIC-shape**
    ``from_dlpack`` CuTe tensors (NOT ``.mark_layout_dynamic()`` / fake tensors),
    the ``stream`` must be REAL (``cutlass.torch.current_stream()``), and
    ``scheduler_args`` should come from ``make_scheduler_args`` with a concrete
    ``max_active_clusters``. Otherwise the host launch SIGFPEs in
    ``fast_divmod.create_divisor`` (the shape-derived divisor reads 0).

    Returns a :class:`CompiledGemmBitcode`. When ``register`` (default), the
    linked nvshmem bitcode is registered with ``library_init`` (required for any
    in-kernel nvshmem device call to resolve); pass ``register=False`` for a
    kernel that issues no nvshmem op (still bitcode-linked, unregistered).

    ``persist`` (optional) turns on cross-process artifact reuse for THIS compile.
    Omit it and the path is byte-identical to before, with no new failure mode. Its
    ``key`` must carry every config the compile depends on that the source
    fingerprint does not — tile/cluster, dtypes, cp, shapes when static — because an
    omitted component is a silently-served WRONG artifact. Do NOT put the rank in it:
    this function adds ``(world_size, pe)`` itself, since forgetting costs a FAULT
    rather than a miss. ``prefix`` is the artifact's symbol prefix; it is baked into
    the image bytes, so it is part of the key and rarely worth changing.

    ``reuse`` (optional) turns on IN-PROCESS sharing for THIS compile. Default False, which is
    byte-identical to before with no new failure mode. When True the compile is looked up in
    `_REUSE` under `_reuse_key` -- functor type + ``compile_key()`` + operand layouts + options +
    register + prefix + device -- and an existing entry is returned with its holder count bumped
    instead of being recompiled. This is what makes a SECOND engine in one process free; without it
    the distributed path recompiles unconditionally, because ``persist`` is None by default and
    nothing else caches these.

    ``reuse`` and ``persist`` are independent and compose: ``persist`` skips the compile across
    PROCESSES by reading an artifact, ``reuse`` skips it (and the artifact read, and the
    ``library_init``) WITHIN one. With both on, the second engine in a process never touches disk.

    Returns:
        A :class:`CompiledGemmBitcode`. ``from_cache`` says which path produced it;
        on a hit ``module`` holds the loaded ``ExternalBinaryModule`` and ``compiled``
        is None, on a miss the reverse. A `_REUSE` hit returns the SAME OBJECT a previous call
        returned -- ``from_cache`` therefore still describes how that object was first produced,
        which is deliberate: it is a property of the artifact, not of this call.

    Raises:
        RuntimeError: if ``register`` and the compiled executor exposes no
            ``jit_module.cuda_library`` handle — there is then nothing to register,
            and letting the kernel launch anyway means an in-kernel nvshmem call
            resolves against unwritten device state, which faults rather than raises.
            Raised identically on the cache-hit path, where the same handle is what
            makes a reloaded artifact usable at all.

    Note:
        A cache MISS is never an error and never propagates: a failed read quarantines
        and compiles, a failed write returns the correctly-compiled kernel anyway. The
        caller cannot tell a broken cache from a cold one except through
        ``from_cache`` and ``artifact_cache.STATS`` — which is the intended blast
        radius for a mechanism whose whole purpose is to be optional.
    """
    if "--enable-tvm-ffi" in (extra_options or ""):
        # THE HOLE THIS CLOSES IS LIVE, not hypothetical: `extra_options` is a free-form string, so
        # this combination reached `cute.compile` with nothing to stop it. What follows is a `.to()`
        # that returns `self` (no CUlibrary, so `library_init` has nothing to register) and an
        # artifact tagged `dump_object` that no loader can read back.
        raise NotImplementedError(
            "extra_options must not contain --enable-tvm-ffi: in-kernel nvshmem needs the eager "
            "base .to() to materialise a CUlibrary, and the TVM-FFI subclass overrides .to() to "
            "`return self`. See compile_kernel() for the same guard on the consolidated surface."
        )

    import cuda.core

    dev_id = cuda.core.Device().device_id
    rank = _rank_scope()

    # ASSEMBLED BEFORE THE LOOKUP, because `options` is a KEY COMPONENT: it carries the link
    # libraries and the tvm-ffi flag, so it must exist before the program key is computed. Under
    # schema 1 it keyed in neither compile path, which is audit row A3.
    bc = bitcode if bitcode is not None else _device_bitcode()
    libs = [bc]
    # Auto-append any op-specific device bitcode the constructed op declares via an optional
    # _extra_link_bitcode() hook (an op may vendor its own device .bc alongside the stock nvshmem one).
    # The cute-DSL link-libraries value is COMMA-separated (base_dsl/dsl.py joins with ','), so we
    # comma-join. An op with no such hook / returning [] -> byte-identical single-bitcode link.
    extra_fn = getattr(op, "_extra_link_bitcode", None)
    if extra_fn is not None:
        try:
            libs.extend([p for p in (extra_fn() or []) if p])
        except Exception:
            pass
    options = f" --link-libraries={','.join(libs)}{extra_options}"  # NO --enable-tvm-ffi (T0.3)

    # --- in-process reuse, BEFORE the disk lookup ----------------------------------------------
    # Ordered first because it is strictly cheaper: a hit here skips the artifact read, the
    # `.to(dev)` and the `library_init` as well as the compile, and it cannot be wrong in a way the
    # disk path could -- the object was produced by THIS process, on THIS device, for THIS rank.
    rkey = None
    if reuse:
        rkey = _reuse_key(op, compile_args, options, register, prefix, dev_id)
        hit = _REUSE.get(rkey)
        if hit is not None:
            hit.holders += 1
            REUSE_STATS["hits"] += 1
            return hit
        REUSE_STATS["misses"] += 1

    # --- cache lookup, BEFORE any compile ------------------------------------------------------
    # Everything here is verified by `artifact_cache.read_artifact` -- schema, backend tag, prefix,
    # pe, world_size, length and sha -- and every one of those checks precedes `library_init`. That
    # ordering is the design: a wrong-rank artifact handed to `library_init` is a
    # CUDA_ERROR_ILLEGAL_ADDRESS inside nvshmem `init.cu:2183` that poisons the context for
    # everything after it, so a check that runs afterwards is not a check.
    keysha = None
    prog = None
    if persist is not None and persist.enabled:
        # SPLIT KEY. The program key is rank-FREE and must be byte-identical on every rank of the
        # job -- which is what lets the agreement check (below) be a one-value comparison. `pe` is
        # added by `artifact_key` and is the ONLY component allowed to differ.
        #
        # `options` is passed so it KEYS: it carries the link libraries and the tvm-ffi flag, and
        # under schema 1 it was in neither path's key at all. It is path-normalised inside
        # `resolve_program_key`, because the bitcode path is absolute and differs per node.
        prog = _ac.resolve_program_key(
            op_factory,
            spec=persist,
            prefix=prefix,
            backend=_ac.BACKEND_DUMP_OBJECT,
            op=op,
            compile_args=compile_args,
            options=options,
            world_size=rank.world_size if rank is not None else None,
            extra=("nvshmem=" + _nvshmem_version(), f"register={bool(register)}"),
        )
        agree_on_program_key(prog)
        keysha = _ac.artifact_key(prog, rank)
        art = _ac.read_artifact(
            keysha, persist, prefix=prefix, backend=_ac.BACKEND_DUMP_OBJECT, rank=rank
        )
        if art is not None:
            try:
                return _from_artifact(art, prefix, dev_id, register, prog, keysha)
            except Exception as exc:
                # A verified artifact that still fails to load is corruption the checks cannot see.
                # Quarantine it so no later process pays the same failure, then fall through and
                # compile -- the caller must get a working kernel whatever the cache does.
                _ac._reject(art.path, f"load failed after verification: {type(exc).__name__}: {exc}")

    compiled = cute.compile(op, *compile_args, options=options)

    executor = compiled.to(dev_id)
    nv_obj = _register(executor, register)
    wrapper = CompiledGemmBitcode(
        executor=executor,
        nvshmem_kernel_obj=nv_obj,
        options=options,
        device_id=dev_id,
        # Keeps the CUDA library loaded for as long as NVSHMEM's registration table references it.
        # Without this the library unloads at THIS `return`. See the field's docstring.
        compiled=compiled,
        program_key=prog,
        artifact_key=keysha,
        artifact_path=(
            _ac.artifact_paths(keysha, persist, rank)[0] if keysha is not None else None
        ),
    )
    if nv_obj is not None:
        _note_registration(wrapper)
    if rkey is not None:
        # Published AFTER `_note_registration`, so a wrapper is never reachable from the table
        # before the sweep knows about it: a second caller could otherwise take a hit on an object
        # whose registration has not been recorded, and then `free()` it -- dropping a `reg_seq`
        # that was never stamped.
        wrapper.reuse_key = rkey
        _REUSE[rkey] = wrapper

    # --- publish, AFTER the kernel is known good -----------------------------------------------
    # Serialising last means a compile that succeeded is never lost to a serialisation failure, and
    # that the bytes written are the bytes of an object that has already been registered.
    if keysha is not None and persist is not None and persist.writable:
        try:
            blob = _ac.serialize(compiled, _ac.BACKEND_DUMP_OBJECT, prefix)
            _ac.write_artifact(
                blob,
                keysha,
                persist,
                prefix=prefix,
                backend=_ac.BACKEND_DUMP_OBJECT,
                rank=rank,
                extra_meta={
                    "options": options,
                    "nvshmem": _nvshmem_version(),
                    "program_key": prog,
                    # The witness. Free here (the module is traced and built), never load-bearing
                    # for this process, and the only thing that can later FALSIFY the composed key.
                    "ir_sha": _ac.ir_sha(compiled),
                },
            )
        except Exception:
            # Never fatal. The caller holds a correct kernel; a cache that can fail the run is worse
            # than no cache. `artifact_cache.STATS.write_failures` records it for anyone looking.
            _ac.STATS.write_failures += 1
    return wrapper


def _register(executor: Any, register: bool) -> Any:
    """Hand a compiled executor's CUDA library to nvshmem, or refuse to pretend it worked.

    Args:
        executor: the result of ``compiled.to(dev)``. Must be the EAGER base ``.to()`` product; a
            TVM-FFI executor returns ``self`` and materialises no library, which is the whole reason
            this route exists.
        register: when False, returns ``None`` without touching nvshmem — for a kernel that links the
            bitcode but issues no device nvshmem op.

    Returns:
        The registered ``NvshmemKernelObject``, or ``None`` when ``register`` is False.

    Raises:
        RuntimeError: when there is no ``jit_module.cuda_library`` to register. Raising is the point:
            an in-kernel nvshmem call against unregistered device state FAULTS rather than raising,
            several frames from here and with the CUDA context already poisoned.
    """
    if not register:
        return None
    jm = getattr(executor, "jit_module", None)
    if jm is None or not getattr(jm, "cuda_library", None):
        raise RuntimeError(
            f"compiled op has no cuda_library handle for library_init (jit_module={jm!r})"
        )
    nv_obj = nvshmem.core.NvshmemKernelObject.from_handle(int(jm.cuda_library[0]))
    nvshmem.core.library_init(nv_obj)
    return nv_obj


def _from_artifact(art, prefix: str, dev_id: int, register: bool, prog=None, keysha=None) -> CompiledGemmBitcode:
    """Turn a VERIFIED artifact into a live, registered kernel.

    Args:
        art: a ``LoadedArtifact`` from ``artifact_cache.read_artifact`` — already checked for schema,
            backend, prefix, pe, world_size, length and sha. Do not call this with anything else.
        prefix: the symbol to ``getattr`` off the loaded module.
        dev_id: the CUDA device to materialise the library on.
        register: as in :func:`compile_nvshmem`.

    Returns:
        A :class:`CompiledGemmBitcode` with ``from_cache=True``, holding the loaded module so its
        CUDA library outlives every launch.

    Raises:
        Exception: anything the loader or the registration raises. The caller catches, quarantines
            and recompiles — a cache must never be able to fail a run.

    Note:
        ``getattr(module, prefix)`` returns the BASE ``CudaDialectJitCompiledFunction``, not a
        TVM-FFI subclass, so ``.to()`` is the eager one that actually materialises a ``CUlibrary``.
        That is the single property this whole route turns on, and it is why the artifact must have
        been written by a tvm-ffi-OFF compile — a fact the backend tag has already established by the
        time execution reaches here.
    """
    module, fn = _ac.deserialize(art, prefix)
    executor = fn.to(dev_id)
    nv_obj = _register(executor, register)
    wrapper = CompiledGemmBitcode(
        executor=executor,
        nvshmem_kernel_obj=nv_obj,
        options=str(art.meta.get("options", "")),
        device_id=dev_id,
        compiled=None,
        module=module,
        from_cache=True,
        program_key=prog,
        artifact_key=keysha,
        artifact_path=art.path,
    )
    if nv_obj is not None:
        _note_registration(wrapper)
    return wrapper


#: BACKWARD-COMPATIBLE ALIAS. `compile_nvshmem` was named `compile_gemm_with_bitcode` until the name
#: was found to be actively misleading: nothing in the function is GEMM-specific, and a *signal*
#: kernel's author read the name as not applying and re-typed the body inline. ~50 call sites across
#: kernels, benchmarks and tests import the old spelling; the alias keeps the rename a pure rename
#: -- one readable diff with a provably nil behaviour change -- instead of a 50-file edit whose real
#: content nobody can see. New code uses `compile_nvshmem`. Removing the alias is its own commit.
compile_gemm_with_bitcode = compile_nvshmem


def make_runner(
    compiled: Any, build_run_args: Callable[[], tuple], *, fold: bool = False
) -> Callable[[], Any]:
    """Build a zero-arg ``run()`` that invokes ``compiled`` with the kernel's positional args.

    ``compiled`` is a :class:`CompiledGemmBitcode` (or any callable taking the kernel's ``__call__``
    positional args). ``build_run_args`` is a zero-arg callable returning the FULL positional tuple
    for ``compiled`` — the ``from_dlpack``'d (and optionally ``mark_layout_dynamic`` /
    ``mark_compact_shape_dynamic``'d) CuTe views of the CURRENT input tensors, plus scalars/stream.

    The ``fold`` flag chooses WHEN those CuTe views are constructed. The kernel COMPILE is unrelated
    and already one-shot (cached) — ``fold`` only controls per-launch host-side argument marshaling.

    ``fold=False`` (DEFAULT — the only deployment-correct mode)
        ``build_run_args`` is called on EVERY launch, so the CuTe views always point at the current
        input tensors. **Mandatory for any real workflow**: an eager forward pass hands a NEW
        activation tensor (new data pointer) each call, and the views MUST be rebuilt to point at it.
        Cost: the per-call ``from_dlpack`` (+ the dynamic ``mark``, slightly pricier than a static
        ``from_dlpack``) host marshaling. This per-call cost is unavoidable for the A2A kernels
        because they cannot use the tvm-ffi launch path (nvshmem incompatibility) that lets the stock
        fold_cp_ops kernels accept a ``torch.Tensor`` directly at <0.5 us. In deployment you may fold the
        FIXED tensors (weights, a reused recv buffer) into a partial closure, but the varying
        activation view is rebuilt here every call.

    ``fold=True`` (BENCHMARKING ONLY — **no real deployment use case**)
        ``build_run_args`` is called ONCE; the resulting CuTe views are CAPTURED and reused on every
        launch. This isolates the DEVICE kernel from the per-call marshaling above — useful only to
        measure the kernel's true device time or to confirm a static-vs-dynamic device-perf tie in a
        benchmark that deliberately hammers ONE fixed input. **It is silently WRONG for any workflow
        whose inputs change**: the captured views hold a fixed data pointer, so a launch after the
        input tensor is replaced computes on the STALE (captured) tensor. (In-place mutation of the
        SAME buffer IS reflected — the pointer is unchanged; a NEW allocation is NOT.) Pinned by
        ``tests/distributed/test_gemm_bitcode_compile.py::test_fold_ignores_new_tensor`` (xfail).
    """
    if fold:
        captured = build_run_args()

        def run():
            return compiled(*captured)

        return run

    def run():
        return compiled(*build_run_args())

    return run


# ---------------------------------------------------------------------------
# Plain D = A @ B convenience (first milestone + the worked recipe template).
# ---------------------------------------------------------------------------
def compile_plain_sm90_gemm_with_bitcode(
    A,
    B,
    D,
    *,
    tile_shape_mn=(128, 256),
    cluster_shape_mnk=(1, 1, 1),
    pingpong: bool = False,
    persistent: bool = True,
    max_swizzle: int = 8,
    register: bool = True,
    GemmCls=GemmDefaultSm90,
    stream: Optional[Any] = None,
):
    """Compile a plain ``D = A @ B`` SM90 GEMM via the bitcode route.

    ``A``/``B``/``D`` are real torch tensors in the PERMUTED ``(m,k,l)`` /
    ``(n,k,l)`` / ``(m,n,l)`` layout the stock ``gemm()`` feeds after ``perm3d``
    (k-major A/B, n-major D; the L batch axis must carry a NON-degenerate stride —
    build via ``randn(L,M,K).permute(1,2,0)``, never ``unsqueeze(-1)``). Returns
    ``(compiled, run)`` where ``compiled`` is a :class:`CompiledGemmBitcode` and
    ``run()`` executes it on the same tensors. Caller keeps ``compiled`` alive.

    Constructs the op + ``epi_args`` / ``make_scheduler_args``
    as ``gemm_interface`` does, with STATIC-shape ``from_dlpack`` tensors — the
    T2.0d recipe. ``GemmCls`` can be swapped for a derived A2A subclass (e.g.
    ``GemmA2ASm90`` after its ``configure_a2a``); the compile path is identical.
    """
    from fold_cp_ops._internal.compile_time.cute_dsl_utils import torch2cute_dtype_map

    a_dtype = torch2cute_dtype_map[A.dtype]
    if stream is None:
        stream = cutlass_torch.current_stream()
    mac = get_max_active_clusters(cluster_shape_mnk[0] * cluster_shape_mnk[1]) if persistent else 0

    gemm_obj = GemmCls(
        Float32,
        a_dtype,
        tile_shape_mn,
        cluster_shape_mnk,
        pingpong=pingpong,
        is_persistent=persistent,
    )
    epi_args = GemmCls.EpilogueArguments(
        alpha=None,
        beta=None,
        mRowVecBroadcast=None,
        mColVecBroadcast=None,
        add_to_output=False,
        rounding_mode=None,
        sr_seed=None,
    )
    scheduler_args = make_scheduler_args(mac, Int32(max_swizzle), None, None)

    cA = from_dlpack(A, assumed_align=16)  # STATIC shape (no mark_layout_dynamic)
    cB = from_dlpack(B, assumed_align=16)
    cD = from_dlpack(D, assumed_align=16)
    compiled = compile_nvshmem(
        gemm_obj,
        cA,
        cB,
        cD,
        None,
        epi_args,
        scheduler_args,
        stream,
        None,
        register=register,
    )

    def run(stream_=None):
        rA = from_dlpack(A, assumed_align=16)
        rB = from_dlpack(B, assumed_align=16)
        rD = from_dlpack(D, assumed_align=16)
        compiled(rA, rB, rD, None, epi_args, scheduler_args, stream_ or stream, None)

    return compiled, run

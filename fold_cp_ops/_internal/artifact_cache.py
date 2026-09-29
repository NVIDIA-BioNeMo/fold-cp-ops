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

"""Cross-process reuse of a compiled CuTe-DSL artifact, opted into with ``persist=``.

WHAT THIS IS FOR. A kernel on this repo's distributed path costs 50-330 s to compile and 1.8-19 ms
to run, and NOTHING of that compile survives the process. Per-cell launch isolation is required for
independent reasons (a wedge in one cell must not take a sweep with it), so in-process reuse cannot
recover it -- only an artifact on disk can. The same applies to any workflow invoked more than once:
a serving process that restarts, a training job re-entering a shape, a multi-process launcher.

WHY IT IS NOT ``@jit_cache``. ``cache_utils.jit_cache`` serialises with ``export_to_c``, which
generates a C header FIRST and fails on any argument it cannot express -- and every kernel here
passes its epilogue as a NamedTuple whose first field is a Tensor. The single-device kernels get away
with it because they compile ``--enable-tvm-ffi``, whose ``export_to_c`` OVERRIDE never touches the
header generator. The A2A kernels cannot compile that way: in-kernel nvshmem needs a materialised
``CUlibrary`` to register device state into, only the EAGER base ``.to()`` produces one, and the
TVM-FFI subclass replaces ``.to()`` with ``return self``. See ``docs/jit_cache_survival.md`` §1.

ONE SURFACE, TWO BACKENDS, AND THE CALLER NEVER PICKS. Measured (§3 corner E): the compile OPTION
decides which serializer can read the artifact back, and the mismatched combination fails on the READ
side, in a LATER PROCESS, after a write that raised nothing:

    compile --enable-tvm-ffi -> dump_to_object   OK 46616 B -> load_module  FAIL (k_args_spec)
    compile (no tvm-ffi)     -> dump_to_object   OK 29600 B -> load_module  OK

So the backend is derived from the compiled object's own class (:func:`backend_for`) and RECORDED in
the artifact's meta; a tag mismatch on load is a reject-and-recompile, never an attempted load. That
check is the difference between "this entry is stale" and a ``DSLRuntimeError`` from inside an
execution engine, raised in a process that did not create the file.

WHAT IS CHECKED, AND WHEN. Every meta field is verified BEFORE any DSL call and before -- for the
nvshmem backend -- ``library_init``. A wrong-rank artifact handed to ``library_init`` is a
``CUDA_ERROR_ILLEGAL_ADDRESS`` inside ``nvshmem init.cu:2183`` that poisons the CUDA context for
everything after it, so a check that runs after the registration is not a check.

NOTHING IS DELETED. A bad entry is renamed to ``.rejected`` with its reason recorded alongside.
Skipping it instead would leave every later process re-reading and re-rejecting the same file
forever; and this repo forbids ``rm`` outright, so a delete was never available.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from fold_cp_ops._internal.cache_security import (
    CacheRoot,
    ensure_private_file,
    is_within,
    validate_cache_root,
)
from fold_cp_ops._internal.cache_utils import (
    FileLock,
    cache_root,
)
from fold_cp_ops._internal.compile_time.template_params import canonical_key_bytes

#: Backend tags. These strings are WRITTEN INTO ARTIFACT META and are therefore part of the on-disk
#: format: renaming one invalidates every existing entry (they will be rejected as a tag mismatch and
#: recompiled, which is safe but not free). They also enter the key, so two backends never collide.
BACKEND_DUMP_OBJECT = "dump_object"
BACKEND_EXPORT_C = "export_c"

#: Bumped whenever the meta schema or the path layout changes in a way a previously-written entry
#: cannot satisfy. It is a KEY component, so an old entry becomes unreachable rather than
#: mis-interpreted -- the cheap, always-correct migration.
#:
#: 1 -> 2 (2026-08-22): the key SPLIT into a rank-free PROGRAM key + `artifact_key` (adds `pe`),
#: `options` entered the key at all for the first time, the operand signature was added, and
#: `ir_sha` joined the meta. Every version-1 entry is unreachable under 2, which is the intent:
#: those entries were minted by a key that could not see the compile options.
ARTIFACT_SCHEMA_VERSION = 2

#: Where artifacts live when a `PersistSpec` does not say. Kept separate from `get_cache_path()`'s
#: `@jit_cache` entries so a human can reason about (and clear) one without touching the other.
ARTIFACT_DIR_ENV = "CPO_JIT_ARTIFACT_DIR"

#: Set to "0" to make every `persist=` a no-op process-wide, exactly as if the argument were omitted.
#: This is the correctness-run switch: a warm artifact cache during a correctness sweep hides the
#: very source edit the sweep exists to test. It mirrors `CPO_CACHE_ENABLED` for the same reason.
ARTIFACT_ENABLED_ENV = "CPO_JIT_ARTIFACT_ENABLED"


def artifacts_enabled() -> bool:
    """Is the artifact cache on at all?

    Read at CALL time, not at import: a test that flips ``CPO_JIT_ARTIFACT_ENABLED`` must not have to
    re-import the module for the flip to take, and a launcher that exports it after Python starts
    would otherwise be silently ignored. (``cache_utils.CACHE_ENABLED`` is an import-time constant and
    that difference has bitten this repo before -- a half-renamed env var left a disk cache enabled
    during a correctness run.)

    Returns True unless the env var is exactly ``"0"``.
    """
    return os.environ.get(ARTIFACT_ENABLED_ENV, "1") != "0"


@dataclass(frozen=True)
class PersistSpec:
    """Where a compiled artifact is stored, and what makes an entry stale.

    Args:
        key: OPTIONAL extra key components, defaulting to empty -- and empty is the NORMAL case.

            **This rule INVERTED at schema 2 and the old one is now wrong.** Under schema 1 the key
            was required and an empty one raised, on the reasoning that "an empty key claims every
            compile of this kernel is interchangeable". That was correct while the caller hand-listed
            identity. It is not correct now: identity is DERIVED by :func:`resolve_program_key` from
            the hash of the functor's EMITTED MLIR, combined across ranks, so a caller supplying
            nothing is relying on the derivation rather than forgetting to describe anything. Supply
            a value only for something the derivation genuinely cannot see.

            Must be picklable and stable across processes; a value whose ``repr`` carries a memory
            address makes the key unstable and turns every lookup into a miss -- slow rather than
            wrong, but still a bug. Do NOT put the rank in it: :func:`artifact_key` adds ``pe``
            itself, because a caller who had to remember would eventually not, and the consequence
            of forgetting is a FAULT rather than a miss.
        dir: directory for artifacts. Defaults to ``$CPO_JIT_ARTIFACT_DIR``, else
            ``<cache_dir>/artifacts``. A shared filesystem is permitted but its coherence is the
            operator's concern; node-local is the default and the tested configuration.
        mode: ``"rw"`` (default -- read and mint), ``"r"`` (read only, NEVER mint: for a perf gate,
            where a run that writes entries makes its own first and second runs different
            measurements), or ``"off"`` (identical to ``persist=None``).

    Raises:
        ValueError: on an unknown ``mode``, or on an empty ``key``. An empty key means "every compile
            of this kernel is interchangeable", which is true for no kernel in this repo, so it is far
            more likely to be an oversight than an intent.
    """

    key: tuple = ()
    dir: Optional[str] = None
    mode: str = "rw"

    def __post_init__(self) -> None:
        if self.mode not in ("rw", "r", "off"):
            raise ValueError(f"PersistSpec.mode must be 'rw'|'r'|'off', got {self.mode!r}")

    @property
    def enabled(self) -> bool:
        """False when this spec is off, or when the process-wide switch is off."""
        return self.mode != "off" and artifacts_enabled()

    @property
    def writable(self) -> bool:
        """True only in ``"rw"``; ``"r"`` reads without ever minting."""
        return self.mode == "rw" and artifacts_enabled()

    def resolve_root(self) -> CacheRoot:
        """The directory this spec's artifacts live in, created ``0700``, and whether it is safe.

        Purpose
            Decide location and permission together. An artifact from this directory is handed to
            ``load_module`` and EXECUTED, so "where" without "may I" is half an answer.

        Functionality & semantics
            Precedence is unchanged: ``self.dir``, then ``$CPO_JIT_ARTIFACT_DIR``, then
            ``<get_cache_path()>/artifacts``. All three go through the same validator, deliberately
            -- an operator-supplied ``CPO_JIT_ARTIFACT_DIR`` is the one MOST likely to point
            somewhere shared, so exempting it would exempt the risky case and check only the safe
            default.

        **SHARED STORAGE IS SAME-UID ONLY, and that is a boundary rather than a limitation to work
        around.** An artifact from this directory is handed to ``load_module`` and EXECUTED, so two
        users pointed at one artifact directory can each hand the other code to run. The validator
        enforces it -- the final root must be owned by the effective uid -- and no option relaxes
        it. A team that wants shared build output should share it through a channel that
        authenticates its contents (a signed artifact store), never through a directory both can
        write. Sharing across MACHINES for one user is fine and is the normal case: the key is
        content-derived, not host-derived.

        **KEYS ARE STABLE UNDER THE EXISTING NORMALIZATION.** The name an artifact is stored under
        comes from :func:`artifact_key`, which hashes `template_params.canonical_key_bytes` -- a
        type-tagged, versioned, MessagePack encoding over text and bytes. It is therefore identical
        across processes, interpreter runs and hosts for the same inputs, which is what makes a
        directory shared between a user's own machines correct rather than merely lucky. It is NOT
        stable across a bump of ``KEY_SCHEMA_VERSION``: that is deliberate, and makes every prior
        name unreachable instead of reinterpreted.

        Returns:
            A `cache_security.CacheRoot`. ``usable`` False means read and write must both be
            skipped; :func:`read_artifact` treats it as a miss and :func:`write_artifact` as a
            skipped write, so an unusable root costs a recompile and never an error.
        """
        if self.dir:
            return validate_cache_root(self.dir)
        env_dir = os.environ.get(ARTIFACT_DIR_ENV)
        if env_dir:
            return validate_cache_root(env_dir)
        # THE DEFAULT PARENT'S VERDICT PROPAGATES, and its `artifacts` child is not created when the
        # parent is rejected. `get_cache_path()` returns a path whether or not the directory is safe
        # -- that is deliberate, so a message can name it -- so building the child from it would
        # mkdir INSIDE a directory this process has just refused, which is the one place it must not
        # write. Asking `cache_root()` instead gets the verdict with the path.
        parent = cache_root()
        if not parent.usable:
            return CacheRoot(
                path=parent.path / "artifacts",
                usable=False,
                reason=f"its parent {parent.path} was refused: {parent.reason}",
            )
        return validate_cache_root(parent.path / "artifacts")

    def resolve_dir(self) -> Path:
        """The directory this spec's artifacts live in, created if absent.

        Returns:
            ``Path`` to the directory. Precedence: ``self.dir``, then ``$CPO_JIT_ARTIFACT_DIR``,
            then ``<get_cache_path()>/artifacts``. Path only -- callers that touch the filesystem
            must consult :meth:`resolve_root` for the verdict first.
        """
        return self.resolve_root().path


@dataclass
class RankScope:
    """The (world_size, pe) an nvshmem artifact is valid for.

    Artifacts carrying in-kernel nvshmem are PER-RANK -- measured: 16 ranks produce 16 distinct
    shas, and handing rank N rank N+1's file faults inside ``nvshmem init.cu:2183`` rather than
    producing a wrong answer. Both fields therefore enter the key AND the filename AND the meta,
    and all three are checked.

    Args:
        world_size: the job's PE count. A different world means a different PE mapping even for the
            same ``pe``, so it is not redundant with ``pe``.
        pe: this process's PE id.

    Raises:
        ValueError: if ``pe`` is outside ``[0, world_size)``, which is always a caller bug and is
            cheaper to catch here than as a cache miss nobody explains.
    """

    world_size: int
    pe: int

    def __post_init__(self) -> None:
        if not (0 <= self.pe < self.world_size):
            raise ValueError(f"pe {self.pe} out of range for world_size {self.world_size}")

    @property
    def infix(self) -> str:
        """The human-readable filename infix, e.g. ``".pe3.of16"``."""
        return f".pe{self.pe}.of{self.world_size}"


@dataclass
class LoadedArtifact:
    """A verified artifact and everything needed to decide what to do with it.

    Attributes:
        path: the ``.o`` that was read.
        meta: the parsed sidecar, already checked against the request.
        blob: the artifact bytes, already hash-verified against ``meta["sha256"]``.
        backend: the backend tag, echoed for the caller that must pick a loader.
    """

    path: Path
    meta: dict
    blob: bytes
    backend: str


def backend_for(compiled: Any) -> str:
    """Which serializer can round-trip this compiled object.

    Reads the object's CLASS rather than a caller-supplied flag, because the caller does not choose
    the compile options at the point ``persist=`` is handled, and a second source of truth about
    tvm-ffi is one more thing that can disagree with reality.

    Args:
        compiled: the return of ``cute.compile(...)``.

    Returns:
        :data:`BACKEND_EXPORT_C` when a TVM-FFI class is in the MRO (its ``export_to_c`` override
        bypasses the C header generator and is the only serializer that reads back), otherwise
        :data:`BACKEND_DUMP_OBJECT` (``dump_to_object`` -> ``load_module(enable_tvm_ffi=False)``).

    Note:
        The mismatched pairing does NOT raise on write. ``dump_to_object`` on a tvm-ffi object emits
        a plausible 46 KB image carrying unresolvable ``__tvm_ffi_*`` references, and the failure
        surfaces only when another process loads it. That is why the tag is stored and checked.
    """
    return (
        BACKEND_EXPORT_C
        if any("TVMFFI" in c.__name__ for c in type(compiled).__mro__)
        else BACKEND_DUMP_OBJECT
    )


def normalize_options(options: str) -> str:
    """Strip node-local absolute paths out of a compile-options string, keeping the flags.

    Args:
        options: the string handed to ``cute.compile(..., options=...)``.

    Returns:
        The same string with every absolute path replaced by its BASENAME.

    Note:
        **Mandatory, not hygiene.** ``options`` decides what is linked and whether tvm-ffi is on, so
        it must key -- but ``nvshmem.core.find_device_bitcode_library()`` returns an ABSOLUTE path and
        mount points differ per node. Keying the raw string would make every rank on a two-node job
        compute a different key, so the cross-rank agreement check (S11b) would fire on a healthy run.
        A check that cries wolf gets switched off, so the normalisation is part of making the check
        usable rather than a nicety.

        The bitcode's IDENTITY is not lost by this: it rides ``nvshmem_version``, which is a separate
        key component. If that ever proves too coarse -- two builds of the same nvshmem version with
        different device bitcode -- the fix is to key a CONTENT hash of the ``.bc``, never the path.
    """
    out = []
    for tok in (options or "").split():
        if "=" in tok:
            flag, _, val = tok.partition("=")
            parts = [os.path.basename(v) if v.startswith("/") else v for v in val.split(",")]
            out.append(f"{flag}={','.join(parts)}")
        else:
            out.append(os.path.basename(tok) if tok.startswith("/") else tok)
    return " ".join(out)


def _extent_token(v: Any) -> str:
    """Render one layout extent as ``"<int>"`` when static, ``"sym"`` when symbolic.

    This is the whole point of the operand signature: a STATIC extent is baked into the IR and a
    SYMBOLIC one is not, so the two produce different programs and must key differently -- but the
    symbolic extent's VALUE must never enter the key, or one artifact per shape is minted and the
    measured `N_token` cubin identity is destroyed.
    """
    try:
        return str(int(v))
    except Exception:
        return "sym"




#: Raised when a peer rank could not produce its MLIR hash. Its own name, so a caller can tell
#: "the distributed key could not be formed" from "this rank's trace failed" -- they need different
#: responses and only the second has a local stack to look at.
class PeerTraceError(RuntimeError):
    """One or more ranks failed to trace, so no all-rank key exists."""


def _rank_mlir_hash(op_factory, compile_args, options: str) -> str:
    """This rank's hash of the emitted MLIR. PRIVATE -- the input to :func:`all_rank_program_key`.

    Purpose
        Half of the identity the artifact cache keys on. **Not a program key on its own**, and it is
        underscore-private for that reason: an A2A kernel bakes in a peer layout, so this value
        identifies what THIS rank compiles and says nothing about whether its peers compiled the
        matching thing. Measured at world 2: `GemmA2ASm90` gives `ee4cb228` on rank 0 and `d9846317`
        on rank 1 for one workflow. The first caller to reach for this directly reintroduces exactly
        the defect :func:`all_rank_program_key` exists to remove.

    Semantics
        Builds a THROWAWAY functor from ``op_factory`` and traces it. The factory is the whole API
        and not a convenience: tracing consumes a functor -- ``_bind_call_params() called twice`` --
        so keying off an already-constructed instance and then compiling it is impossible. With a
        fresh instance the sequence works, measured end to end.

        Costs ~22% of a full compile (0.202 s vs 0.924 s, cutlass 4.7.0, `GemmDefaultSm90`);
        ``get_bitcode()`` is 0.003 s, so the artifact is materialised rather than lazy.

    Input requirements
        op_factory: a ZERO-ARGUMENT callable returning a freshly constructed, UNTRACED functor.
            Passing a constructed instance is the error this signature exists to prevent. **A
            factory whose construction has side effects (allocation, nvshmem state) must not be used
            here** -- it is called an extra time per compile.
        compile_args: the functor's positional ``__call__`` args, exactly as they would be passed to
            ``cute.compile``.
        options: the compile options string, forwarded unchanged.

    Returns:
        A 64-char hex sha256 of the precompiled MLIR bitcode. Deterministic on cutlass 4.7.0
        (measured: identical across two fresh traces); 4.4.2's bytecode writer was NOT, which is why
        this path requires the newer DSL.

    Raises:
        AttributeError: if ``cute.compile.to_precompiled_mlir`` is absent -- i.e. the DSL is too old.
            Left to propagate rather than translated: the caller's front door is where a capability
            refusal belongs.
    """
    import cutlass.cute as cute

    art = cute.compile.to_precompiled_mlir(op_factory(), *compile_args, options=options)
    return hashlib.sha256(art.get_bitcode()).hexdigest()


def all_rank_program_key(op_factory, compile_args=(), *, options: str = "", group=None) -> str:
    """The identity of the WHOLE DISTRIBUTED PROGRAM: every rank's MLIR hash, combined.

    Purpose
        An A2A kernel on one rank is only correct in combination with the specific kernels its PEERS
        compiled -- they exchange data through a baked peer layout. A per-rank key identifies this
        rank's program and cannot see whether the peers' match, so two different world configurations
        can leave one rank's program identical while the others differ, and reusing its artifact is
        then wrong in a way that key could never detect.

        Combining the per-rank hashes fixes that and buys three things the composed key needed
        special cases for:

        * **the result is RANK-FREE and COMPLETE.** Every rank derives the same value from the same
          gathered bytes, so `agree_on_program_key`'s invariant is restored -- and now TRUE, rather
          than an artifact of the key being blind to the peer layout;
        * **agreement is STRUCTURAL.** A rank whose program differs changes the key on EVERY rank, so
          divergence is caught globally instead of each rank privately concluding it agrees;
        * **world size is captured for free** -- the same rank-uniform kernel at world 2 and world 8
          gives different keys, because the gathered tuple is a different length.

    Semantics
        Gathers ``(ok, hash)`` from every rank in ONE ``all_gather_object`` and hashes the
        PE-ORDERED concatenation.

        **Ordered, never a set or a sum**: two different rank->program assignments must not collide,
        and only order distinguishes them. ``all_gather_object`` preserves rank order.

        **The failure is reduced, not raised locally.** This runs a COLLECTIVE on the compile path,
        so a rank that raises and leaves alone blocks every peer until a watchdog kills the job --
        the same hazard, on the same kind of path, that
        `trimul_autotuned._refuse_unsatisfiable_symmetric_request` had to fix. So a local trace
        failure is CAUGHT, carried into the gather as ``ok=False``, and turned into a
        :class:`PeerTraceError` on every rank together.

    Input requirements
        op_factory: zero-argument, returning a fresh UNTRACED functor -- see :func:`_rank_mlir_hash`.
            **Must be rank-uniform in the sense that every rank calls this**; it is collective.
        compile_args: this rank's positional ``__call__`` args. These MAY differ per rank -- that is
            the point.
        options: compile options, forwarded unchanged.
        group: the process group to gather over, or None for the default. Must be the group nvshmem
            was initialised over, since that is the set of ranks whose programs interlock.

    Returns:
        A 64-char hex sha256, IDENTICAL on every rank of the group. With no process group up, this
        rank's own hash, hashed once more so single- and multi-rank keys are the same shape.

    Raises:
        PeerTraceError: if ANY rank failed to trace, on EVERY rank, naming how many failed. The
            local exception is chained on the rank that produced one, so the stack survives.
    """
    ok, own, err = True, "", None
    try:
        own = _rank_mlir_hash(op_factory, compile_args, options)
    except Exception as exc:  # noqa: BLE001 -- re-raised below, symmetrically
        ok, err = False, exc

    try:
        import torch.distributed as dist
    except ImportError:
        dist = None

    if dist is None or not dist.is_available() or not dist.is_initialized():
        if not ok:
            raise PeerTraceError("this rank could not trace, and no group exists to tell") from err
        return hashlib.sha256(own.encode()).hexdigest()

    world = dist.get_world_size(group=group)
    gathered: list = [None] * world
    dist.all_gather_object(gathered, (ok, own), group=group)
    bad = [i for i, g in enumerate(gathered) if not (g and g[0])]
    if bad:
        raise PeerTraceError(
            f"{len(bad)} of {world} ranks could not produce an MLIR hash (ranks {bad[:8]}), so no "
            f"all-rank program key exists. Raised on EVERY rank: the gather is collective, and a "
            f"rank that failed alone would leave its peers blocked in it."
        ) from err
    return hashlib.sha256("".join(g[1] for g in gathered).encode()).hexdigest()


def mlir_keying_available() -> bool:
    """Can this DSL produce an MLIR key at all?

    A RUNTIME capability check, not a version comparison: `to_precompiled_mlir` is absent on cutlass
    4.4.2 and present on 4.7.0, and asserting a version would break the moment a build carries the
    entry point under a number nobody predicted. Checked at CALL time so a test can monkeypatch it.
    """
    import cutlass.cute as cute

    return hasattr(cute.compile, "to_precompiled_mlir")


#: Key components whose value is a property of THIS RANK rather than of the program's shape.
#:
#: The program key must be BYTE-IDENTICAL on every rank -- that is what lets `agree_on_program_key`
#: be a one-value comparison, and what makes `artifact_key`'s ``pe`` the ONLY component allowed to
#: differ. A functor's rank ordinal is folded into the kernel like any other constant, so it turns up
#: in `compile_key()` and would otherwise make each rank's program key unique, breaking the
#: agreement check on every job.
#:
#: Dropping it here does NOT lose the distinction: `artifact_key` adds ``(world_size, pe)``, so a
#: rank still reads and writes its own file. This list is what says "this component is covered
#: THERE", and `test_the_flat_program_key_agrees_across_ranks` is what fails if a NEW rank-carrying
#: component appears without being added.
RANK_SCOPED_KEY_NAMES = ("_a2a_my_cp_rank", "my_cp_rank", "_a2a_pe_table", "pe_table")

#: Matches the heap address a cute Tensor prints in its ``repr``; stripped before keying. See
#: :func:`arg_key`.
_ADDR_RE = None  # bound lazily in `arg_key`, so importing this module needs no `re`


def arg_key(value: Any, depth: int = 0) -> Any:
    """A hashable, allocation-independent description of one compile argument.

    Purpose
        A compile is decided by its functor CONFIGURATION and its OPERANDS. `compile_key()` covers
        the first; this covers the second, so a key built from both is complete without needing the
        emitted MLIR -- which is what makes cross-process reuse possible on a DSL that cannot
        produce an MLIR hash (4.4.2 has no ``to_precompiled_mlir``).

    Functionality & semantics
        * cute ``Tensor`` -> ``("T", element_type, assumed_align, address-stripped repr)``. The repr
          is the only place the DYNAMIC-layout marks appear: measured on 4.4.2, ``shape``,
          ``stride`` and ``_is_dynamic`` are all unchanged by ``mark_layout_dynamic()`` and
          ``.layout`` raises ``NotImplementedError``. ``element_type`` and ``assumed_align`` are
          added because the repr carries neither.
        * dataclass -> type name plus every field, recursively (the epilogue / scheduler packs).
        * tuple / list -> recursive, tagged so the two container kinds differ.
        * scalars, enums, cutlass numeric types, None -> themselves.
        * anything else -> its TYPE NAME only.

    Args:
        value: one compile argument.
        depth: recursion depth; past 6 the value is reduced to its type name so a self-referential
            structure cannot make key construction unbounded.

    Returns:
        A hashable, picklable value carrying no device pointer.

    Note:
        The unknown-object fallback FAILS TOWARD A MISS for the recognised kinds, but toward a
        COLLISION for an unrecognised one that carries compile-relevant content. Every argument this
        repo passes is a recognised kind; a new one must be added above.
    """
    global _ADDR_RE
    import dataclasses as _dc
    import enum as _enum

    if depth > 6:
        return ("depth", type(value).__name__)
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    if isinstance(value, _enum.Enum):
        return ("E", type(value).__name__, value.name)
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, tuple(arg_key(v, depth + 1) for v in value))
    from cutlass.cutlass_dsl import NumericMeta

    if isinstance(value, NumericMeta):
        return ("N", value.__name__)
    # A numeric INSTANCE is not the numeric TYPE above, and the difference is load-bearing:
    # `EpilogueArguments.eps` arrives as `Float32(1e-5)` and is folded into the kernel. Without this
    # branch it fell through to the unknown-object fallback, where MEASURED `Float32(1e-5)` and
    # `Float32(1e-6)` returned the SAME key -- two engines differing only in their LayerNorm epsilon
    # would have shared a compile. `.value` is the Python scalar the DSL wrapped; an instance that
    # carries none falls back to the type, which is correct, because a value the DSL keeps dynamic
    # is a kernel ARGUMENT and one artifact serves every value of it.
    if isinstance(type(value), NumericMeta):
        return ("n", type(value).__name__, getattr(value, "value", None))
    # A callable keys on its IDENTITY. Every plain function is a `builtins.function`, so the
    # fallback below would give every activation ONE key -- `act_fn=gate_fn_map["glu"]` reaches the
    # front store's epilogue and IS folded in. `gate_fn_map` has a single entry today, so there is
    # no live collision; that is why this is worth closing now rather than when the second entry
    # lands and the symptom is a wrong kernel instead of an error. Classes are excluded: they are
    # callable, and the numeric-type branch above already names them.
    if callable(value) and not isinstance(value, type):
        return ("F", getattr(value, "__module__", "?"),
                getattr(value, "__qualname__", None) or repr(value))
    from cutlass import cute

    if isinstance(value, cute.runtime.Tensor):
        if _ADDR_RE is None:
            import re

            _ADDR_RE = re.compile(r"0x[0-9a-fA-F]+")
        return ("T", str(value.element_type), int(value._assumed_align),
                _ADDR_RE.sub("", repr(value)))
    if _dc.is_dataclass(value) and not isinstance(value, type):
        return (type(value).__name__,
                tuple((f.name, arg_key(getattr(value, f.name, None), depth + 1))
                      for f in _dc.fields(value)))
    return ("?", type(value).__module__ + "." + type(value).__name__)


def flat_key_components(op: Any, compile_args: Any, options: str) -> list:
    """The rank-free description a flat program key is hashed from, as an inspectable list.

    Purpose
        Returned as DATA rather than folded straight into a hash so a test can diff two ranks'
        components and NAME the one that differs. A hash that disagrees tells you only that
        something did; the whole cost of a rank-dependent program key is the time spent finding out
        which component made it so.

    Functionality & semantics
        ``[functor module, functor qualname, sorted (name, arg_key(value)) of compile_key() minus
        RANK_SCOPED_KEY_NAMES, arg_key of every compile arg, normalized options]``.

    Args:
        op: the constructed, CONFIGURED functor. Must expose ``compile_key()``.
        compile_args: the full positional ``__call__`` args, including the trailing stream.
        options: the assembled compile options string; normalised here, so a caller need not.

    Returns:
        A JSON-serialisable list.

    Raises:
        AttributeError: if ``op`` has no ``compile_key()``. Deliberate -- keying such a functor on
            its type alone would give every configuration of it one artifact.
    """
    cfg = sorted(
        (k, arg_key(v))
        for k, v in op.compile_key().items()
        if k not in RANK_SCOPED_KEY_NAMES
    )
    return [
        type(op).__module__,
        type(op).__qualname__,
        cfg,
        [arg_key(a) for a in compile_args],
        normalize_options(options),
    ]


def program_key_from_args(op: Any, compile_args: Any, options: str) -> str:
    """A complete, rank-free program key built from the functor's arguments, not its MLIR.

    Purpose
        The MLIR key needs ``cute.compile.to_precompiled_mlir``, which is 4.7.0-and-later and ABSENT
        on the pinned 4.4.2 -- so on the DSL this repo actually runs, `resolve_program_key` raised
        and the artifact cache was simply unavailable. This is the key that makes ``persist=``
        reachable there.

    Functionality & semantics
        ``sha256`` of `flat_key_components`. Complete in the same sense the MLIR key is complete,
        by a different route: the MLIR hash sees every ``const_expr`` gate because it hashes the
        emitted code; this sees them because ``compile_key()`` now READS them (see
        ``COMPILE_GATED_ATTRS``), which is what closed the 37-gate gap.

        It is also strictly CHEAPER: the MLIR key costs a full extra trace of a throwaway functor,
        measured at 22% of a compile. This costs a dict walk.

        Where the two differ is in what they cannot see. The MLIR key cannot miss anything, by
        construction. This one misses anything that changes emitted code WITHOUT changing the
        functor's attributes or the operand layouts -- an environment variable read during tracing,
        or a module-level flag. `_compute_source_fingerprint` covers source edits; codegen-affecting
        env vars (``CUTE_DSL_LINEINFO``, ``CUTE_DSL_ARCH``) are covered by NEITHER, and that hole is
        recorded in CLAUDE.md as narrow and pre-existing.

    Args:
        op: the constructed, CONFIGURED functor.
        compile_args: its full positional ``__call__`` args.
        options: the assembled compile options string.

    Returns:
        A 64-char hex digest, identical on every rank of the job.
    """
    return hashlib.sha256(
        json.dumps(flat_key_components(op, compile_args, options),
                   separators=(",", ":"), default=str).encode()
    ).hexdigest()


def resolve_program_key(
    op_factory,
    *,
    spec: PersistSpec,
    prefix: str,
    backend: str,
    op: Any,
    compile_args: Any,
    options: str,
    world_size: Optional[int] = None,
    extra: tuple = (),
    group=None,
) -> str:
    """The ONE place that decides which key a compile is stored under.

    Purpose
        Both compile seams -- `compile_persisted` and the nvshmem path -- need the same decision, and
        two copies of it is two places for the migration to end up half-done.

    Semantics
        TWO key sources, both COMPLETE, chosen by what the DSL can do:

        * ``op_factory`` AND ``to_precompiled_mlir`` (4.7.0+) -> :func:`all_rank_program_key`, the
          hash of the emitted MLIR combined across ranks. Complete by construction: it hashes the
          code, so no gate can hide from it.
        * otherwise -> :func:`program_key_from_args`, the hash of the functor's ``compile_key()``
          plus the operand layouts. Complete because ``compile_key()`` now reads the
          post-construction ``const_expr`` gates (``COMPILE_GATED_ATTRS``), which is the gap that
          made this route unusable before.

        **This branch used to be a raise, and that is why the artifact cache did nothing here.**
        The composed key was DELETED at M5 -- rightly: it could not see 37 gates and had a
        demonstrated collision at `w8plan/mlirkey/m6_control.py` (one composed key ``51ab5a2f``,
        two MLIR hashes). But the pinned DSL is 4.4.2, which has no ``to_precompiled_mlir``, so the
        result was that every persisted compile raised ``NotImplementedError`` and every
        artifact-cache test skipped. Fixing ``compile_key()`` rather than deleting it removes the
        reason for the raise.

        The MLIR route is still PREFERRED where available: it cannot miss anything, whereas the flat
        key misses anything that changes emitted code without changing the functor's attributes or
        the operand layouts (a codegen-affecting env var is the known, narrow hole).

    Input requirements
        op_factory: zero-arg, returning a fresh UNTRACED functor. OPTIONAL now -- without it, or on
            a DSL with no MLIR keying, the flat key is used.
        op: the already-constructed, CONFIGURED functor. Load-bearing on the flat path: its
            attributes ARE the key.
        group: the process group to combine over, or None for the default.

    Returns:
        A 64-char hex key, identical on every rank of the group. Rank-carrying components are
        excluded by ``RANK_SCOPED_KEY_NAMES`` on the flat path and are restored by
        :func:`artifact_key`, which adds ``(world_size, pe)``.

    Raises:
        AttributeError: on the flat path, if ``op`` exposes no ``compile_key()``.
        PeerTraceError: from :func:`all_rank_program_key` if any rank failed to trace.
    """
    if op_factory is not None and mlir_keying_available():
        base = all_rank_program_key(op_factory, compile_args, options=options, group=group)
    else:
        # THE FLAT KEY. This branch used to be two `NotImplementedError`s, which is why the artifact
        # cache was unavailable on the DSL this repo actually pins: `to_precompiled_mlir` is 4.7.0+
        # and 4.4.2 has none, so EVERY persisted compile raised and every artifact-cache test
        # skipped. The composed key those raises replaced was deleted for a real reason -- it could
        # not see 37 `const_expr` gates and had a demonstrated collision -- but that reason is gone:
        # `compile_key()` now READS those gates (see `COMPILE_GATED_ATTRS`), so the flat key sees
        # what the MLIR hash sees, without the trace.
        #
        # `op`, not `op_factory`: the flat key needs the CONFIGURED functor's attributes, and the
        # factory's product is identical by construction. It also costs no trace, so the guard that
        # refuses a second trace of one instance is never in play.
        base = program_key_from_args(op, compile_args, options)
    # The non-IR terms still key: the backend decides which serializer can read the image back, the
    # prefix was MEASURED to change the bytes, and the caller's `extra` carries the nvshmem version.
    # The MLIR hash covers the PROGRAM; these cover how it was stored.
    #
    # SCHEMA AND ARCH ARE HERE BECAUSE A LATER CHANGE MADE THEM LOAD-BEARING. Both were in the composed key and
    # neither is in the IR, so while the fallback existed they were covered on the path most
    # compiles took. `schema` is separately checked on READ, so it was belt-and-braces; `arch` is
    # WRITTEN to the meta and never checked, so with the fallback gone it would have been in
    # neither the key nor the validation -- and a cache directory shared between two GPU
    # architectures would then hand one arch the other's image. `world_size` is already implicit in
    # the all-rank key (it is the NUMBER of gathered hashes) and is checked on read; it is listed
    # for symmetry with the composed key it replaces, at no cost.
    tail = json.dumps(
        [
            base, ARTIFACT_SCHEMA_VERSION, cuda_arch(), prefix, backend,
            world_size, normalize_options(options), list(extra), spec.key,
        ],
        separators=(",", ":"), default=str,
    )
    return hashlib.sha256(tail.encode()).hexdigest()




def artifact_key(prog_key: str, rank: Optional[RankScope] = None) -> str:
    """The stored artifact's identity: the program, plus which rank it was minted for.

    Args:
        prog_key: from :func:`program_key`. Must be byte-identical on every rank of a job.
        rank: the ``RankScope`` for an nvshmem artifact; ``None`` when the kernel has no in-kernel
            nvshmem and rank is meaningless.

    Returns:
        A 64-char hex sha256 over ``(prog_key, pe)``.

    Note:
        ``pe`` is the ONLY component that may differ across ranks, and it enters HERE rather than in
        the program key so that "do the ranks agree?" is a question the program key can answer. Note
        that ``world_size`` is deliberately NOT added here -- it is already inside ``prog_key``,
        because a different world is a different program, not a different copy of one.
    """
    return hashlib.sha256(
        canonical_key_bytes((prog_key, rank.pe if rank is not None else None))
    ).hexdigest()


def cuda_arch() -> str:
    """The compute capability the artifact is valid for, as ``"sm_90"``.

    Returns:
        ``"sm_<major><minor>"`` for the current device, or ``"unknown"`` when no CUDA device is
        reachable. Never raises: this is called from :func:`artifact_key`, which must work on a
        GPU-free host so the cache's invariants stay testable without an allocation.
    """
    try:
        import torch

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            return f"sm_{major}{minor}"
    except Exception:
        pass
    return "unknown"


def ir_sha(compiled: Any) -> Optional[str]:
    """Hash the traced IR the DSL actually compiled -- the WITNESS that audits the composed key.

    Args:
        compiled: the ``cute.compile`` result. Must be the FRESH object from a miss; a cache hit has
            no ``ir_module`` and there is nothing to witness (nor any need -- the hit is by
            definition serving what a previous miss recorded).

    Returns:
        A 64-char hex sha256 over the module's bytecode plus the compile options, or ``None`` when
        the object exposes no ``ir_module`` (which is not an error -- the witness is optional by
        construction and its absence must never fail a compile).

    Note:
        **Why this is not simply the DSL's own key.** ``get_module_hash`` (``base_dsl/dsl.py:1047``)
        hashes the traced MLIR plus every env var plus the compile options, and is COMPLETE by
        construction. It is computed unconditionally at ``dsl.py:1715`` -- even under
        ``no_cache=True``, which ``cute.compile`` always sets (``compiler.py:1089``) -- and then
        DISCARDED: it reaches only the gated ``jit_cache.set``, and nothing stores it
        (``hasattr(compiled, "module_hash")`` is False, measured). So it cannot be asked for.

        What CAN be asked for is ``compiled.ir_module`` (``jit_executor.py:798``, measured present),
        from which the same serialisation is reproducible for free on the miss path -- the module is
        already built and paid for.

        **Named ``ir_sha`` and NOT ``module_hash`` on purpose.** ``get_module_hash`` runs on the
        module BEFORE ``build_module(...)``; ``compiled.ir_module`` is what was kept AFTER it. The
        two may differ, and that is fine -- a witness only has to be a deterministic function of the
        program -- but calling it the DSL's hash would assert an equality nobody has verified.

        Env vars are deliberately EXCLUDED where the DSL includes them: an env var that does not
        change the IR would make the witness differ between two runs of the same program and produce
        false collisions in the audit, which is the one failure a witness must not have.

        **THE TEXTUAL FORM IS HASHED, NOT THE BYTECODE, AND THAT IS MEASURED RATHER THAN STYLISTIC.**
        ``write_bytecode`` is NOT canonical here: two compiles of the identical program produce
        byte-strings of the SAME LENGTH differing by a permutation of small integers
        (``\x05\x03\x01`` vs ``\x05\x01\x03``) -- the signature of an unordered container being
        serialised in iteration order. Measured on `GemmDefaultSm90`: bytecode UNSTABLE across two
        compiles, textual form STABLE across two and DIFFERENT for a different program.

        A non-deterministic witness cannot falsify anything -- it reports a collision for every
        program compiled twice -- so this choice is what makes the audit meaningful at all.

        Worth knowing, though it is not ours to fix: the DSL's own ``get_module_hash`` hashes that
        same bytecode, so its cache key inherits the instability.
    """
    mod = getattr(compiled, "ir_module", None)
    if mod is None:
        return None
    try:
        h = hashlib.sha256(str(mod).encode())
        opts = getattr(getattr(compiled, "compile_options", None), "to_str", None)
        if callable(opts):
            h.update(str(opts()).encode())
        return h.hexdigest()
    except Exception:
        return None


def artifact_paths(keysha: str, spec: PersistSpec, rank: Optional[RankScope] = None):
    """The ``(.o, .meta.json)`` pair for a key.

    Args:
        keysha: from :func:`artifact_key`.
        spec: supplies the directory.
        rank: when given, its ``.pe<PE>.of<WORLD>`` goes in the FILENAME as well as the hash.

    Returns:
        ``(Path, Path)`` -- the artifact and its sidecar. Neither is created.

    Note:
        The rank infix is redundant with the hash BY CONSTRUCTION, and that is the point. It makes a
        mis-keyed file visible in a directory listing, so a human or a script can spot a rank mismatch
        without parsing a hash. A cache whose entries are all indistinguishable 64-hex names is one
        where this class of bug stays invisible until it faults.
    """
    d = spec.resolve_dir()
    stem = keysha + (rank.infix if rank is not None else "")
    return d / f"{stem}.o", d / f"{stem}.meta.json"


def quarantine(path: Path, reason: str) -> Optional[Path]:
    """Move a bad artifact aside and record why, so no later process re-reads it.

    Args:
        path: the offending ``.o`` (its sidecar, if present, moves too).
        reason: a one-line human explanation, appended to ``<name>.rejected.why``.

    Returns:
        The ``.rejected`` path, or ``None`` if the file had already vanished (a concurrent process
        quarantined it first, which is benign).

    Note:
        Renamed, never deleted. An entry that is merely SKIPPED is re-read and re-rejected by every
        subsequent process forever, paying the read each time; and this repo forbids ``rm``, so the
        alternative was never available. On a name collision the reason is appended rather than the
        file overwritten -- two different corruptions of one key are two facts, not one.

        RENAMING IS ALSO WHAT MAKES THIS SAFE AGAINST A LIVE MODULE. ``load_module`` maps the file,
        so a module loaded from this path is still reading it. A rename leaves that mapping pointing
        at the old inode and is therefore harmless; overwriting the file IN PLACE changes bytes out
        from under a loaded CUDA library and SEGFAULTS. Measured, by a test that did exactly that.
        ``write_artifact`` publishes with ``os.replace`` for the same reason -- never a write into an
        existing path. Any future maintenance here must keep both operations rename-based.
    """
    if not path.exists():
        return None
    dest = path.with_suffix(path.suffix + ".rejected")
    try:
        with FileLock(path.with_suffix(path.suffix + ".lock"), exclusive=True, timeout=5):
            if not path.exists():
                return None
            if dest.exists():
                dest = _next_free(dest)
            os.replace(str(path), str(dest))
            meta = path.with_name(path.stem + ".meta.json")
            if meta.exists():
                os.replace(str(meta), str(dest) + ".meta.json")
            # os.open with an explicit mode, not `open(..., "a")`: the builtin creates at
            # 0666 & ~umask, so the reason file -- which names a key and a corruption -- would be
            # world-readable on a default umask. The mode is ignored when the file already exists,
            # which is what keeps the append-on-collision behaviour above intact.
            why_fd = os.open(str(dest) + ".why", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            with os.fdopen(why_fd, "a") as fh:
                fh.write(reason.rstrip() + "\n")
    except Exception:
        return None
    return dest


def _next_free(dest: Path) -> Path:
    """First unused ``<dest>.<n>``, so a second corruption never overwrites the first."""
    n = 2
    while dest.with_name(f"{dest.name}.{n}").exists():
        n += 1
    return dest.with_name(f"{dest.name}.{n}")


def write_artifact(
    blob: bytes,
    keysha: str,
    spec: PersistSpec,
    *,
    prefix: str,
    backend: str,
    rank: Optional[RankScope] = None,
    extra_meta: Optional[dict] = None,
) -> Optional[Path]:
    """Publish an artifact and its sidecar atomically.

    Args:
        blob: the serialized artifact bytes.
        keysha: from :func:`artifact_key`.
        spec: supplies the directory; a spec that is not ``writable`` makes this a no-op.
        prefix: recorded in the meta so a loader can verify what it is about to ``getattr``.
        backend: recorded in the meta; a loader with a different backend REJECTS rather than loads.
        rank: recorded in the meta AND checked on load, before ``library_init``.
        extra_meta: additional recorded fields (versions, sizes). Never checked unless a loader
            asks for them, so this is the place for diagnostics that should not gate a hit.

    Returns:
        The published ``.o`` path, or ``None`` if the spec is read-only/off or the write failed.
        A failed write is NOT an error: the caller already has a correct compiled kernel, and a cache
        that can fail the run is worse than no cache.

    Note:
        Written to ``<name>.<pid>.tmp`` then ``os.replace``d, so a reader sees the file absent or
        complete and never half. Two processes racing on the same key write IDENTICAL bytes
        (``dump_to_object`` is deterministic -- measured), so the race is benign and needs no lock;
        the sidecar is published BEFORE the artifact so a visible ``.o`` always has its meta.
    """
    if not spec.writable:
        return None
    # A root this process must not touch is a SKIPPED write, not a failed one: `write_failures`
    # means "tried and could not", and a test asserting that counter would otherwise be satisfied
    # by a deliberate refusal, which is a different fact.
    if not spec.resolve_root().usable:
        return None
    o_path, m_path = artifact_paths(keysha, spec, rank)
    meta = {
        "schema": ARTIFACT_SCHEMA_VERSION,
        "backend": backend,
        "prefix": prefix,
        "keysha": keysha,
        "pe": rank.pe if rank is not None else None,
        "world_size": rank.world_size if rank is not None else None,
        "arch": cuda_arch(),
        "bytes": len(blob),
        "sha256": hashlib.sha256(blob).hexdigest(),
    }
    if extra_meta:
        meta.update(extra_meta)
    try:
        # 0600 on the TEMPORARY, before the rename that publishes it. Tightening after `os.replace`
        # would leave the live entry world-readable for that window, and the atomic-publish
        # property this function depends on is exactly what makes the pre-rename chmod free.
        tmp_m = m_path.with_suffix(f".{os.getpid()}.tmp")
        tmp_m.write_text(json.dumps(meta, sort_keys=True, indent=1))
        os.chmod(tmp_m, 0o600)
        os.replace(str(tmp_m), str(m_path))
        tmp_o = o_path.with_suffix(f".{os.getpid()}.tmp")
        tmp_o.write_bytes(blob)
        os.chmod(tmp_o, 0o600)
        os.replace(str(tmp_o), str(o_path))
    except Exception:
        STATS.write_failures += 1
        return None
    STATS.writes += 1
    return o_path


def read_artifact(
    keysha: str,
    spec: PersistSpec,
    *,
    prefix: str,
    backend: str,
    rank: Optional[RankScope] = None,
) -> Optional[LoadedArtifact]:
    """Read and FULLY VERIFY an artifact, or return ``None`` for "compile it".

    Every check below runs before any DSL call and before ``library_init``. Ordered cheapest-first,
    and each failure quarantines rather than skips.

    Args:
        keysha: from :func:`artifact_key`.
        spec: supplies the directory; a disabled spec returns ``None`` immediately.
        prefix: rejected if the meta disagrees -- the prefix is baked into the image bytes.
        backend: rejected if the meta disagrees. This is the check corner E exists for: without it,
            a tvm-ffi-written image handed to ``load_module`` raises a symbol-materialisation error
            from inside an execution engine, in a process that did not create the file.
        rank: rejected if ``pe`` or ``world_size`` disagrees. **This is the check whose absence is a
            fault rather than an error** -- a wrong-rank artifact registered with ``library_init``
            gives ``CUDA_ERROR_ILLEGAL_ADDRESS`` at ``nvshmem init.cu:2183`` and poisons the context.

    Returns:
        A :class:`LoadedArtifact` whose bytes are hash-verified, or ``None`` on any miss, mismatch or
        corruption. ``None`` always means "compile"; it never means "something is wrong" in a way the
        caller must handle.
    """
    if not spec.enabled:
        return None
    root = spec.resolve_root()
    if not root.usable:
        STATS.misses += 1
        return None
    o_path, m_path = artifact_paths(keysha, spec, rank)
    # Three conditions replace the two `exists()` calls, and each rules out a different way the
    # bytes about to be EXECUTED could not be ours: `is_within` refuses a path that escaped the
    # validated root (a keysha or rank infix carrying `..`), and `ensure_private_file` refuses a
    # symlink or a file owned by another user -- while holding both at 0600. A refusal is an
    # ordinary miss, never a rejection: quarantining here would RENAME a file this process has
    # just decided it does not trust the provenance of.
    if not (is_within(root.path, o_path) and is_within(root.path, m_path)):
        STATS.misses += 1
        return None
    if not (ensure_private_file(o_path) and ensure_private_file(m_path)):
        STATS.misses += 1
        return None
    try:
        meta = json.loads(m_path.read_text())
    except Exception as exc:
        _reject(o_path, f"unparseable meta: {exc}")
        return None

    for fieldname, want in (
        ("schema", ARTIFACT_SCHEMA_VERSION),
        ("backend", backend),
        ("prefix", prefix),
        ("pe", rank.pe if rank is not None else None),
        ("world_size", rank.world_size if rank is not None else None),
    ):
        if meta.get(fieldname) != want:
            _reject(o_path, f"{fieldname} mismatch: artifact={meta.get(fieldname)!r} want={want!r}")
            return None

    try:
        blob = o_path.read_bytes()
    except Exception as exc:
        _reject(o_path, f"unreadable: {exc}")
        return None
    if len(blob) != meta.get("bytes") or hashlib.sha256(blob).hexdigest() != meta.get("sha256"):
        _reject(o_path, f"content mismatch: {len(blob)} B vs meta {meta.get('bytes')} B")
        return None
    STATS.hits += 1
    return LoadedArtifact(path=o_path, meta=meta, blob=blob, backend=backend)


def _reject(o_path: Path, reason: str) -> None:
    """Quarantine plus bookkeeping -- the one place a rejection is recorded.

    Args:
        o_path: the artifact to move aside.
        reason: the one-line explanation, recorded on disk AND in :data:`STATS` so a test can assert
            WHY an entry was refused rather than only that it was.
    """
    quarantine(o_path, reason)
    STATS.rejects += 1
    STATS.reject_reasons.append(reason)


@dataclass
class ArtifactStats:
    """Per-process hit/miss/reject counters, for a test to assert on and a human to read.

    A cache that silently never hits passes every positive assertion a suite makes about it, so the
    counters are part of the mechanism rather than instrumentation bolted on: the distributed tests
    assert on ``hits``/``misses`` directly, which is the only way to tell a working cache from an
    inert one.
    """

    hits: int = 0
    misses: int = 0
    rejects: int = 0
    writes: int = 0
    write_failures: int = 0
    reject_reasons: list = field(default_factory=list)

    def reset(self) -> None:
        """Zero every counter. Used by a test that measures one compile in isolation."""
        self.hits = self.misses = self.rejects = self.writes = self.write_failures = 0
        self.reject_reasons.clear()


#: Process-wide counters. Module-level rather than per-store because the interesting question is
#: always "did THIS process compile or load", across every kernel it touched.
STATS = ArtifactStats()


# ---------------------------------------------------------------------------
# The two backends.
# ---------------------------------------------------------------------------
#
# A backend is two functions: serialize a compiled object to bytes, and turn bytes back into
# something callable. They are kept in this module rather than beside their callers because the
# INVARIANT they must jointly satisfy -- that a write and a read agree about the ABI -- is not visible
# from either call site, and corner E is precisely a case where each half looked fine alone.


def serialize(compiled: Any, backend: str, prefix: str) -> bytes:
    """Turn a compiled object into artifact bytes using the backend that can read them back.

    Args:
        compiled: the return of ``cute.compile(...)``.
        backend: from :func:`backend_for`. Passing the other one does not raise here, which is the
            whole problem -- see the Note.
        prefix: the symbol prefix. Part of the image, hence part of the key.

    Returns:
        The artifact bytes.

    Raises:
        ValueError: on an unknown backend tag.
        Exception: whatever the DSL raises. Callers treat a serialize failure as "no cache entry",
            never as a run failure -- they already hold a correct compiled kernel.

    Note:
        ``dump_to_object`` on a TVM-FFI object SUCCEEDS and returns a plausible image that no loader
        can read (corner E). Nothing here can detect that, which is why :func:`backend_for` derives
        the tag from the object instead of trusting a caller, and why the tag is stored and checked
        on read.
    """
    if backend == BACKEND_DUMP_OBJECT:
        return bytes(compiled.dump_to_object(prefix))
    if backend == BACKEND_EXPORT_C:
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "artifact.o")
            compiled.export_to_c(out, prefix, enable_pic=True)
            with open(out, "rb") as fh:
                return fh.read()
    raise ValueError(f"unknown artifact backend {backend!r}")


def deserialize(art: LoadedArtifact, prefix: str):
    """Turn verified artifact bytes back into a callable.

    Args:
        art: a :class:`LoadedArtifact` from :func:`read_artifact` -- already hash-checked and
            tag-checked. Do NOT call this on unverified bytes; the checks are what make the load
            safe, and for the nvshmem backend the unchecked failure is a fault rather than an
            exception.
        prefix: the symbol prefix, which must equal ``art.meta["prefix"]`` (already enforced).

    Returns:
        For :data:`BACKEND_DUMP_OBJECT`, a ``(module, compiled_fn)`` pair: the module MUST be kept
        alive by the caller, because dropping it unloads the CUDA library that nvshmem may still have
        registered and the next launch faults rather than raises. For :data:`BACKEND_EXPORT_C`, the
        ``tvm_ffi`` module (its own lifetime rules are ``cache_utils``'s and unchanged).

    Raises:
        ValueError: on an unknown backend tag.
        Exception: whatever the loader raises; the caller quarantines and recompiles.

    Note:
        The bytes are written to a temp file because both loaders take a PATH. The file is left in
        place for the module's lifetime -- ``load_module`` may map it -- and lands in the artifact
        directory rather than ``/tmp`` so a full ``/tmp`` cannot make a hit fail where a miss would
        have worked.
    """
    import cutlass.cute as cute

    if art.backend == BACKEND_DUMP_OBJECT:
        module = cute.runtime.load_module(str(art.path), enable_tvm_ffi=False)
        return module, getattr(module, prefix)
    if art.backend == BACKEND_EXPORT_C:
        # `cute.runtime.load_module(..., enable_tvm_ffi=True)`, NOT `tvm_ffi.load_module`. The two
        # are not interchangeable and the difference is measured: tvm_ffi's own loader dispatches on
        # the file EXTENSION and has no handler for `.o`, so it raises
        #   "Loader for `.o` files is not registered, resolved to (ffi.Module.load_from_file.o)"
        # -- a message that reads like a missing runtime or a wrong architecture and is neither.
        # `cache_utils` has always used the cute loader here; the prose that says otherwise is the
        # module docstring's, and following the prose instead of the code cost one debug cycle.
        return cute.runtime.load_module(str(art.path), enable_tvm_ffi=True)
    raise ValueError(f"unknown artifact backend {art.backend!r}")


# ---------------------------------------------------------------------------
# The generic entry point: `persist=` for a kernel that needs no nvshmem registration.
# ---------------------------------------------------------------------------
#
# `compile_nvshmem` is this plus the nvshmem half (rank scoping, `library_init`, the module holder).
# Kept separate rather than parameterised because the nvshmem path's failure mode is a FAULT and its
# checks must be ordered against a registration that this function does not perform -- a single
# function with an `if nvshmem:` in it would make that ordering invisible at both call sites.


def backend_for_options(options: str) -> str:
    """Which backend a compile with these options will produce an artifact for.

    Args:
        options: the string handed to ``cute.compile(..., options=...)``.

    Returns:
        :data:`BACKEND_EXPORT_C` when ``--enable-tvm-ffi`` is present, else
        :data:`BACKEND_DUMP_OBJECT`.

    Note:
        This is a SECOND source of truth about tvm-ffi, and that is a real cost accepted for a
        specific reason: the key must be computable BEFORE the compile, and :func:`backend_for` needs
        the compiled object that only exists after. The loop is closed on the miss path, where
        :func:`compile_persisted` asserts this prediction against :func:`backend_for` and refuses to
        WRITE on a disagreement -- so a wrong prediction costs a missed cache entry, never a
        mis-tagged artifact that a later process cannot read.
    """
    return BACKEND_EXPORT_C if "--enable-tvm-ffi" in (options or "") else BACKEND_DUMP_OBJECT


def compile_persisted(
    op: Any,
    *compile_args: Any,
    options: str = "",
    persist: Optional[PersistSpec] = None,
    prefix: str = "func",
    extra: tuple = (),
    op_factory=None,
):
    """``cute.compile`` with cross-process artifact reuse, for a kernel needing no nvshmem registration.

    Args:
        op: the constructed op to compile.
        *compile_args: its full positional ``__call__`` args.
        op_factory: OPTIONAL zero-arg callable returning a fresh untraced functor. Supply it to key
            on the emitted MLIR (complete); omit it to key on the composed key (incomplete, with a
            demonstrated collision -- see :func:`resolve_program_key`). Omitting is the migration
            default and M5 removes it.
        options: passed through to ``cute.compile``; ALSO decides the backend (see
            :func:`backend_for_options`).
        persist: turn the cache on. ``None`` makes this an exact, byte-identical passthrough to
            ``cute.compile`` -- there is no code path where omitting it changes what is compiled.
        prefix: the artifact's symbol prefix. Baked into the image, hence part of the key.
        extra: extra key components a caller knows about and the fingerprint does not.

    Returns:
        A callable with the compiled kernel's calling convention. On a MISS this is the
        ``cute.compile`` result itself. On a HIT under :data:`BACKEND_EXPORT_C` it is a
        POSITIONAL-ONLY closure (``export_to_c`` does not carry the compile-time kwargs wrapper, so
        omitted trailing params are padded from a recorded arity); under
        :data:`BACKEND_DUMP_OBJECT` it is the reloaded compiled function, which does carry its
        args-spec.

    Raises:
        Exception: only what ``cute.compile`` itself raises. Every cache failure -- unreadable,
            mis-tagged, corrupt, unwritable -- falls through to a compile.

    Note:
        Under :data:`BACKEND_DUMP_OBJECT` the caller MUST keep the returned object alive for as long
        as it may launch; the loaded module owns a CUDA library and dropping it unloads that library.
        Callers on that backend should generally use ``compile_nvshmem``, which owns that lifetime
        explicitly in a dataclass field rather than leaving it to a local.
    """
    import cutlass.cute as cute

    if persist is None or not persist.enabled:
        return cute.compile(op, *compile_args, options=options)

    backend = backend_for_options(options)
    prog = resolve_program_key(
        op_factory, spec=persist, prefix=prefix, backend=backend, op=op,
        compile_args=compile_args, options=options, extra=extra,
    )
    keysha = artifact_key(prog)  # no rank: this entry point registers nothing with nvshmem
    art = read_artifact(keysha, persist, prefix=prefix, backend=backend)
    if art is not None:
        try:
            loaded = deserialize(art, prefix)
            if backend == BACKEND_EXPORT_C:
                from fold_cp_ops._internal.cache_utils import EXPORT_FUNC_NAME, _restore_call_abi

                arity = art.meta.get("arity")
                if arity:
                    return _restore_call_abi(loaded[EXPORT_FUNC_NAME], int(arity))
                # No recorded arity means we cannot vouch for the reloaded object's calling
                # convention. That is a MISS, not a best guess: the failure it would otherwise
                # produce (`TypeError: Expects N parameters`) appears only on a warm cache in a
                # second process, which is the hardest possible place to diagnose it.
                _reject(art.path, "no recorded arity: cannot vouch for the reloaded calling convention")
            else:
                return loaded[1]
        except Exception as exc:
            _reject(art.path, f"load failed after verification: {type(exc).__name__}: {exc}")

    compiled = cute.compile(op, *compile_args, options=options)

    actual = backend_for(compiled)
    if actual != backend:
        # The options-based prediction was wrong. Writing now would mint an artifact tagged for a
        # backend that cannot read it -- corner E's failure, manufactured deliberately. Skip the
        # write and say so; the caller still has a correct kernel.
        STATS.write_failures += 1
        STATS.reject_reasons.append(
            f"backend prediction {backend!r} from options != actual {actual!r}; refusing to write"
        )
        return compiled

    if persist.writable:
        try:
            blob = serialize(compiled, backend, prefix)
            # The witness, recorded where it is free: the module is already traced and built. It is
            # never load-bearing for THIS process -- it exists so a later offline audit can prove
            # two different programs never shared one program key.
            meta = {"options": options, "program_key": prog, "ir_sha": ir_sha(compiled)}
            if backend == BACKEND_EXPORT_C:
                meta["arity"] = _measure_arity(blob, persist)
            if backend != BACKEND_EXPORT_C or meta.get("arity"):
                write_artifact(
                    blob, keysha, persist, prefix=prefix, backend=backend, extra_meta=meta
                )
            else:
                # An export we cannot characterise is one a later process could not call safely.
                # Not writing it costs one cold compile per process; writing it costs a TypeError
                # that only ever appears against a warm cache.
                STATS.write_failures += 1
        except Exception:
            STATS.write_failures += 1
    return compiled


def _measure_arity(blob: bytes, spec: PersistSpec) -> Optional[int]:
    """Positional arity an ``export_c`` artifact demands, or None if it cannot be established.

    Args:
        blob: the exported bytes.
        spec: supplies a directory for the temporary file the loader needs (the loader takes a path).

    Returns:
        The arity, or ``None``. ``None`` suppresses the write entirely -- see the caller.

    Note:
        ``export_to_c`` drops the compile-time kwargs wrapper, so a reloaded artifact demands the
        FULL declared arity while the fresh object fills omitted trailing params from their defaults.
        The arity cannot be read off the export side (``c_header_arguments`` bails on the EnvStream
        param and ``get_kwargs_wrapper_spec()`` counts it), so it is MEASURED by making a
        deliberately impossible call: the ``num_args`` check is the first op in ``__tvm_ffi_func``,
        ahead of any parameter decode, so the probe is side-effect free. This mirrors
        ``cache_utils._measure_exported_arity`` -- the same mechanism, recorded in the meta here
        instead of a separate ``.abi`` sidecar, so it is covered by the same sha check as everything
        else.
    """
    from fold_cp_ops._internal.cache_utils import _measure_exported_arity

    root = spec.resolve_root()
    if not root.usable:
        # The probe WRITES a real object file and then loads it. Doing that in a directory this
        # process has refused is the one thing the refusal exists to prevent, so the arity goes
        # unmeasured instead -- which the caller already handles, since `None` is its documented
        # "could not determine" answer.
        return None
    tmp = root.path / f".arity_probe.{os.getpid()}.o"
    try:
        tmp.write_bytes(blob)
        os.chmod(tmp, 0o600)
        return _measure_exported_arity(tmp)
    except Exception:
        return None
    finally:
        try:
            if tmp.exists():
                os.replace(str(tmp), str(tmp) + ".used")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# The consolidated compile surface.
# ---------------------------------------------------------------------------


@dataclass
class CompiledKernel:
    """One return type from both compile paths, so a caller writes the same code either way.

    **The holder is LOAD-BEARING on the nvshmem path and COSMETIC on the tvm-ffi path**, and that
    asymmetry is the whole content of the decision to unify. With in-kernel nvshmem, ``library_init``
    stores the RAW CUDA library handle in nvshmem's own table, so the compiled object must outlive
    every launch or the next one faults rather than raises. With tvm-ffi there is no such hazard --
    ``.to()`` returns ``self`` and nothing holds a raw handle -- so the wrapper buys only uniformity.
    That cost was accepted deliberately: returning DIFFERENT types depending on ``link_nvshmem``
    forces every caller to branch on the flag to learn what it got, which is worse than the two
    separate functions this replaces.

    Attributes:
        executor: what a call forwards to.
        from_cache: True when a disk artifact was loaded rather than a compile run. The single-device
            path had no way to ask this before, and a harness or test needs it.
        options / device_id: what it was compiled with, and where.
        nvshmem_kernel_obj: the registration, or None on the tvm-ffi path -- where ``free()`` is
            then an honest no-op rather than an error.
        program_key / artifact_key / artifact_path: what it was looked up under, so a test reads the
            implementation's own key instead of re-deriving it and drifting.
        _compiled / _module: retained ONLY to keep the CUDA library loaded. Never read.
    """

    executor: Any
    from_cache: bool = False
    options: str = ""
    device_id: int = 0
    nvshmem_kernel_obj: Any = None
    program_key: Optional[str] = None
    artifact_key: Optional[str] = None
    artifact_path: Any = None
    _compiled: Any = None
    _module: Any = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.executor(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Forward unknown attributes to the wrapped executor.

        MANDATORY, not a convenience. ``cache_utils.jit_cache`` calls ``compiled_fn.export_to_c(...)``
        (``cache_utils.py:311``) on exactly what the single-device compile returned, and it catches
        ``Exception`` and merely prints -- so without forwarding, ``@jit_cache`` would stop writing
        artifacts SILENTLY. ``_restore_call_abi``'s docstring warns about this same class ("an
        attribute access would work on a cache MISS and break on a HIT"); here the call site exists.

        Raises:
            AttributeError: when the executor has no such attribute either, so a genuine typo still
                surfaces as one instead of returning None.
        """
        if name.startswith("_"):  # never forward dunder/private lookups into the executor
            raise AttributeError(name)
        return getattr(object.__getattribute__(self, "executor"), name)

    def free(self) -> None:
        """Release the nvshmem registration, if there is one.

        A no-op on the tvm-ffi path, honestly: there is nothing registered to finalise. Must be
        called at a point EVERY RANK reaches -- ``library_finalize`` is collective, so calling it
        from a garbage-collection path deadlocks the job (measured).
        """
        obj = self.nvshmem_kernel_obj
        if obj is None:
            return
        try:
            import nvshmem.core

            nvshmem.core.library_finalize(obj)
        except Exception:
            pass
        self.nvshmem_kernel_obj = None

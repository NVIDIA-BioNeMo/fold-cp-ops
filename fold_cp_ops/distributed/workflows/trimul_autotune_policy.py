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

"""Opt-in distributed autotune of TriMulAutotuned's OWN back/front stores (default-off, freeze-cached).

Faithful + layering-clean: autotunes the ACTUAL ``GemmA2AStore`` (``GemmA2ASm90`` design-E,
M=N=K=N_token, L=Dloc*B) and ``DualGatedGemmDistStore`` (``DualGatedGemmDistSm90``) via the
kernel-agnostic facility (``fold_cp_ops.distributed.distributed_autotune`` — fold_cp_ops->fold_cp_ops, no benchmark import).
Each candidate config builds a TEMPORARY proxy store with SYNTHETIC inputs (perf is value-independent),
times its ``.run()`` with the collective-safe + consensus do_bench, and the proxies are freed after the
pick. The winning config tuple is returned to ``TriMulAutotuned.__init__`` and freeze-cached per shape.

IB-AWARE (docs/trimul_ib_autotune_plan.md). The grid is keyed on the VENUE (``has_ib_peers`` — the SAME
topology probe the real store uses, ``gemm_sm90_a2a.build_p2p_table``) x the N-regime (coalesce crossover
``N*``):

* all-P2P venue (single NVLink domain): the front IB machinery const-elides, so tuning the ``hybrid_ib``
  proxy is byte-identical to the pre-IB NVLink grid — front sweeps ``tile_n`` only; back sweeps the old
  4-config ``cluster_n`` grid. (The store is still built ``hybrid_ib=has_ib_peers`` = False -> NVLink.)
* has-IB venue (>=1 cross-node peer): the proxy IS the real IB drain. Front sweeps ``tile_n`` and, in the
  ENGAGED regime (N>=N*, where the wide-put coalesce is live), the wide-put batch ``W`` (2-STAGE:
  tile_n @ W128, then W{64,128,256} @ winning tile_n — fewer proxy builds live at once, hence fewer
  registrations to retire in order; see ``_dispose_store``).
  Back sweeps ``cluster_n{1,2,4}`` (the cluster_multislot decoupled-drain concentration).

Every returned pick is ORACLE-GATED per shape: ``correctness()`` (a fold_cp_ops-local config-differential
oracle — see the adapters) asserts each swept perf-knob config reproduces the venue-baseline config's
recv, and the driver runs it on the pick before freeze-caching. The ABSOLUTE per-shape anchor is the
existing e2e ``global_oracle`` gate run with autotune ON (the frozen config flows through the full kernel).

Enabled only when ``CPO_DIST_AUTOTUNE=1`` (or ``autotune_config=True``); the default path is byte-identical
(``_resolve_back_config`` / ``_resolve_front_config`` unchanged; W defaults 128 == the store default).
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import os
import sys
import textwrap
import time as _perf

import torch

from fold_cp_ops._internal.autotune import AutotuneConfig
from fold_cp_ops.distributed.distributed_autotune import build_distributed_autotuner

# freeze caches: (cp, N, D[, cp1]) -> resolved config tuple. Populated once per shape per process (in-mem)
# AND persisted to a DISK freeze (see _freeze_*), so a FORM-A pre-freeze (front process, then back process,
# each a separate torchrun) writes the pick and production READS it — front+back never sweep in ONE
# process, so one sweep's proxy registrations are never still live while the other is compiling
# (see _dispose_store for why a live-but-orphaned registration is a double delete, not just clutter).
_BACK_CACHE: dict = {}
_FRONT_CACHE: dict = {}

# ---------------------------------------------------------------------------------------------- #
# DYNAMIC-mode anchor.
# ---------------------------------------------------------------------------------------------- #
# A ``dynamic=True`` TriMulAutotuned is compile-once-any-N: ONE compiled kernel and ONE perf config serve
# EVERY runtime N (the instance cache key literally carries ``None`` in its N slot —
# fused_trimul_cp.py:161). The config therefore MUST NOT depend on which N happened to build the
# instance. Resolving at the build N (what the ctor did) made the pick CALL-ORDER dependent, missed the
# disk pre-freeze (a FORM-A driver writes it at this anchor), and re-ran a fresh multi-GPU sweep per N.
# The raw-tensor perf suite papered over it by hand-building at its own `_ANCHOR_N`; the DTensor entry
# (`fused_trimul_dtensor`, which takes N from `x.shape`) had no such hand and ran an UNCLUSTERED back
# GEMM at every non-anchor N — 28-56% slower on a public API.
#
# This is NOT a new invariant — it is the one the repo ALREADY declares everywhere else. The baked §49
# table `fused_trimul._SM90_IB_CONFIG` is keyed `(D, cp0, cp1)` with NO N, and says so outright: "ONE
# config per (D, cp0, cp1) at the build ANCHOR (N=2048) serves ALL token-N via the dynamic-compile path
# — the tile/cluster are token-N-independent (the front tile_N tiles the FEATURE axis, the back tile_N
# the token-j axis; neither scales with N)". `_resolve_back_config` likewise returns a CONSTANT
# (128,128,F,(1,2,1)) at every N. The autotune freeze key was the only resolver still carrying N, which
# is why turning autotune ON at an off-grid N produced a config WORSE than leaving it off.
#
# 2048 is the PIN BASIS: every committed trimul perf pin (raw AND DTensor) was harvested with the
# dynamic config resolved here, and it is the same anchor the baked table names. Changing this value
# REBASES those pins — re-harvest, never bump casually.
DYNAMIC_ANCHOR_N = 2048


def anchor_n(N, cp_axis_sizes, dynamic: bool) -> int:
    """The N a build resolves its perf config at: ``DYNAMIC_ANCHOR_N`` when dynamic, else ``N``.

    A STATIC build is one shape / one config, so it keeps its own N. A DYNAMIC build snaps, so every N
    in the process resolves the SAME freeze key — the property that makes the DTensor and raw paths
    agree by construction rather than by two lookups happening to coincide. Falls back to the build N
    when the anchor is not shardable by this mesh (the proxy stores are BUILT at it, so it must divide
    every cp axis); that fallback is why the ``cn>=4``-only i-filter in ``_BackProxyAdapter.configs``
    matters — an unsnapped odd N must still be offered a full grid.

    NOTE this anchors ONLY the perf-config lookup. The instance itself is still built at the caller's
    N, so symmetric-recv sizing and memory footprint are untouched: anchoring the whole build would
    force a small-N caller to pay ``DYNAMIC_ANCHOR_N``-sized symmetric allocations.
    """
    N = int(N)
    if not dynamic:
        return N
    a = int(DYNAMIC_ANCHOR_N)
    return a if all(a % int(s) == 0 for s in cp_axis_sizes) else N


# ---------------------------------------------------------------------------------------------------
# FREEZE ENTRY SCHEMA
# ---------------------------------------------------------------------------------------------------
# v1 was `{kind, key, cfg}` — it recorded neither WHICH GRID produced the pick nor HOW DECISIVE the pick
# was, and those two absences are one root cause with two faces (docs/migration_failing_tests.md §14.13,
# §14.15):
#
#   * no grid  -> a change to `configs()` CANNOT invalidate its own cache. Measured blast radius: 91 of
#                 111 donor back keys (82%) hold `cluster_n=1`, minted by the pre-fix i-filter that
#                 pruned the grid to one degenerate candidate at every `N_i_loc % 128 != 0`. The grid fix
#                 changes what a NEW sweep produces and cannot reach an entry already written; a STATIC
#                 build (`anchor_n` returns N when `dynamic=False`) reads the fossil today.
#   * no margin -> a 0.2% coin-flip is indistinguishable from a 30% win. The anchor tie-break that cost
#                 gate D was exactly such a coin-flip, and the margin WAS computed during the sweep and
#                 thrown away.
#
# v2 adds `schema`, `grid_fp`, `margin`, `ts`. The load path is FAIL-CLOSED: an entry whose `grid_fp` is
# absent (v1) or does not match the running grid is STALE — `_freeze_load` returns None and the key
# re-sweeps. A v1 entry is never silently trusted. Nothing is deleted: the donor's v1 entries stay on
# disk as the blast-radius record and are simply ignored.
FREEZE_SCHEMA_VERSION = 2

_GRID_FP_CACHE: dict = {}
_STALE_NOTED: set = set()
#: (kind, key) pairs whose freeze HIT/MISS has already been announced this process. A MISS was
#: SILENT before this existed, which cost a whole multi-cluster experiment: a run that is slow
#: because the autotuner is re-sweeping looks exactly like a run that is slow because the compiler
#: is blowing up, and neither tree logged which one it was. Announcing it makes that a grep.
_FREEZE_NOTED: set = set()


def _norm_ast(src: str) -> str:
    """Formatting/comment/docstring-insensitive normal form of a source fragment.

    `ast.dump(..., include_attributes=False)` already drops comments, line numbers and whitespace, so a
    re-format or a comment edit does NOT invalidate the freeze; docstrings are stripped explicitly for
    the same reason. Anything that changes what the grid EMITS changes the dump.

    The ROOT package of a `from X.Y import Z` is also dropped, so the fingerprint measures the grid and
    not the package it lives in. `_FrontProxyAdapter._valid_tile_ns` imports `_resolve_front_tile_n`
    inside its body, so without this the migrated tree (`fold_cp_ops.…`) would compute a different front
    fingerprint from the donor (`fold_cp_ops.…`) purely because of the rename — a gratuitous difference in an
    observable side effect (the bytes of a written freeze file) for the migration's identity gate to
    trip over. Only the FIRST dotted component is dropped: importing a different MODULE still changes
    the fingerprint, which the tests assert in both directions.
    """
    tree = ast.parse(textwrap.dedent(src))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and "." in node.module:
            node.module = node.module.split(".", 1)[1]
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", None)
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:] or [ast.Pass()]
    return ast.dump(tree, annotate_fields=True, include_attributes=False)


def grid_fingerprint(kind: str) -> str:
    """Fingerprint of the CANDIDATE GRID a `kind` sweep would offer — the field v1 entries lack.

    It hashes the normalized AST of every function that decides which configs are emitted, plus the
    grid-shaping module constants. It deliberately does NOT depend on the shape or the venue: the
    fingerprint answers "was this entry minted by the current `configs()`?", not "what did that grid
    emit here". A shape-dependent fingerprint would need an adapter instance — i.e. the multi-GB proxy
    allocation this function exists to stay off the hot read path.

    `inspect.getsource` failures are NOT caught. A silent fallback marker would make every entry match
    by construction, which is a gate that cannot go red.
    """
    fp = _GRID_FP_CACHE.get(kind)
    if fp is not None:
        return fp
    parts = [f"schema={FREEZE_SCHEMA_VERSION}", f"kind={kind}"]
    if kind == "back":
        srcs = [_BackProxyAdapter._grid, _BackProxyAdapter.configs]
    elif kind == "front":
        from fold_cp_ops.distributed.workflows.trimul_autotuned import _resolve_front_tile_n

        srcs = [
            _FrontProxyAdapter.configs,
            _FrontProxyAdapter._valid_tile_ns,
            _FrontProxyAdapter.engaged,
            _FrontProxyAdapter._n_star,
            _resolve_front_tile_n,
        ]
        parts.append("N_STAR=" + repr(sorted(_FRONT_N_STAR.items())))
        parts.append("W_GRID=" + repr(tuple(_FrontProxyAdapter._W_GRID)))
    else:
        raise ValueError(f"unknown freeze kind {kind!r} (expected 'front' or 'back')")
    for fn in srcs:
        parts.append(_norm_ast(inspect.getsource(fn)))
    fp = hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]
    _GRID_FP_CACHE[kind] = fp
    return fp


def freeze_entry_status(rec, kind: str, expect_fp: str | None = None) -> str:
    """Classify one loaded freeze record. THE product-side classifier — the audit imports this rather
    than re-deriving the rule, so the audit cannot drift from what production actually trusts.

    Returns one of:
      ``live``          — carries a `grid_fp` matching the running grid; production uses it.
      ``stale-grid``    — carries a `grid_fp` from a DIFFERENT grid; ignored, key re-sweeps.
      ``stale-nogrid``  — v1 (`{kind,key,cfg}`, no fingerprint at all); ignored, key re-sweeps.
      ``corrupt``       — not a dict, or no usable `cfg`.
    """
    if not isinstance(rec, dict) or not isinstance(rec.get("cfg"), (list, tuple)):
        return "corrupt"
    got = rec.get("grid_fp")
    if not got:
        return "stale-nogrid"
    want = grid_fingerprint(kind) if expect_fp is None else expect_fp
    return "live" if got == want else "stale-grid"


def _freeze_root():
    """The freeze directory and whether this process may use it.

    Purpose
        The freeze holds the CONSENSUS autotune pick, which every rank then compiles against. An
        entry somebody else can write therefore chooses this job's kernel configuration -- not a
        wrong answer, but a silently mis-tuned one, with nothing pointing at the cause.

    Functionality & semantics
        ``CPO_DIST_AUTOTUNE_FREEZE_DIR`` when set, else ``dist_autotune_freeze`` under the autotune
        cache root. Both go through `cache_security.validate_cache_root`, and when the DEFAULT
        parent is itself unusable that verdict propagates instead of creating a child inside a
        refused directory.

        **This replaces a call that could not have worked.** It read
        ``from fold_cp_ops._internal.autotune import default_cache_dir``, and no such name exists in
        that package -- so the default branch raised ``ImportError`` every time it was taken. It
        survived because the env var is set in the launchers that exercise the freeze, which is
        exactly how a function-local import rots unnoticed.

    Returns:
        A `cache_security.CacheRoot`. ``usable`` False makes a load a MISS and a store a no-op, so
        the sweep simply re-runs.
    """
    from fold_cp_ops._internal.autotune.cache import cache_root_status
    from fold_cp_ops._internal.cache_security import CacheRoot, validate_cache_root

    explicit = os.environ.get("CPO_DIST_AUTOTUNE_FREEZE_DIR", "").strip()
    if explicit:
        return validate_cache_root(explicit)
    parent = cache_root_status()
    if not parent.usable:
        return CacheRoot(
            path=parent.path / "dist_autotune_freeze",
            usable=False,
            reason=f"its parent {parent.path} was refused: {parent.reason}",
        )
    return validate_cache_root(parent.path / "dist_autotune_freeze")


def _freeze_dir() -> str:
    """The freeze directory as a path. Callers that touch the filesystem use :func:`_freeze_root`."""
    return str(_freeze_root().path)


def _freeze_path(kind: str, key) -> str:
    return os.path.join(_freeze_dir(), f"{kind}_" + "_".join(str(k) for k in key) + ".json")


def _note_freeze(kind: str, key, outcome: str) -> None:
    """Announce a freeze-cache HIT or MISS on stderr, once per ``(kind, key)`` per process.

    Purpose
        Make "the autotuner re-swept" visible. A MISS costs a full multi-GPU sweep per key, which is
        the dominant term in a cold run, and before this it was announced nowhere -- so a slow run
        could not be attributed without an experiment. A stale entry already had its own, louder
        line; an absent one had nothing, which is the case that actually happens.

    Functionality & semantics
        Deduplicated through the module-level :data:`_FREEZE_NOTED` set, so a key hammered in a loop
        prints once rather than once per call. Writes to ``stderr`` with ``flush=True``, matching the
        stale-entry announcement, so the two interleave correctly in a captured log. Emits nothing
        else and returns nothing -- this is a side effect, never a control-flow signal.

    Input requirements
        ``kind``     the freeze namespace (``"front"`` / ``"back"``); any str, used only in the text.
        ``key``      the freeze key; must be iterable of str-ables, as :func:`_freeze_path` requires.
                     A non-iterable raises ``TypeError`` from the path builder, not from here.
        ``outcome``  ``"HIT"`` or ``"MISS"``. Not validated -- an unexpected value prints verbatim
                     rather than raising, because a logging helper must never break the caller.

    Returns
        ``None``. Raises nothing of its own.
    """
    tag = (kind, tuple(key))
    if tag in _FREEZE_NOTED:
        return
    _FREEZE_NOTED.add(tag)
    print(f"[freeze] {outcome} {_freeze_path(kind, key)}", file=sys.stderr, flush=True)


def _freeze_load(kind: str, key):
    """Read a disk-frozen pick (a FORM-A driver process wrote it; production reads it so it NEVER re-runs
    the multi-GPU sweep in the hot path). Returns the config tuple, or ``None`` when the entry is missing,
    corrupt, or STALE.

    FAIL-CLOSED on the fingerprint: an entry with no ``grid_fp`` (schema v1) or a ``grid_fp`` from another
    grid is stale and is NOT returned — the caller re-sweeps and mints a v2 entry. This is the whole point
    of the field: a fix to ``configs()`` must invalidate the entries it fixes, and a pre-fingerprint entry
    whose provenance is unknown must be treated as unknown rather than trusted.
    """
    from fold_cp_ops._internal.cache_security import ensure_private_file

    root = _freeze_root()
    path = _freeze_path(kind, key)
    # A refused root and an unreadable entry are BOTH a plain MISS: the caller re-sweeps, which is
    # what it does for an absent entry too. `ensure_private_file` additionally refuses a symlink or
    # a file another user owns or could write -- this entry decides the kernel configuration every
    # rank then compiles, so reading somebody else's is a silently mis-tuned job.
    if not root.usable or not ensure_private_file(path):
        _note_freeze(kind, key, "MISS")
        return None
    try:
        with open(path) as f:
            rec = json.load(f)
    except Exception:
        _note_freeze(kind, key, "MISS")
        return None
    status = freeze_entry_status(rec, kind)
    if status != "live":
        # A stale entry that is silently ignored is indistinguishable from an absent one, which is the
        # failure mode this whole schema exists to close. Announce it once per key per process.
        tag = (kind, tuple(key))
        if tag not in _STALE_NOTED:
            _STALE_NOTED.add(tag)
            print(
                f"[freeze] IGNORED {status} entry {_freeze_path(kind, key)} "
                f"(grid_fp={rec.get('grid_fp') if isinstance(rec, dict) else None!r} "
                f"want={grid_fingerprint(kind)!r}) -- this key will RE-SWEEP",
                file=sys.stderr,
                flush=True,
            )
        return None
    _note_freeze(kind, key, "HIT")
    # JSON turns tuples into lists; restore any nested tuple (the back cluster_shape_mnk).
    return tuple(tuple(x) if isinstance(x, list) else x for x in rec["cfg"])


def _sweep_timings(autotuner):
    """The per-config timings off a finished sweep, plus WHICH attribute they came from.

    Purpose
        Insulate the margin recorder from the autotuner's attribute name, and -- more importantly --
        make "the tuner exposed no timings attribute at all" a DISTINGUISHABLE outcome from "the
        sweep timed nothing". A bare ``getattr(x, name, None) or {}`` collapses those two, and that
        collapse silently emptied ``margin`` on every freeze entry this repo has ever written.

    Functionality & semantics
        Tries the known spellings in order and returns the first that is a non-empty mapping.
        ``last_timings`` is THIS tree's name (``_internal/autotune/tuner.py``); ``configs_timings``
        is the upstream's, kept so a future re-sync of that rename cannot re-break this silently.
        Neither present -> ``({}, None)``, which the caller records rather than smooths over.

    Args:
        autotuner: The tuner object returned by ``build_distributed_autotuner``'s ``run()``. May be
            any object, including None -- probing is by ``getattr``, so a wrong type yields the
            MISSING outcome instead of an AttributeError.

    Returns:
        ``(timings, attr_name)``. ``timings`` maps config -> measurement; ``attr_name`` is the
        attribute it came from, or None when the object exposed neither name.
    """
    for name in ("last_timings", "configs_timings"):
        t = getattr(autotuner, name, None)
        if t:
            return t, name
    return {}, None


def _margin_of(autotuner, best) -> dict | None:
    """Best vs RUNNER-UP from the sweep's own per-config timings — the field v1 discarded.

    Returns ``None`` when fewer than two configs were timed (a one-candidate grid has no runner-up; that
    is itself worth being able to see, and ``n_configs`` records it on the entry). ``rel`` is the fraction
    by which the runner-up is slower, so a coin-flip reads ~0.00x and a decisive win reads ~0.3.
    """
    timings, attr = _sweep_timings(autotuner)
    if attr is None:
        # NOT an empty sweep -- we asked the wrong object. Recorded as such, because the two are
        # indistinguishable in the field this function exists to fill: every entry written before
        # this fix carries n_configs=0 and reads exactly like a grid that timed nothing.
        return {"n_configs": 0, "best_ms": None, "runner_up": None, "timings_attr": "MISSING"}
    ms = []
    for cfg, t in timings.items():
        v = t[0] if isinstance(t, (list, tuple)) else t
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        ms.append((v, cfg))
    ms.sort(key=lambda p: p[0])
    if len(ms) < 2:
        return {
            "n_configs": len(ms),
            "best_ms": ms[0][0] if ms else None,
            "runner_up": None,
            "timings_attr": attr,
        }
    (b_ms, b_cfg), (r_ms, r_cfg) = ms[0], ms[1]
    return {
        "n_configs": len(ms),
        "best_ms": b_ms,
        "runner_up_ms": r_ms,
        "rel": ((r_ms - b_ms) / b_ms) if b_ms > 0 else None,
        "best": b_cfg.all_kwargs(),
        "runner_up": r_cfg.all_kwargs(),
        # The autotuner's min-pick should BE ms[0]; recorded so a disagreement is visible, not assumed away.
        "pick_is_best": (best is None) or (best.all_kwargs() == b_cfg.all_kwargs()),
        "timings_attr": attr,
    }


def _freeze_store(kind: str, key, cfg, dm, *, margin=None, margins=None) -> None:
    """Rank-0 persists the CONSENSUS pick to the disk freeze (barrier'd). Only rank-0 writes — the pick is
    consensus-identical, so one writer suffices and avoids the N-rank write race the facility warns about.
    Atomic (tmp + os.replace) so a concurrent reader never sees a half-written file.

    Writes schema v2: ``grid_fp`` (so a ``configs()`` change invalidates its own cache), ``margin`` (so a
    coin-flip is distinguishable from a decisive win) and ``ts``. ``margins`` carries the per-stage margins
    for the 2-stage front sweep, where ``margin`` is the DECIDING stage."""
    import torch.distributed as dist

    # RANK 0'S ENTIRE WRITE IS WRAPPED, and the barrier below is outside it. That placement is the
    # load-bearing part: any exception escaping here -- a refused root, a full disk, a permission
    # error -- would take rank 0 past the barrier its peers are already blocked in, and the job
    # hangs until a watchdog kills it, with a traceback pointing at whatever the peers were doing.
    # A freeze that cannot be written must cost a re-sweep, never a deadlock.
    if int(getattr(dm, "rank", 0)) == 0:
        try:
            _freeze_store_rank0(kind, key, cfg, margin=margin, margins=margins)
        except Exception as exc:  # noqa: BLE001 - see above; nothing here may reach the barrier
            print(f"[freeze] store SKIPPED for {kind} {tuple(key)}: {exc}", file=sys.stderr,
                  flush=True)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _freeze_store_rank0(kind: str, key, cfg, *, margin=None, margins=None) -> None:
    """The rank-0 half of :func:`_freeze_store`: validate, write privately, publish atomically.

    Split out so the caller's ``try`` can be a single statement wrapping ALL of it -- an inline
    version invites a later edit to add a line outside the guard, which is how a rank stops reaching
    a barrier.

    Args:
        kind: The freeze namespace (``"front"`` / ``"back"``).
        key: The freeze key.
        cfg: The consensus pick.
        margin: The deciding stage's margin, or None.
        margins: Per-stage margins for the 2-stage front sweep, or None.

    Returns:
        None. Writes nothing when the freeze root is unusable.

    Raises:
        OSError: From the filesystem. The caller catches everything and lets the barrier run.
    """
    import tempfile

    root = _freeze_root()
    if not root.usable:
        print(f"[freeze] store SKIPPED: {root.reason}", file=sys.stderr, flush=True)
        return
    payload = {
        "kind": kind,
        "key": list(key),
        "cfg": [list(x) if isinstance(x, tuple) else x for x in cfg],
        "schema": FREEZE_SCHEMA_VERSION,
        "grid_fp": grid_fingerprint(kind),
        "margin": margin,
        "ts": _perf.time(),
        "ts_iso": _perf.strftime("%Y-%m-%dT%H:%M:%S", _perf.gmtime()),
    }
    if margins is not None:
        payload["margins"] = list(margins)
    p = _freeze_path(kind, key)
    # `mkstemp` in the VALIDATED root, which creates 0600 -- so the file is never briefly
    # world-readable, and the mode survives the rename onto the published name. The previous
    # `open(p + ".tmp", "w")` created at 0644 under the usual umask AND used a predictable
    # name, which a second writer could collide on.
    fd, tmp = tempfile.mkstemp(dir=str(root.path), prefix=".freeze.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, p)
    except Exception:
        # Never leave the temporary behind: a directory that accumulates one per failed write
        # is a slow leak nobody notices until it is large.
        try:
            os.replace(tmp, tmp + ".failed")
        except OSError:
            pass
        raise


# MEASURED coalesce-crossover N* keyed by (cp, D): the front wide-put W-sweep ENGAGES only on a has-IB
# venue AND N>=N* (below N* the wide-put coalesce is inert -> W un-swept, tile_n only). (16,256):1792 is
# MEASURED @cp16/D256 (docs/trimul_ib_autotune_plan.md §5 / [[reference_front_wideput_raster_run_i]]).
# An UNMEASURED (cp,D) -> 0 (ENGAGE): the D128 N* sweep must exercise the W dimension to MEASURE its
# crossover; once measured the maintainer pins it here. all-P2P never engages (venue gate wins first).
_FRONT_N_STAR = {(16, 256): 1792}


def _probe_has_ib_peers(pe_map) -> bool:
    """VENUE gate — returns the SAME ``has_ib_peers`` the REAL store computes: ``not all(is_p2p)`` over the
    cp peers. REUSES ``gemm_sm90_a2a.build_p2p_table`` (the nvshmem TEAM_SHARED probe over
    ``pe_map.cp_pe_table``) so the autotune venue decision is byte-identical to the store's IB-vs-NVLink
    decision — a 1-GPU/node cp=2 job is has-IB (cross-node); a single-node cp<=8 job is all-NVLink. This is
    a CORRECTNESS keystone (tuning the wrong kernel for the venue), NOT the cp>8 heuristic (which is
    venue E-8-GPU-node-specific and wrong on a 1-GPU/node cluster)."""
    from fold_cp_ops.distributed.gemm_sm90_a2a import build_p2p_table

    pe_table = tuple(int(x) for x in pe_map.cp_pe_table.tolist())
    return not all(build_p2p_table(pe_table))


def _rel_err(out, ref) -> float:
    """fp32 relative L2 (the parent kernels' correctness metric): ``||out-ref|| / max(||ref||, eps)``."""
    o, r = out.detach().float(), ref.detach().float()
    denom = float(r.norm().item())
    return float((o - r).norm().item() / (denom if denom > 0 else 1.0))


_COMPILE_BAR_S = (
    5.0  # CLAUDE.md HARD RULE: a cold compile > 5s is a BUG (range_constexpr-nested-if blowup).
)


def _compile_check(kind, cfg_key, secs, dm) -> None:
    """HARD ≤5s cold-compile gate — a >5s compile is a BLOCKER (CLAUDE.md: a range_constexpr-nested-if
    blowup to FIX, NEVER warn-and-continue). Fires at the REAL venue during precompile (native sm_90, prod
    Dloc trip counts — a single-node forced-has_ib proxy would inflate them). The per-config time is
    CONSENSUS'd (all_reduce MAX) so the over-bar decision is bit-identical on EVERY rank -> a SYMMETRIC
    raise; a per-rank threshold on a jittery compile time would raise on a subset -> collective deadlock.
    Over-bar -> WARN-only (user override 2026-07-24 ACCEPTED the ~8.2s front-IB D128/Dloc=8 cold compile,
    not fixed); the over-bar time prints but the sweep PROCEEDS + freeze-caches. The consensus (all_reduce
    MAX) keeps the decision bit-identical across ranks."""
    import torch.distributed as dist

    secs = float(secs)
    if dist.is_available() and dist.is_initialized():
        t = torch.tensor([secs], device=getattr(dm, "device", "cuda"), dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)  # consensus = the slowest rank's compile
        secs = float(t.item())
    over = secs > _COMPILE_BAR_S
    if int(getattr(dm, "rank", 0)) == 0 and (over or os.environ.get("CPO_PRINT_AUTOTUNING") == "1"):
        tag = (
            "  *** OVER 5s COLD-COMPILE (WARN — ACCEPTED per user override 2026-07-24) ***"
            if over
            else ""
        )
        print(f"[compile-check] {kind} cfg={cfg_key} cold_compile_s={secs:.2f}{tag}", flush=True)
    # USER OVERRIDE (2026-07-24): a >5s cold compile is ACCEPTED in the autotune (was a HARD raise). The
    # front hybrid_ib D128/Dloc=8 store cold-compiles ~8.2s — a real range_constexpr compile cost the user
    # opted to ACCEPT rather than fix (see [[reference_user_override_autotune_5s_compile_bar]]). WARN-only:
    # the over-bar time is printed above (still visible so a genuine blowup shows), but the sweep PROCEEDS
    # and freeze-caches the pick. Do NOT re-raise.


def _free_recv(store):
    try:
        import nvshmem.core.interop.torch as nvshmem_torch

        nvshmem_torch.free_tensor(store.recv)
    except Exception:
        pass


def _dispose_store(store) -> None:
    """FULLY dispose a swept proxy store so its nvshmem per-module device-state registrations are RELEASED,
    not leaked (TASK #44).

    WHY A LEAKED REGISTRATION IS A DOUBLE DELETE AND NOT MERELY UNTIDY. ``library_init`` stored each proxy
    kernel's CUDA library as a RAW handle in nvshmem's table. A store dropped without ``free()`` leaves the
    CuTe-DSL's garbage collector free to ``cuLibraryUnload`` that library while nvshmem still holds the
    handle — and the next ``library_init`` or ``library_finalize`` dereferences it, SIGSEGV. Ordering the
    two deletes IS the job; ``CompiledGemmBitcode.free`` (nvshmem lets go FIRST, the DSL's reference drops
    SECOND) is where that order lives. There is no registration-COUNT ceiling to stay under.

    Each proxy store compiled its fused A2A GEMM (and per-direction kernels) via ``compile_gemm_with_bitcode``,
    whose ``library_init`` REGISTERS that module's ``nvshmemi_device_state_d`` symbol into nvshmem's host
    registration map (``registered_device_states``); every subsequent ``nvshmemx_culibrary_init`` (the NEXT
    build) re-pushes the host device state to EVERY still-registered module (``nvshmemi_update_device_state``).
    So a stale entry is not inert — it is re-touched by every later build, and once the DSL has collected its
    library that re-touch IS the second delete. The old recv-ONLY ``_free_recv`` set up exactly that: it freed
    the buffers and dropped the store while leaving every proxy kernel registered, so the FRONT sweep's
    orphans survived into the BACK sweep (``TriMulAutotuned.__init__`` runs both).

    Both proxy store classes (``DualGatedGemmDistStore`` / ``GemmA2AStore``) expose a ``free()`` that
    calls ``CompiledGemmBitcode.free()`` -> ``nvshmem.core.library_finalize`` (UNREGISTERS the module) for
    every compiled kernel AND frees the symmetric recv/staging. Calling it here resets the registration
    count between the front and back sweeps, so the in-process PEAK is ``max(|front|, |back|)`` co-resident
    builds (each proven under the wall by the FORM-A driver's per-kernel process), NOT their sum — the same
    reset a fresh process gives, achieved via proper resource release. Falls back to the recv-only
    ``_free_recv`` for a store lacking a ``free()``."""
    f = getattr(store, "free", None)
    if callable(f):
        try:
            f()
            return
        except Exception:
            pass
    _free_recv(store)


class _BackProxyAdapter:
    """Autotune the back design-E store (tile_n + cluster_n; tile_m pinned 128, cluster_m pinned 1).

    VENUE-keyed: all-P2P -> the old 4-config (tile_n, cluster_n) grid (cluster_multislot const-elided);
    has-IB -> the cluster_multislot decoupled drain with cluster_n{1,2,4} concentration (tile pinned
    128x128; cluster_n MUST == cluster_shape_mnk[1], derived by the store from ``cluster_shape_mnk``)."""

    key_names = ["cp", "N", "D"]

    def __init__(
        self, pe_map, B, N, D, dt, dm, *, device_mesh=None, placements=None, has_ib_peers=False
    ):
        from fold_cp_ops.distributed.workflows.trimul_autotuned import GemmA2AStore

        self._Store = GemmA2AStore
        self.pe_map, self.B, self.N, self.D, self.dt, self.dm = pe_map, B, N, D, dt, dm
        self.device_mesh, self.placements = device_mesh, placements
        self.has_ib_peers = bool(has_ib_peers)
        cp = pe_map.cp
        cp_axis = tuple(int(s) for s in pe_map.cp_axis_sizes)
        self.cp0 = cp_axis[0]
        self.cp1 = cp_axis[1] if len(cp_axis) > 1 else 1
        self.N_i_loc = N // self.cp0
        self.N_j_loc = N // self.cp1
        Dloc = D // cp
        dev = dm.device
        torch.manual_seed(4321 + dm.rank)
        M_full = B * N * N
        self.a_dm = torch.randn(Dloc, M_full, device=dev, dtype=dt) * 0.1
        self.b_dm = torch.randn(Dloc, M_full, device=dev, dtype=dt) * 0.1
        self.direction = "outgoing"
        self._stores: dict = {}
        self.key_tensors = (self.a_dm, self.b_dm)
        self.fixed_kwargs = {"cp": cp, "N": N, "D": D}

    def _grid(self):
        # 2-D: conservative (1,1,1). 1-D all-P2P: the 4-config cluster_M==1 grid (cluster_multislot inert).
        # NOTE the tile_m==128 i-alignment question is settled in configs(), not here.
        # 1-D has-IB: cluster_n{1,2,4} @ tile 128x128 (cluster_multislot REQUIRES tile_m==128; cluster_n ==
        # cluster_shape_mnk[1]). Prune cluster_n<=nt_j_pp (j-tiles/peer) AND <8 (NVSwitch domain). cluster_n>=4
        # only on an EVEN-shard N_j (cn>=4-STRADDLE is deferred #45; cn in {1,2} handle straddle via #76d).
        if self.cp1 > 1:
            return [(128, 128, 1)]
        if self.has_ib_peers:
            nt_j_pp = max(1, self.N_j_loc // 128)
            cns = [cn for cn in (1, 2, 4) if cn <= nt_j_pp and cn < 8]
            cns = [cn for cn in cns if cn < 4 or (self.N_j_loc % (128 * cn) == 0)]
            return [(128, 128, cn) for cn in (cns or [1])]
        return [(128, 128, 1), (128, 128, 2), (128, 256, 1), (128, 256, 2)]

    def configs(self):
        out = []
        for tm, tn, cn in self._grid():
            # i-EVEN FILTER — cluster_n>=4 ONLY. The i-straddle constraint (N_i_loc%tile_M==0, one CTA
            # M-tile per peer block) belongs to design-E, and `_build` passes NO `back_store=`, so EVERY
            # proxy here is `GemmA2AStore`'s default **pe_aligned** on BOTH venues — which handles a
            # straddling per-peer block via arbitrary_n / per-peer M-tiling. Production agrees: the
            # i-alignment assert fires only `if back_store == "design_e"` (fused_trimul.py:1281-1292) and
            # so does the (128,128,(1,1,1)) downgrade (:2186-2189).
            #
            # The filter used to read `((not self.has_ib_peers) or cn >= 4)`, i.e. it applied a design-E
            # rule to a pe_aligned proxy on the all-P2P venue. At any N with N_i_loc%128 != 0 that dropped
            # EVERY config and the sweep fell back to the lone conservative (128,128,cn=1) — a SILENTLY
            # untuned back store at every off-grid N (measured: N=7016 cp8 -> 1 config offered instead of
            # 4; the config production actually ships there is (128,128,cn=2), 1.55x faster). cluster_n>=4
            # i-straddle stays DEFERRED (#45), so the filter is kept for cn>=4 only.
            # (cn>=4 j-even-shard is enforced separately in _grid() via N_j_loc; cluster_n clusters along j.)
            if cn >= 4 and self.N_i_loc % tm != 0:
                continue
            if self.cp1 > 1 and self.N_j_loc % tn != 0:
                continue
            out.append(AutotuneConfig(tile_m=tm, tile_n=tn, cluster_n=cn))
        return out or [AutotuneConfig(tile_m=128, tile_n=128, cluster_n=1)]

    def _baseline(self):
        return (128, self.configs()[0].all_kwargs()["cluster_n"])

    def _build(self, tn, cn):
        # has-IB: hybrid_ib=True -> the cluster_multislot decoupled drain (cluster_n from cluster_shape_mnk[1]
        # = cn, satisfying the gemm_sm90_a2a.py:807 A-multicast==drain-width coupling). all-P2P: hybrid_ib=False
        # -> const-collapse to the pe_aligned NVLink TMA store (byte-identical to the pre-IB grid).
        return self._Store(
            self.pe_map,
            self.B,
            self.N,
            self.D,
            self.dt,
            tile_shape_mn=(128, tn),
            cluster_shape_mnk=(1, cn, 1),
            pingpong=False,
            is_persistent=True,
            device_mesh=self.device_mesh,
            placements=self.placements,
            hybrid_ib=self.has_ib_peers,
        )

    def _get(self, tn, cn):
        st = self._stores.get((tn, cn))
        if st is None:
            _t0 = _perf.perf_counter()  # time the COLD BUILD+COMPILE (back compiles lazily in run)
            st = self._build(tn, cn)
            st.run(self.a_dm, self.b_dm, self.direction)  # trigger lazy per-direction compile
            _compile_check("back", (tn, cn), _perf.perf_counter() - _t0, self.dm)
            self._stores[(tn, cn)] = st
        return st

    def precompile(self, dm):
        for cfg in self.configs():
            kw = cfg.all_kwargs()
            self._get(kw["tile_n"], kw["cluster_n"])

    def launch(self, a_dm, b_dm, *, cp, N, D, tile_m, tile_n, cluster_n):
        return self._stores[(tile_n, cluster_n)].run(self.a_dm, self.b_dm, self.direction)

    def _snapshot(self, tn, cn, dm):
        import torch.distributed as dist

        st = self._get(tn, cn)
        st.run(self.a_dm, self.b_dm, self.direction)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        torch.cuda.synchronize(dm.device)
        return st.recv.detach().float().clone()

    def correctness(self, config_kwargs, dm) -> float:
        """Module-local CONFIG-DIFFERENTIAL oracle: tile_n / cluster_n are PERF knobs (same math, different
        drain concentration/tiling — [[feedback_functional_variant_vs_perf_knob]]), so a config's recv MUST
        reproduce the venue-baseline config's recv. The baseline (cluster_n=1) is absolute-validated by the
        e2e ``global_oracle`` gate; this guards every swept knob against it (catches a silent-zero sub-band /
        cluster-drain ULF that corrupts the recv). Module-local — no benchmark/tests import (this module is
        fold_cp_ops->fold_cp_ops). Returns ``rel_err(recv_cfg, recv_baseline)``."""
        tn, cn = config_kwargs["tile_n"], config_kwargs["cluster_n"]
        recv_cfg = self._snapshot(tn, cn, dm)
        btn, bcn = self._baseline()
        recv_base = self._snapshot(btn, bcn, dm)
        return _rel_err(recv_cfg, recv_base)

    def free(self):
        # FULL dispose (finalize each proxy's nvshmem module registration + free recv), NOT recv-only:
        # a recv-only free drops the store while its kernels stay REGISTERED, leaving the DSL free to
        # unload libraries nvshmem still holds raw handles to -- the double delete (TASK #44).
        # See _dispose_store.
        for st in self._stores.values():
            _dispose_store(st)
        self._stores.clear()


class _FrontProxyAdapter:
    """Autotune the staged front store (tile_n dividing Dloc; tile_m pinned 128, cooperative=no pingpong).

    VENUE + REGIME-keyed. all-P2P OR non-engaged (N<N*): sweep ``tile_n`` only (W inert/un-coalesced, fixed
    128). has-IB & ENGAGED (N>=N*): 2-STAGE — stage 1 sweeps ``tile_n`` @ W128 (``_stage=1``); the driver
    pins the winner and stage 2 sweeps the wide-put batch ``W{64,128,256}`` @ that tile_n (``_stage=2``).
    The 2-stage keeps the live build count at |tile_n| + |W-2| instead of the flat tile_n x W cross-product,
    so fewer registrations are alive at once and fewer must be retired in order (``_dispose_store``).
    The store is ALWAYS built ``hybrid_ib=has_ib_peers`` (a
    cross-node job MUST use IB regardless of regime); engagement only decides whether to SWEEP W."""

    key_names = ["cp", "N", "D"]
    _W_GRID = (64, 128, 256)

    def __init__(self, pe_map, B, N, D, dt, dm, *, eps=1e-5, has_ib_peers=False):
        from fold_cp_ops.distributed.workflows.trimul_autotuned import DualGatedGemmDistStore

        self._Store = DualGatedGemmDistStore
        self.pe_map, self.B, self.N, self.D, self.dt, self.dm = pe_map, B, N, D, dt, dm
        self.eps = eps
        self.has_ib_peers = bool(has_ib_peers)
        cp = pe_map.cp
        self.Dloc = D // cp
        cp_axis = tuple(int(s) for s in pe_map.cp_axis_sizes)
        cp0 = cp_axis[0]
        cp1 = cp_axis[1] if len(cp_axis) > 1 else 1
        self.M = B * (N // cp0) * (N // cp1)  # local token count = B * N_i_loc * N_j_loc
        self.K = D
        dev = dm.device
        torch.manual_seed(4321 + dm.rank)
        # synthetic STACKED projections (2D, K) — perf is value-independent.
        self.g_in = torch.randn(2 * D, self.K, device=dev, dtype=dt) * 0.1
        self.p_in = torch.randn(2 * D, self.K, device=dev, dtype=dt) * 0.1
        self.xn = torch.randn(self.M, self.K, device=dev, dtype=dt) * 0.1
        self._stores: dict = {}
        self.key_tensors = (self.xn,)
        self.fixed_kwargs = {"cp": cp, "N": N, "D": D}
        # 2-stage state (mutated by the driver between the two build_distributed_autotuner passes).
        self._stage = 1
        self._winner_tile_n = None

    def _n_star(self):
        return _FRONT_N_STAR.get((self.pe_map.cp, self.D), 0)

    def engaged(self) -> bool:
        """has-IB venue AND N>=N* -> the wide-put coalesce is live -> SWEEP W (else W fixed 128)."""
        return self.has_ib_peers and self.N >= self._n_star()

    def _valid_tile_ns(self):
        # Offer ONLY the tile_ns PRODUCTION runs: map each mult-32 seed through the SAME resolver
        # TriMulAutotuned.__init__ uses (_resolve_front_tile_n = the mult-32 per-peer-tile clamp + best_front_tile
        # C12 residual). Prevents timing/freezing a tile production clamps away — e.g. tile_n=64 @ Dloc=32
        # (postact 32|32 valid, but Dloc%64≠0 → prod clamps to 32) — a freeze/production mismatch. Also covers
        # small-Dloc validity (Dloc=8 → best_front_tile=16). Drift-free: every offered tile is a resolver
        # fixed point (production runs it unchanged).
        from fold_cp_ops.distributed.workflows.trimul_autotuned import _resolve_front_tile_n

        return sorted(
            {_resolve_front_tile_n(self.Dloc, tn) for tn in (256, 128, 64, 32)}, reverse=True
        )

    def configs(self):
        tns = self._valid_tile_ns()
        if not self.engaged():
            # tile_n sweep @ fixed W=128 (all-P2P: W const-elided; non-engaged: coalesce inert below N*).
            return [AutotuneConfig(tile_n=tn, ib_wide_batch=128) for tn in tns]
        if self._stage == 1:
            return [
                AutotuneConfig(tile_n=tn, ib_wide_batch=128) for tn in tns
            ]  # find tile_n @ W128
        # stage 2: W sweep @ the winning tile_n.
        wtn = self._winner_tile_n if self._winner_tile_n is not None else tns[0]
        return [AutotuneConfig(tile_n=wtn, ib_wide_batch=W) for W in self._W_GRID]

    def _baseline_tile_n(self):
        return self._valid_tile_ns()[0]

    def _build(self, tn, W=128):
        # IB-aware: build the REAL IB kernel (hybrid_ib=True + ib_wide_batch=W) so the timed proxy IS the drain
        # the production TriMulAutotuned runs — the old proxy passed NEITHER, tuning the NVLink coupled path, not
        # the IB kernel. On all-P2P (has_ib_peers=False) W const-elides -> identical to the NVLink proxy.
        return self._Store(
            self.pe_map,
            self.g_in,
            self.p_in,
            self.M,
            self.K,
            self.D,
            self.dt,
            eps=self.eps,
            tile_shape_mn=(128, tn),
            pingpong=False,
            is_persistent=True,
            hybrid_ib=self.has_ib_peers,
            ib_wide_batch=W,
        )

    def _get(self, tn, W):
        st = self._stores.get((tn, W))
        if st is None:
            _t0 = (
                _perf.perf_counter()
            )  # time the COLD BUILD+COMPILE (front compiles at construction)
            st = self._build(tn, W)  # compile_gemm_with_bitcode runs HERE (in __init__), not in run
            st.run(self.xn)
            _compile_check("front", (tn, W), _perf.perf_counter() - _t0, self.dm)
            self._stores[(tn, W)] = st
        return st

    def precompile(self, dm):
        for cfg in self.configs():
            kw = cfg.all_kwargs()
            self._get(kw["tile_n"], kw.get("ib_wide_batch", 128))

    def launch(self, xn, *, cp, N, D, tile_n, ib_wide_batch=128):
        return self._stores[(tile_n, ib_wide_batch)].run(self.xn)

    def _snapshot(self, tn, W, dm):
        import torch.distributed as dist

        st = self._get(tn, W)
        st.run(self.xn)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        torch.cuda.synchronize(dm.device)
        return st.recv.detach().float().clone()

    def correctness(self, config_kwargs, dm) -> float:
        """Module-local CONFIG-DIFFERENTIAL oracle (see ``_BackProxyAdapter.correctness``): tile_n / W are
        PERF knobs, so a config's recv MUST reproduce the venue-baseline (largest tile_n @ W128) recv.
        The baseline is absolute-validated by the e2e ``global_oracle`` gate; this guards each swept knob."""
        tn, W = config_kwargs["tile_n"], config_kwargs.get("ib_wide_batch", 128)
        recv_cfg = self._snapshot(tn, W, dm)
        recv_base = self._snapshot(self._baseline_tile_n(), 128, dm)
        return _rel_err(recv_cfg, recv_base)

    def free(self):
        # FULL dispose (finalize each proxy's nvshmem module registration + free recv), NOT recv-only, so
        # the FRONT sweep's builds are UNREGISTERED before TriMulAutotuned.__init__ runs the BACK sweep --
        # otherwise the front's orphaned registrations outlive their CUDA libraries and the back sweep's
        # next library_init re-touches a freed handle (TASK #44). See _dispose_store.
        for st in self._stores.values():
            _dispose_store(st)
        self._stores.clear()


def _oracle_gate(adapter, best, dm, *, tol=2e-2):
    """Oracle-gate the freeze-pick: run ``adapter.correctness`` on the FIRST + LAST grid config + the pick
    (>=2 distinct), assert each rel_L2 < tol. Mirrors benchmark/distributed/autotune_driver.py:59-68. This
    is the per-config in-sweep guard; the ABSOLUTE per-shape anchor is the e2e ``global_oracle`` gate run
    with autotune ON. SPMD-lockstep (same configs on every rank). Returns the {config:rel} gate dict."""
    cfgs = adapter.configs()
    gate = {}
    for cfg in [cfgs[0], cfgs[-1], best]:
        kk = tuple(sorted(cfg.all_kwargs().items()))
        if kk in gate:
            continue
        rel = adapter.correctness(cfg.all_kwargs(), dm)
        gate[kk] = rel
        if int(getattr(dm, "rank", 0)) == 0:
            print(f"[oracle] cfg={cfg.all_kwargs()} rel_L2={rel:.4e}", flush=True)
        assert rel < tol, f"freeze-pick oracle FAIL: config {cfg} rel_L2={rel:.4e} (>= {tol})"
    return gate


def autotune_back_config(
    pe_map,
    B,
    N,
    D,
    dt,
    dm,
    *,
    device_mesh=None,
    placements=None,
    n_iters=30,
    warmup=8,
    oracle_gate=True,
    dynamic=False,
):
    """Return ``(tile_M, tile_N, pingpong, cluster_shape_mnk)`` for the back store, freeze-cached per shape.

    ``dynamic=True`` (one compiled instance serving every N) resolves at ``DYNAMIC_ANCHOR_N`` — key AND
    proxy shapes — so every N shares one pick and one freeze file. See ``anchor_n``."""
    N = anchor_n(N, pe_map.cp_axis_sizes, dynamic)
    cp1 = (tuple(int(s) for s in pe_map.cp_axis_sizes) + (1,))[1]
    key = (pe_map.cp, N, D, cp1)
    if key in _BACK_CACHE:
        return _BACK_CACHE[key]
    frozen = _freeze_load("back", key)
    if frozen is not None:
        _BACK_CACHE[key] = frozen
        return frozen
    has_ib_peers = _probe_has_ib_peers(pe_map)
    adapter = _BackProxyAdapter(
        pe_map,
        B,
        N,
        D,
        dt,
        dm,
        device_mesh=device_mesh,
        placements=placements,
        has_ib_peers=has_ib_peers,
    )
    try:
        best, tuner = build_distributed_autotuner(adapter, dm, n_iters=n_iters, warmup=warmup)()
        margin = _margin_of(tuner, best)
        if oracle_gate:
            _oracle_gate(adapter, best, dm)
        kw = best.all_kwargs()
        cfg = (kw["tile_m"], kw["tile_n"], False, (1, kw["cluster_n"], 1))
    finally:
        adapter.free()
    _BACK_CACHE[key] = cfg
    _freeze_store("back", key, cfg, dm, margin=margin)
    return cfg


def autotune_front_config(
    pe_map, B, N, D, dt, dm, *, eps=1e-5, n_iters=30, warmup=8, oracle_gate=True, dynamic=False
):
    """Return ``(tile_M, tile_N, pingpong, ib_wide_batch)`` for the staged front store, freeze-cached per
    shape. On an engaged has-IB venue this runs the 2-STAGE sweep (tile_n @ W128, then W @ winner).

    ``dynamic=True`` resolves at ``DYNAMIC_ANCHOR_N`` (see ``anchor_n``) — which also pins ``engaged()``,
    the one N-dependent knob on this side (tile_n follows Dloc alone), to the anchor's regime."""
    N = anchor_n(N, pe_map.cp_axis_sizes, dynamic)
    key = (pe_map.cp, N, D)
    if key in _FRONT_CACHE:
        return _FRONT_CACHE[key]
    frozen = _freeze_load("front", key)
    if frozen is not None:
        _FRONT_CACHE[key] = frozen
        return frozen
    has_ib_peers = _probe_has_ib_peers(pe_map)
    adapter = _FrontProxyAdapter(pe_map, B, N, D, dt, dm, eps=eps, has_ib_peers=has_ib_peers)
    try:
        best1, tuner1 = build_distributed_autotuner(adapter, dm, n_iters=n_iters, warmup=warmup)()
        tile_n = best1.all_kwargs()["tile_n"]
        W = best1.all_kwargs().get("ib_wide_batch", 128)
        margins = [_margin_of(tuner1, best1)]
        if oracle_gate:
            _oracle_gate(adapter, best1, dm)
        if adapter.engaged():
            # stage 2: pin the winning tile_n, sweep W. Fresh autotuner over the mutated configs(); the
            # adapter's _stores cache is shared so stage-1 builds are reused (wall-respecting).
            adapter._stage = 2
            adapter._winner_tile_n = tile_n
            best2, tuner2 = build_distributed_autotuner(
                adapter, dm, n_iters=n_iters, warmup=warmup
            )()
            W = best2.all_kwargs()["ib_wide_batch"]
            margins.append(_margin_of(tuner2, best2))
            if oracle_gate:
                _oracle_gate(adapter, best2, dm)
        cfg = (128, tile_n, False, W)
    finally:
        adapter.free()
    _FRONT_CACHE[key] = cfg
    # `margin` is the DECIDING stage (stage 2 when engaged, else the only stage); `margins` keeps both so
    # a decisive tile_n pick followed by a coin-flip W pick is not flattened into one number.
    _freeze_store("front", key, cfg, dm, margin=margins[-1], margins=margins)
    return cfg

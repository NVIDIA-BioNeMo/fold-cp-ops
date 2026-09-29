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


"""Cross-tree COLD-COMPILE comparison: a probe that measures one tree, a comparator that pairs two.

Why this is not a `BenchTarget`
    A cross-tree comparison needs two `fold_cp_ops` on two ``sys.path``s, and those cannot coexist
    in one interpreter. `driver.py` runs targets INSIDE one process across N ranks, so the thing
    simply does not fit that seam. It decomposes instead into a PROBE (one fresh interpreter,
    pointed at one tree, times one ``build``) and a COMPARATOR (alternates the probe across the two
    trees and hands the durations to `bench_utils.compare_cold_compile`).

Why the subprocess half lives HERE and not in `bench_utils`
    `compare_cold_compile` deliberately does not spawn processes: what "fresh" MEANS -- which
    interpreter, which ``sys.path``, which ``CPO_CACHE_ENABLED`` -- belongs to the caller, which is
    what lets the primitive's own test drive it with literal numbers. This module exists to BE that
    caller. Putting the launching in `bench_utils` would make the timing life-line depend on
    process spawning and on tree layout, and would force its test to spawn interpreters.

What is timed, exactly
    ``target.build(ctx)`` for ONE cell, and nothing else in the process. Not the pytest session
    (dominated by node warm-up), not the CLI invocation (dominated by interpreter start, ``import
    torch`` and dist init). The field is called ``build_s`` and NOT ``compile_s`` on purpose:
    ``build`` also allocates the operands and the symmetric recv, so an honest name beats a
    narrower-sounding one that would be wrong. ``build`` is also the only unit that exists under
    one name in BOTH trees -- anything finer would time two different functions and call the ratio
    a result.
"""

from __future__ import annotations

import dataclasses
import json
import os
import socket
import subprocess
import sys
import time
from typing import Any, Callable, Sequence

# NOTHING non-stdlib is imported at module scope, and that is a REQUIREMENT rather than tidiness.
# The probe is invoked by absolute path into THIS file while `PYTHONPATH` points at the tree being
# measured, so every module-scope import resolves in the BASELINE tree. `compare_cold_compile` and
# `MIN_COLD_COMPILE_SAMPLES` are ours-only, so importing them here made the probe unrunnable
# against main -- it failed its own `import` verification, which is how this was found. The
# comparator imports them where it uses them; the probe never needs them. Anything added here must
# exist in EVERY tree this probes, which in practice means the standard library.

#: Prefix of the single stdout line the probe emits. The parent scans for it rather than parsing
#: all of stdout, because a tree's own imports print banners we neither control nor want to match.
PROBE_PREFIX = "COLD_COMPILE_JSON "


@dataclasses.dataclass(frozen=True)
class TreeSpec:
    """One side of the comparison: a label, a tree to measure, and the interpreter to measure it.

    Attributes:
        label: The side's name, e.g. ``"main"``. Passed verbatim to `compare_cold_compile`, so it is
            what the samples are attributed to.
        tree_dir: ABSOLUTE path of the repository root to put on the child's ``PYTHONPATH``. Both
            ``benchmark.*`` and ``fold_cp_ops.*`` resolve into THIS tree; the probe verifies that it
            actually did, because two trees on one box share every module name.
        python: The interpreter to launch. Absolute path -- a bare ``python`` would resolve against
            whatever the child's ``PATH`` happens to be.
    """

    label: str
    tree_dir: str
    python: str = sys.executable


@dataclasses.dataclass(frozen=True)
class ProbeResult:
    """What one probe reports back.

    Every field exists so a bad measurement cannot look like a good one.

    Attributes:
        ok: Whether a duration was produced. False for an import failure, a `supports` skip, or a
            build that raised -- all three of which must be distinguishable from a SLOW build.
        build_s: Wall seconds around ``target.build(ctx)``, or None when `ok` is False.
        imported: Whether the target module resolved and imported AT ALL. Separate from `ok` because
            a tree that cannot be imported must surface as broken rather than as slow: a comparator
            that treats an unimportable baseline as a large `build_s` reports a fabricated ratio.
        tree: The child's ``fold_cp_ops.__file__``. The parent checks it lies under
            `TreeSpec.tree_dir` -- a probe that silently imported the OTHER tree would
            otherwise compare a tree against itself. Reported on EVERY exit path via `_tree_file`,
            including the early failures, because a check fed None cannot answer the question it
            was written for and will answer some other one instead.
        cache_enabled: The value of ``CPO_CACHE_ENABLED`` as READ BY THE CHILD, never as exported by
            the parent. An export that does not reach the child is exactly how a warm-cache
            comparison looks legitimate.
        cache_dir_entries_before: Entry count of the on-disk JIT cache directory before the build.
        cache_dir_entries_after: The same after. A count that GREW while `cache_enabled` says "0"
            proves the flag did not take effect -- an independent check with a real "did not fire"
            state, rather than a restatement of the flag.
        skipped: The target's `supports` reason when the cell is not runnable here, else None.
        digest: The compiled kernel's code-section digest from `fold_cp_ops.testing.cubin_identity`,
            in DIGEST mode; None in timing mode and on every failure. Kept beside `build_s` rather
            than replacing it because the two modes share every verification but answer different
            questions -- one asks how long the build took, the other what it emitted.
        phase: WHICH phase failed -- ``"setup"`` for ctx construction (dist init, nvshmem init,
            placements) or ``"build"`` for ``target.build`` itself. Set only on a failure exit;
            None on success and on the older replies that predate it. Both phases report
            ``ok=False, imported=True`` and are otherwise identical, so without this the parent
            reports a dist-init failure as a BUILD failure -- which cost a real diagnosis its first
            minutes, spent looking for a rendezvous inside a target that has none.
        error_class, error_msg: The child's exception, when there was one.
    """

    ok: bool
    build_s: float | None = None
    digest: dict | None = None
    imported: bool = False
    tree: str | None = None
    cache_enabled: str | None = None
    cache_dir_entries_before: int | None = None
    cache_dir_entries_after: int | None = None
    skipped: str | None = None
    phase: str | None = None
    error_class: str | None = None
    error_msg: str | None = None

    def to_json(self) -> str:
        """Render as the single line the probe prints and the parent parses."""
        return PROBE_PREFIX + json.dumps(dataclasses.asdict(self))


class ProbeFailure(RuntimeError):
    """A side could not produce a usable sample, so NO ratio may be reported.

    Raised out of the `measure` callable, which aborts `compare_cold_compile` rather than letting it
    average over a side that did not run. The message leads with ``BASELINE_BROKEN`` or
    ``CANDIDATE_BROKEN`` because the failure this guards against is a broken baseline being read as
    a slow one -- the same shape as a check with no "did not fire" state.

    Attributes:
        side: The label that failed.
        kind: Which verification failed -- one of ``import``, ``cache_flag``, ``cache_grew``,
            ``wrong_tree``, ``no_tree``, ``skipped``, ``setup``, ``build``. ``setup`` and ``build``
            are separated by `ProbeResult.phase`: both arrive as ``ok=False, imported=True``, and
            calling a dist-init failure a build failure sends the reader into the target's code.
            ``wrong_tree`` and ``no_tree`` are
            deliberately separate: the first says the child named a tree outside the one asked for,
            the second that it named none at all. Collapsing them reported a real build exception as
            "imported fold_cp_ops from None".
        detail: The child's own words.
    """

    def __init__(self, side: str, kind: str, detail: str, *, is_baseline: bool):
        self.side, self.kind, self.detail = side, kind, detail
        role = "BASELINE_BROKEN" if is_baseline else "CANDIDATE_BROKEN"
        super().__init__(
            f"{role}: side {side!r} failed verification {kind!r} -- {detail}\n"
            "NO ratio is reported. A side that cannot produce a full set of usable probes is not a "
            "slow side, and reporting a median over what did run would launder a broken tree into "
            "a performance number."
        )


def probe_env(spec: TreeSpec, base: dict | None = None) -> dict:
    """The child's environment: this tree on the path, the disk cache OFF.

    Args:
        spec: The side to measure. `TreeSpec.tree_dir` becomes the whole ``PYTHONPATH`` -- not a
            prepend -- so nothing from the parent's path can shadow the tree under test.
        base: Environment to start from; ``os.environ`` when None.

    Returns:
        A new dict. ``CPO_CACHE_ENABLED=0`` is set here, and CHECKED in the child's reply: setting
        it is not the same as it arriving.
    """
    env = dict(os.environ if base is None else base)
    env["PYTHONPATH"] = spec.tree_dir
    env["CPO_CACHE_ENABLED"] = "0"
    return env


def probe_argv(spec: TreeSpec, probe_path: str, args: Sequence[str]) -> list[str]:
    """The child's command line: this interpreter, the probe BY ABSOLUTE PATH, then the cell args.

    The absolute path matters. The probe must be the one from the tree that owns this module, while
    everything it imports comes from `TreeSpec.tree_dir`; invoking it as ``-m`` would resolve it out
    of the tree under test, which for the baseline tree does not have it.
    """
    return [spec.python, probe_path, "--probe", *args]


def _launch_subprocess(
    spec: TreeSpec, probe_path: str, args: Sequence[str], *, timeout_s: float
) -> ProbeResult:
    """Run one probe in a fresh interpreter and parse its single reply line.

    Args:
        spec: Which tree, which interpreter.
        probe_path: Absolute path of THIS file.
        args: Cell arguments, forwarded verbatim.
        timeout_s: Hard bound. A probe that hangs is a failed probe, not a slow one.

    Returns:
        The child's `ProbeResult`, or a synthetic ``ok=False`` one when the child crashed, timed
        out, or printed no reply line -- so a launch failure travels the same path as a build
        failure instead of raising something the comparator would not recognise.
    """
    try:
        p = subprocess.run(
            probe_argv(spec, probe_path, args),
            env=probe_env(spec),
            cwd=spec.tree_dir,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return ProbeResult(
            ok=False, error_class="TimeoutExpired", error_msg=f"probe exceeded {timeout_s}s"
        )
    for line in p.stdout.splitlines():
        if line.startswith(PROBE_PREFIX):
            return ProbeResult(**json.loads(line[len(PROBE_PREFIX) :]))
    return ProbeResult(
        ok=False,
        error_class="NoProbeReply",
        error_msg=(
            f"exit={p.returncode}; no {PROBE_PREFIX!r} line. stderr tail: {p.stderr[-400:]!r}"
        ),
    )


def _verify(res: ProbeResult, spec: TreeSpec, *, is_baseline: bool, need: str = "time"):
    """Apply the four verifications and return the duration, or raise `ProbeFailure`.

    The order is deliberate: `imported` is checked FIRST, because an unimportable tree is the
    failure most likely to be misread as a slow one, and every later check would be vacuous on a
    probe that never ran.

    Args:
        res: The child's reply.
        spec: The side that was asked for, used to check the tree it actually imported.
        is_baseline: Which role failed, so the message can say so.
        need: Which measurement the caller requires -- ``"time"`` for `ProbeResult.build_s`,
            ``"digest"`` for `ProbeResult.digest`. EVERY other verification is shared, which is the
            reason digest mode reuses this function instead of restating the tree, cache-flag and
            cache-growth checks it would otherwise have to duplicate and could get subtly wrong.

    Returns:
        `ProbeResult.build_s`, guaranteed finite and positive by `compare_cold_compile` downstream.

    Raises:
        ProbeFailure: On any of: the target did not import; the cell was skipped by `supports`;
            the build raised; the child read a `CPO_CACHE_ENABLED` other than "0"; the child
            imported a `fold_cp_ops` from outside `TreeSpec.tree_dir`; the on-disk cache GREW
            during a build the flag says was uncached; or a successful build named no tree at all.
            The failure is reported as ``setup`` when the child failed constructing the ctx rather
            than running ``build``.
    """
    if not res.imported:
        raise ProbeFailure(
            spec.label, "import", f"{res.error_class}: {res.error_msg}", is_baseline=is_baseline
        )
    if res.skipped:
        raise ProbeFailure(spec.label, "skipped", res.skipped, is_baseline=is_baseline)
    if res.cache_enabled != "0":
        raise ProbeFailure(
            spec.label,
            "cache_flag",
            f"child read CPO_CACHE_ENABLED={res.cache_enabled!r}, not '0'; the disk "
            f"cache may have served this build",
            is_baseline=is_baseline,
        )
    # `wrong_tree` fires ONLY on a tree the child actually named. An ABSENT tree is a different
    # fact and gets `no_tree` below: it says the child never reached the point where it reports
    # one, which on every observed instance meant an exception it had ALREADY captured. Firing
    # `wrong_tree` on None discarded that exception and reported "imported fold_cp_ops from None",
    # which names neither the failure nor a tree -- a check answering a question nobody asked.
    if res.tree and not os.path.abspath(res.tree).startswith(os.path.abspath(spec.tree_dir)):
        raise ProbeFailure(
            spec.label,
            "wrong_tree",
            f"child imported fold_cp_ops from {res.tree!r}, which is not under {spec.tree_dir!r}",
            is_baseline=is_baseline,
        )
    before, after = res.cache_dir_entries_before, res.cache_dir_entries_after
    if before is not None and after is not None and after > before:
        raise ProbeFailure(
            spec.label,
            "cache_grew",
            f"the JIT cache dir grew {before} -> {after} entries during a build the "
            f"flag says was uncached, so the flag did not take effect",
            is_baseline=is_baseline,
        )
    missing = res.build_s is None if need == "time" else not res.digest
    if not res.ok or missing:
        # A reply with no `phase` predates the field and falls through to "build", which is what
        # this check reported for every path before the split -- so an old reply reads exactly as
        # it used to rather than acquiring a new, wrong name.
        raise ProbeFailure(
            spec.label,
            "setup" if res.phase == "setup" else "build",
            f"{res.error_class}: {res.error_msg}",
            is_baseline=is_baseline,
        )
    # Last, and reachable only on an otherwise-clean reply: a build that SUCCEEDED while naming no
    # tree means the probe's own reporting is broken, not the tree. Kept as a check rather than
    # dropped, because the tree check is what stops a cross-tree run comparing one tree to itself,
    # and a silently unenforced identity check is the failure this whole module exists to prevent.
    if not res.tree:
        raise ProbeFailure(
            spec.label,
            "no_tree",
            "child produced a build time but never reported which fold_cp_ops it imported, so "
            "the cross-tree identity check could not run",
            is_baseline=is_baseline,
        )
    return float(res.build_s) if need == "time" else res.digest


def compare_trees(
    baseline: TreeSpec,
    candidate: TreeSpec,
    args: Sequence[str],
    *,
    samples: int | None = None,
    launch: Callable[..., ProbeResult] | None = None,
    out_path: str | None = None,
    probe_path: str | None = None,
    timeout_s: float = 1800.0,
):
    """Measure both trees' cold ``build`` and report the comparison.

    Purpose
        The cross-tree half of the compile bar: two trees, the same cell, alternated, medians.

    Semantics
        Sampling is `compare_cold_compile`'s -- at least five per side, ABBA alternation, medians
        not means. This function supplies the `measure` callable it consumes: one fresh interpreter
        per call, pointed at one tree, verified four ways (see `_verify`) before the duration is
        allowed to count. A failed verification RAISES rather than dropping the sample, because a
        side that cannot run is not a slow side.

        The result is written after EVERY sample, not at the end, so a sweep killed at hour three
        still leaves the samples it took. The file always contains the raw per-sample lists in
        collection order, so a reader can see drift instead of taking a median on trust.

    Args:
        baseline: The reference tree, conventionally `main`. `delta_s` is candidate MINUS baseline,
            so a positive delta means the candidate is slower.
        candidate: The tree under test.
        args: Cell arguments forwarded verbatim to the probe, e.g.
            ``["--target", "front", "--N", "2048"]``.
        samples: Per side; None takes `bench_utils.MIN_COLD_COMPILE_SAMPLES`. Below that floor the
            primitive refuses. Resolved HERE rather than as a default, because a default would
            import the primitive at module scope and break the probe (see the note at the top).
        launch: ``launch(spec, probe_path, args, timeout_s=...) -> ProbeResult``. Injectable so the
            tests drive this with canned replies instead of interpreters; defaults to the real
            subprocess launcher.
        out_path: Where to write the incremental JSON. None disables writing.
        probe_path: Absolute path of this file; derived from ``__file__`` when None.
        timeout_s: Per-probe bound.

    Returns:
        A ``ColdCompileComparison``. Read `resolvable` FIRST -- the bar is "no significant
        regression", which is a question about whether the delta clears the noise, and that
        property answers it.

    Raises:
        ProbeFailure: If either side fails any verification on any sample. No ratio is produced.
        ValueError: From `compare_cold_compile`, if `samples` is below the floor or a duration is
            not finite and positive.
    """
    # Deferred on purpose: see the module-scope note. The probe path never reaches this line.
    from benchmark.distributed.bench_utils import MIN_COLD_COMPILE_SAMPLES, compare_cold_compile

    samples = MIN_COLD_COMPILE_SAMPLES if samples is None else samples
    launch = launch or _launch_subprocess
    probe_path = probe_path or os.path.abspath(__file__)
    specs = {baseline.label: baseline, candidate.label: candidate}
    got: dict[str, list[float]] = {baseline.label: [], candidate.label: []}
    order: list[str] = []

    def _flush(final=None):
        if out_path is None:
            return
        payload: dict[str, Any] = {
            "cell": list(args),
            "baseline": baseline.label,
            "candidate": candidate.label,
            "baseline_tree": baseline.tree_dir,
            "candidate_tree": candidate.tree_dir,
            "samples_requested": samples,
            "order_so_far": list(order),
            "baseline_samples": list(got[baseline.label]),
            "candidate_samples": list(got[candidate.label]),
            "complete": final is not None,
        }
        if final is not None:
            payload.update(
                {
                    "resolvable": final.resolvable,  # the configured bar, first
                    "ratio": final.ratio,
                    "delta_s": final.delta_s,
                    "baseline_median_s": final.baseline_median_s,
                    "candidate_median_s": final.candidate_median_s,
                    "baseline_spread_s": final.baseline_spread_s,
                    "candidate_spread_s": final.candidate_spread_s,
                    # MAD beside the range, never instead of it. `resolvable` above is gated on the
                    # RANGE, so one slow cold compile out of five can make it False and a real
                    # regression then reads as "cannot tell". Emitting both means the first #5 run
                    # can be re-scored from its own payload instead of measured twice.
                    "baseline_mad_s": final.baseline_mad_s,
                    "candidate_mad_s": final.candidate_mad_s,
                    "order": list(final.order),
                    "samples": final.samples,
                    "describe": final.describe(),
                }
            )
        tmp = out_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp, out_path)  # atomic: a reader never sees a half-written file

    def measure(label: str) -> float:
        spec = specs[label]
        res = launch(spec, probe_path, args, timeout_s=timeout_s)
        seconds = _verify(res, spec, is_baseline=(label == baseline.label))
        order.append(label)
        got[label].append(seconds)
        _flush()
        return seconds

    result = compare_cold_compile(
        measure, baseline=baseline.label, candidate=candidate.label, samples=samples
    )
    _flush(result)
    return result


@dataclasses.dataclass(frozen=True)
class DigestVerdict:
    """One config's cross-tree byte-identity verdict.

    Attributes:
        config: The cell's argv tail -- what makes this the SAME cell on both sides.
        identical: True when both trees' code sections match. None when a side did not produce a
            digest, which is a THIRD state and never folded into False: "did not run" and "ran and
            differed" are opposite findings, and only the counts tell them apart.
        differing: Sorted code-section kinds that differ; empty when identical.
        note: Free text -- the failing side's own words when a side did not run.
    """

    config: tuple
    identical: bool | None
    differing: tuple = ()
    note: str = ""


def compare_digests(
    baseline: TreeSpec,
    candidate: TreeSpec,
    configs,
    *,
    launch: Callable[..., ProbeResult] | None = None,
    probe_path: str | None = None,
    timeout_s: float = 1800.0,
) -> tuple:
    """Digest each config in BOTH trees and report byte identity per config.

    Why this is not `compare_trees` with a different measurement
        It nearly is, and it deliberately reuses `_verify` (``need="digest"``) so the tree,
        cache-flag and cache-growth checks are the SAME code rather than restated and subtly
        different. What differs is the reduction: a timing comparison medians many samples, a byte
        comparison is one exact answer per config -- so there is no ABBA alternation and no sample
        count, and adding them would only add launches.

    Reading an IDENTICAL verdict
        It is a real result, and what licenses that is the verification chain rather than optimism:
        `_verify` refuses any reply whose child read ``CPO_CACHE_ENABLED`` as other than ``"0"``,
        and every config runs in its own fresh interpreter, so neither the disk cache nor the
        in-process ``@jit_cache`` memo can serve one artifact to two configs. **Weaken that chain
        and identity becomes unfalsifiable** -- the package's single `compile_key` reads none of the
        A2A knobs, so with the cache ON two of these configs genuinely CAN collide on one ``.o``.

    Args:
        baseline: The tree byte identity is measured AGAINST (``main``).
        candidate: The tree under test.
        configs: Iterable of argv tails, one per config. Must be non-empty and DISTINCT; a repeated
            entry compares a config against itself and reports an identity true of any tree.
        launch: Injected launcher, for testing. Defaults to the subprocess one.
        probe_path: Absolute path of this file for the child. Defaults to this file.
        timeout_s: Per-probe bound. A probe that hangs is a failed probe, not a slow one.

    Returns:
        ``(verdicts, counts)``. The counts are the denominator: a table of all-IDENTICAL rows looks
        the same whether five configs ran or one did.

    Raises:
        ValueError: On an empty or duplicated `configs`.
    """
    cfgs = [tuple(c) for c in configs]
    if not cfgs:
        raise ValueError("compare_digests: no configs -- an empty sweep reports zero problems")
    if len(set(cfgs)) != len(cfgs):
        dupes = sorted({c for c in cfgs if cfgs.count(c) > 1})
        raise ValueError(
            f"compare_digests: duplicate configs {dupes} -- a config compared against itself "
            f"reports an identity that is true of any tree whatsoever"
        )
    launch = launch or _launch_subprocess
    probe_path = probe_path or os.path.abspath(__file__)
    from fold_cp_ops.testing.cubin_identity import differing_kinds

    out, counts = [], {"identical": 0, "differs": 0, "not_run": 0}
    for cfg in cfgs:
        args = list(cfg) + ["--digest"]
        try:
            b = _verify(
                launch(baseline, probe_path, args, timeout_s=timeout_s),
                baseline,
                is_baseline=True,
                need="digest",
            )
            c = _verify(
                launch(candidate, probe_path, args, timeout_s=timeout_s),
                candidate,
                is_baseline=False,
                need="digest",
            )
        except ProbeFailure as e:
            out.append(DigestVerdict(cfg, None, note=str(e)))
            counts["not_run"] += 1
            continue
        diff = tuple(differing_kinds(c, b))
        out.append(DigestVerdict(cfg, not diff, diff))
        counts["identical" if not diff else "differs"] += 1
    return tuple(out), counts


# ----------------------------------------------------------------------------------- the probe ---
def _cache_dir_entries() -> int | None:
    """How many artifacts the on-disk JIT cache holds right now, or None if the dir cannot be read.

    Counted rather than inspected: the check downstream is whether it GREW across a build the flag
    says was uncached, and a count is enough to answer that while costing nothing.

    Which directory, and why not the obvious one
        `jit_cache` writes ``{sha}.o`` / ``{sha}.abi`` / ``{sha}.lock`` into
        ``get_cache_path() / _compute_source_fingerprint()`` -- a SUBDIRECTORY keyed by the source
        fingerprint -- so counting ``get_cache_path()`` itself counts fingerprint DIRS, which do not
        change when a build writes an artifact. Measured: seeding one artifact set took the
        top-level count ``1 -> 1`` and the fingerprint subdir ``0 -> 3``. Counting the top level
        would leave this verification inert while looking correct.

    Why the imports are OUTSIDE the try
        They were inside it under a bare ``except Exception``, against a symbol name
        (``_get_cache_dir``) that does not exist in ANY tree -- so this returned None on every call
        and the cache-growth verification never once fired. A missing symbol must be LOUD. Both
        names below were checked to exist in all 18 worktrees including ``main_oracle``, so raising
        here cannot break the cross-tree probe on a name one side lacks.

    Returns:
        The artifact count, 0 when no build has created the fingerprint dir yet (the true answer for
        a cold probe, and the one that keeps the growth check live where None would skip it), or
        None when the directory exists but cannot be listed.

    Raises:
        ImportError, AttributeError: If `fold_cp_ops` does not expose the cache helpers. Deliberate:
            a verification that cannot run must abort the comparison, not silently pass it.
    """
    from fold_cp_ops._internal.cache_utils import _compute_source_fingerprint, get_cache_path

    try:
        return len(list((get_cache_path() / _compute_source_fingerprint()).iterdir()))
    except FileNotFoundError:
        return 0  # nothing compiled yet -- a real count, not an unanswerable one
    except OSError:
        return None


def _tree_file() -> str | None:
    """The ``fold_cp_ops.__file__`` THIS child resolves, importing it if nothing else has yet.

    Purpose
        Make `ProbeResult.tree` answerable on EVERY exit path, not only the one that got as far as
        timing a build. The identity check downstream asks which tree the child imported, and a
        failure path that reports None forces that check to answer about a tree nobody looked for.

    Semantics
        Imports `fold_cp_ops` explicitly rather than reading ``sys.modules``, so the answer does not
        depend on whether some earlier import happened to pull it in -- targets defer their
        `fold_cp_ops` imports into `build`, so a resident-module read is None on exactly the paths
        that fail early. A tree that genuinely cannot be imported yields None, which the parent
        reports as its own verification rather than as a wrong tree.

    Returns:
        The absolute file of the imported package, or None if it could not be imported at all.
    """
    try:
        import fold_cp_ops

        return fold_cp_ops.__file__
    except Exception:
        return None


def compiled_from_handle(handle: Any) -> Any:
    """Find the compiled kernel inside whatever a `BenchTarget.build` returned.

    Purpose
        `build` returns a target-defined handle, not a compiled object, so the digest mode needs a
        convention for reaching the artifact. Making it a documented convention rather than a
        per-target hook keeps the untested surface to the build call itself.

    Semantics
        Two shapes, in order: a MAPPING carrying a ``"compiled"`` key (what
        ``targets/front_a2a.py`` returns), or anything `cubin_identity.exportable` accepts -- an
        object with an export surface, OR a compile wrapper holding one. A handle matching neither
        raises rather than returning None -- a None would be digested as an empty mapping, and two
        empty mappings compare EQUAL, which is the vacuous-comparison failure this whole path
        exists to avoid.

        The mapping branch stays FIRST and deliberately returns the value unwrapped: what
        ``targets/front_a2a.py`` stores under ``"compiled"`` is itself a wrapper, and `digest_export`
        unwraps it. Reaching through it here as well would work but would put the same knowledge in
        two places, which is exactly how this path broke.

    Args:
        handle: The return of ``target.build(ctx)``. Must carry exactly one compiled kernel; a
            handle holding several would digest only the one under ``"compiled"``, silently.

    Returns:
        The compiled object, ready for ``export_to_c``.

    Raises:
        TypeError: If no compiled kernel can be found, naming what was received.
    """
    if isinstance(handle, dict) and handle.get("compiled") is not None:
        return handle["compiled"]
    # Delegate the "is this exportable, and if not can it be unwrapped" question to the ONE place
    # that answers it. This used to be a local `hasattr(handle, "export_to_c")`, which is the same
    # hand-rolled check that -- spelled once per call site -- left the byte-identity gate handing
    # `digest_export` a wrapper and reading its (wrong) cache diagnosis as fact. A bare
    # `CompiledGemmBitcode` handle exposes neither export method and would have been refused here.
    from fold_cp_ops.testing.cubin_identity import exportable

    try:
        return exportable(handle)
    except TypeError:
        pass
    raise TypeError(
        f"cannot find a compiled kernel in a {type(handle).__name__} handle "
        f"(keys={sorted(handle) if isinstance(handle, dict) else 'n/a'}): digest mode needs the "
        f"object export_to_c is called on. A target must return it under 'compiled', or return it "
        f"directly."
    )


def run_digest(
    build_one: Callable[[], Any], *, cache_enabled: str | None, object_dir: str
) -> ProbeResult:
    """Build one cell and DIGEST what it compiled, instead of timing it.

    Purpose
        The byte-identity half of the cross-tree comparison. Same probe, same verifications, same
        reply protocol as `run_probe` -- only the measurement differs, so a digest cell inherits the
        tree check, the cache-flag check and the cache-growth check rather than restating them.

    Args:
        build_one: Callable returning the target's handle. Injectable so this function's
            reply-assembly is testable with no interpreter, no device and no compile.
        cache_enabled: ``CPO_CACHE_ENABLED`` as read by THIS process, captured before any import
            could mutate it. Must be ``"0"``: with the disk cache on, `jit_cache` returns a reloaded
            callable rather than the compiled object and `digest_export` refuses it by name.
        object_dir: Directory for the exported ``.o``. Must be writable and must be DISTINCT per
            config -- reusing one path across a sweep makes the second export overwrite the first,
            and the resulting "identical" verdict looks exactly like a real one.

    Returns:
        A `ProbeResult` with `digest` set on success, or ``ok=False`` carrying the exception.
    """
    import os as _os

    before = None
    tree = None
    try:
        before = _cache_dir_entries()
        import fold_cp_ops
        from fold_cp_ops.testing.cubin_identity import assert_has_code, digest_export

        tree = fold_cp_ops.__file__
        handle = build_one()
        _os.makedirs(object_dir, exist_ok=True)
        d = digest_export(compiled_from_handle(handle), _os.path.join(object_dir, "cell.o"))
        assert_has_code(d, "the probe's compiled cell")
    except Exception as e:
        return ProbeResult(
            ok=False,
            imported=True,
            tree=tree or _tree_file(),
            cache_enabled=cache_enabled,
            cache_dir_entries_before=before,
            cache_dir_entries_after=_cache_dir_entries(),
            phase="build",
            error_class=type(e).__name__,
            error_msg=repr(e)[:800],
        )
    return ProbeResult(
        ok=True,
        digest=d,
        imported=True,
        tree=tree,
        cache_enabled=cache_enabled,
        cache_dir_entries_before=before,
        cache_dir_entries_after=_cache_dir_entries(),
    )


def run_probe(build_one: Callable[[], Any], *, cache_enabled: str | None) -> ProbeResult:
    """Time one ``build`` and assemble the reply. The injectable core of the probe.

    Purpose
        Separated from `_probe_main` so its result-assembly can be tested with a fake ``build_one``
        -- no interpreter, no device, no dist init.

    Args:
        build_one: Callable performing exactly the work to be timed. It must do the build and
            nothing else; anything else it does lands in `ProbeResult.build_s`.
        cache_enabled: The value of ``CPO_CACHE_ENABLED`` as read by THIS process, captured by the
            caller before anything could mutate it.

    Returns:
        A `ProbeResult` with ``imported=True`` -- reaching this function means the imports
        succeeded, which is what distinguishes a slow tree from a broken one.
    """
    # INSIDE the try, and that placement is the point: `_cache_dir_entries` now RAISES on a missing
    # cache helper instead of swallowing it. Called before the try, that raise would leave the child
    # dead with a traceback and NO reply line -- indistinguishable from a child that was killed,
    # which is the one outcome this probe's contract forbids. Here it becomes a reported failure.
    before = None
    tree = None
    try:
        before = _cache_dir_entries()
        import fold_cp_ops

        tree = fold_cp_ops.__file__
        t0 = time.perf_counter()
        build_one()
        build_s = time.perf_counter() - t0
    except Exception as e:  # a build that raises is a failed probe, never a slow one
        return ProbeResult(
            ok=False,
            imported=True,
            phase="build",
            # `or _tree_file()`: the raise may have come from BEFORE the import above, and a reply
            # naming no tree makes the parent's identity check answer about a tree nobody looked for
            tree=tree or _tree_file(),
            cache_enabled=cache_enabled,
            cache_dir_entries_before=before,
            cache_dir_entries_after=_cache_dir_entries(),
            error_class=type(e).__name__,
            error_msg=repr(e)[:800],
        )
    return ProbeResult(
        ok=True,
        build_s=build_s,
        imported=True,
        tree=tree,
        cache_enabled=cache_enabled,
        cache_dir_entries_before=before,
        cache_dir_entries_after=_cache_dir_entries(),
    )


def _probe_main(argv: Sequence[str]) -> int:
    """Child entry: resolve one target in THIS tree, time its ``build``, print one reply line.

    Every failure mode prints a `ProbeResult` and exits 0. That is deliberate: the parent
    distinguishes an unimportable tree from a slow one by READING the reply, and a child that
    crashed with a traceback and no reply is indistinguishable from a child that was killed.

    Args:
        argv: ``--target NAME --N INT --D INT [--B INT] [--rd INT] [--cp0 INT] [--cp1 INT]``.

    Returns:
        0 always, unless the reply itself could not be printed.
    """
    import argparse

    # FIRST, before any import can mutate it: what this process actually reads for the cache flag.
    cache_enabled = os.environ.get("CPO_CACHE_ENABLED")

    ap = argparse.ArgumentParser(prog="cold_compile --probe")
    ap.add_argument("--probe", action="store_true")
    # DIGEST mode: emit the compiled cell's code-section fingerprint instead of a build time.
    # Same cell, same verifications, same reply line -- only the measurement differs.
    ap.add_argument("--digest", action="store_true")
    ap.add_argument("--target", required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--D", type=int, required=True)
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--rd", type=int, default=2)
    args = ap.parse_args(list(argv))

    try:
        import importlib
        import pkgutil

        import benchmark.distributed.harness.targets as _targets
        from benchmark.distributed.harness.registry import resolve

        for m in pkgutil.iter_modules(_targets.__path__):
            importlib.import_module(f"{_targets.__name__}.{m.name}")
        target = resolve([args.target])[0]
    except Exception as e:
        print(
            ProbeResult(
                ok=False,
                imported=False,
                tree=_tree_file(),
                cache_enabled=cache_enabled,
                error_class=type(e).__name__,
                error_msg=repr(e)[:800],
            ).to_json(),
            flush=True,
        )
        return 0

    try:
        import torch

        from benchmark.distributed.bench_utils import resolve_dist
        from benchmark.distributed.harness.target import Ctx
        from fold_cp_ops.distributed.distributed_manager import DistributedManager
        from tests.distributed.test_gemm_a2a_epi import _cp_axis_sizes, _trimul_placements

        if hasattr(DistributedManager, "_derive_dist_env_from_slurm"):
            DistributedManager._derive_dist_env_from_slurm()
        local = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
        torch.cuda.set_device(local)
        os.environ.setdefault("CPO_DISTRIBUTED_INIT_METHOD", "ENV")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        # No port literal in the repo, and STDLIB ONLY -- this runs in the
        # probe child with PYTHONPATH pointing at the tree being MEASURED, which may be an
        # older tree without `fold_cp_ops._internal.port_selection`. Importing it here would
        # break the probe on exactly the cross-tree comparison it exists to perform, and this
        # module's own test_no_non_stdlib_import_at_module_scope refuses it.
        # Bind 0 and read back: the OS hands out a port it knows is free, which is stronger
        # than deriving one, and there is no number to hardcode. Safe here (unlike a
        # multi-rank rendezvous) because this probe is single-process -- nothing else has to
        # derive the same value independently.
        if "MASTER_PORT" not in os.environ:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as _s:
                _s.bind(("", 0))
                os.environ["MASTER_PORT"] = str(_s.getsockname()[1])
        from collections import OrderedDict

        DistributedManager.initialize(
            OrderedDict(cp=int(os.environ["WORLD_SIZE"])), device_type="cuda"
        )
        dm = DistributedManager()
        DistributedManager.init_nvshmem()
        pm = _trimul_placements(dm)[1]
        axis = _cp_axis_sizes(dm)
        cp0, cp1 = (int(axis[0]), int(axis[1])) if len(axis) == 2 else (int(axis[0]), 1)
        ctx = Ctx(
            dm=dm,
            pm=pm,
            da=resolve_dist(dm),
            N=args.N,
            cp0=cp0,
            cp1=cp1,
            Dloc=max(1, args.D // (cp0 * cp1)),
            B=args.B,
            rd=args.rd,
            device=dm.device,
            rank=dm.rank,
        )
    except Exception as e:
        print(
            ProbeResult(
                ok=False,
                imported=True,
                tree=_tree_file(),
                cache_enabled=cache_enabled,
                phase="setup",
                error_class=type(e).__name__,
                error_msg=repr(e)[:800],
            ).to_json(),
            flush=True,
        )
        return 0

    reason = target.supports(ctx)
    if reason is not None:
        print(
            ProbeResult(
                ok=False,
                imported=True,
                tree=_tree_file(),
                cache_enabled=cache_enabled,
                skipped=reason,
            ).to_json(),
            flush=True,
        )
        return 0

    handle = {}

    def _build_one():
        handle["h"] = target.build(ctx)

    if args.digest:
        import tempfile

        res = run_digest(
            _build_one,
            cache_enabled=cache_enabled,
            # A FRESH directory per child, so two configs can never export over one path -- the
            # failure that makes a byte comparison read identical for the wrong reason.
            object_dir=tempfile.mkdtemp(prefix="cold_compile_digest_"),
        )
    else:
        res = run_probe(_build_one, cache_enabled=cache_enabled)
    if target.teardown is not None and handle.get("h") is not None:
        try:
            target.teardown(handle["h"])
        except Exception:
            pass
    print(res.to_json(), flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch: ``--probe`` is the child, anything else is not offered as a CLI yet.

    The comparator is a library call (`compare_trees`) rather than a CLI, because its two
    `TreeSpec`s are absolute paths that a driver script owns -- baking them into argv here would
    put one machine's layout in the repo.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--probe" in argv:
        return _probe_main(argv)
    print(
        "cold_compile: use --probe (child), or call compare_trees() from a driver.", file=sys.stderr
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

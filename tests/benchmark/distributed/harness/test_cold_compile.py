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


"""Tests for `benchmark.distributed.harness.cold_compile`.

No subprocess and no GPU anywhere here. `compare_trees` takes an injectable ``launch``, for the
same reason `compare_cold_compile` takes an injectable ``measure``: the mechanics of "a fresh
interpreter in that tree" are the one thing that cannot be exercised cheaply, so they are pushed to
the edge and everything else is driven with canned replies.
"""

import json
import os
import sys

import pytest

from benchmark.distributed.harness.cold_compile import (
    ProbeFailure,
    _cache_dir_entries,
    ProbeResult,
    TreeSpec,
    compare_digests,
    compare_trees,
    compiled_from_handle,
    run_digest,
    run_probe,
)
from fold_cp_ops.testing.kernel_matrix import matrix_exempt

pytestmark = matrix_exempt(
    "the subject is host-side process orchestration and its verification rules -- no kernel is "
    "launched, no shape axis exists, and the durations are canned, so no KernelMatrix applies"
)

BASE = TreeSpec(label="main", tree_dir="/trees/main_oracle", python="/py")
CAND = TreeSpec(label="ours", tree_dir="/trees/front_diff", python="/py")
ARGS = ["--target", "front", "--N", "2048", "--D", "128"]


def _reply(spec, build_s, **override):
    """A well-formed probe reply for `spec`, with named fields corrupted on request.

    Args:
        spec: The side being answered, so `tree` lands inside its own `TreeSpec.tree_dir` --
            the default must PASS every verification, or a test that corrupts one field would
            not know which one it was testing.
        build_s: The duration to report.
        **override: Fields to replace, one per verification under test.

    Returns:
        A `ProbeResult`.
    """
    fields = dict(
        ok=True,
        build_s=build_s,
        imported=True,
        tree=os.path.join(spec.tree_dir, "fold_cp_ops", "__init__.py"),
        cache_enabled="0",
        cache_dir_entries_before=3,
        cache_dir_entries_after=3,
    )
    fields.update(override)
    return ProbeResult(**fields)


def _launcher(per_side, corrupt=None, observed=None, out_path=None):
    """A ``launch`` that hands back canned replies, optionally corrupting one side's.

    Args:
        per_side: label -> durations, consumed in call order for that side.
        corrupt: label -> field overrides applied to that side's replies.
        observed: A list the launcher appends the incremental file's sample-count to on every
            call, so a test can watch the file grow DURING the run rather than after it.
        out_path: The file `observed` should be read from.

    Returns:
        ``launch(spec, probe_path, args, *, timeout_s)``.
    """
    it = {k: iter(v) for k, v in per_side.items()}
    corrupt = corrupt or {}

    def launch(spec, probe_path, args, *, timeout_s):
        if observed is not None:
            n = 0
            if out_path and os.path.exists(out_path):
                d = json.load(open(out_path))
                n = len(d["baseline_samples"]) + len(d["candidate_samples"])
            observed.append(n)
        return _reply(spec, next(it[spec.label]), **corrupt.get(spec.label, {}))

    return launch


def test_the_comparator_realizes_the_abba_order_the_primitive_promises():
    """The order the sides were actually measured in is ABBA, and is reported.

    Blocked sampling -- five of one side then five of the other -- hands the whole of any machine
    drift to one side, and plain ABAB still gives one side the first-after-a-gap slot in every
    pair, which is the slot that pays page-cache and import costs. The primitive rotates that; this
    asserts the comparator does not undo it by, say, caching a spec per side.
    """
    res = compare_trees(
        BASE, CAND, ARGS, samples=5, launch=_launcher({"main": [1.0] * 5, "ours": [2.0] * 5})
    )
    assert list(res.order) == ["main", "ours", "ours", "main"] * 2 + ["main", "ours"]
    assert res.samples == 5
    assert res.baseline_median_s == 1.0 and res.candidate_median_s == 2.0
    assert res.ratio == pytest.approx(2.0)


@pytest.mark.parametrize(
    "override,kind",
    [
        (
            {
                "imported": False,
                "ok": False,
                "error_class": "ModuleNotFoundError",
                "error_msg": "No module named 'fold_cp_ops'",
            },
            "import",
        ),
        ({"cache_enabled": "1"}, "cache_flag"),
        ({"tree": "/somewhere/else/fold_cp_ops/__init__.py"}, "wrong_tree"),
        ({"cache_dir_entries_after": 9}, "cache_grew"),
    ],
    ids=["unimportable", "cache_flag_not_set", "imported_wrong_tree", "cache_grew_anyway"],
)
def test_each_verification_fails_on_its_own(override, kind):
    """Four independent ways a comparison can look legitimate while measuring nothing.

    Each is checked separately because each has a different "did not fire" state, and a suite that
    only ever corrupted one field would pass with the other three checks deleted. The default reply
    passes all four, so the corrupted field is the only variable.

    The `import` case is the one that motivated the set: a tree that cannot be imported must surface
    as BROKEN, never as a large `build_s` and never as a missing sample, because a comparator that
    treats an unimportable baseline as a slow one reports a fabricated ratio.
    """
    with pytest.raises(ProbeFailure) as ei:
        compare_trees(
            BASE,
            CAND,
            ARGS,
            samples=5,
            launch=_launcher({"main": [1.0] * 5, "ours": [2.0] * 5}, corrupt={"main": override}),
        )
    assert ei.value.kind == kind
    assert ei.value.side == "main"
    assert "BASELINE_BROKEN" in str(ei.value)
    assert "NO ratio is reported" in str(ei.value)


def test_the_broken_side_is_named_so_a_reader_knows_which_tree_to_fix():
    """A broken CANDIDATE says so, rather than borrowing the baseline's wording."""
    with pytest.raises(ProbeFailure) as ei:
        compare_trees(
            BASE,
            CAND,
            ARGS,
            samples=5,
            launch=_launcher(
                {"main": [1.0] * 5, "ours": [2.0] * 5}, corrupt={"ours": {"cache_enabled": None}}
            ),
        )
    assert ei.value.side == "ours" and "CANDIDATE_BROKEN" in str(ei.value)


def test_the_result_file_is_written_after_every_sample(tmp_path):
    """A sweep killed part-way still leaves the samples it took.

    A cold-compile sweep is minutes per sample and the cells that matter are the slow ones, so
    "write at the end" and "write nothing" are the same file when a job hits its wall clock. The
    launcher watches the file DURING the run: the count it sees on call k must be exactly k, which
    a write-at-the-end implementation fails on the very first call.
    """
    out = str(tmp_path / "cold.json")
    seen = []
    res = compare_trees(
        BASE,
        CAND,
        ARGS,
        samples=5,
        out_path=out,
        launch=_launcher({"main": [1.0] * 5, "ours": [2.0] * 5}, observed=seen, out_path=out),
    )
    assert seen == list(range(10)), f"file did not grow one sample at a time: {seen}"
    d = json.load(open(out))
    assert d["complete"] is True
    assert d["baseline_samples"] == [1.0] * 5 and d["candidate_samples"] == [2.0] * 5
    assert d["resolvable"] is res.resolvable  # the configured bar, emitted first
    assert d["ratio"] == pytest.approx(2.0)
    assert d["order"] == list(res.order)
    assert d["cell"] == ARGS


@pytest.mark.parametrize("samples", [1, 4])
def test_the_five_sample_floor_propagates_from_the_primitive(samples):
    """The comparator does not re-implement the floor, so it cannot drift from it.

    Asserting the primitive's own message is the point: if the comparator ever grew its own check,
    this would still pass while the two floors diverged silently.
    """
    with pytest.raises(ValueError, match=r"below the floor of 5 per side"):
        compare_trees(
            BASE,
            CAND,
            ARGS,
            samples=samples,
            launch=_launcher({"main": [1.0] * 9, "ours": [2.0] * 9}),
        )


def test_run_probe_reports_a_raising_build_as_broken_not_as_slow():
    """A build that raises is a failed probe. The duration field stays empty rather than being
    filled with the time spent before the exception, which would be a plausible small number."""

    def boom():
        raise RuntimeError("compile exploded")

    res = run_probe(boom, cache_enabled="0")
    assert res.ok is False and res.build_s is None
    assert res.imported is True  # we got far enough to import; the tree is not broken
    assert res.error_class == "RuntimeError" and "compile exploded" in res.error_msg


def test_run_probe_times_only_what_it_was_handed():
    """`build_s` measures the callable and nothing around it."""
    import time as _t

    res = run_probe(lambda: _t.sleep(0.02), cache_enabled="0")
    assert res.ok is True and res.imported is True
    assert 0.015 < res.build_s < 1.0


# ── the PROBE's real import surface ────────────────────────────────────────────────────────────
# Every test above drives the comparator with an injected launcher, so none of them ever executes
# the probe path's imports. That gap is not hypothetical: it shipped. The first real cross-tree run
# failed with `BASELINE_BROKEN: import -- cannot import name 'MIN_COLD_COMPILE_SAMPLES' from
# 'benchmark.distributed.bench_utils'`, because the probe is invoked by absolute path into OUR file
# while `PYTHONPATH` points at the tree under measurement, so a module-scope import of an ours-only
# symbol makes the probe unrunnable against the baseline. The verification caught it and refused a
# ratio -- but a test should have caught it first.

COLD_COMPILE_PATH = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
    ),
    "benchmark",
    "distributed",
    "harness",
    "cold_compile.py",
)


def _load_against_a_tree_without_the_primitives(monkeypatch):
    """Exec `cold_compile.py` fresh with `bench_utils` stubbed EMPTY, as the baseline tree has it.

    Args:
        monkeypatch: pytest's, used to swap the cached module for the duration of the test.

    Returns:
        The freshly executed module object.

    Raises:
        ImportError: If the module needs a symbol the baseline tree does not have -- which is the
            whole point of the check.
    """
    import importlib.util
    import types

    empty = types.ModuleType("benchmark.distributed.bench_utils")  # neither ours-only symbol
    monkeypatch.setitem(sys.modules, "benchmark.distributed.bench_utils", empty)
    spec = importlib.util.spec_from_file_location(
        "_cold_compile_in_baseline_tree", COLD_COMPILE_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    # Register before exec: `dataclasses` resolves a field's type via `sys.modules[cls.__module__]`,
    # so a module built from a spec but never registered fails inside @dataclass, not at the import
    # we are actually testing.
    monkeypatch.setitem(sys.modules, spec.name, mod)
    spec.loader.exec_module(mod)
    return mod


def test_the_probe_half_runs_in_a_tree_that_lacks_the_ours_only_primitives(monkeypatch):
    """The probe imports and RUNS where `compare_cold_compile` does not exist.

    Importability alone would be too weak a claim -- a module can import and still reach for a
    missing symbol on the first call -- so this also drives `run_probe` to completion under the
    stub. `PYTHONPATH` points at the measured tree, so anything the probe touches at module scope
    must exist in EVERY tree; in practice that means the standard library.
    """
    mod = _load_against_a_tree_without_the_primitives(monkeypatch)
    res = mod.run_probe(lambda: None, cache_enabled="0")
    assert res.ok is True and res.imported is True and res.build_s > 0.0


def test_the_comparator_half_genuinely_needs_them(monkeypatch):
    """The deferral moved the dependency; it did not delete it.

    Without this, the test above would also pass if someone "fixed" the import by dropping the
    sampling primitive altogether and hand-rolling a median -- which would silently lose the
    five-sample floor and the ABBA alternation. Asserting the comparator still FAILS against a tree
    without the primitives is what distinguishes deferred from discarded.
    """
    mod = _load_against_a_tree_without_the_primitives(monkeypatch)
    with pytest.raises(ImportError, match="MIN_COLD_COMPILE_SAMPLES|compare_cold_compile"):
        mod.compare_trees(
            BASE, CAND, ARGS, samples=5, launch=_launcher({"main": [1.0] * 5, "ours": [2.0] * 5})
        )


def test_no_non_stdlib_import_at_module_scope():
    """A structural guard, cheap and total, for the property the two tests above check behaviourally.

    They exercise the imports that today's code performs; this one fails the moment a NEW
    module-scope import is added, including one whose symbol happens to exist in both trees today
    and stops existing later. Belt and braces, on a file whose whole contract is what it imports.
    """
    import ast

    tree = ast.parse(open(COLD_COMPILE_PATH, encoding="utf-8").read())
    mods = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            mods += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.append(node.module)
    offenders = [m for m in mods if m.split(".")[0] not in sys.stdlib_module_names]
    assert offenders == [], (
        f"module-scope imports outside the standard library: {offenders}. The probe runs with "
        f"PYTHONPATH pointing at the tree being MEASURED, so each of these must exist in every "
        f"such tree. Move it into the function that uses it."
    )


def test_the_tree_is_reported_by_importing_not_by_reading_sys_modules():
    """A probe that has not yet imported `fold_cp_ops` must still name its tree.

    This is the mechanism behind a live venue B failure. Every `BenchTarget` DEFERS its
    `fold_cp_ops` imports into `build` (`targets/front_a2a.py` holds six of them inside functions),
    so on the probe's early exit paths -- target resolution, ctx setup, a `supports` refusal --
    nothing has imported the package yet. An implementation reading ``sys.modules`` returns None on
    exactly those paths, and the parent's tree check then answers about a tree nobody looked for.

    Run in a FRESH interpreter, because this one has `fold_cp_ops` resident from the imports above
    and would let a ``sys.modules`` read pass. The child prints both answers so the difference is
    the measurement rather than an assumption about it.
    """
    import subprocess

    root = os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
    )
    code = (
        "import sys, json;"
        "from benchmark.distributed.harness.cold_compile import _tree_file;"
        "resident = getattr(sys.modules.get('fold_cp_ops'), '__file__', None);"
        "print(json.dumps({'resident': resident, 'imported': _tree_file()}))"
    )
    env = {**os.environ, "PYTHONPATH": root}
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=root, env=env, capture_output=True, text=True, timeout=300
    )
    assert out.returncode == 0, out.stderr[-2000:]
    got = json.loads(out.stdout.strip().splitlines()[-1])
    # The control: in a fresh child the package is genuinely NOT resident, so a sys.modules read is
    # None. Without this the assertion below could pass on a child that had imported it anyway.
    assert got["resident"] is None, "control: the child must not already hold fold_cp_ops"
    assert got["imported"] and got["imported"].endswith(
        os.path.join("fold_cp_ops", "__init__.py")
    ), f"_tree_file must import to answer, got {got['imported']!r}"


def test_a_setup_failure_reports_its_own_error_and_not_a_wrong_tree():
    """The venue B regression: an early exit carries a real exception, which `wrong_tree` ate.

    `_probe_main`'s ctx-setup exit returns ``ok=False, imported=True`` with the exception in
    ``error_class``/``error_msg`` -- and, before this fix, no ``tree``. `_verify` reached its tree
    check first and raised ``wrong_tree: child imported fold_cp_ops from None``, which names
    neither the failure nor a tree, while the child's own words were never printed. The verdict
    must be ``build``, carrying the text the child actually sent.
    """
    with pytest.raises(ProbeFailure) as ei:
        compare_trees(
            BASE,
            CAND,
            ARGS,
            samples=5,
            launch=_launcher(
                {"main": [1.0] * 5, "ours": [None] * 5},
                corrupt={
                    "ours": {
                        "ok": False,
                        "tree": None,
                        "error_class": "RuntimeError",
                        "error_msg": "nvshmem init failed on this node",
                    }
                },
            ),
        )
    assert ei.value.kind == "build", "an absent tree must not outrank the error the child sent"
    assert ei.value.side == "ours"
    assert "nvshmem init failed on this node" in str(ei.value)


def test_a_supports_refusal_outranks_both_tree_checks():
    """A legitimate skip stays a skip even though it also reports no tree.

    The ordering question is worth pinning rather than reading off the source: a `supports` refusal
    and a broken tree both arrive as ``ok=False`` with ``tree=None``, and if the tree check ran
    first every skipped cell would be reported as a broken candidate. This is the silence assertion
    for `wrong_tree` and `no_tree` -- they must NOT fire here.
    """
    with pytest.raises(ProbeFailure) as ei:
        compare_trees(
            BASE,
            CAND,
            ARGS,
            samples=5,
            launch=_launcher(
                {"main": [1.0] * 5, "ours": [None] * 5},
                corrupt={"ours": {"ok": False, "tree": None, "skipped": "needs cp>=2"}},
            ),
        )
    assert ei.value.kind == "skipped"
    assert "needs cp>=2" in str(ei.value)


def test_an_otherwise_clean_reply_that_names_no_tree_is_its_own_failure():
    """`no_tree` is a distinct verdict, so the identity check cannot be silently unenforced.

    Reached only when everything else passed: a build that produced a duration while naming no
    tree. That is the probe's own reporting broken rather than the tree, and it must not be
    reported as ``wrong_tree`` (which would claim a tree was named) nor pass (which would let a
    cross-tree run compare one tree against itself with no check having run).
    """
    with pytest.raises(ProbeFailure) as ei:
        compare_trees(
            BASE,
            CAND,
            ARGS,
            samples=5,
            launch=_launcher(
                {"main": [1.0] * 5, "ours": [2.0] * 5}, corrupt={"ours": {"tree": None}}
            ),
        )
    assert ei.value.kind == "no_tree"
    assert ei.value.side == "ours"


def _seed_cache(tmp_path, monkeypatch):
    """Point the JIT cache at `tmp_path` and return its fingerprint subdir, created but empty.

    Args:
        tmp_path: A writable dir. The real cache is SHARED across every process on the box, so a
            test that counted it would be measuring other people's builds and could not seed it.
        monkeypatch: Used to set `cache_utils.CACHE_DIR`, which `get_cache_path` reads. It is a
            module global captured from the env at import, so setting the env var here would be
            too late.

    Returns:
        ``pathlib.Path`` of ``get_cache_path() / _compute_source_fingerprint()`` -- where
        `jit_cache` actually writes ``{sha}.o``.
    """
    from fold_cp_ops._internal import cache_utils

    monkeypatch.setattr(cache_utils, "CACHE_DIR", str(tmp_path))
    fp = cache_utils.get_cache_path() / cache_utils._compute_source_fingerprint()
    fp.mkdir(parents=True, exist_ok=True)
    return fp


def test_the_cache_count_is_answerable_at_all(tmp_path, monkeypatch):
    """The cache-growth verification was permanently inert, and nothing noticed.

    `_cache_dir_entries` imported a symbol -- ``_get_cache_dir`` -- that exists in NO tree (checked
    across all 18 worktrees including `main_oracle`), inside a bare ``except Exception``. It
    therefore returned None on every call, `_verify`'s growth check reads
    ``before is not None and after is not None``, and the check never fired once since it landed.

    The suite could not have caught it: every other test injects a launcher with canned
    `ProbeResult` payloads, so the real function is never called. This one calls it.
    """
    _seed_cache(tmp_path, monkeypatch)
    n = _cache_dir_entries()
    assert n is not None, "a None count silently disables the growth verification"
    assert isinstance(n, int)


def test_the_cache_count_watches_the_directory_that_actually_grows(tmp_path, monkeypatch):
    """Counting `get_cache_path()` itself would leave the check inert for a SECOND reason.

    `jit_cache` writes into ``get_cache_path() / _compute_source_fingerprint()``, so the top level
    holds fingerprint DIRS and does not change when a build produces an artifact. Measured on the
    real helpers: seeding one artifact set moves the top level ``1 -> 1`` and the fingerprint subdir
    ``0 -> 3``. A fix that only repaired the symbol name would pass the test above and still never
    fire -- which is why the count is asserted to MOVE, not merely to be non-None.
    """
    from fold_cp_ops._internal import cache_utils

    fp = _seed_cache(tmp_path, monkeypatch)
    top = cache_utils.get_cache_path()
    top_before, before = len(list(top.iterdir())), _cache_dir_entries()
    for suffix in (".o", ".abi", ".lock"):  # exactly what a cold build writes
        (fp / f"deadbeef{suffix}").write_text("x")
    top_after, after = len(list(top.iterdir())), _cache_dir_entries()

    assert after > before, f"the counted dir must grow with the artifacts, got {before} -> {after}"
    assert top_after == top_before, (
        "control: the top-level count does NOT move, which is why counting it would be inert"
    )


def test_a_missing_cache_helper_is_loud_rather_than_a_none(tmp_path, monkeypatch):
    """A verification that cannot run must abort the comparison, never quietly pass it.

    This is the defect's own shape: a bare ``except Exception`` around the import turned "this repo
    does not have that symbol" into "no opinion", which `_verify` reads as nothing to check. The
    imports now sit outside the try, so the failure surfaces -- as a reported `build` failure, since
    `run_probe` calls this inside its own try and the child must always print a reply.
    """
    from fold_cp_ops._internal import cache_utils

    monkeypatch.delattr(cache_utils, "get_cache_path")
    with pytest.raises((ImportError, AttributeError)):
        _cache_dir_entries()


def test_a_cold_tree_with_no_fingerprint_dir_counts_zero_rather_than_unknown(tmp_path, monkeypatch):
    """The cold case is the probe's NORMAL case, so it must not be the one that skips the check.

    Before any build the fingerprint subdir does not exist. Returning None there would disable the
    growth check on exactly the runs this module exists to measure; 0 is the true count and keeps it
    live.
    """
    from fold_cp_ops._internal import cache_utils

    monkeypatch.setattr(cache_utils, "CACHE_DIR", str(tmp_path / "never_built"))
    assert _cache_dir_entries() == 0


@pytest.mark.parametrize(
    "phase,kind",
    [("setup", "setup"), ("build", "build"), (None, "build")],
    ids=["ctx_setup_failure", "build_failure", "reply_without_a_phase"],
)
def test_a_setup_failure_and_a_build_failure_are_named_apart(phase, kind):
    """A dist-init failure must not be reported as a BUILD failure.

    `_probe_main`'s ctx-setup exit and `run_probe`'s build exit both produce
    ``ok=False, imported=True`` with the exception in ``error_class``/``error_msg``, and were
    otherwise indistinguishable. Measured cost: a real venue B diagnosis opened by reading `build`
    and searching `targets/front_a2a.py` for a rendezvous it does not contain -- the failure was
    `DistributedManager.init_nvshmem()` during ctx construction, one frame earlier.

    The third case is the compatibility one and is the reason the check reads ``== "setup"`` rather
    than ``!= "build"``: a reply that predates the field carries ``phase=None`` and must keep the
    name it has always had, not acquire a new one that is wrong.
    """
    with pytest.raises(ProbeFailure) as ei:
        compare_trees(
            BASE,
            CAND,
            ARGS,
            samples=5,
            launch=_launcher(
                {"main": [1.0] * 5, "ours": [None] * 5},
                corrupt={
                    "ours": {
                        "ok": False,
                        "phase": phase,
                        "error_class": "RuntimeError",
                        "error_msg": "Expected global_ranks.size() > 1 to be true, but got false.",
                    }
                },
            ),
        )
    assert ei.value.kind == kind
    assert ei.value.side == "ours"
    assert "global_ranks.size() > 1" in str(ei.value), "the child's own words must survive"


# ── the DIGEST mode: byte identity across two trees ───────────────────────────────────────────
def _dreply(spec, digest, **override):
    """A well-formed DIGEST reply for `spec`. Defaults pass every verification."""
    fields = dict(
        ok=True,
        digest=digest,
        imported=True,
        tree=os.path.join(spec.tree_dir, "fold_cp_ops", "__init__.py"),
        cache_enabled="0",
        cache_dir_entries_before=3,
        cache_dir_entries_after=3,
    )
    fields.update(override)
    return ProbeResult(**fields)


def _dlauncher(per_side):
    """A ``launch`` handing back canned digest replies, keyed by side label."""

    def launch(spec, probe_path, args, *, timeout_s):
        v = per_side[spec.label]
        return _dreply(spec, v) if not isinstance(v, ProbeResult) else v

    return launch


_D_SAME = {".text": (100, "aa" * 32), ".nv.constant0": (8, "bb" * 32)}
_D_DIFF = {".text": (100, "cc" * 32), ".nv.constant0": (8, "bb" * 32)}


def test_the_handle_convention_finds_the_compiled_kernel_or_refuses():
    """`build` returns a target-defined handle, so digest mode needs a convention to reach the artifact.

    The refusal is the half that matters: a handle it cannot read must RAISE, never return None. A
    None would be digested as an empty mapping, and two empty mappings compare EQUAL -- so every
    config would report byte-identical while nothing was measured.
    """

    class _Fake:
        def export_to_c(self, **kw):
            pass

    obj = _Fake()
    assert compiled_from_handle({"compiled": obj, "recv": None}) is obj
    assert compiled_from_handle(obj) is obj
    with pytest.raises(TypeError, match=r"cannot find a compiled kernel"):
        compiled_from_handle({"recv": None, "run": None})
    with pytest.raises(TypeError, match=r"cannot find a compiled kernel"):
        compiled_from_handle({"compiled": None})


def test_digest_mode_reports_the_build_exception_rather_than_an_empty_digest(tmp_path):
    """A build that raises must be a FAILED probe, never a probe with nothing to compare.

    Same rule as the timing mode and for a sharper reason: an empty digest does not merely lose a
    sample, it compares EQUAL to another empty digest and reports byte identity.
    """

    def boom():
        raise RuntimeError("nvshmem init failed")

    res = run_digest(boom, cache_enabled="0", object_dir=str(tmp_path))
    assert res.ok is False and res.digest is None
    assert res.phase == "build" and "nvshmem init failed" in res.error_msg


def test_identical_and_differing_configs_are_told_apart(tmp_path):
    """The comparator's two real verdicts, and the kinds it names when they differ."""
    same, _ = compare_digests(
        BASE, CAND, [("--N", "2048")], launch=_dlauncher({"main": _D_SAME, "ours": _D_SAME})
    )
    assert same[0].identical is True and same[0].differing == ()

    diff, counts = compare_digests(
        BASE, CAND, [("--N", "2048")], launch=_dlauncher({"main": _D_SAME, "ours": _D_DIFF})
    )
    assert diff[0].identical is False
    assert diff[0].differing == (".text",), "only .text moved; .nv.constant0 is shared"
    assert counts == {"identical": 0, "differs": 1, "not_run": 0}


def test_a_side_that_did_not_run_is_a_THIRD_state_not_a_difference():
    """ "did not run" and "ran and differed" are opposite findings; folding them loses the sweep.

    A truncated sweep whose failures counted as differences would read as a byte-identity
    regression, and one whose failures counted as identity would read as a clean pass. Neither is
    what happened, so the verdict is None and the denominator carries the count.
    """
    broken = ProbeResult(
        ok=False,
        imported=True,
        cache_enabled="0",
        phase="setup",
        error_class="RuntimeError",
        error_msg="global_ranks.size() > 1",
    )
    v, counts = compare_digests(
        BASE, CAND, [("--N", "2048")], launch=_dlauncher({"main": _D_SAME, "ours": broken})
    )
    assert v[0].identical is None and counts["not_run"] == 1
    assert "global_ranks" in v[0].note


def test_an_empty_or_duplicated_sweep_is_refused():
    """Both shapes report clean while measuring less than they claim, so both raise.

    An empty sweep produces zero problems; a duplicated config compares a cell against itself and
    reports an identity that is true of any tree whatsoever.
    """
    with pytest.raises(ValueError, match=r"an empty sweep reports zero problems"):
        compare_digests(BASE, CAND, [], launch=_dlauncher({}))
    with pytest.raises(ValueError, match=r"compared against itself"):
        compare_digests(
            BASE, CAND, [("--N", "2048"), ("--N", "2048")], launch=_dlauncher({"main": _D_SAME})
        )


def test_the_cache_flag_verification_still_gates_a_digest_reply():
    """What licenses reading IDENTICAL as genuine is the verification chain, not optimism.

    The package's single `compile_key` reads none of the A2A knobs, so with the disk cache ON two
    configs genuinely can be served one ``.o``. The cache-flag check is what makes that
    unreachable, and it must apply in digest mode exactly as in timing mode -- this pins that it is
    shared code and not restated.
    """
    warm = _dreply(CAND, _D_SAME, cache_enabled="1")
    v, counts = compare_digests(
        BASE, CAND, [("--N", "2048")], launch=_dlauncher({"main": _D_SAME, "ours": warm})
    )
    assert v[0].identical is None and counts["not_run"] == 1
    assert "CPO_CACHE_ENABLED" in v[0].note and "cache_flag" in v[0].note


def test_an_ok_reply_carrying_no_digest_is_refused():
    """A child that reported success while digesting NOTHING must not reach the comparison.

    Found by a control, not by design: crippling `_verify`'s digest requirement left every test
    green, because every other fixture reaches the refusal through ``ok=False``. This is the one
    shape that gets past that half -- and it is the dangerous one, since an empty digest compares
    EQUAL to another empty digest and reports byte identity for two cells that emitted nothing.
    """
    for empty in (None, {}):
        v, counts = compare_digests(
            BASE,
            CAND,
            [("--N", "2048")],
            launch=_dlauncher({"main": _D_SAME, "ours": _dreply(CAND, empty)}),
        )
        assert v[0].identical is None, f"digest={empty!r} must not be compared"
        assert counts["not_run"] == 1

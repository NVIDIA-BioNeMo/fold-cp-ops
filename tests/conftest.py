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

# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.
# This file has been modified by NVIDIA CORPORATION & AFFILIATES.
"""pytest configuration for fold-cp-ops kernel tests.

Supports:
  --compile-only    Compile all kernels (populating .o cache), skip actual execution.
                    Uses FakeTensorMode (no GPU memory) so you can use many xdist workers.
                    Works without a GPU if CPO_ARCH and CUTE_DSL_ARCH are set.

Two-pass workflow (after changing kernel source):
  pytest tests/test_softmax.py --compile-only -n 64   # parallel compile, no GPU memory
  pytest tests/test_softmax.py                         # instant .o loads

CPU-only compilation (no GPU needed):
  CPO_ARCH=90 CUTE_DSL_ARCH=sm_90a pytest tests/ --compile-only -n 64

Single-pass workflow (cache already warm):
  pytest tests/test_softmax.py                         # all .o cache hits

Multi-GPU with xdist:
  pytest tests/ -n 4                                   # workers round-robin across GPUs
"""

import os
import shutil
import subprocess
import json
import time
import logging
import tempfile
from pathlib import Path
from getpass import getuser

import pytest


_compile_only = False
_fake_mode = None
#: The throwaway autotune result-cache directory this session created, or None when the operator
#: pinned ``CPO_AUTOTUNE_CACHE_DIR`` themselves. Removed at unconfigure; see `pytest_configure`.
_autotune_cache_dir = None
#: Set by :func:`_write_and_check_coverage_ledger` when strict mode found an unreached facet. Applied
#: by :func:`pytest_sessionfinish`, which is the only hook late enough to hold the exit status and
#: early enough for pytest to use it.
_COVERAGE_STRICT_FAILED = False
#: Lines the ledger check produced, printed by :func:`pytest_terminal_summary`. Buffered rather than
#: printed directly because the check must run in ``pytest_sessionfinish`` (the only hook whose
#: verdict pytest uses for the exit status) and the terminal reporter's own sessionfinish is a
#: HOOKWRAPPER -- so terminal_summary runs AFTER every sessionfinish impl, not before.
_COVERAGE_LINES: list = []


def pytest_addoption(parser):
    parser.addoption(
        "--compile-only",
        action="store_true",
        default=False,
        help="Compile all kernels and export .o cache, skip actual kernel execution. "
        "Use with -n N (pytest-xdist) for parallel compilation.",
    )


def _get_gpu_ids():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return [g.strip() for g in visible.split(",")]
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip().splitlines()
    except (FileNotFoundError,):
        pass
    logging.warning("Failed to get gpu ids, use default '0'")
    return ["0"]


def _setup_worker_logging(worker_id, tmp):
    """Configure per-worker file logging for easier debugging of parallel runs."""
    log_file = tmp / f"tests_{worker_id}.log"
    handler = logging.FileHandler(log_file, mode="w")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    logging.info("Worker %s logging to %s", worker_id, log_file)


def pytest_configure(config):
    global _compile_only, _fake_mode

    try:
        _compile_only = config.getoption("--compile-only", default=False)
    except (ValueError, AttributeError):
        _compile_only = False

    # Assign GPUs to xdist workers round-robin (skip for CPU-only compile).
    #
    # SINGLE-DEVICE SESSIONS ONLY. This hook is a device ASSIGNER: it hands each xdist worker its
    # own GPU through CUDA_VISIBLE_DEVICES. A torchrun session is ALSO a device assigner -- it maps
    # LOCAL_RANK to a device -- and two of them in one process tree do not compose. Each would
    # believe it owns the mapping, so ranks would collide onto one GPU (an OOM, or worse a silently
    # shared device that corrupts every timing) instead of failing somewhere findable.
    #
    # The distributed tests therefore run WITHOUT `-n`: `torchrun --nproc_per_node=N -m pytest`
    # gives one process per rank already, which is the same parallelism by a different owner.
    # Refused loudly rather than silently skipped, because a silent skip leaves every xdist worker
    # on device 0 under torchrun -- the exact resource conflict, minus the diagnostic.
    worker_id = os.environ.get("PYTEST_XDIST_WORKER")
    if worker_id and any(os.environ.get(v) for v in ("LOCAL_RANK", "RANK", "WORLD_SIZE")):
        raise pytest.UsageError(
            "pytest-xdist (-n) cannot be combined with a torchrun session: both assign GPUs, so "
            "the two mappings collide onto one device. Distributed tests get their parallelism "
            "from torchrun itself -- run `torchrun --nproc_per_node=N -m pytest <file>` with NO "
            "-n. Use -n only for single-device tests (tests/kernels, tests/_internal, ...)."
        )
    if worker_id and not (_compile_only and not _has_gpu()):
        tmp = Path(tempfile.gettempdir()) / getuser() / "fold_cp_ops_tests"
        tmp.mkdir(parents=True, exist_ok=True)
        worker_num = int(worker_id.replace("gw", ""))
        cached_gpu_ids = tmp / "gpu_ids.json"
        if worker_num == 0:
            gpu_ids = _get_gpu_ids()
            with cached_gpu_ids.open(mode="w") as f:
                json.dump(gpu_ids, f)
        else:
            while not cached_gpu_ids.exists():
                time.sleep(0.1)
            with cached_gpu_ids.open() as f:
                gpu_ids = json.load(f)
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids[worker_num % len(gpu_ids)]
        _setup_worker_logging(worker_id, tmp)

    # Give this session its own autotune RESULT cache, unless the operator named one.
    #
    # The result cache persists sweep outcomes across processes and defaults ON, matching `main`.
    # That is right for production and wrong for a test suite in both directions: a run would read
    # a developer's real timings (so what the suite exercises depends on what they ran last), and it
    # would WRITE fake ones back (`tests/_internal/autotune/test_tuner.py` builds kernels whose
    # candidates cost 1.0 and 2.0 by construction). Measured, not assumed: that file passes 19/19
    # into a fresh directory and fails 3 on an immediate re-run against the same one, because its
    # fake kernels share a name, a shape and a candidate set, which is the whole key.
    #
    # Per PROCESS rather than per user, so an xdist worker cannot be handed a sibling's entry.
    #
    # AND THE MARKER IS WHAT MAKES THAT TRUE UNDER XDIST. A worker INHERITS the controller's
    # environment, so `CPO_AUTOTUNE_CACHE_DIR` is already set by the time the worker reaches this
    # hook -- the bare "is it set?" test then reads the controller's own directory as an operator
    # pin and every worker shares one cache, which is the opposite of what the line above claims.
    #
    # Measured 2026-08-31: `tests/_internal tests/testing tests/scripts -n 8` failed
    # `test_a_failing_candidate_is_dropped_rather_than_scored` and
    # `test_every_candidate_failing_is_a_loud_error` on 3 of 3 runs, while the same scope SERIAL
    # passed 1266/1266 and the tuner file alone under `-n 8` passed 19/19. The tell was in the
    # assertion: `last_timings` read `{tile=32: 2.0, tile=64: 1.0}` -- the table belonging to
    # `test_the_result_is_memoized_per_request_shape`, not to the failing test. A sibling worker had
    # written it, and the failing tests loaded it and never swept, so their injected failures never
    # fired.
    #
    # The marker distinguishes "we made this directory" from "the operator pinned one", which a
    # path alone cannot: seeing OUR marker, a worker mints its own; seeing none, it respects the pin.
    #
    # THE MARKER CARRIES THE DIRECTORY, NOT THE STRING "1", and that is the whole subtlety. Every
    # descendant inherits this environment -- an xdist worker AND any subprocess a test spawns --
    # so a bare "1" cannot tell them apart, and the two want OPPOSITE answers: a worker must mint
    # its own cache, while a child whose caller pinned CPO_AUTOTUNE_CACHE_DIR must be left alone.
    # Comparing the marker to the CURRENT value separates them, because only the pinning caller
    # changes one without the other:
    #
    #   worker of ours   DIR == marker   -> ours     -> mint a fresh one (the xdist property above)
    #   caller pinned    DIR != marker   -> a pin    -> respect it
    #   nothing set      no DIR          -> fresh    -> mint
    #
    # Measured before the fix: `tests/test_conftest.py::test_the_session_gets_its_own_autotune_
    # result_cache` failed on every run and on clean `main` -- the child inherited "1", read itself
    # as ours, and overrode a pin the test had just set. A test asserting the pin is respected
    # cannot pass while the marker is a constant.
    global _autotune_cache_dir
    _pinned = os.environ.get("CPO_AUTOTUNE_CACHE_DIR", "").strip()
    _ours = bool(_pinned) and os.environ.get("CPO_AUTOTUNE_CACHE_IS_OURS") == _pinned
    if _ours or not _pinned:
        _autotune_cache_dir = tempfile.mkdtemp(prefix="fold_cp_ops_autotune_")
        os.environ["CPO_AUTOTUNE_CACHE_DIR"] = _autotune_cache_dir
        os.environ["CPO_AUTOTUNE_CACHE_IS_OURS"] = _autotune_cache_dir

    # Arm the numeric guard's runtime half for the whole session. Done here rather than in a
    # fixture so it is in place before the first test body runs and before any module-scoped
    # fixture computes a reference -- and so a session that collects nothing still has it, which is
    # what makes "was it disarmed" a question with a definite answer.
    from fold_cp_ops.testing.numeric_guard import install_tripwire

    install_tripwire()

    # Same reasoning for the collective guard: a bare pytest.skip under a LIVE process group is a
    # divergent-rank deadlock, and only at call time is "is a group initialized" knowable. Armed for
    # the whole session, including sessions that collect no distributed tests, so that "was it
    # disarmed" stays a question with a definite answer.
    from fold_cp_ops.testing.collective_guard import install_skip_tripwire

    install_skip_tripwire()

    if _compile_only:
        import torch
        from torch._subclasses.fake_tensor import FakeTensorMode
        import fold_cp_ops._internal.cache_utils

        fold_cp_ops._internal.cache_utils.COMPILE_ONLY = True
        if torch.cuda.is_available():
            torch.cuda.init()
        _fake_mode = FakeTensorMode()
        _fake_mode.__enter__()


def _has_gpu():
    """Check for GPU without initializing CUDA."""
    import torch

    return torch.cuda.is_available()


def pytest_unconfigure(config):
    global _fake_mode, _autotune_cache_dir
    if _fake_mode is not None:
        _fake_mode.__exit__(None, None, None)
        _fake_mode = None
    if _autotune_cache_dir is not None:
        # Best-effort: a leftover directory is harmless (the next session makes its own), but
        # leaking one per run on a shared box is not.
        shutil.rmtree(_autotune_cache_dir, ignore_errors=True)
        # The marker AND the directory both go, together. Leaving the marker behind would make a
        # LATER in-process session mint a throwaway cache over an operator's pin -- the failure
        # this whole block exists to avoid, inverted. Leaving CPO_AUTOTUNE_CACHE_DIR behind is the
        # other half and was previously missed: a later session would then see a set variable with
        # no matching marker, read it as an operator pin, and use the directory just deleted above.
        # Only cleared when it still names OUR directory, so a value changed mid-session survives.
        if os.environ.get("CPO_AUTOTUNE_CACHE_DIR") == _autotune_cache_dir:
            os.environ.pop("CPO_AUTOTUNE_CACHE_DIR", None)
        os.environ.pop("CPO_AUTOTUNE_CACHE_IS_OURS", None)
        _autotune_cache_dir = None


def pytest_collection_finish(session):
    """Print a summary of collected tests grouped by file and function."""
    if not session.items:
        return
    from collections import defaultdict

    counts = defaultdict(lambda: defaultdict(int))
    for item in session.items:
        file_name = item.location[0]
        func_name = item.originalname if hasattr(item, "originalname") else item.name
        counts[file_name][func_name] += 1
    summary = {f: dict(funcs) for f, funcs in sorted(counts.items())}
    total = len(session.items)
    session.config.pluginmanager.get_plugin("terminalreporter").write_line(
        f"Collected {total} tests: {json.dumps(summary, indent=2)}"
    )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_setup(item):
    """Fail matrix violators up front; in --compile-only mode, swallow setup errors.

    The ``matrix_violation`` marker is attached at collection (see
    :func:`pytest_collection_modifyitems`). Failing here rather than there is what keeps the blast
    radius to the offending module instead of the whole session.
    """
    violation = item.get_closest_marker("matrix_violation")
    if violation is not None:
        pytest.fail(
            "kernel test matrix violation (see CLAUDE.md, and "
            "fold_cp_ops/testing/kernel_matrix.py):\n  - " + violation.args[0],
            pytrace=False,
        )
    numeric = item.get_closest_marker("numeric_violation")
    if numeric is not None:
        pytest.fail(
            "numeric comparison violation (see fold_cp_ops/testing/numeric_guard.py):\n  - "
            + numeric.args[0],
            pytrace=False,
        )
    collective = item.get_closest_marker("collective_violation")
    if collective is not None:
        pytest.fail(
            "collective symmetry violation (see fold_cp_ops/testing/collective_guard.py):\n  - "
            + collective.args[0],
            pytrace=False,
        )
    template = item.get_closest_marker("template_param_violation")
    if template is not None:
        pytest.fail(
            "template-parameter violation (see fold_cp_ops/testing/template_param_guard.py):\n  - "
            + template.args[0],
            pytrace=False,
        )
    if not _compile_only:
        yield
        return
    outcome = yield
    if outcome.excinfo is not None:
        outcome.force_result(None)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    """Enforce the numeric guard's runtime halves around the call; in --compile-only, swallow errors.

    Written with pluggy's ``wrapper=True`` rather than the older ``hookwrapper=True`` **because it
    raises**: an exception thrown from an old-style wrapper after its ``yield`` is reported as a
    ``PluggyTeardownRaisedWarning`` attached to a passing-looking test, which is precisely the wrong
    shape for a guard whose whole job is to fail legibly.

    Two checks, both **after** the test body and both skipped when the body already failed, so a
    real failure is never displaced by a meta-failure about how it was asserted:

    * **Coverage.** A test marked ``numeric_required`` (see
      :func:`pytest_collection_modifyitems`) must have run at least one assertion from
      ``fold_cp_ops.testing.numerics``. This is the layer that catches a pooled comparison by its
      ABSENCE of a sanctioned one, which is how the original five slipped past every other check --
      their spelling was novel, so no blacklist would have found them.
    * **Tripwire integrity.** If the test put ``torch.allclose`` back, it fails *here*, naming
      itself. Without this the guard would silently switch off for the rest of the session.

    The record is reset before the body rather than after, so an assertion made in a fixture's setup
    still counts and one test's assertions can never satisfy the next test's gate.
    """
    from fold_cp_ops.testing.collective_guard import (
        drain_violations,
        reset_violations,
        skip_tripwire_problem,
    )
    from fold_cp_ops.testing.numeric_guard import (
        assertions_recorded,
        reset_assertions,
        tripwire_problem,
    )

    reset_assertions()
    reset_violations()
    try:
        result = yield
    except BaseException:
        if _compile_only:
            return None
        # A bare pytest.skip RAISES, so the collective tripwire's finding arrives on THIS path and
        # nowhere else -- checking only after a normal return would leave the guard silent for
        # exactly the call it exists to catch. A divergent skip must FAIL rather than skip: reported
        # as a skip it is indistinguishable from a legitimate one, which is how the hang it causes
        # goes unattributed.
        if not _compile_only:
            _fail_on_collective_violations(drain_violations())
        raise
    if not _compile_only:
        _fail_on_collective_violations(drain_violations())
        if item.get_closest_marker("numeric_required") and not assertions_recorded():
            pytest.fail(
                "numeric coverage violation (see fold_cp_ops/testing/numeric_guard.py): this "
                "test draws shapes from a KernelMatrix -- so it launches a kernel and produces "
                "a value -- but made NO assertion from fold_cp_ops.testing.numerics. Compare "
                "the result with assert_elementwise / assert_bitwise / assert_gemm_exact / "
                'assert_gemm_close, or declare @numeric_exempt("why no comparison applies").',
                pytrace=False,
            )
        problem = tripwire_problem()
        if problem:
            pytest.fail(f"numeric guard: {problem}", pytrace=False)
        problem = skip_tripwire_problem()
        if problem:
            pytest.fail(f"collective guard: {problem}", pytrace=False)
    return result


def _fail_on_collective_violations(violations) -> None:
    """Turn the collective tripwire's findings into a test failure, or do nothing.

    Args:
        violations: What :func:`collective_guard.drain_violations` returned for the test that just
            ran -- already drained, so calling this twice for one test cannot double-report.

    Returns:
        None when there were none.

    Raises:
        ``Failed``, via `pytest.fail`, naming every violation. Raising from a helper rather than
        inline keeps the two call sites (normal return and the exception path a skip arrives on)
        from drifting apart.
    """
    if not violations:
        return
    pytest.fail(
        "collective symmetry violation (see fold_cp_ops/testing/collective_guard.py):\n  - "
        + "\n  - ".join(violations),
        pytrace=False,
    )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item, nextitem):
    """In --compile-only mode, swallow teardown errors."""
    if not _compile_only:
        yield
        return
    outcome = yield
    if outcome.excinfo is not None:
        outcome.force_result(None)


# ── the kernel-matrix audit ───────────────────────────────────────────────────────────────────
def pytest_collection_modifyitems(session, config, items):
    """Fail every test in a module that breaks the kernel-matrix rules -- and only those tests.

    **Why a collection hook and not a test.** A meta-test can be deselected: ``pytest
    tests/kernels/test_foo.py`` never collects it, so a rogue module would run unchallenged. This
    hook fires whenever such a module is collected at all, by any invocation, so there is no way to
    run those tests without it.

    **Nothing here aborts collection.** Violating modules are collected like any other; the audit
    only attaches a marker, and the failure happens per test at setup. An earlier version raised
    ``pytest.UsageError`` here, and one non-conforming file turned ``pytest tests/`` into "no tests
    ran" -- destroying the signal from every unrelated test and blocking anyone part-way through
    adding a kernel. Scope is one module: its own tests fail with the reason, everything else runs,
    exit code still non-zero. A rogue module cannot pass, and cannot take the suite hostage.

    The audit is also wrapped so it cannot itself abort the run: any unexpected exception (an
    unparsable module, an import that explodes) becomes a reported problem for that module rather
    than a collection error for the session.

    Checks are delegated to ``fold_cp_ops.testing.kernel_matrix.audit_test_module`` so that this and
    ``tests/testing/test_kernel_matrix.py`` can never disagree about the rules.

    Args:
        session: The pytest session (unused; part of the hook signature).
        config: The pytest config (unused; part of the hook signature).
        items: Collected test items, modified in place -- offending ones gain a
            ``matrix_violation`` marker that :func:`pytest_runtest_setup` turns into a failure.
            Each module is audited once, so the cost is one AST parse per file, not per test.

    Returns:
        None.
    """
    from fold_cp_ops.testing.kernel_matrix import (
        audit_test_module,
        matrix_scope,
        unsupported_cell_problem,
    )
    from fold_cp_ops.testing.collective_guard import audit_collective_module, collective_scope
    from fold_cp_ops.testing.numeric_guard import audit_numeric_module, numeric_scope

    audited: dict = {}
    for item in items:
        path = Path(str(getattr(item, "path", "") or ""))
        # tests/distributed is in scope too: a mesh is an axis like any other, and `unsupported=`
        # is as mandatory there as anywhere. The one thing that does NOT transfer is coverage --
        # one launch fixes one WORLD_SIZE, so see _write_and_check_coverage_ledger.
        #
        # The predicate lives in kernel_matrix so this hook and tests/testing/test_kernel_matrix.py
        # cannot disagree about WHERE the rules apply, exactly as they already share the rules
        # themselves. It matches on path PARTS, which is what puts tests/distributed/kernels/ and
        # tests/distributed/workflows/ in scope by decision rather than by a leaf-name coincidence.
        if not matrix_scope(path):
            continue
        if path not in audited:
            try:
                audited[path] = audit_test_module(path, getattr(item, "module", None))
            except Exception as exc:  # never let the auditor take the session down
                audited[path] = [f"{path.name}: the matrix audit itself failed: {exc!r}"]
        if audited[path]:
            item.add_marker(pytest.mark.matrix_violation("\n  - ".join(audited[path])))

        # PER-ITEM, and that is the point. Everything above audits a MODULE; this checks the cell
        # this item will actually run. `KernelMatrix.parametrize` can only evaluate a region whose
        # axes ONE call parametrizes, so stacked decorators -- @M.parametrize("N") over
        # @M.parametrize("mesh") -- emit every pair while a region reading both is skipped by each
        # call in turn. Measured: four cells declared unsupported sat in an accepting test's grid.
        # `item.callspec.params` is the bound cell rather than a predicted cross product, so this
        # also catches routes nobody has thought of yet (indirect parametrize, a fixture supplying
        # an axis value). Same failure discipline as the module audit: mark, never abort.
        callspec = getattr(item, "callspec", None)
        if callspec is not None and getattr(item, "module", None) is not None:
            try:
                problem = unsupported_cell_problem(item.module, callspec.params)
            except Exception as exc:  # never let the guard take the session down
                problem = f"{path.name}: the unsupported-cell check itself failed: {exc!r}"
            if problem:
                item.add_marker(pytest.mark.matrix_violation(problem))

    # ── the numeric guard's static half ───────────────────────────────────────────────────────
    # Scoped to every test module, not just tests/kernels + tests/perf: a pooled comparison is
    # wrong wherever it is written, and the module that carried the original defect was at the
    # tests/ root. Same failure discipline as above -- mark the offending module's items, never
    # abort the session, and treat an exception in the auditor as that module's problem.
    guarded: dict = {}
    for item in items:
        path = Path(str(getattr(item, "path", "") or ""))
        if not path.name.startswith("test_") or path.suffix != ".py":
            continue
        if path not in guarded:
            try:
                guarded[path] = (audit_numeric_module(path), numeric_scope(path))
            except Exception as exc:  # never let the auditor take the session down
                guarded[path] = ([f"{path.name}: the numeric audit itself failed: {exc!r}"], set())
        problems, scope = guarded[path]
        if problems:
            item.add_marker(pytest.mark.numeric_violation("\n  - ".join(problems)))
        if getattr(item, "originalname", item.name) in scope:
            item.add_marker(pytest.mark.numeric_required())

    # ── the collective guard's static half ────────────────────────────────────────────────────
    # Scoped by DIRECTORY to tests/distributed/** at any depth, so the A2A kernel tests are covered
    # by the subdirectory they land in rather than by anyone remembering to opt them in. This half
    # sees what the runtime tripwire cannot: a divergent skip on a branch THIS launch did not take,
    # which is a landmine for the next mesh and would otherwise never fire.
    collective: dict = {}
    for item in items:
        path = Path(str(getattr(item, "path", "") or ""))
        if not path.name.startswith("test_") or path.suffix != ".py":
            continue
        if not collective_scope(path):
            continue
        if path not in collective:
            try:
                collective[path] = audit_collective_module(path, getattr(item, "module", None))
            except Exception as exc:  # never let the auditor take the session down
                collective[path] = [f"{path.name}: the collective audit itself failed: {exc!r}"]
        if collective[path]:
            item.add_marker(pytest.mark.collective_violation("\n  - ".join(collective[path])))

    # ── the template-parameter guard ──────────────────────────────────────────────────────────
    # Scoped to every test module, because a functor can be constructed anywhere and the defect is
    # not a property of the directory. It is cheap: the declared names come from the classes the
    # module ALREADY imported, so a module importing no functor is passed without its source being
    # walked. Same failure discipline as the three above -- mark the offending module's items and
    # let setup fail them; never abort collection, which is the mode that turned `pytest tests/`
    # into "no tests ran" the one time an audit raised here.
    from fold_cp_ops.testing.template_param_guard import audit_template_param_module

    templates: dict = {}
    for item in items:
        path = Path(str(getattr(item, "path", "") or ""))
        if not path.name.startswith("test_") or path.suffix != ".py":
            continue
        if path not in templates:
            try:
                templates[path] = audit_template_param_module(path, getattr(item, "module", None))
            except Exception as exc:  # never let the auditor take the session down
                templates[path] = [f"{path.name}: the template-param audit itself failed: {exc!r}"]
        if templates[path]:
            item.add_marker(pytest.mark.template_param_violation("\n  - ".join(templates[path])))


# ── run report: every session writes one, so a failure never has to be reproduced to be read ──
#: Where the run report is written, relative to pytest's own base temp dir. A copy also lands at
#: `_REPORT_LATEST` so the newest run is findable without knowing the numbered directory.
_REPORT_NAME = "run_report.txt"


def _report_paths(config):
    """The two destinations for this session's report.

    Args:
        config: The pytest config, used to reach the base temp dir pytest already made.

    Returns:
        ``(per_run_path, latest_path)``. The first is inside pytest's numbered basetemp, so runs do
        not overwrite each other; the second is a fixed name beside it, so "the last run" is one
        path rather than a directory listing sorted by mtime. Both are returned even when the base
        temp dir cannot be resolved, in which case they are None.
    """
    try:
        base = config._tmp_path_factory.getbasetemp()
    except AttributeError:  # pragma: no cover - only if pytest changes its internals
        return None, None
    return base / _REPORT_NAME, base.parent / f"fold_cp_ops_{_REPORT_NAME}"


#: Filename of the append-as-you-go progress record, written beside the per-run report inside
#: pytest's numbered basetemp. Deliberately NOT the same file as `_REPORT_NAME` -- see
#: :func:`pytest_report_teststatus`.
_PROGRESS_NAME = "run_progress.txt"


@pytest.hookimpl(tryfirst=True)
def pytest_report_teststatus(report, config):
    """Append one flushed line per test phase, so a session that DIES still leaves a record.

    Purpose
        :func:`pytest_terminal_summary` runs at session end, so a session that never reaches the end
        -- a core dump, a SIGKILL, a hung run you interrupt -- writes no report at all, and the file
        on disk is the PREVIOUS run's. That is exactly inverted for the failure you least want to
        reproduce. This hook puts the bytes on disk as each phase finishes, so the record survives
        the process by construction rather than by the process cooperating.

    Semantics
        Called once per report phase (setup / call / teardown). Writes ``<phase> <outcome> <nodeid>``
        for every phase EXCEPT a passing teardown, which carries no information a passing call did
        not already carry. Emitting the ``setup`` line is the point of the design: it lands BEFORE
        the test body runs, so after a hard crash the LAST line of the file names the test that was
        executing -- not the last one that finished, which is what forced a manual reconstruction
        from the collection manifest and the progress-dot count last time this happened.

        Opened in append mode and closed per line, so the bytes reach the OS on every write. A
        cached file handle, or any buffering, would reintroduce the exact failure being fixed: the
        interesting tail is the part still sitting in a userspace buffer when the process dies.
        ``close()`` is sufficient -- the data is in the page cache and outlives the process; only a
        machine crash could lose it, which is not the case being defended against.

        **Written to its own file, not appended into the run report.** Keeping one file
        append-only and the other written-once means neither has to know about the other: the
        summary writer is untouched, and no reader ever meets a half-written report and has to work
        out which half it is looking at. The file lives in pytest's numbered per-run basetemp, so it
        needs no truncation at session start (the directory is new every run) and no second hook.
        Reach the newest one at ``<basetemp>/../pytest-current/run_progress.txt``; the exact path is
        printed in the terminal summary.

    Why this hook and not ``pytest_runtest_logreport``
        That one is the semantically obvious choice and receives only ``report`` -- no ``config``,
        hence no way to reach the basetemp without keeping module state, which is the thing this
        design is trying not to have. ``pytest_report_teststatus`` is the only per-report hook that
        takes both. It is a ``firstresult`` hook, so ``tryfirst=True`` is REQUIRED, not decorative:
        without it a plugin that returns a status ahead of this one would stop the call and silently
        disable the record -- the failure mode would be an empty file exactly when it is needed.

    Args:
        report: The phase report just completed. Read-only here.
        config: The pytest config, for the base temp dir.

    Returns:
        Always ``None``, which is what hands the status character back to the terminal reporter.
        Returning anything else would change pytest's own output. Write errors are swallowed: a
        progress record is diagnostic, and must never be the reason a run fails.
    """
    if report.when == "teardown" and report.outcome == "passed":
        return None
    per_run, _ = _report_paths(config)
    if per_run is None:
        return None
    try:
        with (per_run.parent / _PROGRESS_NAME).open("a") as fh:
            fh.write(f"{report.when:9s} {report.outcome:8s} {report.nodeid}\n")
    except OSError:  # pragma: no cover - a read-only tmp must not fail the run
        pass
    return None


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Write every failure, error and skip of this session to a file, unconditionally.

    Purpose
        So that "which test failed?" is answerable from the last run instead of by reproducing it.
        A perf suite takes ~5 minutes and some of its failures are intermittent, so a lost summary
        can cost more than the run did -- and it is easy to lose: piping pytest through
        ``grep``/``head`` to extract one thing silently truncates everything else, which is exactly
        how a 1-in-N perf failure went unidentified here and had to be re-measured.

    Semantics
        Runs on EVERY session, with no flag to remember, because a report you have to opt into is
        one you do not have when you need it. Written into pytest's own base temp dir (numbered per
        run, so nothing is overwritten) plus a fixed ``fold_cp_ops_run_report.txt`` beside it for
        the latest. The path is printed in the terminal summary so it is discoverable without
        knowing this hook exists.

        Records the full ``longrepr`` for failures and errors -- the assertion text, which is where
        a perf gate puts its measured-vs-pinned numbers and its band -- and one line per skip with
        its reason, since "what was skipped and why" is the other question that otherwise needs a
        re-run.

    Args:
        terminalreporter: The reporter holding this session's collected reports.
        exitstatus: The session's exit status, recorded in the header.
        config: The pytest config, for the base temp dir.

    Returns:
        None; writes the files and prints their location.
    """
    per_run, latest = _report_paths(config)
    if per_run is None:
        return
    for line in _COVERAGE_LINES:
        terminalreporter.write_line(line)
    stats = terminalreporter.stats
    lines = [
        "# fold-cp-ops pytest run report",
        f"# exitstatus : {exitstatus}",
        f"# invocation : {' '.join(config.invocation_params.args)}",
        f"# rootdir    : {config.rootpath}",
        "",
        "## counts",
    ]
    for key in ("passed", "failed", "error", "skipped", "xfailed", "xpassed"):
        if stats.get(key):
            lines.append(f"  {key:8s} {len(stats[key])}")

    for key, title in (("failed", "FAILURES"), ("error", "ERRORS")):
        reports = stats.get(key, [])
        if not reports:
            continue
        lines += ["", f"## {title} ({len(reports)})"]
        for rep in reports:
            lines += ["", f"--- {rep.nodeid}", str(rep.longrepr)]

    skipped = stats.get("skipped", [])
    if skipped:
        lines += ["", f"## SKIPPED ({len(skipped)})"]
        for rep in skipped:
            reason = rep.longrepr[2] if isinstance(rep.longrepr, tuple) else rep.longrepr
            lines.append(f"  {rep.nodeid}: {reason}")

    text = "\n".join(lines) + "\n"
    for path in (per_run, latest):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        except OSError:  # pragma: no cover - a read-only tmp is not worth failing a run over
            pass
    terminalreporter.write_line(f"run report: {latest}")
    progress = per_run.parent / _PROGRESS_NAME
    terminalreporter.write_line(f"run progress (live, survives a crash): {progress}")


def _write_and_check_coverage_ledger(config, default_dir):
    """Persist what this run EXERCISED, summarize it, and under strict mode fail on a hole.

    Purpose
        A mesh axis is the one axis a single launch cannot cover: one launch fixes one ``WORLD_SIZE``
        and skips every spec needing a different rank count. The static matrix check sees all specs
        parametrized and reports full coverage, which is a false claim; failing the run for its own
        world size would be the opposite error. So each run records what it did, and the union across
        runs is where "is the pool covered" is answerable.

    Semantics
        Always writes and always summarizes. **Fails only under** ``CPO_DIST_COVERAGE_STRICT``, and
        only on a facet that every run in the ledger directory skipped -- so an ordinary
        ``torchrun --nproc_per_node=2`` is never failed for not having built a 16-rank mesh, and the
        check bites exactly where a sweep is being claimed.

        The strict failure is applied by raising the session's exit status rather than by failing a
        test: there is no test that owns "the set of launches", and inventing one would make it fail
        for every launch that is not the last.

    Args:
        config: The pytest config, for a discriminator that keeps concurrent writers apart.
        default_dir: Where to write when ``CPO_DIST_LEDGER_DIR`` is unset -- the parent of pytest's
            numbered basetemp, so ledgers from successive runs land in ONE directory and can union.
            Writing inside the numbered dir would give every run its own directory and no union.

    Returns:
        None. A missing directory, an unwritable path or a corrupt sibling ledger degrades to
        silence: a coverage diagnostic must never be the thing that takes a session down.
    """
    from fold_cp_ops.testing import coverage_ledger as cl

    directory = cl.ledger_dir(default_dir)
    # os.getpid() alone is NOT unique here (measured: backgrounded processes all report pid 25), so
    # the discriminator carries the rank, the xdist worker id and the basetemp's own unique name.
    disc = "-".join(
        str(x)
        for x in (
            os.environ.get("RANK", "0"),
            os.environ.get("PYTEST_XDIST_WORKER", "m"),
            Path(str(getattr(config, "rootpath", "r"))).name,
            os.getpid(),
            Path(str(default_dir or "d")).name,
        )
    )
    cl.write_ledger(directory, disc)

    merged = cl.union_ledgers(directory)
    lines = list(cl.format_summary(merged))
    if not lines:
        return
    _COVERAGE_LINES.append("")
    _COVERAGE_LINES.append(f"mesh coverage ledger: {directory}")
    _COVERAGE_LINES.extend(f"  {line}" for line in lines)

    problems = []
    for matrix in _declared_matrices():
        problems.extend(cl.unreached_facets(matrix, merged))
    if not problems:
        return
    if not cl.strict_enabled():
        _COVERAGE_LINES.append(
            f"  {len(problems)} facet(s) not exercised by any run here; set "
            f"{cl.STRICT_ENV}=1 to make that a failure once the sweep is complete"
        )
        return
    _COVERAGE_LINES.extend(f"  COVERAGE HOLE: {p}" for p in problems)
    _COVERAGE_LINES.append(
        f"  {cl.STRICT_ENV} is set, so these are fatal. Run the missing launches into the same "
        f"ledger directory, or waive the facet with a written reason."
    )
    # The exit status is NOT set here. pytest_terminal_summary runs too late to influence it --
    # measured: setting terminalreporter._session.exitstatus here printed every hole and still
    # exited 0. pytest_sessionfinish is the hook whose return value pytest actually uses, so the
    # verdict is stashed and applied there.
    global _COVERAGE_STRICT_FAILED
    _COVERAGE_STRICT_FAILED = True


def _declared_matrices():
    """Every `KernelMatrix` declared by a distributed test module that was imported this session.

    Semantics
        Read off ``sys.modules`` rather than by scanning source, so a matrix is found exactly when
        its module was collected -- which is the same condition under which its values could have
        been exercised. A source scan would report holes for modules this session never ran.

    Returns:
        The matrices, deduplicated by identity. Empty when no distributed module was imported, which
        is why the caller returns early on an empty summary.
    """
    import sys

    from fold_cp_ops.testing.kernel_matrix import KernelMatrix

    seen, out = set(), []
    for name, mod in list(sys.modules.items()):
        if not name.startswith("tests.distributed"):
            continue
        for attr in vars(mod).values():
            if isinstance(attr, KernelMatrix) and id(attr) not in seen:
                seen.add(id(attr))
                out.append(attr)
    return out


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    """Apply the strict coverage verdict to the session's exit status.

    Purpose
        A mesh facet no launch ever exercised is not a test failure -- no single test owns "the set
        of launches" -- so it has to reach the exit status directly. `pytest_terminal_summary` is
        where the holes are computed and printed, but it runs too late to change the status:
        measured, setting it there printed every hole and still exited 0.

    Semantics
        Only ever RAISES the status, never lowers it, so a session that already failed keeps its own
        exit code and a coverage hole cannot mask a real failure.

    Args:
        session: The pytest session whose ``exitstatus`` is the process's return code.
        exitstatus: The status pytest arrived at; left alone unless it is 0.

    Returns:
        None; mutates ``session.exitstatus``.
    """
    try:
        per_run, _latest = _report_paths(session.config)
        if per_run is not None:
            _write_and_check_coverage_ledger(session.config, per_run.parent.parent)
    except Exception:  # noqa: BLE001 - a coverage diagnostic must never take a session down
        pass
    if _COVERAGE_STRICT_FAILED and exitstatus == 0:
        session.exitstatus = 1

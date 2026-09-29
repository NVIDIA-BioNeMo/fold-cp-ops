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

"""pytest fixtures for the single-GPU perf gates under ``tests/perf/``.

Trimmed to what a single-GPU gate needs: the session clock lock and the device fixture. The
upstream file also carried seven torchrun fixtures (``dist_manager``, ``rank``, ``world_size``,
``local_rank``, ``device``, ``device_mesh``, plus mesh-spec parsing); those return with the
distributed perf gates that use them.

NOTE: the sibling helpers (``calibration``, ``pins``, ``_clocklock``) are imported PACKAGE-QUALIFIED
-- ``from tests.perf.pins import ...`` -- not as top-level names. This directory has an
``__init__.py``, so that is simply the correct spelling, and ``pythonpath = ["."]`` in
``pyproject.toml`` puts the repo root on ``sys.path`` for it.

An earlier version instead pushed this directory onto ``sys.path`` from here and imported the bare
names. That worked only while THIS conftest had been loaded, i.e. only when the invocation collected
something under ``tests/perf/``. The cost was silent and specific: ``test_kernel_matrix.py``'s
``test_every_kernel_test_draws_on_the_matrix`` imports every perf module by dotted name to audit it,
so the guard PASSED under ``pytest tests/`` and FAILED with ``ModuleNotFoundError`` under ``pytest
tests/testing/`` -- an enforcement point whose verdict depended on which unrelated directories were
in the same command. It also meant ``pins`` could exist twice, once as ``pins`` and once as
``tests.perf.pins``, with a separate copy of its module state in each.
"""

from __future__ import annotations

import os
import pathlib

import pytest

from tests.perf._clocklock import clock_lock
from tests.wedge_watchdog import install_wedge_watchdog

# Wedge auto-dump: if a run HANGS, dump every thread's traceback after a timeout so the wedge
# self-reports. Pure diagnostic (no behavior change on a passing run); an outer `timeout` still
# kills the process.
#
# This claim used to be FALSE. The previous `faulthandler.dump_traceback_later(t, repeat=True)`
# dumps from a C thread that does NOT hold the GIL, so every tick raced the frame chains of every
# running thread and eventually segfaulted a HEALTHY session -- ~1 in 37 full runs, always landing
# on whatever cell happened to be executing. `install_wedge_watchdog` dumps from a Python thread
# instead, holding the GIL for the walk. See tests/wedge_watchdog.py for the measurement.
install_wedge_watchdog()


#: Seconds of sustained matmul before the first timed cell, in `_gpu_warm`.
#:
#: Raised from ~50 iterations (well under a second). Measured effect across four fresh processes,
#: unlocked: the compute reference's across-process spread fell from 0.71% to 0.29% and LayerNorm's
#: from 0.48% to 0.35%. Small but free, and it removes the "first cell measured on a ramping part"
#: failure mode that this fixture was added for.
#:
#: **It does NOT substitute for the clock lock**, and that was tested rather than assumed: the
#: ``(1e6, 768, 384)`` GEMM is BISTABLE unlocked, reading ~1.09 or ~1.46 ms (2-of-4 each way) with
#: this soak exactly as without it, while locked it reproduces to 0.26%. A soak converges a ramp; it
#: cannot converge a bistable power state.
_WARM_SECONDS = 10.0


def _is_under_torchrun() -> bool:
    """Return True when the process was launched by torchrun.

    Detected from the ``RANK``/``WORLD_SIZE`` pair torchrun always exports; both are required so a
    stray ``RANK`` in an interactive shell does not trigger the multi-rank device pinning below.

    Returns:
        True if both ``RANK`` and ``WORLD_SIZE`` are set in the environment.
    """
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


@pytest.fixture(scope="session", autouse=True)
def clock_locked():
    """Lock GPU graphics+memory clocks to max for the session; reset at teardown.

    Autouse and session-scoped: every perf gate in this directory needs the same clock state, and
    locking per test would add seconds per cell for no benefit. Best-effort by design --
    ``_clocklock`` warns and proceeds UNLOCKED when clock control is denied (the common case
    without root), because an unlocked run with a wider variance band is far more useful than no
    measurement. One locker per node: only ``LOCAL_RANK`` 0 touches the clocks.

    Yields:
        bool -- True if the clocks were actually locked, False if the attempt was denied and the
        session is running unlocked. Tests may read it to widen their tolerance.
    """
    with clock_lock() as locked:
        yield locked


@pytest.fixture(scope="session")
def dev():
    """The CUDA device the single-GPU perf tests run on -- pinned to THIS rank's local GPU.

    MEASUREMENT CORRECTNESS, not cosmetics. Returning
    ``torch.device("cuda", torch.cuda.current_device())`` unconditionally is wrong under torchrun:
    nothing calls ``torch.cuda.set_device`` until a distributed fixture does, and the single-GPU
    perf files do not request one -- so whether their medians were measured one-rank-per-GPU or
    with every rank piled onto GPU 0 depended on whether some other, alphabetically-earlier file
    happened to collect first and leave ``set_device`` behind as a process-wide side effect.

    Measured cost of that accident (venue B, cp=8): run after the distributed files -> 42/42 pass,
    ratios 0.92-1.03; run ALONE -> 10 failed / 31 OOM-skipped, medians inflated 4-8x. Binding to
    ``LOCAL_RANK`` here makes the median independent of collection order.

    Returns:
        A ``torch.device`` for this rank's GPU. Under torchrun it is ``cuda:LOCAL_RANK`` and
        ``set_device`` has been called process-wide; otherwise the current device.

    Raises:
        pytest.skip.Exception: If CUDA is unavailable -- a perf gate has nothing to measure.
    """
    import torch

    if not torch.cuda.is_available():
        pytest.skip("perf-gate benchmark tests require CUDA")
    if _is_under_torchrun():
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
        if local_rank < torch.cuda.device_count():
            torch.cuda.set_device(local_rank)  # process-wide, and now deterministic
            return torch.device("cuda", local_rank)
    return torch.device("cuda", torch.cuda.current_device())


@pytest.fixture(scope="session", autouse=True)
def _gpu_warm(clock_locked, dev):
    """Run throwaway work so the FIRST timed cell is not measured on a ramping GPU.

    MEASUREMENT CORRECTNESS, like ``dev``. Locking the clocks fixes the SM/memory frequency but
    does not bring an idle GPU to a steady state: the first kernel after idle pays power-rail and
    L2 warm-up that later kernels do not. On launch-bound cells -- the small-N ones, ~0.02 ms --
    that lands as a several-percent inflation on whichever cell happens to be collected first,
    which is indistinguishable from a real regression and moves with test ordering.

    Observed before this fixture existed: a full-file run failed exactly its first two cells
    (N=128 and N=256 bf16) while every one of them passed in isolation and on an immediate
    re-run. Widening the tolerance or raising those two pins would have buried a measurement bug
    in the gate's own thresholds; warming the device fixes the measurement instead.

    Args:
        clock_locked: Ordering dependency only -- ensures clocks are pinned BEFORE the warm-up, so
            the warm-up settles the device at the frequency the gate will measure at.
        dev: The device to warm; the same one the gates time on.

    Yields:
        None. Used only for its side effect.
    """
    import time

    import torch

    a = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
    deadline = time.perf_counter() + _WARM_SECONDS
    while time.perf_counter() < deadline:
        for _ in range(20):
            a = torch.mm(a, a.T).to(torch.bfloat16) * 1e-4
        torch.cuda.synchronize(dev)
    del a
    torch.cuda.empty_cache()
    yield


def pytest_collection_modifyitems(session, config, items):
    """Skip the perf gates when the session also collected non-perf tests.

    Purpose
        A pinned median is only meaningful under the conditions it was pinned in. On a box where
        clock locking is DENIED -- ``nvidia-smi -lgc`` returns "The current user does not have
        permission to change clocks", which is the common case outside a dedicated bench host --
        ``_clocklock`` warns and proceeds unlocked, so the SM frequency follows the thermal load.
        Running the correctness suite first puts three minutes of GPU work in front of the gates.

    Semantics
        Observed: under ``pytest tests/`` the tightest cell ``(1001, 2048, 2048, 1, 128, 256)`` read
        0.0271 ms against a 0.0244 pin (+11.5%, band 10%), while under ``pytest tests/perf/`` the
        same cell reads 0.0242-0.0243 across three runs. The kernel is identical.

        **The MECHANISM is not established**, and that is stated rather than guessed at. A controlled
        reproduction -- 1030 correctness tests in-process, then the same measurement -- moved the
        cell only +1.0%, and a 90 s sustained-matmul soak moved it +2.9% with the GPU still at its
        1980 MHz boost clock. So "the correctness suite heats the part" does NOT account for the
        +11.5%; something else in a full session does, and it has not been isolated. What IS
        established is that the two invocations disagree by more than the band, which is enough to
        make the isolated one the only supported way to read these numbers.

        **Calibration does not exempt a gate from this, and used to.** The reasoning was that a gate
        whose pin file carries a spread for every cell divides by a reference measured in the same
        session, so a slow machine cancels. Measured, it does not: the calibrated dual-gated gates
        failed 5 and then 3 cells under `pytest tests/` and passed every one of them in an isolated
        `pytest tests/perf/` minutes later. An in-session reference only cancels a slowdown that
        hits it and the target EQUALLY -- the same reason this repo threw out its single-`matmul`
        drift correction -- and a 2200-test correctness suite ahead of the gate is not that.

        Widening the band is not the alternative. A band wide enough to absorb +11.5% would stop
        catching the regressions these cells exist for -- the front-door work that prompted the
        launch-bound pins to be re-harvested cost ~7%, and would vanish under it. So the gate
        declines to measure rather than measuring badly, and says which command to use.

        ``CPO_PERF_MEASURE=1`` (harvesting) and ``CPO_PERF_ALLOW_SHARED_SESSION=1`` both override,
        the latter for a caller who has locked clocks and knows the session is clean.

    Args:
        session: The pytest session. Unused; part of the hook signature.
        config: The pytest config. Unused.
        items: Every collected item, mutated in place to carry the skip marker.

    Returns:
        None.
    """
    if os.environ.get("CPO_PERF_MEASURE", "0") == "1":
        return
    if os.environ.get("CPO_PERF_ALLOW_SHARED_SESSION", "0") == "1":
        return
    # Only the TIMED gates -- modules named `test_benchmark_perf_*`. `tests/perf/` also holds unit
    # tests for the gates' own machinery (`test_calibration.py`), which measure nothing and must
    # run under `pytest tests/` like any other unit test. Skipping those would remove the check
    # that says the timed cells mean anything, which is the opposite of the intent here.
    # Three-way, and the third case is the one a two-way split got wrong: an item is either an
    # UNCALIBRATED timed cell (skippable), a calibrated cell or a perf unit test (never skipped, but
    # ALSO not evidence of a shared session), or something outside tests/perf/ (the evidence). An
    # earlier version counted the calibrated cells as "other" and so skipped every uncalibrated cell
    # during a plain `pytest tests/perf/` -- 75 of 106, silently.
    perf_dir = pathlib.Path(__file__).parent
    skippable, outside_perf = [], False
    for item in items:
        try:
            path = pathlib.Path(str(item.fspath))
            in_perf = perf_dir in path.parents
            is_timed = in_perf and path.name.startswith("test_benchmark_perf_")
            # There used to be an exemption here: a CALIBRATED module ran anyway, on the grounds
            # that its cells divide by a reference measured in the same session, so a machine
            # running slow cancels instead of reading as a regression. The comment recorded that as
            # verified -- "`pytest tests/` runs those cells right after the correctness suite and
            # they hold".
            #
            # It does not hold. Measured on two separate full `pytest tests/` runs, the calibrated
            # dual-gated gates failed 5 and then 3 cells (`layout`, `blk_k`, `gate3`, `dispatch`);
            # every one of them passed in an isolated `pytest tests/perf/` immediately afterwards.
            # Dividing by an in-session reference cancels a slowdown that hits the reference and the
            # target EQUALLY, and a 2200-test correctness suite does not: it leaves the clocks, the
            # power state and the allocator somewhere the pins were never harvested. That is the
            # same reason the repo forbids a drift correction from a single `torch.matmul`.
            #
            # So calibration buys nothing here and cost every `pytest tests/` a handful of failures
            # that mean nothing -- worse than skipping, because a false failure has to be
            # re-measured before it can be dismissed. The gates measure in an isolated session,
            # which is what CLAUDE.md already told readers they do.
        except (OSError, ValueError):  # pragma: no cover - defensive
            in_perf, is_timed = False, False
        (skippable.append(item) if is_timed else None)
        outside_perf = outside_perf or not in_perf
    if not (skippable and outside_perf):
        return
    skip = pytest.mark.skip(
        reason=(
            "perf gates measure only in an ISOLATED session -- run `pytest tests/perf/`. This "
            "session also collected non-perf tests, and on a box without clock-lock permission "
            "their GPU load shifts the clocks under the gate: measured +11.5% on the tightest "
            "cell, against a 10% band. Set CPO_PERF_ALLOW_SHARED_SESSION=1 to measure anyway."
        )
    )
    for item in skippable:
        item.add_marker(skip)


def _torch():
    """Import torch lazily, so this conftest stays importable on a machine without it."""
    import torch

    return torch


def pytest_collectstart(collector):
    """Refuse a timed perf module that does not read its numbers from its own pin file.

    Purpose
        Four gates had grown four shapes of the same idea -- a 6-tuple holding a float, a 6-tuple
        holding a pair, a 3-tuple, plus loose constants for autotune bands and dispatch ceilings that
        were pinned measurements in everything but name. Nothing related them, so "is this number
        measured or guessed" and "when was it harvested" could only be answered by reading each
        file's prose. This is what stops a fifth shape appearing.

    Semantics
        Applies to modules named ``test_benchmark_perf_*`` under ``tests/perf/`` -- the timed gates.
        Each must expose a module-level ``PINS`` built by :func:`pins.load`, which by construction
        reads ``<same name>.json``. Two failure modes are caught:

        * **no ``PINS``** -- the module either hardcodes numbers or invented its own table;
        * **``PINS`` that is not a ``PinFile``** -- something else was bound to the name.

        It FAILS rather than skips, deliberately. A skip is how a gate that stopped checking anything
        ships quietly, and this check exists precisely because that already happened once here.

    Args:
        collector: The pytest collector. Only ``Module`` collectors are inspected; anything else
            passes through untouched.

    Returns:
        None.

    Raises:
        pytest.UsageError: Naming the module and what it must declare.
    """
    if not isinstance(collector, pytest.Module):
        return
    path = pathlib.Path(str(collector.path))
    if not (
        path.parent == pathlib.Path(__file__).parent
        and path.name.startswith("test_benchmark_perf_")
    ):
        return
    from tests.perf.pins import PinFile, pin_path

    module = collector.obj
    got = getattr(module, "PINS", None)
    if not isinstance(got, PinFile):
        raise pytest.UsageError(
            f"{path.name} is a timed perf gate but does not expose a PINS loaded from its pin "
            f"file. Add `from pins import load as load_pins` and `PINS = load_pins(__file__)`, and "
            f"put every pinned number in {pin_path(str(path)).name}. A gate that keeps its numbers "
            f"in module constants cannot be told apart from one that guessed them."
        )
    _require_dispatch_gate(path, module)


def _require_dispatch_gate(path, module):
    """Refuse a timed perf gate that pins device time but never pins its HOST submit cost.

    Purpose
        Every cell in this directory is now timed with ``mode="device"``, which reports the kernel
        and NOTHING about the per-call Python that submits it. That was the right change -- before
        it, a launch-bound cell reported host dispatch as if it were kernel time, so an 8x8 GEMM
        read 27 us against 7 us of real device time and no mainloop regression at a small shape
        could ever have been seen. But it leaves the submit path unwatched unless something insists,
        and the submit path is where this repo's cost actually accumulates: measured here, `gemm`
        dispatches in 1.31x a bare ``GemmSm90`` launch, `layernorm_fwd` 2.02x, `dual_gated_gemm`
        2.66x. None of that is visible in any device-mode cell.

    Semantics
        A module passes if it declares at least one dispatch pin -- a cell whose key begins with
        ``"dispatch"`` -- or opts out with a module-level ``DISPATCH_EXEMPT`` naming a reason.

        The exemption exists for one real case, and it is expected to be used. **An NVSHMEM
        communication kernel cannot go through TVM-FFI**, so its submit path is a different shape
        and the bare-``GemmSm90`` host reference is the WRONG reference for it -- a reference is only
        valid where it shares a bottleneck class with its target, which is the same rule that made
        the device reference unusable for launch-bound cells in the first place. Such a gate needs
        its own non-TVM-FFI host reference; until one is pinned, it declares the exemption and says
        so, rather than being silently calibrated against a reference that does not apply.

    Args:
        path: The module's path, for the message.
        module: The imported module object.

    Returns:
        None.

    Raises:
        pytest.UsageError: Naming the module and both ways to satisfy the rule.
    """
    if getattr(module, "DISPATCH_EXEMPT", None):
        return
    if any(tuple(k)[:1] == ("dispatch",) for k in module.PINS.keys()):
        return
    raise pytest.UsageError(
        f"{path.name} pins device time but never pins its HOST dispatch cost. Every cell here is "
        f"timed with mode='device', which reports the kernel and nothing about the per-call Python "
        f"that submits it -- so without a dispatch gate that cost is unwatched, and it is where "
        f"this repo's per-call overhead actually accumulates (measured: gemm 1.31x a bare "
        f"GemmSm90 launch, layernorm_fwd 2.02x, dual_gated_gemm 2.66x).\n"
        f"Add a test calling `calibration.assert_host_dispatch(call, ('dispatch', '<entry>'), "
        f"PINS, __file__, device=dev)` with `call` a TINY-shape closure over this gate's entry "
        f"point, then harvest.\n"
        f"OR set `DISPATCH_EXEMPT = '<reason>'`. The reason this exists: an NVSHMEM kernel cannot "
        f"use TVM-FFI, so the bare-GemmSm90 host reference shares no bottleneck class with it and "
        f"would calibrate it wrongly; it needs its own non-TVM-FFI host reference first."
    )

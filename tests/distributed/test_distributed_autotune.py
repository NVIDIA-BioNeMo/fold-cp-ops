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

"""Unit for ``fold_cp_ops/distributed/distributed_autotune.py`` (N1b).

The subject is the wiring, not the arithmetic: `Consensus` and `TimingPolicy` own the reduction and
the timer and are covered by ``tests/_internal/autotune/``. What is tested here is that
:func:`build_distributed_autotuner` composes them so a REAL multi-rank sweep elects one config, and
that :func:`consensus_assert` fires when it does not.

**The timing policy is injected, and that is the point.** A real timer cannot be asked to disagree
across ranks on demand, so the divergence that consensus exists to prevent would never be exercised.
The injected policy keeps ``collective=True`` -- the builder refuses a non-collective one -- and
returns deliberately rank-dependent numbers, which is the only way to drive the failing path. It also
means the whole file runs on gloo with no GPU.

Run::

    CPO_CACHE_ENABLED=0 torchrun --nproc_per_node=2 -m pytest -q \\
        tests/distributed/test_distributed_autotune.py

Single-process, the multi-rank half skips (declared, rank-invariant: WORLD_SIZE is a property of the
launch, identical for every rank, so no rank can skip alone).
"""

from __future__ import annotations

import os
import pathlib

import pytest
import torch

import fold_cp_ops._internal.autotune.timing as timing_mod
from fold_cp_ops._internal.autotune import AutotuneConfig, TimingPolicy
from fold_cp_ops.distributed import DistributedManager
from fold_cp_ops.distributed.distributed_autotune import (
    DistributedAutotuneAdapter,
    build_distributed_autotuner,
    consensus_assert,
    make_collective_timing_policy,
)
from fold_cp_ops.testing.collective_guard import rank_invariant_skip
from fold_cp_ops.testing.kernel_matrix import matrix_exempt
from fold_cp_ops.testing.numeric_guard import numeric_exempt

pytestmark = [
    matrix_exempt(
        "the subject is the autotune WIRING -- there is no kernel, no operand and no tile; the "
        "mesh does not vary either, since consensus is a property of any world size >= 2"
    ),
    numeric_exempt("asserts config identity across ranks, not a computed tensor"),
]

_CFGS = [AutotuneConfig(tile=64), AutotuneConfig(tile=128), AutotuneConfig(tile=256)]

#: This repo's root, for the `-O` subprocess below.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _under_torchrun() -> bool:
    """True only for a genuine multi-rank launch (mirrors ``tests/_internal/autotune``)."""
    return "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1


#: The DM-registered name for this file's CPU process group. Not ``"world"``, which is reserved --
#: and deliberately re-derived rather than cached, because `DistributedManager.reset_grid_groups`
#: keeps ONLY ``"world"`` in ``_group``, so any sibling module's ``apply_mesh`` in a shared session
#: evicts this entry. `cpu_group` is therefore a get-or-create, not a one-shot.
_CPU_GROUP = "world_cpu"


@pytest.fixture(scope="module")
def gloo_group(dist_manager):
    """This file's CPU process group, OWNED BY THE DistributedManager.

    Purpose
        The tests need a gloo group because their buffers are CPU. They must not OWN one: the
        manager exists precisely so process-group lifecycle is not done ad hoc, and this fixture
        used to do it ad hoc in three separate ways.

    Why it is not `dist.init_process_group` / `dist.destroy_process_group` any more
        Every one of those spellings was a hazard, and each has been observed:

        * ``init_process_group(backend="gloo")`` is a SECOND rendezvous behind the manager's back,
          and it is itself a COLLECTIVE that can fail. Measured at WORLD_SIZE=16 across two nodes:
          it failed with ``connectFullMesh ... Connection refused, remote=[127.0.1.1]``, those ranks
          aborted, the NCCL watchdog then killed the peers that were mid-collective with them, and
          10 of 16 ranks core-dumped while the remaining 6 blocked forever. A fixture took down the
          job.
        * ``destroy_process_group()`` with no argument destroys the DEFAULT group -- the one the
          manager and every sibling module are holding. The previous comment here recorded the
          conftest's desync watchdog firing at this fixture's teardown for exactly that reason.
        * ``created_here = not dist.is_initialized()`` decides what to do by consulting a
          process-GLOBAL while taking its other inputs from elsewhere. That is the same shape as the
          `consensus_assert` defect this file's other tests now cover: two sources of truth for one
          fact.

        `DistributedManager.create_group` registers the group in ``_state["_group"]``, so
        `DistributedManager.cleanup` tears it down inside the manager's own bounded,
        barrier-ordered shutdown. Nothing here creates or destroys anything the manager does not
        know about.

    Args:
        dist_manager: the session-scoped manager. REQUIRED, and it is what changed: this file used
            to take no manager at all, which is why it had to bootstrap its own group.

    Yields:
        The ``ProcessGroup`` to pass explicitly as ``group=`` at every collective.

    Note:
        This does NOT by itself make gloo routable across nodes. gloo's device address comes from
        ``createDefaultDevice()``, which resolves the local hostname when ``GLOO_SOCKET_IFNAME`` is
        unset -- and on these nodes ``/etc/hosts`` maps the hostname to ``127.0.1.1``. That is a
        LAUNCH-ENVIRONMENT setting and belongs to the launcher, not here. What the manager fixes is
        that the failure now has ONE owner instead of being rediscovered per fixture.
    """
    import torch.distributed as dist

    if not _under_torchrun():
        rank_invariant_skip(
            "needs a multi-rank torchrun launch",
            because=(
                "WORLD_SIZE is set by the launcher and is identical for every rank of a job, so "
                "every rank reaches the same verdict and no rank can skip alone"
            ),
        )
    # GET-OR-CREATE, and the `create_group` call is COLLECTIVE -- every rank must reach it or the
    # ones that do block. The predicate is the manager's own registry, which is rank-uniform
    # (`reset_grid_groups` runs on every rank or none), so the branch cannot diverge.
    if _CPU_GROUP not in DistributedManager._state["_group"]:
        DistributedManager.create_group(
            _CPU_GROUP, list(range(dist.get_world_size())), backend="gloo"
        )
    group = DistributedManager._state["_group"][_CPU_GROUP]
    yield group
    # NO teardown. The manager owns it; `cleanup()` barriers and destroys once, in its own order.


class _FakeDM:
    """The two attributes this module reads off a distributed manager, and nothing else.

    ``device`` is CPU so ``consensus_assert``'s broadcast buffers work under gloo; ``local_rank`` is
    0 because a gloo barrier here takes no ``device_ids`` that matter. Duck-typed deliberately --
    the module reads attributes rather than importing the manager, which is what lets this run
    without nvshmem.
    """

    def __init__(self, rank=0):
        self.device = torch.device("cpu")
        self.local_rank = 0
        self.rank = rank


class _Adapter:
    """A minimal `DistributedAutotuneAdapter`: three configs, a no-op launch, a compile counter.

    ``launch`` must accept the tuned knob by NAME (``tile``) or `Autotuner` refuses it at
    construction -- which is the check that catches a typo'd knob before the sweep reports "every
    config failed".
    """

    key_names: list = []
    key_tensors: tuple = ()
    fixed_kwargs: dict = {}

    def __init__(self):
        self.precompiled = 0
        self.launched = 0

    def configs(self):
        return list(_CFGS)

    def precompile(self, dm) -> None:
        self.precompiled += 1

    def launch(self, *, tile: int = 64):
        self.launched += 1

    def correctness(self, config_kwargs: dict, dm) -> float:
        return 0.0

    def free(self) -> None:
        pass


def _rank_dependent_policy(order):
    """A collective policy whose measurement depends on the rank, via ``order``.

    Args:
        order: Milliseconds per ``tile`` value, as a dict. Passing a DIFFERENT mapping per rank is
            what manufactures the divergence a real timer cannot be asked for.

    Returns:
        A `TimingPolicy` with ``collective=True`` and an injected measure. The measure ignores the
        callable and reads the config off the adapter's recorded call, so it needs no device.
    """
    state = {"tile": None}

    def measure(fn, *, rounds, warmup, iters, reduce, mode="event"):
        fn()  # let the tuner's binding run exactly as it would in production
        return order[state["tile"]]

    policy = TimingPolicy(collective=True, measure=measure)
    policy._probe_state = state  # the sweep writes the current tile here (see _ProbeAdapter)
    return policy


class _ProbeAdapter(_Adapter):
    """`_Adapter` that publishes the config currently being timed, so the fake measure can price it."""

    def __init__(self, policy):
        super().__init__()
        self._state = policy._probe_state

    def launch(self, *, tile: int = 64):
        self._state["tile"] = tile
        self.launched += 1


def test_the_protocol_is_satisfied_by_a_conforming_adapter():
    """`DistributedAutotuneAdapter` is ``runtime_checkable``, so conformance is assertable.

    A structural check only -- it verifies the attribute and method NAMES, which is exactly the part
    a per-kernel adapter gets wrong when the facility's contract drifts.
    """
    assert isinstance(_Adapter(), DistributedAutotuneAdapter)


def test_a_non_collective_policy_is_refused():
    """Injecting a local policy is refused, naming the desync it would cause.

    Accepting-then-ignoring is how the wrong timer survives review; the whole module exists to keep
    ranks in lockstep, so the one setting that breaks that is rejected at the front door.
    """
    with pytest.raises(ValueError, match="COLLECTIVE timing policy"):
        build_distributed_autotuner(_Adapter(), _FakeDM(), policy=TimingPolicy(collective=False))


@pytest.mark.parametrize("bad", [dict(n_iters=0), dict(n_iters=-3), dict(warmup=-1)])
def test_a_meaningless_iteration_count_is_refused(bad):
    """A zero-length timed window measures nothing and divides by zero downstream."""
    with pytest.raises(ValueError):
        make_collective_timing_policy(**bad)


def test_the_collective_policy_substitutes_only_the_iteration_counts():
    """``n_iters`` / ``warmup`` are overridden; rounds, reduce and mode stay the packaged ones.

    If the override reached ``reduce`` it would drop the slowest-PE reduction, and the sweep would
    silently go back to per-rank minima -- the failure this module exists to prevent, reintroduced
    by its own convenience knob.
    """
    seen = {}

    def spy(fn, **kw):
        seen.update(kw)
        return 1.0

    # Patch the packaged primitive, NOT the policy's `_measure`. Replacing `_measure` would REPLACE
    # the override this test is about, so the spy would observe the policy's own constants and the
    # assertions below would read 10/5 -- passing or failing for reasons unrelated to the override.
    # Measured: that spelling reported iters=10 where 7 was requested.
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(timing_mod, "_default_measure", spy)
    try:
        make_collective_timing_policy(n_iters=7, warmup=2).measure(lambda: None)
    finally:
        monkeypatch.undo()
    assert seen["iters"] == 7 and seen["warmup"] == 2
    assert seen["reduce"] == "max", "the slowest-PE reduction must survive the override"
    assert seen["mode"] == "event", "a collective must not be timed with the adaptive device mode"


def test_precompile_runs_once_and_outside_the_timed_loop(gloo_group):
    """``adapter.precompile`` is called exactly once per ``run()``, before any launch.

    Takes ``gloo_group`` even though nothing here is collective: under a distributed launcher the
    `Autotuner` REFUSES to sweep with no initialized process group, because each rank would tune
    independently and a kernel containing a collective then deadlocks with nothing pointing back at
    the sweep. That refusal is correct, and a test that dodged it by not running under torchrun would
    be exercising a different code path from production.
    """
    policy = _rank_dependent_policy({64: 3.0, 128: 1.0, 256: 2.0})
    a = _ProbeAdapter(policy)
    best, tuner = build_distributed_autotuner(a, _FakeDM(), policy=policy, group=gloo_group)()
    assert a.precompiled == 1, f"precompile ran {a.precompiled} times"
    assert a.launched >= len(_CFGS), "every candidate must be launched at least once"
    assert best.all_kwargs()["tile"] == 128, best


def test_consensus_assert_is_inert_without_a_process_group(monkeypatch):
    """With no group there is nobody to disagree with, so no collective may be issued.

    The predicate is MONKEYPATCHED rather than inherited from the process. Asserting "no group
    exists" by simply not making one is true only while this file runs alone -- in a directory-wide
    run a sibling's session fixture has already initialized NCCL, and this test then drove a real
    broadcast of CPU buffers into it, which is a C++ abort rather than a failure. The property under
    test is "inert when there is no group", and that is what this now states.
    """
    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    consensus_assert(_CFGS[0], _FakeDM())


def test_distributed_every_rank_elects_the_same_config(gloo_group):
    """D1b.1 -- a real two-rank sweep with ASYMMETRIC local timings elects ONE config.

    Rank 0 would locally prefer ``tile=64`` and rank 1 ``tile=256``; the max-reduce makes both see
    the same numbers, and ``tile=128`` -- the best WORST-CASE, which is what a collective actually
    costs -- wins on both. Without the reduction the two ranks compile different kernels and the next
    collective hangs.
    """
    import torch.distributed as dist

    rank = dist.get_rank(gloo_group)
    order = {64: 1.0, 128: 2.0, 256: 9.0} if rank == 0 else {64: 9.0, 128: 2.0, 256: 1.0}
    policy = _rank_dependent_policy(order)
    a = _ProbeAdapter(policy)
    best, _ = build_distributed_autotuner(a, _FakeDM(rank), policy=policy, group=gloo_group)()
    gathered = [None] * dist.get_world_size(gloo_group)
    dist.all_gather_object(gathered, best.all_kwargs(), group=gloo_group)
    assert len(set(map(repr, gathered))) == 1, f"ranks elected different configs: {gathered}"
    assert best.all_kwargs()["tile"] == 128, best


def test_distributed_consensus_assert_fails_on_a_divergent_pick(gloo_group):
    """D1b.2 -- the NEGATIVE CONTROL: force rank 1 to pick differently and require the raise.

    A consensus check never seen to fail has not been tested. Here the ranks are handed configs that
    genuinely differ, bypassing the reduction entirely, so the only thing that can catch it is
    ``consensus_assert`` itself. Rank 0 must NOT raise -- it is the source of truth -- which is what
    distinguishes a working check from one that raises on everybody.
    """
    import torch.distributed as dist

    if dist.get_world_size(gloo_group) < 2:
        rank_invariant_skip(
            "a divergent pick needs at least two ranks",
            because=(
                "world size is fixed by the launcher and is the same number on every rank, so "
                "every rank reaches this verdict together and none can leave its peers in a "
                "collective"
            ),
        )
    rank = dist.get_rank(gloo_group)
    divergent = _CFGS[0] if rank == 0 else _CFGS[2]

    # The raise is CAUGHT rather than asserted through `pytest.raises`, and the barrier is reached
    # UNCONDITIONALLY before any assertion runs. Written the obvious way -- `pytest.raises` around
    # the call, `dist.barrier()` after -- this test DEADLOCKS the moment it does its job: rank 1's
    # assertion failure propagates out of the test body, rank 1 never reaches the barrier, and rank 0
    # blocks in it forever. Measured: the fail-first proof (assert deleted from `consensus_assert`)
    # hung until the outer timeout instead of reporting DID NOT RAISE. A test of a collective must
    # not put a collective downstream of its own assertion.
    raised = None
    try:
        consensus_assert(divergent, _FakeDM(rank), group=gloo_group)
        raised = ""
    except RuntimeError as e:
        # RuntimeError, not AssertionError: the check is an explicit raise so it survives
        # `python -O`. Catching AssertionError here would let a stripped-assertions build report
        # "rank 1 did not raise" as a test failure whose real cause is the flag, not the code.
        raised = str(e)
    dist.barrier(group=gloo_group)
    if rank == 0:
        assert raised == "", f"rank 0 is the source of truth and must not raise; got {raised}"
    else:
        assert "CONSENSUS FAILURE" in raised, (
            f"rank {rank} picked a different config and consensus_assert did not catch it; "
            f"got {raised!r}"
        )


def test_distributed_agreeing_ranks_pass_consensus_assert(gloo_group):
    """The positive control for the test above: identical picks must NOT raise.

    Without it, a ``consensus_assert`` that raised unconditionally would satisfy the negative
    control and look correct.
    """
    consensus_assert(_CFGS[1], _FakeDM(), group=gloo_group)


def test_consensus_assert_uses_no_pickle_and_no_object_collective():
    """A SOURCE-level regression: the config broadcast must never be able to execute what it decodes.

    Purpose
        ``pickle.loads`` on a received buffer constructs whatever the sender's bytes describe, which
        includes calling arbitrary importable objects. Inside one scheduler-admitted job that was
        not reachable by an attacker (see ``SECURITY.md``'s trust boundary), so this is a defence in
        depth -- and defence in depth is exactly the kind of change a later refactor undoes without
        noticing, because nothing it protects is failing today.

    Why AST and not behaviour
        No runtime assertion can distinguish "decoded safely" from "decoded with a codec that HAPPENS
        not to have been handed a malicious payload". The property is about which call is written,
        so the test reads the call. It parses `consensus_assert`'s own body -- not the module -- so
        an unrelated ``pickle`` elsewhere in the file cannot satisfy or break it.

    What it refuses, and why each one
        * ``pickle.dumps`` / ``pickle.loads`` -- the substitution being locked in.
        * ``broadcast_object_list`` / ``all_gather_object`` / ``gather_object`` /
          ``scatter_object_list`` -- PyTorch's object collectives, which pickle internally. Swapping
          the two explicit broadcasts for one of these looks like a simplification and silently
          restores the whole hazard.
        * an unpack missing any of ``raw=False``, ``use_list=False``, ``strict_map_key=True``, or
          carrying an ``object_hook`` / ``object_pairs_hook`` / ``ext_hook``. The hooks are the
          documented way to make MessagePack construct arbitrary objects again, so their absence is
          as load-bearing as the codec choice.
    """
    import ast
    import inspect
    import textwrap

    from fold_cp_ops.distributed import distributed_autotune as da

    tree = ast.parse(textwrap.dedent(inspect.getsource(da.consensus_assert)))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]

    def _dotted(node):
        """``a.b.c`` for an attribute/name call target, else ``""``."""
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
        return ".".join(reversed(parts))

    names = {_dotted(c.func) for c in calls}
    banned = {
        "pickle.dumps",
        "pickle.loads",
        "dist.broadcast_object_list",
        "dist.all_gather_object",
        "dist.gather_object",
        "dist.scatter_object_list",
    }
    assert not (names & banned), (
        f"consensus_assert reintroduced an unsafe deserializer: {sorted(names & banned)}"
    )

    unpacks = [c for c in calls if _dotted(c.func) in ("msgpack.loads", "msgpack.unpackb")]
    assert len(unpacks) == 1, f"expected exactly one MessagePack unpack, found {len(unpacks)}"
    kw = {k.arg: k.value for k in unpacks[0].keywords}
    for name, want in (("raw", False), ("use_list", False), ("strict_map_key", True)):
        assert name in kw, f"the unpack does not pin {name}=; it must not depend on a default"
        assert isinstance(kw[name], ast.Constant) and kw[name].value is want, (
            f"the unpack passes {name}={ast.dump(kw[name])}, expected the literal {want}"
        )
    for hook in ("object_hook", "object_pairs_hook", "ext_hook", "list_hook"):
        assert hook not in kw, f"{hook}= lets the decoder construct arbitrary objects again"

    packs = [c for c in calls if _dotted(c.func) in ("msgpack.dumps", "msgpack.packb")]
    assert len(packs) == 1, f"expected exactly one MessagePack pack, found {len(packs)}"
    pack_kw = {k.arg for k in packs[0].keywords}
    assert "default" not in pack_kw, "default= is the encoder-side hook for arbitrary objects"


def test_distributed_a_config_survives_the_codec_with_its_types_and_key_order(gloo_group):
    """The Gloo counterpart: the values this repo really uses must round-trip UNCHANGED.

    Purpose
        The AST test above pins which codec is called; this pins that the codec preserves what the
        equality assert compares. Both are needed -- a correctly-spelled call that mangled a tuple
        into a list would make `consensus_assert` fire on every rank, turning a safety improvement
        into an outage.

    What is covered, and why each value is there
        ``int`` and ``str`` and ``bool`` are the knob types every shipped ``AutotuneConfig`` uses
        (``tile_m=128``, ``do_transpose="SMEM"``, ``pingpong=False``). The nested TUPLE is the one
        that needs ``use_list=False``: ``cluster=(1, 2)`` is a live spelling in this repo's config
        pools, and MessagePack encodes a tuple as an array, which decodes to a ``list`` by default.

    Insertion order
        The two dicts are built in different orders on purpose. Python dict equality ignores order,
        but MessagePack preserves it on the wire, so the two ranks send DIFFERENT BYTES for the same
        config -- and the assert must still pass. That is the property that makes it safe for ranks
        to build their config packs independently rather than in lockstep.
    """
    import torch.distributed as dist

    rank = dist.get_rank(gloo_group)
    if rank == 0:
        kwargs = dict(tile_m=128, tile_n=256, cluster=(1, 2), do_transpose="SMEM", pingpong=False)
    else:
        kwargs = dict(pingpong=False, do_transpose="SMEM", cluster=(1, 2), tile_n=256, tile_m=128)
    cfg = AutotuneConfig(**kwargs)
    assert cfg.all_kwargs()["cluster"] == (1, 2), "the fixture itself must carry a real tuple"

    consensus_assert(cfg, _FakeDM(rank), group=gloo_group)

    # And the codec really is the one under test: the same pack, decoded here, keeps its types.
    import msgpack

    decoded = msgpack.loads(
        msgpack.dumps(cfg.all_kwargs(), use_bin_type=True),
        raw=False,
        use_list=False,
        strict_map_key=True,
    )
    assert decoded == cfg.all_kwargs(), f"round-trip changed the pack: {decoded}"
    assert isinstance(decoded["cluster"], tuple), (
        f"a tuple knob decoded as {type(decoded['cluster']).__name__}; use_list=False was dropped"
    )
    assert isinstance(decoded["pingpong"], bool), "a bool knob must not decode as an int"


@pytest.mark.parametrize(
    "payload,why",
    [
        (["not", "a", "map"], "an array is not a config"),
        ("a string", "a scalar is not a config"),
        (7, "an integer is not a config"),
        (None, "nil is not a config"),
    ],
)
def test_a_non_map_broadcast_payload_is_REFUSED_by_shape(payload, why, monkeypatch):
    """"It decoded" is not "it is a config", and the decoder can emit any MessagePack type.

    Purpose
        Cover the shape gate without a process group. The payload is injected by monkeypatching the
        decoder, because a real two-rank run cannot send a malformed pack -- the sender is the same
        code -- and the gate exists precisely for the case where the bytes did not come from it.

    Semantics
        Every case must raise ``RuntimeError`` naming the failure, not ``TypeError`` from somewhere
        deeper: an unreadable payload has to be refused where both the rank and the payload can be
        named, not several frames into a launch.
    """
    import torch.distributed as dist

    from fold_cp_ops.distributed import distributed_autotune as da

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "group", type("G", (), {"WORLD": object()})())
    monkeypatch.setattr(dist, "get_backend", lambda g: "gloo")
    monkeypatch.setattr(dist, "broadcast", lambda *a, **k: None)
    monkeypatch.setattr(da.msgpack, "loads", lambda *a, **k: payload)

    with pytest.raises(RuntimeError, match="CONSENSUS FAILURE"):
        consensus_assert(_CFGS[0], _FakeDM(1))


@pytest.mark.parametrize(
    "remote,why",
    [
        ({"tile": True}, "True vs 1 -- equal under ==, different kernels"),
        ({"tile": 1.0}, "1.0 vs 1 -- equal under =="),
        ({"tile": 1, "extra": 2}, "an extra knob"),
        ({}, "a missing knob"),
        ({"tile": (1, 2)}, "a tuple where a scalar was picked"),
        ({"tile": ((1, True),)}, "a type difference NESTED inside a tuple"),
    ],
)
def test_a_type_or_field_difference_is_caught_where_dict_equality_would_not_be(
    remote, why, monkeypatch
):
    """The reason the comparison is a canonical encoding rather than ``==``.

    ``True == 1 == 1.0`` in Python, so ``{"tile": True} == {"tile": 1}`` is True and two ranks
    holding different TYPES for one knob would "agree", trace two different kernels, and deadlock in
    the next collective -- a hang with no attribution, minutes later. The canonical form tags every
    scalar and recurses into tuples, so the difference is visible at any depth.

    The missing/extra cases ride the same mechanism: a differing key set changes the sorted pairs.
    """
    import torch.distributed as dist

    from fold_cp_ops.distributed import distributed_autotune as da

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "group", type("G", (), {"WORLD": object()})())
    monkeypatch.setattr(dist, "get_backend", lambda g: "gloo")
    monkeypatch.setattr(dist, "broadcast", lambda *a, **k: None)
    monkeypatch.setattr(da.msgpack, "loads", lambda *a, **k: remote)

    local = AutotuneConfig(tile=1)
    assert local.all_kwargs() == {"tile": 1}
    with pytest.raises(RuntimeError, match="CONSENSUS FAILURE"):
        consensus_assert(local, _FakeDM(1))


def test_the_consensus_check_still_fires_under_python_dash_O():
    """``assert`` is stripped by ``-O``; this check must not be.

    Purpose
        A safety check that disappears under an optimization flag is worse than no check, because
        nothing announces its absence -- the run simply stops catching divergence and the next
        symptom is a hang in an unrelated collective.

    Semantics
        Runs a real interpreter with ``-O`` in a subprocess. Asserting the source says ``raise``
        would be a weaker test: it would pass for a ``raise`` sitting inside an ``if __debug__``.
    """
    import subprocess
    import sys

    prog = (
        "from fold_cp_ops._internal.autotune import AutotuneConfig\n"
        "from fold_cp_ops.distributed import distributed_autotune as da\n"
        "import torch.distributed as dist\n"
        "dist.is_initialized = lambda: True\n"
        "dist.group = type('G', (), {'WORLD': object()})()\n"
        "dist.get_backend = lambda g: 'gloo'\n"
        "dist.broadcast = lambda *a, **k: None\n"
        "da.msgpack.loads = lambda *a, **k: {'tile': 999}\n"
        "assert __debug__ is False, 'the subprocess is not running under -O'\n"
        "try:\n"
        "    da.consensus_assert(AutotuneConfig(tile=1), type('D', (), {'rank': 1})())\n"
        "except RuntimeError as e:\n"
        "    print('RAISED' if 'CONSENSUS FAILURE' in str(e) else 'WRONG')\n"
        "else:\n"
        "    print('DID NOT RAISE')\n"
    )
    out = subprocess.run(
        [sys.executable, "-O", "-c", prog],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
    )
    assert out.stdout.strip() == "RAISED", f"stdout={out.stdout!r} stderr={out.stderr[-800:]!r}"


@pytest.mark.parametrize(
    "payload,why",
    [
        ({1: 128}, "an integer knob name"),
        ({b"tile": 128}, "a bytes knob name"),
    ],
)
def test_a_config_with_NON_STRING_knob_names_is_REFUSED(payload, why, monkeypatch):
    """Knob names are keyword-argument names, so a non-string one cannot be a config.

    ``strict_map_key=True`` already refuses most of these at DECODE time, but not all -- ``bytes``
    keys are permitted by that option -- and the decoder is not the only way a payload reaches this
    function. The shape gate therefore checks the key types itself rather than relying on a codec
    option to have covered it, which is the difference between a property this code guarantees and
    one it inherits.
    """
    import torch.distributed as dist

    from fold_cp_ops.distributed import distributed_autotune as da

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "group", type("G", (), {"WORLD": object()})())
    monkeypatch.setattr(dist, "get_backend", lambda g: "gloo")
    monkeypatch.setattr(dist, "broadcast", lambda *a, **k: None)
    monkeypatch.setattr(da.msgpack, "loads", lambda *a, **k: payload)

    with pytest.raises(RuntimeError, match="non-string knob names"):
        consensus_assert(_CFGS[0], _FakeDM(1))

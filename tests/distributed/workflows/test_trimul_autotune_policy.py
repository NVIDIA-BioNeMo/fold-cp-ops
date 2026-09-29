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

"""Unit for ``fold_cp_ops/distributed/workflows/trimul_autotune_policy.py`` -- N3.

The module is the A2A-fused TriMul's autotune half: the two per-kernel adapters and the venue-keyed
candidate grids they offer. What is tested here is the GRID, because the grid is where a venue
decision is made and where a silent narrowing costs performance with no error anywhere.

``_grid`` is a pure function of three facts -- ``cp1``, ``has_ib_peers`` and ``N_j_loc`` -- so the
whole file runs with no GPU, no process group and no nvshmem. That is deliberate and it is the point:
the defect this file's history records (a design-E alignment rule applied to a pe_aligned proxy, which
dropped EVERY config at any off-grid N and left the back store silently untuned) was reachable only
through a multi-rank sweep, and is decidable here in microseconds.
"""

from __future__ import annotations

import json

import pytest
import os
import stat

from fold_cp_ops._internal.autotune import AutotuneConfig
from fold_cp_ops.distributed.workflows import trimul_autotune_policy as policy
from fold_cp_ops.testing.kernel_matrix import matrix_exempt

NUMERIC_EXEMPT = (
    "every assertion compares CANDIDATE CONFIG TUPLES -- tile extents and cluster counts -- not a "
    "computed tensor. The grid decides what gets measured; the measuring is the autotuner's job and "
    "is covered by tests/_internal/autotune/."
)

pytestmark = matrix_exempt(
    "the subject is a pure host-side candidate grid; there is no kernel, no operand and no tile to "
    "sweep, and the parametrization below IS the input domain"
)


class _BareBack:
    """Just enough of the back adapter for `_grid` / `configs` -- the three facts they read.

    A real adapter allocates symmetric memory and needs a live pe_map, so constructing one would
    make this a distributed test for no gain: `_grid` reads these attributes and nothing else.
    Written as a class rather than a `SimpleNamespace` so an attribute the grid starts reading
    fails loudly here instead of silently defaulting.
    """

    def __init__(self, *, cp1, has_ib_peers, N_j_loc, N_i_loc=1024):
        self.cp1 = cp1
        self.has_ib_peers = has_ib_peers
        self.N_j_loc = N_j_loc
        self.N_i_loc = N_i_loc

    _grid = policy._BackProxyAdapter._grid
    configs = policy._BackProxyAdapter.configs

    def _grid_as_cfgs(self):
        """configs() on this bare stand-in -- the public shape the sweep consumes."""
        return self.configs()


def _tuples(cfgs):
    """``AutotuneConfig`` list -> comparable ``(tile_m, tile_n, cluster_n)`` tuples."""
    return [
        (c.all_kwargs()["tile_m"], c.all_kwargs()["tile_n"], c.all_kwargs()["cluster_n"])
        for c in cfgs
    ]


def test_a_two_dimensional_mesh_gets_the_conservative_single_config():
    """2-D keeps the validated ``(128, 128, cluster_n=1)`` and nothing else.

    The N-cluster lever is validated for the 1-D store only; on the 2-D peer-unravel it is not
    co-validated, so offering it would have the sweep MEASURE a config production refuses.
    """
    assert _tuples(_BareBack(cp1=8, has_ib_peers=True, N_j_loc=1024)._grid_as_cfgs()) == [
        (128, 128, 1)
    ]


def test_a_two_dimensional_mesh_offers_one_config_or_direction_must_key_the_freeze():
    """The single-config 2-D grid is what makes `direction`'s ABSENCE from the freeze key safe.

    THE FREEZE KEY IS ``(cp, N, D[, cp1])`` AND CARRIES NO DIRECTION, and the sweep proxy hardcodes
    ``self.direction = "outgoing"`` -- so one elected config serves BOTH directions. Whether that is
    correct or a silent gap was an open question, and it resolves on two measurements that only hold
    together:

    * Where the grid HAS choices (1-D meshes), the two directions perform within +-4%. Measured
      across 26 pinned (mesh, D) pairs at N=2048: cp2/cp4/cp8/cp16 give incoming/outgoing ratios of
      0.977-1.040, so an outgoing-elected config is not mis-serving incoming.
    * Where the two directions DO differ materially -- 2-D meshes, ratios 0.81 to 1.70 -- the grid
      offers exactly ONE config, so there is nothing direction could have changed. The asymmetry is
      a property of the peer pattern, not of a mis-elected tile.

    **The whole argument rests on this one line of `_grid`, and nothing else would notice if it
    changed.** Widening the 2-D grid immediately reopens the question, at the meshes where the
    directional spread is largest -- and the symptom would be pins quietly measured against a config
    the autotuner would not have picked for that direction. So the invariant is asserted here rather
    than left as a comment: EITHER the 2-D grid stays single-config, OR `direction` becomes part of
    the freeze key and this test is updated to say so.

    Not a duplicate of the test above it: that one pins WHICH config a 2-D mesh gets, this one pins
    HOW MANY and says what depends on the count.
    """
    for cp1 in (2, 4, 8):
        for ib in (True, False):
            grid = _tuples(_BareBack(cp1=cp1, has_ib_peers=ib, N_j_loc=1024)._grid_as_cfgs())
            assert len(grid) == 1, (
                f"the 2-D back grid (cp1={cp1}, has_ib_peers={ib}) now offers {len(grid)} configs: "
                f"{grid}. With more than one candidate, an election made at direction='outgoing' can "
                "differ from the one 'incoming' would make, and the freeze key -- (cp, N, D[, cp1]) "
                "-- cannot tell them apart. Add `direction` to the freeze key (and re-harvest every "
                "2-D pin), or keep the grid single-config."
            )


def test_an_all_nvlink_one_dimensional_mesh_gets_the_four_config_grid():
    """All-P2P 1-D sweeps ``tile_n {128, 256} x cluster_n {1, 2}``.

    ``cluster_multislot`` is inert without an IB peer -- the store const-elides it -- so the grid is
    about tile and cluster shape alone. Four configs, and the count matters: the history here is a
    filter that silently reduced it to one.
    """
    got = _tuples(_BareBack(cp1=1, has_ib_peers=False, N_j_loc=1024)._grid_as_cfgs())
    assert got == [(128, 128, 1), (128, 128, 2), (128, 256, 1), (128, 256, 2)], got


@pytest.mark.parametrize(
    "N_j_loc,expect_cns",
    [
        (1024, [1, 2, 4]),  # nt_j_pp = 8; cn=4 divides an even-shard N_j
        (512, [1, 2, 4]),  # nt_j_pp = 4; cn=4 still even-shards
        (384, [1, 2]),  # nt_j_pp = 3; cn=4 exceeds it
        (128, [1]),  # nt_j_pp = 1; only the degenerate cluster fits
        (640, [1, 2]),  # nt_j_pp = 5; 640 % 512 != 0 -> cn=4 would STRADDLE, deferred
    ],
)
def test_the_ib_grid_prunes_cluster_n_to_what_the_kernel_can_run(N_j_loc, expect_cns):
    """has-IB 1-D offers ``cluster_n`` in {1,2,4}, pruned by the kernel's own two rules.

    Both rules are the KERNEL's, restated here so the dispatcher and the kernel cannot drift:
    ``cluster_multislot`` caps ``cluster_n <= nt_j_pp`` (a cluster must span at most two adjacent
    peers -- the kernel's error text says "the dispatcher must pick cluster_n<=nt_j_pp"), and
    ``cluster_n >= 4`` additionally requires an EVEN-shard ``N_j`` because the cn>=4 straddle is
    deferred. Tile is pinned 128x128 throughout: ``cluster_multislot`` requires ``tile_m == 128``.

    This is the grid every CROSS-NODE cell of the acceptance grid draws from -- STEP 1 measured
    ``cluster_multislot=True`` on 12 of 28 cell-directions -- so a wrong prune here is a wrong
    production config, not just a wrong benchmark.
    """
    got = _tuples(_BareBack(cp1=1, has_ib_peers=True, N_j_loc=N_j_loc)._grid_as_cfgs())
    assert all(tm == 128 and tn == 128 for tm, tn, _ in got), got
    assert [cn for _, _, cn in got] == expect_cns, got


def test_the_grid_is_never_empty():
    """Every venue yields at least one candidate.

    An empty grid is not a refusal: `ConfigSpace` raises on it, so the sweep dies rather than
    running untuned -- but the fallback exists precisely so a prune that removes everything degrades
    to the always-valid degenerate cluster instead. cluster_n=1 is valid at every N.
    """
    for cp1, ib, njl in ((1, True, 64), (1, False, 64), (8, True, 64), (1, True, 96)):
        got = _tuples(_BareBack(cp1=cp1, has_ib_peers=ib, N_j_loc=njl)._grid_as_cfgs())
        assert got, f"empty grid at cp1={cp1} has_ib={ib} N_j_loc={njl}"
        assert all(isinstance(c, tuple) and len(c) == 3 for c in got)


def test_every_config_the_grid_offers_is_an_autotune_config():
    """`configs()` returns `AutotuneConfig`s, which is what `ConfigSpace` keys and dedups on."""
    cfgs = _BareBack(cp1=1, has_ib_peers=False, N_j_loc=1024)._grid_as_cfgs()
    assert all(isinstance(c, AutotuneConfig) for c in cfgs)
    keys = [c.key() for c in cfgs]
    assert len(set(keys)) == len(keys), f"duplicate configs would be benchmarked twice: {keys}"


# --------------------------------------------------------------------------- #
# D3.2 -- every tuned knob must reach the compiled kernel's identity.
# --------------------------------------------------------------------------- #


def _knob_names():
    """Every knob name the two adapters' grids sweep, as one set."""
    names = set()
    for cfgs in (
        _BareBack(cp1=1, has_ib_peers=False, N_j_loc=1024)._grid_as_cfgs(),
        _BareBack(cp1=1, has_ib_peers=True, N_j_loc=1024)._grid_as_cfgs(),
        _BareBack(cp1=8, has_ib_peers=True, N_j_loc=1024)._grid_as_cfgs(),
    ):
        for c in cfgs:
            names |= set(c.all_kwargs())
    return names


def test_every_tuned_knob_reaches_the_compiled_kernels_identity():
    """D3.2 -- a knob the sweep varies must be part of what identifies the compiled kernel.

    Why this is the gate and not a nicety
        Autotuning measures configs by COMPILING one kernel per config and timing it. If a knob does
        not participate in the artifact's identity, two configs that differ only in that knob resolve
        to the SAME compiled kernel -- so the sweep times one kernel twice, reports a difference that
        is pure noise, and picks a winner that means nothing. **The A/B becomes a tautology.** It
        fails silently and in the direction that looks like success: a tight spread reads as "the
        knob does not matter" rather than as "the knob was never applied".

        This tree has the mirror of that defect on record: 37 ``const_expr`` gates once read
        attributes the key could not see.

    What is checked
        Every knob the grids sweep is either a declared compile-time PARAMETER of the back kernel --
        i.e. a field of its ``Params``, which `compile_key` is built from -- or is listed below with
        the reason it legitimately is not. ``cluster_n`` is the interesting case: it reaches the
        kernel as the SECOND element of ``cluster_shape_mnk``, not under its own name, so it is
        mapped rather than waived.
    """
    from fold_cp_ops.distributed.gemm_a2a_epi import GemmA2ASm90

    params = set(GemmA2ASm90.Params.field_names())
    #: knob -> the compile-time parameter it actually reaches, when the names differ.
    ALIASES = {
        "tile_m": "tile_shape_mn",
        "tile_n": "tile_shape_mn",
        "cluster_n": "cluster_shape_mnk",
    }
    unreached = []
    for knob in sorted(_knob_names()):
        target = ALIASES.get(knob, knob)
        if target not in params:
            unreached.append((knob, target))
    assert not unreached, (
        f"these tuned knobs do not reach the kernel's compile-time identity: {unreached}. "
        f"Declared Params are {sorted(params)}. A knob outside the key makes the sweep time one "
        f"kernel N times and pick noise."
    )
    assert _knob_names(), "the knob set is empty -- this test would pass while checking nothing"


def test_the_alias_targets_are_real_parameters_not_wishful_names():
    """The aliases above must name PARAMETERS, or the mapping is a comment rather than a check.

    An alias to a name the kernel does not declare would let a genuinely-unreached knob pass by
    pointing at a fiction. Asserted separately so the mapping cannot rot into decoration.
    """
    from fold_cp_ops.distributed.gemm_a2a_epi import GemmA2ASm90

    params = set(GemmA2ASm90.Params.field_names())
    for target in ("tile_shape_mn", "cluster_shape_mnk"):
        assert target in params, f"{target!r} is not a declared Param; the alias is fiction"


# ─────────────────────────────────────────────────────── the margin recorder ──


def _tuner_with_timings(timings):
    """A stand-in tuner carrying THIS TREE's timings attribute and nothing else.

    Purpose
        Give the margin tests an object shaped like the one ``build_distributed_autotuner``
        returns, without a GPU or a sweep.

    Input requirements
        ``timings`` maps ``AutotuneConfig`` -> float (or a 1-tuple, which the recorder unwraps).
        It is attached as ``last_timings``, the name ``_internal/autotune/tuner.py`` uses -- writing
        ``configs_timings`` here instead is exactly the mistake these tests exist to catch, so the
        name is deliberately not parameterised.

    Returns:
        An object with a single ``last_timings`` attribute.
    """

    class _T:
        pass

    t = _T()
    t.last_timings = timings
    return t


def test_the_margin_recorder_reads_the_attribute_this_trees_tuner_actually_sets():
    """The recorder must find timings on ``last_timings``.

    This is the regression that motivated the whole group. ``_margin_of`` read ``configs_timings``
    -- correct upstream, where the tuner is ``_internal/autotuner.py`` and sets that name -- but this
    tree restructured the tuner into ``_internal/autotune/`` and renamed the field to
    ``last_timings``. The policy module was carried over verbatim, so the read silently returned
    ``{}`` and EVERY freeze entry written by this repo recorded ``n_configs: 0, rel: None``: the
    schema-v2 field whose entire purpose is telling a 0.2% coin-flip from a 30% win recorded neither.

    Measured before the fix: 10 of 10 entries in a live freeze dir, front and back, every mesh, every
    D. Nothing failed -- the sweeps ran and the picks were sound. Only the evidence was missing.
    """
    a, b = AutotuneConfig(tile=32), AutotuneConfig(tile=64)
    m = policy._margin_of(_tuner_with_timings({a: 1.0, b: 1.5}), a)
    assert m["n_configs"] == 2, f"timings not found; recorder saw {m}"
    assert m["timings_attr"] == "last_timings"
    assert m["rel"] == pytest.approx(0.5), "runner-up is 50% slower"
    assert m["best_ms"] == pytest.approx(1.0)
    assert m["pick_is_best"] is True


def test_a_tuner_exposing_no_timings_is_recorded_as_MISSING_not_as_an_empty_sweep():
    """ "We asked the wrong object" and "the grid timed nothing" must not read alike.

    Both used to land on ``n_configs: 0``, which is why the defect above survived: a reader
    inspecting a freeze entry saw a plausible number for a one-candidate grid. The recorder now
    writes ``timings_attr: "MISSING"``, so the instrument's own failure is visible in the artifact
    rather than inferred from its absence.
    """
    m = policy._margin_of(object(), None)
    assert m["timings_attr"] == "MISSING"
    assert m["n_configs"] == 0
    # An ACTUALLY-empty sweep is the other state, and it must look different.
    empty = policy._margin_of(_tuner_with_timings({AutotuneConfig(tile=32): 1.0}), None)
    assert empty["timings_attr"] == "last_timings" and empty["n_configs"] == 1


def test_the_upstream_attribute_name_still_resolves():
    """``configs_timings`` stays a recognised spelling so a re-sync cannot re-break this silently.

    The fallback is not speculative generality: the upstream tuner sets that name today, and a
    future merge that carries it back would otherwise reintroduce the exact silence this group
    documents -- with no test failing, because the sweep would still succeed.
    """

    class _Upstream:
        configs_timings = {AutotuneConfig(tile=32): 2.0, AutotuneConfig(tile=64): 2.2}

    m = policy._margin_of(_Upstream(), None)
    assert m["timings_attr"] == "configs_timings" and m["n_configs"] == 2
    assert m["rel"] == pytest.approx(0.1)


def test_a_freeze_miss_announces_itself_so_a_reswept_run_is_greppable(
    tmp_path, capsys, monkeypatch
):
    """A MISS must be VISIBLE, because a re-sweeping run and a slow-compiling run look identical.

    This is the regression test for a real cost: distinguishing "the compiler blew up" from "the
    autotuner re-swept" required a multi-cluster experiment purely because neither outcome was
    logged. The assertion is that the MISS path says so on stderr -- and, equally, that a HIT says
    so, since a log that only speaks up on failure cannot confirm the good case either.
    """
    monkeypatch.setenv("CPO_DIST_AUTOTUNE_FREEZE_DIR", str(tmp_path))
    monkeypatch.setattr(policy, "_FREEZE_NOTED", set())
    key = ("front", 2048, 256, 8)

    assert policy._freeze_load("front", key) is None, "no file on disk -> must be a miss"
    err = capsys.readouterr().err
    assert "[freeze] MISS" in err, f"a silent miss is the defect this test exists to catch: {err!r}"
    assert "front_front_2048_256_8.json" in err, "the announcement must name the path that missed"

    # Deduplicated per process: a key hammered in a loop must not flood the log.
    assert policy._freeze_load("front", key) is None
    assert capsys.readouterr().err == "", "the second miss on the same key must stay quiet"


def test_a_freeze_hit_announces_itself_too(tmp_path, capsys, monkeypatch):
    """The HIT half of the pair -- a one-sided log cannot tell 'served from freeze' from 'not called'."""
    monkeypatch.setenv("CPO_DIST_AUTOTUNE_FREEZE_DIR", str(tmp_path))
    monkeypatch.setattr(policy, "_FREEZE_NOTED", set())
    monkeypatch.setattr(policy, "_STALE_NOTED", set())
    key = ("back", 4096)
    cfg = ("a", 128, [1, 2, 1])
    path = tmp_path / ("back_" + "_".join(str(k) for k in key) + ".json")
    path.write_text(json.dumps({"cfg": cfg, "grid_fp": policy.grid_fingerprint("back")}))

    got = policy._freeze_load("back", key)
    assert got is not None, "a live entry with a matching fingerprint must be served, not ignored"
    assert got == ("a", 128, (1, 2, 1)), "JSON lists must come back as tuples"
    err = capsys.readouterr().err
    assert "[freeze] HIT" in err, f"a served entry must announce itself: {err!r}"


# ---------------------------------------------------------------------------
# The distributed-autotune FREEZE cache: private files, safe roots, and -- above all -- a barrier
# that every rank reaches whatever the filesystem does.
# ---------------------------------------------------------------------------


def _freeze_env(tmp_path, monkeypatch):
    """Point the freeze at a private temporary directory and clear the root memo."""
    from fold_cp_ops._internal import cache_security

    d = tmp_path / "freeze"
    monkeypatch.setenv("CPO_DIST_AUTOTUNE_FREEZE_DIR", str(d))
    cache_security.reset_cache()
    return d


class _Rank0:
    """Minimal stand-in for the distributed manager: `_freeze_store` reads only ``.rank``."""

    rank = 0


def test_the_freeze_default_root_no_longer_raises_ImportError(monkeypatch):
    """The default branch imported a name that does not exist, so it raised every time it ran.

    ``from fold_cp_ops._internal.autotune import default_cache_dir`` -- there is no such name in
    that package. The branch is taken whenever ``CPO_DIST_AUTOTUNE_FREEZE_DIR`` is unset, which is
    the DEFAULT configuration, so the default freeze path has never worked; the launchers that
    exercise the freeze all set the variable. A function-local import rots exactly this quietly.
    """
    from fold_cp_ops.distributed.workflows import trimul_autotune_policy as tp

    monkeypatch.delenv("CPO_DIST_AUTOTUNE_FREEZE_DIR", raising=False)
    root = tp._freeze_root()  # must not raise
    assert root.path.name == "dist_autotune_freeze", root.path


def test_a_frozen_pick_round_trips_and_the_file_is_private(tmp_path, monkeypatch):
    """Cold write then warm hit, with the stored entry at 0600.

    The positive control: every refusal test below is satisfied by a freeze that never hits at all,
    so the working path has to be pinned first.
    """
    from fold_cp_ops.distributed.workflows import trimul_autotune_policy as tp

    d = _freeze_env(tmp_path, monkeypatch)
    key = ("cp8", "N2048", "D256")
    cfg = (128, 128, False, (1, 2, 1))
    assert tp._freeze_load("back", key) is None, "the cache must start cold"
    tp._freeze_store("back", key, cfg, _Rank0())

    entries = list(d.glob("*.json"))
    assert entries, "nothing was stored"
    assert stat.S_IMODE(os.stat(entries[0]).st_mode) == 0o600, (
        f"{entries[0].name} is {stat.S_IMODE(os.stat(entries[0]).st_mode):04o}, expected 0600"
    )
    assert stat.S_IMODE(os.stat(d).st_mode) == 0o700, "the freeze root must be 0700"
    assert not list(d.glob("*.tmp")), "a temporary was left behind"
    assert tp._freeze_load("back", key) == cfg, "the warm read did not return the stored pick"


def test_a_symlinked_freeze_entry_is_a_MISS(tmp_path, monkeypatch):
    """This entry chooses the kernel config every rank compiles; it must be the file we wrote."""
    from fold_cp_ops.distributed.workflows import trimul_autotune_policy as tp

    d = _freeze_env(tmp_path, monkeypatch)
    key = ("cp8", "N2048", "D256")
    tp._freeze_store("back", key, (128, 128, False, (1, 2, 1)), _Rank0())
    entry = list(d.glob("*.json"))[0]
    elsewhere = tmp_path / "elsewhere.json"
    elsewhere.write_text(entry.read_text())
    os.replace(str(entry), str(tmp_path / "orig.json"))
    os.symlink(str(elsewhere), str(entry))
    assert tp._freeze_load("back", key) is None, "a symlinked freeze entry was read"


@pytest.mark.parametrize("mode", [0o666, 0o620])
def test_a_WRITABLE_freeze_entry_is_a_MISS(tmp_path, monkeypatch, mode):
    """If others could write it, it may already say whatever they wanted it to say."""
    from fold_cp_ops.distributed.workflows import trimul_autotune_policy as tp

    d = _freeze_env(tmp_path, monkeypatch)
    key = ("cp8", "N2048", "D256")
    tp._freeze_store("back", key, (128, 128, False, (1, 2, 1)), _Rank0())
    entry = list(d.glob("*.json"))[0]
    os.chmod(entry, mode)
    assert tp._freeze_load("back", key) is None, f"a {mode:04o} freeze entry was read"


def test_a_REJECTED_freeze_root_writes_nothing_and_reads_a_miss(tmp_path, monkeypatch):
    """A refused root costs a re-sweep, never an error and never a write into a hostile directory."""
    from fold_cp_ops._internal import cache_security
    from fold_cp_ops.distributed.workflows import trimul_autotune_policy as tp

    hostile = tmp_path / "hostile"
    hostile.mkdir()
    os.chmod(hostile, 0o777)  # AFTER mkdir: the mode argument is umask-masked
    monkeypatch.setenv("CPO_DIST_AUTOTUNE_FREEZE_DIR", str(hostile))
    cache_security.reset_cache()
    try:
        key = ("cp8", "N2048", "D256")
        tp._freeze_store("back", key, (128, 128, False, (1, 2, 1)), _Rank0())
        assert not list(hostile.glob("*.json")), "a refused freeze root was written to"
        assert tp._freeze_load("back", key) is None
    finally:
        cache_security.reset_cache()


@pytest.mark.parametrize(
    "failure",
    ["rejected_root", "write_error"],
    ids=["a refused root", "an OSError mid-write"],
)
def test_EVERY_RANK_REACHES_THE_BARRIER_even_when_rank_0_cannot_write(
    tmp_path, monkeypatch, failure
):
    """THE test in this block: a failed freeze write must not strand the peers in the barrier.

    Purpose
        ``_freeze_store`` ends in a collective. Rank 0 does the writing; every other rank goes
        straight to the barrier and blocks. If ANY exception escapes rank 0's write -- a refused
        root, a full disk, a permission error -- rank 0 never arrives, and the job hangs until a
        watchdog kills it, with a traceback pointing at whatever the peers happened to be doing.
        A freeze that cannot be written must cost a re-sweep, not a deadlock.

    Semantics
        The barrier is monkeypatched to a recorder rather than run for real, because the property
        under test is CONTROL FLOW -- "was the barrier reached" -- and a recorder answers that
        deterministically in one process. A real two-rank run would demonstrate the hang only by
        hanging, which is not a test outcome anyone can collect.

        Both failure shapes are covered: one the code checks for (a refused root) and one it cannot
        anticipate (an ``OSError`` from the filesystem). The second is the one that matters, because
        it is the class of failure a future edit will introduce.
    """
    import torch.distributed as dist

    from fold_cp_ops._internal import cache_security
    from fold_cp_ops.distributed.workflows import trimul_autotune_policy as tp

    reached = []
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "barrier", lambda *a, **k: reached.append(True))

    if failure == "rejected_root":
        hostile = tmp_path / "hostile"
        hostile.mkdir()
        os.chmod(hostile, 0o777)
        monkeypatch.setenv("CPO_DIST_AUTOTUNE_FREEZE_DIR", str(hostile))
        cache_security.reset_cache()
    else:
        _freeze_env(tmp_path, monkeypatch)
        import tempfile as _tf

        def _boom(*a, **k):
            raise OSError("disk full")

        monkeypatch.setattr(_tf, "mkstemp", _boom)

    try:
        tp._freeze_store("back", ("cp8", "N2048", "D256"), (128, 128, False, (1, 2, 1)), _Rank0())
    finally:
        cache_security.reset_cache()

    assert reached == [True], (
        f"rank 0 did not reach the barrier after {failure}; every peer would block forever"
    )

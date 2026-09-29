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

"""Tests for ``tests/perf/pins.py`` -- the one pin format the timed gates may use.

Every check here is about a way a pin file could be WRONG while still parsing, because those are
the ones that produce a green gate comparing against a number nobody meant. Pure JSON handling, so
no GPU.

Also checks the shipped pin files themselves: they are data, and data that nothing validates is data
that drifts.
"""

import json
import pathlib

import pytest

from fold_cp_ops.testing.kernel_matrix import matrix_exempt

from tests.perf import pins

from tests.perf.pins import (
    HARVEST_TAG,
    SCHEMA,
    Pin,
    PinFile,
    emit_harvest,
    load,
    merge_harvest,
    parse_harvest,
    _VALID_ROUNDS,
    pin_path,
)

_PERF_DIR = pathlib.Path(__file__).parent


def _write(tmp_path, payload, name="test_benchmark_perf_x.py"):
    """Write a pin file beside a fake module and return that module's path."""
    mod = tmp_path / name
    mod.write_text("")
    (tmp_path / name.replace(".py", ".json")).write_text(json.dumps(payload))
    return str(mod)


def _ok(**over):
    """A minimal valid payload, with overrides."""
    base = {
        "schema": SCHEMA,
        "kernel": "x",
        "measured_on": "an idle H100",
        "cells": [{"key": [1, 2], "median_ms": 0.5}],
    }
    base.update(over)
    return base


@matrix_exempt("validates a data format; there is no kernel, shape or dtype to sweep")
def test_a_pin_file_round_trips(tmp_path):
    """The base case: keys come back as TUPLES, so a caller can look up with the tuple it built."""
    pf = load(_write(tmp_path, _ok()))
    assert (1, 2) in pf
    assert pf[(1, 2)].median_ms == 0.5
    assert pf[[1, 2]].median_ms == 0.5, "a list key must work too; JSON has no tuples"
    assert len(pf) == 1


@matrix_exempt("validates a data format; there is no kernel, shape or dtype to sweep")
def test_a_missing_pin_file_names_where_to_put_it(tmp_path):
    """The error has to say the path, or the fix is a guess."""
    mod = tmp_path / "test_benchmark_perf_y.py"
    mod.write_text("")
    with pytest.raises(FileNotFoundError, match=r"test_benchmark_perf_y\.json"):
        load(str(mod))


@matrix_exempt("validates a data format; there is no kernel, shape or dtype to sweep")
def test_a_wrong_schema_is_refused_not_best_effort_parsed(tmp_path):
    """A misread pin is a gate comparing against the wrong number, which looks like a pass."""
    with pytest.raises(ValueError, match=r"schema"):
        load(_write(tmp_path, _ok(schema=SCHEMA + 1)))


@matrix_exempt("validates a data format; there is no kernel, shape or dtype to sweep")
def test_an_unknown_key_is_refused_at_both_levels(tmp_path):
    """Adding a field and hoping something reads it is how a gate grows its own format."""
    with pytest.raises(ValueError, match=r"unknown top-level key"):
        load(_write(tmp_path, _ok(extra=1)))
    with pytest.raises(ValueError, match=r"unknown cell key"):
        load(_write(tmp_path, _ok(cells=[{"key": [1], "median_ms": 1.0, "tol": 1.1}])))


@matrix_exempt("validates a data format; there is no kernel, shape or dtype to sweep")
def test_provenance_is_required(tmp_path):
    """A pin nobody can re-harvest is a number nobody can ever move."""
    with pytest.raises(ValueError, match=r"measured_on"):
        load(_write(tmp_path, _ok(measured_on="")))


@matrix_exempt("validates a data format; there is no kernel, shape or dtype to sweep")
def test_a_duplicate_cell_is_refused(tmp_path):
    """Two pins for one cell means whichever is read last silently wins."""
    with pytest.raises(ValueError, match=r"duplicate cell"):
        load(
            _write(
                tmp_path,
                _ok(cells=[{"key": [1], "median_ms": 1.0}, {"key": [1], "median_ms": 2.0}]),
            )
        )


@matrix_exempt("validates a data format; there is no kernel, shape or dtype to sweep")
def test_a_nonsensical_number_is_refused_at_load(tmp_path):
    """Refused where the file is read, not several frames later inside an assertion."""
    with pytest.raises(ValueError, match=r"median_ms must be positive"):
        load(_write(tmp_path, _ok(cells=[{"key": [1], "median_ms": 0.0}])))
    with pytest.raises(ValueError, match=r"rel_std must be non-negative"):
        load(_write(tmp_path, _ok(cells=[{"key": [1], "median_ms": 1.0, "rel_std": -0.1}])))


@matrix_exempt("validates a data format; there is no kernel, shape or dtype to sweep")
def test_a_missing_spread_is_none_and_not_zero(tmp_path):
    """None means "not harvested yet"; zero would claim a perfectly stable cell.

    The difference decides whether a gate may be reference-calibrated, so collapsing them would
    silently give an unharvested cell a zero-width band.
    """
    pf = load(_write(tmp_path, _ok()))
    assert pf[(1, 2)].rel_std is None
    assert not pf.calibrated
    assert Pin(median_ms=1.0, rel_std=0.0).rel_std == 0.0


@matrix_exempt("validates a data format; there is no kernel, shape or dtype to sweep")
def test_calibrated_is_all_or_nothing(tmp_path):
    """A half-calibrated gate would correct some numbers and not others in one run.

    They would then not be comparable with each other, which is worse than correcting none.
    """
    payload = _ok(
        cells=[{"key": [1], "median_ms": 1.0, "rel_std": 0.01}, {"key": [2], "median_ms": 2.0}]
    )
    assert not load(_write(tmp_path, payload)).calibrated
    payload["cells"][1]["rel_std"] = 0.02
    assert load(_write(tmp_path, payload)).calibrated


@matrix_exempt("a property of the shipped data files, one assertion per file")
@pytest.mark.parametrize(
    "path", sorted(_PERF_DIR.glob("test_benchmark_perf_*.json")), ids=lambda p: p.name
)
def test_every_shipped_pin_file_validates(path):
    """The pin files that ship are data, and data nothing validates is data that drifts.

    Also checks the pairing in both directions: a pin file with no test would be dead numbers, and
    the conftest hook already catches a test with no pin file.
    """
    pf = PinFile(path, json.loads(path.read_text()))
    assert len(pf) > 0, f"{path.name} pins nothing"
    assert pf.measured_on
    assert path.with_suffix(".py").exists(), (
        f"{path.name} has no test module; a pin file nothing reads is a number nobody maintains"
    )


@matrix_exempt("a property of the shipped data files; nothing to sweep")
def test_every_timed_gate_has_a_pin_file():
    """The other direction, checked here as well as by the conftest hook.

    The hook fires at collection and is the enforcement; this states the same invariant as a
    readable assertion, so someone reading the tests can see the rule without reading a hook.
    """
    for mod in sorted(_PERF_DIR.glob("test_benchmark_perf_*.py")):
        assert pin_path(str(mod)).exists(), (
            f"{mod.name} is a timed perf gate with no {pin_path(str(mod)).name}"
        )


# -- the harvest line -------------------------------------------------------------------------
@matrix_exempt("formats one line of text; there is no kernel, shape or dtype to sweep")
def test_a_harvest_line_starts_its_own_line(capsys):
    """The emitted line begins with a newline, so ``^PINHARVEST`` can never miss one.

    This is the regression this test exists for, and it is worth more than it looks. Under ``-s``
    pytest writes its per-test progress dot WITHOUT a trailing newline, so a bare ``print`` from a
    gate lands as ``.PINHARVEST ...``. A merge script anchored at the start of a line then reads
    back only the cells that happened to follow a real newline -- and a 16-cell gate whose every
    cell emitted was counted as 1. Nothing fails in that scenario: the harvest exits 0, the tests
    pass, and the missing cells look like cells that were never measured.
    """
    emit_harvest("/some/dir/test_benchmark_perf_x.py", ("shape", 4096, 1), 0.0668, 0.0123)
    out = capsys.readouterr().out
    assert out.startswith("\n"), "a harvest line that shares a line with pytest's dot is unfindable"
    assert len([ln for ln in out.splitlines() if ln.startswith(HARVEST_TAG)]) == 1


@matrix_exempt("parses text into numbers; there is no kernel, shape or dtype to sweep")
def test_a_harvest_line_is_found_even_sharing_a_line_with_pytests_progress_dot():
    """The regression, stated from the READING side as well as the writing side.

    Both halves are pinned because either alone permits the failure: an emitter that stopped adding
    the newline, or a parser anchored at the start of a line. The reader is the half that already
    cost a day -- it read 1 cell of 16 and reported success.
    """
    got = parse_harvest(f'..F.{HARVEST_TAG} test_benchmark_perf_x ["a",1] 0.5 0.02')
    assert got == {"test_benchmark_perf_x": {("a", 1): (0.5, 0.02, None)}}


@matrix_exempt("parses text into numbers; there is no kernel, shape or dtype to sweep")
def test_harvest_parsing_ignores_prose_and_refuses_a_malformed_line():
    """Human-readable output passes through; a BROKEN harvest line stops the merge.

    The asymmetry is the point. A gate's own prints are noise and skipping them is correct, but a
    harvest line that does not parse is a cell that would silently keep its stale pin while the
    merge reported it refreshed.
    """
    assert parse_harvest("    ('shape', 4096): 0.0668 ms  (514.2 TFLOP/s)\n17 passed\n") == {}
    with pytest.raises(ValueError, match=r"malformed harvest line"):
        parse_harvest(f"{HARVEST_TAG} test_x [1] 0.5")
    with pytest.raises(ValueError, match=r"unreadable key"):
        parse_harvest(f"{HARVEST_TAG} test_x (1,2) 0.5 0.02")


@matrix_exempt("merges numbers into a payload; there is no kernel, shape or dtype to sweep")
def test_a_merge_updates_in_place_keeps_notes_and_appends_new_cells():
    """The merge touches the two measured fields and nothing else.

    ``note`` carries the hand-written reason a cell exists and why its number looks the way it does.
    A merge that dropped it would delete the only record of that on every re-harvest, which is how a
    pin file decays into an unexplained list of numbers.
    """
    payload = _ok(
        cells=[
            {"key": [1, 2], "median_ms": 0.5, "note": "the M-tail cell"},
            {"key": [3, 4], "median_ms": 1.5},
        ]
    )
    merged, missing = merge_harvest(payload, {(1, 2): (0.6, 0.03, None), (9, 9): (2.0, 0.01, None)})
    assert [c["key"] for c in merged["cells"]] == [[1, 2], [3, 4], [9, 9]]
    assert merged["cells"][0] == {
        "key": [1, 2],
        "median_ms": 0.6,
        "rel_std": 0.03,
        "note": "the M-tail cell",
    }
    assert merged["cells"][1] == {"key": [3, 4], "median_ms": 1.5}  # untouched, still uncalibrated
    assert missing == [(3, 4)]
    assert payload["cells"][0]["median_ms"] == 0.5, "the input payload must not be mutated"


@matrix_exempt("merges numbers into a payload; there is no kernel, shape or dtype to sweep")
def test_a_partial_merge_leaves_the_file_uncalibrated_and_says_which_cells():
    """A half-harvested file must not read as calibrated, and the caller must be told why.

    This is the guard against the exact mistake that produced the bad merge: a run that covered some
    cells was treated as a completed harvest. `missing` is what makes "did this cover everything"
    answerable without diffing the file.
    """
    payload = _ok(cells=[{"key": [1, 2], "median_ms": 0.5}, {"key": [3, 4], "median_ms": 1.5}])
    merged, missing = merge_harvest(payload, {(1, 2): (0.6, 0.03, None)})
    assert missing == [(3, 4)]
    assert not PinFile(pathlib.Path("x.json"), merged).calibrated


@matrix_exempt("merges numbers into a payload; there is no kernel, shape or dtype to sweep")
def test_provenance_is_replaced_on_a_total_reharvest_and_appended_on_a_partial_one():
    """Provenance follows the CELLS, not the call: replaced when none of the old numbers survive.

    A total re-harvest re-measures every cell, so the old text describes nothing that is still in
    the file and keeping it would name a machine none of the numbers came from. A PARTIAL one
    leaves some cells at their old values, and those were never measured under the new conditions --
    overwriting there silently re-attributes them.

    This is not hypothetical. An earlier revision of the back-A2A pin file recorded "moved a back
    median 15.00 -> 25.96 ms and flipped the elected autotune config" in this field; a later partial
    re-harvest replaced it, and that sentence -- the evidence for one-launch-per-cell -- survived
    only in a shell comment and in git history.
    """
    two = [{"key": [1, 2], "median_ms": 0.5}, {"key": [3, 4], "median_ms": 1.5}]

    # None -> untouched, whatever the coverage.
    assert merge_harvest(_ok(measured_on="an idle H100"), {})[0]["measured_on"] == "an idle H100"

    # TOTAL: every cell re-measured -> replace.
    payload = _ok(cells=two, measured_on="an idle H100")
    merged, missing = merge_harvest(
        payload, {(1, 2): (0.6, 0.01, None), (3, 4): (1.6, 0.01, None)}, "an idle H200"
    )
    assert missing == [] and merged["measured_on"] == "an idle H200"

    # PARTIAL: cell (3,4) keeps its old number -> the old provenance must survive alongside.
    merged, missing = merge_harvest(payload, {(1, 2): (0.6, 0.01, None)}, "an idle H200")
    assert missing == [(3, 4)]
    assert merged["measured_on"] == "an idle H100\nan idle H200"

    # An exact repeat does not accumulate: re-running the same partial harvest twice must not
    # grow the field without bound.
    again, _ = merge_harvest(merged, {(1, 2): (0.7, 0.01, None)}, "an idle H200")
    assert again["measured_on"] == "an idle H100\nan idle H200"


@matrix_exempt("formats one line of text; there is no kernel, shape or dtype to sweep")
def test_a_harvest_line_names_its_pin_file_and_round_trips_its_numbers(capsys):
    """The line carries everything the merge needs: which file, which cell, and both numbers.

    The stem rather than the full path, because the merge writes back to the sibling JSON and an
    absolute path harvested on one machine would not resolve on another. The key goes through JSON
    so a tuple of mixed str/int survives verbatim -- ``str(key)`` would emit Python repr, which
    ``json.loads`` cannot read back.
    """
    emit_harvest("/some/dir/test_benchmark_perf_x.py", ("shape", 4096, 1), 0.0668, 0.0123)
    tag, stem, key, median, rel_std = capsys.readouterr().out.strip().split(" ", 4)
    assert (tag, stem) == (HARVEST_TAG, "test_benchmark_perf_x")
    assert json.loads(key) == ["shape", 4096, 1]
    assert (float(median), float(rel_std)) == (0.0668, 0.0123)


@matrix_exempt("parses one line of text; there is no kernel, shape or dtype to sweep")
def test_a_harvest_line_survives_another_ranks_text_appended_to_it():
    """A multi-rank harvest interleaves stdout mid-line; the record must still be read.

    Eight to sixteen ranks print to one unsynchronized stdout, so a gate's own human-readable
    line lands appended to an emit line whenever two ranks race. Measured on the 2026-08-21
    world-16 harvest: 104 of 1692 lines, every one carrying an intact five-token prefix. The
    parser rejected all 104, which cost nothing only because per-rank duplication happened to
    cover every affected cell from a clean print -- luck that a world-2 harvest would not get.
    """
    raw = (
        'PINHARVEST test_benchmark_perf_x ["back_coupled","cp4",128,2048] 1.151200 0.019846'
        "    ('back_coupled', 'cp4', 128, 2048): median=1.15120 rel_std=0.0198"
    )
    parsed = parse_harvest(raw)
    assert parsed == {
        "test_benchmark_perf_x": {("back_coupled", "cp4", 128, 2048): (1.1512, 0.019846, None)}
    }


@matrix_exempt("parses one line of text; there is no kernel, shape or dtype to sweep")
def test_a_truncated_harvest_line_is_still_refused_loudly():
    """Tolerating trailing text must not weaken the refusal that protects a stale pin.

    A silently dropped line is a cell that keeps its old median while the merge report says it
    was refreshed, so the 4th and 5th tokens are required to be two floats. This is the check
    that keeps the tolerance above from degrading into "accept anything starting with the tag".
    """
    with pytest.raises(ValueError, match="unreadable median/rel_std"):
        parse_harvest('PINHARVEST test_benchmark_perf_x ["shape",4096] 1.15 not_a_float')
    with pytest.raises(ValueError, match="malformed harvest line"):
        parse_harvest('PINHARVEST test_benchmark_perf_x ["shape",4096]')


@matrix_exempt("resolves a file path; there is no kernel, shape or dtype to sweep")
def test_a_pin_file_outside_this_directory_is_found_not_skipped(tmp_path, monkeypatch):
    """The merger must reach ``tests/distributed/perf/``, not only its own directory.

    An earlier ``_main`` resolved every pin file as ``__file__.parent / f"{stem}.json"``, so the
    A2A gates -- whose pin files live one directory over -- hit the not-found branch and were
    skipped SILENTLY with exit 0. Measured: a real harvest line for
    ``test_benchmark_perf_gemm_sm90_a2a`` printed ``SKIP ... no ....json`` and returned 0, which
    would have discarded a 183-cell harvest while reading as a clean merge.
    """
    root = tmp_path / "tests"
    (root / "perf").mkdir(parents=True)
    (root / "distributed" / "perf").mkdir(parents=True)
    monkeypatch.setattr(pins, "_repo_tests_root", lambda: root)
    (root / "distributed" / "perf" / "far.json").write_text('{"cells": []}')
    assert pins._resolve_pin_file("far") == root / "distributed" / "perf" / "far.json"
    with pytest.raises(FileNotFoundError):
        pins._resolve_pin_file("nowhere")


@matrix_exempt("resolves a file path; there is no kernel, shape or dtype to sweep")
def test_a_pin_stem_present_in_two_directories_is_refused_not_picked(tmp_path, monkeypatch):
    """Two pin files sharing a basename must raise, never be resolved by iteration order.

    Searching several directories makes a duplicate basename newly possible, and picking the
    first hit would merge one gate's medians into the other's file -- a wrong-file write no diff
    review would flag, because both files legitimately change during a harvest.
    """
    root = tmp_path / "tests"
    (root / "perf").mkdir(parents=True)
    (root / "distributed" / "perf").mkdir(parents=True)
    monkeypatch.setattr(pins, "_repo_tests_root", lambda: root)
    (root / "perf" / "dup.json").write_text('{"cells": []}')
    (root / "distributed" / "perf" / "dup.json").write_text('{"cells": []}')
    with pytest.raises(RuntimeError, match="ambiguous pin stem"):
        pins._resolve_pin_file("dup")


# ── the harvested round count, carried from emit through parse to the merged file ──────────────
@matrix_exempt("asserts a metadata round trip; no kernel, shape or dtype to sweep")
def test_the_round_count_round_trips_from_emit_through_parse_to_the_merged_cell(capsys):
    """A count that survives emit but is dropped by parse is worse than one never emitted.

    `assert_cell` refuses a pin whose recorded count differs from the derived one, so a count lost in
    transit reads as "harvested at the historical 35" and fails every budgeted cell forever, with the
    failure pointing at the gate rather than at the merger.
    """
    emit_harvest("/d/test_benchmark_perf_x.py", ("s", 4096), 0.0668, 0.0123, 8)
    got = parse_harvest(capsys.readouterr().out)
    assert got == {"test_benchmark_perf_x": {("s", 4096): (0.0668, 0.0123, 8)}}
    merged, _ = merge_harvest(
        {"schema": 1, "kernel": "x", "measured_on": "before", "cells": []},
        got["test_benchmark_perf_x"],
    )
    assert merged["cells"][0]["rounds"] == 8


@matrix_exempt("asserts a metadata round trip; no kernel, shape or dtype to sweep")
def test_omitting_the_round_count_leaves_the_line_exactly_as_it_was():
    """The eleven gates that have not adopted budgeted rounds must emit the four-field line.

    Appended rather than inserted for the same reason: a reader written against the old shape still
    parses every line it used to.
    """
    assert parse_harvest(f'{HARVEST_TAG} test_benchmark_perf_x ["a",1] 0.5 0.02') == {
        "test_benchmark_perf_x": {("a", 1): (0.5, 0.02, None)}
    }


@matrix_exempt("asserts parse tolerance; no kernel, shape or dtype to sweep")
def test_a_FOREIGN_sixth_token_is_not_mistaken_for_a_round_count():
    """Everything past token five may be another rank's text -- that is the premise of the tolerance.

    A multi-rank harvest has 8-16 ranks writing one unsynchronized stdout, and a gate's own
    human-readable line lands appended to an emit line whenever two of them interleave mid-line
    (measured: 104 of 1692 lines on the world-16 harvest). So "there is a sixth token" cannot mean
    "it is mine". Requiring a bare integer that is also a legal rung is what separates them.
    """
    for tail in ("cp16", "42", "7", "0", "(514.2", "35.0"):
        got = parse_harvest(f'{HARVEST_TAG} test_benchmark_perf_x ["a",1] 0.5 0.02 {tail}')
        assert got["test_benchmark_perf_x"][("a", 1)][2] is None, (
            f"{tail!r} was read as a round count; only a bare legal rung may be"
        )
    # ... and a legal rung IS read
    got = parse_harvest(f'{HARVEST_TAG} test_benchmark_perf_x ["a",1] 0.5 0.02 12')
    assert got["test_benchmark_perf_x"][("a", 1)][2] == 12


@matrix_exempt("asserts two constants agree; no kernel, shape or dtype to sweep")
def test_the_valid_rung_set_matches_calibrations():
    """`pins._VALID_ROUNDS` duplicates `calibration._ROUND_RUNGS` to avoid an import cycle.

    A duplicated constant is acceptable only while something fails when the copies drift. Without
    this, adding a rung in `calibration` would make `parse_harvest` silently discard every harvest
    line carrying it -- a count lost in transit, which is the failure the round-trip test above
    describes, arriving by a route that test cannot see.
    """
    from tests.perf.calibration import _ROUND_RUNGS

    assert _VALID_ROUNDS == frozenset(_ROUND_RUNGS)

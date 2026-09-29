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


"""Tests for `benchmark.distributed.harness.compare_runtime`.

Driven with literal JSON payloads written to tmp dirs -- no GPU, no Slurm, no harness run. The
numbers in `_OURS_REAL` are copied from a real venue A cp=(2,8) sweep so the semantics test below
is anchored to a measurement rather than to an invented pair.
"""

import json
import os

import pytest

from benchmark.distributed.harness.compare_runtime import (
    DEFAULT_BAR,
    FAILED_OURS,
    FASTER,
    MISSING_MAIN,
    MISSING_OURS,
    NEUTRAL,
    REGRESSION,
    UNRESOLVABLE,
    compare,
    load_sweep,
    render,
)
from fold_cp_ops.testing.kernel_matrix import matrix_exempt

pytestmark = matrix_exempt(
    "the subject is host-side analysis of emitted JSON -- no kernel is launched and no shape axis "
    "exists; the shapes appearing here are payload CONTENT, not a swept axis"
)

# From the live sweep: time_ms is NOT median(raw_ms), because time_ms is the slowest PE's median
# and raw_ms is the writing rank's own rounds. 9.4935 vs 9.4702 is that gap, not rounding.
_REAL_TIME_MS = 9.4934720993042
_REAL_RAW = [
    9.532480239868164,
    9.487680435180664,
    9.470239639282227,
    9.450912475585938,
    9.460096359252930,
]


def _cell(target="front", *, time_ms=10.0, raw=None, cfg=None, status="ok", N=2048, Dloc=8):
    """One harness cell, shaped like the real emitter's.

    Args:
        target: Cell target name; part of the pairing key.
        time_ms: The slowest-PE median the comparator must use.
        raw: This rank's per-round samples. Defaults to a tight, noise-free series so a test that
            is not ABOUT dispersion cannot accidentally trip the UNRESOLVABLE path.
        cfg: The winning config, for the mismatch check.
        status: Cell status; anything but "ok" must not be compared.
        N, Dloc: Pairing-key fields.
    """
    raw = [time_ms] * 5 if raw is None else raw
    return {
        "target": target,
        "role": "target",
        "N": N,
        "cp0": 2,
        "cp1": 8,
        "Dloc": Dloc,
        "status": status,
        "reason": "" if status == "ok" else "some reason",
        "winner": {"time_ms": time_ms, "raw_ms": raw, "cfg": cfg if cfg is not None else {}},
    }


def _write(tmp_path, name, cells, N=2048):
    """Write one ``bench_N*.json`` and return its directory."""
    d = tmp_path / name
    d.mkdir(exist_ok=True)
    (d / f"bench_N{N}.json").write_text(json.dumps({"N": N, "cp0": 2, "cp1": 8, "cells": cells}))
    return str(d)


def test_the_ratio_uses_time_ms_and_not_a_statistic_of_raw_ms(tmp_path):
    """The measurement is the SLOWEST PE's median, which `raw_ms` cannot reproduce.

    `benchmark_single` medians this rank's rounds and THEN ``all_reduce(MAX)``, so ``time_ms``
    belongs to whichever rank was slowest while ``raw_ms`` belongs to whichever rank wrote the
    file. A comparator that recomputed the median from ``raw_ms`` would compare the two trees'
    luckiest ranks and would look right on every fixture where the two happen to agree -- which is
    why this fixture uses the real numbers, where they do not.
    """
    ours = _write(tmp_path, "ours", [_cell(time_ms=_REAL_TIME_MS, raw=_REAL_RAW)])
    main = _write(tmp_path, "main", [_cell(time_ms=_REAL_TIME_MS, raw=_REAL_RAW)])
    c = compare(ours, main).cells[0]
    assert c.ours_ms == _REAL_TIME_MS, "must read winner.time_ms verbatim"
    assert c.ratio == 1.0
    # the gap the test exists to protect: a median-of-raw implementation would still give 1.0 here,
    # so assert the reported number is time_ms and NOT the raw median.
    import statistics

    assert c.ours_ms != statistics.median(_REAL_RAW)
    assert c.ours_pe_penalty == pytest.approx(_REAL_TIME_MS / statistics.median(_REAL_RAW) - 1)


@pytest.mark.parametrize(
    "ours_ms,expect",
    [(10.0, NEUTRAL), (10.4, NEUTRAL), (10.6, REGRESSION), (9.6, NEUTRAL), (9.4, FASTER)],
    ids=["equal", "inside_bar", "over_bar", "inside_bar_fast", "under_bar"],
)
def test_the_bar_decides_neutral_regression_faster(tmp_path, ours_ms, expect):
    """+-5% around 1.0, with both boundaries exercised rather than only the slow side.

    A one-sided test would pass an implementation that reported every improvement as NEUTRAL, and
    "we got faster" is a result the owner's bar explicitly admits ("neutral or BETTER").
    """
    ours = _write(tmp_path, "ours", [_cell(time_ms=ours_ms)])
    main = _write(tmp_path, "main", [_cell(time_ms=10.0)])
    assert compare(ours, main, bar=DEFAULT_BAR).cells[0].verdict == expect


def test_a_cell_noisier_than_the_bar_is_unresolvable_not_neutral(tmp_path):
    """A cell whose own samples disperse more than the claim cannot support the claim.

    Rounding a dispersed cell to NEUTRAL reports a verdict the data does not carry. Same principle
    as a guard with a "did not fire" state: a bar with no "cannot tell" passes everything it cannot
    see.

    **The FIXTURE changed when `_noise` became the relative MAD rather than `(max-min)/median`, and
    the old one is kept below as the second case because it is now a DIFFERENT finding.** The
    original `[1.83, 1.87, 1.87, 1.92, 1.94]` was taken from a real cell and spans 5.9%, but its
    MAD is ~1.3% -- most samples sit near the median and only the ends are far out. Under a range
    rule that is "too noisy to judge"; under a median-dispersion rule the median is well determined
    and NEUTRAL is the right answer. So this test now needs samples that are genuinely dispersed,
    not merely wide at the extremes.
    """
    noisy = [1.70, 1.80, 1.90, 2.00, 2.10]  # MAD ~5.3% of the median: dispersed THROUGHOUT
    ours = _write(tmp_path, "ours", [_cell(time_ms=1.90, raw=noisy)])
    main = _write(tmp_path, "main", [_cell(time_ms=1.90, raw=noisy)])
    c = compare(ours, main).cells[0]
    assert c.ratio == 1.0 and c.verdict == UNRESOLVABLE
    assert "cannot support a verdict" in c.note


def test_a_wide_but_UNIMODAL_cell_resolves_because_its_median_is_well_determined(tmp_path):
    """The former fixture, which the statistic change deliberately reclassifies.

    `[1.83, 1.87, 1.87, 1.92, 1.94]` spans 5.9% end-to-end but its MAD is ~1.3%. A range rule calls
    it unresolvable; the median it is judging on is in fact solid. Measured consequence on real
    data: `N3008_D512` had 14 of 15 rounds inside 0.5% and one at +11%, and the range rule refused
    a verdict on a cell whose median ratio was 0.992.

    Pinned as its own test so the reclassification is a DECISION with a name, not a silently
    edited fixture in the test above.
    """
    wide = [1.83, 1.87, 1.87, 1.92, 1.94]
    ours = _write(tmp_path, "ours", [_cell(time_ms=1.9177, raw=wide)])
    main = _write(tmp_path, "main", [_cell(time_ms=1.9177, raw=wide)])
    c = compare(ours, main).cells[0]
    assert c.verdict == NEUTRAL, (
        "a wide-but-unimodal cell must resolve under a median-dispersion rule"
    )


def test_a_balanced_MIXTURE_inside_the_bar_is_flagged_by_its_TAIL(tmp_path):
    """THE CASE THE ONE-NUMBER RULE PASSES SILENTLY, which is why `_tail` is reported separately.

    Real cell, `N4096_D256 front_a2a` under contention: ten rounds at 14.0-14.4 then 15.4, 25.0,
    30.1, 73.4, 169.1. Its MAD is **0.97%** -- comfortably inside a 5% bar -- because most samples
    sit near the median. Its `max/min` is **12.06**. No single spread statistic separates that from
    a tight cell, so the median is a summary of a MIXTURE and must not read as clean.

    The flag is deliberately NOT a verdict: contention produces this shape too, and in the case
    above it was four benchmark drivers racing one allocation rather than a kernel defect.
    """
    mixture = [
        14.0,
        14.1,
        14.1,
        14.2,
        14.2,
        14.2,
        14.3,
        14.3,
        14.4,
        14.4,
        15.4,
        25.0,
        30.1,
        73.4,
        169.1,
    ]
    ours = _write(tmp_path, "ours", [_cell(time_ms=14.24, raw=mixture)])
    main = _write(tmp_path, "main", [_cell(time_ms=14.22, raw=mixture)])
    c = compare(ours, main).cells[0]
    assert c.ours_noise < 0.05, "precondition: this cell's dispersion is INSIDE the bar"
    assert c.ours_tail > 10.0, "precondition: its tail is enormous"
    assert "HEAVY TAIL" in c.note, "a mixture inside the bar must still be flagged"


def test_a_config_mismatch_is_flagged_because_the_ratio_compares_two_kernels(tmp_path):
    """Different winning configs make the ratio a parity finding, not a timing one."""
    ours = _write(tmp_path, "ours", [_cell(cfg={"tile_M": 128, "tile_N": 16})])
    main = _write(tmp_path, "main", [_cell(cfg={"tile_M": 256, "tile_N": 128})])
    rep = compare(ours, main)
    c = rep.cells[0]
    assert c.cfg_match is False and "CONFIG MISMATCH" in c.note
    assert rep.config_mismatches == (c,)
    assert "CONFIG MISMATCH" in render(rep)


def test_the_denominator_separates_missing_from_failed_from_compared(tmp_path):
    """Four outcomes, four states. A cell that did not run must not be absent from the table.

    This is the check that makes a truncated sweep legible: all-NEUTRAL over two cells and
    all-NEUTRAL over twelve print the same verdicts, and only the counts distinguish them.
    """
    ours = _write(
        tmp_path,
        "ours",
        [
            _cell("front"),
            _cell("only_ours"),
            _cell("broken", status="error"),
        ],
    )
    main = _write(
        tmp_path,
        "main",
        [
            _cell("front"),
            _cell("only_main"),
            _cell("broken"),
        ],
    )
    rep = compare(ours, main)
    counts = rep.counts()
    assert counts[NEUTRAL] == 1
    assert counts[MISSING_MAIN] == 1 and counts[MISSING_OURS] == 1
    assert counts[FAILED_OURS] == 1
    assert rep.compared == 1 and len(rep.cells) == 4
    out = render(rep)
    assert "DENOMINATOR: 4 cells seen, 1 compared" in out
    for name in ("only_ours", "only_main", "broken"):
        assert name in out, f"{name} must appear in the table rather than being filtered out"


def test_an_empty_side_reads_as_no_data_not_as_no_regression(tmp_path):
    """Zero files on a side is the failure mode a ratio table cannot show by itself."""
    ours = _write(tmp_path, "ours", [_cell()])
    empty = str(tmp_path / "main_empty")
    os.makedirs(empty, exist_ok=True)
    rep = compare(ours, empty)
    assert rep.main_files == 0 and rep.compared == 0
    assert rep.cells[0].verdict == MISSING_MAIN
    assert "NOT 'no regression'" in render(rep)


def test_a_missing_directory_does_not_raise(tmp_path):
    """A path that does not exist yields no data, not a traceback -- the sweep may not have run."""
    rep = compare(str(tmp_path / "nope"), str(tmp_path / "also_nope"))
    assert rep.cells == () and rep.ours_files == 0 and rep.main_files == 0


def _with_configs(cell):
    """The same cell as the driver actually writes it: ``winner`` AND the ``configs`` list it won.

    `_cell` omits ``configs`` because `compare_runtime` never reads it. The scaling aggregator reads
    ONLY ``configs``, so a fixture without it would make the comparison below VACUOUS -- the
    aggregator would return zero for the wrong reason and the test would pass while proving nothing.

    Args:
        cell: A `_cell` payload. Must carry a ``winner``; the config list is derived from it, so the
            two views of the cell cannot drift apart within one fixture.

    Returns:
        A new dict; `cell` is not mutated.
    """
    return {**cell, "configs": [cell["winner"]]}


def test_the_scaling_aggregators_loader_cannot_read_a_driver_out_dir(tmp_path):
    """Pin WHY `load_sweep` exists beside `aggregate_scaling._load` rather than reusing it.

    Both read ``bench_N*.json`` written by the same `driver._flush_json`, which makes them look
    like duplicates and produced a live instruction to retire this module in favour of the other.
    They consume DIFFERENT things and differ by exactly one directory level:

      * `driver._flush_json` always writes FLAT into whatever ``--out-dir`` it is handed.
      * `run_scaling_cell.sh` hands it ``$SC_OUTBASE/$SC_TAG`` -- one tag dir per isolated cell --
        so `aggregate_scaling._load` walks the BASE of a per-cell grid and iterates ``<tag>/``.
      * A cross-tree comparison has ONE out-dir per side, so `load_sweep` reads that dir directly.

    Pointed at a driver out-dir the aggregator therefore finds nothing, silently -- the exact
    shape of a probe that runs zero cells and exits clean. The positive control below is what
    separates "the layouts differ" from "the fixture was malformed".
    """
    from benchmark.distributed.harness import aggregate_scaling as AG

    flat = _write(tmp_path, "ours", [_with_configs(_cell())])

    # POSITIVE CONTROL: the aggregator's own layout, base/<tag>/bench_N*.json -- it must find the
    # cell there, or the zero below says nothing about the level and everything about the payload.
    base = tmp_path / "grid"
    (base / "cell_tag").mkdir(parents=True)
    (base / "cell_tag" / "bench_N2048.json").write_text(
        (tmp_path / "ours" / "bench_N2048.json").read_text()
    )
    assert AG._load(str(base)), "control: the aggregator must read its own base/<tag>/ layout"

    assert AG._load(flat) == {}, "the aggregator reads one level too high for a driver out-dir"
    cells, n_files = load_sweep(flat)
    assert n_files == 1 and len(cells) == 1, (
        "load_sweep reads the layout the driver actually writes"
    )


def test_the_pairing_key_keeps_dloc_which_the_scaling_key_drops(tmp_path):
    """Two sweeps differing only in D must stay two cells -- the aggregator's key collapses them.

    `aggregate_scaling._load` keys on ``(cp0, cp1, N, suffix, base, has_mask)``: no ``Dloc`` and no
    tag. That is correct for a scaling table, where D is fixed per table and the tag identifies the
    cell. For a cross-tree D-sweep it is fatal, and not by a small margin -- measured on the real
    ``w8plan/diag`` tree it collapsed 29 config rows on disk to 9 records, with the surviving entry
    for ``(2, 8, 2048, front)`` coming from ``sweep.main_oracle.D512``. Under that key a comparator
    reads the BASELINE's number out of the CANDIDATE's slot and reports a ratio of 1.0.
    """
    from benchmark.distributed.harness import aggregate_scaling as AG

    base = tmp_path / "grid"
    for tag, dloc, ms in (("D128", 16, 10.0), ("D512", 64, 40.0)):
        (base / tag).mkdir(parents=True)
        cell = _with_configs(_cell(time_ms=ms, Dloc=dloc))
        (base / tag / "bench_N2048.json").write_text(
            json.dumps({"N": 2048, "cp0": 2, "cp1": 8, "cells": [cell]})
        )
    assert len(AG._load(str(base))) == 1, "the scaling key drops Dloc, so the two D collapse to one"

    merged, _ = load_sweep(str(base / "D128"))
    other, _ = load_sweep(str(base / "D512"))
    keys = set(merged) | set(other)
    assert len(keys) == 2, "the comparator's key keeps Dloc, so the two D stay distinct"
    assert {k[-1] for k in keys} == {16, 64}

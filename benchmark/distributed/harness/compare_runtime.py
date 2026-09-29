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


"""Pair two trees' harness sweeps into a per-cell runtime verdict.

What it reads
    The harness's ``bench_N*.json``, one file per token extent, as emitted by
    `driver.run_matrix`. The schema was read off a real venue A sweep rather than assumed:
    ``{N, cp0, cp1, cells: [{target, role, N, cp0, cp1, Dloc, status, winner, ...}]}`` with the
    timing on ``winner``.

Which number is the measurement
    **``winner.time_ms``, and never a statistic recomputed from ``winner.raw_ms``.** They are not
    the same quantity and the difference is not rounding. `bench_timing.benchmark_single` takes
    THIS rank's median of its own rounds and then ``all_reduce(MAX)`` across ranks, so ``time_ms``
    is the SLOWEST PE's median -- the real distributed cost, bit-identical on every rank --
    while ``raw_ms`` stays a per-rank diagnostic. Measured on the live file: the writing rank's
    ``median(raw_ms)`` is 9.4702 ms while ``time_ms`` is 9.4935 ms, because a different rank was
    slower. A comparator that medianed ``raw_ms`` would silently compare two trees' luckiest ranks.

What ``raw_ms`` is still good for
    A LOWER BOUND on dispersion. The within-rank spread is visible; cross-rank spread is not,
    except as the one-sided gap ``time_ms / median(raw_ms) - 1``. Both are reported, because a
    ratio without a noise estimate beside it invites a verdict the data cannot support.
"""

from __future__ import annotations

import dataclasses
import glob
import json
import os
import statistics

#: Default neutrality bar, as a relative deviation from ratio 1.0.
#:
#: **THE VALUE IS PROVISIONAL AND ITS ORIGINAL JUSTIFICATION HAS BEEN WITHDRAWN.** It was calibrated
#: from a sweep later found to have 2-4 benchmark drivers racing one allocation, so the two numbers
#: it cited (0.86% on ``front``, 5.92% on ``front_a2a``) measured CONTENTION, not the instrument.
#: They are removed rather than restated, because a plausible-looking number in a comment rots
#: silently -- nothing re-derives it.
#:
#: **What is measured on clean data** (25 cells, one driver, verified single-step, hybrid cp=(2,8)):
#:
#:     front       MAD median 0.097%  max 0.303%   tail median 1.007  max 1.015
#:     front_a2a   MAD median 0.530%  max 3.042%   tail median 1.029  max 1.281
#:
#: **That is WITHIN-RUN dispersion and it is the WRONG quantity to set this bar from**, which is why
#: no tighter value is set here. The bar must cover RUN-TO-RUN reproducibility -- the same cell,
#: same shape, a fresh launch -- and cross-process variation is known in this repo to dwarf
#: within-run: a recorded worst cell moved 35% across processes at 1 sample and 5.1% at 25, while
#: its within-run spread stayed small. Deriving a 0.5% bar from a 0.1% MAD would produce a gate that
#: fires on ordinary launch-to-launch drift and would be read as a regression.
#:
#: **So 5% stands as a PLACEHOLDER, and it is loose in the direction that matters**: at a within-run
#: MAD of ~0.1% the instrument can see far smaller differences than the bar admits, so a real 2-4%
#: regression currently passes as NEUTRAL. That is the wrong failure direction for a parity gate.
#: Closing it needs a repeat-launch measurement (the same subset re-run in a SECOND launch on a
#: quiet allocation), which has not been taken. Do not tighten this from within-run numbers.
#:
#: The "cannot tell" state is kept for the reason it was introduced -- a bar with no UNRESOLVABLE
#: verdict passes everything it cannot see -- but see `_noise`/`_tail`: the one-number rule it was
#: built on passes exactly the cell that most needs flagging (MAD 0.97%, tail 12.06).
DEFAULT_BAR = 0.05

NEUTRAL, REGRESSION, FASTER = "NEUTRAL", "REGRESSION", "FASTER"
UNRESOLVABLE = "UNRESOLVABLE"

#: ``max/min`` at or above which a cell is flagged as non-unimodal even when it is resolvable.
#: 1.5 is well clear of this venue's observed clean tails (1.00-1.06 on quiet nodes) and well
#: below the 2.26 of the contended cell that motivated the flag.
_TAIL_FLAG = 1.5
MISSING_OURS, MISSING_MAIN = "MISSING_OURS", "MISSING_MAIN"
FAILED_OURS, FAILED_MAIN = "FAILED_OURS", "FAILED_MAIN"


@dataclasses.dataclass(frozen=True)
class CellVerdict:
    """One (target, shape, mesh) cell compared across the two trees.

    Attributes:
        key: ``(target, N, cp0, cp1, Dloc)`` -- what makes a cell the SAME cell on both sides.
        ours_ms, main_ms: ``winner.time_ms`` per side, or None when that side is missing/failed.
        ratio: ``ours_ms / main_ms``; >1 means ours is SLOWER. None unless both sides ran.
        verdict: One of the module constants. Never NEUTRAL by default -- a cell that did not run
            on both sides gets its own state so a truncated sweep cannot read as a clean one.
        ours_cfg, main_cfg: the winning config each side chose.
        cfg_match: whether the two sides ran the SAME config. False makes the ratio a comparison
            between two different kernels, which is a parity finding rather than a timing one.
        ours_noise, main_noise: robust within-rank dispersion (relative MAD) of ``raw_ms``.
            This is what the bar is applied to.
        ours_tail, main_tail: ``max/min`` of ``raw_ms``. Reported beside the dispersion, never
            folded into it -- see `_tail` for why one scalar cannot carry both.
        ours_pe_penalty, main_pe_penalty: ``time_ms/median(raw_ms) - 1``, how much slower the
            slowest PE was than the rank that wrote the file.
        note: Free text for whatever the verdict alone does not say.
    """

    key: tuple
    ours_ms: float | None
    main_ms: float | None
    ratio: float | None
    verdict: str
    ours_cfg: dict | None = None
    main_cfg: dict | None = None
    cfg_match: bool | None = None
    ours_noise: float | None = None
    main_noise: float | None = None
    ours_tail: float | None = None
    main_tail: float | None = None
    ours_pe_penalty: float | None = None
    main_pe_penalty: float | None = None
    note: str = ""


@dataclasses.dataclass(frozen=True)
class Report:
    """The verdicts plus the denominator, because a truncated sweep must not read as a clean one.

    Attributes:
        cells: One `CellVerdict` per cell seen on EITHER side, so a cell missing from one tree
            appears rather than vanishing.
        ours_files, main_files: How many ``bench_N*.json`` were found per side. Zero on a side is
            the difference between "no regression" and "no data", and they look identical in a
            table of ratios.
        bar: The neutrality bar applied.
    """

    cells: tuple[CellVerdict, ...]
    ours_files: int
    main_files: int
    bar: float

    def counts(self) -> dict[str, int]:
        """Verdict -> how many cells, including every not-compared state."""
        out: dict[str, int] = {}
        for c in self.cells:
            out[c.verdict] = out.get(c.verdict, 0) + 1
        return out

    @property
    def compared(self) -> int:
        """Cells where BOTH sides produced a time -- the only cells a ratio exists for."""
        return sum(1 for c in self.cells if c.ratio is not None)

    @property
    def config_mismatches(self) -> tuple[CellVerdict, ...]:
        """Compared cells where the two trees chose DIFFERENT winning configs."""
        return tuple(c for c in self.cells if c.cfg_match is False)


def _noise(raw: list) -> float | None:
    """Robust within-rank dispersion of the MEDIAN: ``median(|x - med|) / med`` (relative MAD).

    **This deliberately is NOT ``(max - min) / median``, and the change has a measured reason.**
    A range is set by the single worst round, so one stall out of 15 condemned a cell whose median
    was perfectly well determined -- `N3008_D512` had 14 rounds inside 0.5% and one at +11%, and the
    range-based rule called it unresolvable while the median ratio was 0.992. The median is robust
    to a lone outlier by construction; scoring it with a range throws that robustness away.

    Args:
        raw: ``winner.raw_ms``, this rank's per-round times. Needs >= 2 entries and a non-zero
            median; anything else yields None rather than a fabricated zero, because a cell with
            no dispersion estimate and a cell with zero dispersion must not look alike.

    Returns:
        Relative MAD, or None.
    """
    if not raw or len(raw) < 2:
        return None
    med = statistics.median(raw)
    if not med:
        return None
    return statistics.median([abs(x - med) for x in raw]) / med


def _tail(raw: list) -> float | None:
    """``max / min`` -- the TAIL indicator, reported BESIDE the dispersion and never as it.

    Why a second number rather than a better single one: **a balanced multi-modal cell and a tight
    cell can share any spread statistic**, so no single scalar separates them. The cell that started
    this ran four rounds on `main`'s number, six intermediate, and five at 2x; its relative MAD is
    modest because half the samples sit near the median, while its ``max/min`` is 2.26. That
    structure is what a bring-back defect looks like when it fires intermittently, and it is exactly
    what a spread statistic averages away -- the same reason this repo forbids a pooled scalar in a
    numerical comparison.

    A large tail is NOT by itself a verdict: contention produces it too, and in the case above the
    cause turned out to be four benchmark drivers racing one allocation. It is reported so the
    reader can see the structure and ask, never silently folded into a pass.

    Args:
        raw: ``winner.raw_ms``. Needs >= 2 entries and a non-zero min.

    Returns:
        ``max/min``, or None.
    """
    if not raw or len(raw) < 2:
        return None
    lo = min(raw)
    return (max(raw) / lo) if lo else None


def load_sweep(dirpath: str) -> tuple[dict, int]:
    """Read every ``bench_N*.json`` in a directory into ``{cell_key: winner-ish dict}``.

    Args:
        dirpath: A harness ``--out-dir``. A path that does not exist yields an empty map and a
            zero file count rather than raising -- the caller reports "no data on this side",
            which is a different verdict from "no regression".

    Returns:
        ``(cells, n_files)``. `cells` maps ``(target, N, cp0, cp1, Dloc)`` to a dict carrying
        ``status``, ``time_ms``, ``cfg``, ``raw_ms``, so the comparison never re-reads the file.
    """
    cells: dict[tuple, dict] = {}
    files = sorted(glob.glob(os.path.join(dirpath, "bench_N*.json")))
    for path in files:
        try:
            doc = json.load(open(path, encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for cell in doc.get("cells", []):
            key = (
                cell.get("target"),
                cell.get("N"),
                cell.get("cp0"),
                cell.get("cp1"),
                cell.get("Dloc"),
            )
            w = cell.get("winner") or {}
            cells[key] = {
                "status": cell.get("status"),
                "reason": cell.get("reason", ""),
                "time_ms": w.get("time_ms"),
                "cfg": w.get("cfg"),
                "raw_ms": w.get("raw_ms") or [],
            }
    return cells, len(files)


def compare(ours_dir: str, main_dir: str, *, bar: float = DEFAULT_BAR) -> Report:
    """Pair two sweeps and judge each cell.

    Semantics
        A cell is compared only when BOTH sides have ``status == "ok"`` and a ``time_ms``. Every
        other outcome gets its own verdict rather than being dropped: a dropped cell is
        indistinguishable from a cell that passed, which is the failure this whole report exists
        to prevent.

        The bar is applied only where the cell's own samples can support it. If either side's
        within-rank spread exceeds `bar`, the verdict is UNRESOLVABLE -- the measurement is
        noisier than the difference being claimed.

    Args:
        ours_dir: Candidate tree's harness ``--out-dir``.
        main_dir: Baseline tree's ``--out-dir``. The ratio is ours/main, so >1 means ours is slower.
        bar: Relative neutrality bar. See `DEFAULT_BAR` for how the default was chosen.

    Returns:
        A `Report`.
    """
    ours, n_ours = load_sweep(ours_dir)
    main, n_main = load_sweep(main_dir)
    out: list[CellVerdict] = []
    for key in sorted(set(ours) | set(main), key=lambda k: tuple(str(x) for x in k)):
        o, m = ours.get(key), main.get(key)
        if o is None:
            out.append(
                CellVerdict(
                    key,
                    None,
                    m.get("time_ms"),
                    None,
                    MISSING_OURS,
                    main_cfg=m.get("cfg"),
                    note="cell absent from the candidate sweep",
                )
            )
            continue
        if m is None:
            out.append(
                CellVerdict(
                    key,
                    o.get("time_ms"),
                    None,
                    None,
                    MISSING_MAIN,
                    ours_cfg=o.get("cfg"),
                    note="cell absent from the baseline sweep",
                )
            )
            continue
        o_ok = o.get("status") == "ok" and o.get("time_ms")
        m_ok = m.get("status") == "ok" and m.get("time_ms")
        common = dict(
            ours_cfg=o.get("cfg"),
            main_cfg=m.get("cfg"),
            ours_noise=_noise(o["raw_ms"]),
            main_noise=_noise(m["raw_ms"]),
            ours_tail=_tail(o["raw_ms"]),
            main_tail=_tail(m["raw_ms"]),
        )
        if not o_ok or not m_ok:
            v = FAILED_OURS if not o_ok else FAILED_MAIN
            bad = o if not o_ok else m
            out.append(
                CellVerdict(
                    key,
                    o.get("time_ms"),
                    m.get("time_ms"),
                    None,
                    v,
                    **common,
                    note=f"status={bad.get('status')!r} reason={bad.get('reason', '')[:120]!r}",
                )
            )
            continue
        ratio = o["time_ms"] / m["time_ms"]
        cfg_match = o.get("cfg") == m.get("cfg")
        noise = max(
            [x for x in (common["ours_noise"], common["main_noise"]) if x is not None] or [0.0]
        )
        if noise > bar:
            verdict, note = (
                UNRESOLVABLE,
                (
                    f"robust dispersion (relative MAD) {noise:.1%} exceeds the bar "
                    f"{bar:.1%}; this cell's own samples cannot support a verdict at that "
                    f"resolution"
                ),
            )
        elif ratio > 1.0 + bar:
            verdict, note = REGRESSION, ""
        elif ratio < 1.0 - bar:
            verdict, note = FASTER, ""
        else:
            verdict, note = NEUTRAL, ""
        tails = [x for x in (common["ours_tail"], common["main_tail"]) if x is not None]
        if tails and max(tails) >= _TAIL_FLAG:
            note = (note + "; " if note else "") + (
                f"HEAVY TAIL max/min ours={common['ours_tail']:.2f} main={common['main_tail']:.2f} "
                f"-- the rounds are not unimodal, so the median is a summary of a MIXTURE. Not a "
                f"verdict by itself (contention does this too), but do not read this cell as clean"
            )
        if not cfg_match:
            note = (note + "; " if note else "") + (
                f"CONFIG MISMATCH ours={o.get('cfg')} main={m.get('cfg')} -- the ratio "
                f"compares two "
                f"DIFFERENT kernels, not two builds of one"
            )
        out.append(
            CellVerdict(
                key,
                o["time_ms"],
                m["time_ms"],
                ratio,
                verdict,
                cfg_match=cfg_match,
                ours_pe_penalty=(o["time_ms"] / statistics.median(o["raw_ms"]) - 1)
                if o["raw_ms"]
                else None,
                main_pe_penalty=(m["time_ms"] / statistics.median(m["raw_ms"]) - 1)
                if m["raw_ms"]
                else None,
                note=note,
                **common,
            )
        )
    return Report(tuple(out), n_ours, n_main, bar)


def render(report: Report) -> str:
    """The paired table, then the denominator, then the parity flags.

    The denominator is printed even when everything passed. "12 cells compared, 0 missing" and
    "2 cells compared, 10 missing" produce the same all-NEUTRAL table, and only the counts tell
    them apart.

    Args:
        report: From `compare`.

    Returns:
        A multi-line string. Nothing is elided -- a cell that could not be compared prints with
        its state and its reason rather than being filtered out of the table.
    """
    L = [
        f"{'target':<12} {'N':>6} {'mesh':>8} {'Dloc':>5} "
        f"{'ours ms':>10} {'main ms':>10} {'ratio':>7}  verdict",
        "-" * 88,
    ]
    for c in report.cells:
        target, N, cp0, cp1, Dloc = c.key
        mesh = f"{cp0}x{cp1}" if cp1 and cp1 > 1 else f"{cp0}"
        om = f"{c.ours_ms:10.4f}" if c.ours_ms else f"{'-':>10}"
        mm = f"{c.main_ms:10.4f}" if c.main_ms else f"{'-':>10}"
        rt = f"{c.ratio:7.4f}" if c.ratio else f"{'-':>7}"
        L.append(
            f"{str(target):<12} {str(N):>6} {mesh:>8} {str(Dloc):>5} {om} {mm} {rt}  {c.verdict}"
        )
        if c.note:
            L.append(f"{'':>12}   note: {c.note}")
        if c.ours_noise is not None or c.main_noise is not None:
            ot = f"{c.ours_tail:.2f}" if c.ours_tail is not None else "n/a"
            mt = f"{c.main_tail:.2f}" if c.main_tail is not None else "n/a"
            on = f"{c.ours_noise:.2%}" if c.ours_noise is not None else "n/a"
            mn = f"{c.main_noise:.2%}" if c.main_noise is not None else "n/a"
            pe_o = f"{c.ours_pe_penalty:+.2%}" if c.ours_pe_penalty is not None else "n/a"
            pe_m = f"{c.main_pe_penalty:+.2%}" if c.main_pe_penalty is not None else "n/a"
            L.append(
                f"{'':>12}   dispersion(MAD) ours={on} main={mn} | tail(max/min) ours={ot} "
                f"main={mt} | slowest-PE penalty ours={pe_o} main={pe_m}"
            )
    counts = report.counts()
    L += [
        "",
        f"BAR: +-{report.bar:.1%} relative; a cell whose own spread exceeds it is "
        f"UNRESOLVABLE, not NEUTRAL.",
        f"DENOMINATOR: {len(report.cells)} cells seen, {report.compared} compared "
        f"(both sides ok), files ours={report.ours_files} main={report.main_files}",
        "  " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())),
    ]
    if report.ours_files == 0 or report.main_files == 0:
        L.append(
            "  *** a side contributed ZERO files -- this is 'no data', NOT 'no regression' ***"
        )
    if report.config_mismatches:
        L.append(
            f"  *** {len(report.config_mismatches)} CONFIG MISMATCH cell(s): the two trees chose "
            f"different winning configs, so those ratios compare different kernels ***"
        )
    return "\n".join(L)


def main(argv=None) -> int:
    """CLI: ``compare_runtime.py <ours_out_dir> <main_out_dir> [bar]``."""
    import sys

    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 2:
        print("usage: compare_runtime.py <ours_out_dir> <main_out_dir> [bar]", file=sys.stderr)
        return 2
    bar = float(argv[2]) if len(argv) > 2 else DEFAULT_BAR
    print(render(compare(argv[0], argv[1], bar=bar)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

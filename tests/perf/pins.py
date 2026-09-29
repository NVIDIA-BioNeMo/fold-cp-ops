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

"""One pin file per perf test, named after it, in a format no gate may invent around.

``test_benchmark_perf_gemm.py`` reads ``test_benchmark_perf_gemm.json`` and nothing else. The
name is the documentation: given a number in a failure message you can find the file it came from
without grepping, and given a pin file you know exactly which test consumes it.

**Why a file rather than a dict in the module.** Four gates had grown four shapes of the same idea
-- one keyed by a 6-tuple holding a float, one holding a ``(median, spread)`` pair, one keyed by a
3-tuple, plus four loose constants for autotune bands and dispatch ceilings that were pinned numbers
in everything but name. Nothing related them, so "is this number measured or guessed" and "when was
it harvested" were answerable only by reading each file's prose. A schema makes both structural.

**What a pin file holds.**

* ``cells`` -- one entry per timed cell: its key, its ``median_ms``, and optionally the ``rel_std``
  that makes it reference-calibratable. A cell WITHOUT a spread is legal and means "not yet
  calibrated"; :func:`PinFile.calibrated` reports it, and the isolation hook uses that to decide
  whether the gate may run in a shared session.
* ``autotune`` -- the same gate's tuning constants: the best-known config per shape, the band it is
  judged at, the dispatch ceiling. These are pinned measurements too, and keeping them here stops
  them drifting into module constants where nobody re-harvests them.
* ``measured_on`` -- free text naming the machine and conditions. A pin without provenance cannot be
  re-harvested by anyone but its author.

Enforced by ``conftest.pytest_collection_modifyitems``: a timed perf module that does not expose a
``PINS`` built by :func:`load` FAILS. Not skips -- a gate that invents its own number format is
exactly the thing this exists to prevent, and a skip would let it ship quietly.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Any, Dict, Optional, Sequence, Tuple

#: Bumped when the on-disk shape changes incompatibly. A pin file carrying a different schema is
#: refused rather than best-effort parsed: a silently-misread pin is a gate comparing against the
#: wrong number, which looks exactly like a passing gate.
SCHEMA = 1

#: The estimator a pin file's numbers were produced by, when the file does not say.
#:
#: **This is a per-TEST property, not a global one, and that is why it is a field and not a schema
#: bump.** Only the distributed TriMul gate moved to `benchmark_extrapolated`; the other eleven pin
#: files are still produced by `benchmark_single` and are perfectly valid. A schema bump would have
#: refused all twelve to express a change affecting one, turning eleven unrelated gates red.
#:
#: Why it must be recorded at all: `benchmark_single(iters=20)` reads `device + C/20` -- it carries
#: the per-call host-dispatch residue -- while `benchmark_extrapolated` solves that term out and
#: reports `device`. Measured across the 13 pinned cp8 cells, 11 of 13 extrapolated readings came in
#: BELOW their `iters=20` pin, mean ratio 0.989. Small, one-sided, and therefore exactly the kind of
#: difference a one-sided gate reads as "everything got slightly faster" forever rather than as a
#: mistake. A file must be calibrated AS A WHOLE, so the mismatch is refused per FILE.
ESTIMATOR_SINGLE = "single"
ESTIMATOR_EXTRAPOLATED = "extrapolated"

#: `benchmark_single(iters=1, subtract_overhead=C)` with ONE `C` probed per process.
#:
#: A THIRD regime, not a synonym for `extrapolated`, and the difference is measured. A global `C`
#: leaves each cell a residual of `(C_cell - C_probe) / device`: at cp8 the probe reads 325.9 us
#: while the smallest cell's own solved `C` is 368 us, so that cell comes in 2.4% high against a
#: 2.5% band -- 12 of 13 cells pass and the smallest fails. Large cells absorb the residual; small
#: ones are exactly where `C` matters, so they cannot.
ESTIMATOR_PROBED = "probed"
DEFAULT_ESTIMATOR = ESTIMATOR_SINGLE

#: Every key a pin file may carry at the top level. An unknown key is refused, because the common
#: way to invent an ad-hoc format is to add a field and hope something reads it.
_TOP_LEVEL = frozenset({"schema", "estimator", "kernel", "measured_on", "cells", "autotune"})

#: Every key a cell entry may carry.
_CELL_KEYS = frozenset({"key", "median_ms", "rel_std", "note", "rounds"})

#: The round counts a harvest line may legitimately carry, mirroring `calibration._ROUND_RUNGS`.
#: DUPLICATED rather than imported, because `calibration` imports THIS module at module scope for
#: `DEFAULT_ESTIMATOR`, and importing back would be a cycle. `tests/perf/test_pins.py` pins the two
#: tuples equal, so the duplication cannot drift silently -- which is the only thing that makes a
#: duplicated constant acceptable.
_VALID_ROUNDS = frozenset({5, 8, 12, 18, 25, 35})


@dataclasses.dataclass(frozen=True)
class Pin:
    """One cell's pinned measurement.

    Attributes:
        median_ms: The measured median, in milliseconds. Positive.
        rel_std: The measured relative spread, or None when the cell has not been harvested with
            one. None is NOT zero -- zero would claim a perfectly stable cell and produce a
            zero-width band, while None means "unknown, so this gate cannot be calibrated yet".
        note: Free text from the file, carried through so a failure message can quote it.
        rounds: The timed-round count this cell was harvested with, or None for a pin written before
            the field existed. **None means 35**, the historical default, and is read that way by
            `calibration.assert_cell` -- not "unknown, so skip the check", because every pin in this
            repo predating the field WAS harvested at 35 and treating them as unverifiable would
            silently exempt exactly the cells the check exists for.
    """

    median_ms: float
    rel_std: Optional[float] = None
    note: str = ""
    rounds: Optional[int] = None

    def __post_init__(self):
        """Refuse a nonsensical pin at load, not at comparison.

        Raises:
            ValueError: If the median is not positive, or the spread is present and negative.
        """
        if not self.median_ms > 0:
            raise ValueError(f"median_ms must be positive; got {self.median_ms}")
        if self.rel_std is not None and self.rel_std < 0:
            raise ValueError(f"rel_std must be non-negative; got {self.rel_std}")
        if self.rounds is not None and self.rounds < 1:
            raise ValueError(f"rounds must be >= 1 when present; got {self.rounds}")


class PinFile:
    """The parsed contents of one ``test_benchmark_perf_<kernel>.json``.

    Args:
        path: Where it was loaded from, for error messages.
        payload: The parsed JSON.

    Raises:
        ValueError: On an unknown schema version, an unknown key, a duplicate cell key, or a cell
            whose numbers do not validate. Every one of these is refused at load so a gate never
            compares against a number nobody checked.
    """

    def __init__(self, path: pathlib.Path, payload: Dict[str, Any]):
        unknown = set(payload) - _TOP_LEVEL
        if unknown:
            raise ValueError(
                f"{path.name}: unknown top-level key(s) {sorted(unknown)}. Allowed: "
                f"{sorted(_TOP_LEVEL)}. Adding a field and hoping something reads it is how a "
                f"gate grows its own format."
            )
        if payload.get("schema") != SCHEMA:
            raise ValueError(
                f"{path.name}: schema {payload.get('schema')!r}, expected {SCHEMA}. Refused rather "
                f"than best-effort parsed -- a misread pin is a gate comparing against the wrong "
                f"number, which looks exactly like a passing gate."
            )
        self.path = path
        #: The estimator named by the file, defaulting to the current one for a file written before
        #: the field existed. Read rather than assumed so a future third estimator can be told apart
        #: from this one without another schema bump.
        self.estimator: str = payload.get("estimator", DEFAULT_ESTIMATOR)
        self.kernel: str = payload.get("kernel", path.stem)
        self.measured_on: str = payload.get("measured_on", "")
        if not self.measured_on:
            raise ValueError(
                f"{path.name}: `measured_on` is required. A pin without provenance cannot be "
                f"re-harvested by anyone but whoever measured it."
            )
        self.autotune: Dict[str, Any] = payload.get("autotune", {})
        self._cells: Dict[Tuple, Pin] = {}
        for entry in payload.get("cells", []):
            bad = set(entry) - _CELL_KEYS
            if bad:
                raise ValueError(f"{path.name}: unknown cell key(s) {sorted(bad)} in {entry}")
            key = tuple(entry["key"])
            if key in self._cells:
                raise ValueError(
                    f"{path.name}: duplicate cell {list(key)}. Two pins for one cell means "
                    f"whichever is read last silently wins."
                )
            self._cells[key] = Pin(
                median_ms=float(entry["median_ms"]),
                rel_std=(None if entry.get("rel_std") is None else float(entry["rel_std"])),
                rounds=(None if entry.get("rounds") is None else int(entry["rounds"])),
                note=entry.get("note", ""),
            )

    def __contains__(self, key: Sequence) -> bool:
        """Whether this cell has a pin. Use before :meth:`__getitem__` to skip-and-harvest."""
        return tuple(key) in self._cells

    def __getitem__(self, key: Sequence) -> Pin:
        """The pin for one cell.

        Args:
            key: The cell's identity, as a sequence; converted to a tuple so a caller may pass a
                list read back from JSON or a tuple built in Python.

        Returns:
            The :class:`Pin`.

        Raises:
            KeyError: If the cell is unpinned. The caller decides whether that is a skip (harvest
                mode) or a failure.
        """
        return self._cells[tuple(key)]

    def __len__(self) -> int:
        """How many cells are pinned."""
        return len(self._cells)

    @property
    def calibrated(self) -> bool:
        """Whether EVERY cell carries a spread, so this gate can be reference-calibrated.

        Returns:
            True only if no cell is missing ``rel_std``. All-or-nothing on purpose: a gate that
            calibrated some cells and not others would apply a drift correction to some numbers and
            not others in the same run, and the two would no longer be comparable with each other.
        """
        return bool(self._cells) and all(p.rel_std is not None for p in self._cells.values())

    def keys(self):
        """The pinned cell keys, for a test that wants to parametrize over exactly what is pinned."""
        return self._cells.keys()


def pin_path(module_file: str) -> pathlib.Path:
    """The pin file belonging to a perf test module.

    Args:
        module_file: The module's ``__file__``.

    Returns:
        The sibling ``.json`` with the same stem. Same-stem-same-directory is the whole convention:
        it makes "which file did this number come from" answerable without grepping.
    """
    return pathlib.Path(module_file).with_suffix(".json")


def load(module_file: str) -> PinFile:
    """Load the pin file for a perf test module.

    Args:
        module_file: The calling module's ``__file__``. Always pass ``__file__`` -- passing a
            literal path is how two modules end up sharing one pin file and silently overwriting
            each other's harvest.

    Returns:
        The parsed :class:`PinFile`.

    Raises:
        FileNotFoundError: If the sibling JSON is missing, naming the path it must be created at.
        ValueError: From :class:`PinFile` on any schema violation.
    """
    path = pin_path(module_file)
    if not path.exists():
        raise FileNotFoundError(
            f"no pin file at {path}. Every timed perf module reads its numbers from a sibling "
            f"JSON of the same name; create it with "
            f"`CPO_PERF_MEASURE=1 pytest tests/perf/ -q -s` and merge the harvest. The `-s` is "
            f"required: pytest captures a passing test's output, so `-q` alone emits nothing."
        )
    return PinFile(path, json.loads(path.read_text()))


#: Prefix of the machine-parseable harvest line. Chosen so a merge script can find these among the
#: gates' own human-readable prints without the two formats having to agree on anything else.
HARVEST_TAG = "PINHARVEST"


def emit_harvest(
    module_file: str, key: Sequence, median_ms: float, rel_std: float, rounds: Optional[int] = None
) -> None:
    """Print one harvested cell in a form a merge script can read back.

    Purpose
        Lets ONE ``CPO_PERF_MEASURE=1`` run refresh every gate's pin file, instead of each gate
        printing a differently-shaped block that a human retypes. Retyping is where a harvest picks
        up a transcription error that nothing downstream can detect.

    Args:
        module_file: The gate's ``__file__``; names which pin file the line belongs to.
        key: The cell's identity.
        median_ms: Its harvested median.
        rel_std: Its harvested relative spread.
        rounds: The timed-round count it was harvested with, or None to omit the field. APPENDED as
            a fifth token rather than inserted, so a reader written against the four-field line still
            parses every line it used to -- the field is optional on the way in as well as out.

    Returns:
        None; writes one line to stdout, ALWAYS starting one.

    Note:
        The leading newline is load-bearing, not cosmetic. Under ``-s`` pytest writes its per-test
        progress dot with no trailing newline, so a bare ``print`` lands as ``.PINHARVEST ...`` --
        and a merge script anchored at ``^PINHARVEST`` then silently reads back only the cells that
        happened to follow a real newline. Measured: a 16-cell gate whose every cell emitted was
        counted as 1, and the three-gate calibration was abandoned on that number. A harvest that
        under-reports looks exactly like a harvest that did not run.
    """
    # Compact separators so the key is ONE whitespace-free token: the line is meant to be read by
    # splitting on spaces, and `json.dumps`'s default `", "` would put the key across several.
    tail = "" if rounds is None else f" {int(rounds)}"
    print(
        f"\n{HARVEST_TAG} {pathlib.Path(module_file).stem} "
        f"{json.dumps(list(key), separators=(',', ':'))} {median_ms:.6f} {rel_std:.6f}{tail}",
        flush=True,
    )


def parse_harvest(text: str) -> Dict[str, Dict[Tuple, Tuple[float, float, Optional[int]]]]:
    """Read every :func:`emit_harvest` line out of a captured run.

    Purpose
        Turns one ``CPO_PERF_MEASURE=1`` log into the numbers each pin file should carry, so a
        re-harvest is a mechanical merge rather than a human retyping medians per gate. Retyping is
        where a harvest picks up a transcription error nothing downstream can detect.

    Args:
        text: The captured stdout. Lines that are not harvest lines are ignored, so the gates' own
            human-readable prints and pytest's progress output can be left in.

    Returns:
        ``{module stem: {cell key tuple: (median_ms, rel_std)}}``. A key harvested twice keeps the
        LAST occurrence, matching what a reader would assume from a log read top to bottom.

    Raises:
        ValueError: If a harvest line is malformed. Refused loudly rather than skipped: a silently
            dropped line is a cell that keeps its stale pin while the report says it was refreshed.
    """
    out: Dict[str, Dict[Tuple, Tuple[float, float]]] = {}
    for raw in text.splitlines():
        idx = raw.find(HARVEST_TAG)
        if idx < 0:
            continue
        # `find` rather than `startswith`: pytest's progress dot shares the line under `-s`, and a
        # harvest that silently skipped those lines is what made an earlier merge read 1 cell of 16.
        # `.split()` collapses runs of whitespace and the FIRST FIVE tokens are the record;
        # anything after them is another writer's text, not corruption. A multi-rank harvest has
        # 8-16 ranks printing to one unsynchronized stdout, so a gate's own human-readable line
        # lands appended to an emit line whenever two ranks interleave mid-line. Measured on the
        # 2026-08-21 world-16 harvest: 104 of 1692 lines, every one carrying an intact 5-token
        # prefix. Refusing them lost nothing only because per-rank duplication happened to cover
        # every affected cell -- that is luck, not a guarantee, and at world-2 it would not hold.
        parts = raw[idx:].split()
        if len(parts) < 5:
            raise ValueError(f"malformed harvest line: {raw!r}")
        _, stem, key_json, median, rel_std = parts[:5]
        # The OPTIONAL sixth token is the round count, and it is validated rather than merely
        # parsed. Everything after token five may be another writer's text -- that is the whole
        # premise of the tolerance above -- so "there is a sixth token" cannot mean "it is mine".
        # Requiring it to be a bare integer AND one of `calibration._ROUND_RUNGS` makes a foreign
        # fragment essentially impossible to mistake for a round count: the measured interleavings
        # are appended human-readable lines that begin with text, and a bare `5`/`8`/`12`/`18`/`25`/
        # `35` is not one of them. If a fragment ever did impersonate one, the consequence is a
        # wrong recorded count, which `assert_cell` then REFUSES -- loud, not silent.
        rounds = None
        if len(parts) >= 6 and parts[5].isdigit() and int(parts[5]) in _VALID_ROUNDS:
            rounds = int(parts[5])
        try:
            key = tuple(json.loads(key_json))
        except json.JSONDecodeError as exc:
            raise ValueError(f"unreadable key in harvest line: {raw!r}") from exc
        try:
            # A pytest progress dot can be FUSED to the last token rather than separated by a
            # space -- `0.005000.` -- so token-level tolerance above does not reach it. Strip a
            # trailing dot run and re-parse. This cannot mask a corrupt value: a float is still
            # required afterwards, and Python already reads a legitimate trailing dot (`5.`) as
            # the same number stripping produces. Measured: 1 line of 2476 on the 2026-08-21 2-D
            # harvest -- rare, but it costs a whole cell whenever that cell printed only once.
            values = (float(median.rstrip(".")), float(rel_std.rstrip(".")), rounds)
        except ValueError as exc:
            # The loud-refusal guarantee survives the tolerance above: a truncated or corrupted
            # record still fails here, because its 4th and 5th tokens are not two floats.
            raise ValueError(f"unreadable median/rel_std in harvest line: {raw!r}") from exc
        out.setdefault(stem, {})[key] = values
    return out


def merge_harvest(
    payload: Dict[str, Any],
    harvested: Dict[Tuple, Tuple[float, float]],
    measured_on: Optional[str] = None,
) -> Tuple[Dict[str, Any], list]:
    """Fold harvested numbers into one pin file's payload, without touching anything else.

    Purpose
        The write half of a re-harvest. Deliberately PURE -- it returns a new payload rather than
        writing -- so the merge logic is testable without a filesystem and so a caller can diff
        before committing numbers that gate a suite.

    Semantics
        Existing cells are updated IN PLACE in the list, preserving order and every field the
        harvest does not measure (notably ``note``). A harvested key with no existing cell is
        APPENDED, so a newly-added cell is picked up on its first harvest. Cells present in the file
        but absent from the harvest are left untouched and REPORTED -- silently keeping a stale
        number while announcing a refresh is the failure this return value exists to prevent.

    Args:
        payload: The parsed pin file. Not mutated.
        harvested: ``{key tuple: (median_ms, rel_std, rounds_or_None)}``, as from
            :func:`parse_harvest`.
        measured_on: Provenance text for THIS harvest. REPLACES the existing value when the
            harvest covered every cell in the file, and is APPENDED to it when some cells kept
            their old numbers -- because those cells were never measured under the new conditions,
            and re-attributing them silently is how a measurement recorded in this field gets lost.
            None leaves the field untouched.

    Returns:
        ``(new payload, unharvested keys)``. The second element is the list of cell keys the harvest
        did not cover, in file order.
    """
    new = dict(payload)
    cells, seen, missing = [], set(), []
    for entry in payload.get("cells", []):
        key = tuple(entry["key"])
        seen.add(key)
        if key in harvested:
            median, rel_std, rounds = harvested[key]
            # `rounds` REPLACES rather than merges, and a harvest that carries none REMOVES the old
            # value. Keeping a stale count beside a fresh median is the one combination that is
            # actively wrong: `assert_cell` compares the recorded count against the count derived
            # from the median, so a mismatched pair produces a refusal nobody can act on.
            fresh = {**entry, "median_ms": round(median, 6), "rel_std": round(rel_std, 6)}
            fresh.pop("rounds", None)
            if rounds is not None:
                fresh["rounds"] = int(rounds)
            cells.append(fresh)
        else:
            cells.append(dict(entry))
            missing.append(key)
    for key, (median, rel_std, rounds) in harvested.items():
        if key not in seen:
            cells.append(
                {
                    "key": list(key),
                    "median_ms": round(median, 6),
                    "rel_std": round(rel_std, 6),
                    **({} if rounds is None else {"rounds": int(rounds)}),
                }
            )
    new["cells"] = cells
    if measured_on is not None:
        # REPLACE on a total re-harvest, APPEND on a partial one -- and the merge already knows
        # which this is, so the caller is not asked to remember.
        #
        # Replacing unconditionally lost a real measurement: an earlier revision of the back-A2A pin
        # file recorded "moved a back median 15.00 -> 25.96 ms and flipped the elected autotune
        # config" in this field, and a later partial re-harvest overwrote it. That sentence is the
        # evidence for one-launch-per-cell, and it survived only in a shell comment and in git.
        #
        # Appending unconditionally is equally wrong the other way: a FULL re-harvest on new silicon
        # would leave a stale first line naming a machine none of the numbers came from.
        #
        # `missing` is empty exactly when every cell in the file was re-measured by this harvest, so
        # it is the discriminator: nothing old survives -> nothing old to attribute.
        prior = str(payload.get("measured_on", "")).strip()
        if not missing or not prior:
            new["measured_on"] = measured_on  # total re-harvest, or nothing to preserve
        elif measured_on.strip() in prior:
            new["measured_on"] = prior  # already recorded -- keep the ACCUMULATED field, do not
            # collapse it to just this line, which is what re-running one partial harvest would do
        else:
            new["measured_on"] = f"{prior}\n{measured_on}"
    return new, missing


def _repo_tests_root() -> pathlib.Path:
    """The ``tests/`` directory this module lives under.

    Purpose
        A single anchor for every pin-file path, so a relocation of ``pins.py`` moves the search
        with it instead of silently narrowing it.

    Returns:
        The ``tests/`` directory (this file is ``tests/perf/pins.py``, so two parents up).
    """
    return pathlib.Path(__file__).resolve().parent.parent


def _pin_dirs() -> list[pathlib.Path]:
    """Every directory that may hold a ``<module stem>.json`` pin file.

    Purpose
        Pin files do NOT all live beside this module. The single-device gates are in
        ``tests/perf/``; the A2A gates are in ``tests/distributed/perf/``. An earlier version
        looked only in ``__file__``'s own directory, so every distributed gate hit the
        not-found branch and was skipped -- SILENTLY, with exit 0. Measured: a real
        ``PINHARVEST`` line for ``test_benchmark_perf_gemm_sm90_a2a`` printed
        ``SKIP ... no ....json`` and returned 0, which would have discarded a 183-cell harvest
        while reading as a clean merge.

    Semantics
        Order is NOT a precedence: ``_resolve_pin_file`` refuses an ambiguous stem rather than
        taking the first hit, so adding a directory here cannot silently shadow an existing pin
        file. Directories that do not exist are dropped, so a partial checkout is not an error.

    Returns:
        The existing pin directories, deepest-listed last. Never empty in a normal checkout.
    """
    root = _repo_tests_root()
    return [d for d in (root / "perf", root / "distributed" / "perf") if d.is_dir()]


def _resolve_pin_file(stem: str) -> pathlib.Path:
    """Locate the one pin file named ``<stem>.json``, refusing an ambiguous name.

    Purpose
        Turn a harvested module stem into the pin file it belongs to, across the several
        directories that hold them.

    Semantics
        A stem matching pin files in MORE THAN ONE directory raises rather than picking one.
        Two gates sharing a basename would otherwise have their medians merged into whichever
        directory sorted first -- a wrong-file write that no diff review would flag, because
        both files legitimately change during a harvest. This mirrors the repo's rule that a
        source basename is unique; here the collision is reported, never resolved by iteration
        order.

    Args:
        stem: A module stem exactly as it appears in a ``PINHARVEST`` line -- a bare module
            name with no directory part and no ``.json`` suffix. A stem carrying either is not
            found, which surfaces as ``FileNotFoundError`` rather than a wrong match.

    Returns:
        The absolute path of the single matching pin file.

    Raises:
        FileNotFoundError: no directory in :func:`_pin_dirs` holds ``<stem>.json``.
        RuntimeError: two or more do, naming every candidate.
    """
    hits = [d / f"{stem}.json" for d in _pin_dirs() if (d / f"{stem}.json").is_file()]
    if not hits:
        raise FileNotFoundError(stem)
    if len(hits) > 1:
        raise RuntimeError(f"ambiguous pin stem {stem!r}: {[str(h) for h in hits]}")
    return hits[0]


def _main(argv) -> int:
    """Merge a captured harvest log into the pin files it names.

    Purpose
        The one command that closes the harvest loop, so refreshing pins is
        ``run -> merge -> review the diff`` rather than a per-gate copy by hand.

    Args:
        argv: ``[log path, measured_on?]``. The log is any captured stdout containing
            ``PINHARVEST`` lines; ``measured_on`` replaces every touched file's provenance and
            should be given whenever the machine or its conditions changed.

    Returns:
        Process exit status: 0 on success, 2 on a usage error. Non-zero is NOT returned for
        un-harvested cells -- those are reported, because a partial harvest is a normal thing to do
        deliberately and the pin file records it faithfully either way.
    """
    if not 1 <= len(argv) <= 2:
        print(f"usage: python {pathlib.Path(__file__).name} <harvest log> [measured_on]")
        return 2
    text = pathlib.Path(argv[0]).read_text()
    measured_on = argv[1] if len(argv) > 1 else None
    unresolved = []
    for stem, harvested in sorted(parse_harvest(text).items()):
        try:
            path = _resolve_pin_file(stem)
        except FileNotFoundError:
            print(f"  UNRESOLVED {stem}: no {stem}.json under {[str(d) for d in _pin_dirs()]}")
            unresolved.append(stem)
            continue
        merged, missing = merge_harvest(json.loads(path.read_text()), harvested, measured_on)
        path.write_text(json.dumps(merged, indent=2) + "\n")
        note = f", {len(missing)} NOT harvested: {[list(k) for k in missing]}" if missing else ""
        print(f"  {path.relative_to(_repo_tests_root())}: {len(harvested)} cells merged{note}")
    if unresolved:
        print(f"  !! {len(unresolved)} harvested gate(s) had NO pin file: {unresolved}")
        return 3
    return 0


if __name__ == "__main__":  # pragma: no cover - a developer entry point, not a test path
    import sys

    raise SystemExit(_main(sys.argv[1:]))

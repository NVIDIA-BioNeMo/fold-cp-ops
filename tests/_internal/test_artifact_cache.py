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

"""Unit tests for ``fold_cp_ops/_internal/artifact_cache.py``.

**Every test here runs without a GPU, and that is a requirement rather than a convenience.** The
store's invariants -- key sensitivity, backend-tag enforcement, atomic publication, quarantine -- are
pure host logic, and a cache whose invariants can only be checked inside a distributed allocation is
a cache whose invariants stop being checked. The parts that genuinely need a device (a bitwise
round-trip, ``library_init``, per-rank artifacts) live in ``tests/distributed/test_artifact_cache.py``.

The negative controls carry the weight. A store that silently never hits satisfies every positive
assertion about correctness, so the tests that matter most are the ones asserting a REFUSAL:
a wrong backend tag, a wrong rank, a truncated file, a changed key component.
"""

import json
import hashlib
import os
import stat
from pathlib import Path

import pytest

from fold_cp_ops._internal import artifact_cache as ac
from fold_cp_ops._internal import cache_security
from fold_cp_ops._internal.artifact_cache import (
    ARTIFACT_ENABLED_ENV,
    ARTIFACT_SCHEMA_VERSION,
    BACKEND_DUMP_OBJECT,
    BACKEND_EXPORT_C,
    PersistSpec,
    RankScope,
    artifact_key,
    artifact_paths,
    normalize_options,
    quarantine,
    read_artifact,
    write_artifact,
)
from fold_cp_ops.testing.kernel_matrix import matrix_exempt

pytestmark = matrix_exempt(
    "The subject is a host-side artifact STORE, not a kernel: it compiles nothing and has no "
    "shape/dtype/tile axes for a KernelMatrix to declare."
)


def _spec(tmp_path, key=("cfg", 1), mode="rw") -> PersistSpec:
    """A spec rooted in a per-test directory.

    Args:
        tmp_path: pytest's per-test directory, so no test can see another's entries.
        key: the config tuple; defaulted because most tests vary something else.
        mode: ``"rw"``/``"r"``/``"off"``.

    Returns:
        A ``PersistSpec`` whose ``dir`` is ``tmp_path``.
    """
    return PersistSpec(key=key, dir=str(tmp_path), mode=mode)


def _key(spec, *, prefix="k", backend=BACKEND_DUMP_OBJECT, rank=None, **kw):
    """An artifact key for the STORAGE tests, with the program half composed HERE, not by the product.

    Purpose
        M5 deleted `program_key`, and the tests below that used it fall into two groups. A few had
        the DERIVATION as their subject and went with it. The rest -- write, read, reject,
        quarantine, mode, the process-wide switch -- only ever needed *a* stable, varying key, and
        composing it locally says so: a change to how the product derives a program key must not be
        able to make a storage test pass or fail. The product's own derivation is covered by the
        `resolve_program_key` tests.

    Args:
        spec: the spec whose (usually empty) extra components take part.
        prefix / backend: keyed here exactly as `resolve_program_key` keys them.
        rank: added by `artifact_key`, never by the program half -- that split is what the tests
            below assert directly, and it is the product's `artifact_key` doing it.
        **kw: any extra terms a test wants to vary (``op``, ``compile_args``, ``options``,
            ``world_size``, ``extra``). Rendered with `repr`, so two different values differ.

    Returns:
        The 64-hex artifact key.
    """
    payload = repr(
        (prefix, backend, sorted((str(k), repr(v)) for k, v in kw.items()), tuple(spec.key))
    )
    return artifact_key(hashlib.sha256(payload.encode()).hexdigest(), rank)


def _put(spec, blob=b"artifact-bytes", *, prefix="k", backend=BACKEND_DUMP_OBJECT, rank=None):
    """Mint one entry and return ``(keysha, o_path)``.

    Args:
        spec: the destination spec.
        blob: the bytes to publish.
        prefix: symbol prefix recorded in the meta.
        backend: backend tag recorded in the meta AND in the key.
        rank: optional ``RankScope``.

    Returns:
        ``(keysha, Path)`` for the published artifact.
    """
    keysha = _key(spec, prefix=prefix, backend=backend, rank=rank)
    path = write_artifact(blob, keysha, spec, prefix=prefix, backend=backend, rank=rank)
    return keysha, path


# ---------------------------------------------------------------------------
# PersistSpec: the argument itself refuses the two ways it is misused.
# ---------------------------------------------------------------------------
def test_an_empty_persist_key_is_ACCEPTED():
    """Empty is the NORMAL case at schema 2, and this test replaces one that asserted the opposite.

    Under schema 1 an empty key raised, on the reasoning that it "claims every compile of this kernel
    is interchangeable". That was right while the CALLER hand-listed identity. It is wrong now:
    identity is derived by `program_key` from the functor's class, its `compile_key()`, the operand
    signature and the compile options, so a caller supplying nothing is relying on the derivation
    rather than failing to describe anything.

    The old test is deleted rather than xfailed on purpose -- an xfail would leave a green suite
    asserting a rule the code no longer has.
    """
    spec = PersistSpec()
    assert spec.key == ()
    assert spec.enabled and spec.writable


def test_an_unknown_mode_is_refused_at_construction():
    """A typo'd mode must not silently read as 'rw' and start minting entries in a gate."""
    with pytest.raises(ValueError, match="rw"):
        PersistSpec(mode="readonly")


def test_mode_off_is_identical_to_persist_none(tmp_path):
    """``mode='off'`` neither reads nor writes -- the documented equivalence to omitting persist."""
    spec = _spec(tmp_path, mode="off")
    assert not spec.enabled and not spec.writable
    keysha, path = _put(spec)
    assert path is None
    assert read_artifact(keysha, spec, prefix="k", backend=BACKEND_DUMP_OBJECT) is None


def test_mode_r_never_writes_but_still_reads(tmp_path):
    """A perf gate must consume entries without minting them, or its runs differ from each other."""
    rw = _spec(tmp_path)
    keysha, path = _put(rw, b"payload")
    assert path is not None

    ro = _spec(tmp_path, mode="r")
    assert ro.enabled and not ro.writable
    got = read_artifact(keysha, ro, prefix="k", backend=BACKEND_DUMP_OBJECT)
    assert got is not None and got.blob == b"payload"

    # and a MISS under mode="r" mints nothing
    other = PersistSpec(key=("cfg", 999), dir=str(tmp_path), mode="r")
    k2 = _key(other, prefix="k", backend=BACKEND_DUMP_OBJECT)
    assert write_artifact(b"nope", k2, other, prefix="k", backend=BACKEND_DUMP_OBJECT) is None
    assert not (tmp_path / f"{k2}.o").exists()


def test_the_process_wide_switch_disables_a_spec_that_says_rw(tmp_path, monkeypatch):
    """``CPO_JIT_ARTIFACT_ENABLED=0`` is the correctness-run switch and overrides every spec.

    Read at CALL time, so a launcher exporting it after import still takes effect -- the failure this
    avoids is a warm cache silently surviving a correctness sweep, which is what the sweep exists to
    rule out.
    """
    spec = _spec(tmp_path)
    monkeypatch.setenv(ARTIFACT_ENABLED_ENV, "0")
    assert not spec.enabled and not spec.writable
    keysha, path = _put(spec)
    assert path is None


# ---------------------------------------------------------------------------
# The key: stable in what must not matter, sensitive to everything that must.
# ---------------------------------------------------------------------------
def test_the_key_is_stable_across_calls(tmp_path):
    """Same inputs -> same key. An unstable key turns every lookup into a miss."""
    spec = _spec(tmp_path)
    a = _key(spec, prefix="k", backend=BACKEND_DUMP_OBJECT)
    b = _key(spec, prefix="k", backend=BACKEND_DUMP_OBJECT)
    assert a == b and len(a) == 64


def test_the_key_ignores_the_directory_and_the_mode(tmp_path):
    """Where an artifact is stored and whether it may be written are not properties of its CONTENT.

    Otherwise a gate running ``mode='r'`` could never hit an entry minted under ``'rw'``, which is
    the exact pairing S6 depends on.
    """
    base = _key(_spec(tmp_path), prefix="k", backend=BACKEND_DUMP_OBJECT)
    other_dir = _key(
        PersistSpec(key=("cfg", 1), dir=str(tmp_path / "elsewhere")),
        prefix="k",
        backend=BACKEND_DUMP_OBJECT,
    )
    read_only = _key(
        PersistSpec(key=("cfg", 1), dir=str(tmp_path), mode="r"),
        prefix="k",
        backend=BACKEND_DUMP_OBJECT,
    )
    assert base == other_dir == read_only


class _FakeOp:
    """A stand-in functor with a declared compile-time surface.

    Kept after M5 deleted the composed key, because the storage tests still want an object to hang
    a varying term on and this one is inert: it touches no DSL and needs no GPU. It no longer stands
    in for anything the product reads -- the program key comes from the emitted MLIR now.
    """

    def __init__(self, **params):
        self._params = params

    def compile_key(self):
        """The functor's declared compile-time surface, as `TemplateParams.compile_key` returns."""
        return dict(self._params)


class _OtherFakeOp(_FakeOp):
    """A DIFFERENT class with an IDENTICAL `compile_key()` -- the qualname term's whole point."""


def test_the_program_key_EXCLUDES_pe_and_the_artifact_key_INCLUDES_it():
    """The split itself, which is what makes cross-rank agreement expressible.

    Every rank of a job computes the SAME program key -- that is the value `agree_on_program_key`
    broadcasts and compares. Only `artifact_key` adds `pe`, so two ranks address different files
    while agreeing on what program those files hold.

    The program half is a LITERAL here. It has to be: `artifact_key` is the subject, `pe` is not one
    of its inputs, and deriving the program half through the product would make this test depend on
    a derivation that has already changed once (M5 replaced it) without its claim changing at all.
    """
    prog = "p" * 64
    assert artifact_key(prog, RankScope(16, 0)) != artifact_key(prog, RankScope(16, 3))
    # ...and it is not a function of pe at all: `artifact_key` is the ONLY place pe enters.
    assert artifact_key(prog, RankScope(16, 0)) == artifact_key(prog, RankScope(16, 0))
    assert artifact_key(prog, None) != artifact_key(prog, RankScope(16, 0))


def test_normalize_options_erases_absolute_paths_but_keeps_flags():
    """Two nodes' bitcode paths must give ONE key, or the agreement check fires on a healthy run.

    That is the failure this normalisation exists to prevent: `find_device_bitcode_library()` is an
    absolute path and mount points differ per node, so keying it raw makes every rank of a two-node
    job disagree. A check that cries wolf gets switched off.
    """
    a = normalize_options(" --link-libraries=/node-a/lib/libnvshmem_device.bc")
    b = normalize_options(" --link-libraries=/completely/other/root/libnvshmem_device.bc")
    assert a == b
    # ...but a genuine flag difference still separates them.
    assert normalize_options("--enable-tvm-ffi") != normalize_options("")
    assert "--enable-tvm-ffi" in normalize_options(" --link-libraries=/x/y.bc --enable-tvm-ffi")


def test_a_different_config_key_changes_the_key(tmp_path):
    """The caller's own config tuple is what distinguishes two compiles of the same kernel."""
    a = _key(_spec(tmp_path, key=("tile", 128)), prefix="k", backend=BACKEND_DUMP_OBJECT)
    b = _key(_spec(tmp_path, key=("tile", 256)), prefix="k", backend=BACKEND_DUMP_OBJECT)
    assert a != b


def test_cuda_arch_never_raises_without_a_device(monkeypatch):
    """The key must be computable on a GPU-free host, or these tests could not exist.

    Returns a sentinel rather than raising; an artifact minted on a real device always carries a real
    arch, so the sentinel can never be served to a device run.
    """
    monkeypatch.setattr(ac, "cuda_arch", ac.cuda_arch)  # no-op: exercise the real function
    assert isinstance(ac.cuda_arch(), str)


# ---------------------------------------------------------------------------
# Paths and rank encoding.
# ---------------------------------------------------------------------------
def test_the_rank_appears_in_the_filename_not_only_in_the_hash(tmp_path):
    """Redundant with the hash BY CONSTRUCTION, and that redundancy is the feature.

    A directory of indistinguishable 64-hex names is one where a rank mismatch stays invisible until
    it faults inside nvshmem's init. The infix lets a human spot it in an ``ls``.
    """
    spec = _spec(tmp_path)
    rank = RankScope(world_size=16, pe=3)
    keysha = _key(spec, prefix="k", backend=BACKEND_DUMP_OBJECT, rank=rank)
    o_path, m_path = artifact_paths(keysha, spec, rank)
    assert o_path.name == f"{keysha}.pe3.of16.o"
    assert m_path.name == f"{keysha}.pe3.of16.meta.json"


def test_a_non_ranked_artifact_has_no_infix(tmp_path):
    """A kernel with no in-kernel nvshmem has no PE, so the infix would be a lie."""
    spec = _spec(tmp_path)
    keysha = _key(spec, prefix="k", backend=BACKEND_DUMP_OBJECT)
    o_path, _ = artifact_paths(keysha, spec, None)
    assert o_path.name == f"{keysha}.o"


def test_two_ranks_write_to_different_paths(tmp_path):
    """16 ranks produce 16 distinct artifacts; sharing one path would race them onto each other."""
    spec = _spec(tmp_path)
    paths = set()
    for pe in range(4):
        rank = RankScope(world_size=4, pe=pe)
        keysha = _key(spec, prefix="k", backend=BACKEND_DUMP_OBJECT, rank=rank)
        paths.add(artifact_paths(keysha, spec, rank)[0])
    assert len(paths) == 4


def test_a_pe_outside_the_world_is_refused():
    """Always a caller bug, and cheaper to catch here than as a miss nobody explains."""
    with pytest.raises(ValueError, match="out of range"):
        RankScope(world_size=4, pe=4)


# ---------------------------------------------------------------------------
# Round-trip, and the checks that make a hit trustworthy.
# ---------------------------------------------------------------------------
def test_a_written_artifact_reads_back_with_its_bytes(tmp_path):
    """The positive control the negative ones are measured against."""
    spec = _spec(tmp_path)
    keysha, path = _put(spec, b"\x00\x01\x02payload")
    got = read_artifact(keysha, spec, prefix="k", backend=BACKEND_DUMP_OBJECT)
    assert got is not None
    assert got.blob == b"\x00\x01\x02payload"
    assert got.path == path
    assert got.meta["schema"] == ARTIFACT_SCHEMA_VERSION
    assert got.meta["backend"] == BACKEND_DUMP_OBJECT


def test_a_missing_entry_is_a_plain_miss(tmp_path):
    """A miss returns None and quarantines nothing -- 'absent' is not 'corrupt'."""
    spec = _spec(tmp_path)
    keysha = _key(spec, prefix="k", backend=BACKEND_DUMP_OBJECT)
    assert read_artifact(keysha, spec, prefix="k", backend=BACKEND_DUMP_OBJECT) is None
    assert list(tmp_path.glob("*.rejected")) == []


def test_a_backend_tag_mismatch_is_refused_and_quarantined(tmp_path):
    """THE corner-E control: a tvm-ffi image must never reach a non-tvm-ffi loader.

    Measured: ``dump_to_object`` on a TVM-FFI object writes 46616 plausible bytes and only the READ
    fails, in a later process, with a symbol-materialisation error from inside an execution engine.
    The tag turns that into a miss.
    """
    spec = _spec(tmp_path)
    keysha, path = _put(spec, backend=BACKEND_DUMP_OBJECT)
    # Rewrite the meta's tag to simulate an entry minted by the other backend at this same path.
    m_path = path.with_name(path.stem + ".meta.json")
    meta = json.loads(m_path.read_text())
    meta["backend"] = BACKEND_EXPORT_C
    m_path.write_text(json.dumps(meta))

    assert read_artifact(keysha, spec, prefix="k", backend=BACKEND_DUMP_OBJECT) is None
    assert not path.exists(), "the offending entry must be moved aside, not merely skipped"
    assert (tmp_path / f"{path.name}.rejected").exists()
    assert any("backend mismatch" in r for r in ac.STATS.reject_reasons)


def test_a_wrong_rank_artifact_is_refused(tmp_path):
    """The check whose absence is a FAULT, not an error.

    A wrong-rank artifact registered with ``library_init`` gives CUDA_ERROR_ILLEGAL_ADDRESS inside
    ``nvshmem init.cu:2183`` and poisons the context for everything after. Here the meta check
    catches it with no device involved at all.
    """
    spec = _spec(tmp_path)
    mine = RankScope(world_size=4, pe=1)
    keysha = _key(spec, prefix="k", backend=BACKEND_DUMP_OBJECT, rank=mine)
    o_path, m_path = artifact_paths(keysha, spec, mine)
    write_artifact(b"bytes", keysha, spec, prefix="k", backend=BACKEND_DUMP_OBJECT, rank=mine)
    meta = json.loads(m_path.read_text())
    meta["pe"] = 2  # someone else's artifact, sitting at my path
    m_path.write_text(json.dumps(meta))

    assert read_artifact(keysha, spec, prefix="k", backend=BACKEND_DUMP_OBJECT, rank=mine) is None
    assert not o_path.exists()


def test_a_wrong_world_size_artifact_is_refused(tmp_path):
    """A different world means a different PE mapping even at the same ``pe``."""
    spec = _spec(tmp_path)
    mine = RankScope(world_size=4, pe=1)
    keysha = _key(spec, prefix="k", backend=BACKEND_DUMP_OBJECT, rank=mine)
    _, m_path = artifact_paths(keysha, spec, mine)
    write_artifact(b"bytes", keysha, spec, prefix="k", backend=BACKEND_DUMP_OBJECT, rank=mine)
    meta = json.loads(m_path.read_text())
    meta["world_size"] = 8
    m_path.write_text(json.dumps(meta))
    assert read_artifact(keysha, spec, prefix="k", backend=BACKEND_DUMP_OBJECT, rank=mine) is None


def test_a_prefix_mismatch_is_refused(tmp_path):
    """The prefix is baked into the image, so serving one for another hands back wrong symbols."""
    spec = _spec(tmp_path)
    keysha, path = _put(spec, prefix="k")
    m_path = path.with_name(path.stem + ".meta.json")
    meta = json.loads(m_path.read_text())
    meta["prefix"] = "different"
    m_path.write_text(json.dumps(meta))
    assert read_artifact(keysha, spec, prefix="k", backend=BACKEND_DUMP_OBJECT) is None


def test_a_truncated_artifact_falls_back_to_compiling(tmp_path):
    """A short read must be a miss, never a load. The sha and the length both catch it."""
    spec = _spec(tmp_path)
    keysha, path = _put(spec, b"0123456789" * 10)
    path.write_bytes(b"0123456789")  # truncated behind the meta's back
    assert read_artifact(keysha, spec, prefix="k", backend=BACKEND_DUMP_OBJECT) is None
    assert not path.exists()


def test_an_unparseable_meta_is_refused(tmp_path):
    """A half-written sidecar is corruption, and corruption is quarantined rather than guessed at."""
    spec = _spec(tmp_path)
    keysha, path = _put(spec)
    path.with_name(path.stem + ".meta.json").write_text("{not json")
    assert read_artifact(keysha, spec, prefix="k", backend=BACKEND_DUMP_OBJECT) is None
    assert not path.exists()


def test_a_schema_bump_makes_old_entries_unreachable(tmp_path):
    """Old entries become invisible rather than mis-interpreted -- the always-correct migration."""
    spec = _spec(tmp_path)
    keysha, path = _put(spec)
    m_path = path.with_name(path.stem + ".meta.json")
    meta = json.loads(m_path.read_text())
    meta["schema"] = ARTIFACT_SCHEMA_VERSION + 1
    m_path.write_text(json.dumps(meta))
    assert read_artifact(keysha, spec, prefix="k", backend=BACKEND_DUMP_OBJECT) is None


# ---------------------------------------------------------------------------
# Quarantine: renamed, never deleted; a second corruption is a second fact.
# ---------------------------------------------------------------------------
def test_a_corrupt_artifact_is_quarantined_not_retried(tmp_path):
    """Skipping instead of moving leaves every later process re-reading the same bad file."""
    spec = _spec(tmp_path)
    keysha, path = _put(spec, b"good bytes")
    path.write_bytes(b"bad")
    assert read_artifact(keysha, spec, prefix="k", backend=BACKEND_DUMP_OBJECT) is None
    rejected = tmp_path / f"{path.name}.rejected"
    assert rejected.exists() and rejected.read_bytes() == b"bad"
    # second call: nothing left to re-read, and nothing new quarantined
    before = len(list(tmp_path.glob("*.rejected*")))
    assert read_artifact(keysha, spec, prefix="k", backend=BACKEND_DUMP_OBJECT) is None
    assert len(list(tmp_path.glob("*.rejected*"))) == before


def test_quarantine_records_the_reason_on_disk(tmp_path):
    """A rejected entry that does not say why makes the next reader re-derive the diagnosis."""
    spec = _spec(tmp_path)
    keysha, path = _put(spec, b"good bytes")
    path.write_bytes(b"bad")
    read_artifact(keysha, spec, prefix="k", backend=BACKEND_DUMP_OBJECT)
    why = tmp_path / f"{path.name}.rejected.why"
    assert why.exists() and "content mismatch" in why.read_text()


def test_a_second_corruption_never_overwrites_the_first(tmp_path):
    """Two corruptions of one key are two facts. Overwriting destroys the earlier evidence."""
    spec = _spec(tmp_path)
    keysha, path = _put(spec, b"first")
    path.write_bytes(b"x")
    read_artifact(keysha, spec, prefix="k", backend=BACKEND_DUMP_OBJECT)
    _put(spec, b"second")
    path.write_bytes(b"y")
    read_artifact(keysha, spec, prefix="k", backend=BACKEND_DUMP_OBJECT)
    assert (tmp_path / f"{path.name}.rejected").read_bytes() == b"x"
    assert (tmp_path / f"{path.name}.rejected.2").read_bytes() == b"y"


def test_quarantining_an_absent_file_is_benign(tmp_path):
    """A concurrent process may have moved it first; that is not an error for either of them."""
    assert quarantine(tmp_path / "nothing.o", "gone") is None


# ---------------------------------------------------------------------------
# Publication is atomic; a reader never sees half a file.
# ---------------------------------------------------------------------------
def test_no_temporary_file_survives_a_successful_write(tmp_path):
    """A leftover ``.tmp`` means the rename did not happen and a reader could see a partial file."""
    spec = _spec(tmp_path)
    _put(spec, b"payload" * 100)
    assert list(tmp_path.glob("*.tmp")) == []


def test_the_meta_is_published_before_the_artifact(tmp_path, monkeypatch):
    """Ordering matters: a visible ``.o`` with no meta reads as corrupt on the very first lookup.

    Simulated by failing the artifact rename and asserting the meta -- not the ``.o`` -- is what got
    published, which is the safe half to have alone (a meta with no ``.o`` is an ordinary miss).
    """
    spec = _spec(tmp_path)
    keysha = _key(spec, prefix="k", backend=BACKEND_DUMP_OBJECT)
    real_replace = os.replace
    calls = []

    def failing_replace(src, dst):
        calls.append(str(dst))
        if str(dst).endswith(".o"):
            raise OSError("simulated failure publishing the artifact")
        return real_replace(src, dst)

    monkeypatch.setattr(ac.os, "replace", failing_replace)
    assert write_artifact(b"bytes", keysha, spec, prefix="k", backend=BACKEND_DUMP_OBJECT) is None
    assert calls[0].endswith(".meta.json"), "meta must be published first"
    assert (tmp_path / f"{keysha}.meta.json").exists()
    assert not (tmp_path / f"{keysha}.o").exists()


def test_a_failed_write_is_not_an_error(tmp_path, monkeypatch):
    """The caller already holds a correct compiled kernel; a cache that can fail the run is worse
    than no cache. ``write_artifact`` returns None and counts it, and never propagates."""
    spec = PersistSpec(key=("cfg",), dir=str(tmp_path / "unwritable"))
    ac.STATS.reset()
    monkeypatch.setattr(ac.Path, "write_bytes", lambda self, b: (_ for _ in ()).throw(OSError("no")))
    keysha = _key(spec, prefix="k", backend=BACKEND_DUMP_OBJECT)
    assert write_artifact(b"bytes", keysha, spec, prefix="k", backend=BACKEND_DUMP_OBJECT) is None
    assert ac.STATS.write_failures == 1


def test_two_writers_of_the_same_key_agree(tmp_path):
    """Corner A measured ``dump_to_object`` deterministic, so a concurrent double-write is benign.

    This pins the consequence the design relies on: the second write leaves the entry readable and
    byte-identical, so the write path needs no lock.
    """
    spec = _spec(tmp_path)
    keysha, _ = _put(spec, b"identical bytes")
    keysha2, _ = _put(spec, b"identical bytes")
    assert keysha == keysha2
    got = read_artifact(keysha, spec, prefix="k", backend=BACKEND_DUMP_OBJECT)
    assert got is not None and got.blob == b"identical bytes"


# ---------------------------------------------------------------------------
# Counters -- how a test tells a working cache from an inert one.
# ---------------------------------------------------------------------------
def test_the_counters_distinguish_a_hit_from_a_miss(tmp_path):
    """A store that silently never hits passes every positive correctness assertion made about it."""
    spec = _spec(tmp_path)
    ac.STATS.reset()
    keysha = _key(spec, prefix="k", backend=BACKEND_DUMP_OBJECT)
    assert read_artifact(keysha, spec, prefix="k", backend=BACKEND_DUMP_OBJECT) is None
    assert (ac.STATS.hits, ac.STATS.misses) == (0, 1)
    write_artifact(b"b", keysha, spec, prefix="k", backend=BACKEND_DUMP_OBJECT)
    assert read_artifact(keysha, spec, prefix="k", backend=BACKEND_DUMP_OBJECT) is not None
    assert (ac.STATS.hits, ac.STATS.misses, ac.STATS.writes) == (1, 1, 1)


# ---------------------------------------------------------------------------
# backend_for reads the object, not a flag.
# ---------------------------------------------------------------------------
class _FakeTVMFFICompiled:
    """Stand-in whose CLASS NAME is what ``backend_for`` inspects, so no compile is needed."""


class _TVMFFIJitCompiledFunctionWithKwargs(_FakeTVMFFICompiled):
    """Named exactly as the real class is, because the selector matches on ``"TVMFFI"`` in the MRO."""


class _CudaDialectJitCompiledFunction:
    """The non-tvm-ffi class the A2A kernels get."""


def test_backend_for_selects_export_c_for_a_tvm_ffi_object():
    """Its ``export_to_c`` OVERRIDE bypasses the C header generator; nothing else can read it back."""
    assert ac.backend_for(_TVMFFIJitCompiledFunctionWithKwargs()) == BACKEND_EXPORT_C


def test_backend_for_selects_dump_object_otherwise():
    """The A2A route: ``dump_to_object`` -> ``load_module(enable_tvm_ffi=False)``."""
    assert ac.backend_for(_CudaDialectJitCompiledFunction()) == BACKEND_DUMP_OBJECT


def test_serialize_refuses_an_unknown_backend():
    """A typo'd tag must not silently fall through to one of the real serializers."""
    with pytest.raises(ValueError, match="unknown artifact backend"):
        ac.serialize(_CudaDialectJitCompiledFunction(), "nonsense", "k")


def test_deserialize_refuses_an_unknown_backend(tmp_path):
    """Same on the read side, and this one is reached with a verified artifact in hand."""
    art = ac.LoadedArtifact(path=Path("/nonexistent.o"), meta={}, blob=b"", backend="nonsense")
    with pytest.raises(ValueError, match="unknown artifact backend"):
        ac.deserialize(art, "k")


# ── the all-rank program key: combination logic, testable without a GPU ────────────────────────
class _FakeDist:
    """A stand-in for `torch.distributed` that replays a fixed set of per-rank gather results.

    Purpose
        The COMBINATION rules -- ordering, world size, failure reduction -- are arithmetic on
        gathered values and need no GPU, no group and no cutlass. Testing them against a real group
        would make them distributed tests, which run in one configuration per launch and so could
        never compare world 2 against world 4 in a single assertion.

    Input requirements
        payload: the list `all_gather_object` should produce, one `(ok, hash)` per rank. Its length
            IS the world size, so a caller changing one must change the other.
    """

    def __init__(self, payload):
        self._payload = list(payload)

    def is_available(self):
        return True

    def is_initialized(self):
        return True

    def get_world_size(self, group=None):
        return len(self._payload)

    def all_gather_object(self, out, obj, group=None):
        out[:] = list(self._payload)


def _key_with(monkeypatch, payload, own="a" * 64):
    """Compute an all-rank key against a faked gather, with this rank's own trace stubbed out.

    Patches BOTH `sys.modules["torch.distributed"]` and the `distributed` ATTRIBUTE of `torch`, and
    the second is the one that works: `import torch.distributed as dist` resolves through
    `getattr(torch, "distributed")`, not through `sys.modules`, so patching only the module table
    left the real `torch.distributed` in place. The symptom was every payload returning one hash --
    the function had silently taken its no-group path, where the stubbed local hash is the only
    input, so four tests asserted four different things about the same constant.
    """
    import sys

    import torch

    fake = _FakeDist(payload)
    monkeypatch.setattr(ac, "_rank_mlir_hash", lambda *a, **k: own)
    monkeypatch.setattr(torch, "distributed", fake)
    monkeypatch.setitem(sys.modules, "torch.distributed", fake)
    return ac.all_rank_program_key(lambda: None, ())


def test_the_all_rank_key_is_ORDER_SENSITIVE(monkeypatch):
    """Two different rank->program assignments must not collide.

    The gather is combined in PE ORDER, never as a set or a sum. A set would make "rank 0 runs A,
    rank 1 runs B" and "rank 0 runs B, rank 1 runs A" the same key -- and for an A2A kernel, whose
    code bakes in a peer layout, those are different programs that would then share one artifact.
    """
    ab = _key_with(monkeypatch, [(True, "A" * 64), (True, "B" * 64)])
    ba = _key_with(monkeypatch, [(True, "B" * 64), (True, "A" * 64)])
    assert ab != ba, "the combine is order-insensitive; a swapped rank assignment collides"


def test_the_all_rank_key_SEPARATES_WORLD_SIZES(monkeypatch):
    """The same rank-uniform kernel at world 2 and world 4 is a different distributed program.

    Free here, where the composed key needed `world_size` threaded in as an explicit term: the
    gathered tuple is simply a different length.
    """
    h = "C" * 64
    w2 = _key_with(monkeypatch, [(True, h)] * 2)
    w4 = _key_with(monkeypatch, [(True, h)] * 4)
    assert w2 != w4


def test_every_rank_derives_the_SAME_key_from_the_same_gather(monkeypatch):
    """The point of combining: the result is rank-free, so `agree_on_program_key` holds and is TRUE.

    Simulated by computing the key with each rank's OWN hash differing -- which is the A2A case,
    measured at world 2 as `ee4cb228` vs `d9846317` -- while the gathered tuple is identical. Every
    rank must land on one value.
    """
    payload = [(True, "D" * 64), (True, "E" * 64)]
    k0 = _key_with(monkeypatch, payload, own="D" * 64)
    k1 = _key_with(monkeypatch, payload, own="E" * 64)
    assert k0 == k1, "ranks with different local programs derived different all-rank keys"


def test_ONE_ranks_trace_failure_fails_EVERY_rank(monkeypatch):
    """The failure this prevents is a HANG, not a wrong key.

    `all_gather_object` is a collective on the compile path. A rank that raises and leaves alone
    blocks every peer in it until a watchdog kills the job -- the same hazard, on the same kind of
    path, that the symmetric-OOM precheck had to fix. So a local failure is carried INTO the gather
    and turned into a refusal on every rank together.
    """
    with pytest.raises(ac.PeerTraceError, match=r"1 of 2 ranks could not produce"):
        _key_with(monkeypatch, [(True, "F" * 64), (False, "")])


def test_a_single_rank_with_no_group_still_gets_a_key(monkeypatch):
    """A single-process compile has nobody to disagree with and must not require a group."""
    import sys

    monkeypatch.setattr(ac, "_rank_mlir_hash", lambda *a, **k: "G" * 64)
    monkeypatch.setitem(sys.modules, "torch.distributed", None)
    k = ac.all_rank_program_key(lambda: None, ())
    assert isinstance(k, str) and len(k) == 64


def test_a_factory_WITHOUT_dsl_support_FALLS_BACK_to_the_flat_key(monkeypatch, tmp_path):
    """With no MLIR keying, key on the functor's arguments -- do not refuse.

    This test used to assert the opposite, and the reason it changed is worth stating rather than
    just editing away. The refusal was correct while the only alternative was the COMPOSED key,
    which could not see 37 ``const_expr`` gates and had a demonstrated collision
    (``m6_control.py``: two functors emitting different code, one key ``51ab5a2f``). Handing a
    caller that key while reporting success is the worst outcome, so refusing named the fix.

    But the pinned DSL is 4.4.2, which has no ``to_precompiled_mlir`` -- so on the DSL this repo
    actually runs, the refusal fired on EVERY persisted compile and the artifact cache did nothing.
    ``compile_key()`` now reads those 37 gates (``COMPILE_GATED_ATTRS``), so the flat key is
    complete in the same sense the MLIR key is, and the reason for the refusal is gone.

    The functor is a stub with a real ``compile_key()``, because that method IS the key on this
    path: a stub without one would test the AttributeError instead.
    """
    monkeypatch.setattr(ac, "mlir_keying_available", lambda: False)

    class _Stub:
        """Minimal functor: one compile-time knob, exposed the way the paradigm exposes them."""

        def __init__(self, knob):
            self.knob = knob

        def compile_key(self):
            """The whole compile-time surface of this stub."""
            return {"knob": self.knob}

    def key(knob):
        return ac.resolve_program_key(
            None, spec=ac.PersistSpec(dir=str(tmp_path)), prefix="k",
            backend=ac.BACKEND_DUMP_OBJECT, op=_Stub(knob), compile_args=(), options="",
        )

    k0 = key(0)
    assert isinstance(k0, str) and len(k0) == 64, f"expected a sha256 hex digest, got {k0!r}"
    assert k0 == key(0), "the flat key must be deterministic, or nothing ever hits"
    assert k0 != key(1), (
        "two configurations differing in one compile-time knob shared a program key -- exactly the "
        "collision the composed key was deleted for"
    )


def test_a_functor_with_NO_compile_key_is_REFUSED_rather_than_keyed_on_its_type(tmp_path):
    """The one thing the flat key must still refuse: a functor whose surface it cannot read.

    The old refusal here was "no ``op_factory``". That is no longer the failure -- the factory is
    optional now -- but the hazard it protected against has simply moved: keying a functor on its
    TYPE alone gives every configuration of it one artifact, which is the same collision with a
    different cause. `flat_key_components` therefore reads ``compile_key()`` unguarded, so a functor
    without one raises ``AttributeError`` at the key rather than silently sharing an entry.

    Deliberately NOT a ``getattr(op, "compile_key", lambda: {})``: a default of "no configuration"
    is indistinguishable from a functor that genuinely has none, and the two must not share an
    artifact.
    """
    with pytest.raises(AttributeError, match="compile_key"):
        ac.resolve_program_key(
            None, spec=ac.PersistSpec(dir=str(tmp_path)), prefix="k",
            backend=ac.BACKEND_DUMP_OBJECT, op=object(), compile_args=(), options="",
        )


def test_the_mlir_key_still_separates_BACKEND_and_PREFIX(monkeypatch, tmp_path):
    """The MLIR hash covers the PROGRAM; it says nothing about how the image was stored.

    `prefix` was MEASURED to change the image bytes and the two backends produce mutually unreadable
    images, so both must still key -- otherwise one artifact is served to a reader that cannot parse
    it. Keeping them is why `resolve_program_key` hashes a tail rather than returning the bare
    all-rank key.
    """
    monkeypatch.setattr(ac, "mlir_keying_available", lambda: True)
    monkeypatch.setattr(ac, "all_rank_program_key", lambda *a, **k: "Z" * 64)
    spec = ac.PersistSpec(dir=str(tmp_path))
    base = dict(spec=spec, op=None, compile_args=(), options="")
    a = ac.resolve_program_key(lambda: None, prefix="k", backend=ac.BACKEND_DUMP_OBJECT, **base)
    b = ac.resolve_program_key(lambda: None, prefix="OTHER", backend=ac.BACKEND_DUMP_OBJECT, **base)
    c = ac.resolve_program_key(lambda: None, prefix="k", backend=ac.BACKEND_EXPORT_C, **base)
    assert len({a, b, c}) == 3, "prefix or backend stopped keying once the MLIR hash was adopted"


def test_the_mlir_key_covers_SCHEMA_and_ARCH_which_the_IR_does_not(monkeypatch, tmp_path):
    """The two terms M5 moved from belt-and-braces to load-bearing.

    Both were in the composed key. Neither is in the emitted IR, so while the fallback existed they
    were covered on the path most compiles took. `schema` is separately checked on READ, so losing
    it from the key would only have cost a wasted read. `arch` is WRITTEN to the meta and never
    checked -- so with the fallback gone it would have been in neither the key nor the validation,
    and one cache directory shared between two GPU architectures would hand one arch the other's
    image. That is a fault, not a miss.
    """
    monkeypatch.setattr(ac, "mlir_keying_available", lambda: True)
    monkeypatch.setattr(ac, "all_rank_program_key", lambda *a, **k: "Z" * 64)
    spec = ac.PersistSpec(dir=str(tmp_path))
    base = dict(
        spec=spec, prefix="k", backend=ac.BACKEND_DUMP_OBJECT, op=None, compile_args=(), options=""
    )
    monkeypatch.setattr(ac, "cuda_arch", lambda: "sm_90")
    sm90 = ac.resolve_program_key(lambda: None, **base)
    monkeypatch.setattr(ac, "cuda_arch", lambda: "sm_100")
    sm100 = ac.resolve_program_key(lambda: None, **base)
    assert sm90 != sm100, "arch stopped keying; a two-arch cache dir would cross-serve images"

    monkeypatch.setattr(ac, "cuda_arch", lambda: "sm_90")
    monkeypatch.setattr(ac, "ARTIFACT_SCHEMA_VERSION", ac.ARTIFACT_SCHEMA_VERSION + 1)
    bumped = ac.resolve_program_key(lambda: None, **base)
    assert bumped != sm90, "a schema bump must re-key, or a stale entry is read under a new format"


# ── the flat program key: `arg_key`, `flat_key_components`, `program_key_from_args` ────────────
@matrix_exempt(
    "asserts a property of the KEY function -- that two operand descriptions differing in one "
    "respect key differently -- which is about the key, not about any declared shape"
)
@pytest.mark.parametrize(
    "left,right,what",
    [
        (("i64", (8,), 8), ("i64", (16,), 8), "extent"),
        (("i64", (8,), 8), ("i32", (8,), 8), "dtype"),
        (("i64", (8,), 8), ("i64", (8,), 16), "assumed_align"),
    ],
)
def test_arg_key_separates_operands_that_differ_in_one_respect(left, right, what):
    """Two operands differing in exactly one compile-relevant respect must key differently.

    This is the bar both keys that use it rest on -- the in-process reuse key and the disk program
    key. A key that collapsed dtype, or alignment, would hand a bf16 caller an fp16 kernel silently:
    the one failure an operand key can have that no test of the kernels themselves would see.

    ``assumed_align`` is parametrized because it is the MEASURED blind spot. The tensor ``repr``
    prints the address, the layout and the dynamic marks and nothing else, so 8-byte and 16-byte
    promises over the same shape produce identical reprs; `arg_key` adds it explicitly for that
    reason, and this cell is what fails if that line is ever removed as redundant.

    Fake tensors, so this needs no CUDA device.
    """
    import cutlass
    import cutlass.cute as cute

    dt = {"i64": cutlass.Int64, "i32": cutlass.Int32}

    def mk(spec):
        kind, shape, align = spec
        return cute.runtime.make_fake_tensor(dt[kind], shape, stride=(1,), assumed_align=align)

    a, b = ac.arg_key(mk(left)), ac.arg_key(mk(right))
    assert a != b, f"arg_key collapsed a difference in {what}: both keyed as {a}"


@matrix_exempt("asserts that the key is stable across allocations, not a shape behaviour")
def test_arg_key_is_stable_across_two_allocations_of_the_same_layout():
    """Two distinct tensors with the same layout must key IDENTICALLY, or nothing ever hits.

    The tensor ``repr`` embeds the heap address, so a naive key changes on every allocation and both
    caches degrade to pure cost with no benefit -- silently, because a cache that never hits still
    returns correct kernels. `arg_key` strips the address; this is what proves it did.
    """
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("from_dlpack needs a real allocation; the fake-tensor cells cover the rest")
    from cutlass.cute.runtime import from_dlpack

    a = torch.empty(8, 16, device="cuda", dtype=torch.bfloat16)
    b = torch.empty(8, 16, device="cuda", dtype=torch.bfloat16)
    assert a.data_ptr() != b.data_ptr(), "the two tensors must not share storage"
    assert ac.arg_key(from_dlpack(a, assumed_align=16)) == ac.arg_key(
        from_dlpack(b, assumed_align=16)
    )


@matrix_exempt("asserts that a dynamic MARK changes the key; independent of any declared shape")
def test_arg_key_separates_a_dynamic_layout_from_a_static_one():
    """``mark_layout_dynamic()`` must change the key -- and it is invisible to every public field.

    Measured on cutlass-dsl 4.4.2: after marking, ``shape``, ``stride`` and ``_is_dynamic`` are all
    UNCHANGED, and ``.layout`` raises ``NotImplementedError``. Only the ``repr`` shows the ``?``
    marks. A key built from the obvious attributes would serve a static-shape kernel to a
    dynamic-shape caller -- a wrong kernel, not a slow one.
    """
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("mark_layout_dynamic needs a real tensor")
    from cutlass.cute.runtime import from_dlpack

    a = torch.empty(8, 16, device="cuda", dtype=torch.bfloat16)
    static = ac.arg_key(from_dlpack(a, assumed_align=16))
    dynamic = ac.arg_key(from_dlpack(a, assumed_align=16).mark_layout_dynamic())
    assert static != dynamic, f"a dynamic mark did not change the key; both are {static}"


@matrix_exempt("asserts the program key's rank-invariance, which no declared shape axis varies")
def test_the_flat_program_key_drops_rank_scoped_components():
    """Two ranks of one job must produce the SAME program key, or the agreement check fails.

    `agree_on_program_key` is a one-value comparison across ranks, and `artifact_key` treats ``pe``
    as the ONLY component allowed to differ. A functor's rank ordinal is folded into its kernel like
    any other constant, so it appears in ``compile_key()`` and would make every rank's program key
    unique -- turning a working cache into a job-wide raise.

    Dropping it loses nothing: `artifact_key` adds ``(world_size, pe)``, so each rank still reads
    and writes its own file. What this test protects is the INVARIANT that the drop is complete --
    a new rank-carrying component added to a functor without being listed in
    ``RANK_SCOPED_KEY_NAMES`` fails here rather than at a 16-rank launch.
    """

    class _Ranked:
        """A functor whose only varying knob is its rank ordinal."""

        def __init__(self, pe):
            self._a2a_my_cp_rank = pe

        def compile_key(self):
            """Configuration as the paradigm reports it, rank ordinal included."""
            return {"tile": 128, "_a2a_my_cp_rank": self._a2a_my_cp_rank}

    keys = {ac.program_key_from_args(_Ranked(pe), (), "") for pe in range(8)}
    assert len(keys) == 1, (
        f"the program key differs across ranks ({len(keys)} distinct over 8 PEs). Every "
        "rank-carrying component must be in RANK_SCOPED_KEY_NAMES; artifact_key restores the "
        "distinction with (world_size, pe)"
    )


@matrix_exempt("asserts the program key's completeness, which no declared shape axis varies")
def test_the_flat_program_key_separates_every_configuration_component():
    """Configuration, operands and options must each move the key on their own.

    Three independent surfaces decide a compile, and a key missing any one of them serves a WRONG
    artifact rather than missing one -- which is the failure the composed key was deleted for. This
    varies each in isolation so a regression names which surface was dropped.
    """
    import cutlass
    import cutlass.cute as cute

    class _Cfg:
        """A functor with one compile-time knob."""

        def __init__(self, knob):
            self.knob = knob

        def compile_key(self):
            """Its whole compile-time surface."""
            return {"knob": self.knob}

    t8 = cute.runtime.make_fake_tensor(cutlass.Int64, (8,), stride=(1,), assumed_align=8)
    t16 = cute.runtime.make_fake_tensor(cutlass.Int64, (16,), stride=(1,), assumed_align=8)

    base = ac.program_key_from_args(_Cfg(0), (t8,), "-a")
    assert base != ac.program_key_from_args(_Cfg(1), (t8,), "-a"), "configuration did not key"
    assert base != ac.program_key_from_args(_Cfg(0), (t16,), "-a"), "operands did not key"
    assert base != ac.program_key_from_args(_Cfg(0), (t8,), "-b"), "options did not key"
    assert base == ac.program_key_from_args(_Cfg(0), (t8,), "-a"), "the key is not deterministic"


@matrix_exempt("asserts that two VALUES of one compile-time argument key apart; not a shape axis")
def test_arg_key_separates_two_values_of_a_cutlass_numeric_instance():
    """``Float32(1e-5)`` and ``Float32(1e-6)`` must not share a key -- ``eps`` is folded in.

    The numeric TYPE (``Float32``, a class) and a numeric INSTANCE are different objects and the
    first version of `arg_key` only recognised the type, so an instance fell through to the
    unknown-object fallback and every ``Float32`` keyed identically. ``EpilogueArguments.eps``
    arrives as one of these, so two engines differing ONLY in their LayerNorm epsilon would have
    shared a compile -- a wrong kernel with no diagnostic.

    The type branch is checked alongside, because the fix must not swallow it: a dtype argument is
    the class and has to keep keying as the class.
    """
    import cutlass

    lo, hi = ac.arg_key(cutlass.Float32(1e-5)), ac.arg_key(cutlass.Float32(1e-6))
    assert lo != hi, f"two eps values collided: both {lo}"
    assert ac.arg_key(cutlass.Float32) != lo, "the numeric TYPE must not key as an instance"
    assert ac.arg_key(cutlass.Float32(1e-5)) == lo, "the instance key must be deterministic"


@matrix_exempt("asserts that two callables key apart; a property of the key, not of a shape")
def test_arg_key_separates_two_callables_rather_than_calling_them_all_a_function():
    """Two activation functions must key differently, or every activation shares one compile.

    Every plain function is a ``builtins.function``, so a key built from the TYPE gives them all one
    value. ``act_fn`` reaches the front store's epilogue and is folded in.

    There is NO live collision today -- ``gate_fn_map`` has a single entry -- and that is the reason
    to pin it now: a test that only fails once a second gate lands is a test that arrives after the
    wrong kernel does. Two locally-defined functions stand in for that second entry.
    """

    def alpha(x):
        """Stand-in for one activation."""
        return x

    def beta(x):
        """Stand-in for another."""
        return x

    assert ac.arg_key(alpha) != ac.arg_key(beta), (
        f"two distinct functions collided: both {ac.arg_key(alpha)}"
    )
    assert ac.arg_key(alpha) == ac.arg_key(alpha), "the callable key must be deterministic"


@matrix_exempt("asserts filesystem modes on published entries; a store property, not a shape")
def test_a_published_artifact_and_its_meta_are_both_0600(tmp_path):
    """An artifact is loaded as EXECUTABLE code, so a world-readable one leaks what the meta says.

    Both files are asserted. The meta is the one that gets forgotten -- it carries no code, so it
    looks harmless, while naming the prefix, the backend, the rank and the SHA of what this user
    compiled.
    """
    spec = _spec(tmp_path)
    _, path = _put(spec, b"payload")
    m_path = path.with_name(path.stem + ".meta.json")
    for p in (path, m_path):
        assert stat.S_IMODE(os.stat(p).st_mode) == 0o600, (
            f"{p.name} is {stat.S_IMODE(os.stat(p).st_mode):04o}, expected 0600"
        )


@matrix_exempt("asserts the artifact root's mode; a store property, not a shape")
def test_the_artifact_directory_is_created_0700(tmp_path):
    """The directory, not just its contents -- a writable dir lets an attacker replace an entry."""
    spec = _spec(tmp_path)
    _put(spec, b"payload")
    assert stat.S_IMODE(os.stat(tmp_path).st_mode) == 0o700, (
        f"artifact root is {stat.S_IMODE(os.stat(tmp_path).st_mode):04o}, expected 0700"
    )


@matrix_exempt("asserts a refusal to read a symlinked entry; a store property, not a shape")
def test_a_symlinked_artifact_is_a_MISS_and_is_NOT_quarantined(tmp_path):
    """Refuse it, but do not rename it -- quarantine is for corruption, not for provenance.

    The distinction matters: `quarantine` MOVES the file, and moving a path this process has just
    decided it does not trust is how a defence starts modifying things outside its own cache. The
    bytes are left identical to the original, so only the ``O_NOFOLLOW`` check can tell them apart.
    """
    spec = _spec(tmp_path)
    keysha, path = _put(spec, b"payload")
    assert read_artifact(keysha, spec, prefix="k", backend=BACKEND_DUMP_OBJECT) is not None

    elsewhere = tmp_path.parent / "elsewhere.o"
    elsewhere.write_bytes(path.read_bytes())
    os.replace(str(path), str(tmp_path.parent / "orig.o"))
    os.symlink(str(elsewhere), str(path))

    assert read_artifact(keysha, spec, prefix="k", backend=BACKEND_DUMP_OBJECT) is None
    assert list(tmp_path.glob("*.rejected")) == [], "a refused-for-provenance entry was quarantined"
    assert os.path.islink(path), "the symlink must be left in place, not replaced"


@matrix_exempt("asserts a refusal to read a foreign-owned entry; a store property, not a shape")
def test_a_foreign_owned_artifact_is_a_MISS(tmp_path, monkeypatch):
    """``chown`` needs privilege, so the comparison is exercised from the ``geteuid`` side."""
    spec = _spec(tmp_path)
    keysha, _ = _put(spec, b"payload")
    assert read_artifact(keysha, spec, prefix="k", backend=BACKEND_DUMP_OBJECT) is not None
    real = os.geteuid()
    monkeypatch.setattr(cache_security.os, "geteuid", lambda: real + 4242)
    assert read_artifact(keysha, spec, prefix="k", backend=BACKEND_DUMP_OBJECT) is None


@matrix_exempt("asserts read/write are skipped on an unsafe root; a store property, not a shape")
def test_an_unsafe_artifact_root_skips_the_write_and_misses_the_read(tmp_path, monkeypatch):
    """A refused root must cost a recompile, never an error, and must leave no files behind."""
    hostile = tmp_path / "hostile"
    hostile.mkdir()
    os.chmod(hostile, 0o777)  # AFTER mkdir: mkdir's mode argument is masked by the umask
    cache_security.reset_cache()
    try:
        spec = PersistSpec(key=("cfg", 1), dir=str(hostile), mode="rw")
        keysha = _key(spec, prefix="k", backend=BACKEND_DUMP_OBJECT)
        with pytest.warns(RuntimeWarning, match="refusing to use cache directory"):
            assert (
                write_artifact(b"payload", keysha, spec, prefix="k", backend=BACKEND_DUMP_OBJECT)
                is None
            )
        assert not [p for p in hostile.iterdir()], "a refused root was written to"
        assert read_artifact(keysha, spec, prefix="k", backend=BACKEND_DUMP_OBJECT) is None
    finally:
        cache_security.reset_cache()


@matrix_exempt("asserts SHA rejection independent of length; a store property, not a shape")
def test_a_SAME_LENGTH_byte_substitution_is_caught_by_the_sha_not_the_size(tmp_path):
    """Without this, every corruption test in this file could be passing on the length check alone.

    The existing truncation test changes both the size and the digest, so it cannot distinguish a
    store that verifies SHA-256 from one that only compares ``len(blob)`` against the meta. One
    flipped byte, same length, separates them.
    """
    spec = _spec(tmp_path)
    blob = b"payload-of-known-length"
    keysha, path = _put(spec, blob)
    meta_before = json.loads(path.with_name(path.stem + ".meta.json").read_text())

    corrupted = bytearray(blob)
    corrupted[0] ^= 0xFF
    assert len(corrupted) == len(blob), "the substitution must not change the length"
    path.write_bytes(bytes(corrupted))

    assert read_artifact(keysha, spec, prefix="k", backend=BACKEND_DUMP_OBJECT) is None
    assert meta_before["bytes"] == len(blob), (
        "the meta's size still matches, so only the SHA can refuse"
    )
    assert not path.exists(), "a content mismatch must be quarantined, not merely skipped"


@matrix_exempt("asserts a rejected parent propagates and creates no child; a store property")
def test_a_REJECTED_default_parent_propagates_and_no_artifacts_CHILD_is_created(
    tmp_path, monkeypatch
):
    """The default artifact dir lives under the JIT root, so a refused JIT root must stop it dead.

    Purpose
        `get_cache_path()` returns a path whether or not the directory is safe -- deliberately, so a
        message can name it. Building ``<that>/artifacts`` from it would ``mkdir`` INSIDE a
        directory this process has just refused, which is the single place it must not write.

    Semantics
        Both halves are asserted, and the second is the one a naive fix misses: the spec is unusable
        AND no ``artifacts`` directory appeared. An implementation that validated the child would
        report it unusable too -- after having created it.
    """
    hostile = tmp_path / "hostile_jit_root"
    hostile.mkdir()
    os.chmod(hostile, 0o777)  # AFTER mkdir: the mode argument is umask-masked
    monkeypatch.setenv("CPO_CACHE_DIR", str(hostile))
    monkeypatch.delenv(ac.ARTIFACT_DIR_ENV, raising=False)
    import fold_cp_ops._internal.cache_utils as cu

    monkeypatch.setattr(cu, "CACHE_DIR", str(hostile))
    cache_security.reset_cache()
    try:
        spec = PersistSpec(key=("cfg", 1), mode="rw")
        with pytest.warns(RuntimeWarning, match="refusing to use cache directory"):
            root = spec.resolve_root()
        assert not root.usable, "a spec under a refused parent was reported usable"
        assert "parent" in root.reason, f"the reason must name the parent: {root.reason!r}"
        assert not (hostile / "artifacts").exists(), (
            "an `artifacts` child was created inside a directory that had just been refused"
        )
    finally:
        cache_security.reset_cache()


@matrix_exempt("asserts pe=None and pe=0 key apart; a key property, not a shape")
def test_a_None_rank_and_pe_zero_do_NOT_share_an_artifact_key():
    """``None`` and ``0`` must not collide, and under a naive encoding they can.

    A non-ranked artifact keys ``pe=None``; a rank-0 nvshmem artifact keys ``pe=0``. Handing rank 0
    the non-ranked file is the failure mode `RankScope` exists to prevent -- it faults inside
    ``nvshmem init.cu`` rather than producing a wrong answer. The type-tagged encoder separates
    them; this pins that it does.
    """
    prog = "a" * 64
    plain = artifact_key(prog, None)
    ranked = artifact_key(prog, RankScope(world_size=2, pe=0))
    assert plain != ranked, f"pe=None and pe=0 produced the same key: {plain}"
    assert artifact_key(prog, RankScope(world_size=2, pe=0)) != artifact_key(
        prog, RankScope(world_size=2, pe=1)
    ), "two ranks share one key"

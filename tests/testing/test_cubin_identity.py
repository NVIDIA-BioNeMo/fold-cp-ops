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

"""Tests for `fold_cp_ops.testing.cubin_identity`.

Two tiers, and both are needed.

The SYNTHETIC tier builds CUDA-ELF-shaped blobs by hand and runs everywhere, including on a box
with no SM90 silicon. It is what pins the properties a real compile cannot isolate -- in
particular that a pure RENAME does not move the digest, which needs two artifacts differing ONLY
in a mangled name and is not something two real configurations can be made to produce on demand.

The COMPILED tier is the known-failing control the harvester is worthless without: two
deliberately different configurations must digest differently. A harvester that reported identity
on everything would pass every syntactic test one might write for it, and would then report a
perfect byte-identity sweep against `main` while measuring nothing.

**Neither tier subsumes the other, and that is MEASURED rather than asserted.** Crippling
`code_sections` two ways and re-running:

* a constant digest (identity on everything) -> the COMPILED control fails, plus two synthetic
  tests;
* a SIZE-ONLY digest -> ``2 failed, 4 passed``, and the compiled control is among the PASSES,
  because the two tile shapes emit ``.text`` of genuinely different length (13,952 vs 16,128 B).

So a size-only harvester sails through a real-compile control and then reports false identity for
any pair emitting the same amount of code with different instructions -- which is precisely what a
bring-back produces when statements are reordered at a fixed tile. Only the synthetic one-byte
case catches it. Do not delete either tier.
"""

import hashlib
import struct

import pytest
import torch

from fold_cp_ops.testing.cubin_identity import (
    EM_CUDA,
    assert_has_code,
    code_sections,
    differing_kinds,
    digest_export,
    exportable,
)
from fold_cp_ops.testing.kernel_matrix import matrix_exempt

pytestmark = matrix_exempt(
    "the subject is a host-side ELF parser and its digest. The compiled tier launches no kernel -- "
    "it compiles two configurations purely as INSTRUMENT INPUT, so the tile shapes here are a "
    "known-answer pair rather than a swept axis, and no numeric output exists to compare"
)

_SM = torch.cuda.get_device_capability(0)[0] if torch.cuda.is_available() else 0
requires_sm90 = pytest.mark.skipif(
    _SM != 9, reason=f"the compiled control needs sm_90; this GPU is sm_{_SM}0"
)


# ── synthetic CUDA-ELF fixtures ───────────────────────────────────────────────────────────────
def _elf(sections, *, machine=EM_CUDA, lead=b"HOSTOBJ\x7fELF-decoy-not-cuda\x00" * 4):
    """A minimal ELF64 image carrying `sections`, prefixed by `lead` so it is not at offset 0.

    The prefix matters: `code_sections` searches from index 1, because in a real ``export_to_c``
    object the HOST ELF sits at offset 0 and the CUDA ELF is embedded later. A fixture written at
    offset 0 would be skipped and every assertion below would be about the wrong bytes. The lead
    also contains a decoy ``\\x7fELF`` with a non-CUDA machine, which is what proves the scan
    selects on ``e_machine`` rather than taking the first magic it finds.

    Args:
        sections: ``[(name, sh_type, body_bytes), ...]``. Names are written verbatim, so a caller
            controls the mangled suffix the helper is supposed to strip. A ``.shstrtab`` is added
            automatically and must not be supplied.
        machine: ``e_machine`` to write. Anything but `EM_CUDA` must be skipped by the scan.
        lead: Bytes placed before the ELF.

    Returns:
        The image as ``bytes``.
    """
    names = [n for n, _, _ in sections] + [".shstrtab"]
    strtab, offsets = b"\x00", {}
    for n in names:
        offsets[n] = len(strtab)
        strtab += n.encode() + b"\x00"
    bodies = [b for _, _, b in sections] + [strtab]
    ehsize, shentsize = 64, 64
    # Bodies first, then the section-header table, so every sh_offset is known before it is written.
    body_off, cur = [], ehsize
    for b in bodies:
        body_off.append(cur)
        cur += len(b)
    shoff = cur

    eh = bytearray(ehsize)
    eh[0:4] = b"\x7fELF"
    eh[4] = 2  # ELFCLASS64
    eh[5] = 1  # ELFDATA2LSB
    struct.pack_into("<H", eh, 18, machine)
    struct.pack_into("<Q", eh, 40, shoff)
    struct.pack_into("<HHH", eh, 58, shentsize, len(bodies), len(bodies) - 1)  # shstrtab is last

    shdrs = bytearray()
    all_secs = list(sections) + [(".shstrtab", 3, strtab)]
    for (name, typ, _b), off, body in zip(all_secs, body_off, bodies):
        sh = bytearray(shentsize)
        struct.pack_into("<IIQQQQ", sh, 0, offsets[name], typ, 0, 0, off, len(body))
        shdrs += sh
    return lead + bytes(eh) + b"".join(bodies) + bytes(shdrs)


def _write(tmp_path, name, image):
    """Write one fixture image and return its path as ``str``."""
    p = tmp_path / name
    p.write_bytes(image)
    return str(p)


_TEXT_A = b"\x01\x02\x03\x04" * 64
_TEXT_B = b"\x01\x02\x03\x05" * 64  # one byte different, in the last word
_PROGBITS = 1


def test_two_different_code_bodies_digest_differently(tmp_path):
    """The property the whole harvester exists for: different instructions, different digest.

    Deliberately a ONE-BYTE difference in bodies of equal length, because a digest keyed on size
    alone -- an easy way to write this wrong -- would report these as identical.
    """
    a = _write(tmp_path, "a.o", _elf([(".text.kernel_Foo", _PROGBITS, _TEXT_A)]))
    b = _write(tmp_path, "b.o", _elf([(".text.kernel_Foo", _PROGBITS, _TEXT_B)]))
    da, db = code_sections(a), code_sections(b)
    assert_has_code(da, "a")
    assert_has_code(db, "b")
    assert da != db
    assert differing_kinds(da, db) == [".text"]
    assert da[".text"][0] == db[".text"][0], "equal SIZE, so size alone cannot be the discriminator"


def test_a_rename_alone_does_not_move_the_digest(tmp_path):
    """A pure rename must compare EQUAL -- the reason this is section content and not the image.

    The two fixtures carry identical code under mangled suffixes of DIFFERENT LENGTH, which is
    exactly what a bring-back rename produces (``...GemmHadamardSm90...`` vs ``...GemmSm90...``).
    A whole-image comparison reports these as different, and during a migration that false
    difference would be read as a real regression. Asserting the two images really do differ is
    what keeps this from passing vacuously.
    """
    a = _write(tmp_path, "a.o", _elf([(".text.kernel_ShortName", _PROGBITS, _TEXT_A)]))
    b = _write(tmp_path, "b.o", _elf([(".text.kernel_AMuchLongerMangledName", _PROGBITS, _TEXT_A)]))
    assert open(a, "rb").read() != open(b, "rb").read(), (
        "control: the two IMAGES must differ, or this test proves nothing about name-independence"
    )
    assert code_sections(a) == code_sections(b)
    assert differing_kinds(code_sections(a), code_sections(b)) == []


def test_every_code_section_kind_is_digested_and_others_are_not(tmp_path):
    """``.text`` and ``.nv.constant*`` are hashed; a named, non-code section is ignored.

    The exclusion is the half that matters: symbol tables and debug sections carry the mangled
    name, so digesting them would reintroduce the rename sensitivity this design removes.
    """
    d = code_sections(
        _write(
            tmp_path,
            "m.o",
            _elf(
                [
                    (".text.kernel_Foo", _PROGBITS, _TEXT_A),
                    (".nv.constant0.kernel_Foo", _PROGBITS, b"\xaa" * 32),
                    (".debug_info.kernel_Foo", _PROGBITS, b"\xbb" * 32),
                    (".symtab", 2, b"\xcc" * 32),  # SHT_SYMTAB: not PROGBITS, not code
                ]
            ),
        )
    )
    assert sorted(d) == [".nv.constant0", ".text"], d
    assert d[".text"][1] == hashlib.sha256(_TEXT_A).hexdigest()


def test_a_non_cuda_elf_is_skipped_rather_than_digested(tmp_path):
    """Selecting on ``e_machine`` is what makes the scan find the CUDA ELF and not the host one."""
    p = _write(tmp_path, "host.o", _elf([(".text.kernel_Foo", _PROGBITS, _TEXT_A)], machine=62))
    with pytest.raises(RuntimeError, match=r"no CUDA ELF"):
        code_sections(p)


def test_an_export_with_no_code_is_refused_rather_than_returning_empty(tmp_path):
    """An empty digest compares EQUAL to another empty digest, so it must not be returnable quietly.

    Two failure modes land here -- an ELF with no code section, and any caller mistake that yields
    ``{}`` -- and both would make a byte-identity assertion pass while measuring nothing. The
    parser returns the empty mapping; `assert_has_code` is the gate that refuses it, and this test
    pins that the gate is what fails rather than the comparison silently succeeding.
    """
    d = code_sections(_write(tmp_path, "e.o", _elf([(".symtab", 2, b"\xcc" * 32)])))
    assert d == {}, "the parser reports what it found"
    assert differing_kinds(d, {}) == [], "and two empty digests DO compare equal -- the hazard"
    with pytest.raises(AssertionError, match=r"no \.text section extracted"):
        assert_has_code(d, "an export that emitted nothing")


# ── the known-failing control, on real compiled code ──────────────────────────────────────────
@pytest.fixture
def disk_cache_off(monkeypatch):
    """Turn the JIT DISK cache off for one test, and prove it took.

    Why an attribute and NOT ``monkeypatch.setenv``
        `cache_utils` binds ``CACHE_ENABLED`` from ``CPO_CACHE_ENABLED`` at line 51, at IMPORT.
        By the time a test body runs, that read has happened, so setting the env var here changes
        nothing. `jit_cache`'s wrapper looks the module global up at CALL time, which is what makes
        the attribute patch effective and the env patch useless.

    Why the fixture exists at all
        The compiled test below used to inherit the setting from whatever shell ran it. It passed
        for its author, who exported ``CPO_CACHE_ENABLED=0``, and failed for everyone else with an
        ``AttributeError`` that named neither the cache nor the cause. Documenting a required
        environment is not the same as establishing it -- the same distinction as setting an env
        var for a subprocess versus verifying it arrived.

    Yields:
        None. The patch is reverted by `monkeypatch` at teardown.
    """
    from fold_cp_ops._internal import cache_utils

    monkeypatch.setattr(cache_utils, "CACHE_ENABLED", False)
    assert cache_utils.CACHE_ENABLED is False, "control: the patch must have taken"
    yield


@requires_sm90
def test_a_cached_compile_is_refused_by_name_rather_than_by_attribute_error(tmp_path):
    """With the disk cache ON, the pipeline returns something that cannot be exported.

    This is the sweep's failure mode in miniature and the strongest evidence for the
    cache-off/fresh-process protocol: on a HIT, `jit_cache` hands back
    `_restore_call_abi`'s reloaded callable rather than the compiled object, so the caller is
    silently given something other than what it asked for.

    The test asserts the DIAGNOSIS, not just the failure. A bare ``AttributeError: 'function'
    object has no attribute 'export_to_c'`` is what this used to produce, and it named neither the
    cache nor the fix; a reviewer had to A/B the environment to find the cause.

    Note the subject is `digest_export`'s refusal, so it is driven with a stand-in rather than by
    warming the real cache -- the disk-cache HIT is STATEFUL and a test that depended on an
    artifact already existing would pass or fail on the state of the machine.

    The matched phrase MOVED, deliberately. The message used to assert the disk cache for EVERY
    unexportable object, including a caller that had simply passed the compile WRAPPER -- which is
    what actually happened, in the byte-identity gate itself, and the wrong diagnosis was believed.
    A bare function still has no wrapper fields, so the cache really is the likely cause here and
    the message still names it; what it no longer does is name it when it cannot be true.
    """
    with pytest.raises(TypeError, match=r"disk-cache HIT"):
        digest_export(lambda *a, **k: None, tmp_path / "never_written.o")
    assert not (tmp_path / "never_written.o").exists(), (
        "the refusal must come BEFORE the export, or a half-written object is left behind"
    )


# ── the unwrap: `exportable` ──────────────────────────────────────────────────────────────────
class _FakeExportable:
    """Stand-in for a ``cute.compile`` result: carries the export surface and nothing else.

    Only the ATTRIBUTE's presence is under test -- `exportable` is pure introspection and never
    calls either method -- so the bodies raise, which is what proves it did not call them.
    """

    def export_to_c(self, *a, **k):
        """Never called by `exportable`; raises so an accidental call is loud."""
        raise AssertionError("exportable must not invoke the export")

    def dump_to_object(self, *a, **k):
        """Never called by `exportable`; raises so an accidental call is loud."""
        raise AssertionError("exportable must not invoke the export")


class _FakeWrapper:
    """Stand-in for `CompiledGemmBitcode` / `CompiledKernel`: holds the compiled object, is not it.

    Mirrors the real shape that caused the defect -- ``compiled`` set on a fresh compile, ``module``
    set instead on a cache hit, never both -- and, like the real one, carries a callable
    ``executor`` with NO export surface, which is the field a reader reaches for first and the one
    that cannot answer.
    """

    def __init__(self, compiled=None, module=None):
        """Build a wrapper in one of the two states the real one has.

        Args:
            compiled: what a FRESH compile puts on the wrapper; ``None`` on a cache hit.
            module: what a disk-cache HIT puts there instead; ``None`` on a fresh compile. Passing
                both is not rejected -- the point of the fixture is to model the real object, and
                the real one leaves that combination unreachable rather than guarded.
        """
        self.compiled = compiled
        self.module = module
        self.executor = lambda *a, **k: None


def test_a_compile_wrapper_is_unwrapped_instead_of_refused():
    """THE REGRESSION TEST: handing `digest_export` the wrapper must work, not raise.

    This is the defect the helper actually had, and it was invisible because the unwrap WAS
    performed -- by hand, at each call site. Three sites wrote ``digest_export(compiled.compiled,
    ...)``; one wrote ``digest_export(compiled, ...)``. The one that forgot was
    `test_the_five_byte_identity_configs_are_distinguishable`, i.e. the byte-identity gate itself,
    so the repo's answer to "can these configurations be told apart in the cubin" was an exception
    for as long as the miss went unnoticed.

    Measured on the real objects, at 2 ranks on sm_90, before the fix::

        CAND wrapper    CompiledGemmBitcode            export_to_c=False dump_to_object=False
        CAND .compiled  CudaDialectJitCompiledFunction export_to_c=True  dump_to_object=True
        CAND .executor  JitExecutor                    export_to_c=False dump_to_object=False

    So nothing was ever lost: the compiled object is retained on the wrapper for an unrelated
    reason (it owns the CUDA library nvshmem holds a raw handle to) and was reachable the whole
    time. That is why the fix is an unwrap and not a plumbing change.

    Asserts IDENTITY of the returned object, not merely that it did not raise: returning the
    wrapper itself would also "not raise" here and would fail three frames later inside the export.
    """
    inner = _FakeExportable()
    assert exportable(_FakeWrapper(compiled=inner)) is inner
    # ...and an object that is ALREADY exportable is returned untouched, which is what keeps the
    # three call sites that unwrap by hand working without a sweep.
    assert exportable(inner) is inner


def test_a_wrapper_with_no_compiled_object_blames_the_cache_and_a_bare_object_does_not():
    """The two failure modes must be told APART, because they have different fixes.

    A wrapper whose ``compiled`` is ``None`` and whose ``module`` is set really is a disk-cache
    HIT, and turning the cache off really is the fix. An object that is not a wrapper at all is a
    caller passing the wrong thing, and the fix is to pass the compiled object. The old message
    gave the first answer to both questions.

    That is not a style complaint. A confidently wrong error is ACTED ON: it sent a reader to prove
    the cache was off -- with ``CACHE_ENABLED`` patched False and one configuration per process --
    and the failure survived every one of those controls, because the cache was never the cause.
    An error that misdirects costs more than no error at all.
    """
    hit = _FakeWrapper(compiled=None, module=object())
    with pytest.raises(TypeError, match=r"compile WRAPPER but none of its fields"):
        exportable(hit)
    with pytest.raises(TypeError, match=r"CACHE_ENABLED"):
        exportable(hit)

    with pytest.raises(TypeError, match=r"is not a compile wrapper"):
        exportable(object())

    # And the wrapper diagnosis must NOT be produced for a bare object, or the two collapse again.
    try:
        exportable(object())
    except TypeError as exc:
        assert "compile WRAPPER but none of its fields" not in str(exc), (
            f"a non-wrapper was diagnosed as a wrapper: {exc}"
        )


def test_the_wrapper_executor_is_never_mistaken_for_the_compiled_object():
    """``executor`` is callable and looks like the answer; it is not, and must not be picked.

    The real ``executor`` is the post-``.to(device)`` ``JitExecutor`` -- the thing you CALL to run
    the kernel, which is exactly why a reader reaches for it -- and it carries no export surface at
    all. `exportable` must skip it rather than return it, or the failure moves from a named
    TypeError here to an ``AttributeError`` inside the export, which is the state this helper's
    error messages exist to prevent.
    """
    inner = _FakeExportable()
    w = _FakeWrapper(compiled=inner)
    assert callable(w.executor), "control: the decoy is callable, which is what makes it tempting"
    assert exportable(w) is inner


@requires_sm90
def test_two_different_configs_produce_different_digests(tmp_path, disk_cache_off):
    """THE CONTROL: two deliberately different tile shapes must digest differently.

    A harvester that reported identity on everything would satisfy every synthetic test above --
    they exercise the parser, not the pipeline -- and would then report a perfect byte-identity
    sweep against `main` while measuring nothing at all. The only thing that rules that out is a
    pair whose answer is known in advance, compiled through the real ``export_to_c`` path.

    Both halves are asserted. The SAME-config half covers EXPORT determinism: the third call
    re-runs ``export_to_c`` to a fresh path and re-hashes, so a timestamp, a path, or a temporary
    name baked into a hashed section would show up here, and every identity result would otherwise
    be a false negative.

    **It does NOT cover COMPILE determinism, and saying so matters.** The `disk_cache_off`
    fixture disables the DISK cache only; the in-process ``@jit_cache`` memo still serves the repeat, which
    is visible as the third compile taking 0.12 s against the first's 1.45 s. Compile determinism
    was measured separately instead -- the same configuration in two FRESH interpreters gave
    ``.text`` sha256 ``5bc432ca...`` both times, matching the in-process value, at 1.74 s and
    1.63 s. That is the same fresh-process-per-configuration discipline a real sweep must follow,
    for the same reason: a memo hit and a reproducible compile are indistinguishable from inside
    one process.

    The tile shape is the instrument's knob, not a swept axis -- see this module's
    `matrix_exempt`. `_compile_gemm` is reached directly because the front door tunes the tile,
    and a tuned pick would defeat the point of choosing two configurations by hand.
    """
    from cutlass import BFloat16

    from fold_cp_ops._internal.rounding import RoundingMode
    from fold_cp_ops.kernels.gemm import _compile_gemm

    cap, cluster = (9, 0), (1, 1, 1)
    head = (BFloat16, BFloat16, BFloat16, None, "k", "k", "n", None)
    tail = (None, None, 0, 0, 0, False, False, cap, RoundingMode.RN, 0, 0)

    def digest(tile, tag):
        compiled = _compile_gemm(*head, tile, cluster, False, True, False, *tail)
        d = digest_export(compiled, tmp_path / f"{tag}.o")
        assert_has_code(d, tag)
        return d

    small = digest((128, 128), "tile_128x128")
    large = digest((128, 256), "tile_128x256")
    again = digest((128, 128), "tile_128x128_again")

    assert small == again, (
        "the SAME configuration compiled twice must digest identically, or the digest is a "
        f"function of the export rather than of the code. Differing: {differing_kinds(small, again)}"
    )
    assert small != large, (
        "KNOWN-FAILING CONTROL FAILED: two different tile shapes produced the SAME digest, so this "
        "harvester cannot distinguish configurations and every identity it reports is vacuous"
    )
    assert ".text" in differing_kinds(small, large), (
        f"the tile change must move .text, not only constants: {differing_kinds(small, large)}"
    )

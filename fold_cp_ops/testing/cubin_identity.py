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

"""Digest the CODE of a compiled CuTe-DSL kernel, so two configurations can be compared by bytes.

Why this exists as a shared helper
    Byte identity against `main` is one of the three bring-back gates, and until now this repo had
    no way to measure it. Five tests carry ``byte_identical`` in their names and every one of them
    compares OUTPUT TENSORS; a kernel can compute the right numbers from different instructions,
    which is exactly how a missing race guard survived an output comparison on this branch. The
    extraction below is PORTED from `tests/kernels/test_gemm_hadamard.py`, where it already worked
    for one kernel pair, rather than written afresh.

Why section CONTENT and never the whole image
    ``export_to_c`` writes a HOST object with the CUDA ELF embedded in ``.lrodata``. Two artifacts
    are never byte-equal as whole files even when their code is identical, because the mangled
    kernel symbol carries the producing Python module and class and the differing NAME LENGTH
    shifts every string after it. A whole-image compare therefore reports "different" for a pure
    rename -- the single most likely thing to happen during a bring-back, and the one case where a
    false difference would be read as a real regression. What must match is the code, and the code
    sections carry no names.

What this CANNOT tell you
    Nothing about numerics, and nothing about a knob the compile key cannot see. Two configurations
    that differ only by such a knob can be served ONE cached artifact, in which case both sides
    digest the same file and identity is trivially true. Callers must compile with the disk cache
    OFF **and in a fresh process per configuration** (the in-process ``@jit_cache`` memo is a
    second, independent collision), and must treat an unexpected identity as a suspected cache
    collision before treating it as a result.

    That protocol is backed by a LIVE OBSERVATION, not only by the compile-key argument. With the
    disk cache on, a cache HIT makes ``jit_cache`` return `_restore_call_abi`'s reloaded callable
    instead of the compiled object -- so the pipeline silently hands back something that is not the
    thing that was asked for, which is the sweep's failure mode in miniature. `digest_export`
    refuses it by name.

    **The disk-cache failure is STATEFUL, which is what makes it dangerous.** On a COLD cache the
    same call MISSES, compiles, and returns a real compiled object, so a cache-on run PASSES until
    an artifact exists and fails afterwards. "It worked on my machine" and "it failed on yours" can
    both be true of one revision in one environment, differing only in what is on disk.

Layering
    Stdlib only -- no torch, no cutlass, no pytest -- so it imports anywhere, including in a probe
    pointed at another tree. It lives under the package root like the rest of
    `fold_cp_ops.testing`, which means editing it busts the JIT disk-cache fingerprint; that is
    correct (the fingerprint hashes every ``.py`` under the package) and worth knowing when a
    one-line edit here triggers recompiles.
"""

from __future__ import annotations

import hashlib
import inspect
import struct
from pathlib import Path

#: ``e_machine`` value identifying a CUDA ELF (EM_CUDA). The host object embeds one; the scan below
#: skips every other ELF in the file, which is why it looks for this rather than for the first
#: ``\x7fELF`` it finds.
EM_CUDA = 190

#: ELF section type ``SHT_PROGBITS`` -- a section that occupies file space and carries real bytes.
#: A section of any other type has nothing to hash.
SHT_PROGBITS = 1

#: Section-name prefixes whose bytes ARE the emitted code. Everything else in the image is symbol
#: tables, relocations and debug data, all of which carry names and therefore move under a rename.
CODE_SECTION_PREFIXES = (".text", ".nv.constant")


def code_sections(object_file: str) -> dict[str, tuple[int, str]]:
    """Map code-section KIND -> ``(size, sha256)`` for one exported artifact.

    Purpose
        Reduce a compiled kernel to a name-independent fingerprint of the instructions it emits,
        so two configurations (or two trees) can be compared for byte identity.

    Semantics
        Scans `object_file` for the single embedded ELF whose ``e_machine`` is `EM_CUDA`, walks its
        section headers, and hashes the bytes of every `SHT_PROGBITS` section whose name starts
        with one of `CODE_SECTION_PREFIXES`. The mangled kernel suffix is stripped from each
        section name (everything from ``.kernel_`` onward) so that two sides line up by KIND rather
        than by a name that encodes the producing class. Section bytes are hashed; section NAMES
        are not.

    Args:
        object_file: Path to a ``.o`` written by ``export_to_c``. Requirements, and what goes wrong
            if they are not met:

            * It must contain EXACTLY ONE CUDA ELF. A fat artifact carrying several would be
              digested from only the first, silently, and a comparison would then be about an arch
              nobody chose.
            * It must be a compiled kernel, not a stub. An export that produced no CUDA ELF raises
              rather than returning ``{}`` -- an empty mapping compares equal to another empty
              mapping, so a vacuous comparison would pass.
            * It must be readable and complete. A file still being written yields a truncated
              parse, most often as a ``struct.error`` or an ``IndexError`` from the section walk.

    Returns:
        ``{kind: (size, sha256_hex)}``, e.g. ``{".text": (12345, "ab12...")}``. Empty only if the
        ELF genuinely carries no code section, which is itself a finding -- see
        `assert_has_code`.

    Raises:
        RuntimeError: If no CUDA ELF is embedded in `object_file`. This is deliberately an
            exception and not an empty result, because the failure it guards against is a
            comparison that trivially passes.
    """
    blob = open(object_file, "rb").read()
    i = blob.find(b"\x7fELF", 1)
    while i != -1 and struct.unpack_from("<H", blob, i + 18)[0] != EM_CUDA:
        i = blob.find(b"\x7fELF", i + 1)
    if i == -1:
        raise RuntimeError(
            f"no CUDA ELF (e_machine={EM_CUDA}) embedded in {object_file}: the export produced "
            f"something other than a compiled kernel, so any comparison against it would be vacuous"
        )
    (shoff,) = struct.unpack_from("<Q", blob, i + 40)
    # ELF64 header: e_shentsize at +58, e_shnum at +60, e_shstrndx at +62.
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", blob, i + 58)
    hdrs = [struct.unpack_from("<IIQQQQ", blob, i + shoff + n * shentsize) for n in range(shnum)]
    stroff = i + hdrs[shstrndx][4]
    out: dict[str, tuple[int, str]] = {}
    for name, typ, _, _, off, size in hdrs:
        end = blob.index(b"\0", stroff + name)
        nm = blob[stroff + name : end].decode()
        if typ == SHT_PROGBITS and size and nm.startswith(CODE_SECTION_PREFIXES):
            body = blob[i + off : i + off + size]
            out[nm.split(".kernel_")[0]] = (size, hashlib.sha256(body).hexdigest())
    return out


#: Fields on a compile WRAPPER that may carry the exportable ``cute.compile`` result, in the order
#: they are tried. `compile_nvshmem` / `compile_gemm_with_bitcode` do not return the compiled object
#: -- they return a `CompiledGemmBitcode` / `CompiledKernel` that holds it, and the two fields are
#: never both set: a FRESH COMPILE has ``compiled`` and no ``module``, a CACHE HIT has ``module``
#: and no ``compiled``. Both are retained for an unrelated reason (they own the CUDA library that
#: nvshmem's ``library_init`` registered a raw handle to), which is precisely why they are still
#: reachable here and nothing has to be plumbed to make this work.
_WRAPPER_EXPORT_FIELDS = ("compiled", "module")


def exportable(obj):
    """The object that can actually be exported, unwrapping a compile wrapper if it is one.

    Purpose
        Make the unwrap a PROPERTY OF THE HELPER instead of a convention each call site repeats.
        It was the convention, and the convention failed exactly the way conventions do: three call
        sites wrote ``digest_export(compiled.compiled, ...)`` and one wrote
        ``digest_export(compiled, ...)``, so `test_the_five_byte_identity_configs_are_distinguishable`
        -- the byte-identity gate itself -- raised instead of measuring, and the repo's answer to
        "are these two configurations distinguishable in the cubin" became an exception nobody read
        as a harness bug.

    Functionality & semantics
        Returns ``obj`` unchanged when it already carries an export surface, so the three call sites
        that unwrap by hand keep working and nothing has to be swept. Otherwise tries each field in
        :data:`_WRAPPER_EXPORT_FIELDS` and returns the first whose value carries one. Purely
        introspective -- nothing is compiled, exported, or written.

        "Carries an export surface" means ``export_to_c`` OR ``dump_to_object``. Both are checked
        because the tvm-ffi-OFF path this repo's A2A kernels take needs ``dump_to_object``, and an
        object could gain one without the other.

    Input requirements
        ``obj`` may be a ``cute.compile`` result, a ``CompiledGemmBitcode`` / ``CompiledKernel``
        wrapper, or anything else -- a wrong type is REPORTED, not assumed. No constraint on the
        compile that produced it; in particular a cache HIT is a valid input and is named as such
        rather than treated as a malformed one.

    Returns
        The object to hand to ``export_to_c`` / ``dump_to_object``.

    Raises
        TypeError: naming the type inspected, every wrapper field and what it held, and the
            precondition that would fix it. The message is deliberately concrete because the
            previous one was concrete AND WRONG -- it asserted the JIT disk cache in every case,
            including the overwhelmingly common one where the caller had simply passed the wrapper,
            and a confidently wrong error is worse than a vague one because it is acted on. Measured:
            it fired with the disk cache provably disabled and one configuration per process.
    """
    if hasattr(obj, "export_to_c") or hasattr(obj, "dump_to_object"):
        return obj
    seen = []
    for field in _WRAPPER_EXPORT_FIELDS:
        if not hasattr(obj, field):
            continue
        inner = getattr(obj, field)
        seen.append(f"{field}={type(inner).__name__}")
        if inner is not None and (
            hasattr(inner, "export_to_c") or hasattr(inner, "dump_to_object")
        ):
            return inner
    kind = type(obj).__name__
    if seen:
        # It IS a wrapper; it just has nothing exportable in it. The only way that happens with the
        # fields present is a cache HIT whose `module` cannot export -- which is the disk-cache
        # story, now scoped to the case where it is actually true.
        raise TypeError(
            f"{kind} is a compile WRAPPER but none of its fields carry an export surface "
            f"({', '.join(seen)}). A FRESH compile populates `compiled`; a disk-cache HIT populates "
            f"`module` instead, and a reloaded artifact cannot always be re-exported. If this is a "
            f"hit, compile with `cache_utils.CACHE_ENABLED = False` (the env var CPO_CACHE_ENABLED "
            f"is read once at import, so exporting it after import is too late) in a FRESH PROCESS "
            f"per configuration -- the in-process memo is a second, independent way to be handed "
            f"something other than a fresh compile."
        )
    raise TypeError(
        f"{kind} has neither `export_to_c` nor `dump_to_object`, and is not a compile wrapper "
        f"(no {' / '.join(_WRAPPER_EXPORT_FIELDS)} field), so there is nothing to digest. Pass the "
        f"`cute.compile` result or the wrapper that holds it. If this is a plain function, it is "
        f"most likely `jit_cache`'s reloaded callable from a disk-cache HIT: set "
        f"`cache_utils.CACHE_ENABLED = False` and compile in a fresh process."
    )


def digest_export(compiled, object_file: str, *, function_name: str = "kernel_entry") -> dict:
    """Export one compiled kernel and digest it in a single step.

    Purpose
        The export and the digest are always done together and getting the pair wrong is silent:
        digesting a path that was never written raises a confusing ``FileNotFoundError``, and
        exporting two configurations to the SAME path makes the second overwrite the first so the
        comparison reads identical.

    Args:
        compiled: The object returned by ``cute.compile``, OR a `compile_nvshmem` /
            `compile_gemm_with_bitcode` wrapper that holds it -- `exportable` unwraps the wrapper,
            so a caller never has to know which it has. Passing ``wrapper.compiled`` by hand still
            works and is what the existing call sites do.
        object_file: Destination path, which must be DISTINCT per configuration -- reusing one path
            across a sweep is the failure this wrapper exists to make visible, since the resulting
            "identical" verdict looks exactly like a real one.
        function_name: Entry symbol handed to ``export_to_c``. It appears in the mangled section
            names, which `code_sections` strips, so it does not affect the digest; it is exposed
            only because ``export_to_c`` requires it.

    Returns:
        `code_sections` of the freshly written object.

    Two export ABIs, and the one that matters is the one this repo's A2A route uses:
        ``cute.compile`` returns a DIFFERENT class depending on whether tvm-ffi is enabled, and the
        two spell ``export_to_c`` incompatibly -- tvm-ffi ON takes ``object_file_path=`` plus a
        ``function_name``, tvm-ffi OFF takes a DIRECTORY ``file_path=`` plus a stem ``file_name=``
        and chooses the extension itself. The single-device ``@jit_cache`` kernels are tvm-ffi ON;
        ``compile_gemm_with_bitcode`` deliberately runs tvm-ffi OFF, so the A2A kernels -- the whole
        subject of the byte-identity gate -- land on the second form. Supporting only the first
        raised ``TypeError: ... got an unexpected keyword argument 'object_file_path'`` on every
        A2A config while the real-compile control kept passing on a single-device one, which is a
        harvester that looks proven and cannot digest the artifacts it exists for. Both are
        dispatched here, on the SIGNATURE rather than on the class name, so a rename upstream does
        not silently reopen the hole.

    Raises:
        TypeError: From `exportable`, if nothing reachable from `compiled` can be exported. The
            message names the type, every wrapper field and what it held, and the precondition that
            would fix it. It no longer asserts the disk cache unconditionally: that diagnosis was
            wrong for the common case (a caller passing the wrapper) and was believed anyway, which
            is the specific harm a confident wrong error does.
        RuntimeError: From `code_sections`, if the export carried no CUDA ELF.
    """
    # UNWRAP FIRST. `compile_nvshmem` / `compile_gemm_with_bitcode` return a wrapper, not the
    # compiled object; `exportable` reaches through it and raises a message naming the actual
    # precondition when it cannot. Doing it here rather than at each call site is the fix for the
    # defect this helper had: the unwrap WAS the convention, and one of the four call sites -- the
    # byte-identity gate itself -- did not follow it.
    compiled = exportable(compiled)
    dest = Path(object_file)
    if not hasattr(compiled, "export_to_c"):
        # `dump_to_object` without `export_to_c`. Nothing in this repo produces that today, but the
        # signature sniff below would raise AttributeError rather than say why, and the whole point
        # of this helper's errors is that they name the thing to fix.
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(compiled.dump_to_object(function_name))
        return code_sections(str(dest))
    params = inspect.signature(compiled.export_to_c).parameters
    if "object_file_path" in params:
        # tvm-ffi ON. Takes a FULL path and names the entry symbol.
        compiled.export_to_c(object_file_path=str(dest), function_name=function_name)
    elif "file_path" in params:
        # tvm-ffi OFF -- and NOT through export_to_c, which also generates a C HEADER and therefore
        # has to describe every argument in C. It cannot: an epilogue's NamedTuple of optional
        # tensors dies as `Unsupported argument for c function argument generation: ... type
        # <class 'tuple'>`, which is the same wall `cache_utils` records for the EnvStream. Since
        # the header is of no interest here -- the subject is the CUBIN -- take `dump_to_object`,
        # which returns the same ELF as bytes and skips header generation entirely. The prefix is
        # pinned to `function_name` so a per-configuration filename cannot leak into the symbol
        # names and therefore cannot move the digest.
        if not hasattr(compiled, "dump_to_object"):
            raise TypeError(
                f"{type(compiled).__name__} takes the tvm-ffi-OFF export_to_c(file_path=...) form "
                f"but has no dump_to_object, and export_to_c alone cannot serve this: it also "
                f"generates a C header and fails on any argument it cannot describe in C."
            )
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(compiled.dump_to_object(function_name))
    else:
        raise TypeError(
            f"{type(compiled).__name__}.export_to_c has an unrecognized signature "
            f"{tuple(params)}; this helper knows the tvm-ffi-ON form (object_file_path=) and the "
            f"tvm-ffi-OFF form (file_path=, file_name=). A new one must be added here rather than "
            f"worked around at the call site, or half the sweep silently stops being exported."
        )
    return code_sections(str(dest))


def differing_kinds(left: dict, right: dict) -> list[str]:
    """Which code-section kinds differ between two digests, sorted.

    Purpose
        A bare ``left == right`` says only that something differs. Naming the kinds is what turns a
        failed byte-identity assertion into a starting point -- ``.text`` differing means the
        instructions changed, while only ``.nv.constant0`` differing points at baked constants.

    Args:
        left, right: Mappings from `code_sections`. A kind present in one and absent from the other
            counts as differing; that asymmetry is a real difference, not a lookup miss, because
            both sides are keyed by stripped KIND rather than by mangled name.

    Returns:
        Sorted list of differing kinds; empty when the two digests are equal.
    """
    return sorted(k for k in set(left) | set(right) if left.get(k) != right.get(k))


def assert_has_code(digest: dict, what: str) -> None:
    """Refuse a digest with no ``.text``, because an empty comparison passes.

    Purpose
        Every failure mode of this harvester -- a wrong path, an export that emitted nothing, a
        name-keying bug that matched no section -- lands as an empty or ``.text``-less mapping, and
        two of those compare EQUAL. This is the non-vacuity gate that every caller must run before
        reading an identity verdict as a result.

    Args:
        digest: A mapping from `code_sections`.
        what: Human name of the side being checked, used in the message so a failure says which of
            the two configurations produced nothing.

    Raises:
        AssertionError: If `digest` is empty or carries no ``.text`` section.
    """
    assert digest and ".text" in digest, (
        f"{what}: no .text section extracted (got {sorted(digest)}) -- an empty digest compares "
        f"EQUAL to another empty digest, so this comparison would pass without measuring anything"
    )

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

# Adapted from https://github.com/triton-lang/triton/blob/main/python/triton/runtime/autotuner.py
# Copyright (C) 2025, Tri Dao.
# Copyright (c) 2025, Tri Dao.
# This file has been modified by NVIDIA CORPORATION & AFFILIATES.

"""Compiling the candidates in parallel subprocesses -- parent and worker in ONE file.

Compiling N candidates serially in the tuning process is the dominant cost of a sweep, because a
cold CuTe compile is seconds and the sweep itself is milliseconds. It cannot be threaded:
``cute.compile`` keeps thread-local MLIR state, and forking after CUDA init segfaults. So the work
goes to spawned subprocesses, each with its own CUDA context, compiling against FAKE tensors built
from the parent's tensor metadata with ``COMPILE_ONLY`` set. The parent then loads every artifact
from the on-disk ``.o`` cache in microseconds.

**Why parent and worker share a file.** Upstream split them, and CLAUDE.md records the result as one
of "two import edges no AST tool can see": the parent spawned ``python -m
fold_cp_ops._internal._compile_worker`` as a STRING, so a rename or a move broke the pair with no
import error. The failure was silent in the worst way -- the pre-compile just fails per config and
falls back to serial, or, if ``COMPILE_ONLY`` never gets set, the worker EXECUTES the kernel. Here
the worker is this same module's ``__main__`` block, so the module string is ``__name__`` and cannot
drift from the file it names. There is nothing left to keep in sync.

**It is an optimization, and it fails soft.** Every failure path -- no workers, a dead pipe, an
argument the wire protocol cannot carry -- returns quietly and lets the tuner compile serially. A
pre-compile that raised would turn a slow sweep into a broken one.
"""

import os
import struct
import subprocess
import sys
import time
from typing import Any, Dict, Optional, Sequence, Tuple

import msgpack

#: How many worker processes to spawn. Each holds a CUDA context, so this trades host memory and
#: device contexts for wall-clock. Read from ``CPO_AUTOTUNE_WORKERS``.
DEFAULT_WORKERS = 8

#: Below this many seconds for the first in-process compile, the artifacts are evidently already
#: cached and spawning workers would cost more than it saves.
_ALREADY_CACHED_S = 0.5


#: Exact built-in types the worker protocol admits. ``type(v) is T``, never ``isinstance``: a
#: SUBCLASS (a ``numpy.bool_``, an ``IntEnum``, a ``str`` subclass) would encode as its base and be
#: rebuilt as the base in the worker, so the config compiled there would differ from the one the
#: caller asked for -- silently, and only in a subprocess. Rejecting sends it down the serial path
#: instead, where its real type survives.
_WIRE_SCALARS = (bool, int, float, str, bytes, type(None))

#: Depth cap for the admissibility walk, so a pathological nesting cannot make preflight unbounded.
_WIRE_MAX_DEPTH = 8


def wire_admissible(value: Any, _depth: int = 0) -> bool:
    """Whether *value* survives the worker protocol UNCHANGED.

    Purpose
        Decide, before any worker is spawned, whether the pre-compile can run at all. The protocol
        moved from ``pickle`` to MessagePack, which represents strictly less -- so the question
        "can this be sent?" became one the caller must ask explicitly rather than discover from an
        exception.

    Functionality & semantics
        Admits exactly ``None``, ``bool``, ``int``, ``float``, ``str``, ``bytes``, ``tuple``, and
        ``str``-keyed ``dict``, recursively. Two rejections are deliberate and are the point:

        * a **list** is refused rather than converted. MessagePack has one array type, so a list
          would come back a tuple (``use_list=False``) or a tuple would come back a list -- either
          way the worker compiles against a type the caller did not pass. Only fold-cp's OWN
          containers are converted to tuples, at the point they are built, where the type is known
          to be an implementation detail.
        * a **subclass** is refused for the same reason, one level down.

        Rejection is never an error: :func:`precompile_configs` skips the pre-compile and the
        candidates compile serially, which is the behaviour that existed before workers.

    Args:
        value: Any described argument, keyword value, or config entry.
        _depth: Recursion depth, internal.

    Returns:
        True if the value can cross the pipe with its type intact.
    """
    if _depth > _WIRE_MAX_DEPTH:
        return False
    t = type(value)
    if t in _WIRE_SCALARS:
        return True
    if t is tuple:
        return all(wire_admissible(v, _depth + 1) for v in value)
    if t is dict:
        return all(
            type(k) is str and wire_admissible(v, _depth + 1) for k, v in value.items()
        )
    return False


def _send(stream, msg: Any) -> None:
    """Write one length-prefixed MessagePack message. Length-prefixed because a pipe is a byte stream.

    The four-byte ``<I`` framing is UNCHANGED from the pickle version, deliberately: it is the part
    the worker loop and every framing test depend on, and only the codec inside it moved.

    Args:
        stream: A binary writable stream (the child's stdin, or the worker's stdout).
        msg: A :func:`wire_admissible` value. The caller checks admissibility in preflight; a value
            that slips through raises here rather than being silently re-typed.
    """
    data = msgpack.packb(msg, use_bin_type=True)
    stream.write(struct.pack("<I", len(data)))
    stream.write(data)
    stream.flush()


def _recv(stream) -> Optional[Any]:
    """Read one length-prefixed MessagePack message, or None at EOF / short read / bad payload.

    ``None`` means "no usable message" and covers every way that can happen: EOF, a truncated
    header, a zero length, a short body, and -- new with MessagePack -- a body that does not decode.
    A decode failure is a PROTOCOL failure, indistinguishable to the caller from a dead worker, and
    both are handled the same way: give up on the worker and compile serially. Raising instead would
    turn a corrupt pipe into a failed autotune rather than a slow one.

    Args:
        stream: A binary readable stream.

    Returns:
        The decoded message, or ``None``.
    """
    header = stream.read(4)
    if len(header) < 4:
        return None
    length = struct.unpack("<I", header)[0]
    if not length:
        return None
    body = stream.read(length)
    if len(body) != length:
        return None
    try:
        return msgpack.unpackb(body, raw=False, use_list=False, strict_map_key=True)
    except Exception:
        return None


def _describe_value(value: Any) -> Any:
    """Reduce ONE argument to what a worker can rebuild: metadata for a tensor, the value itself
    otherwise.

    Purpose
        The worker only ever COMPILES, so it needs a tensor's shape, stride and dtype and nothing
        else. Sending the tensor itself is not a heavier version of the same thing -- it is a
        different operation, with three costs the compile does not need: a device-to-host copy, a
        pickle of the whole payload, and a full device allocation in EVERY worker when it is
        unpickled back onto CUDA.

    Args:
        value: Any argument. A `torch.Tensor` is described; anything else is returned unchanged and
            must satisfy :func:`wire_admissible` (one that does not makes the caller's pre-flight
            check fail, which skips the pre-compile rather than breaking it).

    Returns:
        ``{"__tensor__": {shape, stride, dtype}}`` for a tensor, else `value`.
    """
    import torch

    if isinstance(value, torch.Tensor):
        # TUPLES, not lists. This metadata is fold-cp's own -- the caller never sees it -- so its
        # container type is an implementation detail and converting it changes nothing observable.
        # A CALLER's list is refused instead (`wire_admissible`), because there the type is the
        # caller's and MessagePack cannot round-trip list-vs-tuple. `torch.empty_strided` in the
        # worker takes either.
        return {
            "__tensor__": {
                "shape": tuple(value.shape),
                "stride": tuple(value.stride()),
                "dtype": str(value.dtype).replace("torch.", ""),
            }
        }
    return value


def _describe_args(args: Sequence[Any]) -> list:
    """Reduce the call's positional arguments to what a worker can rebuild.

    Args:
        args: The kernel's positional arguments.

    Returns:
        A tuple the wire protocol admits; see :func:`_describe_value`. A tuple rather than a list
        for the reason given there -- this container is ours, and MessagePack has only one array
        type.
    """
    return tuple(_describe_value(a) for a in args)


def _describe_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Reduce the call's KEYWORD arguments the same way as the positional ones.

    Purpose
        This existed only for positional arguments, and the asymmetry was a live crash. A kernel
        entry that takes an operand by keyword -- `layernorm_gemm(..., gate3=...)` is one, and it is
        the workflow's own call -- had that whole tensor pickled and shipped to every worker for
        every config.

        At the declared workflow shape ``M = 4194304`` (N_token 2048), ``D = 512``, `gate3` is
        ``4194304 x 512`` bf16 = **exactly 2**32 bytes**, one byte more than the ``<I`` length prefix
        in :func:`_send` can express, so `select="autotune"` died with
        ``struct.error: 'I' format requires 0 <= number <= 4294967295``. Below that boundary it did
        not crash, it just moved gigabytes: at D=384 that is 3.2 GB copied to host, pickled, and
        re-allocated on the device in each of eight workers.

        It survived because it is COLD-CACHE ONLY: with artifacts already cached the first compile
        takes under `_ALREADY_CACHED_S` and no worker is ever spawned, so a warm developer box never
        sees it and a first-time user always does.

    Args:
        kwargs: The kernel's keyword arguments, by name.

    Returns:
        A new dict with the same keys; see :func:`_describe_value`.
    """
    return {k: _describe_value(v) for k, v in kwargs.items()}


def precompile_configs(
    fn,
    args: Tuple,
    kwargs: Dict[str, Any],
    configs: Sequence[Any],
    *,
    verbose: bool = False,
) -> None:
    """Compile every candidate in worker subprocesses so the timed region loads from cache.

    Purpose
        Keeps compilation OUT of the measured region. Without it the first invocation of each
        candidate includes a multi-second compile, and while the tuner calls the kernel once before
        timing it, doing that serially for N candidates dominates the sweep.

    Semantics
        Best-effort and side-effect-only: it populates the on-disk artifact cache and returns
        nothing. Every failure -- no artifact cache, one worker, unpicklable arguments, a dead pipe
        -- returns quietly, leaving the tuner to compile serially. It also short-circuits when the
        first in-process compile completes in under half a second, which means the artifacts are
        already cached and workers would be pure overhead.

        NOT collective-aware, deliberately: each rank pre-compiles its own artifacts, and the ranks
        need no agreement here because nothing is measured. The tuner's barrier after the thermal
        warmup is what re-synchronizes them before timing starts.

    Args:
        fn: The kernel entry. Its ``__module__`` and ``__qualname__`` must resolve in a FRESH
            interpreter -- a locally-defined or dynamically-created function cannot be pre-compiled,
            and is skipped rather than raising.
        args: The kernel's positional arguments; tensors are sent as metadata only.
        kwargs: The kernel's keyword arguments. Must be picklable.
        configs: The candidates to compile.
        verbose: Whether to report progress and worker failures.

    Returns:
        None.
    """
    try:
        from fold_cp_ops._internal.cache_utils import CACHE_ENABLED
    except ImportError:
        return
    if not CACHE_ENABLED:
        # Without an artifact cache the workers' compiles are thrown away when they exit.
        return

    max_workers = min(len(configs), int(os.getenv("CPO_AUTOTUNE_WORKERS", str(DEFAULT_WORKERS))))
    if max_workers <= 1:
        return

    t0 = time.time()
    try:
        fn(*args, **dict(kwargs, **configs[0].all_kwargs()))
    except Exception:
        pass
    if time.time() - t0 < _ALREADY_CACHED_S:
        return

    try:
        payload_args = _describe_args(args)
        # KEYWORD arguments go through the same reduction as the positional ones. Describing only
        # `args` shipped whole keyword tensors to every worker -- see `_describe_kwargs`.
        payload_kwargs = _describe_kwargs(kwargs)
        payload_configs = tuple(c.all_kwargs() for c in configs)
        payload = (payload_args, payload_kwargs, payload_configs)
        # TWO checks, and neither subsumes the other. `wire_admissible` is about TYPE -- it refuses
        # a caller's list or a subclass, which MessagePack would happily encode as something else
        # and hand the worker a value the caller never passed. The trial pack is about RANGE and
        # size -- an int past 64 bits, or a str that will not encode -- which no type check sees.
        if not wire_admissible(payload):
            raise TypeError("argument or config type is not carried by the worker protocol")
        msgpack.packb(payload, use_bin_type=True)
    except Exception:
        if verbose:
            print("[autotune] pre-compile skipped: arguments do not survive the worker protocol")
        return

    module, qualname = getattr(fn, "__module__", None), getattr(fn, "__qualname__", None)
    if not module or not qualname or "<locals>" in qualname:
        if verbose:
            print(
                f"[autotune] pre-compile skipped: {qualname} is not importable in a fresh process"
            )
        return

    workers = []
    for _ in range(max_workers):
        try:
            p = subprocess.Popen(
                [sys.executable, "-m", __name__],  # <- the module string IS this file
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=None if verbose else subprocess.DEVNULL,
            )
        except OSError:
            break
        if _recv(p.stdout) != "READY":
            p.kill()
            continue
        workers.append(p)
    if not workers:
        return

    if verbose:
        print(f"[autotune] pre-compiling {len(configs)} configs across {len(workers)} workers")
    t_start = time.time()
    pending = [0] * len(workers)
    try:
        for i, config in enumerate(configs):
            w = workers[i % len(workers)]
            _send(
                w.stdin,
                {
                    "module": module,
                    "qualname": qualname,
                    "args": payload_args,
                    "kwargs": payload_kwargs,
                    "config": config.all_kwargs(),
                },
            )
            pending[i % len(workers)] += 1
        for wi, w in enumerate(workers):
            for _ in range(pending[wi]):
                _recv(w.stdout)
    # `struct.error` joins these two because this module's contract is that a pre-compile
    # NEVER breaks a sweep -- it falls back to serial. An oversized frame length is exactly
    # that kind of failure, and it was reaching the caller as a hard crash.
    except (BrokenPipeError, OSError, struct.error):
        pass
    finally:
        for w in workers:
            try:
                w.stdin.close()
                w.wait(timeout=30)
            except Exception:
                w.kill()
    if verbose:
        print(f"[autotune] pre-compile finished in {time.time() - t_start:.1f}s")


def _worker_main() -> None:
    """The subprocess side: rebuild fake tensors, compile one config, acknowledge, repeat.

    Runs in a FRESH interpreter with its own CUDA context. ``COMPILE_ONLY`` is set BEFORE the kernel
    is imported and is the load-bearing line: without it the worker would EXECUTE the kernel against
    fake tensors rather than only compiling it.

    Protocol: emits ``"READY"``, then for each length-prefixed request emits ``"OK"`` or
    ``("ERR", message)``. Exits when stdin closes.
    """
    import importlib

    import torch

    from fold_cp_ops._internal import cache_utils

    cache_utils.COMPILE_ONLY = True

    _send(sys.stdout.buffer, "READY")
    while True:
        req = _recv(sys.stdin.buffer)
        if req is None:
            return
        try:
            module = importlib.import_module(req["module"])
            target = module
            for part in req["qualname"].split("."):
                target = getattr(target, part)

            def rebuild(v):
                """Turn one described value back into an (uninitialized) device tensor, or pass it
                through. The inverse of `_describe_value`, and it must stay symmetric with it:
                describing a value the worker cannot rebuild would hand the kernel a dict."""
                if isinstance(v, dict) and "__tensor__" in v:
                    meta = v["__tensor__"]
                    return torch.empty_strided(
                        meta["shape"],
                        meta["stride"],
                        dtype=getattr(torch, meta["dtype"]),
                        device="cuda",
                    )
                return v

            rebuilt = [rebuild(a) for a in req["args"]]
            # Keyword arguments are rebuilt too, because they are now described too. The config's
            # own kwargs are plain scalars and pass through `rebuild` untouched.
            kw = {k: rebuild(v) for k, v in req["kwargs"].items()}
            target(*rebuilt, **dict(kw, **req["config"]))
            _send(sys.stdout.buffer, "OK")
        except Exception as exc:  # a failed candidate is the tuner's problem, not the worker's
            _send(sys.stdout.buffer, ("ERR", f"{type(exc).__name__}: {exc}"))


if __name__ == "__main__":
    _worker_main()

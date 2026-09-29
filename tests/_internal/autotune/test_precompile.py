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

"""Tests for ``fold_cp_ops._internal.autotune.precompile`` -- compiling candidates in subprocesses.

The module is **best-effort and side-effect-only**: it warms the on-disk artifact cache and returns
nothing. That shape is the thing to test. Every failure it can hit -- no artifact cache, a
locally-defined kernel whose qualname does not resolve in a fresh interpreter, an unpicklable
argument, a dead pipe -- must return quietly and leave the tuner to compile serially. A
pre-compilation step that can BREAK a tuning run is worse than no pre-compilation step, because it
converts a slow path into a failed one.

The wire protocol gets its own tests for a duller reason: a pipe is a byte stream, so a message
without a length prefix is not a message. A short read that silently returned a truncated object
would hand a worker a config it was never asked to compile.
"""

import io
import msgpack
import pytest

from fold_cp_ops._internal.autotune import AutotuneConfig
from fold_cp_ops._internal.autotune.precompile import (
    _describe_args,
    _describe_kwargs,
    _recv,
    _send,
    precompile_configs,
    wire_admissible,
)


def _wire_len(obj) -> int:
    """Bytes *obj* occupies on the worker pipe.

    The size assertions below are about what actually crosses the pipe, so they must measure the
    codec the pipe uses. They previously measured ``pickle``, which is no longer that codec -- and a
    size test against the wrong serializer can pass while the real payload is orders of magnitude
    larger.
    """
    return len(msgpack.packb(obj, use_bin_type=True))


def _pipe(msg):
    """Round-trip one message through the length-prefixed protocol via an in-memory buffer.

    Args:
        msg: Any picklable object.

    Returns:
        Whatever `_recv` reads back.
    """
    buf = io.BytesIO()
    _send(buf, msg)
    buf.seek(0)
    return _recv(buf)


@pytest.mark.parametrize(
    "msg",
    [
        {"config": {"tile": 128}},
        ("a", 1, None),
        {"nested": {"deep": (1, 2, 3)}},
        {"b": True, "i": 7, "f": 1.5, "s": "x", "y": b"\x00\xff", "n": None},
    ],
)
def test_a_message_round_trips_through_the_pipe(msg):
    """Send then receive reproduces the object, TYPES INCLUDED. The base case for everything below.

    The messages are tuples rather than lists because the codec is MessagePack, which has ONE array
    type: a list would come back a tuple and the assertion would be about the codec's preference
    rather than about the round trip. `wire_admissible` refuses a caller's list for exactly that
    reason, so a list here would also be a message the protocol declines to send.
    """
    got = _pipe(msg)
    assert got == msg
    assert type(got) is type(msg), f"the container type changed: {type(msg)} -> {type(got)}"


def test_a_truncated_body_reads_as_nothing_rather_than_as_a_partial_object():
    """A short read returns None, never a half-decoded message.

    Returning something partial would hand a worker a config it was never asked to compile -- and
    since the worker only warms a cache, the result would be an artifact for the wrong config
    sitting where the right one should be.
    """
    buf = io.BytesIO()
    _send(buf, {"config": {"tile": 128}})
    truncated = io.BytesIO(buf.getvalue()[:-3])
    assert _recv(truncated) is None


def test_a_closed_pipe_reads_as_nothing():
    """EOF is a normal end of stream, not an error. Workers exit; the parent must not raise."""
    assert _recv(io.BytesIO()) is None
    assert _recv(io.BytesIO(b"\x00\x00\x00\x00")) is None


def test_tensors_are_reduced_to_metadata_and_everything_else_is_passed_through():
    """A worker rebuilds shapes, not data: the payload is what makes it cheap.

    Sending real tensors would serialize gigabytes per candidate. Sending metadata is enough because
    a compile depends on shape, stride and dtype and not on values -- which is the same reason the
    tuner compiles against fake tensors in the first place.
    """
    import torch

    described = _describe_args([torch.zeros(4, 8), 7, "k", None])
    assert described[1:] == (7, "k", None)
    meta = described[0]["__tensor__"]
    assert meta["shape"] == (4, 8) and meta["stride"] == (8, 1) and meta["dtype"] == "float32"
    # Shape and stride are TUPLES: this metadata is fold-cp's own, so its container type is an
    # implementation detail and is chosen to survive MessagePack unchanged. A caller's own list is
    # refused instead of converted -- see `test_a_caller_list_is_refused_rather_than_retyped`.
    assert wire_admissible(described), "the described payload must survive the worker protocol"


def test_a_locally_defined_kernel_is_skipped_rather_than_failing():
    """A qualname a fresh interpreter cannot import is a SKIP, not an exception.

    The worker resolves the kernel by ``module`` + ``qualname``. A function defined inside a test
    (or built dynamically) has neither, so pre-compilation is impossible -- and the correct response
    is to fall back to serial compilation, because the tuning run is otherwise fine.
    """

    def local_kernel(x, tile=None):
        """A kernel whose qualname is nested inside this test function."""
        return x

    precompile_configs(local_kernel, (1,), {}, [AutotuneConfig(tile=64)])  # must not raise


def test_an_unpicklable_argument_is_skipped_rather_than_failing():
    """An argument that cannot cross the pipe skips pre-compilation, quietly."""
    precompile_configs(len, (lambda: None,), {}, [AutotuneConfig(tile=64)])  # must not raise


def test_no_configs_is_a_no_op():
    """Nothing to compile means nothing to do -- and no worker pool to pay for."""
    precompile_configs(len, ((),), {}, [])  # must not raise


def test_a_tensor_passed_by_KEYWORD_is_reduced_too_and_not_shipped_whole():
    """The reduction covers keyword arguments, or an operand passed by name crosses the pipe whole.

    This is the asymmetry that made `select="autotune"` CRASH. Only positional arguments were
    described, so a kernel taking an operand by keyword -- `layernorm_gemm(..., gate3=...)`, the
    workflow's own call -- had that entire tensor serialized and sent to every worker for every
    candidate. At the declared workflow shape (M=4194304, D=512) `gate3` is 4194304 x 512 bf16 =
    exactly 2**32 bytes, one byte more than the ``<I`` length prefix in `_send` can express, so the
    sweep died with ``struct.error``. Below that boundary it did not crash, it just moved gigabytes
    per candidate and re-allocated them on the device in every worker.

    It is COLD-CACHE ONLY, which is why it survived: with the artifacts already cached the first
    compile returns before the worker pool is ever spawned, so a warm developer box cannot see it.

    The size assertion is the point of the test. Asserting only the SHAPE of the descriptor would
    still pass if the tensor rode along beside it.
    """
    import torch

    big = torch.zeros(1024, 1024, dtype=torch.float32)  # 4 MiB of data, ~90 bytes of metadata
    described = _describe_kwargs({"gate3": big, "eps": 1e-5, "select": "autotune"})

    assert described["eps"] == 1e-5 and described["select"] == "autotune"
    meta = described["gate3"]["__tensor__"]
    assert (
        meta["shape"] == (1024, 1024) and meta["stride"] == (1024, 1) and meta["dtype"] == "float32"
    )
    assert _wire_len(described) < 1024, "the tensor's DATA is still crossing the pipe"
    assert _wire_len(described) * 100 < big.numel() * big.element_size()


def test_a_described_keyword_tensor_rebuilds_with_the_same_layout():
    """Shape, stride AND dtype survive the trip, or the worker compiles a different kernel.

    A worker that rebuilt a contiguous tensor from a LayoutLeft one would compile against the wrong
    strides and cache the artifact under the parent's key -- a silently wrong compiled kernel rather
    than a crash, which is the worse failure. The case checked here is deliberately off-grid and
    LayoutLeft, since that is where a shape-only round trip would look correct and be wrong.
    """
    import torch

    src = torch.empty_strided((301, 200), (1, 301), dtype=torch.bfloat16)
    described = _describe_kwargs({"x": src})
    assert wire_admissible(described), "the described kwargs must survive the worker protocol"
    packed = msgpack.packb(described, use_bin_type=True)
    meta = msgpack.unpackb(packed, raw=False, use_list=False, strict_map_key=True)["x"]["__tensor__"]
    rebuilt = torch.empty_strided(
        meta["shape"], meta["stride"], dtype=getattr(torch, meta["dtype"])
    )

    assert tuple(rebuilt.shape) == (301, 200)
    assert rebuilt.stride() == (1, 301)
    assert rebuilt.dtype is torch.bfloat16


def test_the_workflow_shape_is_exactly_the_frame_prefix_boundary():
    """The arithmetic that turned this from a slowdown into a crash, pinned at the shape it broke.

    `gate3` at the declared workflow cell (M=4194304 = N_token 2048 squared, D=512) is bf16, so its
    payload is exactly ``2**32`` bytes -- ONE more than ``struct.pack("<I", ...)`` accepts. That is
    why this defect presented as a hard `struct.error` at one specific production shape while merely
    wasting bandwidth at every smaller one, and why a test at a convenient size would not have
    caught it.

    The tensor is built on the ``meta`` device: `_describe_value` reads only shape, stride and
    dtype, so the workflow shape can be described exactly with no 4 GiB allocation.
    """
    import struct

    import torch

    gate3_bytes = 4194304 * 512 * 2
    assert gate3_bytes == 2**32, "the boundary this test is about has moved"
    with pytest.raises(struct.error):
        struct.pack("<I", gate3_bytes)

    gate3 = torch.empty_strided((4194304, 512), (512, 1), dtype=torch.bfloat16, device="meta")
    described = _describe_kwargs({"gate3": gate3})
    assert described["gate3"]["__tensor__"]["shape"] == (4194304, 512)
    # What now crosses the pipe is a length `<I` can express, by four orders of magnitude.
    assert _wire_len(described) < 2**32 - 1
    assert _wire_len(described) < 1024


def test_every_AutotuneConfig_CONSTRUCTED_IN_THE_PACKAGE_survives_the_worker_protocol():
    """Scan the shipped source for real config constructions and check each one can be sent.

    Purpose
        The pre-compile silently degrades to serial when a payload is inadmissible -- correct, but
        invisible. If a SHIPPED config were inadmissible, every autotuned kernel would quietly lose
        worker pre-compilation and the only symptom would be a slower cold sweep, which nobody
        attributes.

    Semantics
        Parses every module under ``fold_cp_ops/`` and evaluates the LITERAL keyword arguments of
        each ``AutotuneConfig(...)`` call. Literal-only on purpose: a computed value cannot be
        resolved without running the producer, and this test's job is to cover what the source
        states, not to re-execute the policy. Calls with no literal keywords are skipped, and the
        count of what WAS checked is asserted -- a scan that silently matched nothing would pass.

        It cannot rot: a new config with a new knob type is picked up the next time this runs,
        without anyone remembering to extend a list.
    """
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3] / "fold_cp_ops"
    checked, offenders = 0, []
    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - a syntactically broken shipped file
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "AutotuneConfig"):
                continue
            kwargs = {}
            for kw in node.keywords:
                if kw.arg is None:
                    break
                try:
                    kwargs[kw.arg] = ast.literal_eval(kw.value)
                except (ValueError, SyntaxError):
                    break
            else:
                if not kwargs:
                    continue
                checked += 1
                if not wire_admissible(kwargs):
                    offenders.append(f"{path.name}:{node.lineno} {kwargs!r}")
    assert checked >= 5, f"the scan only found {checked} literal configs; the walk is broken"
    assert not offenders, (
        "shipped AutotuneConfig values that the worker protocol cannot carry -- these would "
        "silently disable pre-compilation:\n  " + "\n  ".join(offenders)
    )

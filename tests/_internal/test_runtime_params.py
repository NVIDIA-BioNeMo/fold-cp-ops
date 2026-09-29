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


"""Tests for ``fold_cp_ops._internal.runtime_params`` -- host -> kernel parameter marshalling.

Host-side only: the extract/rebuild protocol is exercised with plain Python values rather than MLIR
ones, which is enough to pin the part that actually breaks -- the pairing of the recorded per-field
value counts with the order they are consumed in.
"""

from dataclasses import dataclass
from typing import NamedTuple, Optional

import cutlass
import pytest

from fold_cp_ops._internal.compile_time.template_params import StaticTypes
from fold_cp_ops._internal.runtime_params import (
    ParamsBase,
    _partition_fields,
    _patched_convert_single_arg,
    mlir_namedtuple,
)


@dataclass
class _Mixed(ParamsBase):
    """A params struct with both constant and marshalled fields, for the partition tests."""

    count: int = 3
    flag: bool = True
    label: str = "x"
    absent: Optional[int] = None
    payload: object = None


def test_partition_splits_constants_from_marshalled_fields():
    """Ints, bools, strings and None are baked in; anything else is marshalled."""
    obj = _Mixed(payload=[1, 2])
    const, runtime = _partition_fields(obj)
    assert set(const) == {"count", "flag", "label", "absent"}
    assert set(runtime) == {"payload"}


def test_partition_preserves_declaration_order():
    """Order is the wire format -- the flat value list carries no names.

    A partition that reordered fields would extract and rebuild them in different orders, which
    silently swaps two same-shaped tensors rather than raising.
    """
    obj = _Mixed(payload=object())
    _, runtime = _partition_fields(obj)
    assert list(runtime) == [
        f.name for f in _Mixed.__dataclass_fields__.values() if f.name == "payload"
    ]


def test_partition_treats_a_tuple_as_runtime_not_constant():
    """A tuple of ints is NOT baked in, even though every element would be.

    The test is a flat ``isinstance``, deliberately: ``_new_from_mlir_values`` consumes tuples from
    the value list, so making the partition recurse without changing the rebuild would drop them.
    """

    @dataclass
    class WithTuple(ParamsBase):
        shape: tuple = (128, 4)

    const, runtime = _partition_fields(WithTuple())
    assert "shape" in runtime and "shape" not in const


def test_partition_rejects_a_non_dataclass():
    """``dataclasses.fields`` is used, so a plain object fails loudly rather than partitioning to {}."""
    with pytest.raises(TypeError):
        _partition_fields(object())


def test_rebuild_requires_the_extract_to_have_run():
    """The per-field value counts live on the instance, so the two halves are one protocol."""
    obj = _Mixed(payload=None)
    with pytest.raises(AttributeError, match="_values_pos"):
        obj.__new_from_mlir_values__([])


def test_extract_records_one_count_per_marshalled_field():
    """``_values_pos`` has an entry per runtime field, which is what the rebuild chunks by."""
    obj = _Mixed(payload=None)
    obj.__extract_mlir_values__()
    _, runtime = _partition_fields(obj)
    assert len(obj._values_pos) == len(runtime)


def test_mlir_namedtuple_installs_the_rebuild_hook():
    """The decorator's whole job: give a NamedTuple a ``__new_from_mlir_values__``."""

    class Plain(NamedTuple):
        a: int = 1

    assert not hasattr(Plain, "__new_from_mlir_values__")
    decorated = mlir_namedtuple(Plain)
    assert decorated is Plain, "the decorator mutates in place and returns the same class"
    assert hasattr(Plain, "__new_from_mlir_values__")


def test_namedtuple_rebuild_carries_constants_through_untouched():
    """None and ``StaticTypes`` fields come from the template, consuming no values.

    That is what keeps a ``Constexpr`` field a compile-time constant on the kernel side -- and why
    the caller passes None into its ABI slot at launch.
    """

    @mlir_namedtuple
    class Args(NamedTuple):
        n: int = 7
        absent: Optional[object] = None
        name: str = "k"

    rebuilt = Args().__new_from_mlir_values__([])
    assert rebuilt == Args(7, None, "k")


def test_constexpr_annotation_becomes_a_const_none_slot():
    """The TVM-FFI patch's first case: a ``Constexpr`` parameter must not get a runtime slot.

    Without it the converter would demand a real value for a field whose value is already baked
    into the compiled kernel, and the call would fail on argument count.
    """
    result = _patched_convert_single_arg(123, "n", cutlass.Constexpr[int], None)
    assert type(result).__name__ == "ConstNone"


def test_static_types_is_shared_with_template_params():
    """One classification, imported rather than redefined.

    Two copies would let "constant" come to mean different things in the compile-time and runtime
    halves of the parameter story -- a field baked in by one and marshalled by the other.
    """
    import fold_cp_ops._internal.runtime_params as rp

    assert rp.StaticTypes is StaticTypes

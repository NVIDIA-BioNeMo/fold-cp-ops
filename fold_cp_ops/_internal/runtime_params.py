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

# Copyright (c) 2025, Tri Dao.
# This file has been modified by NVIDIA CORPORATION & AFFILIATES.

"""Parameter structs that carry **runtime** values across the host -> kernel boundary.

The exact complement of `compile_time/template_params.py`. Both describe a functor's parameters;
they differ in what the fields are allowed to hold, and the difference is not cosmetic:

* `TemplateParams` — every field is a compile-time constant, and a tensor in one is a `TypeError`.
* `ParamsBase` / `@mlir_namedtuple` (here) — fields *may* be tensors, layouts or pointers, and are
  marshalled field-by-field into the flat list of MLIR values the DSL threads into the kernel.

The marshalling is the whole content of this module. A struct passed to a `@cute.kernel` is not
copied; the DSL calls `__extract_mlir_values__` on the host side to flatten it into SSA values, and
`__new_from_mlir_values__` inside the kernel to rebuild an object of the same shape around the
values that arrived. Fields that are `StaticTypes` (ints, bools, dtypes, `Constexpr`, None) skip the
round trip entirely and are read off the compile-time template, which is what makes them constants
in the generated code.

**Why the value *count* per field is recorded.** A `cute.Tensor` flattens to several MLIR values,
not one, and how many depends on its layout. `ParamsBase.__extract_mlir_values__` therefore stashes
the per-field counts in `self._values_pos` on the way out, and `_new_from_mlir_values` consumes the
list in those same chunks on the way back. Rebuilding from a `self` whose extract has not run is
consequently an error — the pairing is the protocol, not an optimization.

**This module also installs a global patch on the cutlass TVM-FFI argument converter** (see
`_patched_convert_single_arg`). That is a side effect of import, applied once and idempotent. It is
here rather than in an arbitrary module because both of the things it fixes are marshalling
questions about exactly these structs.
"""

from dataclasses import dataclass, fields
from typing import Any, Dict, Tuple, get_origin

import cutlass
from cutlass.base_dsl.tvm_ffi_builder import spec

# Imported rather than redefined: this is the same classification `TemplateParams` applies as an
# admission test.  One list, so "constant" cannot come to mean two different things in the two
# halves of the parameter story.
from fold_cp_ops._internal.compile_time.template_params import StaticTypes

# ── TVM-FFI argument-converter patch ───────────────────────────────────────────────────────────────
# `cutlass.cute._tvm_ffi_args_spec_converter` builds, per JIT entry, the spec describing how each
# Python argument becomes a TVM-FFI argument.  Two cases it gets wrong for the structs above:
#
#   1. A field annotated `cutlass.Constexpr[T]` is a COMPILE-TIME value, but the stock converter
#      classifies it by its runtime object and emits a runtime argument slot for it.  It must be
#      `spec.ConstNone` — the value is already baked into the compiled kernel, and the caller passes
#      None into the slot.
#   2. A NamedTuple passed where the annotation is a plain `tuple` (or absent) loses its own field
#      type hints, so the converter walks it positionally with no types.  Redirecting to `type(arg)`
#      restores the hints.
#
# Applied at import, once.  It is a process-global monkeypatch of a third-party function, so the
# idempotence guard below matters: importing this module twice under two names (which pytest's
# `prepend` mode can do) would otherwise wrap the patch around itself and hide case 1 behind its
# own recursion.
import cutlass.cute._tvm_ffi_args_spec_converter as _converter_module  # noqa: E402

_PATCH_FLAG = "_fold_cp_ops_constexpr_patch"
_original_convert_single_arg = (
    getattr(_converter_module._convert_single_arg, "__wrapped_original__", None)
    or _converter_module._convert_single_arg
)


def _patched_convert_single_arg(arg: Any, arg_name: str, arg_type: Any, ctx: Any):
    """Classify one JIT argument for TVM-FFI, fixing `Constexpr` and bare-annotated NamedTuples.

    Wraps the stock cutlass converter; every case it does not claim is delegated unchanged, so this
    is additive rather than a reimplementation.

    Args:
        arg: The Python argument value. Only inspected for case 2 (is it a NamedTuple instance).
        arg_name: The parameter's name, used to build the emitted spec entry.
        arg_type: The declared annotation, or None when the entry has none. `cutlass.Constexpr[T]`
            here is what triggers case 1.
        ctx: The converter's context object, passed through untouched.

    Returns:
        `spec.ConstNone(arg_name)` for a `Constexpr`-annotated parameter — the compile-time value is
        already in the kernel, and the caller must pass None in that slot. Otherwise whatever the
        stock converter returns, possibly after redirecting `arg_type` to the NamedTuple's own type.

    Raises:
        Whatever the stock converter raises for an argument neither case claims.
    """
    if arg_type is not None and get_origin(arg_type) is cutlass.Constexpr:
        return spec.ConstNone(arg_name)
    if (
        isinstance(arg, tuple)
        and hasattr(type(arg), "_fields")
        and (arg_type is None or not hasattr(arg_type, "_fields"))
    ):
        return _original_convert_single_arg(arg, arg_name, type(arg), ctx)
    return _original_convert_single_arg(arg, arg_name, arg_type, ctx)


if not getattr(_converter_module._convert_single_arg, _PATCH_FLAG, False):
    setattr(_patched_convert_single_arg, _PATCH_FLAG, True)
    setattr(_patched_convert_single_arg, "__wrapped_original__", _original_convert_single_arg)
    _converter_module._convert_single_arg = _patched_convert_single_arg


# ── Marshalling ────────────────────────────────────────────────────────────────────────────────────
def _partition_fields(obj: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Split a dataclass instance's fields into the ones baked in and the ones marshalled.

    Args:
        obj: A dataclass **instance** (not a class). `dataclasses.fields` is used, so a plain object
            raises `TypeError`.

    Returns:
        `(constexpr, non_constexpr)`, two name -> value dicts partitioned by
        `isinstance(value, StaticTypes)`. Insertion order follows field declaration order, and both
        halves rely on it: the non-constexpr order is the order values are extracted and consumed.

    Note:
        The test is a **flat** `isinstance`, not the recursive `is_compile_time_value`. A tuple field
        therefore lands in `non_constexpr` even when every element is an int, and is marshalled
        rather than baked in. That is deliberate and matched by `_new_from_mlir_values`; changing it
        to recurse here without changing the rebuild would silently drop those fields on the kernel
        side.
    """
    all_fields = {field.name: getattr(obj, field.name) for field in fields(obj)}
    constexpr = {n: f for n, f in all_fields.items() if isinstance(f, StaticTypes)}
    non_constexpr = {n: f for n, f in all_fields.items() if not isinstance(f, StaticTypes)}
    return constexpr, non_constexpr


def _new_from_mlir_values(self: Any, values):
    """Rebuild a `ParamsBase` dataclass inside the kernel from the flattened MLIR values.

    Args:
        self: The **host-side template** instance. Supplies the constexpr fields verbatim, the field
            order, and `self._values_pos` — the per-field value counts recorded by
            `__extract_mlir_values__`. Calling this on an instance whose extract has not run raises
            `AttributeError`; the two halves are one protocol.
        values: The flat sequence of MLIR values the DSL delivered, in extraction order.

    Returns:
        A new instance of the same class, with runtime fields rebuilt from `values` and constexpr
        fields carried over from the template.

    Raises:
        AttributeError: If `self._values_pos` is absent, i.e. extract never ran on this instance.
        IndexError: If `values` is shorter than the recorded counts, which means the caller paired a
            rebuild with a different template than the one that was extracted.
    """
    constexpr_fields, non_constexpr_fields = _partition_fields(self)
    for (name, field), n_items in zip(non_constexpr_fields.items(), self._values_pos):
        non_constexpr_fields[name] = cutlass.new_from_mlir_values(field, values[:n_items])
        values = values[n_items:]
    return self.__class__(**non_constexpr_fields, **constexpr_fields)


def _namedtuple_new_from_mlir_values(self: Any, values):
    """Rebuild a NamedTuple inside the kernel from the flattened MLIR values.

    The NamedTuple counterpart of `_new_from_mlir_values`, installed by `@mlir_namedtuple`. It does
    not need a recorded `_values_pos`: a NamedTuple's fields are positional, so the count for each
    runtime field is re-derived from the template value via `get_mlir_types`.

    Args:
        self: The host-side template NamedTuple. Its field values decide, per position, whether a
            value is consumed (runtime) or carried over (None or `StaticTypes`).
        values: The flat sequence of MLIR values, in field order.

    Returns:
        A new NamedTuple of the same class. Fields that are None or `StaticTypes` on the template
        are copied through unchanged — which is how a `Constexpr`-annotated field stays a compile-
        time constant, and how an absent optional stays absent.

    Raises:
        IndexError: If `values` runs out, meaning the template does not match what was extracted.
    """
    from cutlass.base_dsl.typing import get_mlir_types

    values = list(values)
    new_fields = []
    for field_val in self:
        if field_val is None or isinstance(field_val, StaticTypes):
            new_fields.append(field_val)
        else:
            n_items = len(get_mlir_types(field_val))
            new_fields.append(cutlass.new_from_mlir_values(field_val, values[:n_items]))
            values = values[n_items:]
    return self.__class__(*new_fields)


def mlir_namedtuple(cls):
    """Class decorator making a NamedTuple passable to a `@cute.kernel` as a parameter struct.

    Installs `__new_from_mlir_values__`; the extract side needs no help, because the DSL already
    knows how to flatten a tuple. Use it on any NamedTuple whose fields are handed to a kernel::

        @mlir_namedtuple
        class MyArgs(NamedTuple):
            tensor_arg: cute.Tensor
            const_arg: cutlass.Constexpr[int] = 0

    Args:
        cls: A NamedTuple class (not an instance). Nothing checks that here — the decorator only
            sets an attribute — but applying it to a non-NamedTuple produces a rebuild that iterates
            the wrong thing, so the constraint is real even though it is unenforced.

    Returns:
        `cls`, mutated in place, so it can be used as a decorator.

    Note:
        A `Constexpr`-annotated field pairs with the TVM-FFI patch at the top of this module: the
        value is baked into the compiled kernel, and the caller passes **None** in that slot at call
        time. Without the patch the converter would demand a real runtime value there.
    """
    cls.__new_from_mlir_values__ = _namedtuple_new_from_mlir_values
    return cls


@dataclass
class ParamsBase:
    """Base for a dataclass parameter struct that carries runtime values into a kernel.

    Subclass it and declare fields as an ordinary dataclass; tensors, layouts, pointers and TMA
    atoms are all admissible, alongside constants. The two DSL protocol methods below are inherited,
    so a subclass writes no marshalling code.

    Input requirements are on the *subclass*, and there are two. Fields must be declared in a stable
    order, because order is the wire format — the flat value list carries no names. And the same
    instance that was extracted must be the one rebuilt from, because the per-field value counts are
    stashed on it. Both hold automatically in the normal usage (build the struct on the host, hand
    it to the compiled entry) and are only violated by trying to be clever.

    Use `compile_time/template_params.TemplateParams` instead when every field is a constant: it
    rejects a runtime value at construction, whereas this class will happily marshal one that was
    never meant to cross the boundary.
    """

    def __extract_mlir_values__(self):
        """Flatten the runtime fields into MLIR values, recording each field's value count.

        Returns:
            The concatenated MLIR values of every non-`StaticTypes` field, in declaration order.
            Constexpr fields contribute nothing — they are read from the template on the other side.

        Note:
            Sets `self._values_pos` as a side effect. `_new_from_mlir_values` requires it, so this
            method has to run before any rebuild against this instance.
        """
        _, non_constexpr_fields = _partition_fields(self)
        values, self._values_pos = [], []
        for obj in non_constexpr_fields.values():
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    __new_from_mlir_values__ = _new_from_mlir_values

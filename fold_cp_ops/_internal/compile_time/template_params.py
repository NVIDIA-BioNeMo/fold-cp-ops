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

"""``TemplateParams`` — the declared, validated, immutable compile-time parameter pack.

A CuTe-DSL kernel functor's inputs split in two, and **the DSL decides the split, not the author**:

* **Compile-time** — plain Python scalars (and numeric *types*). Read during tracing and folded
  into the kernel as constants. In C++ CUTLASS these are template parameters; Python has no
  templates, so they live on the functor instance.
* **Runtime** — tensors, layouts, atoms, pointers. These carry MLIR values that **do not survive a
  ``self`` stash across the ``@cute.jit`` -> ``@cute.kernel`` boundary** (recorded in CLAUDE.md for
  peer TMA atoms: they "must be threaded as kernel args"). Stashing one on ``self`` does not raise
  — it produces a kernel reading a dangling or stale value.

Nothing in plain Python distinguishes the two, so the mistake is invisible at the assignment site
and shows up as a wrong result much later. This module makes the distinction **declarative and
checked**:

    class MyParams(TemplateParams):
        dtype: Type[cutlass.Numeric]
        N: int
        cluster_n: int = 1

    class MyKernel(TemplateParamsMixin):
        Params = MyParams
        def __init__(self, dtype, N):
            self._bind_params(dtype=dtype, N=N, cluster_n=4 if N > 8192 else 1)

Declaring a field in ``MyParams`` is the single statement that says "``self.dtype`` / ``self.N`` /
``self.cluster_n`` are compile-time constants, guaranteed immutable for the life of the functor."
Three guarantees follow, all enforced rather than documented:

1. **Runtime values are rejected at construction.** A field holding a tensor/layout/pointer raises
   ``TypeError`` naming the field, instead of silently producing a broken kernel.
2. **Params are immutable.** Rebinding one after construction raises ``AttributeError``. This is
   what stops config from being assigned by whichever method happens to run first.
3. **Params are complete at construction.** There is no second phase, so no method can depend on an
   attribute that does not exist yet.

The complement is ``ParamsBase`` (upstream ``cute_dsl_utils``), for structs that genuinely *do*
carry runtime values across the boundary by marshalling them as MLIR values. Use ``ParamsBase``
when a field must be a tensor; use ``TemplateParams`` when every field is a constant.
"""

import dataclasses
import enum
import functools
from typing import Any, Dict, FrozenSet, Tuple, Type

import cutlass
import msgpack
from cutlass.cutlass_dsl import NumericMeta

#: What :meth:`TemplateParamsMixin.compile_key` records for a declared gate that the instance does
#: not carry. A plain string rather than a sentinel OBJECT because the key is PICKLED: a module-level
#: object pickles by reference and compares unequal across processes, which would turn every
#: cross-process cache lookup into a miss without ever reporting one.
GATE_UNSET = "<unset>"

#: Types the DSL can fold into a kernel as a compile-time constant. Taken from upstream
#: ``cute_dsl_utils.StaticTypes``, which ``ParamsBase`` uses to partition constexpr from runtime
#: fields — the same classification, applied here as an admission test rather than a split.
#: ``NumericMeta`` is the metaclass of the cutlass numeric *types*, so ``cutlass.BFloat16`` (a
#: class, not an instance) qualifies; that is how ``dtype`` is admissible.
#:
#: ``enum.Enum`` is the one addition over upstream, and it is required rather than convenient:
#: ``cutlass.utils.LayoutEnum`` — the operand major mode, which decides the WGMMA atom and every
#: SMEM swizzle — is a plain ``Enum`` (``RoundingMode`` is an ``IntEnum`` and was already admitted
#: by ``int``). An enum member is a module-level singleton read at trace time, exactly like a bool;
#: it carries no MLIR value. Without it the operand majors could not be declared parameters, and
#: the functor's compile-time surface would be incomplete by exactly the four fields that matter
#: most to the compiled artifact.
StaticTypes: Tuple[type, ...] = (
    cutlass.Constexpr,
    NumericMeta,
    enum.Enum,
    int,
    bool,
    str,
    float,
    type(None),
)


def is_compile_time_value(value: Any) -> bool:
    """Return whether ``value`` can be folded into a kernel as a compile-time constant.

    Args:
        value: Any object. Tuples are inspected **element-wise and recursively**, so a shape like
            ``(128, 4)`` is admissible while ``(tensor, 4)`` is not — a tuple is otherwise an
            ``isinstance`` blind spot that would let a runtime value through.

    Returns:
        True if the DSL can treat it as a constant during tracing. False for tensors, layouts,
        atoms, pointers and anything else carrying MLIR values, which must be kernel arguments.
    """
    if isinstance(value, tuple):
        return all(is_compile_time_value(v) for v in value)
    return isinstance(value, StaticTypes)


#: Types a swept instance attribute may hold and still be admitted to :meth:`compile_key`.
#: DELIBERATELY NARROWER than :func:`is_compile_time_value`: this list must also survive
#: ``pickle`` -> ``sha256`` with a value that is IDENTICAL across processes, which is a stronger
#: requirement than "the tracer can fold it". A cutlass dtype (``Float32``) is a class and pickles
#: by module+qualname, so it qualifies; an ``enum.Enum`` member does too.
_KEY_SCALARS = (bool, int, float, str, bytes, type(None))


def is_key_component(value: Any, _depth: int = 0) -> bool:
    """Whether a swept instance attribute belongs in a compile cache key.

    Purpose
        Lets :meth:`TemplateParamsMixin.compile_key` sweep ``__dict__`` for post-construction
        configuration WITHOUT a hand-maintained name list, which is the only version of that sweep
        that cannot rot: ``configure_a2a*`` writes ~39 attributes on one functor here, and a list
        somebody must remember to extend is wrong the first time a knob is added.

    Functionality & semantics
        Accepts scalars, ``enum`` members, cutlass numeric types (classes with a ``NumericMeta``
        metaclass), and tuples/frozensets of those, recursively to ``_depth`` 4. Rejects everything
        else -- tensors, streams, layouts, atoms, dataclass packs, cached derived objects.

        **The rejection direction is the safe one and that is the design.** A compile-relevant value
        wrongly rejected costs an extra cache MISS (a recompile, i.e. today's behaviour). A runtime
        value wrongly ACCEPTED costs a key that varies per call, i.e. also a miss. Neither can serve
        a wrong artifact, which is the failure this key exists to prevent.

    Args:
        value: Any object off the instance ``__dict__``.
        _depth: Recursion depth, internal. A container nested deeper than 4 is rejected rather than
            walked, so a pathological structure cannot make key construction unbounded.

    Returns:
        True if the value may be pickled into a cache key.
    """
    if isinstance(value, _KEY_SCALARS):
        return True
    if isinstance(value, enum.Enum):
        return True
    if isinstance(value, NumericMeta):
        return True
    if isinstance(value, (tuple, frozenset)):
        return _depth < 4 and all(is_key_component(v, _depth + 1) for v in value)
    return False


#: Bumped whenever :func:`normalize_key_component` changes what it emits for an UNCHANGED input.
#: It is the first element of every normalized key, so a bump makes every previously-derived hash
#: unreachable rather than reinterpreted -- one cold rebuild, never a stale artifact served under a
#: key whose meaning moved. Do not reuse a number.
KEY_SCHEMA_VERSION = 1


class UnsupportedKeyComponent(TypeError):
    """A value outside the :func:`is_key_component` domain reached the normalizer.

    Carried as its own type so a caller can tell "this key cannot be formed" from any other
    ``TypeError`` raised while building one. Every caller in this repo responds by BYPASSING the
    disk cache and compiling normally -- a key that cannot be formed is a cache miss, never an
    error the workload sees.
    """


def normalize_key_component(value: Any, _depth: int = 0) -> Any:
    """Canonical, type-TAGGED, MessagePack-safe form of a cache-key component.

    Purpose
        Give ``sha256`` something to hash that (a) survives MessagePack without losing type
        information and (b) is byte-identical across processes and interpreter runs. ``pickle``
        provided neither guarantee for free: this repo removed it from every key path, and a plain
        MessagePack encoding of the raw value would silently merge values that must not share a
        cache entry.

    Functionality & semantics
        Emits a nested tuple whose FIRST element is a short type tag. The tag is what makes the
        encoding injective, and three collisions make it necessary rather than decorative:

        * ``True``, ``1`` and ``1.0`` are all ``==`` in Python and all encode to the same
          MessagePack integer/float families under a naive dump. Tagged, they are ``("b", True)``,
          ``("i", "1")`` and ``("f", "0x1.0000000000000p+0")``.
        * ``(1, 2)`` and ``frozenset({1, 2})`` both dump as an array. Tagged, they differ, and the
          frozenset's members are additionally sorted BY THEIR ENCODED BYTES so that two equal
          frozensets built in different insertion orders normalize identically -- set iteration
          order is not a stable key.
        * an ``enum`` member and its integer value are ``==`` when the enum is an ``IntEnum``.

        Order of the isinstance checks is load-bearing: ``bool`` is a subclass of ``int`` and an
        ``IntEnum`` member is an ``int``, so ``enum`` and ``bool`` are tested BEFORE ``int`` or the
        tags collapse to the wrong one silently.

        ``int`` is emitted as DECIMAL TEXT and ``float`` via ``float.hex()``, both for exactness:
        MessagePack integers are bounded at 64 bits (a larger one raises), and a decimal float
        repr is lossy in the last ulp. ``hex()`` round-trips a double exactly, including
        subnormals, and is stable across platforms.

    Args:
        value: Any object from the :func:`is_key_component` domain -- scalars, ``bytes``, ``None``,
            ``enum`` members, cutlass numeric types, and tuples/frozensets of those.
        _depth: Recursion depth, internal. Beyond 4 the value is REJECTED rather than walked, which
            matches :func:`is_key_component` exactly; the two must agree, or a value that passes the
            admission check could still fail to normalize.

    Returns:
        A tuple of ``str``/``bytes``/``bool``/tuple, containing no Python object that MessagePack
        cannot represent.

    Raises:
        UnsupportedKeyComponent: If *value* is outside the domain, or nested deeper than 4. The
            caller must treat this as "no disk key", not as a failure of the work being done.
    """
    if value is None:
        return ("n",)
    if isinstance(value, enum.Enum):
        # Before int: an IntEnum member IS an int, and keying it as one would merge two different
        # configurations whenever their integer values coincide.
        cls = type(value)
        return ("e", cls.__module__, cls.__qualname__, str(value.name))
    if isinstance(value, bool):
        # Before int: bool is a subclass of int, so this must not fall through to ("i", "1").
        return ("b", value)
    if isinstance(value, int):
        # Decimal TEXT, not a MessagePack integer: an int past 64 bits raises on encode, and a key
        # that cannot be formed for a large-but-legal value would silently disable the cache.
        return ("i", str(value))
    if isinstance(value, float):
        return ("f", float.hex(value))
    if isinstance(value, str):
        return ("s", value)
    if isinstance(value, bytes):
        return ("y", value)
    if isinstance(value, NumericMeta):
        # A cutlass numeric TYPE (``Float32``), not an instance. Keyed by identity, not by any
        # attribute, so a widened class body cannot change existing keys.
        return ("N", value.__module__, value.__qualname__)
    if isinstance(value, tuple):
        if _depth >= 4:
            raise UnsupportedKeyComponent(f"tuple nested deeper than 4: {value!r}")
        return ("t", tuple(normalize_key_component(v, _depth + 1) for v in value))
    if isinstance(value, frozenset):
        if _depth >= 4:
            raise UnsupportedKeyComponent(f"frozenset nested deeper than 4: {value!r}")
        # Sort by ENCODED BYTES, not by the members themselves: the members need not be mutually
        # orderable (``frozenset({1, "a"})`` raises on a plain ``sorted``), while their encodings
        # always are. Two equal frozensets therefore normalize identically whatever order they
        # were built in.
        encoded = [msgpack.packb(normalize_key_component(v, _depth + 1), use_bin_type=True)
                   for v in value]
        encoded.sort()
        return ("F", tuple(encoded))
    raise UnsupportedKeyComponent(f"{type(value).__name__} is not a key component: {value!r}")


def canonical_key_bytes(key: Any) -> bytes:
    """MessagePack bytes for *key*, versioned and type-tagged, ready to hash.

    Purpose
        The single encoder every cache key in this package goes through, so that two key paths
        cannot drift into disagreeing about what a value means.

    Functionality & semantics
        Wraps :func:`normalize_key_component` output in ``(KEY_SCHEMA_VERSION, normalized)`` and
        packs it with ``use_bin_type=True``. Deterministic across processes: the normalizer emits
        only text, bytes, bools and tuples, and MessagePack's encoding of those is fixed -- unlike
        ``pickle``, whose output is stable only within one interpreter version and protocol.

    Args:
        key: Any value in the :func:`is_key_component` domain; in practice a tuple of components.

    Returns:
        The packed bytes. Callers hash these; nothing reads them back.

    Raises:
        UnsupportedKeyComponent: Propagated from the normalizer. Callers bypass the disk cache.
    """
    return msgpack.packb((KEY_SCHEMA_VERSION, normalize_key_component(key)), use_bin_type=True)


class TemplateParams:
    """Base for a kernel functor's compile-time parameter pack. Subclass and declare fields.

    Subclasses are turned into **frozen, keyword-only dataclasses** automatically, so a subclass
    need not repeat the decorator and cannot accidentally declare a mutable or positional one.
    Keyword-only also makes inheritance safe: a base with defaulted fields followed by a subclass
    with required ones is a ``TypeError`` for ordinary dataclasses, and that would make the
    hierarchy this class exists to support impossible.

    Every field is checked at construction (see :func:`is_compile_time_value`); a runtime value
    raises ``TypeError`` there rather than producing a kernel that reads a dangling MLIR value.
    """

    def __init_subclass__(cls, **kwargs):
        """Make every subclass a frozen, keyword-only dataclass.

        Args:
            cls: The parameter-pack subclass being defined.
            **kwargs: Forwarded to ``object.__init_subclass__``.

        Returns:
            None. Rebinds ``cls`` in place as a dataclass.
        """
        super().__init_subclass__(**kwargs)
        dataclasses.dataclass(frozen=True, kw_only=True)(cls)

    def __post_init__(self) -> None:
        """Reject any field that is not a compile-time value.

        Runs automatically after the generated ``__init__``.

        Raises:
            TypeError: If a field holds a tensor, layout, atom, pointer or any other object
                carrying MLIR values. The message names the field and its type, because the whole
                point is that this mistake is otherwise invisible until the kernel misbehaves.
        """
        bad = {
            f.name: type(getattr(self, f.name)).__name__
            for f in dataclasses.fields(self)
            if not is_compile_time_value(getattr(self, f.name))
        }
        if bad:
            raise TypeError(
                f"{type(self).__name__}: these fields are RUNTIME values, not compile-time "
                f"constants: {bad}. A runtime value (tensor, layout, atom, pointer) does not "
                f"survive a `self` stash across the @cute.jit -> @cute.kernel boundary — it does "
                f"not raise there, it yields a kernel reading a stale value. Pass it as an "
                f"explicit kernel argument instead, or use ParamsBase if it must be marshalled."
            )

    @classmethod
    def field_names(cls) -> FrozenSet[str]:
        """Names of the declared parameters.

        Returns:
            A frozenset, empty for ``TemplateParams`` itself (which declares no fields). Used by
            :class:`TemplateParamsMixin` to decide which attributes to protect, and by the guard
            test that checks kernels read only declared parameters.
        """
        if not dataclasses.is_dataclass(cls):
            return frozenset()
        return frozenset(f.name for f in dataclasses.fields(cls))


class TemplateParamsMixin:
    """Gives a kernel functor read-only ``self.<param>`` access to its frozen parameter pack.

    A functor sets ``Params`` to its :class:`TemplateParams` subclass and calls
    :meth:`_bind_params` once, in ``__init__``. Each declared field then reads back as a plain
    attribute — ``self.N`` — so call sites stay ergonomic, while writes to those names raise.

    Only *declared parameter* names are protected. Other attributes remain writable, deliberately:
    the guarantee being made is about values that reach the kernel as compile-time constants, and
    over-freezing the whole instance breaks legitimate per-instance caches (and, measured, breaks
    subclass ``__init__`` ordering in ways that are tedious to work around).

    **Two binding phases, both compile-time.** A kernel functor learns its configuration at
    construction and its *operand* facts -- dtypes, majors, layouts -- only when it is called with
    real tensors. Both are folded into the kernel as constants, because ``__call__`` is ``@cute.jit``
    and every ``self.X`` read during tracing is resolved at trace time. So the compile/runtime
    distinction is NOT "which phase":

        on ``self``          -> compile-time -> belongs in the compile cache key
        passed as argument   -> runtime      -> one artifact serves every value

    `Params` holds what is known at construction and binds in ``__init__``; :attr:`CallParams` holds
    what is read off the operands and binds at the top of ``__call__``. Both are frozen once bound,
    so "complete at binding" holds per phase, and together they are the functor's ENTIRE
    compile-time surface -- which is what lets a cache key be derived from them rather than
    hand-written beside them.

    Anything derived from either pack -- an MMA atom, a SMEM layout, a stage count -- is NOT a
    parameter: those are objects rather than constants, so :func:`is_compile_time_value` rejects
    them. Expose them as ``functools.cached_property``. That is deliberate and stronger than
    stashing them: a derived value cannot desynchronize from what it was derived from, and
    ``cached_property`` writes ``__dict__`` directly, so the guard below does not block it.

    Attributes:
        Params: Construction-phase parameters. Override in every subclass that adds fields.
        CallParams: Call-phase parameters, read off the operands. Defaults to an empty pack, which
            is the single-phase case -- a functor that needs no operand facts simply never calls
            :meth:`_bind_call_params`.
    """

    Params: Type[TemplateParams] = TemplateParams
    CallParams: Type[TemplateParams] = TemplateParams

    #: Attribute names read inside a compile-time gate (``cutlass.const_expr`` /
    #: ``range_constexpr``) that are NOT declared fields of ``Params`` / ``CallParams``.
    #:
    #: **This is the third leg of the compile-time surface, and it exists because the first two do
    #: not cover it.** A `Params` field is frozen at binding and therefore keyed by construction; an
    #: attribute set AFTER construction -- which is what every ``configure_a2a*`` call does -- is
    #: read by the tracer exactly the same way and folded into the kernel exactly the same way, but
    #: is invisible to ``param_dict()``. Measured 2026-08-18: 37 such gates across three functors,
    #: so two functors emitting DIFFERENT code returned the SAME ``compile_key()``.
    #:
    #: Declare every such name here and :meth:`compile_key` picks it up, closing that gap. An
    #: attribute that is absent on the instance contributes the sentinel string ``"<unset>"``, which
    #: is a DISTINCT key component from any value it might later hold -- so "not configured" and
    #: "configured to the default" cannot collide.
    #:
    #: Input requirements: a tuple of ``str`` attribute names, each of which must resolve to a
    #: HASHABLE, PICKLABLE value on a configured instance (the key is pickled). A tensor here makes
    #: every key unhashable and is a bug, not a slow path.
    #: ``tests/_internal/compile_time/test_template_params.py`` pins this tuple against an AST scan
    #: of the class body, so a gate added without a declaration fails at test time rather than
    #: silently un-keying a config.
    COMPILE_GATED_ATTRS: tuple = ()

    @classmethod
    def compile_gated_attrs(cls) -> tuple:
        """The MRO-UNION of every :attr:`COMPILE_GATED_ATTRS` declared on ``cls`` or any base.

        Purpose
            A subclass declaring its own tuple SHADOWS its parent's -- ordinary Python attribute
            lookup, and exactly the wrong semantics here. ``GemmSm90``'s mixins gate on
            ``_a2a_enabled`` / ``_a2a_ib_wide`` / ``_a2a_drain_tail``; if ``GemmSm90A2A`` had to
            restate them to keep them keyed, then adding a gate to the PARENT would silently drop it
            from every subclass's key -- the same silent-collision failure the declaration exists to
            close, one level up.

        Functionality & semantics
            Walks ``cls.__mro__`` reading each class's OWN ``__dict__`` (not ``getattr``, which would
            re-read the inherited value at every level) and unions the tuples. The result is sorted
            for a stable key ordering and memoised on the class under a private name, again in the
            class's own ``__dict__`` so a subclass does not inherit its parent's cached union.

        Args:
            cls: the functor class. No instance is needed -- the declaration is class-level.

        Returns:
            A sorted tuple of attribute names. Empty for a functor that declares none, which is the
            correct answer for a kernel with no post-construction configuration.
        """
        cached = cls.__dict__.get("_COMPILE_GATED_ATTRS_UNION")
        if cached is None:
            names: set = set()
            for klass in cls.__mro__:
                names |= set(klass.__dict__.get("COMPILE_GATED_ATTRS", ()))
            cached = tuple(sorted(names))
            cls._COMPILE_GATED_ATTRS_UNION = cached
        return cached

    @staticmethod
    def _require_compile_time(**kwargs) -> None:
        """Validate raw constructor arguments *before* deriving parameters from them.

        :meth:`_bind_params` already rejects runtime values, but a subclass that computes a derived
        parameter first -- ``cluster_n=_cluster_n_for(dtype, N)`` -- touches the raw argument
        before binding, and a tensor there fails with an opaque ``AttributeError`` from inside the
        derivation instead of the message that names the problem. Call this first when deriving.

        Args:
            **kwargs: Raw constructor arguments, by name.

        Returns:
            None.

        Raises:
            TypeError: If any argument is a runtime value, naming the offenders and their types.
        """
        bad = {k: type(v).__name__ for k, v in kwargs.items() if not is_compile_time_value(v)}
        if bad:
            raise TypeError(
                f"these constructor arguments are RUNTIME values, not compile-time constants: "
                f"{bad}. They cannot become template parameters -- a runtime value does not "
                f"survive a `self` stash across the @cute.jit -> @cute.kernel boundary. Pass them "
                f"as explicit kernel arguments instead."
            )

    def _bind_params(self, **kwargs) -> None:
        """Construct, validate and bind this functor's compile-time parameters. Call once.

        Args:
            **kwargs: One value per field of ``self.Params``, by keyword. A missing required field
                or an unknown name is a ``TypeError`` from the dataclass constructor, i.e. the
                parameter pack is checked for completeness as well as for type.

        Returns:
            None. Sets ``self.params`` and one attribute per declared field.

        Raises:
            TypeError: If a value is a runtime value (see :meth:`TemplateParams.__post_init__`), or
                if the keyword set does not match the declared fields.
            RuntimeError: If called twice. Rebinding would reintroduce exactly the two-phase
                configuration this class exists to prevent.
        """
        if "params" in self.__dict__:
            raise RuntimeError(
                f"{type(self).__name__}._bind_params() called twice. Compile-time parameters are "
                f"established once, in __init__; a value that must change per call is an argument, "
                f"not a parameter — pass it explicitly to the kernel."
            )
        params = self.Params(**kwargs)
        object.__setattr__(self, "params", params)
        for name in type(params).field_names():
            object.__setattr__(self, name, getattr(params, name))

    def _bind_call_params(self, **kwargs) -> None:
        """Bind the CALL-phase parameters -- the operand facts. Call once, at the top of ``__call__``.

        Purpose
            A kernel functor cannot know its operand dtypes, majors or layouts until it is handed
            real tensors, yet those ARE compile-time constants: ``__call__`` is traced, so every
            ``self.X`` read there is folded into the kernel. Binding them declaratively is what makes
            the functor's compile-time surface complete, and therefore what lets a cache key be
            derived from it instead of maintained beside it.

        Semantics
            Identical to :meth:`_bind_params` but for :attr:`CallParams`, with its own once-only
            guard. Once bound, the names are immutable exactly like construction parameters.

            Once-only is safe and is the right contract: one functor instance is traced exactly once
            (the compile entry builds a fresh functor per cache miss), and the compiled artifact is
            then invoked many times without re-entering this Python. A functor that tried to rebind
            would be one being reused across two different operand types -- which must be two
            compiled kernels, not one functor with a changing mind.

        Args:
            **kwargs: One value per field of :attr:`CallParams`, by keyword. A missing required field
                or an unknown name is a ``TypeError``, so the pack is checked for completeness as
                well as for type.

        Returns:
            None. Sets ``self.call_params`` and one attribute per declared field.

        Raises:
            TypeError: If a value is a runtime value -- a tensor, a layout, an atom. That is the
                common mistake here, because the operands ARE tensors: bind
                ``mA.element_type`` (a type, admissible) rather than ``mA``.
            RuntimeError: If called twice.
        """
        if "call_params" in self.__dict__:
            raise RuntimeError(
                f"{type(self).__name__}._bind_call_params() called twice. Call-phase parameters are "
                f"established once, at the top of __call__; one functor is traced once, so a second "
                f"binding means the instance is being reused for operands that need their own "
                f"compiled kernel."
            )
        params = self.CallParams(**kwargs)
        object.__setattr__(self, "call_params", params)
        for name in type(params).field_names():
            object.__setattr__(self, name, getattr(params, name))

    def __setattr__(self, name: str, value: Any) -> None:
        """Block writes to declared compile-time parameters of either phase; allow everything else.

        Args:
            name: Attribute being assigned.
            value: Proposed value.

        Raises:
            AttributeError: If ``name`` is a declared parameter and its phase has already been
                bound. The message points at the fix rather than just refusing.
        """
        if name in self.CallParams.field_names() and "call_params" in self.__dict__:
            raise AttributeError(
                f"{type(self).__name__}.{name} is a call-phase template parameter and is immutable "
                f"once bound. It is read off the operands at the top of __call__ and folded into "
                f"the kernel as a constant; reassigning it desynchronizes the functor from the "
                f"kernel compiled from it. Pass it to _bind_call_params(), or -- if it varies per "
                f"launch -- make it an explicit kernel argument."
            )
        if name in self.Params.field_names() and "params" in self.__dict__:
            raise AttributeError(
                f"{type(self).__name__}.{name} is a compile-time template parameter and is "
                f"immutable after construction. It is folded into the kernel as a constant, so "
                f"reassigning it cannot affect an already-compiled kernel and desynchronizes the "
                f"functor from it. Set it in __init__ via _bind_params(), or — if it varies per "
                f"call — pass it as an explicit kernel argument."
            )
        object.__setattr__(self, name, value)

    def compile_key(self) -> Dict[str, Any]:
        """The functor's ENTIRE compile-time surface: both phases, as one dict.

        Purpose
            This is the whole point of binding both phases declaratively. Every value here is folded
            into the compiled kernel, and nothing folded in is missing -- so a compile cache key
            built from this cannot drift from what the kernel was actually compiled with. A key
            maintained by hand beside the functor can, and when it does the failure is silent: two
            configurations share one artifact, so the kernel that runs is not the one that was
            selected.

        Returns:
            A new dict merging construction-phase parameters, call-phase parameters, and the
            post-construction gates declared in :attr:`COMPILE_GATED_ATTRS`. Empty for either phase
            that has not been bound yet, so calling this before ``__call__`` yields the construction
            parameters alone -- which is the correct key for "what would this functor compile to,
            given operands".

            A declared gate that is ABSENT on the instance contributes ``"<unset>"``. That sentinel
            is deliberate and load-bearing: a functor whose ``configure_a2a`` was never called is a
            DIFFERENT compile from one configured to the same value the ``getattr`` default would
            have produced, and defaulting the two to the same thing here would merge them.

        Raises:
            Nothing. An unbound phase contributes nothing rather than raising, because the partial
            key is meaningful.
        """
        merged = dict(self.param_dict())
        if "call_params" in self.__dict__:
            merged.update(dataclasses.asdict(self.call_params))
        # Declared gates FIRST, because their whole job is to key a name that may live on the CLASS
        # and therefore never appear in `__dict__` -- `getattr(self, "_a2a_ib_wide", False)` reads a
        # class default until some `configure_a2a` writes an instance one, and a sweep of `__dict__`
        # alone is blind to exactly that pre-configure state.
        for name in type(self).compile_gated_attrs():
            merged[name] = getattr(self, name, GATE_UNSET)
        # Then every OTHER instance attribute holding a compile-time value. This is the half that
        # does not need maintaining: `configure_a2a*` writes ~39 attributes on the A2A functor and
        # every one of them is read during tracing, so a hand-kept list is one new knob away from
        # being wrong. Non-keyable values (tensors, streams, cached derived objects) are dropped --
        # they are runtime state, and a value that slipped through would only ever cause an extra
        # MISS, never a false hit.
        for name in sorted(self.__dict__):
            if name in merged or name in ("params", "call_params"):
                continue
            # A ``cached_property`` stores its result straight into ``__dict__``, so the sweep sees
            # it. Skip it: it is DERIVED from values already in the key, so it cannot vary
            # independently of them, and admitting it would make the key claim a derived value is
            # part of the compile surface -- which is the distinction this paradigm rests on.
            if isinstance(getattr(type(self), name, None), (property, functools.cached_property)):
                continue
            value = self.__dict__[name]
            if is_key_component(value):
                merged[name] = value
        return merged

    def param_dict(self) -> Dict[str, Any]:
        """The CONSTRUCTION-phase parameters as a plain dict, for logging and test assertions.

        See :meth:`compile_key` for the full compile-time surface, which is what a cache key needs.

        Returns:
            A new dict of field name to value. Mutating it does not affect the functor.
        """
        return dataclasses.asdict(self.params) if "params" in self.__dict__ else {}

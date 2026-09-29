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

"""The tuner: picks one config per (kernel, shape), and calls the kernel with it.

Composition only -- the space, the timer, the consensus and the cache each own their own decision,
and this file sequences them. The sequence is the contract:

    candidates = space.candidates(request)        # pure function of the request -> same on every rank
    timings    = { c: policy.measure(...) }       # fixed-iteration, event-timed
    failures   = consensus.agree_failures(...)    # a config that failed ANYWHERE is dropped EVERYWHERE
    timings    = consensus.agree_timings(...)     # slowest rank's number, identical on every rank
    winner     = consensus.pick(timings)          # deterministic tie-break

Every step before ``pick`` produces something identical on every rank, so ``pick`` needs no
communication to agree -- which is what makes it correct even if a rank were to reach it early.

**Two pieces of upstream state are deliberately gone.** ``self.nargs``, set in ``__call__`` and read
by the pruning and benchmarking helpers, made the tuner non-re-entrant: a kernel that autotuned
another kernel, or two threads, would overwrite each other's arguments. It is replaced by an
explicit ``request`` dict threaded through. And the bare ``except Exception -> inf`` around each
candidate is replaced by an explicit failure set, because under a collective a swallowed exception
is a HANG on the peers, not a skipped config.
"""

import functools
import inspect
import os
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from fold_cp_ops._internal.autotune.axes import AxisSpace
from fold_cp_ops._internal.autotune.cache import ResultCache
from fold_cp_ops._internal.autotune.config import AutotuneConfig
from fold_cp_ops._internal.autotune.consensus import Consensus
from fold_cp_ops._internal.autotune.precompile import precompile_configs
from fold_cp_ops._internal.autotune.space import ConfigSpace
from fold_cp_ops._internal.autotune.timing import TimingPolicy


def _request_key(bound: Dict[str, Any], key_names: Sequence[str]) -> Tuple:
    """Build the shape/dtype identity a tuning result is cached against.

    Purpose
        Two calls share a tuned config exactly when they would tune to the same answer. That is a
        judgement about which argument properties affect the choice, and it is made HERE rather
        than by stringifying every argument.

    Semantics
        For each named key argument, the value itself. For every tensor argument, its shape, dtype
        and a CONTIGUITY CLASS rather than raw strides -- ``0`` and ``1`` are kept because they mean
        broadcast and unit stride, and anything else collapses to ``2``. That collapse is
        deliberate: a tuned tile does not depend on whether a row pitch is 4096 or 4104, and keying
        on it would make the cache miss on every new shape.

    Args:
        bound: The call's arguments by name.
        key_names: Which non-tensor arguments participate. Named explicitly, because most scalars
            (a stream, a flag) do not change the tuning answer and keying on them only fragments
            the cache.

    Returns:
        A hashable tuple, stable across processes.
    """
    import torch

    parts = [(name, bound[name]) for name in key_names if name in bound]
    for name, value in bound.items():
        if isinstance(value, torch.Tensor):
            strides = tuple(s if s in (0, 1) else 2 for s in value.stride())
            parts.append((name, tuple(value.shape), str(value.dtype), strides))
    return tuple(parts)


class Autotuner:
    """Wraps one kernel entry point, choosing its tunable knobs by measurement.

    Args:
        fn: The kernel entry to tune. Its signature is inspected to bind positional arguments by
            name, so the key and the validity rule can refer to arguments by name.
        space: The candidate configs, with validity and preference separated (`ConfigSpace`).
        key: Names of the non-tensor arguments that participate in the cache key. Tensor arguments
            contribute their shape/dtype/contiguity automatically.
        policy: How candidates are measured. Its ``collective`` flag is the one setting that must be
            right for a distributed kernel; see `TimingPolicy`.
        consensus: How ranks agree. Defaults to auto-detecting an initialized process group.
        cache: Optional on-disk result cache. Defaults to one named after `fn`, off unless
            on by default, disabled with ``CPO_AUTOTUNE_CACHE=0``.
        precompile: Whether to pre-compile candidates in worker subprocesses before timing. Costs
            process startup; saves the first-call compile of every candidate landing in the timed
            region. Ignored when a single candidate survives pruning.
        gate: Name of a KEYWORD-ONLY boolean parameter of `fn` that turns tuning on. When given and
            falsy at the call site, the call goes STRAIGHT to `fn` -- no binding, no request key, no
            candidate list. That is what lets one function be both the fixed-config entry and the
            tuned one: the fixed path costs a single dict lookup (measured free at a 14.5 us kernel),
            so a caller who knows its config, and every pinned perf cell, still measures a kernel
            rather than a sweep. Keyword-only so the check IS just ``kwargs.get`` -- a positionally
            passable gate would be invisible here and silently tune.

    Raises:
        RuntimeError: At call time, if the process was launched under a distributed launcher but no
            process group is initialized -- each rank would then tune independently with no
            consensus, and the divergence surfaces as a hang much later.
    """

    def __init__(
        self,
        fn: Callable,
        *,
        space: ConfigSpace,
        key: Sequence[str] = (),
        policy: Optional[TimingPolicy] = None,
        consensus: Optional[Consensus] = None,
        cache: Optional[ResultCache] = None,
        precompile: bool = True,
        gate: Optional[str] = None,
    ):
        self.fn = fn
        self.space = space
        self.key_names = tuple(key)
        self.policy = policy or TimingPolicy()
        self._consensus = consensus
        self.cache = cache if cache is not None else ResultCache(getattr(fn, "__name__", "kernel"))
        self.precompile = precompile
        self.gate = gate
        self._signature = inspect.signature(fn)
        self._check_knobs_are_parameters()
        self._check_gate_is_keyword_only()
        #: The tuned knob names, as ONE frozen set. Precomputed because the conflict check runs on
        #: every call: rebuilding it walked all N configs and copied each one's kwargs dict, which
        #: is ~N dict allocations per launch and was measurable (see :meth:`__call__`).
        self._knob_names = frozenset(k for c in space.configs for k in c.all_kwargs())
        #: Positional parameter names, in order, so the warm path can bind arguments to names with
        #: a `zip` instead of `inspect.Signature.bind_partial`. Same result for an ordinary call;
        #: `_bind` is still there for the general case.
        self._positional_names = tuple(self._signature.parameters)
        #: Whether any parameter is `*args` / `**kwargs`, which makes positional binding wrong.
        self._simple_signature = all(
            p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
            for p in self._signature.parameters.values()
        )
        #: (request key) -> winning config, for this process. Populated on first tune.
        self.results: Dict[Tuple, AutotuneConfig] = {}
        #: The most recent sweep's timings, for tests and diagnostics.
        self.last_timings: Dict[AutotuneConfig, float] = {}
        self.best_config: Optional[AutotuneConfig] = None

    def _check_knobs_are_parameters(self) -> None:
        """Refuse, at DECORATION time, a knob the kernel cannot accept.

        Purpose
            Without this the mistake surfaces at the first call as "every candidate config failed",
            because each candidate raises ``TypeError: unexpected keyword argument`` independently.
            That message is honest but points at the sweep rather than at the typo.

        Semantics
            Skipped entirely when the kernel takes ``**kwargs``, since then every name is acceptable
            and the check would be a false positive.

        Returns:
            None.

        Raises:
            TypeError: Naming the knobs that are not parameters of the kernel, and listing the
                parameters that are.
        """
        params = self._signature.parameters
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return
        knobs = {k for c in self.space.configs for k in c.all_kwargs()}
        unknown = sorted(knobs - set(params))
        if unknown:
            raise TypeError(
                f"@autotune declares knob(s) {unknown} that {self.fn.__name__} does not accept. "
                f"Its parameters are {sorted(params)}. Left unchecked this surfaces at the first "
                f"call as 'every candidate config failed', which points at the sweep rather than "
                f"at the name."
            )

    def _check_gate_is_keyword_only(self) -> None:
        """Refuse a gate that is not a keyword-only parameter of the kernel.

        Purpose
            The fast path is ``kwargs.get(self.gate)``. A gate that can be passed POSITIONALLY would
            be invisible to that lookup, so ``fn(x, True)`` would silently tune while reading as a
            request not to -- the exact inversion the flag exists to prevent.

        Returns:
            None. No-op when no gate was declared.

        Raises:
            TypeError: If the gate is not a parameter at all, or is not keyword-only.
        """
        if self.gate is None:
            return
        param = self._signature.parameters.get(self.gate)
        if param is None:
            raise TypeError(
                f"@autotune(gate={self.gate!r}) but {self.fn.__name__} has no such parameter. Its "
                f"parameters are {sorted(self._signature.parameters)}."
            )
        if param.kind is not inspect.Parameter.KEYWORD_ONLY:
            raise TypeError(
                f"@autotune(gate={self.gate!r}) must be KEYWORD-ONLY on {self.fn.__name__} (put a "
                f"bare `*` before it). The dispatch check is a `kwargs` lookup, so a positionally "
                f"passable gate would be invisible to it and the call would tune while the call "
                f"site reads as asking it not to."
            )

    @property
    def consensus(self) -> Consensus:
        """The rank-agreement helper, constructed lazily.

        Lazy because a process group is typically initialized AFTER import, so binding one at
        decoration time would capture "no group" for the process's whole life.
        """
        if self._consensus is None:
            self._consensus = Consensus()
        return self._consensus

    def _bind(self, args: Tuple, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Map a call's positional and keyword arguments to names.

        Args:
            args: Positional arguments.
            kwargs: Keyword arguments.

        Semantics
            Takes the fast path -- a `zip` against the precomputed parameter-name tuple -- whenever
            the signature has no ``*args``/``**kwargs``, which is every kernel entry here.
            `inspect.Signature.bind_partial` is correct for the general case but costs several
            microseconds per call, and this runs on EVERY dispatch, including the warm one where the
            config is already known. On a launch-bound shape that was a measurable fraction of the
            kernel itself.

        Returns:
            A dict of every argument the signature could bind. Unbindable extras are ignored rather
            than raising -- the kernel itself will complain about them with a better message.
        """
        if self._simple_signature and len(args) <= len(self._positional_names):
            bound = dict(zip(self._positional_names, args))
            bound.update(kwargs)
            return bound
        try:
            bound = self._signature.bind_partial(*args, **kwargs)
            return dict(bound.arguments)
        except TypeError:
            return dict(kwargs)

    def _sweep(self, args, kwargs, candidates) -> Dict[AutotuneConfig, float]:
        """Measure every candidate, reconciling failures and timings across ranks.

        Semantics
            Collective when a process group is initialized: every rank measures the same candidate
            list in the same order, then the failures are unioned and the timings max-reduced. The
            order matters -- a rank that measured a different sequence would have a different
            thermal history, and comparing across ranks would be comparing different experiments.

        Args:
            args: The kernel's positional arguments.
            kwargs: The kernel's keyword arguments.
            candidates: The configs to time, identical on every rank.

        Returns:
            Config to milliseconds, reconciled. Configs that failed anywhere are absent entirely
            rather than present with ``inf`` -- an absent entry cannot accidentally win, and the
            count of survivors is then meaningful.
        """
        verbose = TimingPolicy.verbose()
        raw: Dict[AutotuneConfig, float] = {}
        if self.precompile:
            precompile_configs(self.fn, args, kwargs, candidates, verbose=verbose)
        self.policy.thermal_warmup(self.consensus.barrier if self.consensus.enabled else None)
        for config in candidates:
            merged = dict(kwargs, **config.all_kwargs())
            try:
                self.fn(*args, **merged)  # compile + correctness warmup, outside the timed region
                raw[config] = self.policy.measure(lambda: self.fn(*args, **merged))
            except Exception as exc:
                # inf, NOT a skip: every rank must still reach the reduction below, or the ranks
                # where this config succeeded block forever waiting for the one that raised.
                raw[config] = float("inf")
                if verbose:
                    print(
                        f"[autotune] {self.fn.__name__} {config} FAILED: "
                        f"{type(exc).__name__}: {exc}"
                    )
        # One collective, doing two jobs: MAX gives the slowest PE's time, and propagates the inf of
        # anything that failed anywhere so it is dropped everywhere.
        timings = self.consensus.agree_timings(raw)
        diverged = self.consensus.divergent_failures()
        if diverged:
            raise RuntimeError(
                f"autotuning {self.fn.__name__}: {len(diverged)} config(s) failed on SOME ranks but "
                f"not all -- {[str(c) for c in diverged]}. Autotuning is SPMD, so this is a setup "
                f"bug, not a tuning outcome: the usual causes are uneven local shards or one rank "
                f"running out of memory. Re-run with CPO_AUTOTUNE_VERBOSE=1 for each rank's error."
            )
        if not timings:
            raise RuntimeError(
                f"every candidate config for {self.fn.__name__} failed to run "
                f"({len(candidates)} tried). Re-run with CPO_AUTOTUNE_VERBOSE=1 for each error."
            )
        if verbose and self.consensus.rank == 0:
            for config, ms in sorted(timings.items(), key=lambda kv: kv[1]):
                print(f"[autotune] {self.fn.__name__} {config} -> {ms:.4f} ms")
        return timings

    def __call__(self, *args, _config: Optional[AutotuneConfig] = None, **kwargs):
        """Tune if needed, then call the kernel with the winning config.

        Args:
            *args: The kernel's positional arguments.
            _config: A frozen config that REPLACES tuning entirely -- no measurement, no cache
                lookup, no collective. This is how a deployment pins a tuned result: autotuning
                is a development-time activity, and re-measuring in every fresh process is both
                slow and non-deterministic. Obtain one from ``wrapper.autotuner.best_config``.
                Passing the knobs as ordinary keyword arguments instead is refused, because a
                caller overriding ONE knob would make the measured winner and the executed
                kernel differ.
            **kwargs: The kernel's keyword arguments. Must NOT include a tuned knob -- that is a
                conflict, not an override, and is refused.

        Returns:
            Whatever the kernel returns.

        Raises:
            ValueError: If a keyword argument collides with a tuned knob.
            RuntimeError: If launched distributed with no initialized process group, or if every
                candidate failed.
        """
        # THE FIXED PATH, and it is deliberately the first thing here. One dict lookup, then
        # straight to the kernel: no argument binding, no request key, no candidate list, not even
        # the distributed check below. That is what lets ONE function be both the fixed-config entry
        # and the tuned one -- which is the whole reason the gate exists, because three entry points
        # for one kernel is three places for a caller to reach the wrong one.
        if self.gate is not None and not kwargs.get(self.gate):
            return self.fn(*args, **kwargs)
        if Consensus.is_distributed_env() and not self.consensus.enabled:
            raise RuntimeError(
                f"autotuning {self.fn.__name__} under a distributed launcher (RANK/WORLD_SIZE are "
                f"set) before torch.distributed is initialized. Each rank would measure and choose "
                f"independently, with no guarantee they agree -- and a kernel containing a "
                f"collective then deadlocks, minutes later, with nothing pointing back here. "
                f"Initialize the process group first, or pass an explicit config."
            )
        if _config is not None:
            self.best_config = _config
            return self.fn(*args, **kwargs, **_config.all_kwargs())
        request = self._bind(args, kwargs)
        # Against the BOUND request, not against `kwargs`. A knob can be passed POSITIONALLY -- and
        # for a kernel whose tile shape is the 6th argument, it usually is. Checking only keywords
        # let `gemm(A, B, D, C, sem, 128, 128, 1, 1, do_autotune=True)` through, and it died several
        # frames later as `TypeError: got multiple values for argument 'tile_M'`, which names the
        # symptom and not the mistake.
        conflicts = self._knob_names.intersection(request)
        if conflicts:
            raise ValueError(
                f"these arguments are tuned knobs and cannot also be passed by the caller: "
                f"{sorted(conflicts)}. Pass a frozen config instead of overriding one knob, or the "
                f"measured winner and the executed kernel would differ."
            )

        # WARM PATH. Everything expensive -- running `validity` over the whole pool, consulting the
        # disk cache, measuring -- is behind the `self.results` lookup, so a repeat call at a known
        # shape costs one key build and one dict hit. It did not used to be: `candidates()` ran the
        # validity rule over every config on EVERY dispatch, which on a launch-bound shape cost
        # more than the kernel. The tuned entry is a production path, so its steady-state overhead
        # is a correctness-of-measurement issue, not just a tuning-time one.
        if not enabled():
            # `enabled()` off -> the FIRST admissible config, which is the author's declaration
            # order and therefore a deterministic default rather than an arbitrary one. This is the
            # path a correctness run and a pinned perf gate take: neither may have the tuner
            # re-measuring underneath it.
            config = self.space.candidates(request)[0]
        else:
            rkey = _request_key(request, self.key_names)
            config = self.results.get(rkey)
            if config is None:
                candidates = self.space.candidates(request)
                if len(candidates) == 1:
                    config = candidates[0]
                else:
                    timings = self.cache.load(rkey, candidates)
                    if timings is None:
                        dropped = self.space.truncated(request)
                        if dropped and TimingPolicy.verbose():
                            print(
                                f"[autotune] {self.fn.__name__}: timing {len(candidates)} of "
                                f"{len(candidates) + dropped} admissible configs (top_k)"
                            )
                        timings = self._sweep(args, kwargs, candidates)
                        self.cache.store(rkey, timings, is_writer=self.consensus.rank == 0)
                    self.last_timings = timings
                    config = Consensus.pick(timings)
                self.results[rkey] = config

        self.best_config = config
        return self.fn(*args, **kwargs, **config.all_kwargs())


def autotune(
    *,
    configs: Optional[Sequence[AutotuneConfig]] = None,
    space: Optional[AxisSpace] = None,
    key: Sequence[str] = (),
    validity: Optional[Callable[[AutotuneConfig, Dict[str, Any]], bool]] = None,
    prefer: Optional[Callable[[AutotuneConfig, Dict[str, Any]], Any]] = None,
    top_k: Optional[int] = None,
    collective: bool = False,
    measure: Optional[Callable[..., float]] = None,
    precompile: bool = True,
    gate: Optional[str] = None,
):
    """Decorate a kernel entry point so its tunable knobs are chosen by measurement.

    Args:
        space: The DECLARED axes (`AxisSpace`), which generate the pool. Preferred over
            `configs=`: it is what makes the tunable axes readable from the kernel code rather
            than recoverable only by scraping a flat list. Mutually exclusive with `configs`.
        configs: The candidate pool as explicit points. The low-level form, kept because a
            genuinely sparse or joint space is sometimes clearest enumerated. Non-empty, no
            duplicates. Mutually exclusive with `space`.
        key: Names of non-tensor arguments that participate in the cache key. Tensor arguments
            contribute shape/dtype/contiguity automatically, so list only the scalars that change
            the answer.
        validity: ``(config, request) -> bool``; False means the config CANNOT run. Must be a pure
            function of the request -- see `ConfigSpace`, and note that a non-pure one is a HANG
            under a collective rather than a wrong answer.
        prefer: ``(config, request) -> sort key``; orders candidates so `top_k` keeps the promising
            ones. May not reject.
        top_k: Cap on how many admissible configs are timed. None times all.
        collective: **Set this True for any kernel containing an in-kernel NVSHMEM or NCCL
            collective.** It selects fixed-iteration, MAX-reduced timing; leaving it False measures
            each rank's own view, and the winner is whichever rank waited least.
        measure: Optional injected timer, mainly for tests that must run without a GPU. The
            default is the packaged ``bench_timing.benchmark_single``, which is always available.
        precompile: Pre-compile candidates in worker subprocesses before timing.
        gate: Name of a KEYWORD-ONLY boolean parameter of `fn` that turns tuning on. Given one, the
            decorated function IS the kernel's single entry point: falsy gate -> the caller's own
            config runs with one dict lookup of overhead; truthy -> the sweep. Without it the
            decorated function always tunes, which is the right shape for a wrapper whose only job
            is tuning.

    Returns:
        A decorator producing a callable that tunes on first use per shape and then dispatches. The
        wrapper exposes ``.autotuner`` and ``.axes``, and **nothing else** -- in particular the
        winning config is ``wrapper.autotuner.best_config``, NOT ``wrapper.best_config``. Reading it
        off the wrapper is an ``AttributeError`` on the first call, and an entry point no test
        exercises will ship that way: this line previously claimed the wrapper carried
        ``.best_config``, and a caller who believed it shipped exactly that bug past a green suite
        and a 190-cell identity sweep, because nothing called the function that read it.
    """

    if (space is None) == (configs is None):
        raise ValueError(
            "@autotune takes exactly one of `space=` (declared axes, preferred) or `configs=` "
            "(explicit points). Giving both leaves two sources of truth for the pool; giving "
            "neither leaves the kernel with nothing to try."
        )
    pool = space.configs() if space is not None else list(configs)

    def decorator(fn):
        tuner = Autotuner(
            fn,
            space=ConfigSpace(pool, validity=validity, prefer=prefer, top_k=top_k),
            key=key,
            policy=TimingPolicy(collective=collective, measure=measure),
            precompile=precompile,
            gate=gate,
        )

        if gate is None:

            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                return tuner(*args, **kwargs)

        else:
            # The gated form gets its OWN wrapper so the fixed path is ONE frame that calls `fn`
            # directly. Routing it through `Autotuner.__call__` and checking there costs a second
            # frame plus a `**kwargs` repack, which measured +2.3 us on a call with eight keyword
            # arguments -- enough to push a 21 us pinned perf cell past its 10% band. The duplicated
            # check in `Autotuner.__call__` stays for callers holding the tuner directly.
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                if not kwargs.get(gate):
                    return fn(*args, **kwargs)
                return tuner(*args, **kwargs)

        wrapper.autotuner = tuner
        wrapper.axes = space  # None when declared as explicit points
        return wrapper

    return decorator


def freeze(config: AutotuneConfig) -> Dict[str, Any]:
    """The knobs of a chosen config, as plain keyword arguments for a NON-tuned entry point.

    Purpose
        For calling the underlying kernel directly, bypassing the decorated wrapper entirely.

    Warning:
        Do NOT pass these to the decorated wrapper -- it refuses tuned knobs as keyword arguments,
        because a caller overriding one knob would make the measured winner and the executed kernel
        differ. To pin a config on the wrapper, use ``wrapper(..., _config=config)``, which replaces
        tuning rather than merging with it.

    Args:
        config: A config, typically read from ``wrapper.autotuner.best_config`` after a tuning run.

    Returns:
        The knobs as keyword arguments.
    """
    return config.all_kwargs()


#: Whether autotuning is permitted at all in this process. A correctness run pins configs and must
#: not have the tuner re-measuring underneath it; a benchmark harness sets this off so a cell's
#: timing is the kernel's, not a sweep's.
def enabled() -> bool:
    """Whether autotuning may run. Reads ``CPO_AUTOTUNE`` (default on).

    Returns:
        False when ``CPO_AUTOTUNE=0``, in which case a decorated kernel runs its FIRST admissible
        config rather than measuring. That is a deterministic fallback, not an arbitrary one: the
        pool's declaration order is the author's default.
    """
    return os.environ.get("CPO_AUTOTUNE", "1") != "0"

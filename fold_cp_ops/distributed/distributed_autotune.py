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

"""Kernel-agnostic distributed (torchrun + nvshmem + CuTe-DSL) config autotuner.

The facility is kernel-agnostic; a per-kernel **adapter** (:class:`DistributedAutotuneAdapter`)
supplies the pruned config grid, a compile-at-config phase folded OUT of the timed loop, a thin
``launch``, and a correctness gate.

Autotuning a kernel that contains a collective is a different problem from autotuning a local one,
and the three ways it goes wrong are all silent: ranks can pick DIFFERENT winners from noise-ordered
ties, a config that raises on one rank can stay in contention on the others, and a rank that finishes
its share of a collective early records its PEERS' slowness as its own speed. All three surface much
later as a hang with nothing pointing back at the sweep.

**What this module is, relative to the upstream.** The upstream carried its own answers to those
three problems inline -- a hand-rolled fixed-N barrier-bracketed event loop
(``make_distributed_do_bench``) and an ``all_reduce(MAX)`` folded into it. This tree had already
restructured the autotuner into :mod:`fold_cp_ops._internal.autotune`, where those same two answers
live as tested components:

===================================  ==================================================
upstream inline piece                the component here
===================================  ==================================================
``make_distributed_do_bench``        ``TimingPolicy(collective=True)`` -- CUDA events, FIXED
                                     iterations, ``reduce="max"``, barrier-bracketed thermal
                                     warmup, and an explicit refusal of the adaptive mode
``all_reduce(MAX)`` inside the timer ``Consensus.agree_timings`` -- max across ranks, plus
                                     ``inf``-propagation so a config failing anywhere is dropped
                                     everywhere, plus ``divergent_failures`` reporting
``min(timings)``                     ``Consensus.pick`` -- deterministic tie-break on the config's
                                     canonical key, not on dict order
===================================  ==================================================

So this module ports the two pieces that genuinely had no counterpart -- the adapter Protocol and the
wiring -- and composes the rest rather than duplicating it. Re-porting the inline versions would have
produced a second implementation of a solved problem, which passes its own tests perfectly while the
one the kernels actually use goes unexercised.

:func:`consensus_assert` IS carried across: the reduction already makes the pick deterministic, but
this is the check at the point of USE, one broadcast before a possibly-divergent kernel is launched
into a collective. Cheap, and the failure it catches is otherwise a timeout with no attribution.

**Where the per-kernel adapters live: under ``benchmark/``, and that is settled by measurement.**
An earlier reading assumed a shipped adapter could not live outside the installed package. It can,
because nothing in the package ever imports one: :func:`build_distributed_autotuner` takes the
adapter as an ARGUMENT. Checked against the upstream tree -- the only importers of its
``back_a2a_autotune_adapter`` are a bench driver and two test modules, while the shipped workflow
autotune half builds its adapter inline.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Protocol, Sequence, runtime_checkable

import msgpack
import torch
import torch.distributed as dist

from fold_cp_ops._internal.autotune import (
    Autotuner,
    AutotuneConfig,
    ConfigSpace,
    Consensus,
    ResultCache,
    TimingPolicy,
)
from fold_cp_ops._internal.autotune import timing as _timing
from fold_cp_ops._internal.compile_time.template_params import (
    UnsupportedKeyComponent,
    canonical_key_bytes,
)

__all__ = [
    "DistributedAutotuneAdapter",
    "build_distributed_autotuner",
    "consensus_assert",
    "make_collective_timing_policy",
]


def _bind_counts(n_iters: int, n_warmup: int) -> Callable[..., float]:
    """Return a ``measure`` callable that substitutes fixed iteration counts.

    Purpose
        Let a caller set the upstream's ``n_iters`` / ``warmup`` without reaching into
        `TimingPolicy`'s packaged constants, which every other kernel shares.

    Semantics
        `TimingPolicy.measure` calls this with its own ``rounds`` / ``warmup`` / ``iters`` /
        ``reduce`` and, because this signature names ``mode``, with ``mode`` too. Only the two
        iteration counts are replaced; ``rounds`` (the median-over-rounds behaviour), ``reduce``
        (the slowest-PE ``all_reduce``) and ``mode`` are forwarded untouched, so the measurement is
        the packaged one at a different repetition count rather than a second timer.

    Args:
        n_iters: Back-to-back launches inside one event window. Must be identical on every rank --
            it is a constant, not derived from a measurement, which is what keeps ranks in lockstep.
        n_warmup: Untimed rounds before the first timed one.

    Returns:
        A callable with `_default_measure`'s keyword interface, returning median milliseconds.
    """

    def _measure(fn, *, rounds, warmup: int, iters: int, reduce, mode: str = "event") -> float:
        # Resolved through the MODULE, not bound at import. A `from ... import _default_measure`
        # would freeze whichever object existed when this module first loaded, so a test double
        # installed on the timing module would be silently bypassed -- and the test would then be
        # measuring the packaged primitive while believing it measured the override.
        return _timing._default_measure(
            fn, rounds=rounds, warmup=n_warmup, iters=n_iters, reduce=reduce, mode=mode
        )

    return _measure


def make_collective_timing_policy(*, n_iters: int = 50, warmup: int = 10) -> TimingPolicy:
    """A :class:`TimingPolicy` for a kernel containing a collective, at an explicit iteration count.

    Purpose
        Give the distributed autotuner the upstream's ``n_iters`` / ``warmup`` knobs without
        reopening the one setting that must never be wrong.

    Functionality & semantics
        ``collective=True`` is what selects CUDA-event timing with a FIXED iteration count, an
        ``all_reduce(MAX)`` for the slowest PE, and a barrier-bracketed thermal warmup. The adaptive
        per-rank repetition count is refused by the policy itself: ranks running different numbers of
        in-kernel puts desynchronize into a hang rather than into a slow number. Only the iteration
        counts are overridden here (see :func:`_bind_counts`).

    Args:
        n_iters: Launches inside one timed window. Must be >= 1; a zero-length window measures
            nothing and divides by zero downstream.
        warmup: Untimed rounds before the first timed one. Must be >= 0.

    Returns:
        A `TimingPolicy` with ``collective=True``.

    Raises:
        ValueError: If ``n_iters < 1`` or ``warmup < 0``.
    """
    if n_iters < 1:
        raise ValueError(
            f"n_iters={n_iters} must be >= 1; a zero-length timed window measures nothing."
        )
    if warmup < 0:
        raise ValueError(f"warmup={warmup} must be >= 0.")
    return TimingPolicy(collective=True, measure=_bind_counts(n_iters, warmup))


@runtime_checkable
class DistributedAutotuneAdapter(Protocol):
    """Per-kernel adapter contract. The facility is kernel-agnostic; the adapter knows the kernel.

    Purpose
        Separate "how ranks agree about a measurement" (this module) from "what this kernel's
        candidate configs are and how to run one" (the adapter). One facility, N kernels.

    Input requirements -- every one of these has a collective consequence if it is wrong:

    ``key_names``
        Names of the non-tensor arguments whose change re-triggers tuning. Must name real
        parameters of ``launch``; `Autotuner` refuses an unknown one at construction.
    ``key_tensors``
        The positional tensors handed to the tuner, contributing shape/dtype to the cache key.
    ``fixed_kwargs``
        Non-config call kwargs (e.g. ``cp``). MUST be disjoint from the config knob names, or a
        candidate's value and a fixed value collide and one silently wins.
    ``configs()``
        The PRUNED valid grid. **Must be built identically, in the same ORDER, on every rank.** A
        rank-dependent grid (a device query, an env var, a wall clock) makes ranks disagree about
        the candidate list, and under a collective that is a hang, not a wrong answer.
    ``precompile(dm)``
        Compile every grid config up front into an internal map, so compile is folded OUT of the
        timed loop. Collective-free, but called inside a barrier bracket to keep ranks in step.
    ``launch(*tensors, **config_kwargs)``
        The tuner's ``fn``: look up the pre-compiled runner and launch it. Ignores tensor VALUES --
        it is timed, not checked.
    ``correctness(config_kwargs, dm)``
        Run one config and return its error against the oracle. Not called by this module; the
        caller decides when a numeric gate runs.
    ``free()``
        Release compiled runners and symmetric buffers.
    """

    key_names: list[str]
    key_tensors: tuple
    fixed_kwargs: dict

    def configs(self) -> Sequence[AutotuneConfig]: ...
    def precompile(self, dm) -> None: ...
    def launch(self, *tensors, **config_kwargs) -> Any: ...
    def correctness(self, config_kwargs: dict, dm) -> float: ...
    def free(self) -> None: ...


def _canonical_config(kwargs: dict) -> bytes:
    """Canonical, type-tagged bytes for one config's knobs, for comparing two ranks' picks.

    Purpose
        Make "did the ranks pick the same config?" a question about VALUES AND TYPES, which
        ``dict.__eq__`` cannot answer.

    Functionality & semantics
        Sorts the ``(name, value)`` pairs by name -- so two ranks that built the same pack in
        different insertion orders compare equal -- and encodes the result with
        `template_params.canonical_key_bytes`, which tags every scalar and recurses into tuples.

        Sorting by NAME only, not by the pair: knob names are unique within a config, so no tie can
        reach the values, and values of mixed types are not mutually orderable.

    Args:
        kwargs: A config's ``all_kwargs()``, or a decoded payload already checked to be a
            string-keyed mapping.

    Returns:
        The encoded bytes. Compared for equality; never decoded.

    Raises:
        UnsupportedKeyComponent: If a knob's value is outside the encodable domain. The caller
            converts this into a consensus failure naming both configs, because a pick that cannot
            be compared must not be launched.
    """
    return canonical_key_bytes(tuple(sorted(kwargs.items(), key=lambda kv: kv[0])))


def consensus_assert(best_config: AutotuneConfig, dm, group=None) -> None:
    """Broadcast rank-0's picked config and assert every rank matches.

    Purpose
        Belt and braces at the point of USE. `Consensus.agree_timings` already makes every rank see
        identical numbers and `Consensus.pick` breaks ties on the config's canonical key, so the
        pick is deterministic -- but this is the last moment before a possibly-divergent kernel is
        launched into a collective, and the failure it catches is otherwise a timeout minutes later
        with nothing pointing back at the sweep.

    Functionality & semantics
        COLLECTIVE: every rank must reach it or the ones that do block in the broadcast. Two
        broadcasts (length, then bytes) because the MessagePack payload's size is not known to the
        receivers. Inert -- an immediate return -- when there is no group to broadcast on, so a
        single-process caller needs no separate path.

        **THE CODEC IS MessagePack, AND THAT BOUNDS WHAT A KNOB MAY HOLD.** A config pack crosses
        the wire as ``msgpack.dumps(..., use_bin_type=True)`` and returns through
        ``msgpack.loads(..., raw=False, use_list=False, strict_map_key=True)``. No object hook, no
        extension hook: the decoder can therefore construct nothing but the primitive types below,
        which is the point -- ``pickle.loads`` on a broadcast buffer is arbitrary-code execution on
        whatever rank 0 sent, and the two-broadcast protocol around it is unchanged precisely so
        the substitution is auditable as a codec swap and nothing else.

        Representable, and round-tripping to the SAME type: ``int`` (64-bit), ``float``, ``bool``,
        ``str``, ``bytes``, ``None``, ``tuple`` (``use_list=False``), and dicts with ``str`` keys --
        which covers every knob any ``AutotuneConfig`` in this repo is built with. Two edges are
        real and are NOT hidden: a ``list`` knob comes back a ``tuple`` and so fails the equality
        check on EVERY rank including rank 0 (loud, never silent -- the tuple case is the one
        preserved, since ``cluster=(1, 2)`` is a knob this repo actually uses), and a value outside
        the set above -- a ``set``, an ``Enum``, a ``torch.dtype``, an int past 64 bits -- raises
        out of ``dumps`` where ``pickle`` would have carried it.

        **THE GROUP IS EXPLICIT AND THE DEVICE FOLLOWS FROM IT.** Both were implicit and they were
        two sources of truth for one fact: the buffers took their device from ``dm`` while the
        collective went to whatever the DEFAULT group happened to be, and nothing checked the two
        agreed. Measured -- a run of one test FILE passes because that file's fixture makes the
        default group ``gloo`` and the buffers are CPU; the same code in a run of the whole
        DIRECTORY hits a ``nccl`` default left by a sibling's session fixture and dies with
        ``RuntimeError: No backend type associated with device type cpu``, then a C++
        ``DistBackendError`` and SIGABRT from the NCCL watchdog. Same code, opposite outcome, decided
        by what ran earlier in the process.

    Args:
        best_config: This rank's winning config. Compared by ``all_kwargs()``, not by identity:
            configs are rebuilt per rank and would never compare equal by address. Every knob it
            carries must be MessagePack-representable (see above); one that is not raises out of
            the encoder on every rank at once rather than diverging silently.
        dm: The distributed manager. ``.rank`` is read for the message, and ``.device`` ONLY when
            the group's backend is NCCL -- see ``group``. Any object carrying those works, which is
            what lets the unit drive this without a real manager.
        group: The process group to broadcast on. ``None`` means the default group, and is inert
            when no default exists. Pass one EXPLICITLY whenever the caller built a group of its own:
            a collective that hardcodes the default cannot be pointed at the group whose backend its
            buffers match, which is the entire defect above.

    Raises:
        RuntimeError: If this rank's pick differs from rank 0's (naming both), or if rank 0's
            payload is not a string-keyed mapping, or is not a constructible ``AutotuneConfig``, or
            carries a knob outside the comparable domain. **A ``raise``, never an ``assert``**:
            ``python -O`` strips assertions, and a divergence check that vanishes under an
            optimization flag is worse than none because nothing announces its absence. The
            comparison is on the canonical TYPE-TAGGED encoding, so it fires for a type difference
            at any depth -- ``True`` vs ``1``, a tuple vs a list -- which ``==`` treats as equal.
        TypeError, OverflowError, ValueError: From the MessagePack codec when a knob falls outside
            the representable set above -- respectively an unsupported type, an int past 64 bits,
            and a non-``str`` key in a nested map. Deterministic and rank-uniform: every rank
            encodes the same pack, so this is a refusal, not a divergence.
        RuntimeError: If the group's backend is NCCL and ``dm.device`` is not a CUDA device. That
            combination cannot work and its native failure is a C++ abort several frames down, so it
            is refused here where both values can be named.
    """
    if not dist.is_available():
        return
    if group is None:
        if not dist.is_initialized():
            return
        group = dist.group.WORLD
    # DEVICE FROM THE BACKEND, not from the manager. `dm.device` is consulted only where it can be
    # right -- a NCCL group needs CUDA buffers, and every other backend here takes CPU ones.
    backend = str(dist.get_backend(group)).lower()
    if "nccl" in backend:
        device = getattr(dm, "device", None)
        if device is None or torch.device(device).type != "cuda":
            raise RuntimeError(
                f"consensus_assert: the group's backend is {backend!r}, which requires CUDA "
                f"buffers, but dm.device is {device!r}. Pass the group whose backend matches the "
                f"manager's device (group=), or hand in a manager on a CUDA device. Left to itself "
                f"this is a C++ DistBackendError and a SIGABRT from the NCCL watchdog."
            )
    else:
        device = torch.device("cpu")
    payload = msgpack.dumps(best_config.all_kwargs(), use_bin_type=True)
    buf = torch.frombuffer(bytearray(payload), dtype=torch.uint8).to(device)
    length = torch.tensor([buf.numel()], device=device, dtype=torch.int64)
    dist.broadcast(length, src=0, group=group)
    if buf.numel() != int(length.item()):
        buf = torch.empty(int(length.item()), device=device, dtype=torch.uint8)
    dist.broadcast(buf, src=0, group=group)
    # No object/extension hook, so the decoder can build only primitives -- the whole point of the
    # swap. `use_list=False` keeps a tuple knob (`cluster=(1, 2)`) a tuple, so the comparison below
    # stays a comparison of picks rather than of container types.
    rank0_cfg = msgpack.loads(
        bytes(buf.cpu().numpy()), raw=False, use_list=False, strict_map_key=True
    )
    rank = getattr(dm, "rank", "?")

    # SHAPE. The decoder can emit any MessagePack type, so "it decoded" is not "it is a config".
    if not isinstance(rank0_cfg, dict):
        raise RuntimeError(
            f"distributed-autotune CONSENSUS FAILURE on rank {rank}: rank 0 sent a "
            f"{type(rank0_cfg).__name__}, not a config mapping. The broadcast payload is malformed; "
            f"every rank must abort rather than launch against a config it cannot read."
        )
    bad_keys = [k for k in rank0_cfg if not isinstance(k, str)]
    if bad_keys:
        raise RuntimeError(
            f"distributed-autotune CONSENSUS FAILURE on rank {rank}: rank 0's config has "
            f"non-string knob names {bad_keys!r}; a config's keys are keyword-argument names."
        )

    # CONSTRUCTIBILITY. Building the config and asking for its key runs the same compile-time-value
    # validation any locally built candidate gets, so a payload that is shaped like a config but
    # carries a value no kernel could be traced with is refused HERE -- where it can be named --
    # rather than several frames into a launch.
    try:
        AutotuneConfig(**rank0_cfg).key()
    except Exception as exc:
        raise RuntimeError(
            f"distributed-autotune CONSENSUS FAILURE on rank {rank}: rank 0's config "
            f"{rank0_cfg!r} is not a constructible AutotuneConfig ({type(exc).__name__}: {exc})."
        ) from exc

    # AGREEMENT, compared on the CANONICAL TYPE-TAGGED encoding rather than with ``==``.
    #
    # Dict equality is not sufficient here and the reason is a real defect class, not pedantry:
    # ``True == 1 == 1.0`` in Python, so ``{"pingpong": True} == {"pingpong": 1}`` -- two ranks
    # holding different types for one knob would agree, then trace two different kernels and
    # deadlock in the next collective. The normalizer tags every scalar and recurses into tuples,
    # so a type difference at ANY depth separates the encodings.
    #
    # And it is an explicit ``raise``, never ``assert``: ``python -O`` strips assertions, and this
    # check exists precisely to prevent a divergent launch -- a safety check that disappears under
    # an optimization flag is worse than none, because nothing announces its absence.
    local_kwargs = best_config.all_kwargs()
    try:
        local_canon = _canonical_config(local_kwargs)
        remote_canon = _canonical_config(rank0_cfg)
    except UnsupportedKeyComponent as exc:
        raise RuntimeError(
            f"distributed-autotune CONSENSUS FAILURE on rank {rank}: a knob is outside the "
            f"comparable domain ({exc}). local={local_kwargs!r} rank0={rank0_cfg!r}"
        ) from exc
    if local_canon != remote_canon:
        raise RuntimeError(
            f"distributed-autotune CONSENSUS FAILURE on rank {rank}: "
            f"local pick {local_kwargs!r} != rank-0 pick {rank0_cfg!r}. Compared on the canonical "
            f"type-tagged encoding, so this fires for a value difference AND for a type difference "
            f"at any depth (True vs 1, a tuple vs a list) that ``==`` would have missed."
        )


def build_distributed_autotuner(
    adapter: DistributedAutotuneAdapter,
    dm,
    *,
    n_iters: int = 50,
    warmup: int = 10,
    cache: Optional[ResultCache] = None,
    policy: Optional[TimingPolicy] = None,
    group=None,
):
    """Wire ``adapter`` into the packaged `Autotuner` with the collective timer and consensus.

    Purpose
        The one place a distributed sweep is assembled, so "the autotuner and the perf gates measure
        the same way" stays a fact about a call site rather than a convention.

    Functionality & semantics
        Returns a zero-argument ``run()`` which: (1) barrier, (2) ``adapter.precompile(dm)``, (3)
        barrier, (4) invokes the tuner over the grid, (5) :func:`consensus_assert`, (6) returns
        ``(best_config, autotuner)``.

        The two barriers bracket a phase that is itself collective-free. They are there so a slow
        compiler on one rank cannot make its peers' first timed config look expensive -- compile
        skew inside the timed region is measured as kernel cost.

        ``precompile=False`` on the tuner is deliberate and is NOT the same knob as
        ``adapter.precompile``: the tuner's own pre-compilation spawns worker SUBPROCESSES, which
        under torchrun would be one set per rank. The adapter compiles in-process instead.

    Args:
        adapter: Anything satisfying :class:`DistributedAutotuneAdapter`.
        dm: The distributed manager. ``.device`` and ``.local_rank`` are read here; the barrier is
            issued with ``device_ids=[local_rank]``.
        n_iters: Launches inside one timed window; forwarded to
            :func:`make_collective_timing_policy`.
        warmup: Untimed rounds before the first timed one.
        cache: On-disk result cache. **Defaults to a DISABLED one**, matching the upstream's
            ``cache_results=False``: N ranks sharing a filesystem would otherwise race to write one
            file. Pass an enabled cache only if its writer election is known to be rank-safe.
        policy: Timing policy override. ``None`` builds the collective one from ``n_iters`` /
            ``warmup``, which is what production wants. Injecting one is how the unit drives a real
            multi-rank sweep with no GPU and with DELIBERATELY divergent per-rank timings -- the
            only way to make the consensus path fail on demand, since a real timer cannot be asked
            to disagree. An injected policy MUST keep ``collective=True``; the builder refuses one
            that does not, because a non-collective policy silently reintroduces the per-rank
            adaptive repetition count this whole module exists to avoid.
        group: The process group EVERY collective here uses -- both barriers and
            :func:`consensus_assert`. ``None`` means the default group and is inert when
            none exists. Pass one explicitly whenever the caller built its own: with an
            implicit group the broadcast buffers and the collective can be sized for
            different backends, which is a C++ abort rather than an error.


    Returns:
        A zero-argument callable returning ``(best_config, autotuner)``.

    Raises:
        ValueError: From `ConfigSpace` if ``adapter.configs()`` is empty or holds duplicates; from
            `Autotuner` if a knob is not a parameter of ``adapter.launch``; here, if an injected
            ``policy`` is not collective.
        AssertionError: From :func:`consensus_assert` if the ranks disagree.
    """
    if policy is not None and not policy.collective:
        raise ValueError(
            "build_distributed_autotuner requires a COLLECTIVE timing policy; got "
            "collective=False. A non-collective policy times each rank independently and may pick "
            "its repetition count per rank, so ranks run different numbers of in-kernel puts and "
            "desynchronize into a hang."
        )
    autotuner = Autotuner(
        fn=adapter.launch,
        space=ConfigSpace(list(adapter.configs())),
        key=list(adapter.key_names),
        policy=policy if policy is not None else make_collective_timing_policy(
            n_iters=n_iters, warmup=warmup
        ),
        consensus=Consensus(),
        cache=cache if cache is not None else ResultCache(
            getattr(adapter, "__class__", type(adapter)).__name__, enabled=False
        ),
        precompile=False,
    )

    def _barrier():
        # Same explicitness as `consensus_assert`: a barrier with no `group=` joins the DEFAULT one,
        # which in a shared process is whatever a sibling module initialized. `device_ids` is passed
        # only for NCCL -- gloo rejects it.
        if not dist.is_available():
            return
        g = group
        if g is None:
            if not dist.is_initialized():
                return
            g = dist.group.WORLD
        if "nccl" in str(dist.get_backend(g)).lower():
            dist.barrier(group=g, device_ids=[int(getattr(dm, "local_rank", 0))])
        else:
            dist.barrier(group=g)

    def run():
        _barrier()
        adapter.precompile(dm)
        _barrier()
        autotuner(*adapter.key_tensors, **adapter.fixed_kwargs)
        best = autotuner.best_config
        consensus_assert(best, dm, group=group)
        return best, autotuner

    return run

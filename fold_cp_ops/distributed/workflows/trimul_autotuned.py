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

"""Part D — the FULLY-FUSED distributed TriMul e2e (front + back A2A fused), GLUE-FREE.

This is the **P2.xgate.sc** TriMul combo (``docs/trimul_autotune_design.md`` §3 P2,
the single most-frequent local-compute winner for our sizes D≥256 & N≥512) with its
TWO compute kernels REPLACED by the Wave-2 A2A-fusion kernels — so each reshard is
fused into a GEMM epilogue — and the front collapsed to ONE einsum-native staged
invocation. The chain == single-device xgate.sc verbatim, with ZERO pytorch on the
data path (no ``.contiguous()``, no ``F.layer_norm``, no torch einsum/matmul):

  P2.xgate.sc (local): ``x_norm = layernorm_fwd(x)`` (shared) -> front ``a,b =
  gated_gemm_gate(x_norm, g_in, p_in, transpose_out, split_out_half)`` -> ``tri =
  _gemm1(a,b,direction)`` -> ``out = layernorm_dual_gated_gemm(tri, Wg=g_out, Wp=p_out,
  norm_weight=norm_out_w, ..., x_gate=x_norm)``.

``TriMulAutotuned`` chains it with BOTH all-to-all reshards fused into GEMM epilogues
(no torch ``redistribute``, no separate A2A kernel, no GMEM staging round-trip):

  1. local LN          : ``x_norm = layernorm_fwd(x)`` — the fold_cp_ops ELEMENT kernel (NOT
                         F.layer_norm), the P2 SHARED activation (the front gate input
                         AND the back out-gate), materialized ONCE. The staged front
                         consumes ``x_norm`` directly (``_normalize=False`` -> NO internal
                         LN) and the dual-x back out-gate reuses the SAME ``x_norm`` -> ONE
                         LN for the whole chain.                              (no comm)
  2. FRONT (fused A2A) : ONE ``DualGatedGemmDistSm90`` invocation (``transpose_out``
                         + dual width ``2D`` + ``_normalize=False``) emitting BOTH a,b: the
                         gated ``glu(x_norm@g_in^T, x_norm@p_in^T)`` postact is stored
                         D-MAJOR into a peer's symmetric recv ``(2*Dloc, M_full)`` (the
                         feature/D scatter ``S(0,1,2) -> S(0,3,3)``). After the exchange MY
                         recv holds ``a = recv[:Dloc]``, ``b = recv[Dloc:]`` (each ``(Dloc,
                         M_full)`` D-major) — MY Dloc feature slice of a AND b over the FULL
                         token grid, ALREADY einsum-native.                   (comm)
  3. einsum (fused A2A): ONE ``GemmA2ASm90`` GEMM-native batched GEMM ``tri[L] =
                         a_dm @ b_dm^T`` (L = d*B+b) whose D-store IS the back A2A: the
                         ``(i, j)`` tile of plane ``L`` is TMA-S2G-stored into a
                         GEMM-native 5-D symmetric recv ``(cp, Dloc, B, N_loc, N)`` on
                         peer ``i//N_loc`` (the ``S3 -> S1`` token reshard). The a/b recv
                         halves feed it as ZERO-COPY STRIDED VIEWS (the per-direction
                         ``.transpose(-1,-2)`` is a view; NO ``.contiguous()`` — proven
                         bit-identical to ``_gemm1``).                        (comm)
  4. local BACK        : ``layernorm_dual_gated_gemm`` reads the d-strided (M, D)
                         LayoutLeft value (= ``back_unpack_gemm_native(recv)``, a view)
                         NATIVELY through its input TMA load and fuses
                         ``LN(tri) -> p_out`` value + ``x_norm -> g_out`` gate (cuEq). (no comm)

THE FRONT IS ONE INVOCATION (einsum-native D-major): the staged front
(``DualGatedGemmDistSm90``, ``docs/layernorm_dual_gated_gemm_staged_a2a_design.md``)
emits BOTH a,b in one launch via ``transpose_out=True`` — the postact is M-major so the
D-major recv ``(2*Dloc, M_full)`` is shape-matched to the SMEM box; ``a = recv[:Dloc]``,
``b = recv[Dloc:]`` are zero-copy views. (This supersedes the wave-1 TWO-invocation
token-major stagec front, which paid a redundant front LN per operand + a separate
token<->D transpose to feed the einsum.)

CHAIN-CONCURRENCY (#29): this chain issues ONE fused front peer store + ONE fused back
peer store + 2 drains in ONE forward. The drains (``quiet`` + ``barrier_all``) after each
fused reshard serialize the comm phases so the back GEMM reads a fully-delivered front
recv.

REUSES (proven, do NOT re-derive): the T2.0d ``compile_nvshmem`` (tvm-ffi
OFF + ``--link-libraries`` + ``library_init``) compile route for BOTH fused kernels;
the staged front kernel's ``configure_a2a`` + EpilogueParams peer-atom route + the
D-major per-half store; the back kernel's ``configure_a2a_gemm_native`` GEMM-native
store + ``ReshardLayout.back_unpack_gemm_native``. The local-compute back-half consumer
is the UNTOUCHED ``fold_cp_ops.kernels.layernorm_dual_gated_gemm`` public entry (reads the d-strided
value natively — verified ``a_major="m"`` from stride ``(1, M)``).

Sharding contract (design doc §2): the local shard is ``(B, N_loc, N, D)`` with token
**rows** (tensor dim 1) cp-split over the cp group, the other token axis (dim 2) +
feature ``D`` whole — the canonical §2 rows-sharded mesh. 1-D cp headline (wave 1);
the staged front's ``configure_a2a_sharded`` is recv-index 2-D-invariant for a future
2-D ``(cp0, cp1)`` mesh (the back store already handles 2-D).
"""

import os
import warnings

import torch
from torch.distributed.tensor import Shard

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass import Float32, Int32
from cutlass.cute.runtime import from_dlpack

from fold_cp_ops._internal.activation import gate_fn_map
from fold_cp_ops._internal.rounding import RoundingMode
# The arch queries live in `_internal.arch` here; `compile_time.cute_dsl_utils` kept only the dtype
# map, because the first two READ THE DEVICE and the compile_time package is the one whose output is
# erased before codegen (it may emit layout algebra and nothing else).
from fold_cp_ops._internal.arch import get_device_capacity, get_max_active_clusters
from fold_cp_ops._internal.compile_time.cute_dsl_utils import torch2cute_dtype_map
from fold_cp_ops.kernels.layernorm import layernorm_fwd
# The upstream's `dual_gated_gemm_stagec` / `dual_gated_gemm_staged` are ONE functor here with a
# compile-time `fusion_variant` selector; `stagec` was renamed **`alg_fold`** and `staged`
# **`prolog_ln`** during extraction (`docs/kernel_variants_map.md` section 5.1). This comment used
# to say `stagec -> prolog_ln`, and `_consume` dispatched on that -- so every forward ran Stage D
# where the upstream runs Stage C, at a measured 1.24x on the out-gate. The behaviour names do not
# read across: Stage C folds the LayerNorm into the weight ALGEBRAICALLY (`alg_fold`), Stage D
# normalizes in the PROLOGUE (`prolog_ln`). The calling convention differs --
# ours takes the output tensor and the CTA tile positionally rather than allocating and heuristing
# internally -- so the two call sites are TRANSLATED, not copied; see `_consume` and `_local_front`.
from fold_cp_ops.kernels.layernorm_dual_gated_gemm import (
    alg_fold_heuristic_config,
    layernorm_dual_gated_gemm,
)
from fold_cp_ops._internal.gemm_tvm_ffi_utils import (
    get_dtypes,
    make_scheduler_args,
    perm3d,
)
from fold_cp_ops.kernels.dual_gated_gemm import interleave_dual_weights
from fold_cp_ops.distributed.gemm_a2a_epi import GemmA2ASm90
from fold_cp_ops.distributed.gemm_bitcode_compile import compile_nvshmem
from fold_cp_ops._internal.compile_time.template_params import (
    TemplateParams,
    TemplateParamsMixin,
)
from fold_cp_ops.distributed.dual_gated_gemm_a2a import DualGatedGemmDistSm90
from fold_cp_ops.distributed.pe_map import PeMap
from fold_cp_ops.distributed.reshard import ReshardLayout

try:  # nvshmem device signal op for the post-quiet DEVICE-signal drain (_A2ADeviceSignal); a2a.py ships it
    import nvshmem.core.device.cute.direct as nvshmem_cute_direct
    _HAS_NVSHMEM_DEVICE = True
except ImportError:  # pragma: no cover - non-nvshmem host
    _HAS_NVSHMEM_DEVICE = False


def _symmetric_empty(shape, *, dtype, device=None):
    """Allocate a SYMMETRIC buffer through the torch MemPool that owns nvshmem in this tree.

    Purpose
        The upstream allocated every symmetric buffer with nvshmem4py's own allocator
        (``nvshmem.core.interop.torch.tensor``). That allocator cannot be used here:
        `DistributedManager` performs the nvshmem bootstrap, so nvshmem4py's ``_is_initialized``
        stays permanently False and its ``tensor()`` raises
        ``NvshmemInvalid: NVSHMEM Library is not initialized`` -- measured, at the first store this
        engine builds. Every allocation in this module therefore routes here instead.

    Semantics
        The pool draws from the nvshmem symmetric heap and RECYCLES: dropping the last reference
        returns the block, so there is no per-tensor free to call (see :func:`_symmetric_free`).
        That is deliberate -- it keeps the collective ``nvshmem_free`` off the object-destruction
        path, where a GC-ordering difference across ranks deadlocks it rather than raising.

    Input requirements
        shape: tuple of ints, IDENTICAL on every rank. The underlying ``nvshmem_malloc`` is
            COLLECTIVE, so a rank-varying shape desynchronizes it and HANGS the job; it does not
            raise, and the hang surfaces at some later barrier with no attribution.
        dtype: a torch dtype. Keyword-only, so the call sites read exactly as the upstream's did.
        device: the calling rank's CUDA device. ``None`` uses the current device. A CPU device
            raises inside the pool.

    Returns:
        An UNINITIALIZED symmetric tensor -- recycled blocks carry the previous tenant's bytes, so a
        caller that needs zeros must zero it.

    Raises:
        torch.cuda.OutOfMemoryError: when the request cannot fit in free device memory. Raised HERE,
            BEFORE `nvshmem_malloc` is reached -- see the capacity precheck below for why the
            allocator must never be handed a request it will refuse.
    """
    from fold_cp_ops.distributed.distributed_manager import DistributedManager

    dev = torch.cuda.current_device() if device is None else device
    _refuse_unsatisfiable_symmetric_request(shape, dtype, dev)
    with torch.cuda.use_mem_pool(DistributedManager.symmetric_mempool(dev)):
        return torch.empty(shape, dtype=dtype, device=dev)


def _refuse_unsatisfiable_symmetric_request(shape, dtype, dev) -> None:
    """Refuse a symmetric allocation that cannot fit, BEFORE ``nvshmem_malloc`` sees it.

    Purpose
        **A FAILED symmetric allocation poisons the CUDA context.** It does not merely return an
        error: the next CUDA call raises `cudaErrorIllegalAddress`, torch turns that into a
        `c10::DistBackendError`, and the process calls `terminate` -- SIGABRT, whole rank gone.
        Reproduced deterministically in ~10 s at world 2 (see
        `tests/distributed/test_distributed_manager.py::test_a_symmetric_OOM_does_not_poison_the_NEXT_allocation`).

        So the OOM cannot be recovered from AFTER the fact, by any handler, on any rank. The only
        place it can be handled is before the allocator is called. That is what this does.

    Semantics
        Compares the request against `torch.cuda.mem_get_info()`'s FREE figure and raises
        `torch.cuda.OutOfMemoryError` if it does not fit. That exception is what the repo's OOM seams
        already expect, so a cell too large to run now SKIPS -- which is the documented rule -- rather
        than aborting the launch and taking every later cell with it.

        **This is not the memory-estimate gate CLAUDE.md forbids.** That rule bans refusing a shape
        because a model predicts it will not fit; capacity is to be discovered at RUNTIME. This
        discovers it at runtime, from the live free-memory figure, and refuses only a request that
        the allocator itself is about to refuse. The difference is that the refusal arrives as a
        catchable Python exception instead of as a dead process.

        Deliberately does NOT reserve headroom or apply a safety factor. A margin would refuse
        allocations that would have succeeded, which is the failure mode of an estimate gate; the
        only requests refused here are ones strictly larger than everything available.

        **The decision is REDUCED across ranks, and that is not a refinement.** Every rank computes
        the same `want` and reads its OWN `free`, so two ranks can disagree -- and `nvshmem_malloc`
        is COLLECTIVE, so a rank that refuses alone leaves its peers blocked in the allocator until
        a watchdog kills the job. An earlier version answered this with "they cannot disagree
        SILENTLY: the raise flows into the caller's `gated_skip`, which all-reduces the decision".
        That is true of the TEST path and FALSE of the production one: `gated_skip` is a pytest
        facility, and `_symmetric_empty` has eighteen call sites in this module's own workflow
        constructors, none of which has a pytest in it. So without the reduce this function converts
        a loud SIGABRT into a silent hang on exactly the path that ships.

        It bites where it activates: the refusal fires only when `want > free`, and near that
        boundary is precisely where two ranks' free figures straddle it. A safety margin is NOT the
        fix -- it makes the divergence rarer and no less silent, and it is the estimate-gate
        behaviour this docstring already refuses above.

    Input requirements
        shape: the requested shape, IDENTICAL on every rank -- as for the caller. This is what makes
            the reduce meaningful: the ranks are voting on the same number.
        dtype: a torch dtype, for the element size.
        dev: the CUDA device whose free memory is queried, and which carries the vote tensor. `None`
            uses the current device. Must be the rank's OWN device -- the reduce goes through NCCL.

    Returns:
        None when the request fits ON EVERY RANK, or when free memory cannot be determined -- an
        unavailable figure must not become a refusal, or a query failure would look like an OOM.
        Note that a measurement failure votes FITS, so one rank that cannot measure never refuses
        the others; it defers to the allocator, which remains the authority.

    Raises:
        torch.cuda.OutOfMemoryError: the request exceeds free device memory on THIS rank or on any
            peer, saying which. Raised on every rank together, so a caller may handle it as a
            collective outcome -- which is what makes the OOM-skip seam usable from a workflow and
            not only from a test. With no process group up, the decision is local, because a single
            rank has nobody to disagree with.
    """
    want = free = total = -1
    fits = True
    try:
        n = 1
        for d in shape:
            n *= int(d)
        want = n * torch.empty((), dtype=dtype).element_size()
        free, total = torch.cuda.mem_get_info(dev)
        fits = want <= free
    except Exception:
        fits = True  # cannot measure -> do not refuse; the allocator remains the authority

    # Make the decision RANK-UNIFORM. Everything above is per-rank -- every rank computes the same
    # `want` but reads its OWN `free` -- and `nvshmem_malloc` is COLLECTIVE, so a rank that refuses
    # alone leaves every peer blocked in the allocator until a watchdog kills the job. MIN over the
    # fits-vote makes one short rank refuse all of them, together.
    #
    # Note where the reduce SITS: after the `except`, not inside the `try`. An early `return` on a
    # measurement failure would skip the collective on that rank alone and hang the others -- the
    # exact defect this reduce exists to remove, reintroduced by the error path.
    i_am_short, peer_is_short = not fits, False
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        vdev = dev if dev is not None else torch.cuda.current_device()
        try:
            vote = torch.tensor([1 if fits else 0], device=vdev, dtype=torch.int32)
            torch.distributed.all_reduce(vote, op=torch.distributed.ReduceOp.MIN)
            agreed = bool(vote.item())
        except Exception:
            agreed = fits  # the group is already broken; nothing can be uniform through it
        peer_is_short, fits = agreed is False and not i_am_short, agreed
    if fits:
        return
    where = (
        "a PEER rank could not fit it (this rank could)"
        if peer_is_short
        else f"this rank has {free / 2**30:.2f} GiB free of {total / 2**30:.2f} GiB total"
    )
    raise torch.cuda.OutOfMemoryError(
        f"symmetric request of {want / 2**30:.2f} GiB refused for shape {tuple(shape)} {dtype}: "
        f"{where}. Refused BEFORE nvshmem_malloc, and refused on EVERY rank together: a failed "
        f"symmetric allocation poisons the CUDA context and the rank then aborts on the next "
        f"collective, so this cannot be caught afterwards -- and a refusal on one rank alone would "
        f"leave its peers blocked in the collective allocator instead."
    )


def _symmetric_free(tensor) -> None:
    """Release a symmetric buffer -- a deliberate NO-OP, kept so the call sites still say it.

    Purpose
        The upstream paired every allocation with ``nvshmem_torch.free_tensor``. Under the MemPool
        the block returns when the last reference is dropped, so there is nothing to call -- but
        DELETING the call sites would erase where the upstream considered a buffer dead, which is
        information the next reader of this port wants.

    Semantics
        Does nothing, on purpose, and must stay that way: ``nvshmem_free`` is COLLECTIVE. Calling it
        from a teardown path means ranks reach it in whatever order their garbage collectors decide,
        and mismatched order is a deadlock rather than an error.

    Args:
        tensor: ignored. Accepted so the call sites read as they did upstream.
    """
    return None


class _A2ASignalDrain:
    """Host-orchestrated per-peer signal-wait A2A drain (PROTOTYPE; ``CPO_A2A_SIGNAL_DRAIN=1``).

    Replaces the post-reshard GLOBAL ``barrier_all`` (every PE waits for every PE) with a
    POINT-TO-POINT completion on a symmetric ``(cp,)`` int64 signal pad, via the nvshmem HOST
    stream ops (all present in ``nvshmem.bindings``)::

        sender (after the fused peer store):
            quiet_on_stream(s)                                # my outstanding puts complete
            signal_op_on_stream(pad+my*8, ctr, SET, peer_r)   # tell peer r "my data landed"  (∀ r)
        receiver:
            signal_wait_until_on_stream(pad+s*8, GE, ctr)     # wait peer s's data            (∀ s)

    The FRONT reshard is a FULL-cp all-to-all (I send to AND receive from every cp peer, self
    included) whose feature peer-table is FLAT-cp (2-D-invariant), so the pe list is
    ``cp_pe_table`` for both 1-D and 2-D. MONOTONIC ``ctr`` (strictly increasing per drain)
    makes SET+GE need NO per-iter reset (drops the a2a.py reset barrier).

    NVLink vs IB (host path): a single ``quiet_on_stream`` forces completion of BOTH transports'
    outstanding puts (NVLink P2P TMA-S2G AND IB nbi); the signals are stream-ordered strictly
    after it, so data-before-signal holds transport-agnostically with no per-transport device
    fence (that fence recipe — ``fence_proxy("async.global")`` + ``fence_acq_rel_sys`` — is only
    load-bearing for an IN-KERNEL producer signal, the Phase-2 evolution in the design note).

    Single-buffer scope: this drops the POST-A2A ``barrier_all`` only; the caller KEEPS the
    PRE-A2A ``_barrier`` (WAR guard) unless recv+pad are double-buffered. With the pre-barrier
    kept, all ranks enter each drain in lockstep so the SET value == the current ``ctr`` on
    every rank.
    """

    def __init__(self, pm):
        import nvshmem.core
        import nvshmem.bindings as nb
        import nvshmem.core.interop.torch as nvshmem_torch
        from nvshmem.core.direct import ComparisonType

        self.cp = int(pm.cp)
        self.my = int(pm.my_cp_rank)
        # COLLECTIVE (cp,) int64 malloc FIRST — no host-sync interposed vs the prior recv malloc
        # (#B rendezvous rule); the d2h .tolist() below runs AFTER this last collective malloc.
        self._pad = _symmetric_empty((self.cp,), dtype=torch.int64)
        self._pad.zero_()
        self._pe_list = [int(p) for p in pm.cp_pe_table.tolist()]  # flat cp peer -> global PE
        self._nvshmem_torch = nvshmem_torch
        self._nb = nb
        # THE SYMMETRIC BASE COMES FROM THE TENSOR, NOT FROM NVSHMEM4PY'S REGISTRY.
        #
        # The upstream read it back through `tensor_get_buffer`, which looks the tensor up in
        # nvshmem4py's own allocation table. `self._pad` is allocated from the torch symmetric
        # MemPool here (see `_symmetric_empty`), so nvshmem4py has NO record of it and the lookup
        # raises:
        #     NvshmemInvalid: Tried to retrieve MemoryResource for GPU with no NVSHMEM Allocations
        #
        # This is the allocator conversion reaching a path only the CROSS-NODE drain takes, which is
        # why it stayed invisible: measured 2026-08-21, all 13 single-node cells of the acceptance
        # grid passed and all 5 two-node cells failed here. An NVLink job never constructs this
        # class.
        #
        # `data_ptr()` is the same device address `tensor_get_buffer(...).handle` returned -- the
        # MemPool draws from the symmetric heap, so the tensor's own pointer IS the symmetric base.
        self._base = int(self._pad.data_ptr())  # symmetric signal base ptr
        self._elem = self._pad.element_size()  # 8
        self._SET = int(nvshmem.core.SignalOp.SIGNAL_SET)
        self._ADD = int(nvshmem.core.SignalOp.SIGNAL_ADD)
        self._GE = int(getattr(ComparisonType, "GE", getattr(ComparisonType, "CMP_GE", 5)))
        # AGG (default): cp ADDs into peer slot-0 + ONE agg wait GE cp*ctr (host calls = cp+1; the
        # blocking per-slot waits that SERIALIZE per-peer skew collapse to one). PER-SLOT
        # (CPO_A2A_SIGNAL_AGG=0): per-peer SET(ctr) + per-slot wait (2*cp calls) — the double-buffer
        # -safe variant (Stage 2). AGG relies on the KEPT pre-A2A barrier for lockstep (a fast peer's
        # FUTURE ADD must not satisfy a PAST agg-wait); per-slot+double-buffer is the no-pre-barrier path.
        self._agg = bool(int(os.environ.get("CPO_A2A_SIGNAL_AGG", "1")))
        self._ctr = 0

    def drain(self) -> None:
        """One post-A2A drain: quiet (complete my puts) + per-peer signal + receiver wait."""
        nb = self._nb
        cstrm = torch.cuda.current_stream().cuda_stream
        self._ctr += 1
        v = self._ctr
        nb.quiet_on_stream(cstrm)  # complete MY outstanding puts (NVLink + IB), stream-ordered
        if self._agg:
            for r in range(self.cp):  # ADD +1 into each peer's aggregate slot 0 (self included)
                nb.signal_op_on_stream(self._base, 1, self._ADD, self._pe_list[r], cstrm)
            nb.signal_wait_until_on_stream(self._base, self._GE, self.cp * v, cstrm)  # ONE agg wait
        else:
            sig_me = self._base + self.my * self._elem
            for r in range(self.cp):  # tell every cp peer (self incl) my data for it has landed
                nb.signal_op_on_stream(sig_me, v, self._SET, self._pe_list[r], cstrm)
            for s in range(self.cp):  # wait until every cp peer SET my pad[s] >= v (its data landed)
                nb.signal_wait_until_on_stream(self._base + s * self._elem, self._GE, v, cstrm)

    def free(self) -> None:
        try:
            if self._pad is not None:
                _symmetric_free(self._pad)
                self._pad = None
        except Exception:
            pass


class _DeviceSignalKernelParams(TemplateParams):
    """The signal kernel's two compile-time constants.

    ``cp`` is the CTA count folded into the launch grid AND the peer-table extent; ``add_op`` is the
    nvshmem ``SignalOp`` value baked into the device ``signal_op``. Both are read during tracing and
    become kernel constants, so both belong in the compile key -- which is what declaring them here
    achieves, via ``TemplateParamsMixin.compile_key()``.

    Input requirements: ``cp`` must be the flat-cp peer count that the signal pad and PE table were
    sized from -- a mismatch launches a grid that indexes past the table, which is an illegal
    address rather than an exception. ``add_op`` must be a valid ``nvshmem.core.SignalOp`` int; an
    invalid one is not diagnosed here and produces a wrong signal semantic at runtime.
    """

    cp: int
    add_op: int


class _DeviceSignalKernel(TemplateParamsMixin):
    """cp-CTA POST-QUIET signal kernel (compiles on sm90/4.5.2 — smoke-verified). CTA r signals peer r's
    aggregate slot 0 (ADD +1) after the fence recipe. The HOST quiet (before this kernel, same stream)
    already drained the store + made the data remotely visible; ``fence_proxy("async.global")`` +
    ``fence_acq_rel_sys`` order this kernel's OWN signal write system-scope. ONE launch (cp threads,
    lane0/CTA) replaces the host-signal drain's cp ``signal_op_on_stream`` host dispatches."""

    Params = _DeviceSignalKernelParams

    def __init__(self, cp: int, add_op: int):
        """Bind the two compile-time constants.

        Args:
            cp: flat-cp peer count. Must equal the extent of both the symmetric signal pad and the
                PE table this kernel is called with; a larger value indexes past them.
            add_op: the ``nvshmem.core.SignalOp`` int the device ``signal_op`` is baked with.

        Raises:
            TypeError: from ``_bind_params`` if either is a runtime value (a tensor) rather than a
                Python int -- which would produce a kernel reading a dangling MLIR value.
        """
        self._bind_params(cp=int(cp), add_op=int(add_op))

    @cute.jit
    def __call__(self, mSignal: cute.Tensor, mPeTable: cute.Tensor, stream):
        cp: cutlass.Constexpr[int] = self.cp
        self.kernel(mSignal, mPeTable).launch(
            grid=(cp, 1, 1), block=(32, 1, 1), cluster=(1, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(self, mSignal: cute.Tensor, mPeTable: cute.Tensor):
        r, _, _ = cute.arch.block_idx()  # CTA r -> peer r
        if cute.arch.warp_idx() == cutlass.Int32(0):
            if cute.arch.lane_idx() == cutlass.Int32(0):
                cute.arch.fence_proxy("async.global")
                cute.arch.fence_acq_rel_sys()
                dst_pe = mPeTable[r]
                sig_slot = cute.local_tile(mSignal, (1,), (cutlass.Int32(0),))  # aggregate slot 0
                nvshmem_cute_direct.signal_op(
                    sig_slot, cutlass.Int64(1), cutlass.Int32(self.add_op), dst_pe
                )



def compile_device_signal_kernel(cp: int, add_op: int, *, persist=None):
    """Compile the cp-CTA device signal kernel from FLAT arguments, reusing across instances.

    Purpose
        The flat compile entry point for this kernel: every value that decides the emitted code is a
        scalar ARGUMENT, and the functor is constructed INSIDE. That is what makes the compile key
        complete by construction rather than by inspection -- two calls agreeing on ``(cp, add_op)``
        compile the same kernel and share it; two differing in either cannot.

    Functionality & semantics
        Builds a `_DeviceSignalKernel`, makes the two FAKE ``(cp,)`` operand descriptors the compile
        traces against (int64 signal pad at 8-byte alignment, int32 PE table at 4-byte), and calls
        `compile_nvshmem` with ``reuse=True`` and ``register=True``.

        Fake tensors rather than real ones on purpose: this kernel indexes ``mPeTable[r]`` and a
        single aggregate slot, so nothing about the trace depends on a device pointer -- and a real
        allocation here would make a compile-only path allocate symmetric memory.

        ``op_factory`` is passed as the same zero-arg lambda, because ``resolve_program_key``
        TRACES a throwaway copy to hash its MLIR and tracing consumes a functor; the copy and the
        compiled one are identical by construction since both capture the same two ints.

        Reuse is ON here and not a parameter. There is exactly one shape of this kernel per
        ``(cp, add_op)`` in a process, it holds no per-instance state, and the second
        `TriangularMultiplication` in a process would otherwise pay a full compile for a kernel
        already registered. A caller that needs an unshared one can call `compile_nvshmem` directly.

    Args:
        cp: flat-cp peer count. Must be >= 1 and must equal the extent of the signal pad and PE
            table the returned kernel is later CALLED with -- the grid is ``(cp, 1, 1)`` and CTA
            ``r`` reads ``mPeTable[r]``, so a smaller runtime table is an out-of-bounds read with no
            diagnostic.
        add_op: the ``nvshmem.core.SignalOp`` int baked into the device ``signal_op``.
        persist: optional ``artifact_cache.PersistSpec`` for cross-PROCESS reuse. None (the default)
            is byte-identical to no disk cache at all.

    Returns:
        A `CompiledGemmBitcode` whose ``executor`` takes ``(mSignal, mPeTable, stream)``. The SAME
        object on a reuse hit, with its holder count incremented -- so every holder must call
        ``free()`` exactly once, and only the last one finalizes.

    Raises:
        RuntimeError: from `compile_nvshmem` when the compiled kernel exposes no
            ``jit_module.cuda_library`` to register. An in-kernel ``signal_op`` against unregistered
            device state faults rather than raising, so refusing here is the cheaper failure.
    """
    factory = lambda: _DeviceSignalKernel(cp, add_op)  # noqa: E731
    f_sig = cute.runtime.make_fake_tensor(cutlass.Int64, (cp,), stride=(1,), assumed_align=8)
    f_pe = cute.runtime.make_fake_tensor(cutlass.Int32, (cp,), stride=(1,), assumed_align=4)
    return compile_nvshmem(
        factory(),
        f_sig,
        f_pe,
        cutlass_torch.current_stream(),
        register=True,
        persist=persist,
        op_factory=factory,
        reuse=True,
    )


class _A2ADeviceSignal:
    """Post-quiet DEVICE-signal A2A drain (PROTOTYPE; ``CPO_A2A_SIGNAL_DEVICE=1``). Same completion
    contract as :class:`_A2ASignalDrain` (host quiet + agg signal-wait) but the cp per-peer signals are
    emitted by ONE device-kernel launch (cp threads) instead of cp HOST ``signal_op_on_stream`` calls —
    removing the host-launch overhead the host-signal path pays per drain (the likely small-N loss).

    drain():  quiet_on_stream (HOST — drains TMA/IB; device quiet is LLVM-dead in cute-DSL, keep host)
              -> device signal kernel (cp CTAs, ADD +1 into peer r's agg slot 0, fence-recipe-ordered)
              -> ONE agg wait (signal_wait_until GE cp*ctr). Monotonic ctr; no reset; count=cp (no
              per-tile, no ring). The kernel is bitcode-linked tvm-ffi-OFF (the a2a.py nvshmem route);
              from_dlpack is done ONCE at construction (cached), so the per-drain cost is 1 kernel launch."""

    def __init__(self, pm, persist=None):
        """Build the drain and compile its device-signal kernel.

        Args:
            pm: the PE map. Supplies ``cp``, the device, the flat-cp -> global PE table and the
                symmetric signal pad; all four must already be valid, as the compile and the
                ``from_dlpack`` views happen HERE and not on first drain.
            persist: optional ``artifact_cache.PersistSpec`` for the signal kernel's compile.
                Defaults to None, which is byte-identical to the pre-consolidation behaviour --
                this constructor is on the workflow's hot construction path, so turning a cache on
                by default here would change what every existing caller does without anyone asking.

        Raises:
            RuntimeError: from ``compile_nvshmem`` when the compiled kernel exposes no
                ``jit_module.cuda_library`` to register. An in-kernel ``signal_op`` against
                unregistered device state faults rather than raising, so refusing here is the
                cheaper failure.
        """
        import nvshmem.core
        import nvshmem.bindings as nb
        import nvshmem.core.interop.torch as nvshmem_torch
        from nvshmem.core.direct import ComparisonType
        import cutlass.torch as cutlass_torch
        from cutlass.cute.runtime import from_dlpack

        self.cp = int(pm.cp)
        self.my = int(pm.my_cp_rank)
        # COLLECTIVE (cp,) int64 agg pad (slot 0 <- every peer's +1 per drain); malloc FIRST (no host-sync
        # interposed vs the prior recv malloc, #B rule), then the .to()/compile below.
        self._pad = _symmetric_empty((self.cp,), dtype=torch.int64)
        self._pad.zero_()
        # device (cp,) int32 flat-cp -> global PE table (the kernel routes signal_op via pe_table[r]).
        self._pe_dev = pm.cp_pe_table.to(device=pm.device, dtype=torch.int32).contiguous()
        self._nvshmem_torch = nvshmem_torch
        self._nb = nb
        # Same as the host-signal drain above: the base comes from the tensor, because the
        # MemPool allocation is invisible to nvshmem4py's registry.
        self._base = int(self._pad.data_ptr())
        self._GE = int(getattr(ComparisonType, "GE", getattr(ComparisonType, "CMP_GE", 5)))
        self._flush = bool(int(os.environ.get("CPO_A2A_DEVSIG_FLUSH", "0")))  # host quiet after kernel (IB attempt)
        self._ctr = 0
        # Set BEFORE the compile: a compile that raises must still leave `free()` callable, and an
        # attribute that only exists on the success path turns a compile error into an AttributeError
        # inside the cleanup that was trying to report it.
        self._sig_compiled = None
        # Compile the device signal kernel (bitcode-linked, tvm-ffi OFF = the a2a.py nvshmem route;
        # signal_op is an nvshmem device symbol needing the --link-libraries bitcode). smoke-verified.
        #
        # This used to be `cute.compile(..., --link-libraries)` -> `.to(dev)` -> `from_handle` ->
        # `library_init` typed out inline, i.e. `compile_nvshmem`'s body re-implemented. It was not
        # re-implemented out of carelessness: the helper was called `compile_gemm_with_bitcode`, and
        # this is a SIGNAL kernel, so its author reasonably read the name as not applying. The
        # function has been renamed because of exactly this; the duplicate is now the shared call.
        # `CompiledGemmBitcode` also owns the retention this code hand-rolled -- the kernel object
        # for `free()` to finalize, AND the compiled object whose garbage collection would otherwise
        # unload a CUDA library nvshmem still holds a raw handle to.
        # M4c: a factory, not just an instance -- `resolve_program_key` traces a THROWAWAY copy to
        # hash its MLIR, and tracing consumes a functor. Both calls capture the same two values, so
        # the copy and the compiled one are identical by construction.
        #
        # M6: the functor construction, the fake operands and the compile all moved into
        # `compile_device_signal_kernel`, which takes the two ints FLAT and therefore keys on them
        # by construction. Two `_A2ADeviceSignal` instances in one process now share one compiled
        # signal kernel instead of each paying for its own.
        self._stream = cutlass_torch.current_stream()
        self._sig_compiled = compile_device_signal_kernel(
            self.cp, int(nvshmem.core.SignalOp.SIGNAL_ADD), persist=persist
        )
        self._run_fn = self._sig_compiled.executor
        self._cute_sig = from_dlpack(self._pad, assumed_align=8)
        self._cute_pe = from_dlpack(self._pe_dev, assumed_align=4)

    def drain(self) -> None:
        import cutlass.torch as cutlass_torch

        nb = self._nb
        cstrm = torch.cuda.current_stream().cuda_stream
        self._ctr += 1
        v = self._ctr
        nb.quiet_on_stream(cstrm)  # HOST quiet: drain TMA/IB, data remotely visible (source-required)
        self._run_fn(self._cute_sig, self._cute_pe, cutlass_torch.current_stream())  # DEVICE signal, cp threads
        if self._flush:  # IB-hang fix attempt: flush the DEVICE-issued IB signal_op (device quiet is LLVM-dead)
            nb.quiet_on_stream(cstrm)
        nb.signal_wait_until_on_stream(self._base, self._GE, self.cp * v, cstrm)  # ONE agg wait

    def free(self) -> None:
        try:
            if self._sig_compiled is not None:  # UNREGISTER the device-signal module (TASK #44 leak fix)
                # `CompiledGemmBitcode.free()` is `library_finalize` plus clearing the handle -- the
                # same two steps this used to do inline, now in the one place that also knows to keep
                # the compiled object alive until after the finalize.
                self._sig_compiled.free()
                self._sig_compiled = None
        except Exception:
            pass
        try:
            if self._pad is not None:
                _symmetric_free(self._pad)
                self._pad = None
        except Exception:
            pass


# F4 — symmetric-recv cache cap. The per-N recv caches below are memoized and were never
# evicted, so a `dynamic=True` instance retained one symmetric buffer per distinct runtime N until
# the heap ran out (54.4 GiB over 12 perf-grid cells; the failing malloc then poisons the heap and
# the next NVSHMEM collective faults). Cap them instead.
#
# The default 2 keeps the current N plus its predecessor, so an alternating two-N workload still
# never reallocates, while an N-sweep stays O(1) in the heap instead of O(#N). Set
# CPO_TRIMUL_RECV_CACHE=0 for "current N only", or a larger value to trade heap for realloc.
_RECV_CACHE_MAX = max(1, int(os.environ.get("CPO_TRIMUL_RECV_CACHE", "2")))


def _pad_operand_k(t: torch.Tensor) -> torch.Tensor:
    """Materialize ``t`` (…, M, K) with its innermost extent padded to a multiple of 64 elements.

    Returns a view whose LOGICAL shape is unchanged (``[..., :K]``) but whose M-row stride is
    ``ceil(K/64)*64`` elements = a multiple of 128 B for bf16. That stride is what the back GEMM's
    mainloop TMA-G2S walks; leaving it 16-mod-32 (every ``K % 16 == 8``) costs a measured
    1.213-1.215x at Dloc=128. Callers must already be materializing — this only widens the
    allocation they were making anyway, so it adds no pass and no second buffer.

    ``K % 64 == 0`` falls through to a plain ``.contiguous()``, byte-identical to the old behaviour.
    """
    k = t.shape[-1]
    k_pad = ((k + 63) // 64) * 64
    if k_pad == k:
        return t.contiguous()
    buf = torch.empty((*t.shape[:-1], k_pad), dtype=t.dtype, device=t.device)
    buf[..., :k].copy_(t)
    return buf[..., :k]


def _recv_cache_put(cache, key, buf, free_fn):
    """Insert `buf` under `key` and free the OLDEST entries beyond `_RECV_CACHE_MAX`.

    COLLECTIVE-SAFETY: `free_fn` is `nvshmem_free`, which is collective, so every rank must free the
    same buffers in the same order. Eviction is FIFO on dict INSERTION order (Python dicts preserve
    it), which is a pure function of the runtime-N sequence -- and every rank rebinds to the same N
    in the same order (`_rebind_runtime` is SPMD). Access order is deliberately NOT used: it would
    make the eviction sequence depend on which store each rank happened to touch.
    """
    cache[key] = buf
    while len(cache) > _RECV_CACHE_MAX:
        oldest = next(iter(cache))
        if oldest == key:  # never evict the buffer we are about to bind
            break
        try:
            free_fn(cache.pop(oldest))
        except Exception as e:  # noqa: BLE001 - never let an evict failure mask the caller's work
            warnings.warn(f"symmetric recv eviction failed for key {oldest!r}: {e!r}", stacklevel=2)


#: The CTA tile M the out-gate consumer runs at. Fixed 128 -- the only CTA-M either dual-gated
#: heuristic sweeps, and the value the upstream's hidden auto-pick also used.
_CONSUMER_TILE_M = 128


def _consumer_tile_n(D: int) -> int:
    """The CTA tile over the out-gate's two-activation width.

    Purpose
        Replace the tile the upstream's public entry chose internally. ``tile_N`` here tiles the
        ``2D``-wide PRE-activation, not the ``D``-wide output, which is why this is not the front's
        tile rule reused.

    Semantics
        Largest multiple of 32 no greater than 128 that DIVIDES ``2D``; 128 when nothing divides.
        Mirrors :func:`_resolve_front_config`'s own fallback, deliberately: two different answers to
        "what tile does a 2D-wide dual take" is how a tile becomes a shape constraint.

    Args:
        D: output feature width. Must be positive; a ``D`` whose ``2D`` has no multiple-of-32
            divisor falls back to 128 rather than raising, because the kernel accepts a partial tile.

    Returns:
        The CTA tile_N.
    """
    two_n = 2 * D
    hi = min(two_n, 128)
    hi -= hi % 32
    return next((t for t in range(hi, 0, -32) if two_n % t == 0), 128)


#: The out-gate consumer seam's OTHER half, next to `_consumer_tile_n` and for the same reason.
#: The upstream picked between two module-level ENTRY POINTS; this tree has one functor and picks a
#: ``fusion_variant`` string, so the choice became an expression instead of a call -- and an inline
#: expression is not a thing a test can name. It was inline, it was INVERTED, and the inversion
#: survived because the seam it belongs to is the one seam the workflow's test module describes and
#: does not pin. Extracted here so `test_the_out_gate_variant_is_the_upstream_entry_it_replaces` has
#: a subject.
_CONSUMER_VARIANT = {"stagec": "alg_fold", "staged_a_in_regs": "prolog_ln"}


def _consumer_fusion_variant(consumer: str) -> str:
    """The `fusion_variant` implementing the upstream out-gate entry that ``consumer`` names.

    Purpose
        Translate the upstream's two-module-entry choice into this tree's one-functor selector, at
        the one place `_consume` reads it.

    Semantics
        The correspondence is by MECHANISM, and the behaviour names deliberately do not read across
        from the upstream's stage letters -- which is exactly how this got inverted once:

        ``stagec`` -> ``alg_fold``
            The upstream's Stage C (`dual_gated_gemm_stagec`) reads A **raw and once**, folds the
            LayerNorm gain into the weight on the host, and repairs the result rank-one in the
            epilogue (``r*acc - s*c + d``). Its two-A class asserts ``not a_in_regs``. Our
            `alg_fold` does the same thing and has no register-source path.

        ``staged_a_in_regs`` -> ``prolog_ln``
            The upstream's Stage D (`dual_gated_gemm_staged`) NORMALIZES A in shared memory before
            the WGMMA and then runs a plain GEMM, so the producer sweeps A **twice** and pass 2 must
            drain the WGMMA before releasing the stage. Its two-A class carries the
            ``_value_a_in_regs`` register mainloop the upstream's name refers to. Our `prolog_ln`
            is that kernel.

        `docs/kernel_variants_map.md` section 5.1 records the rename in this direction. Dispatching
        ``stagec`` to ``prolog_ln`` -- which this workflow did until the inversion was found -- runs
        Stage D on every forward where the upstream runs Stage C, measured at **1.24x** on the
        out-gate at the workflow's own cell (D=256, MN-major value, H100).

    Args:
        consumer: an upstream consumer name. Must be a key of :data:`_CONSUMER_VARIANT`; ``"torch"``
            is handled by `_consume` before this is reached and is NOT accepted here. An unknown
            name raises rather than defaulting, because the default is what hid the inversion: a
            silent ``else`` branch turns a typo, and a mis-translation, into a running kernel.

    Returns:
        A member of the kernel module's ``FUSION_VARIANTS``.

    Raises:
        ValueError: on any name outside :data:`_CONSUMER_VARIANT`.
    """
    try:
        return _CONSUMER_VARIANT[consumer]
    except KeyError:
        raise ValueError(
            f"consumer={consumer!r} names no out-gate kernel; expected one of "
            f"{sorted(_CONSUMER_VARIANT)} (or 'torch', which _consume handles before this point)"
        ) from None


def _resolve_front_config(M, D, device):
    """Source the front (plain-dual LN+DualGatedGEMM) perf-config from its OWN size-heuristic — NO
    hardcoded perf literal. Front is a PLAIN dual (no gate3/x_gate/mask/transpose/split), contraction
    K=D, dual half-width N=D. Returns (tile_M, tile_N, pingpong); tile_M is fixed 128 (the
    heuristic's only swept CTA-M).

    **The heuristic is `alg_fold_heuristic_config`, and the mapping is stated rather than assumed.**
    The upstream called ``dual_gated_gemm_stagec._stagec_heuristic_config``; NO symbol of that name
    exists here, because extraction renamed the ``stagec`` fusion to **``alg_fold``** and folded both
    dual-gated fusions into one functor. So this IS the stagec heuristic under its behaviour name --
    the call below was always right, and only this paragraph was wrong: it used to say ``stagec``
    became ``prolog_ln``, which is the inversion that made `_consume` dispatch the out-gate to Stage
    D. Getting it right here and wrong there is what kept the defect invisible, since the front tile
    the two produce is identical and pinned. The surviving heuristic answers the same question with
    the same RETURN SHAPE (``{}`` | ``{"pingpong": True}`` | ``{"tile_N": 256}``) and the same
    early-out on the fused features, differing only in dropping a ``split_out_half`` parameter this
    call site always passed ``False``.

    That mapping is a claim about behaviour, so it is TESTED rather than argued: the composition of
    this function and :func:`_resolve_front_tile_n` is pinned against the tile configs `main`
    actually resolved over the 40-cell acceptance grid, recorded in ``w8plan/step1/variants.csv``
    (STEP 1). See ``tests/distributed/workflows/test_trimul_autotuned.py``.

    Args:
        M: local token count (the front operand's M extent). Only its magnitude matters.
        D: FULL feature width, not the per-peer slice -- the heuristic is asked about the plain dual
            whose contraction is K=D and whose dual half-width is N=D. The per-peer narrowing to
            ``D_loc`` happens in :func:`_resolve_front_tile_n`, and doing it here instead would ask
            the heuristic about a shape the kernel never sees.
        device: selects the arch table; ``None`` means "the tuned arch", with no warning.

    Returns:
        ``(tile_M, tile_N, pingpong)``.
    """
    hk = alg_fold_heuristic_config(
        M,
        D,
        D,
        has_gate3=False,
        has_xgate=False,
        has_mask=False,
        transpose_out=False,
        device=device,
    )
    tile_M = 128
    if "tile_N" in hk:
        tile_N = hk["tile_N"]
    else:  # heuristic left tile_N to the kernel auto-pick: largest mult-of-32 <= 128 dividing 2D.
        two_n = 2 * D
        hi = min(two_n, 128)
        hi -= hi % 32
        tile_N = next((t for t in range(hi, 0, -32) if two_n % t == 0), 128)
    return tile_M, tile_N, bool(hk.get("pingpong", False))


def _resolve_front_tile_n(D_loc, f_tn):
    """Resolve a candidate front tile_N to the tile_N the store actually RUNS at this ``D_loc``.

    The FRONT-A2A store constraint: the postact tile_N (= ``f_tn // 2``) must DIVIDE ``D_loc`` (a CTA N-tile
    must lie within ONE peer's D-slice for the per-CTA single-peer feature select; asserted in
    ``dual_gated_gemm_staged_a2a.epi_to_underlying_arguments:1063``). The stagec heuristic sizes tile_N to 2D
    and can EXCEED D_loc, so: (1) clamp to the largest mult-of-32 that DIVIDES D_loc (the WGMMA N quantum);
    (2) C12 — if the postact (``f_tn // 2``) STILL does not divide D_loc (``D_loc ≡ 8 (mod 16)``, e.g. Dloc=8
    (D128/cp16) or Dloc=24 (D384/cp16)), fall to the kernel's OWN validity-aware ``best_front_tile`` (valid
    postact for every D_loc >= 8). Fires ONLY where the mult-32 clamp is invalid -> byte-identical for every
    currently-valid cell (PERF-NEUTRAL).

    SHARED by ``TriMulAutotuned.__init__`` (the store build) AND ``_FrontProxyAdapter._valid_tile_ns`` (the
    autotune grid) so the freeze offers / times ONLY tile_ns production RUNS — a divergent copy would let the
    autotuner time a tile (e.g. tile_n=64 @ D_loc=32: postact 32|32 valid but 32%64!=0) that production then
    clamps away to 32, mis-recording the freeze. NOTE: adopting ``best_front_tile`` WHOLESALE (its widest tile,
    e.g. 64 @ D_loc=32) is a DIFFERENT, non-neutral perf change (alters tile_M at Dloc in {64,192}) -> deferred
    to a GPU-paired-median perf follow-up (docs C12), NOT this per-peer-tile clamp."""
    if D_loc % f_tn != 0:
        hi = min(f_tn, D_loc)
        hi -= hi % 32
        f_tn = next((t for t in range(hi, 0, -32) if D_loc % t == 0), 32)
    if D_loc % (f_tn // 2) != 0:
        f_tn = DualGatedGemmDistSm90.best_front_tile(D_loc)[1]
    return f_tn


def front_pad_inner_geometry(B, N_i_loc, N_j_loc, tile_M):
    """P2 — the front recv's PAD-the-innermost-token-extent geometry for ONE runtime token shape.

    Returns ``(Yg_walk, rpp_eff, N_j_pad)``, the SINGLE source of truth shared by the recv sizing, the
    store configure, the runtime ``token_grid_yg`` and every downstream view. Host-side pure int math
    (no GPU) -> unit-testable.

    THE DEFECT it removes. Rank ``r``'s front-recv destination block starts at ``r*rpp*2`` bytes with
    ``rpp = B*N_i_loc*N_j_loc``; when ``2*rpp % 128 != 0`` the block lands mid-sector and EVERY store
    straddles two 32-B sectors (the mod-128 phase law, ``dual_gated_gemm_staged_a2a.py:1239-1255``:
    ~+4% at 64-mod-128, ~+8-11% at 32-mod-128, **+18-37% at 16-mod-32**). Padding the INNERMOST token
    extent to a whole ``tile_M`` makes ``rpp_pad = B*N_i_loc*N_j_pad`` a ``tile_M`` multiple, so every
    rank's base is 128-B clean.

    WHY ``tile_M`` AND NOT 64. A sub-``tile_M`` pad makes an ``epi_m`` store box straddle a *j*-row
    boundary, and no TMA box can express a non-contiguous destination run — the same reason
    ``_configure_transpose_in`` pads ``Xg`` to ``BLK_M`` and not to 8 or 64. ``tile_M in {128, 256}``,
    both multiples of 64, so ``rpp_pad*2 % 128 == 0`` on every rank.

    WHY THE **INNERMOST** EXTENT AND NOT ``rpp`` ITSELF. ``rpp_pad = B*N_i_loc*N_j_pad`` factorises
    through ``N_i_loc`` EXACTLY, so the recv's global-i strides become ``(N_i_loc*N_j_pad, N_j_pad)``
    which collapse to ONE mode of stride ``N_j_pad`` — every downstream reader stays a plain strided
    VIEW (zero copy). Padding ``rpp`` directly gives ``(rpp_pad, N_j_loc)``, which never collapse and
    would force an O(Dloc*N^2) I/O-order ``.contiguous()`` — a frugality breach. That distinction is
    the whole reason this variant is affordable and the ``rpp`` one is not.

    THE GUARD (non-negotiable). Pad ONLY when the shape actually carries the phase penalty:
    ``rpp % 64 != 0``. Without it the pad REGRESSES shapes that never had the defect — e.g.
    ``N_j_loc = 1088`` is ``%64==0`` (already clean) but ``%128==64``, so a blind pad pushes it to
    1152 for +5.9% work and ZERO benefit. Declined shapes come out byte-identical: ``N_j_pad ==
    N_j_loc``, ``rpp_eff == M``, no pad, no extra tiles, no extra bytes.

    THE DECLINE ENCODING (``Yg_walk = M``, so ``Xg = 1``). Under ``dynamic_shape`` ONE compile serves
    many N while the guard is a PER-N decision, so the regime cannot be a compile-time flag. It rides
    the runtime ``Yg``: at ``Yg = M`` the walk has a single X row, ``tile_y == tile_m``, ``tile_x == 0``
    and the A row is ``tile_m*BLK_M + r`` — the unpadded flat-M walk EXACTLY, with the scheduler
    emitting the same ``ceil(M/BLK_M)`` tiles and the existing partial-tile clamp dropping the same
    overshoot.

    COST where it does apply: ``+(N_j_pad-N_j_loc)/N_j_loc`` on the front recv GMEM and the front-GEMM
    work (the pad columns compute glu(0)=0), i.e. O(Dloc*B*N) against an O(Dloc*B*N^2) recv — strictly
    sub-leading along the N_token axis, so the frugality rule holds. It is NOT performance-neutral at a
    padded shape: it replaces a ~1.5x penalty on half the ranks with a uniform ~1.1x one, and the
    barrier-coupled job gets faster while an even-parity rank gets slower in isolation.
    """
    B, N_i_loc, N_j_loc, tile_M = int(B), int(N_i_loc), int(N_j_loc), int(tile_M)
    M = B * N_i_loc * N_j_loc
    # THE GUARD IS VERIFIED, NOT ASSUMED (docs/migration_failing_tests.md §12.5). A forced-pad
    # experiment padded 6 base-CLEAN cells at cp=2 and NONE won: N=2000/4000 have a MISaligned recv
    # row pitch (32/64 mod 128) which the pad fixes, and bought +0.09..+0.47%; N=1984 (base-clean AND
    # pitch-clean, yet still engaging since 1984 is not a tile_M multiple) is the negative control and
    # cost the most, +2.3/+2.5%. Meanwhile N=3512 -- identical except its BASE is misaligned, at LOWER
    # pad cost -- wins 17-21%. So `M % 64` selects on the right quantity: neither the row pitch nor
    # tile-quantization of the walk explains the win, only the per-rank destination base does.
    if M % 64 == 0:  # destination base already 128-B clean on every rank -> DECLINE (byte-identical)
        return M, M, N_j_loc
    N_j_pad = ((N_j_loc + tile_M - 1) // tile_M) * tile_M
    return N_j_loc, B * N_i_loc * N_j_pad, N_j_pad


def _resolve_back_config(B, N, D, cp, dt, device):
    """Resolve the back (batched square einsum) GEMM tile+cluster for the design-E A2A store.

    Returns **(128, 128) cluster_N=2** (``(tile_M, tile_N, pingpong, cluster_shape_mnk)`` =
    ``(128, 128, False, (1, 2, 1))``) — the perf-validated, register-safe 2040 fix.

    Why: the OLD default (128,128) cluster(1,1) under-issued the WGMMA pipeline at non-pow2 square S
    (NCU: SM-compute 41.6% vs 79.5% at S=2040 — same occupancy/grid/waves, so NOT wave-quant) and was
    suboptimal at every S. Adding **cluster_N=2** (B-operand multicast across the 2-CTA N-cluster ->
    deeper-fed MMA) recovers it: bare GEMM S=2040 390 -> 553 TF (1.42x), and lifts aligned S too
    (1024 607 -> 646). cluster_M MUST stay 1 (the pe_aligned per-peer M-tiling guards on cluster_M==1
    — M-multicast and per-peer M-tiling are mutually exclusive). Correctness-validated with pe_aligned
    + the partial-N store (test_back_gemm_native_pe_aligned_fastcfg_correct).

    On tile_N: a larger tile_N=256 gives the FULL lever (S=2040 t_gemm 545 -> 612 TF, coupled 2040/2048
    ratio 1.219 -> 1.168) and **DOES compile + is oracle-clean on the production COUPLED store** across
    cp{2,4} x D{128,256} (27-cell matrix, all correct). The ptxas "insufficient registers (96), needs
    154" reg-fail at tile_N=256 fires ONLY on the c2 (overlap_double_tma, §7.16p) BENCH-ablation store —
    it spins DEDICATED consumer warpgroups draining CONCURRENT with the MMA, so the wide 64x256
    accumulator + the drain-WG live state coexist past the warpgroup reg cap. The coupled store drains
    AFTER the MMA (no concurrent drain WG) -> compiles fine; NO register-target bump is involved (so the
    §7.16m setmaxnreg-realloc deadlock landmine does not apply). Unlike the FRONT (whose tile_N tiles the
    feature axis -> invalid at D128/Dloc=64), the BACK's tile_N tiles the token-j axis (independent of
    Dloc) -> (128,256)c(1,2) is a valid UNIVERSAL back default. (128,128)c(1,2) is the conservative
    minimal-delta value (only cluster_N 1->2 changes vs today; tile_N stays 128 so the partial-N + pe_
    aligned store is byte-identical), trading ~half the 2040 lever for zero small-N GEMM regression; the
    default may move to (128,256). nvMMH is intentionally NOT consulted here; per-shape autotune (with a
    compile-safety gate) is a documented follow-on. ``B,N,D,cp,dt,device`` kept for signature stability."""
    return (128, 128, False, (1, 2, 1))


# --------------------------------------------------------------------------- #
# TASK #49 — BAKED sm90 H100+IB autotune-config table (the empirically-harvested
# production default, analogous to the H200 universal heuristic default).
# --------------------------------------------------------------------------- #
# Harvested + validated on venue B 2-node H100+IB. N-INVARIANT: ONE config per
# (D, cp0, cp1) at the build ANCHOR (N=2048) serves ALL token-N via the dynamic-compile
# path — the tile/cluster are token-N-independent (the front tile_N tiles the FEATURE
# axis, the back tile_N the token-j axis; neither scales with N), so no N is baked into
# the key. Key = (D, cp0, cp1); value = (front_cfg, back_cfg) with
#   front_cfg = (f_tm, f_tn, f_pp, f_W)   back_cfg = (b_tm, b_tn, b_pp, b_cluster).
# Selected as the sm90 production default ONLY when the venue MATCHES the harvest venue:
# arch is sm90 (H100/H200) AND the job has cross-node IB peers (has_ib_peers) AND
# autotune is OFF AND no explicit front_/back_tile_mn override. Any miss (untuned arch,
# single-node NVLink, autotune ON, an unharvested (D, cp0, cp1) key) falls back to the
# H200 universal default (_resolve_front_config / _resolve_back_config) — the baked
# branch NEVER fires off-venue, so the NVLink/non-sm90 path stays byte-identical.
# ARCH-SPECIFIC by construction (feedback_heuristics_arch_specific): these literals are
# the H100+IB optima and are gated off every other arch/topology. Do NOT over-generalize
# (back is uniform (128,128) c(1,1,1) across all 8; front D256 is uniform; front D384
# W=128@cp16-1D vs W=256@2-D) — keep the explicit 8-entry table; unharvested cells fall back.
_SM90_IB_CONFIG = {
    # (D, cp0, cp1): (front (f_tm, f_tn, f_pp, f_W), back (b_tm, b_tn, b_pp, b_cluster))
    (256, 16, 1): ((128, 32, False, 256), (128, 128, False, (1, 1, 1))),
    (256, 2, 8): ((128, 32, False, 256), (128, 128, False, (1, 1, 1))),
    (256, 4, 4): ((128, 32, False, 256), (128, 128, False, (1, 1, 1))),
    (256, 8, 2): ((128, 32, False, 256), (128, 128, False, (1, 1, 1))),
    (384, 16, 1): ((128, 16, False, 128), (128, 128, False, (1, 1, 1))),
    (384, 2, 8): ((128, 16, False, 256), (128, 128, False, (1, 1, 1))),
    (384, 4, 4): ((128, 16, False, 256), (128, 128, False, (1, 1, 1))),
    (384, 8, 2): ((128, 16, False, 256), (128, 128, False, (1, 1, 1))),
}


def _resolve_sm90_ib_config(D, cp0, cp1, arch_is_sm90, has_ib_peers, do_autotune):
    """Return the baked ``(front_cfg, back_cfg)`` for ``(D, cp0, cp1)`` from ``_SM90_IB_CONFIG``, or None.

    Pure gate (GPU-free, unit-testable — NO nvshmem / device build): the harvested H100+IB table is
    consulted ONLY when the venue matches the harvest venue — arch is sm90 AND the job has cross-node IB
    peers AND autotune is OFF. Any miss (untuned arch, single-node NVLink, autotune ON, or an unharvested
    ``(D, cp0, cp1)`` key) returns None so the caller falls back to the existing size-heuristic. The
    explicit ``front_/back_tile_mn`` override is handled by the CALLER (it wins over this table and is
    applied after the store-config resolve), so this helper does not see it. Precedence at the call site:
    explicit override > autotune > this baked table > heuristic. ARCH-SPECIFIC by design
    (feedback_heuristics_arch_specific) — the literals are the H100+IB optima, gated off every other arch."""
    if not (arch_is_sm90 and has_ib_peers and not do_autotune):
        return None
    return _SM90_IB_CONFIG.get((D, cp0, cp1))


# --------------------------------------------------------------------------- #
# FRONT fused store — ONE reusable staged-A2A invocation emitting BOTH a,b into a
# D-MAJOR einsum-native recv. Compiles DualGatedGemmDistSm90 ONCE (bitcode
# route, tvm-ffi OFF, transpose_out + dual width 2D + _normalize=False) and re-runs
# it per call into a persistent (2*Dloc, M_full) symmetric recv. The gated postact
# store IS the front A2A (feature/D scatter S(0,1,2) -> S(0,3,3)); the D-major recv
# delivers a,b already einsum-native so the einsum reads them as zero-copy views
# (no token<->D transpose). See docs/layernorm_dual_gated_gemm_staged_a2a_design.md.
# --------------------------------------------------------------------------- #
class DualGatedGemmDistStore:
    """Reusable fused staged front store: x_norm (M,K) + STACKED (Wg,Wp) (2D,K) -> recv (2*Dloc, M_full).

    ONE invocation emits BOTH a,b (vs the stagec front's TWO width-D invocations). The
    STACKED gate/up weights ``Wg=[Wg_a;Wg_b]``, ``Wp=[Wp_a;Wp_b]`` (each ``(2D, K)``) make
    the dual GEMM N=2D; the glu epilogue halves ``2*2D -> 2D`` and ``transpose_out=True``
    makes the postact M-major so the D-major recv ``(2*Dloc, M_full)`` is shape-matched to
    the SMEM box. After a drain MY recv holds ``a = recv[:Dloc]``, ``b = recv[Dloc:]`` (each
    ``(Dloc, M_full)`` D-major), MY Dloc feature slice of a AND b over the FULL token grid.

    ``_normalize=False``: the kernel takes the PRE-NORMALIZED ``x_norm`` directly (the
    caller runs ``layernorm_fwd`` ONCE upstream — shared with the back gate), so there is
    NO internal LN and NO weight fold (the weights are the RAW stacked projections).
    """

    def __init__(
        self,
        pm: PeMap,
        Wg2,
        Wp2,
        M,
        K,
        D,
        dt,
        *,
        eps=1e-5,
        tile_shape_mn=(128, 128),
        pingpong=False,
        is_persistent=True,
        dynamic_shape=False,
        route2_ni=False,
        composite_k=False,
        B=1,
        b_dynamic=False,
        N_i_loc=None,
        N_j_loc=None,
        hybrid_ib=False,
        ib_wide_batch=128,
        consumer_warpgroups=1,
        ring_depth=2,
        has_mask=False,
        pad_inner=False,
        bg2=None,
        bp2=None,
    ):
        import nvshmem.core  # noqa: F401 (availability probe)
        import nvshmem.core.interop.torch as nvshmem_torch

        # HYBRID NVLink+IB front A2A (opt-in). ON => the postact store becomes the differential is_p2p
        # drain (NVLink peer: coupled TMA-S2G byte-identical; IB peer: symmetric ring + wide-put), via
        # configure_a2a(ib_drain=True, ib_wide=True, ...). Default OFF => the ib kwargs are NEVER passed =>
        # configure_a2a is byte-identical to today (the coupled store). §10 R1: the LAYOUT (route2_ni /
        # transpose_in) axis is ⊥ the TRANSPORT (ib_drain/ib_wide) axis, so route2_ni COMPOSES with the IB
        # drain — the factored store (dual_gated_gemm_staged_a2a.py epi_to_underlying_arguments) runs the
        # SAME transport block over the N_i-stride-1 recv_view, and the wide coalesce runs along N_i within
        # one N_j column. So BOTH D-major and route2_ni thread the ib kwargs identically (no combo reject).
        self._hybrid_ib = bool(hybrid_ib)
        # IB wide-put knobs (only consumed when hybrid_ib): W (32 KiB at 128), drain-warpgroup count,
        # ring depth — the current shipped standalone-front defaults (harness front_a2a target).
        self._ib_wide_batch = int(ib_wide_batch)
        self._ib_consumer_warpgroups = int(consumer_warpgroups)
        self._ib_ring_depth = int(ring_depth)
        self.pm = pm
        self.cp = pm.cp
        # NATIVE 1-D vs 2-D token sharding cp axes (mirror GemmA2AStore): the FRONT feature
        # scatter/peer-table stays FLAT-cp (2-D-invariant), but route2_ni's N_i-stride-1 store needs
        # (cp0, cp1) to position global-i (i-shard base cp0_coord*Xg_pad) + global-j (j-shard base
        # cp1_coord*N_j_loc). 1-D => cp1==1 (cp0==cp, N_j_loc==N) => byte-identical to the flat path.
        cp_axis_sizes = tuple(int(s) for s in pm.cp_axis_sizes)
        self.cp0 = cp_axis_sizes[0]
        self.cp1 = cp_axis_sizes[1] if len(cp_axis_sizes) > 1 else 1
        self.Dloc = D // self.cp
        self.D = D
        self.M = M
        self.dt = dt
        self.eps = eps
        self._nvshmem_torch = nvshmem_torch
        # ROUTE-2 (A) N_i-STRIDE-1 producer store (incoming no-copy): the front WALKS transposed
        # (transpose_in) so the recv comes out with global-i (N_i) STRIDE-1 -> the incoming einsum
        # reads a_major="k" NATIVE. Recv is 3-D (2*Dloc, N_i, N_j) (mirror _b5_route2_ni_probe.py);
        # B=1 + 1-D (N_j==N) only (the store folds no B, positions global-i by flat cp rank). The
        # token grid = (B, Xg=N_i_loc, Yg=N_j_loc). Default OFF = the plain (2*Dloc, M_full) recv.
        self._route2_ni = bool(route2_ni)
        # COMPOSITE-K incoming (§9): transpose_in + the D-MAJOR store (NOT route2_ni's N_i-stride-1 store)
        # -> a PADDED per-rank-CONTIGUOUS recv (2*Dloc, cp*rpp_padded). The back reads the composite
        # K=(cp,Xg_pad). Config-mostly (§9.3): reuses the transpose_in padded walk (route2_ni's token grid
        # machinery) but leaves _a2a_route2_ni OFF so the plain D-major postact store runs. Mutually
        # exclusive with route2_ni (they are two different incoming recv layouts).
        self._composite_k = bool(composite_k)
        if route2_ni and composite_k:
            raise ValueError("route2_ni and composite_k are mutually-exclusive incoming front variants.")
        self._route2_B = int(B)
        # B-DYNAMIC (route2_ni / composite_k only): the token-batch extent is a RUNTIME input, so ONE
        # compiled front serves every batch -- `rebind_M(..., B=)` re-sizes the symmetric recv and the
        # per-launch epilogue carries the extent. OFF => B is baked at construction (a functor built
        # at B=1 that is later launched at B=2 raises rather than unravelling the walk with the wrong
        # Xg). `_b_mode` is the LAYOUT consequence: the recv grows an explicit batch mode INSIDE the
        # feature and OUTSIDE the cp slot, which is the only placement for which the back operand's
        # L = (Dloc, B) collapses to one stride (see DualGatedGemmDistSm90._a2a_b_mode).
        # The TRANSPORT is not a term here, and the `and not self._hybrid_ib` that used to appear
        # in both lines is gone. `hybrid_ib` sets `ib_drain` (:1247), which sets `decoupled`, and
        # `configure_a2a` used to REFUSE `_a2a_decoupled and _a2a_b_mode()` -- so with
        # `_a2a_b_plane` = `transpose_in and (b_plane or b_dynamic)` and `b_dynamic` following
        # `dynamic` (which `TriangularMultiplication` always is), that refusal fired on every
        # cross-node incoming build, at B == 1, where there is no plane to carry at all. The IB
        # drain now carries the plane in its ring metadata, so b-mode composes with every transport
        # and the exclusion is a bug rather than a guard: leaving it would keep the cross-node fast
        # store on the plane-less recv the back read can no longer decode.
        self._b_dynamic = bool(b_dynamic and (route2_ni or composite_k))
        self._b_mode = bool(self._b_dynamic or int(B) > 1) and bool(route2_ni or composite_k)
        self._route2_Xg = int(N_i_loc) if N_i_loc is not None else 0  # local i extent (=N/cp)
        self._route2_Yg = int(N_j_loc) if N_j_loc is not None else 0  # local j extent (=N in 1-D)
        self._route2_tile_M = int(tile_shape_mn[0])  # BLK_M (Xg padding quantum) for route2_ni/composite rebind
        # pad_inner (the PLAIN D-major front only — route2_ni/composite_k are already phase-clean by
        # construction, their rpp derives from Xg_pad). Pads the recv's INNERMOST token extent so every
        # rank's destination base r*rpp*2 is 128-B clean; see front_pad_inner_geometry for the mechanism,
        # the tile_M quantum, the innermost-vs-rpp distinction and the already-clean guard. The per-N
        # geometry (Yg_walk, rpp_eff, N_j_pad) is recomputed on every rebind_M because the guard is a
        # per-N decision and one dynamic compile serves many N. Default OFF => byte-identical.
        self._pad_inner = bool(pad_inner)
        self._pi_yg, self._pi_rpp, self._pi_nj_pad = 0, 0, 0
        self._recv_buf = None  # route2_ni: the contiguous (2*Dloc,N_j,N_i) backing (permuted to recv)
        self._comp_rpp_padded = 0  # composite_k: the PADDED per-peer token extent (recv col count / cp)
        # dynamic_shape: compile ONCE at this anchor M (mark_layout_dynamic operands + postact,
        # mark_compact_shape_dynamic(mode=1) recv so the feature dim 2*Dloc stays static) so ONE
        # executor serves many token counts; run() rebinds the runtime-M operands + recv per call and
        # rebind_M() (re)allocates the symmetric recv per distinct M (cached, SPMD-lockstep). Mirrors
        # benchmark/distributed/dyn_front_validate.py. Default OFF = the byte-identical static path.
        self._dynamic_shape = dynamic_shape
        self._recv_cache = {}
        # MASK: the per-token-row pair mask, applied to the gated glu postact
        # (mMaskColVec, post-glu fp32, pre-store) — the _local_front ``ab *= mask[...,None]``
        # semantics, LOCAL (pre-A2A). COMPILE-TIME variant (the epilogue's mask branch is
        # const_expr): has_mask bakes the masked-glu store; run() rebinds the runtime (1, M) mask.
        # Default OFF => mMaskColVec=None everywhere => byte-identical unmasked path.
        self._has_mask = bool(has_mask)
        self._cur_mask = None  # the CURRENT (1, M) mask cute-source, native dtype (bf16; cast to fp32 in-kernel)

        device_capacity = get_device_capacity(Wg2.device)
        assert device_capacity[0] == 9, f"SM90 only; got {device_capacity}"
        tile_M, tile_N = tile_shape_mn
        twoN = Wg2.shape[0]  # = 2D (the dual GEMM N)
        assert twoN == 2 * D, f"stacked Wg2 N={twoN} must be 2*D={2 * D}"

        # B = interleave(Wg2, Wp2): (K, 2*twoN). No LN fold (_normalize=False -> raw weights).
        #
        # The FRONT PROJECTION BIASES ride the same helper. `Wg2`/`Wp2` are (2D, K), so the dual's own
        # N is `twoN == 2D` and a (2D,) bias is exactly the (N,) the helper documents -- it does the
        # interleave and the row-pitch padding, and the caller does no width arithmetic. Order is
        # load-bearing: the weights go in as (Wg2, Wp2) = (gate, up), so the biases must be
        # (bg2, bp2) = (g_in_b, p_in_b). Swapping them is SILENTLY wrong -- the interleaved column
        # order must match the weight's or the bias is double-counted, which nothing downstream can
        # detect.
        #
        # `bias_i` is a VIEW of a pitch-padded buffer, so the base is held on the instance; dropping
        # it would leave the epilogue reading freed memory. None -> byte-identical to the bias-free
        # path (the epilogue term is const_expr-pruned, not added as zero).
        if bg2 is not None:
            Bw, self._rowvec_base = interleave_dual_weights(
                Wg2, Wp2, bg2, bp2, return_bias=True
            )
            # NOT mark_layout_dynamic: the vector's extent is 2*twoN, a function of D alone, so
            # it is invariant across the M values a dynamic executor serves. `_md` is not even
            # in scope this early -- it is defined ~100 lines below, with the operand marks.
            # The helper pitch-pads to a 4-element multiple, so the fp32 view is 16-B aligned.
            self._cute_rowvec = from_dlpack(self._rowvec_base, assumed_align=16)
        else:
            Bw = interleave_dual_weights(Wg2, Wp2)  # (K, 2*twoN) bf16
            self._rowvec_base = None
            self._cute_rowvec = None
        # transpose_out=True postact: the logical (M, 2D) M-major view of a (2D, M) buffer.
        # rows_per_peer = my whole local token count (M); recv (2*Dloc, M_full) D-major.
        rows_per_peer = M
        # #B rendezvous-deadlock fix (mirror harness front_route2 + front_a2a): materialize pe_table (a
        # cp_pe_table.tolist() d2h CUDA host-sync) HERE, BEFORE the FIRST collective nvshmem malloc (the recv
        # below). Reused at configure_a2a + the ring pe_dev_t, so NO host-sync lands BETWEEN the two collective
        # symmetric-mallocs (recv @290/294, then the hybrid_ib ring @402). The old body did this .tolist()
        # AFTER the recv (BETWEEN the two collectives) -> it smeared the ranks across the two malloc barriers
        # so the nvshmem_malloc rendezvous never converged at large N (the N4096 construction hang; collective
        # even at cp2 all-NVLink). Fine at N1024 where the interposed sync is ~4x shorter.
        pe_table = tuple(int(x) for x in pm.cp_pe_table.tolist())
        if route2_ni:
            # N_i-STRIDE-1 recv (mirror _b5_route2_ni_probe.py): N_i = cp0*Xg_pad (padded FULL global-i,
            # all cp0 i-blocks); N_j = FULL global-N (all cp1 j-blocks; the symmetric recv holds the
            # whole j-extent, the store writes only my j-block at cp1_coord*N_j_loc). WALK extent Yg =
            # N_j_loc (=self._route2_Yg) is SEPARATE (feeds token_grid, NOT the recv N_j alloc) — B10
            # split. 1-D: cp0==cp, cp1==1 => N_i==cp*Xg_pad, N_j==Yg==N => byte-identical. Allocate the
            # contiguous (2*Dloc, N_j, N_i) [N_i inner stride-1] presented (2*Dloc, N_i, N_j).
            self._route2_Xg_pad = ((self._route2_Xg + tile_M - 1) // tile_M) * tile_M
            N_i = self.cp0 * self._route2_Xg_pad
            N_j = self.cp1 * self._route2_Yg  # full global-j extent (== Yg==N when 1-D)
            self._route2_N_i, self._route2_N_j = N_i, N_j
            self._recv_buf, self.recv = self._alloc_route2_recv(N_i, N_j, self._route2_B, dt)
        elif composite_k:
            # COMPOSITE-K incoming: transpose_in + the D-MAJOR store -> a PADDED per-rank-CONTIGUOUS 2-D
            # recv (2*Dloc, cp*rpp_padded), rpp_padded = B*Yg*Xg_pad (each per-Y block padded to a whole
            # BLK_M count so the store box stays 16-B/epi-M-aligned). The recv half reshapes to
            # (Dloc, cp, N_j=Yg, Xg_pad) — the composite K=(cp,Xg_pad) the validated back-read consumes,
            # pad rows [Xg,Xg_pad) == 0 (glu(0)=0). rpp_padded is computed here (the SAME formula
            # _configure_transpose_in overrides _a2a_rows_per_peer to; asserted equal after configure).
            self._route2_Xg_pad = ((self._route2_Xg + tile_M - 1) // tile_M) * tile_M
            self._comp_rpp_padded = self._route2_B * self._route2_Yg * self._route2_Xg_pad
            self.recv = self._alloc_composite_recv(self._comp_rpp_padded, self._route2_B, dt)
        elif self._pad_inner:
            if self._route2_Xg <= 0 or self._route2_Yg <= 0:
                raise ValueError(
                    "pad_inner=True requires the token geometry (B, N_i_loc, N_j_loc) — its guard and "
                    f"recv sizing are per-shape; got N_i_loc={self._route2_Xg}, N_j_loc={self._route2_Yg}."
                )
            # pad_inner: size the recv from the PADDED per-peer extent so every rank's destination base
            # r*rpp_eff*2 is 128-B clean. front_pad_inner_geometry owns the guard: an already-clean
            # shape returns rpp_eff == M and N_j_pad == N_j_loc, i.e. this allocation is byte-identical
            # to the plain one below. configure_a2a is passed rpp_eff (NOT M) so the store's column
            # block, clamp bound and my_col_sub_off all agree with what is allocated here.
            self._pi_yg, self._pi_rpp, self._pi_nj_pad = front_pad_inner_geometry(
                self._route2_B, self._route2_Xg, self._route2_Yg, tile_M
            )
            rows_per_peer = self._pi_rpp
            self.recv = _symmetric_empty((2 * self.Dloc, self.cp * rows_per_peer), dtype=dt)
        else:
            M_full = self.cp * rows_per_peer
            self.recv = _symmetric_empty((2 * self.Dloc, M_full), dtype=dt)
        out_postact = torch.empty(
            twoN, M, device=Wg2.device, dtype=dt
        )  # (2D, M) D-major (compile shape)
        pa_arg = out_postact.mT  # (M, 2D) M-major
        # MASK anchor (has_mask only): a (1, M) per-row col-vec baked at COMPILE so the epilogue's
        # const_expr mask branch is on; run() rebinds the runtime mask into self._cur_mask. l=1 (single
        # batch plane) matches the stagec mMaskColVec=mask.unsqueeze(0) convention. None => unmasked build.
        # DTYPE = kernel dt (bf16): the ColVecLoad loads the mask in its NATIVE dtype and converts to fp32
        # IN-KERNEL (begin_loop .to(acc_dtype)), so NO host fp32 cast is needed — the anchor is bf16 so the
        # compile traces a bf16 mask read, and run() passes the user's native bf16 mask (zero host cast).
        # route2_ni / composite_k: the transpose_in GEMM walks the PADDED (b, Yg, Xg_pad) M-grid, so the
        # mMaskColVec length is B*Yg*Xg_pad (NOT M=B*Xg*Yg) — anchor it padded so the compile traces the
        # padded ColVecLoad (the raw dynamic path anchors at an OFF-GRID N=1000 where Xg_pad>Xg).
        if self._has_mask:
            if self._route2_ni or self._composite_k:
                # ZERO-COPY: the transpose_in front reads the NATIVE (l=1, Xg=N_i_loc, Yg=N_j_loc) mask
                # in-kernel (TransposedMaskColVecLoad) — anchor it at the NATIVE 3-D shape (no host
                # transpose/pad). Xg,Yg are marked dynamic so the op reads them at runtime (dynamic-N).
                # Mode-0 is the BATCH, not a degenerate l: `forward` passes the user's native
                # (B, N_i_loc, N_j_loc) mask straight through on these paths, and the in-kernel read
                # splits the walk's flat (b*Yg + y). At B==1 this is the (1, Xg, Yg) anchor it always
                # was. Under dynamic_shape every extent is marked, so the anchor's B does not bind.
                self._mask_anchor = torch.empty(
                    self._route2_B, self._route2_Xg, self._route2_Yg, device=Wg2.device, dtype=dt
                )
            elif self._pad_inner:
                # the pad_inner walk indexes the mask at the PADDED flat-M, so the (1, M) flat
                # col-vec no longer lines up. Anchor the NATIVE 3-D (1, Xg, Yg_walk) form the
                # _begin_pad_inner branch reads with the same (tile_x, tile_y) unravel as the A-load —
                # still ZERO-COPY (forward() passes mask_local.reshape(1, Xg, Yg), a view of the
                # contiguous (B, N_i_loc, N_j_loc) mask). In the DECLINE regime Yg == M so this is
                # (1, 1, M) and the read reduces to the flat one. Xg,Yg dynamic -> runtime.
                self._mask_anchor = torch.empty(
                    1, M // self._pi_yg, self._pi_yg, device=Wg2.device, dtype=dt
                )
            else:
                self._mask_anchor = torch.empty(1, M, device=Wg2.device, dtype=dt)
        else:
            self._mask_anchor = None
        self._cur_mask = self._mask_anchor

        A = torch.empty(1, M, K, device=Wg2.device, dtype=dt)  # compile-time shape only
        B3 = Bw.mT.unsqueeze(0)  # (1, 2*twoN, K) RAW weight .mT
        PostAct = pa_arg.unsqueeze(0)  # (1, M, 2D) M-major
        A_p, B_p, _, _ = perm3d(A, B3, None, None)
        PostAct_p = perm3d(PostAct, B3, None, None)[0]

        a_dtype, _, _, _ = get_dtypes(A, B3, A, None)
        _md = (lambda c: c.mark_layout_dynamic()) if dynamic_shape else (lambda c: c)
        if not dynamic_shape:
            _md_recv = lambda c: c
        elif route2_ni:
            # ROUTE-2 3-D recv (2*Dloc, N_i, N_j) [N_i stride-1]: BOTH token axes vary with N, so mark
            # N_i (mode 1) AND N_j (mode 2) shape-dynamic (chained); feature (mode 0 = 2*Dloc) STAYS
            # static so the store's int(tensors[0].shape[1])=2*Dloc holds. stride_order (0,2,1) = the
            # (2*Dloc, N_j, N_i)-contiguous-then-permute layout (feature outermost, N_j, N_i innermost),
            # INVARIANT across N. divisibility 8 = 16-B (bf16) TMA-S2G box (N_i is %128, N_j is %8). With
            # this + the a8ecd15 runtime-n_x copy_fn, ONE front compile serves every N (no per-N front).
            if self._b_mode:
                # B-MODE: 4-D (2*Dloc, B, N_i, N_j) over a (2*Dloc, B, N_j, N_i)-contiguous buffer,
                # so the outermost-to-innermost order is (feature, B, N_j, N_i) = modes (0,1,3,2).
                # All THREE of B, N_i and N_j vary at runtime; only the feature stays static (the
                # store reads int(recv.shape[0]) as a Python int). B's divisibility is 1 -- any batch.
                _md_recv = (
                    lambda c: c.mark_compact_shape_dynamic(
                        mode=1, stride_order=(0, 1, 3, 2), divisibility=1
                    )
                    .mark_compact_shape_dynamic(mode=2, stride_order=(0, 1, 3, 2), divisibility=8)
                    .mark_compact_shape_dynamic(mode=3, stride_order=(0, 1, 3, 2), divisibility=8)
                )
            else:
                _md_recv = lambda c: c.mark_compact_shape_dynamic(
                    mode=1, stride_order=(0, 2, 1), divisibility=8
                ).mark_compact_shape_dynamic(mode=2, stride_order=(0, 2, 1), divisibility=8)
        elif composite_k and self._b_mode:
            # B-MODE composite: 3-D (2*Dloc, B, cp*rpp_b) row-major; B and the token extent are both
            # runtime, the feature stays static.
            _md_recv = lambda c: c.mark_compact_shape_dynamic(
                mode=1, stride_order=(0, 1, 2), divisibility=1
            ).mark_compact_shape_dynamic(mode=2, stride_order=(0, 1, 2), divisibility=1)
        else:
            _md_recv = lambda c: c.mark_compact_shape_dynamic(mode=1)
        self._md, self._md_recv = _md, _md_recv
        # HYBRID NVLink+IB differential drain (opt-in): ib_drain (is_p2p per-peer select) + ib_wide
        # (wide-put coalesce). ib_drain forces the decoupled ring; ib_wide rides on it. Default OFF
        # => empty dict => configure_a2a byte-identical to today. Shipped standalone-front defaults
        # (harness front_a2a target): W=128 (32 KiB put), consumer_warpgroups=1, ring_depth=2.
        #
        # THIS COMMENT USED TO SAY "Only for the NON-route2_ni (plain) store -- route2_ni is
        # coupled-only + guarded above". **That is FALSE, and has been since §10 R1 made the LAYOUT
        # axis (route2_ni / transpose_in) compose with the TRANSPORT axis (ib_drain / ib_wide).**
        # The route2_ni branch below passes `**_ib_kw` (see its own R1 comment); there is no guard
        # anywhere in this module -- the only route2_ni `raise` is its mutual exclusion with
        # composite_k -- and `ib_wide` has no opt-out separate from `hybrid_ib`: setting one sets
        # the other, unconditionally, right here.
        #
        # It matters because that sentence is what an auditor asking "can a caller reach route2_ni
        # WITH ib_wide from a public entry point?" would read, and it would answer no. The answer is
        # YES, and no flag is involved -- neither knob is settable (`trimul_tuning.NOT_EXPOSED`
        # refuses both by name), so this configuration is DERIVED:
        #     TriangularMultiplication(layer, "incoming", <2-D mesh>, dm)   # no flags
        #       -> _hybrid_ib_request is None (the public class never sets it)
        #       -> hybrid_ib := has_ib_peers  -> True on any cross-node job     (:2820)
        #       -> _ib_kw = dict(ib_drain=True, ib_wide=True, ...)            # unconditional, below
        #       -> incoming + cp1 > 1 -> incoming_store_variant -> "route2_ni" (:3926)
        # `hybrid_ib=False` with IB peers is REFUSED (it CUDA-faults), so a caller cannot opt out.
        # It also carries shipped numbers: four cells of the published cp16 2-D cross-node incoming
        # ladder are exactly this combination, at 3.4x-4.9x.
        #
        # TWO CORRECTIONS TO AN EARLIER VERSION OF THIS COMMENT, both of which overstated:
        #   1. It cited ":981" as the public kwarg. `:981` is `DualGatedGemmDistStore`, an internal
        #      store builder -- not the public API. The real gate is the auto-detect at `:2820`.
        #   2. It called this "the route2_ni x ib_wide 2-D cross-node HANG". That description does
        #      not survive its own evidence. The wedge was localized to a 16-rank in-process SWEEP:
        #      the wedging cell run ALONE passes, and the per-rank stacks put 6 ranks in the
        #      symmetric ALLOCATION and 10 in the barrier waiting for them, with `run_fn()` never
        #      reached by any rank -- so no kernel ran and it cannot be an in-kernel deadlock.
        #      Separately, the test that named it exercises route2_ni at `cp1 == 1`, because it reads
        #      `dist_manager.device_mesh`, which is FLAT (the 2-D grid lives in
        #      `_device_mesh_subgroups`). So the combination is production-reachable, but the hang
        #      that was attributed to it is neither 2-D nor in the wide drain.
        #
        # What stands: the combination is reachable by default and carries shipped performance, so
        # a defect found in it would fall under CLAUDE.md's production-path rule. Do NOT pre-emptively
        # guard the combo off here -- a scoped-reject is not a resolution, and the maintainers rejected that
        # exact shape once already (the B>1 demotion).
        _ib_kw = {}
        if self._hybrid_ib:
            _ib_kw = dict(
                ib_drain=True, ib_wide=True, ib_wide_batch=self._ib_wide_batch,
                consumer_warpgroups=self._ib_consumer_warpgroups, ring_depth=self._ib_ring_depth,
            )

        def _build_configured_front_gemm():
            """Build a `DualGatedGemmDistSm90` front store and run the configure the caller selected.

            Returns a FRESH, fully configured functor every call. Purely a factory -- it reads the
            enclosing scope and mutates nothing outside the object it builds, so two calls give two
            independent instances that agree on every compile-time parameter and every
            post-construction attribute.

            EXISTS FOR THE MLIR PROGRAM KEY. `to_precompiled_mlir` TRACES the functor, and this
            tree's `template_params` guard refuses a second trace of an already-traced instance --
            so the key must be taken from an instance the caller never compiles. A factory is the
            only way to hand `compile_nvshmem` a throwaway; passing the live object would either
            trip the guard or consume the one that is about to be compiled.

            Input requirements: none of its own -- every value it reads (`a_dtype`, `tile_M`,
            `tile_N`, `pingpong`, `is_persistent`, `K`, `M`, `rows_per_peer`, `pe_table`, `pm`,
            `route2_ni`, `composite_k`, `dynamic_shape`, `_ib_kw`) is already bound in the enclosing
            scope when this is defined. Calling it before those exist raises `NameError`.

            Returns: a configured `DualGatedGemmDistSm90`. Raises `AssertionError` on the
            composite_k branch if configure's rows-per-peer override disagrees with the extent the
            recv was sized from.
            """
            # SINGLE-PHASE CONSTRUCTION. The upstream set eight attributes AFTER building the functor;
            # three of them -- chunk_g, n_dual_tiles, gate3_n3 -- are COMPILE-TIME TEMPLATE PARAMETERS in
            # this tree and are refused post-construction:
            #
            #   AttributeError: DualGatedGemmDistSm90.chunk_g is a compile-time template parameter and is
            #   immutable after construction. It is folded into the kernel as a constant, so reassigning
            #   it cannot affect an already-compiled kernel and desynchronizes the functor from it.
            #
            # That guard is this tree's, not the upstream's, and it is right: the upstream's spelling
            # worked only because nothing re-read the value after compile. The three move into the
            # constructor; the rest are ordinary attributes and stay where they were.
            #
            # `_has_gate3` is NOT set at all any more -- here it is a read-only PROPERTY derived from
            # gate3_n3, so assigning it would shadow the derivation with a stale constant. Passing
            # gate3_n3=0 is what makes it False, which is the value the upstream assigned by hand.
            gemm_obj = DualGatedGemmDistSm90(
                Float32,
                a_dtype,
                (tile_M, tile_N),
                (1, 1, 1),
                pingpong=pingpong,
                is_persistent=is_persistent,
                chunk_g=1,
                n_dual_tiles=0,
                gate3_n3=0,
            )
            gemm_obj._gemm_K = K
            gemm_obj.STATS_TPR = 2
            gemm_obj._do_normalize = False  # pre-normalized x_norm (no internal LN)
            gemm_obj._stats_mode = "streaming"
            # pe_table materialized ABOVE (pre-recv, #B fix) -- reused here for configure_a2a + the ring pe_dev_t.
            # configure_a2a_sharded covers 1-D AND 2-D (recv-index 2-D-invariant); for the
            # wave-1 1-D headline the flat-cp configure_a2a is the cp_axis_sizes==(cp,) case.
            # dynamic_shape marks: operands + postact are mark_layout_dynamic; recv is
            # mark_compact_shape_dynamic(mode=1) (feature dim 2*Dloc STAYS static -> int(recv.shape[0])
            # in the epi stays valid; only the token dim M_full is runtime). Identity when static.
            if route2_ni:
                # transpose_in walk (i_loc inner) + token_grid (B, Xg=N_i_loc, Yg=N_j_loc); engage the
                # N_i-stride-1 producer store. configure OVERRIDES _a2a_rows_per_peer to the PADDED count
                # (the recv above is already sized from the same Xg_pad, so they agree).
                gemm_obj.configure_a2a(
                    cp=self.cp,
                    my_cp_rank=int(pm.my_cp_rank),
                    rows_per_peer=rows_per_peer,
                    pe_table=pe_table,
                    transpose_in=True,
                    token_grid=(self._route2_B, self._route2_Xg, self._route2_Yg),
                    dynamic=dynamic_shape,
                    b_plane=self._b_mode,
                    b_dynamic=self._b_dynamic,
                    **_ib_kw,  # §10 R1: route2_ni COMPOSES with ib_drain/ib_wide (layout ⊥ transport)
                )
                gemm_obj._a2a_route2_ni = True
                # B10 2-D: hand the store copy_fn the (cp0, cp1) token-axis decomposition. The FEATURE
                # peer-table + atoms stay FLAT-cp (configure_a2a set (cp,) above — 2-D-invariant); this
                # overrides _a2a_cp_axis_sizes ONLY for the token dst_i (cp0_coord*Xg_pad) / dst_j
                # (cp1_coord*N_j_loc) addressing. 1-D (cp1==1) => (cp,1) => cp0_coord==rank, cp1_coord==0.
                gemm_obj._a2a_cp_axis_sizes = (self.cp0, self.cp1)
            elif composite_k:
                # COMPOSITE-K: the SAME transpose_in configure as route2_ni (engages the padded i_loc-inner
                # walk via _remap_A_operand_layout, gated only on _transpose_in) but WITHOUT _a2a_route2_ni,
                # so the plain D-MAJOR postact store runs (config-mostly, §9.3). configure OVERRIDES
                # _a2a_rows_per_peer to the padded count -> assert it equals the extent we sized the recv from.
                gemm_obj.configure_a2a(
                    cp=self.cp,
                    my_cp_rank=int(pm.my_cp_rank),
                    rows_per_peer=rows_per_peer,
                    pe_table=pe_table,
                    transpose_in=True,
                    token_grid=(self._route2_B, self._route2_Xg, self._route2_Yg),
                    dynamic=dynamic_shape,
                    b_plane=self._b_mode,
                    b_dynamic=self._b_dynamic,
                    **_ib_kw,  # §10 R1: transpose_in COMPOSES with ib_drain/ib_wide (layout ⊥ transport)
                )
                assert int(gemm_obj._a2a_rows_per_peer) == self._comp_rpp_padded, (
                    f"composite_k rpp mismatch: configure override "
                    f"{int(gemm_obj._a2a_rows_per_peer)} != sized {self._comp_rpp_padded}"
                )
            else:
                gemm_obj.configure_a2a(
                    cp=self.cp,
                    my_cp_rank=int(pm.my_cp_rank),
                    rows_per_peer=rows_per_peer,
                    pe_table=pe_table,
                    dynamic=dynamic_shape,
                    # BUG A (outgoing / plain-front dynamic-anchor straddle): _a2a_should_clamp is const_expr-
                    # BAKED at the anchor compile. Unlike the BACK store (straddle anchor), this FRONT store
                    # anchors a dynamic instance at its FIRST-seen N; if that N is on-grid (rows_per_peer %
                    # tile_M == 0) it bakes should_clamp=FALSE, so a later off-grid runtime N (M=B*N_i_loc*
                    # N_j_loc, M%tile_M!=0, e.g. N1000) overshoots rows_per_peer -> outgoing mis-deliver
                    # (rel_L2~1.0, nomask AND masked). Force the partial-token clamp ON for dynamic_shape so
                    # the clamp branch is baked (byte-identical no-op when aligned: the relative-block atom
                    # drops nothing at rem==0; only the off-grid overshoot is clamped). Static builds anchor
                    # AT their shape so the auto rows_per_peer%tile_M path already covers them (pass-through).
                    partial_token_clamp=dynamic_shape,
                    # pad_inner: rows_per_peer above is ALREADY rpp_eff (the padded extent the recv was
                    # sized from). inner_extent is the runtime WALK extent Yg (N_j_loc in the pad regime,
                    # M in the decline regime) and token_count is the TRUE unpadded A row count M, from
                    # which the kernel derives Xg = M//Yg. Under dynamic_shape the walk extent is re-sent
                    # every launch via EpilogueArguments.token_grid_yg, so this static pair only anchors
                    # the compile. pad_inner=False => every argument below is inert (byte-identical).
                    pad_inner=self._pad_inner,
                    inner_extent=(self._pi_yg if self._pad_inner else None),
                    token_count=(M if self._pad_inner else None),
                    **_ib_kw,
                )
            return gemm_obj

        self._make_gemm_obj = _build_configured_front_gemm
        gemm_obj = _build_configured_front_gemm()
        self._gemm_obj = gemm_obj  # route2_ni dynamic: dynamic_token_grid_yg(N, M) per launch

        max_active_clusters = get_max_active_clusters(1)
        # HYBRID IB-drain ring + device PE table (only when hybrid_ib -> configure set ib_drain +
        # decoupled). The decoupled producer stages each postact subtile into a bounded SYMMETRIC-heap
        # GMEM ring; the consumer warpgroup drains ring->peer recv via put_warp routed by pe_table_dev.
        # Ring slot box = (epi_n_postact, epi_m) with epi_m INNERMOST (stride-1) — epi_m is the recv
        # stride-1 axis in BOTH layouts (token D-major / N_i route2_ni), W· wider on ib_wide (frugal,
        # O(1) in N_token). Mirrors tests/distributed/test_front_a2a_staged._build_staged_front_decoupled.
        # Default OFF => None => the coupled epi (byte-identical). Required whenever _a2a_decoupled (the
        # host epi_to_underlying_arguments raises on a None ring), incl. an all-P2P collapse (unused there).
        self._cute_ring, self._cute_pe_dev, self._ib_ring_t, self._ib_pe_dev_t = None, None, None, None
        if self._hybrid_ib:
            import math as _math
            epi_m = _math.gcd(128, tile_M)
            epi_n_postact = _math.gcd(32, tile_N) // 2
            ring_w = self._ib_wide_batch if bool(getattr(gemm_obj, "_a2a_ib_wide", False)) else 1
            ring_t = _symmetric_empty(
                (max_active_clusters, self._ib_ring_depth, epi_n_postact, epi_m * ring_w), dtype=dt
            )
            ring_t.zero_()
            pe_dev_t = torch.tensor(list(pe_table), device=Wg2.device, dtype=torch.int32).contiguous()
            self._ib_ring_t, self._ib_pe_dev_t = ring_t, pe_dev_t
            self._cute_ring = from_dlpack(ring_t, assumed_align=16)
            self._cute_pe_dev = from_dlpack(pe_dev_t, assumed_align=4)

        def _mk_epi():
            # Reads the CURRENT self.recv (rebindable in dynamic_shape) + the anchor PostAct_p.
            # Used for the anchor COMPILE and the STATIC run; the dynamic run builds a per-M epi.
            return DualGatedGemmDistSm90.EpilogueArguments(
                mPostAct=_md(from_dlpack(PostAct_p, assumed_align=16)),
                act_fn=gate_fn_map["glu"],
                mRowVecBroadcast=self._cute_rowvec,
                mBiasUp=None,
                mBiasGate=None,
                # has_mask: the CURRENT (1, M) fp32 mask (anchor at compile, runtime mask in run()).
                mMaskColVec=(
                    _md(from_dlpack(self._cur_mask, assumed_align=16)) if self._has_mask else None
                ),
                mPostAct3=None,
                act_fn_3=None,
                mRowVecBroadcast3=None,
                mWeight=None,
                mBias=None,
                eps=Float32(eps),
                rounding_mode=RoundingMode.RN,
                recv=_md_recv(from_dlpack(self.recv, assumed_align=16)),
                ring=self._cute_ring,            # None unless hybrid_ib -> coupled epi (byte-identical)
                pe_table_dev=self._cute_pe_dev,  # None unless hybrid_ib
                token_grid_yg=self._token_grid_yg_arg(gemm_obj, dynamic_shape),
                token_grid_b=self._token_grid_b_arg(gemm_obj, dynamic_shape),
            )

        self._mk_epi = _mk_epi
        self.scheduler_args = make_scheduler_args(max_active_clusters, 8, None)
        self.stream = cutlass_torch.current_stream()
        # Persist the constant perm'd B operand for the run path (A is rebuilt per call).
        self._B_p = B_p
        self._M, self._K = M, K
        # Cache the SYMMETRIC backing (route2_ni: recv_buf, the (2*Dloc,[B,]N_j,N_i) tensor whose
        # permuted VIEW is self.recv; plain: self.recv itself) so free() frees the real nvshmem tensor.
        #
        # The KEY must be the one `rebind_M` looks up, and for the two transpose_in variants that is
        # `(B, M)`, not `M`. Seeding under a bare `M` there would MISS on the very first rebind at the
        # anchor shape: the store would allocate a second symmetric recv of the same size, and the
        # anchor's would sit unreachable until `free()` -- a silent doubling of the largest allocation
        # this class makes, on the path that exists to avoid exactly that.
        self._recv_cache[
            (self._route2_B, M) if (self._route2_ni or self._composite_k) else M
        ] = self._recv_buf if self._route2_ni else self.recv
        compiled = compile_nvshmem(
            gemm_obj,
            _md(from_dlpack(A_p, assumed_align=16)),
            _md(from_dlpack(B_p, assumed_align=16)),
            None,
            None,
            _mk_epi(),
            self.scheduler_args,
            # NO varlen_args slot. `main`'s __call__ is
            #   (mA, mB, mD, mC, epilogue_args, scheduler_args, varlen_args, stream, mB2, mB3)
            # and ours is
            #   (mA, mB, mD, mC, epilogue_args, scheduler_args,              stream, mB2)
            # -- extraction dropped `varlen_args` and `mB3`. Leaving the upstream's positional None
            # in place shifts every later argument by one, and the DSL reports it as a TYPE error
            # about the argument that ended up wrong rather than as an arity error:
            #   expects argument #9 (mB2) to be a Tensor or None, but got CUstream
            self.stream,
            op_factory=self._make_gemm_obj,   # M4c: the MLIR program key traces a FRESH instance
            register=True,
            # M6: share this compile with any later engine in this process whose front-store
            # configuration and operand layouts match. The key is the functor's `compile_key()` --
            # which now covers the post-`configure_a2a` surface -- plus the operand layouts, so a
            # second engine that differs in ANY of them still compiles its own.
            reuse=True,
        )
        self._compiled = compiled

    def _alloc_route2_recv(self, N_i, N_j, B, dt):
        """Allocate (or re-shape) the route2_ni N_i-stride-1 symmetric recv for one token geometry.

        Purpose
            ONE place that decides the recv's RANK, so the constructor and ``rebind_M`` cannot drift
            -- they allocate the same layout from the same predicate.

        Semantics
            Returns ``(backing, presented)``. The backing is the CONTIGUOUS symmetric buffer, always
            with ``N_i`` innermost (that is the entire point of route2_ni: global-i stride-1 so the
            back einsum reads ``a_major="k"`` with no copy). The presented view swaps the two token
            axes so the store sees ``(2*Dloc, [B,] N_i, N_j)``.

            * ``_b_mode`` off: ``(2*Dloc, N_j, N_i)`` -> ``(2*Dloc, N_i, N_j)`` -- byte-identical.
            * ``_b_mode`` on: ``(2*Dloc, B, N_j, N_i)`` -> ``(2*Dloc, B, N_i, N_j)``. The batch mode
              sits between the feature and the token axes because the back operand's ``L`` must
              collapse ``(Dloc, B)`` into ONE stride, which requires ``Dloc``'s stride to be exactly
              ``B x`` ``B``'s -- true only when they are the two outermost modes.

        Input requirements
            ``N_i`` is the PADDED full global-i extent (``cp0 * Xg_pad``) and ``N_j`` the full global-j
            extent (``cp1 * Yg``), both positive; ``B >= 1``; ``dt`` a torch dtype. Passing an
            unpadded ``N_i`` silently mis-places every rank's i-block (no error, wrong output).

        Returns
            ``(torch.Tensor, torch.Tensor)`` -- both symmetric-heap views of the same allocation.
            The backing MUST be retained by the caller: it owns the symmetric buffer.
        """
        if self._b_mode:
            buf = _symmetric_empty((2 * self.Dloc, int(B), N_j, N_i), dtype=dt)
            return buf, buf.permute(0, 1, 3, 2)  # (2*Dloc, B, N_i, N_j) N_i stride-1
        buf = _symmetric_empty((2 * self.Dloc, N_j, N_i), dtype=dt)
        return buf, buf.permute(0, 2, 1)  # (2*Dloc, N_i, N_j) N_i stride-1

    def _alloc_composite_recv(self, rpp_padded, B, dt):
        """Allocate the composite_k per-rank-contiguous D-major symmetric recv for one geometry.

        Purpose
            The rank-deciding twin of :meth:`_alloc_route2_recv`, for the composite read.

        Semantics
            ``rpp_padded`` is the per-peer token extent over ALL planes (``B*Yg*Xg_pad``), the same
            number ``configure_a2a`` overrides ``_a2a_rows_per_peer`` to. The ALLOCATION SIZE is
            ``2*Dloc x cp*rpp_padded`` either way -- only the arrangement changes:

            * ``_b_mode`` off: 2-D ``(2*Dloc, cp*rpp_padded)``, slot-major then plane -- byte-identical.
            * ``_b_mode`` on: 3-D ``(2*Dloc, B, cp*rpp_padded//B)``, PLANE-major then slot. The
              inversion is what makes the back operand's ``L=(Dloc,B)`` a single stride and makes
              ``_composite_remap``'s cp stride (``M * stride(M)``) come out right with NO kernel
              change; with the slot outside the plane no flat ``L`` exists at any ``cp > 1``.

        Input requirements
            ``rpp_padded`` must be a positive multiple of ``B`` (it is ``B*Yg*Xg_pad`` by
            construction); a non-multiple raises rather than silently truncating a plane.

        Returns
            ``torch.Tensor`` -- the symmetric recv, 2-D or 3-D per the predicate above.
        """
        if not self._b_mode:
            return _symmetric_empty((2 * self.Dloc, self.cp * int(rpp_padded)), dtype=dt)
        B = int(B)
        if B <= 0 or int(rpp_padded) % B != 0:
            raise ValueError(
                f"composite_k b-mode recv: rows_per_peer={rpp_padded} must be a positive multiple "
                f"of B={B} (it is B*Yg*Xg_pad by construction)."
            )
        return _symmetric_empty(
            (2 * self.Dloc, B, self.cp * (int(rpp_padded) // B)), dtype=dt
        )

    def pad_inner_yg_for(self, N_i_loc, N_j_loc):
        """The pad_inner WALK extent ``Yg`` for a token geometry, computed WITHOUT mutating state.

        Needed because this store's ``rebind_M`` is DEFERRED into ``TriMulAutotuned.front_a2a`` (the F4 heap
        fix: an incoming module must never allocate the plain front's per-N recv), so ``self._pi_yg`` is
        still the PREVIOUS call's value at the point ``forward`` builds the mask col-vec. Reading the
        cached field there silently reshapes the mask with a stale ``Yg`` — a wrong-output bug that only
        fires once a second N is seen. Callers that run BEFORE the rebind must use this."""
        return front_pad_inner_geometry(
            self._route2_B, int(N_i_loc), int(N_j_loc), self._route2_tile_M
        )[0]

    def _token_grid_yg_arg(self, gemm_obj, dynamic_shape):
        """The runtime inner-walk extent for ``EpilogueArguments.token_grid_yg`` — ONE seam, three walks.

        * ``route2_ni`` / ``composite_k`` (transpose_in): ``Yg = N_j_loc``, the native-INNER axis the
          transposed walk holds fixed per M-tile.
        * ``pad_inner`` (P2): ``Yg = self._pi_yg``, the walk's inner extent for THIS runtime N — either
          ``N_j_loc`` (pad regime) or ``M`` (decline regime). Recomputed by ``front_pad_inner_geometry``
          on every ``rebind_M``, which is what lets one dynamic compile serve both regimes.
        * plain static / no variant: ``None`` (byte-identical — the field is unread).
        """
        if not dynamic_shape:
            return None
        if self._route2_ni or self._composite_k:
            return gemm_obj.dynamic_token_grid_yg(self._route2_Yg, self._M, B=self._route2_B)
        if self._pad_inner:
            return gemm_obj.pad_inner_token_grid_yg(self._pi_yg, self._M)
        return None

    def _token_grid_b_arg(self, gemm_obj, dynamic_shape):
        """The runtime token-BATCH extent for ``EpilogueArguments.token_grid_b`` -- the batch twin of
        :meth:`_token_grid_yg_arg`.

        Returns ``None`` for every store that bakes B (plain, pad_inner, static, and any
        ``b_dynamic=False`` transpose_in build), which keeps those launches byte-identical: the field
        stays unset and nothing reads it. For a ``b_dynamic`` transpose_in build it returns
        ``Int32(self._route2_B)`` -- the extent ``rebind_M`` last bound, which is the same number the
        recv was sized from, so the walk and the buffer cannot disagree.

        Args:
            gemm_obj: the configured front functor (it owns the bake-vs-runtime decision and raises
                if asked to run a baked build at a different batch).
            dynamic_shape: this store's dynamic-N flag; False short-circuits to ``None``.

        Returns:
            ``Int32`` or ``None``.
        """
        if not dynamic_shape:
            return None
        if not (self._route2_ni or self._composite_k):
            return None
        return gemm_obj.dynamic_token_grid_b(self._route2_B)

    def run(self, x_norm_2d: torch.Tensor, mask_col: torch.Tensor = None) -> torch.Tensor:
        """x_norm_2d (M, K) bf16 PRE-NORMALIZED token block -> stores resharded a,b into recv.

        The kernel gates ``glu(x_norm@Wg^T, x_norm@Wp^T)`` (NO internal LN — _normalize=
        False) and the D-major postact store reshards both halves by feature into the peer recv.
        dynamic_shape rebinds the runtime-M operands + a per-M postact + the (rebound) recv, all
        mark_layout_dynamic; the static path reuses the anchor epi. Returns the recv (2*Dloc, M_full).

        ``mask_col`` (has_mask builds only): the runtime ``(1, M)`` fp32 per-row mask, multiplied
        into the gated glu postact before the peer store (the ``_local_front`` mask). MUST be
        provided iff the store was built ``has_mask=True`` (the epilogue mask branch is baked).
        """
        if self._has_mask:
            if mask_col is None:
                raise ValueError("front store built has_mask=True but run() got mask_col=None.")
            self._cur_mask = mask_col  # rebind the runtime mask (read by _mk_epi + the dynamic epi)
        elif mask_col is not None:
            raise ValueError("front store built has_mask=False cannot apply a runtime mask_col.")
        # A perm3d = the (1,M,K) -> (M,K,1) transpose (perm3d's .permute(1,2,0)); a VIEW.
        A_p = x_norm_2d.reshape(1, self._M, self._K).permute(1, 2, 0)  # (M, K, 1)
        if self._dynamic_shape:
            # per-M postact (2D, M) -> the M-major (M, 2D, 1) perm the kernel writes; recv rebound by
            # rebind_M. Marks: operands + postact dynamic, recv compact-dynamic(mode=1) (feature fixed).
            out_postact = torch.empty(2 * self.D, self._M, device=x_norm_2d.device, dtype=self.dt)
            PostAct_p = out_postact.mT.unsqueeze(0).permute(1, 2, 0)  # (M, 2D, 1)
            epi = DualGatedGemmDistSm90.EpilogueArguments(
                mPostAct=self._md(from_dlpack(PostAct_p, assumed_align=16)),
                act_fn=gate_fn_map["glu"], mRowVecBroadcast=self._cute_rowvec,
                mBiasUp=None, mBiasGate=None,
                mMaskColVec=(
                    self._md(from_dlpack(mask_col, assumed_align=16)) if self._has_mask else None
                ),
                mPostAct3=None, act_fn_3=None, mRowVecBroadcast3=None,
                mWeight=None, mBias=None, eps=Float32(self.eps), rounding_mode=RoundingMode.RN,
                recv=self._md_recv(from_dlpack(self.recv, assumed_align=16)),
                ring=self._cute_ring, pe_table_dev=self._cute_pe_dev,  # None unless hybrid_ib
                token_grid_yg=self._token_grid_yg_arg(self._gemm_obj, True),
                token_grid_b=self._token_grid_b_arg(self._gemm_obj, True),
            )
        else:
            epi = self._mk_epi()
        self._compiled(
            self._md(from_dlpack(A_p, assumed_align=16)),
            self._md(from_dlpack(self._B_p, assumed_align=16)),
            None,
            None,
            epi,
            self.scheduler_args,
            None,
            self.stream,
        )
        return self.recv

    def rebind_M(self, M: int, *, N_i_loc=None, N_j_loc=None, B=None) -> None:
        """dynamic_shape: point self.recv at the (cached) symmetric recv for this runtime M (allocate
        on first sight). SPMD-collective — all ranks call with the SAME M sequence. run() reads self._M.

        route2_ni: the recv is the 3-D N_i-stride-1 backing; the caller MUST also pass this N's
        (N_i_loc, N_j_loc) so the padded 3-D (2*Dloc, N_j, N_i) recv + the token grid (Xg,Yg) update.

        ``B`` is THIS shape's token-batch extent (route2_ni / composite_k only). ``None`` keeps the
        current one, which is what every N-only rebind wants. It is a parameter because the batch is
        a RUNTIME input exactly like N: it multiplies the recv's token extent and it sets the walk's
        outermost mode, so a store left at the constructor's B sizes the buffer for a different batch
        than the kernel walks -- the recv comes out an exact factor of ``B_ctor/B_runtime`` too small
        and the reshape in the back operand raises with a size that is a clean multiple of the right
        one. The cache key is ``M = B*N_i_loc*N_j_loc``, which already separates two batches at one N.
        """
        assert self._dynamic_shape, "rebind_M requires dynamic_shape=True"
        b_changed = B is not None and int(B) != self._route2_B
        if b_changed:
            if not self._b_dynamic and (self._route2_ni or self._composite_k):
                raise ValueError(
                    f"this front store baked B={self._route2_B} and rebind_M was asked for B={int(B)}. "
                    f"Build it with b_dynamic=True so ONE compile serves every batch; a baked B "
                    f"unravels the transpose_in walk with the wrong Xg and delivers another plane."
                )
            self._route2_B = int(B)
        # (B, M) keys the cache, and the early return tests BOTH. M = B*N_i_loc*N_j_loc alone is not
        # injective once B moves -- (B=1, N) and (B=2, N') can land on the same M with different token
        # geometries, and an M-only key would then hand the second shape the first one's buffer.
        key = (self._route2_B, M)
        if self._route2_ni:
            assert N_i_loc is not None and N_j_loc is not None, "route2_ni rebind needs (N_i_loc,N_j_loc)"
            if M == self._M and not b_changed and self.recv is not None:
                return
            tile_M = self._route2_tile_M
            self._route2_Xg, self._route2_Yg = int(N_i_loc), int(N_j_loc)
            self._route2_Xg_pad = ((self._route2_Xg + tile_M - 1) // tile_M) * tile_M
            N_i = self.cp0 * self._route2_Xg_pad  # padded FULL global-i (all cp0 i-blocks)
            N_j = self.cp1 * self._route2_Yg  # full global-j extent (== Yg==N when 1-D)
            self._route2_N_i, self._route2_N_j = N_i, N_j
            cached = self._recv_cache.get(key)
            if cached is None:
                buf, view = self._alloc_route2_recv(N_i, N_j, self._route2_B, self.dt)
                _recv_cache_put(self._recv_cache, key, buf, _symmetric_free)
            else:
                buf = cached
                view = buf.permute(0, 1, 3, 2) if self._b_mode else buf.permute(0, 2, 1)
            self._recv_buf, self.recv = buf, view
            self._M = M
            return
        if self._composite_k:
            # COMPOSITE-K dynamic-N: rebind the 2-D PADDED per-rank-contiguous recv for THIS N. Needs
            # (N_i_loc, N_j_loc) to recompute Xg_pad (the padding is per-N). recv col = cp*rpp_padded.
            assert N_i_loc is not None and N_j_loc is not None, "composite_k rebind needs (N_i_loc,N_j_loc)"
            if M == self._M and not b_changed and self.recv is not None:
                return
            tile_M = self._route2_tile_M
            self._route2_Xg, self._route2_Yg = int(N_i_loc), int(N_j_loc)
            self._route2_Xg_pad = ((self._route2_Xg + tile_M - 1) // tile_M) * tile_M
            self._comp_rpp_padded = self._route2_B * self._route2_Yg * self._route2_Xg_pad
            r = self._recv_cache.get(key)
            if r is None:
                r = self._alloc_composite_recv(self._comp_rpp_padded, self._route2_B, self.dt)
                _recv_cache_put(self._recv_cache, key, r, _symmetric_free)
            self.recv = r
            self._M = M
            return
        rpp = M
        if self._pad_inner:
            # dynamic-N: re-run the guard for THIS N (it is a per-N decision) and size the recv from
            # the resulting per-peer extent. Needs (N_i_loc, N_j_loc) — the caller passes them; falling
            # back to the ctor anchor would silently size the recv for the WRONG N. rpp_eff == M
            # whenever the guard declines, so the allocation is then identical to the plain one.
            # BEFORE the same-M early return: the cached geometry must never lag the requested shape.
            assert N_i_loc is not None and N_j_loc is not None, "pad_inner rebind needs (N_i_loc,N_j_loc)"
            self._route2_Xg, self._route2_Yg = int(N_i_loc), int(N_j_loc)
            self._pi_yg, self._pi_rpp, self._pi_nj_pad = front_pad_inner_geometry(
                self._route2_B, self._route2_Xg, self._route2_Yg, self._route2_tile_M
            )
            rpp = self._pi_rpp
        if M == self._M and self.recv is not None:
            return
        r = self._recv_cache.get(M)
        if r is None:
            r = _symmetric_empty((2 * self.Dloc, self.cp * rpp), dtype=self.dt)
            _recv_cache_put(self._recv_cache, M, r, _symmetric_free)
        self.recv = r
        self._M = M

    def free(self):
        try:
            self._compiled.free()
        except Exception:
            pass
        for r in self._recv_cache.values():
            try:
                _symmetric_free(r)
            except Exception:
                pass
        self._recv_cache.clear()
        if self._ib_ring_t is not None:  # symmetric-heap IB drain ring (hybrid_ib only)
            try:
                _symmetric_free(self._ib_ring_t)
            except Exception:
                pass
            self._ib_ring_t = None
        # Break the closure cycle. `_make_gemm_obj` / `_mk_epi` are nested defs assigned to self,
        # so they close over `__init__`'s scope and the store is reachable from its own attribute:
        # store -> closure -> cell -> store. MEASURED: without this the store survives `free()` and
        # its symmetric buffers are released only on the next CYCLIC collection, which pytest never
        # forces -- six build/free cycles took the caching allocator 296 -> 1852 MiB and left 5
        # stores alive. The MemPool itself recycles correctly; it was Python holding the reference.
        self._make_gemm_obj = None
        self._mk_epi = None
        self._gemm_obj = None
        self._compiled = None
        self.epi_args = None


# --------------------------------------------------------------------------- #
# BACK fused store — the design-E GEMM-native einsum store (lifted from the
# validated trash_to_be_removed/t32_e2e_design_e.py::FusedBackStore + the kernel's
# own configure_a2a_gemm_native). Compiles GemmA2ASm90 ONCE; re-runs into a
# persistent 5-D symmetric recv. The einsum's D-store IS the back A2A.
# --------------------------------------------------------------------------- #
class GemmA2AStore:
    """Reusable fused einsum-store: a_dm,b_dm (Dloc,M) -> 5-D recv (cp,Dloc,B,N_i_loc,N_j_loc).

    ``back_store`` selects the A2A store variant (both compute the SAME reshard — the einsum's
    D-store IS the back A2A, routing each ``(i, j)`` tile of plane ``L = d*B+b`` to its token-i
    peer, the S3 -> S1 token reshard):

    * ``"pe_aligned"`` (DEFAULT): the committed §0.6.2 pe_aligned per-peer tiling store
      (``arbitrary_n`` + ``pe_aligned_tiling``), compiled DYNAMIC (``mark_layout_dynamic``) at a
      STRADDLE anchor so ONE compile serves any token count (a partial-last per-peer tile is
      TMA-clamped). The compile anchor N_loc MUST straddle (``% cta_tile_M != 0``) or
      ``arbitrary_n`` auto-reduces OFF and pe_aligned goes inert (silently == design-E) — a nearby
      straddle anchor is used when the real N_loc is a multiple of the tile. run() re-binds THIS
      N's dynamic operands + a FRESH dynamic epilogue (the one-compile-many-shapes seam, mirror
      benchmark/distributed/pe_aligned_dyn_validate.py + back_a2a_autotune_adapter.BackA2APeDynAdapter).
    * ``"design_e"`` (fallback, default-off): the GEMM-native design-E TMA-S2G store (a CTA output
      tile maps to ONE peer -> requires each per-peer block CTA-tile-aligned). Byte-identical to the
      pre-rewire path; compiled STATIC per shape, lazily per direction.

    Compiled PER DIRECTION either way (the ``incoming`` operand transpose bakes a different operand
    major-ness into the bitcode descriptor). The recv is allocated as symmetric memory ONCE.
    """

    def __init__(
        self,
        pm: PeMap,
        B,
        N,
        D,
        dt,
        *,
        tile_shape_mn=(128, 128),
        cluster_shape_mnk=(1, 1, 1),
        pingpong=False,
        is_persistent=True,
        device_mesh=None,
        placements=None,
        back_store="pe_aligned",
        dynamic_shape=False,
        route2_ni=False,
        composite_k=False,
        hybrid_ib=False,
        cluster_n=None,
        ring_depth=2,
    ):
        import nvshmem.core  # noqa: F401 (availability probe)
        import nvshmem.core.interop.torch as nvshmem_torch

        assert back_store in ("pe_aligned", "design_e"), f"back_store={back_store!r}"
        # HYBRID NVLink+IB back A2A (opt-in): the cluster_multislot differential drain (the SOLE
        # production IB back drain — decoupled putwarp ring + is_p2p auto-routing). ON => the pe_aligned
        # GEMM-native store rides the cluster_multislot stack; an all-P2P (single-node cp<=8) job
        # const_expr-collapses it back to the pe_aligned NVLink TMA store (has_ib_peers=False,
        # gemm_sm90_a2a.py:778) = byte-identical. Requires pe_aligned (the decoupled stack rides it);
        # design_e (static, no ring) cannot carry it. Default OFF => the ib kwargs are NEVER passed =>
        # configure byte-identical to today.
        self._hybrid_ib = bool(hybrid_ib)
        if hybrid_ib and back_store != "pe_aligned":
            raise ValueError("hybrid_ib=True requires back_store='pe_aligned' (cluster_multislot rides "
                             "the pe_aligned decoupled stack; design_e is static with no ring).")
        # cluster_n (drain concentration) MUST equal the GEMM N-cluster cluster_shape_mnk[1] (asserted in
        # configure_a2a_gemm_native:807 — A-multicast width == drain width). None => take it from the
        # cluster (1-D: 2 for the 2040 fix; 2-D: 1, forced (1,1,1)). ring_depth = the rotating-ring depth.
        self._ib_cluster_n = int(cluster_n) if cluster_n is not None else int(cluster_shape_mnk[1])
        self._ib_ring_depth = int(ring_depth)
        # ROUTE-2 (A) incoming no-copy: the incoming a/b operands are the front's 3-D N_i-stride-1 recv
        # halves (Dloc, N_i, N_j); _operands feeds a permute-VIEW (N_j=M, N_i=K stride-1, Dloc=L) =
        # a_major="k" NATIVE (no .transpose(-1,-2).contiguous()). OUTGOING is UNCHANGED. Requires
        # pe_aligned (the mark_layout_dynamic route handles the runtime K=N_i extent).
        self._route2_ni = bool(route2_ni)
        if route2_ni:
            assert back_store == "pe_aligned", "route2_ni incoming requires back_store='pe_aligned'"
        # COMPOSITE-K incoming (§9): the back reads K=(cp,Xg_pad) from a per-rank-CONTIGUOUS D-major recv
        # (the front's transpose_in+D-major store). The composite K-hoist (gemm_sm90_a2a._composite_remap
        # + _k_tile_cnt) synthesizes the flat-cp mode -> 1-D (cp1==1) + B==1 only (matches the validated
        # back-read test_composite_k_backread.py). Mutually exclusive with route2_ni. pe_aligned only.
        self._a2a_composite_k = bool(composite_k)
        if composite_k:
            assert back_store == "pe_aligned", "composite_k incoming requires back_store='pe_aligned'"
            if route2_ni:
                raise ValueError("route2_ni and composite_k are mutually-exclusive incoming variants.")
        # dynamic_shape (the TriMulAutotuned dynamic-N mode): ONE executor serves many token counts via
        # rebind_N (per-N recv realloc, cached SPMD-lockstep). Requires pe_aligned (the mark_layout_
        # dynamic route); design_e compiles static per shape and CANNOT rebind. Distinct from
        # self._dynamic (pe_aligned's mark_layout_dynamic, on even for a per-shape TriMulAutotuned).
        if dynamic_shape:
            assert back_store == "pe_aligned", "dynamic_shape back store requires back_store='pe_aligned'"
        self._dynamic_shape = dynamic_shape
        self._recv_cache = {}
        self.back_store = back_store
        self._dynamic = back_store == "pe_aligned"  # pe_aligned rides the mark_layout_dynamic route
        self._cluster_shape_mnk = cluster_shape_mnk
        self.pm, self.B, self.N, self.D, self.dt = pm, B, N, D, dt
        self.cp = pm.cp
        self.Dloc = D // self.cp
        # NATIVE 1-D vs 2-D token sharding (NO reshard between them). cp axes from the pe_map:
        # 1-D -> cp1==1 (j unsharded, N_j_loc==N); 2-D -> i split into cp0, j into cp1.
        cp_axis_sizes = tuple(int(s) for s in pm.cp_axis_sizes)
        self.cp0 = cp_axis_sizes[0]
        self.cp1 = cp_axis_sizes[1] if len(cp_axis_sizes) > 1 else 1
        self.N_i_loc = N // self.cp0  # i-axis (token dim 1) peer block
        self.N_j_loc = N // self.cp1  # j-axis (token dim 2) peer block (== N when 1-D)
        self.N_loc = self.N_i_loc  # back-compat alias
        if composite_k and self.cp1 != 1:
            raise ValueError(
                f"composite_k incoming is 1-D (cp1==1) only; got cp_axis_sizes={cp_axis_sizes}. "
                f"cp1==1 is the _composite_remap's flat-cp K-rank synthesis -- a 2-D shard needs the "
                f"j-block to be a separate mode, which is what route2_ni is for. The batch extent is "
                f"NOT a constraint here any more: the recv carries an explicit plane (PLANE-major, "
                f"then slot) and the back operand's L=(Dloc,B) collapses to one stride."
            )
        self.L = self.Dloc * B
        self.M = N  # GEMM M = full token-i axis
        self.K = N  # contraction = inner token axis k
        self.tile_shape_mn = tile_shape_mn
        self._nvshmem_torch = nvshmem_torch
        self._from_dlpack = from_dlpack
        # design-E requires each per-peer block CTA-tile-aligned (a CTA output tile maps to ONE peer).
        # pe_aligned handles a STRADDLING N_loc via arbitrary_n / per-peer M-tiling -> no such assert.
        if back_store == "design_e":
            assert self.N_i_loc % tile_shape_mn[0] == 0, (
                f"N_i_loc={self.N_i_loc} must be a multiple of cta_tile_M={tile_shape_mn[0]} "
                "(design_e store; pe_aligned relaxes this)"
            )
            if self.cp1 > 1:
                assert self.N_j_loc % tile_shape_mn[1] == 0, (
                    f"N_j_loc={self.N_j_loc} must be a multiple of cta_tile_N={tile_shape_mn[1]} "
                    "(design_e 2-D store; pe_aligned relaxes this)"
                )

        device_capacity = get_device_capacity()
        assert device_capacity[0] == 9, f"SM90 only; got {device_capacity}"
        a_dtype = torch2cute_dtype_map[dt]
        pe_table = tuple(int(x) for x in pm.cp_pe_table.tolist())
        if placements is None:
            placements = [Shard(int(d)) for d in pm.cp_shard_tensor_dims]
        # ONE FUNCTOR PER COMPILED KERNEL, and that is the difference from the upstream.
        #
        # `main` builds ONE GemmA2ASm90, configures it, then compiles it TWICE -- mutating
        # `_a2a_composite_k` between the two traces, because composite-K is incoming-only. This tree
        # refuses that:
        #
        #   RuntimeError: GemmA2ASm90._bind_call_params() called twice. Call-phase parameters are
        #   established once, at the top of __call__; one functor is traced once, so a second binding
        #   means the instance is being reused for operands that need their own compiled kernel.
        #
        # The guard is right, and the upstream's spelling worked only because nothing re-read the
        # call-phase state after the first trace. So construction-and-configure becomes a FACTORY:
        # called once here for the store's own functor, and once per direction in the compile loop,
        # so each compiled kernel owns the instance it was traced from. The body below is the
        # upstream's, unchanged and un-reordered; only its indentation and the `return` are new.
        def _build_configured_gemm():
            """Build a `GemmA2ASm90` and run the store-variant configure the caller selected.

            Returns a FRESH, fully configured functor every call. Purely a factory -- it reads the
            enclosing scope and mutates nothing, so two calls give two independent instances that
            agree on every compile-time parameter.
            """
            gemm_obj = GemmA2ASm90(
                Float32,
                a_dtype,
                tile_shape_mn,
                cluster_shape_mnk,
                pingpong=pingpong,
                is_persistent=is_persistent,
            )
            # ---- configure the store variant. pe_aligned rides arbitrary_n + pe_aligned_tiling +
            # dynamic at a STRADDLE anchor (else the aligned-N_loc auto-reduce turns arbitrary_n off ->
            # inert design-E — pe_aligned_dyn_validate.py:129).
            if self._dynamic:
                self._anchor_N = self._straddle_anchor_N(N)
            # HYBRID NVLink+IB back drain (opt-in): the cluster_multislot decoupled-putwarp stack — the sole
            # production IB back drain (test_ib_ring.py:679-685). ib_ring composes pe_aligned's per-peer
            # M-tiling with the decoupled GMEM-ring put_nbi_warp (NVSHMEM auto-routes NVLink P2P / IB per peer);
            # consumer_strided_putwarp is the IB-capable drain; cluster_drain+cluster_multislot the 2-slot
            # per-tile peer routing (arbitrary nt_j_pp). ib_ring (1-D) / cp1>1 (2-D) SUPPRESS the arbitrary_n
            # auto-reduce (gemm_sm90_a2a.py:505-511) so an aligned anchor N_loc stays on the pe_aligned drain
            # (the validated no-straddle sub-case). Default OFF => empty => configure byte-identical to today.
            _back_ib_kw = {}
            if self._hybrid_ib:
                _back_ib_kw = dict(
                    ib_ring=True, ib_quiet=True, decoupled=True, producer_tma=True,
                    consumer_strided=True, consumer_strided_putwarp=True, ring_depth=self._ib_ring_depth,
                    cluster_drain=True, cluster_n=self._ib_cluster_n, cluster_multislot=True,
                )
            if self.cp1 == 1:
                if self._dynamic:
                    anchor_N_loc = self._anchor_N // self.cp0
                    assert anchor_N_loc % tile_shape_mn[0] != 0, (
                        f"pe_aligned anchor N_loc={anchor_N_loc} must straddle cta_tile_M="
                        f"{tile_shape_mn[0]} (else arbitrary_n auto-reduces off -> plain design-E)"
                    )
                    gemm_obj.configure_a2a_gemm_native(
                        cp=self.cp, my_cp_rank=int(pm.my_cp_rank), B=B, N_loc=anchor_N_loc,
                        pe_table=pe_table, arbitrary_n=True, N=self._anchor_N,
                        pe_aligned_tiling=True, dynamic=True, **_back_ib_kw,
                    )
                else:
                    # 1-D native design-E: i-axis peer-block store (j full). The flat configure.
                    gemm_obj.configure_a2a_gemm_native(
                        cp=self.cp, my_cp_rank=int(pm.my_cp_rank), B=B, N_loc=self.N_i_loc,
                        pe_table=pe_table,
                    )
            else:
                # 2-D native: the tile->peer unravel over (cp0, cp1). configure_a2a_sharded reads the
                # per-axis blocks from the pe_map's cp_axis_sizes (gemm_native=True). pe_aligned adds the
                # per-peer tiling + dynamic (straddle anchor); design_e uses the CTA-tile-aligned store.
                # Pass the placements (device_mesh optional — pe_map is authoritative).
                if self._dynamic:
                    if self._hybrid_ib:
                        # C10 (option b): configure_a2a_sharded does NOT forward cluster_multislot and its
                        # own ib_ring path REJECTS 2-D (gemm_sm90_a2a.py:1085); route the 2-D IB drain
                        # straight through configure_a2a_gemm_native(cp_axis_sizes=(cp0,cp1)) — the 2-D
                        # cluster_drain device path already exists (:876) and this is the EXACT call the
                        # passing 2-D cluster_multislot test uses (test_ib_ring.py:679-685). N_loc = N//cp0;
                        # cp1>1 suppresses the arbitrary_n auto-reduce so an aligned anchor stays pe_aligned.
                        gemm_obj.configure_a2a_gemm_native(
                            cp=self.cp, my_cp_rank=int(pm.my_cp_rank), B=B,
                            N_loc=self._anchor_N // self.cp0, pe_table=pe_table, N=self._anchor_N,
                            cp_axis_sizes=(self.cp0, self.cp1), arbitrary_n=True,
                            pe_aligned_tiling=True, dynamic=True, **_back_ib_kw,
                        )
                    else:
                        gemm_obj.configure_a2a_sharded(
                            device_mesh, placements, pe_map=pm, B=B, N=self._anchor_N, gemm_native=True,
                            dynamic=True, pe_aligned_tiling=True,
                        )
                else:
                    gemm_obj.configure_a2a_sharded(
                        device_mesh, placements, pe_map=pm, B=B, N=N, gemm_native=True,
                    )
            # COMPOSITE-K K-hoist (§9) is INCOMING-ONLY: the gemm's _a2a_composite_k flag (which const_expr-
            # gates _remap_A_operand_layout / _gA_local_tile / _k_tile_cnt) is toggled PER-DIRECTION in
            # _compile_pe_aligned_both_directions (True for the incoming compile, False for outgoing) — the
            # OUTGOING operand is the plain (M,K,L) k-major view, so its kernel MUST bake the flag OFF (else the
            # composite remap synthesizes a bogus cp mode on it -> wrong outgoing output). _a2a_cp / _a2a_B are
            # already set by configure_a2a_gemm_native above (the 1-D path). The gemm default is False (byte-
            # identical), so a non-composite store leaves it untouched.
            # persistent symmetric recv (cp, Dloc, B, N_i_loc, N_j_loc) at the REAL runtime shape
            # (1-D: N_j_loc==N -> the wave-1 shape). pe_aligned's dynamic executor re-binds this per call.
            return gemm_obj

        self._make_gemm_obj = _build_configured_gemm
        gemm_obj = _build_configured_gemm()
        self.recv = _symmetric_empty((self.cp, self.Dloc, B, self.N_i_loc, self.N_j_loc), dtype=dt)
        # Seed the recv cache with the anchor/ctor buffer. Keyed (B, N) to match `rebind_N`: keyed
        # on N alone, a second batch extent would silently be handed the first's buffer.
        self._recv_cache[(self.B, self.N)] = self.recv

        max_active_clusters = get_max_active_clusters(
            cluster_shape_mnk[0] * cluster_shape_mnk[1]
        )
        self.scheduler_args = make_scheduler_args(max_active_clusters, Int32(8), None, None)
        # The upstream carried a `varlen_args` attribute here, built by
        # `make_varlen_args(None, None, None)` -- whose first line returns None when both cu_seqlens
        # are None, so the call was a constant. It is not carried at all now: this tree's kernel
        # `__call__` has no varlen_args parameter (see the front store's compile), so the attribute
        # would exist only to be passed to a slot that no longer exists.
        self.stream = cutlass_torch.current_stream()
        self._gemm_obj = gemm_obj
        #: direction -> the functor its compiled kernel was traced from. One functor per
        #: compiled kernel is this tree's rule; see the factory above.
        self._gemm_objs: dict = {}
        self._dev = self.recv.device
        # HYBRID NVLink+IB back drain wiring (cluster_multislot): the pe_aligned decoupled store rides a
        # per-cluster SYMMETRIC-heap staging (n_clusters, ring_depth, epi_m=128, N_j_loc) bf16 as its IB
        # put SOURCE + a device PE table, BOTH routed into GemmA2ASm90.EpilogueArguments via
        # make_differential_epi_args (the #57 dead-descriptor gate: NULLs cluster_stage on an all-P2P
        # COLLAPSE so a single-node run stays byte-identical to pe_aligned). WITHOUT this wiring the
        # kernel's cluster_drain branch reads epilogue_params.cluster_stage=None -> AttributeError at
        # COMPILE — but only with REAL IB peers (has_ib_peers=True), since the collapse elides the
        # consumer warpgroup that reads it (why it slipped the single-node stage). Mirrors the front
        # store's _ib_ring_t alloc + test_ib_ring.py:656-705 (the passing cluster_multislot reference).
        # epi_m is fixed 128: cluster_multislot REQUIRES tile_m==128 (configure_a2a_gemm_native rejects
        # tile_m>epi_m for multislot), so gcd(128,tile_m)==128 for every valid hybrid config. The staging
        # is O(N_token) per rank (frugal, strictly sub-leading vs the O(N_token^2) recv).
        self._cluster_stage_cache = {}
        self._cluster_stage = None
        self._cute_pe_dev, self._ib_pe_dev_t = None, None
        if self._hybrid_ib:
            self._n_clusters = int(max_active_clusters)
            self._ib_pe_dev_t = torch.tensor(
                list(pe_table), device=self._dev, dtype=torch.int32
            ).contiguous()
            self._cute_pe_dev = from_dlpack(self._ib_pe_dev_t, assumed_align=4)
            self._cluster_stage = self._alloc_cluster_stage(self.N_j_loc)
        # Compiled GEMM PER DIRECTION (see class docstring): outgoing operands are k-major
        # (L,i,k)/(L,j,k) -> k-stride-1; incoming are TRANSPOSED (L,k,i)/(L,k,j) -> i-stride-1. The
        # bitcode route bakes the operand stride into the descriptor, so a kernel compiled for one
        # direction MIS-READS the other's (the incoming √2 garbage) -> one kernel per direction.
        self._compiled = {}  # direction -> CompiledGemmBitcode
        if self._dynamic:
            # pe_aligned: EAGER per-direction compile at the straddle anchor (throwaway anchor recv,
            # freed after). All symmetric alloc/free stays in __init__ (SPMD-deterministic). run()
            # re-binds THIS N's dynamic operands + a FRESH dynamic epilogue per call.
            self._compile_pe_aligned_both_directions()
        else:
            # design-E: static per-shape. mD = the GEMM's LOGICAL (M, N, L) output extent (drives
            # only shape/scheduler; the store writes the peer atoms built from recv, so this never
            # reaches GMEM). cute_D + epi are built ONCE from the persistent recv and reused each run;
            # the per-direction kernel is compiled lazily on first run.
            D_logical = torch.as_strided(self.recv, (self.M, N, self.L), (N, 1, self.M * N))
            self._cute_D = from_dlpack(D_logical, assumed_align=16)
            self.epi_args = GemmA2ASm90.EpilogueArguments(
                alpha=None, beta=None, mRowVecBroadcast=None, mColVecBroadcast=None,
                add_to_output=False, rounding_mode=RoundingMode.RN, sr_seed=None,
                recv=from_dlpack(self.recv, assumed_align=16),
            )

    def _straddle_anchor_N(self, N):
        """A full-N compile anchor whose per-cp-axis local extent STRADDLES the CTA tile (so
        arbitrary_n / the pe_aligned partial-tile path is traced), keeping each axis %8 (16-B) AND
        divisible by BOTH cp axes. Real N if it already straddles (either axis), else N + a bump that
        preserves those invariants.

        The bump MUST keep the anchor divisible by cp0 AND cp1 (the 2-D i/j reshard split) and each
        per-axis extent %8 (16-B TMA-S2G). ``tile_m`` (=128) does this ONLY when it divides both cp axes
        with a %8 quotient -- the pow2 cp<=16 case, where this is BYTE-IDENTICAL to the historical
        ``N + tile_m``. For a NON-pow2 cp axis (factor 3/6/12/24) tile_m is NOT a multiple of cp, so
        ``N + tile_m`` breaks anchor%cp (a 2-D reshard ValueError) AND inflates the anchor buffers
        (seen at cp24 non-straddling N: cp4x6/2x12 ValueError, cp6x4/12x2 fault); and for cp>=32
        tile_m//cp is not %8. The general bump ``8*lcm(cp0,cp1)`` is the smallest step preserving %cp0,
        %cp1 AND per-axis %8 at ANY cp; loop until it straddles (one step suffices for every supported
        cp -- the OR-straddle is met via whichever axis's step//cp is not a multiple of tile_m)."""
        tile_m = self.tile_shape_mn[0]
        cp0, cp1 = int(self.cp0), int(self.cp1)

        def straddles(NN):
            return (NN // cp0) % tile_m != 0 or (NN // cp1) % tile_m != 0

        if straddles(N):
            return N
        tile_ok = (tile_m % cp0 == 0 and tile_m % cp1 == 0
                   and (tile_m // cp0) % 8 == 0 and (tile_m // cp1) % 8 == 0)
        if tile_ok:
            step = tile_m  # pow2 cp<=16: BYTE-IDENTICAL to the historical `N + tile_m`
        else:
            from math import lcm
            step = 8 * lcm(cp0, cp1)  # divisible by cp0, cp1 AND per-axis %8 at ANY cp (incl non-pow2/>=32)
        anchor = N + step
        for _ in range(max(tile_m, cp0 * cp1)):  # bounded; 1 step suffices for every supported cp
            if straddles(anchor):
                return anchor
            anchor += step
        raise ValueError(
            f"_straddle_anchor_N: no straddling anchor found for N={N} cp0={cp0} cp1={cp1} step={step} "
            f"(unexpected — please file a bug)."
        )

    def _alloc_cluster_stage(self, n_j_loc):
        """Allocate (+cache per N_j_loc) the per-cluster SYMMETRIC-heap staging buffer
        ``(n_clusters, ring_depth, 128, N_j_loc)`` bf16 — the cluster_multislot drain's IB put SOURCE
        (test_ib_ring.py:665). N_j_loc is the ONLY axis that varies with runtime N; keyed on it so
        rebind_N re-uses a cached buffer. Zero-init (unwritten slots are read as 0). Returns the backing
        tensor; callers build a fresh mark_layout_dynamic'd cute view per launch (one compile serves all N)."""
        key = int(n_j_loc)
        buf = self._cluster_stage_cache.get(key)
        if buf is None:
            buf = _symmetric_empty(
                (self._n_clusters, self._ib_ring_depth, 128, key), dtype=self.dt
            )
            buf.zero_()
            _recv_cache_put(
                self._cluster_stage_cache, key, buf, _symmetric_free
            )
        return buf

    def _compile_pe_aligned_both_directions(self):
        """Compile the pe_aligned-dyn store for BOTH directions at the straddle anchor, then free the
        throwaway anchor recv. Mirrors pe_aligned_dyn_validate.py (compile at a straddle anchor, mark
        operands + recv + D_logical dynamic) + the per-direction split of design-E."""
        anchor_N = self._anchor_N
        anchor_N_i, anchor_N_j = anchor_N // self.cp0, anchor_N // self.cp1
        anchor_recv = _symmetric_empty(
            (self.cp, self.Dloc, self.B, anchor_N_i, anchor_N_j), dtype=self.dt
        )
        # HYBRID: a THROWAWAY anchor cluster-staging (n_clusters, ring_depth, 128, anchor_N_j) matching
        # anchor_recv's j-extent, for the mark_layout_dynamic anchor compile only (freed in finally).
        anchor_stage = (
            _symmetric_empty(
                (self._n_clusters, self._ib_ring_depth, 128, anchor_N_j), dtype=self.dt
            ) if self._hybrid_ib else None
        )
        if anchor_stage is not None:
            anchor_stage.zero_()
        try:
            dummy_a = torch.empty(
                self.Dloc, self.B * anchor_N * anchor_N, device=self._dev, dtype=self.dt
            )
            dummy_b = torch.empty(
                self.Dloc, self.B * anchor_N * anchor_N, device=self._dev, dtype=self.dt
            )
            D_logical = torch.as_strided(
                anchor_recv, (anchor_N, anchor_N, self.L), (anchor_N, 1, anchor_N * anchor_N)
            )
            for direction in ("outgoing", "incoming"):
                # COMPOSITE-K is INCOMING-ONLY: bake the gemm's _a2a_composite_k const_expr PER-DIRECTION
                # (True only for the incoming compile). The OUTGOING operand is the plain (M,K,L) k-major
                # view, so its kernel MUST bake the flag OFF -> the composite remap hooks fall through to
                # the parent default (byte-identical to the non-composite back). Toggling BEFORE
                # _operands + compile so each direction's bitcode captures the right value.
                # A FRESH functor for this direction's kernel. `main` re-used one instance and
                # mutated the flag between traces; here each compiled kernel owns the instance it
                # was traced from (see the factory in __init__). The flag is set BEFORE _operands
                # and the compile, so this direction's bitcode captures the right value -- which is
                # exactly what the upstream's mutation achieved, without the shared instance.
                gemm_d = self._make_gemm_obj()
                gemm_d._a2a_composite_k = bool(
                    self._a2a_composite_k and direction == "incoming"
                )
                self._gemm_objs[direction] = gemm_d
                if self._route2_ni and direction == "incoming":
                    # route2_ni incoming: the operand is the front's 3-D N_i-STRIDE-1 recv-half. Build a
                    # matching dummy the SAME way (contiguous (Dloc, N_j, N_i) -> permute so N_i is
                    # stride-1) so _operands' permute-view bakes a_major="k" (K=N_i stride-1) EXACTLY
                    # like the run feed. (A plain contiguous (Dloc,N_i,N_j) would bake a_major="m" ->
                    # the √2 garbage.) K=N_i (padded) differs from anchor_N -> mark_layout_dynamic.
                    tile_m = self.tile_shape_mn[0]
                    anchor_Ni = self.cp0 * (((anchor_N // self.cp0) + tile_m - 1) // tile_m) * tile_m
                    dummy_buf = torch.empty(
                        self.Dloc, anchor_N, anchor_Ni, device=self._dev, dtype=self.dt
                    )
                    dummy_ni = dummy_buf.permute(0, 2, 1)  # (Dloc, N_i, N_j) N_i stride-1
                    A3, B3 = self._operands(dummy_ni, dummy_ni, direction, N=anchor_N)
                elif self._a2a_composite_k and direction == "incoming":
                    # composite_k incoming: the operand is the front's 2-D PADDED per-rank-contiguous
                    # D-major recv-half (Dloc, cp*N_j*Xg_pad). Build the SAME contiguous dummy so
                    # _operands' reshape+slot-0-permute bakes the (N_j, Xg_pad, Dloc) composite-K view
                    # EXACTLY like the run feed. K=Xg_pad (one rank; cp synthesized) -> mark_layout_dynamic.
                    tile_m = self.tile_shape_mn[0]
                    anchor_Xg_pad = (((anchor_N // self.cp0) + tile_m - 1) // tile_m) * tile_m
                    anchor_N_j = anchor_N // self.cp1
                    dummy_comp = torch.empty(
                        self.Dloc, self.cp * anchor_N_j * anchor_Xg_pad, device=self._dev, dtype=self.dt
                    )
                    A3, B3 = self._operands(dummy_comp, dummy_comp, direction, N=anchor_N)
                else:
                    A3, B3 = self._operands(dummy_a, dummy_b, direction, N=anchor_N)
                if self._hybrid_ib:
                    # cluster_multislot IB drain: route recv + the symheap cluster_stage + device PE
                    # table via make_differential_epi_args (NULLs cluster_stage on the all-P2P collapse
                    # -> byte-identical to pe_aligned; passes through with real IB peers).
                    epi = gemm_d.make_differential_epi_args(
                        recv=from_dlpack(anchor_recv, assumed_align=16).mark_layout_dynamic(),
                        cluster_stage=from_dlpack(anchor_stage, assumed_align=16).mark_layout_dynamic(),
                        pe_table_dev=self._cute_pe_dev,
                    )
                else:
                    epi = GemmA2ASm90.EpilogueArguments(
                        alpha=None, beta=None, mRowVecBroadcast=None, mColVecBroadcast=None,
                        add_to_output=False, rounding_mode=RoundingMode.RN, sr_seed=None,
                        recv=from_dlpack(anchor_recv, assumed_align=16).mark_layout_dynamic(),
                    )
                self._compiled[direction] = compile_nvshmem(
                    gemm_d,
                    from_dlpack(A3, assumed_align=16).mark_layout_dynamic(),
                    from_dlpack(B3, assumed_align=16).mark_layout_dynamic(),
                    from_dlpack(D_logical, assumed_align=16).mark_layout_dynamic(),
                    # varlen_args and the trailing mB3 slot are not in our __call__; see the
                    # front store's compile above for the signature diff and the symptom.
                    None, epi, self.scheduler_args, self.stream,
                    register=True,
                    op_factory=self._make_gemm_obj,
                    reuse=True,  # M6: per-direction, so incoming/outgoing keep separate entries
                )
        finally:
            _symmetric_free(anchor_recv)
            if anchor_stage is not None:
                _symmetric_free(anchor_stage)

    def _operands(self, a_dm: torch.Tensor, b_dm: torch.Tensor, direction: str, N=None):
        """Per-direction (M,K,L)/(N,K,L) operands fed to the GEMM.

        a_dm,b_dm (Dloc, B*N*N) D-major token-order (b,i,k). af=(L,i,k), bf=(L,j,k) VIEWS; for
        ``incoming`` the (i/j, k) axes are transposed (mirror _gemm1). ``N`` overrides the token
        extent (the straddle-anchor compile uses anchor_N; run uses the real self.N). Returns the
        perm3d ``.permute(1,2,0)`` ((L,M,K)->(M,K,L)) the GEMM consumes.

        OUTGOING is a zero-copy VIEW (K-unit-stride, a_major="k"). INCOMING materializes the transpose
        with ``.contiguous()`` (a (C) WORKAROUND, task #16): the pe_aligned per-peer A-row shift
        ``mainloop_remap_mA`` (gemm_sm90_a2a.py:4204) assumes a_major="k" (M=token-i axis, strided) and
        illegal-instruction-CRASHES on the transposed a_major="m" (M unit-stride) operand at partial
        GEMM tiles (M=K=N % 128 != 0). Materializing -> a_major="k" (the proven crash-free layout).
        Correct-by-construction (pure layout change; GEMM is layout-invariant). Cost: one .contiguous()
        copy O(Dloc*N^2), negligible vs the GEMM's O(Dloc*N^3) at N>=1000, and only on incoming. (B)
        [make the per-peer shift valid for a_major="m"] would eliminate the copy -- future no-copy opt.

        ROUTE-2 (A) NO-COPY (route2_ni): the incoming a_dm,b_dm are the front's 3-D N_i-stride-1 recv
        halves (Dloc, N_i, N_j) [N_i = the incoming contraction i, STRIDE-1]. Feed them as a
        permute-VIEW (N_j=M, N_i=K, Dloc=L) so K=N_i is stride-1 = a_major="k" — the producer store
        already laid global-i contiguous, so NO .transpose(-1,-2).contiguous(). N_i (padded per-peer)
        is the contraction; its pad rows are 0 (glu(0)=0) so the sum is exact. B=1 only (L=Dloc).

        COMPOSITE-K NO-COPY (composite_k, §9): the incoming a_dm,b_dm are the front's 2-D D-major recv
        halves (Dloc, cp*N_j*Xg_pad) — per-rank CONTIGUOUS. Reshape to (Dloc, cp, N_j, Xg_pad) then feed
        the SLOT-0 base VIEW (N_j, Xg_pad, Dloc) = (M, K, L); the kernel's _composite_remap synthesizes
        the cp contraction-rank mode (stride N_j*Xg_pad) so the K=(cp,Xg_pad) read spans all cp rank
        blocks. Same M=N_j, L=Dloc as route2_ni — the ONLY difference is the recv layout (per-rank
        contiguous -> wide 32-KiB puts, not the N_i-stride-1 640-B cap). Pad rows [Xg,Xg_pad) are 0
        (glu(0)=0). EXACTLY the view tests/distributed/test_composite_k_backread.py validates. B=1 only.
        """
        if self._route2_ni and direction == "incoming":
            if a_dm.dim() == 4:
                # B-MODE: (Dloc, B, N_i, N_j). flatten(0,1) is a VIEW -- the recv is
                # (2*Dloc, B, N_j, N_i)-contiguous, so Dloc's stride is EXACTLY B x B's, which is the
                # whole reason the batch mode sits there. -> (L=Dloc*B, N_i, N_j) -> (M, K, L). The
                # L order is (dloc, b) with b inner, matching the plain path's flatten and therefore
                # the back store's `d = L//B, b = L%B` decode.
                return (
                    a_dm.flatten(0, 1).permute(2, 1, 0),
                    b_dm.flatten(0, 1).permute(2, 1, 0),
                )
            return a_dm.permute(2, 1, 0), b_dm.permute(2, 1, 0)  # (N_j, N_i, Dloc) = (M, K, L)
        if self._a2a_composite_k and direction == "incoming":
            Nc = self.N if N is None else N
            tile_M = self.tile_shape_mn[0]
            Xg_pad = ((Nc // self.cp0) + tile_M - 1) // tile_M * tile_M  # ceil(N_i_loc/BLK_M)*BLK_M
            N_j = Nc // self.cp1  # full j (== N when 1-D); composite is 1-D so N_j == Nc
            if a_dm.dim() == 3:
                # B-MODE: (Dloc, B, cp*N_j*Xg_pad) -- PLANE-major, then slot. Reshape to
                # (Dloc, B, cp, N_j, Xg_pad), take slot 0, then flatten (Dloc, B) into L: a VIEW,
                # because with the plane OUTSIDE the slot Dloc's stride is exactly B x B's. The
                # kernel's _composite_remap then derives the cp stride as M*stride(M) = N_j*Xg_pad,
                # which is the per-plane per-rank block -- correct with NO kernel change.
                Bc = a_dm.shape[1]
                a5 = a_dm.reshape(self.Dloc, Bc, self.cp, N_j, Xg_pad)
                b5 = b_dm.reshape(self.Dloc, Bc, self.cp, N_j, Xg_pad)
                return (
                    a5[:, :, 0].flatten(0, 1).permute(1, 2, 0),
                    b5[:, :, 0].flatten(0, 1).permute(1, 2, 0),
                )
            a4 = a_dm.reshape(self.Dloc, self.cp, N_j, Xg_pad)  # (Dloc, cp, N_j, Xg_pad)
            b4 = b_dm.reshape(self.Dloc, self.cp, N_j, Xg_pad)
            # slot-0 base + permute -> (N_j, Xg_pad, Dloc) = (M, K, L); kernel synthesizes the cp mode.
            return a4[:, 0].permute(1, 2, 0), b4[:, 0].permute(1, 2, 0)
        N = self.N if N is None else N
        L = self.L
        # a_dm/b_dm arrive either 2-D (Dloc, B*N*N) — the compile dummies and any legacy caller — or
        # already-unpacked 4-D (Dloc, B, N, N) from front_a2a. The 4-D form may be a PADDED strided view
        # (pad_inner: innermost pitch N_j_pad > N), so it is folded with flatten(0,1), which is a VIEW
        # for every layout the front produces (the Dloc stride is exactly B * the b stride), NEVER the
        # reshape below, which would copy. The resulting (M,K,L) operand keeps K unit-stride =>
        # a_major="k" — the same majorness the dummies bake and the only layout the pe_aligned per-peer
        # A-row shift accepts; only the M-stride widens (N -> N_j_pad), which the GEMM already takes at
        # runtime (mark_layout_dynamic; the same widening _pad_operand_k ships on the incoming arm).
        af = a_dm.flatten(0, 1) if a_dm.dim() == 4 else a_dm.reshape(self.Dloc, self.B, N, N).reshape(L, N, N)
        bf = b_dm.flatten(0, 1) if b_dm.dim() == 4 else b_dm.reshape(self.Dloc, self.B, N, N).reshape(L, N, N)
        if direction == "incoming":
            # incoming: tri[i,j] = sum_k a[k,i]*b[k,j] -> transpose the (i/j, k) axes (mirror _gemm1),
            # then materialize so the back feeds a_major="k" (the (C) workaround above).
            #
            # F3 DOWN PAYMENT — pad the materialized innermost (K) extent to a multiple of 64 bf16
            # elements. After the `.permute(1, 2, 0)` below the operand is (M, K, L) and the MAINLOOP
            # TMA-G2S walks it with an M-row stride equal to this K extent; at `2*N % 32 == 16`
            # (i.e. every N congruent to 8 mod 16 — HALF of all legal N) that stride is 16-mod-32 and
            # costs 1.213-1.215x at Dloc=128, measured at FIXED N by varying only the operand stride
            # against a 1.221-1.223x cross-N and a +-0.4 % drift floor. Padding K to %64
            # makes the row stride 128-B clean. NB the destination-side row stride of the back recv is
            # a DIFFERENT quantity and is measured at ZERO (0.999-1.002x) — do not pad that one; see
            # the byte-phase law in gemm_sm90_a2a.py.
            #
            # Free here and only here: this path ALREADY materialises, so the pad only widens an
            # allocation that was happening anyway — no extra pass, no extra buffer, and the logical
            # extent stays N (the returned view is `[..., :N]`). Extra bytes are O(Dloc*N*64) against
            # an O(Dloc*N^2) operand, i.e. strictly sub-leading in N_token. The GEMM needs no change:
            # a padded operand M-stride with the logical extent still N was run end-to-end on the
            # production pe_aligned + arbitrary_n store (`untouched == 0` on all 8 ranks).
            # The OUTGOING operand is a zero-copy view of the front recv and is NOT padded here —
            # doing that requires the front-recv layout change (F1), which is deliberately out of
            # scope; so outgoing still pays the phase at N % 16 == 8.
            af = _pad_operand_k(af.transpose(-1, -2))
            bf = _pad_operand_k(bf.transpose(-1, -2))
        return af.permute(1, 2, 0), bf.permute(1, 2, 0)  # (M,K,L), (N,K,L)

    def _pe_run_args(self, A3, B3):
        """FRESH pe_aligned-dyn run args from the persistent recv (mirror bench_a2a_fusion.
        _back_store_run_args): mark_layout_dynamic on operands + recv + D_logical, a FRESH
        EpilogueArguments per call (the one-compile-many-shapes re-bind seam)."""
        cute_recv = from_dlpack(self.recv, assumed_align=16).mark_layout_dynamic()
        D_logical = torch.as_strided(
            self.recv, (self.M, self.N, self.L), (self.N, 1, self.M * self.N)
        )
        cute_D = from_dlpack(D_logical, assumed_align=16).mark_layout_dynamic()
        if self._hybrid_ib:
            epi = self._gemm_obj.make_differential_epi_args(
                recv=cute_recv,
                cluster_stage=from_dlpack(
                    self._cluster_stage, assumed_align=16
                ).mark_layout_dynamic(),
                pe_table_dev=self._cute_pe_dev,
            )
        else:
            epi = GemmA2ASm90.EpilogueArguments(
                alpha=None, beta=None, mRowVecBroadcast=None, mColVecBroadcast=None,
                add_to_output=False, rounding_mode=RoundingMode.RN, sr_seed=None, recv=cute_recv,
            )
        cute_A = from_dlpack(A3, assumed_align=16).mark_layout_dynamic()
        cute_B = from_dlpack(B3, assumed_align=16).mark_layout_dynamic()
        return (cute_A, cute_B, cute_D, None, epi, self.scheduler_args,
                self.stream, None)

    def _compile_for(self, direction: str):
        """design-E ONLY: compile (+ cache) the static kernel for ``direction`` with DUMMY operands
        matching the runtime view strides (so the baked descriptor reads the strided feed correctly).
        Lazy: called on first run of each direction."""
        # dummy D-major operands of the run shape (Dloc, B*N*N); build the per-direction views from
        # them so the compile-time from_dlpack bakes the SAME stride pattern the run feed will have.
        dummy_a = torch.empty(self.Dloc, self.B * self.N * self.N, device=self._dev, dtype=self.dt)
        dummy_b = torch.empty(self.Dloc, self.B * self.N * self.N, device=self._dev, dtype=self.dt)
        A3, B3 = self._operands(dummy_a, dummy_b, direction)
        compiled = compile_nvshmem(
            self._gemm_obj,
            self._from_dlpack(A3, assumed_align=16),
            self._from_dlpack(B3, assumed_align=16),
            self._cute_D,
            None,
            self.epi_args,
            self.scheduler_args,
            self.stream,
            None,
            register=True,
            op_factory=self._make_gemm_obj,
            reuse=True,  # M6
        )
        self._compiled[direction] = compiled
        return compiled

    def run(self, a_dm: torch.Tensor, b_dm: torch.Tensor, direction: str) -> torch.Tensor:
        """a_dm,b_dm (Dloc, B*N*N) D-major (the staged front recv halves) -> stores into recv.

        Uses the per-direction compiled kernel. The af/bf are VIEWS of a_dm/b_dm + the per-direction
        transpose VIEW — NO .contiguous(). pe_aligned re-binds THIS N's dynamic operands + a fresh
        dynamic epilogue (one-compile-many-shapes); design-E reuses the static cute_D + epi and
        compiles lazily. Returns the persistent recv (cp,Dloc,B,N_i_loc,N_j_loc); caller
        back_unpack_gemm_native.
        """
        if self._dynamic:
            A3, B3 = self._operands(a_dm, b_dm, direction)
            self._compiled[direction](*self._pe_run_args(A3, B3))
            return self.recv
        compiled = self._compiled.get(direction) or self._compile_for(direction)
        A3, B3 = self._operands(a_dm, b_dm, direction)
        compiled(
            self._from_dlpack(A3, assumed_align=16),
            self._from_dlpack(B3, assumed_align=16),
            self._cute_D,
            None,
            self.epi_args,
            self.scheduler_args,
            self.stream,
            None,
        )
        return self.recv

    def rebind_N(self, N: int, *, B: int = None) -> None:
        """dynamic_shape: point self.recv (+ the runtime-N attrs _operands/_pe_run_args read) at this
        token count AND batch extent, (re)allocating the symmetric recv per distinct ``(B, N)``
        (cached, SPMD-lockstep — all ranks call with the SAME sequence). The dynamic executor reads
        shapes at runtime, so no recompile. cluster_M==1 pe_aligned only (asserted at construction).

        Args:
            N: Runtime square token extent. Must be divisible by both cp axes.
            B: Runtime batch extent. ``None`` keeps the constructor's — accepted so existing call
                sites are unchanged, NOT because a stored batch is authoritative. The kernel reads
                its batch extent off recv mode 4 (`gemm_sm90_a2a.py`), so the recv's shape IS the
                contract: allocate it for the batch the caller actually has, or the kernel faithfully
                decodes a shape nobody passed. The cache keys on ``(B, N)`` for the same reason —
                keyed on ``N`` alone, a second batch extent would silently reuse the first's buffer.
        """
        assert self._dynamic_shape, "rebind_N requires dynamic_shape=True"
        B = int(self.B if B is None else B)
        if N == self.N and B == self.B and self.recv is not None:
            return
        N_i_loc, N_j_loc = N // self.cp0, N // self.cp1
        key = (B, N)
        r = self._recv_cache.get(key)
        if r is None:
            r = _symmetric_empty(
                (self.cp, self.Dloc, B, N_i_loc, N_j_loc), dtype=self.dt
            )
            _recv_cache_put(self._recv_cache, key, r, _symmetric_free)
        self.recv = r
        self.B = B
        # `L` is the GEMM's BATCH extent and `_pe_run_args` builds `D_logical` from it, so it must
        # track the runtime batch or the GEMM runs `Dloc*B_ctor` planes instead of `Dloc*B` --
        # silently, at the right output SHAPE, with the tail planes never computed. Measured before
        # this line existed: an engine built at B=1 and forwarded at B=2 agreed with a per-plane
        # reference to a worst ratio of 99.3, while the SAME engine built at B=2 was bitwise exact.
        self.L = self.Dloc * B
        self.N = self.M = self.K = N  # GEMM M=K=full token axis; _operands/_pe_run_args read these
        self.N_i_loc = self.N_loc = N_i_loc
        self.N_j_loc = N_j_loc
        if self._hybrid_ib:  # re-bind the per-cluster symheap staging for this N's N_j_loc (cached)
            self._cluster_stage = self._alloc_cluster_stage(N_j_loc)

    def free(self):
        for c in self._compiled.values():
            try:
                c.free()
            except Exception:
                pass
        for r in self._recv_cache.values():
            try:
                _symmetric_free(r)
            except Exception:
                pass
        self._recv_cache.clear()
        for s in self._cluster_stage_cache.values():  # symheap cluster_multislot staging (hybrid_ib)
            try:
                _symmetric_free(s)
            except Exception:
                pass
        self._cluster_stage_cache.clear()
        # Same closure cycle as the front store: `_make_gemm_obj` is a nested def assigned to self,
        # so it closes over `__init__`'s scope and the store is reachable from its own attribute.
        # Clearing makes the release refcount-driven instead of waiting on a cyclic collection.
        self._compiled = {}
        self._make_gemm_obj = None
        self._gemm_obj = None
        self.epi_args = None


# --------------------------------------------------------------------------- #
# The fully-fused e2e distributed TriMul.
# --------------------------------------------------------------------------- #
class TriMulAutotuned:
    """Fully-fused distributed TriMul: front + back A2A fused into the GEMM epilogues.

    Holds the two fused-front operand stores (a, b), the fused-back einsum store, and
    the ``ReshardLayout`` (for the host-side unpack views). **Reused across calls** —
    the ctor compiles all three fused kernels ONCE and allocates the symmetric recvs
    ONCE, so callers cache an instance per ``(B, N, D, cp, dtype, pe_map, weights)``.

    Requires NVSHMEM to be initialized (``DistributedManager.init_nvshmem``) before
    construction. Weights are bound at construction (the front folds them into the
    GEMM weight); pass the same ``w`` dict shape as ``RealKernelTriMul.forward``.

    Parameters
    ----------
    pe_map : PeMap
        The cp addressing (1-D headline; gives cp / my_cp_rank / cp_pe_table).
    B, N, D : int
        TriMul batch, (square) token extent, feature width. ``N % cp == 0`` and
        ``D % cp == 0``; ``N_loc = N // cp`` must be a multiple of the CTA tile-M.
    w : dict
        Replicated weight dict (keys match ``trimul_autotuned`` / ``RealKernelTriMul``):
        ``norm_in_w/b``, ``p_in_w/b`` (2D,D)/(2D,), ``g_in_w/b``, ``norm_out_w/b``,
        ``p_out_w/b`` (D,D)/(D,), ``g_out_w/b``. (Projection biases ``p_in_b``/``g_in_b``
        are NOT fused into the front store yet — the wave-1 baseline/oracle use bias=0
        for the front projections; a non-None ``p_in_b``/``g_in_b`` raises.)
    dt : torch.dtype
        Compute dtype (bfloat16).
    consumer : str
        The P2 dual-x back-half ``back_v``: ``"stagec"`` (default = the wave-1 "sc",
        cooperative; reads the d-strided value) or ``"staged_a_in_regs"`` (the staged
        a_in_regs path that transposes via ldmatrix into the WGMMA RF — sidesteps
        strided-sA reads at large N). ``"torch"`` uses pure-torch (the reference back
        half). All three compute the SAME xgate.sc math (``x_gate=x_norm``).
    back_store : str
        The back A2A store variant: ``"pe_aligned"`` (DEFAULT — the committed §0.6.2 pe_aligned
        per-peer tiling store, compiled dynamic at a straddle anchor so it is one-compile-many-
        shapes-ready) or ``"design_e"`` (default-off fallback — the byte-identical pre-rewire
        GEMM-native design-E TMA-S2G store, static per shape). Both compute the SAME reshard.
    """

    def __init__(
        self,
        pe_map: PeMap,
        B: int,
        N: int,
        D: int,
        w: dict,
        dt: torch.dtype,
        *,
        consumer: str = "stagec",
        eps: float = 1e-5,
        device_mesh=None,
        placements=None,
        front_tile_mn=None,
        back_tile_mn=None,
        front_pingpong=None,
        back_pingpong=None,
        is_persistent=True,
        autotune_config: bool = None,
        back_store: str = "pe_aligned",
        dynamic: bool = False,
        route2_ni: bool = False,
        composite_k: bool = False,
        hybrid_ib: bool = None,
        back_cluster_n: int = None,
        use_device_signal_nvlink: bool = False,
        has_mask: bool = False,
        front_pad_inner: bool = None,
        front_pad_eager: bool = None,
    ):
        # front_pad_inner: pad the PLAIN front recv's innermost token extent so every rank's
        # destination base is 128-B clean (front_pad_inner_geometry documents the defect, the tile_M
        # quantum and the already-clean guard). Enable with `CPO_FRONT_PAD_INNER=1`.
        #
        # DEFAULT OFF, on measurement rather than caution. The pad is byte-identical on an already-clean
        # shape and worth -6.7 to -9.7% e2e where it engages. But production runs `dynamic=True`, so ONE
        # compiled kernel serves both regimes and the guard is a per-N RUNTIME decision -- a DECLINED
        # shape still runs the 2-mode A operand (Yg,K,Xg,L) with Xg==1 instead of rank-3 (M,K,L) and pays
        # +0.61% at N=2048, +0.76% at N=4096 (pinned harness, 3 reps, non-overlapping arms).
        #
        # WHERE IT ENGAGES, brute-force verified against front_pad_inner_geometry over 7174
        # (cp, N, tile_M) triples (tests/benchmark_test/_phase_grid.py):
        #
        #     cp=2   engages <=> N =  8 (mod 16)   phase 64  mild
        #     cp=4   engages <=> N =  8 (mod 16)   phase 32  mid
        #     cp=8   engages <=> N =  8 (mod 16)   phase 16  WORST   |   N = 16 (mod 32) -> phase 64 mild
        #     cp=16  engages <=> N = 16 (mod 32)   phase 32  mid
        #
        # Two things this table has been mis-read on, both already once:
        #   * "cp=16 never engages" is FALSE -- it engages on HALF its legal token counts (N=1040 is in
        #     the correctness grid). The 0/7 seen on the 9-shape acceptance list is an artifact of that
        #     list. cp=16 is also the width that spans nodes, so its engaging cells are the only ones
        #     where the cost lands on an IB warp-put (no TMA descriptor -> the 32-B sector-straddle the
        #     pad removes may not apply) while the extra bytes are paid on every peer.
        #   * Engaging is NOT the same as winning. The pad's own cost is
        #     (ceil(N_j_loc/tile_M)*tile_M / N_j_loc - 1), i.e. O(tile_M/N): ~2.4% at the N=1000/2008
        #     cells every published measurement used, but +13.3% at N=904 and +0.23% at N=7024. It wins
        #     where the tier is DEEP or N is LARGE and LOSES where the tier is shallow and N small
        #     (cp=2 at N=904: -4% penalty removed against +13.3% pad). best_front_tile returns 256 for
        #     Dloc in [64,128), doubling the quantum.
        #
        # It ships OFF on the NEUTRALITY bar, which is per-shape and not an average: 7 of 9 shapes at
        # every cp were already fine and get ~0.7% SLOWER. On a uniform-N average default-ON would
        # arguably be net-positive at cp=8 -- that is exactly the averaging argument the bar rules out.
        #
        # The fix that earns the default is two front executors compiled under dynamic_shape and
        # dispatched per N from this same guard, so a declined shape is byte-identical again; there is no
        # single-compile route, because the operand's RANK is baked at compile time. It IS implemented
        # below (`_front_regime` / `_select_front_store` / `_make_front_pad`) but UNVERIFIED ON GPU --
        # cold compile is measured, numerics and the decline-cell perf re-check are not.
        self.front_pad_inner = (
            (os.environ.get("CPO_FRONT_PAD_INNER", "0") == "1")
            if front_pad_inner is None else bool(front_pad_inner)
        )
        # (a)'s HONEST COST — a DEFERRED second compile. Documented here rather than left to be
        # discovered at runtime. The two executors are built LAZILY, so the ctor still compiles exactly
        # ONE and `test_cold_compile_le_5s` (which times the construction wall, `_LAST_BUILD_S`) is
        # untouched by construction, not by luck. The other executor compiles on the FIRST forward that
        # hits the other regime. Measured cold (CPO_CACHE_ENABLED=0, fresh process, sm_90a):
        #
        #     coupled NVLink front   1.208 s unpadded / 1.257 s padded  -> deferred +1.25 s : noise
        #     hybrid-IB    front    15.503 s unpadded / 15.452 s padded -> deferred +15.5 s : a
        #         USER-VISIBLE STALL, and it lands MID-RUN rather than at startup.
        #
        # Two things that measurement settled, worth not re-deriving: the pad walk is compile-NEUTRAL
        # (±0.05 s, inconsistent sign, shape-independent — it does NOT hit the range_constexpr-blowup
        # class the ≤5 s rule exists to catch), and the ≤5 s bar is ALREADY breached by the BASELINE
        # (the IB front is 15.5 s on the unpadded default path), the test being WARN-not-fail per the
        # 2026-07-24 user override. So the compile bar never bound on (a).
        #
        # front_pad_eager trades the stall for startup: compile BOTH executors at construction, so a
        # latency-sensitive IB job pays a longer but EXPECTED startup instead of a surprising mid-run
        # pause. OPT-IN, because a job that only ever sees one regime would otherwise pay a whole extra
        # compile (and a second executor's symmetric recv) for something it never uses — and a job that
        # wants neither pays nothing by leaving front_pad_inner off. Inert unless front_pad_inner is on.
        self.front_pad_eager = (
            (os.environ.get("CPO_FRONT_PAD_EAGER", "0") == "1")
            if front_pad_eager is None else bool(front_pad_eager)
        )
        self.pe_map = pe_map
        self.cp = pe_map.cp
        self.B, self.N, self.D, self.dt = B, N, D, dt
        self.consumer = consumer
        self.eps = eps
        self.w = w
        if back_store not in ("pe_aligned", "design_e"):
            raise ValueError(f"back_store must be 'pe_aligned' or 'design_e'; got {back_store!r}.")
        self.back_store = back_store
        # HYBRID NVLink+IB A2A drain (opt-in; default OFF => BYTE-IDENTICAL to today's NVLink e2e).
        # ON => the front store gets ib_drain+ib_wide (is_p2p differential + wide-put coalesce) and the
        # back store gets the cluster_multislot decoupled-putwarp drain — the ib kwargs are gated so when
        # OFF they are NEVER passed (the configure calls are literally today's). An all-P2P (single-node
        # cp<=8) job additionally const_expr-collapses both IB paths back to the coupled/pe_aligned NVLink
        # store (front _a2a_has_ib_peers / back has_ib_peers = not all(is_p2p)). Requires pe_aligned (the
        # decoupled stack rides it). §10 R1: route2_ni (the transpose_in incoming no-copy store) NOW COMPOSES
        # with the IB drain — the front store was factored so the layout (N_i-stride-1) axis is ⊥ the
        # transport (ib_drain/ib_wide) axis; so hybrid_ib + route2_ni is the FAST incoming-hybrid path (no
        # .contiguous() transpose). No combo reject.
        #
        # hybrid_ib=None (the DEFAULT) => AUTO-DETECT the transport from the P2P topology. The coupled
        # (hybrid_ib=False) store is NVLink-ONLY (it TMA-S2Gs to each peer via that peer's nvshmem_ptr,
        # which is NULL for a cross-node non-P2P IB peer -> a silent CUDA illegal-address at the drain
        # barrier). A default cross-node build must therefore route IB peers through the ib_drain ring,
        # not the coupled store. Resolution is DEFERRED to AFTER the host-side shape guards below because
        # the build_p2p_table probe needs nvshmem — the shape guards must fire first so an invalid-shape
        # construction still raises cleanly on sm_120 without touching nvshmem (test_trimul_ib_integration
        # §E asserts the guards on sm_120). See the resolution just past the 2-D guard.
        self._hybrid_ib_request = hybrid_ib
        # back_cluster_n: OPTIONAL override of the back GEMM N-cluster width (cluster_shape_mnk[1]) AND the
        # cluster_multislot drain concentration cluster_n (they MUST match — gemm_sm90_a2a.py:807). None =>
        # the _resolve_back_config default ((1,2,1) 1-D / (1,1,1) 2-D). Used by the §3 pytest to force
        # cluster_n=4 for the C11 straddle verification. Applies to 1-D only (2-D is force-clamped (1,1,1)).
        self._back_cluster_n = int(back_cluster_n) if back_cluster_n is not None else None
        # ROUTE-2 (A) INCOMING no-copy (default OFF = byte-identical to today): the incoming front
        # A2A writes an N_i-stride-1 recv so the incoming einsum reads a_major="k" NATIVE — no
        # .transpose(-1,-2).contiguous() on the back operand. OUTGOING is UNCHANGED. Scope: 1-D
        # (cp1==1, j full) + B==1 (the producer store positions global-i by flat cp rank, no B fold)
        # + pe_aligned back. The transpose_in front is a distinct compile-cache variant -> a SECOND
        # front store (self._front_ni) is built for incoming (mirror the back's per-direction compile).
        self.route2_ni = bool(route2_ni)
        if route2_ni and back_store != "pe_aligned":
            raise ValueError("route2_ni=True requires back_store='pe_aligned' (design_e is per-shape).")
        # COMPOSITE-K INCOMING (§9; default OFF = byte-identical): the FAST incoming path. The front A2A
        # writes a per-rank-CONTIGUOUS D-major recv (transpose_in + D-major store; NO N_i-stride-1) so the
        # producer coalesces the FULL per-rank block into 32-KiB puts (vs route2_ni's b_j-pinned 640-B
        # cap deep in the IB knee); the back reads the composite K=(cp,Xg_pad). ~2x faster incoming.
        # OUTGOING is UNCHANGED (composite is incoming-only). Scope: 1-D (cp1==1) + B==1 + pe_aligned
        # (matches the validated back-read). A SECOND front store (self._front_comp) is built for incoming.
        # Mutually exclusive with route2_ni (they are two different incoming recv layouts).
        self.composite_k = bool(composite_k)
        if composite_k and route2_ni:
            raise ValueError("route2_ni and composite_k are mutually-exclusive incoming variants.")
        if composite_k and back_store != "pe_aligned":
            raise ValueError("composite_k=True requires back_store='pe_aligned' (design_e is per-shape).")
        # dynamic (dynamic-N mode): compile the fused front + back stores ONCE at THIS (B, N, D)
        # ANCHOR (mark_layout_dynamic; the back anchor auto-straddles for pe_aligned), then forward()
        # accepts an ARBITRARY runtime token-N — the symmetric recvs + ReshardLayout are (re)built
        # per distinct N (cached, SPMD-lockstep) and the M-dynamic LN + consumer take the runtime M.
        # ONE compile of ALL 4 kernels serves many token counts. Requires pe_aligned (design_e is
        # static per shape). Default OFF = the byte-identical per-shape path.
        self.dynamic = dynamic
        if dynamic and back_store != "pe_aligned":
            raise ValueError("dynamic=True requires back_store='pe_aligned' (design_e is per-shape).")
        self._rl_cache = {}
        # ---- workflow input-size guards (direction-INVARIANT: direction only transposes the back
        # operand; the size constraints below hold identically for "outgoing" and "incoming"). ----
        # D must split evenly across cp (the feature reshard S(0,1,2)->S(0,3,3)); a silent floor-div
        # would truncate the per-peer D-slice (was UNGUARDED at the Dloc = D//cp sites).
        if D % self.cp != 0:
            raise ValueError(
                f"D={D} must be divisible by cp={self.cp} (feature reshard; got D%cp={D % self.cp})."
            )
        # NATIVE 1-D vs 2-D token sharding (NO runtime reshard between them). The cp axes are
        # the sharded mesh dims (pe_map.cp_axis_sizes, in mesh-dim order); for TriMul they
        # shard token dims 1 (i) and, if 2-D, 2 (j). 1-D: cp_axis_sizes==(cp,) -> i alone is
        # split (N_i_loc=N/cp), j FULL (N_j_loc=N). 2-D: cp_axis_sizes==(cp0,cp1) -> i split
        # into cp0 blocks (N_i_loc=N/cp0) AND j split into cp1 blocks (N_j_loc=N/cp1). The
        # local token block I own is (B, N_i_loc, N_j_loc, D) in BOTH cases (1-D => N_j_loc==N).
        cp_axis_sizes = tuple(int(s) for s in pe_map.cp_axis_sizes)
        if len(cp_axis_sizes) > 2:
            raise ValueError(
                f"TriMulAutotuned supports 1-D or 2-D token sharding; got cp_axis_sizes={cp_axis_sizes}."
            )
        cp0 = cp_axis_sizes[0]
        cp1 = cp_axis_sizes[1] if len(cp_axis_sizes) > 1 else 1
        self.cp0, self.cp1 = cp0, cp1
        self.n_cp_axes = len(cp_axis_sizes)
        if composite_k and cp1 != 1:
            raise ValueError(
                f"composite_k incoming is 1-D (cp1==1) only; got cp_axis_sizes={cp_axis_sizes}. "
                f"cp1==1 is the _composite_remap's flat-cp K-rank synthesis; a 2-D token shard needs "
                f"the j-block as its own mode, which is route2_ni's job. The batch extent is not a "
                f"constraint on either variant any more."
            )
        if N % cp0 != 0 or N % cp1 != 0:
            raise ValueError(f"N={N} must be divisible by both cp axes {cp_axis_sizes}.")
        self.N_i_loc = N // cp0  # my i-axis (token dim 1) local extent
        self.N_j_loc = N // cp1  # my j-axis (token dim 2) local extent (== N when 1-D)
        # pe_aligned back store: 16-B (bf16) stride-1 TMA-S2G -> the token-j (recv col) extent must
        # be %8, and on a 2-D shard EACH per-axis local extent must be %8 (the back 2-D N%8 store-
        # corruption caller-contract — a silent partial-tile store bug the outlier gate misses). This
        # is a per-N constraint (every runtime N once dynamic-N lands in PHASE 2).
        if N % 8 != 0:
            raise ValueError(
                f"N={N} must be a multiple of 8 (pe_aligned back store 16-B stride-1 TMA-S2G)."
            )
        if cp1 > 1 and (self.N_i_loc % 8 != 0 or self.N_j_loc % 8 != 0):
            raise ValueError(
                f"2-D back store requires each per-axis local token extent %8==0 (16-B TMA-S2G on "
                f"both axes): N_i_loc=N//cp0={self.N_i_loc}, N_j_loc=N//cp1={self.N_j_loc}."
            )
        # ---- HYBRID NVLink+IB transport AUTO-DETECT (resolves self._hybrid_ib_request; fixes the
        # coupled-reaches-IB CUDA illegal-address on a default cross-node build). Runs HERE, after the
        # host-side shape guards, because build_p2p_table needs nvshmem (an invalid-shape build must still
        # raise on sm_120 without touching nvshmem). The probe is the SAME module-level build_p2p_table the
        # store's _configure_ib_drain uses (nvshmem TEAM_SHARED translate over cp_pe_table) so the auto
        # decision is BIT-IDENTICAL to the store's IB-vs-NVLink select; it is host-side / buffer-free / no
        # collective / SPMD-consistent (every rank of a symmetric mesh sees the same is_p2p). Semantics:
        #   * hybrid_ib=None (DEFAULT) -> hybrid_ib := has_ib_peers. Single-node all-P2P -> False (coupled,
        #     BYTE-IDENTICAL, the all-P2P const_expr collapse); any cross-node IB peer -> True (ib_drain).
        #   * explicit hybrid_ib=True -> honored (all-P2P collapses to coupled -> still byte-identical).
        #   * explicit hybrid_ib=False WITH IB peers -> a clear raise (coupled CANNOT reach an IB peer),
        #     never the silent IMA. False on an all-P2P job is honored (coupled).
        from fold_cp_ops.distributed.gemm_sm90_a2a import build_p2p_table
        _has_ib_peers = not all(build_p2p_table(tuple(int(x) for x in pe_map.cp_pe_table.tolist())))
        self._has_ib_peers = _has_ib_peers  # §5.5 drain auto-select: cross-node = expensive barrier
        if self._hybrid_ib_request is None:
            hybrid_ib = _has_ib_peers
        else:
            hybrid_ib = bool(self._hybrid_ib_request)
            if not hybrid_ib and _has_ib_peers:
                raise ValueError(
                    "This job has cross-node IB peers (not all cp peers are P2P/NVLink-reachable), but "
                    "hybrid_ib=False selects the coupled NVLink-only A2A store, which faults with a CUDA "
                    "illegal address on a TMA-S2G to an IB peer. Pass hybrid_ib=True, or the default "
                    "hybrid_ib=None to auto-detect the transport from the P2P topology."
                )
        self.hybrid_ib = hybrid_ib
        # NO batch demotion here. A cross-node job at B > 1 used to be downgraded to the PLAIN
        # incoming store because the IB drain's ring metadata carried a FLAT recv column that would
        # have addressed plane 0 for every plane. The record now carries the plane
        # (`DualGatedGemmDistSm90._DECOUPLED_META_B_INDEX`), so the fast store is valid at every
        # batch extent on every transport, and the demotion -- along with the `configure_a2a`
        # refusal it existed to dodge -- is gone. A demotion left standing after its cause is
        # removed is a silent ~2x nobody finds, which is why the two were deleted together.
        if self.hybrid_ib and back_store != "pe_aligned":
            raise ValueError("hybrid_ib=True requires back_store='pe_aligned' (the cluster_multislot / "
                             "front ib_drain machinery rides the pe_aligned decoupled stack; design_e is "
                             "static with no ring).")
        # device_mesh/placements: required for the 2-D back store's configure_a2a_sharded
        # (the tile->peer unravel over (cp0,cp1)). For 1-D the pe_map alone suffices. Default
        # reconstruct placements from the cp shard dims if not supplied.
        self._device_mesh = device_mesh
        if placements is None:
            placements = [Shard(int(d)) for d in pe_map.cp_shard_tensor_dims]
        self._placements = placements
        self.N_loc = self.N_i_loc  # back-compat alias (the i-axis peer block extent)
        self.D_loc = D // self.cp
        self.M = B * self.N_i_loc * self.N_j_loc  # my local token count (front operand M extent)
        # The front store does not yet fuse the up/gate PROJECTION biases (only the LN
        # gain+bias). The wave-1 DTensor baseline + fp32 oracle use bias=0 for the front
        # projections; guard against a silent drop of a non-zero p_in_b/g_in_b.
        # The front projection biases, in the order `interleave_dual_weights` takes them: the
        # weights go in as (g_in, p_in) = (gate, up), so the biases are (bg, bp) = (g_in_b, p_in_b).
        # Fed to every front store below; None -> the epilogue term is const_expr-pruned.
        self._bg_in = w.get("g_in_b")
        self._bp_in = w.get("p_in_b")
        # ---- perf-config sourcing (NO hardcoded perf literal): each fused kernel's tile/pingpong
        # come from its OWN size-heuristic for THIS (B, N, D); explicit front_/back_ overrides win.
        # Front tile from the stagec heuristic (same plain-dual tiling); back (square batched einsum)
        # tile comes from nvMatmulHeuristics, constraint-gated to the design-E store. is_persistent
        # is required True by both fused-store designs (a param, so it is not a buried literal).
        _dev = w["g_in_w"].device
        # OPT-IN distributed autotune (default OFF -> byte-identical to the heuristic path). When ON,
        # the back/front tile+cluster are picked by the distributed autotuner (consensus across ranks)
        # and freeze-cached per shape; explicit front_/back_ overrides still win. See fused_trimul_autotune.
        _do_autotune = (autotune_config if autotune_config is not None
                        else os.environ.get("CPO_DIST_AUTOTUNE") == "1")
        _dm = None
        if _do_autotune:
            from fold_cp_ops.distributed.distributed_manager import DistributedManager
            from fold_cp_ops.distributed.workflows.trimul_autotune_policy import (
                autotune_back_config,
                autotune_front_config,
            )
            _dm = DistributedManager()
        # BAKED sm90 H100+IB config gate (TASK #49): compute arch_is_sm90 ONCE here (reused for the §5.5
        # drain select below, ~:1899) and resolve the harvested (D, cp0, cp1) config into _baked. _baked is
        # None unless the venue matches the harvest venue (sm90 + cross-node IB + autotune OFF) — so the
        # NVLink / non-sm90 / autotune paths are byte-identical (the baked branch never fires). Precedence
        # (highest first): explicit front_/back_tile_mn override > autotune > _baked > heuristic — the
        # override is applied AFTER the resolve below, so _baked is consulted only when the tile arg is None.
        arch_is_sm90 = get_device_capacity(self.pe_map.device)[0] == 9  # sm90 (H100/H200) measured venue
        _baked = _resolve_sm90_ib_config(D, cp0, cp1, arch_is_sm90, self._has_ib_peers, _do_autotune)
        f_W = 128  # front wide-put batch: autotune-picked on an engaged has-IB venue; else the store default
        if _do_autotune and front_tile_mn is None:
            # dynamic=True -> resolve at DYNAMIC_ANCHOR_N, NOT at this build's N. One compiled instance
            # serves every runtime N, so its config must not depend on which N happened to build it;
            # anchoring is also what makes the DTensor entry (N from x.shape) and the raw entry resolve
            # the SAME freeze file, and what lets both hit the FORM-A pre-freeze. See anchor_n.
            f_tm, f_tn, _f_pp_heur, f_W = autotune_front_config(
                pe_map, B, N, D, dt, _dm, eps=eps, dynamic=dynamic
            )
        elif _baked is not None and front_tile_mn is None:
            f_tm, f_tn, _f_pp_heur, f_W = _baked[0]  # harvested H100+IB front (f_tm, f_tn, f_pp, f_W)
        else:
            f_tm, f_tn, _f_pp_heur = _resolve_front_config(self.M, D, _dev)
        # The STAGED front kernel (DualGatedGemmStagedSm90) is COOPERATIVE-ONLY (it asserts
        # `not pingpong`), so force pingpong=False regardless of the (stagec) heuristic's pick.
        f_pp = False
        # FRONT-A2A store constraint: the postact tile_N (= f_tn//2) must lie within ONE peer's D-slice.
        # Resolve f_tn to the tile production RUNS (mult-32 clamp + best_front_tile C12 residual) via the
        # SHARED _resolve_front_tile_n — the SAME resolver _FrontProxyAdapter._valid_tile_ns uses, so the
        # autotune freeze can never offer a tile prod clamps away (see the resolver docstring). PERF-NEUTRAL.
        f_tn = _resolve_front_tile_n(self.D_loc, f_tn)
        if front_tile_mn is not None:
            f_tm, f_tn = front_tile_mn
        if front_pingpong:
            raise ValueError(
                "TriMulAutotuned staged front is cooperative-only; front_pingpong unsupported."
            )
        if _do_autotune and back_tile_mn is None:
            b_tm, b_tn, b_pp, b_cluster = autotune_back_config(
                pe_map, B, N, D, dt, _dm,
                device_mesh=self._device_mesh, placements=self._placements,
                dynamic=dynamic,  # anchored per the front note above
            )
        elif _baked is not None and back_tile_mn is None:
            b_tm, b_tn, b_pp, b_cluster = _baked[1]  # harvested H100+IB back (b_tm, b_tn, b_pp, b_cluster)
        else:
            b_tm, b_tn, b_pp, b_cluster = _resolve_back_config(B, N, D, self.cp, dt, _dev)
        if back_tile_mn is not None:
            b_tm, b_tn = back_tile_mn
        if back_pingpong is not None:
            b_pp = back_pingpong
        # design_e constraint: a CTA tile must map to ONE peer block on BOTH token axes (the back
        # store's tile->peer unravel). i-axis: N_i_loc % tile_M == 0; j-axis (only when split):
        # N_j_loc % tile_N == 0. Force (128,128) cluster(1,1,1) if the heuristic pick straddles a peer
        # block. pe_aligned HANDLES a straddling per-peer block (arbitrary_n / per-peer M-tiling) so
        # it does NOT downgrade here. cluster_N=2 (the 2040-fix lever) is validated for the 1-D store;
        # on the 2-D path it reverts to the conservative (1,1,1) (2-D peer-unravel + N-cluster not
        # co-validated).
        if self.back_store == "design_e" and (
            self.N_i_loc % b_tm != 0 or (cp1 > 1 and self.N_j_loc % b_tn != 0)
        ):
            b_tm, b_tn, b_cluster = 128, 128, (1, 1, 1)
        if self._back_cluster_n is not None:
            # explicit N-cluster override (the §3 C11 cluster_n=4 straddle verification): set BOTH the GEMM
            # N-cluster AND the cluster_multislot drain concentration (the back store derives cluster_n from
            # cluster_shape_mnk[1], and configure_a2a_gemm_native:807 requires they match). 1-D only — the
            # 2-D force below still clamps (1,1,1) (2-D N-cluster not co-validated).
            b_cluster = (1, self._back_cluster_n, 1)
        if cp1 > 1:
            b_cluster = (1, 1, 1)  # 2-D store: keep the validated (1,1,1) (1-D-only cluster_N=2 fix)
        # cluster_M MUST be 1 for the pe_aligned/design-E store (defensive; _resolve already pins it).
        assert b_cluster[0] == 1, f"back A2A GEMM requires cluster_M==1, got {b_cluster}"
        self._front_cfg, self._back_cfg = (f_tm, f_tn, f_pp), (b_tm, b_tn, b_pp, b_cluster)
        self.is_persistent = is_persistent

        # ReshardLayout for the host-side back-recv unpack view (back_unpack_gemm_native —
        # the (M, D) LayoutLeft value the consumer reads). The staged front recv is already
        # einsum-native D-major (a/b are direct recv-slice views), so the front no longer
        # needs front_unpack_dmajor. The fused stores OWN the comm; this just shapes the view.
        # 2-D-aware: ReshardLayout reads pe_map.cp_shard_tensor_dims for the token axes. In dynamic
        # mode this is the anchor rl; forward() swaps in a per-runtime-N rl (cached in _rl_cache).
        self.rl = ReshardLayout(pe_map, B, N, D, feat_width=D)
        # Keyed (B, N) to match `_rebind_runtime` -- an N-only seed would be found by a lookup
        # at a DIFFERENT batch extent and silently return this one's layout.
        self._rl_cache[(B, N)] = self.rl

        # ---- front LN gain + the STACKED projection weights --------------------
        # x_norm = LN(x) is materialized ONCE in forward (layernorm_fwd) and fed to the
        # staged front (_normalize=False). The dual GEMM's Wg = gate proj (g_in_w (2D,K)),
        # Wp = up proj (p_in_w (2D,K)), STACKED [Wa;Wb]: glu(x_norm@Wg^T, x_norm@Wp^T) -> the
        # width-2D postact whose halves are a=[:D], b=[D:] (one invocation emits both).
        K = D  # contraction = feature D
        g_in = w["g_in_w"].to(dt)  # (2D, K) stacked gate proj
        p_in = w["p_in_w"].to(dt)  # (2D, K) stacked up   proj
        self._norm_in_w = w["norm_in_w"].float()  # (D,) front LN gain (for layernorm_fwd)
        self._norm_in_b = w["norm_in_b"].float() if w.get("norm_in_b") is not None else None
        # MASK (mask composition): the pair mask is applied to the gated glu postact (mMaskColVec, LOCAL /
        # pre-A2A). The plain D-major front store (OUTGOING + route2_ni=False INCOMING) reads the mask
        # in native (b, i_loc, j_loc) row order. The transpose_in incoming variants (route2_ni /
        # composite_k) WALK the token-M axis in transposed (b, j_loc, i_loc) order (the (X<->Y)-swapped
        # p-inner walk, _remap_A_operand_layout: flat GEMM-M i reads native b*Xg*Yg+X*Yg+Y), and the
        # mMaskColVec is loaded at that same flat-M i — so the mask col-vec is PRE-TRANSPOSED to (b, j, i)
        # for those paths (built in forward()). Host-side O(M) reorder, sub-leading along N_token (frugal,
        # no I/O-order scratch). So has_mask NOW COMPOSES with route2_ni / composite_k — no combo reject.
        self.has_mask = bool(has_mask)
        self._front = DualGatedGemmDistStore(  # bg2/bp2 forwarded below
            pe_map,
            g_in,
            p_in,
            self.M,
            K,
            D,
            dt,
            eps=eps,
            tile_shape_mn=(f_tm, f_tn),
            pingpong=f_pp,
            is_persistent=is_persistent,
            dynamic_shape=dynamic,
            hybrid_ib=self.hybrid_ib,  # OUTGOING + route2_ni=False INCOMING ride this plain store's IB drain
            ib_wide_batch=f_W,  # autotuned wide-put batch (default 128 == store default -> byte-identical off)
            has_mask=self.has_mask,
            # (a) DUAL EXECUTOR: self._front is ALWAYS the unpadded walk, so a shape the guard declines
            # runs the pre-pad_inner kernel byte-identically. The padded walk is a SECOND executor built lazily
            # in _select_front_store on the first N that engages the guard — which is what keeps the
            # ctor at exactly one front compile. B/N_i_loc/N_j_loc feed the per-N geometry either way.
            pad_inner=False,
            bg2=self._bg_in,
            bp2=self._bp_in,
            B=B,
            N_i_loc=self.N_i_loc,
            N_j_loc=self.N_j_loc,
        )
        self._front_pad = None  # (a) the pad_inner executor; built on first use, never at ctor
        # NOTE: the OPT-IN eager build lives BELOW, after `_front_ni_cfg` is assigned — `_make_front_pad`
        # reads it, so building here would raise AttributeError. Caught by the eager cell on a multi-node run.
        # route2_ni: a SECOND front store with the transpose_in N_i-stride-1 producer store, used for
        # INCOMING (the plain self._front stays for OUTGOING). Mirrors the back's per-direction compile.
        #
        # DYNAMIC-N: ONE front compile serves every runtime N. The committed _a2a_route2_ni copy_fn
        # derives n_x at RUNTIME when _a2a_dynamic (dual_gated_gemm_staged_a2a.py, commit a8ecd15) and
        # the 3-D recv is marked N_i/N_j-shape-dynamic (feature static) — so the dynamic front rebinds
        # its symmetric recv per-N (rebind_M) with NO recompile. Static TriMulAutotuned builds one front for
        # its single shape. (Earlier per-N front recompile was the workaround for the pre-a8ecd15 baked
        # const_expr n_x; retired now that n_x is runtime.)
        self._front_ni = None
        self._front_comp = None  # composite_k incoming front (transpose_in + D-major padded recv)
        self._front_ni_cfg = dict(  # frozen build args for the route2_ni/composite front (_make_front_*)
            pe_map=pe_map, g_in=g_in, p_in=p_in, K=K, D=D, dt=dt, eps=eps,
            tile_shape_mn=(f_tm, f_tn), pingpong=f_pp, is_persistent=is_persistent, B=B, ib_wide_batch=f_W,
            bg_in=self._bg_in, bp_in=self._bp_in,
        )
        if self.front_pad_inner and self.front_pad_eager:
            # OPT-IN eager build: pay the second compile at CONSTRUCTION so no forward ever stalls on
            # it. Worth it only where the deferred cost is large and a mid-run pause is unacceptable —
            # i.e. the hybrid-IB front, whose second compile is ~15.5 s. Built at the ctor's N; it is
            # mark_layout_dynamic and rebinds per N exactly like _front_ni / _front_comp, so the anchor
            # shape does not constrain which N it can later serve.
            #
            # MUST stay below `_front_ni_cfg` — `_make_front_pad` reads it. It was originally 20 lines
            # higher and raised AttributeError on every eager ctor; the lazy path never hit it because
            # `_select_front_store` runs at forward time, long after __init__ completes.
            self._front_pad = self._make_front_pad(
                self.M, self.N_i_loc, self.N_j_loc, dynamic_shape=dynamic
            )
        if self.route2_ni:
            self._front_ni = self._make_front_ni(
                self.M, self.N_i_loc, self.N_j_loc, dynamic_shape=dynamic
            )
        if self.composite_k:
            # COMPOSITE-K: a SECOND front store (transpose_in + D-major padded recv), used for INCOMING
            # (self._front stays for OUTGOING). Same dynamic-N one-compile-many-N story as route2_ni (the
            # transpose_in walk + padded 2-D recv are marked shape-dynamic; rebind_M rebinds per N).
            self._front_comp = self._make_front_comp(
                self.M, self.N_i_loc, self.N_j_loc, dynamic_shape=dynamic
            )
        self._back = GemmA2AStore(
            pe_map,
            B,
            N,
            D,
            dt,
            tile_shape_mn=(b_tm, b_tn),
            cluster_shape_mnk=b_cluster,
            pingpong=b_pp,
            is_persistent=is_persistent,
            device_mesh=self._device_mesh,
            placements=self._placements,
            back_store=back_store,
            dynamic_shape=dynamic,
            route2_ni=self.route2_ni,
            composite_k=self.composite_k,
            hybrid_ib=self.hybrid_ib,  # cluster_multislot drain (cluster_n defaults from b_cluster[1])
        )

        # PER-PEER SIGNAL-WAIT A2A DRAIN (PROTOTYPE; opt-in CPO_A2A_SIGNAL_DRAIN=1). Default OFF =>
        # the barrier path is BYTE-IDENTICAL to today (no pad malloc, no code-path change). When ON the
        # FRONT post-A2A drain (front_a2a) swaps the global quiet+barrier_all for the point-to-point
        # signal-wait (_front_drain). Constructed LAST (after every store's collective recv malloc) so the
        # collective (cp,) pad malloc has NO host-sync interposed vs the prior recv malloc (#B rule); the
        # pe_map is FLAT-cp (the front feature scatter is 2-D-invariant). SPMD: the env is read identically
        # on every rank so all ranks allocate (or none), keeping the collective malloc symmetric.
        self._signal_drain = bool(int(os.environ.get("CPO_A2A_SIGNAL_DRAIN", "0")))  # force-ON override
        # §5.5 per-config AUTO-SELECT (DEFAULT-ON; CPO_A2A_AUTO_DRAIN=0 forces barrier everywhere): engage the
        # host-signal drain by TOPOLOGY, not N. Both drains' overhead is a fixed per-cp/link cost (N-INDEPENDENT).
        # CROSS-NODE (has_ib_peers): the IB global barrier_all rendezvous is expensive (nsys: barrier_on_stream
        # = 13% of GPU time at cp16 N4096, quiet = 0.0%) -> the point-to-point host-signal wins. INTRA-NODE NVLink:
        # barrier_all is cheap -> host-signal's fixed host-dispatch loses (cp8 signal +1.7-2.4% across all N) ->
        # barrier. Arch-keyed (feedback_heuristics_arch_specific): UNTUNED arch (not sm90) -> barrier. The earlier
        # N-crossover was WITHDRAWN (its small-N regressions were noise within the ~15% baseline wobble; the
        # measured N* oscillated 1792 vs 3072). Force-ON bypasses the gate.
        self._auto_drain = bool(int(os.environ.get("CPO_A2A_AUTO_DRAIN", "1")))  # DEFAULT-ON (shipped behavior); CPO_A2A_AUTO_DRAIN=0 forces barrier everywhere
        # CPO_A2A_SIGNAL_DEVICE=1 force-ON => the front-post drain uses the POST-QUIET DEVICE-signal kernel
        # (_A2ADeviceSignal: 1 kernel launch, cp threads) instead of the host-signal drain's cp
        # signal_op_on_stream host dispatches — the same completion contract, removing the host-launch cost.
        self._signal_device = bool(int(os.environ.get("CPO_A2A_SIGNAL_DEVICE", "0")))
        # CAPABILITY BOUNDARY (not a bug-dodge): the POST-QUIET device-signal is NVLink-ONLY. A device-issued
        # IBGDA signal_op has NO completion path in cute-DSL 4.5.2 — device nvshmem quiet is NVVM-rejected
        # (reference_nvshmem_device_global_quiet_ffi) and a HOST stream quiet does NOT flush the device-issued
        # IB QP (MEASURED: cp16 2-node hangs the agg-wait; a host quiet AFTER the kernel is futile). So on
        # cross-node (IB) peers fail-FAST at construction (before any kernel) — never route IB to the hang.
        # The host-signal drain (_A2ASignalDrain, CPO_A2A_SIGNAL_DRAIN=1) IS IB-capable (host quiet flushes
        # host-issued IB) and is the cross-node path; auto-select (below) only ever picks host for IB.
        if self._signal_device and self._has_ib_peers:
            raise ValueError(
                "CPO_A2A_SIGNAL_DEVICE (post-quiet DEVICE-signal A2A drain) is NVLink-only: a device-issued "
                "IBGDA signal_op has no completion path in cute-DSL 4.5.2 (device nvshmem quiet is NVVM-rejected; "
                "a host stream quiet does not flush the device-issued IB QP — measured to hang the receiver's "
                "agg-wait). Use CPO_A2A_SIGNAL_DRAIN=1 (host-signal, IB-capable) for cross-node peers."
            )
        # BACK-POST host-signal opt-in (default OFF): VALIDATED-NEGATIVE at cp16 (erodes the front win) -> never
        # auto-picked; CPO_A2A_BACK_SIGNAL=1 engages it (with host-signal) ONLY as the measurement/record path.
        self._back_signal_optin = bool(int(os.environ.get("CPO_A2A_BACK_SIGNAL", "0")))
        # §5.5 per-config drain AUTO-SELECT -> "barrier" (default) | "host" | "device". TOPOLOGY-keyed (NOT N):
        # force-flags override; else auto engages HOST-signal iff CROSS-NODE (has_ib_peers) on sm90 -- the IB
        # global barrier_all is expensive so the point-to-point signal wins; intra-node NVLink barrier is cheap
        # so signal's fixed host-dispatch loses -> barrier. DEVICE-signal is NEVER auto-picked (it TIES barrier
        # intra-node -> no runtime gain for its ~1-2s construct-time device-kernel compile) and is IB-blocked
        # (guarded above); it stays a NVLink-only FORCE-ON option. Untuned arch (not sm90) -> barrier (never
        # enable an unmeasured signal win). Mirrors feedback_heuristics_arch_specific.
        # use_device_signal_nvlink (clean API arg vs the hacky CPO_A2A_SIGNAL_DEVICE env): SELF-GATING —
        # engages the NVLink-niche DEVICE-signal drain ONLY when ALL PEs are NVLink (not has_ib_peers); on IB it
        # gracefully falls back to barrier/auto (NO raise, unlike the env force-flag). Env force still honored.
        self._device_signal_nvlink = bool(use_device_signal_nvlink)
        # arch_is_sm90 computed ONCE above (the TASK #49 baked-config gate hoisted it before the front/back
        # config resolve); reused here for the §5.5 drain select — same value, no re-query.
        drain_kind = self._choose_drain_kind(
            self._signal_device, self._signal_drain, self._auto_drain, arch_is_sm90, self._has_ib_peers,
            self._device_signal_nvlink,
        )
        self._use_signal_drain = drain_kind != "barrier"
        if drain_kind == "device":
            self._front_sig = _A2ADeviceSignal(self.pe_map)
        elif drain_kind == "host":
            self._front_sig = _A2ASignalDrain(self.pe_map)
        else:
            self._front_sig = None
        # BACK-POST completion drain: VALIDATED-NEGATIVE at cp16 (measured +0.2..+2.5% regression — it ERODES
        # the front-post win: the back barrier is CHEAP, O(N³) back-GEMM-dominated, so host-signal-ing it costs
        # more than it saves). So it is NEVER auto-picked: gated behind a SEPARATE default-OFF opt-in
        # (CPO_A2A_BACK_SIGNAL=1, requires host-signal engaged) kept ONLY as the measurement/record path.
        # Default/auto -> back-post = barrier (the shipped front-post gate is the win). When engaged it REUSES
        # the front's _A2ASignalDrain object (SAME cp group -> identical cp_pe_table peer set + agg handshake;
        # shared monotonic ctr sequences front-post then back-post under the KEPT pre-barriers = zero extra GMEM,
        # no 2nd-malloc #B risk; see §4a for the shared-pad<->pre-barrier coupling the deferred WAR step must
        # honor).
        self._use_back_signal = self._back_signal_optin and drain_kind == "host"
        self._back_sig = self._front_sig if self._use_back_signal else None

        # back-half consumer weights (cuEq: out-gate consumes xn, value = LN(tri)).
        self._Wp_out = w["p_out_w"].to(dt)
        self._Wg_out = w["g_out_w"].to(dt)
        self._bp_out = w["p_out_b"].to(dt) if w.get("p_out_b") is not None else None
        self._bg_out = w["g_out_b"].to(dt) if w.get("g_out_b") is not None else None
        self._norm_out_w = w["norm_out_w"]
        self._norm_out_b = w["norm_out_b"]

    @staticmethod
    def _choose_drain_kind(signal_device, signal_drain, auto_drain, arch_is_sm90, has_ib_peers,
                           device_signal_nvlink=False):
        """Pure §5.5 gate -> 'barrier' (default) | 'host' | 'device'. TOPOLOGY-keyed, N-INDEPENDENT: both drains'
        overhead is a fixed per-cp/link cost, so NOTHING is gated on N. Force-flags override; else auto engages
        HOST-signal iff CROSS-NODE (has_ib_peers) on sm90 -- the IB global barrier_all rendezvous is expensive so
        the point-to-point signal wins (its margin grows with N via the barrier's skew-absorption, but the SIGN is
        topology-set); intra-node NVLink barrier is cheap so signal's fixed host-dispatch loses -> barrier.
        device_signal_nvlink (clean API arg use_device_signal_nvlink=True) SELF-GATES the NVLink-niche DEVICE
        drain to ALL-NVLink (not has_ib_peers) on sm90; on IB it gracefully falls through (NO raise, unlike the
        CPO_A2A_SIGNAL_DEVICE env force-flag). Otherwise DEVICE is never auto-picked. Untuned arch -> barrier.
        PURE function so the routing is unit-testable with NO construction / nvshmem.
        NOTE: the earlier N>=_HOST_CROSSOVER_N gate was WITHDRAWN -- its sub-1% small-N regressions were within the
        ~15% cross-process baseline wobble (noise) and the measured N* oscillated (1792 vs 3072) = not a real
        boundary; the drain choice is set by topology, not N."""
        if signal_device:
            return "device"
        if signal_drain:
            return "host"
        if device_signal_nvlink and arch_is_sm90 and not has_ib_peers:
            return "device"   # SELF-GATING NVLink-only device-signal; graceful barrier fallback on IB
        if auto_drain and arch_is_sm90 and has_ib_peers:
            return "host"
        return "barrier"

    # ---- per-fused-reshard drain (quiet + barrier) -------------------------------
    @staticmethod
    def _drain():
        import nvshmem.core
        import nvshmem.core.rma as nvshmem_rma

        nvshmem_rma.quiet(stream=torch.cuda.current_stream())
        nvshmem.core.barrier_all(stream=torch.cuda.current_stream())

    @staticmethod
    def _barrier():
        import nvshmem.core

        nvshmem.core.barrier_all(stream=torch.cuda.current_stream())

    def _front_drain(self):
        """FRONT post-A2A drain. Uses the host-signal drain when engaged (CPO_A2A_SIGNAL_DRAIN=1
        force-ON, OR the §5.5 auto-select (default-on) picks it for the cross-node (IB) regime where nsys
        proved barrier_all's 13% GLOBAL-SYNC is removable) — dropping the global
        barrier_all for the point-to-point per-peer signal-wait; else the barrier drain (byte-identical).
        The PRE-A2A _barrier (WAR guard) is kept in BOTH paths for now."""
        if self._use_signal_drain:
            self._front_sig.drain()
        else:
            self._drain()

    def _back_drain(self):
        """BACK post-A2A completion drain (§5.5 gate, host-signal only — device is a FRONT NVLink-niche).
        Reuses the SAME host-signal drain object as the front: the back reshards all-to-all over the SAME
        cp group, so the cp_pe_table peer set + the cp-wide agg handshake are identical, and the shared
        MONOTONIC ctr sequences front-post then back-post per forward (SPMD lockstep under the KEPT
        pre-barriers). Else the barrier drain (byte-identical). The PRE-A2A _barrier (WAR guard) is KEPT."""
        if self._use_back_signal:
            self._back_sig.drain()
        else:
            self._drain()

    def _make_front_ni(self, M, N_i_loc, N_j_loc, *, dynamic_shape=False):
        """Build a route2_ni front (the transpose_in N_i-stride-1 producer store) for THIS token
        geometry. ``dynamic_shape`` (dynamic TriMulAutotuned) compiles ONE executor that serves every N via
        rebind_M (the a8ecd15 runtime-n_x copy_fn + the N_i/N_j-dynamic 3-D recv marking); static
        TriMulAutotuned builds one per instance."""
        cfg = self._front_ni_cfg
        return DualGatedGemmDistStore(
            cfg["pe_map"], cfg["g_in"], cfg["p_in"], M, cfg["K"], cfg["D"], cfg["dt"],
            eps=cfg["eps"], tile_shape_mn=cfg["tile_shape_mn"], pingpong=cfg["pingpong"],
            is_persistent=cfg["is_persistent"], dynamic_shape=dynamic_shape,
            route2_ni=True, B=cfg["B"], b_dynamic=dynamic_shape, N_i_loc=N_i_loc, N_j_loc=N_j_loc,
            bg2=cfg["bg_in"], bp2=cfg["bp_in"],
            hybrid_ib=self.hybrid_ib,  # §10 R1: route2_ni composes with ib_drain/ib_wide (N_i coalesce)
            ib_wide_batch=cfg["ib_wide_batch"],  # autotuned wide-put batch (default 128 -> byte-identical off)
            has_mask=self.has_mask,  # mask composes (forward passes the (b,j,i)-transposed col-vec)
        )

    def _front_regime(self, N_i_loc=None, N_j_loc=None):
        """(a) — which front executor serves THIS token geometry, and its padded inner extent.

        ONE source of truth for the per-N regime decision, used by BOTH the mask build in ``forward``
        (which runs BEFORE the store is bound) and the store select + unpack in ``front_a2a``. Returns
        ``(use_pad, Yg, rpp_eff, N_j_pad)``.

        ``use_pad`` is False whenever the guard declines — and under (a) a declined shape is then served
        by ``self._front``, the UNMODIFIED plain executor, so it is byte-identical to the pre-P2 kernel
        rather than paying the 2-mode walk with ``Xg == 1``. That is the whole point of the dual
        executor: the +0.61%/+0.76% measured on declined shapes (log §13.2) came from making one
        compiled kernel serve both regimes, and the only way out is a second compile, because the
        operand's RANK is baked."""
        N_i_loc = self.N_i_loc if N_i_loc is None else int(N_i_loc)
        N_j_loc = self.N_j_loc if N_j_loc is None else int(N_j_loc)
        if not self.front_pad_inner:
            return False, self.B * N_i_loc * N_j_loc, self.B * N_i_loc * N_j_loc, N_j_loc
        Yg, rpp_eff, N_j_pad = front_pad_inner_geometry(
            self.B, N_i_loc, N_j_loc, self._front_cfg[0]
        )
        return (N_j_pad != N_j_loc), Yg, rpp_eff, N_j_pad

    def _select_front_store(self):
        """(a) — the front executor for THIS runtime geometry, building the padded one ON FIRST USE.

        LAZY by design and that laziness is load-bearing: the ctor then compiles exactly ONE front
        executor, so ``test_cold_compile_le_5s`` (which times the TriMulAutotuned construction wall) is
        untouched. The second compile — measured at +1.25 s coupled / +15.5 s hybrid-IB, cold, cache
        off — is deferred to the first forward that actually hits the padded regime, and a job that
        only ever sees one regime never pays it at all."""
        use_pad, _yg, _rpp, _njp = self._front_regime()
        if not use_pad:
            return self._front
        if self._front_pad is None:
            self._front_pad = self._make_front_pad(
                self.M, self.N_i_loc, self.N_j_loc, dynamic_shape=self.dynamic
            )
        return self._front_pad

    def _make_front_pad(self, M, N_i_loc, N_j_loc, *, dynamic_shape=False):
        """Build the P2 PAD-INNER front (the padded-innermost-token-extent walk) for this geometry.

        Mirrors ``_make_front_ni`` / ``_make_front_comp`` — same frozen build cfg, the only difference
        being ``pad_inner=True``. Reached only when ``front_pad_inner`` is on AND the per-N guard
        engages, so the default-OFF tree never constructs it."""
        cfg = self._front_ni_cfg
        return DualGatedGemmDistStore(
            cfg["pe_map"], cfg["g_in"], cfg["p_in"], M, cfg["K"], cfg["D"], cfg["dt"],
            eps=cfg["eps"], tile_shape_mn=cfg["tile_shape_mn"], pingpong=cfg["pingpong"],
            is_persistent=cfg["is_persistent"], dynamic_shape=dynamic_shape,
            pad_inner=True, B=cfg["B"], N_i_loc=N_i_loc, N_j_loc=N_j_loc,
            bg2=cfg["bg_in"], bp2=cfg["bp_in"],
            hybrid_ib=self.hybrid_ib,
            ib_wide_batch=cfg["ib_wide_batch"],
            has_mask=self.has_mask,
        )

    def _make_front_comp(self, M, N_i_loc, N_j_loc, *, dynamic_shape=False):
        """Build a COMPOSITE-K front (transpose_in + the D-MAJOR store -> a padded per-rank-contiguous
        2-D recv) for THIS token geometry (§9). Same one-compile-many-N story as _make_front_ni (the
        transpose_in padded walk + 2-D recv are shape-dynamic; rebind_M rebinds per N). Uses the SAME
        frozen build cfg — the ONLY difference from _make_front_ni is composite_k=True (not route2_ni)."""
        cfg = self._front_ni_cfg
        return DualGatedGemmDistStore(
            cfg["pe_map"], cfg["g_in"], cfg["p_in"], M, cfg["K"], cfg["D"], cfg["dt"],
            eps=cfg["eps"], tile_shape_mn=cfg["tile_shape_mn"], pingpong=cfg["pingpong"],
            is_persistent=cfg["is_persistent"], dynamic_shape=dynamic_shape,
            composite_k=True, B=cfg["B"], b_dynamic=dynamic_shape, N_i_loc=N_i_loc, N_j_loc=N_j_loc,
            bg2=cfg["bg_in"], bp2=cfg["bp_in"],
            hybrid_ib=self.hybrid_ib,  # transpose_in composes with ib_drain/ib_wide (layout ⊥ transport)
            ib_wide_batch=cfg["ib_wide_batch"],  # autotuned wide-put batch (default 128 -> byte-identical off)
            has_mask=self.has_mask,  # mask composes (forward passes the (b,j,i)-transposed col-vec)
        )

    # ---- front: ONE fused staged invocation -> a_dm, b_dm (Dloc, B*N*N) einsum-native ---
    def front_a2a(self, xn_2d: torch.Tensor, direction: str = "outgoing", mask_col: torch.Tensor = None):
        """xn_2d (M, K) PRE-NORMALIZED token block -> a_dm, b_dm einsum-native (per direction).

        Runs the SINGLE staged fused-front kernel: it gates ``glu(x_norm@Wg^T, x_norm@Wp^T)``
        (no internal LN) and the D-major postact store reshards BOTH halves by feature into
        the symmetric recv (2*Dloc, M_full). ONE drain. The recv col is slot-major
        (slot*rows_per_peer + local_token).

        * NATIVE 1-D (cp1==1): slot indexes i-blocks, j full -> ``a = recv[:Dloc]``,
          ``b = recv[Dloc:]`` ALREADY linearize to the full ``(Dloc, B, N, N)`` grid, returned
          as ZERO-COPY VIEWS (the wave-1 path; ``front_unpack_dmajor_2d`` is a pure reshape).
        * NATIVE 2-D (cp1>1): the slot-major col order (i_block, j_block, b, i_local, j_local)
          does NOT linearize to (b, i, j) row-major -> ``front_unpack_dmajor_2d`` reassembles
          the einsum-native grid with ONE host permute + contiguous (design-doc §6 2-D
          host-unpack; host-side data movement, NO kernel change). NOT a 1-D->2-D reshard.

        ROUTE-2 (A) INCOMING: run the transpose_in N_i-stride-1 front (self._front_ni) and return the
        3-D recv halves ``(Dloc, N_i, N_j)`` directly (global-i is already stride-1 = einsum-native
        a_major="k"; NO front_unpack_dmajor reshape). OUTGOING keeps the plain front + reshape.

        COMPOSITE-K INCOMING (§9): run the transpose_in + D-MAJOR front (self._front_comp) and return the
        2-D padded per-rank-contiguous recv halves ``(Dloc, cp*N_j*Xg_pad)`` directly; the back
        ``_operands`` composite branch reshapes them to (Dloc, cp, N_j, Xg_pad) + the slot-0 K-hoist VIEW
        (NO host reshape here). OUTGOING keeps the plain front + reshape.
        """
        if self.route2_ni and direction == "incoming":
            self._barrier()
            # mask_col is the (b,j,i)-transposed col-vec (forward built it to match the transpose_in walk).
            recv = self._front_ni.run(xn_2d, mask_col=mask_col)  # (2*Dloc, N_i, N_j) N_i stride-1
            self._front_drain()
            dloc = self.D_loc
            return recv[:dloc], recv[dloc:]  # a,b halves (Dloc, N_i, N_j) einsum-native
        if self.composite_k and direction == "incoming":
            self._barrier()
            recv = self._front_comp.run(xn_2d, mask_col=mask_col)  # (2*Dloc, cp*N_j*Xg_pad) D-major
            self._front_drain()
            dloc = self.D_loc
            return recv[:dloc], recv[dloc:]  # a,b halves (Dloc, cp*N_j*Xg_pad); back _operands reshapes
        # F4(B): bind the plain front's recv now that we know it is the selected path (collective;
        # every rank takes this branch together). No-op unless a runtime rebind is pending.
        # (a) Pick the executor for THIS N and bind only IT — a declined shape never touches the padded
        # store and vice versa, which also preserves the F4 intent (only the selected store ever
        # allocates its per-N symmetric recv). rebind_M early-returns when M is unchanged, so calling it
        # on the selected store every time is both correct across regime switches and cheap; the old
        # single-shot `_front_pending_M` clear would have left the OTHER store unbound after a switch.
        front = self._select_front_store()
        pending_M = getattr(self, "_front_pending_M", None)
        if pending_M is not None:
            # pad_inner re-runs its per-N guard inside rebind_M, so it needs THIS N's token geometry
            # (they are inert for the unpadded plain front).
            front.rebind_M(pending_M, N_i_loc=self.N_i_loc, N_j_loc=self.N_j_loc)
            if front is self._front:
                self._front_pending_M = None
        self._barrier()
        recv = front.run(xn_2d, mask_col=mask_col)  # (2*Dloc, M_full) D-major; mask pre-store
        self._front_drain()
        a_half = recv[: self.D_loc]  # (Dloc, M_full)
        b_half = recv[self.D_loc :]  # (Dloc, M_full)
        # Reassemble to einsum-native (Dloc, B, N, N). 1-D: free view. Returned 4-D — NOT flattened to
        # (Dloc, B*N*N): under pad_inner the view is STRIDED (innermost pitch N_j_pad > N), so a
        # flatten would force an O(Dloc*N^2) I/O-order .contiguous(). The flatten was a no-op round-trip
        # anyway (`_operands` immediately re-folds it to (Dloc, B, N, N)), so dropping it is a local
        # signature change, not a contract change; `_operands` accepts both ranks.
        njp = getattr(front, "_pi_nj_pad", 0) or None
        a_dm = self.rl.front_unpack_dmajor_2d(a_half, N_j_pad=njp)
        b_dm = self.rl.front_unpack_dmajor_2d(b_half, N_j_pad=njp)
        return a_dm, b_dm

    # ---- back: ONE fused einsum-store -> (M, D) LayoutLeft value -----------------
    def einsum_back_a2a(self, a_dm: torch.Tensor, b_dm: torch.Tensor, direction: str):
        """a_dm,b_dm (Dloc, M) -> the design-E GEMM-native 5-D recv -> (M, D) LayoutLeft value.

        The fused einsum store writes tri's (i,j) tile straight into the peer 5-D recv
        (the back A2A). ONE drain, then ``back_unpack_gemm_native`` = the (M, D) stride
        (1, M) LayoutLeft value the consumer reads natively (NO transpose pass).
        """
        self._barrier()
        recv = self._back.run(a_dm, b_dm, direction)
        self._back_drain()  # §5.5 gate: host-signal (shared front drain) when engaged, else barrier
        return self.rl.back_unpack_gemm_native(recv)  # (M, D) LayoutLeft view

    # ---- back-half consumer: the xgate dual-x back (LN(tri) value, x_norm gate) --
    def _consume(self, tri_value: torch.Tensor, xn_2d: torch.Tensor) -> torch.Tensor:
        """P2.xgate back half: ``out = sigmoid(x_norm@g_out^T + bg) ⊙ (LN(tri)@p_out^T + bp)``.

        The dual-x (``x_gate``) ``dual_gated_gemm_{back_v}`` (xgate.sc by default): the
        VALUE projection runs ``LN(tri)@p_out^T`` (LN fused on ``tri``), the GATE projection
        consumes the SHARED ``x_norm`` RAW (no LN, no rank-1 correction) — cuEq's own
        gate3-in-the-back structure (design doc §3 P2). ``tri_value`` is the (M, D)
        LayoutLeft value (d-strided); the kernel auto-detects ``a_major="m"`` from stride
        (1, M) and reads it via its input TMA — NO torch transpose (the design-E point).
        ``xn_2d`` IS the shared ``x_norm`` = front LN(x). Wave 1: ``back_v = stagec`` ("sc").
        """
        D = self.D
        if self.consumer == "torch":
            # REFERENCE-ONLY path (not the default; not on the fused data path): pure-torch
            # back half via layernorm_fwd (the fold_cp_ops element kernel) + torch matmul/sigmoid.
            trin = layernorm_fwd(
                tri_value,
                self._norm_out_w.float(),
                self._norm_out_b.float() if self._norm_out_b is not None else None,
                self.eps,
            )
            p_out = trin @ self._Wp_out.T
            if self._bp_out is not None:
                p_out = p_out + self._bp_out
            g_out = xn_2d @ self._Wg_out.T
            if self._bg_out is not None:
                g_out = g_out + self._bg_out
            return p_out * torch.sigmoid(g_out)
        # xn_2d IS x_norm (M, D) row-major (layernorm_fwd output) -> pass directly as the raw
        # out-gate input; NO .contiguous() / reshape (already (M, D) contiguous; a different
        # tensor than tri_value so the consumer's x_gate-distinct-from-value guard holds).
        # TRANSLATED, not copied. The upstream picked between two module-level entries
        # (``dual_gated_gemm_stagec`` / ``dual_gated_gemm_staged``) which allocated the output and
        # ran the tile heuristic internally. Here both are ONE functor selected by a compile-time
        # ``fusion_variant``, and the entry takes the output tensor and the CTA tile POSITIONALLY --
        # so the allocation and the tile choice, which the upstream hid, are spelled out. The
        # arithmetic is unchanged; only who allocates moved.
        #
        # `consumer` keeps the upstream's two names so a caller's string still selects the same
        # schedule. STEP 1 measured `consumer='stagec'` on all 28 cell-directions of the acceptance
        # grid, so the other branch is reachable only by an explicit caller -- which is why it is
        # mapped rather than dropped.
        #
        # The mapping is `stagec -> alg_fold`, and it USED TO BE WRITTEN THE OTHER WAY ROUND -- see
        # `_consumer_fusion_variant`, which carries the correspondence and the reason it does not
        # read across from the upstream's stage letters. It is a named function rather than the
        # ternary that used to sit here because the ternary was inverted for the whole life of the
        # port and nothing could name it to test it.
        variant = _consumer_fusion_variant(self.consumer)
        M2, D2 = tri_value.shape[0], tri_value.shape[1]
        out2d = torch.empty(M2, D2, device=tri_value.device, dtype=tri_value.dtype)
        # tile_M is the front's fixed 128 (the only CTA-M either heuristic sweeps); tile_N tiles the
        # 2N-wide PRE-activation and is resolved from the OUTPUT width, not from the contraction.
        layernorm_dual_gated_gemm(
            tri_value,
            self._norm_out_w,
            self._Wg_out,
            self._Wp_out,
            out2d,
            _CONSUMER_TILE_M,
            _consumer_tile_n(D2),
            norm_bias=self._norm_out_b,
            bg=self._bg_out,
            bp=self._bp_out,
            x_gate=xn_2d,
            eps=self.eps,
            fusion_variant=variant,
        )
        return out2d

    # ---- dynamic-N: rebind the runtime token geometry (recvs + rl) before the chain ----
    def _rebind_runtime(self, N: int, B: int = None) -> None:
        """dynamic mode: set THIS forward's runtime ``(B, N)`` geometry (self.B/N/N_i_loc/N_j_loc/M
        + the per-shape ReshardLayout) and rebind the fused stores' symmetric recvs (front by M,
        back by N; cached per distinct shape, SPMD-lockstep). NO recompile — the anchor executors
        are mark_layout_dynamic and the back store reads its batch extent off the recv.

        ``B`` defaults to the constructor's. It is a parameter because the batch extent is a
        RUNTIME input like N: it enters only as a factor of `M = B*N_i_loc*N_j_loc` and of the back
        recv's mode 4, neither of which is baked. Deriving it from the constructor instead is what
        made `x_local.reshape(self.M, D)` a latent RuntimeError on a second batch."""
        if N % self.cp0 != 0 or N % self.cp1 != 0:
            raise ValueError(f"runtime N={N} must be divisible by both cp axes ({self.cp0},{self.cp1}).")
        if N % 8 != 0:
            raise ValueError(f"runtime N={N} must be a multiple of 8 (pe_aligned back 16-B TMA-S2G).")
        N_i_loc, N_j_loc = N // self.cp0, N // self.cp1
        if self.cp1 > 1 and (N_i_loc % 8 != 0 or N_j_loc % 8 != 0):
            raise ValueError(
                f"runtime 2-D per-axis %8: N_i_loc={N_i_loc}, N_j_loc={N_j_loc} (16-B TMA-S2G)."
            )
        self.N, self.N_i_loc, self.N_j_loc, self.N_loc = N, N_i_loc, N_j_loc, N_i_loc
        self.B = int(self.B if B is None else B)
        self.M = self.B * N_i_loc * N_j_loc  # local token count (front operand M extent)
        # Keyed (B, N): the ReshardLayout folds B into its token grid, so an N-only key hands a
        # second batch extent the first's layout -- silently, and with the right shape.
        rl_key = (self.B, N)
        rl = self._rl_cache.get(rl_key)
        if rl is None:
            rl = ReshardLayout(self.pe_map, self.B, N, self.D, feat_width=self.D)
            self._rl_cache[rl_key] = rl
        self.rl = rl
        # F4(B): DEFER the plain front's rebind. `front_a2a` short-circuits to `_front_comp`
        # (composite_k incoming) or `_front_ni` (route2_ni incoming) and never touches `self._front`
        # on those paths, but this used to allocate its per-N symmetric recv anyway -- 12.013 GiB per
        # incoming instance, 44% of the retained heap, never read. `TriangularMultiplication`
        # fixes `direction`
        # at construction, so on an incoming module it is provably unreachable; a raw `TriMulAutotuned`
        # can still be called either way, so the bind is deferred rather than removed.
        # Still collective-safe: the deferred `rebind_M` runs inside `front_a2a`, which every rank
        # reaches together (direction is an SPMD argument of `forward`).
        self._front_pending_M = self.M
        if self.route2_ni:
            # dynamic route2_ni: rebind the ONE dynamic front's 3-D symmetric recv for THIS N (N_i=
            # cp*Xg_pad, N_j=N) — NO recompile (a8ecd15 runtime-n_x + N_i/N_j-dynamic recv marking).
            self._front_ni.rebind_M(self.M, N_i_loc=N_i_loc, N_j_loc=N_j_loc, B=self.B)
        if self.composite_k:
            # dynamic composite_k: rebind the ONE dynamic front's 2-D PADDED recv for THIS N (recv col =
            # cp*N_j*Xg_pad) — NO recompile (transpose_in runtime-Yg + mode-1-dynamic 2-D recv marking).
            self._front_comp.rebind_M(self.M, N_i_loc=N_i_loc, N_j_loc=N_j_loc, B=self.B)
        self._back.rebind_N(N, B=self.B)

    # ---- full e2e ----------------------------------------------------------------
    def forward(
        self,
        x_local: torch.Tensor,
        direction: str = "outgoing",
        mask_local: torch.Tensor = None,
    ) -> torch.Tensor:
        """Full fused e2e on this rank's NATIVE token shard -> out (same shard shape).

        Local shard is ``(B, N_i_loc, N_j_loc, D)`` — NATIVE 1-D (cp1==1 -> N_j_loc==N, the
        row-slab) or NATIVE 2-D (both token axes split). NO reshard between the two: the cp-axis
        structure of the pe_map fixes the geometry at construction.

        ``mask_local`` (default None -> byte-identical to today's unmasked path): the local
        pair-mask shard ``(B, N_i_loc, N_j_loc)`` (== ``x_local``'s token block, no D dim),
        applied to the gated front projection ``ab`` before the a/b split + peer store — the
        ``_local_front`` semantics ``ab *= mask[..., None]`` (LOCAL, pre-A2A; no mask reshard).
        Wired into the front DualGatedGEMM's ``mMaskColVec`` epilogue in Phase 3.

        In ``dynamic`` mode the runtime token-N is INFERRED from ``x_local`` (N_i_loc = shape[1] ->
        N = N_i_loc*cp0) and the recvs + ReshardLayout are rebound per-N (ONE compile serves it).

        Chain (P2.xgate.sc, GLUE-FREE): ``x_norm = layernorm_fwd(x)`` (ONE LN, the fold_cp_ops
        element kernel — shared with the back gate) -> ONE staged fused front (a,b reshard
        on ``x_norm``) -> fused einsum+back reshard -> dual-x back-half (gated by the SAME
        ``x_norm``). ONE fused front peer store + ONE fused back store + 2 drains in one
        forward. The only host data movement off the kernel path is the 2-D front-recv
        reassembly (a no-op view in 1-D).
        """
        # PER-STAGE hang localization (CPO_TRIMUL_STAGE_LOG=1; default OFF -> byte-identical). A
        # timestamped enter/done per stage + a gated cuda.synchronize() so a wedge shows as
        # "STAGE X enter ... [silence]" and the conftest faulthandler stack pins the exact wedged call
        # to THAT stage (without the per-stage sync the async launches race ahead and the stall
        # misattributes to a later sync). Diagnostic-only; no effect when the env flag is unset.
        import os as _os_sl
        import time as _time_sl
        _slog = bool(int(_os_sl.environ.get("CPO_TRIMUL_STAGE_LOG", "0")))

        def _st(msg):
            if _slog:
                print(f"[trimul-stage t={_time_sl.time():.3f} r={_os_sl.environ.get('RANK', '?')}] {msg}",
                      flush=True)

        D = self.D
        # MASK: a non-None mask requires the store was built has_mask=True (the epilogue mask
        # branch is a COMPILE-TIME const_expr — it cannot be turned on per-call). A has_mask=True store
        # with mask_local=None would deref a null mask in-kernel -> require it. Both errors are surfaced
        # (never a silent drop / null deref).
        if mask_local is not None and not self.has_mask:
            raise ValueError(
                "TriMulAutotuned got a mask but was built has_mask=False; build with has_mask=True "
                "(the mMaskColVec front-store epilogue is baked at construction, not per-call)."
            )
        if mask_local is None and self.has_mask:
            raise ValueError(
                "TriMulAutotuned was built has_mask=True but forward() got mask_local=None "
                "(the baked front-store mask epilogue requires a mask every call)."
            )
        _st(f"forward ENTER dir={direction} x_local={tuple(x_local.shape)}")
        if self.dynamic:
            # runtime token-N = my i-axis local extent * cp0 (== N_j_loc*cp1). Rebind recvs + rl.
            # Both extents come from the tensor in hand. Reading N from `x_local` and B from the
            # constructor was the asymmetry that made a second batch a reshape error rather than a
            # second shape.
            self._rebind_runtime(int(x_local.shape[1]) * self.cp0, int(x_local.shape[0]))
        x_2d = x_local.reshape(self.M, D)  # (M, K=D) VIEW (x_local contiguous; the caller's input)
        # x_norm = layernorm_fwd(x): the SHARED activation, materialized ONCE (the fold_cp_ops
        # ELEMENT kernel, NOT F.layer_norm). The staged front consumes x_norm directly
        # (_normalize=False -> no internal LN), and the dual-x back out-gate reuses the SAME
        # x_norm -> ONE LN for the whole chain (was 3: 2 internal front LNs + 1 gate LN).
        _st("STAGE-1 layernorm_fwd (LN) enter")
        xn_2d = layernorm_fwd(x_2d, self._norm_in_w, self._norm_in_b, self.eps)
        if _slog:
            torch.cuda.synchronize()
        _st("STAGE-2 front_a2a (front DualGatedGEMM+LN-epi + front-A2A drain) enter")
        # MASK: flatten the (B, N_i_loc, N_j_loc) local pair mask to the (1, M) fp32 col-vec the front
        # store's mMaskColVec epilogue consumes (M = B*N_i_loc*N_j_loc). LOCAL / pre-A2A (no mask reshard)
        # — applied to the gated glu postact before the a/b split. the transpose_in fast-incoming
        # stores (route2_ni / composite_k) WALK the token-M axis in transposed (b, j_loc, i_loc) order,
        # and the mMaskColVec is loaded at that walk-order flat-M -> pre-TRANSPOSE the mask to (b, j, i)
        # for those paths so mask[flat-M] lines up with the native token the GEMM computes there. OUTGOING
        # + plain (route2_ni=composite_k=False) incoming read native (b, i, j). Runtime shape from
        # mask_local (dynamic-N safe). Host-side O(M) reorder, sub-leading along N_token (frugal).
        # MASK col-vec — ZERO-COPY (Part 2). NO host transpose / zero-pad / fp32-cast on any path:
        #  - fast-incoming (composite_k / route2_ni, transpose_in): pass the user's NATIVE (l=1, Xg, Yg)
        #    bf16 mask straight through; the TransposedMaskColVecLoad reads mask[b, i, tile_y] in-kernel
        #    with the transpose-walk index math + a clamp on the partial-last-x-tile pad rows (their value
        #    is discarded since pad output = 0 via acc=0). B==1 on this path -> mask_local IS (1, Xg, Yg).
        #  - plain (outgoing / non-fast incoming): reshape(1, M) is a zero-copy VIEW of the contiguous mask.
        # The in-kernel begin_loop casts the native bf16 mask -> fp32. Runtime Xg,Yg (mark_layout_dynamic).
        mask_col = None
        if self.has_mask:
            fast_incoming = direction == "incoming" and (self.route2_ni or self.composite_k)
            if fast_incoming:
                mask_col = mask_local
            else:
                # (a): the mask form follows the executor THIS N will select, so it is keyed off the
                # SAME regime helper front_a2a uses -- never off a store's cached state, which is still
                # the previous N's here (the plain front's rebind_M is deferred into front_a2a for F4).
                #  * padded executor -> the NATIVE (1, Xg, Yg) form its _begin_pad_inner branch unravels
                #    with the same (tile_x, tile_y) math as the A-load;
                #  * plain executor  -> the flat (1, M) col-vec, exactly as pre-pad_inner.
                # Both are zero-copy reshapes of the contiguous (B, N_i_loc, N_j_loc) mask.
                use_pad, yg, _rpp, _njp = self._front_regime()
                mask_col = (mask_local.reshape(1, self.M // yg, yg) if use_pad
                            else mask_local.reshape(1, self.M))
        a_dm, b_dm = self.front_a2a(xn_2d, direction, mask_col=mask_col)
        if _slog:
            torch.cuda.synchronize()
        _st("STAGE-3 einsum_back_a2a (back einsum GEMM + back-A2A cluster_multislot drain) enter")
        tri_value = self.einsum_back_a2a(a_dm, b_dm, direction)
        if _slog:
            torch.cuda.synchronize()
        _st("STAGE-4 _consume (out-gate layernorm_dual_gated_gemm) enter")
        out = self._consume(tri_value, xn_2d)
        if _slog:
            torch.cuda.synchronize()
        _st("forward DONE (all 4 stages complete)")
        # back recv unpack -> M = B*N_i_loc*N_j_loc token rows (b, i_local, j_local); reshape
        # to the NATIVE local shard (B, N_i_loc, N_j_loc, D). 1-D: N_j_loc==N (the row-slab).
        return out.reshape(self.B, self.N_i_loc, self.N_j_loc, D)

    def free(self):
        """Release every store's buffers AND drop the references, so no gc pass is needed.

        The second half is not tidiness. MEASURED: building and freeing six engines in one process
        WITHOUT an explicit ``gc.collect()`` leaves 5 stores alive and takes the caching allocator
        from 296 MiB to 1852 MiB; WITH one, 0 alive and 296 MiB flat. The engine object itself
        always reaches 0 -- it is the STORES that survive, because they sit in reference CYCLES
        that only the cyclic collector breaks. ``torch.cuda.empty_cache()`` does not help: it
        cannot free what is still referenced.

        The symmetric MemPool recycles a block correctly when its last reference drops, so this was
        never a pool defect -- it was Python holding the references. pytest does not force a
        collection between tests, so under a long module the stores accumulated until CPython's
        generational thresholds happened to trip, which is why the aborts were INTERMITTENT rather
        than reproducible.

        Clearing the attributes here makes the release refcount-driven, so a caller does not have
        to know to call ``gc.collect()`` -- and a caller who does not know is the case that failed.
        """
        # IDEMPOTENT: clearing the attributes below means a second `free()` would otherwise call
        # `None.free()`. Teardown must tolerate being run twice -- `TriangularMultiplication.free`
        # already calls this inside a try/except, and a caller who frees defensively should not be
        # punished for it.
        for attr in ("_front", "_front_pad", "_front_ni", "_front_comp", "_back", "_front_sig"):
            store = getattr(self, attr, None)
            if store is not None:
                store.free()
        # Break the cycles the stores participate in. Without this the buffers above are released
        # only on the next cyclic collection, which is whenever CPython feels like it.
        self._front = None
        self._front_pad = None
        self._front_ni = None
        self._front_comp = None
        self._back = None
        self._front_sig = None
        self._back_sig = None
        self._rl_cache.clear()
        self.rl = None


# =========================================================================== #
# The PUBLIC DTensor API. DTensor in, DTensor out.
#
# Brought back from the upstream's separate dispatch module and folded in here, because the split
# was never a layering statement: the upstream imported the engine LAZILY from the adapter to break
# an import cycle, and its raw-tensor half was a SECOND public API this tree deliberately does not
# carry. One module, one public entry -- `TriangularMultiplication` / `trimul_a2a` -- engine internal.
#
# NAMES. The upstream's `fused_trimul` family is retired; `_trimul_single_device` and `_get_engine`
# say what they do rather than what they were called.
#
# The nn.Module is `TriangularMultiplication`, named for the LAYER IT REPLACES rather than for the
# optimisation inside it. An earlier name here was `TriMulA2A` -- accurate (a TriMul whose
# all-to-all is fused into the two GEMM epilogues) but it described our implementation, so a caller's
# caller had to learn a new word to find the drop-in for a layer they already had. The class now
# mirrors `TriangleMultiplication{1,2}D`: same constructor shape `(layer, direction, device_mesh,
# ...)`, same six submodule attribute names. `trimul_a2a` keeps its name -- it is the FUNCTION, and
# its A2A-fused-ness is what a caller reaching past the module is asking for.
# =========================================================================== #

import inspect  # noqa: E402  (the API half's imports, kept beside the code they serve)
from typing import Optional  # noqa: E402

import torch.nn as nn  # noqa: E402

from fold_cp_ops.distributed.workflows.trimul_tuning import (  # noqa: E402
    TriMulTuning,
)

from fold_cp_ops.distributed.dtensor_adapter import (  # noqa: E402
    _dtensor_mesh_placements,
    _effective_placements,
    _ensure_nvshmem,
    _is_dtensor,
    _rowmaj_stride,
    _token_split_factor,
    validate_trimul_sharding,
)
from fold_cp_ops.distributed.trimul_weights import W_KEYS, W_PROJ_BIAS_KEYS  # noqa: E402
from fold_cp_ops.workflows.trimul_autotune import trimul_autotuned  # noqa: E402

__all__ = ["TriangularMultiplication", "trimul_a2a"]

# The CP weight dict and `trimul_autotuned`'s weight kwargs are the SAME key set, so the
# single-device fallback splats `w` straight through. Asserted at IMPORT so a rename on either side
# fails loudly here instead of silently dropping a weight at the first cp=1 call.
_W_ALL_KEYS = tuple(W_KEYS) + tuple(W_PROJ_BIAS_KEYS)
_TRIMUL_NON_W = ("x", "mask", "direction", "eps", "out", "select", "_config")
_TRIMUL_W_KWARGS = tuple(
    p for p in inspect.signature(trimul_autotuned).parameters if p not in _TRIMUL_NON_W
)
assert set(_W_ALL_KEYS) == set(_TRIMUL_W_KWARGS), (
    "trimul_weights.W_KEYS + W_PROJ_BIAS_KEYS must be EXACTLY trimul_autotuned's weight kwargs "
    f"(the single-device fallback splats w through). Only in w: "
    f"{sorted(set(_W_ALL_KEYS) - set(_TRIMUL_W_KWARGS))}; only in trimul_autotuned: "
    f"{sorted(set(_TRIMUL_W_KWARGS) - set(_W_ALL_KEYS))}"
)
#: LN gains/biases `trimul_autotuned` asserts are fp32 (the engine .float()s them itself).
_W_LN_KEYS = ("norm_in_w", "norm_in_b", "norm_out_w", "norm_out_b")


def _is_single_device(x) -> bool:
    """True when there is no cp to communicate over -> run the single-device kernel locally.

    Three cases (plan §3 "Trigger condition"), none of which needs a live manager or nvshmem:

    * ``x`` is a plain local ``Tensor`` (no mesh at all);
    * ``x`` is a DTensor on a mesh with a single device (``mesh.numel() == 1``);
    * ``x`` is a DTensor whose EFFECTIVE placements (size-1 Shard axes normalized to Replicate by
      ``_effective_placements``) split NEITHER token axis — i.e. ``cp == 1``.  This also covers a
      real dp-only mesh (``Shard(0)`` on the batch, ``dp > 1``): every rank holds the full square
      ``(N, N)`` token grid, so its own shard is a complete local TriMul (the same dispatch rule as
      ``dtensor_adapter``'s own removed DTensor entry used, whose ``token_split <= 1`` branch
      this replaces -- see that module's note on why it was removed by design rather than merely
      not yet ported).
    """
    if not _is_dtensor(x):
        return True
    mesh, placements = _dtensor_mesh_placements(x)
    if mesh is None or int(mesh.mesh.numel()) == 1:
        return True
    return _token_split_factor(mesh, _effective_placements(mesh, placements)) <= 1


def _rewrap_like(x, out_local):
    """Re-wrap a local result as a DTensor with ``x``'s OWN mesh, placements and global shape.

    Purpose
        The three entry points that produce a distributed result -- `_trimul_single_device`,
        `trimul_a2a` and `TriangularMultiplication.forward` -- must re-wrap identically. When each
        spelled the six lines itself, "identically" was a property nobody checked; this makes it one
        function, so a change to the wrap reaches all three or none.

    Functionality & semantics
        Returns ``out_local`` UNCHANGED when ``x`` is a plain tensor -- plain in, plain out, which is
        the cp=1 contract. Otherwise builds a ``DTensor`` from the local shard carrying ``x``'s mesh,
        ``x``'s placements and ``x``'s global shape, with a row-major stride. ``run_check=False``
        because the caller already validated the sharding and the extents are computed rather than
        inferred; the check would cost a collective per call to re-derive what is known.

    Args:
        x: The INPUT the result corresponds to. Must be the actual input -- its shape becomes the
            output's global shape, so the fused TriMul's shape-preserving property is assumed and a
            shape-CHANGING op must not use this helper.
        out_local: This rank's local result shard, row-major and already at the right local extent.

    Returns:
        ``out_local`` if ``x`` is not a DTensor, else a same-sharding ``DTensor``.
    """
    if not _is_dtensor(x):
        return out_local
    from torch.distributed.tensor import DTensor

    mesh, placements = _dtensor_mesh_placements(x)
    out_shape = tuple(x.shape)
    return DTensor.from_local(
        out_local, mesh, list(placements),
        shape=out_shape, stride=_rowmaj_stride(out_shape), run_check=False,
    )


def _trimul_single_device(x, w: dict, *, direction: str, mask, eps: float = 1e-5):
    """Run the single-device ``trimul_autotuned`` on ``x``; re-wrap iff the input was a DTensor.

    ``w`` splats straight through (the key sets are asserted equal at import).  The four LN
    gains/biases are ``.float()``-ed to satisfy ``trimul_autotuned``'s fp32-LN assert — mirroring
    ``TriMulAutotuned.__init__``, which does the same cast, so ONE ``w`` (e.g. from
    ``weights_from_trimul_module(dtype=bf16)``) works on BOTH paths.  Nothing else is touched, so
    with an fp32-gain ``w`` this is BITWISE the plain ``trimul_autotuned`` call.
    """
    unexpected = [k for k in w if k not in _W_ALL_KEYS]
    if unexpected:
        raise KeyError(
            f"cp=1 TriMul fallback got unexpected weight key(s) {sorted(unexpected)}; "
            f"expected exactly {list(_W_ALL_KEYS)}"
        )
    wk = {k: w.get(k) for k in _W_ALL_KEYS}
    for k in _W_LN_KEYS:
        if wk[k] is not None:
            wk[k] = wk[k].float()
    x_local = x.to_local() if _is_dtensor(x) else x
    mask_local = mask.to_local() if _is_dtensor(mask) else mask
    # A3: the CP mask is (B,N,N) and `trimul_autotuned` reshapes it to (M,) itself — pass through.
    # No B gate — `trimul_autotuned` is B-agnostic (its B==1-only glg is heuristic-gated).
    out_local = trimul_autotuned(
        x_local, **wk, mask=mask_local, direction=direction, eps=eps, select="heuristic"
    )
    return _rewrap_like(x, out_local)


# --------------------------------------------------------------------------- #
# Functional dispatch: DTensor -> TriMulAutotuned -> DTensor, with a cache.
# --------------------------------------------------------------------------- #


def _build_pe_map(x, distributed_manager):
    """Unwrap ``x``'s (mesh, placements), validate the TriMul sharding, build the cp PeMap.

    Returns ``(pe_map, mesh, placements)``. ``placements`` is the ORIGINAL DTensor placements
    (used to re-wrap the output); the PeMap is built from the EFFECTIVE placements (size-1
    Shard axes normalized to Replicate — see ``_effective_placements``).
    """
    mesh, placements = _dtensor_mesh_placements(x)
    validate_trimul_sharding(placements, mesh.ndim)
    pe_placements = _effective_placements(mesh, placements)
    pe_map = PeMap.from_mesh_placements(mesh, pe_placements, distributed_manager=distributed_manager)
    return pe_map, mesh, placements


def _get_engine(cache, pe_map, mesh, placements, B, N, D, w, dt, *, dynamic, has_mask, kw):
    """Return a cached ``TriMulAutotuned``; build (compile-once / alloc-once) on a miss.

    Key = ``(B, None-if-dynamic-else-N, D, cp, cp_axis_sizes, has_mask, id(pe_map))`` — a distinct
    compiled instance per shape/sharding/mask-presence (the mask is a COMPILE-TIME variant of the
    front store, so a masked and an unmasked call build separate instances). In ``dynamic`` mode ONE
    instance serves all N (the key's N slot is ``None``).
    """
    from fold_cp_ops.distributed.workflows.trimul_autotuned import TriMulAutotuned

    # composite_k / route2_ni build a distinct front store -> they key the instance cache too.
    key = (B, None if dynamic else int(N), int(D), pe_map.cp,
           tuple(int(s) for s in pe_map.cp_axis_sizes), bool(has_mask),
           bool(kw.get("composite_k", False)), bool(kw.get("route2_ni", False)))
    inst = cache.get(key)
    if inst is None:
        _ensure_nvshmem(kw.get("distributed_manager"))
        ctor_kw = {k: v for k, v in kw.items() if k != "distributed_manager"}
        inst = TriMulAutotuned(
            pe_map, B, int(N), int(D), w, dt,
            device_mesh=mesh, placements=placements, dynamic=dynamic, has_mask=has_mask, **ctor_kw,
        )
        cache[key] = inst
    return inst


#: The front-store variants the incoming path can run. ``"plain"`` is the D-major store the
#: outgoing path always uses; the other two differ from it ONLY in the layout of the front receive
#: buffer, which is why the choice is a function of the sharding and not of a measurement.
INCOMING_STORE_VARIANTS = ("plain", "composite_k", "route2_ni")


def incoming_store_variant(direction: str, cp1: int, batch: int = 1) -> str:
    """Which front-store variant a configuration runs: ``plain`` | ``composite_k`` | ``route2_ni``.

    Purpose
        State the rule ONCE, as a total function over its inputs, instead of leaving it as a
        condition spread across a call chain. Before this it lived inline in `trimul_a2a` and could
        only be exercised by building an engine; the whole domain is eight cases and is now
        enumerable without a GPU.

    Functionality & semantics
        The DISCRIMINATOR -- which of the two fast variants -- is ``cp1`` ALONE: ``composite_k``
        for a 1-D token shard, ``route2_ni`` for 2-D. The batch extent does not choose between
        them and never has.

        ``batch`` is NOT a gate any more and is retained only so an existing caller keeps working:
        both fast recv layouts now carry an explicit batch plane (``route2_ni`` allocates
        ``(2*Dloc, B, N_j, N_i)``; ``composite_k`` allocates ``(2*Dloc, B, cp*rpp_b)``, PLANE-major
        then slot), and both fronts are built ``b_dynamic`` so ONE compile serves every extent. The
        parameter is VALIDATED (``>= 1``) and otherwise ignored. Deleting it outright would silently
        change the meaning of every positional third argument already written, so it stays until the
        callers are swept.

        NOT inputs, and this is the point of the function: the token count, the mask, the feature
        width, the dtype, the transport, or any caller keyword. Only ``back_store`` interacts, and
        that is a VALIDITY constraint on the chosen variant rather than an input to choosing it --
        folding it in here would return ``"plain"`` for a configuration the caller cannot serve,
        which is the silent-downgrade failure this repo has already paid for once.

    Args:
        direction: ``"outgoing"`` or ``"incoming"``. Anything else raises rather than falling
            through to ``"plain"``, because a typo would otherwise cost ~2x on the incoming path
            with no error.
        cp1: Extent of the SECOND cp axis; ``1`` for a 1-D token shard. Must be >= 1.
        batch: Runtime batch extent; must be >= 1. Does NOT affect the result -- see above.

    Returns:
        One of :data:`INCOMING_STORE_VARIANTS`.

    Raises:
        ValueError: on an unrecognised ``direction``, ``cp1 < 1``, or ``batch < 1``.
    """
    if direction not in ("outgoing", "incoming"):
        raise ValueError(
            f"direction must be 'outgoing' or 'incoming'; got {direction!r}. Falling through to "
            f"the plain store on a typo would silently cost ~2x on the incoming path."
        )
    if int(cp1) < 1:
        raise ValueError(f"cp1 must be >= 1; got {cp1!r}")
    if int(batch) < 1:
        raise ValueError(f"batch must be >= 1; got {batch!r}")
    if direction == "outgoing":
        return "plain"
    return "composite_k" if int(cp1) == 1 else "route2_ni"


def _reject_unknown_engine_kwargs(fused_kwargs: dict) -> None:
    """Refuse a ``fused_kwargs`` key that is not a `TriMulAutotuned` parameter, AT THIS CALL.

    Purpose
        `trimul_a2a` forwards its ``**fused_kwargs`` to the engine constructor, which is built
        lazily on a cache MISS. So a misspelled knob does not fail at the call the caller wrote --
        it fails inside `_get_engine`, after the PeMap is built and nvshmem is up, with a
        ``TypeError`` from a constructor they never named. On a cluster that is a wasted allocation
        and a traceback pointing at the wrong frame.

    Semantics
        Compares against `TriMulAutotuned.__init__`'s live keyword-only signature, so a knob
        renamed on the engine cannot leave a stale allow-list behind. Suggests the closest match,
        because the failure this exists to catch is a typo and the useful part of the message is
        which name was meant.

        Names `trimul_a2a` binds itself (``direction``, ``dynamic``, ``mask``, ...) never reach
        here: Python raises ``got multiple values for keyword argument`` at call time, and that
        message already names both the function and the parameter.

    Args:
        fused_kwargs: The caller's bag, possibly empty.

    Raises:
        TypeError: naming the unknown key, its closest legal neighbour, and the full legal set.
    """
    import difflib
    import inspect

    legal = {
        n
        for n, prm in inspect.signature(TriMulAutotuned.__init__).parameters.items()
        if prm.kind is inspect.Parameter.KEYWORD_ONLY
    }
    unknown = sorted(set(fused_kwargs) - legal)
    if not unknown:
        return
    hints = []
    for k in unknown:
        near = difflib.get_close_matches(k, sorted(legal), n=1)
        hints.append(f"{k!r}" + (f" (did you mean {near[0]!r}?)" if near else ""))
    raise TypeError(
        f"trimul_a2a got unknown fused_kwargs: {', '.join(hints)}. Legal engine knobs are "
        f"{sorted(legal)}. Refused here rather than inside the lazy engine build, which happens "
        f"after the PeMap and nvshmem are up and reports a TypeError from a constructor you did "
        f"not call."
    )


def trimul_a2a(
    x,  # DTensor (B, N, N, D)
    w: dict,
    dt: torch.dtype,
    *,
    direction: str = "outgoing",
    mask=None,  # DTensor (B, N, N) or local Tensor or None
    dynamic: bool = True,
    distributed_manager=None,
    _cache: Optional[dict] = None,
    **fused_kwargs,
):
    """Run the fully-fused distributed TriMul on a DTensor; return a same-sharding DTensor.

    Unwrap ``x`` -> validate the §2 TriMul token-sharding (1-D AND 2-D) -> build/reuse a
    ``TriMulAutotuned`` for this (B, N, D, cp, sharding) -> ``forward(x_local, direction,
    mask_local)`` -> re-wrap with the SAME mesh + placements. Inference only.

    ``w`` is the replicated weight dict (``fold_cp_ops.distributed.trimul_weights.
    weights_from_trimul_module`` builds it from a reference layer). ``mask`` is the pair mask,
    a DTensor with the SAME token sharding as ``x`` (or a local tensor / None); applied
    LOCALLY pre-A2A (Phase 3). ``_cache`` lets a caller (the nn.Module) own the instance
    cache across calls; ``None`` uses a process-global cache. Extra ``fused_kwargs`` (e.g.
    ``hybrid_ib``, ``consumer``, ``back_store``, ``route2_ni``, ``composite_k``) pass through
    to ``TriMulAutotuned.__init__``.

    **cp == 1** (a plain local tensor, a 1-device mesh, or placements that split no token axis)
    short-circuits to ``fold_cp_ops.workflows.trimul_autotune.trimul_autotuned`` BEFORE ``_build_pe_map`` — no
    ``PeMap``, no manager, no nvshmem.  Only ``eps`` is honoured from ``fused_kwargs`` there (the
    rest configure the fused A2A stores, which that path does not build).  A plain-tensor input
    returns a plain tensor; a DTensor input returns a same-sharding DTensor.
    """
    assert direction in ("outgoing", "incoming"), "direction must be 'outgoing'|'incoming'"
    _reject_unknown_engine_kwargs(fused_kwargs)
    if not _is_dtensor(x) and not isinstance(x, torch.Tensor):
        raise TypeError(f"trimul_a2a expects a DTensor or local Tensor x, got {type(x)}.")
    # A1: cp == 1 -> the single-device kernel.  MUST precede `_build_pe_map` (which builds a PeMap
    # and needs a live manager + nvshmem init).
    if _is_single_device(x):
        return _trimul_single_device(x, w, direction=direction, mask=mask, eps=fused_kwargs.get("eps", 1e-5))

    cache = _FUSED_CACHE if _cache is None else _cache
    pe_map, mesh, placements = _build_pe_map(x, distributed_manager)
    B, N, _, D = x.shape
    kw = dict(fused_kwargs)
    kw["distributed_manager"] = distributed_manager
    # Mask presence is a COMPILE-TIME front-store variant -> it keys the instance cache.
    has_mask = mask is not None
    # default the FAST INCOMING store variant so the shipped API's incoming matches the
    # raw-tensor perf path (composite_k 1-D / route2_ni 2-D collapse incoming to ~= outgoing speed;
    # ~2x vs the plain incoming). Both require B==1; OUTGOING ignores them (byte-identical), so only set
    # for incoming. An explicit caller kwarg (composite_k/route2_ni) always wins. A later change lifted the has_mask
    # guard, so masked incoming now rides the fast path too (forward transposes the mask col-vec).
    cp_axis = tuple(int(s) for s in pe_map.cp_axis_sizes)
    cp1 = cp_axis[1] if len(cp_axis) > 1 else 1
    if "composite_k" not in kw and "route2_ni" not in kw:
        variant = incoming_store_variant(direction, cp1, int(B))
        if variant != "plain":
            kw[variant] = True
    ft = _get_engine(cache, pe_map, mesh, placements, B, N, D, w, dt,
                           dynamic=dynamic, has_mask=has_mask, kw=kw)

    x_local = x.to_local()
    mask_local = mask.to_local() if _is_dtensor(mask) else mask
    out_local = ft.forward(x_local, direction, mask_local=mask_local)
    return _rewrap_like(x, out_local)


# Process-global instance cache (used when a caller does not pass its own _cache).
_FUSED_CACHE: dict = {}


# --------------------------------------------------------------------------- #
# The nn.Module wrapper (Phase 4).
# --------------------------------------------------------------------------- #


class TriangularMultiplication(nn.Module):
    """DTensor-in / DTensor-out inference drop-in for reference CP TriMul (1-D and 2-D meshes).

    Purpose
        The shipped nn.Module front door for the A2A-fused TriMul. Its constructor mirrors the CP reference wrappers'
        ``TriangleMultiplication{1,2}D`` -- ``(layer, direction, device_mesh, ...)`` -- and it
        exposes the SAME six submodule names, so it substitutes for the DTensor layer it replaces
        without the caller reshaping anything it already holds.

    Functionality & semantics
        Reads the frozen weights off ``layer`` ONCE at construction (via
        `trimul_weights.weights_from_trimul_module`, which duck-types on the six submodule names and
        imports nothing from them), and owns a `TriMulAutotuned` instance cache -- compiled kernels
        plus symmetric buffers, lifetime == module lifetime. ``forward`` unwraps the DTensor, lazily
        builds or reuses the fused instance for the input's ``(B, N, D, cp, sharding)``, runs the
        fully-fused end-to-end, and re-wraps with the same mesh and placements.

        **Always dynamic.** One compiled instance per direction serves every token-N. There is no
        ``dynamic`` knob and no static mode: a per-N build multiplies the nvshmem CUDA-library
        registrations by the number of distinct N a process sees, and every extra registration is
        another chance for the DSL's collector and nvshmem's ``library_finalize`` to race over the
        same handle (see `gemm_bitcode_compile.CompiledGemmBitcode.free`). Production, the perf
        gates and the published archive all ran dynamic; a knob here would only expose the setting
        nothing is tuned for.

        **``device_mesh`` is recorded and CHECKED, never dispatched on.** ``forward`` takes the mesh
        from the input DTensor exactly as it always has; the constructor's mesh is compared against
        it and a disagreement raises. Making the constructor's mesh authoritative would change how a
        plain-tensor (cp == 1) input routes, which is a semantics change this class does not make.

    Args:
        layer: Any ``nn.Module`` exposing ``norm_in``, ``p_in``, ``g_in``, ``norm_out``, ``p_out``,
            ``g_out``. It does NOT have to be a ``TriangularMultiplication`` -- the weight read
            is by attribute name, which is what keeps the serial code base off this module's
            dependency list. A caller holding only a weight dict builds one with
            `trimul_weights.trimul_module_from_weights`. A missing submodule raises
            ``AttributeError`` naming it. **All SIX biases are fused and carried** -- the two
            LayerNorm biases and all four projection biases (`p_in_b`, `g_in_b`, `p_out_b`,
            `g_out_b`). This docstring used to promise a ``NotImplementedError`` on a projection
            bias; that refusal was removed when the front store's epilogue slot was wired up, and
            the text outlived it.
        direction: ``"outgoing"`` or ``"incoming"``. Anything else raises -- it selects which
            einsum the back half runs, so a typo would otherwise compute the wrong contraction.
        device_mesh: The ``DeviceMesh`` this layer is sharded over. Must equal the mesh of the
            DTensor later passed to ``forward``; see the note above for why it is checked rather
            than used.
        distributed_manager: Live ``DistributedManager``, forwarded to ``PeMap.from_mesh_placements``
            and to nvshmem init. Required for any ``cp > 1`` input; a ``cp == 1`` module never
            touches an nvshmem symbol and tolerates ``None``.
        dtype: Compute dtype for the fused kernels, default ``torch.bfloat16``. NOT derived from
            ``layer`` on purpose -- an fp32 checkpoint would then silently change what the fused
            path computes.
        eps: LayerNorm epsilon. ``None`` (default) takes it from ``layer.norm_in.eps``, falling back
            to ``1e-5``. It is a named argument because it was previously reachable only by knowing
            it happened to be a `TriMulAutotuned` kwarg and passing it through the old
            ``**fused_kwargs``: the fused path therefore used ``1e-5`` for every caller, including
            a checkpoint whose LayerNorm says otherwise. That is a silently wrong VALUE rather than
            a refusal, so it gets a front-door name.
        tuning: Optional `trimul_tuning.TriMulTuning`, the typed per-kernel escape hatch. ``None``
            means the resolved defaults. A default-constructed ``TriMulTuning()`` is equivalent to
            ``None`` -- it flattens to ``{}`` -- so an untuned module calls the engine with exactly
            the arguments it did before this parameter existed.

            **``**fused_kwargs`` is gone, deliberately.** It admitted all twenty of the engine's
            keyword-only parameters unvalidated, including seven names ``trimul_a2a`` binds itself
            (so passing one raised ``got multiple values for keyword argument`` at the FIRST
            FORWARD, naming a function the caller never called), and a typo survived construction
            to fail on hardware. The knobs it exposed that are genuinely tunable live in ``tuning``;
            the rest are derived, and `trimul_tuning.NOT_EXPOSED` records each one with its reason.
            ``hybrid_ib`` is among them: it is auto-detected from the P2P topology, and forcing it
            ``False`` on a job with IB peers is a CUDA illegal address rather than a slower run.
            The functional form `trimul_a2a` still takes ``**fused_kwargs`` for benchmark callers
            that need to override a derived choice.

    Raises:
        ValueError: If ``direction`` is not ``"outgoing"``/``"incoming"``.
        AttributeError: From the weight read when ``layer`` is missing a submodule, per ``layer``.

    Notes
    -----
    Frozen-weights inference only: ``forward`` is ``@torch.no_grad``. If the checkpoint is swapped,
    drop the module -- its cached instances fold the old weights.
    """

    def __init__(
        self,
        layer: nn.Module,
        direction: str,
        device_mesh,
        distributed_manager,
        *,
        placements=None,
        dtype: torch.dtype = torch.bfloat16,
        eps: float | None = None,
        tuning: TriMulTuning | None = None,
    ):
        """Read the weights, resolve the configuration, and build the cp peer map. No compile.

        Every argument is documented on the CLASS, which is where a reader looks; what belongs
        here is the ORDER, because it is the point of the refactor. In sequence: validate
        `direction`; read + fp32-coerce the weights; resolve `eps`; record D from `norm_in`'s gain;
        resolve the placements (`_resolve_placements`); resolve cp, the per-axis cp sizes and the
        `PeMap` (`_resolve_cp`); flatten `tuning` to engine kwargs. Steps 4-7 of that list used to
        run inside `trimul_a2a` on EVERY forward, from inputs this constructor already had.

        Nothing here compiles a kernel or allocates a symmetric buffer -- `prepare` does, and
        `forward` calls it. So this stays cheap, and a module can be built before nvshmem is up.
        """
        super().__init__()
        if direction not in ("outgoing", "incoming"):
            raise ValueError(
                f"direction must be 'outgoing' or 'incoming'; got {direction!r}. It selects the "
                "back-half einsum, so an unrecognised value would compute a different contraction."
            )
        from fold_cp_ops.distributed.trimul_weights import weights_from_trimul_module

        # `dtype` is the COMPUTE dtype, NOT a cast applied to the weight dict. The engine casts the
        # projections itself (`w["p_in_w"].to(dt)`), and it reads the LN gains/biases at fp32 --
        # `tensor_contract.check_tensor` REFUSES anything else, by design, because a narrower gain
        # is a silent loss rather than a conversion. Casting `w` here therefore breaks the LN keys.
        # Measured: passing `dtype=` through raised
        #   ValueError: norm_weight must be torch.float32; got torch.bfloat16
        # on every cross-node cell. So the dict keeps the layer's own dtypes and only the LN keys
        # are coerced -- upward, to fp32, which cannot lose anything and lets a bf16 checkpoint work.
        w = weights_from_trimul_module(layer)
        for k in _W_LN_KEYS:
            if w[k] is not None and w[k].dtype is not torch.float32:
                w[k] = w[k].float()
        self._w = w
        self._dt = dtype
        self._dir = direction
        self._dm = distributed_manager
        # `eps` from the argument, else from the LAYER's own LayerNorm, else 1e-5. It was
        # previously reachable ONLY by knowing it happened to be a `TriMulAutotuned` kwarg and
        # smuggling it through `**fused_kwargs`, so the fused path used 1e-5 for every caller --
        # including a checkpoint whose LayerNorm says otherwise. That is a silent wrong VALUE, not
        # a refusal, which is why it is a named argument now.
        if eps is None:
            eps = float(getattr(getattr(layer, "norm_in", None), "eps", 1e-5))
        self._eps = float(eps)
        # The tuning tree flattens to the engine's own flat keyword names. A default `TriMulTuning`
        # emits `{}`, so an untuned module calls the engine with exactly the arguments it received
        # before this change -- that is a property of the type, asserted in
        # `tests/distributed/workflows/test_trimul_tuning.py`, not something re-measured here.
        self._tuning = tuning
        self._kw = tuning.to_engine_kwargs() if tuning is not None else {}
        self.device_mesh = device_mesh
        self.direction = direction
        # The feature width is a property of the WEIGHTS, so it is known here and never read from
        # the data. `norm_in` normalizes the feature axis, so its gain length IS D. `forward` then
        # CHECKS the input against it (`§4.4`): a tensor of another width would otherwise fail
        # several frames into a compiled kernel, naming a tile instead of the mismatch.
        self._D = int(w["norm_in_w"].shape[0])
        # ---- §4.2 steps 4-7: the configuration that used to run on every forward. -------------
        self._placements = self._resolve_placements(device_mesh, placements)
        self._cp, self._cp_axis_sizes, self._pe_map = self._resolve_cp(device_mesh, self._placements)
        # The variant RULE is fixed here (`direction` and `cp1` are module-lifetime constants); the
        # one input it still needs -- the batch extent -- arrives with the shape, so the rule is
        # APPLIED in `prepare`. See `incoming_store_variant` for why `batch` is a parameter at all.
        self._cp1 = int(self._cp_axis_sizes[1]) if len(self._cp_axis_sizes) > 1 else 1
        # (batch, masked) -> engine. Every other component of the old eight-tuple key is now a
        # module-lifetime constant, and `dynamic=True` erases N -- see `§4.3`.
        self._engines: dict[tuple[int, bool], "TriMulAutotuned"] = {}
        self._engine_variant: dict[tuple[int, bool], str] = {}
        self._warned_second_variant = False
        # The six canonical attribute names, bound to the SOURCE layer's own submodules. Assigning an
        # nn.Module registers it as a child, so `named_children()` reads like the layer this
        # replaces -- which is the whole point of matching the names.
        self.norm_in = layer.norm_in
        self.p_in = layer.p_in
        self.g_in = layer.g_in
        self.norm_out = layer.norm_out
        self.p_out = layer.p_out
        self.g_out = layer.g_out

    @staticmethod
    def _resolve_placements(device_mesh, placements):
        """Return the DTensor placements this module is built for, defaulting by CONVENTION.

        Purpose
            `__init__` must know the placements to derive cp and build the PeMap, but a constructor
            takes a mesh, not a tensor. This resolves the one from the other -- and makes the
            assumption it is making visible, because it IS an assumption.

        Functionality & semantics
            ``None`` -> ``[Shard(i + 1) for i in range(device_mesh.ndim)]``, the convention every
            in-tree caller already writes by hand (`test_trimul_autotuned.py:384`,
            `harness/targets/trimul_e2e.py:363`). It is NOT derivable: `validate_trimul_sharding`
            is placement-GENERIC -- it permits ``Shard(0)`` on the batch axis and requires only that
            at least one token dim is sharded -- so a caller using ``(Shard(0), Shard(1))`` has a
            legal sharding this default does not describe. That is exactly why `forward` compares
            the input's placements against this value and RAISES: the alternative is building a
            second engine for a sharding the first was not compiled for, silently.

            A mesh of 3+ dims gets no default. The convention would produce ``Shard(3)`` on the
            feature axis, which `validate_trimul_sharding` refuses with a message about LayerNorm
            locality -- true, but it names the wrong cause for a caller who passed no placements at
            all. Refusing here names the argument to pass.

        Args:
            device_mesh: The mesh, or ``None`` (a cp=1 / plain-tensor module -- returns ``None``).
            placements: An explicit per-mesh-dim placement sequence, or ``None`` for the default.
                Length must equal ``device_mesh.ndim``; a mismatch raises rather than being padded.

        Returns:
            ``list`` of placements, or ``None`` when ``device_mesh`` is ``None``.

        Raises:
            ValueError: on a length mismatch, or on a 3+-dim mesh with no explicit ``placements``.
        """
        if device_mesh is None:
            return None if placements is None else list(placements)
        if not hasattr(device_mesh, "ndim") or not hasattr(device_mesh, "mesh"):
            raise TypeError(
                f"device_mesh must be a DeviceMesh (it needs .ndim and .mesh) or None; got "
                f"{type(device_mesh).__name__}. The constructor now derives cp, the placements and "
                f"the PeMap from it, so an object that only compares equal is no longer enough."
            )
        ndim = int(device_mesh.ndim)
        if placements is not None:
            pl = list(placements)
            if len(pl) != ndim:
                raise ValueError(
                    f"placements has {len(pl)} entries but device_mesh.ndim is {ndim}. One "
                    f"placement per mesh dim is required; there is no padding rule."
                )
            return pl
        if ndim > 2:
            raise ValueError(
                f"TriangularMultiplication cannot default the placements for a {ndim}-D mesh: the "
                f"convention [Shard(1), Shard(2), ...] would shard the FEATURE axis. Pass "
                f"placements= explicitly (e.g. [Replicate(), Shard(1), Shard(2)] for a dp x cp x cp "
                f"mesh)."
            )
        from torch.distributed.tensor import Shard

        return [Shard(i + 1) for i in range(ndim)]

    def _resolve_cp(self, device_mesh, placements):
        """Return ``(cp, cp_axis_sizes, pe_map)`` for this module's mesh + placements.

        Purpose
            Steps 4-6 of `§4.2`: the cp extent, the per-axis cp sizes and the `PeMap` are all
            functions of ``(device_mesh, placements)``, both fixed at construction, so they are
            computed once here instead of on every `forward`.

        Functionality & semantics
            Mirrors `_build_pe_map` exactly -- validate the TriMul sharding, normalize size-1 Shard
            axes to Replicate (`_effective_placements`), then `PeMap.from_mesh_placements` -- with
            one difference: the placements come from the constructor rather than from a tensor.

            **This touches no nvshmem symbol.** `PeMap.from_mesh_placements` is self-contained (it
            reads the mesh's rank tensor and, for defaults only, the manager's rank/device), so
            hoisting it into `__init__` does NOT make `init_nvshmem()` a construction-time
            precondition. nvshmem is still first required at `prepare`/`forward`, where the engine
            is actually built -- the same ordering as before this refactor.

            A cp of 1 (no mesh, a 1-device mesh, or placements splitting no token axis) returns
            ``(1, (), None)`` and builds nothing: that module routes to the single-device kernel and
            never needs a manager.

        Args:
            device_mesh: The mesh, or ``None``.
            placements: The resolved placements from `_resolve_placements`, or ``None``.

        Returns:
            ``(cp, cp_axis_sizes, pe_map)``; ``pe_map`` is ``None`` iff ``cp == 1``.

        Raises:
            ValueError: from `validate_trimul_sharding` (feature axis sharded, no token axis
                sharded) or from `PeMap.from_mesh_placements`.
        """
        if device_mesh is None or placements is None or int(device_mesh.mesh.numel()) == 1:
            return 1, (), None
        eff = _effective_placements(device_mesh, placements)
        if _token_split_factor(device_mesh, eff) <= 1:
            return 1, (), None
        validate_trimul_sharding(placements, device_mesh.ndim)
        pe_map = PeMap.from_mesh_placements(
            device_mesh, eff, distributed_manager=self._dm
        )
        return int(pe_map.cp), tuple(int(s) for s in pe_map.cp_axis_sizes), pe_map

    @property
    def incoming_variant(self) -> str:
        """Which front-store variant the engines this module has BUILT are running.

        Purpose
            A witness, not a prediction. `benchmark/distributed/harness/targets/trimul_e2e.py`
            records the incoming variant for every benchmark cell; it used to obtain it by decoding
            `self._cache`'s key by negative index, inside a bare ``except Exception``. Collapsing
            that key (`§4.3`) would have made the variant fields silently DISAPPEAR from every
            cell's metadata rather than raise. Reading a decided value beats decoding a key.

        Functionality & semantics
            Three non-variant answers, each a distinct FACT rather than an absence:

            * ``"cp1_fallback"`` -- this module routes to the single-device kernel and has no front
              store at all. Reporting ``"unbuilt"`` there would invite the reader to conclude a
              build was pending when none was ever going to happen.
            * ``"unbuilt"`` -- cp > 1 but nothing has run yet.
            * ``"mixed"`` -- engines of more than one variant. Nothing in the shipped selection
              rule can produce this any more: `incoming_store_variant` is a function of
              ``direction`` and ``cp1`` alone, both module-lifetime constants, and the cross-node
              batch demotion that used to split a module's engines is gone. It is still reported
              rather than asserted away, because it is the honest answer if an explicit per-call
              tuning override ever reintroduces a split.

        Returns:
            One of :data:`INCOMING_STORE_VARIANTS`, or ``"cp1_fallback"`` / ``"unbuilt"`` /
            ``"mixed"``.
        """
        if self._cp == 1:
            return "cp1_fallback"
        vs = set(self._engine_variant.values())
        if not vs:
            return "unbuilt"
        return vs.pop() if len(vs) == 1 else "mixed"

    def prepare(self, *, n_token: int, batch: int = 1, masked: bool = False):
        """Compile the four fused kernels and allocate their symmetric buffers for one shape.

        Purpose
            Move a multi-second, allocating cost to a point the caller chose. The IB front compile
            is ~15.5 s measured, and without this it lands mid-run on the first `forward`.

        Functionality & semantics
            **OPTIONAL. It changes NOTHING about what `forward` accepts or computes.** `forward`
            calls it itself for whatever shape arrives, so a module that never calls it behaves
            exactly as it did before this method existed -- the compile simply happens later.

            Idempotent per ``(batch, masked)``: a second call with the same pair returns the built
            engine and does not recompile, **whatever ``n_token`` says**, because the engine is
            dynamic in N. Those two are the only genuine compile variants left; every other
            component of the old instance-cache key -- direction, dtype, mesh, cp, cp_axis_sizes,
            D -- is a module-lifetime constant, and ``dynamic=True`` erases N (`§4.3`).

            A ``cp == 1`` module builds nothing and returns ``None``: that path runs the
            single-device kernel and never touches an nvshmem symbol.

        Args:
            n_token: Token extent to pre-compile for. **Optimization only: it places NO restriction
                on what `forward` accepts.** One compiled instance serves every token extent
                satisfying ``N % cp_axis == 0`` for each cp axis and ``N % 8 == 0`` (bf16 x 8 = the
                16-byte TMA floor); on a 2-D mesh each per-axis local extent must also be ``% 8``.
                Nothing else -- not a tile multiple, not a power of two.

                **Which value to pass.** Measured: the resolved kernel configuration is a function
                of ``(D, cp)`` alone and does NOT vary with ``n_token`` (720 combinations swept, 0
                differences; `docs/refactor_dtensor_api.md` §3.5.1). Under autotune it snaps to
                ``DYNAMIC_ANCHOR_N = 2048`` regardless. So ``n_token`` does not change which kernel
                you get and there is no "best" value for speed. Pass **the token extent you will
                actually run**, for one reason: the build allocates and caches this shape's
                symmetric receive buffers, and a build at one extent followed by forwards at another
                leaves that allocation resident and unread. The choice is about MEMORY, not
                throughput.
            batch: Batch extent to pre-compile for. **NOT purely an optimization, and this is a
                known limitation rather than a design intent.** A compiled instance is bound to its
                batch extent: the front operand's M is ``batch * N_i_loc * N_j_loc``, the back
                receive buffer is shaped with it, and the default incoming fast-path stores
                (``composite_k`` 1-D / ``route2_ni`` 2-D) are ``batch == 1`` only. A `forward` with
                a different batch therefore **builds a second instance** -- another compile and
                another set of symmetric buffers, which `forward` warns about once. That is the
                behaviour of this package today and is not introduced here. Pass the batch you will
                run. If a workload genuinely varies batch per call, file it: making the batch extent
                dynamic is tracked work, not a configuration option.
            masked: Whether the calls will pass a ``mask``. A genuine COMPILE-TIME variant of the
                front store, so a masked and an unmasked module hold separate engines.

        Returns:
            The built `TriMulAutotuned`, or ``None`` for a ``cp == 1`` module.

        Raises:
            ValueError: if ``n_token`` or ``batch`` is not >= 1.
            RuntimeError: from `_ensure_nvshmem` if nvshmem cannot be brought up for ``cp > 1``.
        """
        if int(n_token) < 1 or int(batch) < 1:
            raise ValueError(f"n_token and batch must be >= 1; got {n_token!r}, {batch!r}")
        if self._cp == 1:
            return None
        key = (int(batch), bool(masked))
        eng = self._engines.get(key)
        if eng is not None:
            return eng
        variant = incoming_store_variant(self._dir, self._cp1, int(batch))
        kw = dict(self._kw)
        # An explicit tuning override always wins; otherwise the rule decides. OUTGOING is
        # byte-identical under either store, so the rule only ever sets these for incoming.
        if variant != "plain" and "composite_k" not in kw and "route2_ni" not in kw:
            kw[variant] = True
        _ensure_nvshmem(self._dm)
        eng = TriMulAutotuned(
            self._pe_map, int(batch), int(n_token), self._D, self._w, self._dt,
            device_mesh=self.device_mesh, placements=self._placements,
            dynamic=True, has_mask=bool(masked), eps=self._eps, **kw,
        )
        self._engines[key] = eng
        # Record what the engine BUILT, not what the rule ASKED FOR. No path diverges today -- the
        # cross-node batch demotion that used to make these differ is GONE (the IB drain now carries
        # the plane in its ring metadata), so this currently always agrees with `variant`. It is kept
        # as a WITNESS rather than collapsed into `variant`: storing the rule's answer would make
        # `incoming_variant` -- and through it every benchmark cell's metadata and the engine-
        # equivalence gate -- report a fast store whenever the rule asked for one, which is the
        # silent-downgrade failure this repo has already paid for once, arriving through the very
        # property added to prevent it. A witness costs one attribute read and cannot go stale.
        self._engine_variant[key] = (
            "composite_k" if eng.composite_k else ("route2_ni" if eng.route2_ni else "plain")
        )
        return eng

    @torch.no_grad()
    def forward(self, x, mask=None):
        """DTensor ``(B, N, N, D)`` -> DTensor ``(B, N, N, D)`` (same sharding). ``mask`` optional.

        Accepts a plain local ``Tensor`` too (no mesh): that -- like a 1-device mesh, or placements
        that split no token axis -- is ``cp == 1`` and routes to the single-device
        ``trimul_autotuned``, returning the same type it was given (plain in / plain out). No
        instance is cached and no nvshmem symbol is touched there, so a cp=1 module never needs a
        live ``DistributedManager``.

        Functionality & semantics
            After `§4.2` there are no decisions left here. The mesh, the placements, the cp map and
            the variant rule were all resolved at construction; what arrives with the tensor is the
            token extent, the batch extent and whether there is a mask. So this method CHECKS the
            three things the tensor could contradict -- mesh, placements, feature width -- looks up
            or builds the ``(batch, masked)`` engine, and re-wraps. Each raise names the CONSTRUCTOR
            argument that disagrees with the tensor, because that is the argument the caller must
            change.

            ``batch`` and ``masked`` are DISCOVERED, not asserted: a second pair builds a second
            engine, exactly as before. What is new is that it says so once (`R8`) instead of
            compiling and allocating in total silence.

        Args:
            x: The pair representation. A ``DTensor`` whose mesh AND placements must equal this
                module's; a mismatch raises rather than silently resharding onto a PE map the
                compiled instance was not built for. Its feature width must equal the layer's
                (``norm_in``'s gain length). A plain ``Tensor`` bypasses all three checks and runs
                single-device.
            mask: Pair mask with the SAME token sharding as ``x``, a local tensor, or ``None``.

        Returns:
            Same type and sharding as ``x``.

        Raises:
            ValueError: If ``x`` is a DTensor on a different mesh or with different placements than
                the constructor was given, or if its feature width is not the layer's.
        """
        x_mesh = getattr(x, "device_mesh", None)
        if x_mesh is not None and self.device_mesh is not None and x_mesh is not self.device_mesh:
            if x_mesh != self.device_mesh:
                raise ValueError(
                    f"TriangularMultiplication was constructed on device_mesh={self.device_mesh}, "
                    f"but forward() received a DTensor on {x_mesh}. The cached fused instance is "
                    "compiled for the constructor's mesh; running it against another one would "
                    "reshard onto a PE map it was not built for."
                )
        # cp == 1 is decided at construction, but a plain tensor (or a DTensor whose own placements
        # split no token axis) is ALSO cp == 1 -- both routed single-device before this refactor and
        # still do, so the per-call check stays rather than becoming a mesh-only decision.
        if self._cp == 1 or _is_single_device(x):
            return _trimul_single_device(
                x, self._w, direction=self._dir, mask=mask, eps=self._eps
            )
        _, x_pl = _dtensor_mesh_placements(x)
        if list(x_pl) != list(self._placements):
            raise ValueError(
                f"TriangularMultiplication was built for placements={self._placements}, but "
                f"forward() received a DTensor placed {list(x_pl)}. The PeMap and the compiled "
                f"stores are derived from the constructor's placements; running against another "
                f"sharding would all-to-all over a different set of peers. Pass placements= to the "
                f"constructor if this module's sharding is not the default "
                f"[Shard(i + 1) for i in range(mesh.ndim)]."
            )
        if int(x.shape[3]) != self._D:
            raise ValueError(
                f"TriangularMultiplication holds weights of feature width D={self._D} (from "
                f"norm_in), but forward() received a tensor of width {int(x.shape[3])}. The "
                f"projections would not contract."
            )
        key = (int(x.shape[0]), mask is not None)
        eng = self._engines.get(key)
        if eng is None:
            self._warn_second_variant(key)
            eng = self.prepare(n_token=int(x.shape[1]), batch=key[0], masked=key[1])
        x_local = x.to_local()
        mask_local = mask.to_local() if _is_dtensor(mask) else mask
        out_local = eng.forward(x_local, self._dir, mask_local=mask_local)
        return _rewrap_like(x, out_local)

    def _warn_second_variant(self, key) -> None:
        """Warn ONCE when a `forward` is about to build a SECOND ``(batch, masked)`` engine.

        Purpose
            A second pair costs another multi-second compile and another full set of symmetric
            receive buffers, and today that happens in total silence -- a caller varying batch per
            call pays it repeatedly with nothing in the log to connect the memory growth to the
            cause.

        Functionality & semantics
            Fires only from the second distinct key onward (the first build is the expected one),
            and only once per module: a warning on every miss would itself become noise in a loop
            that legitimately alternates. It names both keys and both costs. It does NOT raise --
            the behaviour is supported, just expensive, and refusing it would break a caller this
            package has always served.

        Args:
            key: The ``(batch, masked)`` pair about to be built.

        Returns:
            None.
        """
        if self._warned_second_variant or not self._engines:
            return
        self._warned_second_variant = True
        import warnings

        warnings.warn(
            f"TriangularMultiplication is building a SECOND compiled engine: it already holds "
            f"(batch, masked)={sorted(self._engines)} and forward() was called with {key}. Each "
            f"engine is a separate multi-second compile AND a separate set of symmetric receive "
            f"buffers, both held for the module's lifetime. If the batch extent genuinely varies "
            f"per call this is expected; if it does not, the shapes disagree. Warned once.",
            RuntimeWarning,
            stacklevel=3,
        )

    def free(self):
        """Release every built engine's COMPILED KERNEL MODULES. Call before process exit.

        This docstring used to say "free every cached instance's symmetric buffers (call before
        nvshmem finalize)". Both halves were wrong, and both matter:

        * What it frees is the nvshmem-registered **CUDA libraries** -- the chain reaches
          `DualGatedGemmDistStore.free` -> `self._compiled.free()`. It does also call
          `_symmetric_free` on each recv cache, but that function is a DELIBERATE NO-OP: under the
          MemPool a block returns when its last reference drops, and `nvshmem_free` is COLLECTIVE,
          so calling it from a teardown path would have ranks arrive in whatever order their
          garbage collectors chose -- a deadlock rather than an error.
        * `DistributedManager.cleanup()` does NOT finalize nvshmem, so "before nvshmem finalize"
          named an ordering that does not exist there. Finalizing is a one-way door per process and
          runs once from an `atexit` hook that `init_nvshmem()` registers.

        A no-op when only the cp=1 fallback ran -- that path builds no `TriMulAutotuned`, so there
        is nothing to release and ``free()`` neither raises nor touches nvshmem.
        """
        for inst in self._engines.values():
            try:
                inst.free()
            except Exception:
                pass
        self._engines.clear()
        self._engine_variant.clear()

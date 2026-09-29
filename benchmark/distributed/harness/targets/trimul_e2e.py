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

"""trimul_e2e adapter (§6 of docs/trimul_nvshmem_ib_integration_plan.md) — the FULLY-FUSED distributed
TriMul + its two context-parallel BASELINES as harness BenchTargets. Thin plug-ins over the EXISTING,
proven drivers (``TriMulAutotuned``, ``benchmark.distributed.t2_0_dtensor_baseline``,
``benchmark.distributed.trimul_dtensor_baseline``); NO kernel/baseline logic is duplicated here. The build/run
idiom mirrors ``benchmark/distributed/nsys_trimul_dtensor.py`` (the existing driver that drives all three).

Registered names (``direction`` is a CELL dimension via the ``_out``/``_in`` suffix, per §6.1 — the harness
``Ctx`` has no direction field, so ``run_bench.sbatch`` selects the direction per cell by target name):

  * ``trimul_fused_{out,in}``    (role=target)   — ``TriMulAutotuned(hybrid_ib=None -> AUTO, dynamic=True)``.
    ``build`` constructs at the cell (B,D,N,cp0,cp1,direction) + warms up (JIT compile + the symmetric recv
    alloc happen in build -> an OOM/fault is caught + tagged at BUILD, where the driver's oom seam lives).
    ``run`` = the single timed hot ``forward``. ``supports`` = the pure-shape gate. ``teardown`` frees the
    symmetric recvs. ``configs`` = a single ``{}`` (the fused kernel autotunes its OWN back/front stores
    INTERNALLY via the freeze-cached ``trimul_autotune_policy.py`` under ``CPO_DIST_AUTOTUNE=1`` — we
    neither re-sweep that grid, which would leave a pile of proxy builds live inside the TIMED process, nor
    hand-duplicate it). ``comm_bytes`` left None (fused wall-time only — comm overlaps compute, not
    separable, §6.1).
  * ``trimul_dtensor_{out,in}``  (role=baseline) — ``t2_0_dtensor_baseline.distributed_trimul_fwd`` (torch
    DTensor redistribute all-to-all). ``comm_bytes`` = V_a2a (derived-BW column).
  * ``trimul_dtensor_baseline_ring_reducescatter_{out,in}``    (role=baseline) — the vendored reference CP native ring (§5.2), 1-D or 2-D by
    the live mesh. ``comm_bytes`` = V_a2a.

K=N HARD RULE (CLAUDE.md): the fused builds REAL ``(B, N_i_loc, N_j_loc, D)`` token operands, so the back
einsum ``out[b,i,j,d]=Σ_k a[b,i,k,d]·b[b,j,k,d]`` is SQUARE (K=N_token) by construction — never a decoupled
K. The FRONT projection is K=D-exempt.

INPUT PATH — (b) NATIVE LOCAL SHARD, NOT an x_global (load-bearing for a VALID §6.4 OOM filter): ALL three
paths (fused + BOTH baselines) build ONLY this rank's native local block ``(B, N_i_loc, N_j_loc, D)`` =
O(N²/cp) and wrap it as a DTensor via ``DTensor.from_local`` — NO rank ever allocates the O(N²) global
``(B, N, N, D)``. Production-realistic (each rank natively holds its shard; no rank holds x_global in
production) AND REQUIRED for the filter's validity: if a baseline built x_global it would OOM on the INPUT
alloc (a harness artifact) at a SMALLER N than production, misrepresenting §6.4's KEEP verdict ("baseline
OOMs, fused fits") at cells where the baseline would actually FIT. With (b), a baseline OOM reflects its
INTERNAL algorithm memory (e.g. the incoming ~16 GB transpose temp, design §5.5) — a REAL algorithmic loss.
Perf is value-independent, so an arbitrary right-shaped local shard gives the correct timing (comm volume is
shape-determined) — no rank→block-mapping consistency is needed.

OOM filter (§6.4): each ``build`` is wrapped so a ``torch.cuda.OutOfMemoryError`` OR an nvshmem
symmetric-heap alloc failure is caught + NORMALIZED to the driver's build-phase ``status='oom'`` seam
(driver.py:109 ``type().__name__=='OutOfMemoryError'``), which is ALREADY consensus'd across ranks
(``_consensus_ok``). The three-way DROP/KEEP/LABEL verdict is a deterministic function of two
already-consensus'd statuses (``oom_verdict`` below) — a REPORTING function over the per-cell JSON the
driver already writes; NO extra collective, NO driver change (the seam already exists + is tested by
tests/distributed/test_harness_driver.py::test_build_oom_tagged_status_oom). NO memory-estimate pre-gate —
runtime catch only (CLAUDE.md).
"""
import os

from benchmark.distributed.harness.registry import register
from benchmark.distributed.harness.target import BenchTarget

_DTYPE_BYTES = 2   # bf16
_SEED = 20260622   # replicated-weights seed (+ per-rank local-shard offset; mirror nsys_trimul_dtensor.py)

# module-level caches (N-invariant; keyed by cp / D). Per-cell isolation => one process/cell => these are
# populated once, but caching keeps a multi-target smoke cell (fused+dtensor+dtensor_baseline_ring_reducescatter in ONE launch) cheap.
_MESH_CACHE = {}     # (cp0,cp1) -> (pm, fmesh, fpl, ib, jb)  [TriMulAutotuned mesh + my cp coord]
_DMESH_CACHE = {}    # (cp0,cp1) -> the (dp,cp0,cp1) DTensor-baseline mesh
_WEIGHT_CACHE = {}   # D -> replicated TriMul weights dict


# ─────────────────────────────────────────── pure shape math (deterministic on EVERY rank; no GPU) ───
def _cp(ctx):
    return int(ctx.cp0) * int(ctx.cp1)


def _D(ctx):
    # the harness Ctx carries Dloc (= D // cp, floored by __main__); reconstruct the full feature dim.
    return int(ctx.Dloc) * _cp(ctx)


def _supports(ctx):
    """Pure-shape gate = the mirror of ``TriMulAutotuned.__init__``'s host guards (== nsys ``_valid``) PLUS the
    front feature-scatter ``Dloc>=8`` floor (user-scoped C9 SKIP, plan line 289 — a SKIP, NOT an xfail).
    IDENTICAL on every rank (collective symmetry is load-bearing). Returns a human reason to skip, or None.

    NOTE: NOT ``roundpark._shape_check`` — that gate is 1-D-STRICTER (it also requires N//cp0 %8 in 1-D),
    which would wrongly SKIP e.g. N=1000/cp2 (N_i_loc=500) that ``TriMulAutotuned`` 1-D actually runs (it guards
    only N%8 in 1-D; per-axis %8 only when cp1>1). We mirror the kernel's ACTUAL guards to neither over- nor
    under-constrain (CLAUDE.md FIRST PRINCIPLE — never skip an in-principle-supported shape)."""
    cp0, cp1 = int(ctx.cp0), int(ctx.cp1)
    if cp0 < 1 or cp1 < 1:
        return f"invalid cp mesh cp0={cp0} cp1={cp1}"
    cp = cp0 * cp1
    N, D, Dloc = int(ctx.N), _D(ctx), int(ctx.Dloc)
    # D%cp is STRUCTURALLY satisfied here (Ctx carries the already-divided Dloc, so D=Dloc*cp is always
    # %cp==0); kept as defense-in-depth mirroring TriMulAutotuned.__init__:1228. The real D%cp!=0 case (a --D
    # not divisible by cp) is enforced UPSTREAM at the CLI (__main__ floors Dloc=D//cp); the §6.2 grid
    # (D∈{128,256,384,512}) is divisible by every grid cp, so it never fires.
    if D % cp != 0:
        return f"D={D} % cp={cp} != 0 (feature reshard S(0,1,2)->S(0,3,3))"
    if N % cp0 != 0 or N % cp1 != 0:
        return f"N={N} not divisible by both cp axes ({cp0},{cp1}) (not token-shardable)"
    if N % 8 != 0:
        return f"N={N} % 8 != 0 (pe_aligned back store 16-B stride-1 TMA-S2G HARD RULE)"
    if cp1 > 1 and ((N // cp0) % 8 != 0 or (N // cp1) % 8 != 0):
        return (f"2-D back store needs each per-axis local extent %8: N_i_loc={N // cp0}, "
                f"N_j_loc={N // cp1}")
    if Dloc < 8:
        return f"Dloc=D/cp={Dloc} < 8 (front feature-scatter floor; user-scoped C9 SKIP)"
    return None


def _dtensor_baseline_ring_reducescatter_supports(ctx):
    """reference-baseline gate = the shared shape gate PLUS the reference CP 2-D SQUARE-GRID requirement. The vendored
    reference CP 2-D ring (Ring2DComm / triangular_mult) assumes a SQUARE cp grid (cp0==cp1) and raises
    ValueError('group_l...') on a non-square 2-D mesh (memory project_dtensor_adapter_t13; MEASURED on the
    smoke — (2,8) ERRORs, (4,4) OK). Non-square 2-D -> a clean SKIP for dtensor_baseline_ring_reducescatter ONLY (fused + DTensor stay
    valid there, so the cell is still covered); 1-D (cp1==1) and square 2-D (cp0==cp1) are unaffected.
    Deterministic on every rank (collective symmetry)."""
    reason = _supports(ctx)
    if reason is not None:
        return reason
    cp0, cp1 = int(ctx.cp0), int(ctx.cp1)
    if cp1 > 1 and cp0 != cp1:
        return (f"reference CP 2-D ring requires a SQUARE cp grid (cp0==cp1); got ({cp0},{cp1}) non-square "
                f"(baseline limit — fused+DTensor cover this cell)")
    return None


def _comm_bytes(ctx):
    """Per-rank off-diagonal A2A byte volume V_a2a = 2·B·D·N²·(cp−1)/cp² (design §8 / memory
    front_a2a_sol_fgemm; the leading 2 folds the bf16 dtype). Attached to the BASELINES only -> the driver's
    derived-BW column (gbps_total/ib/nvl); the fused target leaves comm_bytes None (§6.1 — the fused comm
    overlaps compute in the GEMM epilogue and is NOT separable from wall time). cp==1 -> 0 (no off-diagonal)."""
    cp = _cp(ctx)
    if cp <= 1:
        return 0
    N, D, B = int(ctx.N), _D(ctx), int(ctx.B)
    return int(2 * B * D * N * N * (cp - 1) / (cp * cp))


def _configs(ctx):
    """Per-cell config = the ``has_mask`` fairness axis (plan §6): mask-OFF (legacy reference) + mask-ON
    (the apples-to-apples headline — the reference CP baseline always runs mask-ON, so only the mask-ON fused cell
    is a fair head-to-head). Two builds/cell, each released by ``teardown`` (which is what keeps the DSL's
    collector from unloading a library nvshmem still holds); each fused build autotunes its OWN back/front stores
    INTERNALLY (freeze-cached ``trimul_autotune_policy.py``, ``CPO_DIST_AUTOTUNE=1``) so this is NOT a
    per-cell autotune RE-sweep. Deterministic + identical on every rank (SPMD)."""
    return [{"has_mask": False}, {"has_mask": True}]


# ──────────────────── strong-/weak-scaling instrumentation (docs/strong_weak_scaling_plan.md §6-§7) ────
# The scaling tables need, PER CELL: the LOCAL shard extents (the weak-scaling evidence that the constant-shard invariant
# `N²/cp = const` really holds), the per-device PEAK memory, and WHICH incoming store the shipped DTensor
# API auto-selected (composite_k 1-D / route2_ni 2-D — fused_trimul_cp.py:130-137). `cell_meta(ctx)` is the
# harness's per-cell extra-fields hook (driver.py:178; covered by test_harness_driver.py::
# test_ok_cell_with_cell_meta_merges_dict): it runs AFTER teardown and receives ONLY ctx, so the build
# stashes the live (non-shape-derivable) numbers here for it to pick up.
_LAST_META = {}


def _mem_reset():
    """Zero the torch caching-allocator high-water mark at BUILD entry, so a cell's peak is its OWN. Per-cell
    isolation gives one cell per process, but a cell runs TWO configs (mask off/on) — without this, cfg1
    would inherit cfg0's peak."""
    import torch
    _LAST_META.clear()
    try:
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _mem_snap(extra=None):
    """Capture device occupancy at the end of BUILD (post-warmup, PRE-teardown). Two numbers, because they
    measure different things: the torch caching-allocator high-water (read later in `_cell_meta`) MISSES the
    nvshmem SYMMETRIC heap — symmetric memory is not torch-allocator memory (memory
    nvshmem_symheap_not_torch_allocator_freeable) — whereas `dev_used_mb` (cudaMemGetInfo total-free) DOES
    include the symmetric recvs + the CUDA context. Sampled here and not in `_cell_meta` because teardown
    frees the symmetric recvs, so a post-teardown cudaMemGetInfo would read back near-empty.

    NOT-A-COMBO-SELECTOR: this cudaMemGetInfo is a REPORTING column for the §7 weak-scaling table (evidence
    the per-device shard is constant). It never feeds a combo/variant/shape decision — nothing branches on
    it. Capacity is handled where it must be, at runtime: `_reraise_as_oom_if_alloc` normalizes a real
    allocation failure to the driver's `status='oom'` seam (CLAUDE.md: runtime OOM catch, never a
    memory-estimate gate)."""
    import torch
    d = dict(extra or {})
    try:
        free, total = torch.cuda.mem_get_info()
        d["dev_used_mb"] = (total - free) / 2**20
    except Exception:
        pass
    _LAST_META.update(d)


def _cell_meta(ctx):
    """Extra per-cell columns for the §6/§7 scaling tables, merged into the ok cell by the driver. Shape math
    is deterministic on every rank; the memory numbers are THIS rank's (rank0 writes the JSON and every rank
    is SPMD-symmetric, so it is the 'per-device peak memory' §7 asks for). Never raises (the driver
    None-guards, but a silent {} would drop the Phase-E evidence column)."""
    import torch
    cp0, cp1, cp = int(ctx.cp0), int(ctx.cp1), _cp(ctx)
    N, D, B = int(ctx.N), _D(ctx), int(ctx.B)
    ni, nj = N // cp0, N // cp1
    m = {"cp": cp, "D": D, "N_i_loc": ni, "N_j_loc": nj,
         # the constant-shard invariant: held CONSTANT down a weak-scaling column by N = 2048·√cp
         "shard_elems": B * ni * nj * D, "shard_bytes": B * ni * nj * D * _DTYPE_BYTES}
    try:
        m["peak_torch_mb"] = torch.cuda.max_memory_allocated() / 2**20
        m["peak_reserved_mb"] = torch.cuda.max_memory_reserved() / 2**20
    except Exception:
        pass
    m.update(_repo_revision())   # per-cell revision stamp: makes a mid-arm tree overlay DETECTABLE
    m.update(_LAST_META)         # build-stashed: dev_used_mb + the selected variant/combo
    return m


def _repo_revision():
    """The revision of the tree THIS cell actually imported, stamped into EVERY cell.

    A checkout can be updated in place during a 46-cell arm, so the tree can change mid-arm. A
    grid-level revision recorded once at startup would then be inaccurate for later cells. Stamping
    per cell makes drift detectable in the results instead of silently mixing revisions in one table.
    Derived from THIS file's location, so it names the tree that was imported, not one passed in by a
    caller."""
    try:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
        with open(os.path.join(root, "REVISION")) as f:
            return {"repo_revision": f.read().strip()[:40]}
    except Exception:
        return {}


def _rank_gpu_affinity(ctx):
    """DECISIVE rank->GPU placement check, run once at BUILD entry.

    A rank-to-GPU COLLISION (two ranks landing on one device) is the nastiest failure class here: it does
    NOT fail loudly, it produces plausible-looking but wrong timings — two ranks sharing an SM budget just
    look "a bit slow", and every derived speedup/efficiency number inherits the error silently. It is a
    live hazard because the two valid pinning models differ in WHERE the pin happens: srun-per-rank
    (LOCAL_RANK == SLURM_LOCALID, all GPUs visible, `set_device` pins) vs per-task CUDA_VISIBLE_DEVICES
    (one GPU visible, LOCAL_RANK == 0 pins trivially). Mixing them is what collides ranks.

    So check it EXACTLY rather than inferring it from a memory ratio: every rank contributes
    ``hostname:GPU-UUID`` and we require the count of DISTINCT devices to equal the world size. A UUID is
    used, not the device index, because under per-task CUDA_VISIBLE_DEVICES every rank reports index 0 and
    an index-based check would be vacuous.

    ``all_gather_object`` is collective; this runs at build entry on every rank in lockstep (the driver's
    build phase is collective-symmetric), so it cannot desync. On a COLLISION it RAISES: the driver
    isolates + consensus'es the build failure into a failed cell, and a loudly failed cell is far better
    than a fast-looking wrong number.
    """
    import socket
    import torch
    import torch.distributed as dist
    if not (dist.is_available() and dist.is_initialized()):
        return {"rank_gpu_check": "n/a (single process)"}
    idx = torch.cuda.current_device()
    try:
        p = torch.cuda.get_device_properties(idx)
        key = str(getattr(p, "uuid", None) or getattr(p, "pci_bus_id", None) or f"idx{idx}")
    except Exception:
        key = f"idx{idx}"
    me = f"{socket.gethostname()}:{key}"
    ws = dist.get_world_size()
    seen = [None] * ws
    dist.all_gather_object(seen, me)
    n = len(set(seen))
    if n != ws:
        raise RuntimeError(
            f"RANK->GPU COLLISION: {ws} ranks mapped onto only {n} distinct GPUs ({sorted(set(seen))}). "
            f"Timings from this layout are silently WRONG (ranks share a device). Check the launcher's "
            f"pinning model: srun-per-rank needs LOCAL_RANK==SLURM_LOCALID with all GPUs visible; a "
            f"per-task CUDA_VISIBLE_DEVICES launcher needs --gpus-per-task=1/--gpu-bind."
        )
    return {"rank_gpu_check": "ok", "distinct_gpus": n, "world_size": ws, "gpu_key": me}


def _selected_variant(mod):
    """Read back WHICH incoming store the module actually built, from its `incoming_variant`.

    Read back rather than re-derived, so the reported config is the one that ran and not a
    duplicate of the selection rule.

    This used to decode the instance-cache KEY by negative index (`key[-2:] == (composite_k,
    route2_ni)`) inside a bare `except Exception: return {}`. When that key collapsed to
    `(batch, masked)`, the failure mode was NOT a crash -- the three variant fields simply stopped
    appearing in every cell's metadata and the downstream tables went on rendering without them.
    The module now exposes the decision it made, so there is no key to decode and nothing to
    swallow: `incoming_variant` returns a VALUE in every state, including `"cp1_fallback"` (this
    module runs the single-device kernel and has no front store), `"unbuilt"` (nothing has run yet)
    and `"mixed"` (engines of more than one variant). An `AttributeError` here means the object is
    not a `TriangularMultiplication`, which is a bug worth raising rather than hiding.
    """
    v = mod.incoming_variant
    return {"incoming_variant": v,
            "composite_k": v == "composite_k", "route2_ni": v == "route2_ni"}


# ───────────────────────────────────────── the §6.4 three-way OOM filter (pure; report-layer) ──────────
def oom_verdict(target_status, baseline_status):
    """§6.4 three-way OOM filter. Inputs = the driver's per-cell statuses for a (target, baseline) PAIR at
    the SAME (D,N,cp,dir) cell; each is ALREADY consensus'd across ranks by ``driver._consensus_ok`` (so it
    is identical on every rank), which makes THIS verdict identical on every rank too — a deterministic
    function of two consensus'd inputs needs NO additional collective. Returns
    ``{keep, target_label, baseline_label, reason}`` where a ``'OOM'`` label marks the OOM'd side of a KEPT
    cell (the harness records the OOM'd target as ``status='oom'``; this pairs them for the results table)."""
    t_oom = (target_status == "oom")
    b_oom = (baseline_status == "oom")
    if t_oom and b_oom:
        return {"keep": False, "target_label": "OOM", "baseline_label": "OOM",
                "reason": "both target AND baseline OOM -> genuinely too large -> DROP from the table"}
    if b_oom and not t_oom:
        return {"keep": True, "target_label": None, "baseline_label": "OOM",
                "reason": "only baseline OOM -> KEEP; report the target (we run where the baseline can't)"}
    if t_oom and not b_oom:
        return {"keep": True, "target_label": "OOM", "baseline_label": None,
                "reason": "only target OOM -> KEEP; report the baseline (an honest loss, never hidden)"}
    return {"keep": True, "target_label": None, "baseline_label": None,
            "reason": "both ran -> normal head-to-head"}


def _reraise_as_oom_if_alloc(exc):
    """Normalize an nvshmem symmetric-heap alloc failure to ``torch.cuda.OutOfMemoryError`` so the driver's
    BUILD-phase oom seam tags ``status='oom'`` (driver.py:109). A torch OOM already IS that class (re-raise
    as-is). A symmetric-over-alloc raises a RuntimeError whose message carries a MEMORY signature -> re-tag
    oom; anything else re-raises UNCHANGED (driver -> 'error'). NEVER returns (always raises). An ASYMMETRIC
    nvshmem alloc HANG is a DIFFERENT failure (a rendezvous desync) -> caught by the build TIMEOUT budget,
    not here (memory front_bench_n2048_cp16_nvshmem_alloc_hang)."""
    import torch
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        raise exc
    msg = (str(exc) or "").lower()
    _MEM_SIG = ("out of memory", "outofmemory", "cudaerrormemoryallocation", "bad_alloc",
                "cannot allocate", "insufficient memory", "symmetric heap", "nvshmem_malloc")
    if any(s in msg for s in _MEM_SIG):
        raise torch.cuda.OutOfMemoryError(f"alloc failure normalized to OOM: {exc!r}") from exc
    raise exc


# ──────────────────────────────────────────── shared build helpers (GPU; lazy-imported) ────────────────
def _fused_mesh(ctx):
    """Reconstruct (pm, fmesh, fpl, ib, jb) the SAME way nsys_trimul_dtensor.py does (the proven path);
    cached by (cp0,cp1) since it is N-invariant. ``init_device_mesh`` / ``from_mesh_placements`` are
    collective -> every rank calls them in lockstep inside the harness's collective-symmetric build()."""
    from torch.distributed.tensor import Shard
    from fold_cp_ops.distributed.layout_map import LayoutRightMap
    from fold_cp_ops.distributed.pe_map import PeMap
    dm = ctx.dm
    key = (int(ctx.cp0), int(ctx.cp1))
    ent = _MESH_CACHE.get(key)
    if ent is None:
        sub = getattr(dm, "device_mesh_subgroups", None)
        fmesh = sub if (getattr(dm, "has_subgroups", False) and sub is not None) else dm.device_mesh
        fpl = [Shard(i + 1) for i in range(fmesh.ndim)]
        pm = PeMap.from_mesh_placements(fmesh, fpl, distributed_manager=dm)
        coord = LayoutRightMap(tuple(int(s) for s in pm.cp_axis_sizes)).unravel(int(pm.my_cp_rank))
        ib = coord[0]
        jb = coord[1] if len(coord) > 1 else 0
        ent = (pm, fmesh, fpl, ib, jb)
        _MESH_CACHE[key] = ent
    return ent


def _dtensor_mesh(cp0, cp1):
    key = (cp0, cp1)
    m = _DMESH_CACHE.get(key)
    if m is None:
        from torch.distributed.device_mesh import init_device_mesh
        m = init_device_mesh("cuda", (1, cp0, cp1), mesh_dim_names=("dp", "cp0", "cp1"))
        _DMESH_CACHE[key] = m
    return m


def _weights(ctx):
    from tests.distributed.correctness_harness import make_weights
    D = _D(ctx)
    w = _WEIGHT_CACHE.get(D)
    if w is None:
        w = make_weights(D, seed=_SEED, device=ctx.dm.device)
        _WEIGHT_CACHE[D] = w
    return w


def _local_block(ctx, dt):
    """This rank's NATIVE token shard ``(B, N_i_loc, N_j_loc, D)`` = O(N²/cp), built DIRECTLY (no O(N²)
    global). The SHARED (b)-path input for ALL three targets — fused consumes it directly; the baselines
    wrap it via ``DTensor.from_local`` (module docstring INPUT PATH). Perf is value-independent so an
    independent per-rank seed is fine (no correctness gate in this perf sweep)."""
    import torch
    cp0, cp1 = int(ctx.cp0), int(ctx.cp1)
    N, D, B = int(ctx.N), _D(ctx), int(ctx.B)
    g = torch.Generator(device="cpu").manual_seed(_SEED + int(ctx.rank))
    return (torch.randn(B, N // cp0, N // cp1, D, generator=g, dtype=torch.float32)
            .to(dt).to(ctx.dm.device))


def _rowmaj_stride(shape):
    """Row-major (C-contiguous) strides for `shape` — the explicit global metadata for DTensor.from_local."""
    st = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        st[i] = st[i + 1] * int(shape[i + 1])
    return tuple(st)


def _dtensor_from_local(local, mesh, placements, global_shape):
    """Wrap this rank's LOCAL shard as a DTensor on `mesh`/`placements` WITHOUT allocating the O(N²) global
    (the (b) input path — module docstring). Explicit global shape+stride so from_local needs NO
    shape-inference collective (mirrors t2_0_dtensor_baseline.py:125's own from_local)."""
    import torch
    from torch.distributed.tensor import DTensor
    gs = torch.Size(tuple(int(s) for s in global_shape))
    return DTensor.from_local(local.contiguous(), mesh, list(placements), shape=gs,
                              stride=_rowmaj_stride(gs))


# ──────────────────────────────────────────────────── FUSED target (role=target) ──────────────────────
def _build_fused(direction):
    def _build(ctx):
        import torch
        from fold_cp_ops.distributed.workflows.trimul_autotuned import TriMulAutotuned
        _mem_reset()
        aff = _rank_gpu_affinity(ctx)   # RAISES on a rank->GPU collision (silently-wrong-timings guard)
        pm, fmesh, fpl, _ib, _jb = _fused_mesh(ctx)
        dt = torch.bfloat16
        w = _weights(ctx)
        N, D, B = int(ctx.N), _D(ctx), int(ctx.B)
        xl = _local_block(ctx, dt)
        # MASK-FAIRNESS config axis: has_mask is a per-cell CONFIG (see _configs -> mask-off +
        # mask-on cells). The reference CP baseline ALWAYS runs mask-ON (a ones mask executing x*mask.unsqueeze(-1)
        # at trimul_dtensor_reference.py:1888) — so the mask-ON fused cell is the apples-to-apples HEADLINE (fused
        # pays its ~free fused-epilogue mask exactly as the reference pays its separate elementwise pass), while the
        # mask-OFF cell is the legacy reference (over-states the fused win by ~one mask-multiply). Default
        # off => byte-identical to the pre-config sweep.
        has_mask = bool(ctx.cfg.get("has_mask", False))
        mask_l = (torch.ones(B, int(xl.shape[1]), int(xl.shape[2]), device=xl.device, dtype=dt)
                  if has_mask else None)
        h = {"ftm": None, "xl": xl, "dir": direction, "mask": mask_l}
        try:
            # hybrid_ib=None (DEFAULT) => AUTO-DETECT the transport from the P2P topology (all-P2P ->
            # coupled NVLink; any cross-node IB peer -> ib_drain). dynamic=True => compile ONCE at this
            # anchor N. consumer="stagec" (the shipped e2e consumer). has_mask from the per-cell config.
            ftm = TriMulAutotuned(pm, B, N, D, w, dt, consumer="stagec",
                              device_mesh=fmesh, placements=fpl, dynamic=True, has_mask=has_mask)
            ftm.forward(xl, direction, mask_local=mask_l)  # WARMUP: JIT compile + symmetric recv alloc land
            torch.cuda.synchronize()     # HERE (a compile/alloc OOM/fault is caught + tagged at BUILD, not run)
            _mem_snap({"path": "TriMulAutotuned(raw local shard)", "consumer": "stagec", **aff})
            h["ftm"] = ftm
            return h
        except Exception as e:
            _teardown_fused(h)           # free the partial symmetric recv on ANY build failure (no leak)
            _reraise_as_oom_if_alloc(e)  # normalize alloc failure -> oom seam; else re-raise unchanged
    return _build


def _run_fused(h):
    h["ftm"].forward(h["xl"], h["dir"], mask_local=h["mask"])  # the single fused e2e forward (TIMED; no alloc)


def _teardown_fused(h):
    ftm = h.get("ftm")
    if ftm is not None:
        try:
            ftm.free()   # frees the front + back symmetric recvs (fused_trimul.py:1690)
        except Exception:
            pass
    h["ftm"] = None


# ──────────────────────── TriangularMultiplication DTensor-API target (role=target) ────────────────
# FAIRNESS: profile the SHIPPED DTensor API TriangularMultiplication(x_dt, mask_dt) — DTensor-in /
# paying the from_local/to_local host overhead the reference CP baseline ALSO pays — so the headline is strictly
# DTensor-API vs DTensor-API (not raw-TriMulAutotuned-on-local-shards vs DTensor-reference). Same local shard, same
# ones mask, same fused mesh/placements as trimul_fused; the ONLY delta vs trimul_fused is the DTensor wrap.
def _build_fusedcp(direction):
    def _build(ctx):
        import torch
        from fold_cp_ops.distributed.trimul_weights import trimul_module_from_weights
        from fold_cp_ops.distributed.workflows.trimul_autotuned import TriangularMultiplication
        _mem_reset()
        aff = _rank_gpu_affinity(ctx)   # RAISES on a rank->GPU collision (silently-wrong-timings guard)
        dt = torch.bfloat16
        w = _weights(ctx)
        N, D, B = int(ctx.N), _D(ctx), int(ctx.B)
        cp = _cp(ctx)
        xl = _local_block(ctx, dt)
        has_mask = bool(ctx.cfg.get("has_mask", False))
        if cp == 1:
            # cp=1 PRODUCTION path (the Phase-A fallback, fused_trimul_cp.py:213): a PLAIN local tensor
            # in -> `_is_single_device` -> `_trimul_cp1` -> trimul_autotuned, touching NO PeMap, NO
            # manager, NO nvshmem. Deliberately NOT wrapped in a DTensor: the whole point of the cp=1
            # denominator is that it needs no mesh, and `_fused_mesh` would demand a live device_mesh that
            # --single-device mode does not have. A plain tensor in => a plain tensor out, hence the
            # `to_local` branch below.
            x_in = xl
            mask_in = torch.ones(B, N, N, device=xl.device, dtype=dt) if has_mask else None
            fmesh = None  # cp=1 needs no mesh; the ctor records it and forward() never dispatches on it
            path = "TriangularMultiplication(cp=1 fallback, plain tensor)"
        else:
            _pm, fmesh, fpl, _ib, _jb = _fused_mesh(ctx)
            x_in = _dtensor_from_local(xl, fmesh, fpl, (B, N, N, D))
            mask_in = None
            if has_mask:
                ml = torch.ones(B, int(xl.shape[1]), int(xl.shape[2]), device=xl.device, dtype=dt)
                mask_in = _dtensor_from_local(ml, fmesh, tuple(fpl), (B, N, N))  # mask placements == x's
            path = "TriangularMultiplication(DTensor API)"
        # One indirection applied to BOTH paths so cp=1 and cp>=2 pay the identical timed-call overhead
        # (a cp=1 output is a plain Tensor and has no .to_local()).
        call = (lambda m, x, mk: m(x, mk)) if cp == 1 else (lambda m, x, mk: m(x, mk).to_local())
        h = {"mod": None, "x_dt": x_in, "mask_dt": mask_in, "call": call}
        try:
            mod = TriangularMultiplication(
                trimul_module_from_weights(w), direction, fmesh,
                (None if cp == 1 else ctx.dm), dtype=dt,
            )
            call(mod, x_in, mask_in)  # WARMUP: JIT compile + symmetric recv alloc + DTensor plumbing
            torch.cuda.synchronize()  # a compile/alloc OOM/fault is caught + tagged at BUILD, not run
            _mem_snap({"path": path, **aff, **_selected_variant(mod)})
            h["mod"] = mod
            return h
        except Exception as e:
            _teardown_fusedcp(h)
            _reraise_as_oom_if_alloc(e)
    return _build


def _run_fusedcp(h):
    # the shipped API forward (TIMED). cp>=2: from_local(x_dt) → to_local → TriMulAutotuned → from_local →
    # to_local. cp==1: the plain-tensor fallback → trimul_autotuned (no DTensor, no nvshmem).
    h["call"](h["mod"], h["x_dt"], h["mask_dt"])


def _teardown_fusedcp(h):
    import torch
    mod = h.get("mod")
    if mod is not None:
        try:
            mod.free()   # free the cached TriMulAutotuned instances' symmetric recvs (no leak)
        except Exception:
            pass
    # MEASURED LEAK (cp=1 N=4096): freeing only `mod` left the INPUT tensors alive in the
    # handle, so the next target in the same process started with them still resident and OOM'd
    # ("Tried to allocate 8.00 GiB ... this process has 72.63 GiB in use"). Harmless at cp>=2 where the
    # local shard is O(N^2/cp), but at cp=1 the "shard" IS the full (B,N,N,D) tensor — 8.6 GB at
    # N=4096/D=256 — so the second target could never fit. Drop every ref, then release the blocks.
    h.clear()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


# ──────────────────────────────────────────────── DTensor baseline (role=baseline) ─────────────────────
def _build_dtensor(direction):
    def _build(ctx):
        import torch
        from torch.distributed.tensor import Shard
        from benchmark.distributed.t2_0_dtensor_baseline import distributed_trimul_fwd
        _mem_reset()
        dt = torch.bfloat16
        w = _weights(ctx)
        cp0, cp1 = int(ctx.cp0), int(ctx.cp1)
        N, D, B = int(ctx.N), _D(ctx), int(ctx.B)
        dmesh = _dtensor_mesh(cp0, cp1)
        token_pl = [Shard(0), Shard(1), Shard(2)]
        xl = _local_block(ctx, dt)   # (b): native local shard -> from_local, NO O(N²) global build
        h = {"x_dt": None, "w": w, "dir": direction, "dt": dt, "fwd": distributed_trimul_fwd, "xl": xl}
        try:
            x_dt = _dtensor_from_local(xl, dmesh, token_pl, (B, N, N, D))
            distributed_trimul_fwd(x_dt, None, w, direction, dt).to_local()  # warmup (sharding-prop spec)
            torch.cuda.synchronize()
            _mem_snap({"path": "t2_0_dtensor_baseline(redistribute A2A)"})
            h["x_dt"] = x_dt
            return h
        except Exception as e:
            _teardown_baseline(h)
            _reraise_as_oom_if_alloc(e)
    return _build


def _run_dtensor(h):
    h["fwd"](h["x_dt"], None, h["w"], h["dir"], h["dt"]).to_local()   # DTensor redistribute A2A (TIMED)


# ──────────────────────────────────────────────────── reference CP baseline (role=baseline) ──────────────────
def _dtensor_baseline_ring_reducescatter_local_fn_from_local(module, bmesh, bpl, xl, global_shape, dt):
    """Thin (b)-path wrapper mirroring trimul_dtensor_baseline.make_trimul_dtensor_baseline_local_fn but seeding the DTensor
    from THIS rank's LOCAL shard (from_local) instead of distribute_tensor(x_global) — so NO O(N²) global is
    built (a reference-baseline OOM must reflect its INTERNAL ring/transpose memory, not an input-build artifact; module
    docstring INPUT PATH). Does NOT touch the vendored reference code — reuses the module/mesh/placements
    build_trimul_dtensor_baseline returns. bf16 autocast + local-shard out == the apples-to-apples timing path."""
    import torch
    from torch.distributed.tensor import DTensor
    B, N = int(global_shape[0]), int(global_shape[1])
    x_dt = _dtensor_from_local(xl, bmesh, bpl, global_shape)
    # mask (B,N,N) with z's placements (bpl); the local mask shard tracks xl's token extents (same shard axes).
    mask_local = torch.ones(B, int(xl.shape[1]), int(xl.shape[2]), dtype=dt, device=xl.device)
    mask_dt = _dtensor_from_local(mask_local, bmesh, tuple(bpl), (B, N, N))

    def fn():
        with torch.autocast("cuda", dtype=dt, enabled=dt in (torch.bfloat16, torch.float16)):
            out = module(x_dt, mask_dt)
        return out.to_local() if isinstance(out, DTensor) else out
    return fn


def _build_dtensor_baseline_ring_reducescatter(direction):
    def _build(ctx):
        import torch
        from benchmark.distributed.trimul_dtensor_baseline import build_trimul_dtensor_baseline
        _mem_reset()
        dt = torch.bfloat16
        w = _weights(ctx)
        N, D, B = int(ctx.N), _D(ctx), int(ctx.B)
        xl = _local_block(ctx, dt)   # (b): native local shard -> from_local, NO O(N²) global build
        h = {"fn": None, "keep": None, "xl": xl}
        try:
            module, bmesh, bpl, _n = build_trimul_dtensor_baseline(ctx.dm, D=D, direction=direction, weights=w, dt=dt)
            fn = _dtensor_baseline_ring_reducescatter_local_fn_from_local(module, bmesh, bpl, xl, (B, N, N, D), dt)
            fn()   # warmup (burn the reference cold sharding-prop spec)
            torch.cuda.synchronize()
            _mem_snap({"path": "reference CP native ring (always mask-ON)"})
            h["fn"], h["keep"] = fn, (module, bmesh, bpl)   # keep refs alive (fn closes over them)
            return h
        except Exception as e:
            _teardown_baseline(h)
            _reraise_as_oom_if_alloc(e)
    return _build


def _run_dtensor_baseline_ring_reducescatter(h):
    h["fn"]()   # reference native ring forward, local-shard out, bf16 autocast (TIMED; apples-to-apples)


def _teardown_baseline(h):
    """DTensor/reference CP baselines hold NO nvshmem symmetric memory (torch/NCCL only) -> just drop refs + free
    the torch caching-allocator blocks so the next per-cell process (or a multi-target smoke cell) starts
    frugal."""
    import torch
    try:
        h.clear()
    except Exception:
        pass
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


# ──────────────────────── cp=1 SINGLE-DEVICE target (the §6/§7 scaling denominator t(1)) ───────────────
# Strong- AND weak-scaling both quote `speedup = t(1)/t(cp)` and `E = t(1)/(cp·t(cp))`, so t(1) is the most
# load-bearing number in both tables. It is measured HERE, by the SAME harness/driver/bench_utils path as
# every cp>=2 cell (identical warmup, rounds, event-window timer, JSON schema) — a t(1) harvested by some
# other script would make every speedup a cross-harness comparison.
#
# The timed call is the SHIPPED single-device entry `trimul_autotuned(select="heuristic")` on the FULL
# (B,N,N,D) tensor: no mesh, no DTensor wrap, no nvshmem. That is exactly what the Phase-A cp=1 fallback in
# fused_trimul_cp.py routes to, so this target measures the same work whether or not that fallback has
# landed — which is why it is a SEPARATE target rather than "run trimul_fusedcp at cp=1". (Running
# trimul_fusedcp at cp=1 as well is still useful and supported: it cross-checks that the fallback is
# transparent, and it prices the DTensor from_local/to_local wrap that the cp>=2 cells also pay.)
#
# K = N_token by construction (CLAUDE.md HARD RULE): x is the SQUARE (B,N,N,D) token tensor, so the back
# einsum contracts over k = N_token. No operand builder is imported from tests/ (the weights come from
# `_weights`, which is the same replicated weight dict every other target in this file uses).
def _single_supports(ctx):
    """cp==1 ONLY. Guarded so this target can never be selected into a cp>=2 cell and silently report a
    single-device time as if it were a sharded one. Deterministic on every rank."""
    if _cp(ctx) != 1:
        return f"trimul_single is the cp=1 scaling denominator only (got cp={_cp(ctx)})"
    return _supports(ctx)


def _heuristic_label(N, D, direction, has_mask, B, device):
    """The combo label the shipped heuristic picked, for the tables' 'config selected' column (plan §6).
    Pure host-side formula (no timing, no memory query); best-effort — a label is a report column, never a
    reason to fail a cell."""
    try:
        from fold_cp_ops.workflows.trimul_autotune import _trimul_heuristic_config
        cfg = _trimul_heuristic_config(int(N), int(D), direction, bool(has_mask), B=int(B), device=device)
    except Exception:
        return {}
    for attr in ("name", "label", "combo"):
        v = getattr(cfg, attr, None)
        if isinstance(v, str):
            return {"combo": v}
    return {"combo": str(cfg)[:80]}


def _build_single(direction):
    def _build(ctx):
        import torch
        from fold_cp_ops.workflows.trimul_autotune import trimul_autotuned
        _mem_reset()
        dt = torch.bfloat16
        w = _weights(ctx)
        N, D, B = int(ctx.N), _D(ctx), int(ctx.B)
        has_mask = bool(ctx.cfg.get("has_mask", False))
        # the FULL square (B,N,N,D) token tensor — at cp=1 the "local shard" IS the global (no sharding)
        g = torch.Generator(device="cpu").manual_seed(_SEED + int(ctx.rank))
        x = torch.randn(B, N, N, D, generator=g, dtype=torch.float32).to(dt).to(ctx.dm.device)
        mask = torch.ones(B, N, N, device=x.device, dtype=dt) if has_mask else None
        h = {"x": x, "mask": mask, "w": w, "dir": direction, "fn": trimul_autotuned}
        try:
            trimul_autotuned(x, **w, mask=mask, direction=direction, select="heuristic")   # WARMUP (JIT)
            torch.cuda.synchronize()   # a compile/alloc OOM is caught + tagged at BUILD, not at run
            _mem_snap({"path": "trimul_autotuned(select=heuristic)",
                       **_heuristic_label(N, D, direction, has_mask, B, x.device)})
            return h
        except Exception as e:
            _teardown_baseline(h)
            _reraise_as_oom_if_alloc(e)
    return _build


def _run_single(h):
    h["fn"](h["x"], **h["w"], mask=h["mask"], direction=h["dir"], select="heuristic")   # TIMED; no alloc


# ───────────────────────────────────────────────────────────────── registry ──────────────────────────
# (base_name, role, build_factory, run, teardown, configs, comm_bytes, supports)
_SPEC = [
    ("trimul_fused", "target", _build_fused, _run_fused, _teardown_fused, _configs, None, _supports),
    ("trimul_fusedcp", "target", _build_fusedcp, _run_fusedcp, _teardown_fusedcp, _configs, None, _supports),
    ("trimul_dtensor", "baseline", _build_dtensor, _run_dtensor, _teardown_baseline, None, _comm_bytes,
     _supports),
    ("trimul_dtensor_baseline_ring_reducescatter", "baseline", _build_dtensor_baseline_ring_reducescatter, _run_dtensor_baseline_ring_reducescatter, _teardown_baseline, None, _comm_bytes,
     _dtensor_baseline_ring_reducescatter_supports),
    # cp=1 scaling denominator (role=target; `supports` gates it to cp==1). Same _configs mask axis as the
    # fused targets so the mask-ON / mask-OFF t(1) each pair with their own cp>=2 column.
    ("trimul_single", "target", _build_single, _run_single, _teardown_baseline, _configs, None,
     _single_supports),
]
_DIRS = (("out", "outgoing"), ("in", "incoming"))


def _factory():
    ts = []
    for base, role, bfac, run, tdn, cfgs, cb, sup in _SPEC:
        for suffix, direction in _DIRS:
            # cell_meta is attached UNIFORMLY: the §6/§7 scaling columns (local shard extents, per-device
            # peak memory, selected variant/combo) are wanted for every target and baseline alike, so the
            # strong/weak tables can compare footprint as well as time.
            ts.append(BenchTarget(f"{base}_{suffix}", build=bfac(direction), run=run, role=role,
                                  supports=sup, teardown=tdn, configs=cfgs, comm_bytes=cb,
                                  cell_meta=_cell_meta))
    return ts


# each direction-suffixed name registers as its own module name (so --targets trimul_fused_out works) + a
# bundle `trimul_e2e` that yields all six (mirror front_a2a's per-name + bundle registration).
for _t in _factory():
    register(_t.name, (lambda nm: (lambda: [t for t in _factory() if t.name == nm]))(_t.name))
register("trimul_e2e", _factory)

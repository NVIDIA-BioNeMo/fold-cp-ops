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

"""Build + drive the reference CP TriMul baseline -- the denominator of every speedup claim here.

The baseline is `trimul_dtensor_reference`'s CP module run AS-IS: this file only picks which of the two
native pipelines matches the input's sharding, seeds the weights, and hands back a callable. No
extra glue runs inside the timed region, and no reshard is ever inserted.

**Why the dispatch is on the SHARDING and not on a caller flag.** The earlier home-grown baseline
called DTensor ``.redistribute()`` three times, which is a feature-axis all-to-all even for a plain
1-D row shard -- so it measured a reshard the real workflow never performs and flattered the fused
path for the wrong reason. Here the reference module that NATIVELY matches the input's sharding
dimensionality is selected, and its ``.forward(z, mask)`` runs verbatim:

* **1-D** (one token axis sharded -- mesh ``(dp, cp)``, placements ``(Shard(0), Shard(1))``, local
  ``(B, N/cp, N, D)`` row slab): `TriangularMultiplication{Outgoing,Incoming}1D`. Forward comm is a
  1-D ring (outgoing) or a tiled reduce-scatter (incoming). **No all-to-all.**
* **2-D** (both token axes sharded -- mesh ``(dp, cp0, cp1)``, placements ``(Shard(0), Shard(1),
  Shard(2))``, local ``(B, N/cp0, N/cp1, D)`` block): `TriangularMultiplication{Outgoing,
  Incoming}2D` over a `Ring2DComm`. Forward comm is a 2-D ring. Requires ``cp0 == cp1``.

These are SEPARATE NATIVE pipelines. The 2-D algorithm is never run on a degenerate ``(1, cp, 1)``
mesh -- that exact pattern, the 2-D feature-reshard applied to a 1-D situation, was the unfair bug
-- and one sharding is never morphed into the other.

**Weight sharing.** `LinearParamsReplicated` / `LayerNormParamsReplicated` SNAPSHOT
``layer.weight.data`` at construction, so :func:`build_trimul_dtensor_baseline` OVERWRITES the
single-device layer's parameters with the same ``make_weights`` dict the fp32 oracle and
`TriMulAutotuned` use, and only THEN wraps it. Wrapping first would silently pin whatever the
constructor left behind. With the weights shared, the reference computes the same twelve ops as
`fold_cp_ops.workflows.trimul_autotune.trimul_ref` -- norm_in, the g_out gate off the normalized
input, g_in/p_in, mask * sigmoid(g), chunk into a/b, einsum, norm_out, p_out, sigmoid gate.

The manager accessors this file reads (``manager.group['cp']``, ``manager.subgroups['cp']``,
``manager.device_mesh``, ``manager.world_size``) are the ones `DistributedManager` exposes, so the
baseline is built directly from whichever live manager the caller passes.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch.distributed.tensor import DTensor, Shard, distribute_tensor


# --------------------------------------------------------------------------- #
# Weight seeding: copy a make_weights dict into the single-device reference layer.
# --------------------------------------------------------------------------- #
def _seed_reference_layer(layer, weights: dict, dt: torch.dtype, device) -> None:
    """Overwrite a `TriangularMultiplication{Outgoing,Incoming}` layer's eight parameters in place.

    Purpose
    -------
    Make the reference baseline compute with the SAME weights as the fp32 oracle and
    `TriMulAutotuned`, so a comparison between them is about the implementation and not about the
    numbers each happened to initialize.

    Semantics
    ---------
    Copies under `torch.no_grad`, in place, into the eight parameters the layer owns. It must run
    BEFORE the CP wrapper is constructed: `LinearParamsReplicated` / `LayerNormParamsReplicated`
    SNAPSHOT ``layer.weight.data`` at construction, so seeding afterwards would change the layer
    and leave the wrapper holding the constructor's defaults -- silently, with no shape error.

    Input requirements
    ------------------
    layer :
        A `TriangularMultiplication` subclass instance, already on ``device`` at ``dt``. Its six
        submodules must be present under the canonical names; a module missing one raises
        `AttributeError` here rather than producing a half-seeded layer.
    weights : dict
        The shared ``correctness_harness.make_weights`` dict: ``norm_in_w``/``norm_in_b`` ``(D,)``,
        ``p_in_w``/``g_in_w`` ``(2D, D)``, ``norm_out_w``/``norm_out_b`` ``(D,)``,
        ``p_out_w``/``g_out_w`` ``(D, D)``. A shape mismatch raises from `Tensor.copy_`. The four
        projection-bias keys must be absent or ``None`` -- see Raises.
    dt : torch.dtype
        Target dtype; each source tensor is cast before the copy.
    device :
        Target device; each source tensor is moved before the copy.

    Raises
    ------
    NotImplementedError
        If ``weights`` carries any non-None projection bias. The reference layer builds its
        projections ``bias=False``, matching this repository's fused TriMul contract, so a bias in
        the shared dict would be applied by the oracle and silently dropped here -- a numeric
        divergence with no exception anywhere else. Refusing is the only way that stays visible.
    """
    with torch.no_grad():
        layer.norm_in.weight.copy_(weights["norm_in_w"].to(dt).to(device))
        layer.norm_in.bias.copy_(weights["norm_in_b"].to(dt).to(device))
        layer.p_in.weight.copy_(weights["p_in_w"].to(dt).to(device))
        layer.g_in.weight.copy_(weights["g_in_w"].to(dt).to(device))
        layer.norm_out.weight.copy_(weights["norm_out_w"].to(dt).to(device))
        layer.norm_out.bias.copy_(weights["norm_out_b"].to(dt).to(device))
        layer.p_out.weight.copy_(weights["p_out_w"].to(dt).to(device))
        layer.g_out.weight.copy_(weights["g_out_w"].to(dt).to(device))
    for k in ("p_in_b", "g_in_b", "p_out_b", "g_out_b"):
        if weights.get(k) is not None:
            raise NotImplementedError(
                f"the reference TriMul layer builds bias=False projections; shared "
                f"weights set {k}. "
                "The fair baseline (and TriMulAutotuned) use bias-free projections."
            )


# --------------------------------------------------------------------------- #
# Build the reference CP baseline for the given sharding dimensionality.
# --------------------------------------------------------------------------- #
def build_trimul_dtensor_baseline(manager, *, D: int, direction: str, weights: dict, dt: torch.dtype):
    """Build the native reference CP TriMul module matching the live manager's cp topology.

    Purpose
    -------
    The single entry point that turns "a live `DistributedManager` plus a weight dict" into the
    baseline module, its mesh, and the placements its input must carry.

    Semantics
    ---------
    Dispatches on the cp-axis structure of ``manager``, NOT on any caller flag -- so a call site
    cannot accidentally ask for the 2-D algorithm on a 1-D shard, which is exactly the unfair
    comparison this baseline replaced.

    * **1 cp axis** -> the 1-D module on a ``(dp, cp)`` mesh with placements
      ``(Shard(0), Shard(1))``.
    * **2 cp axes** -> the 2-D module + `Ring2DComm` on a ``(dp, cp0, cp1)`` mesh with placements
      ``(Shard(0), Shard(1), Shard(2))``.

    A size-1 cp axis is COLLAPSED before that count is taken: a ``cp = N*1`` mesh has an axis that
    splits nothing, so it is genuinely 1-D. Without the collapse it would report two axes, route to
    `Ring2DComm`, and raise ``group_layout (8,1) not square`` on what is a plain row shard. The
    fused side treats ``cp1 == 1`` as 1-D for the same reason, so this keeps the head-to-head
    running on the same topology.

    `init_device_mesh` is a COLLECTIVE. Both branches call it, so every rank must reach this
    function in lockstep; a rank that skips the build hangs its peers here rather than at the
    forward.

    Input requirements
    ------------------
    manager :
        A live `DistributedManager` with an initialized process group. Read for ``.device``,
        ``.group['cp']``, ``.subgroups['cp']``, ``.has_subgroups`` and ``.world_size``.
    D : int
        Feature width. Must match the weight dict's; a mismatch raises from the seeding copy.
    direction : str
        ``"outgoing"`` or ``"incoming"``. Any other value silently selects the incoming layer in
        the class pick but is rejected by the explicit check below.
    weights : dict
        As `_seed_reference_layer`, including its no-projection-bias requirement.
    dt : torch.dtype
        The dtype the module's parameters are held at.

    Returns
    -------
    tuple
        ``(module, mesh, placements, n_cp_axes)``. ``placements`` is what `distribute_tensor` /
        `DTensor.from_local` must be given for the module's input; ``n_cp_axes`` is 1 or 2 and is
        what a caller asserts against its own view of the session topology.

    Raises
    ------
    ValueError
        On an unrecognised ``direction``, or on a cp topology that is neither 1-D nor 2-D.
    """
    import torch.distributed as dist

    from benchmark.distributed.trimul_dtensor_reference import (
        TriangularMultiplicationIncoming,
        TriangularMultiplicationIncoming1D,
        TriangularMultiplicationOutgoing,
        TriangularMultiplicationOutgoing1D,
    )

    if direction not in ("outgoing", "incoming"):
        raise ValueError(f"direction must be 'outgoing' or 'incoming'; got {direction!r}")

    device = manager.device
    has_subgroups = bool(getattr(manager, "has_subgroups", False))
    cp_subgroups = manager.subgroups.get("cp") if has_subgroups else None
    if cp_subgroups is not None:
        _cp_sizes = tuple(dist.get_world_size(g) for g in cp_subgroups)
        _real_axes = [s for s in _cp_sizes if s > 1]  # drop size-1 (non-splitting) axes
        n_cp_axes = len(_real_axes) if _real_axes else 1
    else:
        n_cp_axes = 1

    out_dir = direction == "outgoing"
    cls = TriangularMultiplicationOutgoing if out_dir else TriangularMultiplicationIncoming
    layer = cls(D).to(dtype=dt, device=device)
    _seed_reference_layer(layer, weights, dt, device)

    if n_cp_axes == 1:
        # 1-D native contract: mesh (dp, cp) [2-D], placements (Shard(0), Shard(1)), local row-slab
        # (B, N/cp, N, D). The fold_cp_ops session mesh under CPO_DIST_MESH=cp=N is the 1-D (cp,)
        # mesh, but the 1-D module needs the (dp, cp) 2-D mesh; build it COLLECTIVELY here with
        # dp=1 and take cp_group from ITS cp dim, so mesh and group are mutually consistent. This
        # is the genuine 1-D contract, NOT the degenerate (1, cp, 1) 3-D mesh.
        from torch.distributed.device_mesh import init_device_mesh

        # cp size from the manager's cp group (== world for a cp-only session).
        cp = (dist.get_world_size(manager.group["cp"]) if "cp" in manager.group
              else manager.world_size)
        mesh = init_device_mesh("cuda", (1, cp), mesh_dim_names=("dp", "cp"))
        cp_group = mesh["cp"].get_group()
        cp_cls = (TriangularMultiplicationOutgoing1D if out_dir
                  else TriangularMultiplicationIncoming1D)
        return cp_cls(layer, mesh, cp_group), mesh, (Shard(0), Shard(1)), 1

    if n_cp_axes == 2:
        # 2-D native contract: mesh (dp, cp0, cp1) [3-D], placements (Shard(0),Shard(1),Shard(2)),
        # local ij-block (B, N/cp0, N/cp1, D). The fold_cp_ops session under CPO_DIST_MESH=cp=2*2
        # gives a 2-D (cp0,cp1) subgroup mesh with no dp, but the 2-D module asserts a length-3
        # placement on a (dp,cp0,cp1) mesh. Build that 3-D mesh COLLECTIVELY with dp=1 and derive
        # the Ring2DComm groups (flat 2-D cp group + a column subgroup) and the LayoutMap FROM IT,
        # so mesh, groups and layout are mutually consistent. NOT a 1-D->2-D reshard.
        from torch.distributed.device_mesh import init_device_mesh

        from benchmark.distributed.trimul_dtensor_reference import (
            Ring2DComm,
            TriangularMultiplicationIncoming2D,
            TriangularMultiplicationOutgoing2D,
        )
        from fold_cp_ops.distributed.layout_map import LayoutMap

        cp0, cp1 = (int(s) for s in cp_subgroups_sizes(manager))
        mesh = init_device_mesh("cuda", (1, cp0, cp1), mesh_dim_names=("dp", "cp0", "cp1"))
        # flat 2-D cp grid group (row-major over cp0,cp1) + a column (cp0-axis) subgroup.
        group_2d = mesh["cp0", "cp1"]._flatten("cp_flat").get_group()
        group_col = mesh["cp0"].get_group()  # ranks sharing a cp1 coord (the column group)
        # LayoutMap: row-major (cp0, cp1) -> flat index (strides (cp1, 1)); matches the flatten.
        ring_comm = Ring2DComm(group_2d, group_col, LayoutMap(strides=(cp1, 1), shape=(cp0, cp1)))
        cp_cls = (TriangularMultiplicationOutgoing2D if out_dir
                  else TriangularMultiplicationIncoming2D)
        return cp_cls(layer, mesh, ring_comm), mesh, (Shard(0), Shard(1), Shard(2)), 2

    raise ValueError(
        f"the reference CP baseline supports 1-D or 2-D cp token sharding; got {n_cp_axes} cp axes."
    )


def cp_subgroups_sizes(manager):
    """Report the per-axis cp sizes ``(cp0, cp1)`` of a live manager's cp subgroup mesh.

    Purpose
    -------
    Turn the manager's list of cp subgroups into the plain integer shape the 2-D mesh build needs.

    Semantics
    ---------
    One `dist.get_world_size` per subgroup, in axis order. Reads only -- no collective is issued.

    Input requirements
    ------------------
    manager :
        Must expose ``.subgroups['cp']`` as a sequence of process groups, i.e. must have been
        initialized with a MULTI-AXIS cp mesh. On a 1-D session ``subgroups['cp']`` is absent and
        this raises `TypeError` on the `None`; call it only from the 2-D branch.

    Returns
    -------
    tuple[int, ...]
        The cp axis sizes, in mesh-axis order.
    """
    import torch.distributed as dist

    subs = manager.subgroups.get("cp")
    return tuple(dist.get_world_size(g) for g in subs)


def run_trimul_dtensor_baseline_fwd(
    module, mesh, placements, x_global: torch.Tensor, dt: torch.dtype,
    *, mask_global: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Distribute a global input, run the baseline forward AS-IS, and all-gather the result.

    Purpose
    -------
    The CORRECTNESS path: produce the full ``(B, N, N, D)`` output so it can be compared against a
    single-GPU fp32 oracle.

    Semantics
    ---------
    ``x_global`` is distributed onto the module's own native ``mesh``/``placements`` -- a row slab
    or an ij block, with NO reshard -- and then ``module(z, mask)`` runs verbatim. No autocast is
    applied here: the comparison is against an fp32 oracle, and running the baseline at its natural
    precision keeps the measured gap about the algorithm rather than about rounding. NOTE that
    the 2-D module takes its matmul precision FROM autocast, so without one it runs below the
    precision it is written for, while the 1-D module upcasts itself -- expect a wider 2-D gap. The timing
    path (`make_trimul_dtensor_baseline_local_fn`) makes the opposite choice, deliberately.

    Input requirements
    ------------------
    module, mesh, placements :
        As returned by `build_trimul_dtensor_baseline`. Passing a mesh the module was not built on
        raises inside DTensor, not here.
    x_global : Tensor
        ``(B, N, N, D)``, IDENTICAL on every rank -- `distribute_tensor` slices rank-locally from
        each rank's own copy, so a rank whose copy differs contributes a shard nobody detects.
        Every sharded axis extent must be divisible by its mesh axis size.
    dt : torch.dtype
        Cast applied before distribution.
    mask_global : Tensor or None
        ``(B, N, N)``. ``None`` builds an all-ones mask, because the CP forward requires a mask
        DTensor with the same first three extents and the same placements as ``z``.

    Returns
    -------
    Tensor
        The all-gathered ``(B, N, N, D)`` output. All-gather is a COLLECTIVE, so every rank must
        call this.
    """
    B, N, _, D = x_global.shape
    x_dt = distribute_tensor(x_global.to(dt).to(mesh.device_type), mesh, placements)
    if mask_global is None:
        mask_global = torch.ones(B, N, N, dtype=dt, device=x_global.device)
    # mask placements = z's placements; Shard(0/1/2) all index token axes (< 3), valid for the mask.
    mask_dt = distribute_tensor(mask_global.to(dt).to(mesh.device_type), mesh, tuple(placements))
    out_dt = module(x_dt, mask_dt)  # reference CP forward AS-IS (z, mask)
    if isinstance(out_dt, DTensor):
        return out_dt.full_tensor()
    return out_dt


def make_trimul_dtensor_baseline_local_fn(
    module, mesh, placements, x_global: torch.Tensor, dt: torch.dtype,
    *, mask_global: Optional[torch.Tensor] = None,
):
    """Return a no-arg callable that times ONLY the baseline forward, local-shard out.

    Purpose
    -------
    The apples-to-apples TIMING path: the same granularity the fused path is timed at, so neither
    side pays for a final all-gather the other does not.

    Semantics
    ---------
    The input and mask are distributed ONCE at build time -- setup, not per-call cost -- and the
    returned ``fn()`` runs ``module(z, mask)`` and returns ``.to_local()``. The forward's OWN comm
    (ring / reduce-scatter / `Ring2DComm`) is inside the timed region; only the harness all-gather
    is not.

    ``fn()`` runs under bf16-mixed autocast when ``dt`` is bf16/fp16, and that is a fairness
    decision rather than an implementation detail. The CP module upcasts its operands to
    ``promote_types(dt, fp32)`` to preserve the single-device precision at the einsum -> norm_out
    handoff; autocast downcasts the matmul inputs back to bf16, so the contraction runs on bf16
    TENSOR CORES the way it would in production. WITHOUT autocast that matmul runs fp32 on CUDA
    cores, and the resulting speedup would fold a precision gap into what is supposed to be an
    implementation comparison. The high-precision path stays in `run_trimul_dtensor_baseline_fwd`, where
    the subject is correctness.

    Input requirements
    ------------------
    module, mesh, placements :
        As returned by `build_trimul_dtensor_baseline`.
    x_global : Tensor
        ``(B, N, N, D)``, identical on every rank. **Note the footprint**: every rank materializes
        the full global, so the largest N reachable this way is cp-independent and SMALLER than the
        workflow's. Use `make_trimul_dtensor_baseline_local_fn_sharded` when the subject is peak memory
        or the largest reachable N.
    dt : torch.dtype
        Distribution dtype, and the autocast dtype when it is bf16/fp16.
    mask_global : Tensor or None
        ``(B, N, N)``; ``None`` builds an all-ones mask.

    Returns
    -------
    Callable[[], Tensor]
        A no-arg callable returning this rank's local output shard. It issues COLLECTIVES, so every
        rank must call it the same number of times.
    """
    B, N, _, D = x_global.shape
    x_dt = distribute_tensor(x_global.to(dt).to(mesh.device_type), mesh, placements)
    if mask_global is None:
        mask_global = torch.ones(B, N, N, dtype=dt, device=x_global.device)
    mask_dt = distribute_tensor(mask_global.to(dt).to(mesh.device_type), mesh, tuple(placements))

    def fn():
        """Run one timed baseline forward and return this rank's local output shard."""
        with torch.autocast("cuda", dtype=dt, enabled=dt in (torch.bfloat16, torch.float16)):
            out = module(x_dt, mask_dt)
        return out.to_local() if isinstance(out, DTensor) else out

    return fn


def make_trimul_dtensor_baseline_local_fn_sharded(module, mesh, placements, x_local, dt):
    """Return a no-arg timing callable seeded from THIS rank's local shard -- never a global.

    Purpose
    -------
    The perf/memory path for large N: measure the baseline at the per-rank footprint the real
    workflow has, instead of the inflated one a global-per-rank input creates.

    Semantics
    ---------
    Wraps ``x_local`` with `DTensor.from_local`, which trusts the caller's shard and reconstructs
    the global metadata with ZERO communication. Nothing ever allocates the global ``(B, N, N, D)``,
    so peak memory -- and therefore the largest N a rank can hold -- is the true one. The
    `distribute_tensor` path above puts a cp-independent global on every rank, which understates
    max N and flattens the cp16-vs-cp8 gap.

    Because `from_local` trusts the shard rather than deriving it, the data is arbitrary and this
    path is **NOT** a correctness path. Use `run_trimul_dtensor_baseline_fwd` for that.

    Input requirements
    ------------------
    module, mesh, placements :
        As returned by `build_trimul_dtensor_baseline`.
    x_local : Tensor
        This rank's local block, whose shape must MATCH the native shard exactly -- 1-D:
        ``(B, N//cp, N, D)`` under ``(Shard(0), Shard(1))``; 2-D: ``(B, N//cp0, N//cp1, D)`` under
        ``(Shard(0), Shard(1), Shard(2))``. A wrong local shape is NOT detected here: `from_local`
        infers a global from it, so the run proceeds against a global nobody asked for.
    dt : torch.dtype
        Cast applied to ``x_local``, and the autocast dtype when bf16/fp16.

    Returns
    -------
    Callable[[], Tensor]
        A no-arg callable returning this rank's local output shard; issues COLLECTIVES, so every
        rank must call it the same number of times.
    """
    from torch.distributed.tensor import DTensor

    x_dt = DTensor.from_local(x_local.to(dt), mesh, placements)
    mask_local = torch.ones(*x_local.shape[:3], dtype=dt, device=x_local.device)
    mask_dt = DTensor.from_local(mask_local, mesh, tuple(placements))

    def fn():
        """Run one timed baseline forward and return this rank's local output shard."""
        with torch.autocast("cuda", dtype=dt, enabled=dt in (torch.bfloat16, torch.float16)):
            out = module(x_dt, mask_dt)
        return out.to_local() if isinstance(out, DTensor) else out

    return fn

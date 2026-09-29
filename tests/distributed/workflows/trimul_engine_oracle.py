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

"""R0 — capture the engine-argument oracle for the DTensor API refactor.

Records, for each declared cell, BOTH halves of what the plan's §0 fence is checked against:

  * the EXACT (args, kwargs) `TriMulAutotuned.__init__` receives, captured by wrapping it; and
  * the state it RESOLVES to, read back off the constructed instance.

Read back, never re-derived. Re-calling `_resolve_front_config` / `autotune_front_config` here
would compare two private resolvers that can agree perfectly while the engine used a third value --
the precedence chain in front of them (explicit override > autotune > _baked > heuristic, plus the
`_resolve_front_tile_n` clamp) is exactly where a disagreement would hide.

Sharded by (cp, direction, masked) because engines-per-process is charged against the symmetric
heap (`_symmetric_free` is a no-op; the MemPool recycles by refcount), and ~7 in one process has
been measured to abort a session. Each cell is freed and dropped before the next.
"""
import json
import os
import torch
from torch.distributed.tensor import Shard
from fold_cp_ops.distributed.distributed_manager import DistributedManager
from fold_cp_ops.distributed.pe_map import PeMap


def main() -> None:
    """Capture one shard of the oracle. Reads its cell from the environment; see module doc."""
    DIRECTION = os.environ["R0_DIR"]
    MASKED = os.environ.get("R0_MASKED", "0") == "1"
    D_LIST = [int(x) for x in os.environ.get("R0_D", "128,256,384,512").split(",")]
    N = int(os.environ.get("R0_N", "512"))
    OUT = os.environ["R0_OUT"]

    # R0_MESH: "16" for a flat cp mesh, "4x4" / "2x8" for a factored one. A flat mesh can never
    # exercise `route2_ni` (which needs cp1 > 1) or the 2-D tile->peer unravel, so an oracle captured
    # only on flat meshes is blind to exactly the axes the 2-D drain routes on.
    # NOT `os.environ.get("R0_MESH", str(int(os.environ["WORLD_SIZE"])))`: Python evaluates a
    # default EAGERLY, so that form raised KeyError under bare srun even when R0_MESH was supplied.
    # And bare srun sets SLURM_NTASKS, not WORLD_SIZE -- only torchrun sets the latter.
    _spec = os.environ.get("R0_MESH") or os.environ.get("WORLD_SIZE") or os.environ["SLURM_NTASKS"]
    _cp = tuple(int(x) for x in _spec.split("x")) if "x" in _spec else int(_spec)
    DistributedManager.initialize({"cp": _cp}, device_type="cuda")
    DistributedManager.init_nvshmem()
    dm = DistributedManager()
    # A TUPLE-valued cp group builds TWO meshes: the PARENT (`device_mesh`, flat, shape [prod]) and the
    # factored one (`device_mesh_subgroups`, shape (cp0, cp1)). Reading `device_mesh` unconditionally
    # silently collapses every 2-D spec to a flat mesh -- `cp1` comes back 1, `route2_ni` is never
    # selected, and the captured cells are mislabelled DUPLICATES of the flat run rather than missing.
    # This is the selection `benchmark/.../trimul_e2e.py:361` uses.
    _sub = getattr(dm, "device_mesh_subgroups", None)
    mesh = _sub if (getattr(dm, "has_subgroups", False) and _sub is not None) else dm.device_mesh
    placements = [Shard(i + 1) for i in range(mesh.ndim)]
    pm = PeMap.from_mesh_placements(mesh, placements, distributed_manager=dm)
    import fold_cp_ops.distributed.workflows.trimul_autotuned as T

    dev = torch.device("cuda", torch.cuda.current_device())
    captured = {}
    _orig = T.TriMulAutotuned.__init__

    def _jsonable(v):
        if isinstance(v, (int, float, str, bool)) or v is None:
            return v
        if isinstance(v, (tuple, list)):
            return [_jsonable(x) for x in v]
        if isinstance(v, torch.dtype):
            return str(v)
        if isinstance(v, torch.Tensor):
            return {"__tensor__": [list(v.shape), str(v.dtype)]}
        if isinstance(v, dict):
            return {k: _jsonable(x) for k, x in v.items()}
        return f"<{type(v).__name__}>"

    def wrapper(self, *args, **kwargs):
        # positional order is (pe_map, B, N, D, w, dt); record the SHAPE of w, not its values.
        captured["args"] = {"B": args[1], "N": args[2], "D": args[3],
                            "w_keys": sorted(k for k, v in args[4].items() if v is not None),
                            "dt": str(args[5])}
        captured["kwargs"] = {k: _jsonable(v) for k, v in sorted(kwargs.items())}
        return _orig(self, *args, **kwargs)

    T.TriMulAutotuned.__init__ = wrapper
    g = torch.Generator(device="cpu").manual_seed(20260826)

    def resolved(e):
        """The state the engine RESOLVED to -- read off the instance, never recomputed."""
        sig = getattr(e, "_front_sig", None)
        return {
            "front_cfg": _jsonable(tuple(e._front_cfg)),
            "back_cfg": _jsonable(tuple(e._back_cfg)),
            "hybrid_ib": bool(e.hybrid_ib),
            "has_ib_peers": bool(e._has_ib_peers),
            "route2_ni": bool(e.route2_ni),
            "composite_k": bool(e.composite_k),
            "has_mask": bool(e.has_mask),
            "dynamic": bool(e.dynamic),
            "back_store": e.back_store,
            "use_signal_drain": bool(e._use_signal_drain),
            "drain_obj": type(sig).__name__ if sig is not None else None,
            "front_pad_inner": bool(e.front_pad_inner),
            "front_pad_eager": bool(e.front_pad_eager),
            "geometry": {"cp": int(e.cp), "cp0": int(e.cp0), "cp1": int(e.cp1),
                         "N_i_loc": int(e.N_i_loc), "N_j_loc": int(e.N_j_loc),
                         "M": int(e.M), "D_loc": int(e.D_loc), "L": int(e._back.L)},
            "store_classes": {"front": type(e._front).__name__, "back": type(e._back).__name__,
                              "front_ni": type(e._front_ni).__name__ if e._front_ni is not None else None,
                              "front_comp": type(e._front_comp).__name__ if e._front_comp is not None else None},
        }

    rows = []
    for D in D_LIST:
        if D % int(pm.cp):
            rows.append({"cell": {"cp": int(pm.cp), "mesh": _spec, "direction": DIRECTION, "D": D,
                                  "masked": MASKED, "N": N, "B": 1}, "skipped": f"D={D} not divisible by cp={pm.cp}"})
            continue
        def rn(*sh): return (torch.randn(*sh, generator=g, dtype=torch.float32) * 0.02).to(dev)
        w = dict(norm_in_w=rn(D).float(), norm_in_b=rn(D).float(), p_in_w=rn(2*D, D), g_in_w=rn(2*D, D),
                 norm_out_w=rn(D).float(), norm_out_b=rn(D).float(), p_out_w=rn(D, D), g_out_w=rn(D, D),
                 p_in_b=None, g_in_b=None, p_out_b=None, g_out_b=None)
        # Drive the REAL entry point, not the engine constructor. `trimul_a2a` is what decides
        # `composite_k` / `route2_ni` / `dynamic` / `distributed_manager` and hands them to
        # `TriMulAutotuned.__init__` -- so constructing the engine directly would pin MY kwargs and say
        # nothing about the argument construction R3/R6 actually change. That is the whole subject.
        from torch.distributed.tensor import DTensor
        cp0 = int(pm.cp_axis_sizes[0]); cp1 = int(pm.cp_axis_sizes[1]) if len(pm.cp_axis_sizes) > 1 else 1
        def _rowmaj(sh):
            st = [1] * len(sh)
            for i in range(len(sh) - 2, -1, -1):
                st[i] = st[i + 1] * int(sh[i + 1])
            return tuple(st)
        xl = torch.randn(1, N // cp0, N // cp1, D, generator=g, dtype=torch.float32).to(dev).to(torch.bfloat16)
        gx = (1, N, N, D)
        x = DTensor.from_local(xl.contiguous(), mesh, placements, shape=torch.Size(gx),
                               stride=_rowmaj(gx))
        mk_dt = None
        if MASKED:
            ml = torch.ones(1, N // cp0, N // cp1, device=dev, dtype=torch.bfloat16)
            gm = (1, N, N)
            mk_dt = DTensor.from_local(ml, mesh, placements, shape=torch.Size(gm), stride=_rowmaj(gm))
        captured.clear()
        cache = {}
        T.trimul_a2a(x, w, torch.bfloat16, direction=DIRECTION, mask=mk_dt,
                     distributed_manager=dm, _cache=cache)
        eng = next(iter(cache.values()))
        try:
            rows.append({"cell": {"cp": int(pm.cp), "mesh": _spec, "direction": DIRECTION, "D": D,
                                  "masked": MASKED, "N": N, "B": 1},
                         "init_args": dict(captured), "resolved": resolved(eng)})
        finally:
            for inst in cache.values():
                inst.free()
            cache.clear()
            del eng, w, x, xl, mk_dt
            torch.cuda.empty_cache()

    if dm.rank == 0:
        with open(OUT, "w") as f:
            json.dump(rows, f, indent=1, sort_keys=True)
        print(f"R0WROTE {OUT} cells={len(rows)}")
    DistributedManager.cleanup()


if __name__ == "__main__":
    main()

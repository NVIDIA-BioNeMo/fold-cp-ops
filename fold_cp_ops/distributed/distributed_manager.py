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

"""Distributed state manager: device mesh + process groups + nvshmem bring-up.

**torch owns the nvshmem lifetime; this module only triggers and verifies it.**
``torch.distributed._symmetric_memory`` bootstraps nvshmem LAZILY -- it mints a unique id and
all-gathers it over the process group's own ``Store`` on the FIRST BYTE allocated from the symmetric
allocator. It never FINALIZES it, though (verified: no finalize call anywhere in torch v2.11.0's
sources), which is why :meth:`DistributedManager.cleanup` does. So :meth:`DistributedManager.init_nvshmem` no
longer *performs* an init: it selects the NVSHMEM backend, TRIGGERS torch's bootstrap with a one-byte
collective allocation, and checks the outcome. :meth:`DistributedManager.cleanup` correspondingly has
no finalize to call. That is not tidying for its own sake -- a collective ``nvshmem.core.finalize()``
that had to be ordered against ``destroy_process_group`` was the teardown race, and this removes it
rather than relocating it.

nvshmem4py is still REQUIRED by the package: it provides the in-kernel device API
(``rma``/``amo``/``direct``) and ``library_init``, neither of which torch has. Only the LIFETIME
moved. Here it is optional and used for one thing -- verifying that torch numbered the PEs the way
the PE maps assume.

Ported (surgically) from ``distributed/manager.py`` in the NVIDIA-authored
distributed-CP fork this work originated in. **That fork's files carry NVIDIA's
own SPDX copyright**, so this is a move within one owner's code and not a port of
third-party material -- worth stating because the fork lives inside a
third-party repository and the provenance reads the other way at a glance. For the
distributed-TriMul project (``docs/trimul_nvshmem_design.md`` §3). The public
API (class name, method names, attribute names) matches that fork so that the
sibling bench util (T0.4), the torchrun-pytest harness (T0.2), and the DTensor
adapter (T1.3) can assume the same surface.

Anti-patterns stripped vs that fork (see §1 "AVOID" in the design doc):
  * ``cleanup()`` barrier is now **timeout-bounded** (``CLEANUP_BARRIER_TIMEOUT_S``
    / ``CPO_DIST_CLEANUP_TIMEOUT_S``) and best-effort, so a hung/dead peer
    cannot wedge teardown forever (the fork used an unbounded barrier).
  * No hardcoded ``(Shard(0), Shard(1), Shard(2))`` placement assumptions — this
    class is placement-generic: it builds the mesh + groups + PE maps and never
    inspects DTensor placements. Sharding/placement logic lives in the DTensor
    adapter (T1.3) and the PE-map layout (T1.1).
  * No dependence on ``cute_nvshmem_compile`` or strided-input assumptions.

nvshmem init env is CLUSTER-AWARE (keyed on auto-detected hardware/topology traits, not a cluster
name). ``init_nvshmem`` applies a cluster-agnostic COMMON set + a trait-detected PROFILE, both via
``setdefault`` (a caller can override any single key, or force the whole profile with
``CPO_NVSHMEM_PROFILE``). See ``_NVSHMEM_COMMON_ENV`` / ``_NVSHMEM_PROFILES`` at module top.
  * COMMON (all profiles): ``NVSHMEM_DISABLE_NCCL=1`` (avoid the NCCL bootstrap race),
    ``NVSHMEM_DISABLE_NVLS=1`` (NVLS heap-map crashes here; put needs no multimem),
    ``NVSHMEM_SYMMETRIC_SIZE`` (capped default; override for larger buffers).
  * PROFILE (trait-detected): single-node -> ``single`` (P2P/IPC, no IB); multi-node IB -> ``ib-ibgda``
    (IBGDA + distance-based multi-rail NIC), curated NICs (venue C) or not (venue D ConnectX-7). (#45 dropped
    the venue C ``ib-ibgda-nicmap`` single-rail auto-branch -> explicit-override only; no faster on venue C.)
"""

import logging
import os
from math import prod
from typing import Any, Dict, Optional, OrderedDict, Union
from warnings import warn

import torch

from fold_cp_ops._internal.port_selection import _UNPRIVILEGED_MIN, ephemeral_floor
from fold_cp_ops.distributed.layout_map import LayoutMap, LayoutRightMap

# torch's symmetric-memory allocator OWNS the nvshmem lifetime. Guarded because it is a PRIVATE torch
# API whose signatures have moved between versions: a torch without it must fail at the front door
# with a sentence naming the requirement, not an AttributeError three frames into init_nvshmem().
try:
    import torch.distributed._symmetric_memory as symm_mem

    HAS_SYMM_MEM = True
except ImportError:
    HAS_SYMM_MEM = False

# cuda.core: no longer needed to INIT nvshmem (torch does that), but the CuTe-DSL compile path
# (``cute_compile_helper`` -> ``library_init``) requires a current cuda.core device, and this class is
# where the device identity is known. Optional so a CPU-only / torch-only install still imports.
try:
    import cuda.core

    HAS_CUDA_CORE = True
except ImportError:
    HAS_CUDA_CORE = False

# nvshmem4py: used HERE only to verify torch's PE numbering (see init_nvshmem step 5). The package is
# a hard requirement for the device kernels, but this module no longer needs it to function.
try:
    import nvshmem.bindings

    HAS_NVSHMEM4PY = True
except ImportError:
    HAS_NVSHMEM4PY = False

# grid_group_sizes objects must have (1) .values(), (2) .items().
_GridGroupSizesType = OrderedDict[str, Union[int, tuple[int, ...]]]

# Default symmetric-heap cap + cleanup-barrier timeout (overridable via env).
_DEFAULT_NVSHMEM_SYMMETRIC_SIZE = str(2 * 1024 * 1024 * 1024)  # 2 GiB
CLEANUP_BARRIER_TIMEOUT_S = float(os.environ.get("CPO_DIST_CLEANUP_TIMEOUT_S", "30"))

#: Has nvshmem been brought up in THIS PROCESS? Module-level and never cleared, because the library's
#: lifetime is the PROCESS -- torch never finalizes it at all, and nothing else can un-initialize it
#: earlier. Deliberately NOT `_state["_nvshmem_initialized"]`, which is MANAGER state and is wiped by
#: `cleanup()`: conflating the two loses the fact that the library is still up, so a `cleanup()`
#: called again from `atexit` would skip the one teardown that has to happen before the static
#: destructors run. Two different lifetimes need two different flags.
_NVSHMEM_LIBRARY_UP = False

#: device index -> the ONE ``torch.cuda.MemPool`` this process routes symmetric allocations through.
#: Module-level and NEVER cleared, for the same reason as ``_NVSHMEM_LIBRARY_UP`` above and with a
#: sharper consequence -- though NOT a guarantee, and do not treat it as one.
#:
#: A ``MemPool``'s ``PrivatePool`` starts at ``use_count{1}`` (torch's
#: ``c10/cuda/CUDACachingAllocator.cpp``), held by the Python object. ``release_cached_blocks`` -- the
#: allocator's OOM-retry path -- can only free blocks of a pool in ``graph_pools_freeable``, and a
#: pool enters that set ONLY via ``releasePool()`` when its use count reaches 0. So while SOMETHING
#: holds the handle, no pooled segment can be released, and therefore **no ``nvshmem_free`` can fire
#: implicitly**.
#:
#: Drop the handle and that inverts: the pool becomes freeable, the next release frees its segments,
#: and ``nvshmem_free`` -- which is COLLECTIVE -- runs at whatever moment THIS rank's interpreter
#: happened to choose. Two ranks choose differently, one enters the collective alone, and the job
#: deadlocks. That is the measured wedge.
#:
#: **WHAT THIS DOES NOT DO: it does not stop anyone calling ``symm_mem.get_mem_pool()`` directly,
#: and no code here should ever try.** That API is public, a caller can reach it in one line, and a
#: wrapper or guard around it would be trivially bypassed while costing real maintenance. What this
#: dict buys is narrower and sufficient: OUR allocations go through one pool, and
#: :meth:`release_symmetric_mempools` tears down every symmetric pool in the process at one ordered
#: point -- which works whether or not the pool came from here.
#:
#: Deliberately NOT in ``_state``, which ``cleanup()`` wipes: manager state and process state are
#: different lifetimes, and conflating them here would make ``cleanup()`` arm the deadlock.
_SYMMETRIC_MEMPOOLS: Dict[int, Any] = {}

#: Has the exit teardown been registered? Registration happens once, inside ``init_nvshmem``, so a
#: process that re-initializes after ``cleanup()`` does not stack duplicate hooks.
_NVSHMEM_EXIT_HOOK_REGISTERED = False

#: ``(device_type, dim_names, shape, world_size)`` -> the ``DeviceMesh`` built for that spec.
#:
#: WHY THIS EXISTS, and it is a leak fix rather than an optimization. ``torch``'s
#: ``DeviceMesh._init_process_groups`` has NO construction cache: for a dim that spans the whole
#: world it early-returns the existing default PG (free), but for a GENUINE SUBMESH dim it calls
#: ``split_group``/``new_group`` on EVERY construction. ``_MeshEnv`` holds only ``mesh_stack`` and
#: root-mapping helpers -- there is no lookup by ranks on any path; the only reuse torch offers is
#: SLICING a root mesh, whose slice paths pass ``_init_backend=False``.
#:
#: So a process that builds one ``DeviceMesh`` per test mints fresh NCCL communicators per test.
#: Measured at world 16: ``cp=16`` constructed 6x cost +0 MiB (1-D dim spans the world -> the default
#: PG), while ``cp=(4,4)`` cost +250 MiB on EACH of three constructions. Across a 1060-test session
#: that reached 68 GB of non-torch device memory and exhausted an 80 GB card. NCCL comm buffers live
#: outside torch's allocator, so they are invisible to ``memory_reserved()`` and ``empty_cache()``
#: cannot release them -- which is why the growth reads as "non-PyTorch memory" in an OOM message.
#:
#: KEYED ON THE RANKS, NOT THE AXIS NAME. ``_create_device_mesh_and_groups`` already had a reuse
#: guard (``if name_group in _state["_group"]: continue``) keyed on the axis NAME alone, and every
#: mesh here names its axis ``"cp"`` -- so name-keyed reuse would hand a ``cp=(4,4)`` mesh the
#: ``cp=16`` group, which is the "4-rank cp reported where the spec asked for 8" that
#: :meth:`reset_grid_groups` clears the dict to avoid. The shape is part of the key precisely so a
#: different factorization of the same ranks is a different entry.
#:
#: REUSE, NEVER DESTROY-AND-RECREATE. Destroying subgroups between meshes re-pays ``ncclCommSplit``
#: in both memory and latency on the path a one-launch-per-world-size driver most wants cheap, and
#: it is what :meth:`reset_grid_groups` warns against (a ``DeviceMesh`` that still holds a destroyed
#: group hands a later identical mesh a dead group).
#:
#: CLEARED BY ``cleanup()``, unlike :data:`_SYMMETRIC_MEMPOOLS`. These meshes hold process groups
#: that ``destroy_process_group()`` invalidates, so a mesh cached across a teardown would be handed
#: out dead to the next ``initialize()`` in the same process.
_DEVICE_MESH_CACHE: dict[tuple, Any] = {}

# ---------------------------------------------------------------------------
# Cluster-aware nvshmem env profiles.
#
# nvshmem's transport/heap init is sensitive to the fabric. ``init_nvshmem`` AUTO-DETECTS the
# hardware/topology traits below (host-only reads: env + sysfs, NO nvshmem/CUDA-kernel calls) and
# applies the matching PROFILE via ``os.environ.setdefault`` -> a caller/env can still override any
# single key, or force the whole profile with ``CPO_NVSHMEM_PROFILE``.
#
# Keyed on TRAITS, never a cluster NAME (there will be > 2 clusters -> a name map rots):
#   * node topology : single- vs multi-node (LOCAL_WORLD_SIZE vs WORLD_SIZE / device_count). Single-node
#                     is all-NVLink -> P2P/IPC, NO IB regardless of cluster.
#   * IB fabric     : any InfiniBand HCA visible under ``/sys/class/infiniband``.
#   * NIC curation  : does the cluster curate a valid GPU<->NIC rail set (some publish
#                     ``MELLANOX_VISIBLE_DEVICES``)? Detected + LOGGED ONLY -- it no longer
#                     selects a profile. The round-robin single-rail ``ib-ibgda-nicmap`` it used to force
#                     was no faster than multi-rail ib-ibgda on a curated fabric (6.15 == 6.16 ms) and a 7x footgun on
#                     VF fabrics; distance-based multi-rail ib-ibgda (safer: never lands a PE on an
#                     unusable NIC) is now the default for ALL multi-node IB, curated or not.
# Adding a cluster/fabric = pick (or add) a profile row below; no site-specific branch in the code.
#
# DEFAULT (auto): multi-node IB (curated OR not) -> ``ib-ibgda`` (IBGDA + distance NIC, multi-rail);
# single-node (any cluster) -> ``single`` (P2P/IPC, no IBGDA) -- the cp8 bring-up unblock. The curated-NIC
# ``ib-ibgda-nicmap`` auto-branch was removed (single-rail; no faster, 7x footgun on VF fabrics) -> now
# explicit-override only.
# ---------------------------------------------------------------------------
_NVSHMEM_PROFILE_ENV = "CPO_NVSHMEM_PROFILE"  # override: "auto" | a _NVSHMEM_PROFILES key

# Applied to EVERY profile (cluster-agnostic; == the historical always-on common set).
_NVSHMEM_COMMON_ENV = {
    "NVSHMEM_DISABLE_NCCL": "1",  # avoid the NCCL-bootstrap init race / multi-node deadlock
    "NVSHMEM_SYMMETRIC_SIZE": _DEFAULT_NVSHMEM_SYMMETRIC_SIZE,  # cap the (huge-default) symmetric heap
}
# NVSHMEM_DISABLE_NVLS IS DELIBERATELY ABSENT. The pin it used to carry claimed the NVLS multicast
# heap map crashes; that does not reproduce -- on 8xH200/NVSwitch every rank gets a real multicast
# address where the disabled run gets 0. Pinning it off also forecloses SHARP/NVSwitch reduction, so
# the default is nvshmem's own and an operator can still export NVSHMEM_DISABLE_NVLS=1. If a real
# NVLS fault resurfaces, gate it on a detected TRAIT in _NVSHMEM_PROFILES, never an unconditional pin.

# Profile-specific IB/IBGDA/NIC deltas (setdefault). An absent key => nvshmem's own default for it.
#: sysfs root for RDMA device discovery. A module-level seam so a test can point the probe at a
#: fabricated tree (ACTIVE/DOWN, InfiniBand/Ethernet) instead of whatever the host happens to have --
#: without it the ACTIVE-port logic in `_detect_nvshmem_traits` is assertable but never testable.
_IB_SYSFS_ROOT = "/sys/class/infiniband"

_NVSHMEM_PROFILES = {
    # Single-node (all-NVLink): P2P/IPC only. Bypass the REMOTE transport entirely so no IB endpoint is
    # ever connected (the single-node blocker was connect_endpoints on a round-robin-picked NIC).
    "single": {"NVSHMEM_REMOTE_TRANSPORT": "none"},
    # Multi-node but no IB visible (edge/misconfig): let nvshmem auto-select its remote transport + NIC.
    "default": {},
    # Multi-node IB, IBGDA OFF: host-proxy ibrc + topology-aware (distance) NIC. Override for a fabric
    # whose IBGDA does not init/fall-back cleanly.
    "ib-noibgda": {"NVSHMEM_IB_ENABLE_IBGDA": "0"},
    # Multi-node IB + IBGDA, distance-based NIC (NO naive round-robin). The non-curated-NIC profile
    # (ConnectX-7, no curated NIC set). TENTATIVE : NIC_PE_MAPPING=0 (topology-aware
    # distance, the confirmed root-cause fix) + NIC_HANDLER=auto (let nvshmem pick; the gpu handler may
    # be unavailable on CX-7). NOTE: a DEVICE-initiated cross-node put (the in-kernel put_nbi_warp drain)
    # may still REQUIRE handler=gpu -> confirm on a non-curated fabric at cp16 and update this row per
    # that result (does NOT touch the byte-identical `ib-ibgda-nicmap` below).
    "ib-ibgda": {
        "NVSHMEM_IB_ENABLE_IBGDA": "1",
        "NVSHMEM_IBGDA_NIC_HANDLER": "auto",
        "NVSHMEM_ENABLE_NIC_PE_MAPPING": "0",
    },
    # Multi-node IB + IBGDA + round-robin NIC->PE map (single-rail per PE). EXPLICIT-OVERRIDE ONLY as of
    # (no longer auto-selected): on a curated fabric it was no faster than multi-rail ib-ibgda (6.15 == 6.16 ms) and
    # forcing it is a 7x footgun on VF-NIC fabrics. Keep for deliberate single-rail debug (needs a curated set).
    "ib-ibgda-nicmap": {
        "NVSHMEM_IB_ENABLE_IBGDA": "1",
        "NVSHMEM_IBGDA_NIC_HANDLER": "gpu",
        "NVSHMEM_ENABLE_NIC_PE_MAPPING": "1",
    },
}


class DistributedManager:
    """Borg-style singleton managing torch.distributed + nvshmem state.

    Attributes (read-only via ``__getattr__``; backed by ``_state``):
        rank, world_size, local_rank, node_rank : int
        device : torch.device
        backend : str
        device_mesh : torch.distributed.device_mesh.DeviceMesh
        group, group_rank, group_ranks : dict[str, ...]
        nvshmem_initialized : bool
        method_init : str

    Examples
    --------
    >>> DistributedManager.initialize()
    >>> manager = DistributedManager()
    >>> manager.rank
    0
    >>> manager.world_size
    1
    """

    _state = {}

    def __new__(cls):
        """Return the Borg instance, defaulting all state for single-device use."""
        instance = super().__new__(cls)
        instance.__dict__ = cls._state
        # default the properties so that the default initialize() could work
        if not hasattr(instance, "_initialized"):
            instance._initialized = False
        if not hasattr(instance, "_has_dist"):
            instance._has_dist = False
        if not hasattr(instance, "_rank"):
            instance._rank = 0
        if not hasattr(instance, "_world_size"):
            instance._world_size = 1
        if not hasattr(instance, "_local_rank"):
            instance._local_rank = 0
        if not hasattr(instance, "_node_rank"):
            instance._node_rank = 0
        if not hasattr(instance, "_device"):
            instance._device = torch.device("cpu")
        if not hasattr(instance, "_backend"):
            instance._backend = None
        if not hasattr(instance, "_device_mesh"):
            instance._device_mesh = None
        if not hasattr(instance, "_layout_device_mesh"):
            instance._layout_device_mesh = None
        if not hasattr(instance, "_has_subgroups"):
            instance._has_subgroups = False
        if not hasattr(instance, "_device_mesh_subgroups"):
            instance._device_mesh_subgroups = None
        if not hasattr(instance, "_layout_device_mesh_subgroups"):
            instance._layout_device_mesh_subgroups = None
        if not hasattr(instance, "_group"):
            instance._group = {}
        if not hasattr(instance, "_group_rank"):
            instance._group_rank = {}
        if not hasattr(instance, "_group_ranks"):
            instance._group_ranks = {}
        if not hasattr(instance, "_subgroups"):
            instance._subgroups = {}
        if not hasattr(instance, "_subgroups_rank"):
            instance._subgroups_rank = {}
        if not hasattr(instance, "_subgroups_ranks"):
            instance._subgroups_ranks = {}
        if not hasattr(instance, "_layout_subgroups"):
            instance._layout_subgroups = {}
        if not hasattr(instance, "_method_init"):
            instance._method_init = None
        if not hasattr(instance, "_nvshmem_initialized"):
            instance._nvshmem_initialized = False
        return instance

    @classmethod
    def methods_init_available(cls) -> set[str]:
        """Return the set of available initialization methods."""
        return {"ENV", "SLURM"}

    @classmethod
    def backend_for_device(cls) -> Dict[str, Optional[str]]:
        """Return the mapping of device types to their default backend."""
        backend_for_device = {
            "cuda": "nccl" if torch.distributed.is_nccl_available() else None,
            "cpu": "gloo" if torch.distributed.is_gloo_available() else None,
        }
        return backend_for_device

    @classmethod
    def is_initialized(cls) -> bool:
        """Whether the singleton has been initialized."""
        return cls._state.get("_initialized", False)

    def __init__(self):
        """Validate that the singleton was initialized before instantiation.

        Raises
        ------
        RuntimeError
            If instantiated before ``DistributedManager.initialize()``.
        """
        if not self._initialized:
            raise RuntimeError(
                "A DistributedManager instance is being instantiated before "
                "the singleton class is initialized, which can lead to communication "
                "failure among processes. Please call DistributedManager.initialize() "
                "before instantiating any `DistributedManager` instance. "
            )
        super().__init__()

    def __getattr__(self, name: str) -> Any:
        """Read-only access to the shared ``_state`` (``foo`` -> ``_foo``).

        Raises
        ------
        AttributeError
            If neither ``name`` nor ``_name`` exists.
        """
        # to enable read-only access to the shared _state data
        key_state = f"_{name}"
        has_key_shared_state = key_state in self.__dict__
        has_key = name in self.__dict__
        if has_key_shared_state:
            return self.__dict__[key_state]
        elif has_key:
            return self.__dict__[name]
        else:
            raise AttributeError(f'Attribute "{name}" or "_{name}" not found.')

    def __str__(self):
        """Return a one-line human-readable summary of this rank's state."""
        output = (
            f"Initialized process {self.rank} of {self.world_size} using "
            f"method '{self.method_init}'. Device set to {str(self.device)}. Backend is {self.backend}"
        )
        return output

    @staticmethod
    def _setup(
        grid_group_sizes: Optional[_GridGroupSizesType] = None,
        device_type: str = "cuda",
        backend: Optional[str] = None,
        rank: int = -1,
        node_rank: int = -1,
        world_size: int = -1,
        local_rank: Optional[int] = None,
        local_world_size: Optional[int] = None,
        addr: str = "localhost",
        port: str = "29500",
        method_init: str = "ENV",
        **kwargs_init_pg,
    ):
        """Set up torch.distributed + the singleton state.

        Parameters
        ----------
        grid_group_sizes : OrderedDict, optional
            Group sizes; see ``create_grid_group()``.
        device_type : str
            "cuda" or "cpu".
        backend : str, optional
            Communication backend; defaults to the device's backend.
        rank, node_rank, world_size, local_rank : int
            Process identity (``local_rank`` guessed from GPU count if None).
        local_world_size : int, optional
            Ranks on THIS node. Recorded on the manager AND exported (below), because
            ``_detect_nvshmem_traits`` decides single- vs multi-node from it and its fallback
            (``torch.cuda.device_count()``) is wrong whenever ranks and GPUs decouple.
        addr, port : str
            Rendezvous address/port.
        method_init : str
            "ENV" or "SLURM".
        kwargs_init_pg
            Forwarded to ``torch.distributed.init_process_group``.
        """
        # TODO: could relax this to allow, e.g., "cuda" for "gloo"
        if device_type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"Input device type {device_type} but torch.cuda is not available")

        if world_size != -1 and grid_group_sizes is not None:
            total_size = 1
            assert hasattr(grid_group_sizes, "values")
            for value in grid_group_sizes.values():
                if isinstance(value, tuple) and all(isinstance(v, int) for v in value):
                    total_size *= prod(value)
                elif isinstance(value, int):
                    total_size *= value
                else:
                    raise RuntimeError(
                        f"Values in grid_group_sizes must be either int or tuple[int, ...], got {type(value)}"
                    )

            if world_size != total_size:
                raise RuntimeError(
                    f"Non-default world_size {world_size} != product of grid_group_sizes values ({total_size})"
                )

        backend_for_device = DistributedManager.backend_for_device()

        if backend_for_device["cpu"] is None and backend_for_device["cuda"] is None:
            raise RuntimeError(
                f"No backend available for the supported device types: {backend_for_device.keys()}"
            )

        if device_type not in backend_for_device:
            raise RuntimeError(
                f"Invalid input device type {device_type}: only supports {backend_for_device.keys()}"
            )

        if backend is None:
            backend = backend_for_device[device_type]
        elif backend != backend_for_device[device_type]:
            raise RuntimeError(
                f"Invalid input backend {backend} for input device type {device_type}"
            )

        # Export the WHOLE env:// set, not just the two names init_process_group reads.
        #
        # This is the single point at which the resolved identity reaches `os.environ`, and it is
        # deliberately the LAST thing rather than an input to parsing -- the reverse of what
        # `_derive_dist_env_from_slurm` did, which wrote these from SLURM_* and then read them back,
        # making the env:// method impossible to fail on a Slurm launch.
        #
        # All six are exported because all six are OBSERVABLE TODAY. Under a bare `srun` the old
        # derivation left RANK/WORLD_SIZE/LOCAL_RANK/LOCAL_WORLD_SIZE in `os.environ`, and they were
        # still there after initialize() returned; `_setup` exported only the two below. Deleting
        # the derivation without widening this export would make the other four silently vanish for
        # every caller that reads them afterwards -- and reading them afterwards is ordinary, not
        # exotic: this repository alone does it at 20 sites. A caller reading `os.environ["RANK"]`
        # must not be able to tell which branch initialized the manager.
        os.environ["MASTER_ADDR"] = addr
        os.environ["MASTER_PORT"] = str(port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        if local_rank is not None:
            os.environ["LOCAL_RANK"] = str(local_rank)
        if local_world_size is not None:
            os.environ["LOCAL_WORLD_SIZE"] = str(local_world_size)

        # instantiate the singleton
        DistributedManager._state["_initialized"] = True
        manager = DistributedManager()

        manager._has_dist = torch.distributed.is_available()

        manager._rank = rank
        manager._world_size = world_size
        manager._node_rank = node_rank
        manager._local_world_size = local_world_size
        if device_type == "cuda":
            if (
                manager.world_size > torch.cuda.device_count()
                and manager.world_size % torch.cuda.device_count()
            ):
                warn(
                    "world_size is not a multiple of torch.cuda.device_count() so cuda devices "
                    "could be shared by multiple ranks"
                )
            # will try to guess a local_rank from GPU counts
            if local_rank is None:
                manager._local_rank = manager.rank % torch.cuda.device_count()
            else:
                manager._local_rank = local_rank
            manager._device = torch.device(f"cuda:{manager.local_rank}")
        else:
            if local_rank is not None:
                manager._local_rank = local_rank
            manager._device = torch.device("cpu")

        if not manager.has_dist:
            warn("DistributedManager initialized without torch.distributed package")
            return

        if manager.device.type == "cuda":
            # set device before init_process_group to avoid unintended
            # cuda context and to avoid potential NCCL issues
            torch.cuda.set_device(manager.device)
            torch.cuda.device(manager.device)
            torch.cuda.empty_cache()

        manager._backend = backend

        # initialize torch.distributed
        if manager.device.type == "cuda" and backend == "nccl":
            try:
                # to prevent nccl hang and other potential issues:
                # see e.g., https://github.com/pytorch/pytorch/issues/142356
                torch.distributed.init_process_group(
                    manager.backend,
                    rank=manager.rank,
                    world_size=manager.world_size,
                    device_id=manager.device,
                    **kwargs_init_pg,
                )
            except TypeError:
                torch.distributed.init_process_group(
                    manager.backend,
                    rank=manager.rank,
                    world_size=manager.world_size,
                    **kwargs_init_pg,
                )
        else:
            torch.distributed.init_process_group(
                manager.backend, rank=manager.rank, world_size=manager.world_size, **kwargs_init_pg
            )

        manager._group["world"] = torch.distributed.group.WORLD
        manager._group_rank["world"] = manager.rank
        manager._group_ranks["world"] = torch.distributed.get_process_group_ranks(
            manager.group["world"]
        )

        manager._method_init = method_init

        if grid_group_sizes is not None:
            DistributedManager.create_grid_group(grid_group_sizes)

    @staticmethod
    def _create_device_mesh_and_groups(
        name: list[str], shape: list[int], suffix_mesh: Optional[str] = None
    ) -> None:
        """Create a ``DeviceMesh`` + per-dim process groups for ``(name, shape)``.

        Parameters
        ----------
        name : list[str]
            Dimension names of the mesh.
        shape : list[int]
            Mesh dimension sizes.
        suffix_mesh : str, optional
            Suffix for the mesh state key (e.g. "subgroups").

        Raises
        ------
        RuntimeError
            If the manager is not initialized, torch.distributed is unavailable,
            ``method_init``/``backend``/``device`` are invalid (default-init), or
            ``prod(shape)`` mismatches ``world_size``.
        """
        if not DistributedManager.is_initialized():
            raise RuntimeError(
                "DistributedManager is not initialized upon calling _create_device_mesh_and_groups"
            )
        if not DistributedManager._state["_has_dist"] or not torch.distributed.is_available():
            raise RuntimeError(
                "_create_device_mesh_and_groups requires torch.distributed package, which is not available"
            )
        if (
            DistributedManager._state["_method_init"] is None
            or DistributedManager._state["_method_init"]
            not in DistributedManager.methods_init_available()
        ):
            raise RuntimeError(
                f"Invalid DistributedManager method_init {DistributedManager._state['_method_init']} "
                "(most likely because it was default initialized)"
            )
        if (
            DistributedManager._state["_backend"] is None
            or DistributedManager._state["_backend"]
            not in DistributedManager.backend_for_device().values()
        ):
            raise RuntimeError(
                f"Invalid DistributedManager backend {DistributedManager._state['_backend']} "
                "(most likely because it was default initialized)"
            )
        if (
            DistributedManager._state["_device"] is None
            or DistributedManager._state["_device"].type
            not in DistributedManager.backend_for_device().keys()
        ):
            raise RuntimeError(
                f"Invalid DistributedManager device type {DistributedManager._state['_device'].type} "
                "(most likely because it was default initialized)"
            )

        world_size_expected = prod(shape)

        if world_size_expected != DistributedManager._state["_world_size"]:
            raise RuntimeError(
                f"world_size {DistributedManager._state['_world_size']} does not match the expected world size "
                f"{world_size_expected} computed from the input shape {shape}"
            )

        device_type = DistributedManager._state["_device"].type
        name_mesh = f"_device_mesh_{suffix_mesh}" if suffix_mesh is not None else "_device_mesh"

        # TODO: support arbitrary user-input layout (placement-generic; default LayoutRight).
        layout = LayoutRightMap(tuple(shape))
        DistributedManager._state[f"_layout{name_mesh}"] = layout

        grid2rank = torch.as_strided(
            torch.arange(world_size_expected), size=layout.shape, stride=layout.strides
        )
        # REUSE the DeviceMesh for a spec we have already built. Constructing a new one re-issues
        # `split_group`/`new_group` for every genuine submesh dim -- torch has no construction cache
        # (see `_DEVICE_MESH_CACHE`) -- so a per-test mesh leaks NCCL communicators for the life of
        # the process. A cache HIT returns the same object, so the `get_group` calls below hand back
        # the existing process groups and no communicator is created.
        mesh_key = (device_type, tuple(name), tuple(shape), world_size_expected)
        mesh = _DEVICE_MESH_CACHE.get(mesh_key)
        if mesh is None:
            mesh = torch.distributed.device_mesh.DeviceMesh(
                device_type, grid2rank, mesh_dim_names=tuple(name)
            )
            _DEVICE_MESH_CACHE[mesh_key] = mesh
        DistributedManager._state[name_mesh] = mesh

        for i_group in range(len(name)):
            name_group = name[i_group]

            if name_group in DistributedManager._state["_group"]:
                # skip those already created, e.g., from another call of this function
                continue

            DistributedManager._state["_group"][name_group] = DistributedManager._state[
                name_mesh
            ].get_group(name_group)
            DistributedManager._state["_group_rank"][name_group] = torch.distributed.get_group_rank(
                DistributedManager._state["_group"][name_group], DistributedManager._state["_rank"]
            )
            DistributedManager._state["_group_ranks"][name_group] = (
                torch.distributed.get_process_group_ranks(
                    DistributedManager._state["_group"][name_group]
                )
            )

    @staticmethod
    def create_grid_group(grid_group_sizes: _GridGroupSizesType) -> None:
        """Create the device mesh + (sub)groups from ``grid_group_sizes``.

        Parameters
        ----------
        grid_group_sizes : OrderedDict[str, int | tuple[int, ...]]
            Maps group name -> size. A tuple value partitions that group's ranks
            into a subgrid of the given shape (creating ``<name>_axis_<i>``
            subgroups). The rank layout follows ``LayoutRightMap`` (last group's
            ranks are contiguous in global rank).

        Notes
        -----
        Populates ``_group`` / ``_group_rank`` / ``_group_ranks`` (and the
        ``_subgroups*`` / ``_layout_subgroups`` / ``_device_mesh*`` state for
        tuple-valued groups).

        Examples
        --------
        >>> from collections import OrderedDict
        >>> grid_group_sizes = OrderedDict([("dp", 1), ("cp", (2, 2))])
        >>> DistributedManager.initialize(grid_group_sizes, device_type="cuda")
        >>> manager = DistributedManager()
        >>> # manager.group has 'world', 'dp', 'cp', 'cp_axis_0', 'cp_axis_1'
        >>> # On rank 0 (dp=0, cp_axis_0=0, cp_axis_1=0):
        >>> #   pe_map["cp_axis_0"] == tensor([0, 2]); pe_map["cp_axis_1"] == tensor([0, 1])

        Raises
        ------
        RuntimeError
            If a value is not ``int`` or ``tuple[int, ...]``, or the group /
            subgroup settings are inconsistent.
        """
        # FRONT DOOR. `initialize()` is the first call a program makes; without it there is no
        # process group, rank or world size for a mesh to be built on. This used to be caught
        # several frames deeper by `_create_device_mesh_and_groups`, surfacing as a
        # torch.distributed complaint about an uninitialized default group -- which reads as a bug
        # in the collective rather than as a missing initialize(). Deliberately NOT applied to
        # `symmetric_mempool` / `release_symmetric_mempools` / `_barrier_with_timeout`: those sit on
        # allocation and TEARDOWN paths, and `cleanup()` wipes `_state`, so a guard there would fire
        # during an atexit that runs after it and break a path that works today.
        if not DistributedManager.is_initialized():
            raise RuntimeError(
                "DistributedManager.create_grid_group() needs an initialized manager: there is no "
                "process group, rank or world size to build a mesh on yet. Call "
                "DistributedManager.initialize() first."
            )
        shape_groups = []
        name_groups = []
        shape_subgroups = []
        name_subgroups = []
        group2subgroup = {}
        group2subgroup_axes = {}
        assert hasattr(grid_group_sizes, "items")
        for k, v in grid_group_sizes.items():
            if isinstance(v, tuple) and all(isinstance(v_i, int) for v_i in v):
                # Create a new dimension of the DeviceMesh for each group
                shape_groups.append(prod(v))
                name_groups.append(k)
                # Create a new dimension of the DeviceMesh for each subgroup
                # to allow torch DTensor placement on the subgroups' DeviceMesh,
                # where each subgroup axis is treated as a separate dimension in the mesh
                shape_subgroups.extend(v)
                names_this_subgroup = [f"{k}_axis_{i}" for i in range(len(v))]
                name_subgroups.extend(names_this_subgroup)
                # map each group to its subgroups along each axis
                group2subgroup[k] = names_this_subgroup
                group2subgroup_axes[k] = list(
                    range(len(name_subgroups) - len(v), len(name_subgroups))
                )
            elif isinstance(v, int):
                shape_groups.append(v)
                name_groups.append(k)
                shape_subgroups.append(v)
                name_subgroups.append(k)
            else:
                raise RuntimeError(
                    f"Values in grid_group_sizes must be either int or tuple[int, ...], got {type(v)}"
                )

        # TODO: might not always need the device_mesh for parent groups
        # but one could just create them via create_group()
        DistributedManager._create_device_mesh_and_groups(name_groups, shape_groups)
        if (name_groups == name_subgroups) != (shape_groups == shape_subgroups):
            raise RuntimeError(
                f"Inconsistent group ({name_groups}, {shape_groups}) and "
                f"subgroup ({name_subgroups}, {shape_subgroups}) settings"
            )

        DistributedManager._state["_has_subgroups"] = name_groups != name_subgroups
        if DistributedManager._state["_has_subgroups"]:
            if len(group2subgroup) == 0:
                raise RuntimeError("group2subgroup is empty while _has_subgroups is True")
            DistributedManager._create_device_mesh_and_groups(
                name_subgroups, shape_subgroups, suffix_mesh="subgroups"
            )
            layout = DistributedManager._state["_layout_device_mesh_subgroups"]
            coords = DistributedManager._state["_device_mesh_subgroups"].get_coordinate()
            for name_group, name_subgroups in group2subgroup.items():
                # map the parent process group name to the subgroups
                DistributedManager._state["_subgroups"][name_group] = [
                    DistributedManager._state["_group"][name_subgroup]
                    for name_subgroup in name_subgroups
                ]
                DistributedManager._state["_subgroups_ranks"][name_group] = [
                    DistributedManager._state["_group_ranks"][name_subgroup]
                    for name_subgroup in name_subgroups
                ]
                DistributedManager._state["_subgroups_rank"][name_group] = [
                    DistributedManager._state["_group_rank"][name_subgroup]
                    for name_subgroup in name_subgroups
                ]
                # create the subgroup layout for each parent group
                # TODO: support LayoutMap.reshape to simplify this
                axes_subgroup = group2subgroup_axes[name_group]
                slices = list(coords)
                for axis in axes_subgroup:
                    slices[axis] = slice(None)
                # tuple() form (not layout[*slices]) for Python 3.10 compatibility (this repo's Python floor);
                # LayoutMap.__getitem__ accepts a tuple of slice|int directly.
                layout_subgroup = layout[tuple(slices)]
                # create a LayoutMap for the subgroups with offset 0 to be used for a bijective mapping
                # between rank within the subgroups and the subgrid
                DistributedManager._state["_layout_subgroups"][name_group] = LayoutMap(
                    layout_subgroup.strides, layout_subgroup.shape, offset=0
                )

    @staticmethod
    def reset_grid_groups() -> None:
        """Drop the current device mesh + derived groups, keeping the world group and nvshmem.

        Purpose
            Let one process build SEVERAL device-mesh configurations in sequence on a single world
            group -- which is what a session-scoped process group plus a per-test mesh requires.

        Semantics
            Clears the mesh objects and their layouts, every entry of ``_group`` / ``_group_rank`` /
            ``_group_ranks`` EXCEPT ``"world"``, the ``_subgroups*`` maps, ``_has_subgroups``, and
            the nvshmem PE maps (derived from ``_group_ranks``, so they would otherwise describe the
            previous mesh). Leaves torch.distributed, the world group and nvshmem untouched.

            **Not optional between meshes.** ``_create_device_mesh_and_groups`` SKIPS any group name
            already present, so without this a second mesh silently inherits the first one's group
            of the same name -- a 4-rank ``cp`` reported where the spec asked for 8, with no error.

            The underlying process groups are deliberately NOT destroyed: ``DeviceMesh`` caches the
            groups it creates, and destroying one it still holds would hand a later identical mesh a
            dead group. Dropping our references is enough, and re-creating them is now free --
            :data:`_DEVICE_MESH_CACHE` holds the ``DeviceMesh`` per
            ``(device_type, dim_names, shape, world_size)``, so the ``create_grid_group()`` that
            follows gets a cache HIT and reuses the same process groups.

            **That cache is why this method is now cheap, and it fixed a real leak.** This docstring
            used to claim the cost was "subgroup process groups accumulate for the process's life,
            bounded by the number of distinct meshes built". The bound was wrong: accumulation was
            per ``create_grid_group`` CALL, because torch's ``DeviceMesh`` has no construction cache
            and re-issues ``split_group``/``new_group`` for every genuine submesh dim. Measured at
            world 16, 14 declared mesh specs but 1060 calls: 68 GB of non-torch device memory, which
            exhausted an 80 GB card mid-session. See :data:`_DEVICE_MESH_CACHE`.

        Input requirements
            The manager must be initialized. COLLECTIVE BY CONSEQUENCE: this call communicates
            nothing, but the ``create_grid_group()`` that follows calls ``new_group``, which every
            rank must reach in the same order.

        Returns
            None.
        """
        state = DistributedManager._state
        for key in ("_group", "_group_rank", "_group_ranks"):
            kept = state.get(key, {})
            state[key] = {"world": kept["world"]} if "world" in kept else {}
        for key in ("_subgroups", "_subgroups_rank", "_subgroups_ranks", "_layout_subgroups"):
            state[key] = {}
        for key in (
            "_device_mesh",
            "_device_mesh_subgroups",
            "_layout_device_mesh",
            "_layout_device_mesh_subgroups",
        ):
            state[key] = None
        state["_has_subgroups"] = False

    @staticmethod
    def create_group(name: str, ranks: list[int], **kwargs_dist_ng) -> None:
        """Create and register a new process group ``name`` over ``ranks``.

        Parameters
        ----------
        name : str
            Group name.
        ranks : list[int]
            Global ranks in the group.
        **kwargs_dist_ng
            Forwarded to ``torch.distributed.new_group``.
        """
        DistributedManager._state["_group"][name] = torch.distributed.new_group(
            ranks=ranks, **kwargs_dist_ng
        )
        DistributedManager._state["_group_ranks"][name] = ranks
        DistributedManager._state["_group_rank"][name] = torch.distributed.get_group_rank(
            DistributedManager._state["_group"][name], DistributedManager._state["_rank"]
        )

    @staticmethod
    def _slurm_first_host(nodelist: str) -> Optional[str]:
        """First hostname of a Slurm compact nodelist -- rank 0's node -- without ``scontrol``.

        Pure text so every rank derives the same answer and none can time out. Handles ``n001``,
        ``n-[1-3,7]``, ``n-[1-2]-ib`` and top-level lists ``a,b-[1-2]``; a range yields its lower
        bound, and the literal token is kept so zero padding (``n-[01-09]`` -> ``n-01``) survives.

        Args:
            nodelist: Slurm nodelist string; may be empty, but not ``None`` (guarded by the caller).

        Returns:
            The first hostname, or ``None`` if empty/unparseable -- ``None`` falls through to the
            next source, whereas a wrong host would hang the rendezvous silently.
        """
        s = (nodelist or "").strip()
        if not s:
            return None
        # Split on the first TOP-LEVEL comma (commas inside [...] are part of one group).
        depth, cut = 0, len(s)
        for i, ch in enumerate(s):
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
            elif ch == "," and depth == 0:
                cut = i
                break
        s = s[:cut]
        if "[" not in s:
            return s or None
        prefix, _, rest = s.partition("[")
        group, _, suffix = rest.partition("]")
        first = group.split(",")[0].split("-")[0].strip()
        return f"{prefix}{first}{suffix}" if first else None

    @staticmethod
    def _detect_launcher() -> Optional[str]:
        """Name the launcher that started this process: ``"ENV"``, ``"SLURM"``, or ``None``.

        Purpose
            Decide WHICH namespace holds this process's identity, before any of it is parsed and
            before ``initialize()`` has run. ``initialize()`` uses it to pick a branch; the test
            harness uses it to answer "is there a launcher at all" while deciding whether to skip.

        Functionality & semantics
            Pure inspection of ``os.environ`` plus torch's own launcher predicate. Reads nothing
            else, writes nothing, raises nothing, and can be called any number of times before or
            after ``initialize()``.

            The order is FORCED, not a preference. ``torchrun`` spawns its workers INSIDE an
            ``srun`` step, so both signals are present there, and the ``SLURM_*`` ones describe the
            OUTER step: every worker sees ``SLURM_PROCID=0, SLURM_NTASKS=1``. Parsing those would
            give each worker ``rank=0, world_size=1``, so each would ``init_process_group`` into a
            private one-rank group and every collective would silently become a no-op. The torchrun
            question is therefore asked first.

            ``torch.distributed.is_torchelastic_launched()`` is used rather than reading
            ``TORCHELASTIC_RUN_ID`` directly: it is the only launcher-detection API in
            ``torch.distributed``'s public surface, and delegating means a future hardening of it
            is inherited instead of drifted from. Only ONE direction of that test is sound, and it
            is the one this function relies on. ``torchrun`` ALWAYS sets the variable
            (``local_elastic_agent.py`` writes it in a per-worker dict literal with no branch, and
            is the only writer in torch), so a FALSE result PROVES the launcher is not torchrun --
            which is what makes the ``SLURM`` branch below trustworthy even when a complete but
            stale ``env://`` set is present. The converse is false: anyone can
            ``export TORCHELASTIC_RUN_ID=x``. A forged value can therefore only push a process into
            the ``"ENV"`` branch, never out of it, and that branch still has to find every field.

        Input requirements
            None. Safe on a laptop, under any launcher, and before ``initialize()``.

        Returns
            ``"ENV"`` when torchrun owns the environment, or when ``RANK``/``WORLD_SIZE`` are set
            with no SLURM task env at all (a bare ``env://`` launch: k8s, a manual export).
            ``"SLURM"`` when this is a Slurm task and not a torchrun worker.
            ``None`` when neither is present -- there is no launcher, and the caller decides
            whether that is an error (``initialize()``) or a skip (the test harness).
        """
        if torch.distributed.is_torchelastic_launched():
            return "ENV"
        if "SLURM_PROCID" in os.environ and (
            "SLURM_NTASKS" in os.environ or "SLURM_NPROCS" in os.environ
        ):
            return "SLURM"
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            return "ENV"
        return None

    @staticmethod
    def _initialize_env(*args, **kwargs):
        """Initialize from the ``env://`` namespace ALONE -- torchrun's variables, or a site's.

        Purpose
            Turn the ``env://`` convention into a complete process identity and hand it to
            ``_setup``. This is the branch ``_detect_launcher`` selects for a torchrun launch.

        Functionality & semantics
            Reads ONLY ``env://`` names. It does not consult ``SLURM_*`` for anything, and it does
            not write ``os.environ`` -- ``_setup`` performs the single export once the identity is
            resolved. It used to open with ``_derive_dist_env_from_slurm()``, which mutated
            ``os.environ`` from ``SLURM_*`` and read it back two lines later; that made this method
            impossible to fail on a Slurm launch, which in turn made ``_initialize_slurm`` dead code.

            **Every field is required.** The old form accepted ``RANK`` + ``WORLD_SIZE`` and let the
            rest default, which is why a site that exports five of the seven names (a container hook
            setting ``RANK``/``LOCAL_RANK``/``WORLD_SIZE``/``MASTER_ADDR``/``MASTER_PORT`` but not
            ``LOCAL_WORLD_SIZE``) was accepted here with ``LOCAL_WORLD_SIZE`` left unset. Requiring
            all of them is what lets such a launch fail cleanly and fall through to the Slurm branch,
            which does have every field.

        Input requirements
            ``RANK``, ``WORLD_SIZE``, ``LOCAL_RANK``, ``LOCAL_WORLD_SIZE``, ``MASTER_ADDR`` and
            ``MASTER_PORT`` must all be set and the numeric ones parseable as ``int``.
            ``NODE_RANK`` or ``GROUP_RANK`` supplies the node index; absent both it defaults to 0,
            which is correct on one node and inert elsewhere (nothing reads ``_node_rank`` today).
            ``*args``/``**kwargs`` are forwarded to ``_setup`` unchanged.

        Raises
            RuntimeError naming the missing or non-integer variable, and naming
            ``CPO_DISTRIBUTED_INIT_METHOD`` so a caller can force this branch deliberately.
        """
        required = (
            "RANK",
            "WORLD_SIZE",
            "LOCAL_RANK",
            "LOCAL_WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
        )
        missing = [k for k in required if k not in os.environ]
        if missing:
            raise RuntimeError(
                f"the env:// method needs {required} and this process is missing {tuple(missing)}. "
                "Launch under torchrun, or set CPO_DISTRIBUTED_INIT_METHOD=SLURM to parse SLURM's "
                "task variables instead."
            )
        try:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            local_rank = int(os.environ["LOCAL_RANK"])
            local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
            # From LightningEnvironment.node_rank(): NODE_RANK wins, else GROUP_RANK (torchrun sets
            # the latter, never the former), else 0.
            node_rank = int(os.environ.get("NODE_RANK", os.environ.get("GROUP_RANK", 0)))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"the env:// variables {required} must all be integers where numeric; got "
                f"RANK={os.environ.get('RANK')!r} WORLD_SIZE={os.environ.get('WORLD_SIZE')!r} "
                f"LOCAL_RANK={os.environ.get('LOCAL_RANK')!r} "
                f"LOCAL_WORLD_SIZE={os.environ.get('LOCAL_WORLD_SIZE')!r}"
            ) from exc
        DistributedManager._setup(
            *args,
            rank=rank,
            node_rank=node_rank,
            world_size=world_size,
            local_rank=local_rank,
            local_world_size=local_world_size,
            addr=os.environ["MASTER_ADDR"],
            port=os.environ["MASTER_PORT"],
            method_init="ENV",
            **kwargs,
        )

    @staticmethod
    def _initialize_slurm(*args, **kwargs):
        """Initialize from the ``SLURM_*`` namespace ALONE -- a bare ``srun``, with no torchrun.

        Purpose
            Turn Slurm's task variables into a complete process identity, including a rendezvous
            address and port that Slurm itself does not provide. This is the branch
            ``_detect_launcher`` selects when torchrun is provably absent.

        Functionality & semantics
            Reads ONLY ``SLURM_*``. Any ``env://`` variable present is ignored: torchrun being
            absent means a complete ``env://`` set can only be stale or inherited, and Slurm's own
            variables are the authority for a Slurm task.

            Two values are DERIVED rather than read, because Slurm has no equivalent:

            * ``addr`` is the FIRST HOST of the STEP nodelist, i.e. rank 0's node.
              ``SLURM_LAUNCH_NODE_IPADDR`` is deliberately not used: it names the host where
              ``srun`` was invoked, which is rank 0's node only when srun runs inside the
              allocation. From a login shell it names the LOGIN node, and inside an sbatch step it
              has been measured to DISAGREE ACROSS RANKS (rank 0 seeing ``127.0.0.1`` while its
              peer sees a routable address), which splits a rendezvous rather than failing it.
              STEP, not JOB: a step may be narrower than its allocation.
            * ``port`` is a pure function of the job and step ids, mapped below the host's
              ephemeral floor. It must be PASSED to ``_setup``: omitting it -- which this method
              used to do -- silently applies ``_setup``'s ``"29500"`` default to every job on the
              cluster. The step id is in the seed so consecutive steps in one allocation do not
              collide with a dead predecessor's TIME_WAIT. No probe: every rank would bind the same
              candidate concurrently, one would win and the rest see EADDRINUSE, so a probe
              manufactures the collision it tests for.

        Input requirements
            ``SLURM_PROCID``, ``SLURM_LOCALID`` and a task count (``SLURM_NTASKS``, or the
            deprecated ``SLURM_NPROCS``) must be set and integer-parseable, and a nodelist must be
            present (``SLURM_STEP_NODELIST``, else ``SLURM_JOB_NODELIST``). ``SLURM_NTASKS_PER_NODE``
            gives the local world size; absent it, ``world_size // SLURM_JOB_NUM_NODES`` is used.
            ``*args``/``**kwargs`` are forwarded to ``_setup`` unchanged.

        Raises
            RuntimeError naming the missing or unusable variable and naming
            ``CPO_DISTRIBUTED_INIT_METHOD``.
        """
        procid = os.environ.get("SLURM_PROCID")
        ntasks = os.environ.get("SLURM_NTASKS") or os.environ.get("SLURM_NPROCS")
        localid = os.environ.get("SLURM_LOCALID")
        missing = [
            n
            for n, v in (
                ("SLURM_PROCID", procid),
                ("SLURM_NTASKS", ntasks),
                ("SLURM_LOCALID", localid),
            )
            if v is None
        ]
        if missing:
            raise RuntimeError(
                f"the slurm method needs SLURM_PROCID/SLURM_NTASKS/SLURM_LOCALID and this process is "
                f"missing {tuple(missing)}. Launch under srun, or set CPO_DISTRIBUTED_INIT_METHOD=ENV "
                "to parse the env:// variables instead."
            )
        try:
            rank, world_size, local_rank = int(procid), int(ntasks), int(localid)
            node_rank = int(os.environ.get("SLURM_NODEID", 0))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "SLURM_{PROCID,NTASKS,LOCALID,NODEID} must be integers; got "
                f"PROCID={procid!r} NTASKS={ntasks!r} LOCALID={localid!r} "
                f"NODEID={os.environ.get('SLURM_NODEID')!r}"
            ) from exc
        # SLURM_NTASKS_PER_NODE is "8" or "8(x2)" on a heterogeneous allocation -> leading int.
        local_world_size = None
        ntpn = os.environ.get("SLURM_NTASKS_PER_NODE", "")
        if ntpn:
            try:
                local_world_size = int(ntpn.split("(")[0])
            except ValueError:
                local_world_size = None
        if local_world_size is None:
            nnodes = os.environ.get("SLURM_JOB_NUM_NODES") or os.environ.get("SLURM_NNODES")
            try:
                local_world_size = world_size // int(nnodes) if nnodes else world_size
            except (ValueError, ZeroDivisionError):
                local_world_size = world_size
        nodelist = os.environ.get("SLURM_STEP_NODELIST") or os.environ.get("SLURM_JOB_NODELIST")
        addr = DistributedManager._slurm_first_host(nodelist) if nodelist else None
        if not addr:
            raise RuntimeError(
                "the slurm method needs SLURM_STEP_NODELIST (or SLURM_JOB_NODELIST) to name rank 0's "
                f"node; got {nodelist!r}. SLURM_LAUNCH_NODE_IPADDR is NOT used -- it names the host "
                "srun was invoked from, which is the login node from a login shell and disagrees "
                "across ranks inside an sbatch step. Set MASTER_ADDR and "
                "CPO_DISTRIBUTED_INIT_METHOD=ENV to supply a rendezvous yourself."
            )
        jid = os.environ.get("SLURM_JOB_ID", "")
        sid = os.environ.get("SLURM_STEP_ID", "")
        try:
            seed = int(jid) * 100003 + (int(sid) if sid.isdigit() else 0)
        except ValueError:
            seed = os.getpid()  # no job id (a laptop, a bare shell)
        lo = _UNPRIVILEGED_MIN + 1
        hi = (ephemeral_floor() or 65536) - 1
        port = lo + (abs(seed) % (hi - lo + 1))
        DistributedManager._setup(
            *args,
            rank=rank,
            node_rank=node_rank,
            world_size=world_size,
            local_rank=local_rank,
            local_world_size=local_world_size,
            addr=addr,
            port=str(port),
            method_init="SLURM",
            **kwargs,
        )

    @staticmethod
    def initialize(
        grid_group_sizes: Optional[OrderedDict[str, Union[int, tuple[int, ...]]]] = None,
        device_type: str = "cuda",
        backend: Optional[str] = None,
        **kwargs_init_pg,
    ):
        """Initialize the singleton (idempotent; warns + returns if already done).

        Selects the init method from ``CPO_DISTRIBUTED_INIT_METHOD`` (or
        ``CPO_DISTRIBUTED_INIT_METHOD`` for back-compat) if set, else tries
        ENV then SLURM, else default-initializes to the single-device case.

        Parameters
        ----------
        grid_group_sizes : OrderedDict[str, int | tuple[int, ...]], optional
            See ``create_grid_group()``.
        device_type : str
            "cuda" or "cpu".
        backend : str, optional
            Communication backend; defaults to the device's backend.
        kwargs_init_pg
            Forwarded to ``torch.distributed.init_process_group``.
        """
        if DistributedManager.is_initialized():
            warn("DistributedManager is already initialized. Skip initialize()")
            return
        if backend == "nccl":
            # https://pytorch.org/docs/master/notes/cuda.html#id5
            os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"
        # CPO-scoped only. An upstream-prefixed alias was honoured here as a back-compat
        # fallback until the de-branding removed it: a half-renamed env is worse than none,
        # because a launcher still exporting the old name would set an init method the code
        # says it reads from CPO_*, with nothing reporting the mismatch.
        method_init = os.getenv("CPO_DISTRIBUTED_INIT_METHOD") or os.getenv(
            "CPO_DISTRIBUTED_INIT_METHOD"
        )
        if (
            method_init is not None
            and method_init not in DistributedManager.methods_init_available()
        ):
            raise ValueError(
                f"Unknown value set for *_DISTRIBUTED_INIT_METHOD={method_init}. "
                f"Allowed options are one of {DistributedManager.methods_init_available()}"
            )
        if method_init is None:
            # WHICH launcher, then ONE parse. Not "try env, fall back to slurm": under torchrun BOTH
            # namespaces are populated -- torchrun nests inside the srun step -- and the SLURM_* ones
            # describe the OUTER step, so a Slurm parse there yields rank=0/world_size=1 for every
            # worker. That does not fail; it succeeds wrongly, and every collective becomes a no-op.
            # `_detect_launcher` asks the torchrun question first for exactly that reason, and its
            # FALSE answer is a proof (torchrun always sets the variable) rather than a guess, which
            # is what makes the SLURM branch safe even when a stale env:// set is lying around.
            method_init = DistributedManager._detect_launcher()
            if method_init is None:
                # NO LAUNCHER -> REFUSE. This used to `warn(... "Will default initialize
                # DistributedManager")` and then set `_initialized = True` and nothing else -- no
                # process group, no rank, no world size, no mesh. The manager reported itself
                # initialized while every value it owns stayed unset, so the next call raised
                # "torch.distributed must be initialized ... call initialize() first" at a caller who
                # had just called it. The warning announced an action it did not take and the error
                # blamed the user for the result.
                #
                # A half-built manager is the wrong answer to a missing launcher, and so is inventing
                # a rendezvous: deriving rank 0 / world 1 / localhost / a free port here would let
                # `python script.py` appear to work while quietly meaning something different from
                # every other way this class is used, and would put a launcher's job inside the
                # library. Providing the environment is the launcher's job. Say so, at the point of
                # failure.
                #
                # `tests/distributed/conftest.py` carries explicit workarounds for the old swallow --
                # it checks whether a process group actually exists after initialize() returns. Those
                # became unreachable rather than wrong; leave them, they document what this used to do.
                raise RuntimeError(
                    "DistributedManager.initialize() found no launcher environment: this process is "
                    "not a torchrun worker (torch.distributed.is_torchelastic_launched() is False), "
                    "SLURM_PROCID is unset, and RANK/WORLD_SIZE are unset -- so there is no rank, "
                    "world size or rendezvous to build a process group from.\n"
                    "Launch under torchrun or srun. A single GPU is a supported case and still "
                    "needs a launcher:\n"
                    "    torchrun --nproc_per_node=1 your_script.py\n"
                    "which sets RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT and takes the cp=1 path."
                )
        # `method_init` is now resolved -- either explicitly set, or detected above. Exactly ONE
        # initializer runs, and it reads exactly ONE namespace: nothing here combines the two.
        if method_init == "ENV":
            DistributedManager._initialize_env(
                grid_group_sizes, device_type=device_type, backend=backend, **kwargs_init_pg
            )
        else:
            DistributedManager._initialize_slurm(
                grid_group_sizes, device_type=device_type, backend=backend, **kwargs_init_pg
            )

    @staticmethod
    def _detect_nvshmem_traits():
        """Probe the hardware/topology traits that determine the nvshmem init profile.

        Host-only reads (env + sysfs + torch's already-initialized device count) -- NO nvshmem or
        CUDA-KERNEL calls, so it is safe to run BEFORE the symmetric allocation that brings the
        library up (which is where the env it feeds must already be set). Returns a trait dict
        consumed by :meth:`_select_nvshmem_profile`.

        Trait keys of note:
            ``mixed_fabric``: True when the node exposes BOTH InfiniBand and non-IB (RoCE/Ethernet)
                HCAs, classified by ``ports/*/link_layer``. Drives the NVSHMEM_HCA_LIST restriction.
            ``ib_hca_list``: ``dev:port`` pins for every InfiniBand HCA whose port state is ACTIVE,
                on EVERY fabric (not only a mixed one -- see the inline note). Empty only when no
                ACTIVE IB port exists. A DOWN port here would abort nvshmem bootstrap C-level, which
                is why the state filter is not optional.
            ``n_active_ib_ports``: count of ACTIVE InfiniBand ports regardless of ``mixed_fabric``,
                so a caller can warn about rail spread on an all-IB node where the pin is empty.
        """
        # Node topology: single-node iff every rank is on THIS node. Prefer torchrun's LOCAL_WORLD_SIZE;
        # else fall back to device_count() (1 rank/GPU is the norm; mirrors the manager's own
        # world_size-vs-device_count check). single_node -> all-NVLink -> no IB needed.
        ws = int(DistributedManager._state.get("_world_size", 1) or 1)
        # LOCAL_WORLD_SIZE comes from the MANAGER, which resolved it from whichever namespace owns
        # this launch, not from `os.environ`. The env read this replaced needed a policy for the
        # variable being absent, and the policy it grew -- `torch.cuda.device_count()` -- is wrong
        # the moment ranks and GPUs decouple: 4 ranks/node on 8-GPU nodes across 2 nodes gives
        # lws = 8 >= ws = 8, so `single_node` reads True and an NVLink-only profile is selected for
        # a job that spans two nodes. A resolved value removes the question instead of answering it.
        lws = DistributedManager._state.get("_local_world_size")
        if not (isinstance(lws, int) and lws > 0):
            try:
                ndev = torch.cuda.device_count()
            except Exception:  # noqa: BLE001 — detection must never raise; default to single-node-safe
                ndev = 0
            lws = ndev if ndev > 0 else ws
        single_node = (ws <= 1) or (lws >= ws)
        # IB fabric: any InfiniBand HCA visible.
        try:
            ib_devs = sorted(d for d in os.listdir(_IB_SYSFS_ROOT) if d)
        except OSError:
            ib_devs = []
        # MIXED-FABRIC guard: a node may expose an Ethernet/RoCE HCA (e.g. venue D's mlx5_3) ALONGSIDE its
        # InfiniBand HCAs. nvshmem's NIC selection (round-robin OR distance) can then land a PE on the
        # Ethernet NIC -> QP INIT->RTR fails ("connect EPS failed" / "building transport map failed").
        # Classify each HCA by port link_layer; when the fabric is MIXED (>=1 IB AND >=1 non-IB) we later
        # pin nvshmem to the IB-only HCAs via NVSHMEM_HCA_LIST (== what NCCL_IB_HCA already curates). An
        # all-IB node yields an EMPTY list -> no restriction -> a curated fabric stays byte-identical.
        ib_only, non_ib, ib_active = [], [], []
        for d in ib_devs:
            ll, st, port = "", "", "1"
            for p in ("1", "2"):
                try:
                    with open(f"{_IB_SYSFS_ROOT}/{d}/ports/{p}/link_layer") as _fh:
                        ll = _fh.read().strip()
                except OSError:
                    continue
                port = p
                # PORT STATE, not just link layer. Reading only link_layer admits a DOWN InfiniBand
                # port into the pin below, and a DOWN port in NVSHMEM_HCA_LIST aborts bootstrap
                # C-level (exit 255, NO Python traceback; only the bad-HCA rank dies while its peers
                # hang) -- the failure this restriction exists to PREVENT. sysfs spells it
                # "4: ACTIVE", so match on the word, not on equality.
                try:
                    with open(f"{_IB_SYSFS_ROOT}/{d}/ports/{p}/state") as _fh:
                        st = _fh.read().strip()
                except OSError:
                    st = ""
                break
            (ib_only if ll == "InfiniBand" else non_ib).append(d)
            if ll == "InfiniBand" and "ACTIVE" in st.upper():
                ib_active.append(f"{d}:{port}")
        # mixed_fabric keeps its link-layer meaning (>=1 IB AND >=1 non-IB) and still drives PROFILE
        # selection. Deriving it from port state instead would make a node with one DOWN IB port look
        # "mixed" and switch its whole profile -- a much larger change than anything here.
        mixed_fabric = bool(ib_only) and bool(non_ib)
        # Build the pin on EVERY fabric, not only a mixed one. `mixed_fabric` prevents selection of a
        # wrong-FABRIC NIC but does not provide rail SPREAD. On a homogeneous 8-rail node, an empty
        # list cost 28.6x on a cross-node A2A (23.78 ms vs 0.83 ms once the list was supplied by hand;
        # IBGDA alone moved it only to 20.05 ms).
        # `setdefault` at the call site means an operator-exported value still wins, so this is a
        # DEFAULT, not a policy the launcher cannot override.
        ib_hca_list = ",".join(ib_active)
        # GPUDirect-RDMA peer memory (advisory only): IBGDA needs it, but nvshmem self-falls-back to
        # ibrc if absent, so this is LOGGED not GATED -> keeps the curated byte-identical path off a
        # false-negative module probe.
        peermem = os.path.isdir("/sys/module/nvidia_peermem") or os.path.isdir("/sys/module/gdrdrv")
        # Round-robin NIC->PE map safety: a curated cluster publishes MELLANOX_VISIBLE_DEVICES ->
        # `mype_node % n_devices` lands on a valid NIC; a fabric without it must use distance selection.
        # NOTE: a bare "all" is NOT a curated set (it exposes every NIC incl. a wrong-fabric one) -> the
        # mixed_fabric branch below routes such nodes to distance-NIC + the IB-only HCA restriction.
        nic_curated = "MELLANOX_VISIBLE_DEVICES" in os.environ
        return {
            "single_node": single_node,
            "world_size": ws,
            "local_world_size": lws,
            "ib_present": len(ib_devs) > 0,
            "n_ib_devices": len(ib_devs),
            "ib_devices": ib_devs,
            "mixed_fabric": mixed_fabric,
            "ib_hca_list": ib_hca_list,
            "n_active_ib_ports": len(ib_active),
            "peermem": peermem,
            "nic_curated": nic_curated,
        }

    @staticmethod
    def _select_nvshmem_profile(traits):
        """Map detected ``traits`` -> a :data:`_NVSHMEM_PROFILES` key.

        ``CPO_NVSHMEM_PROFILE`` overrides the auto-detect (``auto`` = detect; any profile key forces
        it). The auto decision tree is intentionally a short, trait-ordered cascade so a future
        fabric slots into one branch (or a new profile row) with no site-specific code.
        """
        override = os.environ.get(_NVSHMEM_PROFILE_ENV, "auto").strip().lower()
        if override and override != "auto":
            if override not in _NVSHMEM_PROFILES:
                raise ValueError(
                    f"{_NVSHMEM_PROFILE_ENV}={override!r} is not a known nvshmem profile "
                    f"{sorted(_NVSHMEM_PROFILES)}; use 'auto' or one of those."
                )
            return override
        if traits["single_node"]:
            return "single"  # all-NVLink -> P2P/IPC only, no IB (any cluster)
        if not traits["ib_present"]:
            return "default"  # multi-node, no visible IB (edge): nvshmem auto-selects
        if traits.get("mixed_fabric"):
            # IB + a stray Ethernet/RoCE HCA (e.g. venue D's mlx5_3): distance NIC (NO round-robin, which
            # can pick the wrong-fabric NIC), and init_nvshmem pins NVSHMEM_HCA_LIST to the IB-only HCAs.
            return "ib-ibgda"
        # #45: multi-node IB (curated OR not) -> the robust multi-rail profile below. The historical
        # ``nic_curated -> ib-ibgda-nicmap`` (single-rail round-robin) auto-branch was REMOVED: it gave
        # ZERO benefit where it was tuned (cp16 D256 N2048: nicmap 6.15 ms == ib-ibgda 6.16 == default
        # 6.15) and is a 7x footgun on a VF-NIC fabric (venue B cp16: 45 ms single-rail vs 6.2 ms multi-rail).
        # Distance-based selection is also strictly SAFER than round-robin (never lands a PE on an unusable
        # NIC). ``ib-ibgda-nicmap`` stays an EXPLICIT override (CPO_NVSHMEM_PROFILE=ib-ibgda-nicmap) for
        # deliberate single-rail measurement/debug only.
        return "ib-ibgda"  # multi-node IB: IBGDA + topology-aware (distance) NIC, multi-rail

    @staticmethod
    def init_nvshmem():
        """Bring nvshmem up over the world group and build the per-group PE maps. Idempotent.

        **This does not initialize nvshmem -- torch does.** ``torch.distributed._symmetric_memory``
        bootstraps the library lazily, on the FIRST BYTE allocated from the symmetric allocator, by
        minting a unique id and all-gathering it over the process group's own ``Store``. So this
        method selects the backend, TRIGGERS that bootstrap with a minimal collective allocation, and
        verifies the outcome. There is no matching finalize, here or in :meth:`cleanup`: torch
        never finalizes it -- :meth:`cleanup` does, which is why that is where the drain lives.

        Must be called after ``initialize()`` (torch.distributed up).

        Sequence -- the order is load-bearing, and each step is named by what breaks without it:

        1. **env.** The cluster-agnostic COMMON set (``NVSHMEM_DISABLE_NCCL=1``,
           ``NVSHMEM_DISABLE_NVLS=1``, ``NVSHMEM_SYMMETRIC_SIZE``) plus a trait-detected PROFILE
           (IB/IBGDA/NIC deltas), both via ``setdefault`` so a caller can override any single key or
           force the whole profile with ``CPO_NVSHMEM_PROFILE``. nvshmem reads its env at library
           init, which now happens inside step 4 -- so this MUST precede it or the profile is
           silently ignored. The choice is recorded in ``_state["_nvshmem_profile"]`` and printed
           once (rank 0).
        2. **``set_backend("NVSHMEM")``, always and unconditionally.** Torch's default resolves to
           ``CUDA``, and the failure is silent rather than loud: measured on this build, an
           ``empty`` + ``rendezvous`` under the default backend SUCCEEDS and hands back non-NULL peer
           pointers (CUDA IPC), while ``nvshmem.bindings.n_pes()`` stays ``0`` -- nvshmem was never
           initialized at all. Every in-kernel ``direct.my_pe()`` would then read 0 on every rank.
           A "peer pointers look fine" check does not catch it; only the backend does.
        3. **Verify the backend took.** It is a process-global that cannot be changed to a different
           value once any symmetric tensor exists, so a caller who allocated first pins this process
           to the wrong one. Verifying converts that into a front-door error instead of a
           silently-degraded run. (Re-setting the SAME value is safe, so step 2 is unconditional.)
        4. **The bootstrap:** ONE byte allocated **through** :meth:`symmetric_mempool` +
           ``rendezvous`` over the world group. COLLECTIVE -- every rank must reach it, in the same
           program order, or it hangs. The pool is taken explicitly rather than left to torch's
           implicit one; see the STEP 4 comment for why the env var makes that a real difference.
        5. **Verify torch's PE numbering** matches what every peer-indexed transfer assumes
           (``my_pe() == rank``, ``n_pes() == world_size``). Under the old hand-rolled init this held
           BY CONSTRUCTION, because ``rank=`` was passed to ``nvshmem.core.init``; under torch's
           bootstrap it is an assumption ABOUT torch, and a wrong PE map misroutes A2A payloads with
           no error anywhere. Skipped with a warning if nvshmem4py is absent.

        The bootstrap tensor is dropped immediately. nvshmem stays up and the block returns to the
        symmetric ``MemPool`` -- measured: a later allocation still rendezvouses with non-NULL peer
        pointers -- so retaining it would only strand a symmetric segment. Returning it to the POOL
        rather than to torch's general allocator is what keeps the collective ``nvshmem_free`` off
        this rank's destruction path: the segment is recycled, and the one free that does happen is
        :meth:`release_symmetric_mempools` at a point every rank reaches together.

        Raises
        ------
        RuntimeError
            If torch.distributed or the singleton is not initialized; if the device is not CUDA; if
            this torch has no ``_symmetric_memory``; if nvshmem is unavailable to torch
            (``is_nvshmem_available()``); if the NVSHMEM backend could not be selected; or if torch's
            PE numbering disagrees with the torch rank.
        ValueError
            If ``CPO_NVSHMEM_PROFILE`` is set to an unknown profile name.
        """
        if DistributedManager._state.get("_nvshmem_initialized", False):
            return  # already initialized

        if not torch.distributed.is_initialized():
            raise RuntimeError(
                "torch.distributed must be initialized before calling init_nvshmem(). "
                "Call DistributedManager.initialize() first."
            )
        if not DistributedManager._state.get("_initialized", False):
            raise RuntimeError(
                "DistributedManager must be initialized before calling init_nvshmem(). "
                "Call DistributedManager.initialize() first."
            )
        device = DistributedManager._state.get("_device", torch.device("cpu"))
        if device.type != "cuda":
            raise RuntimeError(
                f"nvshmem requires CUDA devices but DistributedManager device is {device.type}."
            )
        if not HAS_SYMM_MEM:
            raise RuntimeError(
                f"torch.distributed._symmetric_memory is required for init_nvshmem() but this torch "
                f"({torch.__version__}) does not provide it. It owns the nvshmem lifetime; this "
                "manager has no fallback bootstrap."
            )
        if not symm_mem.is_nvshmem_available():
            raise RuntimeError(
                "torch reports nvshmem is unavailable (is_nvshmem_available() is False): either this "
                "torch build lacks nvshmem support, or the nvshmem runtime is not loadable here. "
                "Install a matching nvidia-nvshmem-cu13."
            )

        # STEP 1. Cluster-aware nvshmem env: apply the cluster-agnostic COMMON set + the trait-detected
        # PROFILE, both via setdefault (a caller/env can still override any single key, or force the whole
        # profile with CPO_NVSHMEM_PROFILE). MUST precede the FIRST SYMMETRIC ALLOCATION below -- that is
        # where torch now initializes the library, and nvshmem reads this env at library init, so setting
        # it afterwards is silently ignored rather than an error.
        # See the _NVSHMEM_PROFILES table at module top. DEFAULT (auto) is
        # BYTE-IDENTICAL to the historical curated-NIC multi-node set (-> ib-ibgda-nicmap: IBGDA + gpu handler
        # + round-robin NIC map); single-node (any cluster) auto-drops to P2P/IPC (NVSHMEM_REMOTE_TRANSPORT
        # =none, no IBGDA/NIC-map) -- the venue D cp8 bring-up unblock. IBGDA rationale (retained): a
        # DEVICE-initiated cross-node put (the in-kernel put_nbi_warp drain) needs GPU-initiated RDMA =
        # IBGDA; the round-robin NIC map is curated-fabric-tuned and UNSAFE on a fabric without a curated NIC set.
        for _k, _v in _NVSHMEM_COMMON_ENV.items():
            os.environ.setdefault(_k, _v)
        _traits = DistributedManager._detect_nvshmem_traits()
        _profile = DistributedManager._select_nvshmem_profile(_traits)
        for _k, _v in _NVSHMEM_PROFILES[_profile].items():
            os.environ.setdefault(_k, _v)
        # Mixed-fabric NIC restriction (e.g. venue D's Ethernet mlx5_3 among the IB HCAs): pin nvshmem to
        # the IB-only HCAs so no PE lands on the wrong-fabric NIC (the QP INIT->RTR failure). Only set on a
        # MIXED fabric (all-IB -> empty -> unset -> byte-identical). setdefault -> a caller/env wins.
        if _traits.get("ib_hca_list"):
            os.environ.setdefault("NVSHMEM_HCA_LIST", _traits["ib_hca_list"])
        DistributedManager._state["_nvshmem_profile"] = _profile
        if int(DistributedManager._state.get("_rank", 0) or 0) == 0:
            print(
                f"[init_nvshmem] nvshmem profile={_profile!r} via {_NVSHMEM_PROFILE_ENV}="
                f"{os.environ.get(_NVSHMEM_PROFILE_ENV, 'auto')} (single_node={_traits['single_node']} "
                f"ib={_traits['n_ib_devices']} peermem={_traits['peermem']} "
                f"nic_curated={_traits['nic_curated']} mixed_fabric={_traits.get('mixed_fabric')} "
                f"hca_list={os.environ.get('NVSHMEM_HCA_LIST', '<unset>')} ws={_traits['world_size']}/"
                f"lws={_traits['local_world_size']})",
                flush=True,
            )
            # RAIL-SPREAD WARNING -- detection only, never policy. CLAUDE.md forbids this repo's
            # Python from being the SOURCE of a NIC/HCA/rail value, so nothing is set here; what is
            # removed is the SILENCE. An all-IB node leaves `ib_hca_list` empty by design (see
            # _detect_nvshmem_traits), and NVSHMEM then does not necessarily spread over every rail.
            # MEASURED on venue B (2 nodes x 8 H100, 8 ACTIVE 400 Gb/s NDR rails), cp=(2,8) N=2048
            # D=128, ours-vs-main paired median on the fused back drain:
            #     NVSHMEM_HCA_LIST unset ............ 23.78 ms  (28.6x main)
            #     + NVSHMEM_IB_ENABLE_IBGDA=1 only .. 20.05 ms  (24.1x main)  <- IBGDA was already on
            #     + NVSHMEM_HCA_LIST=<8 rails> ......  0.83 ms  ( 0.99x main)
            # i.e. a 28x cliff that reports as a clean run. The operator sets the env; this line
            # makes forgetting cost one warning instead of one wrong perf conclusion.
            # Now that DM supplies the pin, "unset" is no longer the interesting state -- an EMPTY
            # effective list is. That happens when no ACTIVE IB port was found, or when an operator
            # exported NVSHMEM_HCA_LIST="" (a present-but-empty value defeats `setdefault`). Either
            # way a multi-node job is about to run without a rail pin, which is the 28x case.
            if not _traits["single_node"] and not os.environ.get("NVSHMEM_HCA_LIST"):
                print(
                    f"[init_nvshmem] WARNING: multi-node job with an EMPTY NVSHMEM_HCA_LIST "
                    f"({_traits.get('n_active_ib_ports', 0)} ACTIVE IB ports detected). nvshmem may "
                    f"not spread across rails -- measured 28x on a cross-node A2A. Either no ACTIVE "
                    f"InfiniBand port was found on this node, or the value was exported EMPTY, which "
                    f"defeats the default. Pin it from the launcher: "
                    f'`eval "$(hca_select.sh --env)"`.',
                    flush=True,
                )

        local_rank = DistributedManager._state.get("_local_rank", 0)
        rank = DistributedManager._state.get("_rank", 0)
        world_size = DistributedManager._state.get("_world_size", 1)

        # Bind the cuda.core current device. NOT needed to initialize nvshmem anymore -- torch does
        # that on its own current device -- but the CuTe-DSL compile path (cute_compile_helper ->
        # library_init) requires a current cuda.core device, and this is where local_rank is known.
        if HAS_CUDA_CORE:
            cuda.core.Device(local_rank).set_current()

        # STEP 1b: WORLD_SIZE == 1 -> there is nothing to bring up, and saying so is the whole
        # point. A single-rank job has no peers, so the symmetric-memory rendezvous below has no
        # counterpart: `symm_mem.rendezvous` asserts `global_ranks.size() > 1` and raises
        # "Expected global_ranks.size() > 1 to be true, but got false" from torch. Before this
        # guard, the README's own cp=1 promise -- "a plain tensor, a 1-device mesh, or placements
        # that split no token axis all route to the single-device kernel WITHOUT TOUCHING NVSHMEM"
        # -- was unreachable through the documented entry point: the tutorial calls init_nvshmem()
        # unconditionally, so a reader with one GPU hit that RuntimeError before any routing
        # decision was made. MEASURED, `torchrun --nproc_per_node=1` on the published block.
        #
        # Returning here is the honest shape rather than a special case: at cp=1 every A2A path is
        # const_expr-elided and the workflow dispatches to the single-device kernel, which allocates
        # ordinary CUDA memory and issues no peer transfer. The device bind above is deliberately
        # kept -- the CuTe-DSL compile path needs a current cuda.core device at every world size --
        # and the backend is deliberately NOT selected, because selecting it is what would make the
        # claim false.
        #
        # INHERITED, not introduced: `main` carries the identical unguarded `symm_mem.rendezvous`
        # at the same line, so cp=1 has never worked through this entry point on either branch.
        if int(world_size) <= 1:
            DistributedManager._state["_nvshmem_initialized"] = True
            print(
                "[init_nvshmem] WORLD_SIZE=1 -> nvshmem NOT initialized (no peers to rendezvous "
                "with). The cp=1 single-device kernel needs none; every A2A path const_expr-elides.",
                flush=True,
            )
            return

        # STEP 2+3: select the NVSHMEM backend and PROVE it took, before anything is allocated.
        # Both failure orders are covered on purpose: torch raises "Backend can not be changed after
        # use." from set_backend when a tensor already exists (measured), and a build that instead
        # ignored the request silently would be caught by the get_backend check below.
        _WHY = (
            "A non-NVSHMEM backend cannot serve the in-kernel nvshmem device API: it is CUDA-IPC, "
            "single-node, and leaves nvshmem UNINITIALIZED (n_pes()==0) while still handing back "
            "plausible non-NULL peer pointers -- so nothing downstream notices. The backend is a "
            "process-global fixed by the FIRST symmetric allocation in the process; call "
            "DistributedManager.init_nvshmem() before any symmetric memory is allocated."
        )
        try:
            symm_mem.set_backend("NVSHMEM")
        except RuntimeError as e:
            raise RuntimeError(
                f"could not select the NVSHMEM symmetric-memory backend ({e}); it is currently "
                f"{symm_mem.get_backend(device)!r}. Something allocated symmetric memory before "
                f"init_nvshmem(). {_WHY}"
            ) from e
        backend = symm_mem.get_backend(device)
        if backend != "NVSHMEM":
            raise RuntimeError(
                f"symmetric-memory backend is {backend!r} after set_backend('NVSHMEM') returned "
                f"without error. {_WHY}"
            )

        # STEP 4: trigger torch's lazy UID bootstrap. COLLECTIVE. One byte is enough -- the handshake
        # is keyed on the FIRST allocation, not on its size. ``group.WORLD`` is the same object as
        # ``_group["world"]``; nvshmem is initialized over the world, which is the premise the PE
        # maps rest on.
        #
        # Allocated through OUR pool, NOT via a bare ``symm_mem.empty``. torch's ``empty`` routes to
        # ``get_mem_pool`` only while ``_should_use_implicit_mempool()`` holds, and that reads
        # ``TORCH_SYMMMEM_IMPLICIT_POOL`` (defaulting to "1", verified in the torch v2.11.0 source) --
        # so an operator exporting 0 silently puts the FIRST allocation every rank makes back on the
        # unpooled path, where the ``from_blob`` deleter calls the COLLECTIVE ``nvshmem_free`` the
        # moment THIS rank's interpreter drops the last reference. Two ranks decide that at different
        # times and the job hangs. Taking the pool explicitly makes the bootstrap independent of that
        # env var and brings it under ``release_symmetric_mempools()``.
        with torch.cuda.use_mem_pool(DistributedManager.symmetric_mempool(device)):
            probe = torch.empty(1, dtype=torch.uint8, device=device)
        handle = symm_mem.rendezvous(probe, torch.distributed.group.WORLD)

        # STEP 5: check the PE-map premise (see the docstring -- it used to hold by construction).
        if HAS_NVSHMEM4PY:
            my_pe, n_pes = int(nvshmem.bindings.my_pe()), int(nvshmem.bindings.n_pes())
            if (my_pe, n_pes) != (rank, world_size):
                raise RuntimeError(
                    f"torch's nvshmem bootstrap numbered this PE {my_pe} of {n_pes}, but "
                    f"torch.distributed has it as rank {rank} of {world_size}. Peer addressing "
                    "assumes PE == global rank (see PeMap in pe_map.py, whose cp_pe_table is built "
                    "from the DeviceMesh rank tensor), so proceeding would misroute "
                    "every peer-indexed transfer with no error raised anywhere. The usual cause is "
                    "a second, hand-rolled nvshmem init racing torch's."
                )
        else:
            warn(
                "nvshmem4py is not importable, so init_nvshmem() cannot verify that torch numbered "
                "the PEs as PE == global rank -- the premise the PE maps are built on. The device "
                "kernels require nvshmem4py regardless; install nvshmem4py-cu13."
            )

        # Drop the bootstrap: nvshmem stays up and the block returns to the symmetric MemPool (NOT
        # to torch's general allocator -- see STEP 4), so retaining it would only strand a symmetric
        # segment. Returning it to the pool means this `del` recycles rather than frees: no
        # collective runs here, which is the whole point of routing STEP 4 through the pool.
        del handle, probe
        global _NVSHMEM_LIBRARY_UP, _NVSHMEM_EXIT_HOOK_REGISTERED
        _NVSHMEM_LIBRARY_UP = True  # process-lifetime fact; survives cleanup()'s state wipe
        # Register the teardown ONCE, here, where nvshmem demonstrably came up. atexit is LIFO and
        # this runs long after torch's own registrations, so ours fires FIRST -- ahead of the static
        # destructors whose collective nvshmem_free is what faults across nodes.
        if not _NVSHMEM_EXIT_HOOK_REGISTERED:
            import atexit

            atexit.register(DistributedManager._drain_and_finalize_at_exit)
            _NVSHMEM_EXIT_HOOK_REGISTERED = True
        DistributedManager._state["_nvshmem_initialized"] = True

    @property
    def nvshmem_initialized(self) -> bool:
        """Whether nvshmem has been initialized."""
        return self._nvshmem_initialized

    @staticmethod
    def symmetric_mempool(device=None):
        """The ONE symmetric-memory ``MemPool`` for a device, owned for the life of the PROCESS.

        Purpose
            Give this process one place to get the symmetric-memory pool, so OUR allocations share
            it. Allocations routed through this pool are recycled rather than freed, which removes
            the collective ``nvshmem_free`` from the object-destruction path.

            **This is a convention, not an enforcement.** ``symm_mem.get_mem_pool()`` is public and
            any caller may use it directly; nothing here prevents that and nothing should try. The
            property that matters is a property of how THIS package allocates, which needs no
            control over what anyone else does.

        Why this exists rather than calling ``symm_mem.get_mem_pool`` at each site
            The pool's protection is a REFCOUNT property, not an API property. Torch frees a pool's
            segments only when the pool reaches ``use_count == 0`` and enters
            ``graph_pools_freeable``; until then even the allocator's OOM-retry path cannot touch
            them. Holding the handle in one place, for the whole process, is what makes that
            condition permanently false. Scattered callers each fetching their own reference would
            still work today, but any one of them dropping the last reference re-arms the deadlock,
            and nothing would flag it -- which is exactly how the wedge arrived.

        Semantics
            Idempotent and collective-free. The first call for a device builds the pool via
            ``symm_mem.get_mem_pool(device)`` -- which presets ``use_on_oom=False`` (so the pool is
            never lent to non-symmetric allocations, a torch-side guard against cross-rank allocation
            desync) and ``no_split=True`` (a segment carries a signal pad, so sharing one between two
            tensors is undefined behaviour). Later calls return the SAME object. The pool is stored
            module-level and is **not** cleared by :meth:`cleanup`, because dropping it is precisely
            the failure this guards against.

            Allocating through it is the caller's job and is a plain torch idiom::

                with torch.cuda.use_mem_pool(DistributedManager.symmetric_mempool()):
                    t = torch.empty(shape, dtype=dtype, device=device)

            Note ``torch.cuda.use_mem_pool``'s ``__exit__`` calls ``releasePool``, but entry
            incremented the count first, so a context enter/exit pair is 1 -> 2 -> 1 and never 0.

        Args:
            device: The CUDA device to get the pool for -- a ``torch.device``, an ``int`` index, or
                ``None`` for the current device. **Must be a CUDA device**; symmetric memory has no
                CPU allocator, and a CPU device raises rather than silently returning a pool that
                cannot back a peer-addressable tensor. A device index that does not match the one the
                caller allocates on yields a pool for the WRONG device: the allocation still
                succeeds, but on a different heap than its peers expect, and the resulting rendezvous
                mismatch surfaces as a hang rather than an error.

        Returns:
            ``torch.cuda.MemPool``: the process-wide pool for that device. The same object on every
            call, so ``is`` comparison holds and ``use_count()`` is >= 1 for the process lifetime.

        Raises:
            RuntimeError: if torch's symmetric memory is unavailable in this build, or the device is
                not CUDA.
        """
        import torch.distributed._symmetric_memory as symm_mem

        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        elif isinstance(device, int):
            device = torch.device("cuda", device)
        else:
            device = torch.device(device)
        if device.type != "cuda":
            raise RuntimeError(
                f"symmetric_mempool requires a CUDA device, got {device!r}; symmetric memory has no "
                "CPU allocator and a non-CUDA pool cannot back a peer-addressable tensor"
            )
        index = device.index if device.index is not None else torch.cuda.current_device()
        if index not in _SYMMETRIC_MEMPOOLS:
            # Hold it forever, deliberately. See `_SYMMETRIC_MEMPOOLS`' comment: this reference is
            # what keeps `use_count` above zero and therefore what keeps `nvshmem_free` off the
            # destruction path.
            _SYMMETRIC_MEMPOOLS[index] = symm_mem.get_mem_pool(torch.device("cuda", index))
        return _SYMMETRIC_MEMPOOLS[index]

    @staticmethod
    def cleanup():
        """Tear down torch.distributed and wipe all state. Safe to call more than once.

        **nvshmem is deliberately NOT torn down here.** Finalizing it is a ONE-WAY DOOR per process:
        torch's ``initialize_nvshmem_with_store`` guards itself with a function-local
        ``static bool is_initialized`` that nothing resets (verified in the torch v2.11.0 AND v2.13.0
        sources), so after a finalize the next ``init_nvshmem()`` fails with
        ``RuntimeError: nvshmem_malloc failed`` -- at a call site with no relation to the one that
        caused it. Putting that behind a ``cleanup()`` flag made the wrong thing reachable by
        accident, so the finalize now lives in :meth:`_drain_and_finalize_at_exit`, registered with
        ``atexit`` by :meth:`init_nvshmem`. It runs exactly once, at the only moment it is ever
        correct, whether or not anyone calls this method.

        That is what makes ``cleanup()`` safe PER TEST: it releases everything with a per-test
        lifetime (mesh, groups, process group, manager state) and touches nothing with a
        process lifetime.

        The pre-destroy barrier stays: it gives every rank a chance to drain in-flight symmetric
        traffic before the process group that torch's bootstrap and rendezvous used disappears. It is
        best-effort -- and see :meth:`_barrier_with_timeout` for the measured limits of "bounded"
        here, which are weaker than the name suggests.

        ``_state`` is wiped unconditionally so the next ``initialize()`` starts clean. That clears
        ``_nvshmem_initialized`` while the LIBRARY stays up, which is correct: the module-level
        :data:`_NVSHMEM_LIBRARY_UP` tracks the process-lifetime fact separately, so the exit hook
        still knows there is something to tear down.
        """
        # A BROKEN group means every collective below is a HANG, not a teardown. `BarrierTimeout`'s
        # contract is that its caller must not enter another collective, and both operations here are
        # collective -- the pre-destroy barrier obviously, and `destroy_process_group` because it
        # synchronizes the ranks it is dismantling. Reaching them after a timeout inverts the
        # diagnostic: the message naming the diverging cell is written at session end, and the
        # session never gets there because this finalizer is where the unwind parks.
        #
        # Measured, per-heartbeat, on a wedged pair: rank0 sat in `CollectiveGate.barrier` at 45 s and
        # 90 s, and by 135 s had moved to `cuda_jit_executor.__del__ -> unload` -- i.e. the 90 s bound
        # DID fire and the unwind died in the finalizers, so `pytest.exit`'s text never printed. The
        # instrument was correct and its output was swallowed here.
        #
        # Whoever sets this flag has already reported the desync itself; skipping is therefore losing
        # nothing but a teardown that cannot succeed. Process exit reclaims the group.
        if DistributedManager._state.get("_group_broken"):
            DistributedManager._state = {}
            _DEVICE_MESH_CACHE.clear()
            return

        if DistributedManager._state.get("_group", {}) != {}:
            if torch.distributed.is_initialized():
                # Drain in-flight symmetric traffic before the group that torch's bootstrap and
                # rendezvous used disappears. nvshmem is NOT torn down here -- see the class note
                # above and _drain_and_finalize_at_exit.
                DistributedManager._barrier_with_timeout()

                # RELEASE EVERY nvshmem-REGISTERED CUDA LIBRARY HERE, and the reason is TIMING, not
                # tidiness. `library_finalize` reaches `nvshmemx_culibrary_finalize` ->
                # `cuLibraryGetGlobal`, and from an ATEXIT hook that faults: measured backtrace
                # `___pthread_rwlock_rdlock (rwlock=0x0)` inside libcuda, i.e. CUDA's own state is
                # already gone by then. Same sweep, same registrations, called from HERE instead:
                # 8/8 ranks exit 0. From atexit: 8/8 exit 139.
                #
                # So this is the last moment a finalize is safe -- after the barrier, so every rank
                # is here together, and before the group goes, because the sweep is collective.
                # `_drain_and_finalize_at_exit` deliberately does NOT do it; a caller that never
                # reaches `cleanup()` leaks the registrations, which is a loaded library nobody
                # calls rather than a fault.
                try:
                    from fold_cp_ops.distributed.gemm_bitcode_compile import (
                        finalize_registered_libraries,
                    )

                    finalize_registered_libraries()
                except Exception as e:  # noqa: BLE001 - teardown must not raise
                    warn(f"nvshmem library finalize sweep did not complete (ignored): {e!r}")

                try:
                    torch.distributed.destroy_process_group()
                except Exception as e:  # best-effort teardown
                    warn(f"destroy_process_group() raised during cleanup (ignored): {e!r}")

        # Wipe all state (including nvshmem flags) unconditionally.
        DistributedManager._state = {}
        # And the mesh cache with it. Its `DeviceMesh` objects hold process groups that the
        # `destroy_process_group()` above has just invalidated, so a mesh surviving a teardown would
        # be handed out DEAD to the next `initialize()` in this process. This is the one place the
        # cache differs from `_SYMMETRIC_MEMPOOLS`, which is deliberately process-lifetime.
        _DEVICE_MESH_CACHE.clear()

    @staticmethod
    def _quiesce_nvshmem() -> None:
        """Hold every PE at one point, on the nvshmem channel, before the heap can be torn down.

        Purpose
            ``nvshmem_finalize`` is COLLECTIVE and torch runs it at process exit, where it cannot be
            reached or ordered (torch exposes ``_nvshmemx_cumodule_init`` and no counterpart, and
            nothing on ``_SymmetricMemory`` releases the library). A PE that tears down its queue
            pairs while a peer is still using them faults that peer -- measured on venue B as a
            segfault on a VARYING subset of ranks after every test had already passed. So the last
            synchronization we can control has to happen here.

        Semantics -- why this is the SECOND barrier and not the only one
            ``torch.distributed.barrier`` and ``nvshmem``'s barrier are different channels: the
            first is a host collective over the process group, the second a stream-level collective
            over TEAM_WORLD (every PE, not the group). Draining one says nothing about the other,
            so both are needed -- but they fail very differently:

            * the torch barrier is BOUNDED and best-effort, so a dead or diverged peer costs
              ``CLEANUP_BARRIER_TIMEOUT_S`` and then lets teardown continue.
            * ``barrier_all`` has NO timeout. One rank that never reaches cleanup hangs every rank
              that does, forever, with no diagnostic.

            Hence the ordering in :meth:`cleanup`: the bounded barrier runs first as a GATE, and
            this one runs only if that gate passed. Reaching it means every rank is alive and
            already inside cleanup, which is exactly the condition under which an unbounded
            collective is safe. Reversing the order would replace a bounded, survivable teardown
            with an unconditional hang -- the failure this whole path exists to avoid.

        Input requirements
            Must be called with nvshmem initialized, torch.distributed still alive, and EVERY rank
            about to call it (the gate above establishes that). ``barrier_all`` is tracker-free, so
            it is valid under torch-owned init even though nvshmem4py never ran the bootstrap.
            Errors are swallowed: a best-effort drain must not convert teardown into a crash.

        Returns
            None.
        """
        if not HAS_NVSHMEM4PY:
            return  # nothing to drain through; torch's exit-time finalize is all there is
        try:
            import nvshmem.core

            nvshmem.core.barrier_all(stream=torch.cuda.current_stream())
            torch.cuda.current_stream().synchronize()
        except Exception as e:  # best-effort: never turn teardown into a crash
            warn(f"nvshmem barrier during cleanup did not complete (ignored): {e!r}")

    @staticmethod
    def release_symmetric_mempools() -> None:
        """Release every symmetric ``MemPool`` at a CONTROLLED point, while nvshmem is still up.

        Purpose
            Turn the pool's teardown from something the interpreter does whenever it feels like it
            into one ordered, collective step. Holding the pool forever does NOT avoid the
            collective free -- it defers it to interpreter shutdown, which runs AFTER
            :meth:`_finalize_nvshmem` and therefore fails with ``NVSHMEM API called before NVSHMEM
            initialization has completed`` (measured: every rank exited 255 that way). Deferring an
            unavoidable collective to the least synchronized moment in the process is strictly worse
            than issuing it deliberately.

        Why this is not a contradiction of the "hold the handle" invariant
            The invariant was never "never release". It is "release at exactly ONE point every rank
            reaches together". An implicit release is rank-dependent and deadlocks; this one is
            called from a single site, after a barrier, so every rank issues the same collective at
            the same step. Same collective, opposite timing discipline.

        Semantics
            Drops BOTH references to each pool -- ours and torch's. ``symm_mem.get_mem_pool``
            caches the pool in its own module-level ``_symm_mem_pools``, so clearing only
            ``_SYMMETRIC_MEMPOOLS`` leaves ``use_count >= 1`` and releases nothing at all; the
            segments would still be freed at shutdown and the exit-255 would remain. Reaching into
            that dict is therefore load-bearing, not tidiness. A ``gc.collect()`` follows so the
            now-unreferenced ``MemPool`` objects are destroyed HERE rather than at an arbitrary
            later collection.

            **COLLECTIVE.** Releasing a pool frees its segments, and an NVSHMEM free is collective,
            so every rank must call this or none. Idempotent: a second call finds nothing to do.

        Returns:
            None.

        Raises:
            Nothing. Teardown must not raise; a failure to release is logged by the caller rather
            than propagated, because an exception here would mask whatever the process was already
            shutting down for.
        """
        import gc

        import torch.distributed._symmetric_memory as symm_mem

        # CLEAR TORCH'S CACHE TOO, AND UNCONDITIONALLY -- not just the pools DM minted.
        # `symm_mem.get_mem_pool()` is a public API any caller may reach directly, and a pool
        # obtained that way is invisible to `_SYMMETRIC_MEMPOOLS`. An earlier version returned
        # early when our registry was empty; measured, that left torch's dict holding the pool to
        # interpreter shutdown and every rank still exited 255. Owning the teardown means owning
        # it for EVERY symmetric pool in the process, however it was obtained.
        torch_cache = getattr(symm_mem, "_symm_mem_pools", None)
        _SYMMETRIC_MEMPOOLS.clear()
        if isinstance(torch_cache, dict):
            torch_cache.clear()
        gc.collect()

    @staticmethod
    def _drain_and_finalize_at_exit() -> None:
        """Hold every PE together and tear nvshmem down, ONCE, at process exit.

        Purpose
            The whole teardown fix, in the one place where it is neither optional nor mis-orderable.
            :meth:`init_nvshmem` registers this with ``atexit`` the first time it brings nvshmem up,
            so no caller has to remember it, no caller can ask for it at the wrong moment, and it
            cannot run twice.

        Why it is HERE and not a parameter on :meth:`cleanup`
            An earlier version took ``cleanup(finalize_nvshmem=...)``. That is a footgun: finalizing
            is a ONE-WAY DOOR (torch's ``initialize_nvshmem_with_store`` guards itself with a
            function-local ``static bool is_initialized`` that nothing resets), so a caller who
            passes it on an intermediate teardown does not fail there -- they fail at the NEXT
            ``init_nvshmem()`` with ``RuntimeError: nvshmem_malloc failed``, a message that names
            neither the cause nor the call that caused it. An API whose misuse is silent and
            delayed should not be an API. Exit is the only moment finalizing is ever correct, so it
            is the only moment it happens, and ``cleanup()`` is free to be called per test.

        Semantics
            No-op unless nvshmem is up. When the process group is still alive the BOUNDED torch
            barrier runs first as a gate, exactly as before -- reaching the unbounded ``barrier_all``
            with a dead peer would hang rather than time out. After ``cleanup()`` has destroyed the
            group there is nothing to gate with, so the drain proceeds ungated: ``barrier_all`` is
            over TEAM_WORLD, not over a torch group, and needs no process group.

            Runs during interpreter shutdown, which is BEFORE the C++ static destructors that would
            otherwise issue a collective ``nvshmem_free`` from ``~NVSHMEMAllocation`` -- the actual
            cause of the cross-node segfault. Getting in front of those is the entire point.

        Returns
            None. Every failure is swallowed: an exit hook that raises turns a clean run into a
            confusing one.
        """
        if not _NVSHMEM_LIBRARY_UP:
            return
        try:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                if not DistributedManager._barrier_with_timeout():
                    return  # a peer is gone; do NOT enter the unbounded barrier below
            # BEFORE the quiesce/finalize, and AFTER the barrier above: this is the one moment
            # where every rank has arrived AND nvshmem is still up, so it is the only correct place
            # to issue the pools' collective free. Leaving it to interpreter shutdown puts it after
            # _finalize_nvshmem and every rank exits 255 (measured).
            # SWEEP FIRST, POOLS SECOND, AND THE ORDER IS THE FIX. `nvshmemx_culibrary_finalize`
            # (nvshmem `src/host/init/init.cu:2177`) dereferences the caller's CUlibrary handle as
            # its FIRST statement, unvalidated:
            #
            #     cuLibraryGetGlobal(&dptr, &size, library, "nvshmemi_device_state_d")
            #
            # and `release_symmetric_mempools()` INVALIDATES those handles. Measured on both sides
            # of that call, same hook, same four registrations, 8 GPUs:
            #
            #     before the pool release   cuCtxGetGetCurrent OK, all 4 handles CUDA_SUCCESS
            #     after  the pool release   handle[0] OK, handle[1] SEGFAULTS
            #     sweep moved above it      all 4 finalizes succeed, zero segfaults
            #
            # The surviving handle is the fresh compile; the ones that die are the artifact-cache
            # HITs, which is why `CPO_JIT_ARTIFACT_ENABLED=0` makes the fault vanish. An earlier
            # draft of this comment blamed "CUDA state is gone at atexit" -- that is WRONG and was
            # refuted by probing the driver here: `cuCtxGetCurrent` succeeds and every handle
            # resolves. Our own pool release is what breaks them, not interpreter shutdown.
            #
            # BACKSTOP, not the primary. `cleanup()` sweeps while the group is up, which is where a
            # well-behaved caller finalizes. This covers the caller that never reaches `cleanup()`,
            # and it is idempotent -- `cleanup()` empties both registries, so this finds nothing.
            try:
                from fold_cp_ops.distributed.gemm_bitcode_compile import (
                    finalize_registered_libraries,
                )

                finalize_registered_libraries()
            except Exception as e:  # noqa: BLE001 - an exit hook must not raise
                warn(f"nvshmem library finalize backstop did not complete (ignored): {e!r}")
            # QUIESCE BEFORE THE POOLS, and the pools LAST before finalize. Measured on venue E at
            # 16 ranks over 2 nodes: with the pool release ahead of the quiesce, the fault MOVED off
            # `library_finalize` and onto
            #
            #     nvshmem/core/collective.py:307 in barrier_all
            #     distributed_manager.py:...    in _quiesce_nvshmem
            #
            # i.e. `release_symmetric_mempools()` breaks more than the CUlibrary handles -- it also
            # breaks the transport state `barrier_all` needs. On ONE node that barrier is local and
            # survives it, which is why 8 ranks on one host stayed green through two wrong orderings.
            #
            # So the rule is not "sweep before pools", it is: EVERYTHING THAT STILL TALKS TO NVSHMEM
            # RUNS FIRST, and the pool release is the last act before the library goes.
            DistributedManager._quiesce_nvshmem()
            DistributedManager.release_symmetric_mempools()
            DistributedManager._finalize_nvshmem()
        except Exception as e:  # noqa: BLE001 - never let an exit hook raise
            warn(f"nvshmem exit teardown did not complete (ignored): {e!r}")

    @staticmethod
    def _finalize_nvshmem() -> None:
        """Tear the nvshmem library down HERE, at a synchronized point, instead of at process exit.

        Purpose
            This is the fix for the cross-node teardown segfault, and it is the ONLY thing measured
            to work.

        Mechanism -- read from the torch v2.11.0 source, because the obvious guess is WRONG.
            **torch never finalizes nvshmem at all**: there is no ``nvshmem_finalize`` /
            ``hostlib_finalize`` call anywhere under ``torch/csrc`` or ``torch/distributed``. What
            runs at exit instead is ``~NVSHMEMAllocation`` (``NVSHMEMSymmetricMemory.cu``), which
            calls **``nvshmem_free`` -- itself a COLLECTIVE** -- for every allocation still held in
            the allocator's ``allocations_`` map. Its only guard is::

                ~NVSHMEMAllocation() {
                  if (is_finalizing()) return;   // avoid CUDA calls after driver shutdown
                  nvshmem_free(ptr);
                }

            and ``is_finalizing_`` is set by ``~AllocatorMap()`` -- another STATIC destructor, whose
            order relative to this one is unspecified across translation units. So depending on
            teardown order a rank either skips the free or issues a collective one, at process exit,
            with no barrier and no agreement between ranks. Over IB that is what faults.

            Finalizing here, early and in lockstep, takes the library down before any of that can
            run. (Why the later ``nvshmem_free`` is then harmless is NOT established from source --
            most likely it no-ops against a finalized library. Measured harmless, not proven so.)

        Evidence (venue B, 2 nodes x 8 GPUs, minimal reproducer: init + one symmetric allocation +
        exit, no pytest involved):

            * let torch finalize at exit  -> **16/16 ranks segfault** (rc=139)
            * barrier + finalize here     -> **16/16 ranks exit 0**

            Barriers ALONE are not enough, tried twice: at fixture teardown (a recorded job, 4 ranks
            still faulted) and at ``atexit`` (a recorded job, 2 still faulted). The finalize itself is
            what is unordered, so no barrier PLACEMENT fixes it -- the finalize has to move to where
            a barrier can precede it, which is here.

        Semantics
            Called only after :meth:`_quiesce_nvshmem` has put every PE at the same point, so the
            collective is entered in lockstep. Uses ``hostlib_finalize`` (the raw binding): it is
            valid under torch-owned init even though nvshmem4py never ran the bootstrap, and after
            it returns torch's own exit-time teardown does not double-fault (verified). Clears
            :data:`_NVSHMEM_LIBRARY_UP`, because the library's process-lifetime has just ended --
            leaving it set would make a later ``cleanup()`` finalize a library that is already gone.

            **Applied unconditionally, not gated on topology.** The fault only appears with IB peers
            (NVLink-only runs are clean either way), but finalizing early is harmless there --
            measured clean on 8xH200 with ``ib_peers=0`` -- and a teardown path that differs by
            interconnect is one that only ever gets tested on one of them.

        Input requirements
            nvshmem must be up and every rank must be about to call this, in the same order. Calling
            it with a peer absent hangs, which is why the bounded torch barrier gates the whole
            sequence. Errors are swallowed: teardown must not become a crash.

        Returns
            None.
        """
        global _NVSHMEM_LIBRARY_UP
        if not HAS_NVSHMEM4PY:
            return
        try:
            import nvshmem.core

            # SILENCE ONE EXACT, ALWAYS-FALSE WARNING -- and only that one.
            #
            # `core.finalize()` calls `memory._free_all_buffers()`, which gates on NVSHMEM4PY's own
            # init flag (`_is_initialized`, set ONLY inside `nvshmem.core.init()`). We never call
            # that: TORCH bootstraps the library and nvshmem4py is read-only here (my_pe/n_pes). So
            # the flag is permanently UNINITIALIZED, the sweeper bails, and every rank prints
            #     "NVSHMEM Library is not initialized. Cannot free buffers"
            # on EVERY run -- baseline included -- while the library is in fact up and being torn
            # down correctly on the next line. The message is false about the library and true only
            # about nvshmem4py's view of it.
            #
            # It is suppressed because it is not merely noise: it cost this project real time twice.
            # It was read as evidence of deferred reclaim once (recorded as "do not carry it into
            # the mechanism story"), and later used as a status flag that reported STILL-255 on runs
            # that had exited 0.
            #
            # ** UN-SUPPRESS THIS THE MOMENT WE ALLOCATE VIA nvshmem4py ** (e.g. `nvshmem_cute.tensor`).
            # `_free_all_buffers()` is a LEAK SWEEPER over `_mr_references`. It is vacuous today
            # because torch owns every allocation, so nothing is lost by skipping it. Once nvshmem4py
            # holds allocations, that skip hides a real leak check and this exact line becomes the
            # only symptom -- so filtering it would convert a genuine defect into silence.
            #
            # EXACT match, scoped to this call, restored in `finally`: any OTHER nvshmem warning,
            # including a future change to this one's wording, still reaches the operator.
            _nvshmem_log = logging.getLogger("nvshmem")
            _silence = lambda rec: (
                rec.getMessage() != ("NVSHMEM Library is not initialized. Cannot free buffers")
            )
            _nvshmem_log.addFilter(_silence)
            try:
                nvshmem.core.finalize()
            finally:
                _nvshmem_log.removeFilter(_silence)
        except Exception as e:
            # Fall back to the raw binding. core.finalize() is a THIN wrapper -- it does
            # `_free_all_buffers()` + `bindings.hostlib_finalize()` + reset nvshmem4py's own init
            # flag -- so the LIBRARY teardown is the identical call and the fallback loses only the
            # bookkeeping. Worth attempting the wrapper first anyway: it frees buffers NVSHMEM4PY
            # allocated, and while torch owns every allocation today (so that tracker is empty and
            # the free is a no-op), the A2A path uses nvshmem4py's device API and may allocate
            # through it -- finalizing over live tracked buffers is how that leaks or faults.
            warn(f"nvshmem.core.finalize() raised ({e!r}); falling back to hostlib_finalize()")
            try:
                nvshmem.bindings.hostlib_finalize()
            except Exception as e2:  # best-effort: never turn teardown into a crash
                warn(f"nvshmem hostlib_finalize() during cleanup raised (ignored): {e2!r}")
        finally:
            _NVSHMEM_LIBRARY_UP = False

    @staticmethod
    def _barrier_with_timeout(timeout_s: float = CLEANUP_BARRIER_TIMEOUT_S) -> bool:
        """Best-effort, GENUINELY timeout-bounded barrier for cleanup.

        **``torch.distributed.barrier(timeout=...)`` does not bound anything on NCCL.** Measured:
        with one rank deliberately skipping cleanup, the other sat in a ``timeout=10s`` barrier for
        over 150 s and never raised -- no warning, no return. The NCCL barrier enqueues a device
        collective and blocks on the stream; the per-call ``timeout`` feeds the watchdog (minutes,
        and ``initialize()`` sets ``TORCH_NCCL_ASYNC_ERROR_HANDLING=0`` for nccl, which disables even
        that), not the call. So the inherited implementation's claim to be an "anti-pattern fix vs
        the reference baseline's unbounded barrier" was not true: it was unbounded in a different way.

        Issuing it ``async_op=True`` and POLLING ``is_completed()`` against a monotonic deadline --
        what :class:`CollectiveGate` does -- is strictly better, so this delegates there rather than
        growing a second copy of that logic. **But measure what it buys before trusting it: it does
        NOT rescue a dead-peer teardown either.** With one rank exiting outright, rank 0 sat inside
        the gate for >90 s and never reached the poll loop, because ``dist.barrier(async_op=True)``
        blocks establishing the NCCL communicator with a peer that is already gone.

        So the honest contract is: **this bounds nothing when a peer has DIED.** Its value is
        ordering the two channels in the normal, symmetric case (every rank present -- measured
        0.6 s), and returning ``False`` on the errors it can observe. Genuinely bounding a dead-peer
        teardown needs a side channel that honours timeouts (a gloo process group), which is not
        wired up here. Treat a hung cleanup as a dead peer and kill the job -- the outer ``timeout``
        every launcher is required to carry is what actually catches this today.

        Args:
            timeout_s: Seconds to wait. Defaults to ``CLEANUP_BARRIER_TIMEOUT_S``
                (``CPO_DIST_CLEANUP_TIMEOUT_S``, 30 s).

        Returns:
            ``True`` if every rank reached the barrier, ``False`` if it timed out or errored. The
            caller uses this as a GATE: ``False`` means some peer is dead or diverged, and the
            unbounded ``barrier_all`` in :meth:`_quiesce_nvshmem` must then be SKIPPED, since it
            would hang forever rather than merely time out.
        """
        from fold_cp_ops.distributed.collective_symmetry import BarrierTimeout, CollectiveGate

        try:
            CollectiveGate().barrier(timeout_s)
        except BarrierTimeout as e:
            warn(f"cleanup barrier did not complete within {timeout_s}s (ignored): {e!r}")
            return False
        except Exception as e:  # best-effort: do not let a hung peer wedge cleanup
            warn(f"cleanup barrier raised (ignored): {e!r}")
            return False
        return True

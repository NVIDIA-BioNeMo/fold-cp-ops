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
"""ComposableEpiMixin: composes EpiOps into epilogue hook methods.

Subclasses declare _epi_ops as a tuple of EpiOp instances. The mixin auto-generates
epi_smem_bytes_per_stage, epi_get_smem_struct, epi_get_smem_tensors, epi_begin,
epi_begin_loop, epi_end, and EpilogueParams by querying each op.

epi_begin and epi_begin_loop return dicts keyed by op name, so epi_visit_subtile
can access values by name (e.g. epi_loop_tensors["alpha"]).

EpilogueParams is auto-generated from _epi_ops (via param_fields()) plus any
_extra_param_fields declared on the subclass. Subclasses still define
EpilogueArguments and epi_to_underlying_arguments manually.
"""

from dataclasses import make_dataclass, MISSING

import cutlass.cute as cute
from cutlass import const_expr

from fold_cp_ops._internal.epi_ops import EpiContext, Scalar


def _compute_smem_map(ops):
    """Pre-compute name → smem tensor index for each non-Scalar op."""
    smem_map = {}
    idx = 0
    for op in ops:
        if not isinstance(op, Scalar):
            smem_map[op.name] = idx
            idx += 1
    return smem_map


def _make_epi_params(epi_ops, extra_fields, bases):
    """Build EpilogueParams dataclass from epi_ops + extra fields.

    Required fields (default=MISSING) are placed first, then optional fields.
    """
    required, optional = [], []
    for op in epi_ops:
        for name, typ, default in op.param_fields():
            (required if default is MISSING else optional).append((name, typ, default))
    for name, typ, default in extra_fields:
        (required if default is MISSING else optional).append((name, typ, default))
    fields = [(n, t) for n, t, _ in required] + [(n, t, d) for n, t, d in optional]
    return make_dataclass("EpilogueParams", fields, bases=bases)


class ComposableEpiMixin:
    """Base mixin that composes EpiOps into the standard epilogue hooks."""

    _epi_ops = ()
    _extra_param_fields = ()  # [(name, type, default), ...] for non-op params (e.g. act_fn)
    _epi_param_bases = ()  # Base classes for EpilogueParams (e.g. (ParamsBase,))
    _epi_smem_map = {}
    _epi_has_async_ops = False

    def __init_subclass__(cls, **kwargs):
        """Derive the per-class epilogue plumbing from the subclass's ``_epi_ops`` declaration.

        Runs once, at class-creation time, so everything it computes is a compile-time constant of
        the class rather than something recomputed per launch.

        Args:
            cls: The subclass being created. Its ``_epi_ops`` (and optionally
                ``_extra_param_fields`` / ``_epi_param_bases``) are read; a subclass that declares
                none inherits the empty defaults and gets no generated params.
            **kwargs: Forwarded to ``super().__init_subclass__``.

        Returns:
            None; ``cls`` is mutated in place.

        Note:
            ``EpilogueParams`` is generated ONLY when the subclass does not define one in its own
            ``__dict__``. Inheriting one from a base does not count -- a subclass that changes
            ``_epi_ops`` must get a params struct matching its own ops, and silently reusing the
            parent's would mismatch the fields the ops write.
        """
        super().__init_subclass__(**kwargs)
        if cls._epi_ops:
            cls._epi_smem_map = _compute_smem_map(cls._epi_ops)
            cls._epi_has_async_ops = any(op.needs_async_fence() for op in cls._epi_ops)
            # Auto-generate EpilogueParams if not explicitly defined on this class
            if "EpilogueParams" not in cls.__dict__:
                cls.EpilogueParams = _make_epi_params(
                    cls._epi_ops, cls._extra_param_fields, cls._epi_param_bases
                )

    # --- Host-side: args → params ---

    def _epi_ops_to_params_dict(self, args):
        """Merge each op's to_params into a single dict. Subclasses call this,
        add custom fields, then construct self.EpilogueParams(**d)."""
        d = {}
        for op in self._epi_ops:
            d.update(op.to_params(self, args))
        return d

    # --- Host-side: smem allocation (queried from ops) ---

    @classmethod
    def epi_smem_bytes_per_stage(cls, args, cta_tile_shape_mnk, epi_tile):
        """Total SMEM one epilogue pipeline stage needs, summed over the declared ops.

        Called during ``_compute_stages`` to decide how many epilogue stages fit, so an
        underestimate here is not a tight fit -- it is a SMEM overrun at launch.

        Args:
            args: The ``EpilogueArguments``. Each op is asked for its own footprint given its own
                argument, so an absent (None) term contributes zero.
            cta_tile_shape_mnk: The CTA tile, which sizes the ops that stage a full tile.
            epi_tile: The epilogue subtile, which sizes the ops that stage one subtile.

        Returns:
            Bytes per stage.
        """
        return sum(
            op.smem_bytes(getattr(args, op.name, None), cta_tile_shape_mnk, epi_tile)
            for op in cls._epi_ops
        )

    def epi_get_smem_struct(self, params):
        """Build the SMEM struct type holding every op's staging buffer.

        Args:
            params: The traced ``EpilogueParams``, which decides per op whether it needs SMEM at
                all -- a term compiled out contributes no field.

        Returns:
            A ``cute.struct`` type. When no op needs SMEM it carries a single zero-length
            ``_epi_empty`` member: a struct with no fields at all is not a valid type, and the
            default epilogue (which loads its broadcast vectors GMEM -> registers directly) is
            exactly that case.
        """
        fields = {}
        for op in self._epi_ops:
            result = op.smem_struct_field(self, params)
            if result is not None:
                name, ftype = result
                fields[name] = ftype
        if not fields:
            # No epilogue op needs SMEM (e.g. default epi loads vecs gmem->regs directly).
            fields["_epi_empty"] = cute.struct.MemRange[cute.Int32, 0]
        EpiSharedStorage = type("EpiSharedStorage", (), {"__annotations__": fields})
        return cute.struct(EpiSharedStorage)

    def epi_get_smem_tensors(self, params, storage):
        """Slice each non-``Scalar`` op's tensor out of the allocated SMEM struct.

        Args:
            params: The traced ``EpilogueParams``.
            storage: The kernel's shared-storage object; ``storage.epi`` is the struct
                ``epi_get_smem_struct`` described.

        Returns:
            One tensor per non-``Scalar`` op, **in declaration order**. That order is the contract:
            ``_epi_smem_map`` indexes into this tuple by name, and reordering ``_epi_ops`` without
            regenerating the map hands an op another op's buffer.
        """
        return tuple(
            op.get_smem_tensor(self, params, storage.epi)
            for op in self._epi_ops
            if not isinstance(op, Scalar)
        )

    def epi_get_tma_atoms(self, params, *, loc=None, ip=None):
        """Collect every TMA atom the declared ops need, for the kernel's descriptor prefetch.

        Args:
            params: The traced ``EpilogueParams``.
            loc: Optional DSL source location.
            ip: Optional DSL insertion point.

        Returns:
            A flat list of atoms, possibly empty. Ops that load through ``cp.async`` or plain
            vector loads contribute none.
        """
        atoms = []
        for op in self._epi_ops:
            atoms.extend(op.tma_atoms(self, params))
        return atoms

    # --- Device-side: kernel execution (delegates to ops) ---

    @cute.jit
    def epi_begin(
        self,
        params,
        epi_smem_tensors,
        epi_tile,
        tiled_copy_t2r,
        tiled_copy_r2s,
        tile_coord_mnkl,
        epilogue_barrier,
        tidx,
        tile_idx=None,
    ):
        """Run every op's per-work-tile setup and return their results keyed by op name.

        The dict return is what lets ``epi_visit_subtile`` write ``epi_loop_tensors["alpha"]``
        instead of unpacking a positional tuple whose order it would have to track.

        Args:
            params: The traced ``EpilogueParams``.
            epi_smem_tensors: The tuple from ``epi_get_smem_tensors``, indexed via
                ``_epi_smem_map``.
            epi_tile: The epilogue subtile shape.
            tiled_copy_t2r: Tensor-memory-to-register copy, or None on SM90.
            tiled_copy_r2s: Register-to-shared copy for the epilogue.
            tile_coord_mnkl: This work tile's coordinate.
            epilogue_barrier: The named barrier all epilogue warps arrive at. Every participating
                thread must reach this call, because the async fence below is collective.
            tidx: Thread index within the CTA.
            tile_idx: Optional linear work-tile index, for ops that need it.

        Returns:
            ``{op.name: op.begin(...)}``.

        Note:
            When any declared op issues ``cp.async``, this commits and waits for the group and then
            arrives at ``epilogue_barrier`` -- once for all such ops, rather than per op. That is
            why the barrier is a parameter and why partial participation deadlocks.
        """
        ctx = EpiContext(
            self,
            epi_tile,
            tiled_copy_t2r,
            tiled_copy_r2s,
            tile_coord_mnkl,
            epilogue_barrier,
            tidx,
            tile_idx,
        )
        smem_map = self._epi_smem_map
        results = {
            op.name: op.begin(
                self,
                getattr(params, op.name, None),
                epi_smem_tensors[smem_map[op.name]] if op.name in smem_map else None,
                ctx,
            )
            for op in self._epi_ops
        }
        if const_expr(self._epi_has_async_ops):
            has_async_data = any(
                getattr(params, op.name, None) is not None
                for op in self._epi_ops
                if op.needs_async_fence()
            )
            if const_expr(has_async_data):
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
                epilogue_barrier.arrive_and_wait()
        return results

    def epi_begin_loop(self, params, epi_tensors, epi_coord):
        """Run every op's per-SUBTILE setup, keyed by op name.

        Args:
            params: The traced ``EpilogueParams``.
            epi_tensors: The dict ``epi_begin`` returned for this work tile.
            epi_coord: This subtile's coordinate within the work tile.

        Returns:
            ``{op.name: op.begin_loop(...)}`` -- the values ``epi_visit_subtile`` combines.
        """
        return {
            op.name: op.begin_loop(self, epi_tensors[op.name], epi_coord) for op in self._epi_ops
        }

    @cute.jit
    def epi_end(
        self,
        params,
        epi_tensors,
        epi_tile,
        tiled_copy_t2r,
        tiled_copy_r2s,
        tile_coord_mnkl,
        tidx,
    ):
        """Run every op's per-work-tile teardown, in declaration order.

        Args:
            params: The traced ``EpilogueParams``.
            epi_tensors: The dict ``epi_begin`` returned.
            epi_tile: The epilogue subtile shape.
            tiled_copy_t2r: Tensor-memory-to-register copy, or None on SM90.
            tiled_copy_r2s: Register-to-shared copy.
            tile_coord_mnkl: This work tile's coordinate.
            tidx: Thread index within the CTA.

        Returns:
            None. Ops that need no teardown -- which is most of them -- do nothing here.
        """
        for op in self._epi_ops:
            op.end(
                self,
                getattr(params, op.name, None),
                epi_tensors[op.name],
                epi_tile,
                tiled_copy_t2r,
                tiled_copy_r2s,
                tile_coord_mnkl,
                tidx,
            )

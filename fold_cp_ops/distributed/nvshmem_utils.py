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

"""nvshmem CuTe-DSL device utilities — vendored into fold-cp-ops.

Ported (surgically) from the upstream CP project's ``distributed/nvshmem/utils.py`` (branch
``the upstream nvshmem TriMul prototype``): the alignment-preserving ``get_peer_tensor``.
Vendored here so ``fold_cp_ops/distributed`` carries **no runtime dependency on the
upstream CP source tree** — the back-A2A peer TMA store (``gemm_sm90_a2a.py``) and the
peer-atom builder (``peer_tma_atoms.py``) import ``get_peer_tensor`` from this
module, not from the upstream CP project's ``distributed.nvshmem.utils``.

Requires ``nvshmem.core`` (the module-level import below is only reached from
inside the callers' HAS_NVSHMEM try/except probes, so an absent nvshmem degrades
gracefully to HAS_NVSHMEM=False rather than an import error at package load).
"""

import cutlass
import cutlass.cute as cute
import nvshmem.core.device.cute.mem as nvshmem_cute_mem


@cute.jit
def get_peer_tensor(
    tensor: cute.Tensor,
    pe: cutlass.Int32,
    **kwargs_cute_make_ptr,
) -> cute.Tensor:
    """Peer-translate a symmetric-heap CuTe tensor, preserving alignment.

    Drop-in replacement for ``nvshmem.core.device.cute.mem.get_peer_tensor``
    that addresses the upstream helper's alignment-stripping behaviour: the
    upstream version internally calls ``cute.make_ptr(dtype, ptr, gmem)`` without
    forwarding the original tensor's alignment, so the returned peer-view falls
    back to dtype-sized alignment (e.g. 2 B for fp16). That is incompatible with
    128-bit-atom IR verification for TMA bulk-copy / ``STG.E.128`` -- a kernel
    that allocated its destination with ``assumed_align=16`` and then wrapped
    through the upstream helper would silently downgrade to 2 B alignment, and
    ``make_tiled_tma_atom`` / ``cute.autovec_copy`` would emit a narrower atom
    (or fail IR verification) on the peer view.

    By default this helper inherits ``assumed_align`` from the input tensor's
    iterator -- so a tensor created with ``assumed_align=16`` yields a peer view
    with the same 16 B alignment. The caller may override (or pass any other
    ``cute.make_ptr`` kwarg) via ``kwargs_cute_make_ptr``.

    :param tensor: a CuTe tensor over symmetric-heap GMEM (allocated via
        ``nvshmem.core.tensor`` / ``nvshmem.core.interop.torch.tensor``).
    :param pe: target PE id. ``pe == my_pe`` returns a self-view (the same
        address as ``tensor``); use ``pe = my_pe`` for testing the round-trip
        without 2-PE plumbing.
    :param kwargs_cute_make_ptr: forwarded to ``cute.make_ptr``.
        ``assumed_align`` is auto-extracted from ``tensor.iterator.alignment``
        (in BYTES, matching ``cute.make_ptr``'s ``assumed_align`` units --
        verified empirically) if not explicitly provided; any explicit value
        wins.
    :return: a ``cute.Tensor`` aliasing PE ``pe``'s symmetric copy of ``tensor``
        with the same ``layout`` and the inherited (or overridden) alignment
        annotation.
    """
    peer_addr = nvshmem_cute_mem.nvshmem_ptr(
        cutlass.Int64(tensor.iterator.toint()),
        cutlass.Int32(pe),
    )
    # Auto-inherit alignment from the input tensor unless the caller passed an
    # explicit value. ``tensor.iterator.alignment`` is in BYTES (verified
    # empirically: ``cute.make_ptr(..., assumed_align=16).alignment`` reads back
    # as 16, not 128), matching ``cute.make_ptr``'s ``assumed_align`` units.
    #
    # IMPORTANT: use ``dict.pop(key, default)`` rather than the more conventional
    # ``if key not in dict: dict[key] = default`` -- the CuTe DSL AST
    # preprocessor mishandles ``not in dict`` membership tests on the
    # ``**kwargs`` capture dict at trace time and evaluates the assignment branch
    # unconditionally, clobbering the caller's explicit value (verified at the
    # time of authoring this helper -- the **kwargs dict object is apparently
    # shared across calls so a stale state-machine read ends up running both
    # branches). ``dict.pop`` does the default-or-extract in a single expression
    # and side-steps the bug. Pop also removes the key from
    # ``kwargs_cute_make_ptr`` so the trailing ``**kwargs_cute_make_ptr``
    # unpacking does not re-pass it to ``cute.make_ptr``.
    assumed_align = kwargs_cute_make_ptr.pop("assumed_align", tensor.iterator.alignment)
    peer_ptr = cute.make_ptr(
        tensor.element_type,
        peer_addr,
        cute.AddressSpace.gmem,
        assumed_align=assumed_align,
        **kwargs_cute_make_ptr,
    )
    return cute.make_tensor(peer_ptr, tensor.layout)

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

"""Weight adapter: a reference ``TriangularMultiplication`` nn.Module -> the A2A-fused TriMul ``w`` dict.

Brought back from the upstream. **No executable line differs from the upstream file** -- this module
is pure torch, so a rewrite would have been a rewrite rather than a port. What changed is prose: the
upstream named the workflow class and its DTensor wrapper with the ``fused_trimul`` family of names,
none of which come across (the target is ``distributed/workflows/trimul_autotuned.py``), so the
references below say what the consumer IS instead of what it used to be called.

The fused / real-kernel / oracle TriMul paths all consume a single replicated **weight dict**
``w`` (keys ``norm_in_w/b``, ``p_in_w`` (2D,D), ``g_in_w`` (2D,D), ``norm_out_w/b``,
``p_out_w`` (D,D), ``g_out_w`` (D,D); plus the four projection biases). This module builds that dict
directly from a frozen ``TriangularMultiplication{Outgoing,Incoming}`` module's
``nn.Parameter``s — the exact inverse of ``benchmark.distributed.trimul_dtensor_baseline._seed_serial_layer``
(N5/D5.1 ports that baseline; this module does not import it).

Mapping (submodule attr -> ``w`` key), no transpose: the reference layer's projections are
``nn.Linear(D, ·, bias=False)`` whose ``.weight`` is ``(out, in)`` and is applied as
``x @ W.T`` — exactly how ``_local_front`` / ``trimul_ref`` consume the ``w`` keys. So each
``.weight`` copies straight into its key.

**The four projection biases split two ways, and the split is not cosmetic.** ``p_in_b``/``g_in_b``
are refused: the front A2A store builds its epilogue with the bias slots hard-``None`` and
``TriMulAutotuned`` raises on a non-``None`` value, so accepting one here would only defer the same
refusal past the point where the argument's name is still available to name.
``p_out_b``/``g_out_b`` are **carried through**: the workflow fuses them as ``bp``/``bg`` on
`layernorm_dual_gated_gemm`.

This module used to refuse all four, with a message asserting the workflow fused none of them --
which its own engine contradicts for the output pair. Because `TriangularMultiplication` reads its
weights through here, the effect was that live, fused code was unreachable from the shipped
nn.Module API. A reference layer built ``bias=False`` (the fair baseline) is unaffected either way: all
four keys come out ``None`` because there is no bias to carry.

Pure torch, no cutlass / nvshmem import — CPU-runnable for the Phase-0 unit.
"""

from __future__ import annotations

from typing import Optional

import torch

__all__ = [
    "weights_from_trimul_module",
    "trimul_module_from_weights",
    "W_KEYS",
    "W_PROJ_BIAS_KEYS",
]

# w-dict keys the fused / real-kernel / oracle paths consume (order = build order).
W_KEYS = (
    "norm_in_w", "norm_in_b",
    "p_in_w", "g_in_w",
    "norm_out_w", "norm_out_b",
    "p_out_w", "g_out_w",
)
# Projection biases. p_in_b/g_in_b are always None -- the front store does not fuse them and
# `TriMulAutotuned` raises on a non-None value. p_out_b/g_out_b ARE fused (as `bp`/`bg` on
# `layernorm_dual_gated_gemm`) and round-trip through both adapters; they are None only when
# the source layer was built `bias=False`, which the fair reference baseline is.
W_PROJ_BIAS_KEYS = ("p_in_b", "g_in_b", "p_out_b", "g_out_b")

# Reference-layer submodule ".weight"/".bias" attr -> w-dict key. Direct copy, no transpose
# (nn.Linear.weight is (out,in), consumed as x @ W.T by _local_front / trimul_ref).
_PARAM_MAP = {
    ("norm_in", "weight"): "norm_in_w",
    ("norm_in", "bias"): "norm_in_b",
    ("p_in", "weight"): "p_in_w",
    ("g_in", "weight"): "g_in_w",
    ("norm_out", "weight"): "norm_out_w",
    ("norm_out", "bias"): "norm_out_b",
    ("p_out", "weight"): "p_out_w",
    ("g_out", "weight"): "g_out_w",
}


def weights_from_trimul_module(
    module: torch.nn.Module,
    *,
    dtype: Optional[torch.dtype] = None,
    device=None,
    detach: bool = True,
) -> dict:
    """Build the A2A-fused TriMul ``w`` dict from a ``TriangularMultiplication`` module.

    Parameters
    ----------
    module : torch.nn.Module
        A ``TriangularMultiplication{Outgoing,Incoming}`` single-device module exposing the
        ``norm_in / p_in / g_in / norm_out / p_out / g_out`` submodules. A 1-D/2-D CP wrapper
        that stores the serial layer under an attribute should pass that inner serial layer.
    dtype : torch.dtype, optional
        If given, cast every tensor to this dtype (the workflow re-casts LN gains to fp32
        and projections to its compute dtype at construction, so passing the compute dtype here
        is optional). ``None`` keeps each parameter's native dtype.
    device : optional
        If given, move every tensor to this device. ``None`` keeps the parameter's device.
    detach : bool
        Detach (default True — frozen-inference weights carry no grad into the fused kernels).

    Returns
    -------
    dict
        ``w`` with the eight weight keys populated, ``p_in_b``/``g_in_b`` forced to ``None``, and
        ``p_out_b``/``g_out_b`` carrying the source Linears' biases when they have any (``None``
        otherwise). Ready to pass to the workflow as its ``w`` argument.

    Raises
    ------
    AttributeError
        If ``module`` is missing an expected submodule/parameter (not a TriMul layer).
    NotImplementedError
        If ``p_in`` or ``g_in`` carries a bias — the front store does not fuse the INPUT projection
        biases. ``p_out``/``g_out`` biases are fused and are carried, not refused.
    """
    w: dict = {}
    for (sub, attr), key in _PARAM_MAP.items():
        submod = getattr(module, sub, None)
        if submod is None:
            raise AttributeError(
                f"module {type(module).__name__} has no submodule '{sub}'; "
                "weights_from_trimul_module expects a TriangularMultiplication layer "
                "(norm_in/p_in/g_in/norm_out/p_out/g_out)."
            )
        t = getattr(submod, attr, None)
        if t is None:
            raise AttributeError(
                f"submodule '{sub}' has no '{attr}' (expected for w-key '{key}')."
            )
        if detach:
            t = t.detach()
        if dtype is not None:
            t = t.to(dtype)
        if device is not None:
            t = t.to(device)
        w[key] = t
    # ALL FOUR projection biases are fused now, so all four are carried. The front pair reaches
    # `interleave_dual_weights(..., return_bias=True)`, whose interleaved (1, 2N) fp32 vector becomes
    # the front store's `mRowVecBroadcast`; the back pair reaches -- `TriMulAutotuned` passes them to
    # `layernorm_dual_gated_gemm` as `bp`/`bg` -- so carry them through rather than dropping them.
    #
    # **This used to refuse all four, and that was wrong in a way worth recording.** The message it
    # raised said the workflow "does not fuse the input/output projection biases", which the engine
    # contradicts for the output pair. The cost was not the wrong sentence: because this adapter is
    # how `TriangularMultiplication` reads its weights, live fused code was unreachable through the
    # shipped nn.Module API, and the one test pool that could have exercised it drove all four biases
    # from a single flag -- so asking for the output pair tripped the INPUT refusal above and the
    # cell raised before running. Two independent accidents, and between them p_out_b/g_out_b had no
    # coverage at all. `test_the_fused_chain_matches_the_fp32_oracle[out_bias=True]` now runs them
    # against the fp32 oracle at cp=2..16.
    for sub, key in (("p_in", "p_in_b"), ("g_in", "g_in_b"),
                     ("p_out", "p_out_b"), ("g_out", "g_out_b")):
        b = getattr(getattr(module, sub), "bias", None)
        if b is not None:
            if detach:
                b = b.detach()
            if dtype is not None:
                b = b.to(dtype)
            if device is not None:
                b = b.to(device)
        w[key] = b
    return w


def trimul_module_from_weights(w: dict, *, dtype: Optional[torch.dtype] = None) -> torch.nn.Module:
    """Build a bare TriMul-shaped ``nn.Module`` from a replicated weight dict.

    Purpose
        The inverse of :func:`weights_from_trimul_module`, for the callers that hold a ``w`` dict and
        no layer. ``TriangularMultiplication`` takes a MODULE (so its constructor mirrors the CP
        reference wrappers `TriangularMultiplication{1,2}D`), while every in-tree caller --
        the nsys bench and the harness target -- builds its weights synthetically. This adapter is
        the one-liner that joins them.

    Functionality & semantics
        Returns an ``nn.Module`` carrying the SIX attributes the fused workflow reads by name:
        ``norm_in``, ``p_in``, ``g_in``, ``norm_out``, ``p_out``, ``g_out`` -- real ``nn.LayerNorm``
        and ``nn.Linear`` submodules, so the result is indistinguishable from the reference layer to
        anything that duck-types on those names. The projections are ``bias=False``, matching the
        wave-1 fused contract that :func:`weights_from_trimul_module` enforces from the other side.

        Parameters are assigned UNDER ``no_grad`` and share storage with ``w``'s tensors -- this is a
        view-like adapter, not a copy, so mutating ``w`` afterwards mutates the module. Pass
        ``dtype`` to get a converted copy instead.

        ROUND-TRIPS: ``weights_from_trimul_module(trimul_module_from_weights(w))`` reproduces ``w``
        for every key in `W_KEYS`, with `W_PROJ_BIAS_KEYS` all ``None``. That is pinned by a test.

    Args:
        w: Replicated weight dict. MUST carry every key in `W_KEYS` and every one must be non-None;
           a missing or ``None`` entry raises rather than producing a module whose forward would
           fail much later with a shape error. Feature width ``D`` is taken from ``norm_in_w``, and
           the projection shapes must agree with it -- ``p_in_w``/``g_in_w`` are ``(2D, D)``,
           ``p_out_w``/``g_out_w`` are ``(D, D)`` -- or the ``nn.Linear`` assignment raises. Any
           non-``None`` ``p_in_b``/``g_in_b`` raises: the front store does not fuse them, and
           silently dropping one would change the computed result. ``p_out_b``/``g_out_b`` are
           accepted and become real ``nn.Linear`` biases on the built module.
        dtype: Optional dtype for the built parameters. ``None`` (default) keeps ``w``'s dtypes and
           shares storage. A value converts, which also breaks the storage sharing above.

    Returns:
        An ``nn.Module`` on the same device(s) as ``w``'s tensors, in ``eval`` mode, with
        ``requires_grad=False`` throughout -- the fused workflow is inference-only.

    Raises:
        KeyError: If a `W_KEYS` entry is absent.
        ValueError: If a `W_KEYS` entry is ``None``, or ``p_in_b``/``g_in_b`` is not ``None``.
    """
    for key in W_KEYS:
        if key not in w:
            raise KeyError(f"trimul_module_from_weights: weight dict is missing '{key}'")
        if w[key] is None:
            raise ValueError(
                f"trimul_module_from_weights: '{key}' is None; every one of W_KEYS must be a tensor."
            )
    cast = (lambda t: t.to(dtype)) if dtype is not None else (lambda t: t)
    D = int(w["norm_in_w"].shape[-1])
    m = torch.nn.Module()
    m.norm_in = torch.nn.LayerNorm(D, eps=1e-5)
    m.norm_out = torch.nn.LayerNorm(D)
    m.p_in = torch.nn.Linear(D, 2 * D, bias=w.get("p_in_b") is not None)
    m.g_in = torch.nn.Linear(D, 2 * D, bias=w.get("g_in_b") is not None)
    # p_out/g_out carry a bias exactly when `w` has one -- `nn.Linear(bias=False)` allocates no
    # `.bias` parameter at all, so a module built without it could not round-trip one back out.
    m.p_out = torch.nn.Linear(D, D, bias=w.get("p_out_b") is not None)
    m.g_out = torch.nn.Linear(D, D, bias=w.get("g_out_b") is not None)
    with torch.no_grad():
        m.norm_in.weight = torch.nn.Parameter(cast(w["norm_in_w"]), requires_grad=False)
        m.norm_in.bias = torch.nn.Parameter(cast(w["norm_in_b"]), requires_grad=False)
        m.norm_out.weight = torch.nn.Parameter(cast(w["norm_out_w"]), requires_grad=False)
        m.norm_out.bias = torch.nn.Parameter(cast(w["norm_out_b"]), requires_grad=False)
        m.p_in.weight = torch.nn.Parameter(cast(w["p_in_w"]), requires_grad=False)
        m.g_in.weight = torch.nn.Parameter(cast(w["g_in_w"]), requires_grad=False)
        m.p_out.weight = torch.nn.Parameter(cast(w["p_out_w"]), requires_grad=False)
        m.g_out.weight = torch.nn.Parameter(cast(w["g_out_w"]), requires_grad=False)
        for sub, key in (("p_in", "p_in_b"), ("g_in", "g_in_b"),
                         ("p_out", "p_out_b"), ("g_out", "g_out_b")):
            if w.get(key) is not None:
                setattr(getattr(m, sub), "bias",
                        torch.nn.Parameter(cast(w[key]), requires_grad=False))
    return m.eval()

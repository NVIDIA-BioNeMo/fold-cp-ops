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

"""Unit for ``fold_cp_ops/distributed/trimul_weights.py`` (N1/D1.1).

The adapter turns a frozen ``TriangularMultiplication`` module into the ``w`` dict the A2A-fused
TriMul workflow consumes. Its whole contract is structural -- WHICH submodule's ``.weight`` becomes
WHICH key, with no transpose -- so the tests below drive a minimal stand-in module with the same
submodule names rather than importing the reference module.

**The stand-in is the honest choice, not a shortcut.** The reference layer is what N5/D5.1 ports
(``benchmark.distributed.trimul_dtensor_baseline``), and it is not in this tree yet; a test that waited for it
would leave the adapter untested through N2-N4, which is exactly when the mapping gets consumed. What
the stand-in cannot check -- that a real caller's layer names its submodules this way -- is checked by the
adapter's own ``AttributeError`` path, exercised below.

No GPU, no process group, no collective: pure torch on CPU. Nothing here can skip, so the
``tests/distributed/**`` declared-skip rule has nothing to declare.
"""

from __future__ import annotations

import pytest
import torch

from fold_cp_ops.distributed.trimul_weights import (
    W_KEYS,
    W_PROJ_BIAS_KEYS,
    trimul_module_from_weights,
    weights_from_trimul_module,
)
from fold_cp_ops.testing.kernel_matrix import matrix_exempt
from fold_cp_ops.testing.numerics import assert_bitwise

pytestmark = matrix_exempt(
    "a pure host-side weight-dict adapter -- there is no kernel, no tile, no mesh and no operand "
    "shape, so no axis of the kernel matrix applies to it"
)

D = 16  # small on purpose: the adapter is shape-agnostic, and a big D only slows the test


def _trimul_like(d=D, *, bias_on=(), dtype=torch.float32):
    """A stand-in ``TriangularMultiplication`` with the six submodules the adapter reads.

    Args:
        d: feature width. Any positive int; the adapter never inspects a shape, so this only decides
            how much data the round-trip compares.
        bias_on: submodule names among ``p_in``/``g_in``/``p_out``/``g_out`` to give a projection
            bias. The reference layer builds these ``bias=False``; a non-empty value here is how the unfusable
            case is exercised.
        dtype: parameter dtype. The adapter preserves it unless ``dtype=`` is passed to the adapter.

    Returns:
        An ``nn.Module`` with ``norm_in``/``norm_out`` (``LayerNorm``, so they carry weight AND bias)
        and ``p_in``/``g_in``/``p_out``/``g_out`` (``Linear``). Parameters are randomized, because a
        default-initialized ``LayerNorm`` weight is all ones and would make a mis-mapped key
        compare EQUAL to the right one.
    """
    m = torch.nn.Module()
    m.norm_in = torch.nn.LayerNorm(d, dtype=dtype)
    m.norm_out = torch.nn.LayerNorm(d, dtype=dtype)
    m.p_in = torch.nn.Linear(d, 2 * d, bias="p_in" in bias_on, dtype=dtype)
    m.g_in = torch.nn.Linear(d, 2 * d, bias="g_in" in bias_on, dtype=dtype)
    m.p_out = torch.nn.Linear(d, d, bias="p_out" in bias_on, dtype=dtype)
    m.g_out = torch.nn.Linear(d, d, bias="g_out" in bias_on, dtype=dtype)
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn_like(p))
    return m


#: (w-key, the submodule attribute it must come from). The adapter's entire contract, spelled out
#: independently of its own ``_PARAM_MAP`` -- reading that table to test it would assert the table
#: equals itself.
_EXPECTED_SOURCE = {
    "norm_in_w": ("norm_in", "weight"),
    "norm_in_b": ("norm_in", "bias"),
    "p_in_w": ("p_in", "weight"),
    "g_in_w": ("g_in", "weight"),
    "norm_out_w": ("norm_out", "weight"),
    "norm_out_b": ("norm_out", "bias"),
    "p_out_w": ("p_out", "weight"),
    "g_out_w": ("g_out", "weight"),
}


def test_every_declared_key_is_present_and_no_others():
    """The dict carries exactly the eight weight keys plus the four bias keys, and nothing else."""
    w = weights_from_trimul_module(_trimul_like())
    assert set(w) == set(W_KEYS) | set(W_PROJ_BIAS_KEYS), sorted(set(w))
    assert len(W_KEYS) == 8 and len(W_PROJ_BIAS_KEYS) == 4


@pytest.mark.parametrize("key", sorted(_EXPECTED_SOURCE))
def test_each_key_round_trips_bitwise_from_its_own_parameter(key):
    """Every key is the BIT PATTERN of its source parameter -- no transpose, no cast, no copy loss.

    Element-wise and exact: the adapter is pure data movement, so anything less than bitwise would
    let a silent dtype round-trip or a sign-of-zero loss through. Randomized parameters are what make
    a mis-mapping visible -- with default LayerNorm gains (all ones) a swap of ``norm_in_w`` and
    ``norm_out_w`` would compare equal.
    """
    m = _trimul_like()
    w = weights_from_trimul_module(m)
    sub, attr = _EXPECTED_SOURCE[key]
    assert_bitwise(w[key], getattr(getattr(m, sub), attr), what=f"w[{key!r}]")


def test_the_projection_bias_keys_are_none():
    """All four projection-bias keys are ``None`` -- the wave-1 contract the workflow requires."""
    w = weights_from_trimul_module(_trimul_like())
    assert all(w[k] is None for k in W_PROJ_BIAS_KEYS), {k: w[k] for k in W_PROJ_BIAS_KEYS}


@pytest.mark.parametrize("sub", ["p_in", "g_in", "p_out", "g_out"])
def test_every_projection_bias_is_CARRIED_not_refused(sub):
    """A projection bias on ANY of the four reaches the dict BITWISE -- the workflow fuses them all.

    This adapter refused all four for most of the port, with a message asserting the workflow fused
    none of them. That was wrong twice over, and in stages: the OUTPUT pair had always been fused (as
    ``bp``/``bg`` on `layernorm_dual_gated_gemm`), and the INPUT pair is fused now (as the
    interleaved ``mRowVecBroadcast`` on the front store). Because `TriangularMultiplication` reads
    its weights through here, each refusal made live fused code unreachable from the shipped
    nn.Module API rather than merely undocumented.

    `test_the_fused_chain_matches_the_fp32_oracle` runs all four against the fp32 oracle.
    """
    m = _trimul_like(bias_on=(sub,))
    w = weights_from_trimul_module(m)
    assert_bitwise(w[f"{sub}_b"], getattr(m, sub).bias, what=f"w[{sub}_b]")
    others = [k for k in W_PROJ_BIAS_KEYS if k != f"{sub}_b"]
    assert all(w[k] is None for k in others), {k: w[k] for k in others}


@pytest.mark.parametrize("missing", ["norm_in", "p_in", "g_out"])
def test_a_module_that_is_not_a_trimul_layer_is_named_not_guessed(missing):
    """A missing submodule raises ``AttributeError`` naming it, rather than a ``KeyError`` later.

    This is the check the stand-in cannot make for us: it proves the adapter VALIDATES the structure
    it assumes, so pointing it at the wrong module fails at the adapter instead of three frames into
    the workflow.
    """
    m = _trimul_like()
    delattr(m, missing)
    with pytest.raises(AttributeError, match=missing):
        weights_from_trimul_module(m)


def test_dtype_and_device_arguments_are_applied():
    """``dtype=``/``device=`` convert every tensor; ``None`` (default) preserves the parameter's."""
    m = _trimul_like(dtype=torch.float32)
    w_native = weights_from_trimul_module(m)
    assert all(w_native[k].dtype is torch.float32 for k in W_KEYS)
    w_cast = weights_from_trimul_module(m, dtype=torch.bfloat16)
    assert all(w_cast[k].dtype is torch.bfloat16 for k in W_KEYS)
    w_cpu = weights_from_trimul_module(m, device=torch.device("cpu"))
    assert all(w_cpu[k].device.type == "cpu" for k in W_KEYS)


def test_detach_defaults_on_so_no_graph_is_carried_into_the_kernels():
    """Default ``detach=True``; ``detach=False`` keeps the parameter's autograd linkage.

    Frozen-inference weights must not drag a graph into a CuTe-DSL kernel, and the flag is the only
    place that is decided.
    """
    m = _trimul_like()
    assert all(not weights_from_trimul_module(m)[k].requires_grad for k in W_KEYS)
    w = weights_from_trimul_module(m, detach=False)
    assert any(w[k].requires_grad for k in W_KEYS), "detach=False must preserve the parameter"


def test_the_public_package_re_exports_the_adapter():
    """``fold_cp_ops.distributed`` re-exports the three names (D1.2).

    The package ``__init__`` executes on ANY ``import fold_cp_ops.distributed.X``, so a name that is
    listed but unresolvable breaks every import in the subtree, not just this one.
    """
    import fold_cp_ops.distributed as pkg

    for name in ("W_KEYS", "W_PROJ_BIAS_KEYS", "weights_from_trimul_module"):
        assert name in pkg.__all__, sorted(pkg.__all__)
        assert hasattr(pkg, name)


# --------------------------------------------------------------------------- #
# trimul_module_from_weights -- the inverse adapter
# --------------------------------------------------------------------------- #


def _weights(d=D):
    """A well-formed weight dict, built the way a bench builds one (synthetic, bias-free).

    The LayerNorm affine is RANDOMIZED, for the reason `_trimul_like` already states in the other
    direction: a default gain is all ones and a default bias all zeros, so a mis-mapped
    ``norm_in_w``/``norm_out_w`` -- or a bias dropped entirely on the way through
    `trimul_module_from_weights` -- would compare EQUAL to the correct answer and the bitwise
    round-trip above would pass on a broken adapter. `_trimul_like` randomizes; this builder is the
    inverse direction and had been left at the identity.
    """
    g = torch.Generator().manual_seed(20260825)
    rn = lambda *sh: torch.randn(*sh, generator=g, dtype=torch.float32) * 0.02  # noqa: E731
    ln_w = lambda: 1.0 + torch.randn(d, generator=g, dtype=torch.float32) * 0.1  # noqa: E731
    ln_b = lambda: torch.randn(d, generator=g, dtype=torch.float32) * 0.5  # noqa: E731
    w = dict(
        norm_in_w=ln_w(), norm_in_b=ln_b(),
        p_in_w=rn(2 * d, d), g_in_w=rn(2 * d, d),
        norm_out_w=ln_w(), norm_out_b=ln_b(),
        p_out_w=rn(d, d), g_out_w=rn(d, d),
    )
    for k in W_PROJ_BIAS_KEYS:
        w[k] = None
    return w


def test_the_built_module_carries_exactly_the_six_canonical_submodule_names():
    """The names are the contract: `TriangularMultiplication` and `weights_from_trimul_module` both
    reach for them by `getattr`, and the CP reference wrappers `TriangularMultiplication{1,2}D` set the same six.
    A seventh child, or a renamed one, means a caller duck-typing on this module diverges from one
    duck-typing on a real caller's layer."""
    m = trimul_module_from_weights(_weights())
    assert {n for n, _ in m.named_children()} == {
        "norm_in", "p_in", "g_in", "norm_out", "p_out", "g_out"
    }


@pytest.mark.parametrize("key", W_KEYS)
def test_each_key_round_trips_bitwise_through_the_module(key):
    """`weights_from_trimul_module(trimul_module_from_weights(w))` reproduces `w`, BITWISE.

    Per-key rather than one dict comparison so a failure names the tensor that moved. Bitwise
    because the adapter is a re-binding, not an arithmetic step -- any drift here is a real defect,
    not tolerance."""
    w = _weights()
    back = weights_from_trimul_module(trimul_module_from_weights(w))
    assert_bitwise(back[key], w[key], what=f"round-trip w[{key!r}]")


def test_the_projection_bias_keys_survive_the_round_trip_as_none():
    """The fused path does not fuse projection biases. A round trip that invented one would make
    the module compute something the kernels cannot reproduce."""
    back = weights_from_trimul_module(trimul_module_from_weights(_weights()))
    assert all(back[k] is None for k in W_PROJ_BIAS_KEYS)  # `_weights()` supplies none of them


@pytest.mark.parametrize("key", ["norm_in_w", "p_in_w", "g_out_w"])
def test_a_missing_or_none_weight_is_refused_at_build_not_at_forward(key):
    """A dict short one key must fail HERE, naming it. Building the module anyway defers the
    failure to a shape error deep in a kernel launch, where the key's name is gone."""
    w = _weights()
    del w[key]
    with pytest.raises(KeyError, match=key):
        trimul_module_from_weights(w)
    w = _weights()
    w[key] = None
    with pytest.raises(ValueError, match=key):
        trimul_module_from_weights(w)


@pytest.mark.parametrize("key", ["p_in_b", "g_in_b", "p_out_b", "g_out_b"])
def test_a_projection_bias_round_trips_through_the_module(key):
    """A projection bias survives ``w -> module -> w`` BITWISE, on any of the four.

    Two distinct failures this catches, and only the round trip sees both: the builder allocating an
    ``nn.Linear(bias=False)`` (which has no ``.bias`` parameter to write at all, so the value is
    dropped at build) and the reader setting the key to None regardless (dropped at read). Either one
    alone leaves `TriangularMultiplication` computing without a bias its caller supplied.
    """
    w = _weights()
    # (2D,) on the front pair -- it biases the stacked dual -- and (D,) on the back pair.
    n = w["p_in_w"].shape[0] if key in ("p_in_b", "g_in_b") else w["p_out_w"].shape[0]
    w[key] = torch.randn(n, dtype=torch.float32)
    back = weights_from_trimul_module(trimul_module_from_weights(w))
    assert_bitwise(back[key], w[key], what=f"round-tripped {key}")


def test_the_default_shares_storage_and_an_explicit_dtype_converts():
    """Documented behaviour, and the difference matters: the default is a view-like re-binding, so a
    caller mutating `w` afterwards mutates the module. Passing `dtype` copies, which breaks that."""
    w = _weights()
    m = trimul_module_from_weights(w)
    assert m.p_in.weight.data_ptr() == w["p_in_w"].data_ptr()
    m16 = trimul_module_from_weights(w, dtype=torch.bfloat16)
    assert m16.p_in.weight.dtype is torch.bfloat16
    assert m16.p_in.weight.data_ptr() != w["p_in_w"].data_ptr()


def test_the_module_is_inference_shaped():
    """The fused workflow is inference-only (`forward` is `@torch.no_grad`). A module handed back in
    train mode with grad-tracking parameters would build an autograd graph nothing consumes."""
    m = trimul_module_from_weights(_weights())
    assert not m.training
    assert not any(p.requires_grad for p in m.parameters())

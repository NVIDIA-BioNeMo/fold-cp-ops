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

"""The cp=1 TriMul dispatcher: chain this package's fused kernels into one triangular update.

**What it computes** -- the twelve-op reference of ``docs/cp1_kernel_bringback_complete.md`` §2,
with the token axis ``(b, i, j)`` flattened to ``M = B * N * N`` throughout::

     1  xn    = LayerNorm(x)                                     (B, N, N, D)
     2  p_in  = xn @ p_in_w^T + p_in_b                           (B, N, N, 2D)
     3  g_in  = xn @ g_in_w^T + g_in_b                           (B, N, N, 2D)
     4  ab    = p_in * sigmoid(g_in)
     5  ab   *= mask                                             optional, DUAL ONLY
     6  a, b  = ab.chunk(2, -1)
     7  tri   = einsum('bikd,bjkd->bijd', a, b)      outgoing
             | einsum('bkid,bkjd->bijd', a, b)      incoming
     8  trin  = LayerNorm(tri)
     9  p_out = trin @ p_out_w^T + p_out_b
    10  g_out = xn @ g_out_w^T + g_out_b                          <- consumes xn, NOT trin
    11  gate  = sigmoid(g_out)
    12  out   = p_out * gate

**Two invariants the combos below are built around.** Op 5 masks the dual ``[a|b]`` ONLY and never
``gate3`` -- that is cuEquivariance's semantics, and it is why a masked shape is not a separate
regime. And ops 10/11 read ``xn``, not ``trin``, which is what makes "gate in the front" and "gate
in the back" two different kernel SHAPES rather than a scheduling choice.

**The combos** (the ``variant`` knob). **A variant IS its kernel chain, in call order** -- a single
underscore belongs to one kernel's own name, a DOUBLE underscore separates two kernels. So
``gemm_layernorm_gemm`` is ONE fused kernel and ``gemm__layernorm_gemm`` is TWO, which is precisely
the difference between those two combos. The FRONT-GATE family computes the output gate in the FRONT
and carries it through as a per-element ``gate3``; the BACK-GATE family computes it in the BACK from
a shared normalized input::

    -- FRONT-GATE (gate3 carried through) ------------------------------------
    gemm_layernorm_gemm
        front -> gemm_layernorm_gemm(a, b, ...)      ops 7,8,9,12 in ONE kernel
    gemm__layernorm_gemm
        front -> gemm -> layernorm_gemm              ops 8,9,12 in one kernel
    gemm__layernorm__gemm_hadamard
        front -> gemm -> layernorm_fwd(transpose=True) -> gemm_hadamard

    -- BACK-GATE (one LayerNorm feeds two consumers) -------------------------
    dual_gated_gemm__gemm__layernorm_dual_gated_gemm
        xn -> dual_gated_gemm -> gemm -> layernorm_dual_gated_gemm(x_gate=xn)
    dual_gated_gemm__gemm__layernorm__gemm_hadamard
        the same, with op 8 moved OUT into a separate transposing LayerNorm pass

These names replaced ``P1.glg`` / ``P1.lng`` / ``P1.q3k`` / ``P2.xgate`` / ``xgate.q3k_lt``. The old
ones encoded only the family (``P1``/``P2``) and an abbreviation, so reading a config told you
nothing about which kernels would run; the family survives as :data:`FRONTGATE_VARIANTS` and
:data:`BACKGATE_VARIANTS` rather than as a prefix.

**Naming.** ``front_v`` / ``lng_inner`` / ``back_v`` name a LayerNorm FUSION, so they take this
package's fusion names -- ``alg_fold`` (the LayerNorm folded into the weight algebraically) and
``prolog_ln`` (run physically in the prologue). The kernel this ports from called them ``stagec``
and ``staged``, after which internal design stage shipped first; those names do not appear here.
``lng2k`` is kept because the plan this implements uses it: it names the front that runs a
STANDALONE LayerNorm and then an unfused dual-gated GEMM, rather than fusing the two.

**One family of host-layer defect runs through all of this: the size heuristic returns a combo the
shape cannot run.** The kernel this ports from records two instances it already fixed -- a launch
corner missing a token floor, and a compute branch returning a combo unconditionally. Two more are
measured here and fixed: op 7 carries a token-alignment floor the upstream states it does not have
(`_gemm1_validity`, 1188 of 7128 enumerated shapes), and the front's width floor does NOT survive
the move to this package's front doors, so carrying it across would prune five widths that in fact
run (`_front_validity`, 20 of 20 cells). A third, `_lt_validity`'s floor on the token count, is a
real constraint that op 7's floor SUBSUMES -- it binds on zero shapes, and it is documented as
subsumed rather than counted.

What closes the family is not any one fix: it is `tests/workflows/test_trimul_autotune.py`'s brute-force
assertion that every combo the heuristic returns survives `_prune` for that same shape, over a wide
grid, so no future edit can reintroduce it one branch at a time.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import Tensor

from fold_cp_ops._internal.arch import get_device_capacity
from fold_cp_ops._internal.autotune import AutotuneConfig, autotune
from fold_cp_ops._internal.heuristic_arch import (
    TUNED_ARCH,
    heuristic_arch,
    warn_arch_suboptimal_once,
)
from fold_cp_ops.kernels.dual_gated_gemm import dual_gated_gemm, heuristic_chunk_g
from fold_cp_ops.kernels.gemm import gemm
from fold_cp_ops.kernels.gemm_hadamard import gemm_hadamard
from fold_cp_ops.kernels.layernorm import layernorm_fwd
from fold_cp_ops.kernels.layernorm_dual_gated_gemm import layernorm_dual_gated_gemm
from fold_cp_ops.kernels.layernorm_gemm import layernorm_gemm

__all__ = [
    "FRONT_VARIANTS",
    "TRIMUL_VARIANTS",
    "TriMulConfig",
    "trimul_autotuned",
    "trimul_freeze",
    "trimul_ref",
]

#: Every combo name the dispatcher recognises. Declared as a constant so a caller can enumerate them
#: and so a typo in a forced ``_config`` is refused by name rather than falling through to an
#: ``unknown variant`` deep inside the timed region.
#: **Reading a variant name.** It is the kernel CHAIN, in call order. A single underscore is part of
#: one kernel's own name; a DOUBLE underscore separates two kernels. So
#: ``gemm_layernorm_gemm`` is ONE fused kernel (ops 7, 8, 9, 12 together) while
#: ``gemm__layernorm_gemm`` is TWO -- `gemm`, then the fused `layernorm_gemm`. That distinction is
#: the entire difference between those two combos, and a single-underscore join would have collapsed
#: them into the same string.
TRIMUL_VARIANTS = (
    "gemm_layernorm_gemm",
    "gemm__layernorm__gemm_hadamard",
    "gemm__layernorm_gemm",
    "dual_gated_gemm__gemm__layernorm_dual_gated_gemm",
    "dual_gated_gemm__gemm__layernorm__gemm_hadamard",
)

#: The two FAMILIES, by where the output gate (ops 10-11) is computed. This grouping used to be
#: carried by the phase-label prefixes; those said nothing about the kernels, so the names now
#: spell the chain and the families are named here instead. The distinction is a kernel SHAPE
#: difference, not a scheduling one: ops 10/11 read ``xn`` and not ``trin``, so computing the gate
#: in the front means carrying it through as a per-element ``gate3``, while computing it in the back
#: means one LayerNorm feeding two consumers.
FRONTGATE_VARIANTS = (
    "gemm_layernorm_gemm",
    "gemm__layernorm__gemm_hadamard",
    "gemm__layernorm_gemm",
)
BACKGATE_VARIANTS = (
    "dual_gated_gemm__gemm__layernorm_dual_gated_gemm",
    "dual_gated_gemm__gemm__layernorm__gemm_hadamard",
)

#: The front choices for the FRONT-GATE combos. ``alg_fold`` and ``prolog_ln`` are the two LayerNorm
#: fusions of `layernorm_dual_gated_gemm`; ``lng2k`` runs a standalone `layernorm_fwd` and then the
#: UNFUSED `dual_gated_gemm` on the already-normalized activation.
FRONT_VARIANTS = ("alg_fold", "prolog_ln", "lng2k")

#: The LayerNorm fusions `layernorm_gemm` implements, i.e. the legal ``lng_inner`` values.
#: ``prolog_ln`` is excluded because the heuristic never returns it. Kept as a tuple rather than
#: inlined so adding it later is a new VALUE and not a signature change.
LNG_INNER_VARIANTS = ("alg_fold",)

#: The LayerNorm fusions the ``x_gate`` back accepts, i.e. the legal ``back_v`` values.
BACK_VARIANTS = ("alg_fold", "prolog_ln")

# Which fields each variant's dispatch actually consumes. The projection keeps a combo-irrelevant
# knob out of the cache key, so two candidates that run the same kernels share one cache entry.
_P1_GLG_FIELDS = ("variant", "front_v")
_P1_Q3K_FIELDS = ("variant", "front_v")
_P1_LNG_FIELDS = ("variant", "front_v", "lng_inner")
_P2_XGATE_FIELDS = ("variant", "back_v")
_XGATE_Q3K_LT_FIELDS = ("variant", "back_v")


@dataclass(frozen=True)
class TriMulConfig:
    """One autotune candidate: which chain of kernels to run, and which fusion at each seam.

    Purpose
        The unit the autotuner sweeps and the size heuristic returns. Frozen, so a config captured
        by `trimul_freeze` cannot be mutated out from under the artifact it selected.

    Semantics
        `variant` is THE combo knob. The other three are per-seam and only some apply to each combo,
        which is what :meth:`all_kwargs` projects away -- a ``gemm_layernorm_gemm`` candidate's
        ``lng_inner`` is not a knob it consumes, and letting it into the cache key would split one
        entry into two that run identical code.

    Attributes:
        variant: A member of :data:`TRIMUL_VARIANTS`.
        front_v: The front for every FRONT-GATE combo; a member of :data:`FRONT_VARIANTS`.
        Irrelevant
            to the BACK-GATE combos, whose front is the unfused `dual_gated_gemm` on a shared
            normalized input.
        lng_inner: The LayerNorm fusion of ``gemm__layernorm_gemm``'s back; a member of
            :data:`LNG_INNER_VARIANTS`.
        back_v: The LayerNorm fusion of a BACK-GATE combo's back; a member of :data:`BACK_VARIANTS`.
    """

    variant: str
    front_v: str = "alg_fold"
    lng_inner: str = "alg_fold"
    back_v: str = "alg_fold"

    def _proj_fields(self) -> Tuple[str, ...]:
        """The field names this combo's dispatch actually reads.

        Returns:
            A tuple of field names. An unrecognised `variant` returns EVERY field rather than a
            guess: an unknown combo must not silently share a cache entry with a known one.
        """
        return {
            "gemm_layernorm_gemm": _P1_GLG_FIELDS,
            "gemm__layernorm__gemm_hadamard": _P1_Q3K_FIELDS,
            "gemm__layernorm_gemm": _P1_LNG_FIELDS,
            "dual_gated_gemm__gemm__layernorm_dual_gated_gemm": _P2_XGATE_FIELDS,
            "dual_gated_gemm__gemm__layernorm__gemm_hadamard": _XGATE_Q3K_LT_FIELDS,
        }.get(self.variant, ("variant", "front_v", "lng_inner", "back_v"))

    def all_kwargs(self) -> dict:
        """This combo's consumed fields, as keyword arguments for the implementation.

        Returns:
            A dict over :meth:`_proj_fields`. Everything else is projected out, so it reaches
            neither the cache key nor the call.
        """
        return {f: getattr(self, f) for f in self._proj_fields()}

    def __str__(self) -> str:
        """Render as ``variant(knob=value, ...)`` over the consumed fields only.

        Returns:
            A short string for a log line or a failure message.
        """
        knobs = ", ".join(f"{k}={v}" for k, v in self.all_kwargs().items() if k != "variant")
        return f"{self.variant}({knobs})"


def _trimul_configs() -> List[TriMulConfig]:
    """The candidate grid the autotuner sweeps: ELEVEN configs.

    Purpose
        The declared pool. It is curated rather than a full Cartesian product because most crossings
        name the same kernels twice -- ``gemm_layernorm_gemm``'s back has no ``lng_inner`` to vary.

    Semantics
        **Eleven, where the kernel this ports from emits fourteen.** ``gemm__layernorm_gemm`` contributes three
        rather than six: `layernorm_gemm` implements one LayerNorm fusion, ``alg_fold``, because the
        heuristic never returned the other. A grid that can hand back a config nothing implements is
        a crash reachable from ``select="autotune"``, so the grid is narrowed with the kernel rather
        than left wide and pruned later.

    Returns:
        A fresh list -- ``gemm_layernorm_gemm`` 3, ``gemm__layernorm__gemm_hadamard`` 3,
        ``gemm__layernorm_gemm`` 3, ``dual_gated_gemm__gemm__layernorm_dual_gated_gemm`` 2.
        ``dual_gated_gemm__gemm__layernorm__gemm_hadamard`` is recognised and prune-valid but
        deliberately NOT in the default grid: it is a candidate to sweep deliberately, not one to
        pay for on every tune.
    """
    configs: List[TriMulConfig] = []
    for fv in FRONT_VARIANTS:
        configs.append(TriMulConfig("gemm_layernorm_gemm", front_v=fv))
        configs.append(TriMulConfig("gemm__layernorm__gemm_hadamard", front_v=fv))
        for li in LNG_INNER_VARIANTS:
            configs.append(TriMulConfig("gemm__layernorm_gemm", front_v=fv, lng_inner=li))
    for bv in BACK_VARIANTS:
        configs.append(TriMulConfig("dual_gated_gemm__gemm__layernorm_dual_gated_gemm", back_v=bv))
    return configs


def _as_autotune_config(c: TriMulConfig) -> AutotuneConfig:
    """Wrap one combo as the autotuner's config, carrying its PROJECTED fields only.

    Purpose
        The bridge between the readable combo object and the tuner's key.

    Semantics
        `AutotuneConfig` refuses a value that is not a compile-time constant, and rightly: a config
        value is folded into the compiled kernel, hashed into the result cache and pickled to a
        pre-compile worker, so a live object there would corrupt all three. So the combo travels as
        its four STRINGS, not as the dataclass -- and only the strings this combo consumes, which is
        what keeps two candidates that run identical code from splitting one cache entry into two.

    Args:
        c: The combo.

    Returns:
        An `AutotuneConfig` over `TriMulConfig.all_kwargs`.
    """
    return AutotuneConfig(**c.all_kwargs())


def _from_autotune_config(conf: AutotuneConfig) -> TriMulConfig:
    """Recover the combo from the tuner's config, filling the projected-away fields with defaults.

    Args:
        conf: An `AutotuneConfig` built by `_as_autotune_config`, or any config carrying at least a
            ``variant``.

    Returns:
        A `TriMulConfig`. A field the projection dropped comes back at its default, which is
        correct precisely because the combo does not read it.

    Raises:
        KeyError: If ``variant`` is absent -- a config with no combo names no chain to run, and
            defaulting one would silently time something the caller did not ask for.
    """
    kw = conf.all_kwargs()
    if "variant" not in kw:
        raise KeyError(
            f"an autotune config for the TriMul dispatcher must carry a 'variant'; got {sorted(kw)}"
        )
    return TriMulConfig(**{k: v for k, v in kw.items() if k in TriMulConfig.__dataclass_fields__})


# ───────────────────────────── validity: what each shape can actually run ─────────────────────────


def dual_tile_n(D: int) -> int:
    """The CTA tile over the front's ``2n = 4D`` pre-activation.

    Purpose
        The front's ONE tiling decision, made here because this package's dual front doors take the
        tile as a required argument rather than auto-picking it.

    Semantics
        The front's two weights are each ``(2D, D)``, so the GLU output width is ``n = 2D`` and the
        interleaved pre-activation the kernel tiles is ``2n = 4D``. `append_gate3_weight` requires
        the region boundary to fall on a work-tile edge -- ``2n % tile_N == 0`` -- so the tile must
        divide ``4D``. It does NOT have to divide the gate width ``n3 = D``: the gate's rows are
        padded up to a whole tile and its TMA descriptor predicates the pad.

        Widest-first among the legal cooperative tiles, because a wider tile is a longer mainloop
        per epilogue. At the 16-byte floor ``D % 8 == 0`` the value 32 always divides ``4D``, so
        this never fails to find one.

    Args:
        D: The feature width. Must satisfy ``D % 8 == 0``; below that the front is refused earlier
            and this would return a tile whose kernel could not load its operands anyway.

    Returns:
        One of 256, 128, 64, 32, 16.
    """
    return next(t for t in (256, 128, 64, 32, 16) if (4 * D) % t == 0)


def _front_validity(D: int) -> bool:
    """Whether the TriMul front can run at this feature width -- for ALL THREE front variants.

    Purpose
        The front's size gate. It is ONE bool rather than one per variant, and that collapse is the
        first of this module's four measured divergences from the kernel it ports.

    Semantics
        The front always carries the output gate (``W3 = g_out_w``, ``n3 = D``), and it is the gate
        that used to constrain the width. Upstream, the fused prologue front auto-picked its
        ``tile_N`` and then required ``n3 % tile_N == 0``, giving it a ``D % 128`` floor, while its
        unfused sibling picked differently and got a looser one. Here BOTH go through front doors
        that take ``tile_N`` explicitly, `dual_tile_n` supplies it, and `append_gate3_weight` pads
        ``n3`` to a whole tile -- so the gate constrains nothing and what remains is the 16-byte
        alignment floor on the contraction extent.

        **MEASURED, not inferred.** 20 cells, both fusions, ``D`` in 64, 96, 128, 136, 192, 200,
        256, 320, 384, 512, each with the gate and a transposed store: all 20 launch and produce
        finite output -- including 64, 136, 192, 200 and 320, which the upstream gate rejects for
        its prologue front. Carrying the upstream rule across would have pruned five widths the
        kernels here run, three of them (136, 200, 320) exactly the off-grid widths this workflow's
        fallback exists for.

    Args:
        D: The feature width, which is also the front's contraction extent.

    Returns:
        True when the front can run. The caller still has to supply `dual_tile_n`'s tile.
    """
    return D % 8 == 0


def _gemm1_validity(N: int) -> bool:
    """Whether the token-pair einsum (op 7) can run at this token count.

    Purpose
        The floor EVERY combo that materializes ``tri`` shares --
        ``gemm__layernorm__gemm_hadamard``, ``gemm__layernorm_gemm``,
        ``dual_gated_gemm__gemm__layernorm_dual_gated_gemm`` and
        ``dual_gated_gemm__gemm__layernorm__gemm_hadamard``. The upstream states there is no such
        floor outside its ``glg`` combo; there is.

    Semantics
        Op 7 is a batched GEMM over ``(N, N)`` operands, so ``N`` is the contiguous extent of both
        the operand rows and the transposed views the incoming direction takes. A 16-bit element
        needs a 16-byte-aligned row pitch, which is ``N % 8 == 0``.

        **MEASURED.** ``N = 100`` and ``N = 99`` are both refused, in both directions, by `gemm`'s
        own front door naming the strides; ``N`` in 128, 136, 64, 40 all run and match a torch
        ``einsum`` reference to bf16 accumulation error. Upstream this floor lives only inside
        ``glg_ok``, while its own comment says the batched GEMM's "only floor is D" -- so at
        ``N = 100, D = 136`` its heuristic returns ``gemm__layernorm_gemm``, its pruner accepts it,
        and op 7 then cannot load its operands. Reachable from the shipped default.

    Args:
        N: The token extent. ``M = B * N * N`` is the flattened token axis, but the alignment is on
            ``N`` itself, not on ``M``: ``N`` is the row pitch of the ``(N, N)`` operands.

    Returns:
        True when op 7 can run.
    """
    return N % 8 == 0


def _lt_validity(M: int, D: int) -> bool:
    """Whether the TRANSPOSING LayerNorm (op 8, out of a LayoutLeft ``tri``) can run.

    Purpose
        Shared by ``gemm__layernorm__gemm_hadamard`` and
        ``dual_gated_gemm__gemm__layernorm__gemm_hadamard``, which are the two combos that do op 8
        as a separate pass over a LayoutLeft input.

    Semantics
        Three floors, all measured against `layernorm_fwd(transpose=True)`:

        * ``D % 16 == 0`` -- the transpose swizzle atom at 16 bits. ``D`` in 136 and 200 are refused
          by name.
        * ``D <= 1024`` -- the whole row is staged in shared memory. Carried from upstream; it could
          not be isolated in the measurement because 1032 trips the ``% 16`` floor first, so it is
          an inherited bound rather than a measured one. Over-restrictive costs a combo; under-
          restrictive would cost a launch.
        * ``M % 8 == 0`` -- the 16-byte floor on ``M``, which is the CONTIGUOUS mode of the
          ``(D, M)`` view the transposing load reads. ``M`` in 513 and 516 are refused, 520 runs.

        **The third is a real floor that the dispatcher can never reach, and saying so is the
        point.** Upstream's back-half gate omits it while its own output-gate gate carries it for
        the same kernel, so the two disagree about one kernel and only one of them is right. But
        every combo that reaches op 8 this way has already run op 7, and `_gemm1_validity` forces
        ``N % 8 == 0``, which makes ``M = B * N^2`` a multiple of 64 and hence of 8. So the floor is
        SUBSUMED: enumerated over 7128 shapes, it binds on exactly zero. It is kept because it is
        this kernel's genuine constraint and this predicate is also callable on its own -- not
        because the dispatcher needs it.

    Args:
        M: The flattened token count ``B * N * N``.
        D: The feature width, which is the axis op 8 reduces.

    Returns:
        True when op 8 can run as a separate transposing pass.
    """
    return (D % 16 == 0) and (D <= 1024) and (M % 8 == 0)


def _back_validity(N: int, D: int, B: int = 1) -> Tuple[bool, bool, bool]:
    """Per-back size gate for the three FRONT-GATE back-halves.

    Purpose
        Says which FRONT-GATE back can run this shape, so `_prune` can drop the rest before anything
        is timed and the heuristic can check its own answer.

    Semantics
        * ``glg`` -- ops 7, 8, 9 and 12 in one kernel, so it runs its OWN op 7 and needs that
          kernel's floors: ``N % 8`` and ``D % 8``. **INHERITED, NOT MEASURED**: the kernel is not
          in this tree yet, so this reproduces the upstream rule and must be re-measured when it
          lands. It is additionally ``B == 1`` only, which `_prune` applies -- that is a signature
          fact, not a size floor, so it does not live here.
        * ``q3k`` -- a separate transposing LayerNorm then a Hadamard-epilogue GEMM, so its floors
          are `_lt_validity`'s. The GEMM itself predicates every extent and adds none: measured at
          ``M`` in 513 and 9801 and ``D`` in 136 and 200, all of which run.
        * ``lng`` -- the fused `layernorm_gemm`, whose only constraint is ``D % 8 == 0``. Measured
          by that kernel's own suite.

        Op 7's own floor is NOT folded in here: it is `_gemm1_validity`, applied by `_prune` to
        every combo that materializes ``tri``, because the BACK-GATE combos needs it too.

    Args:
        N: The token extent.
        D: The feature width.
        B: The batch. Only ``q3k`` reads it, through ``M = B * N * N``; it is a parameter rather
            than an assumption because the transposing LayerNorm's floor is on ``M``, and at
            ``B > 1`` an ``M`` that is fine at ``B = 1`` can stop being so.
    Returns:
        ``(glg_ok, q3k_ok, lng_ok)``.
    """
    glg_ok = (N % 8 == 0) and (D % 8 == 0)
    q3k_ok = _lt_validity(B * N * N, D)
    lng_ok = D % 8 == 0
    return glg_ok, q3k_ok, lng_ok


def _backgate_validity(N: int, D: int) -> bool:
    """Size gate for ``dual_gated_gemm__gemm__layernorm_dual_gated_gemm`` -- the combo that computes
    the output gate in the BACK.

    Semantics
        Three stages, each with a floor. The unfused front on the shared normalized input needs
        ``D % 8``. Op 7 needs `_gemm1_validity`. The back is the two-input dual-gated GEMM whose
        gate consumes a separate raw activation: it tiles the output width ``N == D`` with a
        multiple-of-32 CTA tile and contracts over ``K == D``, giving ``D % 32`` on top of the
        alignment floor.

        **INHERITED, NOT MEASURED.** The two-input back is not in this tree yet, so the ``D % 32``
        rule reproduces the upstream's and must be re-measured when it lands -- the same way the
        front's rule turned out not to survive the move.

    Args:
        N: The token extent, read for op 7's floor.
        D: The feature width.

    Returns:
        True when the combo can run.
    """
    return (D % 8 == 0) and (D % 32 == 0) and _gemm1_validity(N)


def _xgate_q3k_lt_validity(N: int, D: int, B: int = 1) -> bool:
    """Size gate for ``dual_gated_gemm__gemm__layernorm__gemm_hadamard`` --
    ``dual_gated_gemm__gemm__layernorm_dual_gated_gemm`` with op 8 moved out into its own pass.

    Semantics
        Everything `_backgate_validity` requires, plus `_lt_validity` on the separate transposing
        LayerNorm. Only the PLACEMENT of op 8 differs between the two combos, so only its floors are
        added.

    Args:
        N: The token extent.
        D: The feature width.
        B: The batch, for ``M = B * N * N``.

    Returns:
        True when the combo can run.
    """
    return _backgate_validity(N, D) and _lt_validity(B * N * N, D)


def _prune(configs, named_args: dict, **kwargs):
    """Drop the candidates this shape cannot run, before anything is timed.

    Purpose
        The autotuner's validity filter. It must be a PURE function of the shape: the surviving list
        has to be identical on every call, or two runs time different pools and compare numbers for
        different kernels.

    Semantics
        Reads ``x`` from the bound arguments and derives ``(B, N, D)``. Every combo is dropped off
        a non-SM90 device, and then per combo:

        * every combo but ``gemm_layernorm_gemm`` materializes ``tri``, so it needs `_gemm1_validity`;
        * every FRONT-GATE combo needs `_front_validity` and its own arm of `_back_validity`;
        * ``gemm__layernorm_gemm`` additionally needs a ``lng_inner`` this package implements;
        * the BACK-GATE combos needs its own gate.

        **``gemm_layernorm_gemm`` is ``B == 1`` only, and that is a SIGNATURE fact rather than a size floor.**
        Its kernel takes a ``(1, D, N, N)`` operand and runs an op 7 that is not batched over the
        leading axis, so at ``B > 1`` there is nothing to widen -- the other combos cover it.

        **The upstream's docstring for this function contradicts its own code and the prose is
        corrected here rather than carried.** It says ``gemm_layernorm_gemm`` is "VALID ONLY for
        B==1 (gated below)" and then applies no such gate: its ``glg_ok`` is ``N % 8 and D % 8``,
        with an inline comment recording that the kernel gained ``B > 1`` support in a later commit.
        The CODE is what is ported -- the gate below is on the combo's operand shape, which is a
        real constraint -- and the claim that the kernel cannot do ``B > 1`` is dropped.

        **There is no mask gate.** The front masks the dual ``[a|b]`` only and leaves the output
        gate unmasked, which is op 5's semantics, so every combo is mask-capable and a masked shape
        takes the same regime tree as an unmasked one.

    Args:
        configs: The candidate `AutotuneConfig` list, each carrying a combo's projected fields as
            strings (see `_as_autotune_config`).
        named_args: The bound call arguments; ``x`` must be the ``(B, N, N, D)`` input.
        **kwargs: Further bound arguments, merged over `named_args`.

    Returns:
        The surviving list, possibly empty. **Empty is a legal answer here** and is not smoothed
        over with a fallback: at ``N % 8 != 0`` no combo can run op 7, and returning a combo that
        will then refuse the shape is the exact defect this module's tests exist to prevent. The
        front door checks the floor first and raises, so an empty list is unreachable through it.
    """
    args = {**named_args, **kwargs}
    x = args["x"]
    B, N, D = x.shape[0], x.shape[-2], x.shape[-1]
    if get_device_capacity(x.device)[0] != 9:
        return []
    front_ok = _front_validity(D)
    glg_ok, q3k_ok, lng_ok = _back_validity(N, D, B)
    tri_ok = _gemm1_validity(N)
    backgate_ok = _backgate_validity(N, D)
    q3k_lt_ok = _xgate_q3k_lt_validity(N, D, B)
    out = []
    for conf in configs:
        c = _from_autotune_config(conf)
        if c.variant == "dual_gated_gemm__gemm__layernorm_dual_gated_gemm":
            if backgate_ok:
                out.append(conf)
            continue
        if c.variant == "dual_gated_gemm__gemm__layernorm__gemm_hadamard":
            if q3k_lt_ok:
                out.append(conf)
            continue
        if c.variant not in (
            "gemm_layernorm_gemm",
            "gemm__layernorm__gemm_hadamard",
            "gemm__layernorm_gemm",
        ):
            continue
        if not front_ok or c.front_v not in FRONT_VARIANTS:
            continue
        if c.variant == "gemm_layernorm_gemm":
            if not (glg_ok and B == 1):
                continue
        elif c.variant == "gemm__layernorm__gemm_hadamard":
            if not (q3k_ok and tri_ok):
                continue
        else:  # gemm__layernorm_gemm
            if not (lng_ok and tri_ok and c.lng_inner in LNG_INNER_VARIANTS):
                continue
        out.append(conf)
    return out


# ───────────────────────────── the autotune-free size heuristic ──────────────────────────────────
# Maps (B, N, D) straight to a combo by BOUND regime -- no timing, no memory query, no capacity
# gate. VALIDITY FIRST at every branch: the returned combo must survive `_prune` for the same shape,
# and `tests/workflows/test_trimul_autotune.py` asserts exactly that by brute force over a wide grid.
#
# Three regimes, from the upstream's measured sweep:
#   * LAUNCH-tiny (N^2*D <= LAUNCH_MAX_NND): the back is trivial, so the combo with the fewest
#     dispatches wins -- gemm_layernorm_gemm, when its own floors hold and the TOTAL work B*N^2*D is in the corner.
#   * COMPUTE large-D (above the knee, D >= COMPUTE_MIN_D): gemm__layernorm__gemm_hadamard -- a stock-GEMM op 7 at roofline,
#     a memcpy-speed transposing LayerNorm, and a Hadamard-epilogue GEMM.
#   * BULK (above the knee, D <= 256 and D % 32 == 0): dual_gated_gemm__gemm__layernorm_dual_gated_gemm dominates.
#   * FALLBACK: gemm__layernorm_gemm, the broadest FRONT-GATE combo -- the off-grid path this workflow keeps for D % 32 != 0.
#
# The thresholds are arch-dependent and keyed below; the VALIDITY gates are hardware correctness and
# live in the tree, not the dict.
_TRIMUL_PERF = {
    "H200_SXM5": {
        "LAUNCH_MAX_NND": 8.4e6,
        "COMPUTE_MIN_D": 512,
        "BACKGATE_ALG_FOLD_HI_D": 256,
        "BACKGATE_ALG_FOLD_HI_N": 512,
        "BACKGATE_ALG_FOLD_LO_N": 1024,
    },
    # Measured on 8x H100 80GB HBM3 upstream. **Every constant keeps its H200 value**, and the two
    # rejections are the real result: a COMPUTE_MIN_D of 384 was rejected because the q3k onset is
    # N-dependent at that width (four cells over the bar, worst 1.40), and a LAUNCH_MAX_NND of
    # 1.35e7 was rejected because the knee BENDS at larger D (dual_gated_gemm__gemm__layernorm_dual_gated_gemm beats the launch pick by 38%
    # at D=384 and 28% at D=640 inside the widened corner, against a worst 12% cost for keeping
    # 8.4e6). Both candidates came from sampling a threshold at TWO points and inferring a
    # functional form; bisect at three well-separated points spanning the deployed range before
    # concluding anything about a threshold's SHAPE.
    "H100_SXM5": {
        "LAUNCH_MAX_NND": 8.4e6,
        "COMPUTE_MIN_D": 512,
        "BACKGATE_ALG_FOLD_HI_D": 256,
        "BACKGATE_ALG_FOLD_HI_N": 512,
        "BACKGATE_ALG_FOLD_LO_N": 1024,
    },
}


def _trimul_heuristic_config(
    N: int,
    D: int,
    direction: str,
    has_mask: bool,
    B: int = 1,
    device=None,
) -> TriMulConfig:
    """Pick the combo for this shape by bound regime -- no timing, no memory query.

    Purpose
        The default path. ``select="heuristic"`` runs this instead of a sweep, so a caller pays no
        tuning cost and gets a deterministic pick.

    Semantics
        **VALIDITY-GATED AT EVERY BRANCH, and that is the whole reason this function is shaped the
        way it is.** The upstream shipped two branches that returned a combo its own pruner rejects
        -- the launch corner missing ``glg``'s ``N % 8`` floor, and the compute branch returning
        ``gemm__layernorm__gemm_hadamard`` unconditionally, which misfires at 21 distinct widths at
        or above 528. Both are reachable from the shipped default. Both are fixed upstream and
        ported fixed; what is new here is that `tests/workflows/test_trimul_autotune.py` asserts the invariant
        by BRUTE FORCE over a wide grid, so the family cannot come back one branch at a time.

        `direction` and `has_mask` are carried and do not branch. The einsum only swaps which axis
        is contracted, and the masked winner family measured identical to the unmasked one in every
        shared cell of the upstream's re-sweep -- so a separate masked rule would be a claim the
        measurement does not support.

        ARCH-AWARE in the perf layer only: the thresholds were bisected on `TUNED_ARCH`, and any
        other architecture warns ONCE and uses that set. A mis-tuned threshold is slow; a
        mis-derived validity gate is a crash, which is why the gates are not arch-keyed.

    Args:
        N: The token extent.
        D: The feature width.
        direction: ``"outgoing"`` or ``"incoming"``. Carried for signature stability.
        has_mask: Whether a mask is supplied. Carried; see Semantics.
        B: The batch.
        device: The device whose arch selects the thresholds, or None for `TUNED_ARCH` with no
            warning. **Pass ``x.device``** from anything that will then time the result.

    Returns:
        A `TriMulConfig` that survives `_prune` for this shape, provided the shape can run at all --
        which the caller must have established with `_gemm1_validity` and `_front_validity` first.
    """
    arch = heuristic_arch(device) if device is not None else TUNED_ARCH
    if arch != TUNED_ARCH:
        warn_arch_suboptimal_once(arch, "trimul")
    perf = _TRIMUL_PERF.get(arch, _TRIMUL_PERF[TUNED_ARCH])

    glg_ok, q3k_ok, lng_ok = _back_validity(N, D, B)
    backgate_ok = _backgate_validity(N, D)
    # The FRONT-GATE front is `prolog_ln` at EVERY width, and the reason is VALIDITY rather than
    # speed.
    #
    # Upstream this reads "the prologue front when D % 128 == 0, else the fold", and its own comment
    # says why: `staged when D%128==0 (the staged-front gate3 floor N3=D)`. The condition is a
    # LEGALITY test. The rule was never "the fold is faster at these widths" -- it was "use the
    # prologue front wherever it can run, and fall back where it cannot". That floor does not bind
    # here (see `_front_validity`: this package's front doors take `tile_N` explicitly and pad the
    # gate's rows to a whole tile, measured 20/20 across both fusions), so there is nothing left to
    # fall back FROM. `prolog_ln` unconditionally is not a new rule -- it is the upstream's rule
    # evaluated against this tree's validity.
    #
    # CORROBORATION, not the justification. Paired medians (`benchmark_paired`, 41x20, N=128, B=1),
    # ratio to the fold as baseline, whole gemm__layernorm_gemm chain: 0.7596 at D=64, 0.7534 at
    # 128, 0.7585 at 136, 0.7660 at 200, 0.7594 at 256, 0.7581 at 320, 0.7592 at 384, 0.8011 at 512.
    # The prologue front is faster at every width, so the widths the old condition sent to the fold
    # were paying about 24% of the chain for it.
    #
    # **That table has a shelf life and the argument above does not.** Most of the gap is
    # `layernorm_dual_gated_gemm`'s fold path re-doing `build_folded_dual_operands` +
    # `build_folded_gate3_operands` + `append_gate3_weight` on EVERY call -- three host passes over
    # a loop-invariant weight. Hoisting that is an open follow-up, and when it lands this
    # measurement must be re-run; the validity argument is unaffected either way. One cell is
    # unexplained: at D=200 the front measured in ISOLATION reads 0.9453 where every other width
    # reads ~0.58. The chain ratio there is 0.7660, in line with the rest, so the conclusion does
    # not rest on it and it is recorded rather than guessed at.
    frontgate_front_v = "prolog_ln"

    # 1. LAUNCH-tiny corner. gemm_layernorm_gemm has the fewest dispatches and the back is trivial here, but it
    #    is B == 1 only and carries its own N % 8 floor; without that floor an off-grid N inside the
    #    corner returns a combo `_prune` rejects.
    if N * N * D <= perf["LAUNCH_MAX_NND"]:
        in_corner = B * N * N * D <= perf["LAUNCH_MAX_NND"]
        if in_corner and B == 1 and glg_ok and D % 128 == 0:
            front = (
                "lng2k"
                if (
                    N >= 256
                    or D >= perf["COMPUTE_MIN_D"]
                    or (B > 1 and D >= perf["COMPUTE_MIN_D"] // 2)
                )
                else frontgate_front_v
            )
            return TriMulConfig("gemm_layernorm_gemm", front_v=front)
        if backgate_ok:
            return TriMulConfig(
                "dual_gated_gemm__gemm__layernorm_dual_gated_gemm", back_v="prolog_ln"
            )
        if in_corner and B == 1 and glg_ok:
            # This is the SECOND arm of the same legality test `frontgate_front_v` replaced --
            # upstream hard-codes the fold here for the same reason it does in that condition,
            # because this branch is reached at exactly the widths where its prologue front was
            # illegal. The floor does not bind here either, so it takes `frontgate_front_v` too.
            # Missing this site would have left the change half-applied at D=136 and D=200 -- the
            # two widths in the measurement that the old rule sent to the fold AND that reach this
            # branch.
            return TriMulConfig("gemm_layernorm_gemm", front_v=frontgate_front_v)
        if lng_ok and _gemm1_validity(N):
            return TriMulConfig(
                "gemm__layernorm_gemm", front_v=frontgate_front_v, lng_inner="alg_fold"
            )
        return TriMulConfig("gemm__layernorm__gemm_hadamard", front_v=frontgate_front_v)
    # 2. COMPUTE large-D. Validity-gated: this branch used to return gemm__layernorm__gemm_hadamard unconditionally and so
    #    handed back a combo the pruner rejects whenever the transposing LayerNorm's floors fail at
    #    that width. Falling THROUGH is the conservative choice -- a different front might also
    #    serve those widths, but that is unmeasured, and a perf guess does not belong in a validity
    #    fix.
    if D >= perf["COMPUTE_MIN_D"] and q3k_ok and _gemm1_validity(N):
        return TriMulConfig("gemm__layernorm__gemm_hadamard", front_v="lng2k")
    # 3. dual_gated_gemm__gemm__layernorm_dual_gated_gemm bulk. The corner where the wide-tile fold beats the prologue is a measured
    #    rectangle, not a threshold on either axis alone.
    if backgate_ok:
        back = (
            "alg_fold"
            if (
                (D >= perf["BACKGATE_ALG_FOLD_HI_D"] and N >= perf["BACKGATE_ALG_FOLD_HI_N"])
                or (D < perf["BACKGATE_ALG_FOLD_HI_D"] and N >= perf["BACKGATE_ALG_FOLD_LO_N"])
            )
            else "prolog_ln"
        )
        return TriMulConfig("dual_gated_gemm__gemm__layernorm_dual_gated_gemm", back_v=back)
    # 4. Fallback: the broadest FRONT-GATE combo. This is the off-grid path -- every width that is not a multiple
    #    of 32 arrives here, which is what makes `layernorm_gemm` the workflow's universal fallback.
    if lng_ok and _gemm1_validity(N):
        return TriMulConfig("gemm__layernorm_gemm", front_v=frontgate_front_v, lng_inner="alg_fold")
    return TriMulConfig("gemm__layernorm__gemm_hadamard", front_v=frontgate_front_v)


# ───────────────────────────── op 7: the token-pair einsum ───────────────────────────────────────


def _gemm1(a: Tensor, b: Tensor, B: int, N: int, D: int, direction: str) -> Tensor:
    """Op 7 -- ``tri`` from the front's ``a`` and ``b``, batched over feature AND batch.

    Purpose
        The one place the contraction convention is written down. It is derived from the einsum
        rather than transliterated, because this package's `gemm` computes ``A @ B^T`` natively
        while the kernel this ports from computes ``A @ B`` -- so the upstream's transposes do NOT
        carry across, and copying them would produce a plausible, wrong tensor with no error.

    Semantics
        `a` and `b` arrive ``(D, M)`` contiguous with the token axis flattened row-major,
        ``m = ((b * N) + i) * N + j``. Viewing them ``(D, B, N, N)`` and fusing the two leading axes
        gives one batch axis ``L = d * B + b``, which is the single leading dim `gemm` takes.

        Per ``L``, writing ``A`` and ``Bm`` for the two ``(N, N)`` slices::

            outgoing  T[i,j] = sum_k A[i,k] * Bm[j,k]  =  A @ Bm^T   -> gemm(af, bf)
            incoming  T[i,j] = sum_k A[k,i] * Bm[k,j]  =  A^T @ Bm   -> gemm(af.mT, bf.mT)

        The outgoing case takes **NO transpose at all**: `gemm` already contracts the last axis of
        both operands, so the upstream's ``.transpose(-1, -2)`` on the second operand cancels
        against that. The incoming case transposes BOTH, which is not what the upstream writes
        either -- there ``A^T @ Bm`` needs only the first operand moved. Both were verified against
        a torch ``einsum`` at ``(B, N, D)`` of (1,128,64), (1,136,64), (2,64,32) and (3,40,16), in
        both directions, to bf16 accumulation error.

        A transposed view is passed to `gemm` UNCOPIED. Its major mode is derived from the strides,
        so an ``m``-major operand is a different compiled kernel rather than a materialized copy.

    Args:
        a: The front's first half, ``(D, M)`` contiguous. Must be 16-bit; op 7 is a WGMMA.
        b: The front's second half, same shape and dtype.
        B: The batch.
        N: The token extent. Must satisfy `_gemm1_validity` -- at ``N % 8 != 0`` the ``(N, N)``
            operands have a row pitch that is not 16-byte aligned and `gemm` refuses them by name.
        D: The feature width.
        direction: ``"outgoing"`` or ``"incoming"``.

    Returns:
        ``(D * B, N, N)`` in `a`'s dtype, with ``L = d * B + b``. The caller recovers the
        ``(M, D)`` LayoutLeft view op 8 wants as ``tri.reshape(D, M).t()``, which preserves the
        ``(b, i, j)`` token order.

    Raises:
        ValueError: From `gemm`, if `N` breaks the alignment floor.
    """
    af = a.reshape(D, B, N, N).reshape(D * B, N, N)
    bf = b.reshape(D, B, N, N).reshape(D * B, N, N)
    out = torch.empty(D * B, N, N, device=a.device, dtype=a.dtype)
    # TUNED, because `main` is tuned here: its bare `_gemm(af, bf.transpose(-1, -2))` enters
    # `gemm_interface.gemm_out` with `tuned=True`, whose `gemm_tuned` is `@autotune`-decorated and
    # keys on the operand shapes -- so `main` SWEEPS op 7 per shape. `do_autotune=True` is this
    # tree's spelling of that, and it is the parity behaviour, not an enhancement.
    #
    # This replaced a local `_gemm1_tiles(N)` helper `main` has no counterpart for, which pinned a
    # flat (128, 128) and whose own docstring called itself a placeholder with no sweep behind it.
    # Measured paired against `main`, e2e:
    #     N_token=2048 / D=128    1.425x placeholder -> 0.989x
    #     N_token=4096 / D=128    op 7 alone 41537us vs main 23024us = 1.80x on the pinned config
    # The 2048 cell is why a STATIC default looked sufficient: there the sweep's winner and
    # `gemm.default_config()` agree, so pinning the default read as parity. At 4096 they do not.
    # A single-cell perf check cannot tell a tuned dispatch from a lucky constant -- that is the
    # whole argument for measuring the declared matrix rather than one shape.
    if direction == "outgoing":
        gemm(af, bf, out, None, None, do_autotune=True)
    else:
        gemm(af.transpose(-1, -2), bf.transpose(-1, -2), out, None, None, do_autotune=True)
    return out


# `_gemm1_tiles(N)` stood here: a local picker returning a flat (128, 128), which `main` has no
# counterpart for -- there op 7 names no tile and `gemm_tuned` resolves `config is None` to
# `default_config`. It is deleted rather than fixed because a second picker is the defect: two trees
# choosing op 7's config by different rules is how the e2e workflow ran 1.425x `main` while every
# per-kernel gate stayed green. `gemm.default_config()` is now the single rule, in both trees.


# ───────────────────────────── the chains ────────────────────────────────────────────────────────

#: The CTA tile M every front is launched with. `layernorm_gemm` picks its own from its size
#: heuristic and is not covered by this.
#:
#: **These are `main`'s values, not a placeholder.** An earlier version of this note said it stood
#: in for a sweep nobody had run, which was false for every seam it covered except one -- and a
#: false-but-plausible note gets acted on. Measured 2026-08-15:
#:
#:   * **tile_M** -- `main` pins 128 too, one layer down, at `gated_gemm_gate.py:178-179`:
#:     *"tile_M is pinned at 128 (tile_M=256 was 6-14x slower; 192 invalid)"*. `main`'s workflow
#:     passing no tile at its call sites does NOT mean it tunes there; the pin lives in the callee.
#:     **A value's absence at one layer is not evidence of its absence.**
#:   * **tile_N** -- `dual_tile_n(D)` and `_xgate_tile_n(D)` agree with `main`'s auto-picks at all
#:     four declared D once the UNIT is converted: `main`'s `tile_N` is PER-HALF
#:     (`dual_gated_gemm_staged.py:3125`) with `BLK_N = 2*tile_N` (`:3313`), ours is full-2N. Raw,
#:     they read 256 vs 128 and look like a 2x divergence; converted, 8/8 identical.
#:
#: The Hadamard back was the ONE genuine exception -- `main` tunes it via `gemm_elem(tuned=True)`,
#: we pinned, and closing that was worth 1.1549 -> 0.9997 (`1abdfdd`). One comment covered five
#: seams and was right about exactly one of them, which is why the other four are now stated with
#: their provenance instead of an apology for missing data.
_FRONT_TILE_M = 128


def _run_front(
    x2d,
    norm_in_w,
    norm_in_b,
    g_in_w,
    p_in_w,
    g_in_b,
    p_in_b,
    g_out_w,
    g_out_b,
    mask,
    eps,
    M,
    D,
    front_v,
    gated,
):
    """Ops 1-6 (and 10-11 when `gated`): produce ``a``, ``b`` and optionally ``gate3``.

    Purpose
        The one front implementation, shared by all five combos. Which of the three fusions runs is
        the only thing that varies; every one of them emits the SAME ``(a, b[, gate3])``, which is
        what lets the back-cuts consume them unchanged.

    Semantics
        The dual weights are each ``(2D, D)``, so the GLU output width is ``n = 2D`` and the
        pre-activation the kernel tiles is ``2n = 4D``. `PostAct` is allocated as ``(2D, M)`` and
        passed as its TRANSPOSE: the m-major store is selected by STRIDES, so a raw ``(2D, M)``
        tensor would be a shape mismatch reported against a symbol name rather than a transposed
        store. Op 6 is then FREE -- ``a = buf[:D]`` and ``b = buf[D:]`` are contiguous ``(D, M)``
        slices of that same allocation, with no copy and no kernel.

        The mask is applied to the dual ``[a|b]`` ONLY. `gate3` is left unmasked, which is op 5's
        semantics and not an oversight.

    Args:
        x2d: ``(M, D)`` RAW activation. Not pre-normalized -- the fused fronts normalize internally
            and the ``lng2k`` front runs its own `layernorm_fwd` here.
        norm_in_w: ``(D,)`` fp32 LayerNorm gain.
        norm_in_b: ``(D,)`` fp32 LayerNorm bias, or None.
        g_in_w: ``(2D, D)`` gate weight, in `x2d`'s dtype.
        p_in_w: ``(2D, D)`` up weight, same dtype.
        g_in_b: ``(2D,)`` gate bias, or None. Added before the gate's sigmoid (op 3).
        p_in_b: ``(2D,)`` up bias, or None. Added before the product (op 2).
        g_out_w: ``(D, D)`` output-gate weight. Read only when `gated`.
        g_out_b: ``(D,)`` output-gate bias, or None. Read only when `gated`.
        mask: ``(M,)`` per-token mask, or None.
        eps: The LayerNorm variance floor.
        M: The flattened token count.
        D: The feature width. Must satisfy `_front_validity`.
        front_v: A member of :data:`FRONT_VARIANTS`.
        gated: Whether to build ``gate3`` here (every FRONT-GATE combo) or leave it to the back (the
        BACK-GATE combos).

    Returns:
        ``(a, b, gate3)`` -- two ``(D, M)`` contiguous halves and an ``(M, D)`` row-major gate, the
        last None when not `gated`.

    Raises:
        ValueError: From the front door, for a `front_v` outside :data:`FRONT_VARIANTS` or a tile
            the geometry refuses.
    """
    dt = x2d.dtype
    tile_N = dual_tile_n(D)
    buf = torch.empty(2 * D, M, device=x2d.device, dtype=dt)
    post = buf.T  # (M, 2D) m-major: the transposed store is selected by STRIDES
    gate3 = torch.empty(M, D, device=x2d.device, dtype=dt) if gated else None
    w3 = g_out_w if gated else None
    b3 = g_out_b if gated else None
    if front_v == "lng2k":
        x_norm = layernorm_fwd(x2d, norm_in_w, norm_in_b, eps=eps)
        dual_gated_gemm(
            x_norm,
            g_in_w,
            p_in_w,
            post,
            _FRONT_TILE_M,
            tile_N,
            bg=g_in_b,
            bp=p_in_b,
            mask=mask,
            W3=w3,
            b3=b3,
            PostAct3=gate3,
        )
    else:
        layernorm_dual_gated_gemm(
            x2d,
            norm_in_w,
            g_in_w,
            p_in_w,
            post,
            _FRONT_TILE_M,
            tile_N,
            norm_bias=norm_in_b,
            bg=g_in_b,
            bp=p_in_b,
            mask=mask,
            eps=eps,
            W3=w3,
            b3=b3,
            PostAct3=gate3,
            fusion_variant=front_v,
        )
    return buf[:D], buf[D:], gate3


def _tri_layout_left(tri: Tensor, M: int, D: int) -> Tensor:
    """View op 7's output as the ``(M, D)`` LayoutLeft activation op 8 consumes.

    Purpose
        The one place the token order is re-asserted. Getting it wrong permutes the output rows
        without changing any value, which no norm-based check would catch.

    Semantics
        `tri` is ``(D * B, N, N)`` with ``L = d * B + b``, so its flat storage runs ``d`` slowest
        and ``(b, i, j)`` fastest -- i.e. it IS the ``(D, M)`` matrix with the token axis flattened
        row-major. Transposing that gives ``(M, D)`` with stride ``(1, M)``: a VIEW, no copy. Every
        back that follows takes an m-major activation, which is why op 7 is left in this order
        rather than transposed back.

    Args:
        tri: `_gemm1`'s output, ``(D * B, N, N)``.
        M: The flattened token count.
        D: The feature width.

    Returns:
        An ``(M, D)`` view with stride ``(1, M)``. Never a copy.
    """
    return tri.reshape(D, M).t()


def _trimul_config_is_valid(config, request) -> bool:
    """Whether one candidate can run this request. Rejection only -- never preference.

    Purpose
        The autotuner's per-candidate gate. `_prune` answers the same question for a whole list, and
        this is the single-candidate adapter the decorator wants; keeping ONE implementation is what
        stops the sweep and the heuristic's self-check from drifting apart.

    Semantics
        Must be a PURE function of the config and the request. A non-pure one does not merely
        mis-tune: under a collective it makes two ranks measure different pools, which is a hang.

        **The tuner passes an `AutotuneConfig`, and this callback must NOT re-wrap it.** The
        signature says nothing about the type, which is how the mistake got in and stayed: the
        annotation is bare and every `validity=` callback in the package is free to guess. The
        other six guess correctly. This one guessed "dict", and `AutotuneConfig(**config)` on a
        non-mapping raises `TypeError` at the first candidate of every sweep.

    Args:
        config: The candidate, as an `AutotuneConfig` -- which is what the tuner passes and what
            `_prune` takes, so it is forwarded UNWRAPPED. It used to be re-wrapped as
            ``AutotuneConfig(**config)`` against a docstring that called it "the tuner's dict",
            and since ``**`` refuses a non-mapping that raised ``TypeError`` on the first
            candidate of every sweep -- taking out ``select="autotune"`` at every shape, and
            `trimul_freeze` with it. Nothing noticed because no test drove either: a ``_config=``
            pin short-circuits ahead of the validity gate and ``select="heuristic"`` never reaches
            the tuner at all.
        request: The bound call arguments by name; ``x`` supplies the shape and the device.

    Returns:
        True when `_prune` keeps this candidate for this shape.
    """
    return bool(_prune([config], request))


@autotune(
    configs=[_as_autotune_config(c) for c in _trimul_configs()],
    key=["direction", "has_bias", "has_mask"],
    validity=_trimul_config_is_valid,
)
def _trimul_impl(
    x: Tensor,
    norm_in_w: Tensor,
    norm_in_b: Optional[Tensor],
    p_in_w: Tensor,
    g_in_w: Tensor,
    norm_out_w: Tensor,
    norm_out_b: Optional[Tensor],
    p_out_w: Tensor,
    g_out_w: Tensor,
    p_in_b: Optional[Tensor],
    g_in_b: Optional[Tensor],
    p_out_b: Optional[Tensor],
    g_out_b: Optional[Tensor],
    mask: Optional[Tensor],
    direction: str,
    eps: float,
    has_bias: bool,
    has_mask: bool,
    variant: str = "gemm__layernorm_gemm",
    front_v: str = "alg_fold",
    lng_inner: str = "alg_fold",
    back_v: str = "alg_fold",
) -> Tensor:
    """Run one combo end to end, from the raw input to the gated output.

    Purpose
        The timed body. Every cast and every launch happens inside it, so a sweep charges each
        candidate for the host work it actually costs rather than for its mainloop alone.

    Semantics
        The GEMM weights are cast to the activation's dtype HERE rather than by the caller, for the
        same reason: a candidate that needs a cast should pay for it. The LayerNorm gains and biases
        stay fp32, which every kernel here requires.

        Dispatch is on `variant`. Two of the five reach kernels this tree does not have yet and say
        so by name; the rest run.

    Args:
        x: ``(B, N, N, D)`` RAW input. Not pre-normalized.
        norm_in_w, norm_in_b: ``(D,)`` fp32 front LayerNorm gain and bias (bias optional).
        p_in_w, g_in_w: ``(2D, D)`` up and gate projections.
        norm_out_w, norm_out_b: ``(D,)`` fp32 back LayerNorm gain and bias (bias optional).
        p_out_w, g_out_w: ``(D, D)`` output up and gate projections.
        p_in_b, g_in_b: ``(2D,)`` up and gate biases, or None.
        p_out_b, g_out_b: ``(D,)`` output up and gate biases, or None.
        mask: ``(M,)`` per-token mask, or None. Applied to the dual only.
        direction: ``"outgoing"`` or ``"incoming"``.
        eps: The LayerNorm variance floor, shared by both LayerNorms.
        has_bias, has_mask: Cache-key components. Read by the tuner, not by this body -- a tensor
            is not a key, so the presence of one has to travel as a bool.
        variant: The combo; a member of :data:`TRIMUL_VARIANTS`.
        front_v: The front fusion for a FRONT-GATE combo.
        lng_inner: ``gemm__layernorm_gemm``'s back fusion.
        back_v: A BACK-GATE combo's back fusion.

    Returns:
        ``(B, N, N, D)`` in `x`'s dtype.

    Raises:
        ValueError: On an unrecognised `variant`.
        NotImplementedError: On a combo whose kernel has not landed in this tree yet, naming it.
    """
    B, N, _, D = x.shape
    M = B * N * N
    dt = x.dtype
    x2d = x.reshape(M, D)
    g_in_w_c, p_in_w_c = g_in_w.to(dt), p_in_w.to(dt)
    p_out_w_c, g_out_w_c = p_out_w.to(dt), g_out_w.to(dt)

    if variant in (
        "dual_gated_gemm__gemm__layernorm_dual_gated_gemm",
        "dual_gated_gemm__gemm__layernorm__gemm_hadamard",
    ):
        return _run_backgate(
            x2d,
            norm_in_w,
            norm_in_b,
            g_in_w_c,
            p_in_w_c,
            g_in_b,
            p_in_b,
            norm_out_w,
            norm_out_b,
            p_out_w_c,
            g_out_w_c,
            p_out_b,
            g_out_b,
            mask,
            direction,
            eps,
            B,
            N,
            D,
            M,
            variant,
            back_v,
        )

    a, b, gate3 = _run_front(
        x2d,
        norm_in_w,
        norm_in_b,
        g_in_w_c,
        p_in_w_c,
        g_in_b,
        p_in_b,
        g_out_w_c,
        g_out_b,
        mask,
        eps,
        M,
        D,
        front_v,
        gated=True,
    )

    if variant == "gemm_layernorm_gemm":
        # Ops 7, 8, 9 and 12 in ONE kernel, so it takes `a` and `b` rather than a materialized
        # `tri`. The front emits them (D, M) = (D, B, N, N) d-major; that kernel wants (B, D, N, N)
        # so each token's features are contiguous for the LayerNorm over D. At B == 1 the permute is
        # free; above it, it is a real copy and part of what the combo costs.
        try:
            from fold_cp_ops.kernels.gemm_layernorm_gemm import gemm_layernorm_gemm
        except ImportError as e:  # pragma: no cover - exercised only before that kernel lands
            raise NotImplementedError(
                "gemm_layernorm_gemm needs fold_cp_ops.kernels.gemm_layernorm_gemm, which "
                "has not landed in this tree yet. Force gemm__layernorm_gemm or "
                "gemm__layernorm__gemm_hadamard instead -- both run here, and "
                "gemm__layernorm_gemm runs at every 16-byte-aligned width."
            ) from e
        if B == 1:
            a_g, b_g = a.view(1, D, N, N), b.view(1, D, N, N)
        else:
            a_g = a.reshape(D, B, N, N).permute(1, 0, 2, 3).contiguous()
            b_g = b.reshape(D, B, N, N).permute(1, 0, 2, 3).contiguous()
        # No `select=` here, unlike every other kernel in the chain: this front door exposes its
        # block sizes directly and offers `do_autotune` rather than a size heuristic, so the
        # defaults are what a non-tuning caller gets. `proj_w` goes in UNTRANSPOSED -- it is
        # declared ``(d', d)``, so the kernel already contracts its last axis, which is op 9.
        out2d = gemm_layernorm_gemm(
            a_g,
            b_g,
            norm_out_w,
            norm_out_b,
            p_out_w_c,
            p_out_b,
            direction=direction,
            eps=eps,
            gate3=gate3,
        ).reshape(M, D)
    elif variant == "gemm__layernorm__gemm_hadamard":
        # Op 7, then op 8 as a transposing pass (LayoutLeft tri -> row-major LN(tri)), then ops 9
        # and 12 as one Hadamard-epilogue GEMM.
        tri_n = layernorm_fwd(
            _tri_layout_left(_gemm1(a, b, B, N, D, direction), M, D),
            norm_out_w,
            norm_out_b,
            eps=eps,
            transpose=True,
        )
        out2d = torch.empty(M, D, device=x.device, dtype=dt)
        # `gemm_hadamard` computes (A @ B^T + rowvec) * C, so op 9's `trin @ p_out_w^T` needs NO
        # transpose on the weight: the transpose is already in what the kernel computes.
        #
        # `do_autotune=True` rather than a pinned tile, matching what `main`'s workflow does at this
        # seam. The tile is NOT passed alongside it -- this entry raises on a tuned knob supplied
        # with `do_autotune=True`, and the pin it replaces (`_FRONT_TILE_M` plus a width rule) was
        # labelled a placeholder standing in for a sweep nobody had run. Every knob the sweep moves
        # reaches the compile key, so the pick is a real artifact and not a relabelled pinned one:
        # tile and cluster and pingpong are passed to `_compile_gemm_hadamard` directly, and
        # `swap_ab` through the majors, because `D.mT` always exchanges D's two strides.
        gemm_hadamard(
            tri_n.unsqueeze(0),
            p_out_w_c.unsqueeze(0),
            out2d.unsqueeze(0),
            gate3.unsqueeze(0),
            None,
            rowvec_bias=None if p_out_b is None else p_out_b.unsqueeze(0),
            do_autotune=True,
        )
    elif variant == "gemm__layernorm_gemm":
        # Op 7, then ops 8, 9 and 12 as one fused kernel. Its B operand is (K, P) with K = P = D and
        # `LN(x) @ B`, so it takes p_out_w TRANSPOSED -- the opposite of the Hadamard back above,
        # because that one contracts B's last axis and this one contracts its first.
        if lng_inner not in LNG_INNER_VARIANTS:
            raise NotImplementedError(
                f"gemm__layernorm_gemm's lng_inner={lng_inner!r} is not implemented; "
                f"layernorm_gemm builds {LNG_INNER_VARIANTS}."
            )
        out2d = layernorm_gemm(
            _tri_layout_left(_gemm1(a, b, B, N, D, direction), M, D),
            norm_out_w,
            p_out_w_c.t().contiguous(),
            bias=norm_out_b,
            eps=eps,
            gate3=gate3,
            out_bias=p_out_b,
            select="heuristic",
        )
    else:
        raise ValueError(f"unknown variant {variant!r}; expected one of {TRIMUL_VARIANTS}")
    return out2d.reshape(B, N, N, D)


def _run_backgate(
    x2d,
    norm_in_w,
    norm_in_b,
    g_in_w,
    p_in_w,
    g_in_b,
    p_in_b,
    norm_out_w,
    norm_out_b,
    p_out_w,
    g_out_w,
    p_out_b,
    g_out_b,
    mask,
    direction,
    eps,
    B,
    N,
    D,
    M,
    variant,
    back_v,
):
    """The the BACK-GATE combos chain: the output gate computed in the BACK from a shared normalized
    input.

    Purpose
        Split out of `_trimul_impl` because it shares nothing with the FRONT-GATE back-cuts but the
        front and op 7 -- it has no ``gate3`` at all, and its back takes two activations.

    Semantics
        One LayerNorm produces ``xn``, which is used TWICE: by the unfused front for ops 2-4, and by
        the back as the gate's own input for ops 10-11. That sharing is the combo's whole argument;
        it is also why the front here carries no ``W3``.

        ``dual_gated_gemm__gemm__layernorm__gemm_hadamard`` differs from
        ``dual_gated_gemm__gemm__layernorm_dual_gated_gemm`` in the PLACEMENT of op 8 and nothing
        else: it runs the transposing LayerNorm as its own pass and hands the back a pre-normalized
        value with the back's own normalize switched off.

    Args:
        x2d: ``(M, D)`` raw activation.
        norm_in_w, norm_in_b: front LayerNorm gain and bias.
        g_in_w, p_in_w, g_in_b, p_in_b: the front's dual weights and biases.
        norm_out_w, norm_out_b: back LayerNorm gain and bias.
        p_out_w, g_out_w: ``(D, D)`` output value and gate projections.
        p_out_b, g_out_b: their biases, or None.
        mask: ``(M,)`` mask or None; applied in the FRONT only.
        direction: op 7's direction.
        eps: The variance floor.
        B, N, D, M: The shape.
        variant: ``"dual_gated_gemm__gemm__layernorm_dual_gated_gemm"`` or
        ``"dual_gated_gemm__gemm__layernorm__gemm_hadamard"``. back_v: The back's LayerNorm fusion.

    Returns:
        ``(B, N, N, D)``.

    Raises:
        NotImplementedError: For ``dual_gated_gemm__gemm__layernorm__gemm_hadamard``, which is not
        expressible here -- see below. It
            is not in the swept grid, so only an explicit ``_config`` reaches this.
    """
    if variant == "dual_gated_gemm__gemm__layernorm__gemm_hadamard":
        # NOT "not yet written" -- NO front door in this package expresses it, and that is a
        # decision made deliberately elsewhere rather than a gap. The combo wants a two-activation
        # GEMM with NO LayerNorm on either arm, because op 8 has already been done as a separate
        # transposing pass. Upstream reaches that with a `_normalize=False` flag on this same
        # kernel; `layernorm_dual_gated_gemm` refuses to port that flag, on the stated ground that a
        # pre-normalized VALUE makes the whole thing an unfused dual-x GEMM and belongs on
        # `dual_gated_gemm` -- which this package has and upstream did not. But `dual_gated_gemm`
        # takes ONE activation, so the two-A no-LayerNorm shape has no home in either.
        raise NotImplementedError(
            "dual_gated_gemm__gemm__layernorm__gemm_hadamard is not expressible in this "
            "package, and the gap is structural rather than pending. It needs a "
            "TWO-ACTIVATION GEMM with no LayerNorm on either arm (op 8 having been hoisted "
            "into a separate transposing pass). layernorm_dual_gated_gemm deliberately does "
            "not port upstream's `_normalize=False` -- its docstring gives the reason -- and "
            "dual_gated_gemm takes only one activation. Closing this needs an x_gate "
            "parameter on dual_gated_gemm, which is a kernel change and not a dispatcher "
            "one. dual_gated_gemm__gemm__layernorm_dual_gated_gemm computes the SAME output "
            "and does run."
        )

    # ── dual_gated_gemm__gemm__layernorm_dual_gated_gemm. ONE LayerNorm feeds two consumers, which
    # is the combo's whole argument: the front's dual reads `xn` for ops 2-4, and the back's gate
    # reads the SAME `xn` for ops 10-11. That is why the front here carries no W3 -- there is no
    # gate3 to produce.
    x_norm = layernorm_fwd(x2d, norm_in_w, norm_in_b, eps=eps)  # (M, D) row-major = xn
    tile_N = dual_tile_n(D)
    buf = torch.empty(2 * D, M, device=x2d.device, dtype=x2d.dtype)
    # The weight layout, chosen the way `main` chooses it. `chunk_g=16` hands Wg and Wp straight to
    # a two-tensor TMA; `chunk_g=1` makes the entry BUILD an element-interleaved weight per call,
    # which is a whole extra kernel launch `main` never makes here. This front carries no W3 (see
    # above), which is the one fused feature that would force 1.
    #
    # The tile test is a VALIDITY gate, not a preference: `chunk_g > 1` needs the work tile to hold
    # a whole up/gate block pair, and `dual_tile_n` answers to `4D`, not to `chunk_g`. Downgrading
    # to 1 keeps every shape RUNNING -- a shape whose tile cannot host the pairing must still
    # compute, so this may never become a shape constraint.
    chunk_g = heuristic_chunk_g(D, 2 * D, device=x2d.device)
    if tile_N % (2 * chunk_g):
        chunk_g = 1
    dual_gated_gemm(
        x_norm,
        g_in_w,
        p_in_w,
        buf.T,
        _FRONT_TILE_M,
        tile_N,
        bg=g_in_b,
        bp=p_in_b,
        mask=mask,
        chunk_g=chunk_g,
    )
    tri = _gemm1(buf[:D], buf[D:], B, N, D, direction)
    # The value is the MN-major view of op 7's output and the gate is the K-major `xn`. The two
    # majors are SEPARATE compile keys, so this pairing is its own artifact rather than a copy --
    # which is the case the back's register-source value path exists for.
    value = _tri_layout_left(tri, M, D)
    out2d = torch.empty(M, D, device=x2d.device, dtype=x2d.dtype)
    layernorm_dual_gated_gemm(
        value,
        norm_out_w,
        g_out_w,
        p_out_w,
        out2d,
        _FRONT_TILE_M,
        _xgate_tile_n(D),
        norm_bias=norm_out_b,
        bg=g_out_b,
        bp=p_out_b,
        x_gate=x_norm,
        eps=eps,
        fusion_variant=back_v,
    )
    return out2d.reshape(B, N, N, D)


def _xgate_tile_n(D: int) -> int:
    """The CTA tile over the two-activation back's output width.

    Purpose
        The back's one tiling decision. It is a SEPARATE helper from `dual_tile_n` because
        ``tile_N`` means something different on this path and reusing the front's would silently
        halve the useful work per tile.

    Semantics
        Without a second activation the kernel tiles a ``2n`` pre-activation, so the front's tile is
        picked against ``4D``. WITH one there is no pre-activation to halve: the tile covers the
        output width ``n = D`` directly. It carries NO divisibility requirement -- a partial last N
        tile is predicated, and even a tile wider than ``n`` is correct, merely wasteful -- so this
        only has to return something the WGMMA atom accepts.

    Args:
        D: The feature width, which is the back's output width.

    Returns:
        A multiple of 16 in ``[16, 128]``. 16 is the floor rather than a smaller exact divisor
        because that is the narrowest tile the atom builds; at ``D < 16`` the tile exceeds the
        output and predicates the remainder away.
    """
    return min(128, max(16, (D // 16) * 16))


# ───────────────────────────── the fp32 oracle ───────────────────────────────────────────────────


def trimul_ref(
    x: Tensor,
    direction: str,
    mask: Optional[Tensor],
    norm_in_w: Tensor,
    norm_in_b: Optional[Tensor],
    p_in_w: Tensor,
    g_in_w: Tensor,
    norm_out_w: Tensor,
    norm_out_b: Optional[Tensor],
    p_out_w: Tensor,
    g_out_w: Tensor,
    p_in_b: Optional[Tensor] = None,
    g_in_b: Optional[Tensor] = None,
    p_out_b: Optional[Tensor] = None,
    g_out_b: Optional[Tensor] = None,
    eps: float = 1e-5,
) -> Tensor:
    """The eager fp32 TriMul, op for op -- the ground truth every combo is checked against.

    Purpose
        The oracle. It exists in the shipped package rather than only in a test because five
        different chains claim to compute this one function, and "the same as the reference" has to
        mean the same reference for all of them.

    Semantics
        Everything is promoted to fp32 first and stays there, and the twelve ops run in the order
        the module docstring lists. Two of them are where a chain most easily goes wrong and are
        therefore written plainly here: op 5 multiplies the mask into ``ab`` BEFORE the chunk, so it
        reaches the dual and not the output gate; and op 10 contracts ``xn``, the FRONT's normalized
        input, not ``trin``.

        Not a fused-kernel model: it says nothing about rounding order, so a comparison against it
        needs a bound that accounts for the chain's own arithmetic.

    Args:
        x: ``(B, N, N, D)`` raw input, any float dtype.
        direction: ``"outgoing"`` or ``"incoming"`` -- which axis op 7 contracts.
        mask: ``(B, N, N)`` mask, or None.
        norm_in_w, norm_in_b: ``(D,)`` front LayerNorm gain and bias; the bias may be None.
        p_in_w, g_in_w: ``(2D, D)`` up and gate projections.
        norm_out_w, norm_out_b: ``(D,)`` back LayerNorm gain and bias.
        p_out_w, g_out_w: ``(D, D)`` output up and gate projections.
        p_in_b, g_in_b: ``(2D,)`` biases, or None.
        p_out_b, g_out_b: ``(D,)`` biases, or None.
        eps: The variance floor, shared by both LayerNorms. Must match what the kernel was given.

    Returns:
        ``(B, N, N, D)`` fp32.

    Raises:
        ValueError: On an unrecognised `direction`.
    """
    if direction not in ("outgoing", "incoming"):
        raise ValueError(f"direction must be 'outgoing' or 'incoming'; got {direction!r}")
    D = x.shape[-1]
    f = torch.nn.functional
    xn = f.layer_norm(
        x.float(), (D,), norm_in_w.float(), None if norm_in_b is None else norm_in_b.float(), eps
    )
    p_in = xn @ p_in_w.float().T + (0.0 if p_in_b is None else p_in_b.float())
    g_in = xn @ g_in_w.float().T + (0.0 if g_in_b is None else g_in_b.float())
    ab = p_in * torch.sigmoid(g_in)
    if mask is not None:
        ab = ab * mask.float().unsqueeze(-1)
    a, b = ab.chunk(2, dim=-1)
    eq = "bikd,bjkd->bijd" if direction == "outgoing" else "bkid,bkjd->bijd"
    tri = torch.einsum(eq, a, b)
    trin = f.layer_norm(
        tri, (D,), norm_out_w.float(), None if norm_out_b is None else norm_out_b.float(), eps
    )
    p_out = trin @ p_out_w.float().T + (0.0 if p_out_b is None else p_out_b.float())
    g_out = xn @ g_out_w.float().T + (0.0 if g_out_b is None else g_out_b.float())
    return p_out * torch.sigmoid(g_out)


# ───────────────────────────── the public entry ──────────────────────────────────────────────────


def trimul_autotuned(
    x: Tensor,
    norm_in_w: Tensor,
    norm_in_b: Optional[Tensor] = None,
    p_in_w: Tensor = None,
    g_in_w: Tensor = None,
    norm_out_w: Tensor = None,
    norm_out_b: Optional[Tensor] = None,
    p_out_w: Tensor = None,
    g_out_w: Tensor = None,
    p_in_b: Optional[Tensor] = None,
    g_in_b: Optional[Tensor] = None,
    p_out_b: Optional[Tensor] = None,
    g_out_b: Optional[Tensor] = None,
    mask: Optional[Tensor] = None,
    direction: str = "outgoing",
    eps: float = 1e-5,
    out: Optional[Tensor] = None,
    select: str = "heuristic",
    _config: Optional[AutotuneConfig] = None,
) -> Tensor:
    """The fused TriMul: pick a combo for this shape and run it.

    Purpose
        The front door. It validates, resolves a combo, and dispatches. Every kernel in the chain is
        this package's; nothing about the choice is visible in the answer.

    Semantics
        **The token extent is refused before a combo is chosen, and that ordering is the point.**
        Op 7 contracts ``(N, N)`` operands whose row pitch must be 16-byte aligned, so ``N % 8 !=
        0`` is a shape NO combo can run -- and handing back one that will then refuse the input is
        the defect family this module documents. It raises here instead, naming the constraint.

        Selection: ``"heuristic"`` (the default) resolves the combo from the shape by a pure formula
        with no timing; ``"autotune"`` sweeps the valid candidates and caches the winner per shape;
        an explicit `_config` overrides both and is what `trimul_freeze` returns.

    Args:
        x: ``(B, N, N, D)`` RAW input -- do NOT pre-normalize, the front does op 1. ``B >= 1``.
        norm_in_w: ``(D,)`` fp32 front LayerNorm gain.
        norm_in_b: ``(D,)`` fp32 front LayerNorm bias, or None.
        p_in_w, g_in_w: ``(2D, D)`` up and gate projections -- the stacks ``[p_a; p_b]`` and
            ``[g_a; g_b]``, whose halves op 6 splits.
        norm_out_w: ``(D,)`` fp32 back LayerNorm gain.
        norm_out_b: ``(D,)`` fp32 back LayerNorm bias, or None.
        p_out_w, g_out_w: ``(D, D)`` output up and gate projections.
        p_in_b, g_in_b: ``(2D,)`` biases, or None.
        p_out_b, g_out_b: ``(D,)`` biases, or None.
        mask: ``(B, N, N)`` per-token mask, or None. Applied to the DUAL only, never to the output
            gate -- op 5's semantics.
        direction: ``"outgoing"`` or ``"incoming"``.
        eps: The LayerNorm variance floor, shared by both LayerNorms.
        out: Optional ``(B, N, N, D)`` destination, copied into. Allocated when None.
        select: ``"heuristic"``, ``"autotune"`` or ``"default"``.
        _config: A frozen `AutotuneConfig` forcing one combo. Private: it bypasses selection, not
            validation.

    Returns:
        ``(B, N, N, D)`` in `x`'s dtype.

    Raises:
        ValueError: On a non-4-D or non-square `x`; on a token extent that breaks the 16-byte floor;
            on a feature width that breaks it; on a weight of the wrong shape; on a non-fp32
            LayerNorm gain; or on an unrecognised `direction` or `select`. Every one is a ``raise``
            rather than an ``assert`` because ``python -O`` strips asserts, and a stripped check
            here is a kernel-level refusal or a silently wrong answer rather than a sentence naming
            the argument.
        NotImplementedError: From a combo whose kernels have not landed in this tree yet.
    """
    if direction not in ("outgoing", "incoming"):
        raise ValueError(f"direction must be 'outgoing' or 'incoming'; got {direction!r}")
    if select not in ("heuristic", "autotune", "default"):
        raise ValueError(f"select must be 'heuristic', 'autotune' or 'default'; got {select!r}")
    if x.dim() != 4:
        raise ValueError(f"x must be (B, N, N, D); got {tuple(x.shape)}")
    B, N, N2, D = x.shape
    if N != N2:
        raise ValueError(f"x must be (B, N, N, D) with a square token axis; got {N} x {N2}")
    if not _gemm1_validity(N):
        raise ValueError(
            f"N must be 16-byte aligned (N % 8 == 0 for a 16-bit activation); got N={N}. The "
            f"token-pair einsum contracts (N, N) operands, so N is their row pitch, and NO combo "
            f"can run a shape that breaks it."
        )
    if not _front_validity(D):
        raise ValueError(
            f"D must be 16-byte aligned (D % 8 == 0 for a 16-bit activation); got D={D}."
        )
    if norm_in_w.dim() != 1 or norm_in_w.dtype != torch.float32:
        raise ValueError(f"norm_in_w must be a ({D},) float32 tensor; got {tuple(norm_in_w.shape)}")
    if norm_out_w.dim() != 1 or norm_out_w.dtype != torch.float32:
        raise ValueError(
            f"norm_out_w must be a ({D},) float32 tensor; got {tuple(norm_out_w.shape)}"
        )
    if tuple(p_in_w.shape) != (2 * D, D) or tuple(g_in_w.shape) != (2 * D, D):
        raise ValueError(
            f"p_in_w and g_in_w must both be (2D, D) = ({2 * D}, {D}); got "
            f"{tuple(p_in_w.shape)} and {tuple(g_in_w.shape)}"
        )
    if tuple(p_out_w.shape) != (D, D) or tuple(g_out_w.shape) != (D, D):
        raise ValueError(
            f"p_out_w and g_out_w must both be (D, D) = ({D}, {D}); got "
            f"{tuple(p_out_w.shape)} and {tuple(g_out_w.shape)}"
        )
    if mask is not None and tuple(mask.shape[-2:]) != (N, N):
        raise ValueError(f"mask must be (..., N, N) = (..., {N}, {N}); got {tuple(mask.shape)}")

    has_bias = any(t is not None for t in (norm_in_b, g_in_b, p_in_b, norm_out_b, g_out_b, p_out_b))
    has_mask = mask is not None
    if _config is None and select == "heuristic":
        _config = _as_autotune_config(
            _trimul_heuristic_config(
                N,
                D,
                direction,
                has_mask,
                B=B,
                device=x.device,
            )
        )
    impl_args = (
        x,
        norm_in_w,
        norm_in_b,
        p_in_w,
        g_in_w,
        norm_out_w,
        norm_out_b,
        p_out_w,
        g_out_w,
        p_in_b,
        g_in_b,
        p_out_b,
        g_out_b,
        None if mask is None else mask.reshape(-1),
        direction,
        eps,
        has_bias,
        has_mask,
    )
    if _config is not None:
        res = _trimul_impl.__wrapped__(*impl_args, **_config.all_kwargs())
    else:
        res = _trimul_impl(*impl_args)
    if out is not None:
        out.copy_(res)
        return out
    return res


def trimul_freeze(x, norm_in_w, **kwargs):
    """Resolve the winning combo for THIS shape once and bind a callable to it.

    Purpose
        Removes the per-call selection from a steady-state loop. The sweep runs once here; every
        later call dispatches straight to the winner.

    Args:
        x: A REPRESENTATIVE input. The sweep is keyed on its shape and on which optional arguments
            are present, so freezing on a shape the loop will not run returns a combo tuned for
            something else.
        norm_in_w: The front LayerNorm gain.
        **kwargs: Everything else `trimul_autotuned` takes except ``select``, ``out`` and
            ``_config``, forwarded to both the warming call and the frozen one.

    Returns:
        A callable taking ``(x, ..., out=None)`` and dispatching to the frozen combo, with the pick
        on its ``.config`` attribute.
    """
    trimul_autotuned(x, norm_in_w, select="autotune", **kwargs)
    frozen = _trimul_impl.autotuner.best_config

    def frozen_call(x, norm_in_w=norm_in_w, out=None, **call_kwargs):
        """Run the frozen combo.

        Args:
            x: The input.
            norm_in_w: The front LayerNorm gain.
            out: Optional destination.
            **call_kwargs: Overrides for anything captured at freeze time.

        Returns:
            ``(B, N, N, D)``.
        """
        return trimul_autotuned(x, norm_in_w, out=out, _config=frozen, **{**kwargs, **call_kwargs})

    frozen_call.config = frozen
    return frozen_call


# The memory-frugal companion to `trimul_ref`, brought back UNCHANGED from the upstream apart from
# this note. It lives HERE, beside `trimul_ref`, rather than under `benchmark/` where the upstream
# kept it: an oracle that only the benchmark tree can import is an oracle a `pip install`ed user
# cannot check against, and the two functions must not be able to drift apart -- their whole
# contract is that they agree to fp32 round-off.

def trimul_ref_chunked(
    x: torch.Tensor,  # (B, N, N, D)
    direction: str,  # "outgoing" | "incoming"
    mask: Optional[torch.Tensor],  # (B, N, N) or None
    norm_in_w: torch.Tensor,
    norm_in_b: torch.Tensor,
    p_in_w: torch.Tensor,
    g_in_w: torch.Tensor,
    norm_out_w: torch.Tensor,
    norm_out_b: torch.Tensor,
    p_out_w: torch.Tensor,
    g_out_w: torch.Tensor,
    p_in_b: Optional[torch.Tensor] = None,
    g_in_b: Optional[torch.Tensor] = None,
    p_out_b: Optional[torch.Tensor] = None,
    g_out_b: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
    row_tile: int = 512,
    on_tile=None,
) -> Optional[torch.Tensor]:
    """Memory-frugal fp32 TriMul oracle — RESULT-identical to :func:`trimul_ref` (to fp32 round-off)
    but tiles the (i, j) OUTPUT grid so peak transient memory is O(row_tile * N * D) instead of the
    O(N^2 * D) (and O(N^2 * 2D) projection) full-assembly `trimul_ref` builds. For validating LARGE N
    (e.g. N4096) where the full-assembly OOMs a single GPU alongside x_global + the nvshmem symmetric heap.

    The k-CONTRACTION is kept WHOLE (only the i, j output indices are tiled) so the reduction order —
    hence the fp32 result — matches `trimul_ref`; a self-test (tests/test_trimul_ref_chunked.py) asserts
    rel_L2 < 1e-5 vs `trimul_ref` across directions / bias / mask / tile-not-dividing-N. row_tile need not
    divide N (a partial last tile is handled).

    ``on_tile``: optional ``callable(i0, i1, j0, j1, tile_fp32)`` invoked once per (i, j) output tile with
    the fp32 oracle tile ``(B, i1-i0, j1-j0, D)``. When given, the O(N^2*D) fp32 output is **NOT** allocated
    and ``None`` is returned — for STREAMING a reduction/comparison over the oracle without ever
    materializing it (e.g. ``correctness_harness.compute_error_histogram_streamed`` folds the got-vs-oracle
    error per tile, so validating N4096-D384 needs ~O(got + row_err) not +25.8GB). When ``None`` (default),
    behavior is unchanged: the full (B,N,N,D) fp32 oracle is assembled and returned."""
    # Bound locally, exactly as  binds it: the two oracles must read the same,
    # and a module-level alias would be a second name for something used in two functions.
    f = torch.nn.functional
    assert direction in ("outgoing", "incoming"), direction
    B, N, N2, D = x.shape
    assert N2 == N, f"x must be (B,N,N,D); got {tuple(x.shape)}"
    dev = x.device
    T = int(row_tile) if row_tile and int(row_tile) > 0 else N

    niw, nib = norm_in_w.float(), norm_in_b.float()
    now, nob = norm_out_w.float(), norm_out_b.float()
    piw, giw = p_in_w.float(), g_in_w.float()
    pow_, gow = p_out_w.float(), g_out_w.float()
    pib0 = 0.0 if p_in_b is None else p_in_b.float()
    gib0 = 0.0 if g_in_b is None else g_in_b.float()
    pob0 = 0.0 if p_out_b is None else p_out_b.float()
    gob0 = 0.0 if g_out_b is None else g_out_b.float()

    def _proj_half(x_slice, mask_slice, half):
        # x_slice: (B, r, s, D) -> the requested feature-half of ab = proj(LN(x_slice)); full-k intact.
        xn = f.layer_norm(x_slice.float(), (D,), niw, nib, eps)
        ab = (xn @ piw.T + pib0) * torch.sigmoid(xn @ giw.T + gib0)  # (B, r, s, 2D)
        if mask_slice is not None:
            ab = ab * mask_slice.float().unsqueeze(-1)
        return ab[..., :D] if half == 0 else ab[..., D:]

    out = None if on_tile is not None else torch.empty((B, N, N, D), dtype=torch.float32, device=dev)
    for i0 in range(0, N, T):
        i1 = min(i0 + T, N)
        # a-operand tile (proj first-half), full-k. outgoing: a[i,k] -> ROW slice; incoming: a[k,i] -> COL slice.
        if direction == "outgoing":
            a_i = _proj_half(x[:, i0:i1, :, :], None if mask is None else mask[:, i0:i1, :], 0)  # (B,Ti,N,D)
        else:
            a_i = _proj_half(x[:, :, i0:i1, :], None if mask is None else mask[:, :, i0:i1], 0)  # (B,N,Ti,D)
        for j0 in range(0, N, T):
            j1 = min(j0 + T, N)
            if direction == "outgoing":
                b_j = _proj_half(x[:, j0:j1, :, :], None if mask is None else mask[:, j0:j1, :], 1)  # (B,Tj,N,D)
                tri = torch.einsum("bikd,bjkd->bijd", a_i, b_j)  # (B, Ti, Tj, D)
            else:
                b_j = _proj_half(x[:, :, j0:j1, :], None if mask is None else mask[:, :, j0:j1], 1)  # (B,N,Tj,D)
                tri = torch.einsum("bkid,bkjd->bijd", a_i, b_j)  # (B, Ti, Tj, D)
            trin = f.layer_norm(tri, (D,), now, nob, eps)
            p_out = trin @ pow_.T + pob0
            # out-gate consumes xn at the OUTPUT (i,j) position (matches trimul_ref / cuEq): LN(x[i-tile,j-tile]).
            xn_ij = f.layer_norm(x[:, i0:i1, j0:j1, :].float(), (D,), niw, nib, eps)  # (B, Ti, Tj, D)
            g_out = xn_ij @ gow.T + gob0
            tile = p_out * torch.sigmoid(g_out)  # (B, Ti, Tj, D) fp32
            if on_tile is not None:
                on_tile(i0, i1, j0, j1, tile)
            else:
                out[:, i0:i1, j0:j1, :] = tile
    return out  # fp32 (None when on_tile streams the tiles instead of assembling the full oracle)

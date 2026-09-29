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
# Shared utilities for TVM-FFI GEMM compilation.

from functools import partial


import cutlass.cute as cute
from cutlass import Int32, Float32
from cutlass.cute.runtime import make_ptr

from fold_cp_ops._internal.artifact_cache import compile_persisted
from fold_cp_ops._internal.compile_time.compile_utils import make_fake_tensor as fake_tensor
from fold_cp_ops._internal.compile_time.cute_dsl_utils import torch2cute_dtype_map
from fold_cp_ops._internal.tile_scheduler import TileSchedulerOptions


def div_for_dtype(dtype):
    """16-byte alignment: divisibility in elements = 128 // dtype_width_bits."""
    return 128 // dtype.width


def perm3d_single(t):
    """Move one tensor's batch axis from front to back: ``(L, X, Y) -> (X, Y, L)``.

    The kernel addresses its operands batch-last, while torch callers hold them batch-first. This
    is the one place that convention is converted.

    Args:
        t: A tensor or None. Anything that is not exactly 3-D is returned unchanged -- including
           None -- so an optional operand needs no separate guard at the call site.

    Returns:
        A permuted **view** (no copy, so the strides become non-contiguous), or ``t`` itself.
    """
    return t.permute(1, 2, 0) if t is not None and t.ndim == 3 else t


def perm3d(A, B, D, C):
    """Move the batch axis of all four GEMM operands from front to back.

    Args:
        A: The A operand, ``(l, m, k)``.
        B: The B operand, ``(l, n, k)``.
        D: The output, ``(l, m, n)``.
        C: The optional addend, ``(l, m, n)`` or None.

    Returns:
        The four permuted views in the same order. Each is a view, not a copy: a caller that then
        writes through the ORIGINAL tensor is writing through the same storage.
    """
    return tuple(perm3d_single(t) for t in (A, B, D, C))


def get_major(t, dim0, dim1):
    """Name which of a tensor's two matrix axes is contiguous.

    Args:
        t: The **already-permuted** tensor, so its matrix axes are dims 0 and 1. Passing an
            unpermuted one reads the batch stride and names the wrong major.
        dim0: The name to return when dim 0 is contiguous, e.g. ``"m"``.
        dim1: The name to return when dim 1 is contiguous, e.g. ``"k"``.

    Returns:
        ``dim1`` if ``t.stride(1) == 1``, otherwise ``dim0``. Note the asymmetry: dim 1 is *tested*
        and dim 0 is the fallback, so a tensor contiguous in neither is reported as ``dim0``-major.
        ``check_tma_alignment`` rejects that case first, which is why it is safe here.
    """
    return dim1 if t.stride(1) == 1 else dim0


def get_majors(A_p, B_p, D_p, C_p):
    """Name the contiguous axis of each of the four operands.

    Args:
        A_p: Permuted A. Reported as ``"m"`` or ``"k"``.
        B_p: Permuted B. Reported as ``"n"`` or ``"k"``.
        D_p: Permuted D. Reported as ``"m"`` or ``"n"``.
        C_p: Permuted C, or None.

    Returns:
        ``(a_major, b_major, d_major, c_major)``, the last being None when C is absent. These are
        part of the compile cache key, so each distinct combination is a distinct kernel.
    """
    a_major = get_major(A_p, "m", "k")
    b_major = get_major(B_p, "n", "k")
    d_major = get_major(D_p, "m", "n")
    c_major = get_major(C_p, "m", "n") if C_p is not None else None
    return a_major, b_major, d_major, c_major


def _check_and_major(name, t, dim0, dim1, operand=False):
    """Validate one operand's strides and name its contiguous axis, in a SINGLE pass.

    The fused form of ``check_tma_alignment`` + ``get_major``. Both need the same stride tuple and
    the same dtype width, and both run on every call of the host entry, so doing them separately
    fetched the strides twice and cost ~0.8 us per launch -- about 6% of a launch-bound GEMM.
    Fusing them also removes an ordering hazard the two-function form had to document: ``get_major``
    reports ``dim0`` for a tensor contiguous in neither axis, which is only safe because the check
    ran first. Here there is no "first".

    The success path allocates nothing: one ``.stride()``, one dict lookup, and a loop over 2-3
    ints. Every message is built inside a ``_raise_*`` helper, off this path.

    Args:
        name: The operand's name, used only in a message.
        t: The **already-permuted** tensor, or None for an absent operand.
        dim0: The major name when dim 0 is contiguous, e.g. ``"m"``.
        dim1: The major name when dim 1 is contiguous, e.g. ``"k"``.
        operand: Whether this tensor is an MMA operand (A or B) rather than an output. Operands
            additionally have to be 16- or 8-bit; D and C may be fp32.

    Returns:
        ``(major_name, cute_dtype)``, or ``(None, None)`` when ``t`` is None. The dtype is returned
        rather than looked up again by the caller: this is the only place the map is indexed for
        this tensor, and it used to be indexed three more times per call downstream.

    Raises:
        ValueError: If neither matrix axis is contiguous, a non-unit stride breaks the 16-byte
            floor, the base address is not 16-byte aligned, or an operand is not 16- or 8-bit.
    """
    if t is None:
        return None, None
    cute_dtype = torch2cute_dtype_map[t.dtype]
    # Width BEFORE layout: an fp32 A is not an operand at all, so complaining about its strides
    # first would answer a question the caller is not asking. This ordering is what the declared
    # unsupported regions in tests/kernels/test_gemm.py are sequenced against.
    if operand and cute_dtype.width not in (8, 16):
        _raise_bad_operand_width(name, t.dtype)
    strides = t.stride()
    div = 128 // cute_dtype.width
    s1 = strides[1]
    if strides[0] != 1 and s1 != 1:
        _raise_not_contiguous(name, strides)
    for s in strides:
        if s != 1 and s % div:
            _raise_misaligned(name, t, strides, div)
    if t.data_ptr() % 16:
        _raise_misaligned_base(name, t)
    return (dim1 if s1 == 1 else dim0), cute_dtype


def describe_operands(A_p, B_p, D_p, C_p):
    """Name every operand's contiguous axis, map its dtype, and validate it -- ONE pass each.

    What ``kernels/gemm.gemm`` calls, and the reason it is one function: the majors, the cutlass
    dtypes and the alignment checks all need the same two facts per tensor (its strides and its
    element width), and asking for them separately indexed ``torch2cute_dtype_map`` **thirteen**
    times per launch. That is invisible on a device-bound shape and roughly 10% of a launch-bound
    one, which is where the paired perf gate caught it.

    ``get_majors`` / ``get_dtypes`` / ``check_tma_alignment`` remain for callers that want one
    without the others.

    Args:
        A_p: Permuted A. Checked as an MMA operand, so it must be 16- or 8-bit.
        B_p: Permuted B, likewise.
        D_p: Permuted D. May be fp32.
        C_p: Permuted C, or None when there is no addend.

    Returns:
        ``((a_major, b_major, d_major, c_major), (a_dtype, b_dtype, d_dtype, c_dtype))``, with None
        in both tuples for an absent operand.

    Raises:
        ValueError: From the first operand that fails, naming it.
    """
    a = _check_and_major("A", A_p, "m", "k", operand=True)
    b = _check_and_major("B", B_p, "n", "k", operand=True)
    d = _check_and_major("D", D_p, "m", "n")
    c = _check_and_major("C", C_p, "m", "n")
    return (a[0], b[0], d[0], c[0]), (a[1], b[1], d[1], c[1])


def _raise_bad_operand_width(name, torch_dtype):
    """Refuse an MMA operand that is neither 16- nor 8-bit. Off the hot path.

    Args:
        name: ``"A"`` or ``"B"``.
        torch_dtype: The offending dtype, named in the message.

    Returns:
        Never returns.

    Raises:
        ValueError: Always.
    """
    raise ValueError(
        f"unsupported operand dtype {torch_dtype} for {name}: the SM90 WGMMA atom takes 16-bit "
        f"(float16/bfloat16) or 8-bit (float8_e4m3fn/float8_e5m2) operands only. float32 A/B has "
        f"no Hopper MMA form -- cast A and B, or use a torch matmul. D may still be fp32."
    )


def _raise_not_contiguous(name, strides):
    """Build and raise the "no unit stride" message. Off the hot path, so it may allocate.

    Args:
        name: The operand's name.
        strides: The offending stride tuple.

    Returns:
        Never returns.

    Raises:
        ValueError: Always.
    """
    raise ValueError(
        f"{name} must be contiguous along one of its two matrix axes; got strides {strides} "
        f"(after the batch axis is moved last). A TMA descriptor addresses a strided box, so "
        f"one axis has to have unit stride. Call .contiguous() on {name}, or transpose it so "
        f"the intended axis is innermost."
    )


def _raise_misaligned(name, t, strides, div):
    """Build and raise the 16-byte-floor message, naming every offending stride.

    Args:
        name: The operand's name.
        t: The tensor, for its dtype in the message.
        strides: The full stride tuple; every violating entry is listed, not just the first one the
            caller's loop happened to hit.
        div: The required multiple, in elements.

    Returns:
        Never returns.

    Raises:
        ValueError: Always.
    """
    width = torch2cute_dtype_map[t.dtype].width
    bad = [(i, s) for i, s in enumerate(strides) if s != 1 and s % div]
    raise ValueError(
        f"{name} violates the 16-byte alignment floor: stride(s) {bad} are not multiples of "
        f"{div} elements ({t.dtype} is {width} bits, so {div} elements = 16 bytes). This is the "
        f"only shape constraint this GEMM has -- pad the offending extent to a multiple of "
        f"{div}, or make {name} contiguous along the other axis."
    )


def _raise_misaligned_base(name, t):
    """Build and raise the base-address message. Off the hot path, so it may allocate.

    Args:
        name: The operand's name.
        t: The tensor, for the offending address.

    Returns:
        Never returns.

    Raises:
        ValueError: Always.
    """
    raise ValueError(
        f"{name} does not start on a 16-byte boundary (data_ptr() = {t.data_ptr()}, "
        f"{t.data_ptr() % 16} bytes past one). A TMA descriptor's global address must be 16-byte "
        f"aligned. This is almost always a SLICE with a non-multiple offset -- {name}[..., 1:] on "
        f"a 16-bit tensor starts 2 bytes in. Slice on a multiple of {16 * 8 // torch2cute_dtype_map[t.dtype].width} "
        f"elements, or .contiguous() the view."
    )


def check_tma_alignment(name, t) -> None:
    """Refuse an operand whose strides the TMA descriptor cannot express, naming the constraint.

    **The floor is on the PITCH and the base address, never on an EXTENT.** That distinction is the
    whole content of this function, and a contiguous tensor hides it by making the two the same
    number. Concretely, three requirements: one of the two *matrix* axes must be contiguous; every
    other stride must be a multiple of ``128 // dtype.width`` elements (8 at bf16/fp16, 16 at fp8);
    and the base address must be 16-byte aligned. That is what ``make_fake_tensor`` declares via
    ``sym_int64(divisibility=...)`` and ``assumed_align``, so a tensor breaking any of them is
    rejected by the TVM-FFI ABI check as ``Invalid mD.strides[0] on argument #2`` or ``Misaligned
    Tensor data``, several layers down and naming neither the requirement nor the fix.

    **Every extent is free.** MEASURED at bf16, not inferred: N = 1, 3, 7, 201 and 257 and K = 1, 3,
    7, 129 all compute correctly *when the row pitch is padded up to the floor* -- ``torch.empty(l,
    m, 208)[:, :, :201]`` works while ``torch.empty(l, m, 201)`` does not, and the difference
    between them is the pitch, not the 201. Same at fp8 with a 16-element pitch. M and the batch
    extent are unconstrained outright, except that an m-major D makes M the pitch, at which point M
    picks the floor up for that reason and not as a shape constraint.

    Args:
        name: The operand's name (``"A"``, ``"B"``, ``"D"``, ``"C"``), used only in the message.
        t: The **already-permuted** torch tensor, i.e. what ``perm3d`` returned -- the strides
            checked here must be the ones the fake tensor is built from. The matrix axes are then
            dims 0 and 1 and the batch, if any, is last, which is why the contiguity test looks at
            ``strides[:2]`` and not at the trailing pair. None is accepted and ignored, so an absent
            C needs no guard at the call site.

    Returns:
        None.

    Raises:
        ValueError: If neither matrix axis is contiguous, or if any non-unit stride is not a
            multiple of ``128 // dtype.width``.
    """
    if t is None:
        return
    # This runs on EVERY call, so the success path allocates nothing and builds no message: one
    # dict lookup, one `.stride()` (already a tuple), and a loop over 2-3 ints with an early exit.
    # The list comprehension and f-strings this used to evaluate unconditionally cost 1.36 us/call,
    # which is 10% of a small GEMM's total -- a real regression at the shapes the TriMul front and
    # back run at their smallest.  Message construction belongs in `_raise_*`, off this path.
    div = 128 // torch2cute_dtype_map[t.dtype].width
    strides = t.stride()
    if strides[0] != 1 and strides[1] != 1:
        _raise_not_contiguous(name, strides)
    for s in strides:
        if s != 1 and s % div:
            _raise_misaligned(name, t, strides, div)
    if t.data_ptr() % 16:
        _raise_misaligned_base(name, t)


def check_broadcast_alignment(name, t) -> None:
    """Refuse an epilogue broadcast vector whose row pitch is not 4-element aligned.

    The row and column bias tensors are described to the DSL by ``make_fake_tensor(..., leading_dim
    =1, divisibility=4)`` -- a literal 4, not ``128 // width``, because these are loaded with a
    narrower vectorized copy than the matrix operands. The consequence is easy to miss: a **column**
    bias of shape ``(l, m)`` makes M its row pitch, so passing one imposes a 4-element alignment on
    M, an extent that is otherwise entirely free. A caller who works at M = 301 and then adds a
    column bias gets ``Invalid epilogue_args[3].strides[0] ... expected to be divisible by 4``,
    which names an argument index rather than the extent to pad.

    Args:
        name: The argument's name (``"rowvec_bias"`` / ``"colvec_bias"``), used in the message.
        t: The bias tensor, or None (accepted and ignored). A 1-D column bias has no
            non-unit stride and therefore always passes.

    Returns:
        None.

    Raises:
        ValueError: If any non-unit stride is not a multiple of 4 elements.
    """
    if t is None:
        return
    strides = t.stride()
    for s in strides:
        if s != 1 and s % 4:
            bad = [(i, v) for i, v in enumerate(strides) if v != 1 and v % 4]
            raise ValueError(
                f"{name} has stride(s) {bad} that are not multiples of 4 elements. The epilogue "
                f"broadcast load is 4-element vectorized, so the vector's row pitch -- which is M "
                f"for a (l, m) colvec_bias and N for a (l, n) rowvec_bias -- must be a multiple of "
                f"4. Pad that extent, or pass the bias as a flat 1-D tensor."
            )


def get_dtypes(A, B, D, C):
    """Map the four operands' torch dtypes to cutlass numeric types.

    Args:
        A: The A operand. Its dtype must be a key of ``torch2cute_dtype_map``; callers are expected
            to have refused anything else by name already, since a miss here is a bare ``KeyError``.
        B: The B operand.
        D: The output.
        C: The optional addend, or None.

    Returns:
        ``(a_dtype, b_dtype, d_dtype, c_dtype)`` as cutlass types, the last None when C is absent.

    Raises:
        KeyError: If any dtype is not in the map -- which is why ``gemm()`` checks membership first
            and raises a ``ValueError`` naming the operand.
    """
    a_dtype = torch2cute_dtype_map[A.dtype]
    b_dtype = torch2cute_dtype_map[B.dtype]
    d_dtype = torch2cute_dtype_map[D.dtype]
    c_dtype = torch2cute_dtype_map[C.dtype] if C is not None else None
    return a_dtype, b_dtype, d_dtype, c_dtype


def make_scheduler_args(
    max_active_clusters,
    max_swizzle_size,
    tile_count_semaphore,
    batch_idx_permute=None,
    run_j_tiles=0,
):
    """Build the LAUNCH-time tile-scheduler arguments.

    Args:
        max_active_clusters: The persistent grid size in clusters, from ``get_max_active_clusters``.
            0 for a non-persistent launch.
        max_swizzle_size: Rasterization swizzle width, for L2 locality. A performance knob only.
        tile_count_semaphore: The dynamic scheduler's GMEM counter tensor, or None. Must be
            **zeroed before every launch** -- it is an atomic ticket dispenser, and a stale value
            makes the grid skip tiles, silently truncating the output.
        batch_idx_permute: Optional ``(l,)`` int32 permutation of the batch visit order. Scheduling
            only; never changes the result.
        run_j_tiles: Accepted for signature parity with ``make_fake_scheduler_args`` and then
            **deliberately discarded** -- see the note below.

    Returns:
        A ``TileSchedulerOptions`` for the compiled entry.

    Note:
        Several fields are passed as None on purpose. They are ``Constexpr`` on the NamedTuple, so
        their values were baked in at compile time from the *fake* arguments; the FFI prototype has
        those slots ``ConstNone``-erased, and passing a real value would be an ABI mismatch rather
        than an override. That is why ``run_j_tiles`` is ignored here but not there.
    """
    # run_j_tiles is a Constexpr field: its value is BAKED at compile time (via the fake scheduler
    # args -> the const_expr read in get_scheduler_arguments). At call time it must be passed as
    # None so the FFI ABI matches the ConstNone-erased prototype (same contract as raster_order).
    return TileSchedulerOptions(
        max_active_clusters=Int32(max_active_clusters),
        raster_order=None,
        max_swizzle_size=max_swizzle_size,
        tile_count_semaphore=(
            tile_count_semaphore.data_ptr() if tile_count_semaphore is not None else None
        ),
        batch_idx_permute=batch_idx_permute,
        run_j_tiles=None,
        run_j_dynamic=None,
        run_i_tiles=None,  # Constexpr baked at compile (fake args) -> pass None at call (ABI-erased),
        # same contract as run_j_tiles/run_j_dynamic. Omitting it defaulted to 0 (non-None) -> a scheduler_
        # args[7] ABI mismatch for any kernel whose compiled prototype ConstNone-erases run_i_tiles.
    )


def make_fake_scheduler_args(has_semaphore, has_batch_idx_permute, l_sym, run_j_tiles=0):
    """Build the COMPILE-time tile-scheduler arguments, as fake values.

    The counterpart of ``make_scheduler_args``: same NamedTuple, but every runtime field is a
    stand-in whose only job is to declare the argument's shape and type to the tracer, and every
    ``Constexpr`` field carries its real value, which is baked in here and erased from the ABI.

    Args:
        has_semaphore: Whether the compiled kernel should expect a dynamic-scheduler counter. This
            is a compile-time decision, so it must match what the launch will pass.
        has_batch_idx_permute: Whether to expect a batch permutation tensor.
        l_sym: The symbolic batch extent, shared with the operand fake tensors so the compiled
            kernel relates them.
        run_j_tiles: The value to BAKE IN. Unlike at launch, it matters here.

    Returns:
        A ``TileSchedulerOptions`` suitable for ``cute.compile``.
    """
    return TileSchedulerOptions(
        max_active_clusters=Int32(1),
        max_swizzle_size=Int32(8),
        tile_count_semaphore=(
            make_ptr(Int32, 0, cute.AddressSpace.gmem, assumed_align=4) if has_semaphore else None
        ),
        batch_idx_permute=(
            fake_tensor(Int32, (l_sym,), leading_dim=0, divisibility=4)
            if has_batch_idx_permute
            else None
        ),
        run_j_tiles=run_j_tiles,
    )


def make_fake_gemm_tensors(
    a_dtype,
    b_dtype,
    d_dtype,
    c_dtype,
    a_major,
    b_major,
    d_major,
    c_major,
):
    """Build the compile-time stand-ins for the four GEMM operands, sharing symbolic extents.

    The extents are ``cute.sym_int()`` symbols, not literals, which is what lets ONE compiled
    artifact serve every M/N/K/L: shapes never enter the cache key. The four tensors share the
    symbols (A and D share ``m``; A and B share ``k``), so the traced kernel relates their extents
    the way the real call will.

    Args:
        a_dtype: Cutlass element type of A.
        b_dtype: Element type of B.
        d_dtype: Element type of D, or None to get None back for it.
        c_dtype: Element type of C, or None when there is no addend.
        a_major: ``"m"`` or ``"k"`` -- which axis is contiguous. Must match the REAL tensor's
            major, since it sets the declared leading dim and hence the TMA descriptor; a mismatch
            compiles a kernel that reads the operand transposed.
        b_major: ``"n"`` or ``"k"``.
        d_major: ``"m"`` or ``"n"``.
        c_major: ``"m"``, ``"n"``, or None when C is absent.

    Returns:
        ``(mA, mB, mD, mC, m, n, k, l)`` -- the four fake tensors (``mD``/``mC`` are None when
        their dtype was) followed by the four shared symbols, which the caller reuses to declare
        epilogue broadcasts against the same extents.
    """
    a_leading = 1 if a_major == "k" else 0
    b_leading = 1 if b_major == "k" else 0
    d_leading = 1 if d_major == "n" else 0
    c_leading = 1 if c_major == "n" else 0
    m, n, k, l = cute.sym_int(), cute.sym_int(), cute.sym_int(), cute.sym_int()
    div_a = div_for_dtype(a_dtype)
    div_b = div_for_dtype(b_dtype)
    div_d = div_for_dtype(d_dtype) if d_dtype is not None else 1
    div_c = div_for_dtype(c_dtype) if c_dtype is not None else 1
    mA = fake_tensor(a_dtype, (m, k, l), leading_dim=a_leading, divisibility=div_a)
    mB = fake_tensor(b_dtype, (n, k, l), leading_dim=b_leading, divisibility=div_b)
    mD = fake_tensor(d_dtype, (m, n, l), leading_dim=d_leading, divisibility=div_d)
    mC = fake_tensor(c_dtype, (m, n, l), leading_dim=c_leading, divisibility=div_c)
    return mA, mB, mD, mC, m, n, k, l


def compile_gemm_kernel(
    GemmCls,
    a_dtype,
    tile_shape_mn,
    cluster_shape_mnk,
    pingpong,
    persistent,
    is_dynamic_persistent,
    device_capacity,
    mA,
    mB,
    mD,
    mC,
    epi_args,
    scheduler_args,
    post_init=None,
    mSFA=None,
    mSFB=None,
    mB2=None,
    mB3=None,
    persist=None,
):
    """Build GemmCls instance, apply SM90 partial, and cute.compile with TVM-FFI.

    mB2 (SM90 only): optional second B tensor for the two-tensor block-interleaved gated load
    (strategy-1).  When None (default) the trace/signature is byte-identical to before.
    mB3 (SM90 only): optional third B tensor for the fused TriMul output-gate (gate3); appended
    after mB2.  When provided, mB2 is passed too (None if absent) so the positional layout
    (..., mB2, mB3) matches the dual_gated_gemm_staged kernel signature.

    persist (optional `artifact_cache.PersistSpec`): turn on cross-process artifact reuse for THIS
    compile.  This is the SAME parameter `compile_nvshmem` takes, so every kernel in the repo --
    single-device and A2A alike -- is persisted by supplying one argument; only the backend differs,
    and the store picks that itself from the compile options.  Omit it and the path is byte-identical
    to before: `compile_persisted(persist=None)` is an exact passthrough to `cute.compile`.

    Its `key` must carry everything the compile depends on that the source fingerprint does not.
    The shapes are NOT automatically included -- these are FAKE tensors and the compile is
    dynamic-shape, so two different (M,N,K) legitimately share one artifact; but a caller that
    compiles STATIC shapes must put them in the key itself or it will be served the wrong kernel.

    This does NOT replace `@jit_cache`, which caches by `__qualname__` at a different layer and keeps
    working untouched.  The two share `_compute_source_fingerprint`, so they can never disagree about
    what makes an entry stale."""
    # No arch branch: this package is SM90-only, `gemm()` runs `require_sm90` before it gets here
    # and `_compile_gemm` re-checks with `check_arch_supported`, so `device_capacity[0]` is 9. The
    # SM100 arm that used to sit here selected cluster-launch-control persistence and was
    # unreachable from any public entry.
    GemmCls = partial(GemmCls, pingpong=pingpong, is_persistent=persistent)

    def _build():
        """Construct a FRESH functor, identically configured. The factory M4c threads to the key.

        Every argument here is captured, so two calls differ in nothing -- which is what makes the
        MLIR hash of the throwaway instance a valid key for the one that gets compiled. Measured
        free (0.00 ms) at world 2 and world 16, so calling it twice per compile costs nothing.
        """
        obj = GemmCls(Float32, a_dtype, tile_shape_mn, cluster_shape_mnk)
        if post_init:
            post_init(obj)
        return obj

    gemm_obj = _build()
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    sf_args = () if device_capacity[0] == 9 else (mSFA, mSFB)
    # mB2 is appended ONLY when provided (SM90 strategy-1); otherwise the positional signature is
    # unchanged so all existing GEMM compiles keep the same cache key / traced graph.  mB3 (gate3)
    # forces mB2 to be passed too (None if absent) to keep the (..., mB2, mB3) positional order.
    if mB3 is not None:
        b2_args = (mB2, mB3)
    else:
        b2_args = (mB2,) if mB2 is not None else ()
    return compile_persisted(
        gemm_obj,
        mA,
        mB,
        mD,
        mC,
        epi_args,
        scheduler_args,
        stream,
        *sf_args,
        *b2_args,
        options="--enable-tvm-ffi",
        persist=persist,
        op_factory=_build,
    )

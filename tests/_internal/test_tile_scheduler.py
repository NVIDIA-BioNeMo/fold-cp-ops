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


"""Tests for ``fold_cp_ops._internal.tile_scheduler`` -- work-tile distribution.

The delinearization needs a launched grid; what is checkable on the host is the classification the
scheduler is built from -- the persistence modes, the raster heuristic, and the class surface each
variant has to keep for the kernel to drive it uniformly.
"""

from fold_cp_ops._internal.tile_scheduler import (
    PersistenceMode,
    RasterOrder,
    RasterOrderOption,
    TileScheduler,
    TileSchedulerArguments,
    TileSchedulerOptions,
    TriangularTileScheduler,
)


def test_raster_option_and_resolved_order_are_different_types():
    """``Heuristic`` has no counterpart in ``RasterOrder``, which is what forces the resolution.

    If the two enums were one type, a ``Heuristic`` could reach the delinearization unresolved and
    be compared against ``AlongM``/``AlongN`` as an unequal third value -- silently rasterizing
    along neither.
    """
    assert set(RasterOrderOption.__members__) == {"AlongM", "AlongN", "Heuristic"}
    assert set(RasterOrder.__members__) == {"AlongM", "AlongN"}
    assert RasterOrderOption is not RasterOrder


def test_persistence_modes_are_the_four_the_kernel_branches_on():
    """NONE / STATIC / DYNAMIC -- and there is deliberately no fourth.

    A ``CLC`` member used to sit here for Blackwell's cluster-launch-control. It was removed with
    the branches behind it: nothing could construct it (``GemmSm90.arch`` is the constant 90, so the
    flag that selected it could only raise), which made every one of its ~15 scheduler branches
    permanently ``const_expr``-false -- a reading tax with no reachable behaviour. Asserting the
    exact member set is what stops it, or anything like it, from drifting back in unnoticed.
    """
    assert set(PersistenceMode.__members__) == {"NONE", "STATIC", "DYNAMIC"}
    assert PersistenceMode.NONE != PersistenceMode.STATIC


def test_every_scheduler_keeps_the_surface_the_kernel_drives_it_through():
    """The kernel calls the same five names on whichever scheduler it was given.

    A variant missing one would force a branch at the call site, which is exactly the coupling the
    class hierarchy exists to avoid.
    """
    for cls in (TileScheduler, TriangularTileScheduler):
        for name in (
            "Params",
            "to_underlying_arguments",
            "create",
            "get_grid_shape",
            "_delinearize_work_idx",
        ):
            assert hasattr(cls, name), f"{cls.__name__} lost {name}"


def test_the_triangular_scheduler_specializes_rather_than_replaces():
    """It inherits the dense scheduler and overrides only the index -> coordinate mapping.

    This is the property that lets the kernel drive any scheduler through one call site: a variant
    that replaced ``get_current_work`` too would need a branch at every use.
    """
    assert issubclass(TriangularTileScheduler, TileScheduler)
    assert TriangularTileScheduler._delinearize_work_idx is not TileScheduler._delinearize_work_idx
    assert TriangularTileScheduler.get_current_work is TileScheduler.get_current_work, (
        "the advance/read path is shared; only the index -> coordinate mapping differs"
    )


def test_options_default_to_a_non_persistent_dense_launch():
    """Every optional field defaults to the simplest configuration, so a caller opts in to each."""
    opts = TileSchedulerOptions(max_active_clusters=0)
    assert opts.raster_order == RasterOrderOption.Heuristic
    assert opts.tile_count_semaphore is None
    assert opts.batch_idx_permute is None


def test_arguments_carry_the_persistence_mode_the_params_are_built_for():
    """The mode is chosen by the kernel and travels here, rather than being inferred downstream."""
    assert "persistence_mode" in TileSchedulerArguments.__dataclass_fields__

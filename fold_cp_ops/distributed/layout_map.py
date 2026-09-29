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

"""Strided layout algebra for the device-mesh rank<->coordinate mapping.

Ported (numpy-only, surgically) from the upstream CP project's ``distributed/utils.py`` — the
``LayoutMap`` / ``LayoutRightMap`` / ``LayoutLeftMap`` classes only. The upstream CP project's
file additionally carried DTensor/Shard sharding helpers that these classes do
NOT depend on; those are intentionally left out (they belong to the DTensor
adapter / PE-map tasks, not the infrastructure layer).

Modeled on C++ ``std::layout_stride::mapping`` but adds the flat-index ->
multidimensional-index inverse (``unravel``). Used by ``DistributedManager`` to
build the device mesh (``LayoutRightMap``) and to slice per-axis subgroup
layouts (``LayoutMap.__getitem__``).
"""

from __future__ import annotations

import numpy as np


class LayoutMap:
    """A mapping between multidimensional indices and flat indices.

    Based on C++ ``std::layout_stride::mapping`` plus the inverse map
    (flat index -> multidimensional indices).

    Parameters
    ----------
    strides : tuple of ints
        The strides of the layout.
    shape : tuple of ints
        The shape of the layout.
    offset : int, optional
        The offset of the layout.

    Raises
    ------
    ValueError
        If the input strides or shape is invalid (non-positive, mismatched
        length, or non-unique layout).
    """

    def __init__(
        self,
        strides: tuple[int, ...],
        shape: tuple[int, ...],
        offset: int = 0,
    ):
        """Initialize the layout mapping.

        Notes
        -----
        The input strides must be a cumulative product of some permutation of
        the input shape.
        """
        if not all(isinstance(stride, (int, np.int64)) and stride > 0 for stride in strides):
            raise ValueError(f"Input strides contain non-integer or negative values: {strides}")

        self._has_negative_shape = any(s < 0 for s in shape)

        if self._has_negative_shape:
            raise ValueError(f"Input shape contain negative values: {shape}")

        self._has_zero_shape = any(s == 0 for s in shape)

        if self._has_zero_shape:
            # NOTE: the C++ standard does allow the shape to be zero along some axes, which
            # is not useful for our usage case. We nonetheless can relax the condition to
            # allow zero-sized axes but it requires more testing
            raise ValueError(f"Input shape contain zero values: {shape}")

        self._strides = strides
        self._n_axes = len(strides)

        if len(shape) != self._n_axes:
            raise ValueError(f"Shape {shape} and strides {strides} must have the same length")

        self._shape = shape
        self._numel = np.prod(self._shape)
        self._offset = offset

        # singleton axes can confound the uniqueness and exhaustiveness check, e.g.,
        # for layout right of (3, 1, 5), the strides are (5, 5, 1) but direct argsort
        # on the strides will give the permuted exhaustive stride of (1, 5, 15), which
        # corresponds to the strides of (5, 15, 1) (argsort of (2, 0, 1)), which will
        # fail the uniqueness check. This is purely artifact of the stable sorting where
        # the single axis can potentially be arbitrarily placed before or after the
        # other axes with the same stride. The correct thing to do is to handle the ties
        # involving the singleton axes so that we sort by shape if two stride elements are
        # tied.
        shape_and_strides = np.array(
            list(zip(self._shape, self._strides)),
            dtype=np.dtype([("shape", int), ("strides", int)]),
        )
        argsort_ascend_strides_and_shape = np.argsort(shape_and_strides, order=["strides", "shape"])

        self.is_unique = self._is_unique(argsort_ascend_strides_and_shape)
        self.is_exhaustive = self._is_exhaustive(argsort_ascend_strides_and_shape)

        if not self.is_unique:
            raise ValueError(
                f"Input strides {strides} and shape {shape} do not give unique layout."
            )

        self._required_span_size = self._compute_required_span_size()
        self._argsort_descend_strides = argsort_ascend_strides_and_shape[::-1]
        self._argsort_ascend_strides = argsort_ascend_strides_and_shape

    def _compute_required_span_size(self) -> int:
        """Minimal span size to represent the layout in a contiguous buffer.

        See https://eel.is/c++draft/views.multidim#mdspan.layout.stride.expo-1
        """
        if self._n_axes == 0:
            return 1
        if self._has_zero_shape:
            return 0
        return 1 + sum((self._shape[i] - 1) * self._strides[i] for i in range(self._n_axes))

    def _strides_exhaustive(self, permutation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute permuted strides and the expected exhaustive strides.

        For a valid exhaustive layout, ``strides[p[i]] == strides[p[i-1]] *
        shape[p[i-1]]`` for all ``i > 0`` and ``strides[p[0]] == 1``, where
        ``p`` is the ascending-stride permutation.
        """
        strides = np.array(self._strides)
        shape = np.array(self._shape)
        shape_permuted = shape[permutation]
        strides_permuted = strides[permutation]
        shape_shifted = np.concatenate([[1], shape_permuted[:-1]])
        strides_shifted = np.concatenate([[1], strides_permuted[:-1]])
        return strides_permuted, strides_shifted * shape_shifted

    def _is_unique(self, permutation: np.ndarray) -> bool:
        """Whether the index -> flat-index mapping is injective.

        See https://eel.is/c++draft/views.multidim#mdspan.layout.stride.cons
        """
        if self._n_axes == 0:
            return True
        strides, strides_exhaustive = self._strides_exhaustive(permutation)
        ans = np.all(strides >= strides_exhaustive)
        return ans

    def _is_exhaustive(self, permutation: np.ndarray) -> bool:
        """Whether the index -> flat-index mapping is surjective.

        See https://eel.is/c++draft/views.multidim#mdspan.layout.stride.obs-5.2
        """
        if self._n_axes == 0:
            return True
        strides, strides_exhaustive = self._strides_exhaustive(permutation)
        ans = np.all(strides == strides_exhaustive)
        return ans

    @property
    def offset(self) -> int:
        """The offset of the layout."""
        return self._offset

    @property
    def required_span_size(self) -> int:
        """The required span size of the layout."""
        return self._required_span_size

    @property
    def numel(self) -> int:
        """The total number of elements in the layout."""
        return self._numel

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the layout."""
        return self._shape

    @property
    def strides(self) -> tuple[int, ...]:
        """The strides of the layout (a cumulative product of some permutation of the shape)."""
        return self._strides

    def __call__(self, ids: tuple[int, ...]) -> int:
        """Return the flat index for a multidimensional index.

        Raises
        ------
        ValueError
            If the input index is out of range.
        """
        if len(ids) != self._n_axes:
            raise ValueError(f"Expected {self._n_axes} elements in ids but got only {len(ids)}")

        if len(ids) == 0:
            return self._offset

        if self._shape is not None:
            for axis, idx in enumerate(ids):
                if idx < 0 or idx >= self._shape[axis]:
                    raise ValueError(
                        f"Expected ids to satisfy 0 <= ids[{axis}] <= {self._shape[axis] - 1} "
                        f"but found ids[{axis}] == {idx}"
                    )
        return np.dot(ids, self._strides) + self._offset

    def ravel(self, ids: tuple[int, ...]) -> int:
        """Alias of ``__call__``: multidimensional index -> flat index."""
        return self(ids)

    def unravel(self, flat_index: int) -> tuple[int, ...]:
        """Convert a flat index to a multidimensional index.

        Raises
        ------
        TypeError
            If the input is not an integer.
        ValueError
            If the layout is not unique or the index is out of range.
        """
        if not self.is_unique:
            # double check the uniqueness of the layout
            raise ValueError(f"Layout is not unique, cannot unravel {flat_index}")

        if not isinstance(flat_index, (int, np.integer)):
            raise TypeError(f"Expected arg to be an int, but instead got type {type(flat_index)}")

        remaining = flat_index - self._offset

        if remaining < 0 or remaining >= self._required_span_size:
            raise ValueError(
                f"Expected flat_index in range [{self._offset}, {self._offset + self._required_span_size - 1}], "
                f"but instead got {flat_index}"
            )

        indices = [0] * self._n_axes  # Initialize indices

        for i_dim in self._argsort_descend_strides:
            stride = self._strides[i_dim]
            size = self._shape[i_dim]
            indices[i_dim] = (remaining // stride) % size
            remaining -= indices[i_dim] * stride

        if remaining != 0:
            msg = f"Input flat_index {flat_index} is out of the valid range of span."
            if not self.is_exhaustive:
                msg += " Given the layout is not exhaustive, the input flat_index can fall into the unmapped region."
            raise ValueError(msg)

        return tuple(indices)

    def __getitem__(self, slices: tuple[slice | int, ...]) -> "LayoutMap":
        """Create a new LayoutMap by slicing along specified dimensions.

        Integer indices collapse the corresponding dimension; slices transform
        it by ``(start, stop, step)``. Missing trailing dimensions are sliced
        full-range (``:``).

        Raises
        ------
        ValueError
            If slicing with negative/zero step, or ``start >= stop``.
        TypeError
            If a slice element is not ``slice`` or ``int``.

        Examples
        --------
        >>> layout = LayoutMap((12, 4, 1), (2, 3, 4))
        >>> sub_layout = layout[slice(1, 3, 2), :, :]
        >>> sub_layout = layout[:, 1, :]
        >>> sub_layout = layout[1:]  # Equivalent to layout[1:, :, :]
        """
        if not isinstance(slices, tuple) and (isinstance(slices, slice) or isinstance(slices, int)):
            slices = (slices,)

        # Pad slices with full-range slices if needed
        if len(slices) < self._n_axes:
            full_slice = slice(None)  # This is equivalent to ':'
            slices = slices + (full_slice,) * (self._n_axes - len(slices))

        new_shape = []
        new_strides = []
        new_offset = self.offset

        for axis, s in enumerate(slices):
            if isinstance(s, (int, np.int64)):
                # Collapse dimension and adjust offset
                new_offset += s * self.strides[axis]
            elif isinstance(s, slice):
                start, stop, step = s.indices(self.shape[axis])
                if step <= 0:
                    raise ValueError("Unsupported slicing: Negative or zero steps")
                if start >= stop:
                    # NOTE: the start == stop could be supported because we could have
                    # a layout with shape[i] == 0. But it wouldn't be useful for our usage cases.
                    raise ValueError("Unsupported slicing: start not smaller than stop")

                # Calculate new dimension length
                dim_len = (stop - start + step - 1) // step
                dim_len = max(0, dim_len)

                # Update metadata
                new_shape.append(dim_len)
                new_strides.append(self.strides[axis] * step)
                new_offset += start * self.strides[axis]
            else:
                raise TypeError(f"Unsupported slice type: {type(s)}")

        return LayoutMap(tuple(new_strides), tuple(new_shape), new_offset)


class LayoutRightMap(LayoutMap):
    """A right-aligned (row-major / C-contiguous) layout mapping.

    Parameters
    ----------
    shape : tuple of ints
        The shape of the layout.
    """

    def __init__(self, shape: tuple[int, ...]):
        strides = np.ones_like(shape)
        strides[1:] = shape[:0:-1]
        strides = np.cumprod(strides)[::-1]
        super().__init__(tuple(strides), shape=shape)


class LayoutLeftMap(LayoutMap):
    """A left-aligned (column-major / Fortran-contiguous) layout mapping.

    Parameters
    ----------
    shape : tuple of ints
        The shape of the layout.
    """

    def __init__(self, shape: tuple[int, ...]):
        strides = np.ones_like(shape)
        strides[1:] = shape[:-1]
        strides = np.cumprod(strides)
        super().__init__(tuple(strides), shape=shape)

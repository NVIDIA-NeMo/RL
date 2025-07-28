# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Sequence, Union

import numpy as np


class NamedSharding:
    """Represents an N-dimensional arrangement of ranks with named axes, facilitating data sharding, replication, and collection based on these axes.

    Example:
        layout = [
            [[0, 1, 2, 3], [4, 5, 6, 7]],
        ]
        names = ["dp", "pp", "tp"]
        # This represents DP=1, PP=2, TP=4
        sharding = NamedSharding(layout, names)
        print(sharding.shape)  # Output: (1, 2, 4)
        print(sharding.names)  # Output: ['dp', 'pp', 'tp']
        print(sharding.get_ranks(dp=0, pp=1)) # Output: [4, 5, 6, 7]
    """

    def __init__(self, layout: Sequence[Any] | np.ndarray, names: list[str]):
        """Initializes the NamedSharding object.

        Args:
            layout: A nested sequence (e.g., list of lists) representing the ND rank layout.
                    All inner lists must contain integer rank IDs.
            names: A list of strings representing the names of the dimensions,
                   ordered from the outermost to the innermost dimension.
        """
        # Convert to numpy array first, inferring dtype
        try:
            initial_array = np.array(layout)
        except (
            ValueError
        ) as e:  # Catch potential errors during array creation (e.g., ragged arrays)
            raise ValueError(f"Could not create NumPy array from layout: {e}")

        # Check if the inferred dtype is integer-like or float representing integers
        self._layout: np.ndarray[tuple[int, ...], np.dtype[np.int32]]
        if not np.issubdtype(initial_array.dtype, np.integer):
            # Check if all elements are actually integers (handles floats like 1.0)
            if not np.equal(np.mod(initial_array, 1), 0).all():
                raise ValueError("Layout must contain only integer rank IDs.")
            # If they are float but represent integers (e.g., 1.0), cast them
            self._layout = initial_array.astype(np.int32)
        else:
            self._layout = initial_array  # Already integer type

        self._names = list(names)

        if self._layout.ndim != len(self._names):
            raise ValueError(
                f"Number of dimensions in layout ({self._layout.ndim}) "
                f"must match the number of names ({len(self._names)})."
            )

        # Check for duplicate ranks (on the final integer array)
        unique_ranks, counts = np.unique(self._layout, return_counts=True)
        duplicates = unique_ranks[counts > 1]
        if duplicates.size > 0:
            raise ValueError(f"Duplicate ranks found in layout: {duplicates.tolist()}")

        self._name_to_axis = {name: i for i, name in enumerate(self._names)}

    @property
    def shape(self) -> dict[str, int]:
        """Returns the shape of the rank layout."""
        return {name: size for name, size in zip(self._names, self._layout.shape)}

    @property
    def names(self) -> list[str]:
        """Returns the names of the axes."""
        return list(self._names)  # Return a copy

    @property
    def ndim(self) -> int:
        """Returns the number of dimensions."""
        return self._layout.ndim

    @property
    def size(self) -> int:
        """Returns the total number of ranks."""
        return self._layout.size

    @property
    def layout(self) -> np.ndarray[tuple[int, ...], np.dtype[np.int32]]:
        """Returns the underlying NumPy array representing the layout."""
        return self._layout.copy()  # Return a copy

    def get_worker_coords(self, worker_id: int) -> dict[str, int]:
        """Gets the coordinates of a specific worker ID in the sharding layout.

        Args:
            worker_id: The integer ID of the worker.

        Returns:
            A dictionary mapping axis names to their integer coordinates for the given worker_id.

        Raises:
            ValueError: If the worker_id is not found in the layout.
        """
        indices = np.where(self._layout == worker_id)
        if not indices[0].size:  # Check if worker_id was found
            raise ValueError(f"Worker ID {worker_id} not found in sharding layout.")

        coords = {}
        for i, axis_name in enumerate(self._names):
            coords[axis_name] = indices[i].item()
        return coords

    def get_ranks_by_coord(self, **coords: int) -> list[int]:
        """Gets all ranks that match the specified coordinates for named axes.

        Args:
            **coords: Keyword arguments where the key is the axis name (e.g., "dp", "tp")
                      and the value is the integer coordinate along that axis.
                      Axes not specified will match all coordinates along that axis.

        Returns:
            A sorted list of unique rank integers that match the given coordinate criteria.
            Returns an empty list if no ranks match.

        Raises:
            ValueError: If an invalid axis name is provided.
        """
        slicing_indices: list[Any] = [slice(None)] * self.ndim

        for name, index in coords.items():
            if name not in self._name_to_axis:
                raise ValueError(
                    f"Invalid axis name: '{name}'. Valid names are: {self.names}"
                )
            axis_idx = self._name_to_axis[name]
            if not (0 <= index < self.shape[name]):
                # If index is out of bounds for this axis, no ranks will match.
                return []
            slicing_indices[axis_idx] = index

        matching_ranks = self._layout[tuple(slicing_indices)]
        return sorted(np.unique(matching_ranks.flatten()).tolist())

    def get_ranks(self, **kwargs: int) -> Union["NamedSharding", int]:
        """Gets the ranks corresponding to specific indices along named axes.

        Args:
            **kwargs: Keyword arguments where the key is the axis name (e.g., "dp", "tp")
                      and the value is the index along that axis.

        Returns:
            A new NamedSharding instance representing the subset of ranks.
            The shape of the returned sharding corresponds to the axes *not* specified
            in the kwargs. If all axes are specified, an int is returned.

        Raises:
            ValueError: If an invalid axis name is provided or if an index is out of bounds.
        """
        indices: list[Any] = [slice(None)] * self.ndim
        specified_axes = set()

        for name, index in kwargs.items():
            if name not in self._name_to_axis:
                raise ValueError(
                    f"Invalid axis name: '{name}'. Valid names are: {self.names}"
                )
            if not (0 <= index < self.shape[name]):
                raise IndexError(
                    f"Index {index} is out of bounds for axis '{name}' with size {self.shape[name]}"
                )

            axis_index = self._name_to_axis[name]
            indices[axis_index] = index
            specified_axes.add(axis_index)

        # Get the subset of ranks
        subset_layout = self._layout[tuple(indices)]

        # Create a new list of names for the remaining dimensions
        remaining_names = [
            name for i, name in enumerate(self._names) if i not in specified_axes
        ]

        # If all dimensions were specified, we need to handle the 0-dimensional case
        if not remaining_names:
            return subset_layout.item()  # type: ignore

        return NamedSharding(subset_layout, remaining_names)

    def get_axis_index(self, name: str) -> int:
        """Gets the numerical index of a named axis."""
        if name not in self._name_to_axis:
            raise ValueError(
                f"Invalid axis name: '{name}'. Valid names are: {self.names}"
            )
        return self._name_to_axis[name]

    def get_axis_size(self, name: str) -> int:
        """Gets the size of a named axis."""
        return self.shape[name]

    def __repr__(self) -> str:
        shape_str = ", ".join([f"{self.shape[name]}" for name in self.names])
        return f"NamedSharding(shape=({shape_str}), names={self.names}, layout={self._layout})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NamedSharding):
            return NotImplemented
        return (
            np.array_equal(self._layout, other._layout) and self._names == other._names
        )

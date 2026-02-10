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

"""Sequence packing algorithms for efficient batching of variable-length sequences."""

import enum
import math
import random
from abc import ABC, abstractmethod
from bisect import bisect
from typing import Dict, List, Optional, Tuple, Type, Union


class PackingAlgorithm(enum.Enum):
    """Enum for supported sequence packing algorithms."""

    CONCATENATIVE = "concatenative"
    FIRST_FIT_DECREASING = "first_fit_decreasing"
    FIRST_FIT_SHUFFLE = "first_fit_shuffle"
    MODIFIED_FIRST_FIT_DECREASING = "modified_first_fit_decreasing"


class SequencePacker(ABC):
    """Abstract base class for sequence packing algorithms.

    Sequence packing is the process of efficiently arranging sequences of different
    lengths into fixed-capacity bins (batches) to maximize computational efficiency.
    """

    def __init__(
        self,
        bin_capacity: int,
        collect_metrics: bool = False,
        min_bin_count: Optional[int] = None,
        bin_count_multiple: Optional[int] = None,
    ):
        """Initialize the sequence packer.

        Args:
            bin_capacity: The maximum capacity of each bin.
            collect_metrics: Whether to collect metrics across multiple packing operations.
            min_bin_count: Minimum number of bins to create, even if fewer would suffice.
                          If None, no minimum is enforced.
            bin_count_multiple: The total number of bins must be a multiple of this value.
                               If None, no multiple constraint is enforced.

        Raises:
            ValueError: If min_bin_count or bin_count_multiple are invalid.
        """
        self.bin_capacity = bin_capacity
        self.collect_metrics = collect_metrics
        self.min_bin_count = min_bin_count
        self.bin_count_multiple = bin_count_multiple
        self.metrics = None

        # Validate parameters
        if min_bin_count is not None and min_bin_count < 0:
            raise ValueError("min_bin_count must be nonnegative")
        if bin_count_multiple is not None and bin_count_multiple < 1:
            raise ValueError("bin_count_multiple must be positive")

        if collect_metrics:
            from nemo_rl.data.packing.metrics import PackingMetrics

            self.metrics = PackingMetrics()

    @abstractmethod
    def _pack_implementation(self, sequence_lengths: List[int]) -> List[List[int]]:
        """Implementation of the packing algorithm.

        Args:
            sequence_lengths: A list of sequence lengths to pack.

        Returns:
            A list of bins, where each bin is a list of indices into the original
            sequence_lengths list.
        """
        pass

    def _adjust_bin_count(self, bins: List[List[int]]) -> List[List[int]]:
        """Adjust the number of bins to meet minimum and multiple constraints.

        This method preserves the existing bin packing as much as possible and only
        moves sequences one at a time to create additional bins when needed.

        Args:
            bins: The original bins from the packing algorithm.

        Returns:
            Adjusted bins with minimal changes to meet constraints.

        Raises:
            ValueError: If there aren't enough sequences to fill the required number of bins.
        """
        current_bin_count = len(bins)
        target_bin_count = current_bin_count

        if self.min_bin_count is not None:
            target_bin_count = max(target_bin_count, self.min_bin_count)

        if self.bin_count_multiple is not None:
            remainder = target_bin_count % self.bin_count_multiple
            if remainder != 0:
                target_bin_count += self.bin_count_multiple - remainder

        if target_bin_count == current_bin_count:
            return bins

        # Count total sequences
        total_sequences = sum(len(bin_contents) for bin_contents in bins)
        if total_sequences < target_bin_count:
            raise ValueError(
                f"Cannot create {target_bin_count} bins with only {total_sequences} sequences. "
                f"Each bin must contain at least one sequence. "
                f"Either reduce min_bin_count/bin_count_multiple or provide more sequences."
            )

        adjusted_bins = [bin_contents.copy() for bin_contents in bins]
        additional_bins_needed = target_bin_count - current_bin_count
        for _ in range(additional_bins_needed):
            adjusted_bins.append([])

        # Move sequences from existing bins to new bins
        bin_sizes = [
            (len(bin_contents), i)
            for i, bin_contents in enumerate(adjusted_bins[:current_bin_count])
        ]
        bin_sizes.sort(reverse=True)  # Sort by size, largest first

        source_bin_idx = 0

        for new_bin_idx in range(current_bin_count, target_bin_count):
            # Find a bin with at least 2 sequences (so we can move one and leave at least one)
            while source_bin_idx < len(bin_sizes):
                bin_size, original_bin_idx = bin_sizes[source_bin_idx]
                current_size = len(adjusted_bins[original_bin_idx])

                if current_size > 1:
                    # Move one sequence from this bin to the new bin
                    sequence_to_move = adjusted_bins[original_bin_idx].pop()
                    adjusted_bins[new_bin_idx].append(sequence_to_move)
                    break
                else:
                    # This bin only has one sequence, try the next one
                    source_bin_idx += 1
            else:
                # If we get here, we couldn't find any bin with more than 1 sequence
                # This should not happen given our earlier validation, but let's handle it
                raise ValueError(
                    f"Cannot create additional bins: insufficient sequences to redistribute. "
                    f"Need {additional_bins_needed} additional bins but cannot find enough "
                    f"bins with multiple sequences to redistribute from."
                    f"WARNING: Triggering this section of code is a bug. Please report it."
                )

        return adjusted_bins

    def pack(self, sequence_lengths: List[int]) -> List[List[int]]:
        """Pack sequences into bins and update metrics if enabled.

        Args:
            sequence_lengths: A list of sequence lengths to pack.

        Returns:
            A list of bins, where each bin is a list of indices into the original
            sequence_lengths list. The number of bins will satisfy min_bin_count
            and bin_count_multiple constraints if specified.
        """
        # Call the implementation
        bins = self._pack_implementation(sequence_lengths)

        # Adjust bin count to meet constraints
        bins = self._adjust_bin_count(bins)

        # Update metrics if collection is enabled
        if self.collect_metrics and self.metrics:
            self.metrics.update(sequence_lengths, bins, self.bin_capacity)

        return bins

    def reset_metrics(self) -> None:
        """Reset collected metrics."""
        if self.metrics:
            self.metrics.reset()

    def compute_metrics(
        self, sequence_lengths: List[int], bins: List[List[int]]
    ) -> Dict[str, float]:
        """Calculate metrics for a packing solution without updating the metrics tracker.

        Args:
            sequence_lengths: List of sequence lengths
            bins: List of bins, where each bin is a list of indices

        Returns:
            Dictionary of packing metrics
        """
        if self.metrics:
            return self.metrics.calculate_stats_only(
                sequence_lengths, bins, self.bin_capacity
            )
        else:
            # Create a temporary metrics object if not collecting
            from nemo_rl.data.packing.metrics import PackingMetrics

            temp_metrics = PackingMetrics()
            return temp_metrics.calculate_stats_only(
                sequence_lengths, bins, self.bin_capacity
            )

    def get_aggregated_metrics(self) -> Dict[str, float]:
        """Get aggregated metrics across all packing operations.

        Returns:
            Dictionary of aggregated metrics, or empty dict if not collecting
        """
        if self.metrics:
            return self.metrics.get_aggregated_stats()
        else:
            return {}

    def print_metrics(self) -> None:
        """Print the current metrics in a formatted way."""
        if not self.metrics:
            print(
                "Metrics collection is not enabled. Initialize with collect_metrics=True."
            )
            return

        self.metrics.print_aggregated_stats()

    def _validate_sequence_lengths(self, sequence_lengths: List[int]) -> None:
        """Validate that all sequence lengths are within bin capacity.

        Args:
            sequence_lengths: A list of sequence lengths to validate.

        Raises:
            ValueError: If any sequence length exceeds bin capacity.
        """
        for length in sequence_lengths:
            if length > self.bin_capacity:
                raise ValueError(
                    f"Sequence length {length} exceeds bin capacity {self.bin_capacity}"
                )

    def _create_indexed_lengths(
        self, sequence_lengths: List[int], reverse: bool = False
    ) -> List[Tuple[int, int]]:
        """Create a list of (length, index) pairs from sequence lengths.

        Args:
            sequence_lengths: A list of sequence lengths.
            reverse: Whether to sort in descending order (True) or ascending order (False).

        Returns:
            A list of (length, index) pairs, optionally sorted.
        """
        indexed_lengths = [(length, i) for i, length in enumerate(sequence_lengths)]
        if reverse:
            indexed_lengths.sort(reverse=True)  # Sort in descending order
        return indexed_lengths

    def _estimate_bins_needed(self, sequence_lengths: List[int]) -> int:
        """Estimate the number of bins needed based on total length.

        Args:
            sequence_lengths: A list of sequence lengths.

        Returns:
            Estimated number of bins needed.
        """
        total_length = sum(sequence_lengths)
        return max(1, math.ceil(total_length / self.bin_capacity))


class ConcatenativePacker(SequencePacker):
    """Concatenative packing algorithm.

    This algorithm simply concatenates sequences in order until reaching the bin capacity,
    then starts a new bin. It doesn't try to optimize the packing in any way.

    Time complexity: O(n) where n is the number of sequences.

    Example:
    ```python
    >>> examples = {
    ...     "sequence_lengths": [4, 1, 3, 2, 1, 3, 4, 5]
    ... }
    >>> # If packed with seq_length=5:
    ... {"bins": [ [0, 1], [2, 3], [4, 5], [6], [7] ]}
    >>> # If packed with seq_length=8:
    ... {"bins": [ [0, 1, 2], [3, 4, 5], [6], [7] ]}
    """

    # Global class variable to limit the number of sequences packed in a unit
    # -1 disables this limit
    max_sequences_per_bin = -1  # Useful for debugging and testing

    def _pack_implementation(self, sequence_lengths: List[int]) -> List[List[int]]:
        """Pack sequences using the Concatenative algorithm.

        Args:
            sequence_lengths: A list of sequence lengths to pack.

        Returns:
            A list of bins, where each bin is a list of indices into the original
            sequence_lengths list.
        """
        # Validate sequence lengths
        self._validate_sequence_lengths(sequence_lengths)

        bins = []  # List of bins, each bin is a list of sequence indices
        current_bin = []  # Current bin being filled
        current_length = 0  # Current length of sequences in the bin

        for i, length in enumerate(sequence_lengths):
            # Check if adding this sequence would exceed bin capacity or sequence limit
            exceeds_capacity = current_length + length > self.bin_capacity
            exceeds_sequence_limit = (
                self.max_sequences_per_bin != -1
                and len(current_bin) >= self.max_sequences_per_bin
            )

            # If adding this sequence would exceed constraints, start a new bin
            if exceeds_capacity or exceeds_sequence_limit:
                if current_bin:  # Only add the bin if it's not empty
                    bins.append(current_bin)
                current_bin = [i]
                current_length = length
            else:
                # Add the sequence to the current bin
                current_bin.append(i)
                current_length += length

        # Add the last bin if it's not empty
        if current_bin:
            bins.append(current_bin)

        return bins


class FirstFitPacker(SequencePacker):
    """Base class for First-Fit algorithms.

    First-Fit algorithms place each sequence into the first bin where it fits.
    If no bin can fit the sequence, a new bin is created.

    This is an abstract base class that provides the common implementation for
    First-Fit variants. Subclasses must implement the _prepare_sequences method
    to determine the order in which sequences are processed.
    """

    def _prepare_sequences(self, sequence_lengths: List[int]) -> List[Tuple[int, int]]:
        """Prepare sequences for packing.

        This method determines the order in which sequences are processed.
        Subclasses must override this method.

        Args:
            sequence_lengths: A list of sequence lengths to pack.

        Returns:
            A list of (length, index) pairs.
        """
        raise NotImplementedError("Subclasses must implement _prepare_sequences")

    def _pack_implementation(self, sequence_lengths: List[int]) -> List[List[int]]:
        """Pack sequences using the First-Fit algorithm.

        Args:
            sequence_lengths: A list of sequence lengths to pack.

        Returns:
            A list of bins, where each bin is a list of indices into the original
            sequence_lengths list.
        """
        # Prepare sequences for packing (order determined by subclass)
        indexed_lengths = self._prepare_sequences(sequence_lengths)

        bins = []  # List of bins, each bin is a list of sequence indices
        bin_remaining = []  # Remaining capacity for each bin

        for length, idx in indexed_lengths:
            # If the sequence is larger than the bin capacity, it cannot be packed
            if length > self.bin_capacity:
                raise ValueError(
                    f"Sequence length {length} exceeds bin capacity {self.bin_capacity}"
                )

            # Try to find a bin where the sequence fits
            bin_found = False
            for i, remaining in enumerate(bin_remaining):
                if remaining >= length:
                    # Add the sequence to this bin
                    bins[i].append(idx)
                    bin_remaining[i] -= length
                    bin_found = True
                    break

            # If no suitable bin was found, create a new one
            if not bin_found:
                bins.append([idx])
                bin_remaining.append(self.bin_capacity - length)

        return bins


class FirstFitDecreasingPacker(FirstFitPacker):
    """First-Fit Decreasing (FFD) algorithm for sequence packing.

    This algorithm sorts sequences by length in descending order and then
    places each sequence into the first bin where it fits.

    Time complexity: O(n log n) for sorting + O(n * m) for packing,
    where n is the number of sequences and m is the number of bins.
    """

    def _prepare_sequences(self, sequence_lengths: List[int]) -> List[Tuple[int, int]]:
        """Prepare sequences for packing by sorting them in descending order.

        Args:
            sequence_lengths: A list of sequence lengths to pack.

        Returns:
            A list of (length, index) pairs sorted by length in descending order.
        """
        # Create a list of (length, index) pairs
        indexed_lengths = [(length, i) for i, length in enumerate(sequence_lengths)]

        # Sort by length in descending order
        indexed_lengths.sort(reverse=True)

        return indexed_lengths


class FirstFitShufflePacker(FirstFitPacker):
    """First-Fit Shuffle algorithm for sequence packing.

    This algorithm randomly shuffles the sequences and then places each
    sequence into the first bin where it fits.

    Time complexity: O(n * m) for packing, where n is the number of sequences
    and m is the number of bins.
    """

    def _prepare_sequences(self, sequence_lengths: List[int]) -> List[Tuple[int, int]]:
        """Prepare sequences for packing by randomly shuffling them.

        Args:
            sequence_lengths: A list of sequence lengths to pack.

        Returns:
            A list of (length, index) pairs in random order.
        """
        # Create a list of (length, index) pairs
        indexed_lengths = [(length, i) for i, length in enumerate(sequence_lengths)]

        # Shuffle the sequences
        random.shuffle(indexed_lengths)

        return indexed_lengths


class ModifiedFirstFitDecreasingPacker(SequencePacker):
    """Modified First-Fit Decreasing (MFFD) algorithm for sequence packing.

    This algorithm implements the Johnson & Garey (1985) Modified First-Fit-Decreasing
    heuristic. It classifies items into four categories (large, medium, small, tiny)
    and uses a sophisticated 5-phase packing strategy to achieve better bin utilization
    than standard First-Fit Decreasing.

    The algorithm phases:
    1. Classify items by size relative to bin capacity
    2. Create one bin per large item
    3. Add medium items to large bins (forward pass)
    4. Add pairs of small items to bins with medium items (backward pass)
    5. Greedily fit remaining items
    6. Apply FFD to any leftovers

    Time complexity: O(n log n) for sorting + O(n * m) for packing,
    where n is the number of sequences and m is the number of bins.
    """

    def _classify_items(
        self, items: List[Tuple[int, int]]
    ) -> Tuple[
        List[Tuple[int, int]],
        List[Tuple[int, int]],
        List[Tuple[int, int]],
        List[Tuple[int, int]],
    ]:
        """Split items into large / medium / small / tiny classes.

        Follows the classification used by Johnson & Garey:
            large   : (C/2, C]
            medium  : (C/3, C/2]
            small   : (C/6, C/3]
            tiny    : (0  , C/6]

        Args:
            items: List of (index, size) tuples

        Returns:
            Tuple of four lists (large, medium, small, tiny) without additional sorting.
        """
        large, medium, small, tiny = [], [], [], []
        for idx, size in items:
            if size > self.bin_capacity / 2:
                large.append((idx, size))
            elif size > self.bin_capacity / 3:
                medium.append((idx, size))
            elif size > self.bin_capacity / 6:
                small.append((idx, size))
            else:
                tiny.append((idx, size))
        return large, medium, small, tiny

    def _pack_implementation(self, sequence_lengths: List[int]) -> List[List[int]]:
        """Pack sequences using the Modified First-Fit Decreasing algorithm.

        Args:
            sequence_lengths: A list of sequence lengths to pack.

        Returns:
            A list of bins, where each bin is a list of indices into the original
            sequence_lengths list.
        """
        # Validate inputs
        if self.bin_capacity <= 0:
            raise ValueError("bin_capacity must be positive")
        if any(l <= 0 for l in sequence_lengths):
            raise ValueError("sequence lengths must be positive")

        # Validate sequence lengths don't exceed capacity
        self._validate_sequence_lengths(sequence_lengths)

        items: List[Tuple[int, int]] = [(i, l) for i, l in enumerate(sequence_lengths)]

        # Phase-0: classify
        large, medium, small, tiny = self._classify_items(items)

        # Sort according to the rules of MFFD
        large.sort(key=lambda x: x[1], reverse=True)  # descending size
        medium.sort(key=lambda x: x[1], reverse=True)
        small.sort(key=lambda x: x[1])  # ascending size
        tiny.sort(key=lambda x: x[1])

        # Phase-1: start one bin per large item
        bins: List[List[Tuple[int, int]]] = [[item] for item in large]

        # Phase-2: try to add one medium item to each large bin (forward pass)
        for b in bins:
            remaining = self.bin_capacity - sum(size for _, size in b)
            for i, (idx, size) in enumerate(medium):
                if size <= remaining:
                    b.append(medium.pop(i))
                    break

        # Phase-3: backward pass – fill with two small items where possible
        for b in reversed(bins):
            has_medium = any(
                self.bin_capacity / 3 < size <= self.bin_capacity / 2 for _, size in b
            )
            if has_medium or len(small) < 2:
                continue
            remaining = self.bin_capacity - sum(size for _, size in b)
            if small[0][1] + small[1][1] > remaining:
                continue
            first_small = small.pop(0)
            # pick the *largest* small that fits with first_small (so iterate from end)
            second_idx = None
            for j in range(len(small) - 1, -1, -1):
                if small[j][1] <= remaining - first_small[1]:
                    second_idx = j
                    break
            if second_idx is not None:
                second_small = small.pop(second_idx)
                b.extend([first_small, second_small])

        # Phase-4: forward greedy fit of remaining items
        remaining_items = sorted(
            medium + small + tiny, key=lambda x: x[1], reverse=True
        )
        for b in bins:
            while remaining_items:
                rem = self.bin_capacity - sum(size for _, size in b)
                # if even the smallest remaining doesn't fit we break
                if rem < remaining_items[-1][1]:
                    break

                # pick the first (largest) that fits
                chosen_idx = None
                for i, (_, size) in enumerate(remaining_items):
                    if size <= rem:
                        chosen_idx = i
                        break
                if chosen_idx is None:
                    break
                b.append(remaining_items.pop(chosen_idx))

        # Phase-5: FFD on leftovers
        leftovers = remaining_items  # renamed for clarity

        # Original O(n * m) implementation
        """
        ffd_bins: List[List[Tuple[int, int]]] = []
        for idx, size in sorted(leftovers, key=lambda x: x[1], reverse=True):
            placed = False
            for bin_ffd in ffd_bins:
                if size <= self.bin_capacity - sum(s for _, s in bin_ffd):
                    bin_ffd.append((idx, size))
                    placed = True
                    break
            if not placed:
                ffd_bins.append([(idx, size)])
        """

        # New O(n * logn) implementation
        ffd_bins: List[List[Tuple[int, int]]] = [[]]
        ffd_bin_sizes: List[int] = [0]
        for idx, size in sorted(leftovers, key=lambda x: x[1], reverse=True):
            # We only need to check the first bin since we guarantee the order of ffd_bin_sizes to be sorted from smallest to largest.
            if size <= (self.bin_capacity - ffd_bin_sizes[0]):
                new_bin = ffd_bins.pop(0)
                new_bin_size = ffd_bin_sizes.pop(0)
            else:
                new_bin = []
                new_bin_size = 0

            new_bin.append((idx, size))
            new_bin_size += size

            new_idx = bisect(ffd_bin_sizes, new_bin_size)
            ffd_bins.insert(new_idx, new_bin)
            ffd_bin_sizes.insert(new_idx, new_bin_size)

        bins.extend(ffd_bins)

        # Convert to list of index lists (discard sizes)
        return [[idx for idx, _ in b] for b in bins if b]


def get_packer(
    algorithm: Union[PackingAlgorithm, str],
    bin_capacity: int,
    collect_metrics: bool = False,
    min_bin_count: Optional[int] = None,
    bin_count_multiple: Optional[int] = None,
) -> SequencePacker:
    """Factory function to get a sequence packer based on the algorithm.

    Args:
        algorithm: The packing algorithm to use. Can be either a PackingAlgorithm enum value
                  or a string (case-insensitive) matching one of the enum names.
        bin_capacity: The maximum capacity of each bin.
        collect_metrics: Whether to collect metrics across multiple packing operations.
        min_bin_count: Minimum number of bins to create, even if fewer would suffice.
                      If None, no minimum is enforced.
        bin_count_multiple: The total number of bins must be a multiple of this value.
                           If None, no multiple constraint is enforced.

    Returns:
        A SequencePacker instance for the specified algorithm.

    Raises:
        ValueError: If the algorithm is not recognized.
    """
    packers: Dict[PackingAlgorithm, Type[SequencePacker]] = {
        PackingAlgorithm.CONCATENATIVE: ConcatenativePacker,
        PackingAlgorithm.FIRST_FIT_DECREASING: FirstFitDecreasingPacker,
        PackingAlgorithm.FIRST_FIT_SHUFFLE: FirstFitShufflePacker,
        PackingAlgorithm.MODIFIED_FIRST_FIT_DECREASING: ModifiedFirstFitDecreasingPacker,
    }

    # Convert string to enum if needed
    if isinstance(algorithm, str):
        try:
            algorithm = PackingAlgorithm[algorithm.upper()]
        except KeyError:
            available_algorithms = ", ".join([alg.name for alg in PackingAlgorithm])
            raise ValueError(
                f"Unknown packing algorithm: {algorithm}. "
                f"Available algorithms: {available_algorithms}"
            )

    if algorithm not in packers:
        available_algorithms = ", ".join([alg.name for alg in PackingAlgorithm])
        raise ValueError(
            f"Unknown packing algorithm: {algorithm}. "
            f"Available algorithms: {available_algorithms}"
        )

    return packers[algorithm](
        bin_capacity,
        collect_metrics=collect_metrics,
        min_bin_count=min_bin_count,
        bin_count_multiple=bin_count_multiple,
    )

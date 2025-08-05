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

"""Tests for sequence packing algorithms."""

import random
from typing import Dict, List

import pytest

from nemo_rl.data.packing.algorithms import (
    PackingAlgorithm,
    SequencePacker,
    get_packer,
)


def validate_solution(
    sequence_lengths: List[int], bins: List[List[int]], bin_capacity: int
) -> bool:
    """Validate that a packing solution is valid.

    Args:
        sequence_lengths: The original list of sequence lengths.
        bins: The packing solution, where each bin is a list of indices into sequence_lengths.
        bin_capacity: The maximum capacity of each bin.

    Returns:
        True if the packing is valid, False otherwise.
    """
    # Check that all sequences are packed
    all_indices = set()
    for bin_indices in bins:
        all_indices.update(bin_indices)

    if len(all_indices) != len(sequence_lengths):
        return False

    # Check that each bin doesn't exceed capacity
    for bin_indices in bins:
        bin_load = sum(sequence_lengths[idx] for idx in bin_indices)
        if bin_load > bin_capacity:
            return False

    return True


class TestSequencePacker:
    """Test suite for sequence packing algorithms."""

    @pytest.fixture
    def bin_capacity(self) -> int:
        """Fixture for bin capacity."""
        return 100

    @pytest.fixture
    def small_sequence_lengths(self) -> List[int]:
        """Fixture for a small list of sequence lengths."""
        return [10, 20, 30, 40, 50, 60, 70, 80, 90]

    @pytest.fixture
    def medium_sequence_lengths(self) -> List[int]:
        """Fixture for a medium-sized list of sequence lengths."""
        return [25, 35, 45, 55, 65, 75, 85, 95, 15, 25, 35, 45, 55, 65, 75, 85, 95]

    @pytest.fixture
    def large_sequence_lengths(self) -> List[int]:
        """Fixture for a large list of sequence lengths."""
        # Set a seed for reproducibility
        random.seed(42)
        return [random.randint(10, 90) for _ in range(100)]

    @pytest.fixture
    def edge_cases(self) -> Dict[str, List[int]]:
        """Fixture for edge cases."""
        return {
            "empty": [],
            "single_item": [50],
            "all_same_size": [30, 30, 30, 30, 30],
            "max_size": [100, 100, 100],
            "mixed_sizes": [10, 50, 100, 20, 80, 30, 70, 40, 60, 90],
        }

    # TODO(ahmadki): use the function to specify all test algorithms ins tead of lists below
    @pytest.fixture
    def algorithms(self) -> List[PackingAlgorithm]:
        """Fixture for packing algorithms."""
        return [
            PackingAlgorithm.CONCATENATIVE,
            PackingAlgorithm.FIRST_FIT_DECREASING,
            PackingAlgorithm.FIRST_FIT_SHUFFLE,
            PackingAlgorithm.MODIFIED_FIRST_FIT_DECREASING,
        ]

    def test_get_packer(self, bin_capacity: int, algorithms: List[PackingAlgorithm]):
        """Test the get_packer factory function."""
        # Test that each algorithm name returns the correct packer
        for algorithm in algorithms:
            packer = get_packer(algorithm, bin_capacity)
            assert isinstance(packer, SequencePacker)

        # Test with an invalid algorithm value
        with pytest.raises(ValueError):
            # Create a non-existent enum value by using an arbitrary object
            invalid_algorithm = object()
            get_packer(invalid_algorithm, bin_capacity)  # type: ignore

    @pytest.mark.parametrize(
        "algorithm",
        [
            PackingAlgorithm.CONCATENATIVE,
            PackingAlgorithm.FIRST_FIT_DECREASING,
            PackingAlgorithm.FIRST_FIT_SHUFFLE,
            PackingAlgorithm.MODIFIED_FIRST_FIT_DECREASING,
        ],
    )
    def test_small_sequences(
        self,
        bin_capacity: int,
        small_sequence_lengths: List[int],
        algorithm: PackingAlgorithm,
    ):
        """Test packing small sequences with all algorithms."""
        packer = get_packer(algorithm, bin_capacity)
        bins = packer.pack(small_sequence_lengths)

        # Validate the packing
        assert validate_solution(small_sequence_lengths, bins, bin_capacity)

        # Print the number of bins used (for information)
        print(f"{algorithm.name} used {len(bins)} bins for small sequences")

    @pytest.mark.parametrize(
        "algorithm",
        [
            PackingAlgorithm.CONCATENATIVE,
            PackingAlgorithm.FIRST_FIT_DECREASING,
            PackingAlgorithm.FIRST_FIT_SHUFFLE,
            PackingAlgorithm.MODIFIED_FIRST_FIT_DECREASING,
        ],
    )
    def test_medium_sequences(
        self,
        bin_capacity: int,
        medium_sequence_lengths: List[int],
        algorithm: PackingAlgorithm,
    ):
        """Test packing medium-sized sequences with all algorithms."""
        packer = get_packer(algorithm, bin_capacity)
        bins = packer.pack(medium_sequence_lengths)

        # Validate the packing
        assert validate_solution(medium_sequence_lengths, bins, bin_capacity)

        # Print the number of bins used (for information)
        print(f"{algorithm.name} used {len(bins)} bins for medium sequences")

    @pytest.mark.parametrize(
        "algorithm",
        [
            PackingAlgorithm.CONCATENATIVE,
            PackingAlgorithm.FIRST_FIT_DECREASING,
            PackingAlgorithm.FIRST_FIT_SHUFFLE,
            PackingAlgorithm.MODIFIED_FIRST_FIT_DECREASING,
        ],
    )
    def test_large_sequences(
        self,
        bin_capacity: int,
        large_sequence_lengths: List[int],
        algorithm: PackingAlgorithm,
    ):
        """Test packing large sequences with all algorithms."""
        packer = get_packer(algorithm, bin_capacity)
        bins = packer.pack(large_sequence_lengths)

        # Validate the packing
        assert validate_solution(large_sequence_lengths, bins, bin_capacity)

        # Print the number of bins used (for information)
        print(f"{algorithm.name} used {len(bins)} bins for large sequences")

    @pytest.mark.parametrize(
        "algorithm",
        [
            PackingAlgorithm.CONCATENATIVE,
            PackingAlgorithm.FIRST_FIT_DECREASING,
            PackingAlgorithm.FIRST_FIT_SHUFFLE,
            PackingAlgorithm.MODIFIED_FIRST_FIT_DECREASING,
        ],
    )
    # TODO(ahmadki): use the function to specify all test algorithms instead of lists below
    @pytest.mark.parametrize(
        "case_name, sequence_lengths",
        [
            ("single_item", [50]),
            ("all_same_size", [30, 30, 30, 30, 30]),
            ("max_size", [100, 100, 100]),
            ("mixed_sizes", [10, 50, 100, 20, 80, 30, 70, 40, 60, 90]),
        ],
    )
    def test_edge_cases(
        self,
        bin_capacity: int,
        algorithm: PackingAlgorithm,
        case_name: str,
        sequence_lengths: List[int],
    ):
        """Test edge cases with all algorithms."""
        packer = get_packer(algorithm, bin_capacity)
        bins = packer.pack(sequence_lengths)

        # Validate the packing
        assert validate_solution(sequence_lengths, bins, bin_capacity)

        # For single item, check that only one bin is created
        if case_name == "single_item":
            assert len(bins) == 1

    @pytest.mark.parametrize(
        "algorithm",
        [
            PackingAlgorithm.CONCATENATIVE,
            PackingAlgorithm.FIRST_FIT_DECREASING,
            PackingAlgorithm.FIRST_FIT_SHUFFLE,
            PackingAlgorithm.MODIFIED_FIRST_FIT_DECREASING,
        ],
    )
    def test_empty_list(self, bin_capacity: int, algorithm: PackingAlgorithm):
        """Test empty list with algorithms that can handle it."""
        packer = get_packer(algorithm, bin_capacity)
        bins = packer.pack([])

        # For empty list, check that no bins are created
        assert len(bins) == 0

    @pytest.mark.parametrize(
        "algorithm",
        [
            PackingAlgorithm.CONCATENATIVE,
            PackingAlgorithm.FIRST_FIT_DECREASING,
            PackingAlgorithm.FIRST_FIT_SHUFFLE,
            PackingAlgorithm.MODIFIED_FIRST_FIT_DECREASING,
        ],
    )
    def test_error_cases(self, bin_capacity: int, algorithm: PackingAlgorithm):
        """Test error cases with all algorithms."""
        # Test with a sequence length that exceeds bin capacity
        sequence_lengths = [50, 150, 70]  # 150 > bin_capacity (100)

        packer = get_packer(algorithm, bin_capacity)
        with pytest.raises(ValueError):
            packer.pack(sequence_lengths)

    @pytest.mark.parametrize(
        "algorithm",
        [
            PackingAlgorithm.CONCATENATIVE,
            PackingAlgorithm.FIRST_FIT_DECREASING,
            PackingAlgorithm.MODIFIED_FIRST_FIT_DECREASING,
        ],
    )
    def test_deterministic(
        self,
        bin_capacity: int,
        medium_sequence_lengths: List[int],
        algorithm: PackingAlgorithm,
    ):
        """Test that deterministic algorithms produce the same result on multiple runs."""
        packer = get_packer(algorithm, bin_capacity)

        # Run the algorithm twice and check that the results are the same
        bins1 = packer.pack(medium_sequence_lengths)
        bins2 = packer.pack(medium_sequence_lengths)

        # Convert to a format that can be compared (sort each bin and then sort the bins)
        sorted_bins1 = sorted([sorted(bin_indices) for bin_indices in bins1])
        sorted_bins2 = sorted([sorted(bin_indices) for bin_indices in bins2])

        assert sorted_bins1 == sorted_bins2

    @pytest.mark.parametrize(
        "algorithm",
        [
            PackingAlgorithm.FIRST_FIT_SHUFFLE,
        ],
    )
    def test_randomized(
        self,
        bin_capacity: int,
        medium_sequence_lengths: List[int],
        algorithm: PackingAlgorithm,
    ):
        """Test that randomized algorithms can produce different results on multiple runs."""
        # Note: This test might occasionally fail due to randomness

        # Set different seeds to ensure different random behavior
        random.seed(42)
        packer1 = get_packer(algorithm, bin_capacity)
        bins1 = packer1.pack(medium_sequence_lengths)

        random.seed(43)
        packer2 = get_packer(algorithm, bin_capacity)
        bins2 = packer2.pack(medium_sequence_lengths)

        # Convert to a format that can be compared
        sorted_bins1 = sorted([sorted(bin_indices) for bin_indices in bins1])
        sorted_bins2 = sorted([sorted(bin_indices) for bin_indices in bins2])

        # Check if the results are different
        # This is a weak test, as randomness might still produce the same result
        if sorted_bins1 == sorted_bins2:
            print(
                f"Warning: {algorithm.name} produced the same result with different seeds"
            )

    @pytest.mark.parametrize(
        "algorithm",
        [
            PackingAlgorithm.CONCATENATIVE,
            PackingAlgorithm.FIRST_FIT_DECREASING,
            PackingAlgorithm.FIRST_FIT_SHUFFLE,
            PackingAlgorithm.MODIFIED_FIRST_FIT_DECREASING,
        ],
    )
    def test_min_bin_count(
        self,
        bin_capacity: int,
        small_sequence_lengths: List[int],
        algorithm: PackingAlgorithm,
    ):
        """Test minimum bin count functionality."""
        # First get the natural packing
        packer_natural = get_packer(algorithm, bin_capacity)
        bins_natural = packer_natural.pack(small_sequence_lengths)
        natural_bin_count = len(bins_natural)

        # Test with min_bin_count equal to natural count (should be unchanged)
        packer_equal = get_packer(
            algorithm, bin_capacity, min_bin_count=natural_bin_count
        )
        bins_equal = packer_equal.pack(small_sequence_lengths)
        assert len(bins_equal) == natural_bin_count
        assert validate_solution(small_sequence_lengths, bins_equal, bin_capacity)

        # Test with min_bin_count greater than natural count
        min_bins = natural_bin_count + 2
        if min_bins <= len(small_sequence_lengths):  # Ensure we have enough sequences
            packer_more = get_packer(algorithm, bin_capacity, min_bin_count=min_bins)
            bins_more = packer_more.pack(small_sequence_lengths)
            assert len(bins_more) == min_bins
            assert validate_solution(small_sequence_lengths, bins_more, bin_capacity)

            # Verify no empty bins
            for bin_contents in bins_more:
                assert len(bin_contents) > 0, "Found empty bin"

    @pytest.mark.parametrize(
        "algorithm",
        [
            PackingAlgorithm.CONCATENATIVE,
            PackingAlgorithm.FIRST_FIT_DECREASING,
            PackingAlgorithm.FIRST_FIT_SHUFFLE,
            PackingAlgorithm.MODIFIED_FIRST_FIT_DECREASING,
        ],
    )
    def test_bin_count_multiple(
        self,
        bin_capacity: int,
        medium_sequence_lengths: List[int],
        algorithm: PackingAlgorithm,
    ):
        """Test bin count multiple functionality."""
        # Get natural packing
        packer_natural = get_packer(algorithm, bin_capacity)
        bins_natural = packer_natural.pack(medium_sequence_lengths)
        natural_bin_count = len(bins_natural)

        # Test with multiple that doesn't change the count
        if natural_bin_count % 2 == 0:
            multiple = 2
        else:
            multiple = natural_bin_count

        packer_multiple = get_packer(
            algorithm, bin_capacity, bin_count_multiple=multiple
        )
        bins_multiple = packer_multiple.pack(medium_sequence_lengths)
        assert len(bins_multiple) % multiple == 0
        assert validate_solution(medium_sequence_lengths, bins_multiple, bin_capacity)

        # Test with multiple that forces more bins
        multiple = 4
        expected_bins = ((natural_bin_count - 1) // multiple + 1) * multiple
        if expected_bins <= len(
            medium_sequence_lengths
        ):  # Ensure we have enough sequences
            packer_force = get_packer(
                algorithm, bin_capacity, bin_count_multiple=multiple
            )
            bins_force = packer_force.pack(medium_sequence_lengths)
            assert len(bins_force) == expected_bins
            assert len(bins_force) % multiple == 0
            assert validate_solution(medium_sequence_lengths, bins_force, bin_capacity)

            # Verify no empty bins
            for bin_contents in bins_force:
                assert len(bin_contents) > 0, "Found empty bin"

    @pytest.mark.parametrize(
        "algorithm",
        [
            PackingAlgorithm.CONCATENATIVE,
            PackingAlgorithm.FIRST_FIT_DECREASING,
            PackingAlgorithm.MODIFIED_FIRST_FIT_DECREASING,
        ],
    )
    def test_combined_constraints(
        self,
        bin_capacity: int,
        small_sequence_lengths: List[int],
        algorithm: PackingAlgorithm,
    ):
        """Test combined min_bin_count and bin_count_multiple constraints."""
        # Get natural packing
        packer_natural = get_packer(algorithm, bin_capacity)
        bins_natural = packer_natural.pack(small_sequence_lengths)
        natural_bin_count = len(bins_natural)

        min_bins = natural_bin_count + 1
        multiple = 3
        expected_bins = ((min_bins - 1) // multiple + 1) * multiple

        if expected_bins <= len(
            small_sequence_lengths
        ):  # Ensure we have enough sequences
            packer_combined = get_packer(
                algorithm,
                bin_capacity,
                min_bin_count=min_bins,
                bin_count_multiple=multiple,
            )
            bins_combined = packer_combined.pack(small_sequence_lengths)

            assert len(bins_combined) == expected_bins
            assert len(bins_combined) >= min_bins
            assert len(bins_combined) % multiple == 0
            assert validate_solution(
                small_sequence_lengths, bins_combined, bin_capacity
            )

            # Verify no empty bins
            for bin_contents in bins_combined:
                assert len(bin_contents) > 0, "Found empty bin"

    def test_constraint_error_cases(self, bin_capacity: int):
        """Test error cases for bin count constraints."""
        # Test invalid min_bin_count
        with pytest.raises(ValueError):
            get_packer(PackingAlgorithm.CONCATENATIVE, bin_capacity, min_bin_count=-1)

        # Test invalid bin_count_multiple
        with pytest.raises(ValueError):
            get_packer(
                PackingAlgorithm.CONCATENATIVE, bin_capacity, bin_count_multiple=0
            )

        with pytest.raises(ValueError):
            get_packer(
                PackingAlgorithm.CONCATENATIVE, bin_capacity, bin_count_multiple=-5
            )

    def test_insufficient_sequences_for_constraints(self, bin_capacity: int):
        """Test error when there aren't enough sequences to meet constraints."""
        sequence_lengths = [50, 50]  # Only 2 sequences

        # Test min_bin_count constraint with insufficient sequences
        packer = get_packer(
            PackingAlgorithm.CONCATENATIVE, bin_capacity, min_bin_count=3
        )
        with pytest.raises(
            ValueError, match="Cannot create 3 bins with only 2 sequences"
        ):
            packer.pack(sequence_lengths)

        # Test bin_count_multiple constraint with insufficient sequences
        packer = get_packer(
            PackingAlgorithm.CONCATENATIVE, bin_capacity, bin_count_multiple=4
        )
        with pytest.raises(
            ValueError, match="Cannot create 4 bins with only 2 sequences"
        ):
            packer.pack(sequence_lengths)

    @pytest.mark.parametrize(
        "algorithm",
        [
            PackingAlgorithm.CONCATENATIVE,
            PackingAlgorithm.FIRST_FIT_DECREASING,
            PackingAlgorithm.MODIFIED_FIRST_FIT_DECREASING,
        ],
    )
    def test_packing_preservation(
        self,
        bin_capacity: int,
        medium_sequence_lengths: List[int],
        algorithm: PackingAlgorithm,
    ):
        """Test that original packing efficiency is preserved when constraints are applied."""
        # Get natural packing and calculate utilization
        packer_natural = get_packer(algorithm, bin_capacity)
        bins_natural = packer_natural.pack(medium_sequence_lengths)

        def calculate_utilization(bins, sequence_lengths, bin_capacity):
            total_load = sum(sequence_lengths)
            total_capacity = len(bins) * bin_capacity
            return total_load / total_capacity if total_capacity > 0 else 0

        natural_utilization = calculate_utilization(
            bins_natural, medium_sequence_lengths, bin_capacity
        )

        # Force more bins and check that utilization doesn't degrade too much
        min_bins = len(bins_natural) + 1
        if min_bins <= len(medium_sequence_lengths):
            packer_constrained = get_packer(
                algorithm, bin_capacity, min_bin_count=min_bins
            )
            bins_constrained = packer_constrained.pack(medium_sequence_lengths)
            constrained_utilization = calculate_utilization(
                bins_constrained, medium_sequence_lengths, bin_capacity
            )

            # The utilization should decrease, but not dramatically
            # (since we're adding one bin, it should be roughly proportional)
            expected_utilization = (
                natural_utilization * len(bins_natural) / len(bins_constrained)
            )

            # Allow some tolerance due to redistribution effects
            assert constrained_utilization >= expected_utilization * 0.9, (
                f"Utilization degraded too much: {constrained_utilization} vs expected {expected_utilization}"
            )

    def test_factory_function_with_constraints(self, bin_capacity: int):
        """Test that the factory function properly passes constraint parameters."""
        # Test all parameter combinations
        packer1 = get_packer(
            PackingAlgorithm.CONCATENATIVE, bin_capacity, min_bin_count=5
        )
        assert packer1.min_bin_count == 5
        assert packer1.bin_count_multiple is None

        packer2 = get_packer(
            PackingAlgorithm.CONCATENATIVE, bin_capacity, bin_count_multiple=4
        )
        assert packer2.min_bin_count is None
        assert packer2.bin_count_multiple == 4

        packer3 = get_packer(
            PackingAlgorithm.CONCATENATIVE,
            bin_capacity,
            min_bin_count=3,
            bin_count_multiple=2,
        )
        assert packer3.min_bin_count == 3
        assert packer3.bin_count_multiple == 2

    def test_no_constraints_unchanged_behavior(
        self, bin_capacity: int, small_sequence_lengths: List[int]
    ):
        """Test that behavior is unchanged when no constraints are specified."""
        # Create packers with and without explicit None constraints
        packer1 = get_packer(PackingAlgorithm.CONCATENATIVE, bin_capacity)
        packer2 = get_packer(
            PackingAlgorithm.CONCATENATIVE,
            bin_capacity,
            min_bin_count=None,
            bin_count_multiple=None,
        )

        bins1 = packer1.pack(small_sequence_lengths)
        bins2 = packer2.pack(small_sequence_lengths)

        # Results should be identical
        assert bins1 == bins2

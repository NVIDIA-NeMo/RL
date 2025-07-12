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

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

"""Metrics for evaluating sequence packing algorithms."""

import math
import statistics
from typing import Dict, List, Optional


class PackingMetrics:
    """Class for tracking and computing metrics for sequence packing algorithms.

    This class provides methods to calculate various metrics that evaluate the
    efficiency and effectiveness of sequence packing algorithms, such as bin
    utilization, waste, and imbalance.
    """

    def __init__(self):
        """Initialize the metrics tracker."""
        self.reset()

    def reset(self) -> None:
        """Reset all metrics."""
        # Counters for aggregated metrics
        self.total_sequences = 0
        self.total_bins = 0
        self.total_sequence_length = 0
        self.total_bin_capacity = 0
        self.total_waste = 0
        self.bin_utilizations = []
        self.bin_counts = []
        self.packing_times = []

        # Tracking best and worst cases
        self.min_utilization = 1.0
        self.max_utilization = 0.0
        self.min_waste_ratio = 1.0
        self.max_waste_ratio = 0.0

    def update(
        self,
        sequence_lengths: List[int],
        bins: List[List[int]],
        bin_capacity: int,
        packing_time: Optional[float] = None,
    ) -> Dict[str, float]:
        """Update metrics with a new packing solution.

        Args:
            sequence_lengths: List of sequence lengths
            bins: List of bins, where each bin is a list of indices
            bin_capacity: Maximum capacity of each bin
            packing_time: Optional time taken to compute the packing solution

        Returns:
            Dictionary of metrics for this packing solution
        """
        # Calculate metrics for this solution
        stats = self.calculate_stats_only(sequence_lengths, bins, bin_capacity)

        # Update counters
        self.total_sequences += len(sequence_lengths)
        self.total_bins += len(bins)
        self.total_sequence_length += sum(sequence_lengths)
        self.total_bin_capacity += len(bins) * bin_capacity
        self.total_waste += stats["total_waste"]
        self.bin_utilizations.append(stats["average_utilization"])
        self.bin_counts.append(len(bins))

        if packing_time is not None:
            self.packing_times.append(packing_time)

        # Update min/max values
        self.min_utilization = min(self.min_utilization, stats["average_utilization"])
        self.max_utilization = max(self.max_utilization, stats["average_utilization"])
        self.min_waste_ratio = min(self.min_waste_ratio, stats["waste_ratio"])
        self.max_waste_ratio = max(self.max_waste_ratio, stats["waste_ratio"])

        return stats

    def calculate_stats_only(
        self, sequence_lengths: List[int], bins: List[List[int]], bin_capacity: int
    ) -> Dict[str, float]:
        """Calculate metrics for a packing solution without updating the tracker.

        Args:
            sequence_lengths: List of sequence lengths
            bins: List of bins, where each bin is a list of indices
            bin_capacity: Maximum capacity of each bin

        Returns:
            Dictionary of metrics for this packing solution
        """
        if not bins:
            return {
                "num_sequences": 0,
                "num_bins": 0,
                "total_sequence_length": 0,
                "total_bin_capacity": 0,
                "total_waste": 0,
                "average_utilization": 0.0,
                "waste_ratio": 0.0,
                "bin_balance": 0.0,
                "theoretical_min_bins": 0,
                "bin_efficiency": 0.0,
            }

        # Calculate bin loads
        bin_loads = [
            sum(sequence_lengths[idx] for idx in bin_indices) for bin_indices in bins
        ]

        # Calculate basic metrics
        num_sequences = len(sequence_lengths)
        num_bins = len(bins)
        total_sequence_length = sum(sequence_lengths)
        total_bin_capacity = num_bins * bin_capacity
        total_waste = total_bin_capacity - total_sequence_length

        # Calculate utilization metrics
        bin_utilizations = [load / bin_capacity for load in bin_loads]
        average_utilization = total_sequence_length / total_bin_capacity
        waste_ratio = total_waste / total_bin_capacity

        # Calculate bin balance metrics (standard deviation of utilization)
        if num_bins > 1:
            bin_balance = 1.0 - statistics.stdev(bin_utilizations) / average_utilization
        else:
            bin_balance = 1.0

        # Calculate theoretical minimum number of bins
        theoretical_min_bins = math.ceil(total_sequence_length / bin_capacity)

        # Calculate bin efficiency (ratio of theoretical min bins to actual bins)
        bin_efficiency = theoretical_min_bins / num_bins if num_bins > 0 else 0.0

        return {
            "num_sequences": num_sequences,
            "num_bins": num_bins,
            "total_sequence_length": total_sequence_length,
            "total_bin_capacity": total_bin_capacity,
            "total_waste": total_waste,
            "average_utilization": average_utilization,
            "waste_ratio": waste_ratio,
            "bin_balance": bin_balance,
            "theoretical_min_bins": theoretical_min_bins,
            "bin_efficiency": bin_efficiency,
        }

    def get_aggregated_stats(self) -> Dict[str, float]:
        """Get aggregated metrics across all packing operations.

        Returns:
            Dictionary of aggregated metrics
        """
        if not self.bin_utilizations:
            return {}

        # Calculate aggregated metrics
        avg_utilization = (
            self.total_sequence_length / self.total_bin_capacity
            if self.total_bin_capacity > 0
            else 0.0
        )
        avg_waste_ratio = (
            self.total_waste / self.total_bin_capacity
            if self.total_bin_capacity > 0
            else 0.0
        )
        avg_bin_count = (
            sum(self.bin_counts) / len(self.bin_counts) if self.bin_counts else 0.0
        )

        # Calculate theoretical minimum number of bins
        theoretical_min_bins = (
            math.ceil(
                self.total_sequence_length / (self.total_bin_capacity / self.total_bins)
            )
            if self.total_bins > 0
            else 0
        )

        # Calculate bin efficiency (ratio of theoretical min bins to actual bins)
        bin_efficiency = (
            theoretical_min_bins / self.total_bins if self.total_bins > 0 else 0.0
        )

        # Calculate average packing time if available
        avg_packing_time = (
            sum(self.packing_times) / len(self.packing_times)
            if self.packing_times
            else None
        )

        stats = {
            "total_sequences": self.total_sequences,
            "total_bins": self.total_bins,
            "average_utilization": avg_utilization,
            "min_utilization": self.min_utilization,
            "max_utilization": self.max_utilization,
            "average_waste_ratio": avg_waste_ratio,
            "min_waste_ratio": self.min_waste_ratio,
            "max_waste_ratio": self.max_waste_ratio,
            "average_bin_count": avg_bin_count,
            "bin_efficiency": bin_efficiency,
        }

        if avg_packing_time is not None:
            stats["average_packing_time"] = avg_packing_time

        return stats

    def print_aggregated_stats(self) -> None:
        """Print the aggregated metrics in a formatted way."""
        stats = self.get_aggregated_stats()

        if not stats:
            print("No metrics collected yet.")
            return

        print("\n=== Packing Metrics Summary ===")
        print(f"Total sequences packed: {stats['total_sequences']}")
        print(f"Total bins used: {stats['total_bins']}")
        print(
            f"Average bin utilization: {stats['average_utilization']:.4f} (min: {stats['min_utilization']:.4f}, max: {stats['max_utilization']:.4f})"
        )
        print(
            f"Average waste ratio: {stats['average_waste_ratio']:.4f} (min: {stats['min_waste_ratio']:.4f}, max: {stats['max_waste_ratio']:.4f})"
        )
        print(
            f"Bin efficiency (theoretical min bins / actual bins): {stats['bin_efficiency']:.4f}"
        )

        if "average_packing_time" in stats:
            print(f"Average packing time: {stats['average_packing_time']:.6f} seconds")

        print("===============================\n")

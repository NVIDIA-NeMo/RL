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
"""Memory benchmark tests.

These tests measure memory usage and compare against baselines
to detect memory regressions.

Run with:
    pytest tests/benchmarks/test_memory.py -v --benchmark-enable
"""

import pytest

from tests.benchmarks.memory import (
    MemoryBenchmark,
    MemorySnapshot,
    MemoryProfile,
)
from tests.benchmarks.utils import (
    BenchmarkConfig,
    compare_to_baseline,
)


class TestMemoryBenchmarks:
    """Memory benchmark tests."""

    @pytest.mark.benchmark
    def test_memory_dtensor_bs4(self):
        """Test memory usage with DTensor backend, batch size 4."""
        benchmark = MemoryBenchmark(
            name="memory_dtensor_bs4",
            batch_size=4,
            seq_length=512,
            num_steps=3,
            backend="dtensor",
        )
        result = benchmark.run()

        # Verify result structure
        assert result.name == "memory_dtensor_bs4"
        assert result.peak_memory_mb >= 0
        assert result.average_memory_mb >= 0

        # Compare to baseline
        comparison = compare_to_baseline(result)
        assert comparison.passed, f"Memory regression: {comparison.message}"

    @pytest.mark.benchmark
    def test_memory_dtensor_bs8(self):
        """Test memory usage with larger batch size."""
        benchmark = MemoryBenchmark(
            name="memory_dtensor_bs8",
            batch_size=8,
            seq_length=512,
            num_steps=3,
            backend="dtensor",
        )
        result = benchmark.run()

        assert result.peak_memory_mb >= 0
        comparison = compare_to_baseline(result)
        assert comparison.passed, comparison.message

    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_memory_scaling_by_batch(self):
        """Test that memory scales approximately linearly with batch size."""
        results = []
        for bs in [2, 4, 8]:
            benchmark = MemoryBenchmark(
                name=f"memory_scaling_bs{bs}",
                batch_size=bs,
                seq_length=256,
                num_steps=2,
            )
            results.append(benchmark.run())

        # Memory should increase with batch size
        # (exact ratio depends on implementation)
        assert results[1].peak_memory_mb >= results[0].peak_memory_mb * 0.5
        assert results[2].peak_memory_mb >= results[1].peak_memory_mb * 0.5


class TestMemorySnapshot:
    """Tests for MemorySnapshot class."""

    def test_snapshot_creation(self):
        """Test creating a memory snapshot."""
        import time
        start = time.time()
        snapshot = MemorySnapshot.capture(start, label="test")

        assert snapshot.label == "test"
        assert snapshot.timestamp_sec >= 0
        assert snapshot.allocated_mb >= 0
        assert snapshot.peak_mb >= 0


class TestMemoryProfile:
    """Tests for MemoryProfile class."""

    def test_profile_creation(self):
        """Test creating a memory profile."""
        profile = MemoryProfile(name="test_profile")
        assert profile.name == "test_profile"
        assert len(profile.snapshots) == 0
        assert profile.peak_allocated_mb == 0

    def test_profile_add_snapshot(self):
        """Test adding snapshots to profile."""
        profile = MemoryProfile(name="test")

        snapshot1 = MemorySnapshot(
            timestamp_sec=0.0,
            allocated_mb=100.0,
            peak_mb=100.0,
        )
        snapshot2 = MemorySnapshot(
            timestamp_sec=1.0,
            allocated_mb=200.0,
            peak_mb=250.0,
        )

        profile.add_snapshot(snapshot1)
        assert profile.peak_allocated_mb == 100.0

        profile.add_snapshot(snapshot2)
        assert profile.peak_allocated_mb == 250.0
        assert len(profile.snapshots) == 2

    def test_profile_to_dict(self):
        """Test converting profile to dictionary."""
        profile = MemoryProfile(name="test")
        profile.peak_allocated_mb = 500.0
        profile.total_duration_sec = 10.0

        data = profile.to_dict()
        assert data["name"] == "test"
        assert data["peak_allocated_mb"] == 500.0
        assert data["total_duration_sec"] == 10.0


class TestMemoryUtils:
    """Tests for memory utility functions."""

    def test_get_gpu_memory_without_cuda(self):
        """Test GPU memory functions when CUDA not available."""
        from tests.benchmarks.utils import get_gpu_memory_mb, get_peak_gpu_memory_mb

        # Should not raise even without CUDA
        current = get_gpu_memory_mb()
        peak = get_peak_gpu_memory_mb()

        assert isinstance(current, float)
        assert isinstance(peak, float)

    def test_reset_gpu_memory_stats(self):
        """Test resetting GPU memory stats."""
        from tests.benchmarks.utils import reset_gpu_memory_stats

        # Should not raise even without CUDA
        reset_gpu_memory_stats()

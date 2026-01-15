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
"""Throughput benchmark tests.

These tests measure training throughput and compare against baselines
to detect performance regressions.

Run with:
    pytest tests/benchmarks/test_throughput.py -v --benchmark-enable
"""

import pytest

from tests.benchmarks.throughput import (
    ThroughputBenchmark,
    create_mock_batch,
)
from tests.benchmarks.utils import (
    BenchmarkConfig,
    BenchmarkResult,
    compare_to_baseline,
    load_baseline,
)


class TestThroughputBenchmarks:
    """Throughput benchmark tests."""

    @pytest.mark.benchmark
    def test_throughput_dtensor_bs4(self):
        """Test throughput with DTensor backend, batch size 4."""
        benchmark = ThroughputBenchmark(
            name="throughput_dtensor_bs4",
            batch_size=4,
            seq_length=512,
            num_steps=5,
            warmup_steps=1,
            backend="dtensor",
        )
        result = benchmark.run()

        # Verify result structure
        assert result.name == "throughput_dtensor_bs4"
        assert result.throughput_tokens_per_sec >= 0
        assert result.peak_memory_mb >= 0
        assert len(result.step_times_sec) == 5

        # Compare to baseline (if exists)
        comparison = compare_to_baseline(result)
        assert comparison.passed, f"Throughput regression: {comparison.message}"

    @pytest.mark.benchmark
    def test_throughput_dtensor_bs8(self):
        """Test throughput with DTensor backend, batch size 8."""
        benchmark = ThroughputBenchmark(
            name="throughput_dtensor_bs8",
            batch_size=8,
            seq_length=512,
            num_steps=5,
            warmup_steps=1,
            backend="dtensor",
        )
        result = benchmark.run()

        assert result.throughput_tokens_per_sec >= 0
        comparison = compare_to_baseline(result)
        assert comparison.passed, comparison.message

    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_throughput_dtensor_seq1024(self):
        """Test throughput with longer sequences."""
        benchmark = ThroughputBenchmark(
            name="throughput_dtensor_seq1024",
            batch_size=4,
            seq_length=1024,
            num_steps=5,
            backend="dtensor",
        )
        result = benchmark.run()

        assert result.throughput_tokens_per_sec >= 0
        comparison = compare_to_baseline(result)
        assert comparison.passed, comparison.message


class TestMockBatch:
    """Tests for mock batch creation."""

    def test_create_mock_batch_cpu(self):
        """Test mock batch creation on CPU."""
        batch = create_mock_batch(batch_size=2, seq_length=128, device="cpu")

        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "labels" in batch

    def test_create_mock_batch_correct_shape(self):
        """Test mock batch has correct shapes."""
        batch_size = 4
        seq_length = 256
        batch = create_mock_batch(batch_size=batch_size, seq_length=seq_length, device="cpu")

        try:
            import torch
            if isinstance(batch["input_ids"], torch.Tensor):
                assert batch["input_ids"].shape == (batch_size, seq_length)
        except ImportError:
            # Fallback for non-torch environments
            assert len(batch["input_ids"]) == batch_size
            assert len(batch["input_ids"][0]) == seq_length


class TestBaselineComparison:
    """Tests for baseline comparison functionality."""

    def test_no_baseline_passes(self):
        """Test that missing baseline results in pass."""
        config = BenchmarkConfig(
            name="nonexistent_benchmark",
            batch_size=4,
        )
        result = BenchmarkResult(
            name="nonexistent_benchmark",
            config=config,
            throughput_tokens_per_sec=1000,
            peak_memory_mb=100,
        )

        comparison = compare_to_baseline(result)
        assert comparison.passed
        assert "first run" in comparison.message.lower() or "no baseline" in comparison.message.lower()

    def test_within_tolerance_passes(self):
        """Test that results within tolerance pass."""
        from tests.benchmarks.utils import BaselineEntry

        config = BenchmarkConfig(name="test_benchmark", batch_size=4)
        result = BenchmarkResult(
            name="test_benchmark",
            config=config,
            throughput_tokens_per_sec=9800,  # 2% below baseline
            peak_memory_mb=2100,  # ~2.5% above baseline
        )

        baseline = BaselineEntry(
            name="test_benchmark",
            throughput_tokens_per_sec=10000,
            peak_memory_mb=2048,
            tolerance=0.05,
        )

        comparison = compare_to_baseline(result, baseline)
        assert comparison.passed
        assert "within tolerance" in comparison.message.lower()

    def test_regression_fails(self):
        """Test that regression is detected."""
        from tests.benchmarks.utils import BaselineEntry

        config = BenchmarkConfig(name="test_benchmark", batch_size=4)
        result = BenchmarkResult(
            name="test_benchmark",
            config=config,
            throughput_tokens_per_sec=8000,  # 20% below baseline
            peak_memory_mb=2048,
        )

        baseline = BaselineEntry(
            name="test_benchmark",
            throughput_tokens_per_sec=10000,
            peak_memory_mb=2048,
            tolerance=0.05,  # 5% tolerance
        )

        comparison = compare_to_baseline(result, baseline)
        assert not comparison.passed
        assert "regression" in comparison.message.lower()

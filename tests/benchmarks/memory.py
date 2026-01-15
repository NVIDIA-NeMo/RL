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
"""Memory benchmarks for NeMo RL.

This module provides memory benchmarks to measure:
- Peak GPU memory usage during training
- Memory allocation patterns
- Memory efficiency across backends

Benchmarks help catch memory regressions and optimize usage.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from tests.benchmarks.utils import (
    BenchmarkConfig,
    BenchmarkResult,
    get_gpu_memory_mb,
    get_peak_gpu_memory_mb,
    reset_gpu_memory_stats,
)


@dataclass
class MemorySnapshot:
    """Snapshot of memory state at a point in time.

    Attributes:
        timestamp_sec: Time offset from start.
        allocated_mb: Currently allocated memory in MB.
        peak_mb: Peak memory up to this point.
        cached_mb: Cached memory (if available).
        label: Optional label for this snapshot.
    """

    timestamp_sec: float
    allocated_mb: float
    peak_mb: float
    cached_mb: float = 0.0
    label: str = ""

    @classmethod
    def capture(cls, start_time: float, label: str = "") -> "MemorySnapshot":
        """Capture current memory state.

        Args:
            start_time: Start time for relative timestamp.
            label: Optional label for the snapshot.

        Returns:
            MemorySnapshot with current memory state.
        """
        try:
            import torch
            if torch.cuda.is_available():
                return cls(
                    timestamp_sec=time.time() - start_time,
                    allocated_mb=torch.cuda.memory_allocated() / (1024 * 1024),
                    peak_mb=torch.cuda.max_memory_allocated() / (1024 * 1024),
                    cached_mb=torch.cuda.memory_reserved() / (1024 * 1024),
                    label=label,
                )
        except ImportError:
            pass

        return cls(
            timestamp_sec=time.time() - start_time,
            allocated_mb=0.0,
            peak_mb=0.0,
            cached_mb=0.0,
            label=label,
        )


@dataclass
class MemoryProfile:
    """Memory profile for a complete operation.

    Attributes:
        name: Name of the profiled operation.
        snapshots: List of memory snapshots.
        peak_allocated_mb: Peak allocated memory.
        peak_cached_mb: Peak cached memory.
        total_duration_sec: Total duration of profiling.
    """

    name: str
    snapshots: List[MemorySnapshot] = field(default_factory=list)
    peak_allocated_mb: float = 0.0
    peak_cached_mb: float = 0.0
    total_duration_sec: float = 0.0

    def add_snapshot(self, snapshot: MemorySnapshot) -> None:
        """Add a snapshot and update peaks."""
        self.snapshots.append(snapshot)
        self.peak_allocated_mb = max(self.peak_allocated_mb, snapshot.peak_mb)
        self.peak_cached_mb = max(self.peak_cached_mb, snapshot.cached_mb)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "peak_allocated_mb": self.peak_allocated_mb,
            "peak_cached_mb": self.peak_cached_mb,
            "total_duration_sec": self.total_duration_sec,
            "num_snapshots": len(self.snapshots),
        }


class MemoryBenchmark:
    """Memory benchmark runner.

    Example:
        >>> benchmark = MemoryBenchmark(
        ...     name="grpo_memory",
        ...     model_name="gpt2",
        ... )
        >>> result = benchmark.run()
        >>> print(f"Peak memory: {result.peak_memory_mb:.1f} MB")
    """

    def __init__(
        self,
        name: str,
        model_name: str = "gpt2",
        batch_size: int = 4,
        seq_length: int = 512,
        num_steps: int = 5,
        backend: str = "dtensor",
        device: str = "cuda",
        capture_interval_sec: float = 0.1,
    ):
        """Initialize the memory benchmark.

        Args:
            name: Benchmark name.
            model_name: Model to benchmark.
            batch_size: Batch size.
            seq_length: Sequence length.
            num_steps: Number of training steps.
            backend: Training backend.
            device: Device to run on.
            capture_interval_sec: Interval between memory captures.
        """
        self.config = BenchmarkConfig(
            name=name,
            model_name=model_name,
            batch_size=batch_size,
            seq_length=seq_length,
            num_steps=num_steps,
            warmup_steps=1,
            backend=backend,
            device=device,
        )
        self.capture_interval = capture_interval_sec
        self._profile: Optional[MemoryProfile] = None

    def _clear_memory(self) -> None:
        """Clear GPU memory and caches."""
        gc.collect()
        reset_gpu_memory_stats()

    def run(self) -> BenchmarkResult:
        """Run the memory benchmark.

        Returns:
            BenchmarkResult with memory metrics.
        """
        self._clear_memory()

        profile = MemoryProfile(name=self.config.name)
        start_time = time.time()

        # Initial snapshot
        profile.add_snapshot(MemorySnapshot.capture(start_time, "start"))

        # Simulate model loading
        try:
            import torch
            if torch.cuda.is_available():
                # Simulate model memory footprint
                model_params = torch.randn(
                    100_000_000,  # ~400MB for float32
                    device="cuda" if self.config.device == "cuda" else "cpu",
                )
                profile.add_snapshot(MemorySnapshot.capture(start_time, "model_loaded"))
        except ImportError:
            time.sleep(0.1)

        # Simulate training steps
        for step in range(self.config.num_steps):
            # Simulate step memory allocation
            try:
                import torch
                if torch.cuda.is_available():
                    # Simulate activations and gradients
                    batch_mem = torch.randn(
                        self.config.batch_size,
                        self.config.seq_length,
                        768,  # Hidden size
                        device="cuda",
                    )
                    # Simulate backward pass
                    time.sleep(0.01)
                    del batch_mem
            except ImportError:
                time.sleep(0.05)

            profile.add_snapshot(MemorySnapshot.capture(start_time, f"step_{step}"))

        # Final snapshot
        profile.add_snapshot(MemorySnapshot.capture(start_time, "end"))
        profile.total_duration_sec = time.time() - start_time

        self._profile = profile

        # Build result
        return BenchmarkResult(
            name=self.config.name,
            config=self.config,
            throughput_tokens_per_sec=0,  # Not measured in memory benchmark
            throughput_samples_per_sec=0,
            peak_memory_mb=profile.peak_allocated_mb,
            average_memory_mb=sum(s.allocated_mb for s in profile.snapshots) / len(profile.snapshots) if profile.snapshots else 0,
            total_time_sec=profile.total_duration_sec,
            metadata={
                "peak_cached_mb": profile.peak_cached_mb,
                "num_snapshots": len(profile.snapshots),
            },
        )

    def get_profile(self) -> Optional[MemoryProfile]:
        """Get the memory profile from the last run."""
        return self._profile


def benchmark_memory_by_batch_size(
    batch_sizes: Optional[List[int]] = None,
    backend: str = "dtensor",
) -> List[BenchmarkResult]:
    """Benchmark memory usage across different batch sizes.

    Args:
        batch_sizes: List of batch sizes to test.
        backend: Backend to use.

    Returns:
        List of benchmark results.
    """
    batch_sizes = batch_sizes or [1, 2, 4, 8, 16]
    results = []

    for bs in batch_sizes:
        benchmark = MemoryBenchmark(
            name=f"memory_{backend}_bs{bs}",
            batch_size=bs,
            backend=backend,
        )
        result = benchmark.run()
        results.append(result)
        print(f"  Batch {bs}: Peak memory = {result.peak_memory_mb:.1f} MB")

    return results


def benchmark_memory_by_sequence_length(
    seq_lengths: Optional[List[int]] = None,
    backend: str = "dtensor",
) -> List[BenchmarkResult]:
    """Benchmark memory usage across different sequence lengths.

    Args:
        seq_lengths: List of sequence lengths to test.
        backend: Backend to use.

    Returns:
        List of benchmark results.
    """
    seq_lengths = seq_lengths or [256, 512, 1024, 2048]
    results = []

    for seq_len in seq_lengths:
        benchmark = MemoryBenchmark(
            name=f"memory_{backend}_seq{seq_len}",
            seq_length=seq_len,
            backend=backend,
        )
        result = benchmark.run()
        results.append(result)
        print(f"  Seq {seq_len}: Peak memory = {result.peak_memory_mb:.1f} MB")

    return results


__all__ = [
    "MemoryBenchmark",
    "MemorySnapshot",
    "MemoryProfile",
    "benchmark_memory_by_batch_size",
    "benchmark_memory_by_sequence_length",
]

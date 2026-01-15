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
"""Benchmark utilities for performance testing.

This module provides common utilities for running benchmarks:
- Configuration management
- Result collection and comparison
- Baseline management
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Default paths
BENCHMARK_DIR = Path(__file__).parent
BASELINE_FILE = BENCHMARK_DIR / "baselines.json"


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run.

    Attributes:
        name: Name of the benchmark.
        model_name: Model to use (e.g., "gpt2" for small tests).
        batch_size: Batch size for training.
        seq_length: Sequence length.
        num_steps: Number of steps to run.
        warmup_steps: Number of warmup steps (excluded from timing).
        backend: Training backend ("dtensor" or "megatron").
        device: Device to run on ("cuda" or "cpu").
    """

    name: str
    model_name: str = "gpt2"
    batch_size: int = 4
    seq_length: int = 512
    num_steps: int = 10
    warmup_steps: int = 2
    backend: str = "dtensor"
    device: str = "cuda"

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)


@dataclass
class BenchmarkResult:
    """Results from a benchmark run.

    Attributes:
        name: Name of the benchmark.
        config: Configuration used for the benchmark.
        throughput_tokens_per_sec: Tokens processed per second.
        throughput_samples_per_sec: Samples processed per second.
        peak_memory_mb: Peak GPU memory usage in MB.
        average_memory_mb: Average GPU memory usage in MB.
        total_time_sec: Total benchmark time in seconds.
        step_times_sec: List of individual step times.
        timestamp: Timestamp when benchmark was run.
        metadata: Additional metadata.
    """

    name: str
    config: BenchmarkConfig
    throughput_tokens_per_sec: float = 0.0
    throughput_samples_per_sec: float = 0.0
    peak_memory_mb: float = 0.0
    average_memory_mb: float = 0.0
    total_time_sec: float = 0.0
    step_times_sec: List[float] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "name": self.name,
            "config": self.config.to_dict(),
            "throughput_tokens_per_sec": self.throughput_tokens_per_sec,
            "throughput_samples_per_sec": self.throughput_samples_per_sec,
            "peak_memory_mb": self.peak_memory_mb,
            "average_memory_mb": self.average_memory_mb,
            "total_time_sec": self.total_time_sec,
            "step_times_sec": self.step_times_sec,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkResult":
        """Create result from dictionary."""
        config_data = data.pop("config", {})
        config = BenchmarkConfig(**config_data) if config_data else BenchmarkConfig(name=data.get("name", ""))
        return cls(config=config, **data)


@dataclass
class BaselineEntry:
    """A baseline entry for comparison.

    Attributes:
        name: Benchmark name.
        throughput_tokens_per_sec: Baseline throughput.
        peak_memory_mb: Baseline peak memory.
        tolerance: Acceptable regression percentage (e.g., 0.05 = 5%).
        version: Version when baseline was recorded.
        timestamp: When baseline was recorded.
    """

    name: str
    throughput_tokens_per_sec: float
    peak_memory_mb: float
    tolerance: float = 0.05
    version: str = "1.0"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


def load_baseline(name: str, baseline_file: Optional[Path] = None) -> Optional[BaselineEntry]:
    """Load a baseline entry by name.

    Args:
        name: Name of the benchmark to load baseline for.
        baseline_file: Optional path to baseline file.

    Returns:
        BaselineEntry if found, None otherwise.
    """
    baseline_file = baseline_file or BASELINE_FILE
    if not baseline_file.exists():
        return None

    with open(baseline_file) as f:
        baselines = json.load(f)

    if name in baselines:
        return BaselineEntry(**baselines[name])
    return None


def save_baseline(entry: BaselineEntry, baseline_file: Optional[Path] = None) -> None:
    """Save a baseline entry.

    Args:
        entry: Baseline entry to save.
        baseline_file: Optional path to baseline file.
    """
    baseline_file = baseline_file or BASELINE_FILE
    baselines = {}

    if baseline_file.exists():
        with open(baseline_file) as f:
            baselines = json.load(f)

    baselines[entry.name] = entry.to_dict()

    with open(baseline_file, "w") as f:
        json.dump(baselines, f, indent=2)


@dataclass
class ComparisonResult:
    """Result of comparing a benchmark to its baseline.

    Attributes:
        passed: Whether the benchmark passed (within tolerance).
        throughput_ratio: Current / baseline throughput.
        memory_ratio: Current / baseline memory.
        throughput_regression_pct: Percentage throughput regression.
        memory_regression_pct: Percentage memory regression.
        message: Human-readable message.
    """

    passed: bool
    throughput_ratio: float
    memory_ratio: float
    throughput_regression_pct: float
    memory_regression_pct: float
    message: str


def compare_to_baseline(
    result: BenchmarkResult,
    baseline: Optional[BaselineEntry] = None,
    tolerance: float = 0.05,
) -> ComparisonResult:
    """Compare benchmark result to baseline.

    Args:
        result: The benchmark result to compare.
        baseline: Baseline entry to compare against. If None, will try to load.
        tolerance: Acceptable regression percentage (default 5%).

    Returns:
        ComparisonResult with pass/fail status and details.
    """
    if baseline is None:
        baseline = load_baseline(result.name)

    if baseline is None:
        return ComparisonResult(
            passed=True,  # No baseline = pass (first run)
            throughput_ratio=1.0,
            memory_ratio=1.0,
            throughput_regression_pct=0.0,
            memory_regression_pct=0.0,
            message="No baseline found - treating as first run",
        )

    # Calculate ratios
    throughput_ratio = result.throughput_tokens_per_sec / baseline.throughput_tokens_per_sec if baseline.throughput_tokens_per_sec > 0 else 1.0
    memory_ratio = result.peak_memory_mb / baseline.peak_memory_mb if baseline.peak_memory_mb > 0 else 1.0

    # Calculate regression percentages
    throughput_regression = (1.0 - throughput_ratio) * 100  # Positive = regression
    memory_regression = (memory_ratio - 1.0) * 100  # Positive = regression

    # Use baseline tolerance if specified, otherwise use parameter
    tolerance_pct = baseline.tolerance if baseline.tolerance else tolerance

    # Check if within tolerance
    throughput_ok = throughput_regression <= tolerance_pct * 100
    memory_ok = memory_regression <= tolerance_pct * 100
    passed = throughput_ok and memory_ok

    # Build message
    messages = []
    if not throughput_ok:
        messages.append(f"Throughput regression: {throughput_regression:.1f}% (tolerance: {tolerance_pct*100:.1f}%)")
    if not memory_ok:
        messages.append(f"Memory regression: {memory_regression:.1f}% (tolerance: {tolerance_pct*100:.1f}%)")
    if passed:
        messages.append("All metrics within tolerance")

    return ComparisonResult(
        passed=passed,
        throughput_ratio=throughput_ratio,
        memory_ratio=memory_ratio,
        throughput_regression_pct=throughput_regression,
        memory_regression_pct=memory_regression,
        message="; ".join(messages),
    )


class BenchmarkTimer:
    """Context manager for timing benchmark sections.

    Example:
        >>> with BenchmarkTimer() as timer:
        ...     # Do work
        ...     pass
        >>> print(f"Elapsed: {timer.elapsed:.3f}s")
    """

    def __init__(self):
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.elapsed: float = 0.0

    def __enter__(self) -> "BenchmarkTimer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self.end_time = time.perf_counter()
        if self.start_time is not None:
            self.elapsed = self.end_time - self.start_time


def get_gpu_memory_mb() -> float:
    """Get current GPU memory usage in MB.

    Returns:
        Memory usage in MB, or 0 if CUDA not available.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
    except ImportError:
        pass
    return 0.0


def get_peak_gpu_memory_mb() -> float:
    """Get peak GPU memory usage in MB.

    Returns:
        Peak memory usage in MB, or 0 if CUDA not available.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
    except ImportError:
        pass
    return 0.0


def reset_gpu_memory_stats() -> None:
    """Reset GPU memory statistics."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
    except ImportError:
        pass


__all__ = [
    "BenchmarkConfig",
    "BenchmarkResult",
    "BaselineEntry",
    "ComparisonResult",
    "BenchmarkTimer",
    "load_baseline",
    "save_baseline",
    "compare_to_baseline",
    "get_gpu_memory_mb",
    "get_peak_gpu_memory_mb",
    "reset_gpu_memory_stats",
]

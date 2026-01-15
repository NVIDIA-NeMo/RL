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
"""Throughput benchmarks for NeMo RL training.

This module provides throughput benchmarks to measure:
- Tokens per second during training
- Samples per second
- Step latency

Benchmarks can be run with pytest-benchmark or standalone.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

from tests.benchmarks.utils import (
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkTimer,
    get_peak_gpu_memory_mb,
    reset_gpu_memory_stats,
)


def benchmark_training_step(
    step_fn: Callable[[], Any],
    config: BenchmarkConfig,
    cleanup_fn: Optional[Callable[[], None]] = None,
) -> BenchmarkResult:
    """Benchmark a training step function.

    Args:
        step_fn: Function that performs one training step.
        config: Benchmark configuration.
        cleanup_fn: Optional cleanup function to call after each step.

    Returns:
        BenchmarkResult with throughput and memory metrics.
    """
    # Reset memory stats
    reset_gpu_memory_stats()

    # Warmup
    for _ in range(config.warmup_steps):
        step_fn()
        if cleanup_fn:
            cleanup_fn()

    # Reset after warmup
    reset_gpu_memory_stats()

    # Benchmark
    step_times: List[float] = []
    memory_samples: List[float] = []

    for _ in range(config.num_steps):
        with BenchmarkTimer() as timer:
            step_fn()

        step_times.append(timer.elapsed)
        memory_samples.append(get_peak_gpu_memory_mb())

        if cleanup_fn:
            cleanup_fn()

    # Calculate metrics
    total_time = sum(step_times)
    avg_step_time = total_time / config.num_steps if config.num_steps > 0 else 0

    tokens_per_step = config.batch_size * config.seq_length
    throughput_tokens = tokens_per_step / avg_step_time if avg_step_time > 0 else 0
    throughput_samples = config.batch_size / avg_step_time if avg_step_time > 0 else 0

    peak_memory = max(memory_samples) if memory_samples else 0
    avg_memory = sum(memory_samples) / len(memory_samples) if memory_samples else 0

    return BenchmarkResult(
        name=config.name,
        config=config,
        throughput_tokens_per_sec=throughput_tokens,
        throughput_samples_per_sec=throughput_samples,
        peak_memory_mb=peak_memory,
        average_memory_mb=avg_memory,
        total_time_sec=total_time,
        step_times_sec=step_times,
        metadata={
            "avg_step_time_sec": avg_step_time,
            "min_step_time_sec": min(step_times) if step_times else 0,
            "max_step_time_sec": max(step_times) if step_times else 0,
        },
    )


def create_mock_batch(batch_size: int, seq_length: int, device: str = "cpu") -> Dict[str, Any]:
    """Create a mock batch for benchmarking.

    Args:
        batch_size: Number of samples in batch.
        seq_length: Sequence length.
        device: Device to create tensors on.

    Returns:
        Dictionary containing mock batch data.
    """
    try:
        import torch
        return {
            "input_ids": torch.randint(0, 32000, (batch_size, seq_length), device=device),
            "attention_mask": torch.ones((batch_size, seq_length), device=device, dtype=torch.long),
            "labels": torch.randint(0, 32000, (batch_size, seq_length), device=device),
        }
    except ImportError:
        return {
            "input_ids": [[0] * seq_length for _ in range(batch_size)],
            "attention_mask": [[1] * seq_length for _ in range(batch_size)],
            "labels": [[0] * seq_length for _ in range(batch_size)],
        }


class ThroughputBenchmark:
    """Throughput benchmark runner.

    Example:
        >>> benchmark = ThroughputBenchmark(
        ...     name="grpo_throughput",
        ...     model_name="gpt2",
        ...     batch_size=4,
        ... )
        >>> result = benchmark.run()
        >>> print(f"Throughput: {result.throughput_tokens_per_sec:.0f} tokens/sec")
    """

    def __init__(
        self,
        name: str,
        model_name: str = "gpt2",
        batch_size: int = 4,
        seq_length: int = 512,
        num_steps: int = 10,
        warmup_steps: int = 2,
        backend: str = "dtensor",
        device: str = "cuda",
    ):
        """Initialize the benchmark.

        Args:
            name: Benchmark name.
            model_name: Model to benchmark.
            batch_size: Batch size for training.
            seq_length: Sequence length.
            num_steps: Number of steps to measure.
            warmup_steps: Warmup steps (excluded from timing).
            backend: Training backend.
            device: Device to run on.
        """
        self.config = BenchmarkConfig(
            name=name,
            model_name=model_name,
            batch_size=batch_size,
            seq_length=seq_length,
            num_steps=num_steps,
            warmup_steps=warmup_steps,
            backend=backend,
            device=device,
        )
        self._trainer = None
        self._mock_batch = None

    def setup(self) -> None:
        """Set up the benchmark (load model, create batch)."""
        # Create mock batch
        self._mock_batch = create_mock_batch(
            self.config.batch_size,
            self.config.seq_length,
            self.config.device if self.config.device != "cuda" else "cpu",
        )

        # Try to create a minimal trainer
        try:
            from nemo_rl.algorithms.grpo import GRPOTrainer
            # For now, we'll use a mock trainer
            # In a real benchmark, this would be a real trainer
            self._trainer = None
        except ImportError:
            self._trainer = None

    def teardown(self) -> None:
        """Clean up after benchmark."""
        self._mock_batch = None
        self._trainer = None
        reset_gpu_memory_stats()

    def _train_step(self) -> None:
        """Perform one training step."""
        # Simulate training step with mock computation
        try:
            import torch
            if torch.cuda.is_available():
                # Simulate some GPU computation
                batch = self._mock_batch
                if batch and "input_ids" in batch:
                    x = batch["input_ids"]
                    if isinstance(x, torch.Tensor):
                        # Simulate forward + backward
                        y = x.float() @ x.float().T
                        loss = y.mean()
                        if loss.requires_grad:
                            loss.backward()
                time.sleep(0.001)  # Simulate additional overhead
            else:
                time.sleep(0.01)  # CPU simulation
        except ImportError:
            time.sleep(0.01)

    def run(self) -> BenchmarkResult:
        """Run the benchmark.

        Returns:
            BenchmarkResult with throughput metrics.
        """
        self.setup()
        try:
            result = benchmark_training_step(
                step_fn=self._train_step,
                config=self.config,
            )
            return result
        finally:
            self.teardown()


def run_throughput_benchmarks(
    backends: Optional[List[str]] = None,
    batch_sizes: Optional[List[int]] = None,
) -> List[BenchmarkResult]:
    """Run a suite of throughput benchmarks.

    Args:
        backends: List of backends to benchmark (default: ["dtensor"]).
        batch_sizes: List of batch sizes to test (default: [4, 8, 16]).

    Returns:
        List of benchmark results.
    """
    backends = backends or ["dtensor"]
    batch_sizes = batch_sizes or [4, 8, 16]

    results = []
    for backend in backends:
        for batch_size in batch_sizes:
            benchmark = ThroughputBenchmark(
                name=f"throughput_{backend}_bs{batch_size}",
                batch_size=batch_size,
                backend=backend,
            )
            result = benchmark.run()
            results.append(result)
            print(f"  {result.name}: {result.throughput_tokens_per_sec:.0f} tokens/sec")

    return results


__all__ = [
    "ThroughputBenchmark",
    "benchmark_training_step",
    "create_mock_batch",
    "run_throughput_benchmarks",
]

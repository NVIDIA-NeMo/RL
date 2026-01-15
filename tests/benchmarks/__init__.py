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
"""Performance Benchmarking Suite for NeMo RL.

This module provides comprehensive benchmarking tools to measure:
- Training throughput (tokens/second, samples/second)
- Memory usage (peak, average) across backends
- End-to-end training performance

The benchmarks can be run standalone or integrated into CI to catch
performance regressions.

Usage:
    # Run all benchmarks
    pytest tests/benchmarks/ -v --benchmark-enable

    # Run specific benchmark
    pytest tests/benchmarks/test_throughput.py -v --benchmark-enable

    # Save benchmark results
    pytest tests/benchmarks/ -v --benchmark-enable --benchmark-json=results.json
"""

from tests.benchmarks.utils import (
    BenchmarkConfig,
    BenchmarkResult,
    compare_to_baseline,
    load_baseline,
    save_baseline,
)

__all__ = [
    "BenchmarkConfig",
    "BenchmarkResult",
    "compare_to_baseline",
    "load_baseline",
    "save_baseline",
]

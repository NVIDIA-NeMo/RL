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
"""Run performance benchmarks and check for regressions.

This script can be used for CI integration to run benchmarks and
fail the build if performance regresses beyond the tolerance threshold.

Usage:
    python -m tests.benchmarks.run_benchmarks
    python -m tests.benchmarks.run_benchmarks --save-results results.json
    python -m tests.benchmarks.run_benchmarks --tolerance 0.10
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tests.benchmarks.throughput import ThroughputBenchmark
from tests.benchmarks.memory import MemoryBenchmark
from tests.benchmarks.utils import (
    BenchmarkResult,
    BaselineEntry,
    compare_to_baseline,
    load_baseline,
)


def run_throughput_benchmarks() -> List[BenchmarkResult]:
    """Run all throughput benchmarks."""
    benchmarks = [
        ThroughputBenchmark(
            name="throughput_dtensor_bs4",
            batch_size=4,
            seq_length=512,
            num_steps=5,
            warmup_steps=1,
            backend="dtensor",
        ),
        ThroughputBenchmark(
            name="throughput_dtensor_bs8",
            batch_size=8,
            seq_length=512,
            num_steps=5,
            warmup_steps=1,
            backend="dtensor",
        ),
    ]
    
    results = []
    for benchmark in benchmarks:
        print(f"  Running {benchmark.config.name}...")
        result = benchmark.run()
        results.append(result)
        print(f"    Throughput: {result.throughput_tokens_per_sec:.0f} tokens/sec")
    
    return results


def run_memory_benchmarks() -> List[BenchmarkResult]:
    """Run all memory benchmarks."""
    benchmarks = [
        MemoryBenchmark(
            name="memory_dtensor_bs4",
            batch_size=4,
            seq_length=512,
            num_steps=3,
            backend="dtensor",
        ),
        MemoryBenchmark(
            name="memory_dtensor_bs8",
            batch_size=8,
            seq_length=512,
            num_steps=3,
            backend="dtensor",
        ),
    ]
    
    results = []
    for benchmark in benchmarks:
        print(f"  Running {benchmark.config.name}...")
        result = benchmark.run()
        results.append(result)
        print(f"    Peak memory: {result.peak_memory_mb:.1f} MB")
    
    return results


def check_regressions(
    results: List[BenchmarkResult],
    tolerance: float = 0.05,
) -> bool:
    """Check all results for regressions.
    
    Returns:
        True if all benchmarks pass, False if any regression detected.
    """
    all_passed = True
    
    print("\nRegression Check:")
    print("-" * 60)
    
    for result in results:
        comparison = compare_to_baseline(result, tolerance=tolerance)
        
        status = "PASS" if comparison.passed else "FAIL"
        print(f"  [{status}] {result.name}: {comparison.message}")
        
        if not comparison.passed:
            all_passed = False
            if comparison.throughput_regression_pct > tolerance * 100:
                print(f"         Throughput: {comparison.throughput_regression_pct:.1f}% regression")
            if comparison.memory_regression_pct > tolerance * 100:
                print(f"         Memory: {comparison.memory_regression_pct:.1f}% regression")
    
    return all_passed


def save_results(results: List[BenchmarkResult], output_file: Path) -> None:
    """Save benchmark results to JSON file."""
    data = {
        "timestamp": datetime.now().isoformat(),
        "results": [r.to_dict() for r in results],
    }
    
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run NeMo RL performance benchmarks"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="Regression tolerance (default: 0.05 = 5%%)",
    )
    parser.add_argument(
        "--save-results",
        type=str,
        default=None,
        help="Path to save results JSON",
    )
    parser.add_argument(
        "--throughput-only",
        action="store_true",
        help="Run only throughput benchmarks",
    )
    parser.add_argument(
        "--memory-only",
        action="store_true",
        help="Run only memory benchmarks",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("NeMo RL Performance Benchmarks")
    print("=" * 60)
    print(f"Tolerance: {args.tolerance * 100:.0f}%")
    print()
    
    results: List[BenchmarkResult] = []
    
    # Run throughput benchmarks
    if not args.memory_only:
        print("Throughput Benchmarks:")
        print("-" * 60)
        results.extend(run_throughput_benchmarks())
        print()
    
    # Run memory benchmarks
    if not args.throughput_only:
        print("Memory Benchmarks:")
        print("-" * 60)
        results.extend(run_memory_benchmarks())
        print()
    
    # Check for regressions
    all_passed = check_regressions(results, tolerance=args.tolerance)
    
    # Save results if requested
    if args.save_results:
        save_results(results, Path(args.save_results))
    
    print()
    print("=" * 60)
    if all_passed:
        print("BENCHMARK RESULT: PASSED")
        print("All benchmarks within tolerance - no regressions detected")
        return 0
    else:
        print("BENCHMARK RESULT: FAILED")
        print("Performance regression detected - please investigate")
        return 1


if __name__ == "__main__":
    sys.exit(main())

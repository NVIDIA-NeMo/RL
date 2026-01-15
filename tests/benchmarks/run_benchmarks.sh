# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

#!/bin/bash
# Performance benchmark runner for NeMo RL
# This script runs throughput and memory benchmarks and compares against baselines
# to detect performance regressions.
#
# Usage:
#   ./run_benchmarks.sh                    # Run all benchmarks
#   ./run_benchmarks.sh --throughput-only  # Run only throughput benchmarks
#   ./run_benchmarks.sh --memory-only      # Run only memory benchmarks
#   ./run_benchmarks.sh --save-results     # Save results to JSON

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT=$(realpath ${SCRIPT_DIR}/../..)

# Default values
RUN_THROUGHPUT=true
RUN_MEMORY=true
SAVE_RESULTS=false
RESULTS_FILE="${PROJECT_ROOT}/benchmark_results.json"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --throughput-only)
            RUN_MEMORY=false
            shift
            ;;
        --memory-only)
            RUN_THROUGHPUT=false
            shift
            ;;
        --save-results)
            SAVE_RESULTS=true
            shift
            ;;
        --results-file)
            RESULTS_FILE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --throughput-only   Run only throughput benchmarks"
            echo "  --memory-only       Run only memory benchmarks"
            echo "  --save-results      Save results to JSON file"
            echo "  --results-file PATH Path to save results (default: benchmark_results.json)"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

cd ${PROJECT_ROOT}

echo "=============================================="
echo "NeMo RL Performance Benchmarks"
echo "=============================================="
echo ""

# Build pytest command
PYTEST_ARGS="-v"
if [[ "$SAVE_RESULTS" == "true" ]]; then
    PYTEST_ARGS="$PYTEST_ARGS --benchmark-json=$RESULTS_FILE"
fi

# Run benchmarks
BENCHMARK_PASSED=true

if [[ "$RUN_THROUGHPUT" == "true" ]]; then
    echo "Running throughput benchmarks..."
    echo "----------------------------------------------"
    if ! python -m pytest tests/benchmarks/test_throughput.py $PYTEST_ARGS; then
        echo "ERROR: Throughput benchmarks failed!"
        BENCHMARK_PASSED=false
    fi
    echo ""
fi

if [[ "$RUN_MEMORY" == "true" ]]; then
    echo "Running memory benchmarks..."
    echo "----------------------------------------------"
    if ! python -m pytest tests/benchmarks/test_memory.py $PYTEST_ARGS; then
        echo "ERROR: Memory benchmarks failed!"
        BENCHMARK_PASSED=false
    fi
    echo ""
fi

echo "=============================================="
if [[ "$BENCHMARK_PASSED" == "true" ]]; then
    echo "All benchmarks PASSED - No regressions detected"
    exit 0
else
    echo "Benchmark FAILED - Regressions detected!"
    echo "Please check the output above for details."
    exit 1
fi

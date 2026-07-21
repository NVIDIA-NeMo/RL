#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

# Real-engine validation of background prefill on Nemotron 3 Super. Set
# BENCH_MAX_CACHE_PAGES=0 for the legacy uncapped control.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
export REPO_ROOT

export MODEL_PATH="${MODEL_PATH:-nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16}"
export VLLM_TP="${VLLM_TP:-4}"
export BENCH_GPU_MEMORY_UTILIZATION="${BENCH_GPU_MEMORY_UTILIZATION:-0.9}"
export ACCOUNT="${ACCOUNT:-nemotron_sw_post}"
export PARTITION="${PARTITION:-interactive}"
export TIME_LIMIT="${TIME_LIMIT:-00:45:00}"
export BENCH_CACHE_PAGE_SIZE_TOKENS="${BENCH_CACHE_PAGE_SIZE_TOKENS:-2176}"
export BENCH_MAX_CACHE_PAGES="${BENCH_MAX_CACHE_PAGES:-1}"
export BENCH_CANDIDATE_CHUNK_TOKENS="${BENCH_CANDIDATE_CHUNK_TOKENS:-2176}"
export BENCH_OVERLAP_SECONDS="${BENCH_OVERLAP_SECONDS:-0,0.05,0.1,0.25}"
export BENCH_REPEATS="${BENCH_REPEATS:-3}"
export BENCH_WARMUP_REPEATS="${BENCH_WARMUP_REPEATS:-1}"
export BENCH_CONTENTION_REPEATS="${BENCH_CONTENTION_REPEATS:-3}"
export BENCH_CONTENTION_FOREGROUND_TOKENS="${BENCH_CONTENTION_FOREGROUND_TOKENS:-2048}"
export RESULT_ROOT="${RESULT_ROOT:-${REPO_ROOT}/results/streaming_tool_call_background_prefill_benchmark/super-tp4-page-cap-${BENCH_MAX_CACHE_PAGES}}"

exec bash "${REPO_ROOT}/examples/swe_bench/run_background_prefill_admission_benchmark.sh"

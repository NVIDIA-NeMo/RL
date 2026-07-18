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

# Fixed-prompt correctness and latency baseline for continuing a streamed
# Nemotron 3 Super prefill request into its authoritative final decode.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
export REPO_ROOT

export BENCH_MODE="same_request_final_decode"
export MODEL_PATH="${MODEL_PATH:-nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16}"
export VLLM_TP="${VLLM_TP:-8}"
export ACCOUNT="${ACCOUNT:-nemotron_sw_post}"
export PARTITION="${PARTITION:-interactive}"
export TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
export BENCH_CANDIDATE_CHUNK_TOKEN_SWEEP="${BENCH_CANDIDATE_CHUNK_TOKEN_SWEEP:-256,512,1024}"
export BENCH_REPEATS="${BENCH_REPEATS:-3}"
export BENCH_WARMUP_REPEATS="${BENCH_WARMUP_REPEATS:-1}"
export BENCH_FINAL_DECODE_TOOL_OUTPUT_TOKENS="${BENCH_FINAL_DECODE_TOOL_OUTPUT_TOKENS:-4096}"
export BENCH_FINAL_DECODE_MAX_OUTPUT_TOKENS="${BENCH_FINAL_DECODE_MAX_OUTPUT_TOKENS:-8}"
export BENCH_SAME_REQUEST_PREFILL_CHUNKS="${BENCH_SAME_REQUEST_PREFILL_CHUNKS:-1}"
export BENCH_CONCURRENT_SAME_REQUEST_SESSIONS="${BENCH_CONCURRENT_SAME_REQUEST_SESSIONS:-8}"
export RESULT_ROOT="${RESULT_ROOT:-${REPO_ROOT}/results/streaming_tool_call_background_prefill_benchmark/super-same-request}"

exec bash "${REPO_ROOT}/examples/swe_bench/run_background_prefill_admission_benchmark.sh"

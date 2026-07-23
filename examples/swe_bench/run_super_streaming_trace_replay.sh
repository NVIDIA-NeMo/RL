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

# Replay real recorded SWE prompt transitions in paired normal-request and
# same-request-final-decode arms. Prompt and tool text are never written to the
# result; only hashes, lengths, token counts, latency, and output parity remain.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
export REPO_ROOT

export BENCH_MODE="trace_replay"
export MODEL_PATH="${MODEL_PATH:-nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16}"
export VLLM_TP="${VLLM_TP:-8}"
export BENCH_GPU_MEMORY_UTILIZATION="${BENCH_GPU_MEMORY_UTILIZATION:-0.8}"
export ACCOUNT="${ACCOUNT:-nemotron_sw_post}"
export PARTITION="${PARTITION:-interactive,batch}"
export TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
export BENCH_TRACE_LIMIT="${BENCH_TRACE_LIMIT:-16}"
export BENCH_MAX_TRACES_PER_TRAJECTORY="${BENCH_MAX_TRACES_PER_TRAJECTORY:-1}"
export BENCH_SNAPSHOT_CHARS="${BENCH_SNAPSHOT_CHARS:-256}"
export BENCH_SNAPSHOT_CHARS_SWEEP="${BENCH_SNAPSHOT_CHARS_SWEEP:-256,1024,4096}"
export BENCH_TRACE_FINAL_DELAY_SECONDS_SWEEP="${BENCH_TRACE_FINAL_DELAY_SECONDS_SWEEP:-0,0.1,0.25}"
export BENCH_REPEATS="${BENCH_REPEATS:-3}"
export BENCH_WARMUP_REPEATS="${BENCH_WARMUP_REPEATS:-0}"
export RESULT_ROOT="${RESULT_ROOT:-${REPO_ROOT}/results/streaming_tool_call_background_prefill_benchmark/super-trace-replay}"

: "${BENCH_TRAJECTORY_JSONL:?Export BENCH_TRAJECTORY_JSONL with a recorded trajectory_collection.jsonl}"

exec bash "${REPO_ROOT}/examples/swe_bench/run_background_prefill_admission_benchmark.sh"

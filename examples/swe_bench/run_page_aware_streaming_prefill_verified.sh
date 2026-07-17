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

# Compare deterministic streaming-off rollouts with the complete incremental
# tokenizer + page-aware background-prefill path on identical Verified prompts.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
export REPO_ROOT

export VERIFIED_DATA_PATH="${VERIFIED_DATA_PATH:-${REPO_ROOT}/results/swebench_verified/swebench_verified_no_timeout_observed_474_first16.jsonl}"
export EXPECTED_COUNT="${EXPECTED_COUNT:-16}"
export TRAJECTORY_COLLECTION_BATCH_SIZE="${TRAJECTORY_COLLECTION_BATCH_SIZE:-16}"
export NUM_VLLM_REPLICAS="${NUM_VLLM_REPLICAS:-2}"
export ROLLOUT_ONLY_GPP="${ROLLOUT_ONLY_GPP:-4}"
export BASE_CONCURRENCY="${BASE_CONCURRENCY:-16}"
export DETAILED_RUNTIME_METRICS=1
export EXACT_INCREMENTAL_TOKENIZER=1
export FINAL_ONLY_INCREMENTAL_TOKENIZER=1
export FINAL_ONLY_PREFILL=1
export PREFIX_SEEDED_START=1
export PREFILL_AFTER_ADMISSION=1
export BACKGROUND_PREFILL_COMPLETION=1
export STABLE_FIRST_SNAPSHOT_PREFILL=1
export COMPACT_REQUEST_CONTEXT=1
export COUNTERFACTUAL_FULL_TOKENIZER_TIMING=0
export INCREMENTAL_TOKENIZER_CHECKPOINT_INTERVAL="${INCREMENTAL_TOKENIZER_CHECKPOINT_INTERVAL:-8}"
export SNAPSHOT_POLL_INTERVAL_SECONDS="${SNAPSHOT_POLL_INTERVAL_SECONDS:-0.05}"
export SNAPSHOT_LONG_POLL_TIMEOUT_SECONDS="${SNAPSHOT_LONG_POLL_TIMEOUT_SECONDS:-1.0}"
export STREAMING_INITIAL_CHUNK_CHARS="${STREAMING_INITIAL_CHUNK_CHARS:-256}"
export PAIR_ARMS="${PAIR_ARMS:-streaming_off:0:0.05:256:0 page_aware_prefill:1:0.05:256:1:1:1:1:1:256:1.0:1}"
export SEQUENTIAL_ARMS="${SEQUENTIAL_ARMS:-1}"
export SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-nemotron_sw_post}"
# Two replicas require one generation node plus the rollout-only dummy training
# node, fitting the two-node interactive QOS. The two arms run sequentially
# under the per-user node limit. Never add backfill.
export SBATCH_PARTITION="${SBATCH_PARTITION:-interactive,batch_short}"
export SBATCH_TIME="${SBATCH_TIME:-2:0:0}"

exec bash "${REPO_ROOT}/examples/swe_bench/run_streaming_tool_call_verified_pair.sh"

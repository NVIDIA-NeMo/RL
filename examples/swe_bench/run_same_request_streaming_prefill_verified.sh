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

# Compare deterministic streaming-off rollouts with page-aware prefill whose
# authoritative final decode continues on the same vLLM engine request.

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
export PAIR_ARMS="${PAIR_ARMS:-streaming_off:0:0.05:256:0 same_request_prefill:1:0.05:256:1:1:1:1:1:256:1.0:1:1}"
export SEQUENTIAL_ARMS="${SEQUENTIAL_ARMS:-1}"
export AUDIT_PYTHON="${AUDIT_PYTHON:-/usr/bin/python3}"
export CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/coreai/users/ruit/enroot-images/nemo-rl:nightly-071526.squashfs}"
export SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-nemotron_sw_post}"
# The artifact prewarmer runs before either arm and provides a portable current
# uv. Reuse that executable while keeping package/build caches in the shared
# model/container/lock namespace below.
export UV_BIN="${UV_BIN:-${REPO_ROOT}/3rdparty/Gym-workspace/Gym/responses_api_agents/swe_agents/swe_swebench_artifact_prefetch_setup/uv/uv}"
# This is a full 8-GPU allocation on each of two nodes, so the launcher's
# --exclusive flag does not strand GPUs. Never add backfill.
export SBATCH_PARTITION="${SBATCH_PARTITION:-interactive,batch_short}"
export SBATCH_TIME="${SBATCH_TIME:-2:0:0}"

# Linked experiment worktrees should reuse a compatible cache prepared by the
# common repository instead of rebuilding environments and kernels under the
# worktree. An explicit pair cache always wins; otherwise reuse the matching
# namespace only when it already exists.
if [ -z "${PAIR_SHARED_PERSISTENT_CACHE:-}" ] && command -v git >/dev/null 2>&1; then
  COMMON_GIT_DIR="$(git -C "${REPO_ROOT}" rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)"
  if [ -n "${COMMON_GIT_DIR}" ]; then
    COMMON_REPO_ROOT="$(dirname "${COMMON_GIT_DIR}")"
    CACHE_MODEL_KEY="$(printf '%s' "NVIDIA-Nemotron-3-Nano-30B-A3B-BF16" | tr -c '[:alnum:]_.-' '_')"
    CACHE_CONTAINER_KEY="$(printf '%s' "$(basename "${CONTAINER}")" | tr -c '[:alnum:]_.-' '_')"
    CACHE_LOCK_KEY="$(sha256sum "${REPO_ROOT}/uv.lock" | cut -c1-12)"
    COMMON_CACHE="${COMMON_REPO_ROOT}/results/cache/swe_scale/${CACHE_MODEL_KEY}-${CACHE_CONTAINER_KEY}-${CACHE_LOCK_KEY}"
    if [ -d "${COMMON_CACHE}" ]; then
      export PAIR_SHARED_PERSISTENT_CACHE="${COMMON_CACHE}"
    fi
  fi
fi

exec bash "${REPO_ROOT}/examples/swe_bench/run_streaming_tool_call_verified_pair.sh"

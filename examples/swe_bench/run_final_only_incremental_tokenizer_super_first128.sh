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

# Compare streaming off with the post-fix exact incremental tokenizer on the
# first 128 rows of the reviewed 474-row SWE-bench Verified manifest. Launch
# one arm per full two-node srun allocation by setting EXPERIMENT_ARM.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
COMMON_GIT_DIR="$(git -C "${REPO_ROOT}" rev-parse --path-format=absolute --git-common-dir)"
SHARED_REPO_ROOT="$(dirname "${COMMON_GIT_DIR}")"
export REPO_ROOT

if [ "${SUBMIT_MODE:-direct}" = "direct" ] && [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "ERROR: direct mode requires a full two-node srun allocation" >&2
  exit 1
fi

EXPERIMENT_ARM="${EXPERIMENT_ARM:-}"
case "${EXPERIMENT_ARM}" in
  streaming_off)
    export PAIR_ARMS="streaming_off:0:0.05:256:0"
    ;;
  tokenizer_final_only)
    export PAIR_ARMS="tokenizer_final_only:1:0.05:256:1"
    ;;
  *)
    echo "ERROR: EXPERIMENT_ARM must be streaming_off or tokenizer_final_only" >&2
    exit 1
    ;;
esac

export MODEL_PATH="${MODEL_PATH:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/yifuw/hf_home/hub/models--nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-BF16/snapshots/d51eab0d1f979ebc26b546e634a04f450d99158e}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
export CONTAINER="${CONTAINER:-${SHARED_REPO_ROOT}/results/images/nemo-rl-nightly-gym-20260718.squashfs}"
export VERIFIED_DATA_PATH="${VERIFIED_DATA_PATH:-${SHARED_REPO_ROOT}/results/swebench_verified/swebench_verified_no_timeout_observed_474_first128.jsonl}"
export PAIR_SHARED_PERSISTENT_CACHE="${PAIR_SHARED_PERSISTENT_CACHE:-${SHARED_REPO_ROOT}/results/cache/swe_scale/NVIDIA-Nemotron-3-Super-120B-A12B-BF16-nemo-rl-nightly-gym-20260718.squashfs-7c0178463217}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${PAIR_SHARED_PERSISTENT_CACHE}/pip}"
export UV_BIN="${UV_BIN:-${SHARED_REPO_ROOT}/3rdparty/Gym-workspace/Gym/responses_api_agents/swe_agents/swe_swebench_artifact_prefetch_setup/uv/uv}"
export AUDIT_PYTHON="${AUDIT_PYTHON:-/usr/bin/python3}"

export EXPECTED_COUNT=128
export TRAJECTORY_COLLECTION_BATCH_SIZE=128
export NUM_VLLM_REPLICAS=1
export VLLM_TP=8
export ROLLOUT_ONLY_GPP=2
export BASE_CONCURRENCY=16
export AGENT_MAX_TURNS=30

# Both arms share these values. The pair launcher forcibly clears every
# tokenizer/prefill sub-feature for streaming_off. The treatment performs no
# snapshot polling and sends no prefill/generation request for tool output.
export EXACT_INCREMENTAL_TOKENIZER=1
export FINAL_ONLY_INCREMENTAL_TOKENIZER=1
export FINAL_ONLY_PREFILL=0
export PREFIX_SEEDED_START=0
export PREFILL_AFTER_ADMISSION=0
export BACKGROUND_PREFILL_COMPLETION=0
export SAME_REQUEST_FINAL_DECODE=0
export STABLE_FIRST_SNAPSHOT_PREFILL=0
export COMPACT_REQUEST_CONTEXT=0
export DETAILED_RUNTIME_METRICS=1
export COUNTERFACTUAL_FULL_TOKENIZER_TIMING=0
export INCREMENTAL_TOKENIZER_CHECKPOINT_INTERVAL=8
export SNAPSHOT_POLL_INTERVAL_SECONDS=0.05
export STREAMING_INITIAL_CHUNK_CHARS=256

export SEQUENTIAL_ARMS=0
# Artifacts were prewarmed and offline-verified while creating this manifest.
# A nested prewarmer cannot create its own Slurm step inside the direct srun.
export PREWARM_SWEBENCH_ARTIFACTS="${PREWARM_SWEBENCH_ARTIFACTS:-0}"
export SWE_BENCH_ARTIFACT_CACHE_OFFLINE="${SWE_BENCH_ARTIFACT_CACHE_OFFLINE:-1}"
export SUBMIT_MODE="${SUBMIT_MODE:-direct}"
export SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-nemotron_sw_post}"
export SBATCH_PARTITION="${SBATCH_PARTITION:-interactive}"
export SBATCH_TIME="${SBATCH_TIME:-04:00:00}"

exec bash "${REPO_ROOT}/examples/swe_bench/run_streaming_tool_call_verified_pair.sh"

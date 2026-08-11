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

# =============================================================================
# Nano 3.5 V2: SingleController wrapper for nano35_dolphin_launch.sh.
#
# The model, data, GBS=2048 batch shape, and default 64-node split (8 train +
# 40 generation + 16 Gym) stay identical to V1. GenRM and NL2Bash run in PR
# 3511's separate four-node external-service component. This wrapper changes
# the orchestration path to:
#   - examples/run_grpo_single_controller.py
#   - in_order sampling with four-version lookahead
#   - streaming F/B after 32 of the 128 prompt groups are ready
#   - TransferQueue data plane
#
# A four-version lookahead admits five 128-group cohorts, so both inflight and
# buffered capacity default to 128 * (4 + 1) = 640 prompt groups. These counts
# are prompt groups, not the 16 generations in each group.
#
# SC checkpointing is not supported on this branch. Set NRL_MAX_STEPS so the
# intended run fits within one allocation.
#
# Usage (GENRM_BASE_URL must be unset):
#   NRL_MAX_STEPS=10 \
#     bash examples/nemo_gym/nemotron-3.5-nano/nano35_dolphin_launch_sc.sh
#
# Optional:
#   GENRM_REPLICAS=1
#   GENRM_TENSOR_PARALLEL_SIZE=8
#   NL2BASH_REPLICAS=2
#   NL2BASH_TENSOR_PARALLEL_SIZE=4
#   MAX_LOOKAHEAD_VERSIONS=1
#   STREAM_MIN_GROUPS=32
#   MAX_INFLIGHT_PROMPTS=640
#   MAX_BUFFERED_ROLLOUTS=640
#   NUM_STORAGE_UNITS=8
#   NRL_FORCE_REBUILD_VENVS=true
#   DRY_RUN=1
#
# Extra positional arguments are forwarded as Hydra overrides.
# =============================================================================
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

export TRAIN_ENTRYPOINT="${TRAIN_ENTRYPOINT:-./examples/run_grpo_single_controller.py}"
export EXP_NAME="${EXP_NAME:-akamehra-nano35-v2-sc-inorder4-n64-t8-g40-gym16-tp4_cp4_ep8-gpp16-pps128-gbs2048}"

# The SC code in this checkout is newer than the venvs prebaked into the
# container. Reusing those venvs can make isolated Ray workers import the old
# nemo_rl package even though the driver sees the mounted checkout (job 6054272
# failed this way on worker_group_utils). Rebuild each node-local worker venv
# from the mounted source before actor creation. Keep this overridable for a
# future container whose fingerprint exactly matches this checkout.
export NRL_FORCE_REBUILD_VENVS="${NRL_FORCE_REBUILD_VENVS:-true}"

# Keep both large judges outside the Gym actor. The GenRM topology matches the
# currently validated Nano service (one TP=8 replica = two GB200 nodes).
export EXTERNAL_JUDGES="${EXTERNAL_JUDGES:-1}"
export GENRM_REPLICAS="${GENRM_REPLICAS:-1}"
export GENRM_TENSOR_PARALLEL_SIZE="${GENRM_TENSOR_PARALLEL_SIZE:-8}"
export GENRM_REASONING_PARSER_NAME="${GENRM_REASONING_PARSER_NAME:-deepseek_r1}"
export NL2BASH_REPLICAS="${NL2BASH_REPLICAS:-2}"
export NL2BASH_TENSOR_PARALLEL_SIZE="${NL2BASH_TENSOR_PARALLEL_SIZE:-4}"
export EXTERNAL_VLLM_SEGMENT_SIZE="${EXTERNAL_VLLM_SEGMENT_SIZE:-2}"
export EXTERNAL_VLLM_SKIP_PREFLIGHT="${EXTERNAL_VLLM_SKIP_PREFLIGHT:-1}"

MAX_LOOKAHEAD_VERSIONS="${MAX_LOOKAHEAD_VERSIONS:-4}"
STREAM_MIN_GROUPS="${STREAM_MIN_GROUPS:-32}"
NUM_STORAGE_UNITS="${NUM_STORAGE_UNITS:-8}"

for value_name in MAX_LOOKAHEAD_VERSIONS STREAM_MIN_GROUPS NUM_STORAGE_UNITS; do
  value="${!value_name}"
  if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: ${value_name} must be a non-negative integer, got '${value}'." >&2
    exit 1
  fi
done

# rlvr_dolphin.yaml fixes the optimizer cohort at 128 prompt groups.
required_buffer_capacity=$((128 * (MAX_LOOKAHEAD_VERSIONS + 1)))
MAX_INFLIGHT_PROMPTS="${MAX_INFLIGHT_PROMPTS:-${required_buffer_capacity}}"
MAX_BUFFERED_ROLLOUTS="${MAX_BUFFERED_ROLLOUTS:-${required_buffer_capacity}}"

for value_name in MAX_INFLIGHT_PROMPTS MAX_BUFFERED_ROLLOUTS; do
  value="${!value_name}"
  if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: ${value_name} must be a non-negative integer, got '${value}'." >&2
    exit 1
  fi
done

if (( STREAM_MIN_GROUPS < 1 || STREAM_MIN_GROUPS > 128 )); then
  echo "ERROR: STREAM_MIN_GROUPS must be in [1, 128], got ${STREAM_MIN_GROUPS}." >&2
  exit 1
fi
if (( NUM_STORAGE_UNITS < 1 )); then
  echo "ERROR: NUM_STORAGE_UNITS must be at least 1." >&2
  exit 1
fi
if (( MAX_INFLIGHT_PROMPTS < 1 )); then
  echo "ERROR: MAX_INFLIGHT_PROMPTS must be at least 1." >&2
  exit 1
fi
if (( MAX_BUFFERED_ROLLOUTS < required_buffer_capacity )); then
  echo "ERROR: MAX_BUFFERED_ROLLOUTS=${MAX_BUFFERED_ROLLOUTS} is below the in_order floor ${required_buffer_capacity}." >&2
  exit 1
fi

echo "=============================================================="
echo "  Nano 3.5 V2 — SingleController in_order/lookahead-${MAX_LOOKAHEAD_VERSIONS}"
echo "  Streaming threshold: ${STREAM_MIN_GROUPS}/128 prompt groups"
echo "  Inflight/buffered: ${MAX_INFLIGHT_PROMPTS}/${MAX_BUFFERED_ROLLOUTS} prompt groups"
echo "  TransferQueue storage units: ${NUM_STORAGE_UNITS}"
echo "  Rebuild Ray worker venvs: ${NRL_FORCE_REBUILD_VENVS}"
echo "  External judges: GenRM ${GENRM_REPLICAS}xTP${GENRM_TENSOR_PARALLEL_SIZE} + NL2Bash ${NL2BASH_REPLICAS}xTP${NL2BASH_TENSOR_PARALLEL_SIZE}"
echo "=============================================================="

exec bash "${SCRIPT_DIR}/nano35_dolphin_launch.sh" \
  checkpointing.enabled=false \
  grpo.val_period=0 \
  grpo.val_at_start=false \
  grpo.val_at_end=false \
  ++grpo.skip_reference_policy_logprobs_calculation=true \
  ++policy.draft.enabled=false \
  ++async_rl.sampler.name=in_order \
  ++async_rl.sampler.max_lookahead_versions="${MAX_LOOKAHEAD_VERSIONS}" \
  ++async_rl.recompute_kv_cache_after_weight_updates=false \
  ++async_rl.min_groups_for_streaming_train="${STREAM_MIN_GROUPS}" \
  ++async_rl.max_inflight_prompts="${MAX_INFLIGHT_PROMPTS}" \
  ++async_rl.max_buffered_rollouts="${MAX_BUFFERED_ROLLOUTS}" \
  ++async_rl.diagnostics=false \
  ++data_plane.enabled=true \
  ++data_plane.impl=transfer_queue \
  ++data_plane.backend=simple \
  ++data_plane.storage_capacity=1000000 \
  ++data_plane.num_storage_units="${NUM_STORAGE_UNITS}" \
  ++data_plane.claim_meta_poll_interval_s=0.5 \
  ++data_plane.global_segment_size=549755813888 \
  ++data_plane.local_buffer_size=68719476736 \
  "$@"

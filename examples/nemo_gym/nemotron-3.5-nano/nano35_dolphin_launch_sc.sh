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

set -euo pipefail

# =============================================================================
# nano35_dolphin_launch_sc.sh
#
# SingleController variant of nano35_dolphin_launch.sh: the same 64-node
# Nemotron 3.5 Nano RLVR pipeclean, driven by run_grpo_single_controller.py
# with streaming forward/backward, the TransferQueue data plane and
# shard-to-shard NCCL-reshard weight refit.
#
# Shape (GB200, 4 GPUs/node -> 256 GPUs), unchanged from the baseline so the
# two runs are comparable:
#   8 train + 40 generation + 16 gym = 64 nodes  (5:1 generation-to-training)
# GenRM adds 4 nodes from its own allocation, so the campaign footprint is 68.
#
# This is a thin wrapper over nano35_dolphin_launch.sh, which already carries
# every site default (model, blend, judges, container, mounts, caches, Slurm).
# We only swap the config and the driver, so the two runs differ solely in the
# SC wiring.
#
# This branch supports durable trainer checkpoints plus periodic rollout/TQ
# snapshots. The defaults below exercise that production recovery path; set
# ROLLOUT_CHECKPOINT_INTERVAL_S=null to keep trainer checkpointing but disable
# periodic rollout snapshots for an ablation.
#
# Usage:
#   GENRM_BASE_URL=http://<lb-host>:9213/v1 \
#     bash examples/nemo_gym/nemotron-3.5-nano/nano35_dolphin_launch_sc.sh
#
#   DRY_RUN=1 GENRM_BASE_URL=... bash .../nano35_dolphin_launch_sc.sh
#
# GenRM must already be serving — see the GenRM section of the baseline script
# for how to stand up the external pool, or pass EXTERNAL_JUDGES=1 to host
# GenRM and the NL2Bash judge in a hetgroup inside this job instead, which is
# the shape the 6K recipe uses.
#
# Optional:
#   NRL_MAX_STEPS=10             # short pipeclean
#   STREAM_MIN_GROUPS=32         # async_rl.min_groups_for_streaming_train
#   SAMPLER=in_order             # in_order | weight_fifo | windowed
#   MAX_LOOKAHEAD_VERSIONS=4     # the sampler's slack, whatever it spells it
#                                # 1 restores parity with the async-1 baseline
#   BUFFER_RETENTION_MULTIPLIER=2  # max_buffered_rollouts only; gated samplers
#   NUM_STORAGE_UNITS=16         # data_plane.simple.num_storage_units
#   REFIT_TRANSPORT=null         # fall back to the full-tensor NCCL broadcast
#   ROLLOUT_CHECKPOINT_INTERVAL_S=120
#   ROLLOUT_TELEMETRY_INTERVAL_S=30
#   GYM_SOURCE_ROOT=<repo>/3rdparty/Gym-workspace/Gym
#   MOUNT_LOCAL_GYM=0            # selectively overlay token-capture code
#
# Extra positional args are forwarded as Hydra overrides, after ours, so they win.
# =============================================================================

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
_SC_REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)

# The nightly contains prefetched driver and async-vLLM-worker environments,
# but this branch's token-capture code is newer than the Gym copy in the image.
# Overlay only the relevant Gym modules so the container's project metadata and
# lock file stay paired, then add the one missing binary dependency to both
# prefetched environments. Exact pinning also keeps the generated shell command
# free of comparison operators such as `>`.
export GYM_SOURCE_ROOT="${GYM_SOURCE_ROOT:-${_SC_REPO_ROOT}/3rdparty/Gym-workspace/Gym}"
export GYM_CONTAINER_ROOT="${GYM_CONTAINER_ROOT:-/opt/nemo-rl/3rdparty/Gym-workspace/Gym}"
export MOUNT_LOCAL_GYM="${MOUNT_LOCAL_GYM:-0}"

_SC_GYM_PATHS=(
  "nemo_gym"
  "responses_api_models/vllm_model/app.py"
  "responses_api_agents/simple_agent/app.py"
)
for _relative_path in "${_SC_GYM_PATHS[@]}"; do
  if [[ ! -e "${GYM_SOURCE_ROOT}/${_relative_path}" ]]; then
    echo "Required Gym overlay is missing: ${GYM_SOURCE_ROOT}/${_relative_path}" >&2
    exit 1
  fi

  _destination="${GYM_CONTAINER_ROOT}/${_relative_path}"
  _mount="${GYM_SOURCE_ROOT}/${_relative_path}:${_destination}"
  if [[ ",${EXTRA_MOUNTS:-}," != *":${_destination},"* ]]; then
    export EXTRA_MOUNTS="${EXTRA_MOUNTS:+${EXTRA_MOUNTS},}${_mount}"
  fi
done
unset _destination _mount _relative_path _SC_GYM_PATHS

export NRL_DRIVER_PYTHONPATH="${NRL_DRIVER_PYTHONPATH:-${GYM_CONTAINER_ROOT}}"
export NRL_DRIVER_PIP_INSTALL="${NRL_DRIVER_PIP_INSTALL:-orjson==3.11.9}"
export NRL_VLLM_WORKER_PIP_INSTALL="${NRL_VLLM_WORKER_PIP_INSTALL:-orjson==3.11.9}"

export CONFIG_PATH="${CONFIG_PATH:-examples/nemo_gym/nemotron-3.5-nano/rlvr_dolphin_sc.yaml}"

# The SC driver. This branch's Ultra launcher reads NRL_ENTRYPOINT.
# data_plane.enabled=true (set in the config) is mandatory for it.
export NRL_ENTRYPOINT="${NRL_ENTRYPOINT:-./examples/run_grpo_single_controller.py}"

# Distinct from the baseline's EXP_NAME so this starts a new W&B run, run dir
# and singleton job name rather than colliding with the async-1 baseline.
export EXP_NAME="${EXP_NAME:-akamehra-nano35-honest-dolphin-v10-iter6000-rlvr-sc-tp4_cp4_ep8_pp1_gpp16_pps128_gbs2048}"

# The SC knobs worth sweeping without editing the config.
#
# STREAM_MIN_GROUPS starts the optimizer step earlier on partial cohorts.
# NUM_STORAGE_UNITS is the untuned one at this data volume. It is passed as a
# Hydra override unconditionally below, so it beats the recipe and the value in
# rlvr_dolphin_sc.yaml is never what runs — keep the two in step. It shards a
# global pool rather than reserving per unit, so over-provisioning costs only a
# CPU actor each; the windowed sweep's 14336 peak rows are 1.4% of capacity.
# MAX_LOOKAHEAD_VERSIONS is the sampler's slack. in_order spells it
# max_lookahead_versions; weight_fifo/windowed spell it max_staleness_versions.
# The gated samplers also require max_buffered_rollouts >= prompts * (slack + 1).
STREAM_MIN_GROUPS="${STREAM_MIN_GROUPS:-32}"
NUM_STORAGE_UNITS="${NUM_STORAGE_UNITS:-16}"
MAX_LOOKAHEAD_VERSIONS="${MAX_LOOKAHEAD_VERSIONS:-4}"
ROLLOUT_CHECKPOINT_INTERVAL_S="${ROLLOUT_CHECKPOINT_INTERVAL_S:-300}"
ROLLOUT_TELEMETRY_INTERVAL_S="${ROLLOUT_TELEMETRY_INTERVAL_S:-60}"

# Emit only fields belonging to the selected discriminated sampler config.
# Hydra's `+` form is required when switching away from the in_order block
# declared in rlvr_dolphin_sc.yaml.
SAMPLER="${SAMPLER:-in_order}"
case "${SAMPLER}" in
  in_order)
    _SAMPLER_OVERRIDES=(
      "async_rl.sampler.name=in_order"
      "async_rl.sampler.max_lookahead_versions=${MAX_LOOKAHEAD_VERSIONS}"
    )
    ;;
  weight_fifo)
    _SAMPLER_OVERRIDES=(
      "async_rl.sampler.name=weight_fifo"
      "+async_rl.sampler.max_staleness_versions=${MAX_LOOKAHEAD_VERSIONS}"
    )
    ;;
  windowed)
    _SAMPLER_OVERRIDES=(
      "async_rl.sampler.name=windowed"
      "+async_rl.sampler.max_staleness_versions=${MAX_LOOKAHEAD_VERSIONS}"
    )
    ;;
  *)
    echo "SAMPLER must be in_order, weight_fifo or windowed, got '${SAMPLER}'" >&2
    exit 1
    ;;
esac

# Shard-to-shard weight refit, on by default in this variant. It is still
# experimental, so keep the escape hatch one env var away: REFIT_TRANSPORT=null
# restores the full-tensor broadcast that rlvr_dolphin.yaml uses.
REFIT_TRANSPORT="${REFIT_TRANSPORT:-nccl_reshard}"

_NUM_PROMPTS_PER_STEP="${_NUM_PROMPTS_PER_STEP:-128}"

# Generation quota: the current cohort plus every lookahead cohort in flight at
# once. This is the number that must not move, because it is what the arms are
# compared on.
_MAX_INFLIGHT_PROMPTS=$(( _NUM_PROMPTS_PER_STEP * (MAX_LOOKAHEAD_VERSIONS + 1) ))

# Retention headroom, as a multiple of that quota.
#
# _buffer_capacity is a per-group semaphore taken at dispatch and released on
# select, evict, or failure. Zero eviction deletes one of those three release
# paths, so groups that have finished generating but have not yet been trained
# on stay resident holding permits. At a multiplier of 1 they are holding
# permits out of the same pool that admission draws from, so the finished work
# crowds out new generation -- the fix for eviction creates a throughput
# problem one layer down.
#
# A multiplier above 1 gives retention its own headroom, which is what v1 does:
# late_arrival_slack=2 sizes its retention at P*lag*2 against a generation quota
# of P*lag. Retention strictly exceeding what admission can produce is the
# property that stops completed work from starving dispatch.
#
# This is NOT job 6014206 (768 buffered against 384 in flight, 1.8x slower).
# That arm ran WindowedSampler, which derives from BaseSampler and whose admit
# returns None immediately -- "dispatch is bounded by buffer capacity, not by
# version" -- so there the buffer was the only thing limiting dispatch and
# raising it raised dispatch. in_order and weight_fifo are gated samplers, so
# their dispatch windows remain bounded independently of buffer headroom.
BUFFER_RETENTION_MULTIPLIER="${BUFFER_RETENTION_MULTIPLIER:-1}"
_MAX_BUFFERED_ROLLOUTS=$(( _MAX_INFLIGHT_PROMPTS * BUFFER_RETENTION_MULTIPLIER ))

if (( BUFFER_RETENTION_MULTIPLIER < 1 )); then
  echo "BUFFER_RETENTION_MULTIPLIER must be >= 1, got ${BUFFER_RETENTION_MULTIPLIER}." >&2
  echo "Below 1 the buffer sits under the sampler's required floor and the train" >&2
  echo "pump waits for a batch the buffer is too small to ever hold." >&2
  exit 1
fi

if (( BUFFER_RETENTION_MULTIPLIER > 1 )) && [[ "${SAMPLER}" == "windowed" ]]; then
  echo "BUFFER_RETENTION_MULTIPLIER=${BUFFER_RETENTION_MULTIPLIER} with SAMPLER=windowed is the 6014206 trap." >&2
  echo "WindowedSampler.admit returns None, so the buffer is its only dispatch" >&2
  echo "limit and raising it raises dispatch: that arm ran 1.8x slower. Only the" >&2
  echo "gated samplers (in_order and weight_fifo) can take a multiplier." >&2
  exit 1
fi

echo "================================================================"
echo "  Nemotron 3.5 Nano — RLVR SingleController (honest-dolphin)"
echo "================================================================"
echo "  Entrypoint : ${NRL_ENTRYPOINT}"
echo "  Config     : ${CONFIG_PATH}"
echo "  Refit      : ${REFIT_TRANSPORT}"
echo "  Streaming  : min ${STREAM_MIN_GROUPS} of ${_NUM_PROMPTS_PER_STEP} groups per dispatch"
echo "  Sampler    : ${SAMPLER} (slack ${MAX_LOOKAHEAD_VERSIONS})"
echo "  Capacity   : buffer ${_MAX_BUFFERED_ROLLOUTS} groups (x${BUFFER_RETENTION_MULTIPLIER}), ${_MAX_INFLIGHT_PROMPTS} in flight"
echo "  TQ units   : ${NUM_STORAGE_UNITS}"
echo "  Rollout ckpt: interval=${ROLLOUT_CHECKPOINT_INTERVAL_S}s, telemetry=${ROLLOUT_TELEMETRY_INTERVAL_S}s"
echo "================================================================"
echo ""

# The advantage and reward penalties below are inherited from student_rlvr1.yaml
# and cannot work under token capture, which rlvr_dolphin_sc.yaml turns on:
#   - grpo.*_advantage raise NotImplementedError at setup, because the capture
#     finalizer never emits the invalid_tool_call_mask/malformed_thinking_mask
#     train-batch columns the advantage stage would read.
#   - reward_penalties only warn, but capture receipts carry no generated
#     tokens or text at rollout time, so every penalty check is skipped and the
#     penalty-rate metrics read 0. Turning them off keeps the config honest
#     about what the run actually trains on.
# Both use ++ rather than =: only invalid_tool_call_advantage and the four
# penalize_* flags exist in the inherited YAML, so a bare override on the rest
# fails with "not in struct" before the job starts.
#
# allow_openai_version_skew is the mount-overlay counterpart. Gym reads its own
# openai constraint from installed distribution metadata
# (_nemo_gym_openai_requirement -> importlib.metadata.requires), and an
# EXTRA_MOUNTS overlay replaces Gym's sources without regenerating the venv's
# baked dist-info. The parent then runs the openai the mounted tree needs while
# the stale metadata still advertises the image's older cap, and Gym refuses to
# pin an impossible version into its server venvs. The flag drops that pin;
# server venvs install '-e .' from the same mounted tree, so they resolve the
# matching openai on their own. Everything under env.nemo_gym is forwarded
# verbatim as Gym's initial_global_config_dict (build_nemo_gym_config).
exec bash "${SCRIPT_DIR}/nano35_dolphin_launch.sh" \
  "++env.nemo_gym.allow_openai_version_skew=true" \
  "async_rl.min_groups_for_streaming_train=${STREAM_MIN_GROUPS}" \
  "${_SAMPLER_OVERRIDES[@]}" \
  "async_rl.max_inflight_prompts=${_MAX_INFLIGHT_PROMPTS}" \
  "async_rl.max_buffered_rollouts=${_MAX_BUFFERED_ROLLOUTS}" \
  "data_plane.simple.num_storage_units=${NUM_STORAGE_UNITS}" \
  "policy.generation.refit_transport=${REFIT_TRANSPORT}" \
  "rollout_checkpointing.snapshot_attempt_interval_s=${ROLLOUT_CHECKPOINT_INTERVAL_S}" \
  "rollout_checkpointing.telemetry_interval_s=${ROLLOUT_TELEMETRY_INTERVAL_S}" \
  "++grpo.invalid_tool_call_advantage=null" \
  "++grpo.malformed_thinking_advantage=null" \
  "++reward_penalties.penalize_duplicated_reasoning=false" \
  "++reward_penalties.penalize_empty_final_answer=false" \
  "++reward_penalties.penalize_unwanted_tokens=false" \
  "++reward_penalties.penalize_malformed_think_tag=false" \
  "$@"

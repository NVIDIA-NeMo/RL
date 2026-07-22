#!/bin/bash

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

# runs benchmark and reports time to convergence
set -e

# GRPO driver: runs exactly ONCE inside the container on the Ray head node
# (no per-rank torch-distributed env). Ray schedules training/generation/Gym
# actors across the allocation internally.

# Vars without defaults
: "${SEED:?SEED not set}"
: "${DGXNNODES:?DGXNNODES not set}"
: "${DGXNGPU:?DGXNGPU not set}"
: "${GEN_NODES:?GEN_NODES not set}"

# Vars with defaults
: "${RECIPE:=qwen_35/configs/grpo_qwen35_397b_swe_openhands_async_gbs256.yaml}"
: "${MLPERF_MLLOG_FILE:=/results/mllog.log}"
: "${MLPERF_TARGET_ACCURACY:=1.0}"
: "${MLPERF_BENCHMARK_NAME:=qwen35_397b_grpo}"
: "${EXP_NAME:=${SPREFIX:-qwen35_397b_grpo}}"
: "${EXTRA_ARGS:=$@}"

[ "${DEBUG:-0}" = "0" ] || set -x

cd /opt/nemo-rl

# Re-export the input paths at their container-side locations (mounted by
# config_mounts.sh); the recipe and nemo-rl read these names from env.
export CONTAINER_HF_CKPT_PATH="${CONTAINER_HF_CKPT_PATH:?CONTAINER_HF_CKPT_PATH not set (config_mounts.sh)}"
export NRL_MEGATRON_CHECKPOINT_DIR="${CONTAINER_NRL_MEGATRON_CHECKPOINT_DIR:?not set (config_mounts.sh)}"
export NEMO_GYM_SWE_TRAIN_DATA_PATH="${CONTAINER_NEMO_GYM_SWE_TRAIN_DATA_PATH:?not set (config_mounts.sh)}"
export NEMO_GYM_SWE_VALIDATION_DATA_PATH="${CONTAINER_NEMO_GYM_SWE_VALIDATION_DATA_PATH:?not set (config_mounts.sh)}"
export NEMO_GYM_SWE_SIF_DIR="${CONTAINER_NEMO_GYM_SWE_SIF_DIR:?not set (config_mounts.sh)}"
export NEMO_GYM_SWE_WORKSPACE_ROOT="/logs/nemo_gym/workspace"

export HF_HOME="/opt/nemo-rl/.cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export RAY_CGRAPH_get_timeout="${RAY_CGRAPH_get_timeout:-1800}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-0}"
export VLLM_USE_RAY_V2_EXECUTOR_BACKEND="${VLLM_USE_RAY_V2_EXECUTOR_BACKEND:-0}"
export HF_TOKEN="${HF_TOKEN:-}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export MLPERF_SUBMISSION_ORG="${MLPERF_SUBMITTER:-reference}"
export MLPERF_SUBMISSION_PLATFORM="${MLPERF_SYSTEM_NAME:-reference}"

if [ "${FORCE_SUCCESS_STATUS:-0}" -eq 1 ]; then
    _force_success=True
else
    _force_success=False
fi

declare -a OVERRIDES
OVERRIDES+=("++logger.mlperf_enabled=True")
OVERRIDES+=("++logger.mlperf.log_file=${MLPERF_MLLOG_FILE}")
OVERRIDES+=("++logger.mlperf.benchmark=${MLPERF_BENCHMARK_NAME}")
OVERRIDES+=("++logger.mlperf.target_accuracy=${MLPERF_TARGET_ACCURACY}")
OVERRIDES+=("++logger.mlperf.force_success_status=${_force_success}")
OVERRIDES+=("++cluster.num_nodes=${DGXNNODES}")
OVERRIDES+=("++cluster.gpus_per_node=${DGXNGPU}")
if [[ "${COLOCATED_GENERATION:-0}" == "1" ]]; then
    _generation_nodes="${COLOCATED_GENERATION_NODES:-${TRAIN_NODES}}"
    _generation_mode=" colocated"
else
    _generation_nodes="${GEN_NODES}"
    _generation_mode=""
fi
OVERRIDES+=("++policy.generation.colocated.resources.num_nodes=${_generation_nodes}")
OVERRIDES+=("++policy.generation.colocated.resources.gpus_per_node=${DGXNGPU}")
OVERRIDES+=("++logger.wandb.name=${EXP_NAME}")
OVERRIDES+=("++logger.log_dir=/logs")
# Per-experiment subdir: checkpointing auto-resumes and max_num_steps is an
# absolute cumulative count, so NEXP experiments must not share a directory
# (index is stable across walltime windows, keeping resume-chaining intact).
_ckpt_dir="/checkpoint/${_experiment_index:-01}"
mkdir -p "${_ckpt_dir}"
OVERRIDES+=("++checkpointing.checkpoint_dir=${_ckpt_dir}")
OVERRIDES+=("++grpo.seed=${SEED}")

if [[ -n "${MAX_STEPS:-}" ]]; then
    OVERRIDES+=("++grpo.max_num_steps=${MAX_STEPS}")
fi
if [[ -n "${VAL_START_AT:-}" ]]; then
    OVERRIDES+=("++grpo.val_start_at=${VAL_START_AT}")
fi
if [[ -n "${NUM_PROMPTS_PER_STEP:-}" ]]; then
    OVERRIDES+=("++grpo.num_prompts_per_step=${NUM_PROMPTS_PER_STEP}")
fi
if [[ -n "${NUM_GENERATIONS_PER_PROMPT:-}" ]]; then
    OVERRIDES+=("++grpo.num_generations_per_prompt=${NUM_GENERATIONS_PER_PROMPT}")
fi
if [[ -n "${LEARNING_RATE:-}" ]]; then
    OVERRIDES+=("++policy.megatron_cfg.optimizer.lr=${LEARNING_RATE}")
    OVERRIDES+=("++policy.megatron_cfg.optimizer.min_lr=${LEARNING_RATE}")
fi
if [[ -n "${MAX_GRAD_NORM:-}" ]]; then
    OVERRIDES+=("++policy.max_grad_norm=${MAX_GRAD_NORM}")
fi
# cluster.segment_size must PAIR with the sbatch-side --segment
if [[ -n "${SEGMENT:-}" ]]; then
    OVERRIDES+=("++cluster.segment_size=${SEGMENT}")
fi

echo "running GRPO benchmark: recipe=${RECIPE} seed=${SEED}" \
     "nodes=${DGXNNODES} (train=${TRAIN_NODES} gen=${_generation_nodes}${_generation_mode})"
echo "Extra args: ${EXTRA_ARGS}"

# `|| ret_code=$?` (not `; ret_code=$?`): under set -e a failing command
# followed by `;` exits the shell before ret_code is captured
ret_code=0
if [[ -f /workspace/llm/run_grpo_nemo_gym.py ]]; then
    _grpo_driver=/workspace/llm/run_grpo_nemo_gym.py
else
    _grpo_driver=examples/nemo_gym/run_grpo_nemo_gym.py
fi
uv run "${_grpo_driver}" \
    --config "${RECIPE}" \
    "${OVERRIDES[@]}" \
    ${EXTRA_ARGS} \
    || ret_code=$?

set +x
sleep 3
if [[ $ret_code != 0 ]]; then exit $ret_code; fi

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

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/..")

CONTAINER=${CONTAINER:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_genai/users/slikhite/images/nemo-rl-nightly.sqsh}
ACCOUNT=${ACCOUNT:-coreai_dlalgo_genai}
PARTITION=${PARTITION:-batch}
HF_HOME=${HF_HOME:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_genai/users/slikhite/hf_home}
HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
HF_TOKEN=${HF_TOKEN:-xxx}
WANDB_API_KEY=${WANDB_API_KEY:-xxx}

require_env() {
    local var_name="$1"
    if [[ -z "${!var_name:-}" ]]; then
        echo "[ERROR] $var_name must be set." >&2
        exit 1
    fi
}

for var_name in CONTAINER ACCOUNT PARTITION HF_HOME HF_DATASETS_CACHE; do
    require_env "$var_name"
done

CONFIG=${CONFIG:-examples/configs/grpo_math_1B.yaml}
MODEL_NAME=${MODEL_NAME:-Qwen/Qwen2.5-1.5B}
JOB_NAME=${JOB_NAME:-basic-grpo-smoke}
NUM_NODES=${NUM_NODES:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
WALLTIME=${WALLTIME:-4:00:00}
MAX_STEPS=${MAX_STEPS:-3}
LOG_DIR=${LOG_DIR:-logs/basic-grpo-smoke}
CKPT_DIR=${CKPT_DIR:-results/basic-grpo-smoke}
SLURM_OUTPUT=${SLURM_OUTPUT:-slurm-%j-${JOB_NAME}.out}
RAY_LOG_SYNC_FREQUENCY=${RAY_LOG_SYNC_FREQUENCY:-60}
NEMO_RL_VENV_DIR=${NEMO_RL_VENV_DIR:-$PROJECT_ROOT/venvs-basic-grpo}
UV_CACHE_DIR_OVERRIDE=${UV_CACHE_DIR_OVERRIDE:-}
NRL_FORCE_REBUILD_VENVS=${NRL_FORCE_REBUILD_VENVS:-true}
DEEP_GEMM_PREFLIGHT=${DEEP_GEMM_PREFLIGHT:-false}
EXTRA_MOUNTS=${EXTRA_MOUNTS:-${MOUNTS:-}}
EXTRA_OVERRIDES=${EXTRA_OVERRIDES:-}

BASE_MOUNTS="$PROJECT_ROOT:$PROJECT_ROOT,$HF_HOME:$HF_HOME,$HF_DATASETS_CACHE:$HF_DATASETS_CACHE"
BASE_MOUNTS="$BASE_MOUNTS,$NEMO_RL_VENV_DIR:$NEMO_RL_VENV_DIR"
if [[ -n "$UV_CACHE_DIR_OVERRIDE" ]]; then
    BASE_MOUNTS="$BASE_MOUNTS,$UV_CACHE_DIR_OVERRIDE:$UV_CACHE_DIR_OVERRIDE"
fi
if [[ -n "$EXTRA_MOUNTS" ]]; then
    BASE_MOUNTS="$BASE_MOUNTS,$EXTRA_MOUNTS"
fi
MOUNTS="$BASE_MOUNTS"

COMMAND="uv run ./examples/run_grpo.py --config $CONFIG policy.model_name=$MODEL_NAME policy.tokenizer.name=$MODEL_NAME policy.optimizer=null policy.dtensor_cfg.enabled=false policy.megatron_cfg.enabled=true policy.make_sequence_length_divisible_by=1 policy.train_global_batch_size=16 policy.train_micro_batch_size=1 policy.logprob_batch_size=1 policy.max_total_sequence_length=512 policy.generation.max_new_tokens=128 policy.generation.vllm_cfg.max_model_len=512 grpo.max_num_steps=$MAX_STEPS grpo.num_prompts_per_step=4 grpo.num_generations_per_prompt=4 grpo.val_at_start=false grpo.val_at_end=false logger.log_dir=$LOG_DIR logger.wandb_enabled=false logger.tensorboard_enabled=true logger.monitor_gpus=true checkpointing.enabled=false checkpointing.checkpoint_dir=$CKPT_DIR cluster.num_nodes=$NUM_NODES cluster.gpus_per_node=$GPUS_PER_NODE"
if [[ -n "$EXTRA_OVERRIDES" ]]; then
    COMMAND="$COMMAND $EXTRA_OVERRIDES"
fi
if [[ "$DEEP_GEMM_PREFLIGHT" == "true" ]]; then
    DEEP_GEMM_PREFLIGHT_CODE="import torch, deep_gemm; missing=[name for name in ('get_paged_mqa_logits_metadata', 'fp8_paged_mqa_logits') if not callable(getattr(deep_gemm, name, None))]; assert not missing, f'DeepGEMM is missing required vLLM MLA symbols: {missing}'; print(f'DeepGEMM preflight OK: {deep_gemm.__file__}')"
    COMMAND="uv run --locked --extra vllm python -c \"$DEEP_GEMM_PREFLIGHT_CODE\" && $COMMAND"
fi
cd "$PROJECT_ROOT"

mkdir -p "$NEMO_RL_VENV_DIR"
if [[ -n "$UV_CACHE_DIR_OVERRIDE" ]]; then
    mkdir -p "$UV_CACHE_DIR_OVERRIDE"
fi
if [[ -n "$(dirname "$SLURM_OUTPUT")" && "$(dirname "$SLURM_OUTPUT")" != "." ]]; then
    mkdir -p "$(dirname "$SLURM_OUTPUT")"
fi

if [[ -n "${DRYRUN:-}" ]]; then
    echo "[DRYRUN] CONTAINER=$CONTAINER"
    echo "[DRYRUN] ACCOUNT=$ACCOUNT"
    echo "[DRYRUN] PARTITION=$PARTITION"
    echo "[DRYRUN] HF_HOME=$HF_HOME"
    echo "[DRYRUN] HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
    echo "[DRYRUN] MODEL_NAME=$MODEL_NAME"
    echo "[DRYRUN] NUM_NODES=$NUM_NODES"
    echo "[DRYRUN] GPUS_PER_NODE=$GPUS_PER_NODE"
    echo "[DRYRUN] MOUNTS=$MOUNTS"
    echo "[DRYRUN] NEMO_RL_VENV_DIR=$NEMO_RL_VENV_DIR"
    echo "[DRYRUN] UV_CACHE_DIR_OVERRIDE=$UV_CACHE_DIR_OVERRIDE"
    echo "[DRYRUN] NRL_FORCE_REBUILD_VENVS=$NRL_FORCE_REBUILD_VENVS"
    echo "[DRYRUN] DEEP_GEMM_PREFLIGHT=$DEEP_GEMM_PREFLIGHT"
    if [[ -n "$HF_TOKEN" ]]; then
        echo "[DRYRUN] HF_TOKEN=<set>"
    else
        echo "[DRYRUN] HF_TOKEN=<unset>"
    fi
    if [[ -n "$WANDB_API_KEY" ]]; then
        echo "[DRYRUN] WANDB_API_KEY=<set>"
    else
        echo "[DRYRUN] WANDB_API_KEY=<unset>"
    fi
    echo "[DRYRUN] COMMAND=$COMMAND"
    echo "[DRYRUN] sbatch --nodes=$NUM_NODES --gres=gpu:$GPUS_PER_NODE --account=$ACCOUNT --job-name=$JOB_NAME --partition=$PARTITION --time=$WALLTIME --output=$SLURM_OUTPUT ray.sub"
    exit 0
fi

CONTAINER="$CONTAINER" \
MOUNTS="$MOUNTS" \
COMMAND="$COMMAND" \
GPUS_PER_NODE="$GPUS_PER_NODE" \
HF_HOME="$HF_HOME" \
HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
HF_TOKEN="$HF_TOKEN" \
WANDB_API_KEY="$WANDB_API_KEY" \
RAY_LOG_SYNC_FREQUENCY="$RAY_LOG_SYNC_FREQUENCY" \
NEMO_RL_VENV_DIR="$NEMO_RL_VENV_DIR" \
UV_CACHE_DIR_OVERRIDE= \
NRL_FORCE_REBUILD_VENVS="$NRL_FORCE_REBUILD_VENVS" \
sbatch \
    --nodes="$NUM_NODES" \
    --gres="gpu:$GPUS_PER_NODE" \
    --account="$ACCOUNT" \
    --job-name="$JOB_NAME" \
    --partition="$PARTITION" \
    --time="$WALLTIME" \
    --output="$SLURM_OUTPUT" \
    ray.sub

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

# Load secrets (HF_TOKEN, etc.) from a sibling file that should be chmod 600.
# Override the location with SECRETS_FILE=... if you keep secrets elsewhere.
SECRETS_FILE="${SECRETS_FILE:-$SCRIPT_DIR/secrets.env}"
if [[ -f "$SECRETS_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$SECRETS_FILE"
fi

RECIPE=${RECIPE:-examples/configs/recipes/llm/grpo-glm5.1-32n8g-megatron.yaml}
RUN_SCRIPT=${RUN_SCRIPT:-./examples/run_grpo.py}
MAX_STEPS_KEY=${MAX_STEPS_KEY:-grpo.max_num_steps}

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

recipe_num_nodes() {
    awk '
      /^cluster:/ { in_cluster = 1; next }
      /^[^[:space:]]/ { in_cluster = 0 }
      in_cluster && /^[[:space:]]+num_nodes:/ { print $2; exit }
    ' "$PROJECT_ROOT/$RECIPE"
}

for var_name in CONTAINER ACCOUNT PARTITION HF_HOME HF_DATASETS_CACHE; do
    require_env "$var_name"
done

RECIPE_NUM_NODES=$(recipe_num_nodes)
MODEL_NAME=${MODEL_NAME:-}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NUM_NODES=${NUM_NODES:-${RECIPE_NUM_NODES:-8}}
WALLTIME=${WALLTIME:-4:00:00}
JOB_NAME=${JOB_NAME:-glm51-32n8g-grpo}
MAX_STEPS=${MAX_STEPS:-20}
LOG_DIR=${LOG_DIR:-logs/grpo-glm5.1-32n8g-megatron}
CKPT_DIR=${CKPT_DIR:-results/grpo-glm5.1-32n8g-megatron}
WANDB_PROJECT=${WANDB_PROJECT:-nemo-rl-glm}
WANDB_NAME=${WANDB_NAME:-grpo-glm5.1-32n8g-megatron}
WANDB_ENABLED=${WANDB_ENABLED:-True}
TENSORBOARD_ENABLED=${TENSORBOARD_ENABLED:-True}
MONITOR_GPUS=${MONITOR_GPUS:-True}
CHECKPOINTING_ENABLED=${CHECKPOINTING_ENABLED:-True}
CHECKPOINT_SAVE_PERIOD=${CHECKPOINT_SAVE_PERIOD:-20}
DEEP_GEMM_PREFLIGHT=${DEEP_GEMM_PREFLIGHT:-true}
SLURM_OUTPUT=${SLURM_OUTPUT:-slurm-%j-${JOB_NAME}.out}
SLURM_COMMENT=${SLURM_COMMENT:-'{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"240","reason":"other","description":"debugging idle gpu"}}'}
RAY_LOG_SYNC_FREQUENCY=${RAY_LOG_SYNC_FREQUENCY:-60}
GLM51_CACHE_TAG=${GLM51_CACHE_TAG:-py313-onnx121}
GLM51_SEED_UV_CACHE=${GLM51_SEED_UV_CACHE:-true}
GLM51_CACHE_SUFFIX=
if [[ -n "$GLM51_CACHE_TAG" ]]; then
    GLM51_CACHE_SUFFIX="-$GLM51_CACHE_TAG"
fi
NEMO_RL_VENV_DIR=${NEMO_RL_VENV_DIR:-$PROJECT_ROOT/venvs-glm51-grpo$GLM51_CACHE_SUFFIX}
UV_CACHE_DIR_OVERRIDE=${UV_CACHE_DIR_OVERRIDE:-$PROJECT_ROOT/.uv-cache-glm51-grpo$GLM51_CACHE_SUFFIX}
NRL_FORCE_REBUILD_VENVS=${NRL_FORCE_REBUILD_VENVS:-false}
NRL_REFIT_BUFFER_MEMORY_RATIO=${NRL_REFIT_BUFFER_MEMORY_RATIO:-0.02}
PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-${PYTORCH_ALLOC_CONF:-}}
EXTRA_MOUNTS=${EXTRA_MOUNTS:-${MOUNTS:-}}
EXTRA_OVERRIDES=${EXTRA_OVERRIDES:-}

BASE_MOUNTS="/lustre:/lustre"
if [[ -n "$EXTRA_MOUNTS" ]]; then
    BASE_MOUNTS="$BASE_MOUNTS,$EXTRA_MOUNTS"
fi
MOUNTS="$BASE_MOUNTS"

CACHE_SETUP="mkdir -p '$NEMO_RL_VENV_DIR' '$UV_CACHE_DIR_OVERRIDE'; if [[ '$GLM51_SEED_UV_CACHE' == 'true' && ! -f '$UV_CACHE_DIR_OVERRIDE/.nemo_rl_seeded' && -d /root/.cache/uv ]]; then cp -a /root/.cache/uv/. '$UV_CACHE_DIR_OVERRIDE/' && touch '$UV_CACHE_DIR_OVERRIDE/.nemo_rl_seeded'; elif [[ '$GLM51_SEED_UV_CACHE' != 'true' ]]; then touch '$UV_CACHE_DIR_OVERRIDE/.nemo_rl_seeded'; fi"
RUNTIME_ENV="export NEMO_RL_VENV_DIR='$NEMO_RL_VENV_DIR'; export UV_CACHE_DIR='$UV_CACHE_DIR_OVERRIDE'; export NRL_FORCE_REBUILD_VENVS='$NRL_FORCE_REBUILD_VENVS'; export NRL_REFIT_BUFFER_MEMORY_RATIO='$NRL_REFIT_BUFFER_MEMORY_RATIO';"
if [[ -n "$PYTORCH_CUDA_ALLOC_CONF" ]]; then
    RUNTIME_ENV="$RUNTIME_ENV export PYTORCH_CUDA_ALLOC_CONF='$PYTORCH_CUDA_ALLOC_CONF';"
fi
TRAIN_COMMAND="uv run $RUN_SCRIPT --config $RECIPE $MAX_STEPS_KEY=$MAX_STEPS logger.log_dir=$LOG_DIR logger.wandb_enabled=$WANDB_ENABLED logger.wandb.project=$WANDB_PROJECT logger.wandb.name=$WANDB_NAME logger.monitor_gpus=$MONITOR_GPUS logger.tensorboard_enabled=$TENSORBOARD_ENABLED checkpointing.enabled=$CHECKPOINTING_ENABLED checkpointing.checkpoint_dir=$CKPT_DIR checkpointing.save_period=$CHECKPOINT_SAVE_PERIOD cluster.num_nodes=$NUM_NODES cluster.gpus_per_node=$GPUS_PER_NODE"
if [[ -n "$MODEL_NAME" ]]; then
    TRAIN_COMMAND="$TRAIN_COMMAND policy.model_name=$MODEL_NAME policy.tokenizer.name=$MODEL_NAME"
fi
if [[ -n "${FORCE_RECONVERT_FROM_HF:-}" ]]; then
    TRAIN_COMMAND="$TRAIN_COMMAND policy.megatron_cfg.force_reconvert_from_hf=$FORCE_RECONVERT_FROM_HF"
fi
if [[ -n "$EXTRA_OVERRIDES" ]]; then
    TRAIN_COMMAND="$TRAIN_COMMAND $EXTRA_OVERRIDES"
fi
COMMAND="$CACHE_SETUP; $RUNTIME_ENV $TRAIN_COMMAND"
if [[ "$DEEP_GEMM_PREFLIGHT" == "true" ]]; then
    DEEP_GEMM_PREFLIGHT_CODE="import torch, deep_gemm; missing=[name for name in ('get_paged_mqa_logits_metadata', 'fp8_paged_mqa_logits') if not callable(getattr(deep_gemm, name, None))]; assert not missing, f'DeepGEMM is missing required vLLM MLA symbols: {missing}'; print(f'DeepGEMM preflight OK: {deep_gemm.__file__}')"
    COMMAND="$CACHE_SETUP; $RUNTIME_ENV uv run --locked --extra vllm python -c \"$DEEP_GEMM_PREFLIGHT_CODE\" && $TRAIN_COMMAND"
fi

cd "$PROJECT_ROOT"

mkdir -p "$NEMO_RL_VENV_DIR" "$UV_CACHE_DIR_OVERRIDE"
if [[ -n "$(dirname "$SLURM_OUTPUT")" && "$(dirname "$SLURM_OUTPUT")" != "." ]]; then
    mkdir -p "$(dirname "$SLURM_OUTPUT")"
fi

if [[ -n "${DRYRUN:-}" ]]; then
    echo "[DRYRUN] CONTAINER=$CONTAINER"
    echo "[DRYRUN] ACCOUNT=$ACCOUNT"
    echo "[DRYRUN] PARTITION=$PARTITION"
    echo "[DRYRUN] HF_HOME=$HF_HOME"
    echo "[DRYRUN] HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
    if [[ -n "$MODEL_NAME" ]]; then
        echo "[DRYRUN] MODEL_NAME=$MODEL_NAME"
    else
        echo "[DRYRUN] MODEL_NAME=<recipe default>"
    fi
    echo "[DRYRUN] NUM_NODES=$NUM_NODES"
    echo "[DRYRUN] GPUS_PER_NODE=$GPUS_PER_NODE"
    echo "[DRYRUN] SLURM_COMMENT=$SLURM_COMMENT"
    echo "[DRYRUN] MOUNTS=$MOUNTS"
    echo "[DRYRUN] GLM51_CACHE_TAG=$GLM51_CACHE_TAG"
    echo "[DRYRUN] GLM51_SEED_UV_CACHE=$GLM51_SEED_UV_CACHE"
    echo "[DRYRUN] NEMO_RL_VENV_DIR=$NEMO_RL_VENV_DIR"
    echo "[DRYRUN] UV_CACHE_DIR_OVERRIDE=$UV_CACHE_DIR_OVERRIDE"
    echo "[DRYRUN] NRL_FORCE_REBUILD_VENVS=$NRL_FORCE_REBUILD_VENVS"
    echo "[DRYRUN] NRL_REFIT_BUFFER_MEMORY_RATIO=$NRL_REFIT_BUFFER_MEMORY_RATIO"
    echo "[DRYRUN] PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-<unset>}"
    echo "[DRYRUN] DEEP_GEMM_PREFLIGHT=$DEEP_GEMM_PREFLIGHT"
    echo "[DRYRUN] RUN_SCRIPT=$RUN_SCRIPT"
    echo "[DRYRUN] MAX_STEPS_KEY=$MAX_STEPS_KEY"
    if [[ -n "$HF_TOKEN" ]]; then
        echo "[DRYRUN] HF_TOKEN=<set>"
    else
        echo "[DRYRUN] HF_TOKEN=<unset>"
    fi
    echo "[DRYRUN] COMMAND=$COMMAND"
    printf "[DRYRUN] sbatch --nodes=%q --gres=%q --account=%q --job-name=%q --partition=%q --time=%q --output=%q --comment=%q ray.sub\n" \
        "$NUM_NODES" \
        "gpu:$GPUS_PER_NODE" \
        "$ACCOUNT" \
        "$JOB_NAME" \
        "$PARTITION" \
        "$WALLTIME" \
        "$SLURM_OUTPUT" \
        "$SLURM_COMMENT"
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
    --comment="$SLURM_COMMENT" \
    ray.sub

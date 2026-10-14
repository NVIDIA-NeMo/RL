#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# One node / four GPUs; five training steps by default.
# Export NVIDIA_API_KEY (judge) and optionally HF_TOKEN before submission.
# DRY_RUN=1 bash scripts/submit_qwen25_vl_vision2code.sh
# MAX_STEPS=100 bash scripts/submit_qwen25_vl_vision2code.sh --time=04:00:00
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# Optional user-owned shell file containing exports; never put keys in this repo.
if [[ -n "${VISION2CODE_ENV_FILE:-}" ]]; then
    source "$VISION2CODE_ENV_FILE"
fi

export QWEN_VL_MODEL="${QWEN_VL_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
export MM_TRAINER_DATA_PATH="${MM_TRAINER_DATA_PATH:-${REPO_DIR}/RL_data/rl_3000/tasks.jsonl}"
export VISION2CODE_SQSH="${VISION2CODE_SQSH:-${REPO_DIR}/containers/vision2code-render.sqsh}"
export VISION2CODE_JUDGE_URL="${VISION2CODE_JUDGE_URL:-https://inference-api.nvidia.com/v1}"
export VISION2CODE_JUDGE_MODEL="${VISION2CODE_JUDGE_MODEL:-nvidia/qwen/qwen3.8-flash-next}"
export VISION2CODE_JUDGE_KEY="${VISION2CODE_JUDGE_KEY:-${NVIDIA_API_KEY:-}}"
export MAX_STEPS="${MAX_STEPS:-5}"
export WANDB_ENABLED="${WANDB_ENABLED:-false}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export MM_TRAINER_WANDB_ID="${MM_TRAINER_WANDB_ID:-qwen25-vl-vision2code-$(date +%Y%m%d-%H%M%S)-${RANDOM}}"
export MM_TRAINER_RESULTS_DIR="${MM_TRAINER_RESULTS_DIR:-${REPO_DIR}/results/${MM_TRAINER_WANDB_ID}}"
export CONTAINER="${CONTAINER:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/rohitkumarj/data/enroot-containers/rl.nightly.sep30.2026.sqsh}"

# Shared downloads and remote-code imports. No model/cache copying is needed.
export HF_HOME="${HF_HOME:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/slikhite/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${HF_HOME}/modules}"
export NRL_MEGATRON_CHECKPOINT_DIR="${NRL_MEGATRON_CHECKPOINT_DIR:-${HF_HOME}/nemo_rl}"
export UV_CACHE_DIR_OVERRIDE="${UV_CACHE_DIR_OVERRIDE:-${REPO_DIR}/.cache/uv}"
export UV_LINK_MODE=copy

# Reuse the image's CUDA/PyTorch/Megatron/vLLM environments. Gym server venvs
# live separately and are keyed by the image, Gym revision, and manifests.
export UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv
export NEMO_RL_VENV_DIR=/opt/ray_venvs
export NRL_FORCE_REBUILD_VENVS=false
GYM_DIR="${REPO_DIR}/3rdparty/Gym-workspace/Gym"
export NEMO_GYM_EXTRA_ROOTS="$GYM_DIR"
export PYTHONPATH="${HF_MODULES_CACHE}:${REPO_DIR}:${GYM_DIR}:${REPO_DIR}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:${REPO_DIR}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM:${PYTHONPATH:-}"
export RAY_TMPDIR=/tmp
export ENROOT_ALLOW_SUPERUSER=yes
export ENROOT_MAX_PROCESSORS=4
export GPUS_PER_NODE=4
export DEDICATED_RAY_HEAD=0
export QWEN_VISION2CODE_CONFIG=examples/configs/recipes/vlm/vlm_grpo-qwen2.5-vl-3b-instruct-vision2code-1n4g-megatron.yaml

if [[ "${1:-}" == --run ]]; then
    mkdir -p "$MM_TRAINER_RESULTS_DIR" "$HF_HOME" "$HF_HUB_CACHE" \
        "$HF_DATASETS_CACHE" "$HF_MODULES_CACHE" "$NRL_MEGATRON_CHECKPOINT_DIR"
    # ray.sub unsets UV_CACHE_DIR on the host; restore it inside the container.
    export UV_CACHE_DIR="$UV_CACHE_DIR_OVERRIDE"
    # Use the existing training entrypoint; Gym manages its own server venvs.
    exec uv run --no-sync examples/run_vlm_grpo.py --config "$QWEN_VISION2CODE_CONFIG"
fi

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
[[ "$MAX_STEPS" =~ ^[1-9][0-9]*$ ]] || fail 'MAX_STEPS must be a positive integer'
[[ "$WANDB_ENABLED" =~ ^(true|false)$ ]] || fail 'WANDB_ENABLED must be true or false'
for path in "$CONTAINER" "$VISION2CODE_SQSH" "$MM_TRAINER_DATA_PATH"; do
    [[ "$path" == /* && -r "$path" ]] || fail "Expected an absolute, readable file: $path"
done
for path in "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$HF_MODULES_CACHE" \
    "$NRL_MEGATRON_CHECKPOINT_DIR" "$UV_CACHE_DIR_OVERRIDE" "$MM_TRAINER_RESULTS_DIR"; do
    [[ "$path" == /lustre/* && "$path" != *[[:space:],:]* ]] || fail "Cache/results path must be under /lustre without spaces, colons or commas: $path"
done
[[ "$VISION2CODE_JUDGE_URL" == https://*/v1 || "$VISION2CODE_JUDGE_URL" == http://*/v1 ]] || fail 'Judge URL must end in /v1'

if [[ "${DRY_RUN:-0}" != 1 ]]; then
    [[ -n "$VISION2CODE_JUDGE_KEY" ]] || fail 'Export NVIDIA_API_KEY or VISION2CODE_JUDGE_KEY for the vision judge'
fi
if [[ "$WANDB_ENABLED" == true && "$WANDB_MODE" == online && "${DRY_RUN:-0}" != 1 ]]; then
    [[ -n "${WANDB_API_KEY:-}" ]] || fail 'Export WANDB_API_KEY, or use WANDB_MODE=offline'
fi

cache_key=$(
    {
        stat -Lc '%n %s %Y' "$CONTAINER"
        git -C "$GYM_DIR" rev-parse HEAD
        sha256sum "$GYM_DIR/pyproject.toml" "$GYM_DIR/uv.lock" \
            "$GYM_DIR/resources_servers/vision2code/requirements.txt" \
            "$GYM_DIR/responses_api_agents/simple_agent/requirements.txt" \
            "$GYM_DIR/responses_api_models/vllm_model/pyproject.toml"
    } | sha256sum | cut -c1-20
)
export MM_TRAINER_GYM_VENV_DIR="${REPO_DIR}/.cache/vision2code/${cache_key}/gym_venvs"

# The training image has unsquashfs but lacks Enroot. Bind the host shell
# runtime and its three static helpers; do not mount host /etc or site hooks.
export MOUNTS=/lustre:/lustre
for path in /usr/bin/enroot /usr/bin/enroot-nsenter /usr/bin/enroot-mount /usr/bin/enroot-switchroot /usr/lib/enroot /usr/bin/column; do
    [[ -r "$path" ]] || fail "Missing host Enroot runtime: $path"
    MOUNTS+=",${path}:${path}:ro"
done

export COMMAND='bash scripts/submit_qwen25_vl_vision2code.sh --run'
SBATCH_ARGS=(
    --account="${SBATCH_ACCOUNT:-nemotron_n4_omni}"
    --partition="${SBATCH_PARTITION:-batch}"
    --nodes=1 --gres=gpu:4
    --time="${SBATCH_TIME:-01:00:00}"
    --job-name="$MM_TRAINER_WANDB_ID"
    --export=ALL
)
for arg in "$@"; do
    case "$arg" in
        --nodes*|-N*|--gres*|--gpus*) fail 'This launcher requires exactly one node and four GPUs' ;;
    esac
done
printf 'Model: %s\nTasks: %s\nContainer: %s\n' "$QWEN_VL_MODEL" "$MM_TRAINER_DATA_PATH" "$CONTAINER"
printf 'HF home: %s\nHF modules: %s\nGym environments: %s\n' "$HF_HOME" "$HF_MODULES_CACHE" "$MM_TRAINER_GYM_VENV_DIR"
printf 'Results: %s\nSteps: %s\n' "$MM_TRAINER_RESULTS_DIR" "$MAX_STEPS"
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf 'sbatch '
    printf '%q ' "${SBATCH_ARGS[@]}" "$@" ray.sub
    printf '\nDriver: %s\n' "$COMMAND"
    exit 0
fi
exec sbatch "${SBATCH_ARGS[@]}" "$@" ray.sub

#!/usr/bin/env bash
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

# Edit the settings below, then: bash scripts/submit_vlm_grpo.sh
# Preview without submitting: DRY_RUN=1 bash scripts/submit_vlm_grpo.sh
# Extra arguments go to sbatch, e.g. bash scripts/submit_vlm_grpo.sh --time=08:00:00
# This wraps ray.sub, which starts Ray on every node and runs one driver.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

# Optional user-owned shell file containing exports; never put keys in this repo.
if [[ -n "${VISION2CODE_ENV_FILE:-}" ]]; then
    source "$VISION2CODE_ENV_FILE"
fi

# Set these here or export them before running this script. The model must be
# a SuperVL3p5 HF checkpoint directory containing chat_template.jinja. The data
# is prepared by Gym's resources_servers.vision2code.prepare; add
# agent_ref={type: responses_api_agents, name: vision2code_simple_agent} per row.
export MM_TRAINER_MODEL_PATH="${MM_TRAINER_MODEL_PATH:-/lustre/fsw/portfolios/llmservice/users/asohrabizade/workspace/output/super35_GAv1_1006_n32-b512_7e-5-32k_vision2code_chem-ece-doc-clevr-batch1-version2-rep6-75k/checkpoints/tp_1_hf/iter_0000066/canonical/hf}"
export MM_TRAINER_DATA_PATH="${MM_TRAINER_DATA_PATH:-${REPO_DIR}/RL_data/rl_3000/tasks.jsonl}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_MODE="${WANDB_MODE:-online}"

# Vision2code renders policy-generated Python in a separate Enroot image and
# scores the result with an existing vision-capable judge endpoint (/v1).
export VISION2CODE_SQSH="${VISION2CODE_SQSH:-${REPO_DIR}/containers/vision2code-render.sqsh}"
export VISION2CODE_JUDGE_URL="${VISION2CODE_JUDGE_URL:-https://inference-api.nvidia.com/v1}"
export VISION2CODE_JUDGE_MODEL="${VISION2CODE_JUDGE_MODEL:-nvidia/qwen/qwen3.8-flash-next}"
export VISION2CODE_JUDGE_KEY="${VISION2CODE_JUDGE_KEY:-${NVIDIA_API_KEY:-}}"

# nemotron_n4_omni is treated as the Slurm account; batch is the partition.
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-nemotron_n4_omni}"
SBATCH_PARTITION="${SBATCH_PARTITION:-batch}"
SBATCH_TIME="${SBATCH_TIME:-04:00:00}"
export CONTAINER="${CONTAINER:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/rohitkumarj/data/enroot-containers/rl.nightly.sep30.2026.sqsh}"
export MOUNTS="${MOUNTS:-/lustre:/lustre}"

export MM_TRAINER_WANDB_ENTITY="${MM_TRAINER_WANDB_ENTITY:-nvidia}"
export MM_TRAINER_WANDB_PROJECT="${MM_TRAINER_WANDB_PROJECT:-vision2code-grpo-sl}"
export MM_TRAINER_WANDB_ID="${MM_TRAINER_WANDB_ID:-vision2code-$(date +%Y%m%d-%H%M%S)-${RANDOM}}"
export MM_TRAINER_WANDB_NAME="${MM_TRAINER_WANDB_NAME:-${MM_TRAINER_WANDB_ID}}"
export MM_TRAINER_RESULTS_DIR="${MM_TRAINER_RESULTS_DIR:-${REPO_DIR}/results/${MM_TRAINER_WANDB_ID}}"
# Reuse Hugging Face downloads and custom model code across runs. These paths
# must be shared and mounted at the same location on every node.
export HF_HOME="${HF_HOME:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/slikhite/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${HF_HOME}/modules}"
export NRL_MEGATRON_CHECKPOINT_DIR="${NRL_MEGATRON_CHECKPOINT_DIR:-${HF_HOME}/nemo_rl}"
export UV_CACHE_DIR_OVERRIDE="${UV_CACHE_DIR_OVERRIDE:-${REPO_DIR}/.cache/uv}"
export UV_LINK_MODE=copy
export MM_TRAINER_MEDIA_ROOT="${MM_TRAINER_MEDIA_ROOT:-/lustre}"

# Reuse the same image-baked training/worker environments as the Qwen launcher.
export UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv
export NEMO_RL_VENV_DIR=/opt/ray_venvs
export NRL_FORCE_REBUILD_VENVS="${NRL_FORCE_REBUILD_VENVS:-false}"
export RAY_TMPDIR=/tmp
export ENROOT_ALLOW_SUPERUSER=yes
export ENROOT_MAX_PROCESSORS=4
# Match the recipe: 16 training + 16 rollout nodes, each with 4 GPUs.
export GPUS_PER_NODE=4
export DEDICATED_RAY_HEAD=0
GYM_DIR="${REPO_DIR}/3rdparty/Gym-workspace/Gym"
export NEMO_GYM_EXTRA_ROOTS="${GYM_DIR}${NEMO_GYM_EXTRA_ROOTS:+:${NEMO_GYM_EXTRA_ROOTS}}"
# Ray workers need the custom-model cache on sys.path before unpickling.
export PYTHONPATH="${HF_MODULES_CACHE}:${REPO_DIR}:${NEMO_GYM_EXTRA_ROOTS}:${REPO_DIR}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:${REPO_DIR}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM:${PYTHONPATH:-}"

if [[ "${1:-}" == --run ]]; then
    mkdir -p "$MM_TRAINER_RESULTS_DIR" "$HF_HOME" "$HF_HUB_CACHE" \
        "$HF_DATASETS_CACHE" "$HF_MODULES_CACHE" "$NRL_MEGATRON_CHECKPOINT_DIR"
    # ray.sub unsets UV_CACHE_DIR on the host; restore it in the container.
    export UV_CACHE_DIR="$UV_CACHE_DIR_OVERRIDE"
    exec uv run --no-sync examples/run_grpo_single_controller.py \
        --config configs/examples/recipes/vlm/vlm_grpo-supervl3p5-unified-teachers-ready-first-32n4g.yaml
fi

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
for arg in "$@"; do
    case "$arg" in
        --nodes*|-N*|--gres*|--gpus*) fail 'This recipe requires exactly 32 nodes and four GPUs per node' ;;
    esac
done

: "${MM_TRAINER_MODEL_PATH:?Set MM_TRAINER_MODEL_PATH to your HF checkpoint directory}"
: "${MM_TRAINER_DATA_PATH:?Set MM_TRAINER_DATA_PATH to your NeMo Gym training dataset}"
: "${VISION2CODE_SQSH:?Set VISION2CODE_SQSH to the vision2code renderer image (not the training image)}"
: "${VISION2CODE_JUDGE_URL:?Set VISION2CODE_JUDGE_URL to the judge endpoint including /v1}"
: "${VISION2CODE_JUDGE_MODEL:?Set VISION2CODE_JUDGE_MODEL to the served judge model name}"
if [[ "${DRY_RUN:-0}" != 1 ]]; then
    : "${VISION2CODE_JUDGE_KEY:?Set NVIDIA_API_KEY or VISION2CODE_JUDGE_KEY for the judge endpoint}"
    if [[ "${WANDB_MODE}" == "online" ]]; then
        : "${WANDB_API_KEY:?Set WANDB_API_KEY, or set WANDB_MODE=offline}"
    fi
fi
while [[ "${MM_TRAINER_MODEL_PATH}" == */ && "${MM_TRAINER_MODEL_PATH}" != / ]]; do
    MM_TRAINER_MODEL_PATH="${MM_TRAINER_MODEL_PATH%/}"
done
if [[ ! -r "${MM_TRAINER_MODEL_PATH}/chat_template.jinja" ]]; then
    echo "Missing chat template: ${MM_TRAINER_MODEL_PATH}/chat_template.jinja" >&2
    exit 1
fi
if [[ ! -f "${MM_TRAINER_DATA_PATH}" || ! -r "${MM_TRAINER_DATA_PATH}" ]]; then
    echo "Tasks JSONL is not readable: ${MM_TRAINER_DATA_PATH}" >&2
    echo "Use Gym's resources_servers.vision2code.prepare, then add vision2code_simple_agent routing." >&2
    exit 1
fi
if [[ "${VISION2CODE_SQSH}" != /*.sqsh || ! -r "${VISION2CODE_SQSH}" ]]; then
    echo "VISION2CODE_SQSH must be an absolute, readable renderer .sqsh path" >&2
    exit 1
fi
[[ "$CONTAINER" == /* && -r "$CONTAINER" ]] || fail "Expected an absolute, readable training image: $CONTAINER"
[[ "$VISION2CODE_JUDGE_URL" == https://*/v1 || "$VISION2CODE_JUDGE_URL" == http://*/v1 ]] || fail 'Judge URL must end in /v1'

# Key Gym server environments exactly as in the working Qwen launcher. This
# avoids reusing stale image-baked Gym dependencies and shares compatible venvs.
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
export MM_TRAINER_GYM_VENV_DIR="${MM_TRAINER_GYM_VENV_DIR:-${REPO_DIR}/.cache/vision2code/${cache_key}/gym_venvs}"
for path in "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$HF_MODULES_CACHE" \
    "$NRL_MEGATRON_CHECKPOINT_DIR" "$UV_CACHE_DIR_OVERRIDE" "$MM_TRAINER_RESULTS_DIR" \
    "$MM_TRAINER_GYM_VENV_DIR"; do
    [[ "$path" == /lustre/* && "$path" != *[[:space:],:]* ]] || fail "Cache/results path must be under /lustre without spaces, colons or commas: $path"
done

# The training image lacks Enroot. Mount the host runtime and static helpers,
# as in the working Qwen job; Vision2Code supplies its own isolated config.
for path in /usr/bin/enroot /usr/bin/enroot-nsenter /usr/bin/enroot-mount /usr/bin/enroot-switchroot /usr/lib/enroot /usr/bin/column; do
    [[ -r "$path" ]] || fail "Missing host Enroot runtime: $path"
    MOUNTS+=",${path}:${path}:ro"
done

# Keep credentials in the exported environment, outside the logged COMMAND.
export COMMAND='bash scripts/submit_vlm_grpo.sh --run'

SBATCH_ARGS=(
    --account="${SBATCH_ACCOUNT}"
    --partition="${SBATCH_PARTITION}"
    --nodes=32
    --gres=gpu:4
    --time="${SBATCH_TIME}"
    --job-name="${MM_TRAINER_WANDB_NAME}"
    --export=ALL
)
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf 'Repository: %s\nContainer: %s\n' "${REPO_DIR}" "${CONTAINER}"
    printf 'Model: %s\n' "${MM_TRAINER_MODEL_PATH}"
    printf 'Tasks: %s\nRenderer: %s\n' "${MM_TRAINER_DATA_PATH}" "${VISION2CODE_SQSH}"
    printf 'HF home: %s\nHF modules cache: %s\n' "${HF_HOME}" "${HF_MODULES_CACHE}"
    printf 'Gym environments: %s\nResults: %s\n' "${MM_TRAINER_GYM_VENV_DIR}" "${MM_TRAINER_RESULTS_DIR}"
    printf 'Judge: %s at %s\n' "${VISION2CODE_JUDGE_MODEL}" "${VISION2CODE_JUDGE_URL}"
    printf 'sbatch '
    printf '%q ' "${SBATCH_ARGS[@]}" "$@" ray.sub
    printf '\nDriver command:\n%s\n' "${COMMAND}"
    exit 0
fi
exec sbatch "${SBATCH_ARGS[@]}" "$@" ray.sub

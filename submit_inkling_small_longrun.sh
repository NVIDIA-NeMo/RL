#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
DEPENDENCY=()
SUBMIT_OPTIONS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--dependent) DEPENDENCY=("--dependency=afterany:${JOBID:?Set JOBID for a dependent run}") ;;
        --hold) SUBMIT_OPTIONS+=(--hold) ;;
        *) echo "Usage: $0 [-d|--dependent] [--hold]" >&2; exit 1 ;;
    esac
    shift
done

WORK_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RUN=grpo-inkling-small-16n4g-fsdp2-automodel-ep64
MODEL_DIR=${MODEL_DIR:-/lustre/fsw/general_sa/shuangy/models/thinkingmachines/Inkling-Small}
CONTAINER=${CONTAINER:-/lustre/fsw/general_sa/shuangy/images/nemo-rl-inkling-pr3566-tf514-20260916-v5.sqsh}
PROJECT=${PROJECT:-nemo-rl-Inkling}
ENTITY=${ENTITY:-nv-welcome}
CKPTDIR=${CKPTDIR:-$WORK_DIR/results/${PROJECT}_${RUN}}
[[ -f "$CONTAINER" && -f "$MODEL_DIR/model.safetensors.index.json" ]]
set +x
CREDENTIALS_FILE=${NRL_CREDENTIALS_FILE:-$HOME/.env}
if [[ -f "$CREDENTIALS_FILE" ]]; then
    source "$CREDENTIALS_FILE"
fi
if [[ "$ENTITY" == nv-welcome && -n "${WANDB_API_KEY_NVWELCOME:-}" ]]; then
    export WANDB_API_KEY="$WANDB_API_KEY_NVWELCOME"
fi
export WANDB_API_KEY HF_TOKEN
export PYTHONPATH="$WORK_DIR"
unset NRL_IGNORE_VERSION_MISMATCH
export HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 ENROOT_ROOTFS_WRITABLE=1
export GPUS_PER_NODE=4 RAY_LOG_SYNC_FREQUENCY=10
export UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv UV_NO_SYNC=1
export HF_HOME=/lustre/fsw/general_sa/shuangy/src/NeMo-RL/hf
export HF_DATASETS_CACHE=/lustre/fsw/general_sa/shuangy/src/NeMo-RL/hf_datasets
cd "$WORK_DIR"
DRIVER=(uv run examples/run_grpo.py --config "examples/configs/recipes/llm/$RUN.yaml")
if [[ "${INKLING_VALIDATION:-0}" == 1 ]]; then
    DRIVER=(bash "tests/test_suites/llm/$RUN.sh")
    CKPTDIR=${VALIDATION_CKPTDIR:-$WORK_DIR/results/${PROJECT}_${RUN}_validation}
fi
printf -v COMMAND '%q ' "${DRIVER[@]}" \
    "policy.model_name=$MODEL_DIR" "policy.tokenizer.name=$MODEL_DIR" \
    "checkpointing.enabled=True" "checkpointing.checkpoint_dir=$CKPTDIR" \
    "logger.wandb_enabled=True" "logger.wandb.entity=$ENTITY" \
    "logger.wandb.project=$PROJECT" "logger.wandb.name=$RUN"
export COMMAND CONTAINER
export MOUNTS="/lustre:/lustre,$WORK_DIR:/opt/nemo-rl"
sbatch --nodes=16 --ntasks-per-node=1 --exclusive --account=general_sa \
    --partition=batch,tcpo,36x2-a01r --time=03:59:00 \
    --job-name="$RUN" "${DEPENDENCY[@]}" "${SUBMIT_OPTIONS[@]}" ray.sub

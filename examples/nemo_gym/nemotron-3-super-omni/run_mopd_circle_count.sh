#!/usr/bin/env bash
set -euo pipefail

# Thin MOPD wrapper around the shared Super Omni launcher. The launcher
# validates MODEL_PATH, TRAIN_PATH, CONTAINER, SANDBOX_CONTAINER,
# PERSISTENT_CACHE, SLURM_ACCOUNT, and WANDB_API_KEY (for online logging).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXP_NAME="${EXP_NAME:-mopd-super-omni-circle-count}"
export CONFIG_PATH="${CONFIG_PATH:-examples/configs/recipes/vlm/mopd-nemotron-super-omni-120ba12b-10n8g-megatron-tp8ep16cp2-async-gym.v1.yaml}"
export WANDB_PROJ="${WANDB_PROJ:-mopd-nemotron-super-omni}"

if [[ -n "${TEACHER_MODEL_PATH:-}" ]]; then
    while [[ "${TEACHER_MODEL_PATH}" == */ && "${TEACHER_MODEL_PATH}" != "/" ]]; do
        TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH%/}"
    done
    teacher_override="on_policy_distillation.teacher_model_by_agent_name.circle_count_simple_agent=${TEACHER_MODEL_PATH}"
    export EXTRA_HYDRA_ARGS="${EXTRA_HYDRA_ARGS:+${EXTRA_HYDRA_ARGS} }${teacher_override}"
fi

exec "${SCRIPT_DIR}/super_omni_launch.sh"

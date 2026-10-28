#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# DRY_RUN=1 validates assets and prints the submission without allocating nodes.
# Reuse EXP_NAME with RESUME=1 only after the previous allocation has ended,
# or queue with SLURM_DEPENDENCY=afterany:<previous-job> so starts cannot overlap.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
NEMO_RL_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
CONFIG="${NEMO_RL_ROOT}/examples/configs/recipes/vlm/vlm_grpo-supervl3p5-caprl-video-32n4g-megatron-prefix-smoke.yaml"
export CAPRL_EXP_NAME="${EXP_NAME:-supervl35-caprl-prefix-smoke-$(date +%Y%m%d-%H%M%S)}"
export CAPRL_RUN_DIR="${RESULTS_ROOT:-${NEMO_RL_ROOT}/results}/${CAPRL_EXP_NAME}"
export CAPRL_CONVERSION_CACHE="${CAPRL_CONVERSION_CACHE:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/rohitkumarj/rem/unified-teacher-supervl3p5/cache/nemo-rl-omni/megatron-checkpoints-super-vl-35-unified-final-ln-v2}"
CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_ci/nemo_rl_ci/sqsh_files/rl-gym.75221028.sqsh}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-nemotron_sw_post}"
SLURM_PARTITION="${SLURM_PARTITION:-batch}"
SLURM_QOS="${SLURM_QOS:-normal}"
WALLTIME="${WALLTIME:-1:00:00}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://forge.coreweave.com/api/wandb}"
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  export WANDB_MODE="${WANDB_MODE:-offline}"
fi

[[ -f "${CONTAINER}" ]] || { echo "Missing container: ${CONTAINER}" >&2; exit 1; }
[[ "${CAPRL_EXP_NAME}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "Invalid EXP_NAME" >&2; exit 1; }
if [[ "${RESUME:-0}" == 1 ]]; then
  [[ -d "${CAPRL_RUN_DIR}/checkpoints" ]] || { echo "No checkpoint directory to resume" >&2; exit 1; }
elif [[ -d "${CAPRL_RUN_DIR}" ]]; then
  echo "Run directory exists; use RESUME=1 or a new EXP_NAME: ${CAPRL_RUN_DIR}" >&2
  exit 1
fi

# Only lightweight config dependencies are needed on the submit host.
ALLOCATION="$(uv run --no-project --python "${HOST_PYTHON:-3.13}" \
  --with omegaconf --with hydra-core python "${SCRIPT_DIR}/preflight_caprl.py" \
  --allocation --config "${CONFIG}" "$@")"
read -r NUM_NODES GPUS_PER_NODE <<< "${ALLOCATION}"
[[ "${NUM_NODES}" =~ ^[0-9]+$ && "${GPUS_PER_NODE}" =~ ^[0-9]+$ ]] || { echo "Invalid preflight allocation: ${ALLOCATION}" >&2; exit 1; }

export CONTAINER GPUS_PER_NODE
export MOUNTS="/lustre:/lustre,${NEMO_RL_ROOT}:/opt/nemo-rl,${CAPRL_CONVERSION_CACHE}:/opt/caprl-initial-cache:ro"
export BASE_LOG_DIR="${CAPRL_RUN_DIR}"
export RAY_LOG_SYNC_FREQUENCY=60
export NEMO_RL_VENV_DIR=/opt/ray_venvs
export NRL_IGNORE_VERSION_MISMATCH=1
export NRL_FORCE_REBUILD_VENVS=false
export UV_NO_SYNC=1
export NEMO_RL_VIDEO_MEDIA_ROOT=/lustre
export SETUP_COMMAND='bash /opt/nemo-rl/examples/nemo_gym/supervl3p5/scripts/caprl_runtime.sh setup'
printf -v COMMAND 'bash /opt/nemo-rl/examples/nemo_gym/supervl3p5/scripts/caprl_runtime.sh'
for override in "$@"; do
  printf -v COMMAND '%s %q' "${COMMAND}" "${override}"
done
export COMMAND

SUBMIT=(sbatch --parsable --chdir="${NEMO_RL_ROOT}" --nodes="${NUM_NODES}" \
  --gres="gpu:${GPUS_PER_NODE}" --account="${SLURM_ACCOUNT}" \
  --partition="${SLURM_PARTITION}" --qos="${SLURM_QOS}" --time="${WALLTIME}" \
  --job-name="${CAPRL_EXP_NAME}" --output="${CAPRL_RUN_DIR}/slurm-%j.out" \
  --error="${CAPRL_RUN_DIR}/slurm-%j.err" --export=ALL)
if [[ -n "${SLURM_DEPENDENCY:-}" ]]; then
  SUBMIT+=(--dependency="${SLURM_DEPENDENCY}")
fi
SUBMIT+=("${NEMO_RL_ROOT}/ray.sub")
echo "Run: ${CAPRL_EXP_NAME}; nodes=${NUM_NODES}; GPUs/node=${GPUS_PER_NODE}; walltime=${WALLTIME}"
echo "Checkpoint root: ${CAPRL_RUN_DIR}/checkpoints"
echo "Recipe defaults: five steps, media dedup OFF, prefix snapshots 120s, save every trainer step, retain three; overrides below take precedence."
printf 'Effective overrides: '; printf '%q ' "$@"; printf '\n'
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf 'DRY RUN: '; printf '%q ' "${SUBMIT[@]}"; printf '\n'
  echo "Driver command: ${COMMAND}"
  exit 0
fi

mkdir -p "${CAPRL_RUN_DIR}"
JOB_ID="$("${SUBMIT[@]}")"
echo "Submitted job ${JOB_ID}; logs: ${CAPRL_RUN_DIR}/${JOB_ID%%;*}-logs/ray-driver.log"

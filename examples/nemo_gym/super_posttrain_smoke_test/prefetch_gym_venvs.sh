#!/usr/bin/env bash
set -euo pipefail

: "${CONFIG_PATH:?CONFIG_PATH is required}"
: "${CONTAINER:?CONTAINER is required}"
: "${PERSISTENT_CACHE:?PERSISTENT_CACHE is required}"
: "${SLURM_PARTITION:?SLURM_PARTITION is required}"
: "${SLURM_ACCOUNT:?SLURM_ACCOUNT is required}"

CODE_DIR=$(realpath "$PWD")
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
GYM_VENV_DIR="${GYM_VENV_DIR:-${PERSISTENT_CACHE}/gym_venvs}"
RAY_LOG_DIR="${PREFETCH_RAY_LOG_DIR:-/tmp/gym_prefetch_ray_${USER:-user}}"
PREFETCH_TIME_LIMIT="${PREFETCH_TIME_LIMIT:-2:0:0}"
PREFETCH_GRES="${PREFETCH_GRES:-gpu:8}"
PREFETCH_NODES="${PREFETCH_NODES:-1}"
PREFETCH_MOUNTS="${PREFETCH_MOUNTS:-/lustre:/lustre}"
PREFETCH_JOB_NAME="${PREFETCH_JOB_NAME:-${EXP_NAME:-nemo-rl}-gym-venv}"

mkdir -p "${GYM_VENV_DIR}"

echo "========================================"
echo " Gym venv prefetch"
echo " Config     : ${CONFIG_PATH}"
echo " Venv dir   : ${GYM_VENV_DIR}"
echo " Ray logs   : ${RAY_LOG_DIR}"
echo " Container  : ${CONTAINER}"
echo " Partition  : ${SLURM_PARTITION}"
echo " Account    : ${SLURM_ACCOUNT}"
echo " Nodes      : ${PREFETCH_NODES}"
echo " Time       : ${PREFETCH_TIME_LIMIT}"
echo "========================================"

srun \
  --nodes="${PREFETCH_NODES}" \
  --ntasks=1 \
  --gres="${PREFETCH_GRES}" \
  --account="${SLURM_ACCOUNT}" \
  --partition="${SLURM_PARTITION}" \
  --time="${PREFETCH_TIME_LIMIT}" \
  --job-name="${PREFETCH_JOB_NAME}" \
  --export=ALL \
  --no-container-mount-home \
  --container-image="${CONTAINER}" \
  --container-workdir="${CODE_DIR}" \
  --container-mounts="${PREFETCH_MOUNTS}" \
  bash -lc "export NEMO_RL_REPO='${CODE_DIR}'; export NEMO_GYM_VENV_DIR='${GYM_VENV_DIR}'; export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0; uv run --extra nemo_gym python '${SCRIPT_DIR}/prefetch_gym_venvs.py' --config '${CONFIG_PATH}' --venv-dir '${GYM_VENV_DIR}' --log-dir '${RAY_LOG_DIR}'"

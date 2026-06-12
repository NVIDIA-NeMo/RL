#!/usr/bin/env bash
set -euo pipefail

: "${CONTAINER:?CONTAINER is required}"
: "${PERSISTENT_CACHE:?PERSISTENT_CACHE is required}"
: "${SLURM_PARTITION:?SLURM_PARTITION is required}"
: "${SLURM_ACCOUNT:?SLURM_ACCOUNT is required}"

CODE_DIR=$(realpath "$PWD")
NEMO_RL_VENV_DIR="${NEMO_RL_VENV_DIR:-${PERSISTENT_CACHE}/ray_venvs}"
UV_CACHE_DIR_OVERRIDE="${UV_CACHE_DIR_OVERRIDE:-${PERSISTENT_CACHE}/uv_cache}"
UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
PREFETCH_TIME_LIMIT="${PREFETCH_TIME_LIMIT:-1:0:0}"
PREFETCH_GRES="${PREFETCH_GRES:-gpu:1}"
PREFETCH_NODES="${PREFETCH_NODES:-1}"
PREFETCH_MOUNTS="${PREFETCH_MOUNTS:-/lustre:/lustre}"
PREFETCH_JOB_NAME="${PREFETCH_JOB_NAME:-${EXP_NAME:-nemo-rl}-ray-venv}"
RAY_ACTOR_VENV_NAME="${RAY_ACTOR_VENV_NAME:-nemo_rl.environments.nemo_gym.NemoGym}"

mkdir -p "${NEMO_RL_VENV_DIR}" "${UV_CACHE_DIR_OVERRIDE}"

echo "========================================"
echo " Ray actor venv prefetch"
echo " Code dir   : ${CODE_DIR}"
echo " Venv dir   : ${NEMO_RL_VENV_DIR}"
echo " UV cache   : ${UV_CACHE_DIR_OVERRIDE}"
echo " Container  : ${CONTAINER}"
echo " Partition  : ${SLURM_PARTITION}"
echo " Account    : ${SLURM_ACCOUNT}"
echo " Nodes      : ${PREFETCH_NODES}"
echo " Time       : ${PREFETCH_TIME_LIMIT}"
echo " GRES       : ${PREFETCH_GRES}"
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
  bash -lc "set -euo pipefail; \
    export UV_CACHE_DIR='${UV_CACHE_DIR_OVERRIDE}'; \
    export UV_HTTP_TIMEOUT='${UV_HTTP_TIMEOUT}'; \
    export NEMO_RL_VENV_DIR='${NEMO_RL_VENV_DIR}'; \
    VENV_PATH='${NEMO_RL_VENV_DIR}/${RAY_ACTOR_VENV_NAME}'; \
    uv venv --allow-existing \"\${VENV_PATH}\"; \
    export UV_PROJECT_ENVIRONMENT=\"\${VENV_PATH}\"; \
    uv sync --directory '${CODE_DIR}'; \
    uv run --locked --extra nemo_gym --directory '${CODE_DIR}' echo \"Finished creating venv \${VENV_PATH}\"; \
    test -x \"\${VENV_PATH}/bin/python\"; \
    \"\${VENV_PATH}/bin/python\" -c \"import nemo_gym; import nemo_rl.environments.nemo_gym; print('Ray actor venv import check OK')\"; \
    echo \"Ray actor venv prefetch complete: \${VENV_PATH}\""

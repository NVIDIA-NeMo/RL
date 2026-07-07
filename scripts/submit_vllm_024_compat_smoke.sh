#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:?set REPO_DIR to the NeMo-RL checkout}"
CONTAINER="${CONTAINER:?set CONTAINER to the NeMo-RL sqsh image}"
ACCOUNT="${ACCOUNT:-coreai_dlalgo_llm}"
PARTITION="${PARTITION:-batch}"
SEGMENT="${SEGMENT:-1}"
WALLTIME="${WALLTIME:-01:00:00}"
JOB_NAME="${JOB_NAME:-coreai_dlalgo_llm-nemorl.vllm024-smoke}"
LOG_DIR="${LOG_DIR:-${REPO_DIR}/experiments/vllm_024_upgrade/logs}"
VENV_DIR="${VENV_DIR:-${REPO_DIR}/venvs/vllm024-smoke}"
UV_CACHE_DIR="${UV_CACHE_DIR:-${REPO_DIR}/.cache/uv}"
HF_HOME="${HF_HOME:-/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home}"
ENGINE_SMOKE_MODEL="${ENGINE_SMOKE_MODEL:-}"
DRY_RUN="${DRY_RUN:-false}"

SRUN_COMMAND=(
  srun
  --nodes=1
  --ntasks=1
  --no-container-mount-home
  --container-image="${CONTAINER}"
  --container-mounts=/lustre:/lustre
  --container-workdir="${REPO_DIR}"
  env
  "REPO_DIR=${REPO_DIR}"
  "VENV_DIR=${VENV_DIR}"
  "UV_CACHE_DIR=${UV_CACHE_DIR}"
  "HF_HOME=${HF_HOME}"
  "ENGINE_SMOKE_MODEL=${ENGINE_SMOKE_MODEL}"
  bash "${REPO_DIR}/scripts/run_vllm_024_compat_smoke.sh"
)
printf -v WRAPPED_COMMAND '%q ' "${SRUN_COMMAND[@]}"
WRAPPED_COMMAND="${WRAPPED_COMMAND% }"

SBATCH_ARGS=(
  --account="${ACCOUNT}"
  --partition="${PARTITION}"
  --nodes=1
  --ntasks-per-node=1
  --cpus-per-task=32
  --mem=0
  --time="${WALLTIME}"
  --segment="${SEGMENT}"
  --job-name="${JOB_NAME}"
  --output="${LOG_DIR}/slurm-%j.out"
  --comment=metrics
)

if [[ "${DRY_RUN}" == "true" ]]; then
  printf '[DRY-RUN] sbatch'
  printf ' %q' "${SBATCH_ARGS[@]}"
  printf ' --wrap '
  printf '%s' "${WRAPPED_COMMAND}"
  printf '\n'
  exit 0
fi

mkdir -p "${LOG_DIR}" "${VENV_DIR}" "${UV_CACHE_DIR}"
sbatch --parsable "${SBATCH_ARGS[@]}" --wrap "${WRAPPED_COMMAND}"

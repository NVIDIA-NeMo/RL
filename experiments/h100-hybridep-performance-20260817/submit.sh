#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)
MATRIX=${MATRIX:-${SCRIPT_DIR}/matrix.tsv}
BASELINE_COMMIT=${BASELINE_COMMIT:-4a1454bf430624786251d14ba0197169c8e68a5c}
SBATCH_PARTITION=${SBATCH_PARTITION:-batch}
SBATCH_TIME=${SBATCH_TIME:-04:00:00}
GPUS_PER_NODE=8

usage() {
  echo "Usage: $0 <run-id> [--test-only]" >&2
  echo "       $0 --list" >&2
}

if [[ ${1:-} == "--list" ]]; then
  column -t -s $'\t' "${MATRIX}"
  exit 0
fi

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
  exit 2
fi

run_id=$1
test_only=${2:-}
if [[ -n ${test_only} && ${test_only} != "--test-only" ]]; then
  usage
  exit 2
fi

: "${SBATCH_ACCOUNT:?Set SBATCH_ACCOUNT to the scheduler account}"
: "${CONTAINER:?Set CONTAINER to an immutable .sqsh path}"
: "${MOUNTS:?Set MOUNTS to the shared filesystem mounts}"
: "${HF_HOME:?Set HF_HOME to the shared Hugging Face cache}"
: "${RUN_ROOT:?Set RUN_ROOT to an absolute shared-storage result directory}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY without writing it to this script}"
: "${WANDB_PROJECT:?Set WANDB_PROJECT to the destination W&B project}"

if [[ ${RUN_ROOT} != /* ]]; then
  echo "RUN_ROOT must be an absolute path" >&2
  exit 2
fi
if [[ ! -f ${CONTAINER} ]]; then
  echo "CONTAINER does not exist: ${CONTAINER}" >&2
  exit 2
fi
if ! git -C "${PROJECT_ROOT}" cat-file -e "${BASELINE_COMMIT}^{commit}"; then
  echo "BASELINE_COMMIT is unavailable: ${BASELINE_COMMIT}" >&2
  exit 2
fi

row=$(awk -F $'\t' -v id="${run_id}" '
  NR > 1 && $1 == id { print; found = 1 }
  END { if (!found) exit 1 }
' "${MATRIX}") || {
  echo "Unknown run-id: ${run_id}" >&2
  usage
  exit 2
}

IFS=$'\t' read -r _ model_family recipe nodes arm max_steps <<<"${row}"
config_rel="examples/configs/recipes/llm/performance/${recipe}"
config_path="${PROJECT_ROOT}/${config_rel}"
if [[ ! -f ${config_path} ]]; then
  echo "Missing recipe: ${config_path}" >&2
  exit 2
fi

expected_nodes=$(awk '
  /^cluster:/ { in_cluster = 1; next }
  in_cluster && /^  num_nodes:/ { print $2; exit }
' "${config_path}")
if [[ -n ${expected_nodes} && ${expected_nodes} != "${nodes}" ]]; then
  echo "Matrix node count ${nodes} does not match recipe node count ${expected_nodes}" >&2
  exit 2
fi

run_dir="${RUN_ROOT}/${run_id}"
mkdir -p "${run_dir}"

if [[ ${arm} == "baseline" ]]; then
  config_setup="baseline_config=\"${PROJECT_ROOT}/examples/configs/recipes/llm/performance/.${recipe}.baseline-\${SLURM_JOB_ID}.yaml\"
git -C \"${PROJECT_ROOT}\" show \"${BASELINE_COMMIT}:${config_rel}\" > \"\${baseline_config}\"
trap 'rm -f -- \"\${baseline_config}\"' EXIT
config_path=\"\${baseline_config}\""
else
  config_setup="config_path=\"${config_path}\""
fi

command="${config_setup}
UV_NO_SYNC=1 uv run examples/run_grpo.py \\
  --config \"\${config_path}\" \\
  grpo.max_num_steps=${max_steps} \\
  checkpointing.enabled=false \\
  logger.log_dir=\"${run_dir}/nemo-logs\" \\
  logger.wandb_enabled=True \\
  logger.wandb.project=\"${WANDB_PROJECT}\" \\
  logger.wandb.name=\"${run_id}\" \\
  logger.monitor_gpus=True"

export BASE_LOG_DIR="${run_dir}/ray"
export CONTAINER
export GPUS_PER_NODE
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${HF_HOME}/cache}
export HF_HOME
export MOUNTS
export NRL_FORCE_REBUILD_VENVS=${NRL_FORCE_REBUILD_VENVS:-true}
export WANDB_API_KEY

sbatch_args=(
  "--nodes=${nodes}"
  "--account=${SBATCH_ACCOUNT}"
  "--job-name=hybridep-${run_id}"
  "--partition=${SBATCH_PARTITION}"
  "--time=${SBATCH_TIME}"
  "--gpus-per-node=${GPUS_PER_NODE}"
  "--output=${run_dir}/slurm-%j.out"
)
if [[ -n ${SBATCH_SEGMENT:-} ]]; then
  sbatch_args+=("--segment=${SBATCH_SEGMENT}")
fi
if [[ ${test_only} == "--test-only" ]]; then
  sbatch_args+=("--test-only")
fi

echo "Submitting ${run_id}: model=${model_family} arm=${arm} nodes=${nodes} steps=${max_steps}"
cd "${PROJECT_ROOT}"
COMMAND="${command}" sbatch "${sbatch_args[@]}" ray.sub

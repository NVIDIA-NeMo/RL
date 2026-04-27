#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEMORL="${NEMORL:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

if [[ -f "${NEMORL}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${NEMORL}/.env"
  set +a
fi

CONFIG_PATH="${CONFIG_PATH:-examples/omni/nanov3_vision_rl.yaml}"
NUM_NODES="${NUM_NODES:-4}"
# Use a fresh default run name each launch so checkpoints/logs do not resume prior runs.
JOB_NAME_BASE="${JOB_NAME_BASE:-image-grpo}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S-%3N)}"
JOB_NAME="${JOB_NAME:-${JOB_NAME_BASE}-${RUN_ID}}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-${CP_SIZE:-}}"
MODEL_NAME="${IMAGE_GRPO_MODEL_NAME:-${MODEL_NAME:-}}"
CACHE_DIR="${IMAGE_GRPO_CACHE_DIR:-${CACHE_DIR:-}}"
: "${MODEL_NAME:?Set IMAGE_GRPO_MODEL_NAME or MODEL_NAME, or define it in ${NEMORL}/.env}"
: "${CACHE_DIR:?Set IMAGE_GRPO_CACHE_DIR or CACHE_DIR, or define it in ${NEMORL}/.env}"
RESULTS_ROOT="${RESULTS_ROOT:-${NEMORL}/results}"
RESULTS_DIR="${RESULTS_ROOT}/${JOB_NAME}"

SBATCH_ACCOUNT="${SBATCH_ACCOUNT:?Set SBATCH_ACCOUNT or define it in ${NEMORL}/.env}"
SBATCH_PARTITION="${SBATCH_PARTITION:-${PARTITION:-batch}}"
SBATCH_TIME="${SBATCH_TIME:-4:00:00}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"

export CONTAINER="${CONTAINER:?Set CONTAINER or define it in ${NEMORL}/.env}"
export MOUNTS="${MOUNTS:-/lustre:/lustre}"
export NRL_FORCE_REBUILD_VENVS="${NRL_FORCE_REBUILD_VENVS:-false}"
export CACHE_ROOT="${CACHE_ROOT:-${NEMORL}/.cache}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${HF_HOME}/modules}"
export NRL_MEGATRON_CHECKPOINT_DIR="${NRL_MEGATRON_CHECKPOINT_DIR:-${HF_HOME}/nemo_rl}"
export TMPDIR="${TMPDIR:-/tmp/nrl-${RUN_ID}}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${TMPDIR}/triton}"
export NEMO_RL_TRAIN_STEP_MEM_DIAG="${NEMO_RL_TRAIN_STEP_MEM_DIAG:-1}"

export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NVTE_FWD_LAYERNORM_SM_MARGIN="${NVTE_FWD_LAYERNORM_SM_MARGIN:-16}"
export NVTE_BWD_LAYERNORM_SM_MARGIN="${NVTE_BWD_LAYERNORM_SM_MARGIN:-16}"
export NEMO_RL_LOG_GPU_MEMORY="${NEMO_RL_LOG_GPU_MEMORY:-0}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export NRL_IGNORE_VERSION_MISMATCH="${NRL_IGNORE_VERSION_MISMATCH:-true}"

if [[ ! -f "${NEMORL}/ray.sub" ]]; then
  echo "ray.sub not found under NEMORL=${NEMORL}" >&2
  exit 1
fi

if [[ "${CONFIG_PATH}" = /* ]]; then
  CONFIG_ABS_PATH="${CONFIG_PATH}"
else
  CONFIG_ABS_PATH="${NEMORL}/${CONFIG_PATH}"
fi

if [[ ! -f "${CONFIG_ABS_PATH}" ]]; then
  echo "Config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

EXTRA_OVERRIDES=""
if [[ -n "${CONTEXT_PARALLEL_SIZE}" ]]; then
  EXTRA_OVERRIDES+=" policy.megatron_cfg.context_parallel_size=${CONTEXT_PARALLEL_SIZE}"
fi

export COMMAND="\
mkdir -p '${HF_HOME}' '${HF_MODULES_CACHE}' '${NRL_MEGATRON_CHECKPOINT_DIR}' '${TRITON_CACHE_DIR}' '${TMPDIR}' '${RESULTS_DIR}' && \
uv run examples/run_vlm_grpo.py --config '${CONFIG_PATH}' \
cluster.num_nodes=${NUM_NODES} \
policy.model_name='${MODEL_NAME}' \
checkpointing.checkpoint_dir='${RESULTS_DIR}' \
logger.log_dir='${RESULTS_DIR}' \
logger.wandb.name='${JOB_NAME}' \
data.cache_dir='${CACHE_DIR}'\
${EXTRA_OVERRIDES}"

cd "${NEMORL}"

sbatch \
    --nodes=${NUM_NODES} \
    --account=${SBATCH_ACCOUNT} \
    --job-name=${JOB_NAME} \
    --partition=${SBATCH_PARTITION} \
    --time=${SBATCH_TIME} \
    --gres=gpu:${GPUS_PER_NODE} \
    ray.sub

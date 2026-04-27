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

CONFIG_PATH="${CONFIG_PATH:-examples/omni/nanov3_text_rl.yaml}"
NUM_NODES="${NUM_NODES:-4}"
# Use a fresh default run name each launch so checkpoints/logs do not resume prior runs.
JOB_NAME_BASE="${JOB_NAME_BASE:-text-grpo}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S-%3N)}"
JOB_NAME="${JOB_NAME:-${JOB_NAME_BASE}-${RUN_ID}}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-${CP_SIZE:-}}"
MODEL_NAME="${TEXT_GRPO_MODEL_NAME:-${MODEL_NAME:-}}"
TRAIN_DATA_PATH="${TEXT_GRPO_TRAIN_DATA_PATH:-${TRAIN_DATA_PATH:-}}"
VALIDATION_DATA_PATH="${TEXT_GRPO_VALIDATION_DATA_PATH:-${VALIDATION_DATA_PATH:-${TRAIN_DATA_PATH}}}"
: "${MODEL_NAME:?Set TEXT_GRPO_MODEL_NAME or MODEL_NAME, or define it in ${NEMORL}/.env}"
: "${TRAIN_DATA_PATH:?Set TEXT_GRPO_TRAIN_DATA_PATH or TRAIN_DATA_PATH, or define it in ${NEMORL}/.env}"
: "${VALIDATION_DATA_PATH:?Set TEXT_GRPO_VALIDATION_DATA_PATH or VALIDATION_DATA_PATH, or define it in ${NEMORL}/.env}"
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
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${HF_HOME}/modules}"
export NRL_MEGATRON_CHECKPOINT_DIR="${NRL_MEGATRON_CHECKPOINT_DIR:-${HF_HOME}/nemo_rl}"
export TMPDIR="${TMPDIR:-/tmp/nrl-${RUN_ID}}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${TMPDIR}/triton}"
export NEMO_RL_TRAIN_STEP_MEM_DIAG="${NEMO_RL_TRAIN_STEP_MEM_DIAG:-1}"
export NEMO_RL_LOG_GPU_MEMORY="${NEMO_RL_LOG_GPU_MEMORY:-0}"
export RAY_LOG_SYNC_FREQUENCY="${RAY_LOG_SYNC_FREQUENCY:-30}"
export VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE="${VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE:-nccl}"

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
mkdir -p '${CACHE_ROOT}/uv' '${HF_HOME}' '${HF_MODULES_CACHE}' '${NRL_MEGATRON_CHECKPOINT_DIR}' '${TRITON_CACHE_DIR}' '${TMPDIR}' '${RESULTS_DIR}' && \
uv run examples/nemo_gym/run_grpo_nemo_gym.py --config '${CONFIG_PATH}' \
cluster.num_nodes=$NUM_NODES \
policy.model_name='${MODEL_NAME}' \
checkpointing.checkpoint_dir='${RESULTS_DIR}' \
logger.log_dir='${RESULTS_DIR}' \
data.train_jsonl_fpath='${TRAIN_DATA_PATH}' \
data.validation_jsonl_fpath='${VALIDATION_DATA_PATH}' \
logger.wandb.name='${JOB_NAME}'\
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

#!/usr/bin/env bash
# Submit canonical Nemotron Omni image MPO through the repository Ray launcher.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEMORL="${NEMORL:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

if [[ -f "${NEMORL}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${NEMORL}/.env"
  set +a
fi

CONFIG_PATH="${CONFIG_PATH:-examples/configs/recipes/vlm/vlm_mpo-nemotron-omni-30ba3b-mmpr-1n8g-megatron-tp8.v1.yaml}"
NUM_NODES="${NUM_NODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
JOB_NAME="${JOB_NAME:-nemotron-omni-mpo-${RUN_ID}}"
DATA_PATH="${MPO_DATA_PATH:-${DATA_PATH:-}}"
: "${DATA_PATH:?Set MPO_DATA_PATH or DATA_PATH}"
: "${CONTAINER:?Set CONTAINER to the #3290-based release image}"
: "${SBATCH_ACCOUNT:?Set SBATCH_ACCOUNT}"

MODEL_NAME="${MPO_MODEL_NAME:-nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16}"
SBATCH_PARTITION="${SBATCH_PARTITION:-batch}"
SBATCH_TIME="${SBATCH_TIME:-4:00:00}"
RESULTS_ROOT="${RESULTS_ROOT:-${NEMORL}/results}"
RESULTS_NAME="${RESULTS_NAME:-${JOB_NAME}}"
RESULTS_DIR="${RESULTS_ROOT}/${RESULTS_NAME}"
RUNNER="${RUNNER:-uv run --no-sync}"
WANDB_ENTITY="${WANDB_ENTITY:-joc}"
WANDB_PROJECT="${WANDB_PROJECT:-nemotron-omni-main-migration}"
WANDB_NAME="${WANDB_NAME:-${JOB_NAME}}"
WANDB_RUN_ID="${WANDB_RUN_ID:-}"
WANDB_RESUME="${WANDB_RESUME:-allow}"
WANDB_PIN_VERSION="${WANDB_PIN_VERSION:-0.21.1}"
SBATCH_DEPENDENCY="${SBATCH_DEPENDENCY:-}"
MPO_MAX_NUM_STEPS="${MPO_MAX_NUM_STEPS:-}"
MPO_MAX_SAMPLES="${MPO_MAX_SAMPLES:-}"
MPO_TRAIN_GLOBAL_BATCH_SIZE="${MPO_TRAIN_GLOBAL_BATCH_SIZE:-}"

OPTIONAL_OVERRIDES=""
if [[ -n "${MPO_MAX_NUM_STEPS}" ]]; then
  [[ "${MPO_MAX_NUM_STEPS}" =~ ^[1-9][0-9]*$ ]] || {
    echo "MPO_MAX_NUM_STEPS must be a positive integer" >&2
    exit 2
  }
  OPTIONAL_OVERRIDES+=" mpo.max_num_steps=${MPO_MAX_NUM_STEPS}"
fi
if [[ -n "${MPO_MAX_SAMPLES}" ]]; then
  [[ "${MPO_MAX_SAMPLES}" =~ ^[1-9][0-9]*$ ]] || {
    echo "MPO_MAX_SAMPLES must be a positive integer" >&2
    exit 2
  }
  OPTIONAL_OVERRIDES+=" data.train.max_samples=${MPO_MAX_SAMPLES}"
fi
if [[ -n "${MPO_TRAIN_GLOBAL_BATCH_SIZE}" ]]; then
  [[ "${MPO_TRAIN_GLOBAL_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || {
    echo "MPO_TRAIN_GLOBAL_BATCH_SIZE must be a positive integer" >&2
    exit 2
  }
  OPTIONAL_OVERRIDES+=" policy.train_global_batch_size=${MPO_TRAIN_GLOBAL_BATCH_SIZE}"
fi
if [[ -n "${WANDB_RUN_ID}" ]]; then
  [[ "${WANDB_RUN_ID}" =~ ^[A-Za-z0-9_-]+$ ]] || {
    echo "WANDB_RUN_ID may only contain letters, numbers, underscores, and hyphens" >&2
    exit 2
  }
  case "${WANDB_RESUME}" in
    allow | must | never | auto) ;;
    *)
      echo "WANDB_RESUME must be one of: allow, must, never, auto" >&2
      exit 2
      ;;
  esac
  OPTIONAL_OVERRIDES+=" +logger.wandb.id='${WANDB_RUN_ID}' +logger.wandb.resume='${WANDB_RESUME}'"
fi
if [[ -n "${SBATCH_DEPENDENCY}" ]] && [[ ! "${SBATCH_DEPENDENCY}" =~ ^[0-9]+$ ]]; then
  echo "SBATCH_DEPENDENCY must be a numeric Slurm job ID" >&2
  exit 2
fi

export CONTAINER
export GPUS_PER_NODE
export MOUNTS="${MOUNTS:-/lustre:/lustre}"
if [[ -f "${HOME}/.netrc" ]] && [[ "${MOUNTS}" != *"/root/.netrc"* ]]; then
  export MOUNTS="${MOUNTS},${HOME}/.netrc:/root/.netrc:ro"
fi
export HF_HOME="${HF_HOME:-${NEMORL}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${HF_HOME}/modules}"
export NRL_MEGATRON_CHECKPOINT_DIR="${NRL_MEGATRON_CHECKPOINT_DIR:-${HF_HOME}/nemo_rl}"
export NRL_FORCE_REBUILD_VENVS="${NRL_FORCE_REBUILD_VENVS:-false}"
# Ray appends a long session/socket suffix and Linux limits AF_UNIX paths to
# 107 bytes, so TMPDIR must not inherit the full descriptive job name.
export TMPDIR="${TMPDIR:-/tmp/nrl-${RUN_ID##*-}}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${TMPDIR}/triton}"
export WANDB_MODE="${WANDB_MODE:-online}"

[[ -f "${NEMORL}/${CONFIG_PATH}" ]] || {
  echo "Config not found: ${NEMORL}/${CONFIG_PATH}" >&2
  exit 2
}
[[ -f "${NEMORL}/ray.sub" ]] || {
  echo "Ray launcher not found: ${NEMORL}/ray.sub" >&2
  exit 2
}

if [[ -n "${WANDB_PIN_VERSION}" ]]; then
  [[ "${WANDB_PIN_VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || {
    echo "WANDB_PIN_VERSION must be empty or a semantic version" >&2
    exit 2
  }
  # Internal W&B deployments still issue 36-character keys. Newer SDKs reject
  # those keys locally, so mirror the legacy Omni launcher and pin only the
  # container driver environment without changing NeMo-RL's global lockfile.
  # Resolve the interpreter through RUNNER: the container's bare `python` may
  # differ from the /opt/nemo_rl_venv interpreter selected by `uv run`.
  WANDB_PIN_SNIPPET="wandb_python=\$(${RUNNER} python -c \"import sys; print(sys.executable)\") && (\"\${wandb_python}\" -c \"import wandb, sys; sys.exit(0 if wandb.__version__ == '${WANDB_PIN_VERSION}' else 1)\" 2>/dev/null || uv pip install --quiet --no-deps --python \"\${wandb_python}\" 'wandb==${WANDB_PIN_VERSION}') && \"\${wandb_python}\" -c \"import wandb; print('Using wandb', wandb.__version__, 'from', wandb.__file__)\" && "
else
  WANDB_PIN_SNIPPET=""
fi

export COMMAND="\
mkdir -p '${HF_HOME}' '${HF_DATASETS_CACHE}' '${HF_MODULES_CACHE}' '${NRL_MEGATRON_CHECKPOINT_DIR}' '${TRITON_CACHE_DIR}' '${RESULTS_DIR}' && \
cd '${NEMORL}' && \
export PYTHONPATH='${NEMORL}:${NEMORL}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:${NEMORL}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM':\${PYTHONPATH:-} && \
${WANDB_PIN_SNIPPET}${RUNNER} python examples/run_vlm_mpo.py --config '${CONFIG_PATH}' \
cluster.num_nodes=${NUM_NODES} \
policy.model_name='${MODEL_NAME}' \
policy.tokenizer.name='${MODEL_NAME}' \
data.train.data_path='${DATA_PATH}' \
checkpointing.checkpoint_dir='${RESULTS_DIR}' \
logger.wandb_enabled=true \
logger.wandb.entity='${WANDB_ENTITY}' \
logger.wandb.project='${WANDB_PROJECT}' \
logger.wandb.name='${WANDB_NAME}'${OPTIONAL_OVERRIDES}"

cd "${NEMORL}"
SBATCH_ARGS=(
  --nodes="${NUM_NODES}"
  --account="${SBATCH_ACCOUNT}"
  --job-name="${JOB_NAME}"
  --partition="${SBATCH_PARTITION}"
  --time="${SBATCH_TIME}"
  --gres="gpu:${GPUS_PER_NODE}"
)
if [[ -n "${SBATCH_DEPENDENCY}" ]]; then
  SBATCH_ARGS+=(--dependency="afterok:${SBATCH_DEPENDENCY}")
fi
sbatch "${SBATCH_ARGS[@]}" ray.sub

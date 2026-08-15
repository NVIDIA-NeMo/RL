#!/bin/bash
#SBATCH --job-name=zero-kl-precision
#SBATCH --account=your_account
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --output=logs/zero-kl-precision-%j-%x.out
#SBATCH --error=logs/zero-kl-precision-%j-%x.err

# MODEL=qwen1.5b|qwen30ba3b|nanov3  PRECISION=bf16|mxfp8  ZERO_TRAIN_GEN_MISMATCH=false

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
[[ -f "${SCRIPT_DIR}/.env" ]] && set -a && source "${SCRIPT_DIR}/.env" && set +a

: "${RL_DIR:?Set RL_DIR in .env}"
: "${CONTAINER_IMAGE:?Set CONTAINER_IMAGE in .env}"
: "${HF_TOKEN:?Set HF_TOKEN in .env}"
: "${HF_HOME:?Set HF_HOME in .env}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY in .env}"
: "${WANDB_ENTITY:?Set WANDB_ENTITY in .env}"
: "${WANDB_PROJECT:?Set WANDB_PROJECT in .env}"

MODEL="${MODEL:-}"
PRECISION="${PRECISION:-bf16}"
MAX_STEPS="${MAX_STEPS:-2000}"
ZERO_TRAIN_GEN_MISMATCH="${ZERO_TRAIN_GEN_MISMATCH:-true}"
GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-8}}"
NUM_NODES="${SLURM_NNODES:-${NUM_NODES:-1}}"

EXTRA_FLAGS=()
case "${MODEL}" in
    qwen1.5b|qwen-1.5b)
        RUN_PREFIX="qwen1.5b"
        GRPO_CONFIG="examples/configs/recipes/llm/grpo-qwen1.5b-megatron-zero-train-gen-kl.yaml"
        SAVE_PERIOD="${SAVE_PERIOD:-250}"
        ;;
    qwen30ba3b|qwen-30ba3b)
        RUN_PREFIX="qwen30ba3b"
        GRPO_CONFIG="examples/configs/recipes/llm/grpo-dapomath17k-qwen-30ba3b-megatron-zero-train-gen-kl.yaml"
        SAVE_PERIOD="${SAVE_PERIOD:-10}"
        ;;
    nanov3|nano|nanov3-30ba3b)
        RUN_PREFIX="nanov3"
        GRPO_CONFIG="examples/configs/recipes/llm/grpo-nanov3-30ba3b-megatron-zero-train-gen-kl.yaml"
        SAVE_PERIOD="${SAVE_PERIOD:-10}"
        ;;
    *)
        echo "ERROR: MODEL is required (qwen1.5b, qwen30ba3b, nanov3)." >&2
        exit 1
        ;;
esac

[[ "${PRECISION}" == "mxfp8" ]] && EXTRA_FLAGS+=(
    "policy.megatron_cfg.fp8_cfg.enabled=true"
    "policy.megatron_cfg.fp8_cfg.fp8_recipe=mxfp8"
)

[[ "${ZERO_TRAIN_GEN_MISMATCH}" == "false" ]] && EXTRA_FLAGS+=(
    "policy.megatron_cfg.zero_train_gen_mismatch=false"
)

RUN_TAG="${EXP_TAG:-${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}}"
WANDB_RUN_NAME="${RUN_PREFIX}-zero-kl-${PRECISION}-${RUN_TAG}"
CKPT_DIR="${CKPT_DIR:-${RL_DIR}/results/${WANDB_RUN_NAME}}"
LOG_DIR="${LOG_DIR:-${RL_DIR}/logs/${WANDB_RUN_NAME}}"
mkdir -p "${CKPT_DIR}" "${LOG_DIR}" logs

if [[ -n "${NRL_RAY_VENVS_MOUNT_HOST:-}" ]]; then
    mkdir -p "${NRL_RAY_VENVS_MOUNT_HOST}"
    NEMO_RL_VENV_CONTAINER="/opt/ray_venvs"
    NRL_RAY_VENVS_MOUNT_SUFFIX=",${NRL_RAY_VENVS_MOUNT_HOST}:/opt/ray_venvs"
else
    mkdir -p "${RL_DIR}/venvs"
    NEMO_RL_VENV_CONTAINER="/opt/nemo-rl/venvs"
    NRL_RAY_VENVS_MOUNT_SUFFIX=""
fi

GRPO_ARGS=(
    --config "${GRPO_CONFIG}"
    "grpo.max_num_steps=${MAX_STEPS}"
    "cluster.num_nodes=${NUM_NODES}"
    "cluster.gpus_per_node=${GPUS_PER_NODE}"
    "checkpointing.enabled=true"
    "checkpointing.checkpoint_dir=${CKPT_DIR}"
    "checkpointing.save_period=${SAVE_PERIOD}"
    "checkpointing.keep_top_k=${KEEP_TOP_K:-3}"
    "logger.log_dir=${LOG_DIR}"
    "logger.wandb_enabled=true"
    "logger.wandb.project=${WANDB_PROJECT}"
    "logger.wandb.name=${WANDB_RUN_NAME}"
    "${EXTRA_FLAGS[@]}"
    "$@"
)

UV_PREFIX=()
[[ "${NRL_FORCE_REBUILD_VENVS:-false}" == "true" ]] && UV_PREFIX=(env "NRL_FORCE_REBUILD_VENVS=true" uv run --extra mcore)
[[ ${#UV_PREFIX[@]} -eq 0 ]] && UV_PREFIX=(uv run)

cd "${RL_DIR}"
export TORCH_CUDA_ARCH_LIST='9.0 10.0'
export CONTAINER="${CONTAINER_IMAGE}"
export MOUNTS="/lustre:/lustre,${RL_DIR}:/opt/nemo-rl${NRL_RAY_VENVS_MOUNT_SUFFIX}"
if [[ "${RUN_PREFIX}" == "qwen30ba3b" ]]; then
    _dapo_patch="${SCRIPT_DIR}/dapo_zero_kl_patch.py"
    if [[ -f "${_dapo_patch}" ]]; then
        export MOUNTS="${MOUNTS},${_dapo_patch}:/opt/nemo-rl/nemo_rl/environments/dapo_math_verifier.py"
    else
        echo "WARNING: DAPO verifier patch missing: ${_dapo_patch}" >&2
    fi
    unset _dapo_patch
fi
[[ -d /scratch ]] && export MOUNTS="${MOUNTS},/scratch:/scratch"
export GPUS_PER_NODE
export NEMO_RL_VENV_DIR="${NEMO_RL_VENV_CONTAINER}"

CACHE_EXPORT=""
if [[ "${NRL_USE_WARM_UV_CACHE:-false}" == "true" ]]; then
    mkdir -p "${NRL_WARM_UV_CACHE_DIR:-${RL_DIR}/uv_cache}"
    CACHE_EXPORT="export UV_CACHE_DIR=${NRL_WARM_UV_CACHE_DIR:-${RL_DIR}/uv_cache} && "
fi

export COMMAND="${CACHE_EXPORT}\
export NEMO_RL_VENV_DIR=${NEMO_RL_VENV_CONTAINER} && \
export PYTHONUNBUFFERED=1 && \
export UV_HTTP_TIMEOUT=900 && \
export HF_HOME=${HF_HOME} && \
export TORCH_CUDA_ARCH_LIST='${TORCH_CUDA_ARCH_LIST}' && \
export HF_TOKEN=${HF_TOKEN} && \
export WANDB_API_KEY=${WANDB_API_KEY} && \
export WANDB_ENTITY=${WANDB_ENTITY} && \
export CUDA_DEVICE_MAX_CONNECTIONS=1 && \
cd /opt/nemo-rl && \
${UV_PREFIX[*]} examples/run_grpo.py ${GRPO_ARGS[*]}"

echo "MODEL=${RUN_PREFIX} PRECISION=${PRECISION} CONFIG=${GRPO_CONFIG}"
echo "NODES=${NUM_NODES}x${GPUS_PER_NODE}  RUN=${WANDB_RUN_NAME}"
echo "COMMAND: ${COMMAND}"
source ray.sub

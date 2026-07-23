#!/bin/bash
#SBATCH --job-name=qwen1.5b-zero-kl-precision
#SBATCH --account=your_account_here
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --output=logs/qwen1.5b-zero-kl-precision-%j-%x.out
#SBATCH --error=logs/qwen1.5b-zero-kl-precision-%j-%x.err

# =============================================================================
# Qwen2.5-1.5B — Megatron colocated inference, zero-KL precision sweep (BF16 vs MXFP8)
#
# Enables zero_train_gen_mismatch which activates batch-invariant kernels (TE GEMM
# workspace pinned, FA4 num_splits=1) and use_mamba_mem_eff_path=False for hybrid models.
#
# Megatron colocated inference (m-inf) with FlashAttention 4 via TE v2.15.
# zero_train_gen_mismatch wires attention_backend=flash automatically.
#
# PRECISION (default bf16). Both use examples/configs/grpo_math_1B_megatron.yaml;
# mxfp8 adds inline policy.megatron_cfg.fp8_cfg overrides (enabled=true, recipe=mxfp8).
#
# Usage (from RL/ directory):
#   sbatch --export=PRECISION=bf16  run_qwen1.5b_zero_kl_precision.sh
#   sbatch --export=PRECISION=bf16,ZERO_TRAIN_GEN_MISMATCH=false run_qwen1.5b_zero_kl_precision.sh
#   sbatch --export=PRECISION=mxfp8 run_qwen1.5b_zero_kl_precision.sh
#   sbatch --export=PRECISION=mxfp8,ZERO_TRAIN_GEN_MISMATCH=false run_qwen1.5b_zero_kl_precision.sh
#
# RESTART=true (default false) — wipe CKPT_DIR and start a fresh W&B run.
# ZERO_TRAIN_GEN_MISMATCH=true (default) — enable zero-KL train/gen mismatch patches.
# EXP_TAG — optional suffix on run name (default: today's date).
# MAX_STEPS — default 2000.
# NRL_FORCE_REBUILD_VENVS=true — first build / after pyproject or uv.lock changes.
# NRL_USE_WARM_UV_CACHE=true — read wheels from NRL_WARM_UV_CACHE_DIR (default ${RL_DIR}/uv_cache).
# =============================================================================

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"

GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-8}}"
NUM_NODES="${SLURM_NNODES:-1}"

PRECISION="${PRECISION:-bf16}"
MAX_STEPS="${MAX_STEPS:-2000}"
NRL_USE_WARM_UV_CACHE="${NRL_USE_WARM_UV_CACHE:-false}"
ZERO_TRAIN_GEN_MISMATCH="${ZERO_TRAIN_GEN_MISMATCH:-true}"

if [[ -f "${SCRIPT_DIR}/.env" ]]; then
    set -a; source "${SCRIPT_DIR}/.env"; set +a
fi

export TORCH_CUDA_ARCH_LIST='9.0 10.0'
: "${RL_DIR:?Set RL_DIR in .env}"
: "${CONTAINER_IMAGE:?Set CONTAINER_IMAGE in .env}"
: "${HF_TOKEN:?Set HF_TOKEN in .env}"
: "${HF_HOME:?Set HF_HOME in .env}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY in .env}"
: "${WANDB_ENTITY:?Set WANDB_ENTITY in .env}"
: "${WANDB_PROJECT:?Set WANDB_PROJECT in .env}"

NRL_WARM_UV_CACHE_DIR="${NRL_WARM_UV_CACHE_DIR:-${RL_DIR}/uv_cache}"

NRL_RAY_VENVS_MOUNT_HOST="${NRL_RAY_VENVS_MOUNT_HOST:-}"
if [[ -n "${NRL_RAY_VENVS_MOUNT_HOST}" ]]; then
    mkdir -p "${NRL_RAY_VENVS_MOUNT_HOST}"
    NEMO_RL_VENV_CONTAINER="/opt/ray_venvs"
    NRL_RAY_VENVS_MOUNT_SUFFIX=",${NRL_RAY_VENVS_MOUNT_HOST}:/opt/ray_venvs"
else
    mkdir -p "${RL_DIR}/venvs"
    NEMO_RL_VENV_CONTAINER="/opt/nemo-rl/venvs"
    NRL_RAY_VENVS_MOUNT_SUFFIX=""
fi

if [[ ! "$PRECISION" =~ ^(bf16|mxfp8)$ ]]; then
    echo "ERROR: Invalid PRECISION '${PRECISION}'. Must be: bf16, mxfp8" >&2
    exit 1
fi

GRPO_CONFIG="examples/configs/grpo_math_1B_megatron.yaml"
MXFP8_OVERRIDES=""
case "$PRECISION" in
    bf16)
        WANDB_RUN_SUFFIX="bf16"
        ;;
    mxfp8)
        WANDB_RUN_SUFFIX="mxfp8"
        MXFP8_OVERRIDES="\
    policy.megatron_cfg.fp8_cfg.enabled=true \
    policy.megatron_cfg.fp8_cfg.fp8_recipe=mxfp8"
        ;;
esac

NRL_NVTE_DEBUG="${NRL_NVTE_DEBUG:-1}"
NRL_NVTE_DEBUG_LEVEL="${NRL_NVTE_DEBUG_LEVEL:-1}"

_RUN_SUFFIX="${EXP_TAG:-$(date +%Y-%m-%d)}"
WANDB_RUN_NAME="qwen-1.5b-zero-kl-${WANDB_RUN_SUFFIX}-m-inf-${_RUN_SUFFIX}"
CKPT_DIR="${CKPT_DIR:-${RL_DIR}/results/${WANDB_RUN_NAME}}"
LOG_DIR_EXP="${LOG_DIR_EXP:-${RL_DIR}/logs/${WANDB_RUN_NAME}}"
SAVE_PERIOD="${SAVE_PERIOD:-250}"
KEEP_TOP_K="${KEEP_TOP_K:-3}"
RESTART="${RESTART:-false}"
if [[ "${RESTART}" != "true" && "${RESTART}" != "false" ]]; then
    echo "ERROR: RESTART must be true or false (got '${RESTART}')." >&2
    exit 1
fi
if [[ "${ZERO_TRAIN_GEN_MISMATCH}" != "true" && "${ZERO_TRAIN_GEN_MISMATCH}" != "false" ]]; then
    echo "ERROR: ZERO_TRAIN_GEN_MISMATCH must be true or false (got '${ZERO_TRAIN_GEN_MISMATCH}')." >&2
    exit 1
fi

if [[ "${RESTART}" == "true" ]]; then
    if [[ -z "${CKPT_DIR}" || "${CKPT_DIR}" == "/" ]]; then
        echo "ERROR: refusing to RESTART-wipe unsafe CKPT_DIR='${CKPT_DIR}'." >&2
        exit 1
    fi
    if [[ -d "${CKPT_DIR}" ]]; then
        echo "RESTART=true: wiping existing checkpoints in ${CKPT_DIR}" >&2
        rm -rf "${CKPT_DIR}"
    fi
fi

mkdir -p "${CKPT_DIR}" "${LOG_DIR_EXP}"

WANDB_RUN_ID_PIN="${CKPT_DIR}/.wandb_run_id"
if [[ -z "${WANDB_RUN_ID:-}" ]]; then
    _pinned_id=""
    if [[ -f "${WANDB_RUN_ID_PIN}" ]]; then
        _pinned_id="$(head -n1 "${WANDB_RUN_ID_PIN}" 2>/dev/null | tr -d '[:space:]')"
    fi
    if [[ -n "${_pinned_id}" ]]; then
        WANDB_RUN_ID="${_pinned_id}"
    elif [[ "${RESTART}" == "true" ]]; then
        WANDB_RUN_ID="$(printf '%s-%s-%s' "${WANDB_PROJECT}" "${WANDB_RUN_NAME}" "${SLURM_JOB_ID:-$(date +%s)}" | md5sum | cut -c1-32)"
    else
        WANDB_RUN_ID="$(printf '%s-%s' "${WANDB_PROJECT}" "${WANDB_RUN_NAME}" | md5sum | cut -c1-32)"
    fi
fi
printf '%s\n' "${WANDB_RUN_ID}" > "${WANDB_RUN_ID_PIN}"

if [[ "${RESTART}" == "true" ]]; then
    WANDB_RESUME="${WANDB_RESUME:-never}"
else
    WANDB_RESUME="${WANDB_RESUME:-allow}"
fi

echo "=============================================="
echo "Qwen2.5-1.5B  zero-KL precision study"
echo "  Mode:          m-inf (FA4 via TE v2.15)"
echo "  Precision:     ${PRECISION}"
echo "  Config:        ${GRPO_CONFIG}"
echo "  zero_train_gen_mismatch: ${ZERO_TRAIN_GEN_MISMATCH}"
echo "  Max steps:     ${MAX_STEPS}"
echo "  Nodes:         ${NUM_NODES}"
echo "  GPUs/node:     ${GPUS_PER_NODE}"
echo "  Total GPUs:    $((NUM_NODES * GPUS_PER_NODE))"
echo "  RL dir:        ${RL_DIR}"
echo "  Container:     ${CONTAINER_IMAGE}"
echo "  Job ID:        ${SLURM_JOB_ID:-interactive}"
echo "  Nodes:         ${SLURM_NODELIST:-N/A}"
echo "  Run name:      ${WANDB_RUN_NAME}"
echo "  Ckpt dir:      ${CKPT_DIR} (save_period=${SAVE_PERIOD}, keep_top_k=${KEEP_TOP_K}, restart=${RESTART})"
echo "  Log dir:       ${LOG_DIR_EXP}"
echo "  W&B run id:    ${WANDB_RUN_ID} (resume=${WANDB_RESUME})"
echo "  Time:          $(date)"
echo "=============================================="

mkdir -p logs

GRPO_ARGS="\
    --config ${GRPO_CONFIG} \
    grpo.max_num_steps=${MAX_STEPS} \
    cluster.num_nodes=${NUM_NODES} cluster.gpus_per_node=${GPUS_PER_NODE} \
    grpo.val_at_start=true grpo.val_at_end=true grpo.val_period=10 \
    checkpointing.enabled=true \
    checkpointing.checkpoint_dir=${CKPT_DIR} \
    checkpointing.save_period=${SAVE_PERIOD} \
    checkpointing.keep_top_k=${KEEP_TOP_K} \
    logger.log_dir=${LOG_DIR_EXP} \
    logger.wandb_enabled=true \
    logger.wandb.project=${WANDB_PROJECT}"

if [[ "${NRL_FORCE_REBUILD_VENVS:-false}" == "true" ]]; then
    BASE_CMD="NRL_FORCE_REBUILD_VENVS=${NRL_FORCE_REBUILD_VENVS:-true} uv run --extra mcore examples/run_grpo.py ${GRPO_ARGS}"
else
    BASE_CMD="uv run examples/run_grpo.py ${GRPO_ARGS}"
fi

MINF_FLAGS="\
    policy.generation.backend=megatron \
    +policy.megatron_cfg.zero_train_gen_mismatch=${ZERO_TRAIN_GEN_MISMATCH} \
    policy.generation.mcore_generation_config.enable_chunked_prefill=false \
    policy.max_total_sequence_length=1024"

CMD="${BASE_CMD} ${MINF_FLAGS} \
    logger.wandb.name=${WANDB_RUN_NAME}"
if [[ -n "${MXFP8_OVERRIDES}" ]]; then
    CMD="${CMD} ${MXFP8_OVERRIDES}"
fi

if [[ $# -gt 0 ]]; then
    CMD="${CMD} $*"
fi

cd "${RL_DIR}"

export CONTAINER="${CONTAINER_IMAGE}"
export MOUNTS="/lustre:/lustre,${RL_DIR}:/opt/nemo-rl${NRL_RAY_VENVS_MOUNT_SUFFIX}"
if [[ -d /scratch ]]; then
    export MOUNTS="${MOUNTS},/scratch:/scratch"
fi
export GPUS_PER_NODE

if [[ "${NRL_USE_WARM_UV_CACHE}" == "true" ]]; then
    mkdir -p "${NRL_WARM_UV_CACHE_DIR}"
    NRL_WARM_UV_CACHE_EXPORT="export UV_CACHE_DIR=${NRL_WARM_UV_CACHE_DIR} && "
else
    NRL_WARM_UV_CACHE_EXPORT=""
fi

export COMMAND="\
    ${NRL_WARM_UV_CACHE_EXPORT}\
    export NEMO_RL_VENV_DIR=${NEMO_RL_VENV_CONTAINER} && \
    export NVTE_DEBUG=${NRL_NVTE_DEBUG} && \
    export NVTE_DEBUG_LEVEL=${NRL_NVTE_DEBUG_LEVEL} && \
    export PYTHONUNBUFFERED=1 && \
    export UV_HTTP_TIMEOUT=900 && \
    export HF_HOME=${HF_HOME} && \
    export TORCH_CUDA_ARCH_LIST='${TORCH_CUDA_ARCH_LIST}' && \
    export HF_TOKEN=${HF_TOKEN} && \
    export WANDB_API_KEY=${WANDB_API_KEY} && \
    export WANDB_ENTITY=${WANDB_ENTITY} && \
    export WANDB_RUN_ID=${WANDB_RUN_ID} && \
    export WANDB_RESUME=${WANDB_RESUME} && \
    export CUDA_DEVICE_MAX_CONNECTIONS=1 && \
    cd /opt/nemo-rl && \
    ${CMD}"

echo ""
echo "COMMAND:"
echo "${CMD}"
echo ""
echo "=============================================="

export NEMO_RL_VENV_DIR="${NEMO_RL_VENV_CONTAINER}"

source ray.sub

echo ""
echo "=============================================="
echo "Job completed at: $(date)"
echo "=============================================="

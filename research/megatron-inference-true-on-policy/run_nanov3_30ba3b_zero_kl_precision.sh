#!/bin/bash
#SBATCH --job-name=nanov3-30ba3b-zero-kl-precision
#SBATCH --account=your_account_here
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --output=logs/nanov3-30ba3b-zero-kl-precision-%j-%x.out
#SBATCH --error=logs/nanov3-30ba3b-zero-kl-precision-%j-%x.err

# =============================================================================
# Nemotron-3-Nano-30B-A3B (hybrid Mamba/attention MoE) — Megatron colocated
# inference, zero-KL precision sweep (BF16 vs MXFP8).
#
# Enables zero_train_gen_mismatch which activates batch-invariant kernels (TE GEMM
# workspace pinned, FA4 num_splits=1), Mamba kernel unification (use_mamba_mem_eff_path=False,
# MAMBA_DETERMINISTIC=1, mamba_inference_ssm_states_dtype=float32), fixed-order MoE combine
# (NRL_FIXED_ORDER_MOE_COMBINE=1), and all cuBLAS workspace pins.
# Router replay is enabled separately via policy.router_replay.enabled=true.
#
# Overrides the base recipe for colocated sync m-inf study:
#   tensor_model_parallel_size: 2 → 1 (shared train/gen workers, 2n8g = 16 GPUs, TP=1 EP=8)
#   colocated.enabled: false → true
#   mamba_inference_ssm_states_dtype: float32 (match train fp32 chunk-scan boundary states)
#   mamba_inference_conv_states_dtype: float32 (rl-fp4 parity)
#   enable_chunked_prefill: false (BIK reference prefill requires whole-prompt)
#
# CUDA_GRAPHS (default true) — generation-engine CUDA-graph switch. Validated
# compatible with zero_train_gen_mismatch at inference_cuda_graph_scope=block
# (KL ~1e-10, ~10% gen speedup). Requires kv_cache_management_mode=persist.
# Set CUDA_GRAPHS=false to use recompute mode (saves memory during training).
#
# MODE (default m-inf-fa4):
#   m-inf-fa4  — FlashAttention 4 (+policy.megatron_cfg.attention_backend=flash)
#   m-inf      — TE unfused attention
#
# PRECISION (default bf16):
#   bf16   — examples/configs/recipes/llm/grpo-nanov3-30BA3B-2n8g-megatron_generation.yaml
#   mxfp8  — same base yaml + inline fp8 overrides
#
# Usage (from RL/ directory):
#   sbatch --export=PRECISION=bf16  research/zero-kl/run_nanov3_30ba3b_zero_kl_precision.sh
#   sbatch --export=PRECISION=mxfp8 research/zero-kl/run_nanov3_30ba3b_zero_kl_precision.sh
#   sbatch --export=PRECISION=bf16,CUDA_GRAPHS=false research/zero-kl/run_nanov3_30ba3b_zero_kl_precision.sh
#
# RESTART=true (default false) — wipe CKPT_DIR and start a fresh W&B run.
# EXP_TAG — optional suffix on run name (default: today's date).
# MAX_STEPS — default 2000.
# NRL_FORCE_REBUILD_VENVS=true — first build / after pyproject or uv.lock changes.
# NRL_USE_WARM_UV_CACHE=true — read wheels from NRL_WARM_UV_CACHE_DIR (default ${RL_DIR}/uv_cache).
# =============================================================================

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"

GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-8}}"
NUM_NODES="${SLURM_NNODES:-2}"

MODE="${MODE:-m-inf-fa4}"
PRECISION="${PRECISION:-bf16}"
MAX_STEPS="${MAX_STEPS:-2000}"
CUDA_GRAPHS="${CUDA_GRAPHS:-true}"
NRL_USE_WARM_UV_CACHE="${NRL_USE_WARM_UV_CACHE:-false}"

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

if [[ "$MODE" != "m-inf" && "$MODE" != "m-inf-fa4" ]]; then
    echo "ERROR: MODE must be m-inf or m-inf-fa4 (got '${MODE}')." >&2
    exit 1
fi

case "$MODE" in
    m-inf-fa4)
        ATTENTION_FLAGS="+policy.megatron_cfg.attention_backend=flash"
        TE_ATTN_PREFIX='unset NVTE_FUSED_ATTN NVTE_FLASH_ATTN NVTE_UNFUSED_ATTN && export NVTE_FLASH_ATTN=1 && export NVTE_FUSED_ATTN=0 && export NVTE_UNFUSED_ATTN=0 && '
        ;;
    m-inf)
        ATTENTION_FLAGS="+policy.megatron_cfg.attention_backend=unfused"
        TE_ATTN_PREFIX='unset NVTE_FUSED_ATTN NVTE_FLASH_ATTN NVTE_UNFUSED_ATTN && export NVTE_FLASH_ATTN=0 && export NVTE_FUSED_ATTN=0 && export NVTE_UNFUSED_ATTN=1 && '
        ;;
esac

if [[ ! "$PRECISION" =~ ^(bf16|mxfp8)$ ]]; then
    echo "ERROR: Invalid PRECISION '${PRECISION}'. Must be: bf16, mxfp8" >&2
    exit 1
fi

if [[ "${CUDA_GRAPHS}" != "true" && "${CUDA_GRAPHS}" != "false" ]]; then
    echo "ERROR: CUDA_GRAPHS must be true or false (got '${CUDA_GRAPHS}')." >&2
    exit 1
fi

if [[ "${CUDA_GRAPHS}" == "true" ]]; then
    # Validated compatible with zero_train_gen_mismatch at scope=block (KL ~1e-10).
    # BIK Mamba paths run eagerly between graph segments; requires persist cache mode
    # so graph-captured buffer addresses don't dangle across suspend.
    CUDA_GRAPH_FLAGS="\
    +policy.megatron_cfg.moe_pad_experts_for_cuda_graph_inference=true \
    policy.generation.mcore_generation_config.use_cuda_graphs_for_non_decode_steps=True \
    policy.generation.mcore_generation_config.cuda_graph_impl=local \
    policy.generation.mcore_generation_config.num_cuda_graphs=4 \
    policy.generation.mcore_generation_config.inference_cuda_graph_scope=block \
    policy.generation.mcore_generation_config.kv_cache_management_mode=persist"
else
    CUDA_GRAPH_FLAGS="\
    +policy.megatron_cfg.moe_pad_experts_for_cuda_graph_inference=false \
    policy.generation.mcore_generation_config.use_cuda_graphs_for_non_decode_steps=False \
    policy.generation.mcore_generation_config.cuda_graph_impl=none \
    policy.generation.mcore_generation_config.num_cuda_graphs=0 \
    policy.generation.mcore_generation_config.inference_cuda_graph_scope=none \
    policy.generation.mcore_generation_config.kv_cache_management_mode=recompute"
fi

GRPO_CONFIG="examples/configs/recipes/llm/grpo-nanov3-30BA3B-2n8g-megatron_generation.yaml"
MXFP8_OVERRIDES=""
case "$PRECISION" in
    bf16)
        WANDB_RUN_SUFFIX="bf16"
        ;;
    mxfp8)
        WANDB_RUN_SUFFIX="mxfp8"
        MXFP8_OVERRIDES="\
    policy.megatron_cfg.fp8_cfg.enabled=true \
    policy.megatron_cfg.fp8_cfg.fp8_recipe=mxfp8 \
    policy.megatron_cfg.optimizer.use_precision_aware_optimizer=false"
        ;;
esac

NRL_NVTE_DEBUG="${NRL_NVTE_DEBUG:-1}"
NRL_NVTE_DEBUG_LEVEL="${NRL_NVTE_DEBUG_LEVEL:-1}"

_RUN_SUFFIX="${EXP_TAG:-$(date +%Y-%m-%d)}"
WANDB_RUN_NAME="nanov3-30ba3b-zero-kl-${WANDB_RUN_SUFFIX}-${MODE}-${_RUN_SUFFIX}"
CKPT_DIR="${CKPT_DIR:-${RL_DIR}/results/${WANDB_RUN_NAME}}"
LOG_DIR_EXP="${LOG_DIR_EXP:-${RL_DIR}/logs/${WANDB_RUN_NAME}}"
SAVE_PERIOD="${SAVE_PERIOD:-10}"
KEEP_TOP_K="${KEEP_TOP_K:-3}"
RESTART="${RESTART:-false}"
if [[ "${RESTART}" != "true" && "${RESTART}" != "false" ]]; then
    echo "ERROR: RESTART must be true or false (got '${RESTART}')." >&2
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
echo "Nemotron-3-Nano-30B-A3B  zero-KL precision study"
echo "  Mode:          ${MODE}"
echo "  Precision:     ${PRECISION}"
echo "  Config:        ${GRPO_CONFIG}"
echo "  zero_train_gen_mismatch: true"
echo "  Router replay: true"
echo "  CUDA graphs:   ${CUDA_GRAPHS} (validated with zero_train_gen_mismatch at scope=block)"
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

STUDY_OVERRIDES="
    policy.megatron_cfg.moe_token_dispatcher_type=alltoall \
    policy.megatron_cfg.moe_router_dtype=fp32"

GRPO_ARGS="\
    --config ${GRPO_CONFIG} \
    grpo.max_num_steps=${MAX_STEPS} \
    cluster.num_nodes=${NUM_NODES} cluster.gpus_per_node=${GPUS_PER_NODE} \
    ${STUDY_OVERRIDES} \
    grpo.val_at_start=false grpo.val_at_end=true grpo.val_period=10 \
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

# Colocated sync m-inf: training and generation share the same model instance and
# GPU set. Override base recipe defaults (non-colocated, TP=2) for the study topology.
# Keys absent from the recipe/parent YAML must use Hydra '+' (append).
MINF_FLAGS="\
    policy.generation.backend=megatron \
    +policy.megatron_cfg.zero_train_gen_mismatch=true \
    +policy.router_replay.enabled=true \
    policy.generation.colocated.enabled=true \
    policy.megatron_cfg.tensor_model_parallel_size=1 \
    policy.generation.mcore_generation_config.tensor_model_parallel_size=1 \
    +policy.megatron_cfg.cuda_graph_impl=none \
    policy.megatron_cfg.moe_permute_fusion=false \
    +policy.generation.mcore_generation_config.static_kv_memory_pointers=false \
    policy.generation.mcore_generation_config.buffer_size_gb=16 \
    policy.generation.mcore_generation_config.enable_chunked_prefill=false \
    +policy.generation.mcore_generation_config.mamba_inference_ssm_states_dtype=float32 \
    +policy.generation.mcore_generation_config.mamba_inference_conv_states_dtype=float32"

# Append CUDA-graph generation knobs (set by the CUDA_GRAPHS switch above).
MINF_FLAGS="${MINF_FLAGS} ${CUDA_GRAPH_FLAGS}"

CMD="${BASE_CMD} ${MINF_FLAGS} \
    logger.wandb.name=${WANDB_RUN_NAME} \
    ${ATTENTION_FLAGS}"
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

export COMMAND="${TE_ATTN_PREFIX}\
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
    export NRL_INSTALL_FA3=${NRL_INSTALL_FA3:-0} && \
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

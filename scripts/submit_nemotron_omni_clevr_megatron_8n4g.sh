#!/usr/bin/env bash
set -euo pipefail

# Submit the eight-node Omni CLEVR Megatron GRPO recipe through NeMo-RL's
# Ray/Slurm launcher.
#
# Default: non-colocated gen-heavy (6 gen + 2 train). Async colocated is
# supported on this branch (NVIDIA-NeMo/RL#2884); set COLOCATED=true to share
# all GPUs for train+gen. ASYNC_GRPO=true is required for colocated async.
#
# GENERATION_BACKEND=vllm flips inference to Omni vLLM for A/B vs Megatron.
# The A/B defaults disable stop/bad-word filters; native multimodal inputs are
# supported by both synchronous and non-colocated asynchronous rollouts.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEMORL="${NEMORL:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
CONTAINER_NEMORL="${CONTAINER_NEMORL:-/opt/nemo-rl}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-${NEMORL}/workspace}"

CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/coreai/users/cye/enroot/nemo_rl_0807.sqsh}"
MODEL_NAME="${MODEL_NAME:-nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16}"
CONFIG="${CONFIG:-examples/configs/recipes/vlm/vlm_grpo-nemotron-omni-30ba3b-clevr-8n4g-megatron_generation.v1.yaml}"
GENERATION_BACKEND="${GENERATION_BACKEND:-megatron}"
COLOCATED="${COLOCATED:-false}"
ASYNC_GRPO="${ASYNC_GRPO:-true}"

if [[ "${GENERATION_BACKEND}" != "megatron" && "${GENERATION_BACKEND}" != "vllm" ]]; then
  echo "GENERATION_BACKEND must be megatron or vllm (got ${GENERATION_BACKEND})." >&2
  exit 1
fi
if [[ "${COLOCATED}" == "true" && "${ASYNC_GRPO}" == "true" && "${GENERATION_BACKEND}" != "megatron" ]]; then
  echo "Colocated async GRPO requires GENERATION_BACKEND=megatron." >&2
  exit 1
fi

CACHE_ROOT="${CACHE_ROOT:-${WORKSPACE_ROOT}/cache/nemo-rl-omni}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HUGGINGFACE_HUB_CACHE}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${HF_HOME}/modules}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export NRL_MEGATRON_CHECKPOINT_DIR="${NRL_MEGATRON_CHECKPOINT_DIR:-${CACHE_ROOT}/megatron-checkpoints}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}/xdg}"
export TORCH_HOME="${TORCH_HOME:-${CACHE_ROOT}/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${CACHE_ROOT}/triton}"

NUM_NODES="${NUM_NODES:-8}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"

if [[ "${COLOCATED}" == "true" ]]; then
  # Shared train+gen on every GPU (NVIDIA-NeMo/RL#2884).
  TRAIN_WORLD_SIZE=$((NUM_NODES * GPUS_PER_NODE))
  INFERENCE_WORLD_SIZE="${TRAIN_WORLD_SIZE}"
  GEN_GPUS_PER_NODE="${GPUS_PER_NODE}"
  NUM_GEN_NODES="${NUM_NODES}"
  COLOCATED_ENABLED=true
else
  NUM_GEN_NODES="${NUM_GEN_NODES:-6}"
  if (( NUM_NODES == 1 )); then
    if (( NUM_GEN_NODES != 1 )); then
      echo "One-node non-colocated mode requires NUM_GEN_NODES=1." >&2
      exit 1
    fi
    GEN_GPUS_PER_NODE="${GEN_GPUS_PER_NODE:-$((GPUS_PER_NODE / 2))}"
    if (( GPUS_PER_NODE < 4 || GEN_GPUS_PER_NODE <= 0 || GEN_GPUS_PER_NODE >= GPUS_PER_NODE )); then
      echo "One-node non-colocated mode requires at least four GPUs and a strict train/inference split." >&2
      exit 1
    fi
    TRAIN_WORLD_SIZE=$((GPUS_PER_NODE - GEN_GPUS_PER_NODE))
  else
    GEN_GPUS_PER_NODE="${GEN_GPUS_PER_NODE:-${GPUS_PER_NODE}}"
    if (( NUM_GEN_NODES <= 0 || NUM_GEN_NODES >= NUM_NODES )); then
      echo "Multi-node non-colocated mode requires 0 < NUM_GEN_NODES < NUM_NODES." >&2
      exit 1
    fi
    if (( GEN_GPUS_PER_NODE != GPUS_PER_NODE )); then
      echo "Multi-node non-colocated inference must reserve complete GPU nodes." >&2
      exit 1
    fi
    TRAIN_WORLD_SIZE=$(((NUM_NODES - NUM_GEN_NODES) * GPUS_PER_NODE))
  fi
  INFERENCE_WORLD_SIZE=$((NUM_GEN_NODES * GEN_GPUS_PER_NODE))
  COLOCATED_ENABLED=false
fi

# Prefer TP=8; fall back by powers of two for smaller worlds.
DEFAULT_POLICY_TP=8
while (( DEFAULT_POLICY_TP > TRAIN_WORLD_SIZE || TRAIN_WORLD_SIZE % DEFAULT_POLICY_TP != 0 )); do
  DEFAULT_POLICY_TP=$((DEFAULT_POLICY_TP / 2))
done
DEFAULT_INFER_TP=8
while (( DEFAULT_INFER_TP > INFERENCE_WORLD_SIZE || INFERENCE_WORLD_SIZE % DEFAULT_INFER_TP != 0 )); do
  DEFAULT_INFER_TP=$((DEFAULT_INFER_TP / 2))
done
POLICY_TP="${POLICY_TP:-${DEFAULT_POLICY_TP}}"
INFER_TP="${INFER_TP:-${DEFAULT_INFER_TP}}"
POLICY_CP="${POLICY_CP:-1}"

if [[ "${GENERATION_BACKEND}" == "megatron" && "${POLICY_CP}" != "1" ]]; then
  echo "Megatron dynamic inference requires POLICY_CP=1." >&2
  exit 1
fi
if (( TRAIN_WORLD_SIZE % (POLICY_TP * POLICY_CP) != 0 )); then
  echo "Training world size must be divisible by POLICY_TP * POLICY_CP." >&2
  exit 1
fi
if (( INFERENCE_WORLD_SIZE % INFER_TP != 0 )); then
  echo "Inference world size must be divisible by INFER_TP." >&2
  exit 1
fi
TRAIN_DP_SIZE=$((TRAIN_WORLD_SIZE / (POLICY_TP * POLICY_CP)))
INFERENCE_DP_SIZE=$((INFERENCE_WORLD_SIZE / INFER_TP))

# With ETP=1, world_size must be divisible by EP (not by TP*EP).
largest_ep() {
  local world="$1"
  local ep=8
  while (( ep > world || world % ep != 0 )); do
    ep=$((ep / 2))
  done
  printf '%d' "${ep}"
}

POLICY_EP="${POLICY_EP:-$(largest_ep "${TRAIN_WORLD_SIZE}")}"
INFER_EP="${INFER_EP:-$(largest_ep "${INFERENCE_WORLD_SIZE}")}"
if (( TRAIN_WORLD_SIZE % POLICY_EP != 0 )); then
  echo "Training world size must be divisible by POLICY_EP (ETP=1)." >&2
  exit 1
fi
if [[ "${GENERATION_BACKEND}" == "vllm" ]]; then
  # vLLM EP = DP_internal * TP. Async GRPO cannot use internal DP, so EP must
  # equal TP; Ray DP then matches Megatron's world/TP replica count.
  if [[ "${INFER_EP}" != "${INFER_TP}" ]]; then
    echo "Forcing INFER_EP=${INFER_TP} for vLLM (was ${INFER_EP}) for async-safe Ray DP." >&2
  fi
  INFER_EP="${INFER_TP}"
elif (( INFERENCE_WORLD_SIZE % INFER_EP != 0 )); then
  echo "Inference world size must be divisible by INFER_EP (ETP=1)." >&2
  exit 1
fi

MAX_STEPS="${MAX_STEPS:-1000000}"
MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-4096}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
# Two prompt groups per inference DP replica (non-colocated default: 6 gen
# nodes → DP=3 → prompts=6; colocated 8n4g → DP=4 → prompts=8).
NUM_PROMPTS="${NUM_PROMPTS:-$((INFERENCE_DP_SIZE * 2))}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
TRAIN_GBS="${TRAIN_GBS:-$((NUM_PROMPTS * NUM_GENERATIONS))}"
EXPECTED_TRAIN_GBS=$((NUM_PROMPTS * NUM_GENERATIONS))
VAL_GBS="${VAL_GBS:-64}"
VAL_SIZE="${VAL_SIZE:-64}"
if (( TRAIN_GBS != EXPECTED_TRAIN_GBS )); then
  echo "TRAIN_GBS (${TRAIN_GBS}) must equal NUM_PROMPTS * NUM_GENERATIONS (${EXPECTED_TRAIN_GBS})." >&2
  exit 1
fi
if (( TRAIN_GBS % TRAIN_DP_SIZE != 0 )); then
  echo "TRAIN_GBS (${TRAIN_GBS}) must be divisible by training DP size (${TRAIN_DP_SIZE})." >&2
  exit 1
fi
if (( VAL_GBS % TRAIN_DP_SIZE != 0 )); then
  echo "VAL_GBS (${VAL_GBS}) must be divisible by training DP size (${TRAIN_DP_SIZE})." >&2
  exit 1
fi
REFIT_BACKEND="${REFIT_BACKEND:-nccl}"
OPTIMIZER_CPU_OFFLOAD="${OPTIMIZER_CPU_OFFLOAD:-false}"
BUFFER_SIZE_GB="${BUFFER_SIZE_GB:-8}"
OFFLOAD_OPTIMIZER_FOR_LOGPROB="${OFFLOAD_OPTIMIZER_FOR_LOGPROB:-false}"
if [[ "${OPTIMIZER_CPU_OFFLOAD}" == "true" ]]; then
  OPTIMIZER_OFFLOAD_FRACTION="${OPTIMIZER_OFFLOAD_FRACTION:-1.0}"
else
  OPTIMIZER_OFFLOAD_FRACTION="${OPTIMIZER_OFFLOAD_FRACTION:-0.0}"
fi
# Lower-precision Adam moments; set to empty to skip the override.
EXP_AVG_DTYPE="${EXP_AVG_DTYPE:-bfloat16}"
EXP_AVG_SQ_DTYPE="${EXP_AVG_SQ_DTYPE:-bfloat16}"
STORE_PARAM_REMAINDERS="${STORE_PARAM_REMAINDERS:-true}"
# Megatron defaults for stable inference/training parity.
MEGATRON_ENABLE_CHUNKED_PREFILL="${MEGATRON_ENABLE_CHUNKED_PREFILL:-true}"
MEGATRON_CUDA_GRAPH_IMPL="${MEGATRON_CUDA_GRAPH_IMPL:-local}"
MEGATRON_CUDA_GRAPH_SCOPE="${MEGATRON_CUDA_GRAPH_SCOPE:-block}"
MEGATRON_NUM_CUDA_GRAPHS="${MEGATRON_NUM_CUDA_GRAPHS:--1}"
MEGATRON_USE_CUDA_GRAPHS_FOR_NON_DECODE="${MEGATRON_USE_CUDA_GRAPHS_FOR_NON_DECODE:-false}"
MOE_PAD_EXPERTS_FOR_CG="${MOE_PAD_EXPERTS_FOR_CG:-true}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"
MAX_TRAJECTORY_AGE_STEPS="${MAX_TRAJECTORY_AGE_STEPS:-2}"
IN_FLIGHT_WEIGHT_UPDATES="${IN_FLIGHT_WEIGHT_UPDATES:-true}"

JOB_NAME="${JOB_NAME:-nemotron-omni-clevr-${GENERATION_BACKEND}-8n4g}"
EXP_NAME="${EXP_NAME:-${JOB_NAME}}"
PRECISION_RECIPE="${PRECISION_RECIPE:-bf16}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"
WANDB_PROJ="${WANDB_PROJ:-mllm-rl-dev}"
WANDB_GROUP="${WANDB_GROUP:-adlr}"
WANDB_NAME="${WANDB_NAME:-${EXP_NAME}-${PRECISION_RECIPE}-internal-repo}"
RESULTS_DIR="${RESULTS_DIR:-${WORKSPACE_ROOT}/results/nemo-rl-omni/${JOB_NAME}}"
CHECKPOINTING_ENABLED="${CHECKPOINTING_ENABLED:-false}"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-${RESULTS_DIR}/slurm}"

# OCI-HSG Slurm defaults. Override any of these for a reservation or a
# different account/partition.
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-nemotron_sw_post}" # coreai_dlalgo_mcore nemotron_sw_post
SBATCH_PARTITION="${SBATCH_PARTITION:-batch_long}"
SBATCH_QOS="${SBATCH_QOS:-}"
SBATCH_TIME="${SBATCH_TIME:-04:00:00}"
SBATCH_RESERVATION="${SBATCH_RESERVATION:-}"
SBATCH_SEGMENT="${NUM_NODES}"

mkdir -p \
  "${HF_HUB_CACHE}" \
  "${HF_DATASETS_CACHE}" \
  "${HF_MODULES_CACHE}" \
  "${TRANSFORMERS_CACHE}" \
  "${NRL_MEGATRON_CHECKPOINT_DIR}" \
  "${XDG_CACHE_HOME}" \
  "${TORCH_HOME}" \
  "${TRITON_CACHE_DIR}" \
  "${SLURM_LOG_DIR}"

if [[ ! -f "${CONTAINER}" ]]; then
  echo "Container image does not exist: ${CONTAINER}" >&2
  exit 1
fi
if [[ ! -f "${NEMORL}/ray.sub" || ! -f "${NEMORL}/${CONFIG}" ]]; then
  echo "NeMo-RL launcher or config is missing under: ${NEMORL}" >&2
  exit 1
fi

OPTIMIZER_DTYPE_OVERRIDES=""
if [[ -n "${EXP_AVG_DTYPE}" ]]; then
  OPTIMIZER_DTYPE_OVERRIDES+=" ++policy.megatron_cfg.optimizer.exp_avg_dtype=${EXP_AVG_DTYPE}"
fi
if [[ -n "${EXP_AVG_SQ_DTYPE}" ]]; then
  OPTIMIZER_DTYPE_OVERRIDES+=" ++policy.megatron_cfg.optimizer.exp_avg_sq_dtype=${EXP_AVG_SQ_DTYPE}"
fi
if [[ -n "${STORE_PARAM_REMAINDERS}" ]]; then
  OPTIMIZER_DTYPE_OVERRIDES+=" ++policy.megatron_cfg.optimizer.store_param_remainders=${STORE_PARAM_REMAINDERS}"
fi

REFIT_ENV_EXPORTS=""
REFIT_BUFFER_MEMORY_RATIO=""

if [[ "${GENERATION_BACKEND}" == "megatron" ]]; then
  GEN_OVERRIDES="\
++policy.generation.stop_strings=null \
++policy.generation.bad_words=null \
policy.generation.mcore_generation_config.tensor_model_parallel_size=${INFER_TP} \
policy.generation.mcore_generation_config.expert_model_parallel_size=${INFER_EP} \
policy.generation.mcore_generation_config.expert_tensor_parallel_size=1 \
++policy.generation.mcore_generation_config.context_parallel_size=${POLICY_CP} \
++policy.generation.mcore_generation_config.mamba_inference_ssm_states_dtype=float32 \
++policy.generation.mcore_generation_config.mamba_inference_conv_states_dtype=float32 \
++policy.generation.mcore_generation_config.logprobs_mode=raw_logprobs \
policy.generation.mcore_generation_config.enable_chunked_prefill=${MEGATRON_ENABLE_CHUNKED_PREFILL} \
++policy.generation.mcore_generation_config.async_sched_mode=async \
policy.generation.mcore_generation_config.cuda_graph_impl=${MEGATRON_CUDA_GRAPH_IMPL} \
policy.generation.mcore_generation_config.inference_cuda_graph_scope=${MEGATRON_CUDA_GRAPH_SCOPE} \
policy.generation.mcore_generation_config.num_cuda_graphs=${MEGATRON_NUM_CUDA_GRAPHS} \
policy.generation.mcore_generation_config.use_cuda_graphs_for_non_decode_steps=${MEGATRON_USE_CUDA_GRAPHS_FOR_NON_DECODE} \
policy.generation.mcore_generation_config.refit_backend=${REFIT_BACKEND} \
policy.generation.mcore_generation_config.buffer_size_gb=${BUFFER_SIZE_GB} \
policy.generation.mcore_generation_config.moe_pad_experts_for_cuda_graph_inference=${MOE_PAD_EXPERTS_FOR_CG} \
policy.generation.mcore_generation_config.max_model_len=${MAX_SEQUENCE_LENGTH} \
policy.generation.mcore_generation_config.max_tokens=${MAX_SEQUENCE_LENGTH}"
else
  # Non-colocated refit NCCL-broadcasts weights in packed chunks sized at
  # NRL_REFIT_BUFFER_MEMORY_RATIO * total HBM (the 0.02 default is 3.7GiB on
  # GB200), which the ~170GiB-deep trainer cannot spare. Exported globally on
  # purpose: producer and consumer both derive chunk boundaries from it and must
  # agree.
  REFIT_BUFFER_MEMORY_RATIO="${NRL_REFIT_BUFFER_MEMORY_RATIO:-0.005}"
  REFIT_ENV_EXPORTS="export NRL_REFIT_BUFFER_MEMORY_RATIO=${REFIT_BUFFER_MEMORY_RATIO}"

  # Omni vLLM A/B path.
  VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.5}"
  VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-true}"
  # Per-step forward-pass token budget; mirrors mcore_generation_config.max_tokens.
  VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-${MAX_SEQUENCE_LENGTH}}"
  GEN_OVERRIDES="\
++policy.generation.stop_strings=null \
++policy.generation.bad_words=null \
policy.generation.vllm_cfg.async_engine=${ASYNC_GRPO} \
policy.generation.vllm_cfg.skip_tokenizer_init=false \
policy.generation.vllm_cfg.tensor_parallel_size=${INFER_TP} \
policy.generation.vllm_cfg.pipeline_parallel_size=1 \
policy.generation.vllm_cfg.expert_parallel_size=${INFER_EP} \
policy.generation.vllm_cfg.max_model_len=${MAX_SEQUENCE_LENGTH} \
++policy.generation.vllm_cfg.cap_max_tokens_to_context=true \
policy.generation.vllm_cfg.gpu_memory_utilization=${VLLM_GPU_MEMORY_UTILIZATION} \
policy.generation.vllm_cfg.enforce_eager=${VLLM_ENFORCE_EAGER} \
++policy.generation.vllm_cfg.enable_prefix_caching=false \
policy.generation.vllm_cfg.logprobs_mode=raw_logprobs \
++policy.generation.vllm_kwargs.limit_mm_per_prompt.image=2 \
++policy.generation.vllm_kwargs.max_num_batched_tokens=${VLLM_MAX_NUM_BATCHED_TOKENS} \
++policy.generation.vllm_kwargs.mamba_ssm_cache_dtype=float32 \
++policy.generation.vllm_kwargs.skip_mm_profiling=true \
++policy.generation.vllm_kwargs.kernel_config.enable_flashinfer_autotune=false \
++policy.generation.vllm_kwargs.kernel_config.moe_backend=triton"
fi

export NUM_NODES GPUS_PER_NODE CONTAINER
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export NRL_VENVS_TRUST_EXISTING="${NRL_VENVS_TRUST_EXISTING:-1}"
export NRL_FORCE_REBUILD_VENVS="${NRL_FORCE_REBUILD_VENVS:-false}"
export NEMO_RL_VENV_DIR="${NEMO_RL_VENV_DIR:-/opt/ray_venvs}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export FLASHINFER_DISABLE_VERSION_CHECK="${FLASHINFER_DISABLE_VERSION_CHECK:-1}"
export NVTE_FWD_LAYERNORM_SM_MARGIN="${NVTE_FWD_LAYERNORM_SM_MARGIN:-16}"
export NVTE_BWD_LAYERNORM_SM_MARGIN="${NVTE_BWD_LAYERNORM_SM_MARGIN:-16}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

BRIDGE="${CONTAINER_NEMORL}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge"
export COMMAND="\
set -euo pipefail
NRL_SLURM_JOB_ID=\$(basename \"\$(dirname \"\$0\")\")
NRL_SLURM_JOB_ID=\${NRL_SLURM_JOB_ID%%-*}
cd ${CONTAINER_NEMORL}
export HF_HOME=${HF_HOME}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE}
export HF_HUB_CACHE=${HF_HUB_CACHE}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE}
export HF_MODULES_CACHE=${HF_MODULES_CACHE}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE}
export NRL_MEGATRON_CHECKPOINT_DIR=${NRL_MEGATRON_CHECKPOINT_DIR}
export XDG_CACHE_HOME=${XDG_CACHE_HOME}
export TORCH_HOME=${TORCH_HOME}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR}
${REFIT_ENV_EXPORTS}
export PYTHONPATH=${CONTAINER_NEMORL}:${BRIDGE}/src:${BRIDGE}/3rdparty/Megatron-LM\${PYTHONPATH:+:\$PYTHONPATH}
uv run --no-sync python examples/run_vlm_grpo.py --config ${CONFIG} \
cluster.num_nodes=${NUM_NODES} \
cluster.gpus_per_node=${GPUS_PER_NODE} \
policy.model_name=${MODEL_NAME} \
policy.tokenizer.name=${MODEL_NAME} \
policy.is_vlm=true \
policy.megatron_cfg.tensor_model_parallel_size=${POLICY_TP} \
policy.megatron_cfg.expert_model_parallel_size=${POLICY_EP} \
policy.megatron_cfg.expert_tensor_parallel_size=1 \
policy.megatron_cfg.context_parallel_size=${POLICY_CP} \
policy.megatron_cfg.optimizer.optimizer_cpu_offload=${OPTIMIZER_CPU_OFFLOAD} \
policy.megatron_cfg.optimizer.optimizer_offload_fraction=${OPTIMIZER_OFFLOAD_FRACTION} \
policy.offload_optimizer_for_logprob=${OFFLOAD_OPTIMIZER_FOR_LOGPROB} \
${OPTIMIZER_DTYPE_OVERRIDES} \
policy.generation.backend=${GENERATION_BACKEND} \
policy.generation.colocated.enabled=${COLOCATED_ENABLED} \
policy.generation.colocated.resources.num_nodes=${NUM_GEN_NODES} \
policy.generation.colocated.resources.gpus_per_node=${GEN_GPUS_PER_NODE} \
${GEN_OVERRIDES} \
policy.max_total_sequence_length=${MAX_SEQUENCE_LENGTH} \
policy.generation.max_new_tokens=${MAX_NEW_TOKENS} \
grpo.async_grpo.enabled=${ASYNC_GRPO} \
grpo.async_grpo.max_trajectory_age_steps=${MAX_TRAJECTORY_AGE_STEPS} \
grpo.async_grpo.in_flight_weight_updates=${IN_FLIGHT_WEIGHT_UPDATES} \
loss_fn.use_importance_sampling_correction=true \
grpo.num_prompts_per_step=${NUM_PROMPTS} \
grpo.num_generations_per_prompt=${NUM_GENERATIONS} \
grpo.val_batch_size=${VAL_GBS} \
grpo.max_val_samples=${VAL_SIZE} \
policy.train_global_batch_size=${TRAIN_GBS} \
grpo.max_num_steps=${MAX_STEPS} \
checkpointing.enabled=${CHECKPOINTING_ENABLED} \
checkpointing.checkpoint_dir=${RESULTS_DIR} \
logger.log_dir=${RESULTS_DIR} \
logger.wandb_enabled=${WANDB_ENABLED} \
logger.wandb.name=${WANDB_NAME}-\${NRL_SLURM_JOB_ID} \
logger.wandb.project=${WANDB_PROJ} \
+logger.wandb.entity=${WANDB_GROUP} \
${EXTRA_OVERRIDES}"

echo "Submitting ${JOB_NAME}: ${NUM_NODES} node(s), ${GPUS_PER_NODE} GPU(s)/node"
if [[ "${COLOCATED_ENABLED}" == "true" ]]; then
  echo "  layout: colocated async on all ${NUM_NODES} node(s)"
else
  echo "  split: ${NUM_GEN_NODES} gen node(s) / $((NUM_NODES - NUM_GEN_NODES)) train node(s)"
fi
echo "  generation backend: ${GENERATION_BACKEND} colocated=${COLOCATED_ENABLED} async=${ASYNC_GRPO}"
echo "  training world size: ${TRAIN_WORLD_SIZE} (TP=${POLICY_TP}, EP=${POLICY_EP}, DP=${TRAIN_DP_SIZE})"
if [[ "${GENERATION_BACKEND}" == "megatron" ]]; then
  echo "  inference world size: ${INFERENCE_WORLD_SIZE} (TP=${INFER_TP}, EP=${INFER_EP}, DP=${INFERENCE_DP_SIZE})"
  echo "  refit backend: ${REFIT_BACKEND}"
else
  echo "  inference world size: ${INFERENCE_WORLD_SIZE} (TP=${INFER_TP}, EP=${INFER_EP}, DP=${INFERENCE_DP_SIZE})"
  echo "  vLLM: mem_util=${VLLM_GPU_MEMORY_UTILIZATION:-0.5} eager=${VLLM_ENFORCE_EAGER:-true} max_batched_tokens=${VLLM_MAX_NUM_BATCHED_TOKENS:-${MAX_SEQUENCE_LENGTH}}"
  if [[ -n "${REFIT_ENV_EXPORTS}" ]]; then
    echo "  refit: buffer_memory_ratio=${REFIT_BUFFER_MEMORY_RATIO}"
  fi
fi
echo "  seq/new_tokens: ${MAX_SEQUENCE_LENGTH}/${MAX_NEW_TOKENS}"
echo "  prompts/generations/train_gbs: ${NUM_PROMPTS}/${NUM_GENERATIONS}/${TRAIN_GBS}"
echo "  async: max_trajectory_age=${MAX_TRAJECTORY_AGE_STEPS} in_flight_weight_updates=${IN_FLIGHT_WEIGHT_UPDATES}"
echo "  val_gbs/val_size: ${VAL_GBS}/${VAL_SIZE}"
echo "  optimizer moments: exp_avg=${EXP_AVG_DTYPE:-<unset>} exp_avg_sq=${EXP_AVG_SQ_DTYPE:-<unset>} store_param_remainders=${STORE_PARAM_REMAINDERS:-<unset>}"
echo "  W&B: ${WANDB_GROUP}/${WANDB_PROJ}/${WANDB_NAME}-<slurm-job-id> (enabled=${WANDB_ENABLED})"
echo "  Hugging Face cache: ${HF_HUB_CACHE}"
echo "  MCore checkpoint cache: ${NRL_MEGATRON_CHECKPOINT_DIR}"

SBATCH_ARGS=(
  --nodes="${NUM_NODES}"
  --account="${SBATCH_ACCOUNT}"
  --partition="${SBATCH_PARTITION}"
  --job-name="${JOB_NAME}"
  --time="${SBATCH_TIME}"
  --output="${SLURM_LOG_DIR}/%j.out"
  --error="${SLURM_LOG_DIR}/%j.out"
  --gres="gpu:${GPUS_PER_NODE}"
  --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"90","reason":"data_loading","description":"Async GRPO RL training: training GPUs idle during rollout collection (~30min) and validation each step"}}'
  --exclusive
  --mem=0
  --dependency=singleton
  --segment="${SBATCH_SEGMENT}"
)
if [[ -n "${SBATCH_QOS}" ]]; then
  SBATCH_ARGS+=(--qos="${SBATCH_QOS}")
fi
if [[ -n "${SBATCH_RESERVATION}" ]]; then
  SBATCH_ARGS+=(--reservation="${SBATCH_RESERVATION}")
fi

BASE_LOG_DIR="${SLURM_LOG_DIR}" \
MOUNTS="${MOUNTS:-/lustre:/lustre},${NEMORL}:${CONTAINER_NEMORL}" \
sbatch "${SBATCH_ARGS[@]}" "${NEMORL}/ray.sub"

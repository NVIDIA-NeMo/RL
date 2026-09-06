#!/usr/bin/env bash
set -euo pipefail

# Eight-node/four-GPU NeMo-RL v2 SingleController launcher for Nemotron Omni.
# The default non-colocated layout matches the NeMo-RL v1 parity launchers:
# two training nodes and six Megatron generation nodes.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
NEMORL="${NEMORL:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
CONTAINER_NEMORL="${CONTAINER_NEMORL:-/opt/nemo-rl}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-${NEMORL}/workspace}"
TASK="${TASK:-clevr}"
COLOCATED="${COLOCATED:-false}"
ASYNC_GRPO="${ASYNC_GRPO:-true}"

if [[ "${COLOCATED}" != "false" ]]; then
  echo "SingleController currently requires COLOCATED=false." >&2
  exit 1
fi
if [[ "${ASYNC_GRPO}" != "true" ]]; then
  echo "SingleController uses async_rl and currently requires ASYNC_GRPO=true." >&2
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
NUM_GEN_NODES="${NUM_GEN_NODES:-6}"
GEN_GPUS_PER_NODE="${GEN_GPUS_PER_NODE:-${GPUS_PER_NODE}}"
NUM_TRAIN_NODES=$((NUM_NODES - NUM_GEN_NODES))
TRAIN_WORLD_SIZE=$((NUM_TRAIN_NODES * GPUS_PER_NODE))
INFERENCE_WORLD_SIZE=$((NUM_GEN_NODES * GEN_GPUS_PER_NODE))
NUM_STORAGE_UNITS="${NUM_STORAGE_UNITS:-$((2 * NUM_NODES))}"

if (( NUM_NODES < 2 || NUM_GEN_NODES <= 0 || NUM_GEN_NODES >= NUM_NODES )); then
  echo "Non-colocated mode requires 0 < NUM_GEN_NODES < NUM_NODES." >&2
  exit 1
fi
if (( GEN_GPUS_PER_NODE != GPUS_PER_NODE )); then
  echo "Multi-node generation must reserve complete GPU nodes." >&2
  exit 1
fi

POLICY_TP="${POLICY_TP:-8}"
POLICY_EP="${POLICY_EP:-8}"
INFER_TP="${INFER_TP:-8}"
INFER_EP="${INFER_EP:-8}"
if (( TRAIN_WORLD_SIZE % POLICY_TP != 0 || TRAIN_WORLD_SIZE % POLICY_EP != 0 )); then
  echo "Training world size ${TRAIN_WORLD_SIZE} must be divisible by POLICY_TP and POLICY_EP." >&2
  exit 1
fi
if (( INFERENCE_WORLD_SIZE % INFER_TP != 0 || INFERENCE_WORLD_SIZE % INFER_EP != 0 )); then
  echo "Inference world size ${INFERENCE_WORLD_SIZE} must be divisible by INFER_TP and INFER_EP." >&2
  exit 1
fi
TRAIN_DP_SIZE=$((TRAIN_WORLD_SIZE / POLICY_TP))
INFERENCE_DP_SIZE=$((INFERENCE_WORLD_SIZE / INFER_TP))

MODEL_NAME="${MODEL_NAME:-nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16}"
MAX_STEPS="${MAX_STEPS:-1000000}"
NUM_PROMPTS_PER_STEP="${NUM_PROMPTS_PER_STEP:-$((INFERENCE_DP_SIZE * 2))}"
NUM_GENERATIONS_PER_PROMPT="${NUM_GENERATIONS_PER_PROMPT:-8}"
TRAIN_GBS="${TRAIN_GBS:-$((NUM_PROMPTS_PER_STEP * NUM_GENERATIONS_PER_PROMPT))}"
if (( TRAIN_GBS % TRAIN_DP_SIZE != 0 )); then
  echo "TRAIN_GBS ${TRAIN_GBS} must be divisible by training DP size ${TRAIN_DP_SIZE}." >&2
  exit 1
fi
MAX_LOOKAHEAD_VERSIONS="${MAX_LOOKAHEAD_VERSIONS:-1}"
MAX_INFLIGHT_PROMPTS="${MAX_INFLIGHT_PROMPTS:-$((NUM_PROMPTS_PER_STEP * (MAX_LOOKAHEAD_VERSIONS + 1)))}"
MAX_BUFFERED_ROLLOUTS="${MAX_BUFFERED_ROLLOUTS:-$((NUM_PROMPTS_PER_STEP * (MAX_LOOKAHEAD_VERSIONS + 1)))}"
ASYNC_RL_DIAGNOSTICS="${ASYNC_RL_DIAGNOSTICS:-false}"
TOKEN_CAPTURE_ENABLED="${TOKEN_CAPTURE_ENABLED:-false}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"
REFIT_BACKEND="${REFIT_BACKEND:-nccl}"
BUFFER_SIZE_GB="${BUFFER_SIZE_GB:-8}"
OPTIMIZER_CPU_OFFLOAD="${OPTIMIZER_CPU_OFFLOAD:-false}"
OFFLOAD_OPTIMIZER_FOR_LOGPROB="${OFFLOAD_OPTIMIZER_FOR_LOGPROB:-false}"
if [[ "${OPTIMIZER_CPU_OFFLOAD}" == "true" ]]; then
  OPTIMIZER_OFFLOAD_FRACTION="${OPTIMIZER_OFFLOAD_FRACTION:-1.0}"
else
  OPTIMIZER_OFFLOAD_FRACTION="${OPTIMIZER_OFFLOAD_FRACTION:-0.0}"
fi
USE_PRECISION_AWARE_OPTIMIZER="${USE_PRECISION_AWARE_OPTIMIZER:-true}"
EXP_AVG_DTYPE="${EXP_AVG_DTYPE:-bfloat16}"
EXP_AVG_SQ_DTYPE="${EXP_AVG_SQ_DTYPE:-bfloat16}"
STORE_PARAM_REMAINDERS="${STORE_PARAM_REMAINDERS:-true}"
MEGATRON_ENABLE_CHUNKED_PREFILL="${MEGATRON_ENABLE_CHUNKED_PREFILL:-true}"
MEGATRON_TRANSFORMER_IMPL="${MEGATRON_TRANSFORMER_IMPL:-inference_optimized}"
MEGATRON_CUDA_GRAPH_IMPL="${MEGATRON_CUDA_GRAPH_IMPL:-local}"
MEGATRON_CUDA_GRAPH_SCOPE="${MEGATRON_CUDA_GRAPH_SCOPE:-block}"
MEGATRON_NUM_CUDA_GRAPHS="${MEGATRON_NUM_CUDA_GRAPHS:--1}"
MEGATRON_USE_CUDA_GRAPHS_FOR_NON_DECODE="${MEGATRON_USE_CUDA_GRAPHS_FOR_NON_DECODE:-false}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-false}"
if [[ "${MEGATRON_TRANSFORMER_IMPL}" != "inference_optimized" &&
      "${MEGATRON_CUDA_GRAPH_IMPL}" == "local" && "${INFER_EP}" -gt 1 ]]; then
  MOE_PAD_EXPERTS_FOR_CG="${MOE_PAD_EXPERTS_FOR_CG:-true}"
else
  MOE_PAD_EXPERTS_FOR_CG=false
fi

TASK_VIDEO_EXPORTS=""
case "${TASK}" in
  clevr)
    CONFIG="${CONFIG:-examples/configs/recipes/vlm/vlm_grpo-nemotron-omni-30ba3b-clevr-8n4g-megatron-single-controller-async.v1.yaml}"
    MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-4096}"
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
    # SingleController currently rejects validation during setup.
    VAL_PERIOD="${VAL_PERIOD:-0}"
    VAL_AT_START="${VAL_AT_START:-false}"
    VAL_AT_END="${VAL_AT_END:-false}"
    VAL_GBS="${VAL_GBS:-64}"
    VAL_SIZE="${VAL_SIZE:-64}"
    VISION_EMBEDDING_CACHE_MAX_BYTES="${VISION_EMBEDDING_CACHE_MAX_BYTES:-0}"
    OVERLAP_PARAM_GATHER="${OVERLAP_PARAM_GATHER:-true}"
    TASK_ENV="CLEVR_DATASET_MODE=config"
    WANDB_NAME_SUFFIX="nrl_v2_sctq_image"
    TASK_OVERRIDES=""
    ;;
  vstat)
    CONFIG="${CONFIG:-examples/configs/recipes/vlm/vlm_grpo-nemotron-omni-30ba3b-16n8g-megatron-tp4ep4-async-gym-video.v1.yaml}"
    MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-8192}"
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
    # SingleController currently rejects validation during setup.
    VAL_PERIOD="${VAL_PERIOD:-0}"
    VAL_AT_START="${VAL_AT_START:-false}"
    VAL_AT_END="${VAL_AT_END:-false}"
    VAL_GBS="${VAL_GBS:-2}"
    VAL_SIZE="${VAL_SIZE:-2}"
    DATA_ROOT="${DATA_ROOT:-${CONTAINER_NEMORL}/workspace/datasets/vstat-8n4g}"
    NEMO_RL_VIDEO_TRAIN_JSONL="${NEMO_RL_VIDEO_TRAIN_JSONL:-${DATA_ROOT}/train-gym.jsonl}"
    NEMO_RL_VIDEO_VAL_JSONL="${NEMO_RL_VIDEO_VAL_JSONL:-${DATA_ROOT}/val-gym.jsonl}"
    NEMO_RL_VIDEO_MEDIA_ROOT="${NEMO_RL_VIDEO_MEDIA_ROOT:-${DATA_ROOT}/media}"
    HF_DATASET="${HF_DATASET:-ShushengYang/VSTAT}"
    NUM_DATA_ROWS="${NUM_DATA_ROWS:-256}"
    NUM_FRAMES="${NUM_FRAMES:-16}"
    TEMPORAL_PATCH_SIZE="${TEMPORAL_PATCH_SIZE:-2}"
    VIDEO_TARGET_PATCHES="${VIDEO_TARGET_PATCHES:-1024}"
    MIN_GENERATION_TOKENS="${MIN_GENERATION_TOKENS:-2000}"
    VISION_EMBEDDING_CACHE_MAX_BYTES="${VISION_EMBEDDING_CACHE_MAX_BYTES:-536870912}"
    OVERLAP_PARAM_GATHER="${OVERLAP_PARAM_GATHER:-false}"
    ENABLE_THINKING="${ENABLE_THINKING:-true}"
    PREPARE_VSTAT="${PREPARE_VSTAT:-false}"
    WANDB_NAME_SUFFIX="nrl_v2_sctq_video"
    TASK_VIDEO_EXPORTS="\
export NRL_VIDEO_BACKEND=${NRL_VIDEO_BACKEND:-torchcodec}
export NRL_VIDEO_SAMPLING_STYLE=${NRL_VIDEO_SAMPLING_STYLE:-nemotron_vl}
export NRL_VIDEO_TEMPORAL_PATCH_SIZE=${TEMPORAL_PATCH_SIZE}
export VLLM_VIDEO_LOADER_BACKEND=${VLLM_VIDEO_LOADER_BACKEND:-nemotron_vl}
export NEMO_RL_VIDEO_TRAIN_JSONL=${NEMO_RL_VIDEO_TRAIN_JSONL}
export NEMO_RL_VIDEO_VAL_JSONL=${NEMO_RL_VIDEO_VAL_JSONL}
export NEMO_RL_VIDEO_MEDIA_ROOT=${NEMO_RL_VIDEO_MEDIA_ROOT}"
    TASK_ENV="DATA_ROOT=${DATA_ROOT} HF_DATASET=${HF_DATASET} NUM_DATA_ROWS=${NUM_DATA_ROWS} NUM_FRAMES=${NUM_FRAMES} TEMPORAL_PATCH_SIZE=${TEMPORAL_PATCH_SIZE} VIDEO_TARGET_PATCHES=${VIDEO_TARGET_PATCHES} MIN_GENERATION_TOKENS=${MIN_GENERATION_TOKENS} ENABLE_THINKING=${ENABLE_THINKING} PREPARE_VSTAT=${PREPARE_VSTAT}"
    TASK_OVERRIDES="\
policy.tokenizer.chat_template_kwargs.enable_thinking=${ENABLE_THINKING} \
policy.megatron_cfg.env_vars.TORCH_CUDA_ARCH_LIST=\"'${TORCH_CUDA_ARCH_LIST:-10.0}'\" \
policy.megatron_cfg.freeze_vision_model=false \
policy.megatron_cfg.freeze_vision_projection=false \
policy.megatron_cfg.freeze_moe_router=false \
policy.megatron_cfg.mtp_num_layers=0 \
policy.megatron_cfg.mtp_use_repeated_layer=true \
policy.megatron_cfg.mtp_detach_heads=true \
policy.megatron_cfg.mtp_loss_scaling_factor=0.0 \
policy.megatron_cfg.pipeline_model_parallel_size=1 \
policy.megatron_cfg.moe_shared_expert_overlap=false \
policy.megatron_cfg.radio_force_cpe_eval_mode=true \
policy.megatron_cfg.clear_memory_caches_before_refit=true \
policy.megatron_cfg.distributed_data_parallel_config.overlap_grad_reduce=false \
policy.megatron_cfg.distributed_data_parallel_config.overlap_param_gather=false \
policy.megatron_cfg.optimizer.params_dtype=float32 \
policy.megatron_cfg.optimizer.use_precision_aware_optimizer=${USE_PRECISION_AWARE_OPTIMIZER} \
policy.megatron_cfg.optimizer.optimizer_cpu_offload=${OPTIMIZER_CPU_OFFLOAD} \
policy.megatron_cfg.optimizer.optimizer_offload_fraction=${OPTIMIZER_OFFLOAD_FRACTION} \
policy.generation.mcore_generation_config.parsers=[nemotron-v3-reasoning,qwen3-coder-tool] \
++policy.generation.mcore_generation_config.video_num_frames=${NUM_FRAMES} \
++policy.generation.mcore_generation_config.video_temporal_patch_size=${TEMPORAL_PATCH_SIZE} \
++policy.generation.mcore_generation_config.video_target_num_patches=${VIDEO_TARGET_PATCHES} \
data.max_input_seq_length=${MAX_SEQUENCE_LENGTH} \
++data.default.num_frames=${NUM_FRAMES} \
++data.default.video_sampling_style=nemotron_vl \
++data.default.video_temporal_patch_size=${TEMPORAL_PATCH_SIZE} \
++data.default.min_generation_tokens=${MIN_GENERATION_TOKENS} \
data.default.video_maintain_aspect_ratio=true \
data.train.data_path=${NEMO_RL_VIDEO_TRAIN_JSONL} \
data.validation.data_path=${NEMO_RL_VIDEO_VAL_JSONL} \
++env.nemo_gym.policy_model.responses_api_models.vllm_model.chat_template_kwargs.enable_thinking=${ENABLE_THINKING} \
grpo.deduplicate_multimodal_data=false"
    ;;
  *)
    echo "TASK must be clevr or vstat (got ${TASK})." >&2
    exit 1
    ;;
esac

if (( TRAIN_GBS != NUM_PROMPTS_PER_STEP * NUM_GENERATIONS_PER_PROMPT )); then
  echo "TRAIN_GBS must equal NUM_PROMPTS_PER_STEP * NUM_GENERATIONS_PER_PROMPT." >&2
  exit 1
fi

RESULTS_DIR="${RESULTS_DIR:-${WORKSPACE_ROOT}/results/nemo-rl-v2-omni/${TASK}-8n4g}"
CHECKPOINTING_ENABLED="${CHECKPOINTING_ENABLED:-false}"
JOB_NAME="${JOB_NAME:-nemotron-omni-${TASK}-single-controller-8n4g}"
EXP_NAME="${EXP_NAME:-${JOB_NAME}}"
PRECISION_RECIPE="${PRECISION_RECIPE:-bf16}"
WANDB_PROJ="${WANDB_PROJ:-mllm-rl-dev}"
WANDB_GROUP="${WANDB_GROUP:-adlr}"
WANDB_NAME="${WANDB_NAME:-${EXP_NAME}-${PRECISION_RECIPE}-${WANDB_NAME_SUFFIX}}"
CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/coreai/users/cye/enroot/nemo-rl-nightly-gym.sqsh}"
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-nemotron_sw_post}"
SBATCH_PARTITION="${SBATCH_PARTITION:-batch_long}"
SBATCH_QOS="${SBATCH_QOS:-}"
SBATCH_TIME="${SBATCH_TIME:-04:00:00}"
SBATCH_RESERVATION="${SBATCH_RESERVATION:-}"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-${RESULTS_DIR}/slurm}"

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
if [[ ! -f "${NEMORL}/ray.sub" || ! -f "${NEMORL}/${CONFIG}" ||
      ! -f "${NEMORL}/examples/run_grpo_single_controller.py" ||
      ! -f "${NEMORL}/scripts/run_nemotron_omni_multimodal_single_controller_1n4g.sh" ]]; then
  echo "NeMo-RL launcher, config, entrypoint, or helper is missing under ${NEMORL}." >&2
  exit 1
fi
if [[ "${TASK}" == "vstat" && ! -f "${NEMORL}/scripts/prepare_nemotron_omni_vstat.py" ]]; then
  echo "VSTAT preparation script is missing under ${NEMORL}." >&2
  exit 1
fi

export NUM_NODES GPUS_PER_NODE CONTAINER
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export NRL_VENVS_TRUST_EXISTING="${NRL_VENVS_TRUST_EXISTING:-1}"
export NRL_FORCE_REBUILD_VENVS="${NRL_FORCE_REBUILD_VENVS:-false}"
export NEMO_RL_VENV_DIR="${NEMO_RL_VENV_DIR:-/opt/ray_venvs}"
export NEMO_GYM_VENV_DIR="${NEMO_GYM_VENV_DIR:-/opt/gym_venvs}"
export NEMO_GYM_EXTRA_ROOTS="${NEMO_GYM_EXTRA_ROOTS:-${CONTAINER_NEMORL}/3rdparty/Gym-workspace/Gym}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-10.0}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export FLASHINFER_DISABLE_VERSION_CHECK="${FLASHINFER_DISABLE_VERSION_CHECK:-1}"
export NVTE_FWD_LAYERNORM_SM_MARGIN="${NVTE_FWD_LAYERNORM_SM_MARGIN:-16}"
export NVTE_BWD_LAYERNORM_SM_MARGIN="${NVTE_BWD_LAYERNORM_SM_MARGIN:-16}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
if [[ "${TASK}" == "vstat" ]]; then
  export NRL_VIDEO_BACKEND="${NRL_VIDEO_BACKEND:-torchcodec}"
  export NRL_VIDEO_SAMPLING_STYLE="${NRL_VIDEO_SAMPLING_STYLE:-nemotron_vl}"
  export NRL_VIDEO_TEMPORAL_PATCH_SIZE="${TEMPORAL_PATCH_SIZE}"
  export VLLM_VIDEO_LOADER_BACKEND="${VLLM_VIDEO_LOADER_BACKEND:-nemotron_vl}"
  export NEMO_RL_VIDEO_TRAIN_JSONL
  export NEMO_RL_VIDEO_VAL_JSONL
  export NEMO_RL_VIDEO_MEDIA_ROOT
fi

ENABLE_NSYS="${ENABLE_NSYS:-false}"
if [[ "${ENABLE_NSYS}" == "true" ]]; then
  export NRL_NSYS_WORKER_PATTERNS="${NRL_NSYS_WORKER_PATTERNS:-*policy*,*megatron*}"
  export NRL_NSYS_PROFILE_STEP_RANGE="${NRL_NSYS_PROFILE_STEP_RANGE:-1:4}"
  export LD_LIBRARY_PATH="/usr/local/cuda/targets/aarch64-linux/lib:/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib:/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/lib/aarch64-linux-gnu:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
  export NRL_NSYS_EXTRA_OPTIONS="${NRL_NSYS_EXTRA_OPTIONS:-{\"o\":\"${CONTAINER_NEMORL}/workspace/nsys/%p\",\"cpuctxsw\":\"none\",\"force-overwrite\":\"true\"}}"
  mkdir -p "${WORKSPACE_ROOT}/nsys"
fi

export SETUP_COMMAND=""
if [[ "${TASK}" == "vstat" ]]; then
  export SETUP_COMMAND="cd ${CONTAINER_NEMORL} && bash tools/install_audio_deps.sh"
fi

# Reuse the one-node driver for environment setup and task-specific data
# preparation, then place topology/parity overrides last so they take priority.
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
export NEMO_RL_VENV_DIR=${NEMO_RL_VENV_DIR}
export NEMO_GYM_VENV_DIR=${NEMO_GYM_VENV_DIR}
export NEMO_GYM_EXTRA_ROOTS=${NEMO_GYM_EXTRA_ROOTS}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS}
export FLASHINFER_DISABLE_VERSION_CHECK=${FLASHINFER_DISABLE_VERSION_CHECK}
export NVTE_FWD_LAYERNORM_SM_MARGIN=${NVTE_FWD_LAYERNORM_SM_MARGIN}
export NVTE_BWD_LAYERNORM_SM_MARGIN=${NVTE_BWD_LAYERNORM_SM_MARGIN}
export NCCL_DEBUG=${NCCL_DEBUG}
${TASK_VIDEO_EXPORTS}
TASK=${TASK} \
NEMORL=${CONTAINER_NEMORL} \
WORKSPACE_ROOT=${CONTAINER_NEMORL}/workspace \
CONFIG=${CONFIG} \
MODEL_NAME=${MODEL_NAME} \
GPUS_PER_NODE=${GPUS_PER_NODE} \
TRAIN_GPUS=2 GEN_GPUS=2 POLICY_TP=2 POLICY_EP=2 INFER_TP=2 INFER_EP=2 \
MAX_STEPS=${MAX_STEPS} \
MAX_SEQUENCE_LENGTH=${MAX_SEQUENCE_LENGTH} \
MAX_NEW_TOKENS=${MAX_NEW_TOKENS} \
NUM_PROMPTS_PER_STEP=${NUM_PROMPTS_PER_STEP} \
NUM_GENERATIONS_PER_PROMPT=${NUM_GENERATIONS_PER_PROMPT} \
TRAIN_GBS=${TRAIN_GBS} \
MAX_LOOKAHEAD_VERSIONS=${MAX_LOOKAHEAD_VERSIONS} \
MAX_INFLIGHT_PROMPTS=${MAX_INFLIGHT_PROMPTS} \
MAX_BUFFERED_ROLLOUTS=${MAX_BUFFERED_ROLLOUTS} \
ASYNC_RL_DIAGNOSTICS=${ASYNC_RL_DIAGNOSTICS} \
TOKEN_CAPTURE_ENABLED=${TOKEN_CAPTURE_ENABLED} \
REFIT_BACKEND=${REFIT_BACKEND} \
BUFFER_SIZE_GB=${BUFFER_SIZE_GB} \
EXP_AVG_DTYPE=${EXP_AVG_DTYPE} \
EXP_AVG_SQ_DTYPE=${EXP_AVG_SQ_DTYPE} \
STORE_PARAM_REMAINDERS=${STORE_PARAM_REMAINDERS} \
MEGATRON_TRANSFORMER_IMPL=${MEGATRON_TRANSFORMER_IMPL} \
MEGATRON_CUDA_GRAPH_IMPL=${MEGATRON_CUDA_GRAPH_IMPL} \
MOE_PAD_EXPERTS_FOR_CG=${MOE_PAD_EXPERTS_FOR_CG} \
VISION_EMBEDDING_CACHE_MAX_BYTES=${VISION_EMBEDDING_CACHE_MAX_BYTES} \
WANDB_ENABLED=${WANDB_ENABLED} \
RESULTS_DIR=${RESULTS_DIR} \
${TASK_ENV} \
bash scripts/run_nemotron_omni_multimodal_single_controller_1n4g.sh \
cluster.num_nodes=${NUM_NODES} \
cluster.gpus_per_node=${GPUS_PER_NODE} \
policy.megatron_cfg.tensor_model_parallel_size=${POLICY_TP} \
policy.megatron_cfg.expert_model_parallel_size=${POLICY_EP} \
policy.megatron_cfg.distributed_data_parallel_config.overlap_param_gather=${OVERLAP_PARAM_GATHER} \
policy.megatron_cfg.optimizer.optimizer_cpu_offload=${OPTIMIZER_CPU_OFFLOAD} \
policy.megatron_cfg.optimizer.optimizer_offload_fraction=${OPTIMIZER_OFFLOAD_FRACTION} \
++policy.megatron_cfg.optimizer.exp_avg_dtype=${EXP_AVG_DTYPE} \
++policy.megatron_cfg.optimizer.exp_avg_sq_dtype=${EXP_AVG_SQ_DTYPE} \
++policy.megatron_cfg.optimizer.store_param_remainders=${STORE_PARAM_REMAINDERS} \
policy.offload_optimizer_for_logprob=${OFFLOAD_OPTIMIZER_FOR_LOGPROB} \
policy.generation.colocated.resources.num_nodes=${NUM_GEN_NODES} \
policy.generation.colocated.resources.gpus_per_node=${GEN_GPUS_PER_NODE} \
policy.generation.max_new_tokens=${MAX_NEW_TOKENS} \
policy.generation.mcore_generation_config.tensor_model_parallel_size=${INFER_TP} \
policy.generation.mcore_generation_config.expert_model_parallel_size=${INFER_EP} \
++policy.generation.mcore_generation_config.moe_router_dtype=fp32 \
policy.generation.mcore_generation_config.transformer_impl=${MEGATRON_TRANSFORMER_IMPL} \
policy.generation.mcore_generation_config.enable_chunked_prefill=${MEGATRON_ENABLE_CHUNKED_PREFILL} \
policy.generation.mcore_generation_config.enable_prefix_caching=${ENABLE_PREFIX_CACHING} \
++policy.generation.mcore_generation_config.vision_embedding_cache_max_bytes=${VISION_EMBEDDING_CACHE_MAX_BYTES} \
policy.generation.mcore_generation_config.cuda_graph_impl=${MEGATRON_CUDA_GRAPH_IMPL} \
policy.generation.mcore_generation_config.inference_cuda_graph_scope=${MEGATRON_CUDA_GRAPH_SCOPE} \
policy.generation.mcore_generation_config.num_cuda_graphs=${MEGATRON_NUM_CUDA_GRAPHS} \
policy.generation.mcore_generation_config.use_cuda_graphs_for_non_decode_steps=${MEGATRON_USE_CUDA_GRAPHS_FOR_NON_DECODE} \
policy.generation.mcore_generation_config.moe_pad_experts_for_cuda_graph_inference=${MOE_PAD_EXPERTS_FOR_CG} \
policy.generation.mcore_generation_config.refit_backend=${REFIT_BACKEND} \
policy.generation.mcore_generation_config.buffer_size_gb=${BUFFER_SIZE_GB} \
++policy.generation.mcore_generation_config.mamba_inference_ssm_states_dtype=float32 \
++policy.generation.mcore_generation_config.mamba_inference_conv_states_dtype=float32 \
policy.generation.mcore_generation_config.max_model_len=${MAX_SEQUENCE_LENGTH} \
policy.generation.mcore_generation_config.max_tokens=${MAX_SEQUENCE_LENGTH} \
++data_plane.enabled=true \
++data_plane.impl=transfer_queue \
++data_plane.backend=simple \
++data_plane.claim_meta_poll_interval_s=0.5 \
++data_plane.simple.num_storage_units=${NUM_STORAGE_UNITS} \
grpo.val_period=${VAL_PERIOD} \
grpo.val_at_start=${VAL_AT_START} \
grpo.val_at_end=${VAL_AT_END} \
grpo.val_batch_size=${VAL_GBS} \
grpo.max_val_samples=${VAL_SIZE} \
checkpointing.enabled=${CHECKPOINTING_ENABLED} \
checkpointing.checkpoint_dir=${RESULTS_DIR} \
checkpointing.metric_name=null \
logger.wandb_enabled=${WANDB_ENABLED} \
logger.wandb.name=${WANDB_NAME}-\${NRL_SLURM_JOB_ID} \
logger.wandb.project=${WANDB_PROJ} \
+logger.wandb.entity=${WANDB_GROUP} \
${TASK_OVERRIDES} \
${EXTRA_OVERRIDES}"

echo "Submitting ${JOB_NAME}: ${NUM_NODES}x${GPUS_PER_NODE}"
echo "  split: ${NUM_TRAIN_NODES} train nodes / ${NUM_GEN_NODES} generation nodes"
echo "  train TP/EP/world: ${POLICY_TP}/${POLICY_EP}/${TRAIN_WORLD_SIZE}"
echo "  generation TP/EP/world/DP: ${INFER_TP}/${INFER_EP}/${INFERENCE_WORLD_SIZE}/${INFERENCE_DP_SIZE}"
echo "  prompts/generations/train_gbs: ${NUM_PROMPTS_PER_STEP}/${NUM_GENERATIONS_PER_PROMPT}/${TRAIN_GBS}"
echo "  async sampler/lookahead/inflight/buffer: in_order/${MAX_LOOKAHEAD_VERSIONS}/${MAX_INFLIGHT_PROMPTS}/${MAX_BUFFERED_ROLLOUTS}"
echo "  token capture: ${TOKEN_CAPTURE_ENABLED}"
echo "  sequence/new tokens: ${MAX_SEQUENCE_LENGTH}/${MAX_NEW_TOKENS}"
echo "  generation: colocated=${COLOCATED} async=${ASYNC_GRPO} refit=${REFIT_BACKEND}"
echo "  Megatron: transformer=${MEGATRON_TRANSFORMER_IMPL} chunked_prefill=${MEGATRON_ENABLE_CHUNKED_PREFILL} prefix_caching=${ENABLE_PREFIX_CACHING}"
echo "  CUDA graphs: impl=${MEGATRON_CUDA_GRAPH_IMPL} scope=${MEGATRON_CUDA_GRAPH_SCOPE} count=${MEGATRON_NUM_CUDA_GRAPHS} non_decode=${MEGATRON_USE_CUDA_GRAPHS_FOR_NON_DECODE} moe_padding=${MOE_PAD_EXPERTS_FOR_CG}"
echo "  optimizer: cpu_offload=${OPTIMIZER_CPU_OFFLOAD} offload_fraction=${OPTIMIZER_OFFLOAD_FRACTION} logprob_offload=${OFFLOAD_OPTIMIZER_FOR_LOGPROB}"
echo "  config: ${CONFIG}"
if [[ "${TASK}" == "vstat" ]]; then
  echo "  VSTAT: repo=${HF_DATASET} rows=${NUM_DATA_ROWS} prepare=${PREPARE_VSTAT}"
  echo "  datasets: train=${NEMO_RL_VIDEO_TRAIN_JSONL} val=${NEMO_RL_VIDEO_VAL_JSONL} media=${NEMO_RL_VIDEO_MEDIA_ROOT}"
fi
echo "  NSYS: enabled=${ENABLE_NSYS}${NRL_NSYS_PROFILE_STEP_RANGE:+ step_range=${NRL_NSYS_PROFILE_STEP_RANGE}}"
echo "  W&B: ${WANDB_GROUP}/${WANDB_PROJ}/${WANDB_NAME}-<slurm-job-id> (enabled=${WANDB_ENABLED})"

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
  --segment="${NUM_NODES}"
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

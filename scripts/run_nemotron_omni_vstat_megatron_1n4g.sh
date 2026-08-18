#!/usr/bin/env bash
set -euo pipefail

# One-node / four-GPU Nemotron Omni video-GRPO smoke test.
# VSTAT is recommended here because Hugging Face hosts both the MCQ annotations
# and the actual MP4 assets; Video-MME commonly requires a separate video fetch.
#
# ASYNC_GRPO=false selects synchronous GRPO.
# MEGATRON_CUDA_GRAPH_IMPL=local enables CUDA graphs; block scope and MoE
# expert padding are selected automatically unless explicitly overridden.

NEMORL="${NEMORL:-/opt/nemo-rl}"
MODEL_NAME="${MODEL_NAME:-nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-${NEMORL}/workspace}"
DATA_ROOT="${DATA_ROOT:-${WORKSPACE_ROOT}/datasets/vstat-smoke}"
HF_DATASET="${HF_DATASET:-ShushengYang/VSTAT}"
NUM_DATA_ROWS="${NUM_DATA_ROWS:-8}"
PREPARE_VSTAT="${PREPARE_VSTAT:-false}"
NUM_FRAMES="${NUM_FRAMES:-8}"
TEMPORAL_PATCH_SIZE="${TEMPORAL_PATCH_SIZE:-2}"
VIDEO_TARGET_PATCHES="${VIDEO_TARGET_PATCHES:-256}"
MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-4096}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
MIN_GENERATION_TOKENS="${MIN_GENERATION_TOKENS:-2000}"
VISION_EMBEDDING_CACHE_MAX_BYTES="${VISION_EMBEDDING_CACHE_MAX_BYTES:-536870912}"
ENABLE_THINKING="${ENABLE_THINKING:-true}"
MAX_STEPS="${MAX_STEPS:-4}"
NUM_PROMPTS_PER_STEP="${NUM_PROMPTS_PER_STEP:-2}"
NUM_GENERATIONS_PER_PROMPT="${NUM_GENERATIONS_PER_PROMPT:-4}"
TRAIN_GBS="${TRAIN_GBS:-$((NUM_PROMPTS_PER_STEP * NUM_GENERATIONS_PER_PROMPT))}"
GEN_GPUS="${GEN_GPUS:-2}"
INFER_EP="${INFER_EP:-${GEN_GPUS}}"
REFIT_BACKEND="${REFIT_BACKEND:-nccl}"
ASYNC_GRPO="${ASYNC_GRPO:-true}"
MAX_TRAJECTORY_AGE_STEPS="${MAX_TRAJECTORY_AGE_STEPS:-2}"
IN_FLIGHT_WEIGHT_UPDATES="${IN_FLIGHT_WEIGHT_UPDATES:-true}"
MEGATRON_CUDA_GRAPH_IMPL="${MEGATRON_CUDA_GRAPH_IMPL:-local}"
if [[ "${MEGATRON_CUDA_GRAPH_IMPL}" == "none" ]]; then
  DEFAULT_CUDA_GRAPH_SCOPE=none
else
  DEFAULT_CUDA_GRAPH_SCOPE=block
fi
if [[ "${MEGATRON_CUDA_GRAPH_IMPL}" != "none" && "${INFER_EP}" -gt 1 ]]; then
  DEFAULT_MOE_PAD_EXPERTS_FOR_CG=true
else
  DEFAULT_MOE_PAD_EXPERTS_FOR_CG=false
fi
MEGATRON_CUDA_GRAPH_SCOPE="${MEGATRON_CUDA_GRAPH_SCOPE:-${DEFAULT_CUDA_GRAPH_SCOPE}}"
MEGATRON_NUM_CUDA_GRAPHS="${MEGATRON_NUM_CUDA_GRAPHS:--1}"
MEGATRON_USE_CUDA_GRAPHS_FOR_NON_DECODE="${MEGATRON_USE_CUDA_GRAPHS_FOR_NON_DECODE:-false}"
MOE_PAD_EXPERTS_FOR_CG="${MOE_PAD_EXPERTS_FOR_CG:-${DEFAULT_MOE_PAD_EXPERTS_FOR_CG}}"
MEGATRON_ASYNC_SCHED_MODE="${MEGATRON_ASYNC_SCHED_MODE:-async}"
# Lower-precision Adam moments for the 2-GPU train half (HBM-tight on 1n4g).
# Override with EXP_AVG_DTYPE=float32 EXP_AVG_SQ_DTYPE=float32 STORE_PARAM_REMAINDERS=false
# for full-precision optimizer state.
EXP_AVG_DTYPE="${EXP_AVG_DTYPE:-bfloat16}"
EXP_AVG_SQ_DTYPE="${EXP_AVG_SQ_DTYPE:-bfloat16}"
STORE_PARAM_REMAINDERS="${STORE_PARAM_REMAINDERS:-true}"

cd "${NEMORL}"

GPUS_PER_NODE="${GPUS_PER_NODE:-$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)}"
if (( GPUS_PER_NODE != 4 )); then
  echo "This smoke launcher expects exactly four visible GPUs (got ${GPUS_PER_NODE})." >&2
  exit 1
fi
if (( GEN_GPUS <= 0 || GEN_GPUS >= GPUS_PER_NODE )); then
  echo "GEN_GPUS must leave at least one GPU for training." >&2
  exit 1
fi
if (( INFER_EP <= 0 || GEN_GPUS % INFER_EP != 0 )); then
  echo "INFER_EP must be a positive divisor of GEN_GPUS (got ${INFER_EP})." >&2
  exit 1
fi
if [[ "${ASYNC_GRPO}" != "true" && "${ASYNC_GRPO}" != "false" ]]; then
  echo "ASYNC_GRPO must be true or false (got ${ASYNC_GRPO})." >&2
  exit 1
fi
if [[ "${MEGATRON_CUDA_GRAPH_IMPL}" != "none" &&
      "${MEGATRON_CUDA_GRAPH_SCOPE}" == "none" ]]; then
  echo "CUDA graphs require a non-none MEGATRON_CUDA_GRAPH_SCOPE." >&2
  exit 1
fi
if [[ "${MEGATRON_CUDA_GRAPH_IMPL}" != "none" && "${INFER_EP}" -gt 1 &&
      "${MOE_PAD_EXPERTS_FOR_CG}" != "true" ]]; then
  echo "CUDA graphs with expert parallelism require MOE_PAD_EXPERTS_FOR_CG=true." >&2
  exit 1
fi
TRAIN_GPUS=$((GPUS_PER_NODE - GEN_GPUS))

CACHE_ROOT="${CACHE_ROOT:-${WORKSPACE_ROOT}/cache/nemo-rl-omni}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export NRL_MEGATRON_CHECKPOINT_DIR="${NRL_MEGATRON_CHECKPOINT_DIR:-${CACHE_ROOT}/megatron-checkpoints}"
export NEMO_RL_VENV_DIR="${NEMO_RL_VENV_DIR:-/opt/ray_venvs}"
export NEMO_GYM_VENV_DIR="${NEMO_GYM_VENV_DIR:-/opt/ray_venvs}"
export NEMO_GYM_EXTRA_ROOTS="${NEMO_GYM_EXTRA_ROOTS:-${NEMORL}/3rdparty/Gym-workspace/Gym}"
export NRL_VIDEO_BACKEND="${NRL_VIDEO_BACKEND:-torchcodec}"
export NRL_VIDEO_SAMPLING_STYLE="${NRL_VIDEO_SAMPLING_STYLE:-nemotron_vl}"
export NRL_VIDEO_TEMPORAL_PATCH_SIZE="${TEMPORAL_PATCH_SIZE}"
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export NRL_VENVS_TRUST_EXISTING="${NRL_VENVS_TRUST_EXISTING:-1}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export FLASHINFER_DISABLE_VERSION_CHECK="${FLASHINFER_DISABLE_VERSION_CHECK:-1}"

AUDIO_DEPS_SCRIPT="${NEMORL}/tools/install_audio_deps.sh"
MEGATRON_WORKER_PYTHON="${RAY_MEGATRON_PYTHON:-${NEMO_RL_VENV_DIR}/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python}"
NEED_AUDIO_VIDEO_DEPS=false
if ! python -c "import torchcodec" >/dev/null 2>&1; then
  NEED_AUDIO_VIDEO_DEPS=true
fi
if [[ ! -x "${MEGATRON_WORKER_PYTHON}" ]] ||
   ! "${MEGATRON_WORKER_PYTHON}" -c "import av" >/dev/null 2>&1; then
  NEED_AUDIO_VIDEO_DEPS=true
fi
if [[ "${NEED_AUDIO_VIDEO_DEPS}" == "true" ]]; then
  if [[ ! -f "${AUDIO_DEPS_SCRIPT}" ]]; then
    echo "Audio/video dependency installer is missing: ${AUDIO_DEPS_SCRIPT}" >&2
    exit 1
  fi
  echo "Installing missing audio/video dependencies"
  RAY_MEGATRON_PYTHON="${MEGATRON_WORKER_PYTHON}" bash "${AUDIO_DEPS_SCRIPT}"
fi

# NSYS: ENABLE_NSYS=true NRL_NSYS_PROFILE_STEP_RANGE=1:4
ENABLE_NSYS="${ENABLE_NSYS:-false}"
NSYS_ENV=()
if [[ "${ENABLE_NSYS}" == "true" ]]; then
  NRL_NSYS_WORKER_PATTERNS="${NRL_NSYS_WORKER_PATTERNS:-*policy*,*megatron*}"
  NRL_NSYS_PROFILE_STEP_RANGE="${NRL_NSYS_PROFILE_STEP_RANGE:-1:4}"
  LD_LIBRARY_PATH="/usr/local/cuda/targets/aarch64-linux/lib:/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib:/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/lib/aarch64-linux-gnu:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
  NRL_NSYS_EXTRA_OPTIONS="${NRL_NSYS_EXTRA_OPTIONS:-{\"o\":\"${WORKSPACE_ROOT}/nsys/%p\",\"cpuctxsw\":\"none\",\"force-overwrite\":\"true\"}}"
  NSYS_ENV=(
    "NRL_NSYS_WORKER_PATTERNS=${NRL_NSYS_WORKER_PATTERNS}"
    "NRL_NSYS_PROFILE_STEP_RANGE=${NRL_NSYS_PROFILE_STEP_RANGE}"
    "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
    "NRL_NSYS_EXTRA_OPTIONS=${NRL_NSYS_EXTRA_OPTIONS}"
  )
  mkdir -p "${WORKSPACE_ROOT}/nsys"
fi

BRIDGE="${NEMORL}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge"
export PYTHONPATH="${NEMORL}:${NEMO_GYM_EXTRA_ROOTS}:${BRIDGE}/src:${BRIDGE}/3rdparty/Megatron-LM${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${DATA_ROOT}" "${HF_HOME}" "${NRL_MEGATRON_CHECKPOINT_DIR}"
export NEMO_RL_VIDEO_TRAIN_JSONL="${DATA_ROOT}/train-gym.jsonl"
export NEMO_RL_VIDEO_VAL_JSONL="${DATA_ROOT}/val-gym.jsonl"
export NEMO_RL_VIDEO_MEDIA_ROOT="${DATA_ROOT}/media"

VSTAT_PREPARE_SCRIPT="${NEMORL}/scripts/prepare_nemotron_omni_vstat.py"
if [[ ! -f "${VSTAT_PREPARE_SCRIPT}" ]]; then
  echo "VSTAT preparation script is missing: ${VSTAT_PREPARE_SCRIPT}" >&2
  exit 1
fi
if [[ "${PREPARE_VSTAT}" == "true" ||
      ! -s "${NEMO_RL_VIDEO_TRAIN_JSONL}" ||
      ! -s "${NEMO_RL_VIDEO_VAL_JSONL}" ]]; then
  echo "Preparing VSTAT under ${DATA_ROOT}"
  uv run --no-sync python "${VSTAT_PREPARE_SCRIPT}" \
    --output-dir "${DATA_ROOT}" \
    --repo-id "${HF_DATASET}" \
    --num-rows "${NUM_DATA_ROWS}"
fi

RESULTS_DIR="${RESULTS_DIR:-${WORKSPACE_ROOT}/results/nemo-rl-omni/nemotron-omni-vstat-megatron-1n4g}"
mkdir -p "${RESULTS_DIR}"

echo "VSTAT: root=${DATA_ROOT} repo=${HF_DATASET} rows=${NUM_DATA_ROWS} prepare=${PREPARE_VSTAT}"
echo "  async_grpo=${ASYNC_GRPO} scheduler=${MEGATRON_ASYNC_SCHED_MODE}"
echo "  generation: TP=${GEN_GPUS} EP=${INFER_EP}"
echo "  cuda_graph_impl=${MEGATRON_CUDA_GRAPH_IMPL} scope=${MEGATRON_CUDA_GRAPH_SCOPE} non_decode=${MEGATRON_USE_CUDA_GRAPHS_FOR_NON_DECODE}"
exec env "${NSYS_ENV[@]}" uv run --no-sync python examples/nemo_gym/run_grpo_nemo_gym.py \
  --config examples/configs/recipes/vlm/vlm_grpo-nemotron-omni-30ba3b-16n8g-megatron-tp4ep4-async-gym-video.v1.yaml \
  policy.model_name="${MODEL_NAME}" \
  policy.tokenizer.name="${MODEL_NAME}" \
  policy.tokenizer.chat_template_kwargs.enable_thinking="${ENABLE_THINKING}" \
  policy.is_vlm=true \
  policy.generation.backend=megatron \
  ++policy.generation.bad_words=null \
  policy.generation.colocated.enabled=false \
  policy.generation.colocated.resources.num_nodes=1 \
  policy.generation.colocated.resources.gpus_per_node="${GEN_GPUS}" \
  policy.megatron_cfg.tensor_model_parallel_size="${TRAIN_GPUS}" \
  policy.megatron_cfg.expert_model_parallel_size="${TRAIN_GPUS}" \
  policy.megatron_cfg.expert_tensor_parallel_size=1 \
  policy.megatron_cfg.context_parallel_size=1 \
  policy.megatron_cfg.sequence_parallel=true \
  policy.megatron_cfg.bias_activation_fusion=false \
  policy.megatron_cfg.optimizer.optimizer_cpu_offload=false \
  policy.megatron_cfg.optimizer.optimizer_offload_fraction=0.0 \
  ++policy.megatron_cfg.optimizer.exp_avg_dtype="${EXP_AVG_DTYPE}" \
  ++policy.megatron_cfg.optimizer.exp_avg_sq_dtype="${EXP_AVG_SQ_DTYPE}" \
  ++policy.megatron_cfg.optimizer.store_param_remainders="${STORE_PARAM_REMAINDERS}" \
  policy.generation.mcore_generation_config.tensor_model_parallel_size="${GEN_GPUS}" \
  policy.generation.mcore_generation_config.expert_model_parallel_size="${INFER_EP}" \
  policy.generation.mcore_generation_config.expert_tensor_parallel_size=1 \
  ++policy.generation.mcore_generation_config.context_parallel_size=1 \
  policy.generation.mcore_generation_config.transformer_impl=transformer_engine \
  policy.generation.mcore_generation_config.sequence_parallel=true \
  policy.generation.mcore_generation_config.refit_backend="${REFIT_BACKEND}" \
  policy.generation.mcore_generation_config.buffer_size_gb=8 \
  policy.generation.mcore_generation_config.cuda_graph_impl="${MEGATRON_CUDA_GRAPH_IMPL}" \
  policy.generation.mcore_generation_config.inference_cuda_graph_scope="${MEGATRON_CUDA_GRAPH_SCOPE}" \
  policy.generation.mcore_generation_config.num_cuda_graphs="${MEGATRON_NUM_CUDA_GRAPHS}" \
  ++policy.generation.mcore_generation_config.moe_pad_experts_for_cuda_graph_inference="${MOE_PAD_EXPERTS_FOR_CG}" \
  policy.generation.mcore_generation_config.use_cuda_graphs_for_non_decode_steps="${MEGATRON_USE_CUDA_GRAPHS_FOR_NON_DECODE}" \
  policy.generation.mcore_generation_config.max_model_len="${MAX_SEQUENCE_LENGTH}" \
  policy.generation.mcore_generation_config.max_tokens="${MAX_SEQUENCE_LENGTH}" \
  policy.generation.mcore_generation_config.enable_chunked_prefill=true \
  ++policy.generation.mcore_generation_config.async_sched_mode="${MEGATRON_ASYNC_SCHED_MODE}" \
  policy.generation.mcore_generation_config.enable_prefix_caching=false \
  ++policy.generation.mcore_generation_config.vision_embedding_cache_max_bytes="${VISION_EMBEDDING_CACHE_MAX_BYTES}" \
  policy.generation.mcore_generation_config.parsers=[nemotron-v3-reasoning,qwen3-coder-tool] \
  ++policy.generation.mcore_generation_config.video_num_frames="${NUM_FRAMES}" \
  ++policy.generation.mcore_generation_config.video_temporal_patch_size="${TEMPORAL_PATCH_SIZE}" \
  ++policy.generation.mcore_generation_config.video_target_num_patches="${VIDEO_TARGET_PATCHES}" \
  policy.max_total_sequence_length="${MAX_SEQUENCE_LENGTH}" \
  policy.generation.max_new_tokens="${MAX_NEW_TOKENS}" \
  +data.default.num_frames="${NUM_FRAMES}" \
  +data.default.video_sampling_style=nemotron_vl \
  +data.default.video_temporal_patch_size="${TEMPORAL_PATCH_SIZE}" \
  +data.default.min_generation_tokens="${MIN_GENERATION_TOKENS}" \
  data.default.video_target_num_patches="${VIDEO_TARGET_PATCHES}" \
  data.train.data_path="${NEMO_RL_VIDEO_TRAIN_JSONL}" \
  data.validation.data_path="${NEMO_RL_VIDEO_VAL_JSONL}" \
  ++env.nemo_gym.policy_model.responses_api_models.vllm_model.chat_template_kwargs.enable_thinking="${ENABLE_THINKING}" \
  grpo.deduplicate_multimodal_data=false \
  grpo.async_grpo.enabled="${ASYNC_GRPO}" \
  grpo.async_grpo.max_trajectory_age_steps="${MAX_TRAJECTORY_AGE_STEPS}" \
  grpo.async_grpo.in_flight_weight_updates="${IN_FLIGHT_WEIGHT_UPDATES}" \
  loss_fn.use_importance_sampling_correction=true \
  grpo.num_prompts_per_step="${NUM_PROMPTS_PER_STEP}" \
  grpo.num_generations_per_prompt="${NUM_GENERATIONS_PER_PROMPT}" \
  policy.train_global_batch_size="${TRAIN_GBS}" \
  grpo.max_num_steps="${MAX_STEPS}" \
  grpo.val_period=0 \
  grpo.val_at_start=false \
  grpo.val_at_end=false \
  cluster.num_nodes=1 \
  cluster.gpus_per_node="${GPUS_PER_NODE}" \
  checkpointing.enabled=false \
  logger.log_dir="${RESULTS_DIR}" \
  logger.wandb_enabled=false \
  "$@"

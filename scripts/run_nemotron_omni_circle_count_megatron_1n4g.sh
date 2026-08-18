#!/usr/bin/env bash
set -euo pipefail

# 1-node GB200 / 4-GPU smoke for Nemotron Omni on NeMo-Gym's circle_count env.
# circle_count is an IMAGE environment: every row carries one synthetic PNG as a
# base64 data URL plus a "how many <color> circles" question, and the Gym
# resources server rewards an exact match on \boxed{count}.
#
# Unlike the CLEVR launcher this runs the Gym entrypoint
# (examples/nemo_gym/run_grpo_nemo_gym.py): Gym owns the prompts, the reward and
# the generation loop, talking to the policy over the engine's HTTP server.
#
# GENERATION_BACKEND=megatron|vllm (default megatron)
# COLOCATED=true|false (default false) shares every GPU between the trainer and
#   the engine instead of splitting them. Colocated async GRPO is only
#   supported for Megatron generation.
# ASYNC_GRPO=true|false (default true) selects async or sync GRPO.
# GENERATE_ROWS=N regenerates a larger dataset instead of using the 5-row
#   example.jsonl that ships with Gym.
NEMORL="${NEMORL:-/opt/nemo-rl}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-${NEMORL}/workspace}"
GYM="${GYM:-${NEMORL}/3rdparty/Gym-workspace/Gym}"
MODEL_NAME="${MODEL_NAME:-nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16}"
CONFIG="${CONFIG:-examples/configs/recipes/vlm/vlm_grpo-nemotron-omni-30ba3b-circle-count-1n4g-megatron_generation.v1.yaml}"
GENERATION_BACKEND="${GENERATION_BACKEND:-megatron}"
COLOCATED="${COLOCATED:-false}"
ASYNC_GRPO="${ASYNC_GRPO:-true}"

cd "${NEMORL}"

if [[ "${GENERATION_BACKEND}" != "megatron" && "${GENERATION_BACKEND}" != "vllm" ]]; then
  echo "GENERATION_BACKEND must be megatron or vllm (got ${GENERATION_BACKEND})." >&2
  exit 1
fi
if [[ "${COLOCATED}" != "true" && "${COLOCATED}" != "false" ]]; then
  echo "COLOCATED must be true or false (got ${COLOCATED})." >&2
  exit 1
fi
if [[ "${ASYNC_GRPO}" != "true" && "${ASYNC_GRPO}" != "false" ]]; then
  echo "ASYNC_GRPO must be true or false (got ${ASYNC_GRPO})." >&2
  exit 1
fi
if [[ "${COLOCATED}" == "true" && "${ASYNC_GRPO}" == "true" && "${GENERATION_BACKEND}" != "megatron" ]]; then
  echo "Colocated async GRPO requires GENERATION_BACKEND=megatron." >&2
  exit 1
fi
if [[ ! -d "${GYM}" ]]; then
  echo "NeMo-Gym checkout is missing: ${GYM}" >&2
  exit 1
fi

GPUS_PER_NODE="${GPUS_PER_NODE:-$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)}"
if (( GPUS_PER_NODE < 4 )); then
  echo "This launcher requires at least four visible GPUs (got ${GPUS_PER_NODE})." >&2
  exit 1
fi

if [[ "${COLOCATED}" == "true" ]]; then
  # The engine shares the trainer's GPUs and its HTTP server comes up on the
  # policy workers, so both worlds span every GPU.
  TRAIN_WORLD_SIZE="${GPUS_PER_NODE}"
  INFERENCE_WORLD_SIZE="${GPUS_PER_NODE}"
  GEN_GPUS_PER_NODE="${GPUS_PER_NODE}"
  NUM_GEN_NODES=1
  COLOCATED_ENABLED=true
else
  GEN_GPUS_PER_NODE="${GEN_GPUS_PER_NODE:-$((GPUS_PER_NODE / 2))}"
  if (( GEN_GPUS_PER_NODE <= 0 || GEN_GPUS_PER_NODE >= GPUS_PER_NODE )); then
    echo "Non-colocated mode requires a strict train/inference GPU split." >&2
    exit 1
  fi
  TRAIN_WORLD_SIZE=$((GPUS_PER_NODE - GEN_GPUS_PER_NODE))
  INFERENCE_WORLD_SIZE="${GEN_GPUS_PER_NODE}"
  NUM_GEN_NODES=1
  COLOCATED_ENABLED=false
fi

POLICY_TP="${POLICY_TP:-${TRAIN_WORLD_SIZE}}"
INFER_TP="${INFER_TP:-${INFERENCE_WORLD_SIZE}}"
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
  if [[ "${INFER_EP}" != "${INFER_TP}" ]]; then
    echo "Forcing INFER_EP=${INFER_TP} for vLLM (was ${INFER_EP}) for async-safe Ray DP." >&2
  fi
  INFER_EP="${INFER_TP}"
elif (( INFERENCE_WORLD_SIZE % INFER_EP != 0 )); then
  echo "Inference world size must be divisible by INFER_EP (ETP=1)." >&2
  exit 1
fi
if [[ "${COLOCATED_ENABLED}" == "true" ]]; then
  # Colocated Megatron generation reuses the trained model in place, so the
  # engine cannot reshard: its parallel layout has to match the policy's.
  if (( INFER_TP != POLICY_TP || INFER_EP != POLICY_EP )); then
    echo "Colocated mode requires INFER_TP/INFER_EP to match POLICY_TP/POLICY_EP" \
      "(got TP ${INFER_TP} vs ${POLICY_TP}, EP ${INFER_EP} vs ${POLICY_EP})." >&2
    exit 1
  fi
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

BRIDGE="${NEMORL}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge"
export PYTHONPATH="${NEMORL}:${GYM}:${BRIDGE}/src:${BRIDGE}/3rdparty/Megatron-LM${PYTHONPATH:+:${PYTHONPATH}}"
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export NRL_VENVS_TRUST_EXISTING="${NRL_VENVS_TRUST_EXISTING:-1}"
export NRL_FORCE_REBUILD_VENVS="${NRL_FORCE_REBUILD_VENVS:-false}"
export NEMO_RL_VENV_DIR="${NEMO_RL_VENV_DIR:-/opt/ray_venvs}"
# The Gym env servers run in their own venvs; keep them beside the NeMo-RL ones
# so a warm container does not rebuild them on every launch.
export NEMO_GYM_VENV_DIR="${NEMO_GYM_VENV_DIR:-${NEMO_RL_VENV_DIR}}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export NVTE_FWD_LAYERNORM_SM_MARGIN="${NVTE_FWD_LAYERNORM_SM_MARGIN:-16}"
export NVTE_BWD_LAYERNORM_SM_MARGIN="${NVTE_BWD_LAYERNORM_SM_MARGIN:-16}"
export FLASHINFER_DISABLE_VERSION_CHECK="${FLASHINFER_DISABLE_VERSION_CHECK:-1}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-10.0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

MAX_STEPS="${MAX_STEPS:-5}"
MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-4096}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
NUM_PROMPTS="${NUM_PROMPTS:-2}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
TRAIN_GBS="${TRAIN_GBS:-$((NUM_PROMPTS * NUM_GENERATIONS))}"
EXPECTED_TRAIN_GBS=$((NUM_PROMPTS * NUM_GENERATIONS))
if (( TRAIN_GBS != EXPECTED_TRAIN_GBS )); then
  echo "TRAIN_GBS (${TRAIN_GBS}) must equal NUM_PROMPTS * NUM_GENERATIONS (${EXPECTED_TRAIN_GBS})." >&2
  exit 1
fi
REFIT_BACKEND="${REFIT_BACKEND:-nccl}"
JOB_NAME="${JOB_NAME:-nemotron-omni-circle-count-${GENERATION_BACKEND}-1n4g}"
EXP_NAME="${EXP_NAME:-${JOB_NAME}}"
PRECISION_RECIPE="${PRECISION_RECIPE:-bf16}"
WANDB_ENABLED="${WANDB_ENABLED:-false}"
WANDB_PROJ="${WANDB_PROJ:-mllm-rl-dev}"
WANDB_GROUP="${WANDB_GROUP:-adlr}"
WANDB_NAME="${WANDB_NAME:-${EXP_NAME}-${PRECISION_RECIPE}-internal-repo}"
RESULTS_DIR="${RESULTS_DIR:-${WORKSPACE_ROOT}/results/nemo-rl-omni/${JOB_NAME}}"
CHECKPOINTING_ENABLED="${CHECKPOINTING_ENABLED:-false}"
# Host OOM on GB200 when optimizer CPU offload is enabled for this model size.
OPTIMIZER_CPU_OFFLOAD="${OPTIMIZER_CPU_OFFLOAD:-false}"
OFFLOAD_OPTIMIZER_FOR_LOGPROB="${OFFLOAD_OPTIMIZER_FOR_LOGPROB:-false}"
if [[ "${OPTIMIZER_CPU_OFFLOAD}" == "true" ]]; then
  OPTIMIZER_OFFLOAD_FRACTION="${OPTIMIZER_OFFLOAD_FRACTION:-1.0}"
else
  OPTIMIZER_OFFLOAD_FRACTION="${OPTIMIZER_OFFLOAD_FRACTION:-0.0}"
fi
# Lower-precision Adam moments for the 2-GPU train half (HBM-tight on 1n4g).
EXP_AVG_DTYPE="${EXP_AVG_DTYPE:-bfloat16}"
EXP_AVG_SQ_DTYPE="${EXP_AVG_SQ_DTYPE:-bfloat16}"
STORE_PARAM_REMAINDERS="${STORE_PARAM_REMAINDERS:-true}"
BUFFER_SIZE_GB="${BUFFER_SIZE_GB:-8}"
MAX_TRAJECTORY_AGE_STEPS="${MAX_TRAJECTORY_AGE_STEPS:-2}"
IN_FLIGHT_WEIGHT_UPDATES="${IN_FLIGHT_WEIGHT_UPDATES:-true}"
# Keep CUDA graphs disabled and use chunked prefill by default.
MEGATRON_ENABLE_CHUNKED_PREFILL="${MEGATRON_ENABLE_CHUNKED_PREFILL:-true}"
MEGATRON_CUDA_GRAPH_IMPL="${MEGATRON_CUDA_GRAPH_IMPL:-local}" # local none
MEGATRON_CUDA_GRAPH_SCOPE="${MEGATRON_CUDA_GRAPH_SCOPE:-block}" # block layer none
MEGATRON_NUM_CUDA_GRAPHS="${MEGATRON_NUM_CUDA_GRAPHS:--1}"
MEGATRON_USE_CUDA_GRAPHS_FOR_NON_DECODE="${MEGATRON_USE_CUDA_GRAPHS_FOR_NON_DECODE:-false}"
MOE_PAD_EXPERTS_FOR_CG="${MOE_PAD_EXPERTS_FOR_CG:-true}"

# Dataset. Gym ships a 5-row example; GENERATE_ROWS=N synthesizes a larger set
# (N train rows plus a disjoint 25%-sized validation split) under DATA_ROOT.
DATA_ROOT="${DATA_ROOT:-${WORKSPACE_ROOT}/datasets/circle-count-smoke}"
GENERATE_ROWS="${GENERATE_ROWS:-64}"
GYM_CIRCLE_COUNT="${GYM}/resources_servers/circle_count"
TRAIN_JSONL="${TRAIN_JSONL:-${GYM_CIRCLE_COUNT}/data/example.jsonl}"
VAL_JSONL="${VAL_JSONL:-${TRAIN_JSONL}}"
if (( GENERATE_ROWS > 0 )); then
  VAL_ROWS=$(( GENERATE_ROWS / 4 ))
  (( VAL_ROWS > 0 )) || VAL_ROWS=1
  mkdir -p "${DATA_ROOT}"
  TRAIN_JSONL="${DATA_ROOT}/train.jsonl"
  VAL_JSONL="${DATA_ROOT}/val.jsonl"
  echo "Generating ${GENERATE_ROWS} train / ${VAL_ROWS} val circle_count rows under ${DATA_ROOT}"
  uv run --no-sync python "${GYM_CIRCLE_COUNT}/generate_data.py" \
    --n "${GENERATE_ROWS}" --out "${TRAIN_JSONL}" --seed-offset 0
  # Offset the seeds so validation scenes are disjoint from training ones.
  uv run --no-sync python "${GYM_CIRCLE_COUNT}/generate_data.py" \
    --n "${VAL_ROWS}" --out "${VAL_JSONL}" --seed-offset "${GENERATE_ROWS}"
  # generate_data.py omits the agent_ref that NeMo-RL reads off every row to
  # route it to a Gym agent, unlike the example.jsonl it ships beside. The name
  # is the agent key in resources_servers/circle_count/configs/circle_count.yaml.
  uv run --no-sync python scripts/add_nemo_gym_agent_ref.py \
    --agent-name circle_count_simple_agent \
    "${TRAIN_JSONL}" "${VAL_JSONL}"
fi

# NSYS: ENABLE_NSYS=true NRL_NSYS_PROFILE_STEP_RANGE=1:4
ENABLE_NSYS="${ENABLE_NSYS:-false}"
NSYS_ENV=()
if [[ "${ENABLE_NSYS}" == "true" ]]; then
  NRL_NSYS_WORKER_PATTERNS="${NRL_NSYS_WORKER_PATTERNS:-*policy*,*megatron*}"
  NRL_NSYS_PROFILE_STEP_RANGE="${NRL_NSYS_PROFILE_STEP_RANGE:-1:4}"
  LD_LIBRARY_PATH="/usr/local/cuda/targets/aarch64-linux/lib:/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib:/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/lib/aarch64-linux-gnu:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
  NRL_NSYS_EXTRA_OPTIONS="${NRL_NSYS_EXTRA_OPTIONS:-{\"o\":\"/opt/nemo-rl/workspace/nsys/%p\",\"cpuctxsw\":\"none\",\"force-overwrite\":\"true\"}}"
  NSYS_ENV=(
    "NRL_NSYS_WORKER_PATTERNS=${NRL_NSYS_WORKER_PATTERNS}"
    "NRL_NSYS_PROFILE_STEP_RANGE=${NRL_NSYS_PROFILE_STEP_RANGE}"
    "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
    "NRL_NSYS_EXTRA_OPTIONS=${NRL_NSYS_EXTRA_OPTIONS}"
  )
  mkdir -p /opt/nemo-rl/workspace/nsys
fi

mkdir -p \
  "${HF_HUB_CACHE}" \
  "${HF_DATASETS_CACHE}" \
  "${HF_MODULES_CACHE}" \
  "${TRANSFORMERS_CACHE}" \
  "${NRL_MEGATRON_CHECKPOINT_DIR}" \
  "${XDG_CACHE_HOME}" \
  "${TORCH_HOME}" \
  "${TRITON_CACHE_DIR}" \
  "${RESULTS_DIR}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "Config is missing under ${NEMORL}: ${CONFIG}" >&2
  exit 1
fi
for jsonl in "${TRAIN_JSONL}" "${VAL_JSONL}"; do
  if [[ ! -s "${jsonl}" ]]; then
    echo "Gym dataset is missing or empty: ${jsonl}" >&2
    echo "Generate one with: GENERATE_ROWS=64 $0" >&2
    exit 1
  fi
done

COMMON_OVERRIDES=(
  cluster.num_nodes=1
  cluster.gpus_per_node="${GPUS_PER_NODE}"
  policy.model_name="${MODEL_NAME}"
  policy.tokenizer.name="${MODEL_NAME}"
  policy.is_vlm=true
  policy.megatron_cfg.tensor_model_parallel_size="${POLICY_TP}"
  policy.megatron_cfg.expert_model_parallel_size="${POLICY_EP}"
  policy.megatron_cfg.expert_tensor_parallel_size=1
  policy.megatron_cfg.context_parallel_size="${POLICY_CP}"
  policy.megatron_cfg.optimizer.optimizer_cpu_offload="${OPTIMIZER_CPU_OFFLOAD}"
  policy.megatron_cfg.optimizer.optimizer_offload_fraction="${OPTIMIZER_OFFLOAD_FRACTION}"
  policy.offload_optimizer_for_logprob="${OFFLOAD_OPTIMIZER_FOR_LOGPROB}"
)
if [[ -n "${EXP_AVG_DTYPE}" ]]; then
  COMMON_OVERRIDES+=("++policy.megatron_cfg.optimizer.exp_avg_dtype=${EXP_AVG_DTYPE}")
fi
if [[ -n "${EXP_AVG_SQ_DTYPE}" ]]; then
  COMMON_OVERRIDES+=("++policy.megatron_cfg.optimizer.exp_avg_sq_dtype=${EXP_AVG_SQ_DTYPE}")
fi
if [[ -n "${STORE_PARAM_REMAINDERS}" ]]; then
  COMMON_OVERRIDES+=("++policy.megatron_cfg.optimizer.store_param_remainders=${STORE_PARAM_REMAINDERS}")
fi
COMMON_OVERRIDES+=(
  policy.generation.backend="${GENERATION_BACKEND}"
  policy.generation.colocated.enabled="${COLOCATED_ENABLED}"
  policy.generation.colocated.resources.num_nodes="${NUM_GEN_NODES}"
  policy.generation.colocated.resources.gpus_per_node="${GEN_GPUS_PER_NODE}"
  policy.max_total_sequence_length="${MAX_SEQUENCE_LENGTH}"
  policy.generation.max_new_tokens="${MAX_NEW_TOKENS}"
  data.train.data_path="${TRAIN_JSONL}"
  data.validation.data_path="${VAL_JSONL}"
  grpo.async_grpo.enabled="${ASYNC_GRPO}"
  grpo.async_grpo.max_trajectory_age_steps="${MAX_TRAJECTORY_AGE_STEPS}"
  grpo.async_grpo.in_flight_weight_updates="${IN_FLIGHT_WEIGHT_UPDATES}"
  loss_fn.use_importance_sampling_correction=true
  grpo.num_prompts_per_step="${NUM_PROMPTS}"
  grpo.num_generations_per_prompt="${NUM_GENERATIONS}"
  grpo.val_period=0
  grpo.val_at_start=false
  grpo.val_at_end=false
  policy.train_global_batch_size="${TRAIN_GBS}"
  grpo.max_num_steps="${MAX_STEPS}"
  checkpointing.enabled="${CHECKPOINTING_ENABLED}"
  checkpointing.checkpoint_dir="${RESULTS_DIR}"
  logger.log_dir="${RESULTS_DIR}"
  logger.wandb_enabled="${WANDB_ENABLED}"
  logger.tensorboard_enabled=false
  logger.wandb.name="${WANDB_NAME}"
  logger.wandb.project="${WANDB_PROJ}"
  +logger.wandb.entity="${WANDB_GROUP}"
)

GEN_OVERRIDES=()
if [[ "${GENERATION_BACKEND}" == "megatron" ]]; then
  GEN_OVERRIDES=(
    policy.generation.mcore_generation_config.tensor_model_parallel_size="${INFER_TP}"
    policy.generation.mcore_generation_config.expert_model_parallel_size="${INFER_EP}"
    policy.generation.mcore_generation_config.expert_tensor_parallel_size=1
    ++policy.generation.mcore_generation_config.context_parallel_size="${POLICY_CP}"
    policy.generation.mcore_generation_config.enable_chunked_prefill="${MEGATRON_ENABLE_CHUNKED_PREFILL}"
    ++policy.generation.mcore_generation_config.async_sched_mode=async
    policy.generation.mcore_generation_config.cuda_graph_impl="${MEGATRON_CUDA_GRAPH_IMPL}"
    policy.generation.mcore_generation_config.inference_cuda_graph_scope="${MEGATRON_CUDA_GRAPH_SCOPE}"
    policy.generation.mcore_generation_config.num_cuda_graphs="${MEGATRON_NUM_CUDA_GRAPHS}"
    policy.generation.mcore_generation_config.use_cuda_graphs_for_non_decode_steps="${MEGATRON_USE_CUDA_GRAPHS_FOR_NON_DECODE}"
    policy.generation.mcore_generation_config.refit_backend="${REFIT_BACKEND}"
    policy.generation.mcore_generation_config.buffer_size_gb="${BUFFER_SIZE_GB}"
    policy.generation.mcore_generation_config.moe_pad_experts_for_cuda_graph_inference="${MOE_PAD_EXPERTS_FOR_CG}"
    policy.generation.mcore_generation_config.max_model_len="${MAX_SEQUENCE_LENGTH}"
    policy.generation.mcore_generation_config.max_tokens="${MAX_SEQUENCE_LENGTH}"
  )
else
  # Refit packs weights into chunks sized at NRL_REFIT_BUFFER_MEMORY_RATIO *
  # total HBM; exported globally on purpose because producer and consumer must
  # agree on the chunk boundaries.
  export NRL_REFIT_BUFFER_MEMORY_RATIO="${NRL_REFIT_BUFFER_MEMORY_RATIO:-0.005}"
  VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.5}"
  VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-true}"
  VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-${MAX_SEQUENCE_LENGTH}}"
  GEN_OVERRIDES=(
    policy.generation.vllm_cfg.async_engine="${ASYNC_GRPO}"
    policy.generation.vllm_cfg.skip_tokenizer_init=false
    policy.generation.vllm_cfg.tensor_parallel_size="${INFER_TP}"
    policy.generation.vllm_cfg.pipeline_parallel_size=1
    policy.generation.vllm_cfg.expert_parallel_size="${INFER_EP}"
    policy.generation.vllm_cfg.max_model_len="${MAX_SEQUENCE_LENGTH}"
    ++policy.generation.vllm_cfg.cap_max_tokens_to_context=true
    policy.generation.vllm_cfg.gpu_memory_utilization="${VLLM_GPU_MEMORY_UTILIZATION}"
    policy.generation.vllm_cfg.enforce_eager="${VLLM_ENFORCE_EAGER}"
    ++policy.generation.vllm_cfg.enable_prefix_caching=false
    policy.generation.vllm_cfg.logprobs_mode=raw_logprobs
    ++policy.generation.vllm_kwargs.limit_mm_per_prompt.image=1
    ++policy.generation.vllm_kwargs.max_num_batched_tokens="${VLLM_MAX_NUM_BATCHED_TOKENS}"
    ++policy.generation.vllm_kwargs.mamba_ssm_cache_dtype=float32
    ++policy.generation.vllm_kwargs.skip_mm_profiling=true
    ++policy.generation.vllm_kwargs.kernel_config.enable_flashinfer_autotune=false
    ++policy.generation.vllm_kwargs.kernel_config.moe_backend=triton
    # vLLM bans these tokens rather than treating them as stop strings, which
    # Gym clears anyway.
    '++policy.generation.bad_words=["<image>","<img>","</img>","<so_embedding>","<so_start>","<so_end>"]'
  )
fi

echo "Launching ${JOB_NAME}: ${GPUS_PER_NODE} visible GPU(s)"
echo "  env: NeMo-Gym circle_count (images, base64 data URLs) via ${GYM}"
echo "  generation backend: ${GENERATION_BACKEND} colocated=${COLOCATED_ENABLED} async=${ASYNC_GRPO}"
echo "  async: max_trajectory_age=${MAX_TRAJECTORY_AGE_STEPS} in_flight_weight_updates=${IN_FLIGHT_WEIGHT_UPDATES}"
echo "  training world size: ${TRAIN_WORLD_SIZE} (TP=${POLICY_TP}, EP=${POLICY_EP}, ETP=1)"
echo "  inference world size: ${INFERENCE_WORLD_SIZE} (TP=${INFER_TP}, EP=${INFER_EP})"
echo "  data: train=${TRAIN_JSONL} ($(wc -l < "${TRAIN_JSONL}") rows) val=${VAL_JSONL} ($(wc -l < "${VAL_JSONL}") rows)"
echo "  seq/new_tokens: ${MAX_SEQUENCE_LENGTH}/${MAX_NEW_TOKENS}"
echo "  optimizer moments: exp_avg=${EXP_AVG_DTYPE:-<unset>} exp_avg_sq=${EXP_AVG_SQ_DTYPE:-<unset>} store_param_remainders=${STORE_PARAM_REMAINDERS:-<unset>}"
echo "  W&B: ${WANDB_GROUP}/${WANDB_PROJ}/${WANDB_NAME} (enabled=${WANDB_ENABLED})"

exec env "${NSYS_ENV[@]}" uv run --no-sync python examples/nemo_gym/run_grpo_nemo_gym.py \
  --config "${CONFIG}" \
  "${COMMON_OVERRIDES[@]}" \
  "${GEN_OVERRIDES[@]}" \
  "$@"

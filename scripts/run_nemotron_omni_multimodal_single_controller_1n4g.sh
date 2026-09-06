#!/usr/bin/env bash
set -euo pipefail

# One-node/four-GPU NeMo-RL v2 smoke launcher for Nemotron Omni multimodal GRPO.
# TASK=clevr uses native image rollouts; TASK=vstat uses NeMo-Gym video rollouts.
# Both paths use SingleController, TransferQueue, async non-colocated Megatron
# generation (2 train GPUs + 2 generation GPUs), with configurable decode CUDA
# graphs and transformer implementation.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
NEMORL="${NEMORL:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
TASK="${TASK:-clevr}"
MODEL_NAME="${MODEL_NAME:-nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-${NEMORL}/workspace}"

case "${TASK}" in
  clevr)
    DEFAULT_CONFIG="examples/configs/recipes/vlm/vlm_grpo-nemotron-omni-30ba3b-clevr-8n4g-megatron-single-controller-async.v1.yaml"
    MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-2048}"
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
    NUM_PROMPTS_PER_STEP="${NUM_PROMPTS_PER_STEP:-1}"
    NUM_GENERATIONS_PER_PROMPT="${NUM_GENERATIONS_PER_PROMPT:-2}"
    ;;
  vstat)
    DEFAULT_CONFIG="examples/configs/recipes/vlm/vlm_grpo-nemotron-omni-30ba3b-16n8g-megatron-tp4ep4-async-gym-video.v1.yaml"
    MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-4096}"
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
    NUM_PROMPTS_PER_STEP="${NUM_PROMPTS_PER_STEP:-2}"
    NUM_GENERATIONS_PER_PROMPT="${NUM_GENERATIONS_PER_PROMPT:-4}"
    ;;
  *)
    echo "TASK must be clevr or vstat (got ${TASK})." >&2
    exit 1
    ;;
esac

CONFIG="${CONFIG:-${DEFAULT_CONFIG}}"
MAX_STEPS="${MAX_STEPS:-4}"
TRAIN_GBS="${TRAIN_GBS:-$((NUM_PROMPTS_PER_STEP * NUM_GENERATIONS_PER_PROMPT))}"
MAX_LOOKAHEAD_VERSIONS="${MAX_LOOKAHEAD_VERSIONS:-1}"
MAX_INFLIGHT_PROMPTS="${MAX_INFLIGHT_PROMPTS:-$((NUM_PROMPTS_PER_STEP * (MAX_LOOKAHEAD_VERSIONS + 1)))}"
MAX_BUFFERED_ROLLOUTS="${MAX_BUFFERED_ROLLOUTS:-$((NUM_PROMPTS_PER_STEP * (MAX_LOOKAHEAD_VERSIONS + 1)))}"
ASYNC_RL_DIAGNOSTICS="${ASYNC_RL_DIAGNOSTICS:-true}"
TOKEN_CAPTURE_ENABLED="${TOKEN_CAPTURE_ENABLED:-false}"
TRAIN_GPUS="${TRAIN_GPUS:-2}"
GEN_GPUS="${GEN_GPUS:-2}"
POLICY_TP="${POLICY_TP:-${TRAIN_GPUS}}"
POLICY_EP="${POLICY_EP:-${TRAIN_GPUS}}"
INFER_TP="${INFER_TP:-${GEN_GPUS}}"
INFER_EP="${INFER_EP:-${GEN_GPUS}}"
REFIT_BACKEND="${REFIT_BACKEND:-nccl}"
BUFFER_SIZE_GB="${BUFFER_SIZE_GB:-8}"
MEGATRON_TRANSFORMER_IMPL="${MEGATRON_TRANSFORMER_IMPL:-inference_optimized}"
MEGATRON_CUDA_GRAPH_IMPL="${MEGATRON_CUDA_GRAPH_IMPL:-local}"
if [[ "${MEGATRON_TRANSFORMER_IMPL}" != "inference_optimized" &&
      "${MEGATRON_CUDA_GRAPH_IMPL}" == "local" ]]; then
  MOE_PAD_EXPERTS_FOR_CG="${MOE_PAD_EXPERTS_FOR_CG:-true}"
else
  MOE_PAD_EXPERTS_FOR_CG=false
fi
VISION_EMBEDDING_CACHE_MAX_BYTES="${VISION_EMBEDDING_CACHE_MAX_BYTES:-536870912}"
EXP_AVG_DTYPE="${EXP_AVG_DTYPE:-bfloat16}"
EXP_AVG_SQ_DTYPE="${EXP_AVG_SQ_DTYPE:-bfloat16}"
STORE_PARAM_REMAINDERS="${STORE_PARAM_REMAINDERS:-true}"
WANDB_ENABLED="${WANDB_ENABLED:-false}"
RESULTS_DIR="${RESULTS_DIR:-${WORKSPACE_ROOT}/results/nemo-rl-v2-omni/${TASK}}"

cd "${NEMORL}"

GPUS_PER_NODE="${GPUS_PER_NODE:-$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)}"
if (( GPUS_PER_NODE != TRAIN_GPUS + GEN_GPUS )); then
  echo "Expected TRAIN_GPUS + GEN_GPUS = ${GPUS_PER_NODE}, got ${TRAIN_GPUS} + ${GEN_GPUS}." >&2
  exit 1
fi
if (( TRAIN_GBS != NUM_PROMPTS_PER_STEP * NUM_GENERATIONS_PER_PROMPT )); then
  echo "TRAIN_GBS must equal NUM_PROMPTS_PER_STEP * NUM_GENERATIONS_PER_PROMPT." >&2
  exit 1
fi
if (( TRAIN_GPUS % POLICY_TP != 0 || TRAIN_GPUS % POLICY_EP != 0 )); then
  echo "POLICY_TP and POLICY_EP must divide TRAIN_GPUS." >&2
  exit 1
fi
if (( GEN_GPUS % INFER_TP != 0 || GEN_GPUS % INFER_EP != 0 )); then
  echo "INFER_TP and INFER_EP must divide GEN_GPUS." >&2
  exit 1
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "Missing config under ${NEMORL}: ${CONFIG}" >&2
  exit 1
fi

CACHE_ROOT="${CACHE_ROOT:-${WORKSPACE_ROOT}/cache/nemo-rl-omni}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export NRL_MEGATRON_CHECKPOINT_DIR="${NRL_MEGATRON_CHECKPOINT_DIR:-${CACHE_ROOT}/megatron-checkpoints}"
export NEMO_RL_VENV_DIR="${NEMO_RL_VENV_DIR:-/opt/ray_venvs}"
export NEMO_GYM_VENV_DIR="${NEMO_GYM_VENV_DIR:-/opt/gym_venvs}"
export NEMO_GYM_EXTRA_ROOTS="${NEMO_GYM_EXTRA_ROOTS:-${NEMORL}/3rdparty/Gym-workspace/Gym}"
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export NRL_VENVS_TRUST_EXISTING="${NRL_VENVS_TRUST_EXISTING:-1}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export FLASHINFER_DISABLE_VERSION_CHECK="${FLASHINFER_DISABLE_VERSION_CHECK:-1}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-10.0}"

BRIDGE="${NEMORL}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge"
export PYTHONPATH="${NEMORL}:${NEMO_GYM_EXTRA_ROOTS}:${BRIDGE}/src:${BRIDGE}/3rdparty/Megatron-LM${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p \
  "${HF_HOME}" \
  "${NRL_MEGATRON_CHECKPOINT_DIR}" \
  "${RESULTS_DIR}"

TASK_OVERRIDES=()
if [[ "${TASK}" == "clevr" ]]; then
CLEVR_DATASET_MODE="${CLEVR_DATASET_MODE:-smoke}"
if [[ "${CLEVR_DATASET_MODE}" == "smoke" ]]; then
    DATA_ROOT="${DATA_ROOT:-${WORKSPACE_ROOT}/datasets/clevr-smoke}"
    TRAIN_JSONL="${DATA_ROOT}/train.jsonl"
    VAL_JSONL="${DATA_ROOT}/val.jsonl"
    mkdir -p "${DATA_ROOT}"

    if [[ ! -s "${TRAIN_JSONL}" || ! -s "${VAL_JSONL}" ]]; then
      TRAIN_JSONL="${TRAIN_JSONL}" VAL_JSONL="${VAL_JSONL}" \
        uv run --no-sync python - <<'PY'
import base64
import io
import json
import os

from PIL import Image

buffer = io.BytesIO()
Image.new("RGB", (224, 224), color="red").save(buffer, format="PNG")
image_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()

def sample(index: int) -> dict:
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_url},
                    {"type": "text", "text": f"Sample {index}: What color is the image?"},
                ],
            },
            {"role": "assistant", "content": "<answer>red</answer>"},
        ]
    }

for path, count in ((os.environ["TRAIN_JSONL"], 8), (os.environ["VAL_JSONL"], 2)):
    with open(path, "w", encoding="utf-8") as output:
        for index in range(count):
            output.write(json.dumps(sample(index)) + "\n")
PY
    fi

    TASK_OVERRIDES=(
      data.train.dataset_name=ResponseDataset
      ++data.train.data_path="${TRAIN_JSONL}"
      data.train.split=train
      data.validation.dataset_name=ResponseDataset
      ++data.validation.data_path="${VAL_JSONL}"
      data.validation.split=train
      data.num_workers=0
    )
  elif [[ "${CLEVR_DATASET_MODE}" != "config" ]]; then
    echo "CLEVR_DATASET_MODE must be smoke or config (got ${CLEVR_DATASET_MODE})." >&2
    exit 1
  fi
elif [[ "${TASK}" == "vstat" ]]; then
  DATA_ROOT="${DATA_ROOT:-${WORKSPACE_ROOT}/datasets/vstat-smoke}"
  HF_DATASET="${HF_DATASET:-ShushengYang/VSTAT}"
  NUM_DATA_ROWS="${NUM_DATA_ROWS:-8}"
  PREPARE_VSTAT="${PREPARE_VSTAT:-false}"
  NUM_FRAMES="${NUM_FRAMES:-8}"
  TEMPORAL_PATCH_SIZE="${TEMPORAL_PATCH_SIZE:-2}"
  VIDEO_TARGET_PATCHES="${VIDEO_TARGET_PATCHES:-256}"
  MIN_GENERATION_TOKENS="${MIN_GENERATION_TOKENS:-2000}"
  ENABLE_THINKING="${ENABLE_THINKING:-true}"

  export NRL_VIDEO_BACKEND="${NRL_VIDEO_BACKEND:-torchcodec}"
  export NRL_VIDEO_SAMPLING_STYLE="${NRL_VIDEO_SAMPLING_STYLE:-nemotron_vl}"
  export NRL_VIDEO_TEMPORAL_PATCH_SIZE="${TEMPORAL_PATCH_SIZE}"
  mkdir -p "${DATA_ROOT}"

  MEGATRON_WORKER_PYTHON="${RAY_MEGATRON_PYTHON:-${NEMO_RL_VENV_DIR}/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python}"
  if [[ ! -x "${MEGATRON_WORKER_PYTHON}" ]]; then
    echo "Creating the Megatron worker environment before installing PyAV"
    FORCE_REBUILD_VENV="${NRL_FORCE_REBUILD_VENVS:-false}" \
      uv run --no-sync python - <<'PY'
import os

from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.utils.venvs import create_local_venv

create_local_venv(
    PY_EXECUTABLES.MCORE,
    "nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker",
    force_rebuild=os.environ["FORCE_REBUILD_VENV"].lower() == "true",
)
PY
    # The venv was rebuilt above if requested. Do not let Ray rebuild it again
    # after PyAV has been installed, since project sync intentionally excludes av.
    export NRL_FORCE_REBUILD_VENVS=false
  fi
  if ! python -c "import torchcodec" >/dev/null 2>&1 ||
     ! "${MEGATRON_WORKER_PYTHON}" -c "import av" >/dev/null 2>&1; then
    RAY_MEGATRON_PYTHON="${MEGATRON_WORKER_PYTHON}" \
      bash tools/install_audio_deps.sh
  fi

  TRAIN_JSONL="${DATA_ROOT}/train-gym.jsonl"
  VAL_JSONL="${DATA_ROOT}/val-gym.jsonl"
  if [[ "${PREPARE_VSTAT}" == "true" || ! -s "${TRAIN_JSONL}" || ! -s "${VAL_JSONL}" ]]; then
    uv run --no-sync python scripts/prepare_nemotron_omni_vstat.py \
      --output-dir "${DATA_ROOT}" \
      --repo-id "${HF_DATASET}" \
      --num-rows "${NUM_DATA_ROWS}"
  fi

  TASK_OVERRIDES=(
    policy.tokenizer.chat_template_kwargs.enable_thinking="${ENABLE_THINKING}"
    ++policy.generation.mcore_generation_config.video_num_frames="${NUM_FRAMES}"
    ++policy.generation.mcore_generation_config.video_temporal_patch_size="${TEMPORAL_PATCH_SIZE}"
    ++policy.generation.mcore_generation_config.video_target_num_patches="${VIDEO_TARGET_PATCHES}"
    data.default.num_frames="${NUM_FRAMES}"
    data.default.video_sampling_style=nemotron_vl
    data.default.video_temporal_patch_size="${TEMPORAL_PATCH_SIZE}"
    +data.default.min_generation_tokens="${MIN_GENERATION_TOKENS}"
    data.default.video_target_num_patches="${VIDEO_TARGET_PATCHES}"
    data.train.data_path="${TRAIN_JSONL}"
    data.validation.data_path="${VAL_JSONL}"
    ++env.nemo_gym.policy_model.responses_api_models.vllm_model.chat_template_kwargs.enable_thinking="${ENABLE_THINKING}"
    grpo.deduplicate_multimodal_data=false
  )
fi

COMMON_OVERRIDES=(
  ++data_plane.enabled=true
  ++data_plane.impl=transfer_queue
  ++data_plane.backend=simple
  ++data_plane.claim_meta_poll_interval_s=0.5
  ++data_plane.simple.num_storage_units=2
  cluster.num_nodes=1
  cluster.gpus_per_node="${GPUS_PER_NODE}"
  policy.model_name="${MODEL_NAME}"
  policy.tokenizer.name="${MODEL_NAME}"
  policy.is_vlm=true
  policy.max_total_sequence_length="${MAX_SEQUENCE_LENGTH}"
  policy.train_global_batch_size="${TRAIN_GBS}"
  policy.megatron_cfg.tensor_model_parallel_size="${POLICY_TP}"
  policy.megatron_cfg.expert_model_parallel_size="${POLICY_EP}"
  policy.megatron_cfg.expert_tensor_parallel_size=1
  policy.megatron_cfg.context_parallel_size=1
  policy.megatron_cfg.sequence_parallel=true
  policy.megatron_cfg.bias_activation_fusion=false
  policy.megatron_cfg.optimizer.optimizer_cpu_offload=false
  policy.megatron_cfg.optimizer.optimizer_offload_fraction=0.0
  policy.megatron_cfg.distributed_data_parallel_config.overlap_param_gather=false
  ++policy.megatron_cfg.optimizer.exp_avg_dtype="${EXP_AVG_DTYPE}"
  ++policy.megatron_cfg.optimizer.exp_avg_sq_dtype="${EXP_AVG_SQ_DTYPE}"
  ++policy.megatron_cfg.optimizer.store_param_remainders="${STORE_PARAM_REMAINDERS}"
  policy.generation.backend=megatron
  ++policy.generation.stop_strings=null
  ++policy.generation.bad_words=null
  policy.generation.max_new_tokens="${MAX_NEW_TOKENS}"
  policy.generation.colocated.enabled=false
  policy.generation.colocated.resources.num_nodes=1
  policy.generation.colocated.resources.gpus_per_node="${GEN_GPUS}"
  policy.generation.mcore_generation_config.tensor_model_parallel_size="${INFER_TP}"
  policy.generation.mcore_generation_config.expert_model_parallel_size="${INFER_EP}"
  policy.generation.mcore_generation_config.expert_tensor_parallel_size=1
  ++policy.generation.mcore_generation_config.context_parallel_size=1
  ++policy.generation.mcore_generation_config.moe_router_dtype=fp32
  policy.generation.mcore_generation_config.transformer_impl="${MEGATRON_TRANSFORMER_IMPL}"
  policy.generation.mcore_generation_config.sequence_parallel=true
  policy.generation.mcore_generation_config.moe_pad_experts_for_cuda_graph_inference="${MOE_PAD_EXPERTS_FOR_CG}"
  policy.generation.mcore_generation_config.refit_backend="${REFIT_BACKEND}"
  policy.generation.mcore_generation_config.buffer_size_gb="${BUFFER_SIZE_GB}"
  policy.generation.mcore_generation_config.cuda_graph_impl="${MEGATRON_CUDA_GRAPH_IMPL}"
  policy.generation.mcore_generation_config.inference_cuda_graph_scope=block
  policy.generation.mcore_generation_config.num_cuda_graphs=-1
  policy.generation.mcore_generation_config.use_cuda_graphs_for_non_decode_steps=false
  policy.generation.mcore_generation_config.enable_chunked_prefill=true
  policy.generation.mcore_generation_config.kv_cache_management_mode=persist
  ++policy.generation.mcore_generation_config.async_sched_mode=async
  ++policy.generation.mcore_generation_config.logprobs_mode=raw_logprobs
  ++policy.generation.mcore_generation_config.vision_embedding_cache_max_bytes="${VISION_EMBEDDING_CACHE_MAX_BYTES}"
  policy.generation.mcore_generation_config.enable_prefix_caching=false
  grpo.async_grpo=null
  grpo.num_prompts_per_step="${NUM_PROMPTS_PER_STEP}"
  grpo.num_generations_per_prompt="${NUM_GENERATIONS_PER_PROMPT}"
  grpo.max_num_steps="${MAX_STEPS}"
  grpo.val_period=0
  grpo.val_at_start=false
  grpo.val_at_end=false
  grpo.overlong_filtering=false
  loss_fn.use_importance_sampling_correction=true
  ++async_rl.sampler.name=in_order
  ++async_rl.sampler.max_lookahead_versions="${MAX_LOOKAHEAD_VERSIONS}"
  ++async_rl.recompute_kv_cache_after_weight_updates=false
  ++async_rl.min_groups_for_streaming_train="${NUM_PROMPTS_PER_STEP}"
  ++async_rl.max_inflight_prompts="${MAX_INFLIGHT_PROMPTS}"
  ++async_rl.max_buffered_rollouts="${MAX_BUFFERED_ROLLOUTS}"
  ++async_rl.diagnostics="${ASYNC_RL_DIAGNOSTICS}"
  ++token_capture.enabled="${TOKEN_CAPTURE_ENABLED}"
  checkpointing.enabled=false
  logger.log_dir="${RESULTS_DIR}"
  logger.wandb_enabled="${WANDB_ENABLED}"
  logger.tensorboard_enabled=false
)

echo "Launching NeMo-RL v2 Omni ${TASK}: ${TRAIN_GPUS} train + ${GEN_GPUS} generation GPUs"
echo "  SingleController async sampler: in_order, max lookahead ${MAX_LOOKAHEAD_VERSIONS}"
echo "  Token capture: ${TOKEN_CAPTURE_ENABLED}"
echo "  Megatron generation: TP=${INFER_TP} EP=${INFER_EP}, ${MEGATRON_TRANSFORMER_IMPL}, CG=${MEGATRON_CUDA_GRAPH_IMPL}, MoE padding=${MOE_PAD_EXPERTS_FOR_CG}"

exec uv run --no-sync python examples/run_grpo_single_controller.py \
  --config "${CONFIG}" \
  "${COMMON_OVERRIDES[@]}" \
  "${TASK_OVERRIDES[@]}" \
  "$@"

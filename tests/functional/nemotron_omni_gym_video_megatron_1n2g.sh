#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(realpath "${SCRIPT_DIR}/../..")

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "SKIP: HF_TOKEN is required for the Omni checkpoint"
    exit 0
fi

GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if (( GPU_COUNT < 2 )); then
    echo "SKIP: Omni Gym-video Megatron smoke requires at least two GPUs"
    exit 0
fi
DETECTED_CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0)
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-${DETECTED_CUDA_ARCH}}"
MEGATRON_TRANSFORMER_IMPL="${MEGATRON_TRANSFORMER_IMPL:-inference_optimized}"
MEGATRON_CUDA_GRAPH_IMPL="${MEGATRON_CUDA_GRAPH_IMPL:-local}"
if [[ "${MEGATRON_CUDA_GRAPH_IMPL}" == "local" ]]; then
    INFERENCE_CUDA_GRAPH_SCOPE=block
    NUM_CUDA_GRAPHS=-1
else
    INFERENCE_CUDA_GRAPH_SCOPE=none
    NUM_CUDA_GRAPHS=0
fi
if [[ "${MEGATRON_TRANSFORMER_IMPL}" != "inference_optimized" &&
      "${MEGATRON_CUDA_GRAPH_IMPL}" == "local" ]]; then
    MOE_PAD_EXPERTS_FOR_CG=true
else
    MOE_PAD_EXPERTS_FOR_CG=false
fi

EXP_NAME=$(basename "$0" .sh)
EXP_DIR="${SCRIPT_DIR}/${EXP_NAME}"
LOG_DIR="${EXP_DIR}/logs"
DATA_ROOT="${EXP_DIR}/data"
VIDEO_PATH="${DATA_ROOT}/red.mp4"
RAW_TRAIN_PATH="${DATA_ROOT}/train-raw.jsonl"
RAW_VAL_PATH="${DATA_ROOT}/val-raw.jsonl"
TRAIN_PATH="${DATA_ROOT}/train-gym.jsonl"
VAL_PATH="${DATA_ROOT}/val-gym.jsonl"
JSON_METRICS="${EXP_DIR}/metrics.json"
RUN_LOG="${EXP_DIR}/run.log"
rm -rf "${EXP_DIR}"
mkdir -p "${LOG_DIR}" "${DATA_ROOT}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export NRL_VIDEO_BACKEND=torchcodec
export NRL_VIDEO_SAMPLING_STYLE=nemotron_vl
export NRL_VIDEO_TEMPORAL_PATCH_SIZE=2

bash tools/install_audio_deps.sh
ffmpeg -hide_banner -loglevel error -y \
    -f lavfi -i color=c=red:s=224x224:r=8:d=2 \
    -c:v libx264 -pix_fmt yuv420p "${VIDEO_PATH}"

for sample_id in $(seq 1 64); do
    jq -nc \
        --arg prompt "Sample ${sample_id}: What color fills the video? A. Red B. Blue" \
        --arg video "${VIDEO_PATH}" \
        '{prompt: $prompt, video: $video, answer: "A", verifier: "mcqa"}'
done > "${RAW_TRAIN_PATH}"
for sample_id in $(seq 1 2); do
    jq -nc \
        --arg prompt "Validation ${sample_id}: What color fills the video? A. Red B. Blue" \
        --arg video "${VIDEO_PATH}" \
        '{prompt: $prompt, video: $video, answer: "A", verifier: "mcqa"}'
done > "${RAW_VAL_PATH}"

uv run --no-sync examples/nemo_gym/prepare_video_dataset.py convert \
    --input "${RAW_TRAIN_PATH}" \
    --output "${TRAIN_PATH}"
uv run --no-sync examples/nemo_gym/prepare_video_dataset.py convert \
    --input "${RAW_VAL_PATH}" \
    --output "${VAL_PATH}"

# TODO(@cspades): Replace Omni 30B with a smaller pretrained model.
# For now, just use this as a partially-trainable functional test
# (frozen decoder trunk) for inference and multimodal RL.
uv run --no-sync python examples/nemo_gym/run_grpo_nemo_gym.py \
    --config examples/configs/recipes/vlm/vlm_grpo-nemotron-omni-30ba3b-16n8g-megatron-tp4ep4-async-gym-video.v1.yaml \
    cluster.num_nodes=1 \
    cluster.gpus_per_node=2 \
    policy.megatron_cfg.env_vars.TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
    policy.megatron_cfg.tensor_model_parallel_size=2 \
    policy.megatron_cfg.pipeline_model_parallel_size=1 \
    policy.megatron_cfg.expert_model_parallel_size=2 \
    policy.megatron_cfg.expert_tensor_parallel_size=1 \
    policy.megatron_cfg.context_parallel_size=1 \
    policy.megatron_cfg.sequence_parallel=true \
    policy.megatron_cfg.activation_checkpointing=true \
    ++policy.megatron_cfg.freeze_config.freeze_language_model=true \
    policy.megatron_cfg.optimizer.optimizer_cpu_offload=false \
    policy.megatron_cfg.optimizer.optimizer_offload_fraction=0.0 \
    ++policy.megatron_cfg.optimizer.params_dtype=bfloat16 \
    ++policy.megatron_cfg.optimizer.main_grads_dtype=bfloat16 \
    ++policy.megatron_cfg.optimizer.main_params_dtype=float16 \
    ++policy.megatron_cfg.optimizer.exp_avg_dtype=bfloat16 \
    ++policy.megatron_cfg.optimizer.exp_avg_sq_dtype=bfloat16 \
    ++policy.megatron_cfg.optimizer.store_param_remainders=false \
    policy.generation.backend=megatron \
    ++policy.generation.bad_words=null \
    policy.generation.colocated.enabled=true \
    policy.generation.colocated.resources.num_nodes=1 \
    policy.generation.colocated.resources.gpus_per_node=2 \
    policy.generation.max_new_tokens=128 \
    policy.generation.mcore_generation_config.expose_http_server=true \
    policy.generation.mcore_generation_config.tensor_model_parallel_size=2 \
    policy.generation.mcore_generation_config.expert_model_parallel_size=2 \
    policy.generation.mcore_generation_config.expert_tensor_parallel_size=1 \
    ++policy.generation.mcore_generation_config.context_parallel_size=1 \
    ++policy.generation.mcore_generation_config.moe_router_dtype=fp32 \
    policy.generation.mcore_generation_config.transformer_impl="${MEGATRON_TRANSFORMER_IMPL}" \
    policy.generation.mcore_generation_config.sequence_parallel=true \
    policy.generation.mcore_generation_config.refit_backend=nccl \
    policy.generation.mcore_generation_config.buffer_size_gb=2 \
    policy.generation.mcore_generation_config.cuda_graph_impl="${MEGATRON_CUDA_GRAPH_IMPL}" \
    policy.generation.mcore_generation_config.inference_cuda_graph_scope="${INFERENCE_CUDA_GRAPH_SCOPE}" \
    policy.generation.mcore_generation_config.num_cuda_graphs="${NUM_CUDA_GRAPHS}" \
    policy.generation.mcore_generation_config.use_cuda_graphs_for_non_decode_steps=false \
    ++policy.generation.mcore_generation_config.moe_pad_experts_for_cuda_graph_inference="${MOE_PAD_EXPERTS_FOR_CG}" \
    policy.generation.mcore_generation_config.enable_chunked_prefill=true \
    ++policy.generation.mcore_generation_config.async_sched_mode=async \
    policy.generation.mcore_generation_config.enable_prefix_caching=true \
    policy.generation.mcore_generation_config.max_model_len=1024 \
    policy.generation.mcore_generation_config.max_tokens=1024 \
    ++policy.generation.mcore_generation_config.video_num_frames=8 \
    ++policy.generation.mcore_generation_config.video_temporal_patch_size=2 \
    ++policy.generation.mcore_generation_config.video_target_num_patches=256 \
    policy.max_total_sequence_length=1024 \
    +data.default.num_frames=8 \
    +data.default.video_sampling_style=nemotron_vl \
    +data.default.video_temporal_patch_size=2 \
    +data.default.min_generation_tokens=128 \
    data.default.video_target_num_patches=256 \
    data.train.data_path="${TRAIN_PATH}" \
    data.validation.data_path="${VAL_PATH}" \
    grpo.deduplicate_multimodal_data=false \
    grpo.async_grpo.enabled=true \
    grpo.async_grpo.max_trajectory_age_steps=2 \
    grpo.async_grpo.in_flight_weight_updates=true \
    grpo.num_prompts_per_step=1 \
    grpo.num_generations_per_prompt=2 \
    grpo.max_num_steps=1 \
    grpo.val_period=0 \
    grpo.val_at_start=false \
    grpo.val_at_end=false \
    policy.train_global_batch_size=2 \
    policy.train_micro_batch_size=1 \
    logger.tensorboard_enabled=true \
    logger.log_dir="${LOG_DIR}" \
    logger.wandb_enabled=false \
    logger.monitor_gpus=false \
    checkpointing.enabled=false \
    "$@" 2>&1 | tee "${RUN_LOG}"

uv run --no-sync tests/json_dump_tb_logs.py "${LOG_DIR}" --output_path "${JSON_METRICS}"
uv run --no-sync tests/check_metrics.py "${JSON_METRICS}" \
    'max(data["train/gen_kl_error"]) < 0.05' \
    'all_finite(data["train/reward"])'

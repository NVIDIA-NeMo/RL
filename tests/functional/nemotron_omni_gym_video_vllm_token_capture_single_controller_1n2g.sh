#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Mirror of nemotron_omni_gym_video_megatron_single_controller_1n2g.sh with async
# vLLM generation and vLLM media token capture enabled: the worker captures the
# processed video frames it ran on and stages them beside the token delta, and
# the learner trains on the captured patches instead of re-decoding the video.
#
# Frame count must stay even: vLLM emits frames // temporal_patch_size tubelet
# placeholders while the learner builds ceil(frames / temporal_patch_size).
# vLLM's Omni processor sets the per-frame patch budget itself (there is no
# vLLM-side video_target_num_patches), so eight 224x224 frames render to about
# 2.3k prompt tokens; the sequence budget is 4096 rather than the Megatron
# sibling's 1024.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(realpath "${SCRIPT_DIR}/../..")

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "SKIP: HF_TOKEN is required for the Omni checkpoint"
    exit 0
fi

GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if (( GPU_COUNT < 2 )); then
    echo "SKIP: Omni Gym-video vLLM token-capture smoke requires at least two GPUs"
    exit 0
fi
DETECTED_CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0)
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-${DETECTED_CUDA_ARCH}}"

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

# SingleController requires disaggregated generation: one GPU trains the
# frozen-decoder policy, one GPU hosts async vLLM (TP1) with the capture host.
uv run --no-sync python examples/run_grpo_single_controller.py \
    --config examples/configs/recipes/vlm/vlm_grpo-nemotron-omni-30ba3b-16n8g-megatron-tp4ep4-async-gym-video.v1.yaml \
    cluster.num_nodes=1 \
    cluster.gpus_per_node=2 \
    ++cluster.segment_size=1 \
    policy.megatron_cfg.env_vars.TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
    policy.megatron_cfg.tensor_model_parallel_size=1 \
    policy.megatron_cfg.pipeline_model_parallel_size=1 \
    policy.megatron_cfg.expert_model_parallel_size=1 \
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
    policy.generation.backend=vllm \
    ++policy.generation.refit_transport=null \
    ++policy.generation.bad_words=null \
    policy.generation.colocated.enabled=false \
    policy.generation.colocated.resources.num_nodes=1 \
    policy.generation.colocated.resources.gpus_per_node=1 \
    policy.generation.max_new_tokens=128 \
    policy.generation.vllm_cfg.async_engine=true \
    policy.generation.vllm_cfg.expose_http_server=true \
    policy.generation.vllm_cfg.tensor_parallel_size=1 \
    policy.generation.vllm_cfg.expert_parallel_size=1 \
    policy.generation.vllm_cfg.pipeline_parallel_size=1 \
    policy.generation.vllm_cfg.max_model_len=4096 \
    policy.generation.vllm_cfg.gpu_memory_utilization=0.6 \
    policy.generation.vllm_cfg.enforce_eager=true \
    policy.generation.vllm_cfg.enable_prefix_caching=false \
    policy.generation.vllm_cfg.video.sampling_style=nemotron_vl \
    policy.generation.vllm_cfg.video.num_frames=8 \
    policy.generation.vllm_cfg.video.temporal_patch_size=2 \
    policy.generation.vllm_kwargs.allowed_local_media_path="${DATA_ROOT}" \
    policy.generation.vllm_kwargs.limit_mm_per_prompt.video.num_frames=8 \
    ++async_rl.generation_fleet_health.refit_timeout_s=null \
    ++token_capture.enabled=true \
    policy.max_total_sequence_length=4096 \
    data.default.num_frames=8 \
    data.default.video_sampling_style=nemotron_vl \
    data.default.video_temporal_patch_size=2 \
    +data.default.min_generation_tokens=128 \
    data.default.video_target_num_patches=256 \
    data.train.data_path="${TRAIN_PATH}" \
    data.validation.data_path="${VAL_PATH}" \
    grpo.deduplicate_multimodal_data=false \
    grpo.async_grpo=null \
    grpo.num_prompts_per_step=1 \
    grpo.num_generations_per_prompt=2 \
    grpo.max_num_steps=1 \
    grpo.val_period=0 \
    grpo.val_at_start=false \
    grpo.val_at_end=false \
    policy.train_global_batch_size=2 \
    policy.train_micro_batch_size=1 \
    ++data_plane.enabled=true \
    ++data_plane.impl=transfer_queue \
    ++data_plane.backend=simple \
    ++data_plane.claim_meta_poll_interval_s=0.5 \
    ++data_plane.simple.num_storage_units=2 \
    ++async_rl.sampler.name=in_order \
    ++async_rl.sampler.max_lookahead_versions=1 \
    ++async_rl.recompute_kv_cache_after_weight_updates=false \
    ++async_rl.min_groups_for_streaming_train=1 \
    ++async_rl.max_inflight_prompts=2 \
    ++async_rl.max_buffered_rollouts=2 \
    logger.tensorboard_enabled=true \
    logger.log_dir="${LOG_DIR}" \
    logger.wandb_enabled=false \
    logger.monitor_gpus=false \
    checkpointing.enabled=false \
    "$@" 2>&1 | tee "${RUN_LOG}"

uv run --no-sync tests/json_dump_tb_logs.py "${LOG_DIR}" --output_path "${JSON_METRICS}"
# The finalize/* gates prove capture really ran (a finalizer actor did work),
# rejected nothing, and every learner row carried captured video patches.
uv run --no-sync tests/check_metrics.py "${JSON_METRICS}" \
    'max(data["train/finalize/total_ms"]) > 0' \
    'max(data["train/finalize/invalid_row_rate"]) == 0' \
    'max(data["train/finalize/capture_poisoned_rollouts"]) == 0' \
    'min(data["train/finalize/media_row_rate"]) == 1' \
    'max(data["train/gen_kl_error"]) < 0.05' \
    'all_finite(data["train/reward"])'

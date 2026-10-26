#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Mirror of nemotron_omni_clevr_megatron_single_controller_1n2g.sh with async
# vLLM generation and vLLM media token capture enabled. The worker captures the
# processed image pixels it ran on, stages them in the same TQ write as the
# token delta, and the finalizer rebuilds learner rows from captured patches.
#
# Token capture requires the NeMo-Gym rollout path, so the native CLEVR
# ResponseDataset rows of the Megatron sibling become Gym string_match rows.
# The capture path publishes packed patches rather than raw pixel_values, so
# it is not affected by https://github.com/NVIDIA-NeMo/RL/issues/4208.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(realpath "${SCRIPT_DIR}/../..")

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "SKIP: HF_TOKEN is required for the Omni checkpoint"
    exit 0
fi

GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if (( GPU_COUNT < 2 )); then
    echo "SKIP: Omni CLEVR vLLM token-capture smoke requires at least two visible GPUs"
    exit 0
fi
DETECTED_CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0)
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-${DETECTED_CUDA_ARCH}}"

EXP_NAME=$(basename "$0" .sh)
EXP_DIR="${SCRIPT_DIR}/${EXP_NAME}"
LOG_DIR="${EXP_DIR}/logs"
DATA_ROOT="${EXP_DIR}/data"
TRAIN_PATH="${DATA_ROOT}/train.jsonl"
VAL_PATH="${DATA_ROOT}/val.jsonl"
JSON_METRICS="${EXP_DIR}/metrics.json"
RUN_LOG="${EXP_DIR}/run.log"
rm -rf "${EXP_DIR}"
mkdir -p "${LOG_DIR}" "${DATA_ROOT}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Same solid-red 224x224 fixture as the Megatron siblings, emitted as Gym
# string_match rows (base64 input_image + boxed answer) instead of chat rows.
TRAIN_PATH="${TRAIN_PATH}" VAL_PATH="${VAL_PATH}" uv run --no-sync python - <<'PY'
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
        "responses_create_params": {
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_image", "image_url": image_url, "detail": "auto"},
                        {
                            "type": "input_text",
                            "text": (
                                f"Sample {index}: What color is the image? "
                                "Answer with one word inside \\boxed{}."
                            ),
                        },
                    ],
                }
            ]
        },
        "expected_answer": "red",
        "extraction_mode": "boxed",
        "case_sensitive": False,
        "agent_ref": {"type": "responses_api_agents", "name": "string_match_simple_agent"},
    }


for path, count in ((os.environ["TRAIN_PATH"], 64), (os.environ["VAL_PATH"], 2)):
    with open(path, "w") as output:
        for index in range(count):
            output.write(json.dumps(sample(index)) + "\n")
PY

# SingleController requires disaggregated generation: one GPU trains the
# frozen-decoder policy, one GPU hosts async vLLM (TP1) with the capture host.
uv run --no-sync python examples/run_grpo_single_controller.py \
    --config examples/nemo_gym/vlm_grpo_nemotron_omni_clevr_vllm_token_capture_single_controller.yaml \
    cluster.num_nodes=1 \
    cluster.gpus_per_node=2 \
    ++cluster.segment_size=1 \
    policy.megatron_cfg.env_vars.TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
    policy.megatron_cfg.tensor_model_parallel_size=1 \
    policy.megatron_cfg.expert_model_parallel_size=1 \
    policy.megatron_cfg.expert_tensor_parallel_size=1 \
    policy.megatron_cfg.context_parallel_size=1 \
    policy.megatron_cfg.sequence_parallel=true \
    policy.megatron_cfg.activation_checkpointing=true \
    ++policy.megatron_cfg.freeze_config.freeze_language_model=true \
    +policy.megatron_cfg.bias_dropout_fusion=false \
    policy.megatron_cfg.optimizer.optimizer_cpu_offload=false \
    policy.megatron_cfg.optimizer.optimizer_offload_fraction=0.0 \
    ++policy.megatron_cfg.optimizer.params_dtype=bfloat16 \
    ++policy.megatron_cfg.optimizer.main_grads_dtype=bfloat16 \
    ++policy.megatron_cfg.optimizer.main_params_dtype=float16 \
    ++policy.megatron_cfg.optimizer.exp_avg_dtype=bfloat16 \
    ++policy.megatron_cfg.optimizer.exp_avg_sq_dtype=bfloat16 \
    ++policy.megatron_cfg.optimizer.store_param_remainders=false \
    policy.generation.colocated.enabled=false \
    policy.generation.colocated.resources.num_nodes=1 \
    policy.generation.colocated.resources.gpus_per_node=1 \
    policy.generation.max_new_tokens=128 \
    policy.generation.vllm_cfg.tensor_parallel_size=1 \
    policy.generation.vllm_cfg.expert_parallel_size=1 \
    policy.generation.vllm_cfg.pipeline_parallel_size=1 \
    policy.generation.vllm_cfg.max_model_len=1024 \
    policy.generation.vllm_cfg.gpu_memory_utilization=0.6 \
    ++policy.generation.vllm_kwargs.max_num_batched_tokens=1024 \
    ++async_rl.generation_fleet_health.refit_timeout_s=null \
    policy.max_total_sequence_length=1024 \
    data.train.data_path="${TRAIN_PATH}" \
    data.validation.data_path="${VAL_PATH}" \
    grpo.num_prompts_per_step=1 \
    grpo.num_generations_per_prompt=2 \
    grpo.max_num_steps=1 \
    grpo.val_period=0 \
    grpo.val_at_start=false \
    grpo.val_at_end=false \
    policy.train_global_batch_size=2 \
    policy.train_micro_batch_size=1 \
    data_plane.simple.num_storage_units=2 \
    async_rl.min_groups_for_streaming_train=1 \
    async_rl.max_inflight_prompts=2 \
    async_rl.max_buffered_rollouts=2 \
    logger.tensorboard_enabled=true \
    logger.log_dir="${LOG_DIR}" \
    logger.wandb_enabled=false \
    logger.monitor_gpus=false \
    checkpointing.enabled=false \
    "$@" 2>&1 | tee "${RUN_LOG}"

uv run --no-sync tests/json_dump_tb_logs.py "${LOG_DIR}" --output_path "${JSON_METRICS}"
# The finalize/* gates prove capture really ran (a finalizer actor did work),
# rejected nothing, and every learner row carried captured pixels.
uv run --no-sync tests/check_metrics.py "${JSON_METRICS}" \
    'max(data["train/finalize/total_ms"]) > 0' \
    'max(data["train/finalize/invalid_row_rate"]) == 0' \
    'max(data["train/finalize/capture_poisoned_rollouts"]) == 0' \
    'min(data["train/finalize/media_row_rate"]) == 1' \
    'max(data["train/gen_kl_error"]) < 0.05' \
    'all_finite(data["train/reward"])'

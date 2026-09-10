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
    echo "SKIP: Omni CLEVR Megatron smoke requires at least two visible GPUs"
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
TRAIN_PATH="${DATA_ROOT}/train.jsonl"
VAL_PATH="${DATA_ROOT}/val.jsonl"
JSON_METRICS="${EXP_DIR}/metrics.json"
RUN_LOG="${EXP_DIR}/run.log"
rm -rf "${EXP_DIR}"
mkdir -p "${LOG_DIR}" "${DATA_ROOT}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Use a tiny local image dataset. Downloading the full 70K CLEVR training split
# adds several minutes to a one-step smoke and does not improve E2E coverage.
# TODO(@cspades): Replace Omni 30B with a smaller pretrained model.
# For now, just use this as a partially-trainable functional test
# for inference and multimodal RL.
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
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_url},
                    {
                        "type": "text",
                        "text": f"Sample {index}: What color is the image?",
                    },
                ],
            },
            {"role": "assistant", "content": "<answer>red</answer>"},
        ]
    }

for path, count in ((os.environ["TRAIN_PATH"], 64), (os.environ["VAL_PATH"], 2)):
    with open(path, "w") as output:
        for index in range(count):
            output.write(json.dumps(sample(index)) + "\n")
PY

# TODO(@cspades): Replace Omni 30B with a smaller pretrained model.
# For now, just use this as a partially-trainable functional test
# (frozen decoder trunk) for inference and multimodal RL.
uv run --no-sync python examples/run_vlm_grpo.py \
    --config examples/configs/recipes/vlm/vlm_grpo-nemotron-omni-30ba3b-clevr-8n4g-megatron_generation.v1.yaml \
    cluster.num_nodes=1 \
    cluster.gpus_per_node=2 \
    policy.megatron_cfg.env_vars.TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
    policy.megatron_cfg.tensor_model_parallel_size=2 \
    policy.megatron_cfg.expert_model_parallel_size=2 \
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
    policy.generation.backend=megatron \
    policy.generation.colocated.enabled=true \
    policy.generation.colocated.resources.num_nodes=1 \
    policy.generation.colocated.resources.gpus_per_node=2 \
    policy.generation.max_new_tokens=128 \
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
    policy.generation.mcore_generation_config.moe_pad_experts_for_cuda_graph_inference="${MOE_PAD_EXPERTS_FOR_CG}" \
    policy.generation.mcore_generation_config.enable_chunked_prefill=true \
    ++policy.generation.mcore_generation_config.async_sched_mode=async \
    policy.generation.mcore_generation_config.max_model_len=1024 \
    policy.generation.mcore_generation_config.max_tokens=1024 \
    policy.max_total_sequence_length=1024 \
    data.train.dataset_name=ResponseDataset \
    ++data.train.data_path="${TRAIN_PATH}" \
    data.train.split=train \
    data.validation.dataset_name=ResponseDataset \
    ++data.validation.data_path="${VAL_PATH}" \
    data.validation.split=train \
    data.num_workers=0 \
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

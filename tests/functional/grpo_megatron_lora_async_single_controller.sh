#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")
EXP_DIR="$SCRIPT_DIR/grpo_megatron_lora_async_single_controller"
LOG_DIR="$EXP_DIR/logs"
JSON_METRICS="$EXP_DIR/metrics.json"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

rm -rf "$EXP_DIR"
mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT"

# One batch of lookahead matches the legacy async test's trajectory-age limit.
# The in-order sampler still delivers eight complete prompt groups per update.
uv run --group test coverage run -a --data-file="$PROJECT_ROOT/tests/.coverage" --source="$PROJECT_ROOT/nemo_rl" \
    examples/run_grpo_single_controller.py \
    --config examples/configs/grpo_math_1B_megatron_single_controller.yaml \
    policy.model_name=Qwen/Qwen2.5-0.5B \
    policy.precision=bfloat16 \
    policy.megatron_cfg.peft.enabled=true \
    policy.megatron_cfg.peft.dim=32 \
    policy.train_global_batch_size=32 \
    policy.train_micro_batch_size=1 \
    policy.logprob_batch_size=32 \
    policy.generation.backend=vllm \
    policy.generation.lora_refit_mode=merged \
    policy.generation.colocated.enabled=false \
    policy.generation.colocated.resources.gpus_per_node=1 \
    policy.generation.colocated.resources.num_nodes=1 \
    policy.generation.vllm_cfg.async_engine=true \
    cluster.gpus_per_node=2 \
    grpo.max_num_steps=3 \
    grpo.num_prompts_per_step=8 \
    grpo.num_generations_per_prompt=4 \
    data.shuffle=false \
    loss_fn.use_importance_sampling_correction=true \
    async_rl.sampler.name=in_order \
    async_rl.sampler.max_lookahead_versions=1 \
    async_rl.min_groups_for_streaming_train=8 \
    async_rl.max_inflight_prompts=16 \
    async_rl.max_buffered_rollouts=16 \
    checkpointing.enabled=false \
    logger.tensorboard_enabled=true \
    logger.wandb_enabled=false \
    logger.log_dir="$LOG_DIR" \
    "$@" 2>&1 | tee "$EXP_DIR/run.log"

uv run tests/json_dump_tb_logs.py "$LOG_DIR" --output_path "$JSON_METRICS"

# Megatron reports grad_norm after optimizer.step(), not for skipped SC steps.
uv run tests/check_metrics.py "$JSON_METRICS" \
    'sorted(data["train/skipped_step"]) == ["1", "2", "3"]' \
    'max(data["train/skipped_step"]) == 0' \
    'sorted(data["train/grad_norm"]) == ["1", "2", "3"]' \
    'all_finite(data["train/grad_norm"])' \
    'sorted(data["train/reward"]) == ["1", "2", "3"]' \
    'all_finite(data["train/reward"])' \
    'max(data["train/reward"]) > 0.03' \
    'sorted(data["train/gen_kl_error"]) == ["1", "2", "3"]' \
    'all_finite(data["train/gen_kl_error"])' \
    'max(data["train/gen_kl_error"]) < 0.001'

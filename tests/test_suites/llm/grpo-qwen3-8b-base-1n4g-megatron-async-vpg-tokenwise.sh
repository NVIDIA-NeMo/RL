#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS
# OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/common.env"

# ===== BEGIN CONFIG =====
NUM_NODES=1
GPUS_PER_NODE=4
STEPS_PER_RUN=6
MAX_STEPS=6
NUM_RUNS=1
NUM_MINUTES=240
# ===== END CONFIG =====

cd "$PROJECT_ROOT"
uv run examples/run_grpo.py --config "$CONFIG_PATH" \
    grpo.max_num_steps="$MAX_STEPS" \
    grpo.seed=903 grpo.num_prompts_per_step=2 grpo.num_generations_per_prompt=4 \
    grpo.async_grpo.max_trajectory_age_steps=1 \
    policy.train_global_batch_size=8 policy.generation_batch_size=8 \
    policy.max_total_sequence_length=3072 policy.generation.max_new_tokens=2048 \
    ++policy.generation.ignore_eos=true policy.generation.stop_strings=[] \
    policy.megatron_cfg.bias_activation_fusion=true \
    policy.megatron_cfg.distributed_data_parallel_config.grad_reduce_in_fp32=true \
    policy.generation.mcore_generation_config.max_model_len=3072 \
    policy.generation.mcore_generation_config.max_tokens=2048 \
    policy.generation.mcore_generation_config.cuda_graph_max_tokens=128 \
    ++policy.generation.mcore_generation_config.max_requests=64 \
    policy.generation.mcore_generation_config.buffer_size_gb=16 \
    grpo.val_at_start=false grpo.val_at_end=false grpo.val_period=0 \
    policy.generation.mcore_generation_config.enable_prefix_caching=true \
    logger.log_dir="$LOG_DIR" logger.wandb_enabled=false logger.monitor_gpus=false \
    checkpointing.enabled=false checkpointing.checkpoint_dir="$CKPT_DIR" \
    "$@" 2>&1 | tee "$RUN_LOG"

uv run tests/json_dump_tb_logs.py "$LOG_DIR" --output_path "$JSON_METRICS"
uv run tests/check_metrics.py "$JSON_METRICS" \
    'min(data["train/version_point_applied"]) == 1' \
    'max(data["train/version_point_num_versions"]) > 1' \
    'max(data["train/version_point_unstamped_tokens"]) == 0' \
    'max(data["train/version_point_no_snapshot"]) == 0' \
    'max(data["train/version_point_parity"]) == 0' \
    'all_finite(data["train/loss"])' \
    'all_finite(data["train/grad_norm"])'

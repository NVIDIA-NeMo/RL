#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source "$SCRIPT_DIR/common.env"

# Match the non-SingleController convergence run while exercising the
# SingleController/TransferQueue orchestration path.

# ===== BEGIN CONFIG =====
NUM_NODES=8
GPUS_PER_NODE=4
SEGMENT_SIZE=2
STEPS_PER_RUN=50
MAX_STEPS=50
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))
NUM_MINUTES=120
# ===== END CONFIG =====

exit_if_max_steps_reached

cd "$PROJECT_ROOT"

uv run examples/run_grpo_single_controller.py \
    --config "$CONFIG_PATH" \
    grpo.max_num_steps="$MAX_STEPS" \
    policy.megatron_cfg.scheduler.lr_warmup_iters=10 \
    logger.log_dir="$LOG_DIR" \
    logger.wandb_enabled=True \
    logger.wandb.project=nemo-rl \
    logger.wandb.name="$EXP_NAME" \
    logger.monitor_gpus=True \
    logger.tensorboard_enabled=True \
    checkpointing.enabled=False \
    checkpointing.checkpoint_dir="$CKPT_DIR" \
    "$@" \
    2>&1 | tee "$RUN_LOG"

uv run tests/json_dump_tb_logs.py "$LOG_DIR" --output_path "$JSON_METRICS"

uv run tests/check_metrics.py "$JSON_METRICS" \
    'all_finite(data["train/loss"])' \
    'all_finite(data["train/grad_norm"])' \
    'min(data["train/grad_norm"]) > 0' \
    'all_finite(data["train/token_mult_prob_error"])' \
    'mean(data["train/reward"], range_start=-10) > 0.6'

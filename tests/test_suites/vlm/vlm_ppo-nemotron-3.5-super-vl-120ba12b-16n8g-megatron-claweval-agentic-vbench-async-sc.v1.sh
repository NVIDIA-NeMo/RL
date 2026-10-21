#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
source "$SCRIPT_DIR/common.env"

NUM_NODES=16
GPUS_PER_NODE=8
STEPS_PER_RUN=20
MAX_STEPS=20
NUM_RUNS=1
NUM_MINUTES=480

exit_if_max_steps_reached
: "${MODEL_PATH:?Set MODEL_PATH to the Super GA checkpoint}"
: "${TRAIN_PATH:?Set TRAIN_PATH to the same mixed Gym JSONL as GRPO}"
export VAL_PATH="${VAL_PATH:-$TRAIN_PATH}"
export WANDB_MODE=online

cd "$PROJECT_ROOT"
uv run examples/run_grpo_single_controller.py \
    --config "$CONFIG_PATH" \
    ppo.max_num_steps="$MAX_STEPS" \
    logger.log_dir="$LOG_DIR" \
    logger.wandb_enabled=true \
    logger.wandb.project=nemo-rl \
    logger.wandb.name="$EXP_NAME" \
    logger.tensorboard_enabled=true \
    checkpointing.enabled=true \
    checkpointing.checkpoint_dir="$CKPT_DIR" \
    "$@" 2>&1 | tee "$RUN_LOG"

uv run tests/json_dump_tb_logs.py "$LOG_DIR" --output_path "$JSON_METRICS"
uv run tests/check_metrics.py "$JSON_METRICS" \
    '"20" in data["train/loss"]' \
    'all_finite(data["train/loss"])' \
    'all_finite(data["train/critic/loss"])' \
    'max(data["train/critic/grad_norm"]) > 0' \
    'median(data["train/token_mult_prob_error"]) < 1.05' \
    'max(data["train/session_traces/post_admission_trace_count"]) > 0'

#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source "$SCRIPT_DIR/common.env"

# ===== BEGIN CONFIG =====
NUM_NODES=1
GPUS_PER_NODE=4
STEPS_PER_RUN=20
MAX_STEPS=20
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))
NUM_MINUTES=30
# ===== END CONFIG =====

exit_if_max_steps_reached

cd "$PROJECT_ROOT"
uv run examples/run_grpo.py \
    --config "$CONFIG_PATH" \
    grpo.max_num_steps=$MAX_STEPS \
    logger.log_dir="$LOG_DIR" \
    logger.wandb_enabled=True \
    logger.wandb.project=nemo-rl \
    logger.wandb.name="$EXP_NAME" \
    logger.monitor_gpus=True \
    logger.tensorboard_enabled=True \
    checkpointing.enabled=True \
    checkpointing.checkpoint_dir="$CKPT_DIR" \
    "$@" \
    2>&1 | tee "$RUN_LOG"

uv run tests/json_dump_tb_logs.py "$LOG_DIR" --output_path "$JSON_METRICS"

# GB200 baseline: mean KL 0.0765, median token error 4.6442, mean reward 0.0531.
# Allow sampling variance while requiring sustained multi-token decoding.
uv run tests/check_metrics.py "$JSON_METRICS" \
    'sorted(data["train/loss"], key=int)[-1] == "20"' \
    'all_finite(data["train/loss"])' \
    'all_finite(data["train/grad_norm"])' \
    'all_finite(data["train/gen_kl_error"])' \
    'all_finite(data["train/token_mult_prob_error"])' \
    'mean(data["train/gen_kl_error"]) < 0.12' \
    'median(data["train/token_mult_prob_error"]) < 8.0' \
    'mean(data["train/reward"]) > 0.01' \
    'min(data["train/mean_gen_tokens_per_sample"]) > 100' \
    'max(data["train/mean_gen_tokens_per_sample"]) <= 128'

rm -rf "$CKPT_DIR"

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

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/common.env"

# ===== BEGIN CONFIG =====
NUM_NODES=1
STEPS_PER_RUN=100
MAX_STEPS=100
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))
NUM_MINUTES=120
# ===== END CONFIG =====

exit_if_max_steps_reached
cd "$PROJECT_ROOT"
uv run examples/run_grpo_single_controller.py \
    --config "$CONFIG_PATH" \
    grpo.max_num_steps="$MAX_STEPS" \
    logger.log_dir="$LOG_DIR" \
    logger.wandb_enabled=true \
    logger.wandb.name="$EXP_NAME" \
    checkpointing.checkpoint_dir="$CKPT_DIR" \
    "$@" 2>&1 | tee "$RUN_LOG"

uv run tests/json_dump_tb_logs.py "$LOG_DIR" --output_path "$JSON_METRICS"
# Unlike a throughput benchmark, an incomplete regression run must fail.
uv run tests/check_metrics.py "$JSON_METRICS" \
    "set(data[\"train/loss\"]) == {str(i) for i in range(1, $MAX_STEPS + 1)}" \
    'all_finite(data["train/loss"])' \
    'all_finite(data["train/grad_norm"])' \
    'max(data["train/weight_version_lag"]) == 0' \
    'max(data["train/num_masked_seqs_by_logprob_error"]) == 0' \
    'max(data["train/max_seq_mult_prob_error"]) < 1000' \
    'median(data["train/token_mult_prob_error"]) < 1.1'
test -d "$CKPT_DIR/step_$MAX_STEPS/policy/weights"
test -d "$CKPT_DIR/step_$MAX_STEPS/policy/optimizer"

#!/usr/bin/env bash
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

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.env"

# ===== BEGIN CONFIG =====
NUM_NODES=1
GPUS_PER_NODE=4
STEPS_PER_RUN=30
MAX_STEPS=30
NUM_RUNS=1
NUM_MINUTES=120
SNAPSHOT_MEGATRON_BRIDGE=1
# ===== END CONFIG =====

exit_if_max_steps_reached

cd "$PROJECT_ROOT"
uv run --no-sync examples/run_grpo.py \
    --config "$CONFIG_PATH" \
    grpo.max_num_steps="$MAX_STEPS" \
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

uv run --no-sync tests/json_dump_tb_logs.py "$LOG_DIR" --output_path "$JSON_METRICS"

grep -F -q "[zero_train_gen_mismatch] batch-invariant kernels enabled: backend=te_native collective=ordered flash_attention_version=4" "$RUN_LOG"

MAX_RECORDED_STEP=$(jq -r 'if has("train/loss") then (."train/loss" | keys | map(tonumber) | max // 0) else 0 end' "$JSON_METRICS")
if [[ "$MAX_RECORDED_STEP" -lt "$MAX_STEPS" ]]; then
    echo "[ERROR] Expected train/loss through step $MAX_STEPS, found step $MAX_RECORDED_STEP"
    exit 1
fi

uv run --no-sync tests/check_metrics.py "$JSON_METRICS" \
    'min(data["train/num_valid_samples"]) > 0' \
    'all_finite(data["train/loss"])' \
    'all_finite(data["train/grad_norm"])' \
    'max(data["train/grad_norm"]) > 0' \
    'min(data["train/gen_kl_error"]) == 0' \
    'max(data["train/gen_kl_error"]) == 0' \
    'min(data["train/probs_ratio_min"]) == 1' \
    'max(data["train/probs_ratio_max"]) == 1'

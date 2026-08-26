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
NUM_NODES=16
GPUS_PER_NODE=8
STEPS_PER_RUN=1
MAX_STEPS=1
NUM_RUNS=1
NUM_MINUTES=240
# ===== END CONFIG =====

exit_if_max_steps_reached

cd "$PROJECT_ROOT"

# Build a deterministic local video fixture at runtime. This driver is listed
# in disabled.txt because its 16-node topology is too large for nightly.
DATA_DIR="$EXP_DIR/data"
VIDEO_PATH="$DATA_DIR/red.mp4"
RAW_TRAIN_PATH="$DATA_DIR/train_raw.jsonl"
RAW_VALIDATION_PATH="$DATA_DIR/validation_raw.jsonl"
TRAIN_PATH="$DATA_DIR/train.jsonl"
VALIDATION_PATH="$DATA_DIR/validation.jsonl"
mkdir -p "$DATA_DIR"

ffmpeg -hide_banner -loglevel error -y \
    -f lavfi -i color=c=red:s=224x224:r=8:d=2 \
    -c:v libx264 -pix_fmt yuv420p "$VIDEO_PATH"

for sample_id in $(seq 1 4); do
    jq -nc \
        --arg prompt "Sample $sample_id: What color fills the video? A. Red B. Blue" \
        --arg video "$VIDEO_PATH" \
        '{prompt: $prompt, video: $video, answer: "A", verifier: "mcqa"}'
done > "$RAW_TRAIN_PATH"

for sample_id in $(seq 1 2); do
    jq -nc \
        --arg prompt "Validation $sample_id: What color fills the video? A. Red B. Blue" \
        --arg video "$VIDEO_PATH" \
        '{prompt: $prompt, video: $video, answer: "A", verifier: "mcqa"}'
done > "$RAW_VALIDATION_PATH"

uv run examples/nemo_gym/prepare_video_dataset.py convert \
    --input "$RAW_TRAIN_PATH" \
    --output "$TRAIN_PATH"
uv run examples/nemo_gym/prepare_video_dataset.py convert \
    --input "$RAW_VALIDATION_PATH" \
    --output "$VALIDATION_PATH"

uv run examples/nemo_gym/run_grpo_nemo_gym.py \
    --config "$CONFIG_PATH" \
    grpo.max_num_steps="$MAX_STEPS" \
    policy.generation.vllm_kwargs.allowed_local_media_path="$DATA_DIR" \
    data.train.data_path="$TRAIN_PATH" \
    data.validation.data_path="$VALIDATION_PATH" \
    policy.generation.max_new_tokens=256 \
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

RECORDED_STEP=$(jq -r \
    'if has("train/loss") then (."train/loss" | keys | map(tonumber) | max // 0) else 0 end' \
    "$JSON_METRICS")
if [[ "$RECORDED_STEP" -lt 1 ]]; then
    echo "[ERROR] Expected at least one completed training step"
    exit 1
fi

uv run tests/check_metrics.py "$JSON_METRICS" \
    'max(data["train/loss"]) < 1000000.0' \
    'min(data["train/loss"]) > -1000000.0' \
    'max(data["train/max_seq_mult_prob_error"]) < 1000000.0'

rm -rf "$CKPT_DIR"

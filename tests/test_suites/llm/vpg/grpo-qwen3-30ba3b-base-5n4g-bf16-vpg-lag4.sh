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
# The shared environment resolves recipes relative to llm/, including this subdirectory.
EXP_NAME="vpg/$(basename "${BASH_SOURCE[0]}" .sh)"
source "$SCRIPT_DIR/../common.env"

# ===== BEGIN CONFIG =====
NUM_NODES=5
GPUS_PER_NODE=4
STEPS_PER_RUN=6
MAX_STEPS=6
NUM_RUNS=1
NUM_MINUTES=240
# ===== END CONFIG =====

cd "$PROJECT_ROOT"
uv run --no-project --python "${PYTHON:-python3}" python examples/run_grpo.py \
    --config "$CONFIG_PATH" \
    grpo.seed=901 grpo.max_num_steps="$MAX_STEPS" \
    grpo.val_at_start=false grpo.val_at_end=false \
    logger.log_dir="$LOG_DIR" logger.wandb_enabled=false \
    checkpointing.enabled=false checkpointing.checkpoint_dir="$CKPT_DIR" \
    "$@" 2>&1 | tee "$RUN_LOG"

uv run --no-project --python "${PYTHON:-python3}" python \
    examples/version_point_gradients/verify_startup.py "$EXP_DIR" vpg "$MAX_STEPS" 4

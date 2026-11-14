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


SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
source "$SCRIPT_DIR/common.env"

# ===== BEGIN CONFIG =====
NUM_NODES=1
GPUS_PER_NODE=8
NUM_MINUTES=30
NUM_RUNS=1
# ===== END CONFIG =====

: "${COREWEAVE_MODEL_PATH:?Set a local directory containing the deployment model checkpoint}"
cd "$PROJECT_ROOT"
uv run --frozen --no-sync examples/run_grpo.py \
    --config "$CONFIG_PATH" \
    policy.model_name="$COREWEAVE_MODEL_PATH" \
    grpo.max_num_steps=2 \
    logger.log_dir="$LOG_DIR" \
    logger.wandb_enabled=false \
    "$@" 2>&1 | tee "$RUN_LOG"

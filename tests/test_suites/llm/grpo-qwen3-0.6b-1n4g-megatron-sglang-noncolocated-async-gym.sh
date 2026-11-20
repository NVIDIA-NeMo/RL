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

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
source "$SCRIPT_DIR/common.env"

# ===== BEGIN CONFIG =====
NUM_NODES=1
GPUS_PER_NODE=4
STEPS_PER_RUN=10
MAX_STEPS=10
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))
NUM_MINUTES=40
# ===== END CONFIG =====

exit_if_max_steps_reached

DATA_DIR=$EXP_DIR/data
mkdir -p "$DATA_DIR"
cd "$PROJECT_ROOT/3rdparty/Gym-workspace/Gym"
if [[ ! -f env.yaml ]]; then
    if [[ -z "${HF_TOKEN:-}" ]]; then
        echo "[ERROR] HF_TOKEN is not set" >&2
        exit 1
    fi
    echo "hf_token: $HF_TOKEN" >> env.yaml
fi
uv run ng_prepare_data "+config_paths=[resources_servers/workplace_assistant/configs/workplace_assistant.yaml]" \
    +output_dirpath=data/workplace_assistant \
    +mode=train_preparation +should_download=true +data_source=huggingface
cd "$PROJECT_ROOT"

# Match the functional Gym smoke's one-tool prompt budget.
TRAIN_PATH=$DATA_DIR/workplace_assistant_train.jsonl
VALIDATION_PATH=$DATA_DIR/workplace_assistant_validation.jsonl
jq -c '.responses_create_params.tools |= (.[0:1])' \
    3rdparty/Gym-workspace/Gym/data/workplace_assistant/train.jsonl > "$TRAIN_PATH"
jq -c '.responses_create_params.tools |= (.[0:1])' \
    3rdparty/Gym-workspace/Gym/data/workplace_assistant/validation.jsonl > "$VALIDATION_PATH"

PYTHONUNBUFFERED=1 uv run examples/nemo_gym/run_grpo_nemo_gym.py \
    --config "$CONFIG_PATH" \
    grpo.max_num_steps=$MAX_STEPS \
    logger.log_dir="$LOG_DIR" \
    logger.wandb_enabled=true \
    logger.wandb.project=nemo-rl \
    logger.wandb.name="$EXP_NAME" \
    logger.monitor_gpus=true \
    logger.tensorboard_enabled=true \
    data.train.data_path="$TRAIN_PATH" \
    data.validation.data_path="$VALIDATION_PATH" \
    "$@" \
    2>&1 | tee "$RUN_LOG"

uv run tests/json_dump_tb_logs.py "$LOG_DIR" --output_path "$JSON_METRICS"
uv run tests/check_metrics.py "$JSON_METRICS" \
    'all_finite(data["train/loss"])' \
    'median(data["train/gen_kl_error"]) < 1.3' \
    'min(data["timing/train/idle/refit_bubble"]) >= 0' \
    'len(data["train/loss"]) == 10' \
    'len(data["validation/accuracy"]) >= 2'

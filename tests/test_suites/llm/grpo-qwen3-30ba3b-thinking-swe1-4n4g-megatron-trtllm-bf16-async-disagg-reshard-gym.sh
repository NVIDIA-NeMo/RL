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
NUM_NODES=4
GPUS_PER_NODE=4
SEGMENT_SIZE=2
STEPS_PER_RUN=5
MAX_STEPS=5
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))
NUM_MINUTES=120
USES_SANDBOX=1
USE_GYM_CONTAINER=true
# ===== END CONFIG =====

exit_if_max_steps_reached

# NIXL KV transfer must use the RDMA HCAs, not the image's management NIC.
export UCX_NET_DEVICES=all

cd "$PROJECT_ROOT"
uv run examples/nemo_gym/run_grpo_nemo_gym.py \
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
    env.nemo_gym.uv_venv_dir=/opt/gym_venvs \
    "$@" \
    2>&1 | tee "$RUN_LOG"

assert_not_grep \
    "PD disaggregation is only wired for the HTTP/NeMo-Gym rollout" \
    "$RUN_LOG" \
    "Rollouts took the direct engine path instead of the disagg frontends"

uv run tests/json_dump_tb_logs.py "$LOG_DIR" --output_path "$JSON_METRICS"

LAST_STEP=$(jq 'to_entries | .[] | select(.key == "train/loss") | .value | keys | map(tonumber) | max' "$JSON_METRICS")
if [[ "$LAST_STEP" -lt "$MAX_STEPS" ]]; then
    echo "[ERROR] Expected step $MAX_STEPS, reached $LAST_STEP"
    exit 1
fi

uv run tests/check_metrics.py "$JSON_METRICS" \
    'median(data["train/token_mult_prob_error"]) < 1.05' \
    'mean(data["train/gen_kl_error"]) < 0.02' \
    'mean(data["train/total_reward/mean"]) > 0.05' \
    'all_finite(data["train/grad_norm"])'

rm -rf "$CKPT_DIR"

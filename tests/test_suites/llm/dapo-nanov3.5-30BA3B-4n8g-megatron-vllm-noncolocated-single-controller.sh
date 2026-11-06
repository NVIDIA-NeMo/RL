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

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

# ===== BEGIN CONFIG =====
NUM_NODES=4
GPUS_PER_NODE=8
STEPS_PER_RUN=10
MAX_STEPS=10
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
# Dynamic sampling resamples filtered groups, so a step generates more rollouts.
NUM_MINUTES=140  # Initial 4-node budget; calibrate against a completed run.
# ===== END CONFIG =====

EXP_DIR="${BASE_LOG_DIR:-$PROJECT_ROOT/logs/ruit_slurm}/${SLURM_JOB_ID:-local}-logs/nightly"
LOG_DIR="$EXP_DIR/training"
CKPT_DIR="$PROJECT_ROOT/results/$EXP_NAME-${SLURM_JOB_ID:-local}"
JSON_METRICS="$EXP_DIR/metrics.json"
RUN_LOG="$EXP_DIR/run.log"
mkdir -p "$LOG_DIR" "$CKPT_DIR"

exit_if_max_steps_reached

# Run the experiment. Dynamic sampling is only wired up on the SingleController path.
cd $PROJECT_ROOT
uv run examples/run_grpo_single_controller.py \
    --config $CONFIG_PATH \
    grpo.max_num_steps=$MAX_STEPS \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=True \
    logger.wandb.project=nemo-rl \
    logger.wandb.name=$EXP_NAME \
    logger.monitor_gpus=True \
    logger.tensorboard_enabled=True \
    checkpointing.enabled=False \
    checkpointing.checkpoint_dir=$CKPT_DIR \
    $@ \
    2>&1 | tee $RUN_LOG

uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

# The filter has to be doing something and has to be paying for it. A run where
# nothing is filtered proves nothing about dynamic sampling, and one that lets a
# filtered group shrink its step has lost the batch size the feature exists to
# hold. reward is the trained-rows mean here, so unfiltered_reward is the signal
# to gate progress on (v1 reports these two under swapped names; see
# docs/guides/single-controller.md#dynamic-sampling).
uv run tests/check_metrics.py $JSON_METRICS \
    'len(data["train/loss"]) == 10' \
    'all_finite(data["train/loss"])' \
    'all_finite(data["train/token_mult_prob_error"])' \
    'median(data["train/token_mult_prob_error"]) < 1.02' \
    'max(data["train/token_mult_prob_error"]) < 1.025' \
    'median(data["train/unfiltered_reward"]) > 0.48' \
    'max(data["rollout/zero_variance_groups_total"]) >= 50' \
    'min(data["train/skipped_step"]) == max(data["train/skipped_step"]) == 0' \
    'min(data["train/dropped_prompt_groups"]) == max(data["train/dropped_prompt_groups"]) == 0' \
    'median(data["timing/train/total_step_time"]) < 700' \
    'max(data["timing/train/total_step_time"]) < 1100'

# Clean up checkpoint directory after successful run to save space.
rm -rf "$CKPT_DIR"

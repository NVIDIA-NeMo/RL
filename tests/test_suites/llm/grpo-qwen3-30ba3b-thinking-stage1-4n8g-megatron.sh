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
#
# Conversational tool-use pivot RL (PivotRL stage 1, single-step;
# https://arxiv.org/abs/2603.21383) for Qwen3-30B-A3B-Thinking-2507.
# Colocated synchronous GRPO on Megatron against the NeMo-Gym
# single_step_tool_use_with_argument_comparison verifier. Requires the HF
# pivot dataset staged at $HF_HOME/conversational_tool_use_pivot/{train,val}.jsonl
# -- see examples/nemo_gym/prepare_conversational_tool_use_pivot_data.py.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

# ===== BEGIN CONFIG =====
NUM_NODES=4
GPUS_PER_NODE=8
STEPS_PER_RUN=3
MAX_STEPS=3
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
NUM_MINUTES=60
# ===== END CONFIG =====

exit_if_max_steps_reached

cd $PROJECT_ROOT

uv run examples/nemo_gym/run_grpo_nemo_gym.py \
    --config $CONFIG_PATH \
    grpo.max_num_steps=$MAX_STEPS \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=True \
    logger.wandb.project=nemo-rl \
    logger.wandb.name=$EXP_NAME \
    logger.monitor_gpus=True \
    logger.tensorboard_enabled=True \
    checkpointing.enabled=True \
    checkpointing.checkpoint_dir=$CKPT_DIR \
    "$@" \
    2>&1 | tee $RUN_LOG

uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

if [[ $(jq 'to_entries | .[] | select(.key == "train/loss") | .value | keys | map(tonumber) | max' $JSON_METRICS) -ge $MAX_STEPS ]]; then
    uv run tests/check_metrics.py $JSON_METRICS \
        'median(data["train/token_mult_prob_error"]) < 1.02'

    rm -rf "$CKPT_DIR"
fi

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
# MOPD: dense Qwen3-1.7B student distilled from a Qwen3-1.7B
# teacher (student == teacher -> OPD loss ~0), sequence packing ON, 3 nodes
# (1 policy + 1 vLLM + 1 teacher). MOPD is gym-only, so this drives the
# nemo_gym entrypoint. The baked gym venvs and the real nemo_gym
# train/validation datasets are provided by the test harness at runtime
# (see PR #2855); NRL_TRAIN_PATH / NRL_VAL_PATH override the placeholders.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

# ===== BEGIN CONFIG =====
NUM_NODES=3
GPUS_PER_NODE=8
STEPS_PER_RUN=5
MAX_STEPS=5
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
NUM_MINUTES=60
# ===== END CONFIG =====

exit_if_max_steps_reached

# nemo_gym train/validation jsonl (provided by the nightly environment; the
# config placeholders are overridden here). Use `:-` defaults so a dry run /
# accounting pass never aborts on unset vars.
NRL_TRAIN_PATH="${NRL_TRAIN_PATH:-/path/to/train.jsonl}"
NRL_VAL_PATH="${NRL_VAL_PATH:-/path/to/validation.jsonl}"

# Run the experiment
cd $PROJECT_ROOT
uv run examples/nemo_gym/run_grpo_nemo_gym.py \
    --config $CONFIG_PATH \
    grpo.max_num_steps=$MAX_STEPS \
    data.train.data_path=$NRL_TRAIN_PATH \
    data.validation.data_path=$NRL_VAL_PATH \
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

# Convert tensorboard logs to json
uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

# Student == teacher, so the OPD distillation signal is ~0: the policy loss
# should sit near 0 and the train-to-inference probability error near 1.0.
if [[ $(jq 'to_entries | .[] | select(.key == "train/token_mult_prob_error") | .value | keys | map(tonumber) | max' $JSON_METRICS) -ge $MAX_STEPS ]]; then
    uv run tests/check_metrics.py $JSON_METRICS \
        'abs(median(data["train/loss"])) < 0.05' \
        'median(data["train/token_mult_prob_error"]) < 1.1'

    # Clean up checkpoint directory after a successful run to save space.
    rm -rf "$CKPT_DIR"
fi

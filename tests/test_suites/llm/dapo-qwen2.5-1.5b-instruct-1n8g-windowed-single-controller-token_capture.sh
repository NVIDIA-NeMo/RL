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
NUM_NODES=1
GPUS_PER_NODE=8
STEPS_PER_RUN=100
MAX_STEPS=100
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
NUM_MINUTES=130
# ===== END CONFIG =====

exit_if_max_steps_reached

cd $PROJECT_ROOT

# Prepare the NeMo-Gym math_with_judge (OpenMathReasoning) taskset from HF; the
# recipe reads the train rows written here.
cd 3rdparty/Gym-workspace/Gym
if [[ ! -f env.yaml ]]; then
    if [[ -z "${HF_TOKEN:-}" ]]; then
        echo "[ERROR] HF_TOKEN is not set"
        exit 1
    fi
    echo "hf_token: $HF_TOKEN" >> env.yaml
fi
uv run ng_prepare_data "+config_paths=[resources_servers/math_with_judge/configs/math_with_judge.yaml]" \
    +output_dirpath=data/math_with_judge \
    +mode=train_preparation \
    +should_download=true \
    +data_source=huggingface
cd -

# Token capture is only wired up on the SingleController path.
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

uv run tests/check_metrics.py $JSON_METRICS \
    "len(data[\"train/loss\"]) == $MAX_STEPS" \
    'all_finite(data["train/loss"])' \
    'all_finite(data["train/token_mult_prob_error"])' \
    'median(data["train/token_mult_prob_error"]) < 1.02' \
    'max(data["train/token_mult_prob_error"]) < 1.05' \
    'median(data["train/gen_kl_error"]) < 0.001' \
    'median(data["train/unfiltered_reward"]) > 0.03' \
    'max(data["rollout/zero_variance_groups_total"]) >= 1500' \
    'max(data["train/finalize/invalid_row_rate"]) == 0' \
    'max(data["train/finalize/capture_poisoned_rollouts"]) == 0' \
    'min(data["train/skipped_step"]) == max(data["train/skipped_step"]) == 0' \
    'min(data["train/dropped_prompt_groups"]) == max(data["train/dropped_prompt_groups"]) == 0' \
    'max(data["train/replaced_prompt_groups"]) == 0' \
    'median(data["timing/train/total_step_time"]) < 90'

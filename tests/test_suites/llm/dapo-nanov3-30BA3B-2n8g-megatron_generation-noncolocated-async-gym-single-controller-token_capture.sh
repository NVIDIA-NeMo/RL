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
NUM_NODES=2
GPUS_PER_NODE=8
STEPS_PER_RUN=10
MAX_STEPS=3
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
# ~25 min startup (30B-MoE load + CUDA-graph warmup + nemo_gym servers) plus
# ~16 min/step, and dynamic sampling generates extra groups to replace the
# filtered ones. Three steps still cover a refit with requests in flight and a
# checkpoint save through the finalizer.
NUM_MINUTES=120  # Initial budget; calibrate against a completed run.
# ===== END CONFIG =====

exit_if_max_steps_reached

cd $PROJECT_ROOT

# Prepare nemo-gym workplace_assistant dataset (mirrors tests/functional/grpo_async_gym.sh).
DATA_DIR=$EXP_DIR/data
mkdir -p $DATA_DIR
cd 3rdparty/Gym-workspace/Gym
if [[ ! -f env.yaml ]]; then
    if [[ -z "${HF_TOKEN:-}" ]]; then
        echo "[ERROR] HF_TOKEN is not set"
        exit 1
    fi
    echo "hf_token: $HF_TOKEN" >> env.yaml
fi
uv run ng_prepare_data "+config_paths=[resources_servers/workplace_assistant/configs/workplace_assistant.yaml]" \
    +output_dirpath=data/workplace_assistant \
    +mode=train_preparation \
    +should_download=true \
    +data_source=huggingface
cd -

# Trim to a single tool so first prompt fits.
TRAIN_PATH=$DATA_DIR/workplace_assistant_train.jsonl
VALIDATION_PATH=$DATA_DIR/workplace_assistant_validation.jsonl
jq -c '.responses_create_params.tools |= (.[0:1])' 3rdparty/Gym-workspace/Gym/data/workplace_assistant/train.jsonl > $TRAIN_PATH
jq -c '.responses_create_params.tools |= (.[0:1])' 3rdparty/Gym-workspace/Gym/data/workplace_assistant/validation.jsonl > $VALIDATION_PATH

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
    checkpointing.enabled=True \
    checkpointing.checkpoint_dir=$CKPT_DIR \
    checkpointing.save_period=3 \
    data.train.data_path=$TRAIN_PATH \
    data.validation.data_path=$VALIDATION_PATH \
    $@ \
    2>&1 | tee $RUN_LOG

uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

# Gates follow the non-capture DAPO single-controller dynamic-sampling nightly
# (dapo-nanov3.5-30BA3B-4n8g-megatron-vllm-noncolocated-single-controller.sh, #4588),
# plus the capture-only finalize/* health gates this recipe adds.
#
# The dynamic-sampling gates: the filter has to be doing something (>=1 zero-variance
# group filtered across the run) and has to be paying for it without shrinking the step
# (skipped_step and dropped_prompt_groups stay 0 -- the batch size the feature exists to
# hold). unfiltered_reward, not reward, is the progress signal: reward here is the mean
# over the trained (kept) rows, which excludes the all-same-reward groups, so it is the
# wrong denominator to gate progress on (v1 reports the two under swapped names; see
# docs/guides/single-controller.md#dynamic-sampling).
#
# The capture gates: a nonzero finalize/invalid_row_rate or capture_poisoned_rollouts
# means capture or recovery lost a rollout's tokens.
#
# Thresholds are for this recipe's scale (3 steps, 2n8g 30B, Megatron generation):
# total_step_time matches the capture nightly (<1000) plus headroom for the extra
# replacement generation dynamic sampling drives; no max(total_step_time) gate because
# save_period=3 bundles a checkpoint write into the final step. zero_variance >=1 and
# unfiltered_reward >0 are intentionally loose pending calibration against a completed run.
uv run tests/check_metrics.py $JSON_METRICS \
    "len(data[\"train/loss\"]) == $MAX_STEPS" \
    'all_finite(data["train/loss"])' \
    'all_finite(data["train/token_mult_prob_error"])' \
    'median(data["train/token_mult_prob_error"]) < 1.02' \
    'max(data["train/token_mult_prob_error"]) < 1.025' \
    'median(data["train/unfiltered_reward"]) > 0.0' \
    'median(data["timing/train/total_step_time"]) < 1500' \
    'max(data["train/finalize/invalid_row_rate"]) == 0' \
    'max(data["train/finalize/capture_poisoned_rollouts"]) == 0' \
    'max(data["rollout/zero_variance_groups_total"]) >= 1' \
    'min(data["train/skipped_step"]) == max(data["train/skipped_step"]) == 0' \
    'min(data["train/dropped_prompt_groups"]) == max(data["train/dropped_prompt_groups"]) == 0'

# Clean up checkpoint directory after successful run to save space.
rm -rf "$CKPT_DIR"

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
STEPS_PER_RUN=30
MAX_STEPS=30
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
NUM_MINUTES=120
# ===== END CONFIG =====

# Floor of train/reward at step 30, taken from a run of this script. While it is
# empty, the script fails after training, before it checks the metrics.
MIN_REWARD_AT_STEP_30=

exit_if_max_steps_reached

cd $PROJECT_ROOT

# Prepare the NeMo-Gym math_with_judge (OpenMathReasoning) taskset from HF.
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

# Prepend the fixed few-shot block to the first message of every row, so the
# prompt is most of each row and sharing it changes the logprob and training time.
FEW_SHOT_PATH=$SCRIPT_DIR/grpo-nano-v2-12b-1n8g-megatron-single-controller-token_capture.few_shot.txt
TRAIN_PATH=$EXP_DIR/math_with_judge_few_shot_train.jsonl
jq -c --rawfile few_shot $FEW_SHOT_PATH '
    if (.responses_create_params.input[0].content | type) != "string" then
        error("the first input message of a row has no string content")
    else
        .responses_create_params.input[0].content |= ($few_shot + .)
    end' 3rdparty/Gym-workspace/Gym/data/math_with_judge/train.jsonl > $TRAIN_PATH

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
    data.train.data_path=$TRAIN_PATH \
    $@ \
    2>&1 | tee $RUN_LOG

assert_not_grep 'finalize: rollout .* rejected ' "$RUN_LOG" "Token-capture finalizer rejected a rollout"

# Convert tensorboard logs to json
uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

if [[ -z "$MIN_REWARD_AT_STEP_30" ]]; then
    echo "[ERROR] MIN_REWARD_AT_STEP_30 is empty in $(basename ${BASH_SOURCE[0]}). Set it from train/reward at step 30 in $JSON_METRICS."
    exit 1
fi

# A rejected or poisoned rollout becomes a placeholder row, which cannot share its prompt.
uv run tests/check_metrics.py $JSON_METRICS \
    'median(data["train/token_mult_prob_error"]) < 1.05' \
    'data["train/token_mult_prob_error"]["30"] < 1.05' \
    "data[\"train/reward\"][\"30\"] > $MIN_REWARD_AT_STEP_30" \
    'max(data["train/finalize/invalid_row_rate"]) == 0' \
    'max(data["train/finalize/capture_poisoned_rollouts"]) == 0'

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
GPUS_PER_NODE=4
STEPS_PER_RUN=10
MAX_STEPS=10
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
# Small model; nemo_gym server startup + vLLM warmup dominate the 10 steps.
NUM_MINUTES=30
# ===== END CONFIG =====

exit_if_max_steps_reached

cd $PROJECT_ROOT

# Prepare the nemo-gym workplace_assistant dataset (mirrors the SingleController
# sibling grpo-qwen3-1.7b-1n8g-megatron-vllm-gym-single-controller-length_penalty).
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

# Trim to a single tool so the first prompt fits the sequence budget.
TRAIN_PATH=$DATA_DIR/workplace_assistant_train.jsonl
VALIDATION_PATH=$DATA_DIR/workplace_assistant_validation.jsonl
jq -c '.responses_create_params.tools |= (.[0:1])' 3rdparty/Gym-workspace/Gym/data/workplace_assistant/train.jsonl > $TRAIN_PATH
jq -c '.responses_create_params.tools |= (.[0:1])' 3rdparty/Gym-workspace/Gym/data/workplace_assistant/validation.jsonl > $VALIDATION_PATH

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
    data.train.data_path=$TRAIN_PATH \
    data.validation.data_path=$VALIDATION_PATH \
    $@ \
    2>&1 | tee $RUN_LOG

# A disagg replica serves through its frontends only; see the GRPO sibling
# grpo-qwen3-1.7b-2n4g-megatron-vllm-disagg-swe1.
assert_not_grep \
    "PD disaggregation is only wired for the HTTP/NeMo-Gym rollout" \
    "$RUN_LOG" \
    "Rollouts took the direct engine path instead of the disagg frontends"

uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

# Every step refits with rollouts in flight, so token_mult_prob_error is the
# signal that the prefill/decode pause, the weight sync and the resume leave
# the sampler and the trainer in agreement. The run stops at 10 steps because
# under disaggregation the KV caches are kept across in-flight updates
# (async_rl.recompute_kv_cache_after_weight_updates must be false), so the
# mismatch grows with the weight drift on this workload's shared prompt prefix,
# with or without disaggregation; the first 10 steps stay below 1.03.
uv run tests/check_metrics.py $JSON_METRICS \
    'median(data["train/token_mult_prob_error"]) < 1.05' \
    "data['train/token_mult_prob_error']['$MAX_STEPS'] < 1.05" \
    'mean(data["train/gen_kl_error"]) < 0.005' \
    'mean(data["train/reward"]) > 0.1'

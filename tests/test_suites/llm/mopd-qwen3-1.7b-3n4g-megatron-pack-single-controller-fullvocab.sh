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
# GB200 variant of the H100 3n8g full-vocabulary MOPD sanity test. Same
# self-distillation setup as its top-k sibling -- student and teacher are the
# same Qwen3-1.7B checkpoint -- but the objective is the exact reverse KL over
# the whole vocabulary, so the divergence should sit at ~0 rather than merely
# being small.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

# ===== BEGIN CONFIG =====
NUM_NODES=3
GPUS_PER_NODE=4
STEPS_PER_RUN=5
MAX_STEPS=5
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
# The teacher payload transport and the distributed divergence kernels both cost
# more than the top-k estimator, so this gets a wider budget than its sibling.
NUM_MINUTES=25
USES_SANDBOX=1
USE_GYM_CONTAINER=true
# ===== END CONFIG =====

exit_if_max_steps_reached

cd $PROJECT_ROOT
uv run examples/run_grpo_single_controller.py \
    --config $CONFIG_PATH \
    grpo.max_num_steps=$MAX_STEPS \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=True \
    logger.wandb.project=ultra_sc \
    logger.wandb.name=$EXP_NAME \
    logger.monitor_gpus=True \
    logger.tensorboard_enabled=True \
    checkpointing.enabled=False \
    checkpointing.checkpoint_dir=$CKPT_DIR \
    "$@" \
    2>&1 | tee $RUN_LOG

uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

# opd_full returns its own metric set and never emits token_mult_prob_error, so
# gate on the divergence metric instead of the ratio diagnostic.
if [[ $(jq 'to_entries | .[] | select(.key == "train/opd_full_reverse_kl") | .value | keys | map(tonumber) | max' $JSON_METRICS) -ge $MAX_STEPS ]]; then
    uv run tests/check_metrics.py $JSON_METRICS \
        'abs(median(data["train/loss"])) < 0.02' \
        'abs(median(data["train/opd_full_reverse_kl"])) < 0.02' \
        'max(data["train/opd_full_reverse_kl_max"]) < 0.5' \
        'min(data["train/opd_full_reverse_kl_min"]) > -1e-3' \
        'max(data["train/opd_full_decomposition_error"]) < 0.05' \
        'max(data["train/on_policy_distillation/teacher_batches"]) > 0' \
        'max(data["train/on_policy_distillation/teacher_samples"]) > 0' \
        'max(data["train/on_policy_distillation/teacher_model_unique"]) == 1'

    rm -rf "$CKPT_DIR"
fi

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
# GB200 variant of the H100 3n8g full-vocabulary MOPD sanity test: exact reverse
# KL over the whole vocabulary. Qwen3-1.7B self-distillation, divergence ~0.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

# ===== BEGIN CONFIG =====
NUM_NODES=3
GPUS_PER_NODE=4
STEPS_PER_RUN=5
MAX_STEPS=5
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
# Wider than the top-k sibling: payload transport + divergence kernels cost more.
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
    logger.wandb.project=nemo-rl \
    logger.wandb.name=$EXP_NAME \
    logger.monitor_gpus=True \
    logger.tensorboard_enabled=True \
    checkpointing.enabled=False \
    checkpointing.checkpoint_dir=$CKPT_DIR \
    "$@" \
    2>&1 | tee $RUN_LOG

uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

# Gate on train/loss, which every run emits. Keying this on an opd_full metric
# would make "the objective silently never engaged" indistinguishable from "the
# assertions passed": the key would be absent, jq would print nothing, and the
# whole block would be skipped with exit 0. check_metrics.py fails on a missing
# key, so the opd_full assertions below are still required, not optional.
if [[ $(jq 'to_entries | .[] | select(.key == "train/loss") | .value | keys | map(tonumber) | max' $JSON_METRICS) -ge $MAX_STEPS ]]; then
    # Reverse KL is non-negative, so a negative value means the kernel is wrong.
    # -1.0 is aggregate_step_metrics' no-valid-token marker, not a real value.
    # teacher_full_payload_tokens is emitted only by opd_full, so it is the one
    # assertion here that is not satisfied by a run that learned nothing.
    uv run tests/check_metrics.py $JSON_METRICS \
        'abs(median(data["train/loss"])) < 0.02' \
        'abs(median(data["train/opd_full_reverse_kl"])) < 0.02' \
        'max(data["train/opd_full_reverse_kl_max"]) < 0.5' \
        'min({k: v for k, v in data["train/opd_full_reverse_kl_min"].items() if v != -1.0}) > -1e-3' \
        'max(data["train/on_policy_distillation/teacher_full_payload_tokens"]) > 0' \
        'max(data["train/on_policy_distillation/teacher_batches"]) > 0' \
        'max(data["train/on_policy_distillation/teacher_samples"]) > 0' \
        'max(data["train/on_policy_distillation/teacher_model_unique"]) == 1'

    rm -rf "$CKPT_DIR"
fi

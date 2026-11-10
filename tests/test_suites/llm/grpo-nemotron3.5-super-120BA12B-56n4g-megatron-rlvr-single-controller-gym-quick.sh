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
# 48 NeMo RL nodes (16 training, 32 generation) plus 8 nodes in a second hetgroup for the
# GenRM, NL2Bash and safety judge pools in the .pools file next to this script.
NUM_NODES=48
NUM_EXTERNAL_SERVICE_NODES=8
GPUS_PER_NODE=4
SEGMENT_SIZE=16
EXTERNAL_VLLM_SEGMENT_SIZE=2
STEPS_PER_RUN=3
MAX_STEPS=3
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
NUM_MINUTES=120
USE_GYM_CONTAINER=true
USES_SANDBOX=1
JOB_REAPER_COMMENT='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"90","reason":"other","description":"Super 3.5 GRPO with external judge pools idles GPUs during model loading"}}'
# ===== END CONFIG =====

exit_if_max_steps_reached

cd $PROJECT_ROOT

ASSETS_DIR=$HF_HOME/nemotron3.5-super-rlvr

# Competitive coding compiles in a directory that the sandbox container also mounts
# (tools/launch mounts the code snapshot, which holds EXP_DIR, into the sandbox).
export CCC_TEST_FILE=$ASSETS_DIR/ccc/ccc_v2.1_RL_all_competitions_full_metadata.jsonl
export CCC_SHARED_TEMP_DIR=$EXP_DIR/ccc_shared_tmp
mkdir -p $CCC_SHARED_TEMP_DIR

# Runtime settings of the journey launcher this recipe was validated with.
export NRL_VLLM_ASYNC_TIMEOUT_SECONDS=3600
export NRL_WG_USE_RAY_REF=1
export NRL_USE_FASTOKENS=1
export VLLM_USE_FLASHINFER_MOE_FP8=1
export VLLM_FLASHINFER_MOE_BACKEND=latency
export WANDB_INIT_TIMEOUT=300

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

# The bounds are loose on purpose: they catch a broken train/generation match, not small drift.
# One check per judge pool, because a failed judge call scores 0: GenRM, lc_judge (NL2Bash pool) and
# the jailbreak agent (safety pool, in step 1 only).
uv run tests/check_metrics.py $JSON_METRICS \
    'data["train/token_mult_prob_error"]["3"] < 1.1' \
    'max(data["train/gen_kl_error"]) < 0.005' \
    'max(data["train/reward"]) > 0.0' \
    'max(data["train/genrm_simple_agent/reward/mean"]) > 0.0' \
    'max(data["train/lc_judge_simple_agent/reward/mean"]) > 0.0' \
    'max(data["train/jailbreak_refusal_with_explanation/reward/mean"]) > 0.0'

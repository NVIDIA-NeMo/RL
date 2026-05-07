#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

# Functional test for pretrained_checkpoint.format=megatron_bridge.
#
# Phase 1: run a standard GRPO step to force the HF→megatron-bridge conversion.
# Phase 2: re-run pointing checkpointing.pretrained_checkpoint at the bridge
#          checkpoint produced in phase 1, verifying the restore path end-to-end.

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath $SCRIPT_DIR/../..)
git config --global --add safe.directory $PROJECT_ROOT

set -eou pipefail

EXP_NAME=$(basename $0 .sh)
EXP_DIR=$SCRIPT_DIR/$EXP_NAME
LOG_DIR=$EXP_DIR/logs
JSON_METRICS=$EXP_DIR/metrics.json
RUN_LOG=$EXP_DIR/run.log
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}

rm -rf $EXP_DIR $LOG_DIR
mkdir -p $EXP_DIR $LOG_DIR

MODEL_NAME="Qwen/Qwen2.5-0.5B"
EXEMPLAR_CONFIG="$PROJECT_ROOT/examples/configs/grpo_math_1B_megatron.yaml"

# Resolve the megatron checkpoint base directory using the same logic as
# get_megatron_checkpoint_dir(), so we can locate the bridge checkpoint.
if [[ -n "${NRL_MEGATRON_CHECKPOINT_DIR:-}" ]]; then
    MEGATRON_CKPT_BASE="$NRL_MEGATRON_CHECKPOINT_DIR"
elif [[ -n "${HF_HOME:-}" ]]; then
    MEGATRON_CKPT_BASE="$HF_HOME/nemo_rl"
else
    MEGATRON_CKPT_BASE="$HOME/.cache/huggingface/nemo_rl"
fi
BRIDGE_CKPT="${MEGATRON_CKPT_BASE}/${MODEL_NAME}/iter_0000000"

cd $PROJECT_ROOT

COMMON_OVERRIDES=(
    policy.model_name="$MODEL_NAME"
    grpo.num_prompts_per_step=2
    grpo.num_generations_per_prompt=4
    policy.train_global_batch_size=4
    policy.logprob_batch_size=4
    policy.train_micro_batch_size=1
    cluster.gpus_per_node=2
    grpo.max_num_steps=2
    logger.wandb_enabled=false
    logger.monitor_gpus=true
    checkpointing.enabled=false
)

# ---------------------------------------------------------------------------
# Phase 1: standard run — triggers the HF→bridge conversion for MODEL_NAME.
# ---------------------------------------------------------------------------
echo "[INFO] Phase 1: standard run (triggers HF→bridge conversion)"
uv run --no-sync coverage run -a --data-file=$PROJECT_ROOT/tests/.coverage --source=$PROJECT_ROOT/nemo_rl \
    $PROJECT_ROOT/examples/run_grpo.py \
    --config "$EXEMPLAR_CONFIG" \
    "${COMMON_OVERRIDES[@]}" \
    logger.tensorboard_enabled=false \
    logger.log_dir=$LOG_DIR/phase1 \
    $@ \
    2>&1 | tee $RUN_LOG

if [[ ! -f "${BRIDGE_CKPT}/run_config.yaml" ]]; then
    echo "[ERROR] Bridge checkpoint not found at ${BRIDGE_CKPT} after phase 1." \
         "Expected run_config.yaml to exist."
    exit 1
fi
echo "[INFO] Bridge checkpoint verified at ${BRIDGE_CKPT}"

# ---------------------------------------------------------------------------
# Phase 2: restore from the megatron-bridge checkpoint produced in phase 1.
# Uses a temp config that injects checkpointing.pretrained_checkpoint so that
# the struct-mode config loader accepts it without a + override.
# ---------------------------------------------------------------------------
echo "[INFO] Phase 2: restore from megatron-bridge checkpoint"

TEMP_CONFIG=$(mktemp --suffix=.yaml)
trap "rm -f $TEMP_CONFIG" EXIT
cat > "$TEMP_CONFIG" << YAML
defaults: ${EXEMPLAR_CONFIG}
checkpointing:
  pretrained_checkpoint:
    path: "${BRIDGE_CKPT}"
    format: megatron_bridge
YAML

uv run --no-sync coverage run -a --data-file=$PROJECT_ROOT/tests/.coverage --source=$PROJECT_ROOT/nemo_rl \
    $PROJECT_ROOT/examples/run_grpo.py \
    --config "$TEMP_CONFIG" \
    "${COMMON_OVERRIDES[@]}" \
    logger.tensorboard_enabled=true \
    logger.log_dir=$LOG_DIR \
    $@ \
    2>&1 | tee -a $RUN_LOG

uv run --no-sync tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

uv run --no-sync tests/check_metrics.py $JSON_METRICS \
    'max(data["train/token_mult_prob_error"]) < 1.05' \
    'min(data["train/probs_ratio_clamped_min"]) > 0.79' \
    'max(data["train/probs_ratio_clamped_min"]) < 1.21' \
    'min(data["train/probs_ratio_clamped_max"]) > 0.79' \
    'max(data["train/probs_ratio_clamped_max"]) < 1.21'

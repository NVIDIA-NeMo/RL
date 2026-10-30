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

# Two-GPU end-to-end SC smoke and checkpoint/resume mechanics, matching the
# MCore SC functional layer. This does not assert uninterrupted/resumed parity:
# the controller may discard in-flight prompts across a restart.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")
# A fresh directory prevents stale checkpoints/metrics from satisfying assertions.
EXP_DIR=${EXP_DIR:-$(mktemp -d "$SCRIPT_DIR/automodel_sc_full_batch_smoke.XXXXXX")}
mkdir -p "$EXP_DIR"
EXP_DIR=$(realpath "$EXP_DIR")
CKPT_DIR=$EXP_DIR/ckpts
if [[ -e "$CKPT_DIR" ]]; then
    echo "FAIL: use a fresh EXP_DIR; checkpoint directory already exists: $CKPT_DIR"
    exit 1
fi
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}
cd "$PROJECT_ROOT"

TRAIN_CMD=(
    uv run --group test coverage run -a --data-file="$PROJECT_ROOT/tests/.coverage"
    --source="$PROJECT_ROOT/nemo_rl" examples/run_grpo_single_controller.py
    --config examples/configs/recipes/llm/grpo-qwen2.5-1.5b-1n2g-automodel-single-controller-full-batch.yaml
    grpo.max_num_steps=4
    logger.monitor_gpus=false
    checkpointing.checkpoint_dir="$CKPT_DIR"
    checkpointing.save_period=2
)

# Keep max_num_steps identical across both launches. The timeout saves at the
# first step boundary, then the second launch restores and advances to step 4.
"${TRAIN_CMD[@]}" "$@" checkpointing.checkpoint_must_save_by=00:00:00:01 \
    logger.log_dir="$EXP_DIR/logs/run1" 2>&1 | tee "$EXP_DIR/run1.log"
grep -q "Timeout has been reached, stopping training early" "$EXP_DIR/run1.log"

check_checkpoint() {
    local step=$1
    local artifact
    for artifact in training_info.json config.yaml policy/weights policy/optimizer \
        train_dataloader.pt replay_buffer_metadata.pt data_plane; do
        test -e "$CKPT_DIR/step_$step/$artifact" || {
            echo "FAIL: missing step_$step/$artifact"; return 1;
        }
    done
    grep -q "\"current_step\": $step" "$CKPT_DIR/step_$step/training_info.json"
    grep -q '"sampler_name": "in_order"' "$CKPT_DIR/step_$step/training_info.json"
    test ! -e "$CKPT_DIR/step_$step/replay_buffer.pt"
    if compgen -G "$CKPT_DIR/tmp_step_*" > /dev/null; then
        echo "FAIL: checkpoint finalization left tmp_step_*"; return 1
    fi
}
check_checkpoint 1

"${TRAIN_CMD[@]}" "$@" checkpointing.checkpoint_must_save_by=null \
    logger.log_dir="$EXP_DIR/logs/run2" 2>&1 | tee "$EXP_DIR/run2.log"
grep -q "Restoring dataloader state from checkpoint" "$EXP_DIR/run2.log"
grep -q "Restoring replay buffer metadata" "$EXP_DIR/run2.log"
grep -qF "replay group(s) from checkpoint" "$EXP_DIR/run2.log"
check_checkpoint 4

for run in 1 2; do
    expected_steps='{"1"}'
    if [[ $run == 2 ]]; then expected_steps='{"2", "3", "4"}'; fi
    uv run tests/json_dump_tb_logs.py "$EXP_DIR/logs/run$run" \
        --output_path "$EXP_DIR/metrics$run.json" --error-on-conflicts
    uv run tests/check_metrics.py "$EXP_DIR/metrics$run.json" \
        "set(data[\"train/loss\"]) == $expected_steps" \
        'all_finite(data["train/loss"])' \
        'all_finite(data["train/grad_norm"])' \
        'max(data["train/weight_version_lag"]) == 0' \
        'max(data["train/num_masked_seqs_by_logprob_error"]) == 0' \
        'max(data["train/max_seq_mult_prob_error"]) < 1000' \
        'max(data["train/gen_kl_error"]) < 0.002' \
        'min(data["train/probs_ratio_clamped_min"]) > 0.79' \
        'max(data["train/probs_ratio_clamped_max"]) < 1.21'
done
echo "PASS: AutoModel full-batch SC training and checkpoint/resume ($EXP_DIR)"

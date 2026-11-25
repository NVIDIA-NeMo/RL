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
# Forest (context compaction) GRPO: the uninterrupted 10-step BrowseComp smoke
# with its reward and KL checks, then a second run resumes from step 10 to 12.
set -euo pipefail
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
EXP_NAME=$(basename $0 .sh)
EXP_DIR=$SCRIPT_DIR/$EXP_NAME
# Outside the runs' own experiment dirs, which each run deletes.
CKPT_DIR=$EXP_DIR/checkpoints
rm -rf $EXP_DIR
mkdir -p $EXP_DIR

CHECKPOINT_ARGS=(
    checkpointing.enabled=true
    checkpointing.checkpoint_dir=$CKPT_DIR
    checkpointing.metric_name=null
    # Only step 10 and the final step 12 are saved.
    checkpointing.save_period=10
    ++checkpointing.save_data_plane=true
)
# run1 is the driver's full smoke: 10 steps with its compaction, reward and KL checks.
SC_TEST_CONTEXT_COMPACTION=1 \
SC_TEST_EXP_NAME=${EXP_NAME}_run1 \
    bash $SCRIPT_DIR/grpo_async_gym_single_controller.sh \
    grpo.max_num_steps=10 "${CHECKPOINT_ARGS[@]}" "$@"
SC_TEST_CONTEXT_COMPACTION=1 \
SC_TEST_EXP_NAME=${EXP_NAME}_run2 \
RUN_CONVERGENCE_CHECKS=0 \
    bash $SCRIPT_DIR/grpo_async_gym_single_controller.sh \
    grpo.max_num_steps=12 "${CHECKPOINT_ARGS[@]}" "$@"

test -d $CKPT_DIR/step_10/data_plane
test -f $CKPT_DIR/step_10/replay_buffer_metadata.pt
grep -q "Restoring replay buffer metadata" $SCRIPT_DIR/${EXP_NAME}_run2/run.log
test -d $CKPT_DIR/step_12/policy/weights

cd $SCRIPT_DIR/../..
metrics=$EXP_DIR/metrics_run2.json
uv run tests/json_dump_tb_logs.py $SCRIPT_DIR/${EXP_NAME}_run2/logs --output_path $metrics
uv run tests/check_metrics.py $metrics \
    'max(data["train/finalize/total_ms"]) > 0' \
    'max(data["train/token_mult_prob_error"]) < 1.05' \
    'len(data["train/token_mult_prob_error"]) == 2'

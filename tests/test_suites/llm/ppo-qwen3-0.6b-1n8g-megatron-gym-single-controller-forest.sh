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
# Forest PPO (GAE, Megatron critic) on context-compaction rows through the
# SingleController path: two steps with checkpoints, then a second run resumes
# from step 2 to step 4, so Forest recovery is covered with a critic.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

# ===== BEGIN CONFIG =====
NUM_NODES=1
GPUS_PER_NODE=8
STEPS_PER_RUN=4
MAX_STEPS=4
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
# Both training runs (2 steps, then resume to 4) happen inside one job; each
# pays model load plus nemo_gym server startup.
NUM_MINUTES=60
# ===== END CONFIG =====

exit_if_max_steps_reached

cd $PROJECT_ROOT

# Prepare the nemo-gym workplace_assistant dataset.
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

# BrowseComp retains generated history while hiding older tool results.
TRAIN_PATH=$DATA_DIR/workplace_assistant_train.jsonl
VALIDATION_PATH=$DATA_DIR/workplace_assistant_validation.jsonl
DATA_FILTER='del(.task_source) | .agent_ref = {type: "responses_api_agents", name: "browsecomp_agent"}'
jq -c "$DATA_FILTER" 3rdparty/Gym-workspace/Gym/data/workplace_assistant/train.jsonl > $TRAIN_PATH
jq -c "$DATA_FILTER" 3rdparty/Gym-workspace/Gym/data/workplace_assistant/validation.jsonl > $VALIDATION_PATH

# Both runs start from this job's own checkpoints.
rm -rf $CKPT_DIR

for run_spec in "run1 2" "run2 4"; do
    read -r run steps <<< "$run_spec"
    uv run examples/run_grpo_single_controller.py \
        --config $CONFIG_PATH \
        ppo.max_num_steps=$steps \
        logger.log_dir=$LOG_DIR/$run \
        logger.wandb_enabled=True \
        logger.wandb.project=nemo-rl \
        logger.wandb.name=${EXP_NAME}_${run} \
        logger.monitor_gpus=True \
        logger.tensorboard_enabled=True \
        checkpointing.enabled=True \
        checkpointing.checkpoint_dir=$CKPT_DIR \
        checkpointing.metric_name=null \
        checkpointing.save_period=1 \
        ++checkpointing.save_data_plane=true \
        data.train.data_path=$TRAIN_PATH \
        data.validation.data_path=$VALIDATION_PATH \
        $@ \
        2>&1 | tee $EXP_DIR/${run}.log
    if [[ $run == run1 ]]; then
        test -d $CKPT_DIR/step_2/value/weights
        test -d $CKPT_DIR/step_2/data_plane
        test -f $CKPT_DIR/step_2/replay_buffer_metadata.pt
    fi
done

grep -q "Restoring replay buffer metadata" $EXP_DIR/run2.log
test -d $CKPT_DIR/step_4/policy/weights
test -d $CKPT_DIR/step_4/value/weights

# Step 1 trains only the critic (policy_training_start_step=1).
for run_spec in "run1 1" "run2 2"; do
    read -r run expected_policy_steps <<< "$run_spec"
    metrics=$EXP_DIR/metrics_${run}.json
    uv run tests/json_dump_tb_logs.py $LOG_DIR/$run --output_path $metrics
    uv run tests/check_metrics.py $metrics \
        'max(data["train/finalize/total_ms"]) > 0' \
        'len(data["train/critic/loss"]) == 2' \
        "len(data[\"train/loss\"]) == ${expected_policy_steps}" \
        'max(data["train/critic/loss"]) < 1.5' \
        'min(data["train/critic/loss"]) >= 0' \
        'max(data["train/token_mult_prob_error"]) < 1.05'
done

# Eight logical rollouts per step. More valid rows requires a context boundary;
# padding rows have zero sample_mask and cannot satisfy it. Whether the model
# reaches one is random, so any of the three policy steps across both runs counts.
jq -s '.[0] * .[1]' $EXP_DIR/metrics_run1.json $EXP_DIR/metrics_run2.json > $EXP_DIR/metrics_all.json
uv run tests/check_metrics.py $EXP_DIR/metrics_all.json \
    'max(data["train/global_valid_seqs"]) > 8'

# The resumed run's metrics end at MAX_STEPS, so a rerun of this job is skipped.
cp $EXP_DIR/metrics_run2.json $JSON_METRICS

# Clean up checkpoint directory after successful run to save space.
rm -rf "$CKPT_DIR"

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
# PPO counterpart of grpo_async_gym_single_controller_context_compaction.sh: the
# SingleController PPO path (GAE, Megatron critic) on Forest rows from context
# compaction. Runs two steps with checkpoints, then resumes from step 2 to step 4,
# so Forest recovery is covered with a critic.

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath $SCRIPT_DIR/../..)
# Mark the current repo as safe, since wandb fetches metadata about the repo
git config --global --add safe.directory $PROJECT_ROOT

set -eou pipefail

EXP_NAME=${SC_TEST_EXP_NAME:-$(basename $0 .sh)}
EXP_DIR=$SCRIPT_DIR/$EXP_NAME
CKPT_DIR=$EXP_DIR/checkpoints
DATA_DIR=$EXP_DIR/data

export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}

rm -rf $EXP_DIR
mkdir -p $EXP_DIR $DATA_DIR

cd $PROJECT_ROOT
# Follow nemo-gym instructions here to get this data:
# https://docs.nvidia.com/nemo/gym/0.1.0/tutorials/nemo-rl-grpo/setup.html#training-nemo-rl-grpo-setup
cd 3rdparty/Gym-workspace/Gym
# We need HF_TOKEN to download the data from huggingface
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
jq -c "$DATA_FILTER" 3rdparty/Gym-workspace/Gym/data/workplace_assistant/train.jsonl > "$TRAIN_PATH"
jq -c "$DATA_FILTER" 3rdparty/Gym-workspace/Gym/data/workplace_assistant/validation.jsonl > "$VALIDATION_PATH"

TRAIN_CMD=(
    uv run coverage run -a --data-file=$PROJECT_ROOT/tests/.coverage --source=$PROJECT_ROOT/nemo_rl
    $PROJECT_ROOT/examples/run_grpo_single_controller.py
    --config $PROJECT_ROOT/examples/nemo_gym/ppo_qwen3_0_6b_single_controller.yaml
    data.train.data_path=$TRAIN_PATH
    data.validation.data_path=$VALIDATION_PATH
    ++token_capture.enabled=true
    ++token_capture.forest_training=true
    ++async_rl.rollout_failure.min_step_batch_fraction=1
    policy.sequence_packing.enabled=false
    policy.dynamic_batching.enabled=false
    policy.logprob_batch_size=1
    policy.max_total_sequence_length=16384
    value.sequence_packing.enabled=false
    value.dynamic_batching.enabled=false
    value.train_micro_batch_size=2
    ++token_capture.forest_padding_scope=step
    ++policy.generation.vllm_cfg.http_server_serving_chat_kwargs.reasoning_parser=deepseek_r1
    env.nemo_gym.policy_model.responses_api_models.vllm_model.uses_reasoning_parser=true
    env.nemo_gym.policy_model.responses_api_models.vllm_model.extra_body.chat_template_kwargs.enable_thinking=true
    'env.nemo_gym.config_paths=[responses_api_models/vllm_model/configs/vllm_model_for_training.yaml,resources_servers/workplace_assistant/configs/workplace_assistant.yaml,responses_api_agents/browsecomp_agent/configs/browsecomp_agent.yaml]'
    ++env.nemo_gym.browsecomp_agent.responses_api_agents.browsecomp_agent.resources_server.name=workplace_assistant
    ++env.nemo_gym.browsecomp_agent.responses_api_agents.browsecomp_agent.max_steps=6
    ++env.nemo_gym.browsecomp_agent.responses_api_agents.browsecomp_agent.keep_rounds=1
    ++env.nemo_gym.browsecomp_agent.responses_api_agents.browsecomp_agent.max_context_tokens=0
    ++env.nemo_gym.browsecomp_agent.responses_api_agents.browsecomp_agent.nudge_steps=false
    ++env.nemo_gym.browsecomp_agent_environment_server.environment_servers.legacy_agent.entrypoint=app.py
    ++env.nemo_gym.browsecomp_agent_environment_server.environment_servers.legacy_agent.agent_server.type=responses_api_agents
    ++env.nemo_gym.browsecomp_agent_environment_server.environment_servers.legacy_agent.agent_server.name=browsecomp_agent
    ++async_rl.sampler.max_lookahead_versions=0
    ++async_rl.min_groups_for_streaming_train=4
    ++async_rl.max_inflight_prompts=4
    ++async_rl.max_buffered_rollouts=4
    ppo.policy_training_start_step=1
    logger.tensorboard_enabled=true
    logger.wandb_enabled=false
    logger.monitor_gpus=true
    checkpointing.enabled=true
    checkpointing.checkpoint_dir=$CKPT_DIR
    checkpointing.metric_name=null
    checkpointing.save_period=1
    ++checkpointing.save_data_plane=true
)

"${TRAIN_CMD[@]}" \
    ppo.max_num_steps=2 \
    logger.log_dir=$EXP_DIR/logs_run1 \
    "$@" \
    2>&1 | tee $EXP_DIR/run1.log

test -d $CKPT_DIR/step_2/value/weights
test -d $CKPT_DIR/step_2/data_plane
test -f $CKPT_DIR/step_2/replay_buffer_metadata.pt

"${TRAIN_CMD[@]}" \
    ppo.max_num_steps=4 \
    logger.log_dir=$EXP_DIR/logs_run2 \
    "$@" \
    2>&1 | tee $EXP_DIR/run2.log

grep -q "Restoring replay buffer metadata" $EXP_DIR/run2.log
test -d $CKPT_DIR/step_4/policy/weights
test -d $CKPT_DIR/step_4/value/weights

if [[ "${RUN_CONVERGENCE_CHECKS:-1}" == "1" ]]; then
    # Step 1 trains only the critic (policy_training_start_step=1).
    for run_spec in "run1 1" "run2 2"; do
        read -r run expected_policy_steps <<< "$run_spec"
        metrics=$EXP_DIR/metrics_${run}.json
        uv run tests/json_dump_tb_logs.py $EXP_DIR/logs_${run} --output_path $metrics
        # Eight logical rollouts per step. More valid rows requires a context
        # boundary; padding rows have zero sample_mask and cannot satisfy it.
        uv run tests/check_metrics.py $metrics \
            'max(data["train/finalize/total_ms"]) > 0' \
            'max(data["train/global_valid_seqs"]) > 8' \
            'len(data["train/critic/loss"]) == 2' \
            "len(data[\"train/loss\"]) == ${expected_policy_steps}" \
            'max(data["train/critic/loss"]) < 1.5' \
            'min(data["train/critic/loss"]) >= 0' \
            'max(data["train/token_mult_prob_error"]) < 1.05'
    done
fi

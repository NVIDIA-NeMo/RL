#!/bin/bash
# SingleController + NeMo-Gym e2e smoke. Mirrors grpo_async_gym.sh but
# routes everything through the SC path (TransferQueue data plane +
# SingleControllerActor) instead of async_grpo_train.

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath $SCRIPT_DIR/../..)
# Mark the current repo as safe, since wandb fetches metadata about the repo
git config --global --add safe.directory $PROJECT_ROOT

set -eou pipefail

EXP_NAME=${SC_TEST_EXP_NAME:-$(basename $0 .sh)}
EXP_DIR=$SCRIPT_DIR/$EXP_NAME
LOG_DIR=$EXP_DIR/logs
JSON_METRICS=$EXP_DIR/metrics.json
RUN_LOG=$EXP_DIR/run.log
CHECKPOINT_DIR=$EXP_DIR/checkpoints
DATA_DIR=$EXP_DIR/data
SC_ENTRYPOINT=${SC_TEST_ENTRYPOINT:-$PROJECT_ROOT/examples/run_grpo_single_controller.py}
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}

rm -rf $EXP_DIR $LOG_DIR
mkdir -p $EXP_DIR $LOG_DIR $CHECKPOINT_DIR $DATA_DIR

# clean up checkpoint directory on exit
trap "rm -rf $CHECKPOINT_DIR" EXIT

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

# The ordinary smoke uses a small context and only the first tool. The CC test
# retains the full tool set so the agent can execute tasks across model calls.
TRAIN_PATH=$DATA_DIR/workplace_assistant_train.jsonl
VALIDATION_PATH=$DATA_DIR/workplace_assistant_validation.jsonl
DATA_FILTER='.responses_create_params.tools |= (.[0:1])'
CC_OVERRIDES=()
if [[ "${SC_TEST_CONTEXT_COMPACTION:-0}" == "1" ]]; then
    # Keep accepted history while dropping earlier reasoning from the next input.
    DATA_FILTER='del(.task_source) | .agent_ref = {type: "responses_api_agents", name: "simple_agent_with_compaction"}'
    CC_OVERRIDES=(
        ++token_capture.enabled=true
        ++token_capture.context_compaction=true
        ++async_rl.rollout_failure.min_step_batch_fraction=1
        policy.sequence_packing.enabled=false
        policy.dynamic_batching.enabled=false
        policy.logprob_batch_size=1
        policy.max_total_sequence_length=16384
        grpo.calculate_advantages_on_gpu=false
        loss_fn.token_level_loss=true
        loss_fn.sequence_level_importance_ratios=false
        ++policy.generation.vllm_cfg.http_server_serving_chat_kwargs.reasoning_parser=deepseek_r1
        env.nemo_gym.policy_model.responses_api_models.vllm_model.uses_reasoning_parser=true
        env.nemo_gym.policy_model.responses_api_models.vllm_model.extra_body.chat_template_kwargs.enable_thinking=true
        'env.nemo_gym.config_paths=[responses_api_models/vllm_model/configs/vllm_model_for_training.yaml,resources_servers/workplace_assistant/configs/workplace_assistant.yaml,responses_api_agents/simple_agent_with_compaction/configs/simple_agent_with_compaction.yaml]'
        ++env.nemo_gym.simple_agent_with_compaction.responses_api_agents.simple_agent_with_compaction.resources_server.name=workplace_assistant
        ++env.nemo_gym.simple_agent_with_compaction.responses_api_agents.simple_agent_with_compaction.max_steps=6
        ++env.nemo_gym.simple_agent_with_compaction_environment_server.environment_servers.legacy_agent.entrypoint=app.py
        ++env.nemo_gym.simple_agent_with_compaction_environment_server.environment_servers.legacy_agent.agent_server.type=responses_api_agents
        ++env.nemo_gym.simple_agent_with_compaction_environment_server.environment_servers.legacy_agent.agent_server.name=simple_agent_with_compaction
        ++env.nemo_gym.simple_agent_with_compaction.responses_api_agents.simple_agent_with_compaction.context_history.policy.type=recency
        ++env.nemo_gym.simple_agent_with_compaction.responses_api_agents.simple_agent_with_compaction.context_history.policy.config.reasoning.enabled=true
        ++env.nemo_gym.simple_agent_with_compaction.responses_api_agents.simple_agent_with_compaction.context_history.policy.config.reasoning.keep_last_blocks=0
        ++env.nemo_gym.simple_agent_with_compaction.responses_api_agents.simple_agent_with_compaction.context_history.schedule.type=turn_chunked_recency
        ++env.nemo_gym.simple_agent_with_compaction.responses_api_agents.simple_agent_with_compaction.context_history.schedule.actions_per_chunk=1
    )
fi
jq -c "$DATA_FILTER" 3rdparty/Gym-workspace/Gym/data/workplace_assistant/train.jsonl > "$TRAIN_PATH"
jq -c "$DATA_FILTER" 3rdparty/Gym-workspace/Gym/data/workplace_assistant/validation.jsonl > "$VALIDATION_PATH"

uv run coverage run -a --data-file=$PROJECT_ROOT/tests/.coverage --source=$PROJECT_ROOT/nemo_rl \
    $SC_ENTRYPOINT \
    --config $PROJECT_ROOT/examples/nemo_gym/grpo_qwen3_30ba3b_instruct.yaml \
    policy.model_name=Qwen/Qwen3-0.6B \
    policy.automodel_cfg.enabled=false \
    policy.megatron_cfg.enabled=true \
    policy.megatron_cfg.tensor_model_parallel_size=1 \
    policy.megatron_cfg.pipeline_model_parallel_size=1 \
    policy.megatron_cfg.expert_model_parallel_size=1 \
    policy.megatron_cfg.context_parallel_size=1 \
    policy.megatron_cfg.sequence_parallel=false \
    policy.generation.vllm_cfg.tensor_parallel_size=1 \
    policy.generation.vllm_cfg.async_engine=true \
    policy.max_total_sequence_length=512 \
    policy.generation.colocated.enabled=false \
    policy.generation.colocated.resources.num_nodes=1 \
    policy.generation.colocated.resources.gpus_per_node=1 \
    grpo.num_prompts_per_step=4 \
    grpo.num_generations_per_prompt=2 \
    grpo.max_num_steps=10 \
    grpo.val_period=-1 \
    grpo.val_at_start=false \
    grpo.async_grpo=null \
    policy.train_global_batch_size=8 \
    policy.train_micro_batch_size=1 \
    cluster.gpus_per_node=2 \
    loss_fn.reference_policy_kl_penalty=0.01 \
    grpo.skip_reference_policy_logprobs_calculation=false \
    loss_fn.use_importance_sampling_correction=true \
    logger.tensorboard_enabled=true \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=false \
    logger.monitor_gpus=true \
    checkpointing.enabled=false \
    data.train.data_path=$TRAIN_PATH \
    data.validation.data_path=$VALIDATION_PATH \
    ++data_plane.enabled=true \
    ++data_plane.impl=transfer_queue \
    ++data_plane.backend=simple \
    ++data_plane.simple.storage_capacity=1000000 \
    ++data_plane.simple.num_storage_units=2 \
    ++data_plane.claim_meta_poll_interval_s=0.5 \
    ++async_rl.sampler.name=in_order \
    ++async_rl.sampler.max_lookahead_versions=0 \
    ++async_rl.min_groups_for_streaming_train=4 \
    ++async_rl.max_inflight_prompts=4 \
    ++async_rl.max_buffered_rollouts=4 \
    ${CC_OVERRIDES[@]+"${CC_OVERRIDES[@]}"} \
    "$@" \
    2>&1 | tee $RUN_LOG

if [[ "${RUN_CONVERGENCE_CHECKS:-1}" == "1" ]]; then
    uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

    EXTRA_CHECKS=()
    if [[ "$*" == *token_capture.enabled=true* ]]; then
        # Nonzero only when a finalizer actor ran, i.e. capture really was on.
        EXTRA_CHECKS+=('max(data["train/finalize/total_ms"]) > 0')
    fi

    if [[ "${SC_TEST_CONTEXT_COMPACTION:-0}" == "1" ]]; then
        # Eight logical rollouts per step. More valid rows requires a context
        # boundary; padding rows have zero sample_mask and cannot satisfy it.
        EXTRA_CHECKS+=(
            'max(data["train/finalize/total_ms"]) > 0'
            'max(data["train/global_valid_seqs"]) > 8'
            'max(data["train/gen_kl_error"]) < 0.05'
            'max(data["train/token_mult_prob_error"]) < 1.05'
            'len(data["train/token_mult_prob_error"]) == 10'
        )
    fi

    # Observed to be between 0.8-1.3
    uv run tests/check_metrics.py $JSON_METRICS \
        'median(data["train/gen_kl_error"]) < 1.3' \
        'max(data["train/reward"]) > 0' \
        ${EXTRA_CHECKS[@]+"${EXTRA_CHECKS[@]}"}
fi

#!/bin/bash
# Rollout-only simulator e2e on the SingleController + NeMo-Gym smoke recipe.
# Mirrors grpo_async_gym_single_controller.sh (same data, SC recipe and
# overrides) but drives examples/run_single_controller_simulator.py: real vLLM
# generation and Gym, a cpu train backend standing in for the trainer, and a
# no-op refit that still pauses and resumes generation on every step. The
# sampler runs one version ahead so the simulated train step overlaps real
# generation and the idle-skip accounting is exercised.

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath $SCRIPT_DIR/../..)
# Mark the current repo as safe, since wandb fetches metadata about the repo
git config --global --add safe.directory $PROJECT_ROOT

set -eou pipefail

EXP_NAME=$(basename $0 .sh)
EXP_DIR=$SCRIPT_DIR/$EXP_NAME
LOG_DIR=$EXP_DIR/logs
JSON_METRICS=$EXP_DIR/metrics.json
RUN_LOG=$EXP_DIR/run.log
DATA_DIR=$EXP_DIR/data
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}

rm -rf $EXP_DIR $LOG_DIR
mkdir -p $EXP_DIR $LOG_DIR $DATA_DIR

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

# Same trimming as the training smoke: with every tool attached the first prompt is >4000
# tokens, which vLLM answers with nothing at this max_total_sequence_length.
TRAIN_PATH=$DATA_DIR/workplace_assistant_train.jsonl
VALIDATION_PATH=$DATA_DIR/workplace_assistant_validation.jsonl
jq -c '.responses_create_params.tools |= (.[0:1])' 3rdparty/Gym-workspace/Gym/data/workplace_assistant/train.jsonl > $TRAIN_PATH
jq -c '.responses_create_params.tools |= (.[0:1])' 3rdparty/Gym-workspace/Gym/data/workplace_assistant/validation.jsonl > $VALIDATION_PATH

# Every node is an inference node: the recipe's 1-GPU generation split is all this run uses.
uv run coverage run -a --data-file=$PROJECT_ROOT/tests/.coverage --source=$PROJECT_ROOT/nemo_rl \
    $PROJECT_ROOT/examples/run_single_controller_simulator.py \
    --config $PROJECT_ROOT/examples/nemo_gym/grpo_qwen3_30ba3b_instruct.yaml \
    --num-steps 6 \
    --train-time-s 15 \
    --refit-time-s 3 \
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
    ++async_rl.sampler.max_lookahead_versions=1 \
    ++async_rl.min_groups_for_streaming_train=4 \
    ++async_rl.max_inflight_prompts=8 \
    ++async_rl.max_buffered_rollouts=8 \
    $@ \
    2>&1 | tee $RUN_LOG

if [[ "${RUN_CONVERGENCE_CHECKS:-1}" == "1" ]]; then
    uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

    # Liveness: the environment scored at least one rollout. Fidelity: the cpu trainer
    # publishes generation_logprobs as prev_logprobs, so the training/generation ratio is
    # exactly 1, and the held train step is reported with its idle-skipped part bounded
    # by the hold.
    uv run tests/check_metrics.py $JSON_METRICS \
        'max(data["train/reward"]) > 0' \
        'abs(max(data["train/max_seq_mult_prob_error"]) - 1.0) < 1e-6' \
        'min(data["train/simulated_train_time_s"]) >= 0' \
        'max(data["train/simulated_train_idle_skipped_s"]) <= 15'
fi

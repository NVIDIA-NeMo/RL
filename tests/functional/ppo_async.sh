#!/bin/bash
# Async PPO smoke test on a single 2-GPU node (1 train / 1 inference,
# non-colocated). policy_training_start_step=1 + max_num_steps=3 deliberately
# exercises the critic-warmup path (step 0), the warmup->training transition
# (step 1), and a normal async step (step 2) in one run.

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
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}

rm -rf $EXP_DIR $LOG_DIR
mkdir -p $EXP_DIR $LOG_DIR

cd $PROJECT_ROOT
uv run coverage run -a --data-file=$PROJECT_ROOT/tests/.coverage --source=$PROJECT_ROOT/nemo_rl \
    $PROJECT_ROOT/examples/run_ppo.py \
    policy.model_name=Qwen/Qwen2.5-0.5B \
    value.model_name=Qwen/Qwen2.5-0.5B \
    ppo.num_prompts_per_step=2 \
    ppo.num_generations_per_prompt=4 \
    ppo.ppo_epochs=2 \
    ppo.policy_training_start_step=1 \
    policy.train_global_batch_size=4 \
    policy.logprob_batch_size=4 \
    policy.train_micro_batch_size=1 \
    policy.generation.colocated.enabled=false \
    policy.generation.colocated.resources.gpus_per_node=1 \
    policy.generation.colocated.resources.num_nodes=1 \
    policy.generation.vllm_cfg.async_engine=true \
    ppo.async_ppo.enabled=true \
    ppo.async_ppo.max_trajectory_age_steps=1 \
    ppo.async_ppo.in_flight_weight_updates=false \
    loss_fn.use_importance_sampling_correction=true \
    value.train_global_batch_size=4 \
    value.train_micro_batch_size=1 \
    cluster.gpus_per_node=2 \
    ppo.max_num_steps=3 \
    logger.tensorboard_enabled=true \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=false \
    logger.monitor_gpus=true \
    checkpointing.enabled=false \
    $@ \
    2>&1 | tee $RUN_LOG

uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

uv run tests/check_metrics.py $JSON_METRICS \
    'max(data["train/token_mult_prob_error"]) < 1.05' \
    'max(data["train/critic/loss"]) < 6.0' \
    'min(data["train/critic/loss"]) >= 0' \
    'max(data["train/critic/explained_var"]) <= 1.0001' \
    'min(data["train/buffer_size"]) >= 0' \
    'max(data["train/avg_trajectory_age"]) <= 1.0'

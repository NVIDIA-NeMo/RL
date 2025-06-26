#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath $SCRIPT_DIR/../..)
# Mark the current repo as safe, since wandb fetchs metadata about the repo
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
uv run $PROJECT_ROOT/examples/run_grpo_sliding_puzzle.py \
    policy.model_name=Qwen/Qwen3-0.6B \
    cluster.gpus_per_node=2 \
    grpo.max_rollout_turns=5 \
    grpo.max_num_steps=3 \
    grpo.num_prompts_per_step=2 \
    grpo.num_generations_per_prompt=4 \
    policy.max_total_sequence_length=1024 \
    policy.train_global_batch_size=4 \
    policy.train_micro_batch_size=1 \
    policy.generation.top_p=0.99 \
    policy.generation.top_k=8000 \
    logger.tensorboard_enabled=true \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=false \
    checkpointing.enabled=false \
    $@ \
    2>&1 | tee $RUN_LOG

uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

uv run tests/check_metrics.py $JSON_METRICS \
    'max(data["train/token_mult_prob_error"]) < 1.1' \


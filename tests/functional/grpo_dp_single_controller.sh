#!/bin/bash
# Lightweight e2e for examples/run_grpo_single_controller.py — exercises
# setup_handle + setup_single_controller_component + SingleControllerActor
# end-to-end. Same shape as tests/functional/grpo_dp_simple.sh (Qwen3-0.6B,
# 2 GPUs, a handful of steps); data_plane.enabled=true is mandatory for SC.

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath $SCRIPT_DIR/../..)
# Mark the current repo as safe, since wandb fetches metadata about the repo
git config --global --add safe.directory $PROJECT_ROOT

set -eou pipefail

EXP_NAME=$(basename $0 .sh)
EXP_DIR=$SCRIPT_DIR/$EXP_NAME
LOG_DIR=$EXP_DIR/logs
RUN_LOG=$EXP_DIR/run.log
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}

rm -rf $EXP_DIR $LOG_DIR
mkdir -p $EXP_DIR $LOG_DIR

cd $PROJECT_ROOT
uv run coverage run -a --data-file=$PROJECT_ROOT/tests/.coverage --source=$PROJECT_ROOT/nemo_rl \
    $PROJECT_ROOT/examples/run_grpo_single_controller.py \
    policy.model_name=Qwen/Qwen3-0.6B \
    grpo.num_prompts_per_step=2 \
    grpo.num_generations_per_prompt=4 \
    policy.train_global_batch_size=4 \
    policy.train_micro_batch_size=1 \
    cluster.gpus_per_node=2 \
    grpo.max_num_steps=2 \
    logger.tensorboard_enabled=true \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=false \
    logger.monitor_gpus=false \
    checkpointing.enabled=false \
    data_plane.enabled=true \
    data_plane.impl=transfer_queue \
    data_plane.backend=simple \
    single_controller.min_prompt_groups_per_batch=2 \
    single_controller.target_prompt_groups_per_step=2 \
    single_controller.batch_selection_strategy=strict_on_policy \
    single_controller.max_inflight_prompts=4 \
    single_controller.max_buffered_rollouts=4 \
    $@ \
    2>&1 | tee $RUN_LOG

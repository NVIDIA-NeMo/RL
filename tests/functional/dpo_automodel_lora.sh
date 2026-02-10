#!/bin/bash

# clean up checkpoint directory on exit
trap "rm -rf /tmp/lora_dpo_checkpoints" EXIT

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
    $PROJECT_ROOT/examples/run_dpo.py \
    policy.model_name=Qwen/Qwen3-0.6B \
    policy.tokenizer.name=Qwen/Qwen3-0.6B \
    cluster.gpus_per_node=2 \
    dpo.max_num_steps=3 \
    dpo.val_batches=1 \
    dpo.val_global_batch_size=8 \
    ++policy.dtensor_cfg._v2=true \
    policy.train_global_batch_size=8 \
    policy.dtensor_cfg.lora_cfg.enabled=true \
    logger.tensorboard_enabled=true \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=false \
    logger.monitor_gpus=true \
    checkpointing.enabled=true \
    checkpointing.save_period=3 \
    checkpointing.checkpoint_dir=/tmp/lora_dpo_checkpoints \
    "$@" \
    2>&1 | tee $RUN_LOG

uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

uv run tests/check_metrics.py $JSON_METRICS \
  'data["train/loss"]["3"] < 0.8'


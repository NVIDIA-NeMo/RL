#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath $SCRIPT_DIR/../..)

set -eou pipefail

LOG_DIR=$SCRIPT_DIR/$(basename $0 .sh)-logs
JSON_METRICS=$LOG_DIR/$(basename $0 .sh).json
RUN_LOG=$LOG_DIR/$(basename $0 .sh).log
export RAY_DEDUP_LOGS=0
export UV_CACHE_DIR=$PROJECT_ROOT/uv_cache

mkdir -p $LOG_DIR

cd $PROJECT_ROOT
uv run $PROJECT_ROOT/examples/run_sft.py \
    cluster.gpus_per_node=2 \
    sft.max_num_steps=10 \
    logger.tensorboard_enabled=true \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=false \
    $@ \
    2>&1 | tee $RUN_LOG

cd $SCRIPT_DIR
uv run json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

uv run check_metrics.py $JSON_METRICS \
  'data["train/loss"]["9"] < 600' \
  'data["timing/train/sft_train_step"]["9"] < 0.25'


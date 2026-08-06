#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

# ===== BEGIN CONFIG =====
NUM_NODES=8
GPUS_PER_NODE=8
STEPS_PER_RUN=2
MAX_STEPS=2
NUM_RUNS=1
NUM_MINUTES=240
# ===== END CONFIG =====

exit_if_max_steps_reached

# This is initially tracked in disabled.txt while the native Automodel + vLLM
# 0.25.1 DeepSeek V4 correctness run is brought up on GB200 hardware.
cd $PROJECT_ROOT
uv run examples/run_grpo.py \
    --config $CONFIG_PATH \
    grpo.max_num_steps=$MAX_STEPS \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=True \
    logger.wandb.project=nemo-rl \
    logger.wandb.name=$EXP_NAME \
    logger.monitor_gpus=True \
    logger.tensorboard_enabled=True \
    checkpointing.enabled=False \
    "$@" \
    2>&1 | tee $RUN_LOG

uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

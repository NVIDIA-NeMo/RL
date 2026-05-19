#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

# CISPO-vs-GRPO A/B: treatment arm (CISPO loss with paper-default clip).
# See examples/configs/recipes/llm/cispo-ab-qwen2.5-math-1.5b-instruct-1n8g-cispo.yaml.
# NOT in the CISPO PR - local research-validation only.

# ===== BEGIN CONFIG =====
NUM_NODES=1
STEPS_PER_RUN=100
MAX_STEPS=100
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))
NUM_MINUTES=90
# ===== END CONFIG =====

exit_if_max_steps_reached

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
    $@ \
    2>&1 | tee $RUN_LOG

# Dump TB to JSON for offline A/B comparison.
uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

# NOTE: no `tests/check_metrics.py` thresholds here (see the grpo arm's .sh
# for rationale).

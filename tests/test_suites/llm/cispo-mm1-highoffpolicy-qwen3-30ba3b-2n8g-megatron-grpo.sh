#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

# MiniMax-M1 high-off-policy study, GRPO baseline arm (2n8g sized to match the
# proven SAPO sister recipe).
# See examples/configs/recipes/llm/cispo-mm1-highoffpolicy-qwen3-30ba3b-2n8g-megatron-grpo.yaml.

# ===== BEGIN CONFIG =====
NUM_NODES=2
STEPS_PER_RUN=100
MAX_STEPS=100
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))
NUM_MINUTES=$((24 * 60))
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
    logger.tensorboard_enabled=True \
    checkpointing.enabled=False \
    $@ \
    2>&1 | tee $RUN_LOG

uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

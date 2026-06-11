#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

# MiniMax-M1 async lag-1 high-off-policy study, CISPO arm.
# Uses 2 nodes for Megatron policy training plus 1 non-colocated vLLM node.
# See examples/configs/recipes/llm/grpo-cispo-mm1-async-lag1-highoffpolicy-qwen3-30ba3b-3n8g-megatron-cispo.yaml.

# ===== BEGIN CONFIG =====
NUM_NODES=3
STEPS_PER_RUN=10
MAX_STEPS=10
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))
NUM_MINUTES=60
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

# Only run metrics if the target step is reached
if [[ $(jq 'to_entries | .[] | select(.key == "train/loss") | .value | keys | map(tonumber) | max' $JSON_METRICS) -ge $MAX_STEPS ]]; then
    uv run tests/check_metrics.py $JSON_METRICS \
        'mean(data["train/reward"]) > 0.50' \
        'median(data["train/token_mult_prob_error"]) < 1.1' \
        "data[\"train/token_mult_prob_error\"][\"${MAX_STEPS}\"] < 1.1" \
        'mean(data["timing/train/total_step_time"], -6, -1) < 300'
fi

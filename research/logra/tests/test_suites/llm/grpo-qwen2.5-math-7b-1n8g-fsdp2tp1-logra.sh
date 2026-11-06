#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
NEMO_RL_ROOT=$(realpath $SCRIPT_DIR/../../../../..)

# Source common.env from the local test suite
source $SCRIPT_DIR/common.env

# ===== BEGIN CONFIG =====
NUM_NODES=1
GPUS_PER_NODE=8
STEPS_PER_RUN=100
MAX_STEPS=100
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
NUM_MINUTES=240
# ===== END CONFIG =====

exit_if_max_steps_reached

cd $PROJECT_ROOT
uv sync --locked --inexact --group test
uv run --locked run_grpo.py \
    --config configs/recipes/llm/grpo-qwen2.5-math-7b-1n8g-fsdp2tp1-logra.yaml \
    grpo.max_num_steps=$MAX_STEPS \
    logger.log_dir=$LOG_DIR \
    logger.tensorboard_enabled=True \
    checkpointing.checkpoint_dir=$CKPT_DIR \
    $@ \
    2>&1 | tee $RUN_LOG

uv run $NEMO_RL_ROOT/tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

# Only check metrics once the target step is reached
if [[ $(jq 'to_entries | .[] | select(.key == "train/loss") | .value | keys | map(tonumber) | max' $JSON_METRICS) -ge $MAX_STEPS ]]; then
    uv run $NEMO_RL_ROOT/tests/check_metrics.py $JSON_METRICS \
        'mean(data["train/reward"], -10, -1) > 0.60' \
        'median(data["train/token_mult_prob_error"]) < 1.1' \
        'max(data["train/memory/mean_peak_allocated_gib"]) < 30'
fi

#!/bin/bash
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/common.env"

# ===== BEGIN CONFIG =====
GPUS_PER_NODE=2
NUM_NODES=3
STEPS_PER_RUN=2
MAX_STEPS=2
NUM_RUNS=1
NUM_MINUTES=90
# ===== END CONFIG =====

exit_if_max_steps_reached

cd "$PROJECT_ROOT"
uv run examples/run_grpo_single_controller.py \
    --config "$CONFIG_PATH" \
    ppo.max_num_steps=$MAX_STEPS \
    logger.log_dir="$LOG_DIR" \
    logger.tensorboard_enabled=True \
    checkpointing.enabled=False \
    "$@" \
    2>&1 | tee "$RUN_LOG"

uv run tests/json_dump_tb_logs.py "$LOG_DIR" --output_path "$JSON_METRICS"
uv run tests/check_metrics.py "$JSON_METRICS" \
    'len(data["train/critic/loss"]) == 2' \
    'all_finite(data["train/critic/loss"])' \
    'all_finite(data["train/reward"])'

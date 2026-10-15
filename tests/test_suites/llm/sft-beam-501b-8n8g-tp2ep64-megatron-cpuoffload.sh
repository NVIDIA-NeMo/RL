#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source "$SCRIPT_DIR/common.env"

# ===== BEGIN CONFIG =====
NUM_NODES=8
STEPS_PER_RUN=20
MAX_STEPS=20
NUM_RUNS=1
NUM_MINUTES=60
# ===== END CONFIG =====

: "${BEAM_CHECKPOINT:?Set BEAM_CHECKPOINT to a BEAM Hugging Face checkpoint accessible on every worker}"

cd "$PROJECT_ROOT"
uv run examples/run_sft.py \
    --config "$CONFIG_PATH" \
    logger.log_dir="$LOG_DIR" \
    logger.tensorboard_enabled=True \
    logger.monitor_gpus=True \
    "$@" \
    sft.max_num_steps=$MAX_STEPS \
    checkpointing.enabled=False \
    2>&1 | tee "$RUN_LOG"

uv run tests/json_dump_tb_logs.py "$LOG_DIR" --output_path "$JSON_METRICS"
uv run tests/check_metrics.py "$JSON_METRICS" \
    'len(data["train/loss"]) == 20' \
    '"20" in data["train/loss"]' \
    '"20" in data["validation/val_loss"]' \
    'all_finite(data["train/loss"])' \
    'all_finite(data["train/grad_norm"])' \
    'all_finite(data["validation/val_loss"])'

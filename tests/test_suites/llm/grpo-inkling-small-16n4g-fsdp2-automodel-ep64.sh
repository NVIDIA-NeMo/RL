#!/bin/bash
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
source "$SCRIPT_DIR/common.env"

NUM_NODES=18
GPUS_PER_NODE=4
STEPS_PER_RUN=30
MAX_STEPS=30
NUM_RUNS=1
NUM_MINUTES=240

exit_if_max_steps_reached
cd "$PROJECT_ROOT"
uv run examples/run_grpo.py \
    --config "$CONFIG_PATH" \
    grpo.max_num_steps="$MAX_STEPS" \
    logger.log_dir="$LOG_DIR" \
    logger.wandb_enabled=True \
    logger.wandb.entity=nv-welcome \
    logger.wandb.project=nemo-rl-Inkling \
    logger.wandb.name="$EXP_NAME" \
    logger.tensorboard_enabled=True \
    checkpointing.enabled=True \
    checkpointing.checkpoint_dir="$CKPT_DIR" \
    "$@" 2>&1 | tee "$RUN_LOG"

uv run tests/json_dump_tb_logs.py "$LOG_DIR" --output_path "$JSON_METRICS"
uv run tests/check_metrics.py "$JSON_METRICS" \
    'len(data["train/loss"]) >= 30' \
    'median(data["train/token_mult_prob_error"]) < 1.1' \
    'mean(data["train/gen_kl_error"]) < 0.01'

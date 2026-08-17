#!/bin/bash
# One-step functional smoke for Qwen3.8-27B on the Megatron backend.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source "$SCRIPT_DIR/common.env"

# ===== BEGIN CONFIG =====
NUM_NODES=2
GPUS_PER_NODE=8
STEPS_PER_RUN=1
MAX_STEPS=1
NUM_RUNS=1
NUM_MINUTES=120
# ===== END CONFIG =====

exit_if_max_steps_reached

cd "$PROJECT_ROOT"
uv run examples/run_grpo.py \
    --config "$CONFIG_PATH" \
    grpo.max_num_steps=$MAX_STEPS \
    logger.log_dir="$LOG_DIR" \
    logger.wandb_enabled=True \
    logger.wandb.project=nemo-rl \
    logger.wandb.name="$EXP_NAME" \
    logger.monitor_gpus=True \
    logger.tensorboard_enabled=True \
    checkpointing.enabled=False \
    "$@" \
    2>&1 | tee "$RUN_LOG"

uv run tests/json_dump_tb_logs.py "$LOG_DIR" --output_path "$JSON_METRICS"

MAX_RECORDED_STEP=$(jq -r 'if has("train/loss") then (."train/loss" | keys | map(tonumber) | max // 0) else 0 end' "$JSON_METRICS")
if [[ $MAX_RECORDED_STEP -lt $MAX_STEPS ]]; then
    echo "[ERROR] Expected train/loss through step $MAX_STEPS, found step $MAX_RECORDED_STEP"
    exit 1
fi

uv run tests/check_metrics.py "$JSON_METRICS" \
    'abs(data["train/loss"]["1"]) < 100.0' \
    'data["train/num_valid_samples"]["1"] > 0' \
    'data["train/gen_kl_error"]["1"] < 0.1' \
    'data["train/token_mult_prob_error"]["1"] < 1.1'

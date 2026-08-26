#!/bin/bash
# Manual two-step check of the full 8-node TE NVFP4 recipe. The 4-node quick
# variant provides nightly coverage; the full training recipe remains 1500 steps.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source "$SCRIPT_DIR/common.env"

# ===== BEGIN CONFIG =====
NUM_NODES=8
GPUS_PER_NODE=4
STEPS_PER_RUN=2
MAX_STEPS=2
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))
NUM_MINUTES=240
SNAPSHOT_MEGATRON_BRIDGE=1
# ===== END CONFIG =====

exit_if_max_steps_reached

cd "$PROJECT_ROOT"
uv run --no-sync examples/run_grpo.py \
    --config "$CONFIG_PATH" \
    grpo.max_num_steps=$MAX_STEPS \
    logger.log_dir="$LOG_DIR" \
    logger.wandb_enabled=True \
    logger.wandb.project=nemo-rl \
    logger.wandb.name="$EXP_NAME" \
    logger.monitor_gpus=True \
    logger.tensorboard_enabled=True \
    checkpointing.enabled=False \
    checkpointing.checkpoint_dir="$CKPT_DIR" \
    "$@" \
    2>&1 | tee "$RUN_LOG"

uv run --no-sync tests/json_dump_tb_logs.py "$LOG_DIR" --output_path "$JSON_METRICS"

grep -q "\[nvfp4_pertoken\] per-token NVFP4 activation scaling active" "$RUN_LOG"
REFIT_COUNT=$(grep -F -c "[nvfp4_pertoken] refit: quantized" "$RUN_LOG" || true)
if [[ $REFIT_COUNT -lt 2 ]]; then
    echo "[ERROR] Expected at least two quantized refits, found $REFIT_COUNT"
    exit 1
fi

MAX_RECORDED_STEP=$(jq -r 'if has("train/loss") then (."train/loss" | keys | map(tonumber) | max // 0) else 0 end' "$JSON_METRICS")
if [[ $MAX_RECORDED_STEP -lt $MAX_STEPS ]]; then
    echo "[ERROR] Expected train/loss through step $MAX_STEPS, found step $MAX_RECORDED_STEP"
    exit 1
fi

uv run --no-sync tests/check_metrics.py "$JSON_METRICS" \
    'data["train/num_valid_samples"]["1"] > 0' \
    "abs(float(data[\"train/loss\"][\"$MAX_RECORDED_STEP\"])) < 1000000"

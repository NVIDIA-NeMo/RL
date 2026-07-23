#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source "$SCRIPT_DIR/common.env"

# ===== BEGIN CONFIG =====
NUM_NODES=2
GPUS_PER_NODE=4
STEPS_PER_RUN=3
MAX_STEPS=3
NUM_RUNS=1
NUM_MINUTES=60
# ===== END CONFIG =====

exit_if_max_steps_reached

cd "$PROJECT_ROOT"
uv run examples/run_grpo.py \
    --config "$CONFIG_PATH" \
    grpo.max_num_steps=$MAX_STEPS \
    logger.log_dir="$LOG_DIR" \
    logger.wandb_enabled=false \
    logger.tensorboard_enabled=true \
    checkpointing.enabled=false \
    "$@" \
    2>&1 | tee "$RUN_LOG"

uv run tests/json_dump_tb_logs.py "$LOG_DIR" --output_path "$JSON_METRICS"

MAX_RECORDED_STEP=$(jq -r \
    'if has("train/loss") then (."train/loss" | keys | map(tonumber) | max // 0) else 0 end' \
    "$JSON_METRICS")
if [[ $MAX_RECORDED_STEP -lt $MAX_STEPS ]]; then
    echo "[ERROR] Expected train/loss through step $MAX_STEPS, found step $MAX_RECORDED_STEP"
    exit 1
fi

GROUP_READY_COUNT=$(grep -c "NRL_SGLANG_REFIT_GROUP_READY" "$RUN_LOG" || true)
REFIT_SUCCESS_COUNT=$(grep -c "NRL_SGLANG_REFIT_SUCCESS" "$RUN_LOG" || true)
if [[ $GROUP_READY_COUNT -lt 1 ]]; then
    echo "[ERROR] SGLang refit group never became ready"
    exit 1
fi
if [[ $REFIT_SUCCESS_COUNT -lt 2 ]]; then
    echo "[ERROR] Expected at least two completed SGLang refits, found $REFIT_SUCCESS_COUNT"
    exit 1
fi

assert_not_grep \
    "NRL_SGLANG_REFIT_FAILURE" \
    "$RUN_LOG" \
    "SGLang reported a refit failure"

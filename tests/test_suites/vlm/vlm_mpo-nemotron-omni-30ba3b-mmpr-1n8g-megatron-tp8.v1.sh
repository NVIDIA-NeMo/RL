#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source "$SCRIPT_DIR/common.env"

# ===== BEGIN CONFIG =====
NUM_NODES=1
GPUS_PER_NODE=8
STEPS_PER_RUN=2
MAX_STEPS=2
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))
NUM_MINUTES=120
# ===== END CONFIG =====

exit_if_max_steps_reached
: "${MPO_DATA_PATH:?Set MPO_DATA_PATH to the MMPR meta-recipe JSON}"

cd "$PROJECT_ROOT"
uv run examples/run_vlm_mpo.py \
    --config "$CONFIG_PATH" \
    mpo.max_num_steps=$MAX_STEPS \
    data.train.data_path="$MPO_DATA_PATH" \
    logger.log_dir="$LOG_DIR" \
    logger.wandb_enabled=True \
    logger.wandb.project=nemo-rl \
    logger.wandb.name="$EXP_NAME" \
    logger.monitor_gpus=True \
    logger.tensorboard_enabled=True \
    checkpointing.enabled=True \
    checkpointing.checkpoint_dir="$CKPT_DIR" \
    "$@" \
    2>&1 | tee "$RUN_LOG"

uv run tests/json_dump_tb_logs.py "$LOG_DIR" --output_path "$JSON_METRICS"

if [[ $(jq 'to_entries | .[] | select(.key == "train/loss") | .value | keys | map(tonumber) | max' "$JSON_METRICS") -ge $MAX_STEPS ]]; then
    uv run tests/check_metrics.py "$JSON_METRICS" \
        'max(data["train/loss"]) < 1000000' \
        'min(data["train/loss"]) > -1000000'
    rm -rf "$CKPT_DIR"
fi

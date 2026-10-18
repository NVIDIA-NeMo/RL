#!/bin/bash
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/common.env"

# ===== BEGIN CONFIG =====
NUM_NODES=2
STEPS_PER_RUN=40
MAX_STEPS=40
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
NUM_MINUTES=40
# ===== END CONFIG =====

exit_if_max_steps_reached

cd "$PROJECT_ROOT"
uv run examples/run_grpo_single_controller.py \
    --config "$CONFIG_PATH" \
    ppo.max_num_steps=$MAX_STEPS \
    logger.log_dir="$LOG_DIR" \
    logger.wandb_enabled=True \
    logger.wandb.project=nemo-rl \
    logger.wandb.name="$EXP_NAME" \
    logger.monitor_gpus=True \
    logger.tensorboard_enabled=True \
    checkpointing.enabled=True \
    checkpointing.checkpoint_dir="$CKPT_DIR" \
    data_plane.observability.verify_tensor_hash=True \
    "$@" \
    2>&1 | tee "$RUN_LOG"

grep -q "Value init:" "$RUN_LOG"
grep -q "weight_sync=CollectiveWeightSynchronizer" "$RUN_LOG"
grep -qF "PPO: step 39 policy update/refit complete at version 40" "$RUN_LOG"
grep -qF "PPO: step 39 critic train: 256 samples," "$RUN_LOG"

uv run tests/json_dump_tb_logs.py "$LOG_DIR" --output_path "$JSON_METRICS"
# The inherited recipe's reward target was tuned for four policy epochs.
# This one-epoch streaming smoke checks completed updates, finite metrics,
# importance-sampling ratios and data-plane integrity.
uv run tests/check_metrics.py "$JSON_METRICS" \
    'len(data["train/loss"]) == 30' \
    'len(data["train/critic/loss"]) == 40' \
    'len(data["train/reward"]) == 40' \
    'all_finite(data["train/loss"])' \
    'all_finite(data["train/critic/loss"])' \
    'all_finite(data["train/reward"])' \
    'min(data["train/critic/loss"]) >= 0' \
    'median(data["train/token_mult_prob_error"]) < 1.1' \
    'data["train/token_mult_prob_error"]["40"] < 1.1' \
    'median(data["train/max_seq_mult_prob_error"]) < 1.2' \
    'max({**data.get("data_plane/cluster/step/hash/mismatches", {}), **data.get("data_plane/driver/step/hash/mismatches", {})}) == 0' \
    'max({**data.get("data_plane/cluster/step/hash/rows_checked", {}), **data.get("data_plane/driver/step/hash/rows_checked", {})}) > 0'

rm -rf "$CKPT_DIR"

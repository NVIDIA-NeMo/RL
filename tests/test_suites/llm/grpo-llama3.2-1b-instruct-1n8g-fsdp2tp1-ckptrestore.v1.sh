#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

# ===== BEGIN CONFIG =====
NUM_NODES=1
STEPS_PER_RUN=40
MAX_STEPS=40
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
NUM_MINUTES=60
# ===== END CONFIG =====

# First run: Train for 40 steps to create checkpoints
echo "[INFO] Starting first training run for $MAX_STEPS steps"
cd $PROJECT_ROOT
uv run examples/run_grpo_math.py \
    --config $CONFIG_PATH \
    grpo.max_num_steps=$MAX_STEPS \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=False \
    logger.wandb.project=nemo-rl \
    logger.wandb.name=${EXP_NAME}-first \
    logger.monitor_gpus=True \
    logger.tensorboard_enabled=True \
    checkpointing.enabled=True \
    checkpointing.checkpoint_dir=$CKPT_DIR \
    checkpointing.checkpoint_interval=20 \
    $@ \
    2>&1 | tee $RUN_LOG

# Convert tensorboard logs to json
uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

# Backup metrics.json from first run
echo "[INFO] Backing up metrics.json from first run"
cp $JSON_METRICS ${JSON_METRICS}.first_run

# Cleanup checkpoints except step 20
echo "[INFO] Cleaning up checkpoints, keeping only step 20"
find $CKPT_DIR -name "step_*" -not -name "step_20" -type d -exec rm -rf {} + 2>/dev/null || true
find $CKPT_DIR -name "step_*.pt" -not -name "step_20.pt" -exec rm -f {} + 2>/dev/null || true

# Verify step 20 checkpoint exists
if [[ ! -d "$CKPT_DIR/step_20" ]] && [[ ! -f "$CKPT_DIR/step_20.pt" ]]; then
    echo "[ERROR] Step 20 checkpoint not found after cleanup"
    exit 1
fi

echo "[INFO] Checkpoints remaining after cleanup:"
ls -la $CKPT_DIR/

# Clear tensorboard logs for second run
rm -rf $LOG_DIR/*

# Second run: Should restore from step 20 and continue to step 40
echo "[INFO] Starting second training run with checkpoint restore"
uv run examples/run_grpo_math.py \
    --config $CONFIG_PATH \
    grpo.max_num_steps=$MAX_STEPS \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=False \
    logger.wandb.project=nemo-rl \
    logger.wandb.name=${EXP_NAME}-restore \
    logger.monitor_gpus=True \
    logger.tensorboard_enabled=True \
    checkpointing.enabled=True \
    checkpointing.checkpoint_dir=$CKPT_DIR \
    checkpointing.checkpoint_interval=20 \
    $@ \
    2>&1 | tee -a $RUN_LOG

# Convert tensorboard logs to json for second run
uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

# Validation: Compare results between first and second run
echo "[INFO] Validating checkpoint restore functionality"

# Check that second run started from step 20
FIRST_STEP=$(jq 'to_entries | .[] | select(.key == "train/loss") | .value | keys | map(tonumber) | min' $JSON_METRICS)
if [[ $FIRST_STEP -ne 20 ]]; then
    echo "[ERROR] Second run did not start from step 20 (started from step $FIRST_STEP)"
    exit 1
fi

# Check that second run reached target steps
MAX_STEP=$(jq 'to_entries | .[] | select(.key == "train/loss") | .value | keys | map(tonumber) | max' $JSON_METRICS)
if [[ $MAX_STEP -lt $MAX_STEPS ]]; then
    echo "[ERROR] Second run did not reach target steps (reached $MAX_STEP, expected $MAX_STEPS)"
    exit 1
fi

# Compare loss values at step 40 between first and second run
FIRST_RUN_LOSS_40=$(jq '.["train/loss"]["40"]' ${JSON_METRICS}.first_run)
SECOND_RUN_LOSS_40=$(jq '.["train/loss"]["40"]' $JSON_METRICS)

echo "[INFO] Loss at step 40 - First run: $FIRST_RUN_LOSS_40, Second run: $SECOND_RUN_LOSS_40"

# Calculate relative difference (should be very small)
python3 -c "
import sys
first = float('$FIRST_RUN_LOSS_40')
second = float('$SECOND_RUN_LOSS_40')
rel_diff = abs(first - second) / abs(first) if first != 0 else abs(second)
print(f'Relative difference: {rel_diff:.6f}')
if rel_diff > 0.01:  # 1% tolerance
    print(f'ERROR: Loss difference too large: {rel_diff:.6f} > 0.01')
    sys.exit(1)
else:
    print('SUCCESS: Loss values match within tolerance')
"

echo "[INFO] Checkpoint restore test completed successfully"

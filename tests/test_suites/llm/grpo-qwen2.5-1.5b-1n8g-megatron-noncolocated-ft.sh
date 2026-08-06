#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

# ===== BEGIN CONFIG =====
NUM_NODES=1
STEPS_PER_RUN=20
MAX_STEPS=20
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))
NUM_MINUTES=60
# ===== END CONFIG =====

exit_if_max_steps_reached

# Run the experiment
cd $PROJECT_ROOT
uv run examples/run_grpo.py \
    --config $CONFIG_PATH \
    grpo.max_num_steps=$MAX_STEPS \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=True \
    logger.wandb.project=nemo-rl \
    logger.wandb.name=$EXP_NAME \
    logger.tensorboard_enabled=True \
    checkpointing.enabled=False \
    $@ \
    2>&1 | tee $RUN_LOG

# Convert tensorboard logs to json
uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

# Only run assertions if the target step is reached
if [[ $(jq 'to_entries | .[] | select(.key == "train/loss") | .value | keys | map(tonumber) | max' $JSON_METRICS) -ge $MAX_STEPS ]]; then
    # Assert that the killed shard was recovered: dp-2 should reach ready status.
    # dp-0 is killed at t=60s; dp-1 survives; recovery spawns dp-2.
    if ! grep -q "\[RECOVERY\] shard=dp-2 status=ready" $RUN_LOG; then
        echo "[ERROR] Expected fault recovery (dp-2 status=ready) not found in run log"
        exit 1
    fi

    # Assert no async training loop crash from the fault
    assert_not_grep "❌ Error in async loop" $RUN_LOG \
        "Async loop error detected — fault recovery did not complete cleanly"

    echo "[INFO] Fault-tolerant generation test passed: shard recovered and training completed"

    # Clean up logs after successful run to save space
    rm -rf "$CKPT_DIR"
fi

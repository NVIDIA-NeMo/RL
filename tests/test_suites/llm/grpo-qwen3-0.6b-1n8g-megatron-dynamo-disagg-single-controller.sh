#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

# ===== BEGIN CONFIG =====
NUM_NODES=1
STEPS_PER_RUN=30
MAX_STEPS=30
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
NUM_MINUTES=60
# ===== END CONFIG =====

exit_if_max_steps_reached

# Run the experiment
cd $PROJECT_ROOT
uv run examples/run_grpo_single_controller.py \
    --config $CONFIG_PATH \
    grpo.max_num_steps=$MAX_STEPS \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=True \
    logger.wandb.project=nemo-rl \
    logger.wandb.name=$EXP_NAME \
    logger.monitor_gpus=True \
    logger.tensorboard_enabled=True \
    checkpointing.enabled=False \
    checkpointing.checkpoint_dir=$CKPT_DIR \
    $@ \
    2>&1 | tee $RUN_LOG

# Both roles launched, and the frontend saw two of each before serving.
grep -F "'--disaggregation-mode', 'prefill'" $RUN_LOG
grep -F "'--disaggregation-mode', 'decode'" $RUN_LOG
grep -F "frontend ready with generation and RL workers {'prefill': 2, 'backend': 2}" $RUN_LOG
# Decode engines pulled prompt KV from prefill engines over NIXL.
grep -F "NIXL compatibility check passed" $RUN_LOG
# Every sync drains prefill before decode, so no handoff outlives its KV lease
# and no decode engine reads freed blocks.
if grep -E "Releasing expired KV blocks|NIXL transfer failure" $RUN_LOG; then
    echo "A prefill-to-decode KV handoff failed or outlived its lease" >&2
    exit 1
fi
sync_count=$(grep -Fc "_sync_weights: sync done" $RUN_LOG || true)
pause_count=$(grep -Fc "[RL] Engine paused (mode=wait, clear_cache=True)" $RUN_LOG || true)
if [[ "$sync_count" -lt 2 || "$pause_count" -lt $(( 4 * sync_count )) ]]; then
    echo "Expected every weight sync to pause all 4 engines; syncs=$sync_count, pauses=$pause_count" >&2
    exit 1
fi

# Convert tensorboard logs to json
uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

# Only run metrics if the target step is reached
if [[ $(jq 'to_entries | .[] | select(.key == "train/loss") | .value | keys | map(tonumber) | max' $JSON_METRICS) -ge $MAX_STEPS ]]; then
    uv run tests/check_metrics.py $JSON_METRICS \
        'max(data["train/token_mult_prob_error"]) < 1.05' \
        'max(data["train/gen_kl_error"]) < 0.002'

    # Clean up checkpoint directory after successful run to save space.
    rm -rf "$CKPT_DIR"
fi

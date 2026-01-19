#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

# ===== BEGIN CONFIG =====
NUM_NODES=2
GPUS_PER_NODE=4
STEPS_PER_RUN=30
MAX_STEPS=100
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
NUM_MINUTES=140
# ===== END CONFIG =====

exit_if_max_steps_reached

# Run the experiment
cd $PROJECT_ROOT
uv run examples/run_distillation_math.py \
    --config $CONFIG_PATH \
    distillation.max_num_steps=$MAX_STEPS \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=True \
    logger.wandb.project=nemo-rl \
    logger.wandb.name=$EXP_NAME \
    logger.monitor_gpus=True \
    logger.tensorboard_enabled=True \
    checkpointing.enabled=True \
    checkpointing.checkpoint_dir=$CKPT_DIR \
    $@ \
    2>&1 | tee $RUN_LOG

# Convert tensorboard logs to json
uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

# Only run metrics if the target step is reached
if [[ $(jq 'to_entries | .[] | select(.key == "train/loss") | .value | keys | map(tonumber) | max' $JSON_METRICS) -ge $MAX_STEPS ]]; then
    uv run tests/check_metrics.py $JSON_METRICS \
        'data["train/loss"]["1"] < 1.5' \
        'data["train/loss"]["100"] < 0.25' \
        'data["validation/accuracy"]["100"] > 0.2' \
        'mean(data["timing/train/total_step_time"], -6, -1) < 1600'

    # Clean up checkpoint directory after successful run to save space.
    rm -rf "$CKPT_DIR"
fi


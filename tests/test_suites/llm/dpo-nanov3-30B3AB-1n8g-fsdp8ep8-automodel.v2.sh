#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

# ===== BEGIN CONFIG =====
NUM_NODES=1
STEPS_PER_RUN=15
MAX_STEPS=15
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
NUM_MINUTES=30
# ===== END CONFIG =====

exit_if_max_steps_reached

# Run the EP8 experiment
cd $PROJECT_ROOT
uv run examples/run_dpo.py \
    --config $CONFIG_PATH \
    dpo.max_num_steps=$MAX_STEPS \
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
    # Step 1 runs before the first update, so policy == reference and both DPO
    # rewards are 0: loss = -log(sigmoid(0)) = ln 2. This holds for
    # preference_loss=dpo, preference_loss_weight=1 and sft_loss_weight=0, and
    # needs automodel_cfg.deterministic so the two forwards match bitwise.
    # loss[11] < 0.61 sits between the 95% and 99% one-sided prediction bounds
    # of 10 non-deterministic runs (mean 0.5768, sd 0.0124), so container or
    # kernel changes that move the deterministic trajectory still pass.
    uv run tests/check_metrics.py $JSON_METRICS \
        'abs(data["train/loss"]["1"] - 0.6931) < 0.0005' \
        'data["train/loss"]["11"] < 0.61' \
        'abs(data["train/preference_loss"]["1"] - 0.6931) < 0.0005' \
        'data["train/preference_loss"]["11"] < 0.61' \
        'mean(data["timing/train/total_step_time"], -5, -1) < 6.5'

    # Clean up checkpoint directory after successful run to save space.
    rm -rf "$CKPT_DIR"
fi

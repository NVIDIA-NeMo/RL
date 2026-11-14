#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

# ===== BEGIN CONFIG =====
NUM_NODES=2
STEPS_PER_RUN=100
MAX_STEPS=100
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
NUM_MINUTES=120
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

# Convert tensorboard logs to json
uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

# Only run metrics if the target step is reached
if [[ $(jq 'to_entries | .[] | select(.key == "train/loss") | .value | keys | map(tonumber) | max' $JSON_METRICS) -ge $MAX_STEPS ]]; then
    uv run tests/check_metrics.py $JSON_METRICS \
        "len(data[\"train/loss\"]) == $MAX_STEPS" \
        'all_finite(data["train/loss"])' \
        'all_finite(data["train/token_mult_prob_error"])' \
        'median(data["train/token_mult_prob_error"]) < 1.05' \
        'data["train/token_mult_prob_error"]["100"] < 1.05' \
        'mean(data["train/grad_norm"], 2, 0) > 0.06' \
        'median(data["train/unfiltered_reward"]) > 0.35' \
        'max(data["rollout/zero_variance_groups_total"]) >= 100' \
        'min(data["train/skipped_step"]) == max(data["train/skipped_step"]) == 0' \
        'min(data["train/dropped_prompt_groups"]) == max(data["train/dropped_prompt_groups"]) == 0' \
        'median(data["timing/train/total_step_time"]) < 180'

    # Clean up checkpoint directory after successful run to save space.
    rm -rf "$CKPT_DIR"
fi

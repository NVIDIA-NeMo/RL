#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

# ===== BEGIN CONFIG =====
NUM_NODES=32
GPUS_PER_NODE=4
STEPS_PER_RUN=60
MAX_STEPS=60
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
NUM_MINUTES=240
# ===== END CONFIG =====

exit_if_max_steps_reached

# Use the DeepSeek-V3 checkpoint converted to BF16.
if [[ -z "$NRL_DEEPSEEK_V3_BF16_CKPT" ]]; then
    echo "Need to set NRL_DEEPSEEK_V3_BF16_CKPT to the path of DeepSeek-V3 checkpoint converted to BF16. See docs/guides/deepseek.md for more details."
    exit 1
fi

# Run the experiment
cd $PROJECT_ROOT
uv run examples/run_grpo_math.py \
    --config $CONFIG_PATH \
    policy.model_name=$NRL_DEEPSEEK_V3_BF16_CKPT \
    grpo.max_num_steps=$MAX_STEPS \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=True \
    logger.wandb.project=nemo-rl \
    logger.wandb.name=$EXP_NAME \
    logger.monitor_gpus=True \
    logger.tensorboard_enabled=True \
    checkpointing.enabled=True \
    checkpointing.checkpoint_dir=$CKPT_DIR \
    "$@" \
    2>&1 | tee $RUN_LOG

# Convert tensorboard logs to json
uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

# Only run metrics if the target step is reached
if [[ $(jq 'to_entries | .[] | select(.key == "train/loss") | .value | keys | map(tonumber) | max' $JSON_METRICS) -ge $MAX_STEPS ]]; then
    uv run tests/check_metrics.py $JSON_METRICS \
        'mean(data["train/gen_kl_error"]) < 0.002' \
        'data["train/reward"]["60"] > 0.60' \
        'mean(data["timing/train/total_step_time"], -6, -1) < 210'

    # Clean up checkpoint directory after successful run to save space.
    rm -rf "$CKPT_DIR"
fi


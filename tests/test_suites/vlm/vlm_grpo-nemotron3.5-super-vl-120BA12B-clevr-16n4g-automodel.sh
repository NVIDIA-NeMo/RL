#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

# ===== BEGIN CONFIG =====
NUM_NODES=16
GPUS_PER_NODE=4
STEPS_PER_RUN=5
MAX_STEPS=5
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
NUM_MINUTES=180
# ===== END CONFIG =====

exit_if_max_steps_reached

# Run the experiment
# Nightly smoke: 5 image GRPO steps on 16 x 4-GPU GB200 nodes (~10 min/step
# plus ~10 min setup). CLEVR-CoGenT validation is skipped (val_at_start=false,
# val_period > MAX_STEPS): 5 steps is too few to move accuracy; the refit-health
# metrics below are what this test guards, including the vision-weight loading
# path (final vision LayerNorm + RADIO loader patches on stock vLLM).
cd $PROJECT_ROOT
uv run examples/run_vlm_grpo.py \
    --config $CONFIG_PATH \
    grpo.max_num_steps=$MAX_STEPS \
    grpo.val_at_start=False \
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
        'median(data["train/token_mult_prob_error"]) < 1.1' \
        "data['train/token_mult_prob_error']['$MAX_STEPS'] < 1.1" \
        'mean(data["train/gen_kl_error"]) < 0.02' \
        'max(data["train/reward"]) > 0.4'

    # Clean up checkpoint directory after successful run to save space.
    rm -rf "$CKPT_DIR"
fi

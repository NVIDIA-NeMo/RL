#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

# ===== BEGIN CONFIG =====
NUM_NODES=4
STEPS_PER_RUN=20
MAX_STEPS=20
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
NUM_MINUTES=240
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
    # With router replay the rollout routing is reproduced in training, so the logprob
    # mismatch stays low on every step; gen_kl_error separates it from the non-R3 run.
    uv run tests/check_metrics.py $JSON_METRICS \
        'data["validation/accuracy"]["20"] > 0.4' \
        'median(data["train/token_mult_prob_error"]) < 1.02' \
        'mean(data["train/gen_kl_error"]) < 0.0006'
    # R3 run: token_mult_prob_error 1.009-1.011 on every step, mean gen_kl_error 0.00037.
    # Without R3: median 1.016 with spikes of 1e6-3e9 on ~20% of steps, mean gen_kl_error
    # 0.00086-0.00087.

    # Clean up checkpoint directory after successful run to save space.
    rm -rf "$CKPT_DIR"
fi

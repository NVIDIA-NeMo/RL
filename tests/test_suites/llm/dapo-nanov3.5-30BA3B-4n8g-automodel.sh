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
    # Rare MoE routing mismatches can dominate the mean token_mult_prob_error.
    # Check the bulk error and cap severe outlier tokens at every training step.
    # Require both metrics at all 20 steps; missing or non-finite values must fail.
    uv run tests/check_metrics.py $JSON_METRICS \
        'data["validation/accuracy"]["20"] > 0.4' \
        'set(data["train/token_mult_prob_error_p999"]) == set(map(str, range(1, 21)))' \
        'set(data["train/num_tokens_logprob_error_above_10_nats"]) == set(map(str, range(1, 21)))' \
        'all_finite(data["train/token_mult_prob_error_p999"])' \
        'all_finite(data["train/num_tokens_logprob_error_above_10_nats"])' \
        'max(data["train/token_mult_prob_error_p999"]) < 1.55' \
        'max(data["train/num_tokens_logprob_error_above_10_nats"]) <= 3' \
        'mean(data["train/gen_kl_error"]) < 0.001'

    # Clean up checkpoint directory after successful run to save space.
    rm -rf "$CKPT_DIR"
fi

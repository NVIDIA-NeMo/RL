#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

# ===== BEGIN CONFIG =====
NUM_NODES=4
STEPS_PER_RUN=10
MAX_STEPS=10
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
NUM_MINUTES=120
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
    # Without router replay, MoE top-k flips between vLLM and training leave 0-2 tokens
    # per step tens of nats off, which spikes the token_mult_prob_error mean on ~20% of
    # steps (too often even for its median over 10 steps). Gate on the per-step p99.9
    # over tokens and allow a few such spike tokens instead. Steps 1-10 are LR warmup,
    # so accuracy only gets a floor that catches a broken run.
    uv run tests/check_metrics.py $JSON_METRICS \
        'data["validation/accuracy"]["10"] > 0.25' \
        'max(data["train/token_mult_prob_error_p999"]) < 1.55' \
        'max(data["train/num_tokens_logprob_error_above_10_nats"]) <= 3' \
        'mean(data["train/gen_kl_error"]) < 0.001'
    # Over steps 1-10 of 3 runs: validation accuracy 0.332 at step 0 and 0.344-0.352 at
    # step 10, token_mult_prob_error_p999 1.407-1.435, 0-2 tokens per step above 10 nats,
    # mean gen_kl_error 0.00082-0.00083.

    # Clean up checkpoint directory after successful run to save space.
    rm -rf "$CKPT_DIR"
fi

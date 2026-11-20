#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

# ===== BEGIN CONFIG =====
NUM_NODES=2
STEPS_PER_RUN=90
MAX_STEPS=90
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
NUM_MINUTES=30
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

# Every engine of a disaggregated replica starts in a PD mode (SGLang logs its
# ServerArgs); a run that silently came up aggregated must not pass the gates.
assert_grep "disaggregation_mode='prefill'" "$RUN_LOG" \
    "No prefill engine started: the rollouts ran aggregated, not disaggregated"
assert_grep "disaggregation_mode='decode'" "$RUN_LOG" \
    "No decode engine started: the rollouts ran aggregated, not disaggregated"

# Convert tensorboard logs to json
uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

# Only run metrics if the target step is reached
if [[ $(jq 'to_entries | .[] | select(.key == "train/loss") | .value | keys | map(tonumber) | max' $JSON_METRICS) -ge $MAX_STEPS ]]; then
    # token_mult_prob_error is the gate that matters here: the prefill->decode
    # KV handoff changes how the sampled tokens are produced, and a disagreement
    # with the trainer's recomputed logprobs shows up in it first.
    # gen_kl_error threshold: the 90-step PD and aggregated GB200 runs of this
    # recipe both averaged 1.3e-4 (max 1.6e-4); 1e-3 leaves ~8x headroom.
    uv run tests/check_metrics.py $JSON_METRICS \
        'median(data["train/token_mult_prob_error"]) < 1.1' \
        "data['train/token_mult_prob_error']['$MAX_STEPS'] < 1.1" \
        'mean(data["train/gen_kl_error"]) < 0.001' \
        'mean(data["timing/train/total_step_time"], 2) < 25'

    # Clean up checkpoint directory after successful run to save space.
    rm -rf "$CKPT_DIR"
fi

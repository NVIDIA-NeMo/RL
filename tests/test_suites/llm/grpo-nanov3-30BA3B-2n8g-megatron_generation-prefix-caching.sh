#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

# Nightly coverage for Megatron generation prefix caching.

# ===== BEGIN CONFIG =====
NUM_NODES=2
GPUS_PER_NODE=8
STEPS_PER_RUN=10
MAX_STEPS=10
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
NUM_MINUTES=60
# ===== END CONFIG =====

exit_if_max_steps_reached

# Run the experiment
cd $PROJECT_ROOT
uv run examples/run_grpo.py \
    --config $CONFIG_PATH \
    grpo.max_num_steps=$MAX_STEPS \
    data.default.prompt_file=$PROJECT_ROOT/tests/test_suites/fixtures/prefix_caching_cot_prompt.txt \
    policy.generation.backend=megatron \
    policy.generation.mcore_generation_config.enable_prefix_caching=true \
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
        'max(data["train/reward"]) > 0.0'

    # Clean up checkpoint directory after successful run to save space.
    rm -rf "$CKPT_DIR"
fi

# Every generated prompt shares a preamble longer than one 256-token KV block.
# Require repeated prefix-cache reuse rather than accepting a single smoke hit.
hits=$(
    grep -aoE 'mcore prefix cache \(cumul\): [0-9]+ hits' "$RUN_LOG" \
        | grep -oE '[0-9]+' \
        | awk 'END { print }' \
        || true
)
if [[ -n "$hits" && "$hits" -ge 10 ]]; then
    echo "PASS: prefix-cache hits (cumulative) = $hits"
else
    echo "FAIL: expected a cumulative prefix-cache hit count >= 10, got '${hits:-none}'."
    exit 1
fi

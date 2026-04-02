#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

# ===== BEGIN CONFIG =====
NUM_NODES=1
GPUS_PER_NODE=4
STEPS_PER_RUN=50
MAX_STEPS=50
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
NUM_MINUTES=180
# ===== END CONFIG =====

exit_if_max_steps_reached

POLICY_MODEL=${NRL_EAGLE3_POLICY_MODEL:-Qwen/Qwen3-1.7B}
DRAFT_MODEL=${NRL_EAGLE3_DRAFT_MODEL:-AngelSlim/Qwen3-1.7B_eagle3}

# Run the experiment
cd $PROJECT_ROOT
uv run examples/run_grpo.py \
    --config $CONFIG_PATH \
    policy.model_name="$POLICY_MODEL" \
    policy.tokenizer.name="$POLICY_MODEL" \
    policy.draft.model_name="$DRAFT_MODEL" \
    policy.generation.vllm_kwargs.speculative_config.model="$DRAFT_MODEL" \
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

if grep -q "Speculative decoding is enabled without draft refit sync" "$RUN_LOG"; then
    echo "Unexpected startup-weight warning for refit-backed Eagle3 path"
    exit 1
fi

grep -q "Draft Loss:" "$RUN_LOG"

# Convert tensorboard logs to json
uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

# Only run metrics if the target step is reached
if [[ $(jq 'to_entries | .[] | select(.key == "train/loss") | .value | keys | map(tonumber) | max' $JSON_METRICS) -ge $MAX_STEPS ]]; then
    uv run tests/check_metrics.py $JSON_METRICS \
        'min(data["train/draft_loss"]) > 0'

    # Clean up checkpoint directory after successful run to save space.
    rm -rf "$CKPT_DIR"
fi

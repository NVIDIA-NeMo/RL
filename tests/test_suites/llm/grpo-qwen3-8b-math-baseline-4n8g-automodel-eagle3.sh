#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

# ===== BEGIN CONFIG =====
NUM_NODES=4
GPUS_PER_NODE=8
STEPS_PER_RUN=30
MAX_STEPS=30
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))
NUM_MINUTES=240
# ===== END CONFIG =====

exit_if_max_steps_reached

cd $PROJECT_ROOT
uv run examples/run_grpo.py \
    --config $CONFIG_PATH \
    grpo.max_num_steps=$MAX_STEPS \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=true \
    logger.wandb.project=nemo-rl \
    logger.wandb.name=$EXP_NAME \
    logger.monitor_gpus=true \
    logger.tensorboard_enabled=true \
    checkpointing.enabled=true \
    checkpointing.checkpoint_dir=$CKPT_DIR \
    $@ \
    2>&1 | tee $RUN_LOG

if grep -q "Speculative decoding is enabled without draft refit sync" "$RUN_LOG"; then
    echo "Unexpected startup-weight warning for refit-backed EAGLE3 path"
    exit 1
fi

uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

if [[ $(jq 'to_entries | .[] | select(.key == "train/loss") | .value | keys | map(tonumber) | max' $JSON_METRICS) -ge $MAX_STEPS ]]; then
    # Metric scales anchored to the math-baseline 4n8g run: the TTT draft
    # loss (per-step soft-CE, summed over microbatches) started ~209 and
    # dropped below 40 by step 30; acceptance length at k=3 was stable at
    # 1.8-2.2 across refits from step 1.
    uv run tests/check_metrics.py $JSON_METRICS \
        'min(data["train/draft_loss"]) > 0' \
        'data["train/draft_loss"]["30"] < 150' \
        'data["train/vllm/spec_acceptance_length"]["30"] > 1.8'

    rm -rf "$CKPT_DIR"
fi

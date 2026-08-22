#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

# ===== BEGIN CONFIG =====
NUM_NODES=2
# 160 SC steps == the 160 optimizer steps the legacy async recipe takes in 40
# steps of 1024 prompts at gbs=256. SC maps one RL step to one optimizer step,
# so the same total data and the same number of updates need 4x the steps.
STEPS_PER_RUN=160
MAX_STEPS=160
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
NUM_MINUTES=90
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
    checkpointing.enabled=True \
    checkpointing.checkpoint_dir=$CKPT_DIR \
    $@ \
    2>&1 | tee $RUN_LOG

# The critic came up, GAE is driving the advantages, and the policy refits
# across clusters (non-colocated generation).
grep -q "Initializing value model for GAE" $RUN_LOG
grep -q "Using GAE advantage estimator" $RUN_LOG
grep -q "weight_sync=CollectiveWeightSynchronizer" $RUN_LOG

# Convert tensorboard logs to json
uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

# Only run metrics if the target step is reached. SC has no validation loop, so
# the convergence signal here is train reward rather than validation accuracy.
if [[ $(jq 'to_entries | .[] | select(.key == "train/loss") | .value | keys | map(tonumber) | max' $JSON_METRICS) -ge $MAX_STEPS ]]; then
    uv run tests/check_metrics.py $JSON_METRICS \
        'median(data["train/token_mult_prob_error"]) < 1.1' \
        'data["train/token_mult_prob_error"]["160"] < 1.1' \
        'median(data["train/max_seq_mult_prob_error"]) < 1.2' \
        'len(data["train/critic/loss"]) == 160' \
        'min(data["train/critic/loss"]) >= 0' \
        'data["train/reward"]["160"] > 0.75'

    # Clean up checkpoint directory after successful run to save space.
    rm -rf "$CKPT_DIR"
fi

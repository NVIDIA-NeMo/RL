#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

# ===== BEGIN CONFIG =====
NUM_NODES=1
STEPS_PER_RUN=30
MAX_STEPS=30
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
NUM_MINUTES=60
# ===== END CONFIG =====

exit_if_max_steps_reached

# PP requires automodel with update_seq_len (PR #1689).
# Install from submodule into the DTensor v2 worker venv if the container's version is older.
WORKER_VENV=/opt/ray_venvs/nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2
$WORKER_VENV/bin/python3 -c "from nemo_automodel.components.distributed.pipelining.autopipeline import AutoPipeline; assert hasattr(AutoPipeline, 'update_seq_len')" 2>/dev/null \
  || $WORKER_VENV/bin/pip install -e $PROJECT_ROOT/3rdparty/Automodel-workspace/Automodel --no-deps -q

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
# Slightly relaxed thresholds vs non-PP baseline
if [[ $(jq 'to_entries | .[] | select(.key == "train/loss") | .value | keys | map(tonumber) | max' $JSON_METRICS) -ge $MAX_STEPS ]]; then
    uv run tests/check_metrics.py $JSON_METRICS \
        'mean(data["train/gen_kl_error"]) < 0.001' \
        'data["train/gen_kl_error"]["30"] < 0.001 ' \
        'data["train/reward"]["30"] > 0.3' \
        'data["train/grad_norm"]["30"] < 0.25' \
        'data["train/grad_norm"]["30"] > 0.08'

    # Clean up checkpoint directory after successful run to save space.
    rm -rf "$CKPT_DIR"
fi

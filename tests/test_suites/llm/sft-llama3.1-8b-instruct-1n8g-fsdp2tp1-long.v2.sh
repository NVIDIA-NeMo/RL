#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

# TODO: @ashors real convergence run (dataset only has 2737)
# ===== BEGIN CONFIG =====
NUM_NODES=1
STEPS_PER_RUN=2730
MAX_STEPS=2730
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
NUM_MINUTES=120
# ===== END CONFIG =====

exit_if_max_steps_reached

# Run the experiment
cd $PROJECT_ROOT
uv run examples/run_sft.py \
    --config $CONFIG_PATH \
    sft.max_num_steps=$MAX_STEPS \
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

# TODO: the memory check is known to OOM. see https://github.com/NVIDIA/NeMo-RL/issues/263
# Only run metrics if the target step is reached
if [[ $(jq 'to_entries | .[] | select(.key == "train/loss") | .value | keys | map(tonumber) | max' $JSON_METRICS) -ge $MAX_STEPS ]]; then
    # TODO: FIGURE OUT CORRECT METRICS
    uv run tests/check_metrics.py $JSON_METRICS \
        'data["train/loss"]["1"] < 5' \
        'data["train/loss"]["2730"] < 0.3' \
        'max(data["ray/node.0.gpu.0.mem_gb"]) < 50'
fi

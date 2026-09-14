#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

# ===== BEGIN CONFIG =====
NUM_NODES=1
GPUS_PER_NODE=8
STEPS_PER_RUN=100
MAX_STEPS=100
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
NUM_MINUTES=90
# ===== END CONFIG =====

exit_if_max_steps_reached

# Run the experiment. The recipe keeps checkpointing disabled: a 30B-A3B save is
# ~42GB and this test only guards the Energon SFTv2 training path.
cd $PROJECT_ROOT
uv run examples/run_sft_v2.py \
    --config $CONFIG_PATH \
    sft.max_num_steps=$MAX_STEPS \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=True \
    logger.wandb.project=nemo-rl \
    logger.wandb.name=$EXP_NAME \
    logger.monitor_gpus=True \
    logger.tensorboard_enabled=True \
    $@ \
    2>&1 | tee $RUN_LOG

# Convert tensorboard logs to json
uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

# SFTv2 logs metrics without the "train/" prefix used by the v1 SFT entrypoint.
if [[ $(jq 'to_entries | .[] | select(.key == "loss") | .value | keys | map(tonumber) | max' $JSON_METRICS) -ge $MAX_STEPS ]]; then
    uv run tests/check_metrics.py $JSON_METRICS \
        'data["loss"]["1"] > 1.0' \
        'data["loss"]["100"] < 0.5' \
        'mean(data["total_step_time"], 2) < 40'
fi

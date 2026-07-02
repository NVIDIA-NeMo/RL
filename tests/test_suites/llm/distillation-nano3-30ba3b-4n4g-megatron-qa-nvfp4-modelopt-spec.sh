#!/bin/bash
# Nightly low-cost signal for the Nano3 hybrid MoE/Mamba ModelOpt layer-spec
# path. This one-step smoke deliberately keeps
# policy.disable_modelopt_layer_spec=false so the Megatron import path exercises
# modelopt_mamba_stack_spec instead of the faster standard Megatron stack spec.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

# ===== BEGIN CONFIG =====
NUM_NODES=4
GPUS_PER_NODE=4
STEPS_PER_RUN=1
MAX_STEPS=1
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
NUM_MINUTES=45
# ===== END CONFIG =====

exit_if_max_steps_reached

# Run the experiment
cd $PROJECT_ROOT
uv run examples/nemo_gym/run_distillation_nemo_gym.py \
    --config $CONFIG_PATH \
    distillation.max_num_steps=$MAX_STEPS \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=True \
    logger.wandb.project=nemo-rl \
    logger.wandb.name=$EXP_NAME \
    logger.monitor_gpus=True \
    logger.tensorboard_enabled=True \
    checkpointing.enabled=False \
    checkpointing.checkpoint_dir=$CKPT_DIR \
    $@ \
    2>&1 | tee $RUN_LOG

# Convert tensorboard logs to json
uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

MAX_RECORDED_STEP=$(jq -r 'if has("train/loss") then (."train/loss" | keys | map(tonumber) | max // 0) else 0 end' $JSON_METRICS)
if [[ $MAX_RECORDED_STEP -lt $MAX_STEPS ]]; then
    echo "[ERROR] Expected train/loss through step $MAX_STEPS, found step $MAX_RECORDED_STEP"
    exit 1
fi

uv run tests/check_metrics.py $JSON_METRICS \
    'data["train/loss"]["1"] > 0.0' \
    'data["train/loss"]["1"] < 0.12' \
    'data["train/mean_gen_tokens_per_sample"]["1"] > 100' \
    'data["train/mean_gen_tokens_per_sample"]["1"] < 2048'

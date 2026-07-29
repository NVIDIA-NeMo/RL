#!/bin/bash
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
source "$SCRIPT_DIR/common.env"

# ===== BEGIN CONFIG =====
NUM_NODES=6
GPUS_PER_NODE=4
STEPS_PER_RUN=4
MAX_STEPS=4
NUM_RUNS=1
NUM_MINUTES=240
USES_SANDBOX=1
USE_GYM_CONTAINER=true
# ===== END CONFIG =====

exit_if_max_steps_reached

: "${MODEL_PATH:?Set MODEL_PATH to the Nemotron Nano V3.5 checkpoint}"
: "${TRAIN_PATH:?Set TRAIN_PATH to the SWE training JSONL}"
: "${VAL_PATH:?Set VAL_PATH to the SWE validation JSONL}"
: "${SIF_FORMATTERS:?Set SIF_FORMATTERS to a JSON list of SWE image format strings}"

cd "$PROJECT_ROOT"

uv run examples/nemo_gym/run_grpo_nemo_gym.py \
    --config "$CONFIG_PATH" \
    grpo.max_num_steps=$MAX_STEPS \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=True \
    logger.wandb.project=nemo-rl \
    logger.wandb.name=$EXP_NAME \
    logger.monitor_gpus=True \
    checkpointing.enabled=True \
    checkpointing.checkpoint_dir=$CKPT_DIR \
    "$@" \
    2>&1 | tee "$RUN_LOG"

#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)

# ===== BEGIN CONFIG =====
SUITE=nightly
SKU=h100
# Mirrors grpo-qwen3-1.7b-1n8g-megatron-eagle3.sh (delegated base).
NUM_NODES=1
GPUS_PER_NODE=8
STEPS_PER_RUN=50
MAX_STEPS=50
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))
NUM_MINUTES=180
# ===== END CONFIG =====

source "${SCRIPT_DIR%%/tests/test_suites/*}/tests/test_suites/common-tq.env"
# Run base script under this wrapper's identity (own log/ckpt dirs, wandb name).
# The matching TQ YAML inherits from <base>.yaml and turns on data_plane.
export EXP_NAME="$TQ_EXP_NAME"
bash "$SCRIPT_DIR/$BASE_RECIPE.sh" "$@"

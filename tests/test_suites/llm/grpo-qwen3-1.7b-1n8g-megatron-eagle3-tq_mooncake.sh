#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)

# ===== BEGIN CONFIG =====
# Mirrors grpo-qwen3-1.7b-1n8g-megatron-eagle3.sh (delegated base).
NUM_NODES=1
GPUS_PER_NODE=8
STEPS_PER_RUN=50
MAX_STEPS=50
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))
NUM_MINUTES=180
# ===== END CONFIG =====

source "$SCRIPT_DIR/common-tq.env"
# Run base script under this wrapper's identity (own log/ckpt dirs, wandb name).
# The matching TQ YAML inherits from <base>.yaml and turns on data_plane.
export EXP_NAME="$TQ_EXP_NAME"
bash "$SCRIPT_DIR/$BASE_RECIPE.sh" "$@"

# The wire guard only counts; assert it found nothing.
cd "$SCRIPT_DIR/../../.."
uv run tests/check_metrics.py "$SCRIPT_DIR/$TQ_EXP_NAME/metrics.json" \
    'max(data["data_plane/cluster/step/hash/mismatches"]) == 0' \
    'max(data["data_plane/cluster/step/hash/guard_failures"]) == 0'

#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)

# ===== BEGIN CONFIG =====
# Mirrors grpo-qwen2.5-math-1.5b-instruct-1n8g-fsdp2tp1.v3.sh (delegated base).
NUM_NODES=1
STEPS_PER_RUN=450
MAX_STEPS=450
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
NUM_MINUTES=120
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

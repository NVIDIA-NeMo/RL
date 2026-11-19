#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)

# ===== BEGIN CONFIG =====
# Mirrors the MXFP8 rollout test driver (delegated base).
NUM_NODES=4
GPUS_PER_NODE=4
STEPS_PER_RUN=10
MAX_STEPS=10
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
NUM_MINUTES=120
# ===== END CONFIG =====

export EXP_NAME="$(basename "$0" .sh)"
bash "$SCRIPT_DIR/grpo-nanov3-30BA3B-4n4g-megatron_generation-noncolocated-mxfp8-rollouts.sh" "$@"

#!/bin/bash
export EXP_NAME="$(basename "$0" .sh)"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# ===== BEGIN CONFIG =====
NUM_NODES=2
STEPS_PER_RUN=40
MAX_STEPS=40
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))
NUM_MINUTES=40
# ===== END CONFIG =====

exec bash "$SCRIPT_DIR/ppo-qwen2.5-1.5b-gsm8k-2n8g-megatron-valuetp2sp-dynbatch-noncolocated-async-single-controller-streaming32.sh" "$@"

#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
export EXP_NAME=$(basename "$0" .sh)

# ===== BEGIN CONFIG =====
NUM_NODES=1
GPUS_PER_NODE=8
STEPS_PER_RUN=100
MAX_STEPS=100
NUM_RUNS=1
NUM_MINUTES=90
# ===== END CONFIG =====

source "$SCRIPT_DIR/vlm_sft-nemotron-omni-30ba3b-clevr-1n8g-megatron-tp8ep8-energon.v1.sh" "$@"

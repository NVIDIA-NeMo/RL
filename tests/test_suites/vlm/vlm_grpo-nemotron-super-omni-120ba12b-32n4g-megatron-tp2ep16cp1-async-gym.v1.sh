#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)

export EXP_NAME="$(basename "$0" .sh)"

# ===== BEGIN CONFIG =====
NUM_NODES=32
GPUS_PER_NODE=4
SEGMENT_SIZE=8
STEPS_PER_RUN=10
MAX_STEPS=50
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))
NUM_MINUTES=840
# ===== END CONFIG =====
export NUM_NODES GPUS_PER_NODE SEGMENT_SIZE

exec "$SCRIPT_DIR/vlm_grpo-nemotron-super-omni-120ba12b-16n8g-megatron-tp8ep16cp2-async-gym.v1.sh" "$@"

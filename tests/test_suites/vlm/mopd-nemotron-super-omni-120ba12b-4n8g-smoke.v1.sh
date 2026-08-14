#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)

export EXP_NAME="$(basename "$0" .sh)"

# ===== BEGIN CONFIG =====
NUM_NODES=4
GPUS_PER_NODE=8
STEPS_PER_RUN=3
MAX_STEPS=3
NUM_RUNS=1
NUM_MINUTES=480
# ===== END CONFIG =====
export NUM_NODES NUM_MINUTES

exec "$SCRIPT_DIR/mopd-nemotron-super-omni-120ba12b-10n8g-megatron-tp8ep16cp2-async-gym.v1.sh" "$@"

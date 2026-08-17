#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)

# ===== BEGIN CONFIG =====
# Mirrors grpo-nano-v2-12b-2n4g-fsdp2tp1.sh (delegated base).
# The only multi-node mooncake coverage on 4-GPU nodes: every other
# -tq_mooncake recipe is 8 GPUs per node, so nothing exercised the RDMA data
# plane across nodes on this topology, which is where each node must announce
# its own MC_TCP_BIND_ADDRESS rather than the driver's.
NUM_NODES=2
GPUS_PER_NODE=4
STEPS_PER_RUN=30
MAX_STEPS=30
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
NUM_MINUTES=60
# ===== END CONFIG =====

source "$SCRIPT_DIR/common-tq.env"
# Run base script under this wrapper's identity (own log/ckpt dirs, wandb name).
# The matching TQ YAML inherits from <base>.yaml and turns on data_plane.
export EXP_NAME="$TQ_EXP_NAME"
bash "$SCRIPT_DIR/$BASE_RECIPE.sh" "$@"

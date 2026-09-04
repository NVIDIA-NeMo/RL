#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)

# ===== BEGIN CONFIG =====
# Mirrors vlm_grpo-nemotron-omni-30ba3b-clevr-2n8g-megatron-tp8ep8.v1.sh, the
# delegated base, which common-tq.env derives by stripping -tq_mooncake.
NUM_NODES=2
GPUS_PER_NODE=8
STEPS_PER_RUN=10
MAX_STEPS=10
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
NUM_MINUTES=120
# ===== END CONFIG =====

source "$SCRIPT_DIR/common-tq.env"
# Run base script under this wrapper's identity (own log/ckpt dirs, wandb name).
# The matching TQ YAML inherits from <base>.yaml and turns on data_plane.
export EXP_NAME="$TQ_EXP_NAME"
bash "$SCRIPT_DIR/$BASE_RECIPE.sh" "$@"

# No TQ-specific gate: turning the data plane on must not change what the recipe
# is held to, so the logprob check lives on the base recipe, where both the
# data-plane and no-data-plane paths inherit it.

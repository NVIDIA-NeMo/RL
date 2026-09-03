#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)

# ===== BEGIN CONFIG =====
# Mirrors vlm_grpo-nemotron-omni-30ba3b-clevr-1n8g-megatron-tp8ep8.v1.sh (delegated
# base) except NUM_NODES -- see cluster.num_nodes in the matching yaml.
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
# is held to, so this wrapper passes exactly when the base recipe's own
# max(train/reward) > 0.5 passes.
#
# An earlier version added 'max(train/token_mult_prob_error) < 1.02', a bound
# taken from the automodel sibling (measured 1.0129-1.0155 there). It does not
# transfer to this recipe: the legacy control -- same node count, same sequence
# length, data_plane.enabled=false -- measured 1.035-1.063 across nine steps
# (job 17686235), so the gate would fail the no-data-plane path too. A check
# that the control cannot pass is testing the backend, not the data plane.

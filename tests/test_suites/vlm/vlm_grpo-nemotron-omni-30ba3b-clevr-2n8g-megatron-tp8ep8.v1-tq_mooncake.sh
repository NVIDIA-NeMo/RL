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

# TQ-owned logprob gate: the base recipe's gate runs under this wrapper too,
# but it is not a data-plane check -- it would pass or fail the same way with
# data_plane.enabled=false. This one covers the mooncake_cpu wire itself.
#
# median, not max or mean: a single long outlier sequence dominates both. On
# this recipe mooncake measured 1.031-1.041 over nine steps with one at 1.371,
# median ~1.033, and the no-data-plane control sits in the same band -- so 1.05
# passes both and still catches a wire that corrupts the logprob inputs.
source "$SCRIPT_DIR/common.env"
uv run tests/check_metrics.py "$JSON_METRICS" \
    'median(data["train/token_mult_prob_error"]) < 1.05'

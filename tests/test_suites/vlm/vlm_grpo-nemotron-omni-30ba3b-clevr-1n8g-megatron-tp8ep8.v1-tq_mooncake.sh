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

# TQ-specific gate, on top of the base recipe's own reward check.
#
# token_mult_prob_error compares rollout logprobs against prev_logprobs, both
# computed over the images, so it detects a wire round-trip that alters pixel
# data. Measured 1.0129-1.0155 on the automodel sibling across nine runs
# spanning the legacy (data_plane.enabled=false) path, both backends and four
# wire formats -- a 0.0026 spread, which is why the 1.02 bound is loose enough
# not to flake and tight enough to catch corruption.
#
# probs_ratio is NOT gated here: this recipe takes several inner steps per
# rollout, so the ratio measures policy drift rather than data fidelity. Its max
# ranged 2.32-29.21 across runs of identical code, with the no-data-plane
# control in the same spread.
source "$SCRIPT_DIR/common.env"
uv run tests/check_metrics.py "$JSON_METRICS" \
    'max(data["train/token_mult_prob_error"]) < 1.02'

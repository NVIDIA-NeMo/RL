#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)

# ===== BEGIN CONFIG =====
# Mirrors vlm_grpo-nemotron-omni-30ba3b-clevr-1n8g-automodel-ep8.v1.sh (delegated base).
NUM_NODES=1
STEPS_PER_RUN=10
MAX_STEPS=10
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
NUM_MINUTES=120
# ===== END CONFIG =====

source "$SCRIPT_DIR/common-tq.env"
export EXP_NAME="$TQ_EXP_NAME"
bash "$SCRIPT_DIR/$BASE_RECIPE.sh" "$@"

# TQ-specific gate, on top of the base recipe's own reward check.
#
# token_mult_prob_error compares rollout logprobs against prev_logprobs, both
# computed over the images, so it detects a wire round-trip that alters pixel
# data. Measured 1.0137-1.0155 across seven runs spanning the legacy
# (data_plane.enabled=false) path and three wire formats -- a 0.0018 spread,
# which is why the 1.02 bound is loose enough not to flake and tight enough to
# catch corruption.
#
# probs_ratio is NOT gated here: this recipe takes 16 inner steps per rollout,
# so the ratio measures policy drift rather than data fidelity. Its max ranges
# 5.85-29.21 across runs of identical code, and the no-data-plane control sits
# in the same spread.
source "$SCRIPT_DIR/common.env"
uv run tests/check_metrics.py "$JSON_METRICS" \
    'max(data["train/token_mult_prob_error"]) < 1.02'

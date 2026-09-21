#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)

# ===== BEGIN CONFIG =====
# Mirrors vlm_grpo-qwen3.5-35ba3b-geo3k-2n8g-automodel-ep16.sh (delegated base).
NUM_NODES=2
STEPS_PER_RUN=20
MAX_STEPS=20
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
# 150, not the base recipe's 240: a 20-step run of this recipe takes ~86 min.
NUM_MINUTES=150
# ===== END CONFIG =====

source "$SCRIPT_DIR/common-tq.env"
# Run base script under this wrapper's identity (own log/ckpt dirs, wandb name).
# The matching TQ YAML inherits from <base>.yaml and turns on data_plane.
export EXP_NAME="$TQ_EXP_NAME"
bash "$SCRIPT_DIR/$BASE_RECIPE.sh" "$@"

# TQ-specific gate, on top of the base recipe's own reward and
# median(train/token_mult_prob_error) checks.
#
# This recipe trains one inner step per rollout (train_global_batch_size ==
# num_prompts_per_step * num_generations_per_prompt), so the training forward
# runs on the very weights that produced prev_logprobs and the importance ratio
# is an identity. Any deviation means the two passes saw different data --
# exactly what a data-plane defect looks like. Measured 1.000000 on all 20
# steps across three wire formats.
#
# Deliberately NOT applied to recipes with >1 inner step: there the ratio
# measures policy drift, and its max swings 5.85-29.21 run to run on identical
# code (it does so on the non-data-plane path too).
#
# The logprob bound is a median, not a mean: on this recipe the per-step value
# is ~1.02 but spikes on a third of the steps (measured 20-step data-plane run:
# 5.05, 5.09, 4.13, 1.79, 1.34, 1.17), so the mean is 1.65 and a mean bound
# would have to sit above 2.0 to pass. The median is 1.026.
source "$SCRIPT_DIR/common.env"
uv run tests/check_metrics.py "$JSON_METRICS" \
    'max(data["train/probs_ratio_max"]) < 1.0001' \
    'min(data["train/probs_ratio_min"]) > 0.9999' \
    'median(data["train/token_mult_prob_error"]) < 1.05'

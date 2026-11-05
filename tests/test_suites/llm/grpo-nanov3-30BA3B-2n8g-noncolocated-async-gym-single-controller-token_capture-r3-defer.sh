#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

# ===== BEGIN CONFIG =====
NUM_NODES=2
GPUS_PER_NODE=8
STEPS_PER_RUN=3
MAX_STEPS=3
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
NUM_MINUTES=60
# ===== END CONFIG =====

# Same driver and gates as the token_capture nightly, run on this recipe.
export EXP_NAME
bash "$SCRIPT_DIR/grpo-nanov3-30BA3B-2n8g-megatron_generation-noncolocated-async-gym-single-controller-token_capture.sh" "$@"

# The deferred route path ran, covered every valid row, and never fell back.
cd $PROJECT_ROOT
uv run tests/check_metrics.py $JSON_METRICS \
    'min(data["train/finalize/route_plan_span_count"]) > 0' \
    'min(data["train/finalize/routed_experts_row_coverage"]) == 1' \
    'not any("routed_experts_deferred_fallback/" in k for k in data)'

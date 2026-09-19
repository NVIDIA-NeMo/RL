#!/bin/bash
# Full GenRM recipe; disabled in CI because it requires an external judge.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
source "$SCRIPT_DIR/common.env"

# ===== BEGIN CONFIG =====
NUM_NODES=1
STEPS_PER_RUN=300
MAX_STEPS=300
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))
NUM_MINUTES=1800
# ===== END CONFIG =====

exit_if_max_steps_reached
: "${GENRM_BASE_URL:?Set GENRM_BASE_URL to the external OCR judge endpoint}"
run_flow_grpo_recipe "$@"

uv run "$NEMO_RL_ROOT/tests/check_metrics.py" "$JSON_METRICS" \
    "median(data['train/mean_ratio']) > 0.5" \
    "median(data['train/mean_ratio']) < 1.5" \
    "max(data['train/grad_norm']) < 100"

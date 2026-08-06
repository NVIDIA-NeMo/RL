#!/bin/bash
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)
NEMO_RL_ROOT=$(realpath "${SCRIPT_DIR}/../../../../..")

source "${SCRIPT_DIR}/common.env"

# ===== BEGIN CONFIG =====
NUM_NODES=1
GPUS_PER_NODE=1
STEPS_PER_RUN=10
MAX_STEPS=10
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))
NUM_MINUTES=30
# ===== END CONFIG =====

cd "${PROJECT_ROOT}"
uv run run_grpo_turn_credit.py \
    --config configs/recipes/llm/grpo-qwen2.5-0.5b-instruct-1n1g-dtensor2tp1-turn-credit.yaml \
    "$@" 2>&1 | tee "${RUN_LOG}"

grep -q "Adding native turn-level credit" "${RUN_LOG}"
grep -q "Max number of steps has been reached" "${RUN_LOG}"
echo '{"succeed": "yes"}' >"${JSON_METRICS}"
uv run "${NEMO_RL_ROOT}/tests/check_metrics.py" \
    "${JSON_METRICS}" \
    'data["succeed"] == "yes"'

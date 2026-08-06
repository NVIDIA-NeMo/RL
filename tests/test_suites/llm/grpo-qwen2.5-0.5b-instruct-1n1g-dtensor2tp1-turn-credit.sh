#!/bin/bash
set -euo pipefail

# ===== BEGIN CONFIG =====
NUM_NODES=1
GPUS_PER_NODE=1
STEPS_PER_RUN=10
MAX_STEPS=10
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))
NUM_MINUTES=30
# ===== END CONFIG =====

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)
NEMO_RL_ROOT=$(realpath "${SCRIPT_DIR}/../../..")

exec bash \
    "${NEMO_RL_ROOT}/research/turn_level_credit/tests/test_suites/llm/grpo-qwen2.5-0.5b-instruct-1n1g-dtensor2tp1-turn-credit.sh" \
    "$@"

#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)
PROJECT_ROOT=$(realpath "${SCRIPT_DIR}/../..")
RUN_LOG=$(mktemp)
trap 'rm -f "${RUN_LOG}"' EXIT

cd "${PROJECT_ROOT}"
uv run run_grpo_turn_credit.py \
    --config configs/grpo_math_0.5b_turn_credit.yaml \
    grpo.max_num_steps=1 \
    checkpointing.enabled=false \
    "$@" 2>&1 | tee "${RUN_LOG}"

grep -q "Adding native turn-level credit" "${RUN_LOG}"
grep -q "Max number of steps has been reached" "${RUN_LOG}"

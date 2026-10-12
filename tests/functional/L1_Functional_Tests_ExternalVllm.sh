#!/bin/bash
# Separate serving environment: independent vLLM process, shared HF refit.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")
cd "$PROJECT_ROOT"

if [[ "${FAST:-0}" == 1 ]]; then
    echo "FAST: external vLLM E2E runs in the full functional tier"
    exit 0
fi

# vLLM 0.29.0 is independently resolved, keeping its dependencies separate
# from the training environment's pinned Megatron/Gym stack. Reuse an explicitly
# supplied serving environment when the runner already has one.
if [[ -z "${EXTERNAL_VLLM_TEST_PYTHON:-}" ]]; then
    EXTERNAL_VLLM_TEST_PYTHON=$(uv run --no-project --python 3.13 \
        --with vllm==0.29.0 python -c 'import sys; print(sys.executable)')
    export EXTERNAL_VLLM_TEST_PYTHON
fi
time uv run --no-sync bash "$SCRIPT_DIR/grpo_external_vllm_gym_single_controller.sh"

#!/bin/bash
# Separate serving environment: independent vLLM process, direct NCCL refit.
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
    # A Python path from `uv run --with` can belong to a temporary environment
    # removed when that command exits. Keep the server environment for the test.
    SERVING_VENV="${EXTERNAL_VLLM_TEST_RUN_DIR:-$SCRIPT_DIR/grpo_external_vllm_gym_single_controller}/serving-venv"
    if [[ ! -x "$SERVING_VENV/bin/python" ]]; then
        uv venv --no-project --python 3.13 "$SERVING_VENV"
    fi
    uv pip install --python "$SERVING_VENV/bin/python" 'vllm==0.29.0'
    EXTERNAL_VLLM_TEST_PYTHON="$SERVING_VENV/bin/python"
    export EXTERNAL_VLLM_TEST_PYTHON
fi
time uv run --no-sync bash "$SCRIPT_DIR/grpo_external_vllm_gym_single_controller.sh"
time uv run --no-sync bash "$SCRIPT_DIR/grpo_external_vllm_prefix_gym.sh"
time uv run --no-sync bash "$SCRIPT_DIR/grpo_external_vllm_capture_gym.sh"

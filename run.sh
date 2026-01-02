#!/bin/bash
set -e

VENV_NAME=".venv_test"
CONFIG_FILE="examples/configs/grpo_math_1B_sglang.yaml"

if [ -d "$VENV_NAME" ]; then
    echo "Removing existing virtual environment..."
    rm -rf "$VENV_NAME"
fi

uv venv "$VENV_NAME"
source "$VENV_NAME/bin/activate"
uv pip install -e ".[sglang]"

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"


python examples/run_grpo_math.py --config "$CONFIG_FILE"


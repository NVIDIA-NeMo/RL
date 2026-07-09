#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")

cd "$PROJECT_ROOT"
uv run --extra vllm coverage run -a --data-file="$PROJECT_ROOT/tests/.coverage" --source="$PROJECT_ROOT/nemo_rl" \
    tools/model_diagnostics/2.long_generation_decode_vs_prefill.py \
    --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16 \
    --prompts arc \
    --max-tokens 8192 \
    --num-batches 4 \
    --tensor-parallel-size 2 \

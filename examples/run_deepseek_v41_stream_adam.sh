#!/bin/bash
# Colocated DAPO recipe. Requires matching CUDA/AutoModel/vLLM dependencies.
set -euo pipefail
: "${DS41_FULL_CHECKPOINT:?Set a verified full checkpoint directory}"
cd "$(dirname "$0")/.."
exec uv run examples/run_grpo.py \
  --config examples/configs/recipes/llm/dapo-deepseek-v4.1-12n4g-fsdp2tp1-ep48-colocated-stream-adam.yaml "$@"

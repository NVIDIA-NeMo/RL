#!/bin/bash
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$PROJECT_ROOT"
uv sync --locked --inexact --group test
uv run --locked run_grpo.py --config configs/recipes/llm/grpo-qwen2.5-math-7b-1n8g-fsdp2tp1-logra.yaml "$@"

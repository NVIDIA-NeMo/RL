#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"
uv run --extra vllm python tests/functional/vllm_fp8_per_token_head.py "$@"

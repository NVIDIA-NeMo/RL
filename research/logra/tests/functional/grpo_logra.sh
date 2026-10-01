#!/bin/bash
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$PROJECT_ROOT"
uv run --locked run_grpo.py --config configs/grpo_logra_smoke.yaml "$@"

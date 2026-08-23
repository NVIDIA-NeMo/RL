#!/usr/bin/env bash
set -euo pipefail

readonly TESTS=(
  tests/unit/models/megatron/test_dflash_model.py
  tests/unit/models/megatron/test_dflash_asymmetric_tp_export.py
  tests/unit/models/megatron/test_dspark_training_provider.py
  tests/unit/test_qwen_moe_draft_linux_runner_contract.py
)

uv run --frozen --no-sync python -c 'import pytest, sys; print(f"LOCKED_PYTEST interpreter={sys.executable} prefix={sys.prefix} pytest={pytest.__file__}")'
exec uv run --frozen --no-sync python -m pytest -q "${TESTS[@]}" "$@"

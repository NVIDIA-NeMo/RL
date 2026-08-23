#!/usr/bin/env bash
set -euo pipefail

readonly MCORE_TESTS=(
  tests/unit/models/megatron/test_dflash_model.py
  tests/unit/models/megatron/test_dflash_asymmetric_tp_export.py
  tests/unit/models/megatron/test_dspark_training_provider.py
)
readonly CONTRACT_TEST=tests/unit/test_qwen_moe_draft_linux_runner_contract.py

uv run --frozen --no-sync python -c 'import pytest, sys; print(f"LOCKED_PYTEST interpreter={sys.executable} prefix={sys.prefix} pytest={pytest.__file__}")'
uv run --frozen --no-sync python -m pytest -q --mcore-only "${MCORE_TESTS[@]}" "$@"
exec uv run --frozen --no-sync python -m pytest -q "${CONTRACT_TEST}" "$@"

#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

readonly MCORE_TESTS=(
  tests/unit/models/megatron/test_dflash_model.py
  tests/unit/models/megatron/test_dflash_asymmetric_tp_export.py
  tests/unit/models/megatron/test_dspark_training_provider.py
)
readonly CONTRACT_TEST=tests/unit/test_qwen_moe_draft_linux_runner_contract.py

uv run --frozen --no-sync python -c 'import pytest, sys; print(f"LOCKED_PYTEST interpreter={sys.executable} prefix={sys.prefix} pytest={pytest.__file__}")'
uv run --frozen --no-sync python -m pytest -q --mcore-only "${MCORE_TESTS[@]}" "$@"
exec uv run --frozen --no-sync python -m pytest -q "${CONTRACT_TEST}"

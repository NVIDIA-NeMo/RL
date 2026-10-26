# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

#!/bin/bash
# Shard: three small suites that each spend far less time in pytest than in job
# setup, so they share one GPU runner instead of six:
#   1. Base model tests (formerly L0_Unit_Tests_Models_1..4)
#   2. vLLM + NeMo Gym tests (formerly L0_Unit_Tests_Vllm_Nemo_Gym)
#   3. TensorRT-LLM tests (formerly L0_Unit_Tests_Trtllm)
#
# Order matters: the base run uses the image's venv as shipped (--no-sync), and
# each later `uv run --extra ...` re-syncs the venv to that extra's set.
# Every suite runs even if an earlier one fails, so one failure doesn't hide
# another; the shard fails at the end if any suite failed.

source "$(dirname "${BASH_SOURCE[0]}")/run_unit_shard_common.sh"

failed=()

# 1. Base (unmarked) tests from every models/ subdirectory except generation/.
# Tests carrying the mcore or automodel marker are excluded by conftest.py filtering.
uv run --no-sync bash -x ./tests/run_unit.sh \
    "unit/models/" \
    "--ignore=unit/models/generation/" \
    "${EXCLUDED_UNIT_TESTS[@]}" \
    --cov=nemo_rl \
    --cov-report=term-missing \
    --cov-report=json \
    --hf-gated || failed+=("models")

# 2. Tests requiring both vLLM and NeMo Gym.
uv run --extra vllm --extra nemo_gym --group test bash -x ./tests/run_unit.sh \
    "unit/" \
    "${EXCLUDED_UNIT_TESTS[@]}" \
    --vllm-only \
    --nemo-gym-only \
    --cov=nemo_rl \
    --cov-append \
    --cov-report=term-missing \
    --cov-report=json || failed+=("vllm_nemo_gym")

# 3. TensorRT-LLM async-engine tests, including marked cross-cutting tests.
# The release image persists the build-time wheel at this path. Require it so
# this suite fails quickly instead of silently starting a source compilation.
TRTLLM_WHEEL_CACHE_DIR=/opt/trtllm_wheels \
TRTLLM_REQUIRE_CACHED_WHEEL=1 \
uv run --extra trtllm bash -x ./tests/run_unit.sh \
    "unit/" \
    "${EXCLUDED_UNIT_TESTS[@]}" \
    --cov=nemo_rl \
    --cov-append \
    --cov-report=term-missing \
    --cov-report=json \
    --hf-gated \
    --trtllm-only || failed+=("trtllm")

if [[ ${#failed[@]} -gt 0 ]]; then
    echo "[ERROR]: Failed suites in this shard: ${failed[*]}"
    exit 1
fi

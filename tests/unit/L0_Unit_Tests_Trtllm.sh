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
# Shard: TensorRT-LLM async-engine tests, including marked cross-cutting tests.

source "$(dirname "${BASH_SOURCE[0]}")/run_unit_shard_common.sh"

# The release image persists the build-time wheel at this path. Require it so
# this shard fails quickly instead of silently starting a source compilation.
TRTLLM_WHEEL_CACHE_DIR=/opt/trtllm_wheels \
TRTLLM_REQUIRE_CACHED_WHEEL=1 \
uv run --extra trtllm bash -x ./tests/run_unit.sh \
    "unit/" \
    "${EXCLUDED_UNIT_TESTS[@]}" \
    --cov=nemo_rl \
    --cov-report=term-missing \
    --cov-report=json \
    --hf-gated \
    --trtllm-only

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
# Shard: Model tests not covered by mcore/automodel/generation shards
# Picks up base (unmarked) tests from models/policy/, models/dtensor/, models/huggingface/
# Tests in models/megatron/ (all mcore) and models/automodel/ (all automodel) are excluded
# by conftest.py filtering since this is a base run.

source "$(dirname "${BASH_SOURCE[0]}")/run_unit_shard_common.sh"

uv run --no-sync bash -x ./tests/run_unit.sh "unit/models/" "--ignore=unit/models/generation/" "${EXCLUDED_UNIT_TESTS[@]}" --cov=nemo_rl --cov-report=term-missing --cov-report=json --hf-gated

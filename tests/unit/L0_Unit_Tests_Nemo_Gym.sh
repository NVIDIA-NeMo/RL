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
# Shard: All nemo_gym-marked tests anywhere in the codebase

source "$(dirname "${BASH_SOURCE[0]}")/run_unit_shard_common.sh"

uv run --extra nemo_gym bash -x ./tests/run_unit.sh "unit/" "${EXCLUDED_UNIT_TESTS[@]}" --cov=nemo_rl --cov-report=term-missing --cov-report=json --nemo-gym-only -vv

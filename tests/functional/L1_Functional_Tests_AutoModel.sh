# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
source "$(dirname "${BASH_SOURCE[0]}")/run_functional_shard_common.sh"

run_test      uv run --no-sync bash ./tests/functional/dpo_automodel_lora.sh
run_test      uv run --no-sync bash ./tests/functional/grpo_automodel_lora.sh
run_test      uv run --no-sync bash ./tests/functional/grpo_automodel_lora_async.sh
run_test      uv run --no-sync bash ./tests/functional/grpo_automodel_lora_non_colocated.sh
run_test      uv run --no-sync bash ./tests/functional/sft_automodel_lora.sh
run_test      uv run --no-sync bash ./tests/functional/test_automodel_extra_installed_correctly.sh

combine_functional_coverage

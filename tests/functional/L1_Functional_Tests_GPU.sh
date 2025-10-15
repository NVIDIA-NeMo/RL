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
set -xeuo pipefail # Exit immediately if a command exits with a non-zero status

cd /opt/nemo-rl
time uv run --no-sync bash ./tests/functional/sft.sh
time uv run --no-sync bash ./tests/functional/grpo.sh
time uv run --no-sync bash ./tests/functional/grpo_async.sh
time uv run --no-sync bash ./tests/functional/grpo_megatron.sh
time uv run --no-sync bash ./tests/functional/grpo_multiturn.sh
time uv run --no-sync bash ./tests/functional/grpo_non_colocated.sh
time uv run --no-sync bash ./tests/functional/dpo.sh
time uv run --no-sync bash ./tests/functional/rm.sh
time uv run --no-sync bash ./tests/functional/eval.sh
time uv run --no-sync bash ./tests/functional/eval_async.sh
time uv run --no-sync bash ./tests/functional/test_mcore_extra_installed_correctly.sh
time uv run --no-sync bash ./tests/functional/test_automodel_extra_installed_correctly.sh
time uv run --no-sync bash ./tests/functional/vlm_grpo.sh
time uv run --no-sync bash ./tests/functional/distillation.sh

# Research functional tests (self-discovery)
for test_script in research/*/tests/functional/*.sh; do
    project_dir=$(dirname $(dirname $test_script))
    pushd $project_dir
    time uv run --no-sync bash $test_script
    popd
done

cd /opt/nemo-rl/tests
coverage combine .coverage*

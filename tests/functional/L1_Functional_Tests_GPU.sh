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

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT=$(realpath ${SCRIPT_DIR}/../..)

cd ${PROJECT_ROOT}
# This test is intentionally not run with uv run --no-sync to verify that the frozen environment is working correctly.
time bash ./tests/functional/grpo_frozen_env.sh
time bash ./tests/functional/test_frozen_env.sh

time uv run --no-sync bash ./tests/functional/sft.sh
time uv run --no-sync bash ./tests/functional/sft_resume_diamond.sh
time uv run --no-sync bash ./tests/functional/grpo.sh
time uv run --no-sync bash ./tests/functional/grpo_async.sh
time uv run --no-sync bash ./tests/functional/grpo_megatron.sh
time uv run --no-sync bash ./tests/functional/grpo_megatron_generation.sh
time uv run --no-sync bash ./tests/functional/grpo_multiturn.sh
time uv run --no-sync bash ./tests/functional/grpo_non_colocated.sh
time uv run --no-sync bash ./tests/functional/grpo_sglang.sh
time uv run --no-sync bash ./tests/functional/dpo_automodel_lora.sh
time uv run --no-sync bash ./tests/functional/dpo.sh
time uv run --no-sync bash ./tests/functional/rm.sh
time uv run --no-sync bash ./tests/functional/eval.sh
time uv run --no-sync bash ./tests/functional/eval_async.sh
time uv run --no-sync bash ./tests/functional/test_mcore_extra_installed_correctly.sh
time uv run --no-sync bash ./tests/functional/test_automodel_extra_installed_correctly.sh
time uv run --no-sync bash ./tests/functional/vlm_grpo.sh
time uv run --no-sync bash ./tests/functional/distillation.sh
time uv run --no-sync bash ./tests/functional/distillation_megatron.sh
time uv run --no-sync bash ./tests/functional/sft_automodel_lora.sh
time uv run --no-sync bash ./tests/functional/sft_megatron_lora.sh

# Research functional tests (self-discovery)
for test_script in research/*/tests/functional/*.sh; do
    project_dir=$(echo $test_script | cut -d/ -f1-2)
    pushd $project_dir
    time uv run --no-sync bash $(echo $test_script | cut -d/ -f3-)
    popd
done

cd ${PROJECT_ROOT}/tests
coverage combine .coverage*

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
source "$(dirname "${BASH_SOURCE[0]}")/run_functional_shard_common.sh"

# This test is intentionally not run with uv run --no-sync to verify that the frozen environment is working correctly.
run_test      bash ./tests/functional/test_frozen_env.sh

run_test fast uv run --no-sync bash ./tests/functional/test_converters.sh
run_test      uv run --no-sync bash ./tests/functional/test_decode_vs_prefill.sh
run_test      uv run --no-sync bash ./tests/functional/test_mcore_extra_installed_correctly.sh

# Research functional tests (self-discovery)
if [[ "${FAST:-0}" != "1" ]]; then
    for test_script in research/*/tests/functional/*.sh; do
        project_dir=$(echo $test_script | cut -d/ -f1-2)
        pushd $project_dir
        time uv run --no-sync bash $(echo $test_script | cut -d/ -f3-)
        popd
    done
fi

combine_functional_coverage

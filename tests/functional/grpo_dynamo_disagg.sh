#!/bin/bash
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

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
project_root=$(realpath "${script_dir}/../..")

export DYNAMO_EXP_NAME=grpo_dynamo_disagg
export DYNAMO_CONFIG_PATH="${project_root}/examples/configs/grpo_math_1B_dynamo_disagg.yaml"
export DYNAMO_EXPECT_DISAGG=1

exec bash "${script_dir}/grpo_dynamo.sh"

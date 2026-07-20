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
set -xeuo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT=$(realpath ${SCRIPT_DIR}/../..)

cd ${PROJECT_ROOT}

# All TRT-LLM tests use the async engine. These scenarios deliberately cover
# the supported axes without creating a full backend/deployment-mode matrix.
time uv run --no-sync bash ./tests/functional/grpo_trtllm_fsdp2_colocated.sh
time uv run --no-sync bash ./tests/functional/grpo_trtllm_mcore_colocated.sh
time uv run --no-sync bash ./tests/functional/grpo_trtllm_mcore_non_colocated_async.sh

cd ${PROJECT_ROOT}/tests
if compgen -G ".coverage*" > /dev/null; then
    coverage combine .coverage*
fi

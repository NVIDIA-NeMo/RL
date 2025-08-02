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
uv run --no-sync bash -x ./tests/run_unit.sh unit/models/policy/ --cov=nemo_rl --cov-report=term-missing --cov-report=json --hf-gated

exit_code=$(pytest tests/unit/models/policy/ --collect-only --hf-gated --mcore-only -q >/dev/null 2>&1; echo $?)
if [[ $exit_code -eq 5 ]]; then
    echo "No mcore tests to run"
    exit 0
else
    uv run --extra mcore bash -x ./tests/run_unit.sh unit/models/policy/ --cov=nemo_rl --cov-append --cov-report=term-missing --cov-report=json --hf-gated --mcore-only
fi

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

# Common boilerplate for unit test shard scripts.
# Source this file at the top of each L0_Unit_Tests_*.sh shard script.
# It sets up: SCRIPT_DIR, PROJECT_ROOT, FAST exclusions, and test assets.

set -xeuo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT=$(realpath ${SCRIPT_DIR}/../..)

cd ${PROJECT_ROOT}

# Source exclusion list for FAST mode
EXCLUDED_UNIT_TESTS=()
if [[ "${FAST:-0}" == "1" ]]; then
    source ${SCRIPT_DIR}/excluded_unit_tests.sh
fi

uv run tests/unit/prepare_unit_test_assets.py

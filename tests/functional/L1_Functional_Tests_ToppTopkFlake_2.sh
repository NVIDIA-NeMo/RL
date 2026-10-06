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
# TEMP, DO NOT MERGE: one shard of the top-p/top-k flake hunt.
set -xeuo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT=$(realpath ${SCRIPT_DIR}/../..)
cd ${PROJECT_ROOT}

SHARD=$(basename "$0" .sh)
nvidia-smi --query-gpu=name --format=csv,noheader | head -1
uv run --no-sync bash ./tests/functional/topp_topk_flake_loop.sh 5 "${SHARD##*_}"

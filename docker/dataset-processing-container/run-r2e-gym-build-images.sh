#!/usr/bin/env bash
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

STATE_DIR=${STATE_DIR:-/workspace/state}
MAX_WORKERS=${MAX_WORKERS:-1}
INSTANCE_FILE="${1:-/workspace/r2e-gym-instances-to-build.txt}"

cd /workspace/repos/r2e-gym-arm-build
uv run --with datasets python src/r2egym/repo_analysis/build_arm64_dockers.py \
    --instance-file "${INSTANCE_FILE}" \
    --state-file "${STATE_DIR}/r2e_gym_build_push_state.json" \
    --max-workers "${MAX_WORKERS}" \
    --cleanup-local \
    --push \
    --retry-failed \
    --registry "${DOCKER_REGISTRY}/r2e-gym"

#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Opt-in entrypoint for using vllm-project/router on the rollout generation
# path. The default run_in_allocation.sh entrypoint remains the original
# Python-proxy deployment.

set -euo pipefail

: "${EXTERNAL_VLLM_TOOLS_DIR_HOST:?EXTERNAL_VLLM_TOOLS_DIR_HOST is required}"
export EXTERNAL_VLLM_ROUTER_POOL="${EXTERNAL_VLLM_ROUTER_POOL:-ROLLOUT}"
exec "${EXTERNAL_VLLM_TOOLS_DIR_HOST}/run_in_allocation.sh"

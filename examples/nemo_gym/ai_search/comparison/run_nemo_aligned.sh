#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

COMPARISON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_DIR="$(cd "${COMPARISON_DIR}/.." && pwd)"
CONFIG_PATH="${PLUGIN_DIR}/grpo_qwen2_5_3b_aligned_benchmark.yaml"

if ! curl --fail --silent --show-error http://127.0.0.1:8100/health >/dev/null; then
  echo "The shared retrieval server is not healthy on port 8100." >&2
  exit 1
fi

export AI_SEARCH_CONFIG="${CONFIG_PATH}"
export AI_SEARCH_MAX_STEPS="${AI_SEARCH_MAX_STEPS:-4}"
export AI_SEARCH_NUM_PROMPTS="${AI_SEARCH_NUM_PROMPTS:-4}"
export AI_SEARCH_NUM_GENERATIONS="${AI_SEARCH_NUM_GENERATIONS:-4}"
# The dashboard is unrelated to training and can fail independently when its
# metrics subprocess is unavailable on a batch node.
export NEMO_RL_RAY_DASHBOARD=0

exec bash "${PLUGIN_DIR}/run_ai_search.sh" "$@"

#!/bin/bash
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

# ===== BEGIN CONFIG =====
NUM_NODES=1
GPUS_PER_NODE=8
STEPS_PER_RUN=100
MAX_STEPS=100
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
NUM_MINUTES=240
# ===== END CONFIG =====

set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$PROJECT_ROOT"
CONFIG_PATH=configs/recipes/llm/grpo-qwen2.5-math-7b-1n8g-fsdp2tp1-logra.yaml
if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "[ERROR] Config file $CONFIG_PATH not found" >&2
    exit 1
fi
if [[ -n "${TEST_DRYRUN:-}" ]]; then
    echo "[INFO] TEST_DRYRUN mode: config $CONFIG_PATH found"
    exit 0
fi
uv sync --locked --inexact --group test
uv run --locked run_grpo.py --config "$CONFIG_PATH" "$@"

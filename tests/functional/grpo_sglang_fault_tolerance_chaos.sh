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

# Kill a busy engine in real, synchronous GRPO on two colocated GPUs.
# EXPECT=survival requires replacement, trainer weight transfer, and completion.
# EXPECT=bounded_failure requires the specific restart-budget error at refit.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")
EXPECT=${EXPECT:-survival}
ARTIFACT_DIR=${ARTIFACT_DIR:-$(mktemp -d "/tmp/sglang-grpo-${EXPECT}.XXXXXX")}
mkdir -p "$ARTIFACT_DIR"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"

echo "[chaos] Real GRPO ${EXPECT}; artifacts: $ARTIFACT_DIR"
uv run --no-sync python tests/functional/_sglang_grpo_chaos.py \
    --expect "$EXPECT" --artifact-dir "$ARTIFACT_DIR" "$@" \
    2>&1 | tee "$ARTIFACT_DIR/harness.log"

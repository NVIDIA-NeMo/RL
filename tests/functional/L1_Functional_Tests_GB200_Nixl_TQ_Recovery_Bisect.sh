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

# GPU_COUNT=2
# The 120-minute job reserves 30 minutes for image/setup and result publication.
# Each revision runs the original recovery test; source trials have a 90-minute budget.
set -euo pipefail
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$PROJECT_ROOT"
source tests/scripts/detect_rdma.sh
if ! rdma_device_available; then
    echo "[FAIL] Historical NIXL comparison requires RDMA" >&2
    exit 1
fi
RESULT_DIR=$(mktemp -d "$PROJECT_ROOT/tests/functional/nixl-bisect.XXXXXX")
uv run --no-sync python -c 'import importlib.metadata, platform; print(platform.platform()); print({n: importlib.metadata.version(n) for n in ("nixl", "nixl-cu13", "torch", "ray")})' | tee "$RESULT_DIR/runtime.log"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | tee -a "$RESULT_DIR/runtime.log"
uv run --no-sync python tools/nixl_recovery_bisect.py --output "$RESULT_DIR"

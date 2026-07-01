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

SIF_DIR=${SIF_DIR:-/opt/data}
WORK_DIR=${WORK_DIR:-/workspace/sif}
STATE_DIR=${STATE_DIR:-/workspace/state}
MAX_WORKERS=${MAX_WORKERS:-1}

INSTANCE_FILE="${1:-/workspace/r2e-gym-instances-to-build.txt}"

python /workspace/build_swe_sif_images.py \
    --r2e-gym-ids-file "${INSTANCE_FILE}" \
    --max-workers "${MAX_WORKERS}" \
    --sif-dir "${SIF_DIR}" \
    --work-dir "${WORK_DIR}" \
    --registry "${DOCKER_REGISTRY}"

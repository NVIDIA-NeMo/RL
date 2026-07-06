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

# Prewarm the files SWE-bench otherwise downloads from raw.githubusercontent.com.

set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 <swe-bench-jsonl>" >&2
  exit 2
fi

DATASET_PATH="$1"
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
GYM_CODE="${GYM_CODE:-${REPO_ROOT}/3rdparty/Gym-workspace/Gym}"
SWE_AGENTS_DIR="${GYM_CODE}/responses_api_agents/swe_agents"
SETUP_DIR="${SWE_BENCH_PREFETCH_SETUP_DIR:-${SWE_AGENTS_DIR}/swe_swebench_artifact_prefetch_setup}"
SWEBENCH_DIR="${SETUP_DIR}/SWE-bench"
PATCH_PATH="${SWE_AGENTS_DIR}/patches/swebench_artifact_cache.patch"
CACHE_DIR="${SWE_BENCH_ARTIFACT_CACHE_DIR:-${SWE_AGENTS_DIR}/swebench_artifact_cache}"

if [[ "${SWEBENCH_PREWARM_IN_SRUN:-0}" != "1" ]]; then
  PREWARM_ACCOUNT="${PREWARM_ACCOUNT:-${SBATCH_ACCOUNT:-nemotron_sw_post}}"
  PREWARM_PARTITION="${PREWARM_PARTITION:-interactive}"
  PREWARM_GPUS="${PREWARM_GPUS:-1}"
  PREWARM_TIME="${PREWARM_TIME:-01:00:00}"

  exec srun \
    --account="${PREWARM_ACCOUNT}" \
    --partition="${PREWARM_PARTITION}" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=8 \
    --gres="gpu:${PREWARM_GPUS}" \
    --time="${PREWARM_TIME}" \
    --immediate=60 \
    env SWEBENCH_PREWARM_IN_SRUN=1 bash "$0" "${DATASET_PATH}"
fi

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "Dataset does not exist: ${DATASET_PATH}" >&2
  exit 2
fi

mkdir -p "${SETUP_DIR}"

if [[ ! -x "${SWEBENCH_DIR}/venv/bin/python" ]]; then
  SETUP_DIR="${SETUP_DIR}" \
  UV_DIR="${SETUP_DIR}/uv" \
  PYTHON_DIR="${SETUP_DIR}/python" \
  SWEBENCH_DIR="${SWEBENCH_DIR}" \
  SWEBENCH_REPO="https://github.com/HeyyyyyyG/SWE-bench.git" \
  SWEBENCH_COMMIT="HEAD" \
  SWEBENCH_PATCH="${PATCH_PATH}" \
  "${SWE_AGENTS_DIR}/setup_scripts/swebench.sh"
else
  if git -C "${SWEBENCH_DIR}" apply --reverse --check "${PATCH_PATH}"; then
    echo "SWE-bench artifact-cache patch already applied"
  else
    git -C "${SWEBENCH_DIR}" apply --check "${PATCH_PATH}"
    git -C "${SWEBENCH_DIR}" apply "${PATCH_PATH}"
  fi
fi

export SWE_BENCH_ARTIFACT_CACHE_DIR="${CACHE_DIR}"
export SWE_BENCH_ARTIFACT_CACHE_OFFLINE=0
"${SWEBENCH_DIR}/venv/bin/python" -m swebench.harness.prefetch_artifacts \
  --dataset "${DATASET_PATH}" \
  --cache-dir "${CACHE_DIR}"

export SWE_BENCH_ARTIFACT_CACHE_OFFLINE=1
"${SWEBENCH_DIR}/venv/bin/python" -m swebench.harness.prefetch_artifacts \
  --dataset "${DATASET_PATH}" \
  --cache-dir "${CACHE_DIR}" \
  --offline

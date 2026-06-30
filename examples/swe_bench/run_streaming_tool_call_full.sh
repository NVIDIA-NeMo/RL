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

# Submit a full streaming-tool-call SWE run with real training and weight refit.
# This wraps run_grpo_swe2_scale_gen.sh, which submits through sbatch + ray.sub.
#
# Defaults use the exact 16-node reproduction shape for one training step:
#   32 vLLM replicas = 8 generation nodes + 8 training nodes, 64 rollouts/step.
# Use NUM_VLLM_REPLICAS=16 for the smallest valid full-training shape (8 nodes).
# Set MAX_NUM_STEPS=all to use the recipe's uncapped training duration.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
LAUNCHER="${REPO_ROOT}/examples/swe_bench/run_grpo_swe2_scale_gen.sh"

if [[ -z "${HF_HOME:-}" ]]; then
  echo "ERROR: export HF_HOME before submitting." >&2
  exit 1
fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "ERROR: export WANDB_API_KEY before submitting." >&2
  exit 1
fi
if [[ ! -f "${LAUNCHER}" ]]; then
  echo "ERROR: launcher not found: ${LAUNCHER}" >&2
  exit 1
fi

NUM_VLLM_REPLICAS="${NUM_VLLM_REPLICAS:-32}"
STREAMING_TOOL_CALL="${STREAMING_TOOL_CALL:-1}"
MAX_NUM_STEPS_VALUE="${MAX_NUM_STEPS:-1}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d-%H%M%S)}"

if [[ "${STREAMING_TOOL_CALL}" == "1" ]]; then
  MODE="streaming"
elif [[ "${STREAMING_TOOL_CALL}" == "0" ]]; then
  MODE="baseline"
else
  echo "ERROR: STREAMING_TOOL_CALL must be 0 or 1." >&2
  exit 1
fi

export REPO_ROOT
export NUM_VLLM_REPLICAS
export STREAMING_TOOL_CALL
export SKIP_TRAINING=0
export TEMPERATURE="${TEMPERATURE:-1.0}"
export SBATCH_TIME="${SBATCH_TIME:-6:00:00}"
export WANDB_GROUP="${WANDB_GROUP:-streaming-tool-call-full-e2e}"
export EXP_SUFFIX="${EXP_SUFFIX:-streaming-tool-call-full-${MODE}-r${NUM_VLLM_REPLICAS}-steps${MAX_NUM_STEPS_VALUE}-${RUN_STAMP}}"

if [[ "${MAX_NUM_STEPS_VALUE}" == "all" ]]; then
  unset MAX_NUM_STEPS
else
  export MAX_NUM_STEPS="${MAX_NUM_STEPS_VALUE}"
fi

echo "Submitting full SWE run:"
echo "  mode=${MODE}"
echo "  replicas=${NUM_VLLM_REPLICAS}"
echo "  max_steps=${MAX_NUM_STEPS_VALUE}"
echo "  wandb_group=${WANDB_GROUP}"
echo "  experiment=${EXP_SUFFIX}"

bash "${LAUNCHER}"

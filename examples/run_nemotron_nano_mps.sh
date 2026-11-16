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

# Nemotron Nano MPS: single-node, eight-GPU GRPO with shared Megatron weights.
# Requires an exclusive node with eight high-memory GPUs (GB200-class capacity),
# NVLink within each four-GPU expert group, CUDA MPS, and the NeMo RL Megatron
# environment installed as described in docs/about/installation.md. The public
# model and DAPOMath17K dataset must be accessible through Hugging Face.
#
# Run from any directory:
#   bash examples/run_nemotron_nano_mps.sh
#   bash examples/run_nemotron_nano_mps.sh grpo.max_num_steps=3
#
# This starts a private MPS daemon before the driver creates its local Ray cluster.
# The recipe caps training at 58%; inference is uncapped. These are overlapping
# MPS limits, not disjoint SM reservations. No custom scheduler is enabled.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ -n "${RAY_ADDRESS:-}" || -v CUDA_VISIBLE_DEVICES ||
      -n "${CUDA_MPS_PIPE_DIRECTORY:-}" || -n "${CUDA_MPS_ACTIVE_THREAD_PERCENTAGE:-}" ||
      -n "${CUDA_MPS_SM_PARTITION:-}" ]]; then
    echo "Use a fresh shell on an exclusive eight-GPU node, without Ray/MPS/GPU-selection overrides." >&2
    exit 1
fi
gpu_uuids=$(nvidia-smi --query-gpu=uuid --format=csv,noheader)
if [[ $(wc -l <<< "$gpu_uuids") -ne 8 ]]; then
    echo "This example requires all eight GPUs on one node." >&2
    exit 1
fi
if uv run ray status >/dev/null 2>&1; then
    echo "A Ray cluster is already running; use a fresh exclusive node for this example." >&2
    exit 1
fi

mps_directory=$(mktemp -d /tmp/nemotron-nano-mps.XXXXXXXX)
export CUDA_MPS_PIPE_DIRECTORY="$mps_directory/pipe"
export CUDA_MPS_LOG_DIRECTORY="$mps_directory/log"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d
cleanup() {
    local result=$?
    trap - EXIT
    if ! printf 'quit\n' | nvidia-cuda-mps-control; then
        echo "MPS shutdown failed; inspect $CUDA_MPS_LOG_DIRECTORY." >&2
        if [[ "$result" -eq 0 ]]; then result=1; fi
    fi
    exit "$result"
}
trap cleanup EXIT

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
uv run python examples/run_grpo_single_controller.py \
    --config examples/configs/recipes/llm/grpo-nemotron-nano-30b-a3b-mps.yaml \
    "$@"

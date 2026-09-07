#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

set -xeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath "${SCRIPT_DIR}/../..")

cd "${PROJECT_ROOT}"

GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if (( GPU_COUNT < 2 )); then
    echo "SKIP: Nemotron Omni functional tests require at least two GB200 GPUs"
    exit 0
fi

# Both tests colocate TP2/EP2 training and generation on two GB200 GPUs.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

time uv run --no-sync bash ./tests/functional/nemotron_omni_clevr_megatron_1n2g.sh
time uv run --no-sync bash ./tests/functional/nemotron_omni_gym_video_megatron_1n2g.sh

cd "${PROJECT_ROOT}/tests"
if compgen -G ".coverage*" > /dev/null; then
    coverage combine .coverage*
fi

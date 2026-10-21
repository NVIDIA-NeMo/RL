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
umask 077
cd /flint-work/RL
export PYTHONPATH=/flint-work/RL:/flint-work/tools:/flint-work/sglang-plugin/src
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=disabled RAY_USAGE_STATS_ENABLED=0
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export SGLANG_EXTERNAL_MODEL_PACKAGE=flint_sglang FLASHINFER_USE_CUDA_NORM=1
export FLINT_TRITON_CACHE_BASE=/flint-work/cache/triton
export TMPDIR=/tmp/flint-grpo-8075956
export XDG_CACHE_HOME=/flint-work/cache
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
unset RAY_ADDRESS NRL_IGNORE_VERSION_MISMATCH
flint_config=examples/configs/recipes/llm/grpo-flint-4layer-1n4g-automodel-sglang-quick.yaml
if [[ "${FLINT_SYNTHETIC_SMOKE:-0}" == 1 ]]; then
  flint_config=examples/configs/recipes/llm/grpo-flint-4layer-1n4g-automodel-sglang-synthetic-quick.yaml
fi
exec /opt/flint/bin/python-driver /flint-work/tools/launch_grpo.py \
  --config "$flint_config" "$@"

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
cd -- "${BENCH_SOURCE}"
# Credentials and W&B endpoint, if needed, are inherited from the submit environment.
export PYTHONPATH="${BENCH_SOURCE}:${BENCH_SOURCE}/3rdparty/Gym-workspace/Gym"
BENCH_UV=${BENCH_UV:-uv}
"${BENCH_UV}" run --no-project --python "${BENCH_PYTHON}" \
    python -m tools.model_diagnostics.prefix_checkpoint_benchmark \
    --model "${BENCH_MODEL}" --output "${BENCH_RUN}/run" \
    --worker-python "${BENCH_PYTHON}" "$@" --preflight
exec "${BENCH_UV}" run --no-project --python "${BENCH_PYTHON}" \
    python -m tools.model_diagnostics.prefix_checkpoint_benchmark \
    --model "${BENCH_MODEL}" --output "${BENCH_RUN}/run" \
    --worker-python "${BENCH_PYTHON}" "$@"

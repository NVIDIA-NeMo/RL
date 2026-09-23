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
export LD_LIBRARY_PATH="${CUDA_COMPAT_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
generation_args=()
if [[ -n "${GENERATION_PYTHON:-}" ]]; then
  generation_args=(--generation-python "${GENERATION_PYTHON}")
fi
/home/snorouzi/.local/bin/uv run --no-project --python "${POLICY_PYTHON}" \
  python research/just_grpo/run_just_grpo.py --config "${CONFIG}" \
  --model "${MODEL}" --output-dir "${OUTPUT_DIR}" "${generation_args[@]}"

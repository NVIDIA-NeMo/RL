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

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath "${SCRIPT_DIR}/../..")
EXP_NAME=$(basename "$0" .sh)
EXP_DIR=${SCRIPT_DIR}/${EXP_NAME}
LOG_DIR=${EXP_DIR}/logs
RUN_LOG=${EXP_DIR}/run.log
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}

rm -rf "${EXP_DIR}"
mkdir -p "${LOG_DIR}"
git config --global --add safe.directory "${PROJECT_ROOT}"

dynamo_python=/opt/dynamo_venv/bin/python
"${dynamo_python}" -c \
  'import importlib.metadata as m; assert m.version("ai-dynamo") == "1.3.0.post1"; assert m.version("vllm") == "0.23.0"'
grep -Fqx \
  'vllm PR #44814 merge commit c9e5bf813530fb9ce06024e075da0f520b0718c8' \
  /opt/dynamo_venv/VLLM_BACKPORTS
/opt/dynamo_venv/bin/etcd --version
/opt/dynamo_venv/bin/nats-server --version

cd "${PROJECT_ROOT}"
uv run --no-sync coverage run -a \
  --data-file="${PROJECT_ROOT}/tests/.coverage" \
  --source="${PROJECT_ROOT}/nemo_rl" \
  "${PROJECT_ROOT}/examples/run_grpo.py" \
  --config "${PROJECT_ROOT}/examples/configs/grpo_math_1B_dynamo.yaml" \
  policy.model_name=Qwen/Qwen3-0.6B \
  policy.tokenizer.name=Qwen/Qwen3-0.6B \
  logger.log_dir="${LOG_DIR}" \
  2>&1 | tee "${RUN_LOG}"

grep -F "Performing policy generation refit" "${RUN_LOG}"
grep -F "Invalidated generation backend KV caches after weight update" "${RUN_LOG}"

if pgrep -f '[d]ynamo.frontend|[d]ynamo.vllm|[/]opt/dynamo_venv/bin/etcd|[/]opt/dynamo_venv/bin/nats-server'; then
  echo "Managed Dynamo processes remain after GRPO shutdown" >&2
  exit 1
fi

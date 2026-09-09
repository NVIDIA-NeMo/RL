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

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
source "${SCRIPT_DIR}/common.env"

# ===== BEGIN CONFIG =====
NUM_NODES=1
GPUS_PER_NODE=8
STEPS_PER_RUN=2
MAX_STEPS=2
NUM_RUNS=1
NUM_MINUTES=30
# ===== END CONFIG =====

# The H100 suite allocates eight GPUs; the recipe uses only three.
exit_if_max_steps_reached

dynamo_python=/opt/dynamo_venv/bin/python
"${dynamo_python}" -c \
  'import importlib.metadata as m; assert m.version("ai-dynamo") == "1.3.0.post1"; assert m.version("vllm") == "0.23.0"; assert m.version("nixl") == "1.1.0"; assert m.version("nixl-cu13") == "1.1.0"'
grep -Fqx \
  'vllm PR #44814 merge commit c9e5bf813530fb9ce06024e075da0f520b0718c8' \
  /opt/dynamo_venv/VLLM_BACKPORTS
/opt/dynamo_venv/bin/etcd --version
/opt/dynamo_venv/bin/nats-server --version

cd "${PROJECT_ROOT}"
uv run --no-sync \
  "${PROJECT_ROOT}/examples/run_grpo.py" \
  --config "${CONFIG_PATH}" \
  grpo.max_num_steps="${MAX_STEPS}" \
  logger.log_dir="${LOG_DIR}" \
  2>&1 | tee "${RUN_LOG}"

grep -F "'--disaggregation-mode', 'decode'" "${RUN_LOG}"
grep -F "'--disaggregation-mode', 'prefill'" "${RUN_LOG}"
grep -F "'VLLM_NIXL_SIDE_CHANNEL_PORT':" "${RUN_LOG}"
component_ready_line=$(grep -F \
    "frontend ready with managed components" "${RUN_LOG}")
grep -Fq "'backend': 1" <<< "${component_ready_line}"
grep -Fq "'prefill': 1" <<< "${component_ready_line}"

grep -F "Performing policy generation refit" "${RUN_LOG}"
grep -F "Invalidated generation backend KV caches after weight update" "${RUN_LOG}"

refit_count=$(grep -Fc "Performing policy generation refit" "${RUN_LOG}" || true)
cache_success_count=$(grep -Fc \
  "Invalidated generation backend KV caches after weight update" \
  "${RUN_LOG}" || true)
if [[ "${refit_count}" -eq 0 || "${cache_success_count}" -ne "${refit_count}" ]]; then
  echo "Expected one successful cache invalidation per refit; refits=${refit_count}, successes=${cache_success_count}" >&2
  exit 1
fi
if grep -Fq \
  -e "Failed to invalidate generation backend KV caches" \
  -e "KV cache invalidation not supported or only partially applied" \
  "${RUN_LOG}"; then
  echo "The Dynamo run reported a cache invalidation failure" >&2
  exit 1
fi

uv run --no-sync tests/json_dump_tb_logs.py \
  "${LOG_DIR}" \
  --output_path "${JSON_METRICS}" \
  --require-tag-prefix "generation_metrics/"
last_step=$(jq '[.["train/loss"] | keys[] | tonumber] | max' "${JSON_METRICS}")
if [[ "${last_step}" != "${MAX_STEPS}" ]]; then
  echo "Expected ${MAX_STEPS} training steps, found ${last_step}" >&2
  exit 1
fi
uv run --no-sync tests/check_metrics.py \
  "${JSON_METRICS}" \
  'max(data["train/token_mult_prob_error"]) < 1.05'

if pgrep -f '[d]ynamo.frontend|[d]ynamo.vllm|[/]opt/dynamo_venv/bin/etcd|[/]opt/dynamo_venv/bin/nats-server'; then
  echo "Managed Dynamo processes remain after GRPO shutdown" >&2
  exit 1
fi

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
DATA_DIR=${EXP_DIR}/data
GYM_ROOT=${PROJECT_ROOT}/3rdparty/Gym-workspace/Gym
SOURCE_DATA_DIR=${WORKPLACE_ASSISTANT_DATA_DIR:-${GYM_ROOT}/data/workplace_assistant}
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}

rm -rf "${EXP_DIR}"
mkdir -p "${LOG_DIR}" "${DATA_DIR}"
git config --global --add safe.directory "${PROJECT_ROOT}"

dynamo_python=/opt/dynamo_venv/bin/python
"${dynamo_python}" -c \
  'import importlib.metadata as m; assert m.version("ai-dynamo") == "1.3.0.post1"; assert m.version("vllm") == "0.23.0"'
grep -Fqx \
  'vllm PR #44814 merge commit c9e5bf813530fb9ce06024e075da0f520b0718c8' \
  /opt/dynamo_venv/VLLM_BACKPORTS

if [[ ! -f "${SOURCE_DATA_DIR}/train.jsonl" || ! -f "${SOURCE_DATA_DIR}/validation.jsonl" ]]; then
  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "Workplace-assistant data is missing and HF_TOKEN is not set" >&2
    exit 1
  fi
  cd "${GYM_ROOT}"
  if [[ ! -f env.yaml ]]; then
    printf 'hf_token: %s\n' "${HF_TOKEN}" > env.yaml
  fi
  uv run ng_prepare_data \
    "+config_paths=[resources_servers/workplace_assistant/configs/workplace_assistant.yaml]" \
    +output_dirpath=data/workplace_assistant \
    +mode=train_preparation \
    +should_download=true \
    +data_source=huggingface
  SOURCE_DATA_DIR=${GYM_ROOT}/data/workplace_assistant
fi

TRAIN_PATH=${DATA_DIR}/workplace_assistant_train.jsonl
VALIDATION_PATH=${DATA_DIR}/workplace_assistant_validation.jsonl
jq -c \
  '.responses_create_params.tools |= (.[0:1]) | .responses_create_params.tool_choice = "auto"' \
  "${SOURCE_DATA_DIR}/train.jsonl" > "${TRAIN_PATH}"
jq -c \
  '.responses_create_params.tools |= (.[0:1]) | .responses_create_params.tool_choice = "auto"' \
  "${SOURCE_DATA_DIR}/validation.jsonl" > "${VALIDATION_PATH}"
jq -s -e '
  all(.[];
    ((.responses_create_params.tools | type) == "array") and
    ((.responses_create_params.tools | length) == 1) and
    (.responses_create_params.tool_choice == "auto")
  )
' "${TRAIN_PATH}" "${VALIDATION_PATH}" > /dev/null

cd "${PROJECT_ROOT}"
uv run --no-sync coverage run -a \
  --data-file="${PROJECT_ROOT}/tests/.coverage" \
  --source="${PROJECT_ROOT}/nemo_rl" \
  "${PROJECT_ROOT}/examples/nemo_gym/run_grpo_nemo_gym.py" \
  --config "${PROJECT_ROOT}/examples/nemo_gym/grpo_dynamo_gym_smoke.yaml" \
  logger.log_dir="${LOG_DIR}" \
  data.train.data_path="${TRAIN_PATH}" \
  data.validation.data_path="${VALIDATION_PATH}" \
  "$@" \
  2>&1 | tee "${RUN_LOG}"

metrics_json=${EXP_DIR}/metrics.json
uv run --no-sync tests/json_dump_tb_logs.py \
  "${LOG_DIR}" \
  --output_path "${metrics_json}" \
  --require-tag-prefix "train/"
uv run --no-sync tests/check_metrics.py \
  "${metrics_json}" \
  '"2" in data["train/loss"]' \
  '"2" in data["train/global_valid_seqs"]'

batch_count=$(grep -Fc "Got trajectory batch (size: 4)" "${RUN_LOG}" || true)
if [[ "${batch_count}" -ne 2 ]]; then
  echo "Expected two four-rollout training batches; found ${batch_count}" >&2
  exit 1
fi
echo "Completed training rollouts: 8"

refit_count=$(grep -Fc "Performing policy generation refit" "${RUN_LOG}" || true)
cache_success_count=$(grep -Fc \
  "Invalidated generation backend KV caches after weight update" \
  "${RUN_LOG}" || true)
if [[ "${refit_count}" -eq 0 || "${cache_success_count}" -ne "${refit_count}" ]]; then
  echo "Expected one successful cache invalidation per refit; refits=${refit_count}, successes=${cache_success_count}" >&2
  exit 1
fi
if grep -Fq "/tokenize" "${RUN_LOG}"; then
  echo "Dynamo response used the unsupported /tokenize fallback" >&2
  exit 1
fi
if ! grep -Fq "'DYN_ENABLE_EXPERIMENTAL_PARSERS_V2': '1'" "${RUN_LOG}"; then
  echo "Managed Dynamo did not enable its nvext-preserving v2 tool parser" >&2
  exit 1
fi
if ! grep -Fq "'--dyn-tool-call-parser', 'qwen3_coder'" "${RUN_LOG}"; then
  echo "Managed Dynamo did not launch the qwen3_coder v2 tool parser" >&2
  exit 1
fi
if pgrep -f '[d]ynamo.frontend|[d]ynamo.vllm|[/]opt/dynamo_venv/bin/etcd|[/]opt/dynamo_venv/bin/nats-server'; then
  echo "Managed Dynamo processes remain after NeMo-Gym GRPO shutdown" >&2
  exit 1
fi

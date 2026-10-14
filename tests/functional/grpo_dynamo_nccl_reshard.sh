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

# Two-GPU smoke for nccl_reshard refit into managed Dynamo: one Megatron trainer
# GPU, one Dynamo vLLM engine. The driver reaches NeMo RL's worker extension inside
# the engine through Dynamo's /engine/* RL routes; the weights move over NCCL.
# Train TP1 -> gen TP1 makes the bulk reshard an identity mapping, so this covers
# the Dynamo transport, not the resharding math (see grpo_nccl_reshard_refit.sh).

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

cd "${PROJECT_ROOT}"
uv run --no-sync coverage run -a \
  --data-file="${PROJECT_ROOT}/tests/.coverage" \
  --source="${PROJECT_ROOT}/nemo_rl" \
  "${PROJECT_ROOT}/examples/run_grpo.py" \
  --config "${PROJECT_ROOT}/examples/configs/grpo_math_1B_dynamo.yaml" \
  policy.model_name=Qwen/Qwen3-0.6B \
  policy.tokenizer.name=Qwen/Qwen3-0.6B \
  policy.automodel_cfg.enabled=false \
  policy.megatron_cfg.enabled=true \
  policy.megatron_cfg.tensor_model_parallel_size=1 \
  policy.megatron_cfg.pipeline_model_parallel_size=1 \
  policy.generation.refit_transport=nccl_reshard \
  logger.log_dir="${LOG_DIR}" \
  "$@" \
  2>&1 | tee "${RUN_LOG}"

grep -F "nccl_reshard bulk comm group" "${RUN_LOG}"
refit_count=$(grep -Fc "Performing policy generation refit" "${RUN_LOG}" || true)
if [[ "${refit_count}" -eq 0 ]]; then
  echo "Expected at least one nccl_reshard refit" >&2
  exit 1
fi

metrics_json=${EXP_DIR}/metrics.json
uv run --no-sync tests/json_dump_tb_logs.py \
  "${LOG_DIR}" \
  --output_path "${metrics_json}"
# A broken refit leaves the engine on stale or partial weights, and the train/gen
# logprob ratio drifts from 1.
uv run --no-sync tests/check_metrics.py \
  "${metrics_json}" \
  'max(data["train/token_mult_prob_error"]) < 1.05'

if pgrep -f "[d]ynamo.frontend|[d]ynamo.vllm|[V]LLM::EngineCore|[n]emorl_dynamo_etcd_|[n]emorl_dynamo_nats_"; then
  echo "Managed Dynamo processes remain after GRPO shutdown" >&2
  exit 1
fi

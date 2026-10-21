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

# Two-GPU SingleController + NeMo-Gym run with managed Dynamo generation.
# Mirrors grpo_megatron_generation_gym_single_controller.sh with the generation
# backend swapped to managed Dynamo; Gym's requests go straight to its frontend.
# Pass ++token_capture.enabled=true to stage captured calls from the engines.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath "${SCRIPT_DIR}/../..")
EXP_NAME=$(basename "$0" .sh)
EXP_DIR=${SCRIPT_DIR}/${EXP_NAME}
LOG_DIR=${EXP_DIR}/logs
RUN_LOG=${EXP_DIR}/run.log
DATA_DIR=${EXP_DIR}/data
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}

rm -rf "${EXP_DIR}"
mkdir -p "${LOG_DIR}" "${DATA_DIR}"
git config --global --add safe.directory "${PROJECT_ROOT}"

# Managed Dynamo runs from the dynamo extra's actor venv (vLLM + ai-dynamo + nemo_rl),
# built here if needed; etcd and nats-server come from NEMO_RL_DYNAMO_BIN_DIR or PATH.
dynamo_python=$(uv run --no-sync python -c \
  'from nemo_rl.models.generation.dynamo.venv import get_dynamo_python; print(get_dynamo_python())' | tail -n 1)
"${dynamo_python}" -c \
  'import importlib.metadata as m; assert m.version("ai-dynamo") == "1.5.1"; assert m.version("vllm") == "0.29.0"'
uv run --no-sync python -c \
  'from nemo_rl.models.generation.dynamo.venv import get_dynamo_service_executable as e; e("etcd"); e("nats-server")'

# Follow nemo-gym instructions here to get this data:
# https://docs.nvidia.com/nemo/gym/0.1.0/tutorials/nemo-rl-grpo/setup.html#training-nemo-rl-grpo-setup
cd "${PROJECT_ROOT}/3rdparty/Gym-workspace/Gym"
# We need HF_TOKEN to download the data from huggingface
if [[ ! -f env.yaml ]]; then
  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "[ERROR] HF_TOKEN is not set" >&2
    exit 1
  fi
  echo "hf_token: ${HF_TOKEN}" >> env.yaml
fi
uv run ng_prepare_data "+config_paths=[resources_servers/workplace_assistant/configs/workplace_assistant.yaml]" \
  +output_dirpath=data/workplace_assistant \
  +mode=train_preparation \
  +should_download=true \
  +data_source=huggingface
cd "${PROJECT_ROOT}"

# Keep only the first tool: with every tool attached, the first prompt runs past this
# smoke's 512-token sequence length.
TRAIN_PATH=${DATA_DIR}/workplace_assistant_train.jsonl
VALIDATION_PATH=${DATA_DIR}/workplace_assistant_validation.jsonl
jq -c '.responses_create_params.tools |= (.[0:1])' 3rdparty/Gym-workspace/Gym/data/workplace_assistant/train.jsonl > "${TRAIN_PATH}"
jq -c '.responses_create_params.tools |= (.[0:1])' 3rdparty/Gym-workspace/Gym/data/workplace_assistant/validation.jsonl > "${VALIDATION_PATH}"

uv run --no-sync coverage run -a \
  --data-file="${PROJECT_ROOT}/tests/.coverage" \
  --source="${PROJECT_ROOT}/nemo_rl" \
  "${PROJECT_ROOT}/examples/run_grpo_single_controller.py" \
  --config "${PROJECT_ROOT}/examples/nemo_gym/grpo_qwen3_0_6b_dynamo_single_controller.yaml" \
  grpo.max_num_steps=3 \
  policy.generation.max_new_tokens=128 \
  policy.max_total_sequence_length=512 \
  logger.tensorboard_enabled=true \
  logger.log_dir="${LOG_DIR}" \
  logger.wandb_enabled=false \
  logger.monitor_gpus=false \
  data.train.data_path="${TRAIN_PATH}" \
  data.validation.data_path="${VALIDATION_PATH}" \
  "$@" \
  2>&1 | tee "${RUN_LOG}"

# Gym reached the Dynamo frontend, and every weight sync froze Dynamo in place (pause)
# and restored it afterwards (resume).
grep -F "[Dynamo] Forwarding rollouts to" "${RUN_LOG}"
sync_count=$(grep -Fc "_sync_weights: sync done" "${RUN_LOG}" || true)
pause_count=$(grep -Fc "[RL] Engine paused (mode=keep, clear_cache=True)" "${RUN_LOG}" || true)
resume_count=$(grep -Fc "[RL] Engine resumed" "${RUN_LOG}" || true)
if [[ "${sync_count}" -lt 2 || "${pause_count}" -lt "${sync_count}" || "${resume_count}" -lt "${sync_count}" ]]; then
  echo "Expected every weight sync to pause and resume Dynamo; syncs=${sync_count}, pauses=${pause_count}, resumes=${resume_count}" >&2
  exit 1
fi

metrics_json=${EXP_DIR}/metrics.json
uv run --no-sync tests/json_dump_tb_logs.py "${LOG_DIR}" --output_path "${metrics_json}"
EXTRA_CHECKS=()
if [[ "$*" == *token_capture.enabled=true* ]]; then
  # Nonzero only when a finalizer actor ran, i.e. capture really was on.
  EXTRA_CHECKS+=('max(data["train/finalize/total_ms"]) > 0')
fi
# The worker starts from dummy weights, so these hold only if every refit landed.
uv run --no-sync tests/check_metrics.py "${metrics_json}" \
  'max(data["train/token_mult_prob_error"]) < 1.05' \
  'median(data["train/gen_kl_error"]) < 1.3' \
  'max(data["train/reward"]) > 0' \
  "${EXTRA_CHECKS[@]}"

if pgrep -f "[d]ynamo.frontend|[d]ynamo.vllm|[d]ynamo.engine_main|[V]LLM::EngineCore|[n]emorl_dynamo_etcd_|[n]emorl_dynamo_nats_"; then
  echo "Managed Dynamo processes remain after SingleController shutdown" >&2
  exit 1
fi

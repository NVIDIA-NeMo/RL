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

# Two-GPU SingleController + TransferQueue run with managed Dynamo generation.

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

# Managed Dynamo runs from the dynamo extra's actor venv (vLLM + ai-dynamo + nemo_rl),
# built here if needed; etcd and nats-server come from NEMO_RL_DYNAMO_BIN_DIR or PATH.
dynamo_python=$(uv run --no-sync python -c \
  'from nemo_rl.models.generation.dynamo.venv import get_dynamo_python; print(get_dynamo_python())' | tail -n 1)
"${dynamo_python}" -c \
  'import importlib.metadata as m; assert m.version("ai-dynamo") == "1.5.1"; assert m.version("vllm") == "0.29.0"'
uv run --no-sync python -c \
  'from nemo_rl.models.generation.dynamo.venv import get_dynamo_service_executable as e; e("etcd"); e("nats-server")'

cd "${PROJECT_ROOT}"
uv run --no-sync coverage run -a \
  --data-file="${PROJECT_ROOT}/tests/.coverage" \
  --source="${PROJECT_ROOT}/nemo_rl" \
  "${PROJECT_ROOT}/examples/run_grpo_single_controller.py" \
  --config "${PROJECT_ROOT}/examples/configs/grpo_math_1B_dynamo_single_controller.yaml" \
  policy.model_name=Qwen/Qwen3-0.6B \
  policy.tokenizer.name=Qwen/Qwen3-0.6B \
  grpo.num_prompts_per_step=2 \
  grpo.num_generations_per_prompt=4 \
  grpo.seq_logprob_error_threshold=1000 \
  grpo.max_num_steps=3 \
  policy.train_global_batch_size=8 \
  policy.train_micro_batch_size=1 \
  policy.logprob_batch_size=1 \
  policy.generation.max_new_tokens=128 \
  async_rl.sampler.name=in_order \
  async_rl.sampler.max_lookahead_versions=1 \
  async_rl.min_groups_for_streaming_train=2 \
  async_rl.max_inflight_prompts=2 \
  async_rl.max_buffered_rollouts=4 \
  data_plane.backend=simple \
  logger.log_dir="${LOG_DIR}" \
  logger.tensorboard_enabled=true \
  logger.wandb_enabled=false \
  logger.monitor_gpus=false \
  checkpointing.enabled=false \
  "$@" \
  2>&1 | tee "${RUN_LOG}"

# Every weight sync freezes Dynamo in place (pause) and restores it afterwards
# (resume). Dynamo accepts allow_unpaused updates, so a lost pause would not
# fail the refit.
sync_count=$(grep -Fc "_sync_weights: sync done" "${RUN_LOG}" || true)
pause_count=$(grep -Fc "[RL] Engine paused (mode=keep, clear_cache=True)" "${RUN_LOG}" || true)
resume_count=$(grep -Fc "[RL] Engine resumed" "${RUN_LOG}" || true)
if [[ "${sync_count}" -lt 2 || "${pause_count}" -lt "${sync_count}" || "${resume_count}" -lt "${sync_count}" ]]; then
  echo "Expected every weight sync to pause and resume Dynamo; syncs=${sync_count}, pauses=${pause_count}, resumes=${resume_count}" >&2
  exit 1
fi

metrics_json=${EXP_DIR}/metrics.json
uv run --no-sync tests/json_dump_tb_logs.py "${LOG_DIR}" --output_path "${metrics_json}"
# The worker starts from dummy weights, so these hold only if every refit landed.
uv run --no-sync tests/check_metrics.py "${metrics_json}" \
  'max(data["train/token_mult_prob_error"]) < 1.05' \
  'max(data["train/gen_kl_error"]) < 0.002'

if pgrep -f "[d]ynamo.frontend|[d]ynamo.vllm|[V]LLM::EngineCore|[n]emorl_dynamo_etcd_|[n]emorl_dynamo_nats_"; then
  echo "Managed Dynamo processes remain after SingleController shutdown" >&2
  exit 1
fi

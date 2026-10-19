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

# Three-GPU SingleController run: one Megatron trainer GPU, one Megatron
# prefill engine, and one Megatron decode engine behind a Dynamo frontend.

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

dynamo_venv_dir=${NEMO_RL_DYNAMO_VENV_DIR:-/opt/dynamo_venv}
"${dynamo_venv_dir}/bin/python" -c \
  'import importlib.metadata as m; assert m.version("ai-dynamo") == "1.5.1"'
# The adapter sidecar imports Megatron-LM's Dynamo integration from this tree.
megatron_root=${NEMO_RL_MEGATRON_ROOT:-${PROJECT_ROOT}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM}
test -d "${megatron_root}/megatron/inference/integrations/dynamo"

cd "${PROJECT_ROOT}"
uv run --no-sync coverage run -a \
  --data-file="${PROJECT_ROOT}/tests/.coverage" \
  --source="${PROJECT_ROOT}/nemo_rl" \
  "${PROJECT_ROOT}/examples/run_grpo_single_controller.py" \
  --config "${PROJECT_ROOT}/examples/configs/grpo_math_1B_dynamo_pd_megatron_single_controller.yaml" \
  policy.model_name=Qwen/Qwen3-0.6B \
  policy.tokenizer.name=Qwen/Qwen3-0.6B \
  policy.generation.dynamo_cfg.megatron_root="${megatron_root}" \
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

# Every weight sync quiesces generation, pauses both engines, and resumes them.
sync_count=$(grep -Fc "_sync_weights: sync done" "${RUN_LOG}" || true)
pause_count=$(grep -Fc "paused inference engine" "${RUN_LOG}" || true)
resume_count=$(grep -Fc "resumed inference engine" "${RUN_LOG}" || true)
# Two single-rank engines: one pause and one resume line per engine per sync.
# The first sync only builds the engines, so it contributes resumes but no pause.
if [[ "${sync_count}" -lt 2 || "${pause_count}" -lt $(( 2 * (sync_count - 1) )) || "${resume_count}" -lt $(( 2 * sync_count )) ]]; then
  echo "Expected every weight sync after the first to pause and resume both engines; syncs=${sync_count}, pauses=${pause_count}, resumes=${resume_count}" >&2
  exit 1
fi
# The frontend saw both roles register before the first rollout.
if ! grep -Fq "[Dynamo] frontend ready" "${RUN_LOG}"; then
  echo "Expected the Dynamo frontend to report both engine roles ready" >&2
  exit 1
fi

metrics_json=${EXP_DIR}/metrics.json
uv run --no-sync tests/json_dump_tb_logs.py "${LOG_DIR}" --output_path "${metrics_json}"
# The engines start from dummy weights, so these hold only if every refit landed.
uv run --no-sync tests/check_metrics.py "${metrics_json}" \
  'max(data["train/token_mult_prob_error"]) < 1.05' \
  'data["train/token_mult_prob_error"]["3"] < 1.05'

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
NUM_NODES=3
GPUS_PER_NODE=8
STEPS_PER_RUN=4
MAX_STEPS=4
NUM_RUNS=1
NUM_MINUTES=240
USE_GYM_CONTAINER=true
# ===== END CONFIG =====

exit_if_max_steps_reached

cd "${PROJECT_ROOT}"
uv run examples/nemo_gym/run_grpo_nemo_gym.py \
    --config "${CONFIG_PATH}" \
    grpo.max_num_steps="${MAX_STEPS}" \
    logger.log_dir="${LOG_DIR}" \
    logger.wandb_enabled=true \
    logger.wandb.project=nemo-rl \
    logger.wandb.name="${EXP_NAME}" \
    logger.tensorboard_enabled=true \
    "$@" \
    2>&1 | tee "${RUN_LOG}"

uv run tests/json_dump_tb_logs.py "${LOG_DIR}" \
    --output_path "${JSON_METRICS}" \
    --require-tag-prefix generation_metrics/

last_step=$(jq 'to_entries | .[] | select(.key == "train/loss") | .value | keys | map(tonumber) | max' "${JSON_METRICS}")
if [[ ${last_step} -lt ${MAX_STEPS} ]]; then
    echo "[ERROR] Expected step ${MAX_STEPS}, but the last train/loss step is ${last_step}"
    exit 1
fi

uv run tests/check_metrics.py "${JSON_METRICS}" \
    'median(data["train/token_mult_prob_error"]) < 1.1' \
    "data['train/token_mult_prob_error']['${MAX_STEPS}'] < 1.1" \
    'mean(data["train/gen_kl_error"]) < 0.02'

refit_count=$(grep -c "✅ Ready for refit" "${RUN_LOG}" || true)
cache_invalidation_count=$(grep -c \
    "✅ Invalidated generation backend KV caches after weight update" \
    "${RUN_LOG}" || true)
cache_invalidation_failure_count=$(grep -cE \
    "Failed to invalidate generation backend KV caches|Dynamo KV cache invalidation failed" \
    "${RUN_LOG}" || true)
if [[ ${cache_invalidation_failure_count} -ne 0 ]]; then
    echo "[ERROR] Found ${cache_invalidation_failure_count} cache invalidation failure(s)"
    exit 1
fi
if [[ ${refit_count} -ne ${cache_invalidation_count} ]]; then
    echo "[ERROR] Expected one cache invalidation per refit, but found " \
        "${cache_invalidation_count} invalidations for ${refit_count} refits"
    exit 1
fi

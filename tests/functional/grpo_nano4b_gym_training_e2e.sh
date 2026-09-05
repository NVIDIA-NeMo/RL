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

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT=$(realpath "${SCRIPT_DIR}/../..")
EXP_NAME=$(basename "$0" .sh)
EXP_DIR="${SCRIPT_DIR}/${EXP_NAME}"
LOG_DIR="${EXP_DIR}/logs"
JSON_METRICS="${EXP_DIR}/metrics.json"
RUN_LOG="${EXP_DIR}/run.log"
CONFIG_PATH="${SCRIPT_DIR}/grpo_nano4b_gym_training_e2e.yaml"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

rm -rf "${EXP_DIR}"
mkdir -p "${LOG_DIR}"

cd "${PROJECT_ROOT}"

uv run coverage run -a --data-file="${PROJECT_ROOT}/tests/.coverage" --source="${PROJECT_ROOT}/nemo_rl" \
    "${PROJECT_ROOT}/examples/nemo_gym/run_grpo_nemo_gym.py" \
    --config "${CONFIG_PATH}" \
    logger.log_dir="${LOG_DIR}" \
    "$@" \
    2>&1 | tee "${RUN_LOG}"

grep -Fq "Running synchronous GRPO training" "${RUN_LOG}"

uv run tests/json_dump_tb_logs.py "${LOG_DIR}" --output_path "${JSON_METRICS}"

# The fixture intentionally contains one accepted and one rejected rollout. In
# addition to testing both verifier outcomes, this gives Reinforce++ a non-zero
# advantage so grad_norm proves that an optimizer step was actually exercised.
uv run tests/check_metrics.py "${JSON_METRICS}" \
    'len(data["train/loss"]) == 1' \
    'all_finite(data["train/loss"])' \
    'all_finite(data["train/grad_norm"])' \
    'min(data["train/grad_norm"]) > 0' \
    'all_finite(data["train/advantages/min"])' \
    'all_finite(data["train/advantages/max"])' \
    'min(data["train/advantages/min"]) < 0' \
    'max(data["train/advantages/max"]) > 0' \
    'data["train/total_reward/min"]["1"] == 0' \
    'data["train/total_reward/max"]["1"] == 1' \
    'data["train/total_reward/mean"]["1"] == 0.5' \
    'all_finite(data["train/token_mult_prob_error"])' \
    'max(data["train/token_mult_prob_error"]) < 1.05' \
    'data["timing/train/generation"]["1"] > 0' \
    'data["validation/accuracy"]["1"] == 0.5' \
    'data["timing/validation/total_validation_time"]["1"] > 0'

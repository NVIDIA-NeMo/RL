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

#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT=$(realpath ${SCRIPT_DIR}/../..)
git config --global --add safe.directory ${PROJECT_ROOT}

EXP_NAME=$(basename $0 .sh)
EXP_DIR=${SCRIPT_DIR}/${EXP_NAME}
LOG_DIR=${EXP_DIR}/logs
JSON_METRICS=${EXP_DIR}/metrics.json
RUN_LOG=${EXP_DIR}/run.log
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}

rm -rf ${EXP_DIR}
mkdir -p ${LOG_DIR}

cd ${PROJECT_ROOT}
uv run --group test coverage run -a \
    --data-file=${PROJECT_ROOT}/tests/.coverage \
    --source=${PROJECT_ROOT}/nemo_rl \
    ${PROJECT_ROOT}/examples/run_grpo.py \
    --config ${PROJECT_ROOT}/examples/configs/grpo_math_1B_trtllm.yaml \
    policy.model_name=Qwen/Qwen3-0.6B \
    policy.max_total_sequence_length=512 \
    policy.generation.max_new_tokens=128 \
    policy.generation.colocated.enabled=true \
    policy.generation.trtllm_cfg.async_engine=true \
    policy.generation.trtllm_cfg.tensor_parallel_size=1 \
    grpo.async_grpo.enabled=false \
    grpo.num_prompts_per_step=2 \
    grpo.num_generations_per_prompt=4 \
    grpo.max_num_steps=2 \
    policy.train_global_batch_size=4 \
    policy.train_micro_batch_size=1 \
    policy.logprob_batch_size=4 \
    cluster.gpus_per_node=2 \
    cluster.num_nodes=1 \
    logger.tensorboard_enabled=true \
    logger.log_dir=${LOG_DIR} \
    logger.wandb_enabled=false \
    logger.monitor_gpus=true \
    checkpointing.enabled=false \
    "$@" \
    2>&1 | tee ${RUN_LOG}

uv run tests/json_dump_tb_logs.py ${LOG_DIR} --output_path ${JSON_METRICS}
uv run tests/check_metrics.py ${JSON_METRICS} \
    'max(data["train/token_mult_prob_error"]) < 1.05'

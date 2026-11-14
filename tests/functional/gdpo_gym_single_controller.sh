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
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
EXP_DIR="${REPO_ROOT}/tests/functional/gdpo_gym_single_controller"
mkdir -m 755 -p "${EXP_DIR}"
RUN_DIR=$(mktemp -d "${EXP_DIR}/run-XXXXXX")
chmod 755 "${RUN_DIR}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"

GYM_ROOT="${REPO_ROOT}/3rdparty/Gym-workspace/Gym"
(
    cd "${GYM_ROOT}"
    uv run python resources_servers/example_tool_call_multireward/create_examples.py
)
jq -c '. + {agent_ref: {name: "example_tool_call_multireward_simple_agent"}}' \
    "${GYM_ROOT}/resources_servers/example_tool_call_multireward/data/example.jsonl" \
    > "${RUN_DIR}/validation.jsonl"
for _ in $(seq 1 60); do
    cat "${RUN_DIR}/validation.jsonl"
done > "${RUN_DIR}/train.jsonl"

uv run python "${REPO_ROOT}/examples/run_grpo_single_controller.py" \
    --config "${REPO_ROOT}/examples/nemo_gym/gdpo_example_tool_call_multireward.yaml" \
    policy.model_name=Qwen/Qwen2.5-1.5B-Instruct \
    policy.automodel_cfg.enabled=false \
    policy.megatron_cfg.enabled=true \
    policy.megatron_cfg.tensor_model_parallel_size=1 \
    policy.megatron_cfg.sequence_parallel=false \
    policy.generation.vllm_cfg.tensor_parallel_size=1 \
    policy.generation.vllm_cfg.http_server_serving_chat_kwargs.tool_parser=hermes \
    policy.generation.colocated.enabled=false \
    policy.generation.colocated.resources.num_nodes=1 \
    policy.generation.colocated.resources.gpus_per_node=1 \
    policy.max_total_sequence_length=2048 \
    policy.generation.max_new_tokens=256 \
    policy.train_global_batch_size=8 \
    policy.train_micro_batch_size=1 \
    cluster.num_nodes=1 cluster.gpus_per_node=2 \
    grpo.num_prompts_per_step=2 grpo.num_generations_per_prompt=4 \
    grpo.max_num_steps=3 grpo.val_period=0 grpo.val_at_start=false \
    grpo.val_at_end=false grpo.async_grpo=null \
    logger.tensorboard_enabled=true logger.wandb_enabled=false \
    logger.log_dir="${RUN_DIR}/logs" \
    checkpointing.enabled=false \
    data.train.data_path="${RUN_DIR}/train.jsonl" \
    data.validation.data_path="${RUN_DIR}/validation.jsonl" \
    ++token_capture.enabled=false \
    ++data_plane.enabled=true ++data_plane.impl=transfer_queue \
    ++data_plane.backend=simple ++data_plane.simple.storage_capacity=1000000 \
    ++data_plane.simple.num_storage_units=2 \
    ++async_rl.sampler.name=in_order ++async_rl.sampler.max_lookahead_versions=0 \
    ++async_rl.min_groups_for_streaming_train=2 \
    ++async_rl.max_inflight_prompts=2 ++async_rl.max_buffered_rollouts=2 \
    "$@" 2>&1 | tee "${RUN_DIR}/run.log"

uv run "${REPO_ROOT}/tests/json_dump_tb_logs.py" "${RUN_DIR}/logs" \
    --output_path "${RUN_DIR}/metrics.json"
uv run "${REPO_ROOT}/tests/check_metrics.py" "${RUN_DIR}/metrics.json" \
    'len(data["train/loss"]) == 3' \
    'all_finite(data["train/loss"])' \
    'max(data["train/reward"]) > 0' \
    'min(data["train/global_valid_seqs"]) > 0'

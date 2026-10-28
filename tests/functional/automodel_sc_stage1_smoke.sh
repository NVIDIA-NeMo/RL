#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

# Stage 1 integration smoke: run in the NeMo RL driver environment on 8 GPUs.
# Model paths, W&B run names, and output directories can be overridden via "$@".
set -euo pipefail
uv run examples/run_grpo_single_controller.py \
  --config examples/configs/recipes/llm/dapo-gemma4-e2b-it-1n8g-automodel.yaml \
  grpo.async_grpo=null \
  grpo.max_num_steps=5 \
  grpo.val_period=0 \
  grpo.num_prompts_per_step=8 \
  grpo.num_generations_per_prompt=4 \
  grpo.use_dynamic_sampling=false \
  grpo.reward_scaling.enabled=false \
  grpo.reward_shaping.enabled=false \
  policy.train_global_batch_size=32 \
  policy.train_micro_batch_size=1 \
  policy.dynamic_batching.enabled=false \
  policy.sequence_packing.enabled=false \
  policy.generation.colocated.enabled=false \
  policy.generation.colocated.resources.num_nodes=1 \
  policy.generation.colocated.resources.gpus_per_node=4 \
  policy.generation.vllm_cfg.async_engine=true \
  data_plane.enabled=true \
  ++async_rl.sampler.name=in_order \
  ++async_rl.sampler.max_lookahead_versions=0 \
  ++async_rl.min_groups_for_streaming_train=8 \
  ++async_rl.max_inflight_prompts=8 \
  ++async_rl.max_buffered_rollouts=8 \
  checkpointing.enabled=false \
  logger.wandb_enabled=true \
  logger.wandb.project=nemorl-gemma4 \
  logger.wandb.name=grpo-gemma4-e2b-it-1n8g-automodel-sc-stage1 \
  logger.log_dir=logs/grpo-gemma4-e2b-it-1n8g-automodel-sc-stage1 \
  "$@"

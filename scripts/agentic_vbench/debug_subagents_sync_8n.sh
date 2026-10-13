#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# First production gate for subagent training: 8 nodes, synchronous/in-order
# rollouts, explicit delegation, and one checkpoint per optimizer update.
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export CONFIG_IN_CONTAINER=${CONFIG_IN_CONTAINER:-examples/configs/recipes/vlm/vlm_grpo-nemotron-super-omni-120ba12b-16n8g-megatron-agentic-vbench-sc-256k.v1.yaml}
export NUM_NODES=${NUM_NODES:-8}
export NUM_GEN_NODES=${NUM_GEN_NODES:-4}
export SEGMENT_SIZE=${SEGMENT_SIZE:-4}
export NUM_PROMPTS=${NUM_PROMPTS:-8}
export NUM_GENERATIONS=${NUM_GENERATIONS:-4}
export MAX_STEPS=${MAX_STEPS:-1000000}
# max_num_epochs is null in launch_sc.sh, so the dataloader cycles this compact
# manifest independently of the production optimizer horizon.
export TRAIN_REPEATS=${TRAIN_REPEATS:-4}
export SAVE_PERIOD=${SAVE_PERIOD:-1}
export CHECKPOINT_KEEP_TOP_K=${CHECKPOINT_KEEP_TOP_K:-2}
export JOB_CYCLES=${JOB_CYCLES:-6}
export SLURM_QOS=${SLURM_QOS:-short}
export SLURM_TIME_LIMIT=${SLURM_TIME_LIMIT:-02:00:00}
# Twelve turns give the 8-node parity gate useful tool/subagent coverage while
# keeping synchronous straggler time below the full 32-turn qualification run.
export AGENTIC_VBENCH_MAX_TURNS=${AGENTIC_VBENCH_MAX_TURNS:-12}
export BASE_NAME=${BASE_NAME:-agentic_vbench_v2_subagents_sync_8n_$(date -u +%Y%m%d-%H%M%S)-$$}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-$BASE_NAME}

exec bash "$script_dir/smoke_subagents_fast_step.sh" \
  async_rl.sampler.name=in_order \
  async_rl.sampler.max_lookahead_versions=0 \
  "async_rl.min_groups_for_streaming_train=${NUM_PROMPTS}" \
  "async_rl.max_inflight_prompts=${NUM_PROMPTS}" \
  "async_rl.max_buffered_rollouts=${NUM_PROMPTS}" \
  async_rl.rollout_failure.on_dropped_prompt=replace \
  async_rl.rollout_failure.max_replacement_attempts=4 \
  async_rl.rollout_failure.replacement_reserve_prompts=1 \
  "$@"

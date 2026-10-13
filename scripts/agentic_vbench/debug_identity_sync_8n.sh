#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# First production-debug gate for the integrated NeMo-RL v2 branch.
#
# Runs the framework-owned identity capture path on 8 nodes: 4 learner + 4
# generation. OpenCode summarization/pruning stays off, so no context rewrite
# occurs, but NeMo-RL still reconstructs the served call chain. Ordinary token
# capture cannot reliably resolve stock OpenCode's replayed history.
#
# The in_order sampler has zero lookahead, so policy version N must produce one
# complete step and train before version N+1 can dispatch. This is
# SingleController's synchronous/on-policy contract. The control uses 12 turns,
# which is the validated four-learner-node size; the 16-node gates exercise the
# full 32-turn agent budget.
#
# Expected use with the Super-GA checkpoint:
#   MODEL_PATH=/lustre/fsw/portfolios/nemotron/users/ehosseiniasl/checkpoints/\
#super35-journey-mopd2-identity-upsampling-from-my-step30-yifuw-001_boosted_mtp \
#     bash scripts/agentic_vbench/debug_identity_sync_8n.sh
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export CONFIG_IN_CONTAINER=${CONFIG_IN_CONTAINER:-examples/configs/recipes/vlm/vlm_grpo-nemotron-super-omni-120ba12b-16n8g-megatron-agentic-vbench-sc-256k.v1.yaml}
export NUM_NODES=${NUM_NODES:-8}
export NUM_GEN_NODES=${NUM_GEN_NODES:-4}
export SEGMENT_SIZE=${SEGMENT_SIZE:-4}
export NUM_PROMPTS=${NUM_PROMPTS:-8}
export NUM_GENERATIONS=${NUM_GENERATIONS:-4}
export MAX_STEPS=${MAX_STEPS:-1}
export SAVE_PERIOD=${SAVE_PERIOD:-1}
export JOB_CYCLES=${JOB_CYCLES:-1}
export SLURM_QOS=${SLURM_QOS:-short}
export SLURM_TIME_LIMIT=${SLURM_TIME_LIMIT:-02:00:00}
export CONTEXT_COMPACTION=true
export AGENTIC_VBENCH_COMPACTION_AUTO=false
export AGENTIC_VBENCH_COMPACTION_PRUNE=false
export AGENTIC_VBENCH_MAX_TURNS=${AGENTIC_VBENCH_MAX_TURNS:-12}
export BASE_NAME=${BASE_NAME:-agentic_vbench_v2_identity_sync_8n_$(date -u +%Y%m%d-%H%M%S)-$$}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-$BASE_NAME}

exec bash "$script_dir/smoke_off_32turns_256k.sh" \
  async_rl.sampler.name=in_order \
  async_rl.sampler.max_lookahead_versions=0 \
  "async_rl.min_groups_for_streaming_train=${NUM_PROMPTS}" \
  "async_rl.max_inflight_prompts=${NUM_PROMPTS}" \
  "async_rl.max_buffered_rollouts=${NUM_PROMPTS}" \
  async_rl.rollout_failure.on_dropped_prompt=replace \
  async_rl.rollout_failure.max_replacement_attempts=4 \
  async_rl.rollout_failure.replacement_reserve_prompts=1 \
  "$@"

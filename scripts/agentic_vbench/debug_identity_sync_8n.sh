#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# First production-debug gate for the integrated NeMo-RL v2 branch.
#
# Runs plain identity token capture (no OpenCode or framework compaction) on
# 8 nodes: 4 learner + 4 generation. The in_order sampler has zero lookahead,
# so policy version N must produce one complete step and train before version
# N+1 can dispatch. This is SingleController's synchronous/on-policy contract.
#
# Expected use with the Super-GA checkpoint:
#   MODEL_PATH=/lustre/fsw/portfolios/nemotron/users/ehosseiniasl/checkpoints/\
#super35-journey-mopd2-identity-upsampling-from-my-step30-yifuw-001_boosted_mtp \
#     bash scripts/agentic_vbench/debug_identity_sync_8n.sh
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export CONFIG_IN_CONTAINER=${CONFIG_IN_CONTAINER:-examples/configs/recipes/vlm/vlm_grpo-nemotron-super-omni-120ba12b-8n8g-megatron-agentic-vbench-sc.v2.yaml}
export NUM_NODES=${NUM_NODES:-8}
export NUM_GEN_NODES=${NUM_GEN_NODES:-4}
export SEGMENT_SIZE=${SEGMENT_SIZE:-4}
export NUM_PROMPTS=${NUM_PROMPTS:-8}
export NUM_GENERATIONS=${NUM_GENERATIONS:-4}
export MAX_STEPS=${MAX_STEPS:-2}
export SAVE_PERIOD=${SAVE_PERIOD:-1}
export JOB_CYCLES=${JOB_CYCLES:-1}
export BASE_NAME=${BASE_NAME:-agentic_vbench_v2_identity_sync_8n_$(date -u +%Y%m%d-%H%M%S)-$$}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-$BASE_NAME}

exec bash "$script_dir/smoke_identity_32turns_256k.sh" \
  async_rl.sampler.name=in_order \
  async_rl.sampler.max_lookahead_versions=0 \
  "async_rl.min_groups_for_streaming_train=${NUM_PROMPTS}" \
  "async_rl.max_inflight_prompts=${NUM_PROMPTS}" \
  "async_rl.max_buffered_rollouts=${NUM_PROMPTS}" \
  "$@"

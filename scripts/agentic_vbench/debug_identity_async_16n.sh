#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Second production-debug gate for the integrated NeMo-RL v2 branch.
#
# Keeps the same framework-owned identity objective and 256k policy window as
# the 8-node sync gate, then expands to 16 nodes (8 learner + 8 generation) and enables the
# v2 ready-first asynchronous sampler with up to two stale policy versions and
# three steps of prompt groups in flight. Run only after the sync gate has sane
# finite optimizer metrics, nonempty rows/rewards and raw TMPE near one.
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export CONFIG_IN_CONTAINER=${CONFIG_IN_CONTAINER:-examples/configs/recipes/vlm/vlm_grpo-nemotron-super-omni-120ba12b-16n8g-megatron-agentic-vbench-sc-256k.v1.yaml}
export NUM_NODES=${NUM_NODES:-16}
export NUM_GEN_NODES=${NUM_GEN_NODES:-8}
export SEGMENT_SIZE=${SEGMENT_SIZE:-8}
export NUM_PROMPTS=${NUM_PROMPTS:-8}
export NUM_GENERATIONS=${NUM_GENERATIONS:-4}
export MAX_STEPS=${MAX_STEPS:-1000000}
# The dataloader cycles because max_num_epochs is null; keep the materialized
# manifest compact instead of coupling its size to the production horizon.
export TRAIN_REPEATS=${TRAIN_REPEATS:-4}
export SAVE_PERIOD=${SAVE_PERIOD:-1}
export JOB_CYCLES=${JOB_CYCLES:-6}
export SLURM_QOS=${SLURM_QOS:-short}
export SLURM_TIME_LIMIT=${SLURM_TIME_LIMIT:-02:00:00}
export CONTEXT_COMPACTION=true
export AGENTIC_VBENCH_COMPACTION_AUTO=false
export AGENTIC_VBENCH_COMPACTION_PRUNE=false
export BASE_NAME=${BASE_NAME:-agentic_vbench_v2_identity_async_16n_$(date -u +%Y%m%d-%H%M%S)-$$}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-$BASE_NAME}

inflight_groups=$((NUM_PROMPTS * 3))
exec bash "$script_dir/smoke_off_32turns_256k.sh" \
  async_rl.sampler.name=ready_first \
  ~async_rl.sampler.max_lookahead_versions \
  +async_rl.sampler.max_staleness_versions=2 \
  "async_rl.min_groups_for_streaming_train=${NUM_PROMPTS}" \
  "async_rl.max_inflight_prompts=${inflight_groups}" \
  "async_rl.max_buffered_rollouts=${inflight_groups}" \
  "$@"

#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# 16-node compaction run shaped so step 1 (and its checkpoint) lands inside the
# ~30-40 min window after which the cluster scheduler cancels >=8-node jobs.
#
# Run 2182723 (same tasks, 32 turns, 8192 output tokens, 64k declared OpenCode
# context) finished 59 episodes in 38 min without a single optimizer step:
#   * in_order with one version of lookahead dispatched 16 prompt groups (64
#     rollouts) onto a 32-slot Harbor backend, so no 8 groups completed together;
#   * 10/69 trials ended in a judge crash on a list-shaped solution.json, which
#     the agent reported as an infrastructure error, so those prompts were
#     re-dispatched instead of scored 0;
#   * finished episodes took 8.6 min (median), 14 min (p90), 24.5 min (max).
# Here: zero lookahead with exactly one step of groups in flight (32 rollouts =
# 32 backend slots), Harbor's agent cap at 0.4 x 1800 s = 12 min (the verifier
# still scores what was delivered), judge crashes scored 0 (Gym bdeafb452), and
# per-step checkpoints with chained cycles.
#   bash scripts/agentic_vbench/smoke_cc_16n_fast_step.sh
set -euo pipefail
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
code_dir=$(cd "$script_dir/../.." && pwd)
export AGENTIC_VBENCH_BACKEND_DIR=${AGENTIC_VBENCH_BACKEND_DIR:-$code_dir/data/agentic_vbench/backend-cc-4}
export NUM_NODES=${NUM_NODES:-16}
export NUM_GEN_NODES=${NUM_GEN_NODES:-8}
export SEGMENT_SIZE=${SEGMENT_SIZE:-8}
export NUM_PROMPTS=${NUM_PROMPTS:-8}
export NUM_GENERATIONS=${NUM_GENERATIONS:-4}
# One step of rollouts fills the backend exactly; nothing queues behind it.
export AGENTIC_VBENCH_CONCURRENCY=${AGENTIC_VBENCH_CONCURRENCY:-$((NUM_PROMPTS * NUM_GENERATIONS))}
# OpenCode compacts when its declared window fills: 64k so compaction happens in
# ordinary episodes while the policy's real window stays 256k.
export AGENTIC_VBENCH_CONTEXT_TOKENS=${AGENTIC_VBENCH_CONTEXT_TOKENS:-65536}
# Harbor stops OpenCode at 0.4 x the task's 1800 s and verifies what was delivered.
export AGENTIC_VBENCH_AGENT_TIMEOUT_MULTIPLIER=${AGENTIC_VBENCH_AGENT_TIMEOUT_MULTIPLIER:-0.4}
export SAVE_PERIOD=${SAVE_PERIOD:-1}
export JOB_CYCLES=${JOB_CYCLES:-6}
export BASE_NAME=${BASE_NAME:-agentic_vbench_ut_cc_16n_fast_step_$(date -u +%Y%m%d-%H%M%S)}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-$BASE_NAME}
exec bash "$script_dir/smoke_cc_32turns_256k.sh" \
  async_rl.sampler.max_lookahead_versions=0 \
  "async_rl.max_inflight_prompts=${NUM_PROMPTS}" \
  "async_rl.max_buffered_rollouts=${NUM_PROMPTS}" \
  "$@"

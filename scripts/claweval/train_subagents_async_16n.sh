#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Second ClawEval gate: 16-node ready-first v2 training. Run only after the 8-node
# synchronous gate has finite rewards/optimizer metrics, TMPE near one, and a
# loadable checkpoint.
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export NUM_NODES=${NUM_NODES:-16}
export NUM_GEN_NODES=${NUM_GEN_NODES:-8}
export SEGMENT_SIZE=${SEGMENT_SIZE:-8}
export NUM_PROMPTS=${NUM_PROMPTS:-8}
export NUM_GENERATIONS=${NUM_GENERATIONS:-4}
export MAX_STEPS=${MAX_STEPS:-1000000}
export SAVE_PERIOD=${SAVE_PERIOD:-1}
export JOB_CYCLES=${JOB_CYCLES:-1}
export SLURM_TIME_LIMIT=${SLURM_TIME_LIMIT:-04:00:00}
export CONTEXT_COMPACTION=true
export CLAW_EVAL_COMPACTION_AUTO=false
export CLAW_EVAL_COMPACTION_PRUNE=false
export CLAW_EVAL_TRAINING_MODE=subagent
export CLAW_EVAL_SUBAGENTS_ENABLED=true
export CLAW_EVAL_CONTEXT_TOKENS=${CLAW_EVAL_CONTEXT_TOKENS:-262144}
export CLAW_EVAL_MAX_TURNS=${CLAW_EVAL_MAX_TURNS:-40}
export CLAW_EVAL_CONCURRENCY=${CLAW_EVAL_CONCURRENCY:-$((NUM_PROMPTS * NUM_GENERATIONS))}
export CLAW_EVAL_DELEGATION_PROMPT=${CLAW_EVAL_DELEGATION_PROMPT:-You may delegate suitable subtasks to subagents when useful. Review and integrate their results before producing the final answer.}
export BASE_NAME=${BASE_NAME:-claweval_subagents_async_16n_$(date -u +%Y%m%d-%H%M%S)}
export WANDB_PROJECT=nemorl-v2-subagent-comapction-debug
export WANDB_ENTITY=${WANDB_ENTITY:-adlr}

inflight_groups=$((NUM_PROMPTS * 3))
agent=env.nemo_gym.claweval_agent.responses_api_agents.claweval_agent
exec bash "$script_dir/run_sc.sh" \
  async_rl.sampler.name=ready_first \
  ~async_rl.sampler.max_lookahead_versions \
  +async_rl.sampler.max_staleness_versions=2 \
  "async_rl.min_groups_for_streaming_train=${NUM_PROMPTS}" \
  "async_rl.max_inflight_prompts=${inflight_groups}" \
  "async_rl.max_buffered_rollouts=${inflight_groups}" \
  +env.nemo_gym.training_mode=subagent \
  "+${agent}.opencode_training_mode=subagent" \
  "+${agent}.opencode_subagents_enabled=true" \
  "+${agent}.opencode_subagent_max_depth=1" \
  "+${agent}.opencode_subagent_max_active=null" \
  "+${agent}.opencode_compaction_auto=false" \
  "+${agent}.opencode_compaction_prune=false" \
  "$@"

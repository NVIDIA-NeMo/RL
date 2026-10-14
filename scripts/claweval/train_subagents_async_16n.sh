#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Second ClawEval gate: 16-node ready-first v2 training. Run only after the 8-node
# synchronous gate has finite rewards/optimizer metrics, TMPE near one, and a
# loadable checkpoint.
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd "$script_dir/../.." && pwd)

export NUM_NODES=${NUM_NODES:-16}
export NUM_GEN_NODES=${NUM_GEN_NODES:-8}
export SEGMENT_SIZE=${SEGMENT_SIZE:-8}
export NUM_PROMPTS=${NUM_PROMPTS:-8}
export NUM_GENERATIONS=${NUM_GENERATIONS:-4}
export MAX_STEPS=${MAX_STEPS:-1000000}
export SAVE_PERIOD=${SAVE_PERIOD:-1}
export JOB_CYCLES=${JOB_CYCLES:-6}
export SLURM_QOS=${SLURM_QOS:-normal}
export SLURM_TIME_LIMIT=${SLURM_TIME_LIMIT:-04:00:00}
export CONTEXT_COMPACTION=true
export CLAW_EVAL_COMPACTION_AUTO=false
export CLAW_EVAL_COMPACTION_PRUNE=false
export CLAW_EVAL_TRAINING_MODE=subagent
export CLAW_EVAL_SUBAGENTS_ENABLED=true
export CLAW_EVAL_CONTEXT_TOKENS=${CLAW_EVAL_CONTEXT_TOKENS:-262144}
export CLAW_EVAL_OUTPUT_TOKENS=${CLAW_EVAL_OUTPUT_TOKENS:-8192}
export CLAW_EVAL_MAX_TURNS=${CLAW_EVAL_MAX_TURNS:-40}
export CLAW_EVAL_MAX_ROW_TOKENS=${CLAW_EVAL_MAX_ROW_TOKENS:-65536}
export CLAW_EVAL_CONCURRENCY=${CLAW_EVAL_CONCURRENCY:-$((NUM_PROMPTS * NUM_GENERATIONS))}
# Match the sync mechanism gate exactly; async validation changes only dispatch
# order and bounded staleness, not whether parent/child rows are exercised.
debug_delegation_prompt="During this task call the task tool exactly once near the beginning to delegate one suitable bounded subtask to the general subagent. "
debug_delegation_prompt+="Do not delegate the full task. Review and integrate the child result. Continue until every required deliverable exists. "
debug_delegation_prompt+="Before producing the final answer verify the deliverables with bash. Do not produce a final answer while required deliverables are missing."
export CLAW_EVAL_DELEGATION_PROMPT=${CLAW_EVAL_DELEGATION_PROMPT:-"${debug_delegation_prompt}"}
export BASE_NAME=${BASE_NAME:-claweval_subagents_async_16n_$(date -u +%Y%m%d-%H%M%S)}
export WANDB_PROJECT=nemorl-v2-subagent-comapction-debug
export WANDB_ENTITY=${WANDB_ENTITY:-adlr}
export NRL_ROUTER_REPLAY_VALIDATE=${NRL_ROUTER_REPLAY_VALIDATE:-1}
export NRL_R3_TRACE=${NRL_R3_TRACE:-1}
export NRL_R3_TRACE_STEPS=${NRL_R3_TRACE_STEPS:-2}
export NRL_R3_TRACE_SAMPLES=${NRL_R3_TRACE_SAMPLES:-2}
export NRL_R3_TRACE_MICROBATCHES=${NRL_R3_TRACE_MICROBATCHES:-2}
export NRL_R3_TRACE_VERIFY_FORWARD=${NRL_R3_TRACE_VERIFY_FORWARD:-1}
export NRL_R3_TRACE_DIR=${NRL_R3_TRACE_DIR:-$repo_dir/results/$BASE_NAME/r3_trace}

inflight_groups=$((NUM_PROMPTS * 3))
agent=env.nemo_gym.claweval_agent.responses_api_agents.claweval_agent
exec bash "$script_dir/run_sc.sh" \
  +policy.router_replay.enabled=true \
  async_rl.sampler.name=ready_first \
  ~async_rl.sampler.max_lookahead_versions \
  +async_rl.sampler.max_staleness_versions=2 \
  "async_rl.min_groups_for_streaming_train=${NUM_PROMPTS}" \
  "async_rl.max_inflight_prompts=${inflight_groups}" \
  "async_rl.max_buffered_rollouts=${inflight_groups}" \
  "+token_capture.max_train_row_tokens=${CLAW_EVAL_MAX_ROW_TOKENS}" \
  +env.nemo_gym.training_mode=subagent \
  "+${agent}.opencode_training_mode=subagent" \
  "+${agent}.opencode_subagents_enabled=true" \
  "+${agent}.opencode_subagent_max_depth=1" \
  "+${agent}.opencode_subagent_max_active=null" \
  "+${agent}.max_turns=${CLAW_EVAL_MAX_TURNS}" \
  "+${agent}.model_context_tokens=${CLAW_EVAL_CONTEXT_TOKENS}" \
  "+${agent}.model_output_capability_tokens=${CLAW_EVAL_OUTPUT_TOKENS}" \
  "+${agent}.opencode_delegation_prompt=${CLAW_EVAL_DELEGATION_PROMPT}" \
  "+${agent}.opencode_compaction_auto=false" \
  "+${agent}.opencode_compaction_prune=false" \
  "$@"

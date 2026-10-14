#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Subagent-RL smoke: the parent OpenCode agent may delegate with the `task` tool
# to depth-one `general` children; every session's calls are captured as their
# own framework-owned context segment and the whole episode trains as one
# scored logical owner (parent segment + one segment per child). OpenCode
# compaction is off (identity arm): the ledger roots a new chain for each child
# the same way it does for a compaction rewrite.
#
# Trainer contract: env.nemo_gym.training_mode=subagent plus the agent override
# block NeMo-RL validates against it (validate_session_training_config). The
# same knobs reach the Harbor runner through the environment so the yaml's
# ${oc.env} defaults and the override block agree.
#
# Same step shaping as smoke_cc_16n_fast_step.sh (zero lookahead, one step of
# groups in flight, 12-minute agent cap, per-step checkpoints, chained cycles).
#   bash scripts/agentic_vbench/smoke_subagents_fast_step.sh
set -euo pipefail
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
code_dir=$(cd "$script_dir/../.." && pwd)
export AGENTIC_VBENCH_BACKEND_DIR=${AGENTIC_VBENCH_BACKEND_DIR:-$code_dir/data/agentic_vbench/backend-cc-4}
# Framework-owned segments on, OpenCode compaction/pruning off, subagents on.
export CONTEXT_COMPACTION=true
export AGENTIC_VBENCH_COMPACTION_AUTO=false
export AGENTIC_VBENCH_COMPACTION_PRUNE=false
export AGENTIC_VBENCH_TRAINING_MODE=subagent
export AGENTIC_VBENCH_SUBAGENTS_ENABLED=true
# Keep the pinned task row verbatim; OpenCode applies this only to the primary
# build agent so the dedicated smoke reliably exercises child-trace training.
delegation_prompt_default="You may delegate suitable subtasks to subagents when useful. "
delegation_prompt_default+="Review and integrate their results before producing the final answer."
export AGENTIC_VBENCH_DELEGATION_PROMPT=${AGENTIC_VBENCH_DELEGATION_PROMPT:-"${delegation_prompt_default}"}
subagent_prompt_default="Complete only the delegated bounded subtask. Use tools early and keep every response concise. "
subagent_prompt_default+="Return a concise result to the parent before the output limit."
export AGENTIC_VBENCH_SUBAGENT_PROMPT=${AGENTIC_VBENCH_SUBAGENT_PROMPT:-"${subagent_prompt_default}"}
export NUM_NODES=${NUM_NODES:-16}
export NUM_GEN_NODES=${NUM_GEN_NODES:-8}
export SEGMENT_SIZE=${SEGMENT_SIZE:-8}
export NUM_PROMPTS=${NUM_PROMPTS:-8}
export NUM_GENERATIONS=${NUM_GENERATIONS:-4}
# Keep GRPO allocation-bounded. Per-step checkpoints plus chained Slurm cycles
# provide recovery; an artificial smoke-step cap must not terminate training.
if [[ ${MAX_STEPS:-1000000} != 1000000 ]]; then
  echo "ERROR: subagent training requires MAX_STEPS=1000000; 20 steps is an observation gate, not a stop" >&2
  exit 2
fi
export MAX_STEPS=1000000
# Dataset cardinality is independent of the optimizer horizon.  launch_sc sets
# max_num_epochs=null, so the dataloader cycles these rows without materializing
# one copy per possible GRPO step.
export TRAIN_REPEATS=${TRAIN_REPEATS:-4}
# Children run concurrently with each other inside one episode, so a step's
# episodes can hold more containers than rollouts; Harbor backend slots bound it.
export AGENTIC_VBENCH_CONCURRENCY=${AGENTIC_VBENCH_CONCURRENCY:-$((NUM_PROMPTS * NUM_GENERATIONS))}
# No compaction: the declared window is the policy's real 256k window.
export AGENTIC_VBENCH_CONTEXT_TOKENS=${AGENTIC_VBENCH_CONTEXT_TOKENS:-262144}
export AGENTIC_VBENCH_OUTPUT_TOKENS=${AGENTIC_VBENCH_OUTPUT_TOKENS:-8192}
export AGENTIC_VBENCH_MAX_TURNS=${AGENTIC_VBENCH_MAX_TURNS:-32}
# Full task budget (1800 s per episode); the GPU reaper exemption in launch_sc.sh
# removes the kill-window reason for a shorter cap.
export AGENTIC_VBENCH_AGENT_TIMEOUT_MULTIPLIER=${AGENTIC_VBENCH_AGENT_TIMEOUT_MULTIPLIER:-1.0}
# Without compaction a 32-turn episode is one training row of its whole context.
# An 81,920-token ceiling was still unsafe on the 8-node TP8/EP16/CP1 learner:
# run 2238533 completed step 0, then a near-80k multimodal row OOM'd the MoE
# forward pass in step 1.  Keep the smoke at the production-tested 64k envelope;
# rejected owners train nothing in that step instead of crashing all ranks.
export SUBAGENT_MAX_ROW_TOKENS=${SUBAGENT_MAX_ROW_TOKENS:-65536}
export SAVE_PERIOD=${SAVE_PERIOD:-1}
export JOB_CYCLES=${JOB_CYCLES:-6}
export SLURM_QOS=${SLURM_QOS:-short}
export SLURM_TIME_LIMIT=${SLURM_TIME_LIMIT:-02:00:00}
export BASE_NAME=${BASE_NAME:-agentic_vbench_subagents_16n_fast_step_$(date -u +%Y%m%d-%H%M%S)}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-$BASE_NAME}
agent=env.nemo_gym.agentic_vbench_agent.responses_api_agents.agentic_vbench_agent
exec bash "$script_dir/smoke_off_32turns_256k.sh" \
  async_rl.sampler.max_lookahead_versions=0 \
  "async_rl.max_inflight_prompts=${NUM_PROMPTS}" \
  "async_rl.max_buffered_rollouts=${NUM_PROMPTS}" \
  "+token_capture.max_train_row_tokens=${SUBAGENT_MAX_ROW_TOKENS}" \
  +env.nemo_gym.training_mode=subagent \
  "+${agent}.opencode_training_mode=subagent" \
  "+${agent}.opencode_subagents_enabled=true" \
  "+${agent}.opencode_subagent_max_depth=1" \
  "+${agent}.opencode_subagent_max_active=null" \
  "+${agent}.max_turns=${AGENTIC_VBENCH_MAX_TURNS}" \
  "+${agent}.model_context_tokens=${AGENTIC_VBENCH_CONTEXT_TOKENS}" \
  "+${agent}.model_output_capability_tokens=${AGENTIC_VBENCH_OUTPUT_TOKENS}" \
  "+${agent}.opencode_delegation_prompt=${AGENTIC_VBENCH_DELEGATION_PROMPT}" \
  "+${agent}.opencode_subagent_prompt=${AGENTIC_VBENCH_SUBAGENT_PROMPT}" \
  "+${agent}.opencode_compaction_auto=false" \
  "+${agent}.opencode_compaction_prune=false" \
  "$@" \
  grpo.max_num_steps=1000000 \
  grpo.max_num_epochs=null

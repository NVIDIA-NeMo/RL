#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Second production gate for subagent training. Run only after the 8-node sync
# gate keeps every observed raw TMPE value at or below 1.03.
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd "$script_dir/../.." && pwd)

export CONFIG_IN_CONTAINER=${CONFIG_IN_CONTAINER:-examples/configs/recipes/vlm/vlm_grpo-nemotron-super-omni-120ba12b-16n8g-megatron-agentic-vbench-sc-256k.v1.yaml}
export NUM_NODES=${NUM_NODES:-16}
export NUM_GEN_NODES=${NUM_GEN_NODES:-8}
export SEGMENT_SIZE=${SEGMENT_SIZE:-8}
export NUM_PROMPTS=${NUM_PROMPTS:-8}
export NUM_GENERATIONS=${NUM_GENERATIONS:-4}
export MAX_STEPS=${MAX_STEPS:-1000000}
export TRAIN_REPEATS=${TRAIN_REPEATS:-4}
# Keep the first async comparison on the same four-task causal-gate manifest as
# sync v19. Eight prompt groups draw repeated rows from that unchanged set.
export TRAIN_LIMIT=${TRAIN_LIMIT:-4}
export SAVE_PERIOD=${SAVE_PERIOD:-1}
export CHECKPOINT_KEEP_TOP_K=${CHECKPOINT_KEEP_TOP_K:-2}
export JOB_CYCLES=${JOB_CYCLES:-6}
export SLURM_QOS=${SLURM_QOS:-normal}
export SLURM_TIME_LIMIT=${SLURM_TIME_LIMIT:-24:00:00}
export MIN_SLURM_TIME_SECONDS=${MIN_SLURM_TIME_SECONDS:-86400}
export MIN_CHECKPOINT_BUDGET_SECONDS=${MIN_CHECKPOINT_BUDGET_SECONDS:-82800}
export REQUIRE_RESUMABLE_JOB_CYCLES=${REQUIRE_RESUMABLE_JOB_CYCLES:-true}
export AGENTIC_VBENCH_MAX_TURNS=${AGENTIC_VBENCH_MAX_TURNS:-32}
# Keep the first async gate behaviorally identical to the qualified sync gate.
# Only the sampler topology and permitted policy staleness change below.
debug_delegation_prompt="Your first action must be exactly one task tool call before bash read or any other tool. "
debug_delegation_prompt+="Delegate only mechanical media preparation to the general subagent. Ask it to write a duration table at /workspace/work/subagent_durations.tsv and labeled contact sheets under /workspace/work/subagent_contact_sheets. "
debug_delegation_prompt+="Also ask it to write and syntax check /workspace/work/build_solution.py which accepts one chosen clip ID per storyboard slot and mechanically creates solution.mp4 plus a truthful solution.json from the full clips. Require solution.json to be a top-level JSON object with a segments list and never a bare JSON list. Tell the child not to inspect or choose clips and not to create either deliverable itself. "
debug_delegation_prompt+="When the child returns read its duration table and inspect every prepared contact sheet exactly once in at most two parallel read batches. Choose the best clip for every slot once without revisiting decisions. "
debug_delegation_prompt+="Hard transition rule: after the tool results containing the final contact sheets your very next response must contain a bash tool call invoking python /workspace/work/build_solution.py followed by the chosen clip IDs in slot order. Do not write clip by clip analysis or comparison before that tool call and keep private selection reasoning under 200 words. "
debug_delegation_prompt+="Then use one bash tool call to verify both output files exist and use Python to assert solution.json parses as a top-level JSON object with a segments list containing only objects. Keep every response short. "
debug_delegation_prompt+="Never end while a required deliverable is missing. After verification succeeds return one brief final sentence."
export AGENTIC_VBENCH_DELEGATION_PROMPT=${AGENTIC_VBENCH_DELEGATION_PROMPT:-"${debug_delegation_prompt}"}
debug_subagent_prompt="Complete only mechanical media preparation and never choose candidate clips. Never create solution.mp4 or solution.json. "
debug_subagent_prompt+="Use ffprobe in one shell workflow to write /workspace/work/subagent_durations.tsv with one clip and duration per line. "
debug_subagent_prompt+="Use one batched frame extraction workflow to create clearly labeled compact contact sheets for every candidate under /workspace/work/subagent_contact_sheets. "
debug_subagent_prompt+="Write /workspace/work/build_solution.py which accepts exactly one numeric clip ID per storyboard slot. It must concatenate those full clips in order into /workspace/output/solution.mp4 and write the required contiguous full-duration segments to /workspace/output/solution.json as a top-level JSON object with a segments list and never a bare JSON list. Syntax check the script but do not run it. "
debug_subagent_prompt+="Do not view or analyze any extracted image and do not discuss the storyboard. Verify the duration table contact sheets and builder exist. "
debug_subagent_prompt+="Return only their paths and counts in at most four short lines then stop."
export AGENTIC_VBENCH_SUBAGENT_PROMPT=${AGENTIC_VBENCH_SUBAGENT_PROMPT:-"${debug_subagent_prompt}"}
export BASE_NAME=${BASE_NAME:-agentic_vbench_v2_subagents_async_16n_$(date -u +%Y%m%d-%H%M%S)-$$}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-$BASE_NAME}
export WANDB_PROJECT=nemorl-v2-subagent-comapction-debug
export WANDB_ENTITY=${WANDB_ENTITY:-adlr}
# Super-GA is MoE: preserve generation-time expert routes when Megatron scores
# and trains stale async rows. Trace the first two steps; never mask bad TMPE.
export NRL_ROUTER_REPLAY_VALIDATE=${NRL_ROUTER_REPLAY_VALIDATE:-1}
export NRL_R3_TRACE=${NRL_R3_TRACE:-1}
export NRL_R3_TRACE_STEPS=${NRL_R3_TRACE_STEPS:-2}
export NRL_R3_TRACE_SAMPLES=${NRL_R3_TRACE_SAMPLES:-2}
export NRL_R3_TRACE_MICROBATCHES=${NRL_R3_TRACE_MICROBATCHES:-2}
export NRL_R3_TRACE_VERIFY_FORWARD=${NRL_R3_TRACE_VERIFY_FORWARD:-1}
export NRL_R3_TRACE_DIR=${NRL_R3_TRACE_DIR:-$repo_dir/results/$BASE_NAME/r3_trace}

inflight_groups=$((NUM_PROMPTS * 3))
exec bash "$script_dir/smoke_subagents_fast_step.sh" \
  +policy.router_replay.enabled=true \
  async_rl.sampler.name=ready_first \
  ~async_rl.sampler.max_lookahead_versions \
  +async_rl.sampler.max_staleness_versions=2 \
  "async_rl.min_groups_for_streaming_train=${NUM_PROMPTS}" \
  "async_rl.max_inflight_prompts=${inflight_groups}" \
  "async_rl.max_buffered_rollouts=${inflight_groups}" \
  "$@"

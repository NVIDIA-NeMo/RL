#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# First production gate for subagent training: 8 nodes, synchronous/in-order
# rollouts, explicit delegation, and one checkpoint per optimizer update.
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd "$script_dir/../.." && pwd)

export CONFIG_IN_CONTAINER=${CONFIG_IN_CONTAINER:-examples/configs/recipes/vlm/vlm_grpo-nemotron-super-omni-120ba12b-16n8g-megatron-agentic-vbench-sc-256k.v1.yaml}
export NUM_NODES=${NUM_NODES:-8}
export NUM_GEN_NODES=${NUM_GEN_NODES:-4}
export SEGMENT_SIZE=${SEGMENT_SIZE:-4}
export NUM_PROMPTS=${NUM_PROMPTS:-4}
export NUM_GENERATIONS=${NUM_GENERATIONS:-4}
export MAX_STEPS=${MAX_STEPS:-1000000}
# max_num_epochs is null in launch_sc.sh, so the dataloader cycles this compact
# manifest independently of the production optimizer horizon.
export TRAIN_REPEATS=${TRAIN_REPEATS:-4}
# Keep the mechanism gate short and reward-bearing.  The first four assembly
# tasks include task10 and task12, the two prompts that produced the positive
# verifier rewards in the validated comparison run.  The full assembly set is
# restored for the longer qualification run after this gate passes.
export TRAIN_LIMIT=${TRAIN_LIMIT:-4}
export SAVE_PERIOD=${SAVE_PERIOD:-1}
export CHECKPOINT_KEEP_TOP_K=${CHECKPOINT_KEEP_TOP_K:-2}
# Slurm allocations are resumable execution segments, not the GRPO horizon.
# The old 2 h short-QoS default derived a 1 h 40 min controller deadline and
# cleanly stopped v19 after step 4.  Use the longest batch_long allocation and
# normal QoS; per-step checkpoints plus the serialized array continue the same
# 1,000,000-step run across allocation boundaries.
export JOB_CYCLES=${JOB_CYCLES:-6}
export SLURM_QOS=${SLURM_QOS:-normal}
export SLURM_TIME_LIMIT=${SLURM_TIME_LIMIT:-7-00:00:00}
export MIN_SLURM_TIME_SECONDS=${MIN_SLURM_TIME_SECONDS:-86400}
export MIN_CHECKPOINT_BUDGET_SECONDS=${MIN_CHECKPOINT_BUDGET_SECONDS:-86400}
export REQUIRE_RESUMABLE_JOB_CYCLES=${REQUIRE_RESUMABLE_JOB_CYCLES:-true}
# Use the recipe's full turn budget.  A 12-turn shortcut caused every completed
# v8 rollout to terminate before solving the task, with zero reward and no
# delegation, so it cannot serve as the subagent qualification gate.
export AGENTIC_VBENCH_MAX_TURNS=${AGENTIC_VBENCH_MAX_TURNS:-32}
# This wrapper is specifically the mechanism gate, so require one bounded child
# call.  The production recipe remains opt-in ("may delegate") after the parent
# and child training rows have been proven healthy here.
# Keep this Hydra override free of commas and apostrophes.  The launcher passes
# the value as one argv entry, but Hydra still parses punctuation in the value;
# an apostrophe in the v9 prompt made the live driver fail before setup.
debug_delegation_prompt="Your first action must be exactly one task tool call before bash read or any other tool. "
debug_delegation_prompt+="Delegate only mechanical media preparation to the general subagent. Ask it to write a duration table at /workspace/work/subagent_durations.tsv and labeled contact sheets under /workspace/work/subagent_contact_sheets. "
debug_delegation_prompt+="Also ask it to write and syntax check /workspace/work/build_solution.py which accepts one chosen clip ID per storyboard slot and mechanically creates solution.mp4 plus a truthful solution.json from the full clips. Tell the child not to inspect or choose clips and not to create either deliverable itself. "
debug_delegation_prompt+="When the child returns read its duration table and inspect every prepared contact sheet exactly once in at most two parallel read batches. Choose the best clip for every slot once without revisiting decisions. "
debug_delegation_prompt+="Hard transition rule: after the tool results containing the final contact sheets your very next response must contain a bash tool call invoking python /workspace/work/build_solution.py followed by the chosen clip IDs in slot order. Do not write clip by clip analysis or comparison before that tool call and keep private selection reasoning under 200 words. "
debug_delegation_prompt+="Then use one bash tool call to verify both output files exist and solution.json parses. Keep every response short. "
debug_delegation_prompt+="Never end while a required deliverable is missing. After verification succeeds return one brief final sentence."
export AGENTIC_VBENCH_DELEGATION_PROMPT=${AGENTIC_VBENCH_DELEGATION_PROMPT:-"${debug_delegation_prompt}"}
debug_subagent_prompt="Complete only mechanical media preparation and never choose candidate clips. Never create solution.mp4 or solution.json. "
debug_subagent_prompt+="Use ffprobe in one shell workflow to write /workspace/work/subagent_durations.tsv with one clip and duration per line. "
debug_subagent_prompt+="Use one batched frame extraction workflow to create clearly labeled compact contact sheets for every candidate under /workspace/work/subagent_contact_sheets. "
debug_subagent_prompt+="Write /workspace/work/build_solution.py which accepts exactly one numeric clip ID per storyboard slot. It must concatenate those full clips in order into /workspace/output/solution.mp4 and write the required contiguous full-duration segments to /workspace/output/solution.json. Syntax check the script but do not run it. "
debug_subagent_prompt+="Do not view or analyze any extracted image and do not discuss the storyboard. Verify the duration table contact sheets and builder exist. "
debug_subagent_prompt+="Return only their paths and counts in at most four short lines then stop."
export AGENTIC_VBENCH_SUBAGENT_PROMPT=${AGENTIC_VBENCH_SUBAGENT_PROMPT:-"${debug_subagent_prompt}"}
export BASE_NAME=${BASE_NAME:-agentic_vbench_v2_subagents_sync_8n_$(date -u +%Y%m%d-%H%M%S)-$$}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-$BASE_NAME}
export WANDB_PROJECT=nemorl-v2-subagent-comapction-debug
export WANDB_ENTITY=${WANDB_ENTITY:-adlr}
# Super is a 512-expert MoE, while this recipe generates with vLLM TP4 and
# scores/trains with Megatron TP8/EP16.  Replay vLLM's exact expert choices so
# rollout-vs-policy parity does not depend on two different router executions.
# Keep strict route and forward checks on for the first two qualification steps;
# tools/check_r3_trace.py validates the emitted evidence after the run.
export NRL_ROUTER_REPLAY_VALIDATE=${NRL_ROUTER_REPLAY_VALIDATE:-1}
export NRL_R3_TRACE=${NRL_R3_TRACE:-1}
export NRL_R3_TRACE_STEPS=${NRL_R3_TRACE_STEPS:-2}
export NRL_R3_TRACE_SAMPLES=${NRL_R3_TRACE_SAMPLES:-2}
export NRL_R3_TRACE_MICROBATCHES=${NRL_R3_TRACE_MICROBATCHES:-2}
export NRL_R3_TRACE_VERIFY_FORWARD=${NRL_R3_TRACE_VERIFY_FORWARD:-1}
export NRL_R3_TRACE_DIR=${NRL_R3_TRACE_DIR:-$repo_dir/results/$BASE_NAME/r3_trace}

exec bash "$script_dir/smoke_subagents_fast_step.sh" \
  +policy.router_replay.enabled=true \
  async_rl.sampler.name=in_order \
  async_rl.sampler.max_lookahead_versions=0 \
  "async_rl.min_groups_for_streaming_train=${NUM_PROMPTS}" \
  "async_rl.max_inflight_prompts=${NUM_PROMPTS}" \
  "async_rl.max_buffered_rollouts=${NUM_PROMPTS}" \
  async_rl.rollout_failure.on_dropped_prompt=replace \
  async_rl.rollout_failure.max_replacement_attempts=4 \
  async_rl.rollout_failure.replacement_reserve_prompts=1 \
  "$@"

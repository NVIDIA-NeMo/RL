#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# First production gate for subagent training: 8 nodes, synchronous/in-order
# rollouts, explicit delegation, and full recovery checkpoints every ten updates.
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd "$script_dir/../.." && pwd)

export SLURM_ACCOUNT=${SLURM_ACCOUNT:-nemotron_n4_omni}
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
export SAVE_PERIOD=${SAVE_PERIOD:-20}
export CHECKPOINT_KEEP_TOP_K=${CHECKPOINT_KEEP_TOP_K:-0}
# Slurm allocations are resumable execution segments, not the GRPO horizon.
# The old 2 h short-QoS default derived a 1 h 40 min controller deadline and
# cleanly stopped v19 after step 4. Use day-long normal-QoS segments; ten-update
# recovery checkpoints and the array continue the same 1,000,000-step run
# across allocation boundaries. Six segments fit inside the dedicated 7-day
# cpu-long task backend while leaving roughly one day of scheduling cushion.
export JOB_CYCLES=${JOB_CYCLES:-6}
export SLURM_QOS=${SLURM_QOS:-normal}
export SLURM_TIME_LIMIT=${SLURM_TIME_LIMIT:-24:00:00}
export MIN_SLURM_TIME_SECONDS=${MIN_SLURM_TIME_SECONDS:-86400}
export MIN_CHECKPOINT_BUDGET_SECONDS=${MIN_CHECKPOINT_BUDGET_SECONDS:-82800}
export REQUIRE_RESUMABLE_JOB_CYCLES=${REQUIRE_RESUMABLE_JOB_CYCLES:-true}
# Use the recipe's full turn budget.  A 12-turn shortcut caused every completed
# v8 rollout to terminate before solving the task, with zero reward and no
# delegation, so it cannot serve as the subagent qualification gate.
export AGENTIC_VBENCH_MAX_TURNS=${AGENTIC_VBENCH_MAX_TURNS:-32}
# Bound only the observation preview returned to OpenCode. The complete tool
# output remains on disk, so commands and verifier behavior are unchanged.
export AGENTIC_VBENCH_TOOL_OUTPUT_MAX_LINES=${AGENTIC_VBENCH_TOOL_OUTPUT_MAX_LINES:-400}
export AGENTIC_VBENCH_TOOL_OUTPUT_MAX_BYTES=${AGENTIC_VBENCH_TOOL_OUTPUT_MAX_BYTES:-8192}
# This wrapper is specifically the mechanism gate, so require one child to own
# the benchmark task end to end.  In v24, splitting media preparation from
# selection made 8/16 rollouts exhaust the parent response budget before either
# deliverable was written.  Letting the child write the artifacts also proves
# that child-session tool calls and side effects are preserved for scoring.
# The production recipe remains opt-in ("may delegate") after the parent and
# child training rows have been proven healthy here.
# Keep this Hydra override free of commas and apostrophes.  The launcher passes
# the value as one argv entry, but Hydra still parses punctuation in the value;
# an apostrophe in the v9 prompt made the live driver fail before setup.
debug_delegation_prompt="Your first action must be exactly one task tool call before bash read or any other tool. "
debug_delegation_prompt+="Delegate the complete video assembly task to the general subagent. Include the original storyboard slot requirements and require the child to inspect the supplied clips choose every slot create solution.mp4 and create a truthful solution.json before returning. Require solution.json to be a top-level JSON object with a segments list and never a bare JSON list. "
debug_delegation_prompt+="When the child returns do not repeat its visual analysis. Immediately use one bash tool call to verify both output files exist and use Python to assert solution.json parses as a top-level JSON object with a segments list containing only objects. If either artifact is missing or invalid call the general subagent once more with a repair request that forbids new visual inspection and requires immediate creation from existing work in at most four tool calls. "
debug_delegation_prompt+="Keep every response short. Never end while a required deliverable is missing. After verification succeeds return one brief final sentence."
export AGENTIC_VBENCH_DELEGATION_PROMPT=${AGENTIC_VBENCH_DELEGATION_PROMPT:-"${debug_delegation_prompt}"}
debug_subagent_prompt="Complete the delegated AgenticVBench task end to end and create both required deliverables before returning. Begin with tools and do not narrate candidate comparisons. "
debug_subagent_prompt+="Use one batched ffprobe command and one batched ffmpeg -hide_banner -loglevel error -nostats frame extraction workflow while suppressing routine output. Create no more than four labeled overview JPEGs under /workspace/work/subagent_contact_sheets. Each JPEG must be at most 1280 by 720 pixels and each candidate must have four chronological thumbnails with its numeric clip ID. Read every overview exactly once in one parallel read batch. After that read batch never extract or read another frame and never create diagnostic or comparison images. "
debug_subagent_prompt+="Choose the best full clip for every storyboard slot silently. Your very next response after the overview read results must call bash to create /workspace/output/solution.mp4 and /workspace/output/solution.json. Concatenate each selected full clip once. Use source ffprobe durations for truthful contiguous JSON ranges and do not tune timestamps frame counts or sub-frame MP4 duration rounding. The JSON must be a top-level JSON object with a segments list and never a bare JSON list. "
debug_subagent_prompt+="Do not use todowrite and do not retry cosmetic labels. Keep private selection reasoning under 200 words. Use one final bash call to verify both artifacts exist and that solution.json parses. Then return only a brief completion result."
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

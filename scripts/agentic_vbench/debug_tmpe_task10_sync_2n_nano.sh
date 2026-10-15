#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Exact-row Nano replay for the nine-image Agentic-VBench TMPE outlier.
#
# This deliberately restores the v7 per-candidate contact-sheet instructions:
# task10 then places nine 1280x720 images in one parent trace.  It is a parity
# reproducer, not the bounded-media production prompt.
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd "$script_dir/../.." && pwd)

export TRAIN_PATH=${TRAIN_PATH:-$repo_dir/data/agentic_vbench/train/assembly_task10_x4.jsonl}
export VAL_PATH=${VAL_PATH:-$TRAIN_PATH}
export AGENTIC_VBENCH_TASKS=agentic-vbench-assembly-task10
export TRAIN_LIMIT=1
export TRAIN_REPEATS=4
if [[ ! -s $TRAIN_PATH ]]; then
  python3 "$script_dir/build_train_dataset.py" \
    --benchmark-root "$repo_dir/data/agentic_vbench/agentic-vbench" \
    --tasks agentic-vbench-assembly-task10 \
    --repeats 4 \
    --output "$TRAIN_PATH"
fi

replay_delegation_prompt="Your first action must be exactly one task tool call before bash read or any other tool. "
replay_delegation_prompt+="Delegate only mechanical media preparation to the general subagent. Ask it to write a duration table at /workspace/work/subagent_durations.tsv and labeled contact sheets under /workspace/work/subagent_contact_sheets. "
replay_delegation_prompt+="Also ask it to write and syntax check /workspace/work/build_solution.py which accepts one chosen clip ID per storyboard slot and mechanically creates solution.mp4 plus a truthful solution.json from the full clips. Require solution.json to be a top-level JSON object with a segments list and never a bare JSON list. Tell the child not to inspect or choose clips and not to create either deliverable itself. "
replay_delegation_prompt+="When the child returns read its duration table and inspect every prepared contact sheet exactly once in at most two parallel read batches. Choose the best clip for every slot once without revisiting decisions. "
replay_delegation_prompt+="Hard transition rule: after the tool results containing the final contact sheets your very next response must contain a bash tool call invoking python /workspace/work/build_solution.py followed by the chosen clip IDs in slot order. Do not write clip by clip analysis or comparison before that tool call and keep private selection reasoning under 200 words. "
replay_delegation_prompt+="Then use one bash tool call to verify both output files exist and use Python to assert solution.json parses as a top-level JSON object with a segments list containing only objects. Keep every response short. Never end while a required deliverable is missing. After verification succeeds return one brief final sentence."
export AGENTIC_VBENCH_DELEGATION_PROMPT=${AGENTIC_VBENCH_DELEGATION_PROMPT:-"$replay_delegation_prompt"}

replay_subagent_prompt="Complete only mechanical media preparation and never choose candidate clips. Never create solution.mp4 or solution.json. "
replay_subagent_prompt+="Use ffprobe in one shell workflow to write /workspace/work/subagent_durations.tsv with one clip and duration per line. "
replay_subagent_prompt+="Use one batched frame extraction workflow to create clearly labeled compact contact sheets for every candidate under /workspace/work/subagent_contact_sheets. "
replay_subagent_prompt+="Write /workspace/work/build_solution.py which accepts exactly one numeric clip ID per storyboard slot. It must concatenate those full clips in order into /workspace/output/solution.mp4 and write the required contiguous full-duration segments to /workspace/output/solution.json as a top-level JSON object with a segments list and never a bare JSON list. Syntax check the script but do not run it. "
replay_subagent_prompt+="Do not view or analyze any extracted image and do not discuss the storyboard. Verify the duration table contact sheets and builder exist. Return only their paths and counts in at most four short lines then stop."
export AGENTIC_VBENCH_SUBAGENT_PROMPT=${AGENTIC_VBENCH_SUBAGENT_PROMPT:-"$replay_subagent_prompt"}

export AGENTIC_VBENCH_TOOL_OUTPUT_MAX_LINES=${AGENTIC_VBENCH_TOOL_OUTPUT_MAX_LINES:-400}
export AGENTIC_VBENCH_TOOL_OUTPUT_MAX_BYTES=${AGENTIC_VBENCH_TOOL_OUTPUT_MAX_BYTES:-8192}
export NRL_R3_TRACE_STEPS=${NRL_R3_TRACE_STEPS:-20}
export NRL_R3_TRACE_SAMPLES=${NRL_R3_TRACE_SAMPLES:-16}
export NRL_R3_TRACE_MICROBATCHES=${NRL_R3_TRACE_MICROBATCHES:-16}

export BASE_NAME=${BASE_NAME:-agentic_vbench_subagents_nano_sync2_$(date -u +%Y%m%d-%H%M%S)_task10_radio_tp_fix}
export RESULTS_DIR=${RESULTS_DIR:-$repo_dir/results/$BASE_NAME}
export CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-$RESULTS_DIR}
export NEMO_RL_TMPE_OUTLIER_DUMP_PATH=${NEMO_RL_TMPE_OUTLIER_DUMP_PATH:-$RESULTS_DIR/tmpe_outliers}
export NEMO_RL_TMPE_OUTLIER_THRESHOLD=${NEMO_RL_TMPE_OUTLIER_THRESHOLD:-1.03}
export RUN_TMPE_OUTLIER_TESTS=${RUN_TMPE_OUTLIER_TESTS:-1}

exec bash "$script_dir/debug_subagents_sync_2n_nano.sh" "$@"

#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Context-compaction training on all 100 AgenticVBench tasks (assembly, repair,
# repurpose, sequencing) with 16 prompts x 8 generations = 128 rows per
# optimizer step, 16 nodes (8 train + 8 generation).
#
# Shaping, from run 2183521 (8x4, steps ~22 min: ~10 min generation at 32
# concurrent episodes, ~4 min training for 32 rows, 10 min startup):
#   * 128 rollouts per step at 32 concurrent episodes = four waves (~40-50 min);
#     the backend (64 CPUs / 256 GB) could not serve 64 containers reliably;
#   * training 128 rows ~15 min; one version of lookahead (recipe default:
#     two steps of groups in flight) overlaps the next step's generation with it;
#   * 12 h cycles chained 4x with per-step checkpoints; the controller stops
#     itself on the checkpoint budget and the next cycle resumes.
# Non-assembly families are scored by the Hub judges (nvinference-hybrid:
# Claude Opus 4.7 + Gemini 3.5 Flash), which need ~/.config/agentic-vbench/nvinference.env.
#   bash scripts/agentic_vbench/train_cc_all_16x8.sh
set -euo pipefail
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
code_dir=$(cd "$script_dir/../.." && pwd)
export AGENTIC_VBENCH_BACKEND_DIR=${AGENTIC_VBENCH_BACKEND_DIR:-$code_dir/data/agentic_vbench/backend-cc-7}
export AGENTIC_VBENCH_TASKS=${AGENTIC_VBENCH_TASKS:-all}
# The assembly smoke scripts this launcher chains through blank the judge
# protocol (assembly has a native judge). Repurpose/sequencing verifiers need
# the Hub judge: without it their native test.sh dies in its diagnostic block
# under set -euo pipefail and writes no reward (run 2210161: 33/33 repurpose
# trials). The protocol also patches that diagnostic and injects the Hub key.
export AGENTIC_VBENCH_JUDGE_PROTOCOL=${AGENTIC_VBENCH_JUDGE_PROTOCOL:-nvinference-hybrid}
export NUM_NODES=${NUM_NODES:-16}
export NUM_GEN_NODES=${NUM_GEN_NODES:-8}
export SEGMENT_SIZE=${SEGMENT_SIZE:-8}
export NUM_PROMPTS=${NUM_PROMPTS:-16}
export NUM_GENERATIONS=${NUM_GENERATIONS:-8}
export MAX_STEPS=${MAX_STEPS:-60}
# 100 tasks x 10 repeats = 1000 rows >= 60 steps x 16 prompts (data.shuffle=false,
# so each step is 16 distinct tasks and an epoch is ~6 steps).
export TRAIN_REPEATS=${TRAIN_REPEATS:-10}
# The chained smoke script limits the task list to NUM_PROMPTS tasks (its
# "first N" smoke dataset); pin the full all-task dataset explicitly instead.
export TRAIN_PATH=${TRAIN_PATH:-$code_dir/data/agentic_vbench/train/all_x${TRAIN_REPEATS}.jsonl}
if [[ ! -s $TRAIN_PATH ]]; then
  python3 "$script_dir/build_train_dataset.py" --benchmark-root "$code_dir/data/agentic_vbench/agentic-vbench" \
    --tasks all --repeats "$TRAIN_REPEATS" --output "$TRAIN_PATH"
fi
# Backend capacity: 64 concurrent episodes saturate even the whole-node backend
# (120 CPUs at load 94, container API unresponsive for 90 s, probe/env-start
# timeouts; run 2212447) because the video tasks run ffmpeg/Whisper in their
# containers. 40 keeps the API responsive; 128 rollouts = ~3 waves (~75 min),
# within the training GPUs' 240-min idle exemption.
export AGENTIC_VBENCH_CONCURRENCY=${AGENTIC_VBENCH_CONCURRENCY:-40}
# OpenCode compacts when its declared 64k window fills; the policy keeps 256k.
export AGENTIC_VBENCH_CONTEXT_TOKENS=${AGENTIC_VBENCH_CONTEXT_TOKENS:-65536}
# Full task budget (1800 s per episode): the GPU reaper exemption in launch_sc.sh
# removes the reason to shorten episodes. 0.4 (12 min) had stopped 41% of
# episodes at 64 concurrent; 1.0 scores every episode on merit.
export AGENTIC_VBENCH_AGENT_TIMEOUT_MULTIPLIER=${AGENTIC_VBENCH_AGENT_TIMEOUT_MULTIPLIER:-1.0}
export SLURM_TIME_LIMIT=${SLURM_TIME_LIMIT:-12:00:00}
export SAVE_PERIOD=${SAVE_PERIOD:-1}
export JOB_CYCLES=${JOB_CYCLES:-4}
export BASE_NAME=${BASE_NAME:-agentic_vbench_ut_cc_all_16x8_$(date -u +%Y%m%d-%H%M%S)}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-$BASE_NAME}
exec bash "$script_dir/smoke_cc_32turns_256k.sh" "$@"

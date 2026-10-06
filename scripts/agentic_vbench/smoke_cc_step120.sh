#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Two optimizer steps of SingleController GRPO with context compaction from the
# step-120 weights on pinned repair/assembly tasks, through the rootless Podman
# backend started by submit_backend.sh. OpenCode is told a short context so it
# compacts inside a 12-turn episode and the rewritten-context reconstruction is
# exercised, not just ordinary capture.
#   AGENTIC_VBENCH_BACKEND_DIR=$PWD/data/agentic_vbench/backend-cc-smoke \
#     DRY_RUN=1 bash scripts/agentic_vbench/smoke_cc_step120.sh
set -euo pipefail
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
code_dir=$(cd "$script_dir/../.." && pwd)
export AVB_DATA_ROOT=${AVB_DATA_ROOT:-$code_dir/data/agentic_vbench}
export AGENTIC_VBENCH_BACKEND_DIR=${AGENTIC_VBENCH_BACKEND_DIR:-$AVB_DATA_ROOT/backend-cc-smoke}
export AGENTIC_VBENCH_TASKS=${AGENTIC_VBENCH_TASKS:-"repair assembly"}
# Repair/assembly verifiers are native checks; no Hub judge (and no credentials) needed.
export AGENTIC_VBENCH_JUDGE_PROTOCOL=${AGENTIC_VBENCH_JUDGE_PROTOCOL-}
export MAX_STEPS=${MAX_STEPS:-2}
export NUM_PROMPTS=${NUM_PROMPTS:-8}
export NUM_GENERATIONS=${NUM_GENERATIONS:-4}
export TRAIN_LIMIT=${TRAIN_LIMIT:-$NUM_PROMPTS}
export TRAIN_REPEATS=${TRAIN_REPEATS:-$((MAX_STEPS + 2))}
export ENV_FILE=${ENV_FILE:-/dev/null}
export MODEL_PATH=${MODEL_PATH:-/lustre/fsw/portfolios/nemotron/users/arushig/workspace/output/stage2_text_rl_with_r3_step120}
# 8 nodes (4 training / 4 generation) matched the 2026-10-01 GRPO smoke layout.
export NUM_NODES=${NUM_NODES:-8}
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export NUM_GEN_NODES=${NUM_GEN_NODES:-4}
export SEGMENT_SIZE=${SEGMENT_SIZE:-4}
export SAVE_PERIOD=${SAVE_PERIOD:-2}
export CHECKPOINT_KEEP_TOP_K=${CHECKPOINT_KEEP_TOP_K:-1}
export JOB_CYCLES=${JOB_CYCLES:-1}
export SLURM_TIME_LIMIT=${SLURM_TIME_LIMIT:-04:00:00}
export CONTEXT_COMPACTION=${CONTEXT_COMPACTION:-true}
export AGENTIC_VBENCH_COMPACTION_AUTO=${AGENTIC_VBENCH_COMPACTION_AUTO:-$CONTEXT_COMPACTION}
export AGENTIC_VBENCH_COMPACTION_PRUNE=${AGENTIC_VBENCH_COMPACTION_PRUNE:-false}
export AGENTIC_VBENCH_CONCURRENCY=${AGENTIC_VBENCH_CONCURRENCY:-8}
export AGENTIC_VBENCH_MAX_TURNS=${AGENTIC_VBENCH_MAX_TURNS:-12}
# OpenCode compacts once a call's total tokens reach context - output (here
# 16384), far below the policy's 49152-token limit, so a multi-turn smoke episode
# exercises at least one context rewrite. output is also the per-call max_tokens.
export AGENTIC_VBENCH_CONTEXT_TOKENS=${AGENTIC_VBENCH_CONTEXT_TOKENS:-20480}
export AGENTIC_VBENCH_OUTPUT_TOKENS=${AGENTIC_VBENCH_OUTPUT_TOKENS:-4096}
export BASE_NAME=${BASE_NAME:-agentic_vbench_sc_cc_smoke_step120_$(date -u +%Y%m%d-%H%M%S)-$$}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-$BASE_NAME}
exec bash "$script_dir/run_sc.sh" data.shuffle=false "$@"

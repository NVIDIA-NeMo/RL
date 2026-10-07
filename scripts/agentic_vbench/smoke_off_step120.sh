#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Baseline smoke without context compaction, sized like the pivot checkout's
# scripts/agentic_vbench/smoke_grpo_step120_rootless.sh (16 nodes: 8 training /
# 8 generation, 8 assembly tasks x 4 generations, 2 optimizer steps, 8 agent
# turns of up to 4096 tokens, checkpoint every step, 2 h) so training progress
# can be compared with the pivot run before enabling compaction. Differences
# from pivot are the harness (eval-factory Harbor + stock OpenCode, remote
# Podman backend) and the SingleController path with framework-owned token
# capture and OpenCode compaction/pruning off (no context rewrite).
#   AGENTIC_VBENCH_BACKEND_DIR=$PWD/data/agentic_vbench/backend-cc-smoke-2 \
#     bash scripts/agentic_vbench/smoke_off_step120.sh
set -euo pipefail
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
code_dir=$(cd "$script_dir/../.." && pwd)
export AVB_DATA_ROOT=${AVB_DATA_ROOT:-$code_dir/data/agentic_vbench}
export AGENTIC_VBENCH_BACKEND_DIR=${AGENTIC_VBENCH_BACKEND_DIR:-$AVB_DATA_ROOT/backend-cc-smoke}
# Pivot trained on the assembly family only (native verifier, no judge).
export AGENTIC_VBENCH_TASKS=${AGENTIC_VBENCH_TASKS:-assembly}
export AGENTIC_VBENCH_JUDGE_PROTOCOL=${AGENTIC_VBENCH_JUDGE_PROTOCOL-}
export MAX_STEPS=${MAX_STEPS:-2}
export NUM_PROMPTS=${NUM_PROMPTS:-8}
export NUM_GENERATIONS=${NUM_GENERATIONS:-4}
export TRAIN_LIMIT=${TRAIN_LIMIT:-$NUM_PROMPTS}
export TRAIN_REPEATS=${TRAIN_REPEATS:-$((MAX_STEPS + 2))}
export ENV_FILE=${ENV_FILE:-/dev/null}
export MODEL_PATH=${MODEL_PATH:-/lustre/fsw/portfolios/nemotron/users/arushig/workspace/output/stage2_text_rl_with_r3_step120}
export NUM_NODES=${NUM_NODES:-16}
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export NUM_GEN_NODES=${NUM_GEN_NODES:-8}
export SEGMENT_SIZE=${SEGMENT_SIZE:-8}
export SAVE_PERIOD=${SAVE_PERIOD:-2}
export CHECKPOINT_KEEP_TOP_K=${CHECKPOINT_KEEP_TOP_K:-1}
export JOB_CYCLES=${JOB_CYCLES:-1}
export SLURM_TIME_LIMIT=${SLURM_TIME_LIMIT:-02:00:00}
# No context rewrite anywhere: OpenCode compaction and pruning off. The capture
# route stays framework-owned (token_capture.context_compaction=true): with it off,
# Gym resolves each call's parent by fingerprinting OpenCode's replayed history,
# which fails for stock OpenCode and poisons the calls (66 poisoned calls, every
# rollout masked, in the 2026-10-07 attempt); the framework-owned route matches
# the served completions by content and is what the compaction run validated.
# This is the "identity" arm of compare_cc_20steps.sh: one row per rollout.
export CONTEXT_COMPACTION=true
export AGENTIC_VBENCH_COMPACTION_AUTO=false
export AGENTIC_VBENCH_COMPACTION_PRUNE=false
# Pivot harness: max_steps 8, max_tokens 4096, context 49152, concurrency 8.
# (Pivot's max_total_tokens 8192 was a fork-only cap; stock OpenCode has none.)
export AGENTIC_VBENCH_CONCURRENCY=${AGENTIC_VBENCH_CONCURRENCY:-8}
export AGENTIC_VBENCH_MAX_TURNS=${AGENTIC_VBENCH_MAX_TURNS:-8}
export AGENTIC_VBENCH_CONTEXT_TOKENS=${AGENTIC_VBENCH_CONTEXT_TOKENS:-49152}
export AGENTIC_VBENCH_OUTPUT_TOKENS=${AGENTIC_VBENCH_OUTPUT_TOKENS:-4096}
export BASE_NAME=${BASE_NAME:-agentic_vbench_sc_off_smoke_step120_$(date -u +%Y%m%d-%H%M%S)-$$}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-$BASE_NAME}
exec bash "$script_dir/run_sc.sh" data.shuffle=false "$@"

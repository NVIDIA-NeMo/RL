#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Agentic-VBench identity smoke on the unified-teacher branch: evaluation harness
# budget (32 turns, 8192 output tokens), the policy's native 256k window, ordinary
# token capture (no context rewrite: OpenCode compaction and pruning off), 8 nodes
# (4 training / 4 generation), 8 assembly tasks x 4 generations, 2 steps,
# checkpoint every step, chained cycles so a cluster cancellation resumes.
#   AGENTIC_VBENCH_BACKEND_DIR=$PWD/data/agentic_vbench/backend-cc-3 \
#     bash scripts/agentic_vbench/smoke_identity_32turns_256k.sh
set -euo pipefail
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
code_dir=$(cd "$script_dir/../.." && pwd)
export AVB_DATA_ROOT=${AVB_DATA_ROOT:-$code_dir/data/agentic_vbench}
export AGENTIC_VBENCH_BACKEND_DIR=${AGENTIC_VBENCH_BACKEND_DIR:-$AVB_DATA_ROOT/backend-cc-3}
# Pivot trained on the assembly family only (native verifier, no judge).
export AGENTIC_VBENCH_TASKS=${AGENTIC_VBENCH_TASKS:-assembly}
export AGENTIC_VBENCH_JUDGE_PROTOCOL=${AGENTIC_VBENCH_JUDGE_PROTOCOL-}
export MAX_STEPS=${MAX_STEPS:-2}
export NUM_PROMPTS=${NUM_PROMPTS:-8}
export NUM_GENERATIONS=${NUM_GENERATIONS:-4}
export TRAIN_LIMIT=${TRAIN_LIMIT:-$NUM_PROMPTS}
export TRAIN_REPEATS=${TRAIN_REPEATS:-$((MAX_STEPS + 2))}
export ENV_FILE=${ENV_FILE:-/dev/null}
# Gym server venvs are built from this branch's Gym tree; keep them apart from the cc-pr4200 ones.
export GYM_VENV_DIR=${GYM_VENV_DIR:-$AVB_DATA_ROOT/gym_venvs_unified_teacher}
export MODEL_PATH=${MODEL_PATH:-/lustre/fsw/portfolios/nemotron/users/arushig/workspace/output/stage2_text_rl_with_r3_step120}
export NUM_NODES=${NUM_NODES:-8}
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export NUM_GEN_NODES=${NUM_GEN_NODES:-4}
export SEGMENT_SIZE=${SEGMENT_SIZE:-4}
export SAVE_PERIOD=${SAVE_PERIOD:-1}
export CHECKPOINT_KEEP_TOP_K=${CHECKPOINT_KEEP_TOP_K:-1}
export JOB_CYCLES=${JOB_CYCLES:-3}
export SLURM_TIME_LIMIT=${SLURM_TIME_LIMIT:-04:00:00}
# Ordinary token capture on this branch: no context rewrite anywhere.
export CONTEXT_COMPACTION=false
export AGENTIC_VBENCH_COMPACTION_AUTO=false
export AGENTIC_VBENCH_COMPACTION_PRUNE=false
export AGENTIC_VBENCH_CONCURRENCY=${AGENTIC_VBENCH_CONCURRENCY:-16}
export AGENTIC_VBENCH_MAX_TURNS=${AGENTIC_VBENCH_MAX_TURNS:-32}
export AGENTIC_VBENCH_CONTEXT_TOKENS=${AGENTIC_VBENCH_CONTEXT_TOKENS:-262144}
export AGENTIC_VBENCH_OUTPUT_TOKENS=${AGENTIC_VBENCH_OUTPUT_TOKENS:-8192}
export BASE_NAME=${BASE_NAME:-agentic_vbench_ut_identity_32t_256k_$(date -u +%Y%m%d-%H%M%S)-$$}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-$BASE_NAME}
exec bash "$script_dir/run_sc.sh" data.shuffle=false "$@"

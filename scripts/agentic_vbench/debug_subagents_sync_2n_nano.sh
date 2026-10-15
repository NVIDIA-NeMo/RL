#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Two-node Nano Omni diagnostic for the exact Agentic first4_x4 subagent gate.
# This wrapper deliberately delegates the prompt/tool/depth and synchronous R3
# contract to debug_subagents_sync_8n.sh; it changes only model/topology and
# gives the diagnostic its own result/checkpoint/W&B identity.
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd "$script_dir/../.." && pwd)

nano_snapshot_default=/lustre/fsw/portfolios/nemotron/users/ehosseiniasl/.cache/huggingface/hub/models--nvidia--Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16/snapshots/e5e9932441de940c9a62185c870ea5bcd4cd24e2
export MODEL_PATH=${MODEL_PATH:-$nano_snapshot_default}
export CONTAINER=${CONTAINER:-/lustre/fsw/portfolios/nemotron/users/arushig/images/nemo-rl-cc-pr4200.sqsh}
export CONFIG_IN_CONTAINER=${CONFIG_IN_CONTAINER:-examples/configs/recipes/vlm/vlm_grpo-nemotron-nano-omni-30ba3b-2n8g-megatron-agentic-vbench-sc-256k.v1.yaml}
export TOKENIZER_CHAT_TEMPLATE=${TOKENIZER_CHAT_TEMPLATE:-$MODEL_PATH/chat_template.jinja}
export VLLM_CHAT_TEMPLATE=${VLLM_CHAT_TEMPLATE:-$MODEL_PATH/chat_template.jinja}

export SLURM_ACCOUNT=${SLURM_ACCOUNT:-nemotron_n4_omni}
export NUM_NODES=2
export NUM_GEN_NODES=1
export SEGMENT_SIZE=1
export NUM_PROMPTS=4
export NUM_GENERATIONS=4
if [[ ${MAX_STEPS:-1000000} != 1000000 ]]; then
  echo "ERROR: Nano subagent diagnostic requires MAX_STEPS=1000000; the scheduler allocation is the run boundary" >&2
  exit 2
fi
export MAX_STEPS=1000000

export AGENTIC_VBENCH_TASKS=${AGENTIC_VBENCH_TASKS:-assembly}
export TRAIN_LIMIT=${TRAIN_LIMIT:-4}
export TRAIN_REPEATS=${TRAIN_REPEATS:-4}
export TRAIN_PATH=${TRAIN_PATH:-$repo_dir/data/agentic_vbench/train/assembly_first4_x4.jsonl}
export VAL_PATH=${VAL_PATH:-$TRAIN_PATH}
export AGENTIC_VBENCH_BACKEND_DIR=${AGENTIC_VBENCH_BACKEND_DIR:-$repo_dir/data/agentic_vbench/backend-subagent-sync8-20261014-v20-long}

export BASE_NAME=${BASE_NAME:-agentic_vbench_subagents_nano_sync2_20261014_v1_diagnostic}
export RESULTS_DIR=${RESULTS_DIR:-$repo_dir/results/$BASE_NAME}
export CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-$RESULTS_DIR}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-$BASE_NAME}
export WANDB_PROJECT=nemorl-v2-subagent-comapction-debug
export WANDB_ENTITY=${WANDB_ENTITY:-adlr}

# Both launch and Agentic judge credentials are read without printing values.
credentials_default=/lustre/fsw/portfolios/nemotron/users/ehosseiniasl/codex/credentials.env
export ENV_FILE=${ENV_FILE:-$credentials_default}
export AGENTIC_VBENCH_CREDENTIALS_FILE=${AGENTIC_VBENCH_CREDENTIALS_FILE:-$credentials_default}

# Preserve the production horizon and its resumable, allocation-bounded cycles.
export JOB_CYCLES=${JOB_CYCLES:-6}
export SLURM_QOS=${SLURM_QOS:-normal}
export SLURM_TIME_LIMIT=${SLURM_TIME_LIMIT:-24:00:00}
export MIN_SLURM_TIME_SECONDS=${MIN_SLURM_TIME_SECONDS:-86400}
export MIN_CHECKPOINT_BUDGET_SECONDS=${MIN_CHECKPOINT_BUDGET_SECONDS:-82800}
export REQUIRE_RESUMABLE_JOB_CYCLES=${REQUIRE_RESUMABLE_JOB_CYCLES:-true}

exec bash "$script_dir/debug_subagents_sync_8n.sh" "$@"

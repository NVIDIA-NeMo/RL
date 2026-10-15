#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Exact source-cluster launcher for the ClawEval Super subagent Gate 1.
# Twenty healthy optimizer steps are an observation gate, not a trainer stop.
set -euo pipefail

if (( $# )); then
  echo "ERROR: this exact gate launcher accepts no Hydra overrides" >&2
  exit 2
fi
if [[ -n ${EXTRA_OVERRIDES:-} ]]; then
  echo "ERROR: this exact gate launcher requires EXTRA_OVERRIDES to be empty" >&2
  exit 2
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd "$script_dir/../.." && pwd)

super_checkpoint=/lustre/fsw/portfolios/nemotron/users/ehosseiniasl/checkpoints/super35-journey-mopd2-identity-upsampling-from-my-step30-yifuw-001_boosted_mtp
if [[ -n ${MODEL_PATH:-} && $MODEL_PATH != "$super_checkpoint" ]]; then
  echo "ERROR: Super Gate 1 requires MODEL_PATH=$super_checkpoint" >&2
  exit 2
fi
export MODEL_PATH=$super_checkpoint
export CONFIG_IN_CONTAINER=examples/configs/recipes/vlm/vlm_grpo-nemotron-super-omni-120ba12b-8n8g-megatron-claweval-sc.v2.yaml

# Four learner and four generation nodes, with all eight GPUs on every node.
export NUM_NODES=8
export GPUS_PER_NODE=8
export NUM_GEN_NODES=4
export SEGMENT_SIZE=4
export NUM_PROMPTS=8
export NUM_GENERATIONS=4

# The sync run is allocation-bounded and remains live after the 20-step audit.
export MAX_STEPS=1000000
export JOB_CYCLES=6
export SLURM_ACCOUNT=nemotron_n4_omni
export SLURM_QOS=normal
export SLURM_TIME_LIMIT=24:00:00
export MIN_SLURM_TIME_SECONDS=86400
export MIN_CHECKPOINT_BUDGET_SECONDS=82800
export REQUIRE_RESUMABLE_JOB_CYCLES=true

# Keep exact 20-step milestone donors forever while retaining two recent
# per-step recovery checkpoints between milestones.
export SAVE_PERIOD=20
export CHECKPOINT_KEEP_TOP_K=null

# Framework trace reconstruction is required for parent/child ownership. The
# OpenCode context-rewriting features remain disabled during this first gate.
export CONTEXT_COMPACTION=true
export CLAW_EVAL_COMPACTION_AUTO=false
export CLAW_EVAL_COMPACTION_PRUNE=false
export CLAW_EVAL_TRAINING_MODE=subagent
export CLAW_EVAL_SUBAGENTS_ENABLED=true

export WANDB_PROJECT=nemorl-v2-subagent-comapction-debug
export WANDB_ENTITY=${WANDB_ENTITY:-adlr}
export CLAW_EVAL_BACKEND_DIR=${CLAW_EVAL_BACKEND_DIR:-$repo_dir/data/agentic_vbench/backend-subagent-sync8-20261014-v20-long}
export CLAW_EVAL_CREDENTIALS_FILE=${CLAW_EVAL_CREDENTIALS_FILE:-/lustre/fsw/portfolios/nemotron/users/ehosseiniasl/codex/credentials.env}

export BASE_NAME=${BASE_NAME:-claweval_subagents_super_sync8_$(date -u +%Y%m%d-%H%M%S)_current_head}
export RESULTS_DIR=${RESULTS_DIR:-$repo_dir/results/$BASE_NAME}
export CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-$RESULTS_DIR}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-$BASE_NAME}

# Diagnostic only: retain exact rows above the hard gate without masking them
# or changing reward, advantage, loss, optimizer, or GRPO behavior.
export NEMO_RL_TMPE_OUTLIER_DUMP_PATH=$RESULTS_DIR/tmpe_outliers
export NEMO_RL_TMPE_OUTLIER_THRESHOLD=1.03
export RUN_TMPE_OUTLIER_TESTS=1

exec bash "$script_dir/train_subagents_sync_8n.sh" \
  checkpointing.save_period=20 \
  checkpointing.keep_top_k=null \
  +checkpointing.ft_save_period=1 \
  +checkpointing.ft_keep_latest_k=2

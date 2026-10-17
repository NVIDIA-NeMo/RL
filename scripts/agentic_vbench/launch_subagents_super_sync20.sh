#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Exact source-cluster launcher for the Agentic-VBench Super subagent Gate 1.
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
if [[ ${AGENTIC_VBENCH_BACKEND:-remote} != remote ]]; then
  echo "ERROR: exact AgenticVBench Gate 1 requires AGENTIC_VBENCH_BACKEND=remote" >&2
  exit 2
fi
export AGENTIC_VBENCH_BACKEND=remote

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd "$script_dir/../.." && pwd)

for reserved_variable in RESULTS_DIR CHECKPOINT_ROOT CHECKPOINT_DIR TOKEN_CAPTURE_DIR AGENTIC_VBENCH_OUTPUT_ROOT WANDB_RUN_ID SYNC_POLICY_CHECKPOINT SYNC_GATE_ATTESTATION; do
  if [[ -n ${!reserved_variable+x} ]]; then
    echo "ERROR: exact Super Gate 1 owns $reserved_variable; unset it before launch" >&2
    exit 2
  fi
done
if ! git -C "$repo_dir" diff --quiet --ignore-submodules=all || \
   ! git -C "$repo_dir" diff --cached --quiet --ignore-submodules=all; then
  echo "ERROR: exact Super Gate 1 requires committed superproject source" >&2
  exit 2
fi

super_checkpoint=/lustre/fsw/portfolios/nemotron/users/ehosseiniasl/checkpoints/super35-journey-mopd2-identity-upsampling-from-my-step30-yifuw-001_boosted_mtp
if [[ -n ${MODEL_PATH:-} && $MODEL_PATH != "$super_checkpoint" ]]; then
  echo "ERROR: Super Gate 1 requires MODEL_PATH=$super_checkpoint" >&2
  exit 2
fi
export MODEL_PATH=$super_checkpoint
export CONFIG_IN_CONTAINER=examples/configs/recipes/vlm/vlm_grpo-nemotron-super-omni-120ba12b-16n8g-megatron-agentic-vbench-sc-256k.v1.yaml
export CONTAINER=/lustre/fsw/portfolios/nemotron/users/arushig/images/nemo-rl-cc-pr4200.sqsh
export IMMUTABLE_SOURCE_SNAPSHOT=true
export REQUIRE_FRESH_PROVENANCE=true

# Four learner and four generation nodes, with all eight GPUs on every node.
export NUM_NODES=8
export GPUS_PER_NODE=8
export NUM_GEN_NODES=4
export SEGMENT_SIZE=4
export NUM_PROMPTS=4
export NUM_GENERATIONS=4
# This is the currently validated base topology. The live v27 experiment tests
# TP8 separately; do not claim or select TP8 here until that run proves it.
export GENERATION_TENSOR_PARALLEL_SIZE=4

# The sync run is allocation-bounded and remains live after the 20-step audit.
export MAX_STEPS=1000000
export TRAIN_LIMIT=4
export TRAIN_REPEATS=4
export AVB_DATA_ROOT=$repo_dir/data/agentic_vbench
export AGENTIC_VBENCH_ROOT=$AVB_DATA_ROOT/agentic-vbench
export AGENTIC_VBENCH_TASKS=assembly
export TRAIN_PATH=$AVB_DATA_ROOT/train/assembly_first4_x4.jsonl
export VAL_PATH=$TRAIN_PATH
export JOB_CYCLES=6
export SLURM_ACCOUNT=nemotron_n4_omni
export SLURM_QOS=normal
export SLURM_TIME_LIMIT=24:00:00
export MIN_SLURM_TIME_SECONDS=86400
export MIN_CHECKPOINT_BUDGET_SECONDS=82800
export REQUIRE_RESUMABLE_JOB_CYCLES=true

# A full checkpoint is still written every step for recovery. Step 20 and later
# 20-step milestones are periodic checkpoints retained indefinitely; only the
# two newest non-milestone recovery checkpoints are kept.
export SAVE_PERIOD=20
export CHECKPOINT_KEEP_TOP_K=null

# Framework capture stays on for session reconstruction, but neither OpenCode
# compaction nor pruning rewrites the context in this subagent-only gate.
export CONTEXT_COMPACTION=true
export AGENTIC_VBENCH_COMPACTION_AUTO=false
export AGENTIC_VBENCH_COMPACTION_PRUNE=false
export AGENTIC_VBENCH_TRAINING_MODE=subagent
export AGENTIC_VBENCH_SUBAGENTS_ENABLED=true

export WANDB_PROJECT=nemorl-v2-subagent-comapction-debug
export WANDB_ENTITY=adlr
export WANDB_CLUSTER_NAME=nsc-svg-slurm-1
export AGENTIC_VBENCH_BACKEND_DIR=${AGENTIC_VBENCH_BACKEND_DIR:-$repo_dir/data/agentic_vbench/backend-subagent-sync8-20261014-v20-long}
export ENV_FILE=${ENV_FILE:-/lustre/fsw/portfolios/nemotron/users/ehosseiniasl/codex/credentials.env}

export BASE_NAME=${BASE_NAME:-agentic_vbench_subagents_super_sync8_$(date -u +%Y%m%d-%H%M%S)_current_head}
[[ $BASE_NAME =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "ERROR: BASE_NAME must contain only letters, digits, dot, underscore, or dash" >&2
  exit 2
}
export RESULTS_DIR=$repo_dir/results/$BASE_NAME
[[ ! -e $RESULTS_DIR ]] || {
  echo "ERROR: exact Super Gate 1 requires a fresh result directory: $RESULTS_DIR" >&2
  exit 2
}
export CHECKPOINT_ROOT=$RESULTS_DIR
export WANDB_RUN_NAME=$BASE_NAME
unset AGENTIC_VBENCH_DELEGATION_PROMPT AGENTIC_VBENCH_SUBAGENT_PROMPT
export AGENTIC_VBENCH_MAX_TURNS=32
export AGENTIC_VBENCH_CONTEXT_TOKENS=262144
export AGENTIC_VBENCH_OUTPUT_TOKENS=8192
export AGENTIC_VBENCH_TOOL_OUTPUT_MAX_LINES=400
export AGENTIC_VBENCH_TOOL_OUTPUT_MAX_BYTES=8192
export SUBAGENT_MAX_ROW_TOKENS=65536

# This diagnostic is observational only: it records exact rows above the hard
# audit bound and never masks them or changes the loss.
export NEMO_RL_TMPE_OUTLIER_DUMP_PATH=$RESULTS_DIR/tmpe_outliers
export NEMO_RL_TMPE_OUTLIER_THRESHOLD=1.03
export RUN_TMPE_OUTLIER_TESTS=1

exec bash "$script_dir/debug_subagents_sync_8n.sh" \
  policy.generation.vllm_cfg.tensor_parallel_size="$GENERATION_TENSOR_PARALLEL_SIZE" \
  checkpointing.save_period=20 \
  checkpointing.keep_top_k=null \
  +checkpointing.ft_save_period=1 \
  +checkpointing.ft_keep_latest_k=2

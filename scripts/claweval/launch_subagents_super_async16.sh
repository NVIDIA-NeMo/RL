#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Exact ClawEval Super Gate 2 launcher. Requires a freshly audited sync
# step-20 donor and creates a new async result/checkpoint/W&B identity.
set -euo pipefail

if (( $# )); then
  echo "ERROR: this exact async launcher accepts no Hydra overrides" >&2
  exit 2
fi
if [[ -n ${EXTRA_OVERRIDES:-} ]]; then
  echo "ERROR: this exact async launcher requires EXTRA_OVERRIDES to be empty" >&2
  exit 2
fi
if [[ ${CLAW_EVAL_BACKEND:-remote} != remote ]]; then
  echo "ERROR: exact ClawEval Gate 2 requires CLAW_EVAL_BACKEND=remote" >&2
  exit 2
fi
export CLAW_EVAL_BACKEND=remote
for reserved_variable in CHECKPOINT_MUST_SAVE_BY SBATCH_COMMENT NRL_R3_TRACE_DIR; do
  if [[ -n ${!reserved_variable+x} ]]; then
    echo "ERROR: exact Super Gate 2 owns $reserved_variable; unset it before launch" >&2
    exit 2
  fi
done
: "${CLAW_EVAL_BACKEND_DIR:?Set CLAW_EVAL_BACKEND_DIR to a fresh dedicated live async backend}"
: "${SYNC_POLICY_CHECKPOINT:?Set SYNC_POLICY_CHECKPOINT to the qualified sync step_20/policy/weights directory}"

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd "$script_dir/../.." && pwd)

if ! git -C "$repo_dir" diff --quiet --ignore-submodules=all || \
   ! git -C "$repo_dir" diff --cached --quiet --ignore-submodules=all; then
  echo "ERROR: exact Super Gate 2 requires committed superproject source" >&2
  exit 2
fi
canonical_auditor=$repo_dir/../../nemo-rl-resources/recipes/subagent/audit_single_controller_health.py
canonical_auditor_sha256=f1889a3deb62a92c982cfc015d44f6f42440efe790c4d846143081ebf926130d
if [[ -n ${SYNC_HEALTH_AUDITOR+x} && $(realpath -m "$SYNC_HEALTH_AUDITOR") != $(realpath -m "$canonical_auditor") ]]; then
  echo "ERROR: exact Super Gate 2 does not allow a substituted health auditor" >&2
  exit 2
fi
[[ -s $canonical_auditor ]] || { echo "ERROR: canonical health auditor is missing: $canonical_auditor" >&2; exit 2; }
actual_auditor_sha256=$(sha256sum "$canonical_auditor" | cut -d' ' -f1)
[[ $actual_auditor_sha256 == "$canonical_auditor_sha256" ]] || {
  echo "ERROR: canonical health auditor hash mismatch: $actual_auditor_sha256" >&2
  exit 2
}
export SYNC_HEALTH_AUDITOR=$canonical_auditor
export SYNC_HEALTH_AUDITOR_SHA256=$canonical_auditor_sha256

super_checkpoint=/lustre/fsw/portfolios/nemotron/users/ehosseiniasl/checkpoints/super35-journey-mopd2-identity-upsampling-from-my-step30-yifuw-001_boosted_mtp
if [[ -n ${MODEL_PATH:-} && $(realpath "$MODEL_PATH") != $(realpath "$super_checkpoint") ]]; then
  echo "ERROR: Super Gate 2 requires MODEL_PATH=$super_checkpoint" >&2
  exit 2
fi
export MODEL_PATH=$super_checkpoint
export CONFIG_IN_CONTAINER=examples/configs/recipes/vlm/vlm_grpo-nemotron-super-omni-120ba12b-8n8g-megatron-claweval-sc.v2.yaml
export CONTAINER=/lustre/fsw/portfolios/nemotron/users/arushig/images/nemo-rl-cc-pr4200.sqsh
export IMMUTABLE_SOURCE_SNAPSHOT=true
export REQUIRE_FRESH_PROVENANCE=true
export CLAW_EVAL_SOURCE=/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_omni/users/arushig/claw-eval
export CLAW_EVAL_ORIGINAL_TASK_ROOT=$CLAW_EVAL_SOURCE/data_generation/runs/kimi_video_scaled_100/generated/tasks
export CLAW_EVAL_BRIEF_TASK_ROOT=$CLAW_EVAL_SOURCE/data_generation/runs/kimi_video_scaled_100_brief_eval/generated/tasks
export CLAW_EVAL_B_TASK_ROOT=$CLAW_EVAL_SOURCE/data_generation/runs/kimi_video_brief_1000/generated/tasks
export CLAW_EVAL_SFT=/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_omni/users/ehosseiniasl/github_repos/ehosseiniasl_vlm2_claweval/data/claw_kimi_all_passing_631_opencode_v3/energon_sft.jsonl
export CLAW_EVAL_DATA_ROOT=$repo_dir/data/claweval
export TRAIN_PATH=$CLAW_EVAL_DATA_ROOT/train/opencode_v3_all_619.jsonl
export VAL_PATH=$TRAIN_PATH

export NUM_NODES=16
export GPUS_PER_NODE=8
export NUM_GEN_NODES=8
export SEGMENT_SIZE=8
export NUM_PROMPTS=8
export NUM_GENERATIONS=4
export GENERATION_TENSOR_PARALLEL_SIZE=4
export CLAW_EVAL_CONCURRENCY=32
export TQ_NUM_STORAGE_UNITS=32
export TOKENIZER_CHAT_TEMPLATE=default
export VLLM_CHAT_TEMPLATE=null
export VLLM_LOAD_FORMAT=auto
export SKIP_REPLAY_RESTORE=1
export NRL_IGNORE_VERSION_MISMATCH=1
export MAX_STEPS=1000000
export JOB_CYCLES=6
export SLURM_ACCOUNT=nemotron_n4_omni
export SLURM_PARTITION=batch_long
export SLURM_QOS=normal
export SLURM_TIME_LIMIT=24:00:00
export CHECKPOINT_MARGIN_SECONDS=1200
export MIN_SLURM_TIME_SECONDS=86400
export MIN_CHECKPOINT_BUDGET_SECONDS=82800
export REQUIRE_RESUMABLE_JOB_CYCLES=true
export SAVE_PERIOD=20
export CHECKPOINT_KEEP_TOP_K=0

export CONTEXT_COMPACTION=true
export CLAW_EVAL_COMPACTION_AUTO=false
export CLAW_EVAL_COMPACTION_PRUNE=false
export CLAW_EVAL_TRAINING_MODE=subagent
export CLAW_EVAL_SUBAGENTS_ENABLED=true

export WANDB_PROJECT=nemorl-v2-subagent-comapction-debug
export WANDB_ENTITY=adlr
export WANDB_MODE=online
export WANDB_CLUSTER_NAME=nsc-svg-slurm-1
export WANDB_RUN_GROUP=cluster-nsc-svg-slurm-1
export CLAW_EVAL_CREDENTIALS_FILE=${CLAW_EVAL_CREDENTIALS_FILE:-/lustre/fsw/portfolios/nemotron/users/ehosseiniasl/codex/credentials.env}
export BASE_NAME=${BASE_NAME:-claweval_subagents_super_async16_$(date -u +%Y%m%d-%H%M%S)_current_head}
export WANDB_RUN_NAME=$BASE_NAME
unset CLAW_EVAL_DELEGATION_PROMPT CLAW_EVAL_SUBAGENT_PROMPT
export CLAW_EVAL_CONTEXT_TOKENS=262144
export CLAW_EVAL_OUTPUT_TOKENS=8192
export CLAW_EVAL_MAX_TURNS=40
export CLAW_EVAL_TOOL_OUTPUT_MAX_LINES=400
export CLAW_EVAL_TOOL_OUTPUT_MAX_BYTES=8192
export CLAW_EVAL_MAX_ROW_TOKENS=65536
export CLAW_EVAL_JUDGE_BASE_URL=https://inference-api.nvidia.com/v1
export CLAW_EVAL_JUDGE_MODEL=gcp/google/gemini-3-flash-preview
export CLAW_EVAL_JUDGE_API_KEY_ENV=NV_INFERENCE_API_KEY
export NEMO_RL_TMPE_OUTLIER_DUMP_PATH=$repo_dir/results/$BASE_NAME/tmpe_outliers
export NEMO_RL_TMPE_OUTLIER_THRESHOLD=1.03
export RUN_TMPE_OUTLIER_TESTS=1
export NRL_ROUTER_REPLAY_VALIDATE=1
export NRL_R3_TRACE=1
export NRL_R3_TRACE_STEPS=2
export NRL_R3_TRACE_SAMPLES=2
export NRL_R3_TRACE_MICROBATCHES=2
export NRL_R3_TRACE_VERIFY_FORWARD=1

exec bash "$script_dir/train_subagents_async_16n.sh" \
  checkpointing.save_period=20 \
  checkpointing.keep_top_k=0 \
  ++checkpointing.ft_save_period=10 \
  ++checkpointing.ft_keep_latest_k=5

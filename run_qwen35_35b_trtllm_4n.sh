#!/usr/bin/env bash

# Launch Qwen3.5-35B-A3B GRPO with TRT-LLM generation on an existing 4-node
# GB200 Ray allocation: 2 Megatron train nodes + 2 TRT-LLM gen nodes.
#
# Purpose: E2E real-training BF16 refit test for the GDN+MoE+Vision code path
# (same architecture class as 397B) without requiring a 36-node allocation.
# MTP is disabled in the recipe (mtp_num_layers=0). Vision weights load but
# vision forward pass is not exercised by the SWE text-only task.
#
# Refit test progression:
#   Phase 1 (default): TRTLLM_TP=4  — single-node TP, 2 replicas, BF16 engine
#   Phase 2:           TRTLLM_TP=8  — cross-node TP, 1 replica
#
# Usage: run via <job>-attach.sh on the Ray head node.
# Override env vars:
#   TRTLLM_TP=8           Cross-node TP (phase 2)

set -euo pipefail

export TRTLLM_TP="${TRTLLM_TP:-4}"
export TRTLLM_MEP="${TRTLLM_MEP:-${TRTLLM_TP}}"
RUN_TAG="${RUN_TAG:-qwen35-35b-a3b-refit-jinja-fix-tp${TRTLLM_TP}-erinh}"
export EXP_NAME="${RUN_TAG}"
export SEED="${SEED:-42}"

source /lustre/fsw/coreai_comparch_trtllm/erinh/env.sh
: "${WANDB_API_KEY:?WANDB_API_KEY is required}"

# Skip uv venv creation for TRT-LLM workers — the 'trtllm' extra is not in
# nemo-rl's optional-dependencies. Point directly at the installed venv.
export NEMO_RL_PY_EXECUTABLES_TRTLLM="${NEMO_RL_PY_EXECUTABLES_TRTLLM:-/opt/nemo_rl_venv/bin/python}"

# Model paths — both point to BF16 so Megatron's BF16 update stream matches
# the TRT-LLM engine parameter format.
export CONTAINER_HF_CKPT_PATH=/lustre/fsw/coreai_comparch_trtllm/erinh/llm-models/Qwen3.5-35B-A3B
export TRTLLM_MODEL_PATH=/lustre/fsw/coreai_comparch_trtllm/erinh/llm-models/Qwen3.5-35B-A3B

# Megatron checkpoint dir — writable lustre dir; NeMo-RL will HF→Megatron
# convert on first run and cache the result here for subsequent runs.
export CONTAINER_NRL_MEGATRON_CHECKPOINT_DIR=/lustre/fsw/coreai_comparch_trtllm/erinh/megatron_checkpoints/qwen35_35b_grpo
mkdir -p "${CONTAINER_NRL_MEGATRON_CHECKPOINT_DIR}"

export CONTAINER_NEMO_GYM_SWE_SIF_DIR=/lustre/share/coreai_dlalgo_ci/artifacts/dataset/r2e-gym/r2e-gym/easy-sif
export CONTAINER_NEMO_GYM_SWE_TRAIN_DATA_PATH=/lustre/share/coreai_dlalgo_ci/artifacts/dataset/r2e-gym/r2e-gym/easy-subset-hf-e4f5af9_orig/benchmark_r2e_gym_easy_train.jsonl
export CONTAINER_NEMO_GYM_SWE_VALIDATION_DATA_PATH=/lustre/share/coreai_dlalgo_ci/artifacts/dataset/r2e-gym/r2e-gym/easy-subset-hf-e4f5af9_orig/benchmark_r2e_gym_easy_val.jsonl

source /workspace/llm/config_GB200_4x4_t2g2_tp2pp1ep4gtp4_trtllm.sh

export TRAIN_NODES=2
export GEN_NODES=2
export DGXNNODES=4
export COLOCATED_GENERATION=0

export GEN_BACKEND=trtllm
export SKIP_TRAINING=${SKIP_TRAINING:-0}

unset MAX_STEPS

# Absolute path inside the container (conf/ is mounted under /workspace/llm/).
export RECIPE=/workspace/llm/conf/grpo_qwen35_35b_a3b_swe_openhands_async_trtllm.yaml

export MLPERF_MLLOG_FILE="/logs/${RUN_TAG}-mllog.log"
export CHECKPOINT_DIR="/logs/checkpoint-${RUN_TAG}"
unset EXTRA_ARGS

bash /workspace/llm/run_and_time.sh \
  2>&1 | tee "/logs/${RUN_TAG}-driver.log"

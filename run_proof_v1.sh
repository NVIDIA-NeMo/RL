#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

export CONFIG_PATH="${CONFIG_PATH:-examples/configs/grpo_proof_rl_64n.yaml}"
export ENABLE_MTP_INFERENCE="${ENABLE_MTP_INFERENCE:-1}"
export NRL_MAX_STEPS="${NRL_MAX_STEPS:-2000}"
export NRL_VLLM_ASYNC_TIMEOUT_SECONDS="${NRL_VLLM_ASYNC_TIMEOUT_SECONDS:-7200}"

export NRL_CONTEXT_PARALLEL_SIZE="${NRL_CONTEXT_PARALLEL_SIZE:-16}"
export NRL_MAX_CONTEXT_LENGTH="${NRL_MAX_CONTEXT_LENGTH:-131072}"

export WANDB_NAME="${WANDB_NAME:-nemotron-3-ultra-imo-proof-v1}"
export EXP_SUFFIX="${EXP_SUFFIX:-${WANDB_NAME}}"

# 128 training nodes + 128 rollout nodes on 4-GPU GB200 systems.
export NUM_ACTOR_NODES="${NUM_ACTOR_NODES:-256}"
export GENERATION_NUM_NODES="${GENERATION_NUM_NODES:-128}"

# Eight independent two-node proof-judge services.
export USE_HET_SERVERS="${USE_HET_SERVERS:-1}"
export HET_SERVER_COUNT="${HET_SERVER_COUNT:-8}"
export HET_SERVER_NODES="${HET_SERVER_NODES:-2}"
export HET_SERVER_GPUS_PER_NODE="${HET_SERVER_GPUS_PER_NODE:-4}"
export PROOF_JUDGE_MODEL="${PROOF_JUDGE_MODEL:-deepseek-ai/DeepSeek-Math-V2}"
export PROOF_JUDGE_PORT="${PROOF_JUDGE_PORT:-5000}"

# Point this at the converted Ultra SFT checkpoint described in the guide.
export NRL_MODEL_PATH="${NRL_MODEL_PATH:-${MODEL_PATH:-}}"
export NRL_TRAIN_PATH="${NRL_TRAIN_PATH:-${TRAIN_PATH:-}}"
export NRL_VAL_PATH="${NRL_VAL_PATH:-${VAL_PATH:-${NRL_TRAIN_PATH}}}"

exec "${SCRIPT_DIR}/launch_ultra_proofs.sh" \
  "policy.megatron_cfg.context_parallel_size=${NRL_CONTEXT_PARALLEL_SIZE}" \
  "policy.max_total_sequence_length=${NRL_MAX_CONTEXT_LENGTH}" \
  "$@"

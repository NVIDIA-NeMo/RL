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
export NRL_EXPERT_PARALLEL_SIZE="${NRL_EXPERT_PARALLEL_SIZE:-32}"
export NRL_MAX_CONTEXT_LENGTH="${NRL_MAX_CONTEXT_LENGTH:-65536}"

export WANDB_NAME="${WANDB_NAME:-nemotron-3-ultra-imo-verification-v1}"
export EXP_SUFFIX="${EXP_SUFFIX:-${WANDB_NAME}}"

# 64 training nodes + 48 rollout nodes on 4-GPU GB200 systems.
export NUM_ACTOR_NODES="${NUM_ACTOR_NODES:-112}"
export GENERATION_NUM_NODES="${GENERATION_NUM_NODES:-48}"
export SIMP_SAMP_MAX_PARALLEL_TRAJECTORIES="${SIMP_SAMP_MAX_PARALLEL_TRAJECTORIES:-128}"

# Six independent two-node proof-judge services.
export USE_HET_SERVERS="${USE_HET_SERVERS:-1}"
export HET_SERVER_COUNT="${HET_SERVER_COUNT:-6}"
export HET_SERVER_NODES="${HET_SERVER_NODES:-2}"
export HET_SERVER_GPUS_PER_NODE="${HET_SERVER_GPUS_PER_NODE:-4}"
export PROOF_JUDGE_MODEL="${PROOF_JUDGE_MODEL:-deepseek-ai/DeepSeek-Math-V2}"
export PROOF_JUDGE_PORT="${PROOF_JUDGE_PORT:-5000}"

# Verification starts from an explicit Hugging Face export of a retained
# proof-generation checkpoint, so no model path is guessed here.
export NRL_MODEL_PATH="${NRL_MODEL_PATH:-${MODEL_PATH:-}}"
export NRL_TRAIN_PATH="${NRL_TRAIN_PATH:-${TRAIN_PATH:-}}"
export NRL_VAL_PATH="${NRL_VAL_PATH:-${VAL_PATH:-${NRL_TRAIN_PATH}}}"

exec "${SCRIPT_DIR}/launch_ultra_proofs.sh" \
  "policy.megatron_cfg.context_parallel_size=${NRL_CONTEXT_PARALLEL_SIZE}" \
  "policy.max_total_sequence_length=${NRL_MAX_CONTEXT_LENGTH}" \
  "policy.megatron_cfg.expert_model_parallel_size=${NRL_EXPERT_PARALLEL_SIZE}" \
  "grpo.async_grpo.simp_samp_max_parallel_trajectories=${SIMP_SAMP_MAX_PARALLEL_TRAJECTORIES}" \
  "grpo.reward_shaping.enabled=true" \
  "grpo.reward_shaping.overlong_buffer_length=8192" \
  "checkpointing.force_keep_steps=5" \
  "$@"

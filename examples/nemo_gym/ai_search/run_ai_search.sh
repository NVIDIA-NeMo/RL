#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

AI_SEARCH_PLUGIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export AI_SEARCH_PREPARE_TRAINING_ENV=1
source "${AI_SEARCH_PLUGIN_DIR}/prepare_ai_search.sh"

AI_SEARCH_MAX_STEPS="${AI_SEARCH_MAX_STEPS:-1}"
AI_SEARCH_NUM_PROMPTS="${AI_SEARCH_NUM_PROMPTS:-2}"
AI_SEARCH_NUM_GENERATIONS="${AI_SEARCH_NUM_GENERATIONS:-4}"
AI_SEARCH_CONFIG="${AI_SEARCH_CONFIG:-${AI_SEARCH_PLUGIN_DIR}/grpo_qwen2_5_7b.yaml}"
AI_SEARCH_RUN_DIR="${AI_SEARCH_RUN_DIR:-${AI_SEARCH_RUNTIME_DIR}/runs/grpo-ai-search-qwen2.5-7b}"

mkdir -p "${AI_SEARCH_RUN_DIR}/checkpoints" "${AI_SEARCH_RUN_DIR}/logs"

# The example installs all required policy, rollout, and Gym dependencies into one
# node-local environment. Reusing it avoids compiling optional MoE-only kernels.
export NEMO_RL_PY_EXECUTABLES_SYSTEM=1

cd "${AI_SEARCH_REPO_DIR}"
exec "${UV_BIN}" run --no-sync \
  python examples/nemo_gym/run_grpo_nemo_gym.py \
  --config "${AI_SEARCH_CONFIG}" \
  "grpo.max_num_steps=${AI_SEARCH_MAX_STEPS}" \
  "grpo.num_prompts_per_step=${AI_SEARCH_NUM_PROMPTS}" \
  "grpo.num_generations_per_prompt=${AI_SEARCH_NUM_GENERATIONS}" \
  "checkpointing.checkpoint_dir=${AI_SEARCH_RUN_DIR}/checkpoints" \
  "logger.log_dir=${AI_SEARCH_RUN_DIR}/logs" \
  "$@"

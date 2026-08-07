#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# =============================================================================
# launch_6k_pipeclean.sh
#
# Thin wrapper around ultra_launch.sh for the 6K-GPU (1536-node) Ultra
# pipeclean recipe. Sets the validated node split and config path, then
# forwards to the launcher.
#
# Shape (GB200 NVL72, 4 GPUs/node → 6144 GPUs):
#   Training 512 / vLLM 960 / Gym 64
#
# Usage — the site block below supplies model, data, container and Slurm
# defaults for the GB200 cluster this recipe was validated on, so a bare
# invocation works there:
#
#   WANDB_API_KEY=$WANDB_API_KEY \
#   bash examples/nemo_gym/nemotron-3-ultra/launch_6k_pipeclean.sh
#
# On any other cluster, export the site variables yourself (they are all
# ${VAR:-default} and every one of them wins over the default).
#
# Optional:
#   NRL_MAX_STEPS=4              # short pipeclean
#   WALLTIME=4:00:00
#   CONTEXT_PARALLEL_SIZE=16     # default; raise only if CP=16 still OOMs
#   DRY_RUN=1
#   NUM_TRAIN_NODES / NUM_GEN_NODES / NUM_GYM_NODES  # override the 6K split
#
# Extra positional args are forwarded as Hydra overrides to ultra_launch.sh.
# =============================================================================
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

# Default config — callers may still override CONFIG_PATH explicitly.
export CONFIG_PATH="${CONFIG_PATH:-${SCRIPT_DIR}/pipeclean_6k.yaml}"

# =============================================================================
# Site defaults
# =============================================================================
# Paths on the GB200 cluster where this recipe was validated: the checkpoint,
# blend and judges the 6K runs used, so a bare invocation reproduces them.
# Export any of these to point elsewhere.
# =============================================================================
export MODEL_PATH="${MODEL_PATH:-/lustre/fsw/portfolios/llmservice/users/jiaqiz/models/ultra_stage2sft_step300}"
export TRAIN_PATH="${TRAIN_PATH:-/lustre/fsw/portfolios/llmservice/users/jiaqiz/data/gym/rl-data-tools/blends/curriculum_v35_inescapable-sawfly.train.efforts0p15_qamathcode.jsonl}"
# The reference runs validate on the training blend; there is no separate split.
export VAL_PATH="${VAL_PATH:-${TRAIN_PATH}}"

# Must carry vLLM 0.25.1 in the RL venvs to match this branch's code; a
# pre-bump image fails at import with "cannot import name ServingTokenization".
export CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/coreai/users/yifuw/enroot-images/gitlab-master.nvidia.com/yifuw/images/nemo-rl:main_ultra_recipes_prebaked_venvs_20260730.squashfs}"
export SANDBOX_CONTAINER="${SANDBOX_CONTAINER:-/lustre/fsw/portfolios/llmservice/users/igitman/images/nemo-skills-sandbox-latest.sqsh}"
export EXTRA_MOUNTS="${EXTRA_MOUNTS:-/lustre:/lustre}"
export PERSISTENT_CACHE="${PERSISTENT_CACHE:-/lustre/fsw/portfolios/llmservice/users/${USER}/.cache/nemotron_ultra}"
export HF_HOME="${HF_HOME:-/lustre/fsw/portfolios/llmservice/users/${USER}/hf_home}"

export GENRM_MODEL="${GENRM_MODEL:-/lustre/fsw/portfolios/llmservice/users/ansubramania/models/qwen235b_principle_comparison_genrm_step1230}"
export NL2BASH_JUDGE_MODEL="${NL2BASH_JUDGE_MODEL:-/lustre/fsw/portfolios/llmservice/users/ansubramania/models/Qwen3-235B-A22B-Instruct-2507-FP8}"
export SAFETY_JUDGE_MODEL="${SAFETY_JUDGE_MODEL:-/lustre/fsw/portfolios/llmservice/users/ansubramania/super_v3/model_checkpoints/Nemotron-Content-Safety-Reasoning-4B}"

export SLURM_ACCOUNT="${SLURM_ACCOUNT:-nemotron_sw_pre}"
export SLURM_PARTITION="${SLURM_PARTITION:-batch}"
# A 6K allocation may additionally need SLURM_QOS, SLURM_RESERVATION and
# EXCLUDE_NODES. They are left unset because a reservation you do not hold
# makes sbatch fail outright; see ultra_launch.sh for the variable names.

# 6K node split (~33% train / ~63% gen / ~4% gym). Callers may override.
export NUM_TRAIN_NODES="${NUM_TRAIN_NODES:-512}"
export NUM_GEN_NODES="${NUM_GEN_NODES:-960}"
export NUM_GYM_NODES="${NUM_GYM_NODES:-64}"

# CP=16 is baked into pipeclean_6k.yaml; allow an override for memory experiments.
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-16}"

# Sensible defaults for short 6K hero / pipeclean allocations when unset.
export EXP_NAME="${EXP_NAME:-ultra-6k-pipeclean}"
export WALLTIME="${WALLTIME:-4:00:00}"

exec bash "${SCRIPT_DIR}/ultra_launch.sh" \
  "policy.megatron_cfg.context_parallel_size=${CONTEXT_PARALLEL_SIZE}" \
  "$@"

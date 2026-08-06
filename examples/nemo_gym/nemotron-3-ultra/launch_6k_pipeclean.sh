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
# pipeclean recipe. Sets the validated node split and config path from the
# internal 6k-pipecleaning recipe, then forwards to the public launcher.
#
# Shape (GB200 NVL72, 4 GPUs/node → 6144 GPUs):
#   Training 512 / vLLM 960 / Gym 64
#
# Usage (required env vars are the same as ultra_launch.sh):
#
#   EXP_NAME=ultra-6k-pipeclean \
#   MODEL_PATH=/path/to/ultra_sft_checkpoint \
#   TRAIN_PATH=$DATA_DIR/rlvr1.train.jsonl \
#   VAL_PATH=$DATA_DIR/rlvr1.val.jsonl \
#   CONTAINER=/path/to/nemo-rl-container \
#   SANDBOX_CONTAINER=/path/to/nemo-skills-sandbox.sqsh \
#   PERSISTENT_CACHE=/path/to/persistent/cache \
#   SLURM_PARTITION=$SLURM_PARTITION \
#   SLURM_ACCOUNT=$SLURM_ACCOUNT \
#   GENRM_MODEL=nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-GenRM \
#   NL2BASH_JUDGE_MODEL=Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 \
#   SAFETY_JUDGE_MODEL=nvidia/Nemotron-Content-Safety-Reasoning-4B \
#   bash examples/nemo_gym/nemotron-3-ultra/launch_6k_pipeclean.sh
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

# 6K node split (2× the internal 768n ratio). Callers may override any of these.
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

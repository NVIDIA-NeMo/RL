#!/usr/bin/env bash
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
#
# CMH reproduction of the malformed-v3 combined SA-V + CapRL experiment.
# The source branch/subdirectory name records every required PR/MR pin:
# PR 4116, Megatron-Bridge MR 6067, and Megatron-Core PRs 6912/7278/7279.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run_stamp="${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}"
cluster_name="${SLURM_CLUSTER_NAME:-aws-cmh-slurm-1-v1}"
experiment="super35_0912_malformed_v3_legacy_loo_np_async_r3_frozen_text_r3_step120_32n_pr4116_mbridge6067_mcore6912_7278_7279_${run_stamp}"

export RUN_ID="${run_stamp}"
export SLURM_CLUSTER_NAME="${cluster_name}"
export BASE_NAME="${BASE_NAME:-async_grpo_${experiment}}"
export WANDB_PROJECT="${WANDB_PROJECT:-Nemotron-omni-RL-debug}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${cluster_name}_${experiment}}"
export ENV_FILE="${ENV_FILE:-/lustre/fsw/portfolios/nemotron/users/ehosseiniasl/.codex/credentials.env}"
# Keep multi-terabyte checkpoints and their HF exports outside the source tree.
# /lustre and /scratch are aliases for the same CMH quota, so this is an
# organizational/safety boundary rather than a separate storage allocation.
export RESULTS_DIR="${RESULTS_DIR:-/lustre/fsw/portfolios/nemotron/projects/nemotron_omni_vision/users/ehosseiniasl/results/${BASE_NAME}}"

export CONTAINER="${CONTAINER:-/scratch/fsw/portfolios/nemotron/projects/nemotron_omni_vision/users/ehosseiniasl/images/rl-gym.67009223-gym_ln_fix.sqsh}"
export MODEL_PATH="${MODEL_PATH:-/lustre/fsw/portfolios/nemotron/users/arushig/workspace/output/stage2_text_rl_with_r3_step120}"
export DATA_PATH="${DATA_PATH:-/lustre/fsw/portfolios/nemotron/users/arushig/nemo_gym_rl_video_0803/nemo_rl/results/combined_sav_caprl_20260822/train_sav_all_tracks_plus_caprl_exclude6215_cluster_paths.jsonl}"
export CONFIG_PATH="${CONFIG_PATH:-examples/configs/recipes/vlm/vlm_grpo-nemotron-super-omni-120ba12b-32n4g-megatron-sav-caprl-r3-malformed-v3.yaml}"

export R3_ENABLED=true
export PROFILE_BAND_ENABLED=false
export LENGTH_PENALTY_ENABLED=false
export MAX_STEPS=60
export JOB_CYCLES=20
export NUM_NODES=32
export NUM_GEN_NODES=16
export GPUS_PER_NODE=4
export SEGMENT_SIZE=8
export NUM_PROMPTS=128
export NUM_GENERATIONS=16
export SAVE_PERIOD=5
export CHECKPOINT_KEEP_TOP_K=null
export SLURM_TIME_LIMIT=04:00:00

export TOKENIZER_CHAT_TEMPLATE=default
export VLLM_CHAT_TEMPLATE=null
export IN_FLIGHT_WEIGHT_UPDATES=true
export RECOMPUTE_KV_CACHE_AFTER_WEIGHT_UPDATES=false
export SLURM_ACCOUNT="${SLURM_ACCOUNT:-nemotron_omni_vision}"
export SLURM_PARTITION="${SLURM_PARTITION:-batch_long}"

export PERSISTENT_CACHE="${PERSISTENT_CACHE:-/scratch/fsw/portfolios/nemotron/projects/nemotron_omni_vision/users/ehosseiniasl/nemo_rl_cache/super35_0912_malformed_v3_pr4116_mbridge6067_mcore6912_7278_7279}"
export GYM_VENV_DIR="${GYM_VENV_DIR:-${PERSISTENT_CACHE}/gym_venvs_r3}"
export PREFETCH_GYM_VENVS="${PREFETCH_GYM_VENVS:-true}"

exec "${script_dir}/async_grpo_super35_sav_caprl_step120_32n.sh" "$@"

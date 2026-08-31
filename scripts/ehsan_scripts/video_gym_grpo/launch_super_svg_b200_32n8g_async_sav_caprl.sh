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

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
code_dir="$(cd "${script_dir}/../../.." && pwd)"
cd "${code_dir}"

credentials_env="${CREDENTIALS_ENV:-/lustre/fsw/portfolios/nemotron/users/ehosseiniasl/codex/credentials.env}"
if [[ ! -r "${credentials_env}" ]]; then
    echo "ERROR: credentials environment is not readable: ${credentials_env}" >&2
    exit 1
fi
set -a
source "${credentials_env}"
set +a

export EXP_NAME="${EXP_NAME:-super-svg-b200-32n8g-async-sav-caprl}"
export WANDB_ENTITY="${WANDB_ENTITY:-adlr}"
export WANDB_PROJ="${WANDB_PROJ:-Nemotron-omni-RL-debug}"
export WANDB_MODE="${WANDB_MODE:-online}"
export MODEL_PATH="${MODEL_PATH:-/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_omni/users/arushig/workspace/output/full_generalist_iter10000}"
export CHAT_TEMPLATE="${CHAT_TEMPLATE:-/scratch/fsw/portfolios/nemotron/projects/nemotron_omni_vision/users/pulkitk/workspace/output/sav_qwen_rt_v1/checkpoints/tp_1_hf/iter_0000864/mcore_to_hf_540p/chat_template_mm.jinja}"
export TRAIN_PATH="${TRAIN_PATH:-${code_dir}/results/auto_research/20260829-super35-sav-caprl/data/train_64450.jsonl}"
export VAL_PATH="${VAL_PATH:-${code_dir}/results/auto_research/20260829-super35-sav-caprl/data/validation_1024.jsonl}"
export NEMO_RL_VIDEO_MEDIA_ROOT="${NEMO_RL_VIDEO_MEDIA_ROOT:-/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_omni/users/arushig/nemo_gym_rl_video_0803/nemo_rl/results/video_frame_cache_caprl_passrate_n5_easy_to_hard_lt60s_20260806_f64}"
export CONFIG_PATH="${CONFIG_PATH:-examples/configs/recipes/vlm/vlm_grpo-nemotron-super-omni-120ba12b-svg-b200-32n8g-megatron-tp8ep16cp2-async-gym-video-sav-caprl.v1.yaml}"
export CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/llmservice/users/smohsenitahe/images/nemo-rl-main-20260807-super-ultra-omni-prefetched.sqsh}"
export SANDBOX_CONTAINER="${SANDBOX_CONTAINER:-/lustre/fsw/portfolios/llmservice/users/igitman/images/nemo-skills-sandbox-latest.sqsh}"
export PERSISTENT_CACHE="${PERSISTENT_CACHE:-${code_dir}/.cache/nemo-rl-super-omni-svg-b200}"
export NRL_MEGATRON_CHECKPOINT_DIR="${NRL_MEGATRON_CHECKPOINT_DIR:-${PERSISTENT_CACHE}/megatron_ckpt_cache_temporal2}"
export USE_CONTAINER_MEGATRON="${USE_CONTAINER_MEGATRON:-false}"
export SLURM_ACCOUNT="${SLURM_ACCOUNT:-nemotron_omni_vision}"
export SLURM_PARTITION="${SLURM_PARTITION:-batch}"
export SLURM_TIME_LIMIT="${SLURM_TIME_LIMIT:-4:0:0}"
export JOB_CYCLES="${JOB_CYCLES:-20}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export EXTRA_MOUNTS="${EXTRA_MOUNTS:-/scratch:/scratch,/lustre:/lustre}"
export BASE_LOG_DIR="${BASE_LOG_DIR:-${code_dir}/results/${EXP_NAME}/slurm}"

exec bash examples/nemo_gym/nemotron-3-super-omni/super_omni_launch.sh

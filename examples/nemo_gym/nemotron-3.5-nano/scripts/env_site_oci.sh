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

# Public asset map for the validated OCI site. No credentials.
# Every value can be overridden by the calling environment.
export EXTERNAL_VLLM_SHARED_ROOT="${EXTERNAL_VLLM_SHARED_ROOT:-/lustre}"
export RESULTS_ROOT="${RESULTS_ROOT:-${NEMO_RL_ROOT}/results}"
export PERSISTENT_CACHE="${PERSISTENT_CACHE:-${NEMO_RL_ROOT}/.cache/nano35}"
export HF_HOME="${HF_HOME:-${PERSISTENT_CACHE}/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export SLURM_ACCOUNT="${SLURM_ACCOUNT:-nemotron_sw_post}"
export SLURM_PARTITION="${SLURM_PARTITION:-batch}"
export SLURM_QOS="${SLURM_QOS:-normal}"
export CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_ci/nemo_rl_ci/sqsh_files/rl-gym.75221028.sqsh}"
export MODEL_PATH="${MODEL_PATH:-/lustre/fsw/portfolios/llmservice/users/venkats/nemo-evaluator-rundirs/nano_v35_sft/conversions/upsampled-iter6000/hf}"
export TRAIN_PATH="${TRAIN_PATH:-/lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/geshen/rl-data-tools/blends/curriculum_amplified_dolphin_v41_cheery_umbrette.train.jsonl}"
export VAL_PATH="${VAL_PATH:-${TRAIN_PATH}}"
export GENRM_MODEL="${GENRM_MODEL:-/lustre/fsw/portfolios/coreai/users/amahishi/projects/nemo-rl-workspace/ultra-smoke-test/cache/hf_judge_models/hub/models--nvidia--NVIDIA-Nemotron-3-Ultra-550B-A55B-GenRM/snapshots/116af7cb1a23ce9017b2c412945b0623252655d2}"
export GENRM_REASONING_PARSER="${GENRM_REASONING_PARSER:-/lustre/fsw/portfolios/llmservice/users/lvega/evals/ultra_v3_reasoning_parser.py}"
export NL2BASH_JUDGE_MODEL="${NL2BASH_JUDGE_MODEL:-/lustre/fsw/portfolios/llmservice/users/ansubramania/models/Qwen3-235B-A22B-Instruct-2507-FP8}"
export SAFETY_JUDGE_MODEL="${SAFETY_JUDGE_MODEL:-/lustre/fsw/portfolios/llmservice/users/ansubramania/super_v3/model_checkpoints/Nemotron-Content-Safety-Reasoning-4B}"
export SANDBOX_CONTAINER="${SANDBOX_CONTAINER:-/lustre/fsw/portfolios/llmservice/users/geshen/mopd_nano_fast/images/nemo-skills-sandbox-no-sync.sqsh}"

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

# Copy to an ignored scripts/site.local.sh and replace ALL /shared placeholders.
# The checkout, results, caches and assets must be on storage visible to all nodes
# under EXTERNAL_VLLM_SHARED_ROOT. These defaults are deliberately not runnable.
# Set HF_TOKEN / WANDB_API_KEY in your shell or a secret manager, not here.
export EXTERNAL_VLLM_SHARED_ROOT="/shared"
export RESULTS_ROOT="/shared/my-user/nano35/results"
export PERSISTENT_CACHE="/shared/my-user/nano35/cache"
export HF_HOME="${PERSISTENT_CACHE}/hf_home"
export HF_HUB_CACHE="${HF_HOME}/hub"
export CONTAINER="/shared/images/compatible-rl-gym.sqsh"
export SANDBOX_CONTAINER="/shared/images/nemo-skills-sandbox.sqsh"
export MODEL_PATH="/shared/models/nano-v35-sft"
export TRAIN_PATH="/shared/data/curriculum_amplified_dolphin_v41_cheery_umbrette.train.jsonl"
export VAL_PATH="${TRAIN_PATH}"
export GENRM_MODEL="/shared/models/NVIDIA-Nemotron-3-Ultra-550B-A55B-GenRM"
export GENRM_REASONING_PARSER="/shared/parsers/ultra_v3_reasoning_parser.py"
export NL2BASH_JUDGE_MODEL="/shared/models/Qwen3-235B-A22B-Instruct-2507-FP8"
export SAFETY_JUDGE_MODEL="/shared/models/Nemotron-Content-Safety-Reasoning-4B"
export SLURM_ACCOUNT="your-account"
export SLURM_PARTITION="your-partition"
export SLURM_QOS="normal"
# Hardware/scheduler must support this recipe: AArch64, four GPUs per node,
# heterogeneous Slurm jobs, --segment, and Pyxis/Enroot. On a different site,
# adapt scheduler flags and X11 library mounts in nano35_rlvr_common.sh first.
# This is not an automatic hardware/topology port.

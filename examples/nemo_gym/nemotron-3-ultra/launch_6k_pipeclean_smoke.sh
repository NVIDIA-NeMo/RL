#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

# Smaller-scale validation wrapper for launch_6k_pipeclean.sh.
#
# It preserves the production model and external judge deployment while
# reducing policy training, generation, Gym, batch size, and sequence length:
#
#   Hetgroup 0: 32 training + 14 generation + 2 Gym = 48 nodes (segment 16)
#   Hetgroup 1: 14 GenRM + 2 NL2Bash = 16 nodes (segment 2)
#   Batch:      32 prompts x 16 generations = 512 samples
#   Sequence:   49,152 tokens
#   Total:      64 nodes / 256 GPUs
#
# The safety judge remains in Gym. The 16-node training segments match the
# HybridEP assumption of 64 ranks per NVLink domain. The external pool keeps
# the production TP sizes but uses two fewer GenRM replicas to stay at 64 nodes.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

export NUM_TRAIN_NODES="${NUM_TRAIN_NODES:-32}"
export NUM_GEN_NODES="${NUM_GEN_NODES:-14}"
export NUM_GYM_NODES="${NUM_GYM_NODES:-2}"
export SEGMENT_SIZE="${SEGMENT_SIZE:-16}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"

# Keep production TP sizes while fitting the external pool into 16 nodes.
export EXTERNAL_JUDGES=1
export GENRM_REPLICAS=14
export GENRM_TENSOR_PARALLEL_SIZE=4
export NL2BASH_REPLICAS=2
export NL2BASH_TENSOR_PARALLEL_SIZE=4
export EXTERNAL_VLLM_SEGMENT_SIZE=2

NUM_PROMPTS_PER_STEP="${NUM_PROMPTS_PER_STEP:-32}"
TRAIN_GLOBAL_BATCH_SIZE="${TRAIN_GLOBAL_BATCH_SIZE:-$((NUM_PROMPTS_PER_STEP * 16))}"
MAX_TOTAL_SEQUENCE_LENGTH="${MAX_TOTAL_SEQUENCE_LENGTH:-49152}"

export EXP_NAME="${EXP_NAME:-ultra-6k-pipeclean-external-judges-smoke-py31314}"
export NRL_MAX_STEPS="${NRL_MAX_STEPS:-2}"
export WALLTIME="${WALLTIME:-2:00:00}"
export SLURM_ACCOUNT="${SLURM_ACCOUNT:-nemotron_sw_post}"
export SLURM_PARTITION="${SLURM_PARTITION:-batch}"
export SLURM_QOS="${SLURM_QOS:-short}"
export USE_SNAPSHOT="${USE_SNAPSHOT:-0}"

echo "Ultra 6K pipeclean external-judge smoke profile"
echo "  hetgroup 0: train=${NUM_TRAIN_NODES}, generation=${NUM_GEN_NODES}, Gym=${NUM_GYM_NODES}"
echo "  hetgroup 1: GenRM=${GENRM_REPLICAS}xTP${GENRM_TENSOR_PARALLEL_SIZE}, NL2Bash=${NL2BASH_REPLICAS}xTP${NL2BASH_TENSOR_PARALLEL_SIZE}"
echo "  batch: prompts=${NUM_PROMPTS_PER_STEP}, generations=16, global=${TRAIN_GLOBAL_BATCH_SIZE}"
echo "  max sequence length: ${MAX_TOTAL_SEQUENCE_LENGTH}"

exec bash "${SCRIPT_DIR}/launch_6k_pipeclean.sh" \
  "grpo.num_prompts_per_step=${NUM_PROMPTS_PER_STEP}" \
  "policy.train_global_batch_size=${TRAIN_GLOBAL_BATCH_SIZE}" \
  "policy.max_total_sequence_length=${MAX_TOTAL_SEQUENCE_LENGTH}" \
  "policy.generation.max_new_tokens=${MAX_TOTAL_SEQUENCE_LENGTH}" \
  "policy.generation.vllm_cfg.max_model_len=${MAX_TOTAL_SEQUENCE_LENGTH}" \
  "$@"

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

# bf16_009 convergence recipe with prefix recovery; no submission unless run.
# Start a NEW lineage, then reuse EXP_NAME to continue toward 100 total steps.
# Usage from the NeMo-RL root:
#   EXP_NAME=nano35-bf16-009-prefix-100step bash \
#     examples/nemo_gym/nemotron-3.5-nano/scripts/launch_nano35_bf16_009_prefix_checkpoint.sh
# DRY_RUN=1 resolves the submission without allocating nodes.
# Trailing arguments are forwarded as Hydra overrides by the common launcher.
set -euo pipefail

SCRIPT_DIR="$(cd -L -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -L)"
export NEMO_RL_ROOT="${NEMO_RL_ROOT:-$(cd -L -- "${SCRIPT_DIR}/../../../.." && pwd -L)}"

# Use a private local site profile if present, otherwise the public OCI map.
# The shared launcher sources it; credentials should be exported separately.
if [[ -z "${SITE_OVERLAY:-}" ]]; then
  if [[ -f "${SCRIPT_DIR}/.env" ]]; then
    export SITE_OVERLAY="${SCRIPT_DIR}/.env"
  else
    export SITE_OVERLAY="${SCRIPT_DIR}/env_site_oci.sh"
  fi
fi

export ARM_LABEL="bf16_009 with prefix-rollout checkpoints"
export RUN_LABEL="${RUN_LABEL:-Nano 3.5 BF16/MTP prefix-checkpoint convergence}"
export CONFIG_PATH_HOST="${CONFIG_PATH_HOST:-${SCRIPT_DIR}/../configs/rlvr_sc_bf16_009_prefix_checkpoint.yaml}"
export TRAIN_ENTRYPOINT="${TRAIN_ENTRYPOINT:-./examples/run_grpo_single_controller.py}"
export RESULTS_ROOT="${RESULTS_ROOT:-${NEMO_RL_ROOT}/results}"
export CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_ci/nemo_rl_ci/sqsh_files/rl-gym.75221028.sqsh}"

# Match the reference allocation, not the earlier 88-node diagnostic run.
export NUM_TRAIN_NODES="${NUM_TRAIN_NODES:-32}"
export NUM_GEN_NODES="${NUM_GEN_NODES:-32}"
export NUM_GYM_NODES="${NUM_GYM_NODES:-2}"
export SEGMENT_SIZE="${SEGMENT_SIZE:-2}"
export EXTERNAL_VLLM_SEGMENT_SIZE="${EXTERNAL_VLLM_SEGMENT_SIZE:-2}"
export GENRM_REPLICAS="${GENRM_REPLICAS:-8}"
export NL2BASH_REPLICAS="${NL2BASH_REPLICAS:-4}"
export NRL_MAX_STEPS="${NRL_MAX_STEPS:-100}"
export NRL_CHECKPOINT_SUBSET="${NRL_CHECKPOINT_SUBSET:-0}"
export WALLTIME="${WALLTIME:-1:30:00}"
export SLURM_ACCOUNT="${SLURM_ACCOUNT:-nemotron_sw_post}"
export SLURM_QOS="${SLURM_QOS:-normal}"
export WANDB_PROJ="${WANDB_PROJ:-nano35-bf16-prefix-convergence}"

# Run mounted checkpoint-capable sources using the compatible image's venvs.
export USE_IMAGE_GYM="${USE_IMAGE_GYM:-0}"
export GYM_VENV_DIR="${GYM_VENV_DIR:-/opt/gym_venvs}"
export USE_SNAPSHOT="${USE_SNAPSHOT:-0}"
export NRL_IGNORE_VERSION_MISMATCH="${NRL_IGNORE_VERSION_MISMATCH:-1}"
export NRL_FORCE_REBUILD_VENVS="${NRL_FORCE_REBUILD_VENVS:-false}"
export UV_NO_SYNC="${UV_NO_SYNC:-1}"

echo "Reference differences: malformed-thinking reward penalty OFF; invalid-tool-call advantage override OFF."
echo "BF16, MTP=5, router replay OFF; target 100 steps across 90-minute allocations."
echo "If WALLTIME is overridden, also adjust checkpointing.checkpoint_must_save_by."

# Resolve a relative site path before changing cwd. Slurm should submit from
# the mounted NeMo-RL checkout, even when invoked elsewhere.
if [[ "${SITE_OVERLAY}" != /* ]]; then
  export SITE_OVERLAY="${PWD}/${SITE_OVERLAY}"
fi
cd "${NEMO_RL_ROOT}"

# shellcheck source=./nano35_rlvr_common.sh
source "${SCRIPT_DIR}/nano35_rlvr_common.sh"

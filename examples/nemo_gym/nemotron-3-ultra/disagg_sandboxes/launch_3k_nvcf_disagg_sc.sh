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
# launch_3k_nvcf_disagg_sc.sh
#
# The primary 3K test arm: SingleController, NVCF-hosted judges, disaggregated
# sandboxes. Judges and tool execution both leave the allocation, so every GPU
# in the job belongs to policy training or generation.
#
# Shape (GB200 NVL72, 4 GPUs/node):
#   256 training + 488 generation + 8 Gym = 752 nodes / 3008 GPUs, one hetgroup
#
# Compared with the v1 baseline (launch_3k_pipeclean.sh, 762 nodes): the 10-node
# external judge hetgroup is gone because GenRM and NL2Bash are hosted, and Gym's
# GPUs go unused because the safety judge is hosted too. Training and the batch
# are identical, so loss and reward curves remain comparable; judge latency and
# tool-execution latency do not, since both are now network round trips.
#
# The backup arm if NVCF proves unusable is launch_3k_disagg_sc.sh, which keeps
# the same SC + disagg stack but serves the judges on the allocation.
#
# NO CHECKPOINTING (inherited from the SC overlay). Size NRL_MAX_STEPS to fit
# one allocation.
#
# Required, in addition to the usual site variables:
#   NVIDIA_API_KEY         NVCF credential, read by the config as
#                          ${oc.env:NVIDIA_API_KEY}. Keep it in the environment
#                          or a sourced creds file — never on a command line.
#   OPENSANDBOX_BASE_URL   sandbox service endpoint
#   OPENSANDBOX_API_KEY    sandbox service credential, same handling
#   NS_SANDBOX_IMAGE       nemo-skills sandbox image the service can pull
#
# Recommended (see disagg_overrides.yaml for sizing):
#   NS_SANDBOX_POOL_REF=<pool-name>   claim PREWARMED pods from a server-side
#                                     Pool created before the job; set empty
#                                     ("") for direct creates at Gym startup
#   NS_SANDBOX_POOL_SIZE=128          keep sessions/pod ≈ CPUs/pod
#
# Judge endpoints and model names live in nvcf_judges.yaml; sandbox backends live
# in disagg_overrides.yaml. Neither is duplicated here, so they cannot drift.
#
# The Gym overlay must provide the sandbox_pool backend (NVIDIA-NeMo/Gym branch
# hemild/rlvr-osb-473f446f (the 6K Gym pin + sandbox commit) until merged).
#
# Optional: NRL_MAX_STEPS, WALLTIME, CONTEXT_PARALLEL_SIZE, DRY_RUN,
# NUM_TRAIN_NODES / NUM_GEN_NODES / NUM_GYM_NODES, STREAM_MIN_GROUPS,
# NUM_STORAGE_UNITS, REFIT_TRANSPORT — as launch_3k_pipeclean_sc.sh.
#
# Extra positional args are forwarded as Hydra overrides to ultra_launch.sh.
# =============================================================================
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

# Default config — callers may still override CONFIG_PATH explicitly.
export CONFIG_PATH="${CONFIG_PATH:-${SCRIPT_DIR}/pipeclean_3k_nvcf_disagg_sc.yaml}"

# The SC driver. data_plane.enabled=true (set in the config) is mandatory for it.
export TRAIN_ENTRYPOINT="${TRAIN_ENTRYPOINT:-./examples/run_grpo_single_controller.py}"

# =============================================================================
# NVCF judges
# =============================================================================
: "${NVIDIA_API_KEY:?NVIDIA_API_KEY is required (NVCF credential read by the config as \${oc.env:NVIDIA_API_KEY})}"
# Deployment-specific NVCF function routes consumed by nvcf_judges.yaml.
: "${GENRM_NVCF_MODEL:?GENRM_NVCF_MODEL is required (NVCF route for the GenRM judge)}"
: "${NL2BASH_NVCF_MODEL:?NL2BASH_NVCF_MODEL is required (NVCF route for the NL2Bash judge)}"
: "${SAFETY_NVCF_MODEL:?SAFETY_NVCF_MODEL is required (NVCF route for the safety judge)}"
export GENRM_NVCF_MODEL NL2BASH_NVCF_MODEL SAFETY_NVCF_MODEL
export NVIDIA_API_KEY

# Hand the judge wiring entirely to nvcf_judges.yaml. ultra_launch.sh emits a
# `.model=` or `.base_url=` Hydra override for any of these that is non-empty,
# and a CLI override beats the config — it would point a judge back at a local
# checkpoint while the YAML still looked correct. Empty rather than unset also
# defends against values inherited from the caller's shell.
export EXTERNAL_JUDGES=0
export GENRM_MODEL=""
export NL2BASH_JUDGE_MODEL=""
export SAFETY_JUDGE_MODEL=""
export GENRM_BASE_URL=""
export NL2BASH_BASE_URL=""

# =============================================================================
# Disaggregated sandbox service
# =============================================================================
export NO_COLOCATED_SANDBOX=1
: "${OPENSANDBOX_BASE_URL:?OPENSANDBOX_BASE_URL is required (sandbox service endpoint)}"
: "${OPENSANDBOX_API_KEY:?OPENSANDBOX_API_KEY is required (sandbox service credential)}"
: "${NS_SANDBOX_IMAGE:?NS_SANDBOX_IMAGE is required (sandbox image the service can pull)}"
export OPENSANDBOX_BASE_URL OPENSANDBOX_API_KEY NS_SANDBOX_IMAGE
# Unset-only default (`-` not `:-`): export NS_SANDBOX_POOL_REF="" forces direct
# creates at Gym startup, matching the header contract.
export NS_SANDBOX_POOL_REF="${NS_SANDBOX_POOL_REF-ns-tools-warm}"
export LEAN_SANDBOX_POOL_REF="${LEAN_SANDBOX_POOL_REF-math-lean-warm}"
# Half of 6K's 256, tracking the halved 8192-sample cohort.
export NS_SANDBOX_POOL_SIZE="${NS_SANDBOX_POOL_SIZE:-128}"

# =============================================================================
# Site defaults — the checkpoint and blend of the other 3K arms
# =============================================================================
export MODEL_PATH="${MODEL_PATH:-/lustre/fsw/portfolios/llmservice/users/jiaqiz/models/ultra_stage2sft_step300}"
export TRAIN_PATH="${TRAIN_PATH:-/lustre/fsw/portfolios/llmservice/users/jiaqiz/data/gym/rl-data-tools/blends/curriculum_v35_inescapable-sawfly.train.efforts0p15_qamathcode.jsonl}"
# The reference runs validate on the training blend; there is no separate split.
export VAL_PATH="${VAL_PATH:-${TRAIN_PATH}}"

# Image with the Gym venvs prebaked (python interpreter included — the stock
# nightly ships a venv tree whose interpreter is absent) and the sandbox deps
# installed. Pinned to a resolved image, not the moving nightly symlink.
#
# This is the disagg image the 6K disagg recipe uses. It is older than the
# nightly the pipeclean arms pin, but verified equivalent where it matters: its
# /opt/ray_venvs generation workers carry vLLM 0.25.1, which is what this
# branch's code needs (a pre-bump venv fails at import with "cannot import name
# ServingTokenization"). Its /opt/gym_venvs judge venvs are on 0.20.0 — the same
# as the nightly's, and unused on this arm anyway, since NVCF judges mean no
# local vLLM is ever launched.
#
# Byte-identical restripe of hemild/scale-6k/...plus_sandbox_venvs.sqsh. The
# original is laid out over 8 OSTs at 1 MB, and 752 nodes pulling 63 GB through
# 8 OSTs is the I/O concentration that took down an earlier run. This copy
# inherits the 350-OST / 16 MB layout the other arms' image already uses.
export CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/llmservice/users/sauramishra/images-striped/main_ultra_recipes_prebaked_venvs_20260730.plus_sandbox_venvs.sqsh}"
# Part of the sandbox-image delta: the image predates the tree, and ultra_launch
# mounts nemo_rl/ and examples/configs/ but NOT examples/, so both the Gym
# examples and the SC driver have to be bind-mounted in or the image's stale
# copies run against the mounted newer library.
GYM_EXAMPLES_DIR=$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)
export EXTRA_MOUNTS="${EXTRA_MOUNTS:-/lustre:/lustre,${GYM_EXAMPLES_DIR}:/opt/nemo-rl/examples/nemo_gym,${GYM_EXAMPLES_DIR}/../run_grpo_single_controller.py:/opt/nemo-rl/examples/run_grpo_single_controller.py}"
export PERSISTENT_CACHE="${PERSISTENT_CACHE:-/lustre/fsw/portfolios/llmservice/users/${USER}/.cache/nemotron_ultra}"
export HF_HOME="${HF_HOME:-/lustre/fsw/portfolios/llmservice/users/${USER}/hf_home}"

export SLURM_ACCOUNT="${SLURM_ACCOUNT:-nemotron_sw_pre}"
export SLURM_PARTITION="${SLURM_PARTITION:-batch}"

# Training matches every other 3K arm. Gym keeps 8 nodes, as the NVCF reference
# run did: no judge is served here, so their GPUs sit idle, but Gym still needs
# CPU capacity of its own for the resource servers, and borrowing it from the
# training or generation nodes would contend with the thing being measured.
# Generation absorbs the remainder to keep the total on the 16-node segment
# boundary, which costs it 6 nodes against the other two arms.
export NUM_TRAIN_NODES="${NUM_TRAIN_NODES:-256}"
export NUM_GEN_NODES="${NUM_GEN_NODES:-488}"
export NUM_GYM_NODES="${NUM_GYM_NODES:-8}"

# CP=16 is inherited from pipeclean_6k.yaml; allow an override for memory
# experiments.
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-16}"

# Same SC knobs as the other SC arms, so they differ only in judge and sandbox
# placement.
STREAM_MIN_GROUPS="${STREAM_MIN_GROUPS:-128}"
NUM_STORAGE_UNITS="${NUM_STORAGE_UNITS:-32}"
REFIT_TRANSPORT="${REFIT_TRANSPORT:-nccl_reshard}"

# Data-plane backend. mooncake_cpu carries rollouts over CPU RDMA and has no TCP
# fallback — it raises at data-plane setup if no RDMA device is usable, rather
# than degrading silently. DP_BACKEND=simple reverts to the SimpleStorage Ray
# actors, the only backend NUM_STORAGE_UNITS applies to.
DP_BACKEND="${DP_BACKEND:-mooncake_cpu}"

export EXP_NAME="${EXP_NAME:-ultra-3k-sc-nvcf-disagg}"
export WALLTIME="${WALLTIME:-4:00:00}"

# Reuse the container's prebaked venvs instead of building in-tree on first run
# (GYM_VENV_DIR= empty restores the stock in-tree behavior).
GYM_VENV_DIR="${GYM_VENV_DIR:-/opt/gym_venvs}"
VENV_OVERRIDE=()
if [ -n "${GYM_VENV_DIR}" ]; then
  VENV_OVERRIDE=("++env.nemo_gym.uv_venv_dir=${GYM_VENV_DIR}")
fi

# Preflight the sandbox dependency this variant introduces: ultra_launch overlays
# $PWD/3rdparty/Gym-workspace/Gym whenever that directory exists (even empty —
# an uninitialized submodule silently masks the container's baked Gym).
GYM_OVERLAY="$PWD/3rdparty/Gym-workspace/Gym"
if [ ! -s "${GYM_OVERLAY}/resources_servers/ns_tools/sandbox_pool.py" ]; then
  echo "ERROR: ${GYM_OVERLAY} is missing the sandbox_pool backend (empty or wrong Gym overlay?)." >&2
  echo "       Populate it with NVIDIA-NeMo/Gym branch hemild/rlvr-osb-473f446f and re-run." >&2
  exit 1
fi

exec bash "${SCRIPT_DIR}/../ultra_launch.sh" \
  "policy.megatron_cfg.context_parallel_size=${CONTEXT_PARALLEL_SIZE}" \
  "async_rl.min_groups_for_streaming_train=${STREAM_MIN_GROUPS}" \
  "data_plane.backend=${DP_BACKEND}" \
  "data_plane.simple.num_storage_units=${NUM_STORAGE_UNITS}" \
  "policy.generation.refit_transport=${REFIT_TRANSPORT}" \
  "${VENV_OVERRIDE[@]}" \
  "$@"

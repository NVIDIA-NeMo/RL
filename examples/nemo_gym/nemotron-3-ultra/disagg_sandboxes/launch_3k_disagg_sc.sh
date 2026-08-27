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
# launch_3k_disagg_sc.sh
#
# The BACKUP 3K test arm: SingleController with disaggregated sandboxes, but the
# judges served on the allocation instead of from NVCF. Use this if NVCF proves
# unusable; the primary arm is launch_3k_nvcf_disagg_sc.sh.
#
# Shape (GB200 NVL72, 4 GPUs/node), deliberately identical to the v1 baseline
# launch_3k_pipeclean.sh:
#   Hetgroup 0: 256 training + 494 generation + 2 Gym = 752 nodes (segment 16)
#   Hetgroup 1: 8 GenRM + 2 NL2Bash                   =  10 nodes (segment 2)
#   Total:      762 nodes / 3048 GPUs
#
# Judges are deployed exactly as the baseline deploys them — GenRM and NL2Bash on
# the external hetgroup, the safety judge in Gym — so this arm differs from the
# baseline only in the execution path (v1 -> SC) and the sandbox backend
# (colocated -> disaggregated), and differs from the NVCF arm only in judge
# placement. That makes NVCF a single isolated variable.
#
# This departs from the 6K disagg recipe, which serves all three judges inside
# Gym on 64 nodes. Reproducing that here would both change the node total and
# carry the 6K judge parallelism (GenRM TP4 x DP16 = 64 GPUs) into a half-size
# job, giving this arm roughly double the baseline's judge capacity. Set
# EXTERNAL_JUDGES=0 with NUM_GYM_NODES=32 to get the old shape back.
#
# NO CHECKPOINTING. The SC path raises if checkpointing.enabled is true, so size
# NRL_MAX_STEPS to fit one allocation.
#
# Required, in addition to the usual site variables:
#   OPENSANDBOX_BASE_URL   sandbox service endpoint
#   OPENSANDBOX_API_KEY    keep in the environment or a sourced creds file —
#                          never on the command line or in configs
#   NS_SANDBOX_IMAGE       nemo-skills sandbox image the service can pull
#
# Recommended at this scale (see disagg_overrides.yaml for sizing):
#   NS_SANDBOX_POOL_REF=<pool-name>   claim PREWARMED pods from a server-side
#                                     Pool created before the job
#   NS_SANDBOX_POOL_SIZE=128          keep sessions/pod ≈ CPUs/pod
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
export CONFIG_PATH="${CONFIG_PATH:-${SCRIPT_DIR}/pipeclean_3k_disagg_sc.yaml}"

# The SC driver. data_plane.enabled=true (set in the config) is mandatory for it.
export TRAIN_ENTRYPOINT="${TRAIN_ENTRYPOINT:-./examples/run_grpo_single_controller.py}"

# =============================================================================
# Disaggregated sandbox service
# =============================================================================
export NO_COLOCATED_SANDBOX=1
: "${OPENSANDBOX_BASE_URL:?OPENSANDBOX_BASE_URL is required (sandbox service endpoint)}"
: "${OPENSANDBOX_API_KEY:?OPENSANDBOX_API_KEY is required (sandbox service credential)}"
: "${NS_SANDBOX_IMAGE:?NS_SANDBOX_IMAGE is required (sandbox image the service can pull)}"
export OPENSANDBOX_BASE_URL OPENSANDBOX_API_KEY NS_SANDBOX_IMAGE
# Unset-only default (`-` not `:-`): export NS_SANDBOX_POOL_REF="" forces
# direct creates at Gym startup, matching the header contract.
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
# Byte-identical restripe of hemild/scale-6k/...plus_sandbox_venvs.sqsh. The
# original is laid out over 8 OSTs at 1 MB, and 752 nodes pulling 63 GB through
# 8 OSTs is the I/O concentration that took down an earlier run. This copy
# inherits the 350-OST / 16 MB layout the other arms' image already uses.
export CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/llmservice/users/sauramishra/images-striped/main_ultra_recipes_prebaked_venvs_20260730.plus_sandbox_venvs.sqsh}"
# Same stale-image delta as the v1 disagg wrapper, plus the SC driver itself:
# ultra_launch does not mount examples/, so run_grpo_single_controller.py has to
# be bind-mounted in or the image's older copy runs against the newer library.
GYM_EXAMPLES_DIR=$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)
export EXTRA_MOUNTS="${EXTRA_MOUNTS:-/lustre:/lustre,${GYM_EXAMPLES_DIR}:/opt/nemo-rl/examples/nemo_gym,${GYM_EXAMPLES_DIR}/../run_grpo_single_controller.py:/opt/nemo-rl/examples/run_grpo_single_controller.py}"
export PERSISTENT_CACHE="${PERSISTENT_CACHE:-/lustre/fsw/portfolios/llmservice/users/${USER}/.cache/nemotron_ultra}"
export HF_HOME="${HF_HOME:-/lustre/fsw/portfolios/llmservice/users/${USER}/hf_home}"

export GENRM_MODEL="${GENRM_MODEL:-/lustre/fsw/portfolios/llmservice/users/ansubramania/models/qwen235b_principle_comparison_genrm_step1230}"
export NL2BASH_JUDGE_MODEL="${NL2BASH_JUDGE_MODEL:-/lustre/fsw/portfolios/llmservice/users/ansubramania/models/Qwen3-235B-A22B-Instruct-2507-FP8}"
export SAFETY_JUDGE_MODEL="${SAFETY_JUDGE_MODEL:-/lustre/fsw/portfolios/llmservice/users/ansubramania/super_v3/model_checkpoints/Nemotron-Content-Safety-Reasoning-4B}"

export SLURM_ACCOUNT="${SLURM_ACCOUNT:-nemotron_sw_pre}"
export SLURM_PARTITION="${SLURM_PARTITION:-batch}"

# Serve GenRM and NL2Bash on the external hetgroup and the safety judge in Gym,
# at the baseline's pool sizes, so judge capacity matches the arm this is
# compared against. Gym needs only 2 nodes because the safety judge at TP=4 x
# DP=2 is its sole GPU consumer.
export EXTERNAL_JUDGES="${EXTERNAL_JUDGES:-1}"
export GENRM_REPLICAS="${GENRM_REPLICAS:-8}"
export GENRM_TENSOR_PARALLEL_SIZE="${GENRM_TENSOR_PARALLEL_SIZE:-4}"
export GENRM_REASONING_PARSER_NAME="${GENRM_REASONING_PARSER_NAME:-deepseek_r1}"
export GENRM_ENABLE_EXPERT_PARALLEL="${GENRM_ENABLE_EXPERT_PARALLEL:-0}"
export NL2BASH_REPLICAS="${NL2BASH_REPLICAS:-2}"
export NL2BASH_TENSOR_PARALLEL_SIZE="${NL2BASH_TENSOR_PARALLEL_SIZE:-4}"
export EXTERNAL_VLLM_SEGMENT_SIZE="${EXTERNAL_VLLM_SEGMENT_SIZE:-2}"
export EXTERNAL_VLLM_SKIP_PREFLIGHT="${EXTERNAL_VLLM_SKIP_PREFLIGHT:-1}"

# Node split, identical to launch_3k_pipeclean.sh. Keep the Ray total a multiple
# of 16 and NUM_TRAIN_NODES a multiple of 32 (TP * CP * PP = 128 GPUs).
export NUM_TRAIN_NODES="${NUM_TRAIN_NODES:-256}"
export NUM_GEN_NODES="${NUM_GEN_NODES:-494}"
export NUM_GYM_NODES="${NUM_GYM_NODES:-2}"

# CP=16 is inherited from pipeclean_6k.yaml; allow an override for memory
# experiments.
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-16}"

# The 6K values halved with the cohort: 128 is a quarter of the 512-group
# cohort, and storage units track per-step volume.
STREAM_MIN_GROUPS="${STREAM_MIN_GROUPS:-128}"
NUM_STORAGE_UNITS="${NUM_STORAGE_UNITS:-32}"

# Data-plane backend. mooncake_cpu carries rollouts over CPU RDMA and has no TCP
# fallback — it raises at data-plane setup if no RDMA device is usable, rather
# than degrading silently. DP_BACKEND=simple reverts to the SimpleStorage Ray
# actors, the only backend NUM_STORAGE_UNITS applies to.
DP_BACKEND="${DP_BACKEND:-mooncake_cpu}"

# Shard-to-shard weight refit, on by default in this variant. REFIT_TRANSPORT=null
# restores the full-tensor broadcast.
REFIT_TRANSPORT="${REFIT_TRANSPORT:-nccl_reshard}"

export EXP_NAME="${EXP_NAME:-ultra-3k-disagg-sandboxes-sc}"
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

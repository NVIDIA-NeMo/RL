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
# launch_6k_disagg.sh
#
# 6K-GPU Ultra pipeclean with DISAGGREGATED sandboxes: identical to
# launch_6k_pipeclean.sh except tool execution (ns_tools, math_formal_lean)
# runs on pods of an external sandbox service instead of colocated per-node
# nemo-skills containers — NO_COLOCATED_SANDBOX=1 skips the sandbox image
# requirement and the per-node sandbox deployment entirely.
#
# Shape (GB200 NVL72, 4 GPUs/node → 6144 GPUs), unchanged for comparability:
#   Training 512 / vLLM 960 / Gym 64
#
# Required, in addition to the usual site variables:
#   OPENSANDBOX_BASE_URL   sandbox service endpoint
#   OPENSANDBOX_API_KEY    keep in the environment or a sourced creds file —
#                          never on the command line or in configs
#   NS_SANDBOX_IMAGE       nemo-skills sandbox image the service can pull
#
# Recommended at this scale (see disagg_overrides.yaml for sizing):
#   NS_SANDBOX_POOL_REF=<pool-name>   claim PREWARMED pods from a server-side
#                                     Pool created before the job (sub-second
#                                     per-pod allocation); set empty ("") = direct
#                                     creates at Gym startup
#   NS_SANDBOX_POOL_SIZE=256          keep sessions/pod ≈ CPUs/pod
#
# The Gym overlay must provide the sandbox_pool backend (NVIDIA-NeMo/Gym
# branch hemild/rlvr-osb-473f446f (the 6K Gym pin + sandbox commit) until merged).
#
# Optional: NRL_MAX_STEPS, WALLTIME, CONTEXT_PARALLEL_SIZE, DRY_RUN,
# NUM_TRAIN_NODES / NUM_GEN_NODES / NUM_GYM_NODES — as launch_6k_pipeclean.sh.
#
# Extra positional args are forwarded as Hydra overrides to ultra_launch.sh.
# =============================================================================
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

# Default config — callers may still override CONFIG_PATH explicitly.
export CONFIG_PATH="${CONFIG_PATH:-${SCRIPT_DIR}/pipeclean_6k_disagg.yaml}"

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
export NS_SANDBOX_POOL_SIZE="${NS_SANDBOX_POOL_SIZE:-256}"

# =============================================================================
# Site defaults — identical to launch_6k_pipeclean.sh (minus SANDBOX_CONTAINER)
# =============================================================================
# Kept byte-identical to the baseline wrapper on purpose: the disagg comparison
# is only meaningful if both runs read the same checkpoint, blend and judges.
# =============================================================================
export MODEL_PATH="${MODEL_PATH:-/lustre/fsw/portfolios/llmservice/users/jiaqiz/models/ultra_stage2sft_step300}"
export TRAIN_PATH="${TRAIN_PATH:-/lustre/fsw/portfolios/llmservice/users/jiaqiz/data/gym/rl-data-tools/blends/curriculum_v35_inescapable-sawfly.train.efforts0p15_qamathcode.jsonl}"
# The reference runs validate on the training blend; there is no separate split.
export VAL_PATH="${VAL_PATH:-${TRAIN_PATH}}"

# Image with the Gym venvs prebaked (python interpreter included — the stock
# nightly ships a venv tree whose interpreter is absent) and the sandbox deps
# installed. Pinned to a resolved image, not the moving nightly symlink.
export CONTAINER="${CONTAINER:-/lustre/fs1/portfolios/nemotron/projects/nemotron_sw_post/users/hemild/scale-6k/main_ultra_recipes_prebaked_venvs_20260730.plus_sandbox_venvs.sqsh}"
# Part of the sandbox-image delta: this image predates the tree, and
# ultra_launch mounts nemo_rl/ and examples/configs/ but NOT examples/nemo_gym/,
# so without this the driver executes the IMAGE's stale run_grpo_nemo_gym.py
# against the mounted newer library (TypeError: 'GRPOConfig' is not
# subscriptable — hit live by the 8-node smoke).
GYM_EXAMPLES_DIR=$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)
export EXTRA_MOUNTS="${EXTRA_MOUNTS:-/lustre:/lustre,${GYM_EXAMPLES_DIR}:/opt/nemo-rl/examples/nemo_gym}"
export PERSISTENT_CACHE="${PERSISTENT_CACHE:-/lustre/fsw/portfolios/llmservice/users/${USER}/.cache/nemotron_ultra}"
export HF_HOME="${HF_HOME:-/lustre/fsw/portfolios/llmservice/users/${USER}/hf_home}"

export GENRM_MODEL="${GENRM_MODEL:-/lustre/fsw/portfolios/llmservice/users/ansubramania/models/qwen235b_principle_comparison_genrm_step1230}"
export NL2BASH_JUDGE_MODEL="${NL2BASH_JUDGE_MODEL:-/lustre/fsw/portfolios/llmservice/users/ansubramania/models/Qwen3-235B-A22B-Instruct-2507-FP8}"
export SAFETY_JUDGE_MODEL="${SAFETY_JUDGE_MODEL:-/lustre/fsw/portfolios/llmservice/users/ansubramania/super_v3/model_checkpoints/Nemotron-Content-Safety-Reasoning-4B}"

export SLURM_ACCOUNT="${SLURM_ACCOUNT:-nemotron_sw_pre}"
export SLURM_PARTITION="${SLURM_PARTITION:-batch}"
# A 6K allocation may additionally need SLURM_QOS, SLURM_RESERVATION and
# EXCLUDE_NODES; see ultra_launch.sh for the variable names.

# 6K node split, identical to the baseline pipeclean. Callers may override.
export NUM_TRAIN_NODES="${NUM_TRAIN_NODES:-512}"
export NUM_GEN_NODES="${NUM_GEN_NODES:-960}"
export NUM_GYM_NODES="${NUM_GYM_NODES:-64}"

# CP=16 is baked into pipeclean_6k.yaml; allow an override for memory experiments.
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-16}"

# Sensible defaults for short 6K hero / pipeclean allocations when unset.
export EXP_NAME="${EXP_NAME:-ultra-6k-disagg-sandboxes}"
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
  "${VENV_OVERRIDE[@]}" \
  "$@"

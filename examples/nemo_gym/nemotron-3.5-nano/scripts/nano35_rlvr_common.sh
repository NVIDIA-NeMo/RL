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
#
# Shared native launcher for the Nano 3.5 prefix-checkpoint recipe.
# Uses only this NeMo-RL checkout and tools/external_gym_vllm/.
set -euo pipefail

SCRIPT_DIR="$(cd -L -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -L)"
RECIPE_ROOT="$(cd -L -- "${SCRIPT_DIR}/.." && pwd -L)"
NEMO_RL_ROOT="${NEMO_RL_ROOT:-$(cd -L -- "${SCRIPT_DIR}/../../../.." && pwd -L)}"

if [[ ! -f "${NEMO_RL_ROOT}/ray.sub" ]]; then
  echo "ERROR: NeMo RL checkout is unavailable at ${NEMO_RL_ROOT}" >&2
  exit 1
fi

# -----------------------------------------------------------------------------
# Site
# -----------------------------------------------------------------------------
# A site profile supplies cluster-specific assets; explicit exports win.
# No pipeline checkout or private pipeline .env is read.
SITE_OVERLAY="${SITE_OVERLAY:-}"
SITE_OVERLAY_USED="none (explicit environment)"
if [[ -n "${SITE_OVERLAY}" ]]; then
  if [[ ! -f "${SITE_OVERLAY}" ]]; then
    echo "ERROR: SITE_OVERLAY is not a file: ${SITE_OVERLAY}" >&2
    exit 1
  fi
  source "${SITE_OVERLAY}"
  SITE_OVERLAY_USED="${SITE_OVERLAY}"
fi

# -----------------------------------------------------------------------------
# Required inputs
# -----------------------------------------------------------------------------
: "${EXP_NAME:?Set EXP_NAME to identify the run and its checkpoint lineage}"
: "${CONTAINER:?Set CONTAINER in the site profile or environment}"
export CONTAINER
if [[ ! -f "${CONTAINER}" ]]; then
  echo "ERROR: CONTAINER does not exist: ${CONTAINER}" >&2
  exit 1
fi

# -----------------------------------------------------------------------------
# Code provenance: report it, do not enforce it
# -----------------------------------------------------------------------------
# The run executes whatever the checkout currently is, so the commits are
# recorded and printed rather than checked against an expected value. A dirty
# tree is worth saying out loud, because USE_SNAPSHOT=0 means the run reads the
# checkout live.
NEMO_RL_COMMIT="$(git -C "${NEMO_RL_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
NEMO_RL_BRANCH="$(git -C "${NEMO_RL_ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
GYM_ROOT="${NEMO_RL_ROOT}/3rdparty/Gym-workspace/Gym"
if [[ ! -f "${GYM_ROOT}/responses_api_models/local_vllm_model/local_vllm_model_actor.py" ]]; then
  echo "ERROR: Gym checkout is unavailable at ${GYM_ROOT}" >&2
  echo "  Initialize it with: git -C ${NEMO_RL_ROOT} submodule update --init 3rdparty/Gym-workspace/Gym" >&2
  exit 1
fi
GYM_COMMIT="$(git -C "${GYM_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"

# -----------------------------------------------------------------------------
# Gym venvs: the image's prebuilt trees when they provably match, else rebuild
# -----------------------------------------------------------------------------
# The image bakes one venv per Gym server under /opt/gym_venvs, and each
# installs nemo_gym *editable* against /opt/nemo-rl/3rdparty/Gym-workspace/Gym.
# Reusing them skips the ~71 concurrent `uv sync` calls at spinup that job
# 3668422 died inside of, and costs nothing at runtime: with the recipe's
# `skip_venv_if_present: true`, Gym's setup_env_command emits a bare
# `source <venv>/bin/activate` once it finds bin/python and bin/activate, so it
# never invokes uv and never writes to the read-only squashfs.
#
# The catch is that an editable install ties those venvs to the one Gym
# revision they were built from, while the image is versioned independently of
# this checkout. So the fast path is taken by proof, not by assumption: read
# the Gym commit out of the image (single-file `unsquashfs -cat`, ~0.1s even on
# an 89 GiB image) and use it only when it equals what we pin. Getting this
# wrong in the silent direction is expensive -- skew surfaces as an ImportError
# at Gym spinup, ~20 minutes into a held allocation -- whereas rebuilding into
# PERSISTENT_CACHE is merely slow, so an unknown answer falls back.
#
# A dirty Gym checkout also falls back: commit equality cannot see uncommitted
# edits. Note that even a clean tree only proves the *sources* match, and the
# venvs additionally encode dependencies, so set USE_IMAGE_GYM=0 by hand after
# changing any server's pyproject.toml or requirements.txt.
_IMAGE_GYM_COMMIT="unknown"
if [[ "${CONTAINER}" == *.sqsh ]] && command -v unsquashfs >/dev/null 2>&1; then
  # `|| true` is load-bearing: unsquashfs exits 2 when the path is absent (an
  # image without the submodule gitdir), and under `set -euo pipefail` that
  # kills the launcher outright instead of falling back to the slow path.
  _IMAGE_GYM_COMMIT="$(unsquashfs -cat "${CONTAINER}" \
    'opt/nemo-rl/.git/modules/3rdparty/Gym-workspace/Gym/HEAD' 2>/dev/null \
    | tr -d '[:space:]' || true)"
  # A detached submodule gitdir holds the raw sha; anything else (an unresolved
  # "ref: refs/..." line, or an image without the gitdir) is not comparable.
  [[ "${_IMAGE_GYM_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || _IMAGE_GYM_COMMIT="unknown"
fi
# `|| true` again: under pipefail, `head -1` closing the pipe early can leave
# git with SIGPIPE on a tree with many changes -- exactly the dirty case this
# is meant to detect.
GYM_DIRTY="$(git -C "${GYM_ROOT}" status --porcelain 2>/dev/null | head -1 || true)"
if [[ -z "${USE_IMAGE_GYM:-}" ]]; then
  if [[ "${_IMAGE_GYM_COMMIT}" != "unknown" \
        && "${_IMAGE_GYM_COMMIT}" == "${GYM_COMMIT}" \
        && -z "${GYM_DIRTY}" ]]; then
    USE_IMAGE_GYM=1
  else
    USE_IMAGE_GYM=0
  fi
fi
export USE_IMAGE_GYM

# -----------------------------------------------------------------------------
# Workload config
# -----------------------------------------------------------------------------
# The recipe configs are mounted together; inheritance is relative to this
# directory, not the caller's working directory. The directory is mounted at
# /opt/nano35-config, and rlvr_sc.yaml reaches rlvr.yaml through its
# defaults: key, so the two must stay side by side.
export RECIPES_DIR_HOST="${RECIPES_DIR_HOST:-$(cd -L -- "${SCRIPT_DIR}/../configs" && pwd -L)}"
for required in rlvr.yaml rlvr_sc.yaml; do
  if [[ ! -f "${RECIPES_DIR_HOST}/${required}" ]]; then
    echo "ERROR: workload config missing: ${RECIPES_DIR_HOST}/${required}" >&2
    exit 1
  fi
done

: "${CONFIG_PATH_HOST:?CONFIG_PATH_HOST is required (set by the arm script)}"
[[ "${CONFIG_PATH_HOST}" == /* ]] || CONFIG_PATH_HOST="${NEMO_RL_ROOT}/${CONFIG_PATH_HOST}"
if [[ ! -f "${CONFIG_PATH_HOST}" ]]; then
  echo "ERROR: CONFIG_PATH_HOST does not exist: ${CONFIG_PATH_HOST}" >&2
  exit 1
fi
CONFIG_MOUNT_DIR_HOST="$(cd -L -- "$(dirname -- "${CONFIG_PATH_HOST}")" && pwd -L)"
CONFIG_PATH_HOST="${CONFIG_MOUNT_DIR_HOST}/$(basename -- "${CONFIG_PATH_HOST}")"
CONFIG_PATH="/opt/nano35-config/$(basename -- "${CONFIG_PATH_HOST}")"

# The driver is invoked as `cd /opt/nemo-rl && uv run ${TRAIN_ENTRYPOINT}`.
# Overlaying nemo_rl/ does not replace the entrypoint file itself, so a
# container-baked run_grpo_single_controller.py would keep winning over the
# checkout even though TRAIN_ENTRYPOINT was validated against it. The mount
# list below binds the live entrypoints over the image copies.
TRAIN_ENTRYPOINT="${TRAIN_ENTRYPOINT:-./examples/nemo_gym/run_grpo_nemo_gym.py}"
if [[ ! -f "${NEMO_RL_ROOT}/${TRAIN_ENTRYPOINT#./}" ]]; then
  echo "ERROR: training entrypoint not found: ${NEMO_RL_ROOT}/${TRAIN_ENTRYPOINT#./}" >&2
  exit 1
fi

# -----------------------------------------------------------------------------
# Results, checkpoints and logs
# -----------------------------------------------------------------------------
# Reusing EXP_NAME resumes from the same checkpoint directory. Choose a new name
# to start a separate lineage.
#
# Results, caches and the checkout must be visible under the site's shared
# mount on every node. Reusing EXP_NAME resumes the same checkpoint lineage.
export EXTERNAL_VLLM_SHARED_ROOT="${EXTERNAL_VLLM_SHARED_ROOT:-/scratch/fsw}"
export RESULTS_ROOT="${RESULTS_ROOT:-${NEMO_RL_ROOT}/results}"
export RESULTS_DIR="${RESULTS_DIR:-${RESULTS_ROOT}/${EXP_NAME}}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-${RESULTS_DIR}}"
export BASE_LOG_DIR="${BASE_LOG_DIR:-${RESULTS_DIR}}"

# An override that lands off the shared root fails late and expensively --
# partway into a rung, with the job holding 86 nodes -- so reject it here
# instead. run_in_allocation.sh performs the same check, but from inside the
# allocation; this one costs a shell prompt.
# The checkout must also be visible as Slurm's submission directory.
for _dir_var in NEMO_RL_ROOT RESULTS_DIR CHECKPOINT_DIR BASE_LOG_DIR; do
  case "${!_dir_var}" in
    "${EXTERNAL_VLLM_SHARED_ROOT}"/*) ;;
    *)
      echo "ERROR: ${_dir_var}=${!_dir_var} is not under EXTERNAL_VLLM_SHARED_ROOT=${EXTERNAL_VLLM_SHARED_ROOT}." >&2
      case "${_dir_var}" in
        NEMO_RL_ROOT)
          echo "  Only that root is mounted into the container, so a checkout" >&2
          echo "  elsewhere is invisible on the compute nodes. Move this one" >&2
          echo "  under it and launch from there." >&2
          ;;
        *)
          echo "  Checkpoints for this recipe are terabytes; point RESULTS_ROOT (or" >&2
          echo "  ${_dir_var}) at a path under that root." >&2
          ;;
      esac
      exit 1
      ;;
  esac
done
unset _dir_var

RUN_DIR="${RESULTS_DIR}/runs/$(date +%Y%m%d-%H%M)"
LOG_DIR="${RUN_DIR}/logs"
SLURM_LOG_DIR="${RUN_DIR}/slurm"
mkdir -p "${CHECKPOINT_DIR}" "${LOG_DIR}" "${SLURM_LOG_DIR}" "${RESULTS_DIR}/runs"
ln -sfn "${RUN_DIR}" "${RESULTS_DIR}/runs/latest"

# -----------------------------------------------------------------------------
# Model, blend and judges
# -----------------------------------------------------------------------------
# Site assets are explicit: moving the recipe must not silently select a
# different policy, blend or reward model from another cluster.
: "${MODEL_PATH:?Set MODEL_PATH in the site profile}"
: "${TRAIN_PATH:?Set TRAIN_PATH in the site profile}"
export MODEL_PATH TRAIN_PATH
if [[ "${NRL_CHECKPOINT_SUBSET:-0}" == "1" ]]; then
  if [[ -z "${CHECKPOINT_TRAIN_PATH:-}" || ! -r "${CHECKPOINT_TRAIN_PATH}" || ! -s "${CHECKPOINT_TRAIN_PATH}" ]]; then
    echo "ERROR: Set CHECKPOINT_TRAIN_PATH to a readable, nonempty checkpoint-filtered JSONL." >&2
    echo "  Provide a filtered JSONL, or set NRL_CHECKPOINT_SUBSET=0 for the original blend." >&2
    exit 1
  fi
  export TRAIN_PATH="${CHECKPOINT_TRAIN_PATH}"
fi
export VAL_PATH="${VAL_PATH:-${TRAIN_PATH}}"

: "${GENRM_MODEL:?Set GENRM_MODEL in the site profile}"
: "${GENRM_REASONING_PARSER:?Set GENRM_REASONING_PARSER in the site profile}"
: "${NL2BASH_JUDGE_MODEL:?Set NL2BASH_JUDGE_MODEL in the site profile}"
: "${SAFETY_JUDGE_MODEL:?Set SAFETY_JUDGE_MODEL in the site profile}"
: "${SANDBOX_CONTAINER:?Set SANDBOX_CONTAINER in the site profile}"
export GENRM_MODEL GENRM_REASONING_PARSER NL2BASH_JUDGE_MODEL SAFETY_JUDGE_MODEL SANDBOX_CONTAINER
export GENRM_REASONING_PARSER_NAME="${GENRM_REASONING_PARSER_NAME:-ultra_v3}"
# Keep each node's sandbox self-contained. Left to itself the sandbox image
# auto-detects multi-node from SLURM_JOB_NODELIST and elects one nginx master
# that fronts every uWSGI worker in the allocation by hostname, while the other
# nodes run nginx as a bare proxy to that master. Two consequences at 66 Ray
# nodes: the master's config carries nproc*nodes = 9504 hostname upstreams, and
# nginx resolves all of them at parse time and treats a single failed lookup as
# a fatal [emerg] -- so the master exits non-zero, ray.sub's
# --kill-on-bad-exit=1 sees the sidecar die, and the job is torn down about
# three minutes in, before training starts. Failure rate tracks the lookup
# count: 864 upstreams survived 5/5 runs, 2112 survived 1/2, 9504 survived 0/2.
# The master is also a funnel for every sandbox request in the job and a single
# point of failure.
#
# Forcing single-node mode gives each node its own nginx over its own local
# workers on 127.0.0.1, so there are no cross-node lookups to lose and no
# funnel. This is what ray.sub already assumes: it starts one sidecar per node
# and probes localhost:6000 on each. It also matches the nemo-skills-sandbox
# build this recipe was validated against, which has no multi-node mode at all
# and ran 66 nodes x 140 workers in 43-49s.
#
# NUM_WORKERS is deliberately left unset so it keeps tracking nproc --all;
# with local-only upstreams the worker count no longer drives any DNS risk.
export SANDBOX_COMMAND="${SANDBOX_COMMAND:-SANDBOX_FORCE_SINGLE_NODE=1 /start-with-nginx.sh}"
export NEMO_SKILLS_SANDBOX_PORT="${NEMO_SKILLS_SANDBOX_PORT:-6000}"
export PERSISTENT_CACHE="${PERSISTENT_CACHE:-${NEMO_RL_ROOT}/.cache/nano35}"
export HF_HOME="${HF_HOME:-${PERSISTENT_CACHE}/hf_home}"

# -----------------------------------------------------------------------------
# External-vLLM tools staging
# -----------------------------------------------------------------------------
# The external-vLLM helper bind-mounts its own tools directory into the judge
# containers and rejects any source off the shared root. It derives that
# directory from the checkout root, so a checkout that lives off the shared root
# fails the check seconds into an 86-node allocation.
#
# Stage the tools beside the run's own results instead: on Lustre, in the
# lineage that reads them, refreshed each submit so a fix to the checkout is
# picked up. Refreshing under a running job is the same exposure USE_SNAPSHOT=0
# already accepts for nemo_rl itself, and the mount is read-only.
export EXTERNAL_VLLM_TOOLS_DIR_HOST="${EXTERNAL_VLLM_TOOLS_DIR_HOST:-${RESULTS_DIR}/external_gym_vllm}"
if [[ "${EXTERNAL_VLLM_TOOLS_DIR_HOST}" == "${RESULTS_DIR}/external_gym_vllm" ]]; then
  mkdir -p "${EXTERNAL_VLLM_TOOLS_DIR_HOST}"
  cp -a "${NEMO_RL_ROOT}/tools/external_gym_vllm/." "${EXTERNAL_VLLM_TOOLS_DIR_HOST}/"
fi

# Everything the external-service helper mounts, checked here rather than
# there. Its own check runs inside the allocation, so a bad path costs 86 nodes
# and a place in the queue.
for _external_path in EXTERNAL_VLLM_TOOLS_DIR_HOST GENRM_MODEL GENRM_REASONING_PARSER NL2BASH_JUDGE_MODEL; do
  case "${!_external_path}" in
    "" | "${EXTERNAL_VLLM_SHARED_ROOT}"/*) ;;
    *)
      echo "ERROR: ${_external_path}=${!_external_path} is not under ${EXTERNAL_VLLM_SHARED_ROOT}." >&2
      echo "  The external judge container mounts it and refuses anything else." >&2
      exit 1
      ;;
  esac
done
unset _external_path

# -----------------------------------------------------------------------------
# Job shape: 86 nodes = 32 train + 32 generation + 2 Gym + 20 external services
# -----------------------------------------------------------------------------
# The external hetgroup holds 16 GenRM nodes and 4 NL2Bash nodes; the two
# remaining Gym nodes serve the local safety judge, which stays in-cluster.
#
# GPUS_PER_NODE must be exported before any pool is registered: registration
# sizes hetgroup 1 from it, and it is exported to ray.sub, so it has to match
# the physical GPUs per node for both components.
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
export NUM_TRAIN_NODES="${NUM_TRAIN_NODES:-32}"
export NUM_GEN_NODES="${NUM_GEN_NODES:-32}"
export NUM_GYM_NODES="${NUM_GYM_NODES:-2}"
export SEGMENT_SIZE="${SEGMENT_SIZE:-2}"
export EXTERNAL_VLLM_SEGMENT_SIZE="${EXTERNAL_VLLM_SEGMENT_SIZE:-2}"

NUM_ACTOR_NODES=$((NUM_TRAIN_NODES + NUM_GEN_NODES))
NUM_RAY_NODES=$((NUM_ACTOR_NODES + NUM_GYM_NODES))

for _n in NUM_TRAIN_NODES NUM_GEN_NODES; do
  if (( ${!_n} <= 0 )); then
    echo "ERROR: ${_n} must be > 0 (got ${!_n})" >&2; exit 1
  fi
done
unset _n
if (( NUM_RAY_NODES % SEGMENT_SIZE != 0 )); then
  echo "ERROR: hetgroup 0 is ${NUM_RAY_NODES} nodes, not a multiple of SEGMENT_SIZE=${SEGMENT_SIZE}." >&2
  echo "  train=${NUM_TRAIN_NODES} + generation=${NUM_GEN_NODES} + gym=${NUM_GYM_NODES}" >&2
  exit 1
fi

# -----------------------------------------------------------------------------
# External judge pools
# -----------------------------------------------------------------------------
# Registered through NeMo RL's own interface, so adding or resizing a pool needs
# no change to run_in_allocation.sh. Serving settings are deployment concerns and
# belong here rather than in the wrapper.
GENRM_BASE_URL="__GENRM_BASE_URL__"
NL2BASH_BASE_URL="__NL2BASH_BASE_URL__"
export GENRM_REPLICAS="${GENRM_REPLICAS:-8}"
export GENRM_TENSOR_PARALLEL_SIZE="${GENRM_TENSOR_PARALLEL_SIZE:-8}"
export NL2BASH_REPLICAS="${NL2BASH_REPLICAS:-4}"
export NL2BASH_TENSOR_PARALLEL_SIZE="${NL2BASH_TENSOR_PARALLEL_SIZE:-4}"
GENRM_SERVED_MODEL_NAME="${GENRM_SERVED_MODEL_NAME:-model}"
NL2BASH_SERVED_MODEL_NAME="${NL2BASH_SERVED_MODEL_NAME:-model}"
GENRM_CONTAINER="${GENRM_CONTAINER:-${CONTAINER}}"
NL2BASH_CONTAINER="${NL2BASH_CONTAINER:-${GENRM_CONTAINER}}"
GENRM_VLLM_PYTHON="${GENRM_VLLM_PYTHON:-/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/bin/python}"
NL2BASH_VLLM_PYTHON="${NL2BASH_VLLM_PYTHON:-${GENRM_VLLM_PYTHON}}"
GENRM_TOOL_CALL_PARSER="${GENRM_TOOL_CALL_PARSER:-qwen3_coder}"
NL2BASH_TOOL_CALL_PARSER="${NL2BASH_TOOL_CALL_PARSER:-hermes}"
NL2BASH_ATTENTION_BACKEND="${NL2BASH_ATTENTION_BACKEND:-TRITON_ATTN}"
GENRM_COMPILATION_CONFIG="${GENRM_COMPILATION_CONFIG:-{\"pass_config\":{\"fuse_allreduce_rms\":false}}}"
GENRM_MODEL_LOADER_EXTRA_CONFIG="${GENRM_MODEL_LOADER_EXTRA_CONFIG:-{\"enable_multithread_load\":true,\"num_threads\":96}}"
NL2BASH_COMPILATION_CONFIG="${NL2BASH_COMPILATION_CONFIG:-{\"cudagraph_capture_sizes\":[1,2,4,8,16,32,64,128,256]}}"
NL2BASH_MODEL_LOADER_EXTRA_CONFIG="${NL2BASH_MODEL_LOADER_EXTRA_CONFIG:-{\"enable_multithread_load\":true,\"num_threads\":112}}"

source "${NEMO_RL_ROOT}/tools/external_gym_vllm/pool_config.sh"
EXTERNAL_VLLM_POOLS=""
export EXTERNAL_VLLM_LB_PYTHON="${EXTERNAL_VLLM_LB_PYTHON:-/opt/nemo_rl_venv/bin/python}"

# --shared-path makes the parser plugin readable inside the pool container; the
# vLLM flag below is what actually loads it. Both are conditional because the
# flag reaches the container's vLLM CLI unchanged (serve_vllm_on_ray.py just
# delegates), so an image whose vLLM lacks it would fail on an empty value.
register_external_vllm_pool GENRM \
  --display-name GenRM \
  --model "${GENRM_MODEL}" \
  --container "${GENRM_CONTAINER}" \
  --python "${GENRM_VLLM_PYTHON}" \
  --replicas "${GENRM_REPLICAS}" \
  --tensor-parallel-size "${GENRM_TENSOR_PARALLEL_SIZE}" \
  --served-model-name "${GENRM_SERVED_MODEL_NAME}" \
  --vllm-port "${GENRM_VLLM_PORT:-8000}" \
  --lb-port "${GENRM_LB_PORT:-9213}" \
  --startup-timeout "${GENRM_STARTUP_TIMEOUT:-3600}" \
  --url-placeholder "${GENRM_BASE_URL}" \
  ${GENRM_REASONING_PARSER:+--shared-path "${GENRM_REASONING_PARSER}"}
# mnnvl, not trtllm: a GenRM replica at TP=8 spans two nodes, and vLLM rejects
# the trtllm flashinfer allreduce for multi-node collectives ("Flashinfer
# allreduce is not supported for multi-node allreduce with 'trtllm' backend"),
# failing in determine_available_memory after the weights are already loaded.
# The two nodes are placed in one NVLink domain by EXTERNAL_VLLM_SEGMENT_SIZE,
# which is what mnnvl needs. NL2Bash is unaffected: TP=4 stays on one node.
external_vllm_pool_env GENRM \
  "FLASHINFER_WORKSPACE_BASE=/tmp" \
  "VLLM_FLASHINFER_ALLREDUCE_BACKEND=mnnvl" \
  "VLLM_ALLREDUCE_USE_SYMM_MEM=0"
genrm_vllm_args=(
  --trust-remote-code
  --dtype bfloat16
  --kv-cache-dtype fp8
  --max-num-seqs 256
  --gpu-memory-utilization 0.95
  --enable-prefix-caching
  --reasoning-parser "${GENRM_REASONING_PARSER_NAME}"
  --enable-auto-tool-choice
  --tool-call-parser "${GENRM_TOOL_CALL_PARSER}"
  --compilation-config "${GENRM_COMPILATION_CONFIG}"
  --model-loader-extra-config "${GENRM_MODEL_LOADER_EXTRA_CONFIG}"
)
[[ -n "${GENRM_REASONING_PARSER}" ]] && genrm_vllm_args+=(--reasoning-parser-plugin "${GENRM_REASONING_PARSER}")
[[ "${GENRM_ENABLE_EXPERT_PARALLEL:-1}" == "1" ]] && genrm_vllm_args+=(--enable-expert-parallel)
external_vllm_pool_args GENRM "${genrm_vllm_args[@]}"

register_external_vllm_pool NL2BASH \
  --display-name NL2Bash \
  --model "${NL2BASH_JUDGE_MODEL}" \
  --container "${NL2BASH_CONTAINER}" \
  --python "${NL2BASH_VLLM_PYTHON}" \
  --replicas "${NL2BASH_REPLICAS}" \
  --tensor-parallel-size "${NL2BASH_TENSOR_PARALLEL_SIZE}" \
  --served-model-name "${NL2BASH_SERVED_MODEL_NAME}" \
  --vllm-port "${NL2BASH_VLLM_PORT:-8000}" \
  --lb-port "${NL2BASH_LB_PORT:-9214}" \
  --startup-timeout "${NL2BASH_STARTUP_TIMEOUT:-3600}" \
  --url-placeholder "${NL2BASH_BASE_URL}"
external_vllm_pool_env NL2BASH \
  "FLASHINFER_WORKSPACE_BASE=/tmp" \
  "VLLM_USE_FLASHINFER_MOE_FP16=0" \
  "VLLM_USE_FLASHINFER_MOE_FP8=0" \
  "VLLM_USE_DEEP_GEMM=0" \
  "VLLM_MOE_USE_DEEP_GEMM=0" \
  "NCCL_MNNVL_ENABLE=1"
nl2bash_vllm_args=(
  --dtype bfloat16
  --pipeline-parallel-size 1
  --max-model-len 131072
  --max-num-seqs 256
  --gpu-memory-utilization 0.85
  --enable-prefix-caching
  --enable-chunked-prefill
  --enable-auto-tool-choice
  --tool-call-parser "${NL2BASH_TOOL_CALL_PARSER}"
  --attention-backend "${NL2BASH_ATTENTION_BACKEND}"
  --compilation-config "${NL2BASH_COMPILATION_CONFIG}"
  --model-loader-extra-config "${NL2BASH_MODEL_LOADER_EXTRA_CONFIG}"
)
[[ "${NL2BASH_ENABLE_EXPERT_PARALLEL:-1}" == "1" ]] && nl2bash_vllm_args+=(--enable-expert-parallel)
external_vllm_pool_args NL2BASH "${nl2bash_vllm_args[@]}"

# Registration computes this from every pool's replicas x TP / GPUS_PER_NODE.
NUM_EXTERNAL_SERVICE_NODES="${EXTERNAL_VLLM_NUM_NODES}"
NUM_TOTAL_NODES=$((NUM_RAY_NODES + NUM_EXTERNAL_SERVICE_NODES))

export \
  EXTERNAL_VLLM_LB_PYTHON \
  EXTERNAL_VLLM_POOLS \
  EXTERNAL_VLLM_SHARED_ROOT \
  EXTERNAL_VLLM_TOOLS_DIR_HOST

# -----------------------------------------------------------------------------
# Slurm
# -----------------------------------------------------------------------------
export SLURM_ACCOUNT="${SLURM_ACCOUNT:-nemotron_sw_post}"
export SLURM_PARTITION="${SLURM_PARTITION:-batch}"
export SLURM_QOS="${SLURM_QOS:-normal}"
export WALLTIME="${WALLTIME:-4:00:00}"
EXCLUDE_NODES="${EXCLUDE_NODES:-}"
JOB_NAME="${JOB_NAME:-${EXP_NAME}}"
if [[ -z "${SLURM_COMMENT:-}" ]]; then
  export SLURM_COMMENT='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"90","reason":"other","description":"Nano 3.5 amplified-dolphin RLVR prefix-checkpoint convergence"}}'
fi
SLURM_COMMENT_ARGS=()
[[ -n "${SLURM_COMMENT:-}" ]] && SLURM_COMMENT_ARGS=(--comment="${SLURM_COMMENT}")

# Runs to step 100, with checkpoints every 10 steps for eval rungs.
# Compared against the restored step count, so a requeued job continues toward
# 100 rather than running 100 more steps. Reaching 100 takes several
# allocations: job 3698210 managed roughly 28 steps inside a 4h WALLTIME, so
# relaunching under the SAME EXP_NAME is what resumes -- CHECKPOINT_DIR is
# derived from it, and a new name starts from step 0.
export NRL_MAX_STEPS="${NRL_MAX_STEPS:-100}"
export WANDB_ENTITY="${WANDB_ENTITY:-nvidia}"
export WANDB_PROJ="${WANDB_PROJ:-nano35-rlvr-main-tot}"
WANDB_NAME="${WANDB_NAME:-${EXP_NAME}}"

# W&B needs the key in the environment; a host-side `wandb login` does not
# carry in, because ray.sub:259 adds --no-container-mount-home to every srun
# and ~/.netrc is therefore invisible inside the container. Enabling the logger
# without a key is not a soft failure: SingleControllerActor builds Logger in
# __init__, so wandb.init raises "No API key configured" and the actor dies
# *after* the whole cluster is up -- job 3678974 lost 700s of setup (662s of it
# NeMo-Gym) and a 61GiB refit to exactly that. So default off unless a key is
# present, matching tools/launch.sh:393. WANDB_MODE=offline needs no key and
# counts as usable. An explicit WANDB_ENABLED=True without either is honoured
# as a *request* but still downgraded, since there is nothing to authenticate.
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  export WANDB_API_KEY
  WANDB_ENABLED="${WANDB_ENABLED:-True}"
elif [[ "${WANDB_MODE:-}" =~ ^(offline|dryrun|disabled)$ ]]; then
  export WANDB_MODE
  WANDB_ENABLED="${WANDB_ENABLED:-True}"
else
  if [[ "${WANDB_ENABLED:-}" == "True" ]]; then
    echo "[WARN] WANDB_ENABLED=True but WANDB_API_KEY is unset; disabling W&B" >&2
    echo "       rather than crashing the run once the cluster is up." >&2
    echo "       export WANDB_API_KEY=<key>, or set WANDB_MODE=offline." >&2
  else
    echo "[WARN] WANDB_API_KEY is not set — W&B logging will be disabled." >&2
  fi
  WANDB_ENABLED=False
fi

# Run from the live checkout; there is no pin to snapshot against.
export USE_SNAPSHOT="${USE_SNAPSHOT:-0}"
export USE_CUSTOM_VLLM="${USE_CUSTOM_VLLM:-0}"
export RAY_LOG_SYNC_FREQUENCY="${RAY_LOG_SYNC_FREQUENCY:-60}"

# -----------------------------------------------------------------------------
# Mounts
# -----------------------------------------------------------------------------
CODE_ROOT="/opt/nemo-rl"
MOUNTS="${EXTERNAL_VLLM_SHARED_ROOT}:${EXTERNAL_VLLM_SHARED_ROOT}"
_append_mount() { MOUNTS="${MOUNTS:+${MOUNTS},}$1"; }

_append_mount "${NEMO_RL_ROOT}/nemo_rl:${CODE_ROOT}/nemo_rl"
_append_mount "${NEMO_RL_ROOT}/examples/configs:${CODE_ROOT}/examples/configs"
_append_mount "${CONFIG_MOUNT_DIR_HOST}:/opt/nano35-config"
# USE_IMAGE_GYM selects source code, not the venv location. Image venvs use an
# editable install at this path, so they can also run mounted local Gym code
# when its dependencies remain compatible with the image.
if [[ "${USE_IMAGE_GYM}" != "1" ]]; then
  _append_mount "${GYM_ROOT}:${CODE_ROOT}/3rdparty/Gym-workspace/Gym"
fi
# See the TRAIN_ENTRYPOINT note above: nemo_rl/ does not cover these files.
for _entrypoint in \
  examples/run_grpo_single_controller.py \
  examples/run_grpo.py \
  examples/nemo_gym/run_grpo_nemo_gym.py
do
  [[ -f "${NEMO_RL_ROOT}/${_entrypoint}" ]] && \
    _append_mount "${NEMO_RL_ROOT}/${_entrypoint}:${CODE_ROOT}/${_entrypoint}"
done
unset _entrypoint

# X11 libs for the ether0 Gym server, taken from the host because this image
# ships none (an unsquashfs -l of the nightly finds zero libX*).
#
# ether0.rewards -> exmol -> synspace -> rdkit.Chem.Draw, and rdkit's vendored
# libcairo links libX11/libXext/libXrender from the system. Its libxcb-* are
# vendored inside the wheel (hashed sonames); these are not. Missing them is
# fatal to the whole run, not just to ether0: one server that exits fails
# NemoGym._spinup for the batch (job 3669906, ImportError: libXrender.so.1).
#
# This is the transitive closure, not just the three cairo names -- libX11
# pulls libxcb, and libXdmcp pulls libbsd/libmd. libbsd and libmd are
# deliberately absent from this list: the image already provides them, and
# shadowing them with host copies risks breaking unrelated consumers.
#
# Mounted individually rather than as a directory so nothing else in the
# host's /usr/lib is exposed. Missing a lib is a hard error: silently skipping
# would just move the failure to Gym spinup, twenty minutes later.
X11_LIB_DIR="${X11_LIB_DIR:-/usr/lib/aarch64-linux-gnu}"
X11_LIBS=(
  libX11.so.6
  libXau.so.6
  libXdmcp.so.6
  libXext.so.6
  libXrender.so.1
  libxcb.so.1
)
for _lib in "${X11_LIBS[@]}"; do
  _src="$(readlink -f "${X11_LIB_DIR}/${_lib}" 2>/dev/null || true)"
  if [[ -z "${_src}" || ! -f "${_src}" ]]; then
    echo "ERROR: ${_lib} not found under ${X11_LIB_DIR} on $(hostname)." >&2
    echo "  The ether0 Gym server needs it; without it NemoGym spinup fails." >&2
    echo "  Install the X11 runtime libs, or drop" >&2
    echo "  resources_servers/ether0/configs/ether0.yaml from env.nemo_gym.config_paths." >&2
    exit 1
  fi
  # Target keeps the soname, not the versioned filename, so the loader finds it.
  _append_mount "${_src}:${X11_LIB_DIR}/${_lib}"
done
unset _lib _src

[[ -n "${EXTRA_MOUNTS:-}" ]] && _append_mount "${EXTRA_MOUNTS}"
export MOUNTS

export RAY_SUB="${RAY_SUB:-${NEMO_RL_ROOT}/ray.sub}"
BATCH_SCRIPT="${BATCH_SCRIPT:-${NEMO_RL_ROOT}/tools/external_gym_vllm/run_in_allocation.sh}"
for _f in RAY_SUB BATCH_SCRIPT; do
  if [[ ! -f "${!_f}" ]]; then
    echo "ERROR: ${_f} not found at ${!_f}" >&2; exit 1
  fi
done
unset _f

# -----------------------------------------------------------------------------
# Caches
# -----------------------------------------------------------------------------
# Lustre holds the warm cache; JIT writes go to node-local /tmp to keep parallel
# compilation off Lustre's metadata path. Local paths carry the job id so a
# cancelled job cannot poison the next one.
LOCAL_CACHE_TAG='${SLURM_JOB_ID:-manual}'
LUSTRE_VLLM_CACHE="${PERSISTENT_CACHE}/vllm_compile_cache_bf16"
LUSTRE_INDUCTOR_CACHE="${PERSISTENT_CACHE}/inductor_cache"
LUSTRE_TRITON_CACHE="${PERSISTENT_CACHE}/triton_cache"
LUSTRE_FLASHINFER_CUBIN_CACHE="${PERSISTENT_CACHE}/flashinfer_cubins"
# Gym venvs. USE_IMAGE_GYM (resolved above) picks the image's baked trees when
# they provably match our pin; otherwise we build our own and persist them.
#
# Explicit GYM_VENV_DIR=/opt/gym_venvs also reuses image venvs with local code:
# skip_venv_if_present checks their existence, not the source commit. The user
# must ensure dependency compatibility. Missing venvs still require setup.
# Building all venvs at spinup can mean 71 concurrent syncs, 30 sharing the same
# responses_api_agents/simple_agent venv, and Gym takes no lock around venv
# creation -- so a sibling's sync can uninstall nemo_gym between another
# server's install and its import. Job 3668422 died exactly that way, with one
# server of 71 raising ModuleNotFoundError: No module named 'nemo_gym'.
#
# Pointing the fallback at PERSISTENT_CACHE (the pattern used by
# examples/nemo_gym/nemotron-3-super/stage3_rlhf.yaml) makes the tree outlive
# the job, so the recipe's `skip_venv_if_present: true` short-circuits the sync
# on every later run. Prefetch it once with prefetch_gym_venvs.sh before the
# first real run; see the README.
#
# Not job-scoped like the /tmp caches above: sharing it across runs is the
# entire point, and uv_venv_dir trees do not relocate once built.
if [[ "${USE_IMAGE_GYM}" == "1" ]]; then
  GYM_VENV_DIR="${GYM_VENV_DIR:-/opt/gym_venvs}"
  GYM_VENV_SOURCE="image ${GYM_VENV_DIR} (Gym ${_IMAGE_GYM_COMMIT}, prebuilt)"
  GYM_SOURCE="image, Gym ${_IMAGE_GYM_COMMIT} (not mounted)"
  # Reachable only when USE_IMAGE_GYM=1 was forced by hand, since the automatic
  # path requires equality. Worth shouting about: the run then executes the
  # image's Gym, not the revision this checkout pins, and the recipe's verified
  # image/Gym pairing (see the ether0 rdkit note in the README) no longer holds.
  if [[ "${_IMAGE_GYM_COMMIT}" != "${GYM_COMMIT}" ]]; then
    echo "[WARN] USE_IMAGE_GYM=1 was forced, but the image's Gym" >&2
    echo "       (${_IMAGE_GYM_COMMIT}) is not the pinned ${GYM_COMMIT}." >&2
    echo "       The run will use the image's Gym and ignore ${GYM_ROOT}." >&2
    echo "       Unset USE_IMAGE_GYM to build venvs against the pin instead." >&2
  fi
else
  GYM_VENV_DIR="${GYM_VENV_DIR:-${PERSISTENT_CACHE}/gym_venvs}"
  GYM_VENV_SOURCE="checkout ${GYM_VENV_DIR}"
  GYM_SOURCE="${GYM_ROOT} (mounted)"
  # Say why the slow path was taken; otherwise a 20-minute venv build at spinup
  # looks like a hang rather than a decision.
  if [[ "${GYM_VENV_DIR}" == "/opt/gym_venvs" ]]; then
    GYM_VENV_SOURCE="image ${GYM_VENV_DIR} (prebuilt, local Gym source)"
    echo "[INFO] Reusing existing image Gym venvs with the mounted checkout; commit differences do not trigger setup." >&2
  elif [[ -n "${GYM_DIRTY}" ]]; then
    echo "[INFO] Gym checkout is dirty — building venvs into ${GYM_VENV_DIR}" >&2
    echo "       rather than reusing the image's prebuilt /opt/gym_venvs." >&2
  elif [[ "${_IMAGE_GYM_COMMIT}" == "unknown" ]]; then
    echo "[INFO] Could not read the image's Gym commit from ${CONTAINER};" >&2
    echo "       building venvs into ${GYM_VENV_DIR}. Prefetch them first." >&2
  elif [[ "${_IMAGE_GYM_COMMIT}" != "${GYM_COMMIT}" ]]; then
    echo "[INFO] Image Gym ${_IMAGE_GYM_COMMIT} != pinned ${GYM_COMMIT};" >&2
    echo "       building venvs into ${GYM_VENV_DIR}. Prefetch them first." >&2
  fi
fi
NRL_VLLM_LOCAL_CACHE_DIR="/tmp/nemo_rl_${LOCAL_CACHE_TAG}_vllm_cache"
NRL_VLLM_CACHE_SEED_DIR="/tmp/nemo_rl_${LOCAL_CACHE_TAG}_vllm_cache_warm"
INDUCTOR_CACHE_DIR="/tmp/nemo_rl_${LOCAL_CACHE_TAG}_inductor_cache"
TRITON_CACHE_DIR="/tmp/nemo_rl_${LOCAL_CACHE_TAG}_triton_cache"
CUDA_CACHE_PATH="/tmp/nemo_rl_${LOCAL_CACHE_TAG}_cuda_cache"
FLASHINFER_CUBIN_CACHE="/tmp/nemo_rl_${LOCAL_CACHE_TAG}_flashinfer_cubins"
FLASHINFER_WS_BASE="/tmp/nemo_rl_${LOCAL_CACHE_TAG}_flashinfer_workspace"
mkdir -p "${LUSTRE_VLLM_CACHE}" "${LUSTRE_INDUCTOR_CACHE}" \
         "${LUSTRE_TRITON_CACHE}" "${LUSTRE_FLASHINFER_CUBIN_CACHE}"
export LUSTRE_VLLM_CACHE LUSTRE_INDUCTOR_CACHE LUSTRE_TRITON_CACHE \
       NRL_VLLM_LOCAL_CACHE_DIR INDUCTOR_CACHE_DIR TRITON_CACHE_DIR \
       FLASHINFER_CUBIN_CACHE FLASHINFER_WS_BASE CUDA_CACHE_PATH
export CACHE_SYNC_FREQUENCY="${CACHE_SYNC_FREQUENCY:-120}"

read -r -d '' SETUP_COMMAND <<SETUPEOF || true
set -u
echo "[CACHE SEED] Clearing stale /tmp caches and seeding from Lustre..."
rm -rf "${NRL_VLLM_CACHE_SEED_DIR}" "${INDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}" \
       "${CUDA_CACHE_PATH}" "${FLASHINFER_CUBIN_CACHE}" "${FLASHINFER_WS_BASE}"
mkdir -p "${NRL_VLLM_CACHE_SEED_DIR}" "${INDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}" \
         "${CUDA_CACHE_PATH}" "${FLASHINFER_CUBIN_CACHE}" "${FLASHINFER_WS_BASE}"
_seed() {
  if [ -d "\$1" ] && [ -n "\$(ls -A "\$1" 2>/dev/null)" ]; then
    cp -a "\$1/." "\$2/" 2>/dev/null \
      && echo "[CACHE SEED] \$3: seeded" \
      || echo "[CACHE SEED] \$3: copy failed (non-fatal)"
  else
    echo "[CACHE SEED] \$3: no warm cache on Lustre yet"
  fi
}
if [ "${CACHE_FORCE_FRESH:-0}" = "1" ]; then
  echo "[CACHE SEED] CACHE_FORCE_FRESH=1; skipping warm cache extraction."
else
  _seed "${LUSTRE_VLLM_CACHE}" "${NRL_VLLM_CACHE_SEED_DIR}" "vLLM (bf16)"
  _seed "${LUSTRE_INDUCTOR_CACHE}" "${INDUCTOR_CACHE_DIR}" "Inductor"
  _seed "${LUSTRE_TRITON_CACHE}" "${TRITON_CACHE_DIR}" "Triton"
  _seed "${LUSTRE_FLASHINFER_CUBIN_CACHE}" "${FLASHINFER_CUBIN_CACHE}" "FlashInfer cubins"
fi
echo "[CACHE SEED] Done."
SETUPEOF
export SETUP_COMMAND

# -----------------------------------------------------------------------------
# Training command
# -----------------------------------------------------------------------------
# Hyperparameters live in CONFIG_PATH. Only per-run facts are passed here:
# cluster shape, paths, judge endpoints, logging.
#
# UV_HTTP_TIMEOUT is generous on purpose. Every job re-resolves whenever the
# checkout's uv.lock differs from the one the container was built against, so
# each start pulls a few hundred MiB of wheels per node; a ten-second per-request
# timeout is not enough for a 238 MiB vllm wheel and fails the job mid-extract
# with the whole allocation held. Raise it further on a slow or flaky mirror.
# 3rdparty/vllm/nemo-rl.env exists only in images built with
# BUILD_CUSTOM_VLLM=1, which generates both the directory and the file; it is
# not in the repo and not in a stock or nightly image. Sourcing it
# unconditionally puts `source <missing> && ` at the head of TRAIN_CMD, and the
# && then swallows the entire driver invocation -- the job takes its 86 nodes,
# brings up Ray, and never runs a step. So it is keyed off USE_CUSTOM_VLLM,
# the same way lightning35_launch.sh does it.
case "${USE_CUSTOM_VLLM}" in
  1) VLLM_ENV_SOURCE="source ${CODE_ROOT}/3rdparty/vllm/nemo-rl.env && " ;;
  0) VLLM_ENV_SOURCE="" ;;
  *) echo "ERROR: USE_CUSTOM_VLLM must be 0 or 1, got: ${USE_CUSTOM_VLLM}" >&2; exit 1 ;;
esac
# No mooncake port preflight here. NeMo-RL reserves all three of the master's
# listening ports -- metadata server, RPC, and metrics -- from a band below the
# ephemeral floor and binds each one to prove it is free, so there is no fixed
# port left to check; the 9003 default that killed job 3678431 twelve minutes in
# is no longer reachable. That fix is a mandated cherry-pick, and like the MTP
# one it is unchecked on purpose: skipping it fails loudly in TQ bootstrap rather
# than silently, so a launcher-side probe would only duplicate the diagnosis.
TRAIN_CMD="cd ${CODE_ROOT} && date ; \
${VLLM_ENV_SOURCE}\
OMP_NUM_THREADS=16 \
RAY_DEDUP_LOGS=1 \
WANDB_INIT_TIMEOUT=300 \
VLLM_CACHE_ROOT=${NRL_VLLM_LOCAL_CACHE_DIR} \
NRL_VLLM_CACHE_SEED_DIR=${NRL_VLLM_CACHE_SEED_DIR} \
DG_JIT_CACHE_DIR=${NRL_VLLM_LOCAL_CACHE_DIR}/deep_gemm \
TORCHINDUCTOR_CACHE_DIR=${INDUCTOR_CACHE_DIR} \
TRITON_CACHE_DIR=${TRITON_CACHE_DIR} \
CUDA_CACHE_PATH=${CUDA_CACHE_PATH} \
FLASHINFER_CUBIN_DIR=${FLASHINFER_CUBIN_CACHE} \
FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WS_BASE} \
UV_CACHE_DIR=${PERSISTENT_CACHE}/uv \
UV_LOCK_TIMEOUT=1800 \
NEMO_GYM_VENV_DIR=${GYM_VENV_DIR} \
RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 \
UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-300} \
VLLM_USE_FLASHINFER_MOE_FP8=1 \
VLLM_FLASHINFER_MOE_BACKEND=latency \
NRL_VLLM_ASYNC_TIMEOUT_SECONDS=1800 \
NRL_WG_USE_RAY_REF=1 \
HF_HOME=${HF_HOME} \
HF_TOKEN=\${HF_TOKEN:-} \
NRL_USE_FASTOKENS=${NRL_USE_FASTOKENS:-1} \
uv run ${TRAIN_ENTRYPOINT} \
--config ${CONFIG_PATH} \
policy.model_name=${MODEL_PATH} \
cluster.gpus_per_node=${GPUS_PER_NODE} \
cluster.num_nodes=${NUM_ACTOR_NODES} \
cluster.segment_size=${SEGMENT_SIZE} \
policy.generation.colocated.resources.num_nodes=${NUM_GEN_NODES} \
env.nemo_gym.num_gpu_nodes=${NUM_GYM_NODES} \
checkpointing.checkpoint_dir=${CHECKPOINT_DIR} \
data.train.data_path=${TRAIN_PATH} \
data.validation.data_path=${VAL_PATH} \
++env.nemo_gym.genrm_model.responses_api_models.genrm_model.base_url=${GENRM_BASE_URL} \
++env.nemo_gym.genrm_model.responses_api_models.genrm_model.model=${GENRM_SERVED_MODEL_NAME} \
++env.nemo_gym.nl2bash_judge_model.responses_api_models.local_vllm_model.base_url=${NL2BASH_BASE_URL} \
++env.nemo_gym.nl2bash_judge_model.responses_api_models.local_vllm_model.model=${NL2BASH_SERVED_MODEL_NAME} \
env.nemo_gym.safety_judge_model.responses_api_models.local_vllm_model.model=${SAFETY_JUDGE_MODEL} \
env.nemo_gym.nemo_gym_log_dir=${LOG_DIR}/nemo_gym \
${EXTERNAL_SERVICE_READINESS_PLACEHOLDER} \
logger.log_dir=${LOG_DIR} \
logger.wandb_enabled=${WANDB_ENABLED} \
logger.wandb.name=${WANDB_NAME} \
logger.wandb.project=${WANDB_PROJ} \
${NRL_MAX_STEPS:+grpo.max_num_steps=${NRL_MAX_STEPS}} \
${*}"

export COMMAND="${TRAIN_CMD}"
# Checks the placeholders, shared paths, tool files and node count before the
# allocation is requested.
validate_external_vllm_submission "${COMMAND}" "${NUM_EXTERNAL_SERVICE_NODES}"

# -----------------------------------------------------------------------------
# Summary and provenance
# -----------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  ${RUN_LABEL:-Nano 3.5 RLVR} — ${EXP_NAME}"
echo "================================================================"
echo "  Arm:          ${ARM_LABEL:-unspecified}"
echo "  Entrypoint:   ${TRAIN_ENTRYPOINT}"
echo "  Config:       ${CONFIG_PATH_HOST}"
echo "  NeMo RL:      ${NEMO_RL_COMMIT} (${NEMO_RL_BRANCH})"
echo "  Gym pinned:   ${GYM_COMMIT}"
echo "  Gym running:  ${GYM_SOURCE}"
echo "  Gym venvs:    ${GYM_VENV_SOURCE}"
echo "  Container:    ${CONTAINER}"
echo "  Site overlay: ${SITE_OVERLAY_USED}"
echo ""
echo "  Nodes:        ${NUM_TOTAL_NODES} total"
echo "    train:      ${NUM_TRAIN_NODES}"
echo "    generation: ${NUM_GEN_NODES}"
echo "    gym:        ${NUM_GYM_NODES}"
echo "    external:   ${NUM_EXTERNAL_SERVICE_NODES}  (pools: ${EXTERNAL_VLLM_POOLS})"
echo "  Walltime:     ${WALLTIME}"
echo ""
echo "  Checkpoints:  ${CHECKPOINT_DIR}"
echo "  Run dir:      ${RUN_DIR}"
echo "  W&B:          ${WANDB_ENTITY}/${WANDB_PROJ} / ${WANDB_NAME} (enabled=${WANDB_ENABLED})"
echo "================================================================"
echo ""

if [[ -n "$(git -C "${NEMO_RL_ROOT}" status --porcelain 2>/dev/null)" ]]; then
  echo "  NOTE: the NeMo RL checkout is dirty and USE_SNAPSHOT=0, so the run"
  echo "        reads it live. provenance.txt records the diff stat."
  echo ""
fi

{
  echo "timestamp: $(date -Iseconds)"
  echo "arm: ${ARM_LABEL:-unspecified}"
  echo "train_entrypoint: ${TRAIN_ENTRYPOINT}"
  echo "recipe_root: ${RECIPE_ROOT}"
  echo "nemo_rl_branch: ${NEMO_RL_BRANCH}"
  echo "nemo_rl_commit: ${NEMO_RL_COMMIT}"
  echo "nemo_rl_dirty:"
  git -C "${NEMO_RL_ROOT}" status --porcelain 2>/dev/null | head -40 || true
  echo "gym_commit: ${GYM_COMMIT}"
  # Which tree actually ran, since USE_IMAGE_GYM makes gym_commit ambiguous on
  # its own: under the fast path Gym comes from the image, not from GYM_ROOT.
  echo "gym_source: ${GYM_SOURCE}"
  echo "gym_image_commit: ${_IMAGE_GYM_COMMIT}"
  echo "gym_venvs: ${GYM_VENV_SOURCE}"
  echo "use_image_gym: ${USE_IMAGE_GYM}"
  echo "container: ${CONTAINER}"
  echo "site_overlay: ${SITE_OVERLAY_USED}"
  echo "config_host: ${CONFIG_PATH_HOST}"
  echo "config_container: ${CONFIG_PATH}"
  echo "mounts: ${MOUNTS}"
  echo "model: ${MODEL_PATH}"
  echo "train_data: ${TRAIN_PATH}"
  echo "genrm_model: ${GENRM_MODEL}"
  echo "genrm_reasoning_parser: ${GENRM_REASONING_PARSER}"
  echo "nl2bash_model: ${NL2BASH_JUDGE_MODEL}"
  echo "safety_model: ${SAFETY_JUDGE_MODEL}"
  echo "external_pools: ${EXTERNAL_VLLM_POOLS}"
  echo "nodes_total: ${NUM_TOTAL_NODES}"
  echo "command: ${TRAIN_CMD}"
} > "${RUN_DIR}/provenance.txt"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1 — resolved command below; nothing submitted."
  echo ""
  echo "--- TRAIN_CMD ---"
  echo "${TRAIN_CMD}"
  echo "--- end ---"
  echo ""
  echo "Provenance: ${RUN_DIR}/provenance.txt"
  exit 0
fi

# -----------------------------------------------------------------------------
# Submit
# -----------------------------------------------------------------------------
# Two components: hetgroup 0 is the NeMo RL Ray cluster and the per-pool load
# balancers, hetgroup 1 is the external replicas. run_in_allocation.sh starts
# ray.sub on hetgroup 0 only, after every pool reports healthy and its URL has
# replaced the placeholder in COMMAND.
#
# singleton serialises same-name submissions so a resubmission after preemption
# resumes from the latest checkpoint instead of racing the job it replaced.
SLURM_DEPENDENCY="${SLURM_DEPENDENCY:-}"
DEPENDENCY="singleton"
[[ -n "${SLURM_DEPENDENCY}" ]] && DEPENDENCY="singleton,${SLURM_DEPENDENCY}"

SBATCH_OUTPUT=$(sbatch \
  --nodes="${NUM_RAY_NODES}" \
  --account="${SLURM_ACCOUNT}" \
  --job-name="${JOB_NAME}" \
  --partition="${SLURM_PARTITION}" \
  --time="${WALLTIME}" \
  --gres=gpu:"${GPUS_PER_NODE}" \
  --exclusive \
  --mem=0 \
  --dependency="${DEPENDENCY}" \
  --segment="${SEGMENT_SIZE}" \
  --output="${SLURM_LOG_DIR}/%j.out" \
  --error="${SLURM_LOG_DIR}/%j.err" \
  ${SLURM_QOS:+--qos="${SLURM_QOS}"} \
  ${EXCLUDE_NODES:+--exclude="${EXCLUDE_NODES}"} \
  ${SLURM_RESERVATION:+--reservation="${SLURM_RESERVATION}"} \
  "${SLURM_COMMENT_ARGS[@]}" \
  : \
  --nodes="${NUM_EXTERNAL_SERVICE_NODES}" \
  --account="${SLURM_ACCOUNT}" \
  --job-name="${JOB_NAME}-services" \
  --partition="${SLURM_PARTITION}" \
  --time="${WALLTIME}" \
  --gres=gpu:"${GPUS_PER_NODE}" \
  --exclusive \
  --mem=0 \
  --segment="${EXTERNAL_VLLM_SEGMENT_SIZE}" \
  ${SLURM_QOS:+--qos="${SLURM_QOS}"} \
  ${EXCLUDE_NODES:+--exclude="${EXCLUDE_NODES}"} \
  ${SLURM_RESERVATION:+--reservation="${SLURM_RESERVATION}"} \
  "${BATCH_SCRIPT}")

echo "${SBATCH_OUTPUT}"
JOB_ID="$(echo "${SBATCH_OUTPUT}" | grep -oE '[0-9]+$')" || true
if [[ -n "${JOB_ID}" ]]; then
  echo ""
  echo "  Ray logs:   ${BASE_LOG_DIR}/${JOB_ID}-logs/"
  echo "  Slurm logs: ${SLURM_LOG_DIR}/${JOB_ID}.out"
  echo ""
fi

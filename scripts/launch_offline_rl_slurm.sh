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
set -euo pipefail

# =============================================================================
# launch_offline_rl_slurm.sh
#
# Submits a multi-node OAPL or EIWOR training job to Slurm via `sbatch ray.sub`,
# following the pattern in README.md ("OAPL Multi-node" / "EIWOR Multi-node")
# plus a few conventions borrowed from nemo-rl-internal/scripts/launch_nano_e2e_opencode.sh
# (code snapshotting, HF cache setup, --exclusive/--dependency=singleton,
# interactive attach mode, DRY_RUN secret redaction). Unlike that script, this
# one has no generation/rollout/vLLM machinery — OAPL/EIWOR train on a fixed
# offline dataset, so there's only one node group.
#
# Run this from the ROOT of the Offline-RL-Internal repo, on the cluster login
# node (or non-interactively over ssh — see scripts/remote_launch_offline_rl.sh
# for a Mac-side wrapper that does that for you).
#
# Required env vars:
#   ALGO              "oapl", "eiwor", "sft", or "grpo" (also covers CISPO --
#                      CISPO is config-driven via loss_fn.use_cispo, not a
#                      separate script; use ALGO=grpo either way)
#   TRAIN_DATA_PATH   Path to the training JSONL, as visible INSIDE the
#                      container (i.e. under one of the MOUNTS paths).
#   CONTAINER         Docker image / squashfs for the Ray cluster.
#   SLURM_ACCOUNT     Slurm account.
#   PARTITION         Slurm partition.
#
# Common optional env vars (defaults shown):
#   VAL_DATA_PATH=              Validation JSONL path (skips validation config if unset)
#   NUM_NODES=2                 cluster.num_nodes (== --nodes for sbatch; head is
#                               colocated on worker 0, per README)
#   GPUS_PER_NODE=8              cluster.gpus_per_node (use 4 for GB200)
#   MODEL_NAME=                  override policy.model_name (defaults to the value
#                               baked into the config file)
#   BETA=                        (EIWOR only) override eiwor.beta
#   OAPL_BETA1=                  (OAPL only) override oapl.beta1 and the
#                               dataset's value-estimation temperature
#   OAPL_BETA2=                  (OAPL only) override oapl.beta2
#   NUM_COMPLETIONS_PER_PROMPT=  (EIWOR only) override eiwor.num_completions_per_prompt
#   BIRD_SQL_DB_DIR=              (GRPO/BIRD only) shared-storage path to the BIRD
#                               SQLite databases, as visible INSIDE the container.
#                               Must be laid out as
#                               <dir>/dev_20240627/dev_databases/<db_id>/<db_id>.sqlite  (BIRD_SQL_SPLIT=dev)
#                               <dir>/train/train_databases/<db_id>/<db_id>.sqlite       (BIRD_SQL_SPLIT=train)
#                               (matches resources_servers/bird_sql/setup_bird_sql.py's
#                               idempotency checks, so it's used as-is instead of
#                               triggering an auto-download). It is mounted read-only for
#                               training; SEED_GYM_VENVS=1 creates it (read-write) and downloads the
#                               BIRD split into it if missing (~9 GB for train).
#                               Overrides env.nemo_gym.bird_sql_resources_server
#                               .resources_servers.bird_sql.bird_sql_dir and is
#                               auto-mounted read-only.
#   BIRD_SQL_SPLIT=               (GRPO/BIRD only) "dev" or "train": which BIRD split's
#                               databases the bird_sql reward server uses. Overrides
#                               ...bird_sql.split. The server has ONE split, so it
#                               must match the db_ids in TRAIN_DATA_PATH/VAL_DATA_PATH.
#   GYM_VENV_DIR=                 (GRPO only) shared-storage dir for the NeMo-Gym server venvs. With it set,
#                               training reuses ready venvs instead of installing packages at startup
#                               (+env.nemo_gym.uv_venv_dir, skip_venv_if_present=true). Seed it once with
#                               SEED_GYM_VENVS=1. A venv records its own path, so build it where it is used
#                               and never copy it; rebuild it if the container image changes.
#   GYM_UV_CACHE_DIR=             (GRPO only) shared-storage uv cache (+env.nemo_gym.uv_cache_dir, and
#                               UV_CACHE_DIR for SETUP_COMMAND).
#   UV_RUN_NO_SYNC=0              set to 1 to run the training command as `uv run --frozen --no-sync`, i.e. use the container's baked
#                               environment as is instead of syncing it to the extra first (the sync can stall on a compute node:
#                               it fetches git dependencies). Only valid when the image was built from this checkout's uv.lock.
#   UV_PYTHON_INSTALL_DIR=        (GRPO only) shared-storage dir for a uv-managed Python that satisfies the repo's
#                               requires-python, for containers whose uv is too old to download it ("No interpreter
#                               found for Python 3.13.14"). Created if missing, mounted read-write and exported for
#                               the seed job and the training command. SEED_GYM_VENVS=1 fills it: it installs uv
#                               ${UV_VERSION} (main's Dockerfile pin) to a node-local temp dir and runs
#                               `uv python install ${UV_PYTHON_VERSION}` into it (skipped if already there).
#   SEED_MAX_JOBS=8               parallel compile jobs per CUDA extension build in the seed (MAX_JOBS); lower if the seed is
#                               OOM-killed. SEED_CONCURRENT_BUILDS=2: how many extensions uv builds at once.
#   SEED_SKIP_ENV_SYNC=0          set to 1 when the container already holds this checkout's environment (image built from the same
#                               uv.lock): the seed then skips `uv sync` and the model download and only builds the Gym venvs.
#   SEED_BUILD_ENV=               extra VAR=value words for the seed's source builds, e.g. FLASH_ATTN_CUDA_ARCHS=100 to compile
#                               flash-attn for one GPU architecture instead of its default 80;90;100;120 (about 4x less work).
#                               Only use an architecture that matches the training GPUs (GB200 is sm_100).
#   UV_VERSION=0.11.28            uv used only to download that Python.
#   UV_PYTHON_VERSION=3.13.14     Python to install; keep >= requires-python in pyproject.toml.
#   SEED_GYM_VENVS=0              set to 1 to submit a short 1-node job that builds the venvs into
#                               GYM_VENV_DIR, downloads the BIRD databases into BIRD_SQL_DB_DIR and the
#                               model (MODEL_NAME) into HF_HOME, and exits (no training). Requests no GPUs unless SEED_GPUS=N;
#                               SEED_CPUS=4, SEED_MEM=16G (small requests start sooner), SEED_WALLTIME=1:00:00.
#   BIRD_SQL_DIALECT=             (GRPO/BIRD only) "sqlite" or "spark": the SQL dialect
#                               the reward server executes. Overrides ...bird_sql.dialect.
#                               Must match the prompts in the data ("Database Engine").
#   CONFIG=                      path to config yaml (defaults to
#                               examples/configs/${ALGO}.yaml)
#   CHECKPOINT_DIR=results/${ALGO}_${NUM_NODES}n   checkpointing.checkpoint_dir
#   JOB_NAME=${ALGO}-multinode   Slurm job name (also used as the W&B run name default)
#   WALLTIME=4:0:0                Slurm walltime
#   SLURM_QOS=                    optional --qos
#   SLURM_COMMENT=                optional --comment (site-specific scheduler JSON, if required)
#   SEGMENT_SIZE=                 optional --segment (topology-aware placement; see docs/cluster.md)
#   SINGLETON=1                   pass --dependency=singleton (blocks a second concurrent
#                               submission with the same JOB_NAME+user; matches nemo-rl-internal
#                               convention). Set 0 to disable.
#   MOUNTS=$PWD:$PWD               extra paths can be appended, comma-separated,
#                               e.g. MOUNTS="$PWD:$PWD,/lustre/data:/lustre/data:ro"
#   HF_HOME, HF_TOKEN, HF_HUB_CACHE, HF_DATASETS_CACHE
#                                forwarded to ray.sub if set (needed for gated HF models)
#   USE_SNAPSHOT=1 (batch) / 0 (interactive or DRY_RUN)
#                                copy git-tracked files into code_snapshots/<job_name>
#                                via tools/code_snapshot.sh, so later local edits don't
#                                affect a queued/running job. Set 0 to run the live tree.
#                                Set to "config" for a lightweight alternative that only
#                                freezes CONFIG (not the whole tracked tree/submodules) --
#                                protects against SYNC_CONFIGS changing a queued job's
#                                config, but NOT against source code changing under it.
#   CONTAINER_CODE_ROOT=         optional absolute container path for a code
#                               bind mount, e.g. /opt/nemo-rl
#   CONTAINER_CODE_MOUNT_MODE=ro code bind mode: ro or rw. Use rw only with a
#                               run-local code copy.
#   UV_PROJECT_ENVIRONMENT=      writable uv environment path; required when
#                               USE_LOCKED_DEPS_ENV=1
#   UV_SYNC_CACHE_DIR=           optional cache used only by uv sync/run inside
#                               the training command and by the seed job's `uv sync`. Put it on shared
#                               storage so CUDA extensions compiled from source (flash-attn, mamba-ssm,
#                               deep-ep, ...) survive a failed or timed-out job; unlike
#                               UV_CACHE_DIR_OVERRIDE, this does not cover the
#                               container's own /root/.cache/uv
#   USE_LOCKED_DEPS_ENV=0        set to 1 to sync the selected locked extra into
#                               UV_PROJECT_ENVIRONMENT without installing this
#                               project or separate Bridge/MCore packages, then
#                               run the mounted checkout directly from PYTHONPATH
#   REQUIRE_RUN_LOCAL_UV_PROJECT_ENVIRONMENT=0
#                               set to 1 to require the writable uv environment
#                               to be exactly <dirname(CHECKPOINT_DIR)>/
#                               uv-project-environment. Production offline-RL
#                               submitters enable this to prevent cross-run
#                               environment sharing.
#   PRECOMPILE_MCORE_DATASET_HELPER=0
#                               set to 1 for a writable Megatron code copy. The
#                               driver builds MCore's helper once under a lock
#                               before distributed workers initialize and makes
#                               that Python's versioned config utility available
#                               under the name expected by MCore's Makefile.
#
# W&B (all optional; if WANDB_API_KEY is set, logging is enabled by default):
#   WANDB_PROJECT=${ALGO}-dev
#   WANDB_NAME=${JOB_NAME}
#   WANDB_API_KEY=              forwarded to ray.sub; if unset, wandb logging is
#                               disabled and training logs locally only.
#
#   EXTRA_ARGS="..."            extra raw Hydra overrides, appended verbatim to COMMAND
#   DRY_RUN=0                   set to 1 to print the sbatch command without submitting
#   INTERACTIVE=0                set to 1 for a persistent, idle Ray allocation you attach
#                               to and run/re-run commands in without requeueing (useful
#                               for a first run while you're still validating your dataset).
#   INTERACTIVE_WAIT=1           when INTERACTIVE=1, block in the foreground until Ray is ready
#
# Usage:
#   ALGO=oapl \
#   TRAIN_DATA_PATH=/data/oapl_train.jsonl \
#   VAL_DATA_PATH=/data/oapl_val.jsonl \
#   CONTAINER=nvcr.io/nvidia/nemo-rl:latest \
#   SLURM_ACCOUNT=my_account \
#   PARTITION=batch \
#   NUM_NODES=4 \
#   WANDB_API_KEY=xxxx \
#   ./scripts/launch_offline_rl_slurm.sh
#
#   # Dry run first (recommended): just prints the sbatch/COMMAND, no submit
#   DRY_RUN=1 ALGO=oapl TRAIN_DATA_PATH=... CONTAINER=... SLURM_ACCOUNT=... PARTITION=... \
#   ./scripts/launch_offline_rl_slurm.sh
# =============================================================================

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd -P)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd -P)
cd "${PROJECT_ROOT}"

DRY_RUN="${DRY_RUN:-0}"
INTERACTIVE="${INTERACTIVE:-0}"
INTERACTIVE_WAIT="${INTERACTIVE_WAIT:-1}"

# ---------- Required args ----------
ALGO="${ALGO:?Set ALGO=oapl, ALGO=eiwor, ALGO=sft, or ALGO=grpo}"
if [[ "${ALGO}" != "oapl" && "${ALGO}" != "eiwor" && "${ALGO}" != "sft" && "${ALGO}" != "grpo" ]]; then
  echo "ERROR: ALGO must be 'oapl', 'eiwor', 'sft', or 'grpo' (got '${ALGO}')" >&2
  exit 1
fi
if [[ "${ALGO}" == "oapl" && -n "${BETA+x}" ]]; then
  echo "ERROR: BETA is EIWOR-only; use OAPL_BETA1 and OAPL_BETA2 for OAPL." >&2
  exit 1
fi
if [[ "${ALGO}" == "eiwor" && ( -n "${OAPL_BETA1+x}" || -n "${OAPL_BETA2+x}" ) ]]; then
  echo "ERROR: OAPL_BETA1 and OAPL_BETA2 are OAPL-only; use BETA for EIWOR." >&2
  exit 1
fi

TRAIN_DATA_PATH="${TRAIN_DATA_PATH:?Set TRAIN_DATA_PATH to your training JSONL path (as visible inside the container)}"
CONTAINER="${CONTAINER:?Set CONTAINER to a docker image or squashfs path}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:?Set SLURM_ACCOUNT to your Slurm account}"
PARTITION="${PARTITION:?Set PARTITION to your Slurm partition}"

# ---------- Defaults ----------
VAL_DATA_PATH="${VAL_DATA_PATH:-}"
NUM_NODES="${NUM_NODES:-2}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
CONFIG="${CONFIG:-examples/configs/${ALGO}.yaml}"
MODEL_NAME="${MODEL_NAME:-}"
BETA="${BETA:-}"
OAPL_BETA1="${OAPL_BETA1:-}"
OAPL_BETA2="${OAPL_BETA2:-}"
NUM_COMPLETIONS_PER_PROMPT="${NUM_COMPLETIONS_PER_PROMPT:-}"
BIRD_SQL_DB_DIR="${BIRD_SQL_DB_DIR:-}"
BIRD_SQL_SPLIT="${BIRD_SQL_SPLIT:-}"
BIRD_SQL_DIALECT="${BIRD_SQL_DIALECT:-}"
JOB_NAME="${JOB_NAME:-${ALGO}-multinode}"
WALLTIME="${WALLTIME:-4:0:0}"
SLURM_QOS="${SLURM_QOS:-}"
SLURM_COMMENT="${SLURM_COMMENT:-}"
SEGMENT_SIZE="${SEGMENT_SIZE:-}"
SINGLETON="${SINGLETON:-1}"
MOUNTS="${MOUNTS:-$PWD:$PWD}"
BASE_LOG_DIR="${BASE_LOG_DIR:-}"  # ray.sub default: $SLURM_SUBMIT_DIR (this repo's root). Set to change where <jobid>-logs/ lands.
EXTRA_ARGS="${EXTRA_ARGS:-}"
CONTAINER_CODE_ROOT="${CONTAINER_CODE_ROOT:-}"
CONTAINER_CODE_MOUNT_MODE="${CONTAINER_CODE_MOUNT_MODE:-ro}"
UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-}"
UV_SYNC_CACHE_DIR="${UV_SYNC_CACHE_DIR:-}"
USE_LOCKED_DEPS_ENV="${USE_LOCKED_DEPS_ENV:-0}"
REQUIRE_RUN_LOCAL_UV_PROJECT_ENVIRONMENT="${REQUIRE_RUN_LOCAL_UV_PROJECT_ENVIRONMENT:-0}"
PRECOMPILE_MCORE_DATASET_HELPER="${PRECOMPILE_MCORE_DATASET_HELPER:-0}"
GYM_VENV_DIR="${GYM_VENV_DIR:-}"
GYM_UV_CACHE_DIR="${GYM_UV_CACHE_DIR:-}"
UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-}"
UV_RUN_NO_SYNC="${UV_RUN_NO_SYNC:-0}"
UV_VERSION="${UV_VERSION:-0.11.28}"
UV_PYTHON_VERSION="${UV_PYTHON_VERSION:-3.13.14}"
SEED_GYM_VENVS="${SEED_GYM_VENVS:-0}"
SEED_GPUS="${SEED_GPUS:-0}"
SEED_CPUS="${SEED_CPUS:-4}"
SEED_MEM="${SEED_MEM:-16G}"
SEED_PARTITION="${SEED_PARTITION:-}"  # defaults to PARTITION; set e.g. cpu so the GPU-free seed does not use a GPU partition
SEED_BUILD_ENV="${SEED_BUILD_ENV:-}"
SEED_SKIP_ENV_SYNC="${SEED_SKIP_ENV_SYNC:-0}"
SEED_MAX_JOBS="${SEED_MAX_JOBS:-8}"
SEED_CONCURRENT_BUILDS="${SEED_CONCURRENT_BUILDS:-2}"
SEED_WALLTIME="${SEED_WALLTIME:-1:00:00}"
if [[ "${SEED_GYM_VENVS}" != "0" && "${SEED_GYM_VENVS}" != "1" ]]; then
  echo "ERROR: SEED_GYM_VENVS must be 0 or 1, got ${SEED_GYM_VENVS}" >&2
  exit 1
fi
if [[ "${SEED_GYM_VENVS}" == "1" && ( "${ALGO}" != "grpo" || -z "${GYM_VENV_DIR}" || -z "${BIRD_SQL_DB_DIR}" ) ]]; then
  echo "ERROR: SEED_GYM_VENVS=1 needs ALGO=grpo plus GYM_VENV_DIR and BIRD_SQL_DB_DIR." >&2
  exit 1
fi
if [[ "${ALGO}" != "grpo" && ( -n "${GYM_VENV_DIR}" || -n "${GYM_UV_CACHE_DIR}" ) ]]; then
  echo "ERROR: GYM_VENV_DIR/GYM_UV_CACHE_DIR are GRPO-only; unset them or set ALGO=grpo." >&2
  exit 1
fi
if [[ "${CONTAINER_CODE_MOUNT_MODE}" != "ro" && "${CONTAINER_CODE_MOUNT_MODE}" != "rw" ]]; then
  echo "ERROR: CONTAINER_CODE_MOUNT_MODE must be ro or rw" >&2
  exit 1
fi
if [[ "${USE_LOCKED_DEPS_ENV}" != "0" && "${USE_LOCKED_DEPS_ENV}" != "1" ]]; then
  echo "ERROR: USE_LOCKED_DEPS_ENV must be 0 or 1, got ${USE_LOCKED_DEPS_ENV}" >&2
  exit 1
fi
if [[ "${USE_LOCKED_DEPS_ENV}" == "1" && -z "${UV_PROJECT_ENVIRONMENT}" ]]; then
  echo "ERROR: USE_LOCKED_DEPS_ENV=1 requires UV_PROJECT_ENVIRONMENT" >&2
  exit 1
fi
if [[ "${REQUIRE_RUN_LOCAL_UV_PROJECT_ENVIRONMENT}" != "0" && "${REQUIRE_RUN_LOCAL_UV_PROJECT_ENVIRONMENT}" != "1" ]]; then
  echo "ERROR: REQUIRE_RUN_LOCAL_UV_PROJECT_ENVIRONMENT must be 0 or 1" >&2
  exit 1
fi
if [[ "${PRECOMPILE_MCORE_DATASET_HELPER}" != "0" && "${PRECOMPILE_MCORE_DATASET_HELPER}" != "1" ]]; then
  echo "ERROR: PRECOMPILE_MCORE_DATASET_HELPER must be 0 or 1" >&2
  exit 1
fi
if [[ "${PRECOMPILE_MCORE_DATASET_HELPER}" == "1" && "${CONTAINER_CODE_MOUNT_MODE}" != "rw" ]]; then
  echo "ERROR: MCore helper precompilation requires CONTAINER_CODE_MOUNT_MODE=rw" >&2
  exit 1
fi

if [[ ! -f "${CONFIG}" ]]; then
  echo "ERROR: config file not found: ${CONFIG}" >&2
  exit 1
fi
# Omega configs need this checkout to have the Omega code (omega-algo branch). Without it the config schema
# silently ignores grpo.loss_function and the job runs plain GRPO instead.
if grep -qE '^[[:space:]]*loss_function:[[:space:]]*omega' "${CONFIG}" \
    && [[ ! -f nemo_rl/algorithms/loss/omega_loss.py ]]; then
  echo "ERROR: ${CONFIG} sets grpo.loss_function: omega, but $(pwd) has no nemo_rl/algorithms/loss/omega_loss.py." >&2
  echo "       Check out the omega-algo branch there first, or the run would silently be GRPO." >&2
  exit 1
fi

# Default checkpoints into BASE_LOG_DIR (same directory as ray.sub's
# <jobid>-logs/ and the slurm-%j.out file below) when set, so everything for
# a run lives in one place; otherwise fall back to the old results/ default.
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${BASE_LOG_DIR:+${BASE_LOG_DIR}/checkpoints}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-results/${ALGO}_${NUM_NODES}n}"
if [[ "${REQUIRE_RUN_LOCAL_UV_PROJECT_ENVIRONMENT}" == "1" ]]; then
  if [[ "${USE_LOCKED_DEPS_ENV}" != "1" ]]; then
    echo "ERROR: run-local uv environment enforcement requires USE_LOCKED_DEPS_ENV=1" >&2
    exit 1
  fi
  EXPECTED_RUN_LOCAL_UV_PROJECT_ENVIRONMENT="$(dirname -- "${CHECKPOINT_DIR%/}")/uv-project-environment"
  if [[ "${UV_PROJECT_ENVIRONMENT}" != "${EXPECTED_RUN_LOCAL_UV_PROJECT_ENVIRONMENT}" ]]; then
    echo "ERROR: UV_PROJECT_ENVIRONMENT must be run-local: expected ${EXPECTED_RUN_LOCAL_UV_PROJECT_ENVIRONMENT}, got ${UV_PROJECT_ENVIRONMENT}" >&2
    exit 1
  fi
fi
if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${CHECKPOINT_DIR}"
  CHECKPOINT_DIR="$(cd "${CHECKPOINT_DIR}" && pwd -P)"
fi

# Resolve a file path to its physical (symlink-free) form, so a path that
# reaches this script via a symlinked prefix (e.g. REMOTE_REPO_DIR built on
# /lustre/... that's actually a symlink into /scratch/...) matches whatever
# gets bind-mounted into the container, rather than pointing at a location
# that was never mounted.
resolve_physical_path() {
  local p="$1"
  local dir base
  dir=$(cd "$(dirname "${p}")" &>/dev/null && pwd -P) || { echo "${p}"; return; }
  base=$(basename "${p}")
  echo "${dir}/${base}"
}
[[ -n "${TRAIN_DATA_PATH}" ]] && TRAIN_DATA_PATH=$(resolve_physical_path "${TRAIN_DATA_PATH}")
[[ -n "${VAL_DATA_PATH}" ]] && VAL_DATA_PATH=$(resolve_physical_path "${VAL_DATA_PATH}")

# BIRD_SQL_DB_DIR is a directory (not a file), like BASE_LOG_DIR -- resolve via
# `cd && pwd -P` rather than resolve_physical_path's file-oriented dirname/basename
# split. Unlike BASE_LOG_DIR, never mkdir it: it must already contain the
# pre-populated BIRD SQLite databases, so a missing path is a misconfiguration
# to surface, not something to silently create empty.
if [[ -n "${BIRD_SQL_DB_DIR}" ]]; then
  if [[ "${DRY_RUN}" != "1" ]]; then
    if [[ "${SEED_GYM_VENVS}" == "1" ]]; then
      mkdir -p "${BIRD_SQL_DB_DIR}"  # the seed job downloads the databases into it
    elif [[ ! -d "${BIRD_SQL_DB_DIR}" ]]; then
      echo "ERROR: BIRD_SQL_DB_DIR does not exist: ${BIRD_SQL_DB_DIR} (run once with SEED_GYM_VENVS=1 to download it)" >&2
      exit 1
    fi
    BIRD_SQL_DB_DIR="$(cd "${BIRD_SQL_DB_DIR}" && pwd -P)"
  else
    BIRD_SQL_DB_DIR=$(resolve_physical_path "${BIRD_SQL_DB_DIR}")
  fi
fi

# BASE_LOG_DIR must also be physical: ray.sub writes a shared-filesystem
# canary under it and checks visibility from the head node, which fails if
# the string still uses a symlinked prefix that was never bind-mounted.
if [[ -n "${BASE_LOG_DIR}" ]]; then
  if [[ "${DRY_RUN}" != "1" ]]; then
    mkdir -p "${BASE_LOG_DIR}"
    BASE_LOG_DIR="$(cd "${BASE_LOG_DIR}" && pwd -P)"
  else
    BASE_LOG_DIR=$(resolve_physical_path "${BASE_LOG_DIR}")
  fi
fi

# Auto-mount the dataset's parent directory (read-only) if it isn't already
# under one of the mounted paths, so the JSONL is reachable inside the container.
add_data_mount() {
  local data_path="$1"
  [[ -z "${data_path}" ]] && return
  local data_dir
  data_dir=$(dirname "${data_path}")
  if [[ "${MOUNTS}" != *"${data_dir}"* ]]; then
    MOUNTS="${MOUNTS},${data_dir}:${data_dir}:ro"
  fi
}
add_data_mount "${TRAIN_DATA_PATH}"
add_data_mount "${VAL_DATA_PATH}"

# BASE_LOG_DIR is a directory itself (not a file), so mount it directly
# rather than via add_data_mount (which mounts a file's *parent*).
if [[ -n "${BASE_LOG_DIR}" && "${MOUNTS}" != *"${BASE_LOG_DIR}"* ]]; then
  MOUNTS="${MOUNTS},${BASE_LOG_DIR}:${BASE_LOG_DIR}"
fi
if [[ -n "${BIRD_SQL_DB_DIR}" && "${MOUNTS}" != *"${BIRD_SQL_DB_DIR}"* ]]; then
  # read-only for training; the seed job writes the downloaded databases into it
  MOUNTS="${MOUNTS},${BIRD_SQL_DB_DIR}:${BIRD_SQL_DB_DIR}$([[ "${SEED_GYM_VENVS}" == "1" ]] || echo ":ro")"
fi

# Persistent NeMo-Gym venvs + uv cache on shared storage (see GYM_VENV_DIR in the header).
# Created if missing, resolved to physical paths (like BASE_LOG_DIR) and mounted read-write.
resolve_physical_dir() {
  local p="$1"
  if [[ "${DRY_RUN}" != "1" ]]; then
    mkdir -p "${p}"
    (cd "${p}" && pwd -P)
  else
    resolve_physical_path "${p}"
  fi
}
[[ -n "${GYM_VENV_DIR}" ]] && GYM_VENV_DIR=$(resolve_physical_dir "${GYM_VENV_DIR}")
[[ -n "${GYM_UV_CACHE_DIR}" ]] && GYM_UV_CACHE_DIR=$(resolve_physical_dir "${GYM_UV_CACHE_DIR}")
[[ -n "${UV_SYNC_CACHE_DIR}" ]] && UV_SYNC_CACHE_DIR=$(resolve_physical_dir "${UV_SYNC_CACHE_DIR}")
[[ -n "${UV_PYTHON_INSTALL_DIR}" ]] && UV_PYTHON_INSTALL_DIR=$(resolve_physical_dir "${UV_PYTHON_INSTALL_DIR}")
for _gym_dir in "${GYM_VENV_DIR}" "${GYM_UV_CACHE_DIR}" "${UV_PYTHON_INSTALL_DIR}" "${UV_SYNC_CACHE_DIR}"; do
  if [[ -n "${_gym_dir}" && "${MOUNTS}" != *"${_gym_dir}"* ]]; then
    MOUNTS="${MOUNTS},${_gym_dir}:${_gym_dir}"
  fi
done
# ray.sub runs SETUP_COMMAND on every node before Ray starts; point its uv at the shared cache
# so downloads are done once, not per node and per job. (Physical path, so it exists in the container.)
if [[ -n "${SETUP_COMMAND:-}" && -n "${GYM_UV_CACHE_DIR}" ]]; then
  export SETUP_COMMAND="export UV_CACHE_DIR='${GYM_UV_CACHE_DIR}'; ${SETUP_COMMAND}"
fi

# Point Slurm's own stdout/stderr file (normally slurm-%j.out in the
# submission dir) at BASE_LOG_DIR too, so it lands alongside ray.sub's
# <jobid>-logs/ and the checkpoint dir instead of cluttering the repo root.
SBATCH_OUTPUT_ARGS=()
if [[ -n "${BASE_LOG_DIR}" ]]; then
  SBATCH_OUTPUT_ARGS=(--output="${BASE_LOG_DIR}/slurm-%j.out")
fi

# ---------- HuggingFace cache/token (forwarded to ray.sub if set) ----------
export HF_TOKEN="${HF_TOKEN:-${HF_API_TOKEN:-}}"
export HF_HOME="${HF_HOME:-}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-}"

# HF_HOME must be physical and on the shared filesystem too: the Megatron
# backend converts HF checkpoints to Megatron format on rank 0 and writes the
# result under $HF_HOME; other ranks/nodes then read it back. If HF_HOME
# isn't shared (or is a symlinked path never bind-mounted), those reads fail
# with "Pretrained run config not found... not mounted on this node."
if [[ -n "${HF_HOME}" ]]; then
  if [[ "${DRY_RUN}" != "1" ]]; then
    mkdir -p "${HF_HOME}"
    HF_HOME="$(cd "${HF_HOME}" && pwd -P)"
  else
    HF_HOME=$(resolve_physical_path "${HF_HOME}")
  fi
  export HF_HOME
  if [[ "${MOUNTS}" != *"${HF_HOME}"* ]]; then
    MOUNTS="${MOUNTS},${HF_HOME}:${HF_HOME}"
  fi
fi

# ---------- W&B ----------
WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_API_KEY
if [[ -n "${WANDB_API_KEY}" ]]; then
  WANDB_ENABLED=True
else
  WANDB_ENABLED=False
  echo "WARNING: WANDB_API_KEY is not set — wandb logging disabled, training will log locally only." >&2
fi
WANDB_PROJECT="${WANDB_PROJECT:-${ALGO}-dev}"
WANDB_NAME="${WANDB_NAME:-${JOB_NAME}}"

# ---------- Code snapshot ----------
# Batch mode: snapshot by default so the job runs on code frozen at submission
# time, unaffected by edits you make while it's queued/running.
# Interactive/DRY_RUN: live directory by default, for fast iteration.
if [[ "${INTERACTIVE}" == "1" || "${DRY_RUN}" == "1" ]]; then
  USE_SNAPSHOT="${USE_SNAPSHOT:-0}"
else
  USE_SNAPSHOT="${USE_SNAPSHOT:-1}"
fi

CODE_ROOT="${PROJECT_ROOT}"
if [[ "${USE_SNAPSHOT}" == "1" ]]; then
  SNAPSHOT_DIR=$(bash "${PROJECT_ROOT}/tools/code_snapshot.sh" "${JOB_NAME}-$(date +%Y%m%d-%H%M%S)")
  echo "Code snapshot: ${SNAPSHOT_DIR}"
  CODE_ROOT="${SNAPSHOT_DIR}"
  MOUNTS="${MOUNTS},${SNAPSHOT_DIR}:${SNAPSHOT_DIR}"
elif [[ "${USE_SNAPSHOT}" == "config" ]]; then
  # Lightweight alternative to the full code snapshot: freezes only the
  # CONFIG file, not the whole tracked tree (rsyncing every submodule makes
  # the full snapshot slow). CODE_ROOT stays live, so this only protects
  # against CONFIG changing under a queued job (e.g. via SYNC_CONFIGS) --
  # NOT against the actual source code changing under it. Use USE_SNAPSHOT=1
  # instead if you also SYNC=1 (full repo) between back-to-back launches.
  CONFIG_SNAPSHOT_DIR="${PROJECT_ROOT}/code_snapshots/${JOB_NAME}-$(date +%Y%m%d-%H%M%S)-config"
  mkdir -p "${CONFIG_SNAPSHOT_DIR}"
  cp "${CONFIG}" "${CONFIG_SNAPSHOT_DIR}/$(basename "${CONFIG}")"
  CONFIG="${CONFIG_SNAPSHOT_DIR}/$(basename "${CONFIG}")"
  echo "Config snapshot: ${CONFIG}"
fi
UV_SYNC_PROJECT_ROOT="${CODE_ROOT}"
if [[ -n "${CONTAINER_CODE_ROOT}" ]]; then
  if [[ "${CONTAINER_CODE_ROOT}" != /* ]]; then
    echo "ERROR: CONTAINER_CODE_ROOT must be absolute: ${CONTAINER_CODE_ROOT}" >&2
    exit 1
  fi
  # Keep the host-side source path available for locked dependency preparation.
  # Training runs through the canonical container path below.
  if [[ "${MOUNTS}" != *"${UV_SYNC_PROJECT_ROOT}:${UV_SYNC_PROJECT_ROOT}"* ]]; then
    MOUNTS="${MOUNTS},${UV_SYNC_PROJECT_ROOT}:${UV_SYNC_PROJECT_ROOT}"
  fi
  if [[ "${CONTAINER_CODE_MOUNT_MODE}" == "rw" ]]; then
    MOUNTS="${MOUNTS},${CODE_ROOT}:${CONTAINER_CODE_ROOT}"
  else
    MOUNTS="${MOUNTS},${CODE_ROOT}:${CONTAINER_CODE_ROOT}:ro"
  fi
  CODE_ROOT="${CONTAINER_CODE_ROOT}"
fi

# ---------- Build the Hydra override list ----------
OVERRIDES=(
  "--config" "${CONFIG}"
  "cluster.num_nodes=${NUM_NODES}"
  "cluster.gpus_per_node=${GPUS_PER_NODE}"
  "data.train.data_path='${TRAIN_DATA_PATH}'"
  "checkpointing.checkpoint_dir='${CHECKPOINT_DIR}'"
  "logger.wandb_enabled=${WANDB_ENABLED}"
  "logger.wandb.project='${WANDB_PROJECT}'"
  "logger.wandb.name='${WANDB_NAME}'"
)

if [[ -n "${VAL_DATA_PATH}" ]]; then
  OVERRIDES+=("data.validation.data_path='${VAL_DATA_PATH}'")
fi
if [[ -n "${MODEL_NAME}" ]]; then
  OVERRIDES+=("policy.model_name='${MODEL_NAME}'" "policy.tokenizer.name='${MODEL_NAME}'")
fi
if [[ "${ALGO}" == "eiwor" && -n "${BETA}" ]]; then
  OVERRIDES+=("eiwor.beta=${BETA}")
fi
if [[ "${ALGO}" == "oapl" && -n "${OAPL_BETA1}" ]]; then
  OVERRIDES+=("oapl.beta1=${OAPL_BETA1}" "data.train.beta1=${OAPL_BETA1}")
  [[ -n "${VAL_DATA_PATH}" ]] && OVERRIDES+=("data.validation.beta1=${OAPL_BETA1}")
fi
if [[ "${ALGO}" == "oapl" && -n "${OAPL_BETA2}" ]]; then
  OVERRIDES+=("oapl.beta2=${OAPL_BETA2}")
fi
if [[ "${ALGO}" == "eiwor" && -n "${NUM_COMPLETIONS_PER_PROMPT}" ]]; then
  OVERRIDES+=("eiwor.num_completions_per_prompt=${NUM_COMPLETIONS_PER_PROMPT}")
fi
if [[ "${ALGO}" != "grpo" && -n "${BIRD_SQL_DB_DIR}" ]]; then
  echo "ERROR: BIRD_SQL_DB_DIR is GRPO-only; unset it or set ALGO=grpo." >&2
  exit 1
fi
if [[ "${ALGO}" == "grpo" && -n "${BIRD_SQL_DB_DIR}" ]]; then
  OVERRIDES+=("env.nemo_gym.bird_sql_resources_server.resources_servers.bird_sql.bird_sql_dir='${BIRD_SQL_DB_DIR}'")
fi

if [[ -n "${BIRD_SQL_SPLIT}" && "${BIRD_SQL_SPLIT}" != "dev" && "${BIRD_SQL_SPLIT}" != "train" ]]; then
  echo "ERROR: BIRD_SQL_SPLIT must be 'dev' or 'train' (got '${BIRD_SQL_SPLIT}')" >&2
  exit 1
fi
if [[ -n "${BIRD_SQL_DIALECT}" && "${BIRD_SQL_DIALECT}" != "sqlite" && "${BIRD_SQL_DIALECT}" != "spark" ]]; then
  echo "ERROR: BIRD_SQL_DIALECT must be 'sqlite' or 'spark' (got '${BIRD_SQL_DIALECT}')" >&2
  exit 1
fi
if [[ "${ALGO}" != "grpo" && ( -n "${BIRD_SQL_SPLIT}" || -n "${BIRD_SQL_DIALECT}" ) ]]; then
  echo "ERROR: BIRD_SQL_SPLIT/BIRD_SQL_DIALECT are GRPO-only; unset them or set ALGO=grpo." >&2
  exit 1
fi
if [[ "${ALGO}" == "grpo" && -n "${BIRD_SQL_SPLIT}" ]]; then
  OVERRIDES+=("env.nemo_gym.bird_sql_resources_server.resources_servers.bird_sql.split=${BIRD_SQL_SPLIT}")
fi
if [[ "${ALGO}" == "grpo" && -n "${BIRD_SQL_DIALECT}" ]]; then
  OVERRIDES+=("env.nemo_gym.bird_sql_resources_server.resources_servers.bird_sql.dialect=${BIRD_SQL_DIALECT}")
fi
# New keys (not in the yaml), hence the Hydra "+" prefix. NeMo-RL only fills uv_venv_dir/uv_cache_dir
# with setdefault, so these win over the container defaults.
if [[ "${ALGO}" == "grpo" && -n "${GYM_VENV_DIR}" ]]; then
  OVERRIDES+=("+env.nemo_gym.uv_venv_dir='${GYM_VENV_DIR}'" "+env.nemo_gym.skip_venv_if_present=true")
fi
if [[ "${ALGO}" == "grpo" && -n "${GYM_UV_CACHE_DIR}" ]]; then
  OVERRIDES+=("+env.nemo_gym.uv_cache_dir='${GYM_UV_CACHE_DIR}'")
fi

RUN_SCRIPT="examples/run_${ALGO}.py"
# GRPO here always uses NeMo-Gym, which needs its own runner: it binds
# task_to_env["nemo_gym"], clears stop_token_ids after configure_generation_config
# and sets the validation sizes. examples/run_grpo.py does none of these.
if [[ "${ALGO}" == "grpo" ]]; then
  RUN_SCRIPT="examples/nemo_gym/run_grpo_nemo_gym.py"
fi
# --frozen: install from the committed uv.lock as-is, skipping a fresh resolve
# (needed because a live resolve currently fails — see the sglang/nemo-gym
# openai-version conflict in pyproject.toml's [tool.uv] conflicts list).
# --extra automodel: OAPL/EIWOR default to the DTensor backend, which only
# needs the automodel extra; this also avoids ever resolving the sglang extra.
# The base DTensor recipes use `automodel`; set UV_EXTRA=mcore for Megatron
# recipes so the frozen environment includes Bridge runtime dependencies.
UV_EXTRA="${UV_EXTRA:-automodel}"
# The container's baked-in submodule/pyproject fingerprint won't match a
# manually-cloned or freshly-snapshotted checkout; bypass the resulting
# version-mismatch check rather than block the run on it. Set inline (not
# just exported here) since ray.sub only forwards a fixed allowlist of env
# vars into the container - this guarantees it's present regardless.
NRL_IGNORE_VERSION_MISMATCH="${NRL_IGNORE_VERSION_MISMATCH:-1}"

# ---------- Seed the Gym venvs (SEED_GYM_VENVS=1): one short job, no GPUs ----------
# Builds every NeMo-Gym server venv into GYM_VENV_DIR (and fills GYM_UV_CACHE_DIR) using the repo's
# own dry-run prefetch (examples/nemo_gym/prefetch_venvs.py), then exits. Training jobs then reuse
# the venvs (skip_venv_if_present) instead of downloading packages while holding GPUs.
if [[ "${SEED_GYM_VENVS}" == "1" ]]; then
  SEED_DIR="${BASE_LOG_DIR:-${PROJECT_ROOT}}"
  SEED_SCRIPT="${SEED_DIR}/seed_gym_venvs_cmd.sh"
  # prefetch_venvs.py takes no overrides and even a dry run constructs each server, and bird_sql's
  # constructor looks for (and otherwise downloads) the BIRD databases under bird_sql_dir. So run it
  # on a copy of the config whose placeholder bird_sql_dir points at the real, mounted database dir.
  SEED_CONFIG="${SEED_DIR}/seed_gym_venvs_config.yaml"
  # Same actor-venv upgrade SETUP_COMMAND does (the container's NemoGym venv has an old openai),
  # then the dry-run prefetch with the venv dir / cache dir on shared storage. The nemo-rl env sync
  # (`uv sync`) must NOT see GYM_UV_CACHE_DIR: with that empty cache uv rebuilds transformer-engine,
  # grouped-gemm and deep-ep from source instead of using the container's cache. Only the Gym parts
  # (actor-venv upgrade, prefetch) use the shared cache; `uv run --no-sync` keeps it from re-syncing.
  # The prefetch gets its own Ray head on a free port (RAY_ADDRESS points at it): init_ray() first tries
  # ray.init(address="auto"), and on a shared CPU node that attached to another job's Ray cluster and crashed
  # ("Failed to connect to socket ... raylet"). RAY_ADDRESS from the environment wins over "auto".
  # uv in the image may predate the pinned Python ("No interpreter found"), so fetch it with a pinned newer uv.
  PY_BOOTSTRAP=""
  if [[ -n "${UV_PYTHON_INSTALL_DIR}" ]]; then
    PY_BOOTSTRAP="(ls ${UV_PYTHON_INSTALL_DIR} 2>/dev/null | grep -q cpython-${UV_PYTHON_VERSION} || (curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh | env UV_UNMANAGED_INSTALL=/tmp/uv-bootstrap-\${SLURM_JOB_ID:-0} sh && /tmp/uv-bootstrap-\${SLURM_JOB_ID:-0}/uv python install ${UV_PYTHON_VERSION})) && uv python find '>=${UV_PYTHON_VERSION}' && "
  fi
  # BIRD databases (stdlib-only downloader shipped with the Gym server; idempotent) and the model, so a
  # training job spends no GPU time downloading either.
  BIRD_FN="ensure_bird_train"
  [[ "${BIRD_SQL_SPLIT}" == "dev" ]] && BIRD_FN="ensure_bird_sql"
  DB_STEP="python3 -c \"import sys; sys.path.insert(0, '${CODE_ROOT}/3rdparty/Gym-workspace/Gym/resources_servers/bird_sql'); import setup_bird_sql; setup_bird_sql.${BIRD_FN}('${BIRD_SQL_DB_DIR}')\" && "
  MODEL_STEP=""
  if [[ -n "${MODEL_NAME}" && -n "${HF_HOME}" ]]; then
    MODEL_STEP="HF_HOME=${HF_HOME} uv run --frozen --no-sync --extra ${UV_EXTRA} python -c \"from huggingface_hub import snapshot_download; snapshot_download('${MODEL_NAME}')\" && "
  fi
  # Same environment as the training command, so the seed checks (or fills) the env training will actually use.
  SEED_ENV_PREFIX="${UV_SYNC_CACHE_DIR:+UV_CACHE_DIR=${UV_SYNC_CACHE_DIR} }${UV_PROJECT_ENVIRONMENT:+UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT} }"
  ENV_SYNC_STEP="${SEED_ENV_PREFIX}uv sync --frozen --extra ${UV_EXTRA} && ${MODEL_STEP}"
  # Skip mode: no sync/model step, but run the training command's own `uv run` (with its sync) once and import the
  # packages the run needs, so a stale or mismatched image environment fails here, on CPU, not on the GPU nodes.
  # Megatron and Transformer Engine are only located, not imported: TE loads libcuda.so.1, which CPU nodes lack.
  SEED_NO_SYNC_FLAG=""
  [[ "${UV_RUN_NO_SYNC}" == "1" ]] && SEED_NO_SYNC_FLAG="--no-sync "
  if [[ "${SEED_SKIP_ENV_SYNC}" == "1" ]]; then
    ENV_SYNC_STEP="${SEED_ENV_PREFIX}uv run --frozen ${SEED_NO_SYNC_FLAG}--extra ${UV_EXTRA} python -c \"import importlib.util as u, torch, ray, nemo_rl.algorithms.grpo, nemo_rl.environments.nemo_gym, nemo.lens; m = [n for n in ('megatron.core', 'megatron.bridge', 'transformer_engine') if u.find_spec(n) is None]; assert not m, m; print('env ok: torch', torch.__version__, 'ray', ray.__version__)\" && ${MODEL_STEP}"
  fi
  SEED_CMD="cd ${CODE_ROOT} && export NRL_IGNORE_VERSION_MISMATCH=${NRL_IGNORE_VERSION_MISMATCH} PYTHONUNBUFFERED=1 MAX_JOBS=${SEED_MAX_JOBS} UV_CONCURRENT_BUILDS=${SEED_CONCURRENT_BUILDS} ${SEED_BUILD_ENV} ${UV_PYTHON_INSTALL_DIR:+UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR}} && ${DB_STEP}${PY_BOOTSTRAP}${GYM_UV_CACHE_DIR:+UV_CACHE_DIR=${GYM_UV_CACHE_DIR}} uv pip install --python /opt/ray_venvs/nemo_rl.environments.nemo_gym.NemoGym/bin/python -r ${CODE_ROOT}/3rdparty/Gym-workspace/Gym/pyproject.toml && ${ENV_SYNC_STEP}unset RAY_ADDRESS && export RAY_TMPDIR=/tmp/seedray_\${SLURM_JOB_ID:-0} && mkdir -p \$RAY_TMPDIR && SEED_RAY_PORT=\$(python3 -c \"import random, socket; print(next(p for p in random.sample(range(40000, 50000), 1000) if socket.socket().connect_ex(('127.0.0.1', p))))\") && uv run --frozen --no-sync --extra ${UV_EXTRA} ray start --head --node-ip-address=127.0.0.1 --port=\$SEED_RAY_PORT --temp-dir=\$RAY_TMPDIR --num-cpus=${SEED_CPUS} --include-dashboard=false --disable-usage-stats && export RAY_ADDRESS=127.0.0.1:\$SEED_RAY_PORT && { SEED_RC=0; ${GYM_UV_CACHE_DIR:+UV_CACHE_DIR=${GYM_UV_CACHE_DIR}} NEMO_GYM_VENV_DIR='${GYM_VENV_DIR}' uv run --frozen --no-sync --extra ${UV_EXTRA} python examples/nemo_gym/prefetch_venvs.py ${SEED_CONFIG} || SEED_RC=\$?; exit \$SEED_RC; }"
  SEED_GRES=()
  if [[ "${SEED_GPUS}" -gt 0 ]]; then
    SEED_GRES=(--gres=gpu:"${SEED_GPUS}")
  fi
  echo "=============================================================="
  echo "SEED_GYM_VENVS=1: building Gym venvs (no training job)"
  echo "GYM_VENV_DIR:     ${GYM_VENV_DIR}"
  echo "GYM_UV_CACHE_DIR: ${GYM_UV_CACHE_DIR}"
  echo "CONFIG:           ${CONFIG} (copied to ${SEED_CONFIG} with bird_sql_dir=${BIRD_SQL_DB_DIR})"
  echo "COMMAND:          ${SEED_CMD}"
  echo "=============================================================="
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN=1: not submitting."
    exit 0
  fi
  mkdir -p "${SEED_DIR}"
  sed "s#/path/to/local/bird_sql_databases#${BIRD_SQL_DB_DIR}#" "${CONFIG}" > "${SEED_CONFIG}"
  if ! grep -q "bird_sql_dir: ${BIRD_SQL_DB_DIR}" "${SEED_CONFIG}"; then
    echo "ERROR: could not set bird_sql_dir in ${SEED_CONFIG}; is the placeholder missing from ${CONFIG}?" >&2
    exit 1
  fi
  printf '#!/bin/bash\nset -ex\n%s\n' "${SEED_CMD}" > "${SEED_SCRIPT}"
  sbatch \
    --nodes=1 --ntasks=1 --cpus-per-task="${SEED_CPUS}" --mem="${SEED_MEM}" \
    --account="${SLURM_ACCOUNT}" \
    --job-name="seed-gym-venvs" \
    --partition="${SEED_PARTITION:-${PARTITION}}" \
    --time="${SEED_WALLTIME}" \
    --no-requeue \
    "${SEED_GRES[@]}" \
    --output="${SEED_DIR}/seed-gym-venvs-%j.out" \
    --wrap "srun --no-container-mount-home --container-mounts=${MOUNTS} --container-image=${CONTAINER} --container-workdir=${CODE_ROOT} bash ${SEED_SCRIPT}"
  echo "Seeding job submitted. Output: ${SEED_DIR}/seed-gym-venvs-<jobid>.out"
  exit 0
fi
UV_PYTHON_PREFIX=""
if [[ -n "${UV_PYTHON_INSTALL_DIR}" ]]; then
  UV_PYTHON_PREFIX="UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR} "
fi
UV_NO_SYNC_FLAG=""
[[ "${UV_RUN_NO_SYNC}" == "1" ]] && UV_NO_SYNC_FLAG="--no-sync "
UV_PROJECT_ENVIRONMENT_PREFIX=""
if [[ -n "${UV_PROJECT_ENVIRONMENT}" ]]; then
  UV_PROJECT_ENVIRONMENT_PREFIX="UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT} "
fi
UV_SYNC_CACHE_PREFIX=""
if [[ -n "${UV_SYNC_CACHE_DIR}" ]]; then
  UV_SYNC_CACHE_PREFIX="UV_CACHE_DIR=${UV_SYNC_CACHE_DIR} "
fi
if [[ "${USE_LOCKED_DEPS_ENV}" == "1" ]]; then
  LOCAL_AUTOMODEL="${CODE_ROOT}/3rdparty/Automodel-workspace/Automodel"
  LOCAL_NEMO_GYM="${CODE_ROOT}/3rdparty/Gym-workspace/Gym"
  LOCAL_MEGATRON_BRIDGE="${CODE_ROOT}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge"
  LOCAL_MEGATRON_BRIDGE_SRC="${LOCAL_MEGATRON_BRIDGE}/src"
  LOCAL_MEGATRON_LM="${LOCAL_MEGATRON_BRIDGE}/3rdparty/Megatron-LM"
  RUNTIME_BIN_DIR="${CODE_ROOT}/.runtime-bin"
  SYNC_ENV_SCRIPT="${CODE_ROOT}/scripts/sync_offline_training_environment.sh"
  PRECOMPILE_CMD=""
  if [[ "${PRECOMPILE_MCORE_DATASET_HELPER}" == "1" ]]; then
    PRECOMPILE_SCRIPT="${CODE_ROOT}/scripts/precompile_mcore_dataset_helper.sh"
    PRECOMPILE_CMD="env VIRTUAL_ENV=${UV_PROJECT_ENVIRONMENT} PYTHONPATH=${CODE_ROOT}:${LOCAL_MEGATRON_BRIDGE_SRC}:${LOCAL_MEGATRON_LM} ${PRECOMPILE_SCRIPT} ${CODE_ROOT} ${UV_PROJECT_ENVIRONMENT}/bin/python && "
  fi
  # Prepare dependencies against the run-local host path, prebuild the only
  # in-source MCore artifact once when requested, then source-select the
  # run-local Bridge/MCore trees through PYTHONPATH.
  TRAIN_CMD="${UV_SYNC_CACHE_PREFIX}UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT} bash ${SYNC_ENV_SCRIPT} ${UV_SYNC_PROJECT_ROOT} ${UV_PROJECT_ENVIRONMENT} ${UV_EXTRA} && cd ${CODE_ROOT} && ${PRECOMPILE_CMD}PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 NRL_IGNORE_VERSION_MISMATCH=${NRL_IGNORE_VERSION_MISMATCH} NEMO_RL_PY_EXECUTABLES_SYSTEM=1 VIRTUAL_ENV=${UV_PROJECT_ENVIRONMENT} UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT} PATH=${RUNTIME_BIN_DIR}:${UV_PROJECT_ENVIRONMENT}/bin:\${PATH} PYTHONPATH=${CODE_ROOT}:${LOCAL_AUTOMODEL}:${LOCAL_NEMO_GYM}:${LOCAL_MEGATRON_BRIDGE_SRC}:${LOCAL_MEGATRON_LM} ${UV_PROJECT_ENVIRONMENT}/bin/python ./${RUN_SCRIPT} ${OVERRIDES[*]} ${EXTRA_ARGS}"
else
  TRAIN_CMD="cd ${CODE_ROOT} && PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 NRL_IGNORE_VERSION_MISMATCH=${NRL_IGNORE_VERSION_MISMATCH} ${UV_PYTHON_PREFIX}${UV_SYNC_CACHE_PREFIX}${UV_PROJECT_ENVIRONMENT_PREFIX}uv run --frozen ${UV_NO_SYNC_FLAG}--extra ${UV_EXTRA} ./${RUN_SCRIPT} ${OVERRIDES[*]} ${EXTRA_ARGS}"
fi

RAY_SUB="${RAY_SUB:-${PROJECT_ROOT}/ray.sub}"
if [[ ! -f "${RAY_SUB}" ]]; then
  echo "ERROR: ray.sub not found at ${RAY_SUB}" >&2
  exit 1
fi

export CONTAINER
export MOUNTS
export GPUS_PER_NODE
[[ -n "${BASE_LOG_DIR}" ]] && export BASE_LOG_DIR

echo "=============================================================="
echo "ALGO:            ${ALGO}"
echo "NUM_NODES:       ${NUM_NODES}"
echo "GPUS_PER_NODE:   ${GPUS_PER_NODE}"
echo "TRAIN_DATA_PATH: ${TRAIN_DATA_PATH}"
echo "VAL_DATA_PATH:   ${VAL_DATA_PATH:-<none>}"
echo "CHECKPOINT_DIR:  ${CHECKPOINT_DIR}"
echo "CONTAINER:       ${CONTAINER}"
echo "MOUNTS:          ${MOUNTS}"
echo "WANDB_ENABLED:   ${WANDB_ENABLED} (project=${WANDB_PROJECT}, name=${WANDB_NAME})"
echo "CODE_ROOT:       ${CODE_ROOT}"
echo "CODE_MOUNT_MODE: ${CONTAINER_CODE_MOUNT_MODE}"
echo "UV_PROJECT_ENV:  ${UV_PROJECT_ENVIRONMENT:-<default>}"
echo "UV_SYNC_CACHE:    ${UV_SYNC_CACHE_DIR:-<container default>}"
echo "LOCKED_DEPS_ENV: ${USE_LOCKED_DEPS_ENV}"
echo "RUN_LOCAL_UV_ENV:${REQUIRE_RUN_LOCAL_UV_PROJECT_ENVIRONMENT}"
echo "PRECOMPILE_MCORE:${PRECOMPILE_MCORE_DATASET_HELPER}"
echo "COMMAND:         ${TRAIN_CMD}"
echo "=============================================================="

if [[ "${DRY_RUN}" == "1" ]]; then
  echo ""
  echo "DRY_RUN=1: not submitting to Slurm. Command:"
  printf '%s\n' "${TRAIN_CMD}" | sed -E \
    -e 's/(HF_TOKEN=)[^[:space:]]*/\1<redacted>/g' \
    -e 's/(WANDB_API_KEY=)[^[:space:]]*/\1<redacted>/g'
  exit 0
fi

if [[ "${INTERACTIVE}" == "1" ]]; then
  # Ensure COMMAND is not in the environment; ray.sub does COMMAND=${COMMAND:-},
  # so unset -> empty string -> idle mode (creates attach script, sleeps forever).
  unset COMMAND 2>/dev/null || true

  echo ""
  echo "  INTERACTIVE MODE — submitting a ${NUM_NODES}-node allocation (walltime ${WALLTIME})."
  echo "  Ray cluster will start and idle until you attach."
  echo ""

  submission_output=$(sbatch \
    --nodes="${NUM_NODES}" \
    --account="${SLURM_ACCOUNT}" \
    --job-name="interactive-${JOB_NAME}" \
    --partition="${PARTITION}" \
    --time="${WALLTIME}" \
    --no-requeue \
    --gres=gpu:"${GPUS_PER_NODE}" \
    --exclusive \
    --mem=0 \
    ${SLURM_COMMENT:+--comment="${SLURM_COMMENT}"} \
    ${SEGMENT_SIZE:+--segment="${SEGMENT_SIZE}"} \
    ${SLURM_QOS:+--qos="${SLURM_QOS}"} \
    "${SBATCH_OUTPUT_ARGS[@]}" \
    "${RAY_SUB}")
  echo "${submission_output}"

  if [[ "${submission_output}" =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
    JOB_ID="${BASH_REMATCH[1]}"
  else
    echo "ERROR: Could not parse job ID from sbatch output." >&2
    exit 1
  fi

  LAUNCH_DIR="$(pwd)"
  ATTACH_SCRIPT="${LAUNCH_DIR}/${JOB_ID}-attach.sh"
  CMD_FILE="${LAUNCH_DIR}/${JOB_ID}-run-cmd.sh"
  printf '%s\n' "${TRAIN_CMD}" > "${CMD_FILE}"
  chmod +x "${CMD_FILE}"

  echo ""
  echo "  Saved training command to: ${CMD_FILE}"
  echo "  Attach once Ray is ready:  bash ${ATTACH_SCRIPT}"
  echo "  Then run:                  source ${CMD_FILE}"
  echo "  Or non-interactively:      COMMAND=\"\$(cat ${CMD_FILE})\" bash ${ATTACH_SCRIPT}"
  echo "  Edit + re-run w/o requeue: vim ${CMD_FILE} && source ${CMD_FILE}"
  echo "  Cancel:                    scancel ${JOB_ID}"

  if [[ "${INTERACTIVE_WAIT}" == "1" ]]; then
    echo ""
    echo "  Waiting in foreground for Ray to be ready (Ctrl+C to stop waiting)..."
    prev_state=""
    while [[ ! -f "${ATTACH_SCRIPT}" ]]; do
      state=$(squeue -j "${JOB_ID}" -h -o "%T" 2>/dev/null || true)
      if [[ -z "${state}" ]]; then
        echo "  Job ${JOB_ID} is no longer in the queue. Check: sacct -j ${JOB_ID}"
        exit 1
      fi
      if [[ "${state}" != "${prev_state}" ]]; then
        echo "  [$(date +%H:%M:%S)] Job state: ${state}"
        prev_state="${state}"
      fi
      sleep 15
    done
    echo ""
    echo "  Ray cluster is ready! Attach with: bash ${ATTACH_SCRIPT}"
  fi
  exit 0
fi

# ---------- Batch mode ----------
export COMMAND="${TRAIN_CMD}"

SINGLETON_ARGS=()
if [[ "${SINGLETON}" == "1" ]]; then
  SINGLETON_ARGS=(--dependency=singleton)
fi

sbatch \
  --nodes="${NUM_NODES}" \
  --account="${SLURM_ACCOUNT}" \
  --job-name="${JOB_NAME}" \
  --partition="${PARTITION}" \
  --time="${WALLTIME}" \
  --no-requeue \
  --gres=gpu:"${GPUS_PER_NODE}" \
  --exclusive \
  --mem=0 \
  "${SINGLETON_ARGS[@]}" \
  ${SLURM_COMMENT:+--comment="${SLURM_COMMENT}"} \
  ${SEGMENT_SIZE:+--segment="${SEGMENT_SIZE}"} \
  ${SLURM_QOS:+--qos="${SLURM_QOS}"} \
  "${SBATCH_OUTPUT_ARGS[@]}" \
  "${RAY_SUB}"

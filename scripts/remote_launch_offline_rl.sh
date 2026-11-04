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
# remote_launch_offline_rl.sh
#
# Run this ON YOUR MACBOOK. It SSHes into a Slurm cluster's login node and
# submits a multi-node OAPL/EIWOR job there via
# scripts/launch_offline_rl_slurm.sh (which wraps `sbatch ray.sub`, per the
# README's "OAPL Multi-node" / "EIWOR Multi-node" instructions).
#
# It does NOT copy your local repo by default — it assumes REMOTE_REPO_DIR
# already holds a clone of Offline-RL-Internal on the cluster (shared
# filesystem). Set SYNC=1 to rsync your local repo over first (careful: this
# overwrites files under REMOTE_REPO_DIR that match your local tree; add
# extra excludes via RSYNC_EXTRA_ARGS if needed).
#
# Required:
#   LOGIN_HOST         ssh target for the login node, e.g. user@cluster-login.example.com
#   REMOTE_REPO_DIR     Absolute path to Offline-RL-Internal on the cluster
#   ALGO                oapl | eiwor
#   TRAIN_DATA_PATH     JSONL path, as visible on the CLUSTER filesystem
#   CONTAINER, SLURM_ACCOUNT, PARTITION   (same as launch_offline_rl_slurm.sh)
#
# All other env vars accepted by scripts/launch_offline_rl_slurm.sh are
# forwarded through automatically if set, including: VAL_DATA_PATH, NUM_NODES,
# GPUS_PER_NODE, MODEL_NAME, BETA, OAPL_BETA1, OAPL_BETA2,
# NUM_COMPLETIONS_PER_PROMPT, BIRD_SQL_DB_DIR, BIRD_SQL_SPLIT, BIRD_SQL_DIALECT,
# SETUP_COMMAND (run by ray.sub inside every node's container before Ray starts), CONFIG,
# CHECKPOINT_DIR, JOB_NAME, WALLTIME, SLURM_QOS, SLURM_COMMENT, SEGMENT_SIZE,
# SINGLETON, MOUNTS, HF_HOME, HF_TOKEN, HF_HUB_CACHE, HF_DATASETS_CACHE,
# WANDB_PROJECT, WANDB_NAME, WANDB_API_KEY, EXTRA_ARGS, DRY_RUN, INTERACTIVE,
# INTERACTIVE_WAIT, USE_SNAPSHOT, CONTAINER_CODE_ROOT,
# CONTAINER_CODE_MOUNT_MODE, UV_PROJECT_ENVIRONMENT, UV_SYNC_CACHE_DIR, UV_EXTRA,
# NRL_IGNORE_VERSION_MISMATCH, USE_LOCKED_DEPS_ENV,
# PRECOMPILE_MCORE_DATASET_HELPER, RAY_LOG_SYNC_FREQUENCY, HOSTMON_ENABLED,
# HOSTMON_INTERVAL.
#
# Usage:
#   LOGIN_HOST=dmisra@cluster-login \
#   REMOTE_REPO_DIR=/home/dmisra/Offline-RL-Internal \
#   ALGO=oapl \
#   TRAIN_DATA_PATH=/lustre/data/oapl_train.jsonl \
#   VAL_DATA_PATH=/lustre/data/oapl_val.jsonl \
#   CONTAINER=nvcr.io/nvidia/nemo-rl:latest \
#   SLURM_ACCOUNT=my_account \
#   PARTITION=batch \
#   NUM_NODES=4 \
#   WANDB_API_KEY=xxxx \
#   ./scripts/remote_launch_offline_rl.sh
#
#   # Dry run first (recommended): just prints the sbatch command, no submit
#   DRY_RUN=1 ... ./scripts/remote_launch_offline_rl.sh
#
#   # Interactive debugging allocation (idle Ray cluster you attach to and
#   # re-run commands in without requeueing):
#   INTERACTIVE=1 ... ./scripts/remote_launch_offline_rl.sh
# =============================================================================

LOGIN_HOST="${LOGIN_HOST:?Set LOGIN_HOST, e.g. user@cluster-login.example.com}"
REMOTE_REPO_DIR="${REMOTE_REPO_DIR:?Set REMOTE_REPO_DIR to the Offline-RL-Internal path on the cluster}"
SYNC="${SYNC:-0}"
SYNC_CONFIGS="${SYNC_CONFIGS:-1}"  # lightweight: always sync the BIRD GRPO configs by default. Set 0 to disable.
SYNC_CONFIGS_INCLUDE="${SYNC_CONFIGS_INCLUDE:-grpo_birdbench*.yaml}"  # only these files: the remote checkout may be a release branch whose other configs must stay as they are

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
LOCAL_REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

if [[ "${SYNC}" == "1" ]]; then
  echo "Syncing ${LOCAL_REPO_ROOT} -> ${LOGIN_HOST}:${REMOTE_REPO_DIR} ..."
  # --checksum: see the SYNC_CONFIGS branch below for why this matters --
  # same-length edits can otherwise be silently skipped based on size+mtime.
  # shellcheck disable=SC2086
  rsync -avz --delete --checksum \
    --exclude '.git' --exclude '.venv' --exclude '__pycache__' --exclude 'results' \
    ${RSYNC_EXTRA_ARGS:-} \
    "${LOCAL_REPO_ROOT}/" "${LOGIN_HOST}:${REMOTE_REPO_DIR}/"
elif [[ "${SYNC_CONFIGS}" == "1" ]]; then
  echo "Syncing configs only: ${LOCAL_REPO_ROOT}/examples/configs/ -> ${LOGIN_HOST}:${REMOTE_REPO_DIR}/examples/configs/ ..."
  # --checksum: compare file content, not just size+mtime. Without it, a
  # same-length edit (e.g. "1.0" -> "0.5") can be silently skipped if the
  # remote file's mtime happens to be >= the local one (clock skew between
  # your Mac and the cluster), even though the content actually differs.
  rsync -avz --checksum --include "${SYNC_CONFIGS_INCLUDE}" --exclude "*" \
    "${LOCAL_REPO_ROOT}/examples/configs/" "${LOGIN_HOST}:${REMOTE_REPO_DIR}/examples/configs/"
fi

# Env vars understood by launch_offline_rl_slurm.sh; forward whichever are set.
FORWARD_VARS=(
  ALGO TRAIN_DATA_PATH VAL_DATA_PATH CONTAINER SLURM_ACCOUNT PARTITION
  NUM_NODES GPUS_PER_NODE MODEL_NAME BETA OAPL_BETA1 OAPL_BETA2
  NUM_COMPLETIONS_PER_PROMPT BIRD_SQL_DB_DIR BIRD_SQL_SPLIT BIRD_SQL_DIALECT SETUP_COMMAND
  GYM_VENV_DIR GYM_UV_CACHE_DIR UV_PYTHON_INSTALL_DIR SEED_GYM_VENVS SEED_SKIP_ENV_SYNC SEED_PARTITION SEED_GPUS SEED_CPUS SEED_MEM SEED_WALLTIME SEED_MAX_JOBS SEED_CONCURRENT_BUILDS SEED_BUILD_ENV
  CONFIG CHECKPOINT_DIR JOB_NAME WALLTIME SLURM_QOS SLURM_COMMENT SEGMENT_SIZE
  SINGLETON MOUNTS BASE_LOG_DIR HF_HOME HF_TOKEN HF_HUB_CACHE HF_DATASETS_CACHE
  WANDB_PROJECT WANDB_NAME WANDB_API_KEY EXTRA_ARGS DRY_RUN
  INTERACTIVE INTERACTIVE_WAIT USE_SNAPSHOT CONTAINER_CODE_ROOT
  CONTAINER_CODE_MOUNT_MODE PRECOMPILE_MCORE_DATASET_HELPER
  UV_PROJECT_ENVIRONMENT UV_SYNC_CACHE_DIR UV_RUN_NO_SYNC UV_EXTRA NRL_IGNORE_VERSION_MISMATCH
  USE_LOCKED_DEPS_ENV RAY_LOG_SYNC_FREQUENCY HOSTMON_ENABLED HOSTMON_INTERVAL
)

REMOTE_ENV=""
for var in "${FORWARD_VARS[@]}"; do
  if [[ -n "${!var:-}" ]]; then
    REMOTE_ENV+="$(printf '%q=%q ' "${var}" "${!var}")"
  fi
done

# Fail fast locally on missing required vars, with the same messages the
# remote script would give, instead of waiting on a round-trip ssh error.
: "${ALGO:?Set ALGO=oapl or ALGO=eiwor}"
: "${TRAIN_DATA_PATH:?Set TRAIN_DATA_PATH}"
: "${CONTAINER:?Set CONTAINER}"
: "${SLURM_ACCOUNT:?Set SLURM_ACCOUNT}"
: "${PARTITION:?Set PARTITION}"

echo "Submitting on ${LOGIN_HOST} (repo: ${REMOTE_REPO_DIR}) ..."
# shellcheck disable=SC2029
ssh "${LOGIN_HOST}" "cd '${REMOTE_REPO_DIR}' && ${REMOTE_ENV} bash scripts/launch_offline_rl_slurm.sh"

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

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)

cd "${PROJECT_ROOT}"

if [[ -n "$(git status --porcelain)" ]]; then
  printf 'Refusing to update a dirty checkout.\n' >&2
  git status --short >&2
  exit 2
fi

HEAD_BEFORE_PULL=$(git rev-parse HEAD)
git pull --ff-only
git submodule sync --recursive
git submodule update --init --recursive
HEAD_AFTER_PULL=$(git rev-parse HEAD)

if [[ "${HEAD_BEFORE_PULL}" != "${HEAD_AFTER_PULL}" ]]; then
  printf 'Checkout advanced from %s to %s; restarting the updated launcher.\n' \
    "${HEAD_BEFORE_PULL}" "${HEAD_AFTER_PULL}"
  exec "${BASH_SOURCE[0]}" "$@"
fi

if [[ -n "$(git status --porcelain)" ]]; then
  printf 'Refusing to submit from a dirty checkout.\n' >&2
  git status --short >&2
  exit 2
fi

if [[ -n "$(git rev-list '@{upstream}..HEAD')" ]]; then
  printf 'Refusing to submit commits that have not been pushed upstream.\n' >&2
  exit 2
fi

MODEL_CONFIG=${1:-"${SCRIPT_DIR}/models/qwen3-30ba3b-4n4g.env"}
if [[ ! -f "${MODEL_CONFIG}" ]]; then
  printf 'Model config does not exist: %s\n' "${MODEL_CONFIG}" >&2
  exit 2
fi

# shellcheck source=/dev/null
source "${MODEL_CONFIG}"

: "${MODEL_ID:?MODEL_ID is required}"
: "${CONFIG_PATH:?CONFIG_PATH is required}"
: "${NUM_ACTOR_NODES:?NUM_ACTOR_NODES is required}"
: "${GPUS_PER_NODE:?GPUS_PER_NODE is required}"
: "${SEGMENT_SIZE:?SEGMENT_SIZE is required}"
: "${MAX_STEPS:?MAX_STEPS is required}"
: "${TIME_LIMIT:?TIME_LIMIT is required}"
: "${DEFAULT_DEEPEP_COMMIT:?DEFAULT_DEEPEP_COMMIT is required}"

PARTITION=${PARTITION:-batch}
DEEPEP_COMMIT_EXPLICIT=${DEEPEP_COMMIT-}
DEEPEP_COMMIT=${DEEPEP_COMMIT:-"${DEFAULT_DEEPEP_COMMIT}"}
DEEPEP_WHEEL=${DEEPEP_WHEEL:-}
RUN_SUFFIX=${RUN_SUFFIX:-"$(date +%Y%m%d-%H%M%S)"}
RUN_NAME=${RUN_NAME:-"${MODEL_ID}-hybridep-${DEEPEP_COMMIT:0:8}-${RUN_SUFFIX}"}
WANDB_ENABLED=${WANDB_ENABLED:-False}
WANDB_PROJECT=${WANDB_PROJECT:-sna-async-grpo-gb200}
DISPATCHER_MODE=${DISPATCHER_MODE:-hybridep}
PADDING_LOG_ENABLED=${NEMO_RL_HYBRIDEP_LOG_PACKING:-0}
PADDING_LOG_MAX_CALLS=${NEMO_RL_HYBRIDEP_LOG_PACKING_MAX_CALLS:-4096}
PADDING_LOG_RANKS=${NEMO_RL_HYBRIDEP_LOG_PACKING_RANKS:-0}
PADDING_LOG_REDUCE=${NEMO_RL_HYBRIDEP_LOG_PACKING_REDUCE:-1}
CONTAINER=${CONTAINER:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/containers/qwen30-hybridep-oci-20260727/nemo_rl_nightly.sqsh}
HF_HOME=${HF_HOME:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home}
HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-"${HF_HOME}/cache"}
RUN_ROOT=${RUN_ROOT:-"${PROJECT_ROOT}/exp_logs/hybridep/${MODEL_ID}/${RUN_NAME}"}

case "${DISPATCHER_MODE}" in
  hybridep | recipe) ;;
  *)
    printf 'DISPATCHER_MODE must be either hybridep or recipe.\n' >&2
    exit 2
    ;;
esac

if [[ ! "${DEEPEP_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
  printf 'DEEPEP_COMMIT must be a full lowercase 40-character SHA.\n' >&2
  exit 2
fi

if [[ ! "${RUN_SUFFIX}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  printf 'RUN_SUFFIX may contain only letters, digits, dot, underscore, and hyphen.\n' >&2
  exit 2
fi

if [[ ! "${RUN_NAME}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  printf 'RUN_NAME may contain only letters, digits, dot, underscore, and hyphen.\n' >&2
  exit 2
fi

if [[ -n "${DEEPEP_WHEEL}" && -z "${DEEPEP_COMMIT_EXPLICIT}" ]]; then
  printf 'Set DEEPEP_COMMIT to a non-empty SHA whenever DEEPEP_WHEEL is set.\n' >&2
  exit 2
fi

if [[ "${DEEPEP_COMMIT}" != "${DEFAULT_DEEPEP_COMMIT}" && -z "${DEEPEP_WHEEL}" ]]; then
  printf 'A non-default DEEPEP_COMMIT requires DEEPEP_WHEEL.\n' >&2
  exit 2
fi

if [[ -z "${DEEPEP_WHEEL}" ]] && ! grep -Fq \
  "rev=${DEEPEP_COMMIT}#${DEEPEP_COMMIT}" uv.lock; then
  printf 'The default DEEPEP_COMMIT is not present in uv.lock: %s\n' "${DEEPEP_COMMIT}" >&2
  exit 2
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  printf 'Recipe does not exist: %s\n' "${CONFIG_PATH}" >&2
  exit 2
fi

if [[ ! -f "${CONTAINER}" ]]; then
  printf 'Container does not exist: %s\n' "${CONTAINER}" >&2
  exit 2
fi

if [[ "${WANDB_ENABLED}" == "True" && -z "${WANDB_API_KEY:-}" ]]; then
  printf 'WANDB_API_KEY must be set when WANDB_ENABLED=True.\n' >&2
  exit 2
fi

FAIRSHARE_ROWS=$(sshare -a --user="$(id -un)" -o Account,User,FairShare -n -P)
read -r AUTO_ACCOUNT AUTO_FAIRSHARE < <(
  awk -F'|' -v user="$(id -un)" '
    $2 == user {
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1)
      if (!seen || $3 + 0 > best) {
        account = $1
        best = $3 + 0
        fairshare = $3
        seen = 1
      }
    }
    END {
      if (account != "") {
        print account, fairshare
      }
    }
  ' <<< "${FAIRSHARE_ROWS}"
) || true
if [[ -z "${AUTO_ACCOUNT:-}" ]]; then
  printf 'Could not resolve a user-level FairShare account.\n' >&2
  exit 2
fi

ACCOUNT=${ACCOUNT:-"${AUTO_ACCOUNT}"}
ACCOUNT_FAIRSHARE=$(awk -F'|' -v user="$(id -un)" -v account="${ACCOUNT}" '
  $2 == user {
    gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1)
    if ($1 == account) {
      print $3
      exit
    }
  }
' <<< "${FAIRSHARE_ROWS}")
if [[ -z "${ACCOUNT_FAIRSHARE}" ]]; then
  printf 'No user-level FairShare row found for account %s.\n' "${ACCOUNT}" >&2
  exit 2
fi

RL_COMMIT=$(git rev-parse HEAD)
BRIDGE_COMMIT=$(git -C 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge rev-parse HEAD)
MEGATRON_LM_COMMIT=$(git -C 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM rev-parse HEAD)
CONTAINER_SHA256=$(sha256sum "${CONTAINER}" | cut -d' ' -f1)

driver_args=(
  uv run examples/run_grpo.py
  --config "${CONFIG_PATH}"
  "grpo.max_num_steps=${MAX_STEPS}"
  "cluster.num_nodes=${NUM_ACTOR_NODES}"
  "cluster.gpus_per_node=${GPUS_PER_NODE}"
  "cluster.segment_size=${SEGMENT_SIZE}"
  "logger.log_dir=${RUN_ROOT}/training"
  "logger.wandb_enabled=${WANDB_ENABLED}"
  "logger.wandb.project=${WANDB_PROJECT}"
  "logger.wandb.name=${RUN_NAME}"
  logger.monitor_gpus=True
  logger.tensorboard_enabled=True
  checkpointing.enabled=false
  "++deepep_override=${DEEPEP_COMMIT}"
)
if [[ "${DISPATCHER_MODE}" == "hybridep" ]]; then
  driver_args+=(
    policy.megatron_cfg.moe_token_dispatcher_type=flex
    ++policy.megatron_cfg.moe_flex_dispatcher_backend=hybridep
    ++policy.megatron_cfg.moe_hybridep_num_sms=32
  )
  if [[ "${PADDING_LOG_ENABLED}" == "1" ]]; then
    driver_args+=(
      "++policy.megatron_cfg.env_vars.NEMO_RL_HYBRIDEP_LOG_PACKING='${PADDING_LOG_ENABLED}'"
      "++policy.megatron_cfg.env_vars.NEMO_RL_HYBRIDEP_LOG_PACKING_MAX_CALLS='${PADDING_LOG_MAX_CALLS}'"
      "++policy.megatron_cfg.env_vars.NEMO_RL_HYBRIDEP_LOG_PACKING_RANKS='${PADDING_LOG_RANKS}'"
      "++policy.megatron_cfg.env_vars.NEMO_RL_HYBRIDEP_LOG_PACKING_REDUCE='${PADDING_LOG_REDUCE}'"
    )
  fi
fi
printf -v driver_command '%q ' "${driver_args[@]}"

SETUP_COMMAND=
DEEPEP_OVERLAY=
DEEPEP_WHEEL_SHA256=
if [[ -n "${DEEPEP_WHEEL}" ]]; then
  if [[ ! -f "${DEEPEP_WHEEL}" ]]; then
    printf 'DeepEP wheel does not exist: %s\n' "${DEEPEP_WHEEL}" >&2
    exit 2
  fi

  DEEPEP_WHEEL=$(readlink -f -- "${DEEPEP_WHEEL}")
  case "${DEEPEP_WHEEL}" in
    /lustre/*) ;;
    *)
      printf 'DEEPEP_WHEEL must resolve under the shared /lustre mount: %s\n' "${DEEPEP_WHEEL}" >&2
      exit 2
      ;;
  esac

  DEEPEP_OVERLAY="/tmp/nemo-rl-deepep-${DEEPEP_COMMIT:0:12}-${RUN_SUFFIX}"
  DEEPEP_WHEEL_SHA256=$(sha256sum "${DEEPEP_WHEEL}" | cut -d' ' -f1)
  printf -v overlay_assignment 'overlay=%q' "${DEEPEP_OVERLAY}"
  printf -v wheel_assignment 'wheel=%q' "${DEEPEP_WHEEL}"
  printf -v wheel_sha_assignment 'expected_wheel_sha256=%q' "${DEEPEP_WHEEL_SHA256}"
  printf -v SETUP_COMMAND '%s\n' \
    'set -euo pipefail' \
    "${overlay_assignment}" \
    "${wheel_assignment}" \
    "${wheel_sha_assignment}" \
    '[[ "${overlay}" == /tmp/nemo-rl-deepep-* && "${overlay}" != /tmp/nemo-rl-deepep- ]]' \
    'actual_wheel_sha256=$(sha256sum "${wheel}" | cut -d" " -f1)' \
    '[[ "${actual_wheel_sha256}" == "${expected_wheel_sha256}" ]]' \
    'rm -rf -- "${overlay}"' \
    'mkdir -p "${overlay}"' \
    'unset UV_CONFIG_FILE' \
    'UV_NO_CONFIG=1 uv pip install --python /opt/nemo_rl_venv/bin/python --target "${overlay}" --reinstall --no-deps --no-index "${wheel}"' \
    'PYTHONPATH="${overlay}" /opt/nemo_rl_venv/bin/python -c "import importlib.metadata as md, os; import deep_ep, deep_ep_cpp, hybrid_ep_cpp; root = os.path.realpath(os.environ[\"PYTHONPATH\"]); paths = [os.path.realpath(deep_ep.__file__), os.path.realpath(deep_ep_cpp.__file__), os.path.realpath(hybrid_ep_cpp.__file__)]; assert all(os.path.commonpath([root, path]) == root for path in paths), paths; print(\"DEEPEP_RUNTIME_VERSION\", md.version(\"deep_ep\")); print(\"DEEPEP_RUNTIME_PATHS\", *paths)"'
fi

COMMAND="${driver_command}"

mkdir -p "${RUN_ROOT}"

metadata_path="${RUN_ROOT}/submission.env"
{
  printf 'run_name=%q\n' "${RUN_NAME}"
  printf 'model_id=%q\n' "${MODEL_ID}"
  printf 'config_path=%q\n' "${CONFIG_PATH}"
  printf 'account=%q\n' "${ACCOUNT}"
  printf 'account_fairshare=%q\n' "${ACCOUNT_FAIRSHARE}"
  printf 'highest_fairshare_account=%q\n' "${AUTO_ACCOUNT}"
  printf 'highest_fairshare=%q\n' "${AUTO_FAIRSHARE}"
  printf 'nodes=%q\n' "${NUM_ACTOR_NODES}"
  printf 'gpus_per_node=%q\n' "${GPUS_PER_NODE}"
  printf 'segment_size=%q\n' "${SEGMENT_SIZE}"
  printf 'max_steps=%q\n' "${MAX_STEPS}"
  printf 'dispatcher_mode=%q\n' "${DISPATCHER_MODE}"
  printf 'padding_log_enabled=%q\n' "${PADDING_LOG_ENABLED}"
  printf 'padding_log_max_calls=%q\n' "${PADDING_LOG_MAX_CALLS}"
  printf 'padding_log_ranks=%q\n' "${PADDING_LOG_RANKS}"
  printf 'padding_log_reduce=%q\n' "${PADDING_LOG_REDUCE}"
  printf 'nccl_nvls_enable=%q\n' "${NCCL_NVLS_ENABLE:-}"
  printf 'rl_commit=%q\n' "${RL_COMMIT}"
  printf 'bridge_commit=%q\n' "${BRIDGE_COMMIT}"
  printf 'megatron_lm_commit=%q\n' "${MEGATRON_LM_COMMIT}"
  printf 'deepep_commit=%q\n' "${DEEPEP_COMMIT}"
  printf 'deepep_wheel=%q\n' "${DEEPEP_WHEEL}"
  printf 'deepep_wheel_sha256=%q\n' "${DEEPEP_WHEEL_SHA256}"
  printf 'container=%q\n' "${CONTAINER}"
  printf 'container_sha256=%q\n' "${CONTAINER_SHA256}"
  printf 'submitted_at=%q\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')"
} > "${metadata_path}"

export COMMAND
export CONTAINER
export HF_DATASETS_CACHE
export HF_HOME
export MOUNTS="${PROJECT_ROOT}:${PROJECT_ROOT},/lustre:/lustre"
export NRL_FORCE_REBUILD_VENVS=true
export GPUS_PER_NODE
export SETUP_COMMAND
export BASE_LOG_DIR="${RUN_ROOT}/ray"
PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -n "${DEEPEP_OVERLAY}" ]]; then
  PYTHONPATH="${DEEPEP_OVERLAY}:${PYTHONPATH}"
fi
export PYTHONPATH

sbatch_args=(
  --export=ALL
  --nodes="${NUM_ACTOR_NODES}"
  --segment="${SEGMENT_SIZE}"
  --account="${ACCOUNT}"
  --job-name="${ACCOUNT}.${RUN_NAME}"
  --partition="${PARTITION}"
  --time="${TIME_LIMIT}"
  --gres="gpu:${GPUS_PER_NODE}"
  --output="${RUN_ROOT}/slurm-%j.out"
  ray.sub
)

printf 'Validating schedule with sbatch --test-only...\n'
sbatch --test-only "${sbatch_args[@]}"

job_id=$(sbatch --parsable "${sbatch_args[@]}")
printf 'job_id=%q\n' "${job_id}" >> "${metadata_path}"
printf 'Submitted %s\nMetadata: %s\n' "${job_id}" "${metadata_path}"

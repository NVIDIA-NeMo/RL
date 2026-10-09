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

# Run external Gym vLLM pools beside NeMo RL in a single, two-component
# Slurm heterogeneous job. Pool definitions are supplied by the caller through
# EXTERNAL_VLLM_POOLS and consistently named, exported environment variables.

set -euo pipefail

# Some HSG-Park compute-node batch environments omit the site Slurm directory
# from PATH even though the clients are installed under /cm/local/apps/slurm.
# Discover it before the first scontrol/srun call and export it to ray.sub.
if ! command -v scontrol >/dev/null 2>&1 || ! command -v srun >/dev/null 2>&1; then
  for slurm_bin_dir in /cm/local/apps/slurm/*/bin; do
    if [[ -x "${slurm_bin_dir}/scontrol" && -x "${slurm_bin_dir}/srun" ]]; then
      export PATH="${slurm_bin_dir}:${PATH}"
      break
    fi
  done
fi
for slurm_command in scontrol srun; do
  if ! command -v "${slurm_command}" >/dev/null 2>&1; then
    echo "[FATAL] ${slurm_command} is not available in the batch environment" >&2
    exit 1
  fi
done
unset slurm_bin_dir slurm_command

: "${SLURM_JOB_ID:?This script must run inside a Slurm allocation}"
: "${SLURM_HET_SIZE:?This script requires a Slurm heterogeneous job}"
: "${SLURM_JOB_NODELIST_HET_GROUP_0:?Hetgroup 0 nodelist is required}"
: "${SLURM_JOB_NODELIST_HET_GROUP_1:?Hetgroup 1 nodelist is required}"
: "${SLURM_JOB_ACCOUNT:?SLURM_JOB_ACCOUNT is required}"
: "${SLURM_JOB_PARTITION:?SLURM_JOB_PARTITION is required}"
: "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
: "${BASE_LOG_DIR:?BASE_LOG_DIR is required}"
: "${CONTAINER:?CONTAINER is required}"
: "${MOUNTS:?MOUNTS is required}"
: "${COMMAND:?COMMAND is required}"
: "${EXTERNAL_VLLM_POOLS:?EXTERNAL_VLLM_POOLS is required}"
: "${EXTERNAL_VLLM_TOOLS_DIR_HOST:?EXTERNAL_VLLM_TOOLS_DIR_HOST is required}"

if [[ "${SLURM_HET_SIZE}" != "2" ]]; then
  echo "[FATAL] Expected exactly two Slurm hetgroups, got ${SLURM_HET_SIZE}" >&2
  exit 1
fi

RAY_SUB="${RAY_SUB:-${SLURM_SUBMIT_DIR}/ray.sub}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
EXTERNAL_VLLM_LB_PYTHON="${EXTERNAL_VLLM_LB_PYTHON:-/opt/nemo_rl_venv/bin/python}"
EXTERNAL_VLLM_SHARED_ROOT="${EXTERNAL_VLLM_SHARED_ROOT:-/lustre}"

if [[ ! -f "${RAY_SUB}" ]]; then
  echo "[FATAL] ray.sub does not exist: ${RAY_SUB}" >&2
  exit 1
fi
for required_file in vllm_backend_registry.sh vllm_pool_lb.py lb_watchdog.sh serve_vllm_on_ray.py; do
  if [[ ! -f "${EXTERNAL_VLLM_TOOLS_DIR_HOST}/${required_file}" ]]; then
    echo "[FATAL] Missing ${EXTERNAL_VLLM_TOOLS_DIR_HOST}/${required_file}" >&2
    exit 1
  fi
done
EXTERNAL_VLLM_ROUTER_POOL="${EXTERNAL_VLLM_ROUTER_POOL:-}"
if [[ -n "${VLLM_ROUTER_WHEEL:-}" && -n "${VLLM_ROUTER_SITE_PACKAGES:-}" ]]; then
  echo "[FATAL] Set only one of VLLM_ROUTER_WHEEL and VLLM_ROUTER_SITE_PACKAGES" >&2
  exit 1
fi
if [[ -n "${EXTERNAL_VLLM_ROUTER_POOL}" ]] &&
  [[ ! -f "${EXTERNAL_VLLM_TOOLS_DIR_HOST}/vllm_router_from_registry.sh" ]]; then
  echo "[FATAL] Missing ${EXTERNAL_VLLM_TOOLS_DIR_HOST}/vllm_router_from_registry.sh" >&2
  exit 1
fi
if [[ ! "${GPUS_PER_NODE}" =~ ^[0-9]+$ ]] || (( GPUS_PER_NODE <= 0 )); then
  echo "[FATAL] GPUS_PER_NODE must be a positive integer" >&2
  exit 1
fi
if [[ "${EXTERNAL_VLLM_SHARED_ROOT}" != /* ]]; then
  echo "[FATAL] EXTERNAL_VLLM_SHARED_ROOT must be absolute" >&2
  exit 1
fi

read -r -a pool_names <<< "${EXTERNAL_VLLM_POOLS}"
if (( ${#pool_names[@]} == 0 )); then
  echo "[FATAL] EXTERNAL_VLLM_POOLS must name at least one pool" >&2
  exit 1
fi

pool_value() {
  local pool="$1"
  local suffix="$2"
  local default_value="${3-}"
  local variable_name="${pool}_${suffix}"
  printf '%s' "${!variable_name-${default_value}}"
}

require_pool_value() {
  local pool="$1"
  local suffix="$2"
  local value
  value=$(pool_value "${pool}" "${suffix}")
  if [[ -z "${value}" ]]; then
    echo "[FATAL] ${pool}_${suffix} is required" >&2
    return 1
  fi
  printf '%s' "${value}"
}

declare -A display_names=()
declare -A models=()
declare -A containers=()
declare -A vllm_pythons=()
declare -A launch_modes=()
declare -A vllm_executables=()
declare -A replicas=()
declare -A tensor_parallel_sizes=()
declare -A data_parallel_sizes=()
declare -A replicas_per_node=()
declare -A prefill_replicas=()
declare -A nodes_per_replica=()
declare -A node_offsets=()
declare -A node_counts=()
declare -A served_model_names=()
declare -A backend_ports=()
declare -A lb_ports=()
declare -A control_lb_ports=()
declare -A startup_timeouts=()
declare -A placeholders=()
declare -A control_placeholders=()
declare -A group_ids=()
declare -A pool_log_dirs=()
declare -A state_dirs=()
declare -A lb_state_dirs=()
declare -A pool_urls=()
declare -A control_pool_urls=()

total_external_nodes=0
max_startup_timeout=0
declare -A seen_pool_names=()
declare -A seen_lb_ports=()
declare -A seen_placeholders=()

for pool in "${pool_names[@]}"; do
  if [[ ! "${pool}" =~ ^[A-Z][A-Z0-9_]*$ ]]; then
    echo "[FATAL] Invalid pool name '${pool}'; use uppercase shell identifiers" >&2
    exit 1
  fi
  if [[ -n "${seen_pool_names[${pool}]-}" ]]; then
    echo "[FATAL] Duplicate external vLLM pool: ${pool}" >&2
    exit 1
  fi
  seen_pool_names["${pool}"]=1

  display_names["${pool}"]=$(pool_value "${pool}" DISPLAY_NAME "${pool}")
  models["${pool}"]=$(require_pool_value "${pool}" MODEL)
  containers["${pool}"]=$(require_pool_value "${pool}" CONTAINER)
  launch_modes["${pool}"]=$(pool_value "${pool}" LAUNCH_MODE nemo-rl-ray)
  vllm_pythons["${pool}"]=$(pool_value "${pool}" VLLM_PYTHON)
  vllm_executables["${pool}"]=$(pool_value "${pool}" VLLM_EXECUTABLE vllm)
  replicas["${pool}"]=$(require_pool_value "${pool}" REPLICAS)
  tensor_parallel_sizes["${pool}"]=$(require_pool_value "${pool}" TENSOR_PARALLEL_SIZE)
  data_parallel_sizes["${pool}"]=$(pool_value "${pool}" DATA_PARALLEL_SIZE 1)
  replicas_per_node["${pool}"]=$(pool_value "${pool}" REPLICAS_PER_NODE 1)
  prefill_replicas["${pool}"]=$(pool_value "${pool}" PREFILL_REPLICAS 0)
  served_model_names["${pool}"]=$(pool_value "${pool}" SERVED_MODEL_NAME model)
  backend_ports["${pool}"]=$(pool_value "${pool}" VLLM_PORT 8000)
  lb_ports["${pool}"]=$(require_pool_value "${pool}" LB_PORT)
  control_lb_ports["${pool}"]=$(pool_value "${pool}" CONTROL_LB_PORT)
  startup_timeouts["${pool}"]=$(pool_value "${pool}" STARTUP_TIMEOUT 3600)
  placeholders["${pool}"]=$(require_pool_value "${pool}" URL_PLACEHOLDER)
  control_placeholders["${pool}"]=$(pool_value "${pool}" CONTROL_URL_PLACEHOLDER)
  if { [[ -n "${control_lb_ports[${pool}]}" ]] && [[ -z "${control_placeholders[${pool}]}" ]]; } ||
    { [[ -z "${control_lb_ports[${pool}]}" ]] && [[ -n "${control_placeholders[${pool}]}" ]]; }; then
    echo "[FATAL] ${pool} must set both control endpoint fields" >&2
    exit 1
  fi
  group_ids["${pool}"]=$(pool_value "${pool}" GROUP_ID "inline-${pool,,}-${SLURM_JOB_ID}")
  if [[ ! "${group_ids[${pool}]}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "[FATAL] ${pool}_GROUP_ID may contain only letters, digits, '.', '_', and '-'" >&2
    exit 1
  fi

  # srun inherits the pool contract through exported variables. Re-export the
  # normalized values so optional defaults are also visible inside replicas.
  export "${pool}_DISPLAY_NAME=${display_names[${pool}]}"
  export "${pool}_MODEL=${models[${pool}]}"
  export "${pool}_CONTAINER=${containers[${pool}]}"
  export "${pool}_VLLM_PYTHON=${vllm_pythons[${pool}]}"
  export "${pool}_LAUNCH_MODE=${launch_modes[${pool}]}"
  export "${pool}_VLLM_EXECUTABLE=${vllm_executables[${pool}]}"
  export "${pool}_REPLICAS=${replicas[${pool}]}"
  export "${pool}_TENSOR_PARALLEL_SIZE=${tensor_parallel_sizes[${pool}]}"
  export "${pool}_DATA_PARALLEL_SIZE=${data_parallel_sizes[${pool}]}"
  export "${pool}_REPLICAS_PER_NODE=${replicas_per_node[${pool}]}"
  export "${pool}_PREFILL_REPLICAS=${prefill_replicas[${pool}]}"
  export "${pool}_SERVED_MODEL_NAME=${served_model_names[${pool}]}"
  export "${pool}_VLLM_PORT=${backend_ports[${pool}]}"
  export "${pool}_CONTROL_LB_PORT=${control_lb_ports[${pool}]}"
  export "${pool}_CONTROL_URL_PLACEHOLDER=${control_placeholders[${pool}]}"
  export "${pool}_ENV_VARS=$(pool_value "${pool}" ENV_VARS)"
  export "${pool}_VLLM_ARGS=$(pool_value "${pool}" VLLM_ARGS)"

  for numeric_suffix in REPLICAS TENSOR_PARALLEL_SIZE DATA_PARALLEL_SIZE REPLICAS_PER_NODE VLLM_PORT LB_PORT STARTUP_TIMEOUT; do
    case "${numeric_suffix}" in
      REPLICAS) numeric_value="${replicas[${pool}]}" ;;
      TENSOR_PARALLEL_SIZE) numeric_value="${tensor_parallel_sizes[${pool}]}" ;;
      DATA_PARALLEL_SIZE) numeric_value="${data_parallel_sizes[${pool}]}" ;;
      REPLICAS_PER_NODE) numeric_value="${replicas_per_node[${pool}]}" ;;
      VLLM_PORT) numeric_value="${backend_ports[${pool}]}" ;;
      LB_PORT) numeric_value="${lb_ports[${pool}]}" ;;
      STARTUP_TIMEOUT) numeric_value="${startup_timeouts[${pool}]}" ;;
    esac
    if [[ ! "${numeric_value}" =~ ^[0-9]+$ ]] || (( numeric_value <= 0 )); then
      echo "[FATAL] ${pool}_${numeric_suffix} must be a positive integer" >&2
      exit 1
    fi
    if [[ "${numeric_suffix}" == "VLLM_PORT" || "${numeric_suffix}" == "LB_PORT" ]] && (( numeric_value > 65535 )); then
      echo "[FATAL] ${pool}_${numeric_suffix} must be at most 65535" >&2
      exit 1
    fi
  done
  case "${launch_modes[${pool}]}" in
    nemo-rl-ray)
      if [[ -z "${vllm_pythons[${pool}]}" ]]; then
        echo "[FATAL] ${pool}_VLLM_PYTHON is required in nemo-rl-ray mode" >&2
        exit 1
      fi
      ;;
    native)
      if [[ -z "${vllm_executables[${pool}]}" ]]; then
        echo "[FATAL] ${pool}_VLLM_EXECUTABLE is required in native mode" >&2
        exit 1
      fi
      if (( data_parallel_sizes[${pool}] != 1 )); then
        echo "[FATAL] ${pool} native mode requires data parallel size 1" >&2
        exit 1
      fi
      if (( tensor_parallel_sizes[${pool}] * replicas_per_node[${pool}] > GPUS_PER_NODE )); then
        echo "[FATAL] ${pool} native replicas require more than GPUS_PER_NODE=${GPUS_PER_NODE}" >&2
        exit 1
      fi
      if (( GPUS_PER_NODE % tensor_parallel_sizes[${pool}] != 0 )); then
        echo "[FATAL] ${pool}_TENSOR_PARALLEL_SIZE must divide GPUS_PER_NODE in native mode" >&2
        exit 1
      fi
      ;;
    *)
      echo "[FATAL] ${pool}_LAUNCH_MODE must be 'nemo-rl-ray' or 'native'" >&2
      exit 1
      ;;
  esac
  if [[ ! "${prefill_replicas[${pool}]}" =~ ^[0-9]+$ ]]; then
    echo "[FATAL] ${pool}_PREFILL_REPLICAS must be a nonnegative integer" >&2
    exit 1
  fi
  if (( prefill_replicas[${pool}] > 0 )); then
    if (( prefill_replicas[${pool}] >= replicas[${pool}] )); then
      echo "[FATAL] ${pool}_PREFILL_REPLICAS must be less than ${pool}_REPLICAS" >&2
      exit 1
    fi
    if (( data_parallel_sizes[${pool}] != 1 )); then
      echo "[FATAL] ${pool}_DATA_PARALLEL_SIZE must be 1 with prefill/decode disaggregation" >&2
      exit 1
    fi
  fi
  if [[ "${launch_modes[${pool}]}" != "native" ]] &&
    (( replicas_per_node[${pool}] != 1 )); then
    echo "[FATAL] ${pool}_REPLICAS_PER_NODE is supported only in native mode" >&2
    exit 1
  fi
  if [[ "${launch_modes[${pool}]}" != "native" ]] &&
    (( tensor_parallel_sizes[${pool}] % GPUS_PER_NODE != 0 )); then
    echo "[FATAL] ${pool}_TENSOR_PARALLEL_SIZE must be divisible by GPUS_PER_NODE" >&2
    exit 1
  fi
  if [[ -n "${seen_lb_ports[${lb_ports[${pool}]}]-}" ]]; then
    echo "[FATAL] Multiple pools use load-balancer port ${lb_ports[${pool}]}" >&2
    exit 1
  fi
  seen_lb_ports["${lb_ports[${pool}]}"]="${pool}"
  if [[ -n "${control_lb_ports[${pool}]}" ]]; then
    if [[ ! "${control_lb_ports[${pool}]}" =~ ^[0-9]+$ ]] ||
      (( control_lb_ports[${pool}] <= 0 || control_lb_ports[${pool}] > 65535 )); then
      echo "[FATAL] ${pool}_CONTROL_LB_PORT must be a valid TCP port" >&2
      exit 1
    fi
    if [[ -n "${seen_lb_ports[${control_lb_ports[${pool}]}]-}" ]]; then
      echo "[FATAL] Multiple endpoints use port ${control_lb_ports[${pool}]}" >&2
      exit 1
    fi
    seen_lb_ports["${control_lb_ports[${pool}]}"]="${pool}-control"
  fi
  if [[ -n "${seen_placeholders[${placeholders[${pool}]}]-}" ]]; then
    echo "[FATAL] Multiple pools use URL placeholder ${placeholders[${pool}]}" >&2
    exit 1
  fi
  seen_placeholders["${placeholders[${pool}]}"]="${pool}"
  if [[ "${COMMAND}" != *"${placeholders[${pool}]}"* ]]; then
    echo "[FATAL] Driver command is missing ${placeholders[${pool}]} for ${display_names[${pool}]}" >&2
    exit 1
  fi
  if [[ -n "${control_placeholders[${pool}]}" ]]; then
    if [[ -n "${seen_placeholders[${control_placeholders[${pool}]}]-}" ]]; then
      echo "[FATAL] Multiple pools use URL placeholder ${control_placeholders[${pool}]}" >&2
      exit 1
    fi
    if [[ "${COMMAND}" != *"${control_placeholders[${pool}]}"* ]]; then
      echo "[FATAL] Driver command is missing ${control_placeholders[${pool}]} for ${display_names[${pool}]}" >&2
      exit 1
    fi
    seen_placeholders["${control_placeholders[${pool}]}"]="${pool}-control"
  fi

  if [[ "${launch_modes[${pool}]}" == "native" ]]; then
    nodes_per_replica["${pool}"]=1
    node_counts["${pool}"]=$(((replicas[${pool}] + replicas_per_node[${pool}] - 1) / replicas_per_node[${pool}]))
  else
    # Each private Ray cluster owns whole nodes. This makes its fixed Ray port
    # safe to reuse across replicas because no two replicas share a host.
    nodes_per_replica["${pool}"]=$((tensor_parallel_sizes[${pool}] * data_parallel_sizes[${pool}] / GPUS_PER_NODE))
    node_counts["${pool}"]=$((replicas[${pool}] * nodes_per_replica[${pool}]))
  fi
  node_offsets["${pool}"]="${total_external_nodes}"
  total_external_nodes=$((total_external_nodes + node_counts[${pool}]))
  if (( startup_timeouts[${pool}] > max_startup_timeout )); then
    max_startup_timeout="${startup_timeouts[${pool}]}"
  fi
done

if [[ -n "${EXTERNAL_VLLM_ROUTER_POOL}" ]]; then
  if [[ -z "${seen_pool_names[${EXTERNAL_VLLM_ROUTER_POOL}]-}" ]]; then
    echo "[FATAL] EXTERNAL_VLLM_ROUTER_POOL is not registered: ${EXTERNAL_VLLM_ROUTER_POOL}" >&2
    exit 1
  fi
  if [[ -z "${control_lb_ports[${EXTERNAL_VLLM_ROUTER_POOL}]}" ||
    -z "${control_placeholders[${EXTERNAL_VLLM_ROUTER_POOL}]}" ]]; then
    echo "[FATAL] ${EXTERNAL_VLLM_ROUTER_POOL} requires a separate control port and URL placeholder" >&2
    exit 1
  fi
fi
for pool in "${pool_names[@]}"; do
  if [[ -n "${control_lb_ports[${pool}]}" && "${pool}" != "${EXTERNAL_VLLM_ROUTER_POOL}" ]]; then
    echo "[FATAL] Only EXTERNAL_VLLM_ROUTER_POOL may define a separate control endpoint" >&2
    exit 1
  fi
done

shared_paths=("${BASE_LOG_DIR}" "${EXTERNAL_VLLM_TOOLS_DIR_HOST}")
if [[ -n "${VLLM_ROUTER_WHEEL:-}" ]]; then
  shared_paths+=("${VLLM_ROUTER_WHEEL}")
fi
if [[ -n "${VLLM_ROUTER_SITE_PACKAGES:-}" ]]; then
  shared_paths+=("${VLLM_ROUTER_SITE_PACKAGES}")
fi
for pool in "${pool_names[@]}"; do
  if [[ "${models[${pool}]}" == /* ]]; then
    shared_paths+=("${models[${pool}]}")
  fi
  while IFS= read -r shared_path; do
    [[ -n "${shared_path}" ]] && shared_paths+=("${shared_path}")
  done <<< "$(pool_value "${pool}" SHARED_PATHS)"
done
for shared_path in "${shared_paths[@]}"; do
  if [[ "${shared_path}" != "${EXTERNAL_VLLM_SHARED_ROOT}" && "${shared_path}" != "${EXTERNAL_VLLM_SHARED_ROOT}"/* ]]; then
    echo "[FATAL] Path must be under ${EXTERNAL_VLLM_SHARED_ROOT} for the external-service container mount: ${shared_path}" >&2
    exit 1
  fi
done

mapfile -t ray_nodes < <(
  scontrol show hostnames "${SLURM_JOB_NODELIST_HET_GROUP_0}" | sort
)
mapfile -t external_nodes < <(
  scontrol show hostnames "${SLURM_JOB_NODELIST_HET_GROUP_1}" | sort
)
if (( ${#ray_nodes[@]} == 0 )); then
  echo "[FATAL] Slurm hetgroup 0 contains no NeMo RL nodes" >&2
  exit 1
fi
if (( ${#external_nodes[@]} != total_external_nodes )); then
  echo "[FATAL] Slurm hetgroup 1 has ${#external_nodes[@]} nodes, expected ${total_external_nodes}" >&2
  for pool in "${pool_names[@]}"; do
    echo "[FATAL]   ${display_names[${pool}]}: ${node_counts[${pool}]} nodes" >&2
  done
  exit 1
fi

# Must match ray.sub's LOG_DIR: ENDED is the teardown channel between them.
if [[ -n "${SLURM_RESTART_COUNT:-}" ]]; then
  LOG_DIR="${BASE_LOG_DIR}/${SLURM_JOB_ID}-${SLURM_RESTART_COUNT}-logs"
else
  LOG_DIR="${BASE_LOG_DIR}/${SLURM_JOB_ID}-logs"
fi
mkdir -p "${LOG_DIR}"
rm_args=()
for pool in "${pool_names[@]}"; do
  pool_key="${pool,,}"
  pool_log_dirs["${pool}"]="${LOG_DIR}/external_${pool_key}"
  state_dirs["${pool}"]="${pool_log_dirs[${pool}]}/state"
  lb_state_dirs["${pool}"]="/tmp/external-vllm-state-${pool_key}"
  mkdir -p "${pool_log_dirs[${pool}]}" "${state_dirs[${pool}]}"
  rm_args+=(
    "${state_dirs[${pool}]}/.registry_${group_ids[${pool}]}"
    "${state_dirs[${pool}]}/.registry_${group_ids[${pool}]}.lock"
    "${LOG_DIR}/${pool_key}_url"
  )
done
rm -f "${rm_args[@]}"
for pool in "${pool_names[@]}"; do
  rm -f "${pool_log_dirs[${pool}]}"/head_ip_*
done

{
  for pool in "${pool_names[@]}"; do
    echo "[external_${pool,,}]"
    offset="${node_offsets[${pool}]}"
    count="${node_counts[${pool}]}"
    printf '%s\n' "${external_nodes[@]:offset:count}"
  done
  echo "[nemo_rl_ray]"
  printf '%s\n' "${ray_nodes[@]}"
} > "${LOG_DIR}/node-allocation.txt"

echo "[INFO] Heterogeneous-job external-vLLM topology"
echo "[INFO]   Hetgroup 0, NeMo RL Ray: ${#ray_nodes[@]} nodes (${SLURM_JOB_NODELIST_HET_GROUP_0})"
for pool in "${pool_names[@]}"; do
  echo "[INFO]   Hetgroup 1, ${display_names[${pool}]}: ${node_counts[${pool}]} nodes, ${replicas[${pool}]} TP=${tensor_parallel_sizes[${pool}]}/DP=${data_parallel_sizes[${pool}]} replicas"
done

declare -a service_step_pids=()
declare -a service_step_labels=()
declare -a lb_step_pids=()
declare -a lb_step_labels=()
ray_sub_pid=""

cleanup() {
  local status=$?
  trap - EXIT TERM INT

  touch "${LOG_DIR}/ENDED" 2>/dev/null || true
  if [[ -n "${ray_sub_pid}" ]] && kill -0 "${ray_sub_pid}" 2>/dev/null; then
    kill "${ray_sub_pid}" 2>/dev/null || true
  fi
  for pid in "${lb_step_pids[@]}" "${service_step_pids[@]}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done

  wait 2>/dev/null || true
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 143' TERM INT

check_service_steps() {
  local index
  for index in "${!service_step_pids[@]}"; do
    if ! kill -0 "${service_step_pids[${index}]}" 2>/dev/null; then
      echo "[FATAL] ${service_step_labels[${index}]} exited unexpectedly" >&2
      return 1
    fi
  done
  for index in "${!lb_step_pids[@]}"; do
    if ! kill -0 "${lb_step_pids[${index}]}" 2>/dev/null; then
      echo "[FATAL] ${lb_step_labels[${index}]} exited unexpectedly" >&2
      return 1
    fi
  done
}

resolve_node_ip() {
  local node="$1" ip
  ip=$(getent ahostsv4 "${node}" 2>/dev/null | awk 'NR == 1 { print $1 }' || true)
  if [[ -z "${ip}" ]]; then
    ip=$(host "${node}" 2>/dev/null | awk '/has address/ { print $4; exit }' || true)
  fi
  if [[ -z "${ip}" ]]; then
    echo "[FATAL] Could not resolve an IPv4 address for ${node}" >&2
    return 1
  fi
  echo "${ip}"
}

VLLM_SERVER_BODY=$(cat <<'VLLM_SERVER_BODY_EOF'
set -euo pipefail

: "${POOL_PREFIX:?POOL_PREFIX is required}"
: "${REPLICA_ID:?REPLICA_ID is required}"
: "${REPLICA_INDEX:?REPLICA_INDEX is required}"
: "${EXTERNAL_VLLM_TOOLS_DIR:?EXTERNAL_VLLM_TOOLS_DIR is required}"
: "${EXTERNAL_VLLM_STATE_DIR:?EXTERNAL_VLLM_STATE_DIR is required}"
: "${EXTERNAL_VLLM_GROUP_ID:?EXTERNAL_VLLM_GROUP_ID is required}"
: "${HEAD_IP_FILE:?HEAD_IP_FILE is required}"
: "${LOG_FILE:?LOG_FILE is required}"

pool_value() {
  local variable_name="${POOL_PREFIX}_$1"
  printf '%s' "${!variable_name-}"
}

MODEL=$(pool_value MODEL)
VLLM_PYTHON=$(pool_value VLLM_PYTHON)
LAUNCH_MODE=$(pool_value LAUNCH_MODE)
VLLM_EXECUTABLE=$(pool_value VLLM_EXECUTABLE)
VLLM_HTTP_PORT=$(pool_value VLLM_PORT)
TENSOR_PARALLEL_SIZE=$(pool_value TENSOR_PARALLEL_SIZE)
DATA_PARALLEL_SIZE=$(pool_value DATA_PARALLEL_SIZE)
PREFILL_REPLICAS=$(pool_value PREFILL_REPLICAS)
SERVED_MODEL_NAME=$(pool_value SERVED_MODEL_NAME)
DISPLAY_NAME=$(pool_value DISPLAY_NAME)
REPLICA_LOCAL_SLOT="${REPLICA_LOCAL_SLOT:-0}"
if [[ "${LAUNCH_MODE}" == "native" ]]; then
  VLLM_HTTP_PORT=$((VLLM_HTTP_PORT + REPLICA_LOCAL_SLOT))
fi
[[ -n "${SERVED_MODEL_NAME}" ]] || SERVED_MODEL_NAME=model
[[ -n "${DISPLAY_NAME}" ]] || DISPLAY_NAME="${POOL_PREFIX}"
[[ -n "${DATA_PARALLEL_SIZE}" ]] || DATA_PARALLEL_SIZE=1
[[ -n "${PREFILL_REPLICAS}" ]] || PREFILL_REPLICAS=0
[[ -n "${LAUNCH_MODE}" ]] || LAUNCH_MODE=nemo-rl-ray
[[ -n "${VLLM_EXECUTABLE}" ]] || VLLM_EXECUTABLE=vllm

BACKEND_ROLE=standard
if (( PREFILL_REPLICAS > 0 )); then
  if (( REPLICA_INDEX < PREFILL_REPLICAS )); then
    BACKEND_ROLE=prefill
  else
    BACKEND_ROLE=decode
  fi
fi

source "${EXTERNAL_VLLM_TOOLS_DIR}/vllm_backend_registry.sh"

# Match ray.sub's sub-ephemeral port layout. Some GB200 nodes use
# 9000-65000 for ephemeral ports, so Ray's former 10002-19999 worker range
# could race vLLM's bind-probe-and-release TCPStore allocation. Every replica
# owns disjoint nodes, so these fixed ports can be reused across replicas.
RAY_PORT=1200
RAY_CLIENT_SERVER_PORT=1201
NODE_MANAGER_PORT=1301
OBJECT_MANAGER_PORT=1303
RUNTIME_ENV_AGENT_PORT=1305
DASHBOARD_AGENT_GRPC_PORT=1307
METRICS_EXPORT_PORT=1309
DASHBOARD_AGENT_LISTEN_PORT=1311
MIN_WORKER_PORT=2000
MAX_WORKER_PORT=2999
VLLM_ENGINE_PORT=$((7000 + REPLICA_LOCAL_SLOT * 100))

cleanup_replica() {
  if [[ "${SLURM_PROCID:-0}" -eq 0 ]]; then
    registry_remove "${REPLICA_ID}" || true
  fi
  if [[ "${LAUNCH_MODE}" == "nemo-rl-ray" ]]; then
    ray stop 2>/dev/null || true
  fi
}
trap cleanup_replica EXIT
trap 'trap - EXIT; cleanup_replica; exit 143' TERM INT

if [[ "${SLURM_PROCID:-0}" -eq 0 ]]; then
  rm -f "${HEAD_IP_FILE}"
  HEAD_IP=$(hostname -I | awk '{ print $1 }')
  if [[ -z "${HEAD_IP}" ]]; then
    HEAD_IP=$(getent ahostsv4 "$(hostname)" | awk 'NR == 1 { print $1 }')
  fi
  if [[ -z "${HEAD_IP}" ]]; then
    echo "[${REPLICA_ID}] ERROR: could not determine the head-node IP" >&2
    exit 1
  fi

  echo "${HEAD_IP}" > "${HEAD_IP_FILE}"
  if [[ "${LAUNCH_MODE}" == "nemo-rl-ray" ]]; then
    echo "[${REPLICA_ID}] Starting private Ray head at ${HEAD_IP}:${RAY_PORT}"
    ray start \
      --head \
      --node-ip-address="${HEAD_IP}" \
      --port="${RAY_PORT}" \
      --ray-client-server-port="${RAY_CLIENT_SERVER_PORT}" \
      --min-worker-port="${MIN_WORKER_PORT}" \
      --max-worker-port="${MAX_WORKER_PORT}" \
      --node-manager-port="$((NODE_MANAGER_PORT + 1))" \
      --object-manager-port="$((OBJECT_MANAGER_PORT + 1))" \
      --runtime-env-agent-port="$((RUNTIME_ENV_AGENT_PORT + 1))" \
      --dashboard-agent-grpc-port="$((DASHBOARD_AGENT_GRPC_PORT + 1))" \
      --dashboard-agent-listen-port="$((DASHBOARD_AGENT_LISTEN_PORT + 1))" \
      --metrics-export-port="$((METRICS_EXPORT_PORT + 1))" \
      --disable-usage-stats
  fi

  while IFS= read -r assignment; do
    [[ -n "${assignment}" ]] || continue
    variable_name="${assignment%%=*}"
    if [[ ! "${variable_name}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || [[ "${assignment}" != *=* ]]; then
      echo "[${REPLICA_ID}] ERROR: invalid environment assignment: ${assignment}" >&2
      exit 1
    fi
    export "${assignment}"
  done <<< "$(pool_value ENV_VARS)"

  # Keep vLLM's TCPStore and MessageQueue ports inside the reserved
  # 7000-7999 band. serve_vllm_on_ray.py applies the NeMo RL compatibility
  # patch that offsets the TCPStore search within this per-engine window.
  export VLLM_PORT="${VLLM_ENGINE_PORT}"

  vllm_args=()
  while IFS= read -r argument; do
    [[ -n "${argument}" ]] && vllm_args+=("${argument}")
  done <<< "$(pool_value VLLM_ARGS)"

  data_parallel_args=()
  if (( DATA_PARALLEL_SIZE > 1 )); then
    data_parallel_args+=(
      --data-parallel-size "${DATA_PARALLEL_SIZE}"
      --data-parallel-size-local 1
      --data-parallel-backend ray
      --api-server-count 1
    )
  fi

  kv_transfer_args=()
  if [[ "${BACKEND_ROLE}" == "prefill" ]]; then
    export VLLM_NIXL_SIDE_CHANNEL_HOST="${VLLM_NIXL_SIDE_CHANNEL_HOST:-${HEAD_IP}}"
    export VLLM_NIXL_SIDE_CHANNEL_PORT="${VLLM_NIXL_SIDE_CHANNEL_PORT:-5557}"
    export UCX_NET_DEVICES="${UCX_NET_DEVICES:-all}"
    kv_transfer_args+=(
      --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}'
    )
  elif [[ "${BACKEND_ROLE}" == "decode" ]]; then
    export VLLM_NIXL_SIDE_CHANNEL_HOST="${VLLM_NIXL_SIDE_CHANNEL_HOST:-${HEAD_IP}}"
    export VLLM_NIXL_SIDE_CHANNEL_PORT="${VLLM_NIXL_SIDE_CHANNEL_PORT:-5557}"
    export UCX_NET_DEVICES="${UCX_NET_DEVICES:-all}"
    kv_transfer_args+=(
      --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}'
    )
  fi

  server_command=()
  distributed_backend=ray
  if [[ "${LAUNCH_MODE}" == "native" ]]; then
    server_command=("${VLLM_EXECUTABLE}" serve)
    distributed_backend=mp
  else
    server_command=("${VLLM_PYTHON}" "${EXTERNAL_VLLM_TOOLS_DIR}/serve_vllm_on_ray.py" serve)
  fi

  echo "[${REPLICA_ID}] Starting ${DISPLAY_NAME} ${BACKEND_ROLE} vLLM server with ${LAUNCH_MODE} at TP=${TENSOR_PARALLEL_SIZE}/DP=${DATA_PARALLEL_SIZE}"
  "${server_command[@]}" "${MODEL}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --distributed-executor-backend "${distributed_backend}" \
    "${data_parallel_args[@]}" \
    "${kv_transfer_args[@]}" \
    --port "${VLLM_HTTP_PORT}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    "${vllm_args[@]}" \
    > "${LOG_FILE}" 2>&1 &
  VLLM_PID=$!

  health_url="http://${HEAD_IP}:${VLLM_HTTP_PORT}/health"
  server_is_healthy() {
    if [[ "${LAUNCH_MODE}" == "native" ]]; then
      curl --fail --silent --show-error --max-time 2 "${health_url}" >/dev/null
    else
      "${VLLM_PYTHON}" -c \
        'import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=2).close()' \
        "${health_url}" >/dev/null
    fi
  }
  while ! server_is_healthy 2>/dev/null; do
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
      echo "[${REPLICA_ID}] ERROR: vLLM exited before becoming healthy" >&2
      exit 1
    fi
    sleep 5
  done

  if [[ "${NEMO_RL_VLLM_PREFIX_PLUGIN_REQUIRED:-0}" == "1" ]]; then
    capability_path="${NEMO_RL_VLLM_PREFIX_CAPABILITY_PATH:-/v1/nemo-rl/prefix-token-capability}"
    capability_url="http://${HEAD_IP}:${VLLM_HTTP_PORT}${capability_path}"
    if [[ "${LAUNCH_MODE}" == "native" ]]; then
      capability_json=$(curl --fail --silent --show-error --max-time 5 "${capability_url}")
    else
      capability_json=$("${VLLM_PYTHON}" -c \
        'import sys, urllib.request; print(urllib.request.urlopen(sys.argv[1], timeout=5).read().decode())' \
        "${capability_url}")
    fi
    if ! printf '%s' "${capability_json}" | grep -Eq \
      '"active"[[:space:]]*:[[:space:]]*true' \
      || ! printf '%s' "${capability_json}" | grep -Eq \
        '"required_prefix_token_ids"[[:space:]]*:[[:space:]]*true' \
      || ! printf '%s' "${capability_json}" | grep -Eq \
        '"ng_capture"[[:space:]]*:[[:space:]]*true' \
      || ! printf '%s' "${capability_json}" | grep -Eq \
        '"external_staging"[[:space:]]*:[[:space:]]*true' \
      || ! printf '%s' "${capability_json}" | grep -Eq \
        '"stock_chat_routes_replaced"[[:space:]]*:[[:space:]]*[1-9][0-9]*'; then
      echo "[${REPLICA_ID}] ERROR: incomplete NeMo RL token-capture capability: ${capability_json}" >&2
      exit 1
    fi
    echo "[${REPLICA_ID}] Verified NeMo RL token-capture API capability"
  fi

  registry_add "${REPLICA_ID}" "${HEAD_IP}" "${VLLM_HTTP_PORT}" "${BACKEND_ROLE}"
  echo "[${REPLICA_ID}] Registered healthy ${BACKEND_ROLE} backend ${HEAD_IP}:${VLLM_HTTP_PORT}"
  if wait "${VLLM_PID}"; then
    vllm_status=0
  else
    vllm_status=$?
  fi
  echo "[${REPLICA_ID}] ERROR: vLLM exited with status ${vllm_status}" >&2
  if (( vllm_status == 0 )); then
    exit 1
  fi
  exit "${vllm_status}"
else
  if [[ "${LAUNCH_MODE}" == "native" ]]; then
    echo "[${REPLICA_ID}] ERROR: native mode does not support multi-node replicas" >&2
    exit 1
  fi
  for _ in $(seq 1 120); do
    [[ -s "${HEAD_IP_FILE}" ]] && break
    sleep 1
  done
  if [[ ! -s "${HEAD_IP_FILE}" ]]; then
    echo "[${REPLICA_ID}] ERROR: private Ray head IP was not published" >&2
    exit 1
  fi

  HEAD_IP=$(cat "${HEAD_IP_FILE}")
  joined=0
  for _ in $(seq 1 120); do
    if ray start \
      --address="${HEAD_IP}:${RAY_PORT}" \
      --min-worker-port="${MIN_WORKER_PORT}" \
      --max-worker-port="${MAX_WORKER_PORT}" \
      --node-manager-port="${NODE_MANAGER_PORT}" \
      --object-manager-port="${OBJECT_MANAGER_PORT}" \
      --runtime-env-agent-port="${RUNTIME_ENV_AGENT_PORT}" \
      --dashboard-agent-grpc-port="${DASHBOARD_AGENT_GRPC_PORT}" \
      --dashboard-agent-listen-port="${DASHBOARD_AGENT_LISTEN_PORT}" \
      --metrics-export-port="${METRICS_EXPORT_PORT}" \
      --disable-usage-stats; then
      joined=1
      break
    fi
    sleep 2
  done
  if (( joined == 0 )); then
    echo "[${REPLICA_ID}] ERROR: failed to join private Ray cluster" >&2
    exit 1
  fi
  tail -f /dev/null
fi
VLLM_SERVER_BODY_EOF
)
bash -n <(printf '%s' "${VLLM_SERVER_BODY}") || {
  echo "[FATAL] Generated vLLM server script has syntax errors" >&2
  exit 1
}

ray_head_node="${ray_nodes[0]}"
lb_mounts="${MOUNTS},${EXTERNAL_VLLM_TOOLS_DIR_HOST}:/opt/external-vllm-tools:ro"
external_service_mount="${EXTERNAL_VLLM_SHARED_ROOT}:${EXTERNAL_VLLM_SHARED_ROOT}"
for pool in "${pool_names[@]}"; do
  lb_mounts+=",${state_dirs[${pool}]}:${lb_state_dirs[${pool}]}"
done

for pool in "${pool_names[@]}"; do
  echo "[INFO] Launching ${display_names[${pool}]} replicas"
  for (( replica_index = 0; replica_index < replicas[${pool}]; replica_index++ )); do
    if [[ "${launch_modes[${pool}]}" == "native" ]]; then
      first_node_index=$((node_offsets[${pool}] + replica_index / replicas_per_node[${pool}]))
      replica_local_slot=$((replica_index % replicas_per_node[${pool}]))
    else
      first_node_index=$((node_offsets[${pool}] + replica_index * nodes_per_replica[${pool}]))
      replica_local_slot=0
    fi
    replica_node_count="${nodes_per_replica[${pool}]}"
    replica_nodes=("${external_nodes[@]:first_node_index:replica_node_count}")
    replica_nodelist=$(IFS=,; echo "${replica_nodes[*]}")
    replica_id="${SLURM_JOB_ID}-${pool,,}-${replica_index}"
    head_ip_file="${pool_log_dirs[${pool}]}/head_ip_${replica_index}"
    vllm_log="${pool_log_dirs[${pool}]}/vllm_${replica_index}.log"

    echo "[INFO] ${display_names[${pool}]} replica ${replica_index}: ${replica_nodelist}"
    srun_resource_args=(--overlap --gres="gpu:${GPUS_PER_NODE}")
    if (( replicas_per_node[${pool}] > 1 )); then
      # --exact keeps persistent packed steps from reserving every CPU on the
      # node; Slurm assigns each step a disjoint TP-sized GPU allocation.
      srun_resource_args=(
        --exact
        --cpus-per-task="${VLLM_CPUS_PER_REPLICA:-16}"
        --gpus-per-task="${tensor_parallel_sizes[${pool}]}"
      )
    fi
    srun \
      --het-group=1 \
      --no-container-mount-home \
      --container-image="${containers[${pool}]}" \
      --container-mounts="${external_service_mount}" \
      --mpi=pmix \
      "${srun_resource_args[@]}" \
      --kill-on-bad-exit=1 \
      --nodelist="${replica_nodelist}" \
      --nodes="${nodes_per_replica[${pool}]}" \
      --ntasks="${nodes_per_replica[${pool}]}" \
      --ntasks-per-node=1 \
      --export="ALL,POOL_PREFIX=${pool},REPLICA_ID=${replica_id},REPLICA_INDEX=${replica_index},REPLICA_LOCAL_SLOT=${replica_local_slot},EXTERNAL_VLLM_TOOLS_DIR=${EXTERNAL_VLLM_TOOLS_DIR_HOST},EXTERNAL_VLLM_STATE_DIR=${state_dirs[${pool}]},EXTERNAL_VLLM_GROUP_ID=${group_ids[${pool}]},HEAD_IP_FILE=${head_ip_file},LOG_FILE=${vllm_log}" \
      --output="${pool_log_dirs[${pool}]}/replica_${replica_index}_%t.log" \
      bash -c "${VLLM_SERVER_BODY}" &
    service_step_pids+=("$!")
    service_step_labels+=("${display_names[${pool}]} replica ${replica_index}")
  done
done

ray_head_ip=$(resolve_node_ip "${ray_head_node}")
for pool in "${pool_names[@]}"; do
  lb_mode=load-balance
  if (( prefill_replicas[${pool}] > 0 )); then
    lb_mode=disaggregated-prefill
  fi
  pool_urls["${pool}"]="http://${ray_head_ip}:${lb_ports[${pool}]}/v1"
  proxy_port="${lb_ports[${pool}]}"
  proxy_label="load balancer"
  if [[ "${pool}" == "${EXTERNAL_VLLM_ROUTER_POOL}" ]]; then
    proxy_port="${control_lb_ports[${pool}]}"
    control_pool_urls["${pool}"]="http://${ray_head_ip}:${proxy_port}/v1"
    proxy_label="control fan-out proxy"
    lb_mode=control-fanout
  fi
  echo "[INFO] Starting ${display_names[${pool}]} ${proxy_label} on port ${proxy_port}"
  srun \
    --het-group=0 \
    --no-container-mount-home \
    --container-name="external-vllm-lb-${pool,,}-${SLURM_JOB_ID}" \
    --container-image="${CONTAINER}" \
    --container-mounts="${lb_mounts}" \
    --container-workdir="${SLURM_SUBMIT_DIR}" \
    --mpi=pmix \
    -A "${SLURM_JOB_ACCOUNT}" \
    -p "${SLURM_JOB_PARTITION}" \
    --overlap \
    --nodelist="${ray_head_node}" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=2 \
    --output="${pool_log_dirs[${pool}]}/load_balancer.log" \
    bash -lc "PYTHON='${EXTERNAL_VLLM_LB_PYTHON}' /opt/external-vllm-tools/lb_watchdog.sh '${proxy_port}' '${lb_state_dirs[${pool}]}' '${group_ids[${pool}]}' '${lb_mode}'" &
  lb_step_pids+=("$!")
  lb_step_labels+=("${display_names[${pool}]} ${proxy_label}")

  if [[ "${pool}" == "${EXTERNAL_VLLM_ROUTER_POOL}" ]]; then
    echo "[INFO] Starting ${display_names[${pool}]} Rust router at ${pool_urls[${pool}]}"
    router_mounts="${external_service_mount},${EXTERNAL_VLLM_TOOLS_DIR_HOST}:/opt/external-vllm-tools:ro,${state_dirs[${pool}]}:${lb_state_dirs[${pool}]}"
    srun \
      --het-group=0 \
      --no-container-mount-home \
      --container-name="external-vllm-router-${pool,,}-${SLURM_JOB_ID}" \
      --container-image="${containers[${pool}]}" \
      --container-mounts="${router_mounts}" \
      --container-workdir="${SLURM_SUBMIT_DIR}" \
      --mpi=pmix \
      -A "${SLURM_JOB_ACCOUNT}" \
      -p "${SLURM_JOB_PARTITION}" \
      --overlap \
      --nodelist="${ray_head_node}" \
      --nodes=1 \
      --ntasks=1 \
      --cpus-per-task="${VLLM_ROUTER_CPUS:-8}" \
      --output="${pool_log_dirs[${pool}]}/vllm_router_step.log" \
      --export="ALL,EXTERNAL_VLLM_TOOLS_DIR=/opt/external-vllm-tools" \
      bash /opt/external-vllm-tools/vllm_router_from_registry.sh \
        "${lb_ports[${pool}]}" \
        "${lb_state_dirs[${pool}]}" \
        "${group_ids[${pool}]}" \
        "${replicas[${pool}]}" \
        "${prefill_replicas[${pool}]}" \
        "${startup_timeouts[${pool}]}" \
        "${pool_log_dirs[${pool}]}/vllm_router.log" &
    lb_step_pids+=("$!")
    lb_step_labels+=("${display_names[${pool}]} Rust router")
  fi
done

deadline=$((SECONDS + max_startup_timeout))
while true; do
  all_ready=1
  for pool in "${pool_names[@]}"; do
    if ! ready=$(
      EXTERNAL_VLLM_STATE_DIR="${state_dirs[${pool}]}" \
      EXTERNAL_VLLM_TOOLS_DIR="${EXTERNAL_VLLM_TOOLS_DIR_HOST}" \
      EXTERNAL_VLLM_GROUP_ID="${group_ids[${pool}]}" \
      bash -c 'source "${EXTERNAL_VLLM_TOOLS_DIR}/vllm_backend_registry.sh"; registry_count_ready'
    ); then
      echo "[WARN] Could not read ${display_names[${pool}]} registry; retrying" >&2
      ready=0
    fi
    echo "[INFO] ${display_names[${pool}]} ready: ${ready}/${replicas[${pool}]}"
    if (( ready != replicas[${pool}] )); then
      all_ready=0
    fi
  done
  (( all_ready == 1 )) && break
  check_service_steps
  if (( SECONDS >= deadline )); then
    echo "[FATAL] Timed out waiting for all external vLLM pools" >&2
    exit 1
  fi
  sleep 15
done

for pool in "${pool_names[@]}"; do
  until curl -sfm 10 "${pool_urls[${pool}]}/models" >/dev/null 2>&1; do
    check_service_steps
    if (( SECONDS >= deadline )); then
      echo "[FATAL] ${display_names[${pool}]} load balancer failed its end-to-end /models probe" >&2
      exit 1
    fi
    sleep 5
  done
  echo "${pool_urls[${pool}]}" > "${LOG_DIR}/${pool,,}_url"
  COMMAND="${COMMAND//${placeholders[${pool}]}/${pool_urls[${pool}]}}"
  if [[ -n "${control_placeholders[${pool}]}" ]]; then
    until curl -sfm 10 "${control_pool_urls[${pool}]}/models" >/dev/null 2>&1; do
      check_service_steps
      if (( SECONDS >= deadline )); then
        echo "[FATAL] ${display_names[${pool}]} control proxy failed its /models probe" >&2
        exit 1
      fi
      sleep 5
    done
    echo "${control_pool_urls[${pool}]}" > "${LOG_DIR}/${pool,,}_control_url"
    COMMAND="${COMMAND//${control_placeholders[${pool}]}/${control_pool_urls[${pool}]}}"
  fi
done
export COMMAND

echo "[INFO] External vLLM pools are healthy; starting NeMo RL"
# ray.sub predates hetjobs and consumes the unsuffixed allocation variables.
# Restrict those variables to component 0; srun also defaults to hetgroup 0.
# `env` execs bash directly, so ray_sub_pid is the process that owns its traps.
env \
  SLURM_JOB_NODELIST="${SLURM_JOB_NODELIST_HET_GROUP_0}" \
  SLURM_JOB_NUM_NODES="${#ray_nodes[@]}" \
  bash "${RAY_SUB}" &
ray_sub_pid=$!

while kill -0 "${ray_sub_pid}" 2>/dev/null; do
  if ! check_service_steps; then
    touch "${LOG_DIR}/ENDED"
    kill "${ray_sub_pid}" 2>/dev/null || true
    wait "${ray_sub_pid}" 2>/dev/null || true
    exit 1
  fi
  sleep 5
done

set +e
wait "${ray_sub_pid}"
status=$?
set -e
ray_sub_pid=""
exit "${status}"

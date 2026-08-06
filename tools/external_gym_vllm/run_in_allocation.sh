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
declare -A replicas=()
declare -A tensor_parallel_sizes=()
declare -A nodes_per_replica=()
declare -A node_offsets=()
declare -A node_counts=()
declare -A served_model_names=()
declare -A backend_ports=()
declare -A lb_ports=()
declare -A startup_timeouts=()
declare -A placeholders=()
declare -A group_ids=()
declare -A pool_log_dirs=()
declare -A state_dirs=()
declare -A lb_state_dirs=()
declare -A pool_urls=()

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
  vllm_pythons["${pool}"]=$(require_pool_value "${pool}" VLLM_PYTHON)
  replicas["${pool}"]=$(require_pool_value "${pool}" REPLICAS)
  tensor_parallel_sizes["${pool}"]=$(require_pool_value "${pool}" TENSOR_PARALLEL_SIZE)
  served_model_names["${pool}"]=$(pool_value "${pool}" SERVED_MODEL_NAME model)
  backend_ports["${pool}"]=$(pool_value "${pool}" VLLM_PORT 8000)
  lb_ports["${pool}"]=$(require_pool_value "${pool}" LB_PORT)
  startup_timeouts["${pool}"]=$(pool_value "${pool}" STARTUP_TIMEOUT 3600)
  placeholders["${pool}"]=$(require_pool_value "${pool}" URL_PLACEHOLDER)
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
  export "${pool}_REPLICAS=${replicas[${pool}]}"
  export "${pool}_TENSOR_PARALLEL_SIZE=${tensor_parallel_sizes[${pool}]}"
  export "${pool}_SERVED_MODEL_NAME=${served_model_names[${pool}]}"
  export "${pool}_VLLM_PORT=${backend_ports[${pool}]}"
  export "${pool}_ENV_VARS=$(pool_value "${pool}" ENV_VARS)"
  export "${pool}_VLLM_ARGS=$(pool_value "${pool}" VLLM_ARGS)"

  for numeric_suffix in REPLICAS TENSOR_PARALLEL_SIZE VLLM_PORT LB_PORT STARTUP_TIMEOUT; do
    case "${numeric_suffix}" in
      REPLICAS) numeric_value="${replicas[${pool}]}" ;;
      TENSOR_PARALLEL_SIZE) numeric_value="${tensor_parallel_sizes[${pool}]}" ;;
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
  if (( tensor_parallel_sizes[${pool}] % GPUS_PER_NODE != 0 )); then
    echo "[FATAL] ${pool}_TENSOR_PARALLEL_SIZE must be divisible by GPUS_PER_NODE" >&2
    exit 1
  fi
  if [[ -n "${seen_lb_ports[${lb_ports[${pool}]}]-}" ]]; then
    echo "[FATAL] Multiple pools use load-balancer port ${lb_ports[${pool}]}" >&2
    exit 1
  fi
  seen_lb_ports["${lb_ports[${pool}]}"]="${pool}"
  if [[ -n "${seen_placeholders[${placeholders[${pool}]}]-}" ]]; then
    echo "[FATAL] Multiple pools use URL placeholder ${placeholders[${pool}]}" >&2
    exit 1
  fi
  seen_placeholders["${placeholders[${pool}]}"]="${pool}"
  if [[ "${COMMAND}" != *"${placeholders[${pool}]}"* ]]; then
    echo "[FATAL] Driver command is missing ${placeholders[${pool}]} for ${display_names[${pool}]}" >&2
    exit 1
  fi

  # Each private Ray cluster owns whole nodes. This makes its fixed Ray port safe
  # to reuse across replicas because no two replicas ever share a host.
  nodes_per_replica["${pool}"]=$((tensor_parallel_sizes[${pool}] / GPUS_PER_NODE))
  node_counts["${pool}"]=$((replicas[${pool}] * nodes_per_replica[${pool}]))
  node_offsets["${pool}"]="${total_external_nodes}"
  total_external_nodes=$((total_external_nodes + node_counts[${pool}]))
  if (( startup_timeouts[${pool}] > max_startup_timeout )); then
    max_startup_timeout="${startup_timeouts[${pool}]}"
  fi
done

shared_paths=("${BASE_LOG_DIR}" "${EXTERNAL_VLLM_TOOLS_DIR_HOST}")
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
  echo "[INFO]   Hetgroup 1, ${display_names[${pool}]}: ${node_counts[${pool}]} nodes, ${replicas[${pool}]} TP=${tensor_parallel_sizes[${pool}]} replicas"
done

declare -a service_step_pids=()
declare -a service_step_labels=()
declare -a lb_step_pids=()
declare -a lb_step_labels=()
declare -a preflight_pids=()
declare -a preflight_labels=()
ray_sub_pid=""

cleanup() {
  local status=$?
  trap - EXIT TERM INT

  touch "${LOG_DIR}/ENDED" 2>/dev/null || true
  if [[ -n "${ray_sub_pid}" ]] && kill -0 "${ray_sub_pid}" 2>/dev/null; then
    kill "${ray_sub_pid}" 2>/dev/null || true
  fi
  for pid in "${lb_step_pids[@]}" "${service_step_pids[@]}" "${preflight_pids[@]}"; do
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
VLLM_HTTP_PORT=$(pool_value VLLM_PORT)
TENSOR_PARALLEL_SIZE=$(pool_value TENSOR_PARALLEL_SIZE)
SERVED_MODEL_NAME=$(pool_value SERVED_MODEL_NAME)
DISPLAY_NAME=$(pool_value DISPLAY_NAME)
[[ -n "${SERVED_MODEL_NAME}" ]] || SERVED_MODEL_NAME=model
[[ -n "${DISPLAY_NAME}" ]] || DISPLAY_NAME="${POOL_PREFIX}"

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
VLLM_ENGINE_PORT=7000

cleanup_replica() {
  if [[ "${SLURM_PROCID:-0}" -eq 0 ]]; then
    registry_remove "${REPLICA_ID}" || true
  fi
  ray stop 2>/dev/null || true
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

  echo "[${REPLICA_ID}] Starting ${DISPLAY_NAME} vLLM server at TP=${TENSOR_PARALLEL_SIZE}/DP=1"
  "${VLLM_PYTHON}" "${EXTERNAL_VLLM_TOOLS_DIR}/serve_vllm_on_ray.py" serve "${MODEL}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --distributed-executor-backend ray \
    --port "${VLLM_HTTP_PORT}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    "${vllm_args[@]}" \
    > "${LOG_FILE}" 2>&1 &
  VLLM_PID=$!

  while ! "${VLLM_PYTHON}" -c \
    'import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=2).close()' \
    "http://${HEAD_IP}:${VLLM_HTTP_PORT}/health" >/dev/null 2>&1; do
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
      echo "[${REPLICA_ID}] ERROR: vLLM exited before becoming healthy" >&2
      exit 1
    fi
    sleep 5
  done

  registry_add "${REPLICA_ID}" "${HEAD_IP}" "${VLLM_HTTP_PORT}"
  echo "[${REPLICA_ID}] Registered healthy backend ${HEAD_IP}:${VLLM_HTTP_PORT}"
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
  offset="${node_offsets[${pool}]}"
  first_pool_node="${external_nodes[${offset}]}"
  echo "[INFO] Validating the ${display_names[${pool}]} container and Python environment"
  srun \
    --het-group=1 \
    --no-container-mount-home \
    --container-image="${containers[${pool}]}" \
    --container-mounts="${external_service_mount}" \
    --mpi=pmix \
    --gres="gpu:${GPUS_PER_NODE}" \
    --overlap \
    --kill-on-bad-exit=1 \
    --nodelist="${first_pool_node}" \
    --nodes=1 \
    --ntasks=1 \
    bash -lc \
    "command -v ray >/dev/null && '${vllm_pythons[${pool}]}' -c 'import ray, vllm; from nemo_rl.models.generation.vllm.patches import _apply_vllm_patches'" \
    >/dev/null &
  preflight_pids+=("$!")
  preflight_labels+=("${display_names[${pool}]} container")
done

echo "[INFO] Validating the load-balancer container and Python environment"
srun \
  --het-group=0 \
  --no-container-mount-home \
  --container-image="${CONTAINER}" \
  --container-mounts="${lb_mounts}" \
  --container-workdir="${SLURM_SUBMIT_DIR}" \
  --mpi=pmix \
  -A "${SLURM_JOB_ACCOUNT}" \
  -p "${SLURM_JOB_PARTITION}" \
  --overlap \
  --kill-on-bad-exit=1 \
  --nodelist="${ray_head_node}" \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=1 \
  bash -lc \
  "test -x /opt/external-vllm-tools/lb_watchdog.sh && '${EXTERNAL_VLLM_LB_PYTHON}' -c 'import aiohttp'" \
  >/dev/null &
preflight_pids+=("$!")
preflight_labels+=("load balancer container")

preflight_failed=0
for preflight_index in "${!preflight_pids[@]}"; do
  preflight_pid="${preflight_pids[${preflight_index}]}"
  if ! wait "${preflight_pid}"; then
    echo "[FATAL] External-vLLM preflight failed: ${preflight_labels[${preflight_index}]} (pid=${preflight_pid})" >&2
    preflight_failed=1
  fi
done
preflight_pids=()
preflight_labels=()
if (( preflight_failed != 0 )); then
  exit 1
fi

for pool in "${pool_names[@]}"; do
  echo "[INFO] Launching ${display_names[${pool}]} replicas"
  for (( replica_index = 0; replica_index < replicas[${pool}]; replica_index++ )); do
    first_node_index=$((node_offsets[${pool}] + replica_index * nodes_per_replica[${pool}]))
    replica_node_count="${nodes_per_replica[${pool}]}"
    replica_nodes=("${external_nodes[@]:first_node_index:replica_node_count}")
    replica_nodelist=$(IFS=,; echo "${replica_nodes[*]}")
    replica_id="${SLURM_JOB_ID}-${pool,,}-${replica_index}"
    head_ip_file="${pool_log_dirs[${pool}]}/head_ip_${replica_index}"
    vllm_log="${pool_log_dirs[${pool}]}/vllm_${replica_index}.log"

    echo "[INFO] ${display_names[${pool}]} replica ${replica_index}: ${replica_nodelist}"
    srun \
      --het-group=1 \
      --no-container-mount-home \
      --container-image="${containers[${pool}]}" \
      --container-mounts="${external_service_mount}" \
      --mpi=pmix \
      --gres="gpu:${GPUS_PER_NODE}" \
      --overlap \
      --kill-on-bad-exit=1 \
      --nodelist="${replica_nodelist}" \
      --nodes="${nodes_per_replica[${pool}]}" \
      --ntasks="${nodes_per_replica[${pool}]}" \
      --ntasks-per-node=1 \
      --export="ALL,POOL_PREFIX=${pool},REPLICA_ID=${replica_id},EXTERNAL_VLLM_TOOLS_DIR=${EXTERNAL_VLLM_TOOLS_DIR_HOST},EXTERNAL_VLLM_STATE_DIR=${state_dirs[${pool}]},EXTERNAL_VLLM_GROUP_ID=${group_ids[${pool}]},HEAD_IP_FILE=${head_ip_file},LOG_FILE=${vllm_log}" \
      --output="${pool_log_dirs[${pool}]}/replica_${replica_index}_%t.log" \
      bash -c "${VLLM_SERVER_BODY}" &
    service_step_pids+=("$!")
    service_step_labels+=("${display_names[${pool}]} replica ${replica_index}")
  done
done

ray_head_ip=$(resolve_node_ip "${ray_head_node}")
for pool in "${pool_names[@]}"; do
  pool_urls["${pool}"]="http://${ray_head_ip}:${lb_ports[${pool}]}/v1"
  echo "[INFO] Starting ${display_names[${pool}]} load balancer at ${pool_urls[${pool}]}"
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
    bash -lc "PYTHON='${EXTERNAL_VLLM_LB_PYTHON}' /opt/external-vllm-tools/lb_watchdog.sh '${lb_ports[${pool}]}' '${lb_state_dirs[${pool}]}' '${group_ids[${pool}]}'" &
  lb_step_pids+=("$!")
  lb_step_labels+=("${display_names[${pool}]} load balancer")
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

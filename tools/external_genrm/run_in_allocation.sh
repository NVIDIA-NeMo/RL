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

# Run an external GenRM pool beside NeMo RL in one Slurm heterogeneous job.

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
: "${GENRM_CONTAINER:?GENRM_CONTAINER is required}"
: "${GENRM_MODEL:?GENRM_MODEL is required}"
: "${GENRM_TOOLS_DIR_HOST:?GENRM_TOOLS_DIR_HOST is required}"
: "${GENRM_VLLM_PYTHON:?GENRM_VLLM_PYTHON is required}"

RAY_SUB="${RAY_SUB:-${SLURM_SUBMIT_DIR}/ray.sub}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
GENRM_REPLICAS="${GENRM_REPLICAS:-8}"
GENRM_TENSOR_PARALLEL_SIZE="${GENRM_TENSOR_PARALLEL_SIZE:-8}"
GENRM_REASONING_PARSER="${GENRM_REASONING_PARSER:-}"
GENRM_REASONING_PARSER_NAME="${GENRM_REASONING_PARSER_NAME:-}"
GENRM_TOOL_CALL_PARSER="${GENRM_TOOL_CALL_PARSER:-}"
GENRM_ENABLE_EXPERT_PARALLEL="${GENRM_ENABLE_EXPERT_PARALLEL:-0}"
GENRM_COMPILATION_CONFIG="${GENRM_COMPILATION_CONFIG:-}"
GENRM_MODEL_LOADER_EXTRA_CONFIG="${GENRM_MODEL_LOADER_EXTRA_CONFIG:-}"
GENRM_SERVED_MODEL_NAME="${GENRM_SERVED_MODEL_NAME:-model}"
GENRM_GROUP_ID="${GENRM_GROUP_ID:-inline-${SLURM_JOB_ID}}"
GENRM_VLLM_PORT="${GENRM_VLLM_PORT:-8000}"
GENRM_LB_PORT="${GENRM_LB_PORT:-9213}"
GENRM_LB_PYTHON="${GENRM_LB_PYTHON:-/opt/nemo_rl_venv/bin/python}"
GENRM_STARTUP_TIMEOUT="${GENRM_STARTUP_TIMEOUT:-3600}"
GENRM_URL_PLACEHOLDER="${GENRM_URL_PLACEHOLDER:-__GENRM_BASE_URL__}"
export \
  GENRM_COMPILATION_CONFIG \
  GENRM_ENABLE_EXPERT_PARALLEL \
  GENRM_MODEL_LOADER_EXTRA_CONFIG \
  GENRM_REASONING_PARSER_NAME \
  GENRM_SERVED_MODEL_NAME \
  GENRM_TOOL_CALL_PARSER

LOG_DIR="${BASE_LOG_DIR}/${SLURM_JOB_ID}-logs"
GENRM_URL_FILE="${LOG_DIR}/genrm_url"
GENRM_LOG_DIR="${LOG_DIR}/external_genrm"
GENRM_STATE_DIR="${GENRM_LOG_DIR}/state"

if [[ "${SLURM_HET_SIZE}" != "2" ]]; then
  echo "[FATAL] Expected exactly two Slurm hetgroups, got ${SLURM_HET_SIZE}" >&2
  exit 1
fi
if [[ ! -f "${RAY_SUB}" ]]; then
  echo "[FATAL] ray.sub does not exist: ${RAY_SUB}" >&2
  exit 1
fi
if [[ "${COMMAND}" != *"${GENRM_URL_PLACEHOLDER}"* ]]; then
  echo "[FATAL] Driver command is missing ${GENRM_URL_PLACEHOLDER}" >&2
  exit 1
fi
for required_file in genrm_registry.sh genrm_lb.py lb_watchdog.sh serve_vllm_on_ray.py; do
  if [[ ! -f "${GENRM_TOOLS_DIR_HOST}/${required_file}" ]]; then
    echo "[FATAL] Missing ${GENRM_TOOLS_DIR_HOST}/${required_file}" >&2
    exit 1
  fi
done
if [[ -n "${GENRM_REASONING_PARSER}" && ! -f "${GENRM_REASONING_PARSER}" ]]; then
  echo "[FATAL] GenRM reasoning parser does not exist: ${GENRM_REASONING_PARSER}" >&2
  exit 1
fi
if { [[ -n "${GENRM_REASONING_PARSER}" ]] && [[ -z "${GENRM_REASONING_PARSER_NAME}" ]]; } ||
  { [[ -z "${GENRM_REASONING_PARSER}" ]] && [[ -n "${GENRM_REASONING_PARSER_NAME}" ]]; }; then
  echo "[FATAL] GENRM_REASONING_PARSER and GENRM_REASONING_PARSER_NAME must be set together" >&2
  exit 1
fi
if [[ "${GENRM_ENABLE_EXPERT_PARALLEL}" != "0" && "${GENRM_ENABLE_EXPERT_PARALLEL}" != "1" ]]; then
  echo "[FATAL] GENRM_ENABLE_EXPERT_PARALLEL must be 0 or 1" >&2
  exit 1
fi
shared_paths=("${BASE_LOG_DIR}" "${GENRM_TOOLS_DIR_HOST}")
if [[ "${GENRM_MODEL}" == /* ]]; then
  shared_paths+=("${GENRM_MODEL}")
fi
if [[ -n "${GENRM_REASONING_PARSER}" ]]; then
  shared_paths+=("${GENRM_REASONING_PARSER}")
fi
for shared_path in "${shared_paths[@]}"; do
  if [[ "${shared_path}" != "/lustre" && "${shared_path}" != /lustre/* ]]; then
    echo "[FATAL] Path must be under /lustre for the GenRM container mount: ${shared_path}" >&2
    exit 1
  fi
done
if (( GENRM_TENSOR_PARALLEL_SIZE % GPUS_PER_NODE != 0 )); then
  echo "[FATAL] GENRM_TENSOR_PARALLEL_SIZE must be divisible by GPUS_PER_NODE" >&2
  exit 1
fi

GENRM_NODES_PER_REPLICA=$((GENRM_TENSOR_PARALLEL_SIZE / GPUS_PER_NODE))
EXPECTED_GENRM_NODES=$((GENRM_REPLICAS * GENRM_NODES_PER_REPLICA))

mapfile -t ray_nodes < <(
  scontrol show hostnames "${SLURM_JOB_NODELIST_HET_GROUP_0}" | sort
)
mapfile -t genrm_nodes < <(
  scontrol show hostnames "${SLURM_JOB_NODELIST_HET_GROUP_1}" | sort
)
if (( ${#ray_nodes[@]} == 0 )); then
  echo "[FATAL] Slurm hetgroup 0 contains no NeMo RL nodes" >&2
  exit 1
fi
if (( ${#genrm_nodes[@]} != EXPECTED_GENRM_NODES )); then
  echo "[FATAL] Slurm hetgroup 1 has ${#genrm_nodes[@]} nodes, expected ${EXPECTED_GENRM_NODES} for ${GENRM_REPLICAS} TP=${GENRM_TENSOR_PARALLEL_SIZE} replicas" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}" "${GENRM_LOG_DIR}" "${GENRM_STATE_DIR}"
rm -f \
  "${GENRM_STATE_DIR}/.registry_${GENRM_GROUP_ID}" \
  "${GENRM_STATE_DIR}/.registry_${GENRM_GROUP_ID}.lock" \
  "${GENRM_URL_FILE}"

{
  echo "[external_genrm]"
  printf '%s\n' "${genrm_nodes[@]}"
  echo "[nemo_rl_ray]"
  printf '%s\n' "${ray_nodes[@]}"
} > "${LOG_DIR}/node-allocation.txt"

echo "[INFO] Heterogeneous-job external GenRM topology"
echo "[INFO]   Hetgroup 0, NeMo RL Ray: ${#ray_nodes[@]} nodes (${SLURM_JOB_NODELIST_HET_GROUP_0})"
echo "[INFO]   Hetgroup 1, GenRM: ${#genrm_nodes[@]} nodes, ${GENRM_REPLICAS} TP=${GENRM_TENSOR_PARALLEL_SIZE} replicas"

declare -a genrm_step_pids=()
lb_step_pid=""
ray_sub_pid=""

cleanup() {
  local status=$?
  trap - EXIT TERM INT

  touch "${LOG_DIR}/ENDED" 2>/dev/null || true
  if [[ -n "${ray_sub_pid}" ]] && kill -0 "${ray_sub_pid}" 2>/dev/null; then
    kill "${ray_sub_pid}" 2>/dev/null || true
  fi
  if [[ -n "${lb_step_pid}" ]] && kill -0 "${lb_step_pid}" 2>/dev/null; then
    kill "${lb_step_pid}" 2>/dev/null || true
  fi
  for pid in "${genrm_step_pids[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
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
  for index in "${!genrm_step_pids[@]}"; do
    if ! kill -0 "${genrm_step_pids[$index]}" 2>/dev/null; then
      echo "[FATAL] GenRM replica step ${index} exited unexpectedly" >&2
      return 1
    fi
  done
  if [[ -n "${lb_step_pid}" ]] && ! kill -0 "${lb_step_pid}" 2>/dev/null; then
    echo "[FATAL] GenRM load-balancer step exited unexpectedly" >&2
    return 1
  fi
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

GENRM_BODY=$(cat <<'GENRM_BODY_EOF'
set -euo pipefail

: "${REPLICA_ID:?REPLICA_ID is required}"
: "${MODEL:?MODEL is required}"
: "${VLLM_PYTHON:?VLLM_PYTHON is required}"
: "${VLLM_PORT:?VLLM_PORT is required}"
: "${TENSOR_PARALLEL_SIZE:?TENSOR_PARALLEL_SIZE is required}"
: "${GENRM_TOOLS_DIR:?GENRM_TOOLS_DIR is required}"
: "${GENRM_SERVING_DIR:?GENRM_SERVING_DIR is required}"
: "${GENRM_GROUP_ID:?GENRM_GROUP_ID is required}"
: "${HEAD_IP_FILE:?HEAD_IP_FILE is required}"
: "${LOG_FILE:?LOG_FILE is required}"

source "${GENRM_TOOLS_DIR}/genrm_registry.sh"
RAY_PORT=6379

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
    --disable-usage-stats

  export FLASHINFER_WORKSPACE_BASE=/tmp
  export VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm
  export VLLM_ALLREDUCE_USE_SYMM_MEM=0

  reasoning_args=()
  if [[ -n "${REASONING_PARSER:-}" ]]; then
    reasoning_args=(
      --reasoning-parser-plugin "${REASONING_PARSER}"
      --reasoning-parser "${GENRM_REASONING_PARSER_NAME}"
    )
  fi

  tool_call_args=()
  if [[ -n "${GENRM_TOOL_CALL_PARSER}" ]]; then
    tool_call_args=(
      --enable-auto-tool-choice
      --tool-call-parser "${GENRM_TOOL_CALL_PARSER}"
    )
  fi

  expert_parallel_args=()
  if [[ "${GENRM_ENABLE_EXPERT_PARALLEL}" == "1" ]]; then
    expert_parallel_args=(--enable-expert-parallel)
  fi

  compilation_args=()
  if [[ -n "${GENRM_COMPILATION_CONFIG}" ]]; then
    compilation_args=(--compilation-config "${GENRM_COMPILATION_CONFIG}")
  fi

  model_loader_args=()
  if [[ -n "${GENRM_MODEL_LOADER_EXTRA_CONFIG}" ]]; then
    model_loader_args=(
      --model-loader-extra-config "${GENRM_MODEL_LOADER_EXTRA_CONFIG}"
    )
  fi

  echo "[${REPLICA_ID}] Starting native vLLM server at TP=${TENSOR_PARALLEL_SIZE}/DP=1"
  "${VLLM_PYTHON}" "${GENRM_TOOLS_DIR}/serve_vllm_on_ray.py" serve "${MODEL}" \
    --trust-remote-code \
    --dtype bfloat16 \
    --kv-cache-dtype fp8 \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --max-num-seqs 256 \
    --gpu-memory-utilization 0.95 \
    --enable-prefix-caching \
    --distributed-executor-backend ray \
    --port "${VLLM_PORT}" \
    --served-model-name "${GENRM_SERVED_MODEL_NAME}" \
    "${reasoning_args[@]}" \
    "${tool_call_args[@]}" \
    "${expert_parallel_args[@]}" \
    "${compilation_args[@]}" \
    "${model_loader_args[@]}" \
    > "${LOG_FILE}" 2>&1 &
  VLLM_PID=$!

  while ! "${VLLM_PYTHON}" -c \
    'import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=2).close()' \
    "http://${HEAD_IP}:${VLLM_PORT}/health" >/dev/null 2>&1; do
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
      echo "[${REPLICA_ID}] ERROR: vLLM exited before becoming healthy" >&2
      exit 1
    fi
    sleep 5
  done

  registry_add "${REPLICA_ID}" "${HEAD_IP}" "${VLLM_PORT}"
  echo "[${REPLICA_ID}] Registered healthy backend ${HEAD_IP}:${VLLM_PORT}"
  wait "${VLLM_PID}"
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
    if ray start --address="${HEAD_IP}:${RAY_PORT}" --disable-usage-stats; then
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
GENRM_BODY_EOF
)
bash -n <(printf '%s' "${GENRM_BODY}") || {
  echo "[FATAL] Generated GenRM server script has syntax errors" >&2
  exit 1
}

ray_head_node="${ray_nodes[0]}"
lb_state_dir="/tmp/external-genrm-state"
lb_mounts="${MOUNTS},${GENRM_TOOLS_DIR_HOST}:/opt/external-genrm-tools:ro,${GENRM_STATE_DIR}:${lb_state_dir}"

echo "[INFO] Launching external GenRM replicas"
for (( replica_index = 0; replica_index < GENRM_REPLICAS; replica_index++ )); do
  first_node_index=$((replica_index * GENRM_NODES_PER_REPLICA))
  replica_nodes=("${genrm_nodes[@]:first_node_index:GENRM_NODES_PER_REPLICA}")
  replica_nodelist=$(IFS=,; echo "${replica_nodes[*]}")
  replica_id="${SLURM_JOB_ID}-genrm-${replica_index}"
  head_ip_file="${GENRM_LOG_DIR}/head_ip_${replica_index}"
  vllm_log="${GENRM_LOG_DIR}/vllm_${replica_index}.log"

  echo "[INFO] GenRM replica ${replica_index}: ${replica_nodelist}"
  srun \
    --het-group=1 \
    --no-container-mount-home \
    --container-image="${GENRM_CONTAINER}" \
    --container-mounts=/lustre:/lustre \
    --mpi=pmix \
    --gres="gpu:${GPUS_PER_NODE}" \
    -A "${SLURM_JOB_ACCOUNT}" \
    -p "${SLURM_JOB_PARTITION}" \
    --overlap \
    --kill-on-bad-exit=1 \
    --nodelist="${replica_nodelist}" \
    --nodes="${GENRM_NODES_PER_REPLICA}" \
    --ntasks="${GENRM_NODES_PER_REPLICA}" \
    --ntasks-per-node=1 \
    --export="ALL,REPLICA_ID=${replica_id},VLLM_PORT=${GENRM_VLLM_PORT},TENSOR_PARALLEL_SIZE=${GENRM_TENSOR_PARALLEL_SIZE},MODEL=${GENRM_MODEL},VLLM_PYTHON=${GENRM_VLLM_PYTHON},REASONING_PARSER=${GENRM_REASONING_PARSER},GENRM_TOOLS_DIR=${GENRM_TOOLS_DIR_HOST},GENRM_SERVING_DIR=${GENRM_STATE_DIR},GENRM_GROUP_ID=${GENRM_GROUP_ID},HEAD_IP_FILE=${head_ip_file},LOG_FILE=${vllm_log}" \
    --output="${GENRM_LOG_DIR}/replica_${replica_index}_%t.log" \
    bash -c "${GENRM_BODY}" &
  genrm_step_pids+=("$!")
done

ray_head_ip=$(resolve_node_ip "${ray_head_node}")
GENRM_LB_URL="http://${ray_head_ip}:${GENRM_LB_PORT}/v1"

echo "[INFO] Starting GenRM load balancer at ${GENRM_LB_URL}"
srun \
  --het-group=0 \
  --no-container-mount-home \
  --container-name="genrm-lb-${SLURM_JOB_ID}" \
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
  --output="${GENRM_LOG_DIR}/load_balancer.log" \
  bash -lc "PYTHON='${GENRM_LB_PYTHON}' /opt/external-genrm-tools/lb_watchdog.sh '${GENRM_LB_PORT}' '${lb_state_dir}' '${GENRM_GROUP_ID}'" &
lb_step_pid=$!

deadline=$((SECONDS + GENRM_STARTUP_TIMEOUT))
while ! curl -sf "http://${ray_head_ip}:${GENRM_LB_PORT}/health" >/dev/null 2>&1; do
  check_service_steps
  if (( SECONDS >= deadline )); then
    echo "[FATAL] Timed out waiting for the GenRM load balancer to bind" >&2
    exit 1
  fi
  sleep 2
done

echo "[INFO] Waiting for all ${GENRM_REPLICAS} GenRM servers"
while true; do
  ready=$(
    GENRM_SERVING_DIR="${GENRM_STATE_DIR}" \
    GENRM_TOOLS_DIR="${GENRM_TOOLS_DIR_HOST}" \
    GENRM_GROUP_ID="${GENRM_GROUP_ID}" \
    bash -c 'source "${GENRM_TOOLS_DIR}/genrm_registry.sh"; registry_count_ready'
  )
  ready=${ready:-0}
  echo "[INFO] GenRM ready: ${ready}/${GENRM_REPLICAS}"
  if (( ready == GENRM_REPLICAS )); then
    break
  fi
  check_service_steps
  if (( SECONDS >= deadline )); then
    echo "[FATAL] Timed out waiting for the complete GenRM pool" >&2
    exit 1
  fi
  sleep 15
done

until curl -sfm 10 "${GENRM_LB_URL}/models" >/dev/null 2>&1; do
  check_service_steps
  if (( SECONDS >= deadline )); then
    echo "[FATAL] GenRM load balancer failed its end-to-end /models probe" >&2
    exit 1
  fi
  sleep 5
done

echo "${GENRM_LB_URL}" > "${GENRM_URL_FILE}"
COMMAND="${COMMAND//${GENRM_URL_PLACEHOLDER}/${GENRM_LB_URL}}"
export COMMAND

echo "[INFO] External GenRM pool is healthy; starting NeMo RL"
(
  # ray.sub predates hetjobs and consumes the unsuffixed allocation variables.
  # Restrict those variables to component 0; srun also defaults to hetgroup 0.
  export SLURM_JOB_NODELIST="${SLURM_JOB_NODELIST_HET_GROUP_0}"
  export SLURM_JOB_NUM_NODES="${#ray_nodes[@]}"
  bash "${RAY_SUB}"
) &
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

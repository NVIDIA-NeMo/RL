#!/bin/bash

# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
# =============================================================================
# k8s_entry.sh - In-container Ray bootstrap for NeMo RL on Kubernetes.
#
# This is the Kubernetes counterpart of ray.sub (Slurm). It is meant to be the
# *container entrypoint command* of every pod in a multi-node K8s training job.
# It brings up (or attaches to) a Ray cluster matching what init_ray() expects,
# then -- on the head/rank-0 pod only -- runs the training command.
#
# It auto-detects the two mainstream ways to run distributed jobs on K8s, so no
# bespoke YAML is required from NeMo RL:
#
#   * KubeRay (RayCluster / RayJob) -- Ray is already running (detected via
#     KUBERAY_GEN_RAY_START_CMD or a reachable RAY_ADDRESS / live `ray status`).
#     The script skips cluster setup and just runs the training command, which
#     attaches with ray.init(address="auto").
#
#   * Kubeflow PyTorchJob (and any launcher that injects torch-distributed style
#     env vars: RANK / WORLD_SIZE / MASTER_ADDR) -- the script sets up Ray across
#     all pods (rank 0 == head) and runs the training command on rank 0. Workers
#     block until the head calls `ray stop`, then exit 0 so the job reports
#     success.
#
# The Ray resource tags ("worker_units" + an externally-managed marker) match
# ray.sub so that nemo_rl.distributed.virtual_cluster.init_ray() takes its
# "connect to existing externally-managed cluster" path on every backend.
#
# -----------------------------------------------------------------------------
# Usage
#
#   The training command may be passed either as positional args or via the
#   COMMAND environment variable (positional args take precedence):
#
#     bash k8s_entry.sh uv run --locked examples/run_grpo.py --config ...
#     COMMAND="uv run --locked examples/run_grpo.py --config ..." bash k8s_entry.sh
#
#   In a Kubeflow PyTorchJob, every pod (Master + Worker) runs the *same*
#   command; this script uses RANK to decide head vs. worker. As a KubeRay
#   RayJob entrypoint, the script detects the managed cluster and only runs the
#   training command as the driver.
#
# -----------------------------------------------------------------------------
# Environment variables (all optional unless noted):
#
#   COMMAND          Training command to run on the head/rank-0 pod. Ignored if
#                    positional args are given. If neither is set, the head
#                    idles (sleep infinity) for interactive `kubectl exec`.
#   SETUP_COMMAND    Optional command run on every pod before starting Ray.
#   GPUS_PER_NODE    GPUs advertised per pod to Ray as worker_units (default: 8).
#   NUM_NODES        Total pods in the job. Defaults to WORLD_SIZE, then 1.
#   RANK             This pod's rank (0 == head). Defaults to RANK, then
#                    PET_NODE_RANK (torchrun), then JOB_COMPLETION_INDEX
#                    (indexed Jobs / JobSet), then 0.
#   MASTER_ADDR      Hostname/IP of the head pod (injected by PyTorchJob).
#   RAY_PORT                 Ray GCS port (default: 6379).
#   RAY_DASHBOARD_PORT       Dashboard port (default: 8265).
#   RAY_CLIENT_SERVER_PORT   Ray client server port (default: 10001).
#   GCS_WAIT_TIMEOUT         Seconds a worker waits for the head GCS (default: 600).
#   CLUSTER_WAIT_TIMEOUT     Seconds the head waits for all workers (default: 1800).
# =============================================================================

set -eou pipefail

# ---------------------------------------------------------------------------
# Logging helpers (match the [INFO]/[WARN]/[ERROR] convention used elsewhere).
# ---------------------------------------------------------------------------
log_info()  { echo "[INFO]  $*"; }
log_warn()  { echo "[WARN]  $*" >&2; }
log_error() { echo "[ERROR] $*" >&2; }
die()       { log_error "$*"; exit 1; }

# ---------------------------------------------------------------------------
# Resolve the training command. Positional args win; otherwise fall back to the
# COMMAND env var. An empty result means "idle" (interactive head).
# ---------------------------------------------------------------------------
if [[ "$#" -gt 0 ]]; then
  TRAIN_CMD=("$@")
elif [[ -n "${COMMAND:-}" ]]; then
  # shellcheck disable=SC2206  # intentional word-splitting of the COMMAND string
  TRAIN_CMD=(bash -c "${COMMAND}")
else
  TRAIN_CMD=()
fi

# ---------------------------------------------------------------------------
# Defaults. Every value can be overridden by exporting it before invocation.
# We deliberately read the same env vars that Kubeflow's training-operator,
# torchrun, and indexed Jobs / JobSet inject, so no manifest changes are needed.
# ---------------------------------------------------------------------------
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
RAY_PORT="${RAY_PORT:-6379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
RAY_CLIENT_SERVER_PORT="${RAY_CLIENT_SERVER_PORT:-10001}"
GCS_WAIT_TIMEOUT="${GCS_WAIT_TIMEOUT:-600}"
CLUSTER_WAIT_TIMEOUT="${CLUSTER_WAIT_TIMEOUT:-1800}"
SETUP_COMMAND="${SETUP_COMMAND:-}"

# Rank: first env var that is set wins (PyTorchJob/torchrun/indexed-Job/JobSet).
RANK="${RANK:-${PET_NODE_RANK:-${JOB_COMPLETION_INDEX:-0}}}"

# World size / node count: WORLD_SIZE is set by PyTorchJob & torchrun.
NUM_NODES="${NUM_NODES:-${WORLD_SIZE:-1}}"

# Head address: MASTER_ADDR is set by PyTorchJob & torchrun.
MASTER_ADDR="${MASTER_ADDR:-}"

# After ray>=2.47 a per-task uv venv feature is on by default and conflicts with
# NeMo RL's once-per-node venv handling (see ray.sub for the same setting).
export RAY_ENABLE_UV_RUN_RUNTIME_ENV="${RAY_ENABLE_UV_RUN_RUNTIME_ENV:-0}"

# Fixed Ray component ports (mirrors ray.sub). On Kubernetes the head and worker
# run in separate pods, so every Ray component must bind a *known* port for the
# worker raylet to register with the head GCS -- otherwise Ray picks random
# high ports that a headless Service / NetworkPolicy may not allow, and the
# worker times out with "RPC error: Deadline Exceeded" during startup. The head
# uses the base port + 1 (matching ray.sub) so a co-located worker on the head
# pod would not collide; worker pods use the base ports.
NODE_MANAGER_PORT="${NODE_MANAGER_PORT:-53001}"
OBJECT_MANAGER_PORT="${OBJECT_MANAGER_PORT:-53003}"
RUNTIME_ENV_AGENT_PORT="${RUNTIME_ENV_AGENT_PORT:-53005}"
DASHBOARD_AGENT_GRPC_PORT="${DASHBOARD_AGENT_GRPC_PORT:-53007}"
DASHBOARD_AGENT_LISTEN_PORT="${DASHBOARD_AGENT_LISTEN_PORT:-52365}"
METRICS_EXPORT_PORT="${METRICS_EXPORT_PORT:-53009}"
MIN_WORKER_PORT="${MIN_WORKER_PORT:-10002}"
MAX_WORKER_PORT="${MAX_WORKER_PORT:-11000}"


# ---------------------------------------------------------------------------
# Resolve a hostname to an IP, trying several tools (mirrors ray.sub robustness).
# Echoes the original value if resolution fails.
# ---------------------------------------------------------------------------
resolve_host() {
  local host="$1" ip=""
  [[ -z "$host" ]] && return 0
  ip=$(getent hosts "$host" 2>/dev/null | awk '{print $1}' | head -n1 || true)
  if [[ -z "$ip" ]] && command -v host >/dev/null 2>&1; then
    ip=$(host "$host" 2>/dev/null | awk '/has address/ {print $4}' | head -n1 || true)
  fi
  if [[ -z "$ip" ]] && command -v nslookup >/dev/null 2>&1; then
    ip=$(nslookup "$host" 2>/dev/null | awk '/^Address: / {print $2}' | head -n1 || true)
  fi
  echo "${ip:-$host}"
}

# Best-effort detection of this pod's primary IP for `--node-ip-address`.
detect_self_ip() {
  local ip=""
  ip=$(hostname -I 2>/dev/null | awk '{print $1}' || true)
  if [[ -z "$ip" ]]; then
    ip=$(ip route get 1 2>/dev/null | awk '{print $(NF-2); exit}' || true)
  fi
  echo "${ip:-127.0.0.1}"
}

# ---------------------------------------------------------------------------
# Ray best-practice: raise the open-file soft limit (session-scoped only).
# ---------------------------------------------------------------------------
raise_ulimit() {
  local hard
  hard=$(ulimit -Hn)
  if [[ "$hard" == "unlimited" ]] || [[ 65535 -lt "$hard" ]]; then
    ulimit -Sn 65535 || true
  fi
}

# ---------------------------------------------------------------------------
# Wait until the Ray cluster reports >= expected worker_units (one per GPU),
# matching the bringup gate that ray.sub performs before launching the driver.
# ---------------------------------------------------------------------------
wait_for_worker_units() {
  local expected="$1" timeout="$2" start
  start=$(date +%s)
  log_info "Waiting for ${expected} worker_units to come online (timeout ${timeout}s)..."
  while true; do
    local status online
    status=$(ray status 2>/dev/null || true)
    online=$(echo "$status" | grep "worker_units" | awk -F'[/. ]' '{print $4}' | head -n1 || true)
    online="${online:-0}"
    [[ "$online" =~ ^[0-9]+$ ]] || online=0
    log_info "worker_units online: ${online}/${expected}"
    if (( online >= expected )); then
      log_info "All worker_units online. Cluster is ready."
      return 0
    fi
    if (( $(date +%s) - start > timeout )); then
      die "Timed out waiting for worker_units (${online}/${expected})."
    fi
    sleep 5
  done
}

# ---------------------------------------------------------------------------
# Wait until the head's GCS answers health checks (used by worker pods).
# ---------------------------------------------------------------------------
wait_for_gcs() {
  local address="$1" timeout="$2" start
  start=$(date +%s)
  log_info "Waiting for Ray GCS at ${address} (timeout ${timeout}s)..."
  while true; do
    if ray health-check --address "$address" >/dev/null 2>&1; then
      log_info "Ray GCS at ${address} is ready."
      return 0
    fi
    if (( $(date +%s) - start > timeout )); then
      die "Timed out waiting for Ray GCS at ${address}."
    fi
    sleep 5
  done
}

# ---------------------------------------------------------------------------
# Block until the head's GCS becomes unreachable (head ran `ray stop`), so the
# worker pod can exit 0 and let the PyTorchJob report success.
# ---------------------------------------------------------------------------
wait_until_head_stops() {
  local address="$1"
  log_info "Worker joined cluster. Waiting until head GCS at ${address} stops..."
  while ray health-check --address "$address" >/dev/null 2>&1; do
    sleep 10
  done
  log_info "Head GCS at ${address} is gone; worker exiting 0."
}

# ---------------------------------------------------------------------------
# Run the training command (or idle) on the head pod.
# ---------------------------------------------------------------------------
run_driver_or_idle() {
  if [[ "${#TRAIN_CMD[@]}" -gt 0 ]]; then
    log_info "Launching training command on head pod: ${TRAIN_CMD[*]}"
    "${TRAIN_CMD[@]}"
  else
    log_info "No training command given; cluster is idle. Use 'kubectl exec' to attach."
    sleep infinity
  fi
}

# ===========================================================================
# Main
# ===========================================================================
raise_ulimit

if [[ -n "$SETUP_COMMAND" ]]; then
  log_info "Running SETUP_COMMAND on this pod..."
  bash -c "$SETUP_COMMAND"
fi

log_info "===================================================="
log_info "NeMo RL Kubernetes Ray bootstrap (k8s_entry.sh)"
log_info "  RANK            = ${RANK}"
log_info "  NUM_NODES       = ${NUM_NODES}"
log_info "  GPUS_PER_NODE   = ${GPUS_PER_NODE}"
log_info "  MASTER_ADDR     = ${MASTER_ADDR:-<unset>}"
log_info "  RAY_PORT        = ${RAY_PORT}"
log_info "===================================================="

# ---------------------------------------------------------------------------
# Mode A: a Ray cluster is already managed for us (KubeRay / RayJob). Do not
# start a second Ray; just run the training command as the driver. KubeRay sets
# KUBERAY_GEN_RAY_START_CMD on its pods; RAY_ADDRESS is the "attach here" hint.
# ---------------------------------------------------------------------------
externally_managed=false
if [[ -n "${KUBERAY_GEN_RAY_START_CMD:-}" ]]; then
  externally_managed=true
  log_info "Detected KubeRay-managed Ray cluster (KUBERAY_GEN_RAY_START_CMD set)."
elif [[ -n "${RAY_ADDRESS:-}" ]] && ray health-check --address "${RAY_ADDRESS}" >/dev/null 2>&1; then
  externally_managed=true
  log_info "Detected reachable Ray cluster via RAY_ADDRESS=${RAY_ADDRESS}."
fi

if [[ "$externally_managed" == "true" ]]; then
  run_driver_or_idle
  exit 0
fi

# ---------------------------------------------------------------------------
# Mode B: we own the Ray bringup (Kubeflow PyTorchJob / JobSet / indexed Job).
# Rank 0 is the head; all other ranks are workers that attach to MASTER_ADDR.
# ---------------------------------------------------------------------------
if [[ -z "$MASTER_ADDR" ]]; then
  if [[ "$RANK" == "0" ]]; then
    MASTER_ADDR=$(detect_self_ip)
    log_warn "MASTER_ADDR not set; head auto-detected its IP as ${MASTER_ADDR}."
  else
    die "MASTER_ADDR must be set for worker pods (rank ${RANK})."
  fi
fi

head_ip=$(resolve_host "$MASTER_ADDR")
gcs_address="${head_ip}:${RAY_PORT}"

# Advertise the same resource tags ray.sub uses so init_ray() treats this as an
# externally-managed cluster and reuses it (rather than starting a local one).
ray_resources="{\"worker_units\": ${GPUS_PER_NODE}, \"slurm_managed_ray_cluster\": 1}"

if [[ "$RANK" == "0" ]]; then
  self_ip=$(detect_self_ip)
  log_info "Starting Ray HEAD on ${self_ip} (advertising as ${head_ip})..."
  ray start --head \
    --disable-usage-stats \
    --resources="${ray_resources}" \
    --num-gpus="${GPUS_PER_NODE}" \
    --node-ip-address="${self_ip}" \
    --port="${RAY_PORT}" \
    --ray-client-server-port="${RAY_CLIENT_SERVER_PORT}" \
    --dashboard-host=0.0.0.0 \
    --dashboard-port="${RAY_DASHBOARD_PORT}" \
    --node-manager-port=$((NODE_MANAGER_PORT + 1)) \
    --object-manager-port=$((OBJECT_MANAGER_PORT + 1)) \
    --runtime-env-agent-port=$((RUNTIME_ENV_AGENT_PORT + 1)) \
    --dashboard-agent-grpc-port=$((DASHBOARD_AGENT_GRPC_PORT + 1)) \
    --dashboard-agent-listen-port=$((DASHBOARD_AGENT_LISTEN_PORT + 1)) \
    --metrics-export-port=$((METRICS_EXPORT_PORT + 1)) \
    --include-dashboard=True

  wait_for_worker_units "$((NUM_NODES * GPUS_PER_NODE))" "$CLUSTER_WAIT_TIMEOUT"

  # Run the driver, then stop Ray so worker pods unblock and exit 0.
  set +e
  run_driver_or_idle
  driver_rc=$?
  set -e
  log_info "Training command finished (rc=${driver_rc}); stopping Ray head."
  ray stop >/dev/null 2>&1 || true
  exit "$driver_rc"
else
  log_info "Starting Ray WORKER (rank ${RANK}) attaching to ${gcs_address}..."
  wait_for_gcs "$gcs_address" "$GCS_WAIT_TIMEOUT"
  worker_ip=$(detect_self_ip)
  ray start --address="${gcs_address}" \
    --disable-usage-stats \
    --resources="${ray_resources}" \
    --num-gpus="${GPUS_PER_NODE}" \
    --node-ip-address="${worker_ip}" \
    --min-worker-port="${MIN_WORKER_PORT}" \
    --max-worker-port="${MAX_WORKER_PORT}" \
    --node-manager-port="${NODE_MANAGER_PORT}" \
    --object-manager-port="${OBJECT_MANAGER_PORT}" \
    --runtime-env-agent-port="${RUNTIME_ENV_AGENT_PORT}" \
    --dashboard-agent-grpc-port="${DASHBOARD_AGENT_GRPC_PORT}" \
    --dashboard-agent-listen-port="${DASHBOARD_AGENT_LISTEN_PORT}" \
    --metrics-export-port="${METRICS_EXPORT_PORT}"
  # Block until the head stops Ray, then exit 0 so the job is marked successful.
  wait_until_head_stops "$gcs_address"
fi

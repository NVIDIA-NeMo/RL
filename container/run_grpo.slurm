#!/bin/bash
#SBATCH --job-name=nemo-rl-grpo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --partition=hopper-prod
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --requeue
#SBATCH --time=3-00:00:00

# NeMo-RL GRPO Training with Ray Cluster (containerized)
#
# This script starts a Ray cluster inside a container and runs GRPO training.
# Based on NeMo-RL's ray.sub and adapted for this cluster.
#
# Usage:
#   Single node (8 GPUs, default 1B model):
#     sbatch container/run_grpo.slurm
#
#   Multi-node (2 nodes, 16 GPUs):
#     sbatch --nodes=2 container/run_grpo.slurm
#
#   Custom config:
#     CONFIG_FILE=/path/to/config.yaml sbatch container/run_grpo.slurm
#
#   Custom config with overrides:
#     EXTRA_OVERRIDES="grpo.max_num_steps=5 policy.model_name=Qwen/Qwen3-4B" sbatch container/run_grpo.slurm
#
#   Idle mode (just start Ray cluster, attach manually):
#     COMMAND="" sbatch --nodes=2 container/run_grpo.slurm
#
#   Attach to running idle cluster:
#     bash logs/nemo-rl-grpo-<jobid>.out  # prints attach instructions
#     bash <jobid>-attach.sh              # shell into head node
#     bash <jobid>-attach.sh 1            # shell into worker 1

set -eoux pipefail

# ==============================================================================
# Configuration
# ==============================================================================

# Container
CONTAINER="${CONTAINER:-/fsx/edward/docker_images/nemo-rl.sqsh}"
NEMO_RL_SOURCE="${NEMO_RL_SOURCE:-/fsx/edward/work/scale-rl/nemo-rl}"
MOUNTS="/fsx:/fsx,/scratch:/scratch,${NEMO_RL_SOURCE}:/opt/nemo-rl"

# Training config
CONFIG_FILE="${CONFIG_FILE:-}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/fsx/edward/checkpoints/nemo-rl/grpo}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

# Build the training command
if [[ -n "${COMMAND+x}" && -z "$COMMAND" ]]; then
    # COMMAND="" explicitly set — idle mode
    COMMAND=""
elif [[ -n "${COMMAND:-}" ]]; then
    # COMMAND set to something — use as-is
    :
else
    # Default: run GRPO
    COMMAND="cd /opt/nemo-rl && uv run python examples/run_grpo.py"
    if [[ -n "$CONFIG_FILE" ]]; then
        COMMAND+=" --config ${CONFIG_FILE}"
    fi
    COMMAND+=" cluster.num_nodes=${SLURM_NNODES:-1}"
    COMMAND+=" cluster.gpus_per_node=8"
    COMMAND+=" checkpointing.checkpoint_dir='${CHECKPOINT_DIR}'"
    if [[ -n "$EXTRA_OVERRIDES" ]]; then
        COMMAND+=" ${EXTRA_OVERRIDES}"
    fi
fi

# Ray venv cache on persistent storage
export NEMO_RL_VENV_DIR="/fsx/edward/.cache/nemo-rl/ray_venvs"

# Ray settings
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export RAY_USAGE_STATS_ENABLED=0
export RAY_USE_MULTIPROCESSING_CPU_COUNT=1

# Ray logging
export RAY_BACKEND_LOG_LEVEL=${RAY_BACKEND_LOG_LEVEL:-info}
export RAY_LOG_TO_STDERR=${RAY_LOG_TO_STDERR:-0}
export RAY_logging_level=${RAY_logging_level:-INFO}
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}

# ==============================================================================
# Ray cluster ports
# ==============================================================================
NODE_MANAGER_PORT=${NODE_MANAGER_PORT:-53001}
OBJECT_MANAGER_PORT=${OBJECT_MANAGER_PORT:-53003}
RUNTIME_ENV_AGENT_PORT=${RUNTIME_ENV_AGENT_PORT:-53005}
DASHBOARD_AGENT_GRPC_PORT=${DASHBOARD_AGENT_GRPC_PORT:-53007}
METRICS_EXPORT_PORT=${METRICS_EXPORT_PORT:-53009}
PORT=${PORT:-54514}
RAY_CLIENT_SERVER_PORT=${RAY_CLIENT_SERVER_PORT:-10001}
DASHBOARD_PORT=${DASHBOARD_PORT:-8265}
DASHBOARD_AGENT_LISTEN_PORT=${DASHBOARD_AGENT_LISTEN_PORT:-52365}
MIN_WORKER_PORT=${MIN_WORKER_PORT:-54001}
MAX_WORKER_PORT=${MAX_WORKER_PORT:-54513}
RAY_LOG_SYNC_FREQUENCY=${RAY_LOG_SYNC_FREQUENCY:-}
RAY_DEBUGGER_ARGS=
if [ "${RAY_DEBUG:-}" = "legacy" ]; then
  RAY_DEBUGGER_ARGS="--ray-debugger-external"
fi

# ==============================================================================
# Setup
# ==============================================================================

# Unset UV_CACHE_DIR to avoid local cache interfering with container cache
unset UV_CACHE_DIR

# Log directory
mkdir -p logs
LOG_DIR="logs/${SLURM_JOB_ID}-logs"
mkdir -p "$LOG_DIR"

GPUS_PER_NODE=8

# ulimit per Ray best practices
if [[ $(ulimit -Hn) == "unlimited" ]] || [[ 65535 -lt $(ulimit -Hn) ]]; then
  ulimit -Sn 65535
fi

# srun common args
COMMON_SRUN_ARGS=""
COMMON_SRUN_ARGS+=" --gres=gpu:${GPUS_PER_NODE}"
COMMON_SRUN_ARGS+=" --no-container-mount-home"
COMMON_SRUN_ARGS+=" --mpi=pmix"
COMMON_SRUN_ARGS+=" --container-mounts=$MOUNTS"
COMMON_SRUN_ARGS+=" --container-image=$CONTAINER"
COMMON_SRUN_ARGS+=" --container-workdir=/opt/nemo-rl"

num_retries=3

# Track backgrounded srun PIDs
declare -A SRUN_PIDS

check_srun_processes() {
  for name in "${!SRUN_PIDS[@]}"; do
    pid="${SRUN_PIDS[$name]}"
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[ERROR] Background srun '$name' died (pid=$pid)." >&2
      touch "$LOG_DIR/ENDED"
      exit 1
    fi
  done
}

# ==============================================================================
# Resolve node IPs
# ==============================================================================
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
ip_addresses_array=()

for node in $nodes; do
    ip_address=""
    ip_address=$(host $node 2>/dev/null | awk '/has address/ { print $4 }' | head -1 || true)
    if [[ -z "$ip_address" ]]; then
        ip_address=$(getent hosts $node 2>/dev/null | awk '{ print $1 }' | head -1 || true)
    fi
    if [[ -z "$ip_address" ]]; then
        ip_address=$node
    fi
    echo "[INFO] Node: $node -> IP: $ip_address"
    ip_addresses_array+=("$ip_address")
done

head_node=${nodes_array[0]}
head_node_ip=${ip_addresses_array[0]}
ip_head=$head_node_ip:$PORT

echo "========================================="
echo "NeMo-RL GRPO Training"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "GPUs per node: $GPUS_PER_NODE"
echo "Head node: $head_node ($head_node_ip)"
echo "Container: $CONTAINER"
echo "Config: ${CONFIG_FILE:-default (grpo_math_1B.yaml)}"
echo "Command: ${COMMAND:-<idle>}"
echo "Log dir: $LOG_DIR"
echo "Started at: $(date)"
echo "========================================="

# ==============================================================================
# Start Ray head node
# ==============================================================================
head_cmd=$(cat <<EOF
touch $LOG_DIR/STARTED_RAY_HEAD
export NEMO_RL_VENV_DIR=$NEMO_RL_VENV_DIR

exit-dramatically() {
    pkill -P \$\$ || true
    kill -TERM 0 || true
    exit 1
}
export -f exit-dramatically

# Monitor sidecar
(set +x; while true; do sleep 60; if [[ -f "$LOG_DIR/ENDED" ]]; then echo "ENDED detected"; exit-dramatically; fi; done) &

# Log sync sidecar
if [[ -n "$RAY_LOG_SYNC_FREQUENCY" ]]; then
  (set +x; while true; do
    sleep $RAY_LOG_SYNC_FREQUENCY
    mkdir -p $LOG_DIR/ray
    for d in /tmp/ray/session_[0-9]*/logs; do
      [[ -d "\$d" ]] && rsync -a "\$d/" "$LOG_DIR/ray/\$(basename \$(dirname \$d))/logs/" 2>/dev/null || true
    done
    [[ -f "$LOG_DIR/ENDED" ]] && break
  done) &
fi

ray start --head \
    --disable-usage-stats \
    --resources='{"worker_units": $GPUS_PER_NODE, "slurm_managed_ray_cluster": 1}' \
    --node-ip-address="$head_node_ip" \
    --port=${PORT} \
    --ray-client-server-port=${RAY_CLIENT_SERVER_PORT} \
    --dashboard-port=${DASHBOARD_PORT} \
    --dashboard-host="$head_node_ip" \
    --include-dashboard=True \
    --node-manager-port=$((NODE_MANAGER_PORT + 1)) \
    --object-manager-port=$((OBJECT_MANAGER_PORT + 1)) \
    --runtime-env-agent-port=$((RUNTIME_ENV_AGENT_PORT + 1)) \
    --dashboard-agent-grpc-port=$((DASHBOARD_AGENT_GRPC_PORT + 1)) \
    --dashboard-agent-listen-port=$((DASHBOARD_AGENT_LISTEN_PORT + 1)) \
    --metrics-export-port=$((METRICS_EXPORT_PORT + 1)) \
    $RAY_DEBUGGER_ARGS \
    --block
EOF
)

srun $COMMON_SRUN_ARGS --container-name=ray-head --nodes=1 --ntasks=1 -w "$head_node" -o $LOG_DIR/ray-head.log bash -x -c "$head_cmd" &
SRUN_PIDS["ray-head"]=$!

NUM_ACTORS=$((GPUS_PER_NODE * SLURM_NNODES))

# ==============================================================================
# Start Ray worker nodes (if multi-node)
# ==============================================================================
for ((i = 1; i < SLURM_NNODES; i++)); do
  node_i=${nodes_array[$i]}

  worker_cmd=$(cat <<EOF
export NEMO_RL_VENV_DIR=$NEMO_RL_VENV_DIR

exit-dramatically() {
    pkill -P \$\$ || true
    kill -TERM 0 || true
    exit 1
}

(set +x; while true; do sleep 60; if [[ -f "$LOG_DIR/ENDED" ]]; then echo "ENDED detected"; exit-dramatically; fi; done) &

if [[ -n "$RAY_LOG_SYNC_FREQUENCY" ]]; then
  (set +x; while true; do
    sleep $RAY_LOG_SYNC_FREQUENCY
    mkdir -p $LOG_DIR/ray/$node_i
    for d in /tmp/ray/session_[0-9]*/logs; do
      [[ -d "\$d" ]] && rsync -a "\$d/" "$LOG_DIR/ray/$node_i/\$(basename \$(dirname \$d))/logs/" 2>/dev/null || true
    done
    [[ -f "$LOG_DIR/ENDED" ]] && break
  done) &
fi

ray start --address "$ip_head" \
          --disable-usage-stats \
          --resources='{"worker_units": $GPUS_PER_NODE, "slurm_managed_ray_cluster": 1}' \
          --min-worker-port=${MIN_WORKER_PORT} \
          --max-worker-port=${MAX_WORKER_PORT} \
          --node-manager-port=${NODE_MANAGER_PORT} \
          --object-manager-port=${OBJECT_MANAGER_PORT} \
          --runtime-env-agent-port=${RUNTIME_ENV_AGENT_PORT} \
          --dashboard-agent-grpc-port=${DASHBOARD_AGENT_GRPC_PORT} \
          --dashboard-agent-listen-port=${DASHBOARD_AGENT_LISTEN_PORT} \
          --metrics-export-port=${METRICS_EXPORT_PORT} \
          $RAY_DEBUGGER_ARGS \
          --block
EOF
)
  srun $COMMON_SRUN_ARGS --container-name=ray-worker-$i --nodes=1 --ntasks=1 -w "$node_i" -o $LOG_DIR/ray-worker-$i.log bash -x -c "$worker_cmd" &
  SRUN_PIDS["ray-worker-$i"]=$!
  sleep 3
done

# ==============================================================================
# Wait for Ray cluster to be ready
# ==============================================================================
while check_srun_processes && ! srun --overlap --nodes=1 --ntasks=1 -w $head_node test -f $LOG_DIR/STARTED_RAY_HEAD; do
  echo "[INFO][$(date)] Waiting for head node container to start..."
  sleep 2
done

extract_worker_units() {
  status_output=$(srun --overlap --container-name=ray-head --nodes=1 --ntasks=1 -w "$head_node" ray status 2>/dev/null) || true
  if echo "$status_output" | grep -q "worker_units"; then
    echo "$status_output" | grep "worker_units" | awk -F'[/. ]' '{print $4}'
  else
    echo 0
  fi
}

while true; do
  worker_units=$(extract_worker_units)
  echo "[INFO] Ray workers online: $worker_units/$NUM_ACTORS"
  if [[ "$worker_units" -eq "$NUM_ACTORS" ]]; then
    break
  fi
  check_srun_processes
  sleep 2
done

echo "All Ray workers connected!"

# ==============================================================================
# Run training (or idle)
# ==============================================================================
CONTAINER_CWD=/opt/nemo-rl

if [[ -n "$COMMAND" ]]; then
  echo "Running: $COMMAND"
  srun --no-container-mount-home --overlap --container-name=ray-head --container-workdir=$CONTAINER_CWD --nodes=1 --ntasks=1 -w "$head_node" -o $LOG_DIR/ray-driver.log bash -c "$COMMAND"
else
  echo "[INFO] Ray cluster is idle. To attach:"
  cat <<EOF >${SLURM_JOB_ID}-attach.sh
WORKER_NUM=\${1:-}
if [[ -z "\$WORKER_NUM" ]]; then
  if [[ -n "\${COMMAND:-}" ]]; then
    srun --no-container-mount-home --gres=gpu:${GPUS_PER_NODE} --overlap --container-name=ray-head --container-workdir=$CONTAINER_CWD --nodes=1 --ntasks=1 -w "$head_node" --jobid $SLURM_JOB_ID bash -c "\$COMMAND"
  else
    srun --no-container-mount-home --gres=gpu:${GPUS_PER_NODE} --overlap --container-name=ray-head --container-workdir=$CONTAINER_CWD --nodes=1 --ntasks=1 -w "$head_node" --jobid $SLURM_JOB_ID --pty bash
  fi
else
  nodes_array=($nodes)
  if [[ -n "\${COMMAND:-}" ]]; then
    srun --no-container-mount-home --gres=gpu:${GPUS_PER_NODE} --overlap --container-name=ray-worker-\$WORKER_NUM --container-workdir=$CONTAINER_CWD --nodes=1 --ntasks=1 -w "\${nodes_array[\$WORKER_NUM]}" --jobid $SLURM_JOB_ID bash -c "\$COMMAND"
  else
    srun --no-container-mount-home --gres=gpu:${GPUS_PER_NODE} --overlap --container-name=ray-worker-\$WORKER_NUM --container-workdir=$CONTAINER_CWD --nodes=1 --ntasks=1 -w "\${nodes_array[\$WORKER_NUM]}" --jobid $SLURM_JOB_ID --pty bash
  fi
fi
EOF
  chmod +x ${SLURM_JOB_ID}-attach.sh
  echo "  bash ${SLURM_JOB_ID}-attach.sh          # head node shell"
  echo "  bash ${SLURM_JOB_ID}-attach.sh 1         # worker 1 shell"
  echo "  COMMAND='ray status' bash ${SLURM_JOB_ID}-attach.sh  # run command"
  sleep infinity
fi

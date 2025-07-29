# NeMo RL GRPO | DGX Create (Lepton) Guide

This guide outlines the steps needed to perform multi-node GRPO (Group Relative Policy Optimization) training on a DGX Lepton cluster using NeMo RL.

## Table of Contents

1. [Context](#context)
2. [Clone NeMo RL Repository](#clone-nemo-rl-repository)
3. [Build the NeMo RL Container](#build-the-nemo-rl-container)
4. [Launch NeMo RL GRPO using the UI](#launch-nemo-rl-grpo-using-the-ui)

## Context

This document provides step-by-step instructions for setting up and running multi-node GRPO training on a DGX Lepton cluster. The guide covers container building, deployment through the UI, and command-line interface usage.

## Clone NeMo RL Repository

### 1. Navigate to Desired Directory

```bash
cd <FOLDER_TO_CLONE_NEMO_RL_REPO>
```

### 2. Clone the Repository

```bash
git clone https://github.com/NVIDIA-NeMo/RL.git nemo-rl
```

## Build the NeMo RL Container

### 1. Navigate to Docker Directory

```bash
cd nemo-rl/docker/
```

### 2. Build the Container

**Important:** Make sure to increase your container's memory limit to ensure enough memory allocation space.

```bash
docker buildx build --target release -t nemo-rl-release1 --platform linux/amd64 -f Dockerfile ..
```

### 3. Tag the Image for NGC Registry

```bash
docker tag nemo-rl-release1 nvcr.io/<INSERT_NGC_ORG_NAME>/nemo-rl-release1:25.05
```

**Note:** To find your NGC Org Name:
1. Log into NGC
2. Click on your username
3. Click on "Contact Admin"
4. Your NGC Org Name will display in the pop-up

### 4. Login to Docker via NGC Registry

```bash
docker login nvcr.io
```

Use the following credentials:
- **Username:** `$oauthtoken`
- **Password:** `<NGC_API_TOKEN>`

### 5. Push Container to NGC Registry

```bash
docker push nvcr.io/<INSERT_NGC_ORG_ID>/nemo-rl-release1:25.05
```

## Launch NeMo RL GRPO using the UI

### 1. Login to Lepton Dashboard

Navigate to: https://dashboard.dgxc-lepton.nvidia.com

### 2. Create Batch Job

1. Click **"Batch Jobs"**
2. Click **"Create Job"**
3. Leave **"Custom"** as-is (you're using a custom container)
4. Enter a job name

### 3. Configure Resources

Configure the following resources:
- **Node group:** 
- **GPU type:** `A100-80GB`
- **# of GPUs per node:** `1x`
- **Worker:** `4`

### 4. Container Configuration

1. Select **"Custom Container"**
2. Paste your container image name
3. Attach your NGC private registry auth

### 5. Run Command Script

Paste the following script in the **"Run Command"** field:

```bash
set -e
export NCCL_DEBUG=INFO
export NCCL_IGNORE_CPU_AFFINITY=1

# Lepton specific environment variable
SERVICE_PREFIX="${LEPTON_JOB_SERVICE_PREFIX:-$LEPTON_JOB_NAME}"
SUBDOMAIN="${LEPTON_SUBDOMAIN:-$LEPTON_JOB_NAME-job-svc}"
export MASTER_ADDR="${SERVICE_PREFIX}-0.${SUBDOMAIN}"
export THIS_ADDR="${SERVICE_PREFIX}-${LEPTON_JOB_WORKER_INDEX}.${SUBDOMAIN}"
export LOCAL_IP="$(hostname -I | awk '{print $1}')"
export WORLD_SIZE="${LEPTON_JOB_TOTAL_WORKERS}"
export WORLD_RANK="${LEPTON_JOB_WORKER_INDEX}"
export NGPUS="${LEPTON_RESOURCE_ACCELERATOR_NUM}"

# NeMo RL config
CLUSTER_NUM_NODES="${CLUSTER_NUM_NODES:-$WORLD_SIZE}"
CLUSTER_GPU_PER_NODE="${CLUSTER_GPU_PER_NODE:-$NGPUS}"
RAY_HEAD_PORT="${RAY_HEAD_PORT:-6379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
WORKING_DIR="/opt/nemo-rl"
RAY_VENV="/opt/nemo_rl_venv/bin"
SCRIPT_PATH="${SCRIPT_PATH:-}"
CONFIG_PATH="${CONFIG_PATH:-}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"

echo "========================================="
echo "Initializing NeMo RL GRPO setup..."
echo "Batch job configuration"
echo "Lepton Job:          ${LEPTON_JOB_NAME}"
echo "Host:                $(hostname)"
echo "Service Prefix:      ${SERVICE_PREFIX}"
echo "Subdomain:           ${SUBDOMAIN}"
echo "Master Addr:         ${MASTER_ADDR}"
echo "World Size / Rank:   ${WORLD_SIZE} / ${WORLD_RANK}"
echo "Local IP:            ${LOCAL_IP}"
echo "Ray Ports:           ${RAY_HEAD_PORT} / ${RAY_DASHBOARD_PORT}"
echo "Cluster Nodes:       ${CLUSTER_NUM_NODES}"
echo "GPUs per Node:       ${CLUSTER_GPU_PER_NODE}"
echo "Script Path:         ${SCRIPT_PATH}"
echo "Config Path:         ${CONFIG_PATH}"
echo "Checkpoint Dir:      ${CHECKPOINT_DIR}"
echo "Batch job configuration output completed"
echo "========================================="
echo "Performing additional validation checks..."
# Validate required environment variables
missing=()
for var in SCRIPT_PATH CONFIG_PATH CHECKPOINT_DIR; do
  [[ -z "${!var}" ]] && missing+=("$var")
done
if ((${#missing[@]})); then
  echo "ERROR: Missing required env vars:"
  printf '  - %s\n' "${missing[@]}"
  echo "Supply them via --env when launching the job."
  exit 1
fi

# Function to check if this is the master node (rank 0)
is_master_node() {
    echo "Checking node role..."
    echo "World Rank: $WORLD_RANK"
    echo "Lepton Worker Index: ${LEPTON_JOB_WORKER_INDEX}"
    
    if [ "$WORLD_RANK" -eq 0 ]; then
        echo "Detected as MASTER node (Rank $WORLD_RANK)"
        return 0
    fi
    
    echo "✗ Detected as WORKER node (Rank $WORLD_RANK)"
    return 1
}

# Function to check if Ray is already running
is_ray_running() {
    if $RAY_VENV/ray status >/dev/null 2>&1; then
        echo "Ray is already running on this node"
        return 0
    fi
    return 1
}

# Get master IP
echo "Resolving Master IP..."
MASTER_IP=""
while [ -z "$MASTER_IP" ]; do
    MASTER_IP=$(getent hosts -- $MASTER_ADDR | awk '{ print $1 }' || true)
    if [ -z "$MASTER_IP" ]; then
        sleep 5
    fi
done
export MASTER_IP
echo "Master IP: $MASTER_IP"

# Function to wait for Ray head to be ready
wait_for_ray_head() {
    echo "Waiting for Ray head node to be ready..."   
    local max_attempts=60
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if $RAY_VENV/ray status --address=$MASTER_IP:$RAY_HEAD_PORT >/dev/null 2>&1; then
            echo "Ray head node is ready!"
            return 0
        fi
        echo "Attempt $((attempt+1))/$max_attempts: Ray head not ready yet..."
        sleep 5
        attempt=$((attempt+1))
    done
    
    echo "ERROR: Ray head node failed to start within expected time"
    return 1
}

# Function to wait for all worker nodes to connect
wait_for_workers() {
    echo "Ray head node initialization completed...."
    echo "Waiting for all worker nodes to connect..."
    local expected_nodes=$CLUSTER_NUM_NODES
    local max_attempts=100
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        local connected_nodes=$($RAY_VENV/ray status --address=$MASTER_IP:$RAY_HEAD_PORT 2>/dev/null | grep -c "node_" || echo "0")
        echo "Connected nodes: $connected_nodes/$expected_nodes"
        
        if [ "$connected_nodes" -ge "$expected_nodes" ]; then
            echo "Master and all worker nodes successfully connected!"
            return 0
        fi
        
        sleep 15
        attempt=$((attempt+1))
    done
    
    echo "WARNING: Unable to connect master to all worker nodes within expected time, proceeding anyway..."
    return 0
}

# Check if Ray is already running and stop it
if is_ray_running; then
    echo "Stopping existing Ray instance..."
    $RAY_VENV/ray stop --force
    sleep 5
fi

# Adjust NCCL settings based on GPU count
echo "Verifying optimal NCCL_SOCKET_IFNAME setting"
if [[ "${CLUSTER_GPU_PER_NODE}" -ne 8 ]]; then
  echo "8x GPU node configuration NOT detected - unsetting NCCL_SOCKET_IFNAME"
  unset NCCL_SOCKET_IFNAME
fi

# Main execution logic
if is_master_node; then
    echo "Validations completed"
    echo "Master node initializing Ray head node and training"
    
    # Start Ray head node
    echo "Starting Ray head node on $MASTER_IP:$RAY_HEAD_PORT"
    $RAY_VENV/ray start --head \
        --port=$RAY_HEAD_PORT \
        --dashboard-host=0.0.0.0 \
        --dashboard-port=$RAY_DASHBOARD_PORT \
        --resources='{"nrl_tag_ALL": 1}' 
    
    # Wait a bit for head node to fully initialize
    sleep 15
    
    # Wait for worker nodes to connect
    if [ "$CLUSTER_NUM_NODES" -gt 1 ]; then
        wait_for_workers
    fi
    
    # Display Ray cluster status
    echo "Ray cluster status:"
    $RAY_VENV/ray status --address=$MASTER_IP:$RAY_HEAD_PORT
    
    echo "Starting GRPO training..."
    echo "Current directory: $(pwd)"
    
    # Verify the script and config exist
    [[ -f "$SCRIPT_PATH"  ]] || { echo "ERROR: $SCRIPT_PATH not found"; exit 1; }
    [[ -f "$CONFIG_PATH"  ]] || { echo "ERROR: $CONFIG_PATH not found"; exit 1; }
    
    # Create checkpoint directory if it doesn't exist
    mkdir -p "$(dirname "$CHECKPOINT_DIR")"
    
    # Start the training
    echo "Launching GRPO training with the following parameters:"
    echo "  - Script: $SCRIPT_PATH"
    echo "  - Config: $CONFIG_PATH"
    echo "  - Nodes: $CLUSTER_NUM_NODES"
    echo "  - GPUs per node: $CLUSTER_GPU_PER_NODE"
    echo "  - Checkpoint dir: $CHECKPOINT_DIR"
    
    uv run $SCRIPT_PATH --config $CONFIG_PATH \
        cluster.num_nodes=$CLUSTER_NUM_NODES \
        cluster.gpus_per_node=$CLUSTER_GPU_PER_NODE \
        checkpointing.checkpoint_dir=$CHECKPOINT_DIR
        
else
    echo "Validations completed"
    echo "Attempting to connect worker node to Ray head node"
    
    # Wait for Ray head to be ready before connecting
    wait_for_ray_head
    
    # Start Ray worker node
    echo "Starting Ray worker node, connecting to $MASTER_IP:$RAY_HEAD_PORT"
    $RAY_VENV/ray start \
        --address=$MASTER_IP:$RAY_HEAD_PORT \
        --resources='{"worker_units": 1, "nrl_tag_ALL": 1}'
    
    # Keep worker alive
    echo "Worker node connected successfully. Keeping alive..."
    while true; do
        sleep 30
        # Check if Ray is still running
        if ! $RAY_VENV/ray status --address=$MASTER_IP:$RAY_HEAD_PORT >/dev/null 2>&1; then
            echo "Ray head node appears to be down. Exiting..."
            break
        fi
    done
fi

echo "Script execution completed."
```

### 6. Advanced Configuration

Click **"Advanced Configuration"** above the Create and Create with Schedule buttons to add Environment Variables:

| Variable | Value |
|----------|-------|
| `CLUSTER_NUM_NODES` | `4` |
| `CLUSTER_GPU_PER_NODE` | `1` |
| `RAY_DASHBOARD_PORT` | `8625` |
| `SCRIPT_PATH` | `examples/run_grpo_math.py` |
| `CONFIG_PATH` | `examples/configs/grpo_math_8B.yaml` |
| `CHECKPOINT_DIR` | `results/llama8b_4nodes` |
| `HF_TOKEN` | `<YOUR_HF_HUB_TOKEN>` |

### 7. Mount Storage

1. Click **"+ Mount Storage"**
2. Configure:
   - **Volume:** `Static NFS az-files-nfs-vol`
   - **From path:** `/`
   - **Mount path:** `/mnt`

### 8. Launch Job

Click **"Create"** to launch your job.

### 9. Verify Logs

Monitor the job execution through the provided logging interface.

## Troubleshooting

### Common Issues

1. **Container Build Failures:** Ensure sufficient memory allocation for Docker build process
2. **NGC Authentication:** Verify your NGC API token is valid and has appropriate permissions
3. **Ray Cluster Issues:** Check network connectivity between nodes and verify Ray ports are accessible
4. **Environment Variables:** Ensure all required environment variables are properly set

### Log Analysis

- Monitor Ray cluster status through the dashboard
- Check NCCL debug output for network-related issues
- Verify GPU utilization and memory usage
- Review training logs for convergence and performance metrics

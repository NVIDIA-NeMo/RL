<div align="center">

# DGXC Create (Run:ai) NeMo-RL GRPO Training

</div>

# Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Git Clone NeMo-RL Repo](#git-clone-nemo-rl-repo)
4. [Create a Run:ai Data Source (PVC)](#create-a-runai-data-source-ie-persistent-volume-claim-pvc)
   - [Capture your PVC's Kubernetes name](#capture-your-pvcs-kubernetes-name)
5. [Create Run:ai Credential](#create-runai-credential)
6. [Launch a Workspace with PVC Attached](#launch-a-workspace-with-pvc-attached)
7. [SSH into Your Workspace and Create Launch Script](#ssh-into-your-workspace-and-create-launch-script)
   - [Create the GRPO Launch Script](#create-the-grpo-launch-script)
   - [Make Script Executable](#make-script-executable)
   - [Verify Script Creation](#verify-script-creation)
8. [Launch NeMo RL Container](#launch-nemo-rl-container)
   - [Export HuggingFace API Key](#export-huggingface-api-key)
   - [Submit Multi-Node PyTorch Training Job](#submit-multi-node-pytorch-training-job)
   - [Verify Training Job Launch](#verify-training-job-launch)
   - [View Logs](#view-logs)

# Overview
This guide outlines the steps needed to perform GRPO on a DGX Create (Run:ai) Cluster

# Prerequisites
A DGXC Create (Run:ai) cluster with kubectl, kubeconfig, Run:ai CLI, users, and projects set up
A NGC account with private registry and API key
Docker or similar OCI compliant runtime

## Git clone NeMo-RL repo
```bash
## Change to desired folder directory
cd <FOLDER_TO_CLONE_NEMO_RL_REPO>

## Git clone NeMo RL repo
git clone https://github.com/NVIDIA-NeMo/RL.git nemo-rl

## Change directory to the Dockerfile
cd nemo-rl/docker/

## Build your image
docker buildx build --target release -t nemo-rl-release --platform linux/amd64 -f Dockerfile ..

## Tag your image to the NGC private registry                                                                                      
docker tag nemo-rl-release nvcr.io/<INSERT_NGC_ORG_NAME>/nemo-rl-release:latest

## Authenticae with nvcr.io
docker login nvcr.io
Username: $oauthtoken
Password: <NGC_API_TOKEN>

## Push your image to your NGC private registry
docker push nvcr.io/<INSERT_NGC_ORG_NAME>/nemo-rl-release1:latest
```    

## Create a Run:ai Data source i.e. Persistent Volume Claim (PVC)
<img width="494" height="473" alt="Screenshot 2025-07-11 at 2 09 24 PM" src="https://github.com/user-attachments/assets/629ce609-b19f-4d0b-981d-cb13ad3a5527" />

1. Navigate to the **Data sources** page illustrated in the image above (steps #1 - #3).
2. Click + NEW DATA SOURCE and select PVC from the menu. You will be taken to the PVC creation page.
3. Select the Scope for your PVC at the project level.
4. Enter a name for the PVC.
5. Select the New PVC radio button.
6. Select Storage class based on CSP i.e. AWS: Create a PVC w/ storage class dgxc-enterprise-file. GCP: Create a PVC w/ storage class zonal-rwx. 
7. Select Access mode Read-write by many nodes
8. Enter a Claim size with Units  
9. Leave Volume mode Filesystem as-is
10. Enter preferred container mount path e.g. /data, /home, etc.
11. Click on CREATE DATA SOURCE

Capture your PVC's Kuberenetes name via user interface:
<img width="673" height="175" alt="Screenshot 2025-07-29 at 1 44 42 PM" src="https://github.com/user-attachments/assets/c656a16b-b868-48c2-aab1-8f9658ef7963" />

1. Navigate to **Data sources** page

Capture your PVC's Kuberenetes name via kubectl:
```bash

kubectl get pvc

Claim Name                Run:AI Managed  
───────────────────────────────────────────
pvc1-name-project         Yes             
pvc2-name-project         Yes   
```
## Create Run:ai Credential
<img width="476" height="414" alt="Screenshot 2025-07-29 at 1 50 48 PM" src="https://github.com/user-attachments/assets/3b12ae62-3235-4a41-9e0b-60cb2aa85837" />

1. Navigate to the **Credentials** page illustrated in the image above (steps #1 - #3).
2. Click + NEW CREDENTIALS and select Docker registry from the drop down menu. You will be taken to the New credential creation page.
3. Select the Scope for your new NGC credential.
4. Enter a name for the credential.
5. Select the New secret radio button.
6. For username, use $oauthtoken.
7. For password, paste your NGC Personal API token.
8. Under Docker Registry URL, enter nvcr.io.
9. Click CREATE CREDENTIALS. Your credentials will now be saved in the cluster and shall be used when you pull a container from your private registry.

Refer to [NVIDIA documentation](https://docs.nvidia.com/dgx-cloud/run-ai/latest/user-guide.html#setting-up-credentials-for-accessing-registries-and-data) for more information

Based on your outline and details, here's the remainder of the instructions in markdown format:

## Launch a Workspace with PVC Attached

If you prefer to use the Run:ai UI, follow the [Interactive Workload Examples](https://docs.nvidia.com/dgx-cloud/run-ai/latest/interactive-examples.html#interactive-workload-examples) to launch a NeMo container with Jupyter Labs. 

To create a workspace with a PVC attached using the CLI, set up the GRPO launch script using the CLI

```bash

runai workspace submit -i nvcr.io/nvidia/nemo:25.02 \
    --existing-pvc claimname=<CLAIM_NAME>,path=<PATH> \
    --command -- sleep infinity
```

Output:
```
Creating workspace <WORKSPACE_NAME>...
To track the workload's status, run 'runai workspace describe <WORKSPACE_NAME>'
```

## SSH into Your Workspace and Create a GRPO Launch Script

Access your workspace and create the GRPO launch script:

```bash
## Using kubectl
kubectl exec -it <POD_NAME> -- bash

## Using Runai CLI
runai workspace bash <WORKSPACE_NAME>
```

### Create the GRPO Launch Script
>**Note:** If you're following along from the UI, copy the script from "#!/bin/bash" to "echo Script execution completed."

Update the PVC mount path in the command below and create the launch script:

```bash
cat > /<PVC_MOUNT_PATH>/launch_grpo.sh << 'EOF'
#!/bin/bash
# launch_grpo.sh - Automated GRPO training setup with configurable environment variables

set -euo pipefail

#!/usr/bin/env bash
set -euo pipefail

CLUSTER_NUM_NODES=${CLUSTER_NUM_NODES:-2} # Set # of nodes for training
CLUSTER_GPU_PER_NODE=${CLUSTER_GPU_PER_NODE:-8} # Set # of GPUs per node
SCRIPT_PATH=${SCRIPT_PATH} # Set script path
CONFIG_PATH=${CONFIG_PATH} # Set config path
WORKING_DIR="/opt/nemo-rl"
RAY_VENV="/opt/nemo_rl_venv/bin"
RAY_HEAD_PORT=${MASTER_PORT:-6379} # Ray head port
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:=8265} # Ray dashboard port


# Function to check if this is the master node
is_master_node() {
    local hostname=$(hostname)
    echo "Checking node..."
    echo "Local Address: $hostname"
    echo "Master Address: $MASTER_ADDR"
    
    # We default to a worker role (rank 1) if not set.
    if [[ "${RANK:-1}" -eq 0 ]]; then
        echo "✓ Detected as Master (RANK=0)"
        return 0
    else
        echo "✗ Detected as Worker (RANK=${RANK:-unspecified})"
        return 1
    fi
}

# Function to check if Ray is already running
is_ray_running() {
    if $RAY_VENV/ray status >/dev/null 2>&1; then
        echo "Ray is already running on this node"
        return 0
    fi
    return 1
}

# Function to wait for Ray head to be ready
wait_for_ray_head() {
    echo "Waiting for Ray head node to be ready..."
    local max_attempts=60
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if $RAY_VENV/ray status --address=$MASTER_ADDR:$RAY_HEAD_PORT >/dev/null 2>&1; then
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
    echo "Waiting for all worker nodes to connect..."
    local expected_nodes=$CLUSTER_NUM_NODES
    local max_attempts=30
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        # Corrected line
        local connected_nodes
        connected_nodes=$($RAY_VENV/ray status --address="$MASTER_ADDR:$RAY_HEAD_PORT" 2>/dev/null | (grep "node_" || true) | wc -l)
        echo "Connected nodes: $connected_nodes/$expected_nodes"
        
        if [ "$connected_nodes" -ge "$expected_nodes" ]; then
            echo "All worker nodes connected!"
            return 0
        fi
        echo "Attempt $((attempt+1))/$max_attempts: Checking in 15 seconds..."
        sleep 15
        attempt=$((attempt+1))
    done
    
    echo "WARNING: Not all worker nodes connected within expected time, proceeding anyway..."
    exit 1
}

# Check if Ray is already running and stop it
if is_ray_running; then
    echo "Stopping existing Ray instance..."
    $RAY_VENV/ray stop --force
    sleep 5
fi

# Main execution logic
if is_master_node; then
    
    # Start Ray head node
    echo "Starting Ray head node on $MASTER_ADDR:$RAY_HEAD_PORT"
    $RAY_VENV/ray start --head \
        --port=$RAY_HEAD_PORT \
        --dashboard-host=0.0.0.0 \
        --dashboard-port=$RAY_DASHBOARD_PORT \
        --resources='{"nrl_tag_ALL": 1}' 
    
    # Wait for worker nodes to connect
    wait_for_workers
    
    # Display Ray cluster status
    echo "Ray cluster status:"
    $RAY_VENV/ray status --address=$MASTER_ADDR:$RAY_HEAD_PORT
    
    # Verify the script and config exist
    if [ ! -f "$SCRIPT_PATH" ]; then
        echo "ERROR: Script not found at $SCRIPT_PATH"
        echo "Contents of examples directory:"
        ls -la examples/ || echo "directory not found"
        exit 1
    fi
    
    if [ ! -f "$CONFIG_PATH" ]; then
        echo "ERROR: Config not found at $CONFIG_PATH"
        echo "Contents of examples/configs directory:"
        ls -la examples/configs/ || echo "examples/configs directory not found"
        exit 1
    fi
    
    # Start the training
    echo "Executing GRPO training script..."
    
    uv run $SCRIPT_PATH --config $CONFIG_PATH policy.model_name="/nemo-workspace/models/deepseek-r1-distill-llama-70b" cluster.num_nodes=8
        
else
    
    # Wait for Ray head to be ready before connecting
    wait_for_ray_head
    
    # Start Ray worker node
    echo "Starting Ray worker... Connecting to $MASTER_ADDR:$RAY_HEAD_PORT"
    $RAY_VENV/ray start --address="$MASTER_ADDR:$RAY_HEAD_PORT"
    
    # Worker node started successfully. Now, wait indefinitely.
    # This keeps the container alive. If the container is killed, this process stops.
    echo "Worker node connected. Tailing /dev/null to keep process alive..."
    tail -f /dev/null
fi

echo "Script execution completed."

EOF
```

### Make Script Executable

**IMPORTANT:** Make sure to run the following command to make your script executable upon container initialization:

```bash

chmod +x /<PVC_MOUNT_PATH>/launch_grpo.sh
```

### Verify Script Creation

Verify the script has been created successfully:

```bash

cd <PVC_MOUNT_PATH>
ls -l
```

## Launch NeMo RL Container

### Export HuggingFace API Key

```bash

export HF_TOKEN=<YOUR_API_KEY>
echo $HF_TOKEN
```

### Submit Multi-Node PyTorch Training Job

Submit a Run:ai multi-node PyTorch training job to initialize the NeMo RL container:

> **NOTE:** `--workers 1` will use 2 nodes total (1 master + 1 worker)

```bash
runai training pytorch submit <NEMO_RL_TRAINING_JOB_NAME> \
    -p <PROJECT_NAME> \
    -g 8 --workers 1 \
    -i nvcr.io/<INSERT_NGC_ORG_ID>/nemo-rl-release1:25.05 \
    --existing-pvc claimname=<RUNAI_PVC_NAME>,path=<MOUNT_PATH> \
    -e HF_TOKEN=$HF_TOKEN \
    -e CLUSTER_NUM_NODES=2 \
    -e CLUSTER_GPU_PER_NODE=8 \
    -e SCRIPT_PATH="./examples/run_grpo_math.py" \
    -e CONFIG_PATH="examples/configs/grpo_math_8B.yaml"
    --command -- /<PVC_MOUNT_PATH>/launch_grpo.sh
```

### Verify Training Job Launch

Check that your training job and pods have launched successfully:

```bash
runai training pytorch describe <NEMO_RL_TRAINING_JOB_NAME>
```

Or alternatively:

```bash
kubectl get pods
```

### View Logs

You can view training logs through either the UI or CLI:

**UI Method:**
1. Navigate to Workload Manager
2. Go to Workloads
3. Click on your workload
4. Select "Show Details"
5. Click "Logs"

**kubectl Method:**
```bash
kubectl logs -f <POD_NAME>
```
                         
**CLI Method:**
```bash
runai training pytorch logs <TRAINING_JOB_NAME>
```

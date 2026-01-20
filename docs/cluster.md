# Set Up Clusters

This guide explains how to run NeMo RL with Ray on Slurm or Kubernetes.

## Use Slurm for Batched and Interactive Jobs

 The following code provides instructions on how to use Slurm to run batched job submissions and run jobs interactively.

### Batched Job Submission

```sh
# Run from the root of NeMo RL repo
NUM_ACTOR_NODES=1  # Total nodes requested (head is colocated on ray-worker-0)

COMMAND="uv run ./examples/run_grpo_math.py" \
CONTAINER=YOUR_CONTAINER \
MOUNTS="$PWD:$PWD" \
sbatch \
    --nodes=${NUM_ACTOR_NODES} \
    --account=YOUR_ACCOUNT \
    --job-name=YOUR_JOBNAME \
    --partition=YOUR_PARTITION \
    --time=1:0:0 \
    --gres=gpu:8 \
    ray.sub
```

> [!TIP]
> Depending on your Slurm cluster configuration, you may or may not need to include the `--gres=gpu:8` option in the `sbatch` command.

> [!NOTE]
> For GB200 systems with 4 GPUs per node, use `--gres=gpu:4` instead of `--gres=gpu:8`.

Upon successful submission, Slurm will print the `SLURM_JOB_ID`:
```text
Submitted batch job 1980204
```
Make a note of the job submission number. Once the job begins, you can track its process in the driver logs which you can `tail`:
```sh
tail -f 1980204-logs/ray-driver.log
```

### Interactive Launching

> [!TIP]
> A key advantage of running interactively on the head node is the ability to execute multiple multi-node jobs without needing to requeue in the Slurm job queue. This means that during debugging sessions, you can avoid submitting a new `sbatch` command each time. Instead, you can debug and re-submit your NeMo RL job directly from the interactive session.

To run interactively, launch the same command as [Batched Job Submission](#batched-job-submission), but omit the `COMMAND` line:
```sh
# Run from the root of NeMo RL repo
NUM_ACTOR_NODES=1  # Total nodes requested (head is colocated on ray-worker-0)

CONTAINER=YOUR_CONTAINER \
MOUNTS="$PWD:$PWD" \
sbatch \
    --nodes=${NUM_ACTOR_NODES} \
    --account=YOUR_ACCOUNT \
    --job-name=YOUR_JOBNAME \
    --partition=YOUR_PARTITION \
    --time=1:0:0 \
    --gres=gpu:8 \
    ray.sub
```

> [!NOTE]
> For GB200 systems with 4 GPUs per node, use `--gres=gpu:4` instead.

Upon successful submission, Slurm will print the `SLURM_JOB_ID`:
```text
Submitted batch job 1980204
```
Once the Ray cluster is up, a script will be created to attach to the Ray head node. Run this script to launch experiments:
```sh
bash 1980204-attach.sh
```
Now that you are on the head node, you can launch the command as follows:
```sh
uv run ./examples/run_grpo_math.py
```

### Slurm Environment Variables

All Slurm environment variables described below can be added to the `sbatch`
invocation of `ray.sub`. For example, `GPUS_PER_NODE=8` can be specified as follows:

```sh
GPUS_PER_NODE=8 \
... \
sbatch ray.sub \
   ...
```
#### Common Environment Configuration
``````{list-table}
:header-rows: 1

* - Environment Variable
  - Explanation
* - `CONTAINER`
  - (Required) Specifies the container image to be used for the Ray cluster.
    Use either a docker image from a registry or a squashfs (if using enroot/pyxis).
* - `MOUNTS`
  - (Required) Defines paths to mount into the container. Examples:
    ```md
    * `MOUNTS="$PWD:$PWD"` (mount in current working directory (CWD))
    * `MOUNTS="$PWD:$PWD,/nfs:/nfs:ro"` (mounts the current working directory and `/nfs`, with `/nfs` mounted as read-only)
    ```
* - `COMMAND`
  - Command to execute after the Ray cluster starts. If empty, the cluster idles and enters interactive mode (see the [Slurm interactive instructions](#interactive-launching)).
* - `HF_HOME`
  - Sets the cache directory for huggingface-hub assets (e.g., models/tokenizers).
* - `WANDB_API_KEY`
  - Setting this allows you to use the wandb logger without having to run `wandb login`.
* - `HF_TOKEN`
  - Setting the token used by huggingface-hub. Avoids having to run the `huggingface-cli login`
* - `HF_DATASETS_CACHE`
  - Sets the cache dir for downloaded Huggingface datasets.
``````

> [!TIP]
> When `HF_TOKEN`, `WANDB_API_KEY`, `HF_HOME`, and `HF_DATASETS_CACHE` are set in your shell environment using `export`, they are automatically passed to `ray.sub`. For instance, if you set:
>
> ```sh
> export HF_TOKEN=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
> ```
> this token will be available to your NeMo RL run. Consider adding these exports to your shell configuration file, such as `~/.bashrc`.

#### Advanced Environment Configuration
``````{list-table}
:header-rows: 1

* - Environment Variable
    (and default)
  - Explanation
* - `UV_CACHE_DIR_OVERRIDE`
  - By default, this variable does not need to be set. If unset, `ray.sub` uses the 
    `UV_CACHE_DIR` defined within the container (defaulting to `/root/.cache/uv`). 
    `ray.sub` intentionally avoids using the `UV_CACHE_DIR` from the user's host 
    environment to prevent the host's cache from interfering with the container's cache. 
    Set `UV_CACHE_DIR_OVERRIDE` if you have a customized `uv` environment (e.g., 
    with pre-downloaded packages or specific configurations) that you want to persist 
    and reuse across container runs. This variable should point to a path on a shared 
    filesystem accessible by all nodes (head and workers). This path will be mounted 
    into the container and will override the container's default `UV_CACHE_DIR`.
* - `CPUS_PER_WORKER=128`
  - CPUs each Ray worker node claims. Default is `16 * GPUS_PER_NODE`.
* - `GPUS_PER_NODE=8`
  - Number of GPUs each Ray worker node claims. To determine this, run `nvidia-smi` on a worker node.
* - `BASE_LOG_DIR=$SLURM_SUBMIT_DIR`
  - Base directory for storing Ray logs. Defaults to the Slurm submission directory ([SLURM_SUBMIT_DIR](https://slurm.schedmd.com/sbatch.html#OPT_SLURM_SUBMIT_DIR)).
* - `NODE_MANAGER_PORT=53001`
  - Port for the Ray node manager on worker nodes.
* - `OBJECT_MANAGER_PORT=53003`
  - Port for the Ray object manager on worker nodes.
* - `RUNTIME_ENV_AGENT_PORT=53005`
  - Port for the Ray runtime environment agent on worker nodes.
* - `DASHBOARD_AGENT_GRPC_PORT=53007`
  - gRPC port for the Ray dashboard agent on worker nodes.
* - `METRICS_EXPORT_PORT=53009`
  - Port for exporting metrics from worker nodes.
* - `PORT=6379`
  - Main port for the Ray head node.
* - `RAY_CLIENT_SERVER_PORT=10001`
  - Port for the Ray client server on the head node.
* - `DASHBOARD_GRPC_PORT=52367`
  - gRPC port for the Ray dashboard on the head node.
* - `DASHBOARD_PORT=8265`
  - Port for the Ray dashboard UI on the head node. This is also the port
    used by the Ray distributed debugger.
* - `DASHBOARD_AGENT_LISTEN_PORT=52365`
  - Listening port for the dashboard agent on the head node.
* - `MIN_WORKER_PORT=54001`
  - Minimum port in the range for Ray worker processes.
* - `MAX_WORKER_PORT=54257`
  - Maximum port in the range for Ray worker processes.
``````

> [!NOTE]
> For the most part, you will not need to change ports unless these
> are already taken by some other service backgrounded on your cluster.

## Kubernetes

This guide outlines the process of migrating NemoRL training jobs from a Slurm environment to a Kubernetes cluster utilizing Ray orchestration and NVIDIA GPUs.

---

## Prerequisites

Before beginning, ensure the following requirements are met:

* **Cluster Access:** You must have access to the K8s cluster from a client machine via `kubectl`.

> [!IMPORTANT]
> **Authentication Required**:
> Simply installing `kubectl` on your local machine is not sufficient. You must work with your **Infrastructure Administrator** to obtain a valid `KUBECONFIG` file (usually placed at `~/.kube/config`) or authentication token. This file contains the endpoint and credentials required to connect your local client to the specific remote GPU cluster.
> 
* **Operators:** The cluster must have the [**NVIDIA Operator**](https://github.com/NVIDIA/gpu-operator) (for GPU provisioning) and the [**KubeRay Operator**](https://github.com/ray-project/kuberay) (for Ray Cluster lifecycle management) installed.
* **Registry Access:** Ability to push/pull Docker images to a registry (e.g., nvcr.io or Docker Hub).

### 1. Test Cluster Access
Verify your connection and operator status:

```bash
kubectl get pods -o wide -w
```

### 2. Build and Push the Docker Container
We will use the NVIDIA cloud registry (`nvcr.io`) for this guide. From your client machine:

**Login to the Registry**
```bash
# Set up Docker and nvcr.io with your NGC_API_KEY
docker login nvcr.io

# Username: $oauthtoken
# Password: <NGC_API_KEY>
```

**Build and Push**
Clone the NemoRL repository and build the container.

```bash
# Clone recursively
git clone [https://github.com/NVIDIA-NeMo/RL](https://github.com/NVIDIA-NeMo/RL) --recursive
cd RL

# If you already cloned without --recursive, update submodules:
git submodule update --init --recursive

# Set your organization
export NGC_ORG=<YOUR_NGC_ORG>

# Self-contained build (default: builds from main)
docker buildx build --target release -f docker/Dockerfile --tag nvcr.io/${NGC_ORG}/nemo-rl:latest --push .
```

---

## Phase 1: Infrastructure Setup

### 1. Configure Shared Storage (NFS)
This tutorial uses a NFS-based `ReadWriteMany` volume to ensure the Head node and Worker nodes see the exact same files (code, data, checkpoints). This prevents "File Not Found" errors.

> **Note:** This is a cluster-wide resource. If your admin has already provided an NFS storage class, you only need to create this PVC once.

**File:** `shared-pvc.yaml`

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: nemo-shared-workspace
spec:
  accessModes:
    - ReadWriteMany     # Critical: Allows RW access from multiple nodes
  storageClassName: nfs-client
  resources:
    requests:
      storage: 2Ti      # Adjust based on dataset and model size
```

**Apply the configuration:**
```bash
kubectl apply -f shared-pvc.yaml
```

### 2. Create Registry Secret
This secret allows the cluster to pull the private image you built earlier.

```bash
kubectl create secret docker-registry nvcr-secret \
  --docker-server=nvcr.io \
  --docker-username='$oauthtoken' \
  --docker-password=YOUR_NGC_API_KEY_HERE \
  --docker-email=admin@example.com
```

---

## Phase 2: Ray Cluster Configuration

We will create a Ray cluster with **1x Head node** and **1x Worker node** (with 8x GPUs each).

**Key Configuration Notes:**
* **Networking:** Uses `bond0` to bypass virtual ethernet overhead (check with your admin regarding the correct interface for NCCL).
* **Memory:** Disables Ray's OOM killer to prevent false positives.
* **Caching:** Redirects HuggingFace cache to the shared PVC.
* **Version Match:** The `rayVersion` spec must match the version in `RL/pyproject.toml`. Check this example [version snapshot](https://github.com/NVIDIA-NeMo/RL/blob/b2e4265d4f2424c0467691f2f0f864cdebe1ab0f/pyproject.toml#L25).
* **Container image:** Replace the image name `nvcr.io/nvidian/nemo-rl:latest` with your actual image, e.g., `nvcr.io/YOUR_NGC_ORG/nemo-rl:latest`.

> [!WARNING]
> **Check Your Node Capacity & Resource Limits**
> The resource requests in the manifest below (e.g., `cpu: "128"`, `memory: "1500Gi"`) are configured for high-end H100 nodes. If these numbers exceed your physical node's available capacity, your pods will remain in a **Pending** state indefinitely.
>
> Additionally, the shared memory volume is backed by actual node RAM:
> ```yaml
> volumes:
>   - name: dshm
>     emptyDir:
>       medium: Memory
>       sizeLimit: "1000Gi" # Counts against Node RAM
> ```
> You must ensure your physical node has enough memory to cover the container `requests` **plus** the `sizeLimit` of this volume. Please adjust these values to match your specific hardware compute shape.

**File:** `nemo-rl-h100.yaml`

```yaml
apiVersion: ray.io/v1
kind: RayCluster
metadata:
  name: nemo-h100-cluster
spec:
  rayVersion: '2.49.2'

  ######################
  # HEAD NODE (Uniform with Workers)
  ######################
  headGroupSpec:
    rayStartParams:
      dashboard-host: '0.0.0.0'
      block: 'true' 
      num-gpus: "8"
    template:
      spec:
        imagePullSecrets:
          - name: nvcr-secret
        
        hostNetwork: true 
        dnsPolicy: ClusterFirstWithHostNet

        tolerations:
          - key: "nvidia.com/gpu"
            operator: "Exists"
            effect: "NoSchedule"
        
        containers:
        - name: ray-head
          image: nvcr.io/nvidian/nemo-rl:latest
          imagePullPolicy: Always
          resources:
            limits:
              nvidia.com/gpu: 8 
              cpu: "128"
              memory: "1500Gi"
            requests:
              nvidia.com/gpu: 8
              cpu: "128"
              memory: "1500Gi"
          env:
            - name: NVIDIA_VISIBLE_DEVICES
              value: "all"
             # IMPORTANT: Verify the correct network interface with your cluster admin
             # Common values: bond0, eth0, ib0 (for InfiniBand)
             # Run 'ip addr' or 'ifconfig' on a node to identify available interfaces
            - name: NCCL_SOCKET_IFNAME
              value: bond0
            - name: NCCL_SHM_DISABLE
              value: "0"
            - name: RAY_memory_monitor_refresh_ms
              value: "0"
            - name: HF_HOME
              value: "/shared/huggingface"
          volumeMounts:
            # All code and data now live here
            - mountPath: /shared
              name: shared-vol
            - mountPath: /dev/shm
              name: dshm
        volumes:
          - name: shared-vol
            persistentVolumeClaim:
              claimName: nemo-shared-workspace
          - name: dshm
            emptyDir:
              medium: Memory
              sizeLimit: "1000Gi"

  ##########################
  # WORKER NODES (H100)
  ##########################
  workerGroupSpecs:
  - replicas: 1
    minReplicas: 1
    maxReplicas: 1
    groupName: gpu-group-h100
    rayStartParams:
      block: 'true'
      num-gpus: "8"
    template:
      spec:
        imagePullSecrets:
          - name: nvcr-secret
        
        hostNetwork: true 
        dnsPolicy: ClusterFirstWithHostNet
        
        affinity:
          podAntiAffinity:
            requiredDuringSchedulingIgnoredDuringExecution:
            - labelSelector:
                matchExpressions:
                - key: ray.io/node-type
                  operator: In
                  values: ["worker", "head"]
              topologyKey: "kubernetes.io/hostname"

        containers:
        - name: ray-worker
          image: nvcr.io/nvidian/nemo-rl:latest
          imagePullPolicy: Always
          resources:
            limits:
              nvidia.com/gpu: 8 
              cpu: "128"
              memory: "1500Gi"
            requests:
              nvidia.com/gpu: 8
              cpu: "128"
              memory: "1500Gi"
          env:
             # IMPORTANT: Verify the correct network interface with your cluster admin
             # Common values: bond0, eth0, ib0 (for InfiniBand)
             # Run 'ip addr' or 'ifconfig' on a node to identify available interfaces
            - name: NCCL_SOCKET_IFNAME
              value: bond0
            - name: NCCL_SHM_DISABLE
              value: "0"
            - name: RAY_memory_monitor_refresh_ms
              value: "0"
            - name: HF_HOME
              value: "/shared/huggingface"
          volumeMounts:
            - mountPath: /shared
              name: shared-vol
            - mountPath: /dev/shm
              name: dshm
        
        tolerations:
          - key: "nvidia.com/gpu"
            operator: "Exists"
            effect: "NoSchedule"
        volumes:
          - name: shared-vol
            persistentVolumeClaim:
              claimName: nemo-shared-workspace
          - name: dshm
            emptyDir:
              medium: Memory
              sizeLimit: "1000Gi"

```

**Cluster Management Commands:**

* **Startup:** `kubectl create -f nemo-rl-h100.yaml`
* **Shutdown:** `kubectl delete -f nemo-rl-h100.yaml`

---

## Phase 3: Run Sample NemoRL Workloads

Once the cluster is running, you can interact with the Ray head node to submit jobs.

### 1. Access the Head Node
```bash
kubectl exec -it $(kubectl get pod -l ray.io/node-type=head -o jsonpath='{.items[0].metadata.name}') -- /bin/bash
```

### 2. Setup Code on Shared Volume
Inside the pod, clone the code to the shared PVC (`/shared`). This ensures workers can see the code.

```bash
cd /shared
git clone [https://github.com/NVIDIA-NeMo/RL](https://github.com/NVIDIA-NeMo/RL) --recursive
cd RL
git submodule update --init --recursive
```

### 3. Submit a Job
Move to the code directory, edit your configuration, and run the job.

```bash
cd /shared/RL

# Edit config (e.g., paths, model config)
vim examples/configs/grpo_math_1B.yaml 

# Set environment variables
export HF_TOKEN=...
export WANDB_API_KEY=...

# Run the job
uv run examples/run_grpo_math.py \
  --config examples/configs/grpo_math_1B.yaml
```

### 4. Configuration Adjustments
To run across multiple nodes, or to ensure logs/checkpoints persist, update your YAML config file (`examples/configs/grpo_math_1B.yaml`):

**Cluster Size:**
```yaml
cluster:
  gpus_per_node: 8
  num_nodes: 2
```

**Logging & Checkpointing:**
Redirect these to `/shared` so they persist after the pod is deleted.

```yaml
checkpointing:
  enabled: true
  checkpoint_dir: "/shared/results/grpo"

# ...

logger:
  log_dir: "/shared/logs"  # Base directory for all logs
  wandb_enabled: true
  wandb:
    project: "grpo-dev"
    name: "grpo-dev-logger"
```

### 5. Monitoring
* **Console:** Watch job progress directly in the terminal where you ran `uv run`.
* **WandB:** If enabled, check the Weights & Biases web interface.

---

## Utility: PVC Busybox Helper

Use a lightweight "busybox" pod to inspect the PVC or copy data in/out without spinning up a heavy GPU node.

**Create the Busybox Pod:**

```bash
# Variables
PVC_NAME=nemo-shared-workspace
MOUNT_PATH=/shared

kubectl create -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: nemo-workspace-busybox
spec:
  containers:
  - name: busybox
    image: busybox
    command: ["sleep", "infinity"]
    volumeMounts:
    - name: workspace
      mountPath: ${MOUNT_PATH}
  volumes:
  - name: workspace
    persistentVolumeClaim:
      claimName: ${PVC_NAME}
EOF
```

**Usage:**

* **Inspect files:**
    ```bash
    kubectl exec -it nemo-workspace-busybox -- sh
    # inside the pod:
    ls /shared/results/grpo/
    ```

* **Copy data (Local -> PVC):**
    ```bash
    kubectl cp ./my-nemo-code nemo-workspace-busybox:/shared/
    ```

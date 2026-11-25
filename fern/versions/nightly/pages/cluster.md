---
title: "Set Up Clusters"
---
This guide explains how to run NeMo RL with Ray on Slurm or Kubernetes.

## Use Slurm for Batched and Interactive Jobs

 The following code provides instructions on how to use Slurm to run batched job submissions and run jobs interactively.

### Batched Job Submission

```sh
# Run from the root of NeMo RL repo
NUM_ACTOR_NODES=1  # Total nodes requested (head is colocated on ray-worker-0)

COMMAND="uv run ./examples/run_grpo.py" \
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
uv run ./examples/run_grpo.py
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
<table>
  <thead>
    <tr>
      <th>Environment Variable</th>
      <th>Explanation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>CONTAINER</code></td>
      <td>(Required) Specifies the container image to be used for the Ray cluster.<br />Use either a docker image from a registry or a squashfs (if using enroot/pyxis).</td>
    </tr>
    <tr>
      <td><code>MOUNTS</code></td>
      <td>(Required) Defines paths to mount into the container. Examples:<br /><code>MOUNTS="$PWD:$PWD"</code> (mount in current working directory (CWD))<br /><code>MOUNTS="$PWD:$PWD,/nfs:/nfs:ro"</code> (mounts the current working directory and <code>/nfs</code>, with <code>/nfs</code> mounted as read-only)</td>
    </tr>
    <tr>
      <td><code>COMMAND</code></td>
      <td>Command to execute after the Ray cluster starts. If empty, the cluster idles and enters interactive mode (see the [Slurm interactive instructions](#interactive-launching)).</td>
    </tr>
    <tr>
      <td><code>HF_HOME</code></td>
      <td>Sets the cache directory for huggingface-hub assets (e.g., models/tokenizers).</td>
    </tr>
    <tr>
      <td><code>WANDB_API_KEY</code></td>
      <td>Setting this allows you to use the wandb logger without having to run <code>wandb login</code>.</td>
    </tr>
    <tr>
      <td><code>HF_TOKEN</code></td>
      <td>Setting the token used by huggingface-hub. Avoids having to run the <code>hf auth login</code></td>
    </tr>
    <tr>
      <td><code>HF_DATASETS_CACHE</code></td>
      <td>Sets the cache dir for downloaded Huggingface datasets.</td>
    </tr>
  </tbody>
</table>

> [!TIP]
> When `HF_TOKEN`, `WANDB_API_KEY`, `HF_HOME`, and `HF_DATASETS_CACHE` are set in your shell environment using `export`, they are automatically passed to `ray.sub`. For instance, if you set:
>
> ```sh
> export HF_TOKEN=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
> ```
> this token will be available to your NeMo RL run. Consider adding these exports to your shell configuration file, such as `~/.bashrc`.

#### Advanced Environment Configuration
<table>
  <thead>
    <tr>
      <th>Environment Variable<br />(and default)</th>
      <th>Explanation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>UV_CACHE_DIR_OVERRIDE</code></td>
      <td>By default, this variable does not need to be set. If unset, <code>ray.sub</code> uses the<br /><code>UV_CACHE_DIR</code> defined within the container (<code>/opt/nemo_rl_cache/uv</code> in images built from <code>docker/Dockerfile</code>; set <code>CONTAINER_UV_CACHE_DIR</code> for a container that uses another path).<br /><code>ray.sub</code> intentionally avoids using the <code>UV_CACHE_DIR</code> from the user&#x27;s host<br />environment to prevent the host&#x27;s cache from interfering with the container&#x27;s cache.<br />Set <code>UV_CACHE_DIR_OVERRIDE</code> if you have a customized <code>uv</code> environment (e.g.,<br />with pre-downloaded packages or specific configurations) that you want to persist<br />and reuse across container runs. This variable should point to a path on a shared<br />filesystem accessible by all nodes (head and workers). This path will be mounted<br />into the container and will override the container&#x27;s default <code>UV_CACHE_DIR</code>.</td>
    </tr>
    <tr>
      <td><code>CPUS_PER_WORKER</code></td>
      <td>CPUs each Ray worker node claims. If unset, <code>ray.sub</code> auto-detects this from<br />Slurm (the <code>CPUTot</code> of the first allocated node) so that Ray sees every CPU<br />on the node. Detection assumes a homogeneous allocation (all nodes have the<br />same CPU count) and only queries a single node to avoid hammering the Slurm<br />controller with one RPC per node; set this explicitly for a heterogeneous<br />allocation or to override detection.</td>
    </tr>
    <tr>
      <td><code>GPUS_PER_NODE=8</code></td>
      <td>Number of GPUs each Ray worker node claims. To determine this, run <code>nvidia-smi</code> on a worker node.</td>
    </tr>
    <tr>
      <td><code>DEDICATED_RAY_HEAD=0</code></td>
      <td>Set to <code>1</code> to run the first allocated node as a GPU-free Ray head: it hosts<br />only the Ray control plane and the driver, advertises <code>--num-gpus=0</code>, and<br />carries neither <code>worker_units</code> nor the topology resources, so no GPU actor<br />or topology-constrained placement group can land on it. Use this when driver<br />or head-node memory pressure is destabilizing training.<br />**You must shrink the driver&#x27;s node count yourself.** <code>ray.sub</code> cannot see<br />inside your <code>COMMAND</code>, so it only prints the effective GPU-node count. With<br /><code>--nodes=N</code> and <code>DEDICATED_RAY_HEAD=1</code>, <code>cluster.num_nodes</code> (plus<br /><code>env.nemo_gym.num_gpu_nodes</code> when set) must sum to <code>N - 1</code>. If you leave it<br />at <code>N</code>, cluster bringup still succeeds and the driver instead fails minutes<br />later with <code>Maximum number of retries reached (6). Cluster resources may be<br />insufficient...</code>.<br />**The head&#x27;s GPUs are idled, not freed.** Under <code>#SBATCH --exclusive</code> the<br />head node is still allocated with <code>--gres</code>, so this costs one node&#x27;s worth<br />of GPUs.<br />**Watch segment divisibility.** With topology-aware placement the allocation<br />becomes <code>--nodes=N+1</code>, which can trip the &quot;<code>SEGMENT_SIZE</code> must evenly divide<br /><code>NUM_NODES</code>&quot; check in <code>tools/launch</code>.</td>
    </tr>
    <tr>
      <td><code>NRL_TARGET_NOFILE=131072</code></td>
      <td>Open-file soft limit <code>ray.sub</code> asks for. Ray&#x27;s documented 65535 is a floor,<br />not a target, so the soft limit is raised toward this value rather than<br />pinned at the floor: to this value when the hard limit permits it, to the<br />hard limit itself when that falls between 65535 and this value, and left<br />alone with a warning below 65535. A soft limit that already clears the<br />target is never lowered. Raise the target for a workload whose connection<br />pool is configured above it — NeMo-Gym&#x27;s <code>global_aiohttp_connector_limit</code><br />defaults to 102400, and a pool larger than the descriptor ceiling surfaces<br />as <code>OSError: [Errno 24]</code> mid-request rather than as backpressure. The<br />resolved pair is printed as <code>[INFO] file descriptor limit: soft=... hard=...</code>.</td>
    </tr>
    <tr>
      <td><code>NRL_MOUNT_HOST_ENROOT</code></td>
      <td>Unset by default; set to <code>1</code> to bind-mount the compute node&#x27;s own enroot<br />installation (every <code>/usr/bin/enroot*</code> file and the enroot library, share<br />and configuration directories that exist) into the training container, so a<br />process inside it, such as a NeMo Gym agent server that runs each agent in<br />its own sandbox, can start containers. Any value other than unset, <code>0</code> or<br /><code>1</code> is rejected when the batch script starts on the first allocated node.<br />pyxis validates every mount source on each node where a step starts and<br />fails with <code>pyxis: --container-mounts: source path does not exist</code>, so every<br />allocated node needs the same enroot layout. Only enroot&#x27;s own files are<br />mounted: the training container must still provide the tools enroot&#x27;s<br />subcommands invoke (<code>unsquashfs</code> for <code>enroot create</code>, <code>squashfuse</code> and<br /><code>fuse-overlayfs</code> for starting a <code>.sqsh</code> image directly). With<br /><code>tools/launch</code>, pass the variable through <code>EXTRA_ENV</code>.</td>
    </tr>
    <tr>
      <td><code>BASE_LOG_DIR=$SLURM_SUBMIT_DIR</code></td>
      <td>Base directory for storing Ray logs. Defaults to the Slurm submission directory ([SLURM_SUBMIT_DIR](https://slurm.schedmd.com/sbatch.html#OPT_SLURM_SUBMIT_DIR)).</td>
    </tr>
    <tr>
      <td><code>LOKI_PUSH_URL</code></td>
      <td>Set this to stream logs to a Loki endpoint with<br />[Grafana Alloy](https://grafana.com/docs/alloy/), which runs alongside each<br />Ray container. Unset by default, which leaves log export off. Four streams<br />are shipped, distinguished by the <code>log_kind</code> label: Ray&#x27;s own logs under<br /><code>/tmp/ray/session_*/logs</code> (<code>ray-internal</code>), which are otherwise lost when<br />the container exits; this container&#x27;s wrapper log (<code>wrapper</code>); the driver<br />log (<code>driver</code>); and the sandbox sidecar logs (<code>sandbox</code>). Unlike<br /><code>RAY_LOG_SYNC_FREQUENCY</code>, which copies the same files to <code>BASE_LOG_DIR</code>,<br />this sends them over the network and so does not add per-file writes to the<br />shared filesystem.<br />It is separate from <code>telemetry.logs_enabled</code><br />([observability](/observability/observability-stack)), which sends Python<br />log records from NeMo-RL processes over OTLP and does not see these files.<br />Requires an image providing <code>/usr/local/bin/alloy</code> (the included<br /><code>docker/Dockerfile</code> installs it) and a readable <code>alloy/config.alloy</code> under<br />the submission directory.<br />The shipped config authenticates with mTLS, so it also needs<br /><code>LOKI_CA_FILE</code>, <code>LOKI_CERT_FILE</code>, and <code>LOKI_KEY_FILE</code>. These are read inside<br />the container, so their directory must be listed in <code>MOUNTS</code>. For any other<br />scheme Loki supports, such as basic auth, a bearer token, or a tenant ID,<br />copy the config, edit its <code>loki.write</code> block, and point <code>ALLOY_CONFIG</code> at<br />the copy; the certificate variables are then not required.<br />If anything is missing, or if Alloy fails to start, <code>ray.sub</code> logs a warning<br />and the job runs normally without exporting logs.</td>
    </tr>
    <tr>
      <td><code>ALLOY_CONFIG=$SLURM_SUBMIT_DIR/alloy/config.alloy</code></td>
      <td>Path to the Alloy configuration, read inside the container. Override it to<br />use a config with different auth or different log sources.</td>
    </tr>
    <tr>
      <td><code>CLUSTER_NAME=$SLURM_CLUSTER_NAME</code></td>
      <td>Value of the <code>cluster</code> label on exported logs. Defaults to the cluster name<br />Slurm reports, so it only needs setting to label streams with something else.</td>
    </tr>
    <tr>
      <td><code>NODE_MANAGER_PORT=1301</code></td>
      <td>Port for the Ray node manager on worker nodes.</td>
    </tr>
    <tr>
      <td><code>OBJECT_MANAGER_PORT=1303</code></td>
      <td>Port for the Ray object manager on worker nodes.</td>
    </tr>
    <tr>
      <td><code>RUNTIME_ENV_AGENT_PORT=1305</code></td>
      <td>Port for the Ray runtime environment agent on worker nodes.</td>
    </tr>
    <tr>
      <td><code>DASHBOARD_AGENT_GRPC_PORT=1307</code></td>
      <td>gRPC port for the Ray dashboard agent on worker nodes.</td>
    </tr>
    <tr>
      <td><code>METRICS_EXPORT_PORT=1309</code></td>
      <td>Port for exporting metrics from worker nodes.</td>
    </tr>
    <tr>
      <td><code>PORT=1200</code></td>
      <td>Main port for the Ray head node.</td>
    </tr>
    <tr>
      <td><code>RAY_CLIENT_SERVER_PORT=1201</code></td>
      <td>Port for the Ray client server on the head node.</td>
    </tr>
    <tr>
      <td><code>DASHBOARD_GRPC_PORT=52367</code></td>
      <td>gRPC port for the Ray dashboard on the head node.</td>
    </tr>
    <tr>
      <td><code>DASHBOARD_PORT=8265</code></td>
      <td>Port for the Ray dashboard UI on the head node. This is also the port<br />used by the Ray distributed debugger.</td>
    </tr>
    <tr>
      <td><code>DASHBOARD_AGENT_LISTEN_PORT=1311</code></td>
      <td>Listening port for the dashboard agent on the head node.</td>
    </tr>
    <tr>
      <td><code>MIN_WORKER_PORT=2000</code></td>
      <td>Minimum port in the range for Ray worker processes.</td>
    </tr>
    <tr>
      <td><code>MAX_WORKER_PORT=2999</code></td>
      <td>Maximum port in the range for Ray worker processes.</td>
    </tr>
  </tbody>
</table>

> [!NOTE]
> For the most part, you will not need to change ports unless these
> are already taken by some other service backgrounded on your cluster.
> The defaults above are the source-of-truth port layout defined in `ray.sub`;
> keep this table in sync with that block if the defaults ever change.

### Topology-Aware Placement for MoE (avoiding cross-rack stalls)

On clusters where one NVLink domain spans a fixed set of nodes (e.g. a GB200
NVL72 rack = 18 nodes = one fabric domain), a MoE job's expert-parallel (EP)
`all_to_all` can intermittently stall under sustained load when an EP group is
split across two NVLink domains (two racks). The fix is to keep every EP group
inside a single NVLink domain. This requires **two coordinated knobs**, both
sized to the EP group's node span:

```text
segment_size (nodes) = expert_model_parallel_size / gpus_per_node
```

For example `expert_model_parallel_size=8` on a 4-GPU/node cluster spans 2
nodes, so the segment size is `2`.

* **`cluster.segment_size`** (recipe YAML) — application/Ray side. After the
  Slurm allocation exists, NeMo RL selects topology-aligned nodes (complete
  `segment_size`-node segments per NVLink domain) and orders global ranks so
  each EP group lands on one segment. It is a *post-allocation selector*: it
  cannot change what Slurm allocated, and raises `ResourceInsufficientError` if
  the allocated nodes cannot form complete segments. The requested node count
  must be evenly divisible by `segment_size`.

  ```yaml
  cluster:
    gpus_per_node: 4
    segment_size: 2   # = EP group node span; keeps each EP group intra-rack
  ```

* **`sbatch --segment=<size>`** (Slurm side, block topology) — allocation side.
  Tells Slurm to allocate nodes in segments of `<size>` nodes, each segment
  guaranteed within a single topology block (rack); segments themselves may
  span blocks. This makes the physical allocation match what
  `cluster.segment_size` expects (and forces an even node count per rack, so
  odd splits like 5+3 cannot occur). The node count must be evenly divisible by
  the segment size.

  ```sh
  ... \
  sbatch --nodes=8 --segment=2 ray.sub \
     ...
  ```

> [!NOTE]
> Set both to the same value. `--segment` alone guarantees the physical layout
> but not the rank→node mapping; `cluster.segment_size` alone aligns ranks but
> can fail (or silently fall back) if Slurm hands you a layout that cannot form
> complete segments. The guarantee also relies on EP being the contiguous
> (inner) parallel dimension in the global rank order, which is the default for
> `tp-...-ep-dp-pp` layouts with TP/PP not interleaving EP.

For nightly/release tests launched with `tools/launch`, you do not pass
`--segment` by hand: add `SEGMENT_SIZE=<size>` to the script's
`# ===== BEGIN CONFIG =====` block (mirroring `cluster.segment_size` in the
recipe) and `tools/launch` validates divisibility and appends
`--segment=<size>` to the `sbatch` invocation automatically.

```sh
# ===== BEGIN CONFIG =====
NUM_NODES=8
GPUS_PER_NODE=4
SEGMENT_SIZE=2     # nodes per NVLink-domain segment; -> sbatch --segment
...
# ===== END CONFIG =====
```

## Kubernetes

This guide outlines the process of migrating NeMo RL training jobs from a Slurm environment to a Kubernetes cluster utilizing Ray orchestration and NVIDIA GPUs.

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
Clone the NeMo RL repository and build the container.

```bash
# Clone recursively
git clone https://github.com/NVIDIA-NeMo/RL.git --recursive
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
* **Container image:** Replace `<NGC_ORG>` in `nvcr.io/<NGC_ORG>/nemo-rl:latest` with the NGC org you pushed to in the build step.

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
          image: nvcr.io/<NGC_ORG>/nemo-rl:latest
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
          image: nvcr.io/<NGC_ORG>/nemo-rl:latest
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

## Phase 3: Run Sample NeMo RL Workloads

Once the cluster is running, you can interact with the Ray head node to submit jobs.

### 1. Access the Head Node
```bash
kubectl exec -it $(kubectl get pod -l ray.io/node-type=head -o jsonpath='{.items[0].metadata.name}') -- /bin/bash
```

### 2. Setup Code on Shared Volume
Inside the pod, clone the code to the shared PVC (`/shared`). This ensures workers can see the code.

```bash
cd /shared
git clone https://github.com/NVIDIA-NeMo/RL.git --recursive
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
uv run examples/run_grpo.py \
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

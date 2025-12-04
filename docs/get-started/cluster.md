---
description: "Step-by-step guide to running NeMo RL on multi-node Slurm clusters"
categories: ["getting-started"]
tags: ["cluster", "slurm", "kubernetes", "multi-node", "ray"]
personas: ["mle-focused", "cluster-administrator-focused"]
difficulty: "intermediate"
content_type: "tutorial"
---

(gs-cluster)=

# Set Up a Training Cluster

Scaling from a single GPU to a multi-node cluster allows you to train larger models (70B+) and process data much faster.

NeMo RL uses **Ray** to manage distributed computing.

:::{card}
**Goal**: Submit a multi-node training job (e.g., GRPO) to a Slurm cluster.

^^^

**Steps**:

1. **Understand the Architecture**: How Ray and Slurm work together.
2. **Submit a Job**: Choose between Interactive (Debug) or Batch (Production) modes.
3. **Verify**: Check logs and the Ray Dashboard.
:::

:::{button-ref} index
:color: secondary
:outline:
:ref-type: doc

← Previous: Quickstart Guide
:::

---

## 1. Understand the Architecture

When you submit a job, two layers of orchestration happen:

1. **Slurm (The Hardware Layer)**: Allocates physical nodes (e.g., 4 nodes with 8 GPUs each).
2. **Ray (The Application Layer)**: Connects those nodes into a unified cluster.
   * **Head Node**: Runs the driver script (e.g., `run_grpo.py`) and manages the cluster state.
   * **Worker Nodes**: Execute the heavy lifting (model training, generation).

You don't need to manually configure Ray. NeMo RL provides a helper script, `ray.sub`, that handles the bootstrapping for you.

---

## 2. Submit a Job

The submission process is identical for SFT, DPO, Reward Model (RM), and GRPO. You swap the Python script in the `COMMAND` variable.

### Command Cheatsheet

Copy the command for your training type:

| Type | Command |
| :--- | :--- |
| **SFT** | `uv run examples/run_sft.py` |
| **DPO** | `uv run examples/run_dpo.py` |
| **Reward Model** | `uv run examples/run_rm.py` |
| **GRPO** | `uv run examples/run_grpo_math.py` |

::::{tab-set}

:::{tab-item} Interactive Mode (Recommended for Debugging)
Interactive mode launches the cluster and gives you a shell on the **Head Node**. This is perfect for debugging because you can run scripts, check files, and kill/restart jobs without re-queueing.

**1. Submit the Request**
Ask for the resources you need (e.g., 1 node, 8 GPUs).

```bash
# Run from the root of NeMo RL repo
NUM_ACTOR_NODES=1  # Total nodes requested (head is colocated on ray-worker-0)

CONTAINER=nvcr.io/nvidia/nemo:latest \
MOUNTS="$PWD:$PWD" \
sbatch \
    --nodes=${NUM_ACTOR_NODES} \
    --account=YOUR_ACCOUNT \
    --partition=YOUR_PARTITION \
    --gres=gpu:8 \
    --time=1:0:0 \
    --job-name=YOUR_JOBNAME \
    ray.sub
```

:::{tip}
- Replace `YOUR_ACCOUNT` and `YOUR_PARTITION` with values from your cluster. Run `sacctmgr show associations user=$USER` to find your account.
- Depending on your Slurm cluster configuration, you may need `--gres=gpu:8` or `--gpus-per-node=8`. Check with your cluster admin if jobs don't receive GPUs.
:::

**2. Attach to the Cluster**
Once the job starts, Slurm creates an attach script (e.g., `12345-attach.sh`). Run it:

```bash
bash <JOB_ID>-attach.sh
```

**3. Run Your Training**
You are now inside the container on the head node. Run your command (see Cheatsheet above):

```bash
uv run examples/run_sft.py
```

:::

:::{tab-item} Batch Mode (Production)
Batch mode is "fire and forget." You specify the command upfront, and the cluster shuts down automatically when it finishes.

**1. Submit the Job**
Include the `COMMAND` variable in your submission. Replace the command below with the one from the Cheatsheet.

```bash
# Run from the root of NeMo RL repo
NUM_ACTOR_NODES=1  # Total nodes requested (head is colocated on ray-worker-0)

COMMAND="uv run examples/run_sft.py" \
CONTAINER=nvcr.io/nvidia/nemo:latest \
MOUNTS="$PWD:$PWD" \
sbatch \
    --nodes=${NUM_ACTOR_NODES} \
    --account=YOUR_ACCOUNT \
    --partition=YOUR_PARTITION \
    --gres=gpu:8 \
    --time=1:0:0 \
    --job-name=YOUR_JOBNAME \
    ray.sub
```

**2. Check Status**
Slurm will write the output to a log file (e.g., `12345-logs/ray-driver.log`).
:::

::::

---

## 3. Verify Your Cluster

Once your job is running, you have two main ways to see what's happening.

### 1. Slurm Logs

The `ray.sub` script creates a log directory named after your Job ID (e.g., `1980204-logs/`).

* **`ray-driver.log`**: The stdout/stderr of your Python script. Check this for training progress (loss values).
* **`ray-worker-*.log`**: Logs for individual worker nodes (useful for debugging specific node failures).
* **`dashboard.log`**: Debug info for the Ray dashboard.

```bash
tail -f 1980204-logs/ray-driver.log
```

### 2. Ray Dashboard

Ray provides a visual dashboard to see GPU usage, memory usage, and actor status.

1. Find the dashboard port in the logs (printed at startup).
2. Forward the port to your local machine:

```bash
ssh -L 8265:localhost:8265 user@cluster-login-node
```

3. Open `http://localhost:8265` in your browser.

---

## Common Environment Variables

You can pass these variables to `sbatch` to configure the environment:

| Variable | Description |
| :--- | :--- |
| **`CONTAINER`** | Docker image to use (required). |
| **`MOUNTS`** | Paths to mount (e.g., `"$PWD:$PWD,/data:/data"`). |
| **`HF_TOKEN`** | Hugging Face token (for downloading gated models). |
| **`HF_HOME`** | Cache directory for Hugging Face models and tokenizers. |
| **`HF_DATASETS_CACHE`** | Cache directory for Hugging Face datasets. |
| **`WANDB_API_KEY`** | Weights & Biases key (for logging). |
| **`GPUS_PER_NODE`** | Number of GPUs per node (default: 8). |

:::{tip}
Export secrets like `HF_TOKEN` in your shell profile (`~/.bashrc`) so you don't have to type them every time.
:::

:::{dropdown} Advanced Environment Configuration
The following variables allow for deeper customization of the Ray cluster. Most users will not need to change these defaults.

| Variable (and default) | Explanation |
| :--- | :--- |
| `UV_CACHE_DIR_OVERRIDE` | Override the UV cache directory to a shared filesystem location. Essential for persisting package caches across jobs. |
| `CPUS_PER_WORKER=128` | CPUs each Ray worker node claims. Default is `16 * GPUS_PER_NODE`. |
| `GPUS_PER_NODE=8` | Number of GPUs each Ray worker node claims. |
| `BASE_LOG_DIR=$SLURM_SUBMIT_DIR` | Base directory for storing Ray logs. |
| `NODE_MANAGER_PORT=53001` | Port for the Ray node manager on worker nodes. |
| `OBJECT_MANAGER_PORT=53003` | Port for the Ray object manager on worker nodes. |
| `RUNTIME_ENV_AGENT_PORT=53005` | Port for the Ray runtime environment agent on worker nodes. |
| `DASHBOARD_AGENT_GRPC_PORT=53007` | gRPC port for the Ray dashboard agent on worker nodes. |
| `METRICS_EXPORT_PORT=53009` | Port for exporting metrics from worker nodes. |
| `PORT=54514` | Main port for the Ray head node (default: 54514; Ray's standard default is 6379, but this script uses 54514 for multi-node compatibility). |
| `RAY_CLIENT_SERVER_PORT=10001` | Port for the Ray client server on the head node. |
| `DASHBOARD_GRPC_PORT=52367` | gRPC port for the Ray dashboard on the head node. |
| `DASHBOARD_PORT=8265` | Port for the Ray dashboard UI on the head node. |
| `DASHBOARD_AGENT_LISTEN_PORT=52365` | Listening port for the dashboard agent on the head node. |
| `MIN_WORKER_PORT=54001` | Minimum port in the range for Ray worker processes. |
| `MAX_WORKER_PORT=54257` | Maximum port in the range for Ray worker processes. |
:::

# Debugging in NeMo RL

This guide explains how to debug NeMo RL applications, covering two scenarios. It first outlines the procedure for debugging distributed Ray worker/actor processes using the Ray Distributed Debugger within a SLURM environment, and then details debugging the main driver script.

## Debugging in the Worker/Actors (on SLURM)

Since Ray programs can spawn many workers/actors, we need to use the Ray Distributed Debugger
to properly jump to the breakpoint on each worker.

### Prerequisites

* Install [Ray Debugger VS Code/Cursor extension](https://docs.ray.io/en/latest/ray-observability/ray-distributed-debugger.html).
* Launch the [interactive cluster](./cluster.md#interactive-launching) with `ray.sub`.
* Launch VS Code/Cursor on the SLURM login node (where `squeue`/`sbatch` is available).
* Add `breakpoint()` in your code under actors & tasks (i.e. classes or functions decorated with `@ray.remote`).
* **Ensure** `RAY_DEBUG=legacy` is not set since this debugging requires the default distributed debugger.

### Port-forwarding from the Head Node

From the SLURM login node, query the nodes used by the interactive `ray.sub` job as follows:

```sh
teryk@slurm-login:~$ squeue --me
             JOBID PARTITION        NAME     USER ST       TIME  NODES NODELIST(REASON)
           2504248     batch ray-cluster   terryk  R      15:01      4 node-12,node-[22,30],node-49
```

The first node is always the head node, so we need to port forward the dashboard port to the login node:

```sh
# Traffic from the login node's $LOCAL is forwarded to node-12:$DASHBOARD_PORT
# - If you haven't changed the default DASHBOARD_PORT in ray.sub, it is likely 8265
# - Choose a LOCAL_PORT that isn't taken. If the cluster is multi-tenant, 8265
#   on the login node is likely taken by someone else.
ssh -L $LOCAL_PORT:localhost:$DASHBOARD_PORT -N node-12

# Example chosing a port other than 8265 for the LOCAL_PORT
ssh -L 52640:localhost:8265 -N node-12
```

Example output from the port-forwarding with `ssh` may print logs like this, where the warning is expected:

```text
Warning: Permanently added 'node-12' (ED25519) to the list of known hosts.
bind [::1]:52640: Cannot assign requested address
```

### Open the Ray Debugger Extension

In VS Code/Cursor, open the Ray Debugger extension by clicking on the Ray icon in the activity bar or by searching for "View: Show Ray Debugger" in the command palette (Ctrl+Shift+P or Cmd+Shift+P).

![Ray Debugger Extension Step 1](./assets/ray-debug-step1.png)

### Add the Ray Cluster

Click on the "Add Cluster" button in the Ray Debugger panel.

![Ray Debugger Extension Step 2](./assets/ray-debug-step2.png)

Enter the address and port you set up in the port forwarding step. If you followed the example above using port 52640, you would enter:

![Ray Debugger Extension Step 3](./assets/ray-debug-step3.png)

### Add a Breakpoint and Run Your Program

All breakpoints that are reached while the program is running will be visible in the Ray Debugger Panel dropdown for the cluster `127.0.0.1:52640`. Click
`Start Debugging` to jump to one worker's breakpoint.

Note that you can jump between breakpoints across all workers with this process.

![Ray Debugger Extension Step 4](./assets/ray-debug-step4.png)

## Debugging in the Driver Script

By default, setting breakpoints in the driver script (outside of  `@ray.remote`) will not pause program execution when using Ray. To enable pausing at these breakpoints, set the environment variable to `RAY_DEBUG=legacy`:

```sh
RAY_DEBUG=legacy uv run ....
```

## Nsight Profiling (nsys)

NeMo RL supports nsight profiling for Ray workers through environment variable pattern matching. This allows you to selectively profile specific worker types without modifying code or affecting performance of workers that don't need profiling.

**Note**: To prevent profile files from becoming too large, consider running for a small number of steps (e.g., 10 steps) when profiling.

### Prerequisites

* Install NVIDIA Nsight Systems (`nsys`) on the compute nodes where workers will run. For Ubuntu installation instructions, see the [NVIDIA Nsight Systems Installation Guide](https://docs.nvidia.com/nsight-systems/InstallationGuide/index.html#:~:text=Ubuntu%20(minimal%20setup%20for%20containers)). **Note: If you're using NeMo RL containers, `nsys` is already installed.**
* Ensure the workers you want to profile have GPU access

### Environment Variable Configuration

Set the `NRL_NSYS_WORKER_PATTERNS` environment variable with a comma-separated list of patterns to match worker names:

```bash
export NRL_NSYS_WORKER_PATTERNS="*policy*,*vllm*"
```

#### Pattern Format

- Use shell-style wildcards (`*`, `?`, `[seq]`, `[!seq]`)
- Patterns are matched against worker names using `fnmatch`
- Multiple patterns are separated by commas
- Whitespace around patterns is automatically stripped
- Empty patterns are ignored

#### Supported Workers

Currently supported worker types:
- **DTensorPolicyWorker**: Pattern matched against `"dtensor_policy_worker"`
- **VllmGenerationWorker**: Pattern matched against `"vllm_generation_worker"`

### Example Usage

#### Profile Only Policy Workers
```bash
NRL_NSYS_WORKER_PATTERNS="*policy*" uv run examples/run_grpo_math.py grpo.max_num_steps=10
```

#### Profile Multiple Worker Types
```bash
NRL_NSYS_WORKER_PATTERNS="*policy*,*vllm*" uv run examples/run_grpo_math.py grpo.max_num_steps=10
```

#### Profile Workers with Exact Names
```bash
NRL_NSYS_WORKER_PATTERNS="dtensor_policy_worker,vllm_generation_worker" uv run examples/run_grpo_math.py grpo.max_num_steps=10
```

### Profile Output

When profiling is enabled:

1. **Logging**: You'll see log messages indicating which workers have profiling enabled:
   ```
   Nsight profiling enabled for worker 'dtensor_policy_worker' (matched pattern '*policy*')
   ```

2. **Profile Files**: Each profiled worker generates a `.nsys-rep` file with naming pattern:
   ```
   dtensor_policy_worker_<PID>.nsys-rep
   vllm_generation_worker_<PID>.nsys-rep
   ```

3. **File Location**: Profile files are saved in `/tmp/ray/session*/logs/nsight/` directory on each worker node.

**Note for SLURM users with `ray.sub`**: When using `ray.sub` on SLURM, set `RAY_LOG_SYNC_FREQUENCY=$NUM_SEC` (e.g., `RAY_LOG_SYNC_FREQUENCY=30`) to ensure that the nsight profile files get copied from the container's ephemeral filesystem (`/tmp/ray`) to the persistent `$SLURM_JOB_ID-logs/ray` directory.

### Analyzing Profiles

To analyze the generated profile files, load the `.nsys-rep` files into the NVIDIA Nsight Systems desktop application, which you can download from the [NVIDIA Nsight Systems Get Started page](https://developer.nvidia.com/nsight-systems/get-started).

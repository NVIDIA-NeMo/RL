# Profile GPU with Nsys

NeMo RL supports Nsight profiling for Ray workers through environment variable pattern matching. This allows you to selectively profile specific worker types without modifying code or affecting the performance of workers that don't need profiling.

**Note**: To prevent profile files from becoming too large, consider limiting profiling to a smaller number of steps (e.g., 10 steps).

## Prerequisites

* Install NVIDIA Nsight Systems (`nsys`) on the compute nodes where workers will run. For Ubuntu installation instructions, see the [NVIDIA Nsight Systems Installation Guide](https://docs.nvidia.com/nsight-systems/InstallationGuide/index.html#:~:text=Ubuntu%20(minimal%20setup%20for%20containers)).

**Note: If you're using NeMo RL containers, `nsys` is already installed.**

* Ensure the workers you want to profile have GPU access

## Configure the Environment Variables

Set the `NRL_NSYS_WORKER_PATTERNS` environment variable with a comma-separated list of patterns to match worker names:

```bash
export NRL_NSYS_WORKER_PATTERNS="*policy*,*vllm*"
```

Set the `NRL_NSYS_PROFILE_STEP_RANGE` environment variable to control which training steps the profiler captures. Its
format is colon separated integers representing `start:stop`, where `start` is inclusive and `stop` is exclusive
(same as slice syntax `arr[start:stop]`). Note that the `start` is 1-index, so `NRL_NSYS_PROFILE_STEP_RANGE=0:10` would error.

```bash
export NRL_NSYS_PROFILE_STEP_RANGE=3:5
```

### Pattern Format

- Use shell-style wildcards (`*`, `?`, `[seq]`, `[!seq]`)
- Patterns are matched against worker names using `fnmatch`
- Multiple patterns are separated by commas
- Whitespace around patterns is automatically stripped
- Empty patterns are ignored

### Supported Workers

The supported worker types are:
- **DTensorPolicyWorker**: Pattern matched against `"dtensor_policy_worker"`
- **VllmGenerationWorker**: Pattern matched against `"vllm_generation_worker"`

## Example Usage

### Profile Only Policy Workers
```bash
NRL_NSYS_PROFILE_STEP_RANGE=2:3 NRL_NSYS_WORKER_PATTERNS="*policy*" uv run examples/run_grpo_math.py grpo.max_num_steps=5
```

### Profile Multiple Worker Types

```bash
NRL_NSYS_PROFILE_STEP_RANGE=1:2 NRL_NSYS_WORKER_PATTERNS="*policy*,*vllm*" uv run examples/run_grpo_math.py grpo.max_num_steps=5
```

### Profile Workers with Exact Names

```bash
NRL_NSYS_PROFILE_STEP_RANGE=3:10 NRL_NSYS_WORKER_PATTERNS="dtensor_policy_worker,vllm_generation_worker" uv run examples/run_grpo_math.py grpo.max_num_steps=5
```

### Profile Megatron Workers

:::{important}
To profile a Megatron worker, you should set `LD_LIBRARY_PATH` as follows, otherwise you will get errors when loading `libtransformer_engine.so`.
:::

```bash
LD_LIBRARY_PATH="/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib:/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/lib/x86_64-linux-gnu" \
NRL_NSYS_PROFILE_STEP_RANGE=2:3 NRL_NSYS_WORKER_PATTERNS="megatron_policy_worker,vllm_generation_worker" uv run examples/run_grpo_math.py --config examples/configs/grpo_math_1B_megatron.yaml grpo.max_num_steps=5
```

## Profile Output

When profiling is enabled, it generates the following logs and files:

1. **Logging**: You'll see log messages indicating which workers have profiling enabled:
   ```
   Nsight profiling enabled for worker 'dtensor_policy_worker' (matched pattern '*policy*')
   ```

2. **Profile Files**: Each profiled worker generates a `.nsys-rep` file with naming pattern:
   ```
   dtensor_policy_worker_<NRL_NSYS_PROFILE_STEP_RANGE>_<PID>.nsys-rep
   vllm_generation_worker_<NRL_NSYS_PROFILE_STEP_RANGE>_<PID>.nsys-rep
   ```

3. **File Location**: Profile files are saved in `/tmp/ray/session*/logs/nsight/` directory on each worker node.

**Note for SLURM users with `ray.sub`**: When using `ray.sub` on SLURM, set `RAY_LOG_SYNC_FREQUENCY=$NUM_SEC` (e.g., `RAY_LOG_SYNC_FREQUENCY=30`) to ensure that the nsight profile files get copied from the container's ephemeral filesystem (`/tmp/ray`) to the persistent `$SLURM_JOB_ID-logs/ray` directory.

## Analyze Profile Files

To analyze the generated profile files, load the `.nsys-rep` files into the NVIDIA Nsight Systems desktop application, which you can download from the [NVIDIA Nsight Systems Get Started page](https://developer.nvidia.com/nsight-systems/get-started).

## How We Patched Nsight Support in Ray

Ray's Nsight profiling support had a bug where it hardcoded the Python executable path instead of using the actual Python executable from the runtime environment. This caused issues when using virtual environments or custom Python installations (`py_executables`).

### The Problem

In Ray's `nsight.py` file, the original code was:

```python
context.py_executable = " ".join(self.nsight_cmd) + " python"
```

This hardcoded `" python"` instead of correctly preserving the intended Python executable path.

### The Fix

To fix this problem, we patched the following line to preserve the original `context.py_executable`:

```python
context.py_executable = " ".join(self.nsight_cmd) + f" {context.py_executable}"
```

### Where We Applied the Patch

We applied this patch in two locations to cover different deployment scenarios:

1. **In `ray.sub` (SLURM clusters)**: The patch is applied before Ray's control plane starts up on both head and worker nodes:
   ```bash
   sed -i 's/context\.py_executable = " "\.join(self\.nsight_cmd) + " python"/context.py_executable = " ".join(self.nsight_cmd) + f" {context.py_executable}"/g' /opt/nemo_rl_venv/lib64/python*/site-packages/ray/_private/runtime_env/nsight.py
   ```

2. **In `nemo_rl/__init__.py` (Local clusters)**: The patch is applied automatically when NeMo RL is imported, making it work seamlessly for local development and testing environments.

### Why We Needed Both Locations

- **`ray.sub`**: Required for SLURM-managed clusters where Ray processes start in containers before Python imports happen. The patch must be applied at the filesystem level before Ray's control plane initializes.

- **`__init__.py`**: Required for local clusters and development environments where users start Ray clusters directly. The patch is applied when `nemo_rl` is imported, ensuring the fix is in place before any Ray processes are spawned.

This dual approach ensures that Nsight profiling works correctly regardless of how the Ray cluster is deployed.

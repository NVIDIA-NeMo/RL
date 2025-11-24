# Instruction to repoduce the error

## Steps

### 1. Clone and install nemo-rl repo as described in the repo


```bash
git clone git@github.com:NVIDIA-NeMo/RL.git nemo-rl --recursive
cd nemo-rl

# If you are already cloned without the recursive option, you can initialize the submodules recursively
git submodule update --init --recursive
```

---

### 2. Copy dir inside nemo-rl

```bash
cp -r /oom_bug_reproduce /path/to/nemo-rl/
cd /path/to/nemo-rl/oom_bug_reproduce
```

---

### 3. Add Ether0 dependency for reward computation

Edit `pyproject.yaml` in the `nemo-rl` repo and add:

```bash
dependencies = [
  ...
  "ether0 @ git+https://github.com/Future-House/ether0.git",
  ]
```

---

### 4. Install dependencies

```bash
# DEPRECATED - No longer needed
# uv sync --all-extras
``` 

No need to install manually as ether0 has its own uv environment.

---

### 6. Run experiments

From inside the copied `/path/to/nemo-rl/oom_bug_reproduce` directory:

```bash
# DEPRECATED - No longer needed as we need container to handle env setup
# bash run.sh configs/config_qwen3_235B_instruct.yaml 16 qwen3_235B_instruct_results

# small scale
COMMAND="bash nemorl_oom_bug_reproduce/local_run_small_scale.sh" NNODES=2 bash user_run_slurm.sh

# large scale
COMMAND="bash nemorl_oom_bug_reproduce/local_run_large_scale.sh" NNODES=16 LONG_RUN=true bash user_run_slurm.sh
```

Note: you will need to change the slurm partition name inside run.sh !

### 7. Expected behavior

> CPU ram will slowly increase. For us, after ~10-12 full steps, it raises an OOM error  

Tried but not able to reproduce.
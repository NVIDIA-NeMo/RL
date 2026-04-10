# NeMo-RL on HFC (HuggingFace Cluster)

**Upstream**: https://github.com/NVIDIA-NeMo/RL

## Requirements

- NVIDIA H100/B200 (compute capability 9.0+)
- SLURM with Enroot + Pyxis
- Docker access on login node (for building)

## Container Details

- Base: `nvcr.io/nvidia/cuda-dl-base:25.05-cuda12.9-devel-ubuntu24.04`
- Python 3.13.11, PyTorch 2.10.0, CUDA 12.9
- Backends: vLLM, Megatron-Core, AutoModel (sglang skipped)
- Registry tag: `nemo-rl:edward-20260410`

## Pre-built Image

A pre-built `.sqsh` file is available on the shared filesystem:

```bash
/fsx/edward/docker_images/nemo-rl.sqsh
```

Or pull from the cluster registry:

```bash
# Direct use (no local copy needed)
srun --container-image="docker://registry.hpc-cluster-hopper.hpc.internal.huggingface.tech/library/nemo-rl:edward-20260410" ...

# Create your own local .sqsh copy
enroot import "docker://registry.hpc-cluster-hopper.hpc.internal.huggingface.tech/library/nemo-rl:edward-20260410"
```

## Building from Source

Run on the login node (requires Docker):

```bash
cd container/
export REGISTRY="registry.hpc-cluster-hopper.hpc.internal.huggingface.tech"

# Build, push to registry, and create .sqsh file
./build_container.sh --sqsh-output nemo-rl.sqsh

# Skip sglang to speed up the build (~30 min instead of ~2h)
SKIP_SGLANG_BUILD=1 ./build_container.sh --sqsh-output nemo-rl.sqsh
```

## Testing the Container

```bash
sbatch container/test_container.slurm --sqsh /fsx/edward/docker_images/nemo-rl.sqsh
cat nemo_rl_test_<job-id>.out
```

## Usage

### Quick Start: Single-Node GRPO (sbatch)

```bash
# Default config (Qwen2.5-1.5B, 8 GPUs, OpenMathInstruct-2)
sbatch container/run_grpo.slurm
```

### Multi-Node GRPO

```bash
# 2 nodes, 16 GPUs
sbatch --nodes=2 container/run_grpo.slurm

# With custom config
CONFIG_FILE=container/configs/grpo-qwen3-4b-thinking-2n8g-fsdp2tp1-noncolocated-fast.yaml \
  sbatch --nodes=2 container/run_grpo.slurm
```

### Custom Config Overrides

```bash
# Override any config value via Hydra syntax
EXTRA_OVERRIDES="grpo.max_num_steps=5 policy.model_name=Qwen/Qwen2.5-1.5B" \
  sbatch container/run_grpo.slurm
```

### Interactive: Piggyback on Existing Allocation

If you already have a SLURM allocation (e.g., from `salloc` or a dev node):

```bash
# Find your job ID
squeue -u $USER

# Run GRPO on your existing allocation
srun --jobid=<JOBID> --overlap --gres=gpu:8 --nodes=1 --ntasks=1 \
     --container-image="/fsx/edward/docker_images/nemo-rl.sqsh" \
     --container-mounts="/fsx:/fsx,/scratch:/scratch,/fsx/edward/work/scale-rl/nemo-rl:/opt/nemo-rl" \
     --no-container-mount-home \
     bash -c "cd /opt/nemo-rl && uv run python examples/run_grpo.py cluster.num_nodes=1 cluster.gpus_per_node=8"
```

### Interactive: Get a Shell Inside the Container

```bash
srun --jobid=<JOBID> --overlap --gres=gpu:8 --nodes=1 --ntasks=1 \
     --container-image="/fsx/edward/docker_images/nemo-rl.sqsh" \
     --container-mounts="/fsx:/fsx,/scratch:/scratch,/fsx/edward/work/scale-rl/nemo-rl:/opt/nemo-rl" \
     --no-container-mount-home \
     --pty bash
```

Then inside the container:

```bash
cd /opt/nemo-rl

# Run GRPO (default 1B model)
uv run python examples/run_grpo.py cluster.num_nodes=1 cluster.gpus_per_node=8

# Run SFT
uv run python examples/run_sft.py

# Run DPO
uv run python examples/run_dpo.py

# Run unit tests
uv run --group test pytest tests/unit -x -q
```

### Idle Ray Cluster (Start Cluster, Attach Later)

Useful for debugging or running multiple experiments on the same cluster:

```bash
# Start a 2-node Ray cluster, but don't run anything
COMMAND="" sbatch --nodes=2 container/run_grpo.slurm

# Check logs for attach instructions
cat logs/nemo-rl-grpo-<JOBID>.out

# Attach to the head node
bash <JOBID>-attach.sh

# Attach to worker 1
bash <JOBID>-attach.sh 1

# Run a command without a shell
COMMAND="ray status" bash <JOBID>-attach.sh
```

### Multi-Node with ray_patched.sub

For more control over the Ray cluster (custom ports, logging, etc.):

```bash
# Edit run_multinode.slurm to customize CONFIG_FILE, CHECKPOINT_DIR, etc.
sbatch container/run_multinode.slurm

# Or attach to an existing allocation
bash container/run_multinode.slurm <JOBID>
```

### Full Unit Tests

```bash
sbatch container/run_full_tests.slurm --sqsh /fsx/edward/docker_images/nemo-rl.sqsh
```

## Available Configs

| Config | Model | Nodes | Description |
|--------|-------|-------|-------------|
| (default) | Qwen2.5-1.5B | 1 | `examples/configs/grpo_math_1B.yaml` |
| `grpo-qwen3-4b-...-fast.yaml` | Qwen3-4B | 2 | Fast test (small batches, 5 steps) |
| `grpo-qwen3-4b-...-noncolocated.yaml` | Qwen3-4B | 2 | Full config (non-colocated vLLM) |

See `examples/configs/` for all upstream configs (8B, 70B, Megatron, etc.).

## Developing with the Container

Mount your local source to iterate without rebuilding:

```bash
# The mount overlay makes /opt/nemo-rl point to your local clone
--container-mounts="/fsx:/fsx,/scratch:/scratch,/fsx/edward/work/scale-rl/nemo-rl:/opt/nemo-rl"
```

Edit code locally, re-run inside the container -- changes are reflected immediately.

## Notes

- Build runs on login node (Docker); training runs on compute nodes (Enroot/Pyxis)
- vLLM runs in a separate Ray worker venv (pre-built inside the container)
- `NEMO_RL_VENV_DIR` is set to `/fsx/edward/.cache/nemo-rl/ray_venvs` for persistent caching
- For gated models (Llama, etc.), set `HF_TOKEN` before submitting

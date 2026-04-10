# NeMo-RL

**Repository**: https://github.com/NVIDIA-NeMo/RL

## Requirements

- CUDA 12.9, Python 3.12, PyTorch 2.8.0
- NVIDIA H100/B200 (compute capability 9.0+)
- SLURM with Enroot + Pyxis

## Setup

### 1. Build Container (Login Node)

```bash
export REGISTRY="registry.hpc-cluster-hopper.hpc.internal.huggingface.tech"
./build_container.sh --sqsh-output nemo-rl.sqsh
```

Build time: ~50-60 minutes.

### 2. Test Container (Compute Node)

```bash
sbatch test_container.slurm --sqsh /fsx/edward/docker_images/nemo-rl.sqsh
```

Test time: ~30 seconds. Check output: `cat nemo_rl_test_<job-id>.out`

## Usage

### Training Examples

Single-node GRPO (interactive):
```bash
export CONTAINER_IMAGE="/fsx/edward/docker_images/nemo-rl.sqsh"

srun --gpus-per-node=8 \
     --container-image="${CONTAINER_IMAGE}" \
     --container-mounts="/fsx:/fsx,/scratch:/scratch" \
     --no-container-mount-home \
     bash -c "cd /opt/nemo-rl && uv run python examples/run_grpo_math.py"
```

Multi-node GRPO (batch):
```bash
# Edit run_multinode.slurm to set HF_TOKEN (for gated models), then:
sbatch run_multinode.slurm
bash run_multinode.slurm 21639840
```

### Full Unit Tests (2 GPUs)

```bash
sbatch run_full_tests.slurm --sqsh /fsx/edward/docker_images/nemo-rl.sqsh
```

## Notes

- Build runs on login node (Docker), tests run on compute nodes (Enroot/Pyxis)
- `run_multinode.slurm` uses GRPO with Llama 3.1 8B (2-node config, HF token included)
- Other examples: `run_sft.py`, `run_dpo.py`, `run_rm.py` (see `examples/`)
- vLLM is optional (requires `uv sync --extra vllm`)
- Official docs: https://github.com/NVIDIA-NeMo/RL

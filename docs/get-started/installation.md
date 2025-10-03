---
description: "Complete installation guide for NeMo RL across different platforms including Ubuntu, Windows, macOS, and cluster environments"
categories: ["getting-started"]
tags: ["installation", "setup", "docker", "conda", "gpu-accelerated", "configuration"]
personas: ["mle-focused", "admin-focused", "devops-focused"]
difficulty: "beginner"
content_type: "tutorial"
modality: "universal"
---

# Install NeMo RL

This guide covers installing NeMo RL on various platforms and environments.

## Prerequisites

### System Requirements
- **Python**: 3.12 or higher
- **CUDA**: 11.8 or higher (for GPU support)
- **Memory**: Minimum 16 GB RAM, 32 GB+ recommended
- **Storage**: At least 50GB free space for models and datasets

### Hardware Requirements

(gpu-requirements)=

- **GPU**: NVIDIA GPU with 8 GB+ VRAM (16 GB+ recommended)
- **CPU**: Multi-core processor (8+ cores recommended)
- **Network**: Stable internet connection for downloading models

## Installation Methods

### Method 1: Clone and Install (Recommended)

1. **Clone the repository**:
   ```bash
   git clone git@github.com:NVIDIA/NeMo-RL.git nemo-rl
   cd nemo-rl
   ```

2. **Initialize submodules**:
   ```bash
   git submodule update --init --recursive
   ```

3. **Install with uv** (recommended):
   ```bash
   uv sync
   ```
   
   > **Note**: NeMo RL uses `uv` for dependency management. This is the recommended installation method as it ensures all dependencies are properly resolved and locked.

4. **Install with pip** (alternative):
   ```bash
   pip install -e .
   ```

### Method 2: Docker Installation

1. **Pull the Docker image**:
   ```bash
   docker pull nvcr.io/nvidia/nemo-rl:latest
   ```

2. **Run the container**:
   ```bash
   docker run --gpus all -it nvcr.io/nvidia/nemo-rl:latest
   ```

### Method 3: Conda Installation

1. **Create a new conda environment**:
   ```bash
   conda create -n nemo-rl python=3.9
   conda activate nemo-rl
   ```

2. **Install PyTorch**:
   ```bash
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   ```

3. **Install NeMo RL**:
   ```bash
   pip install nemo-rl
   ```

## Environment Setup

### Environment Variables

Set the following environment variables:

```bash
# NeMo RL specific
export NRL_VIRTUAL_CLUSTER_MAX_RETRIES=6
export NRL_RAY_OBJECT_STORE_MEMORY=1000000000

# Hugging Face
export HF_HOME="/path/to/huggingface/cache"
export HF_DATASETS_CACHE="/path/to/datasets/cache"

# Weights and Biases (optional)
export WANDB_API_KEY="your_wandb_api_key"

# CUDA
export CUDA_VISIBLE_DEVICES="0,1,2,3"  # Specify GPUs to use
```

### Hugging Face Authentication

(model-access)=

For models requiring authentication (e.g., Llama models):

```bash
huggingface-cli login
```

## Optional Dependencies

### For Development
```bash
uv sync --group dev
```

### For Documentation
```bash
pip install -r requirements-docs.txt
```

### For Testing
```bash
uv sync --group test
```

## Optional Backend Dependencies

The [`3rdparty/`](https://github.com/NVIDIA/NeMo-RL/tree/main/3rdparty) directory contains optional third-party dependencies maintained as separate git submodules:

- **Megatron-LM** - For large-scale transformer training with NVIDIA Megatron backend
- **Megatron-Bridge** - Adapter layer for Megatron framework integration

:::{note}
Most users can use the default HuggingFace/DTensor backend. Only initialize submodules with `git submodule update --init --recursive` if you need Megatron backend support.
:::

## Hardware-Specific Dependencies

This section covers additional system dependencies that may be required for specific backends or bare-metal installations.

### cuDNN (Required for Megatron Backend)

If you are using the Megatron backend on bare metal (outside of a container), you'll need cuDNN headers installed.

**Check if cuDNN is already installed:**
```bash
dpkg -l | grep cudnn.*cuda
```

**Install cuDNN** (Ubuntu/Debian):

1. **Add NVIDIA CUDA repository**:
   ```bash
   # Download the CUDA keyring package
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
   sudo dpkg -i cuda-keyring_1.1-1_all.deb
   sudo apt update
   ```

2. **Install cuDNN**:
   ```bash
   # Install cuDNN meta package (latest version)
   sudo apt install cudnn
   
   # Or install specific version for CUDA 12.x
   sudo apt install cudnn9-cuda-12
   
   # Or install specific version for CUDA 12.8
   sudo apt install cudnn9-cuda-12-8
   ```

:::{tip}
Find the exact version for your system at [NVIDIA cuDNN Downloads](https://developer.nvidia.com/cudnn-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_network).
:::

**When do you need this?**
- Running Megatron backend on bare metal (not in Docker)
- Training large models with Megatron-LM
- Using advanced parallelism features (TP/PP/CP)

### libibverbs-dev (Required for vLLM on Bare Metal)

If you encounter issues installing vLLM's `deepspeed` dependency on bare metal, you may need to install InfiniBand verbs library.

**Error you might see:**
```
ERROR: Could not build wheels for deepspeed
```

**Install libibverbs-dev:**
```bash
sudo apt-get update
sudo apt-get install libibverbs-dev
```

**When do you need this?**
- Installing vLLM generation backend on bare metal
- Using high-performance networking (InfiniBand)
- Encountering deepspeed installation errors

### Flash Attention 2 (Performance Optimization)

Flash Attention 2 significantly improves training performance and memory efficiency. However, it can take considerable time to build from source.

**Pre-build and cache Flash Attention** (recommended for bare metal):

```bash
# This builds flash-attn and caches it in uv's cache directory
bash tools/build-flash-attn-in-uv-cache.sh
```

**Build time expectations:**
- ~45 minutes with 48 CPU hyperthreads
- ~90+ minutes with fewer cores
- Only needs to be built once (then cached)

**Expected output on success:**
```
✅ flash-attn successfully added to uv cache
```

**Why pre-build?**
- First-time builds can be very slow
- Subsequent installs use the cached build (much faster)
- Avoids timeout issues during training setup
- Reduces iteration time for development

**When do you need this?**
- Working outside Docker containers
- First-time setup on bare metal
- Improving training performance with packed sequences
- Using long context lengths efficiently

:::{note}
The NeMo RL Docker container already includes pre-built Flash Attention in the uv cache. You only need to manually build if working on bare metal.
:::

**Verification:**
After the build completes, verify Flash Attention is available:
```bash
uv run python -c "import flash_attn; print(flash_attn.__version__)"
```

## Platform-Specific Instructions

### Ubuntu/Debian

1. **Install system dependencies**:
   ```bash
   sudo apt update
   sudo apt install build-essential git curl
   ```

2. **Install CUDA** (if not already installed):
   ```bash
   # Follow NVIDIA's CUDA installation guide
   # https://docs.nvidia.com/cuda/cuda-installation-guide-linux/
   ```

3. **Install NeMo RL**:
   ```bash
   git clone git@github.com:NVIDIA/NeMo-RL.git nemo-rl
   cd nemo-rl
   git submodule update --init --recursive
   uv sync
   ```

### Windows

1. **Install WSL2** (recommended):
   ```bash
   # Follow Microsoft's WSL2 installation guide
   # https://docs.microsoft.com/en-us/windows/wsl/install
   ```

2. **Install CUDA on Windows**:
   - Download and install CUDA Toolkit from NVIDIA
   - Install cuDNN library

3. **Install NeMo RL in WSL2**:
   ```bash
   git clone git@github.com:NVIDIA/NeMo-RL.git nemo-rl
   cd nemo-rl
   git submodule update --init --recursive
   uv sync
   ```

### macOS

1. **Install Homebrew** (if not already installed):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **Install Python and dependencies**:
   ```bash
   brew install python@3.9
   brew install git
   ```

3. **Install NeMo RL**:
   ```bash
   git clone git@github.com:NVIDIA/NeMo-RL.git nemo-rl
   cd nemo-rl
   git submodule update --init --recursive
   uv sync
   ```

## Cluster Setup

### Slurm Cluster

1. **Load required modules**:
   ```bash
   module load cuda/11.8
   module load python/3.9
   ```

2. **Install in shared directory**:
   ```bash
   git clone git@github.com:NVIDIA/NeMo-RL.git /shared/nemo-rl
   cd /shared/nemo-rl
   git submodule update --init --recursive
   uv sync
   ```

### Kubernetes Cluster

1. **Create a Docker image**:
   ```dockerfile
   FROM nvcr.io/nvidia/pytorch:23.12-py3
   
   RUN git clone https://github.com/NVIDIA/NeMo-RL.git /workspace/nemo-rl
   WORKDIR /workspace/nemo-rl
   RUN git submodule update --init --recursive
   RUN pip install -e .
   ```

2. **Deploy to Kubernetes**:
   ```bash
   kubectl apply -f k8s/nemo-rl-deployment.yaml
   ```

## Verification

### Basic Installation Test

Run a simple test to verify the installation:

```bash
uv run python -c "import nemo_rl; print('NeMo RL installed successfully!')"
```

### GPU Test

Verify GPU support:

```bash
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

### Example Run

Test with a simple example:

```bash
cd examples
uv run python run_sft.py --config configs/sft.yaml cluster.gpus_per_node=1
```

## Troubleshooting

### Common Installation Issues

1. **CUDA Version Mismatch**:
   - Ensure CUDA version matches PyTorch requirements
   - Check `nvidia-smi` for driver version
   - Verify PyTorch CUDA installation

2. **Memory Issues**:
   - Increase system memory or use swap
   - Reduce batch sizes in training
   - Use gradient checkpointing

3. **Permission Errors**:
   - Use `sudo` for system-wide installation
   - Check file permissions in installation directory

### Get Help

- **Documentation**: Check the [troubleshooting guide](../guides/troubleshooting)
- **Community**: Join the [NeMo Discord](https://discord.gg/nvidia-nemo)
- **Issues**: Report problems on [GitHub](https://github.com/NVIDIA/NeMo-RL/issues)

## Next Steps

After installation, proceed to:
- [Quickstart Guide](quickstart) - Get started with your first training run
- [Cluster Setup](cluster.md) - Set up distributed training 
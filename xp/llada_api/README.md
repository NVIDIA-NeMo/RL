# LLaDA/Nemotron OpenAI API Server

This directory contains a complete OpenAI-compatible API server implementation for diffusion language models with full support for DCP (Distributed Checkpoint) format checkpoints, SLURM job submission, and **multi-GPU inference**.

**Supported Models:**
- **LLaDA** (Large Language Diffusion Models) with Fast-dLLM and dInfer acceleration
- **Nemotron** models with native diffusion generation

**Key Features:**
- ✅ Batch processing for 3-5x throughput improvement
- ✅ Multi-GPU support for additional parallelism
- ✅ Multiple inference engines (dInfer, Fast-dLLM, Nemotron)
- ✅ DCP checkpoint support
- ✅ SLURM integration for cluster deployments

## Directory Structure

```
xp/llada_api/
├── llada_openai_server.py      # Main FastAPI server implementation
├── scripts/
│   ├── start_llada_batch_server.sh   # Batch server launcher (local & SLURM)
│   └── connect_to_llada_server.sh  # Connection helper for SLURM jobs
├── examples/
│   ├── llada_api_client.py     # Python client examples
│   └── llada_slurm_example.sh  # End-to-end SLURM demo
└── docs/
    └── README.md               # Complete documentation
```

## Quick Start

### From Project Root (Recommended)

Use the convenient wrapper scripts from the NeMo-RL project root:

```bash
# Batch server with LLaDA (auto-selects dInfer engine - 10x+ faster)
./scripts/start_llada_batch_server.sh --local --model-path GSAI-ML/LLaDA-8B-Instruct

# Batch server with explicit Fast-dLLM engine
./scripts/start_llada_batch_server.sh --local --model-path GSAI-ML/LLaDA-8B-Instruct --engine fast-dllm

# Batch server with Nemotron (auto-selects nemotron engine)
./scripts/start_llada_batch_server.sh --local --model-path nvidia/Nemotron-Diffusion-Research-4B-v0

# With specific algorithm within engine
./scripts/start_llada_batch_server.sh --local --model-path GSAI-ML/LLaDA-8B-Instruct --algorithm dinfer_hierarchy

# Multi-GPU inference with 4 GPUs (higher throughput)
./scripts/start_llada_batch_server.sh --local --model-path GSAI-ML/LLaDA-8B-Instruct --num-gpus 4 --batch-size 32

# Local with DCP checkpoint
./scripts/start_llada_batch_server.sh --local --dcp-path /path/to/checkpoint.dcp --base-model GSAI-ML/LLaDA-8B-Instruct

# SLURM execution
export ACCOUNT=your_slurm_account
./scripts/start_llada_batch_server.sh --dcp-path /path/to/checkpoint.dcp --base-model GSAI-ML/LLaDA-8B-Instruct

# SLURM with multi-GPU (8 GPUs for maximum throughput)
# Note: --num-gpus automatically matches --gpus for SLURM
export ACCOUNT=your_slurm_account
./scripts/start_llada_batch_server.sh --model-path GSAI-ML/LLaDA-8B-Instruct --gpus 8 --batch-size 64

# Connect to SLURM server
./connect_to_llada_server.sh --job-id 12345
```

## Performance Guide

### Throughput Optimization

For maximum throughput, combine multiple optimizations:

1. **Use dInfer engine** (10x+ faster than Fast-dLLM for LLaDA)
2. **Enable batch processing** (3-5x additional speedup)
3. **Use multi-GPU** (near-linear scaling with GPU count)

Example optimal configuration:
```bash
./scripts/start_llada_batch_server.sh \
  --local \
  --model-path GSAI-ML/LLaDA-8B-Instruct \
  --engine dinfer \
  --num-gpus 4 \
  --batch-size 32 \
  --max-wait-time 0.05
```

Expected throughput improvements:
- Single GPU, no batching: **1x baseline**
- Single GPU + batching: **3-5x**
- Single GPU + batching + dInfer: **30-50x**
- 4 GPUs + batching + dInfer: **100-200x**

See [docs/MULTI_GPU.md](docs/MULTI_GPU.md) for detailed multi-GPU guide.

**Note**: Multi-GPU mode automatically handles left-padding issues when batch size equals GPU count. See [docs/multi_gpu_padding_fix.md](docs/multi_gpu_padding_fix.md) for technical details.

```

**Inference Engines**:
- `dinfer` - dInfer for LLaDA (10x+ faster, auto-selected)
- `fast-dllm` - Fast-dLLM for LLaDA (alternative)
- `nemotron` - Native Nemotron (auto-selected for Nemotron models)

### From This Directory

Run scripts directly from the llada_api directory:

```bash
cd xp/llada_api

# Local execution with DCP checkpoint
scripts/start_llada_batch_server.sh --local --dcp-path /path/to/checkpoint.dcp --base-model GSAI-ML/LLaDA-8B-Instruct

# Local execution with HuggingFace model (batch mode)
scripts/start_llada_batch_server.sh --local --model-path GSAI-ML/LLaDA-8B-Instruct

# Local execution with streaming
scripts/start_llada_batch_server.sh --local --streaming --model-path GSAI-ML/LLaDA-8B-Instruct

# SLURM execution
export ACCOUNT=your_slurm_account
scripts/start_llada_batch_server.sh --dcp-path /path/to/checkpoint.dcp --base-model GSAI-ML/LLaDA-8B-Instruct

# Connection helper
scripts/connect_to_llada_server.sh --job-id 12345
```

## Features

- **OpenAI API Compatibility**: Drop-in replacement for OpenAI API endpoints
- **Dual Model Support**: Supports both LLaDA and Nemotron diffusion models
- **Fast-dLLM Acceleration**: Enhanced performance for LLaDA with KV caching and parallel decoding
- **Native Nemotron Generation**: Uses Nemotron's built-in diffusion generation
- **DCP Checkpoint Support**: Automatic conversion from DCP to HuggingFace format for both model types
- **SLURM Integration**: Run as containerized SLURM jobs with GPU resources
- **Local Execution**: Run directly on your local machine
- **uv Package Management**: Uses uv for fast, reliable dependency management
- **Batch Processing**: High-throughput batch processing for evaluation workloads
- **Streaming Support**: Real-time streaming responses
- **Connection Helpers**: Automatic SSH tunnel setup for SLURM jobs
- **Comprehensive Examples**: Complete workflow demonstrations

## Key Components

### 1. Main Server
- **`llada_batch_server.py`**: High-throughput batch processing server (RECOMMENDED)
  - Engine-based architecture (fast-dllm, dinfer, nemotron)
  - dInfer support for 10x+ performance improvement
  - Batch processing for 3-5x additional speedup
  - Per-request algorithm switching
  - Systematic model/engine validation
- **`llada_openai_server.py`**: ⚠️ DEPRECATED - Single-request streaming server (use batch server instead)

Features:
- FastAPI-based OpenAI-compatible API with model-specific optimizations
- Supports both HuggingFace and DCP checkpoint loading for LLaDA and Nemotron
- LLaDA: Fast-dLLM and dInfer acceleration with KV caching and parallel decoding
- Nemotron: Native diffusion generation using model's built-in methods
- Includes health checks, error handling, and comprehensive logging

### 2. Server Launcher (`scripts/start_llada_batch_server.sh`)
- Unified launcher for local and SLURM execution modes
- Automatic dependency checking and environment setup
- Configurable SLURM job parameters (GPUs, memory, time, etc.)
- Container-based execution with proper mount points

### 3. Connection Helper (`scripts/connect_to_llada_server.sh`)
- Automatic compute node discovery from job IDs
- SSH tunnel setup instructions
- Connection testing and validation
- Real-time guidance for accessing SLURM-hosted servers

### 4. Examples (`examples/`)
- **`llada_api_client.py`**: Comprehensive Python client with multiple usage patterns
- **`llada_slurm_example.sh`**: End-to-end automated SLURM workflow demo

## Documentation

See [`docs/README.md`](docs/README.md) for complete documentation including:
- Detailed API reference
- Configuration options
- Troubleshooting guides
- Integration examples
- Performance tuning tips

## Integration with NeMo-RL

This implementation integrates seamlessly with the NeMo-RL ecosystem:

- Uses existing DCP checkpoint utilities from `nemo_rl.utils.native_checkpoint`
- Compatible with NeMo-RL container infrastructure (`.sqsh` containers with uv)
- Follows NeMo-RL patterns for SLURM job submission and package management
- Uses uv for dependency management following project standards
- Supports the same model architectures and checkpoint formats

## Development

The server is designed to be:
- **Extensible**: Easy to add new model formats or API endpoints
- **Maintainable**: Clear separation of concerns and comprehensive error handling
- **Testable**: Includes examples and validation scripts
- **Production-ready**: Proper logging, monitoring, and resource management

## Getting Help

1. Check the full documentation in [`docs/README.md`](docs/README.md)
2. Run the demo script: `examples/llada_slurm_example.sh`
3. Use `--help` flag on any script for detailed usage information
4. Check server logs for debugging information

## License

This implementation follows the same license as the NeMo-RL project.

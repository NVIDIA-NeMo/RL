# LLaDA OpenAI API Server

This directory contains a complete OpenAI-compatible API server implementation for LLaDA (Large Language Diffusion Models) with full support for DCP (Distributed Checkpoint) format checkpoints and SLURM job submission.

## Directory Structure

```
xp/llada_api/
├── llada_openai_server.py      # Main FastAPI server implementation
├── scripts/
│   ├── start_llada_server.sh   # Server launcher (local & SLURM)
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
# Easiest local execution (no setup required)
./start_llada_server.sh --local --model-path GSAI-ML/LLaDA-8B-Instruct

# Local with DCP checkpoint (requires dependencies)
./start_llada_server.sh --local --dcp-path /path/to/checkpoint.dcp --base-model GSAI-ML/LLaDA-8B-Instruct

# SLURM execution (full NeMo-RL environment)
export ACCOUNT=your_slurm_account
./start_llada_server.sh --dcp-path /path/to/checkpoint.dcp --base-model GSAI-ML/LLaDA-8B-Instruct

# Connect to SLURM server
./connect_to_llada_server.sh --job-id 12345
```

### From This Directory

Run scripts directly from the llada_api directory:

```bash
cd xp/llada_api

# Local execution with DCP checkpoint
scripts/start_llada_server.sh --local --dcp-path /path/to/checkpoint.dcp --base-model GSAI-ML/LLaDA-8B-Instruct

# Local execution with HuggingFace model
scripts/start_llada_server.sh --local --model-path GSAI-ML/LLaDA-8B-Instruct

# SLURM execution
export ACCOUNT=your_slurm_account
scripts/start_llada_server.sh --dcp-path /path/to/checkpoint.dcp --base-model GSAI-ML/LLaDA-8B-Instruct

# Connection helper
scripts/connect_to_llada_server.sh --job-id 12345
```

## Features

- **OpenAI API Compatibility**: Drop-in replacement for OpenAI API endpoints
- **DCP Checkpoint Support**: Automatic conversion from DCP to HuggingFace format
- **SLURM Integration**: Run as containerized SLURM jobs with GPU resources
- **Local Execution**: Run directly on your local machine
- **uv Package Management**: Uses uv for fast, reliable dependency management
- **LLaDA-Specific Parameters**: Full support for diffusion generation parameters
- **Streaming Support**: Real-time streaming responses
- **Connection Helpers**: Automatic SSH tunnel setup for SLURM jobs
- **Comprehensive Examples**: Complete workflow demonstrations

## Key Components

### 1. Main Server (`llada_openai_server.py`)
- FastAPI-based OpenAI-compatible API server
- Supports both HuggingFace and DCP checkpoint loading
- Implements LLaDA diffusion generation with configurable parameters
- Includes health checks, error handling, and comprehensive logging

### 2. Server Launcher (`scripts/start_llada_server.sh`)
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

# LLaDA OpenAI API Server

A complete OpenAI-compatible API server for LLaDA (Large Language Diffusion Models) with DCP checkpoint support and SLURM integration.

## Location

The LLaDA API server implementation is located in:
```
xp/llada_api/
```

## Quick Usage

### Convenient Scripts (Project Root)

For ease of use, convenient wrapper scripts are provided at the project root:

```bash
# Easiest: Start server locally with HuggingFace model (no setup required)
./start_llada_server.sh --local --model-path GSAI-ML/LLaDA-8B-Instruct

# Start server locally with DCP checkpoint (requires NeMo-RL dependencies)
./start_llada_server.sh --local --dcp-path /path/to/checkpoint.dcp --base-model GSAI-ML/LLaDA-8B-Instruct

# Start server on SLURM
export ACCOUNT=your_slurm_account
./start_llada_server.sh --dcp-path /path/to/checkpoint.dcp --base-model GSAI-ML/LLaDA-8B-Instruct

# Connect to SLURM server
./connect_to_llada_server.sh --job-id 12345
```

### Direct Usage

```bash
cd xp/llada_api

# Use scripts directly
scripts/start_llada_server.sh --local --model-path GSAI-ML/LLaDA-8B-Instruct  # HF Hub model
scripts/start_llada_server.sh --local --model-path /path/to/model              # Local model
scripts/connect_to_llada_server.sh --job-id 12345

# Run examples
python examples/llada_api_client.py
examples/llada_slurm_example.sh
```

## Features

- ✅ **OpenAI API Compatible** - Drop-in replacement for OpenAI endpoints
- ✅ **DCP Checkpoint Support** - Automatic conversion from DCP format
- ✅ **SLURM Integration** - Run as containerized jobs with GPU resources  
- ✅ **Local Execution** - Run on your local machine
- ✅ **uv Package Management** - Uses uv for fast, reliable dependency management
- ✅ **LLaDA Parameters** - Full diffusion generation control (steps, temperature, CFG)
- ✅ **Streaming Support** - Real-time response streaming
- ✅ **Connection Helpers** - Automatic SSH tunnel setup for SLURM
- ✅ **Complete Examples** - End-to-end workflow demonstrations

## Documentation

See the complete documentation at: [`xp/llada_api/docs/README.md`](xp/llada_api/docs/README.md)

## Directory Structure

```
xp/llada_api/
├── llada_openai_server.py          # Main FastAPI server
├── scripts/
│   ├── start_llada_server.sh       # Server launcher (local & SLURM)
│   └── connect_to_llada_server.sh  # SLURM connection helper
├── examples/
│   ├── llada_api_client.py         # Python client examples
│   └── llada_slurm_example.sh      # End-to-end SLURM demo
├── docs/
│   └── README.md                   # Complete documentation
└── README.md                       # Quick reference
```

## Integration with NeMo-RL

This server integrates seamlessly with NeMo-RL:
- Uses `nemo_rl.utils.native_checkpoint` for DCP conversion
- Compatible with existing container infrastructure
- Follows NeMo-RL SLURM patterns and conventions
- Supports all NeMo-RL trained LLaDA checkpoints

## Getting Started

1. **Read the docs**: [`xp/llada_api/docs/README.md`](xp/llada_api/docs/README.md)
2. **Try the demo**: `xp/llada_api/examples/llada_slurm_example.sh`
3. **Use convenience scripts**: `./start_llada_server.sh --help`

The server provides a production-ready way to serve your LLaDA models with full OpenAI API compatibility!

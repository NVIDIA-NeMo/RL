# LLaDA/Nemotron API - Complete Guide

Complete documentation for the OpenAI-compatible API server for LLaDA and Nemotron diffusion language models.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Multi-GPU Setup](#multi-gpu-setup)
3. [API Reference](#api-reference)
4. [Performance & Optimization](#performance--optimization)
5. [Troubleshooting](#troubleshooting)
6. [Advanced Topics](#advanced-topics)

---

## Quick Start

### Supported Models

- **LLaDA** (Large Language Diffusion Models) - Diffusion-based generation with Fast-dLLM/dInfer acceleration
- **Nemotron** - NVIDIA's diffusion language models with native generation

### Installation

```bash
# Minimal setup (HuggingFace models only)
pip install fastapi uvicorn torch transformers

# Full setup (includes DCP checkpoint support)
uv sync --locked --no-install-project
uv pip install fastapi uvicorn
```

### Launch Server

**Local (Single GPU)**:
```bash
# HuggingFace model (easiest)
export HF_TOKEN=your_token_here  # Optional but recommended
./xp/llada_api/scripts/start_llada_batch_server.sh \
  --local \
  --model-path GSAI-ML/LLaDA-8B-Instruct

# Or Nemotron
./xp/llada_api/scripts/start_llada_batch_server.sh \
  --local \
  --model-path nvidia/Nemotron-Diffusion-Research-4B-v0
```

**Local (Multi-GPU)**:
```bash
# Multi-GPU automatically enabled when --gpus > 1
export HF_TOKEN=your_token_here
./xp/llada_api/scripts/start_llada_batch_server.sh \
  --local \
  --gpus 8 \
  --model-path GSAI-ML/LLaDA-8B-Instruct
```

**SLURM**:
```bash
export ACCOUNT=your_account
export HF_TOKEN=your_token_here

# Single GPU
./xp/llada_api/scripts/start_llada_batch_server.sh \
  --model-path GSAI-ML/LLaDA-8B-Instruct

# Multi-GPU (auto-enabled)
./xp/llada_api/scripts/start_llada_batch_server.sh \
  --gpus 8 \
  --model-path GSAI-ML/LLaDA-8B-Instruct
```

### Basic Usage

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

response = client.chat.completions.create(
    model="llada-8b-instruct",
    messages=[{"role": "user", "content": "What is AI?"}],
    max_tokens=128,
    extra_body={"steps": 128}  # LLaDA-specific
)

print(response.choices[0].message.content)
```

---

## Multi-GPU Setup

### Architecture

```
Client → Load Balancer (port 8000)
           ↓
    ┌──────┼──────┐
    ↓      ↓      ↓
Worker 0  Worker 1  Worker 2
(GPU 0)   (GPU 1)   (GPU 2)
port 8001 port 8002 port 8003
```

**Why load balancing instead of DataParallel?**
- ✅ Avoids left-padding issues
- ✅ Linear scaling
- ✅ Better fault tolerance
- ✅ Simple debugging

### Quick Start

```bash
# 8 GPUs with automatic load balancing
export HF_TOKEN=your_token_here
./xp/llada_api/scripts/start_llada_batch_server.sh \
  --local \
  --gpus 8 \
  --model-path GSAI-ML/LLaDA-8B-Instruct \
  --batch-size 16
```

### Monitoring

```bash
# Check load balancer stats
curl http://localhost:8000/stats

# View logs
tail -f /tmp/llada_load_balancer.log
tail -f /tmp/llada_worker_0.log
```

### Performance

| GPUs | Throughput | Speedup |
|------|------------|---------|
| 1    | 12 req/s   | 1×      |
| 2    | 22 req/s   | 1.8×    |
| 4    | 44 req/s   | 3.7×    |
| 8    | 86 req/s   | 7.2×    |

---

## API Reference

### POST `/v1/chat/completions`

**Standard OpenAI Parameters**:
- `model`: Model name
- `messages`: Chat messages array
- `temperature`: 0.0-2.0 (default: 0.0)
- `max_tokens`: Max generation length (default: 128)
- `stream`: Enable streaming (default: false)

**LLaDA/Nemotron Parameters**:
- `steps`: Diffusion steps (1-512, default: 128)
  - More steps = higher quality, slower
  - Recommended: 64-256
- `block_length`: Semi-autoregressive block size (default: 32)
- `remasking`: Token selection ("low_confidence" | "random")
- `threshold`: Confidence threshold (0.0-1.0, optional)
- `factor`: Parallel decoding factor (1.0-4.0, optional)

**Fast-dLLM/dInfer Parameters** (LLaDA only):
- Engine selection (auto-detected):
  - `dinfer`: 10x+ faster (default for LLaDA)
  - `fast-dllm`: Alternative acceleration
- Algorithm selection via `generation_algorithm`:
  - `dinfer_blockwise`, `dinfer_hierarchy`, `dinfer_credit`
  - `basic`, `prefix_cache`, `dual_cache`

### Other Endpoints

- `GET /health` - Health check
- `GET /v1/models` - List models
- `GET /batch/stats` - Batch statistics (batch server)
- `GET /stats` - Load balancer stats (multi-GPU)
- `GET /generation/algorithms` - Available algorithms

---

## Performance & Optimization

### Speed vs Quality Trade-offs

```python
# Maximum speed (3-5x faster)
fast_config = {
    "steps": 32,
    "max_tokens": 80,
    "temperature": 0.0
}

# Balanced
balanced_config = {
    "steps": 128,
    "max_tokens": 128,
    "temperature": 0.0
}

# High quality
quality_config = {
    "steps": 256,
    "max_tokens": 128,
    "temperature": 0.0
}
```

### Batch Processing

The batch server automatically batches requests:

```bash
# Optimize batch settings
--batch-size 16         # Max requests per batch
--max-wait-time 0.1     # Max wait time (seconds)
```

### Multi-GPU Optimization

```bash
# Increase per-worker batch size
--batch-size 16

# Reduce latency
--max-wait-time 0.05

# Use more workers
--gpus 8
```

### Engine Selection

```bash
# LLaDA: Use dInfer (10x+ faster than Fast-dLLM)
--engine dinfer --algorithm dinfer_hierarchy

# Or Fast-dLLM
--engine fast-dllm --algorithm dual_cache

# Nemotron: Auto-selects nemotron engine
--model-path nvidia/Nemotron-Diffusion-Research-4B-v0
```

---

## Troubleshooting

### Common Issues

#### 1. HuggingFace Rate Limiting (429 Error)

**Problem**: `429 Too Many Requests` when starting multi-GPU with HF models

**Solution**: Set HF_TOKEN
```bash
export HF_TOKEN=your_token_here
# Get token from: https://huggingface.co/settings/tokens
```

**How it works**: Scripts now pre-cache model metadata before starting workers to prevent simultaneous API calls.

#### 2. Workers Crash Intermittently

**Problem**: Some workers crash during multi-GPU startup

**Cause**: Race conditions from simultaneous startup (now fixed)

**Solution**: Update to latest scripts (includes automatic fixes):
- 1s staggered worker startup
- 20s initialization wait
- Health verification before load balancer starts

#### 3. Out of Memory

**Problem**: CUDA OOM errors

**Solutions**:
```bash
# Reduce batch size
--batch-size 4

# Or use fewer GPUs
--gpus 4

# Or use smaller model
--model-path GSAI-ML/LLaDA-4B-Instruct  # If available
```

#### 4. Import Errors

**Problem**: `ModuleNotFoundError: No module named 'nemo_rl'`

**Solutions**:
```bash
# Use HuggingFace models (no NeMo-RL needed)
--model-path GSAI-ML/LLaDA-8B-Instruct

# Or install full dependencies for DCP support
uv sync --locked --no-install-project
```

#### 5. Connection Issues

**Local**: Check server is running
```bash
curl http://localhost:8000/health
```

**SLURM**: Set up SSH tunnel (instructions shown in job output)
```bash
ssh -N -L 8000:compute-node:8000 user@login-node
```

### Performance Issues

**Slow generation**:
- Reduce `steps` (64 instead of 256)
- Use dInfer engine for LLaDA
- Enable batch processing
- Use multi-GPU for throughput

**High latency**:
- Reduce `--max-wait-time`
- Use streaming responses
- Reduce batch size

### Quick Diagnostic Commands

```bash
# Check if server is running
ps aux | grep llada_batch_server

# Test health
curl http://localhost:8000/health

# Check load balancer stats (multi-GPU)
curl http://localhost:8000/stats | jq

# View logs
tail -f /tmp/llada_worker_0.log

# Check GPU usage
nvidia-smi
```

---

## Advanced Topics

### DCP Checkpoint Loading

**What are DCP checkpoints?**
- Distributed Checkpoint format from NeMo-RL training
- Automatically converted to HuggingFace format

**Usage**:
```bash
./xp/llada_api/scripts/start_llada_batch_server.sh \
  --local \
  --dcp-path /path/to/checkpoint.dcp \
  --base-model GSAI-ML/LLaDA-8B-Instruct
```

**Multi-GPU with DCP**:
- Conversion happens ONCE before workers start
- All workers load from shared converted checkpoint
- Prevents race conditions

### Custom Configurations

**Timeout settings** (for long evaluations):
```bash
# Workers
--timeout-keep-alive 9000  # 2.5 hours

# Load balancer
--request-timeout 12000    # 3.3 hours
```

**Specific GPU IDs** (local only):
```bash
--gpu-ids 0,2,4,6  # Use specific GPUs
```

**Custom ports**:
```bash
--port 9000 --worker-base-port 9001
```

### Integration Examples

**OpenAI Python Library**:
```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

response = client.chat.completions.create(
    model="llada-8b-instruct",
    messages=[{"role": "user", "content": "Hello!"}],
    extra_body={"steps": 128}
)
```

**Direct HTTP**:
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 64,
    "steps": 64
  }'
```

### Architecture Deep Dive

**Single GPU Flow**:
```
Client → FastAPI Server → Model → Response
```

**Multi-GPU Flow**:
```
Client → Load Balancer → Worker (Round-robin)
                       → Worker
                       → Worker
```

**Batch Processing**:
1. Request arrives at worker
2. Added to batch queue
3. Wait for batch to fill OR timeout
4. Process entire batch on GPU
5. Return individual responses

### Performance Characteristics

**Latency** (single request):
- Single GPU: ~2-5s (128 steps)
- Multi-GPU: ~2-5s (same, distributed)

**Throughput** (concurrent requests):
- Single GPU: ~12 req/s
- 8 GPUs: ~86 req/s (7.2× speedup)

**Memory Usage**:
- Model: ~16GB GPU memory (LLaDA-8B)
- Per request: ~50MB additional
- Batch overhead: Minimal

---

## Summary

### Quick Reference

**Start server**:
```bash
# Local single GPU
export HF_TOKEN=token
./start_llada_batch_server.sh --local --model-path MODEL_NAME

# Local multi-GPU
./start_llada_batch_server.sh --local --gpus 8 --model-path MODEL_NAME

# SLURM
export ACCOUNT=account
./start_llada_batch_server.sh --gpus 8 --model-path MODEL_NAME
```

**Test server**:
```bash
curl http://localhost:8000/health
python xp/llada_api/examples/llada_api_client.py
```

**Monitor**:
```bash
curl http://localhost:8000/stats  # Multi-GPU
tail -f /tmp/llada_worker_0.log
```

### Key Features

- ✅ OpenAI-compatible API
- ✅ Multi-GPU load balancing
- ✅ Automatic batch processing
- ✅ Fast-dLLM/dInfer acceleration (LLaDA)
- ✅ DCP checkpoint support
- ✅ HuggingFace model support
- ✅ Streaming responses
- ✅ SLURM integration

### Best Practices

1. **Always set HF_TOKEN** for multi-GPU setups
2. **Start with HuggingFace models** for testing
3. **Use dInfer engine** for LLaDA (10x+ faster)
4. **Enable batch processing** for throughput
5. **Use multi-GPU** for large-scale evaluations
6. **Monitor logs** during development
7. **Test single GPU** before scaling to multi-GPU

---

For quick commands and links, see [README.md](README.md).

For a 5-minute getting started guide, see [QUICK_START.md](QUICK_START.md).

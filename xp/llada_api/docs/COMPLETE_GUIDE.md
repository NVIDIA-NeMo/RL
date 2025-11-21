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
           ↓ Centralized Batching
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

### Centralized Batching

The load balancer implements **centralized batching** for optimal GPU utilization:

**How it works**:
1. **Request Accumulation**: Load balancer collects incoming requests
2. **Batch Formation**: Forms batches (default: 8 requests, 20ms timeout)
3. **Parallel Dispatch**: Sends entire batch to least-loaded worker
4. **Round-Robin Distribution**: Batches distributed across all workers
5. **True Parallelization**: Multiple workers process different batches simultaneously

**Benefits**:
- ✅ **Better GPU utilization**: Workers receive full batches instead of single requests
- ✅ **Parallel processing**: All workers active simultaneously
- ✅ **Load balancing**: Batches distributed evenly across workers
- ✅ **Lower latency**: No artificial wait times, immediate dispatch when batch ready

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

# Check worker health and activity
curl http://localhost:8000/worker-status

# Use dedicated monitoring tool
python xp/llada_api/check_worker_status.py

# Monitor continuously
python xp/llada_api/check_worker_status.py --monitor

# View logs
tail -f /tmp/llada_load_balancer.log
tail -f /tmp/llada_worker_0.log
```

**Key metrics to monitor**:
- `healthy_workers`: Number of active workers
- `avg_batch_size`: Batching efficiency (should be close to max batch size)
- `batch_size_histogram`: Distribution of batch sizes
- `system_status`: Overall system health ("normal" vs "overloaded")
- `worker_status`: Individual worker states ("busy", "idle", "unhealthy")

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
  - Recommended: 32-64 for optimal performance with dInfer v0.1
  - Larger values (64) may improve throughput but increase memory usage
- `remasking`: Token selection ("low_confidence" | "random")
- `threshold`: Confidence threshold (0.0-1.0, optional)
- `factor`: Parallel decoding factor (1.0-4.0, optional)

**Fast-dLLM/dInfer Parameters** (LLaDA only):
- Engine selection (auto-detected):
  - `dinfer`: 10x+ faster (default for LLaDA, updated to v0.1)
  - `fast-dllm`: Alternative acceleration
- Algorithm selection via `generation_algorithm`:
  - dInfer algorithms (recommended):
    - `dinfer_blockwise` - Threshold-based parallel decoding (recommended)
    - `dinfer_hierarchy` - Hierarchical parallel decoding
    - `dinfer_credit` - Credit-based parallel decoding with EMA fusion
    - `dinfer_soft` - Soft token sampling (experimental)
  - Fast-dLLM algorithms:
    - `basic`, `prefix_cache`, `dual_cache`

### Other Endpoints

- `GET /health` - Health check
- `GET /v1/models` - List models
- `GET /batch/stats` - Batch statistics (batch server)
- `GET /stats` - Load balancer stats (multi-GPU)
- `GET /worker-status` - Worker health and activity status (multi-GPU)
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
# LLaDA: Use dInfer v0.1 (10x+ faster than Fast-dLLM)
--engine dinfer --algorithm dinfer_blockwise

# Or with hierarchical decoding for enhanced parallel strategy
--engine dinfer --algorithm dinfer_hierarchy

# Or Fast-dLLM
--engine fast-dllm --algorithm dual_cache

# Nemotron: Auto-selects nemotron engine
--model-path nvidia/Nemotron-Diffusion-Research-4B-v0
```

**dInfer v0.1 Updates**:
- New credit-based decoding algorithm (`dinfer_credit`)
- Improved hierarchical decoding with segment-based strategies
- Enhanced KV-cache management with vicinity refresh
- Recommended `block_length` values: 32-64 for optimal performance

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

# For dInfer: Use recommended block_length (32-64)
# Very small block_length values (e.g., 8) can increase memory usage
# due to more diffusion iterations per generation
```

**Note**: With dInfer v0.1, using `block_length` between 32-64 provides optimal memory/performance trade-off. Values too small (<16) or too large (>128) may cause OOM or performance issues.

#### 4. "No Healthy Workers Available" (Multi-GPU)

**Problem**: `503: No healthy workers available` error during high load

**Cause**: Workers marked unhealthy when they're just busy processing batches

**Solution**: System now automatically handles busy workers (v2.0+):
- Health checks are more lenient for recently active workers
- Workers stay healthy even during long batch processing
- Automatic recovery when all workers appear unavailable

**Manual check**:
```bash
# Check worker status
python xp/llada_api/check_worker_status.py

# Look for "busy" workers (this is good - means system is working)
curl http://localhost:8000/worker-status
```

#### 5. Import Errors

**Problem**: `ModuleNotFoundError: No module named 'nemo_rl'`

**Solutions**:
```bash
# Use HuggingFace models (no NeMo-RL needed)
--model-path GSAI-ML/LLaDA-8B-Instruct

# Or install full dependencies for DCP support
uv sync --locked --no-install-project
```

#### 6. Connection Issues

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
- Use dInfer v0.1 engine for LLaDA (10x+ faster than Fast-dLLM)
  - Recommended: `--engine dinfer --algorithm dinfer_blockwise`
- Use optimal `block_length` (32-64)
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

# Check worker status and activity (multi-GPU)
curl http://localhost:8000/worker-status | jq
python xp/llada_api/check_worker_status.py

# Monitor worker health continuously
python xp/llada_api/check_worker_status.py --monitor

# View logs
tail -f /tmp/llada_worker_0.log
tail -f /tmp/llada_load_balancer.log

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

**Multi-GPU Flow (Centralized Batching)**:
```
Client → Load Balancer (Batch Formation) → Worker 0 (Full Batch)
       → Load Balancer (Batch Formation) → Worker 1 (Full Batch)  
       → Load Balancer (Batch Formation) → Worker 2 (Full Batch)
```

**Traditional Round-Robin** (legacy):
```
Client → Load Balancer → Worker (Single Request)
                       → Worker (Single Request)
                       → Worker (Single Request)
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
- ✅ Fast-dLLM/dInfer v0.1 acceleration (LLaDA)
  - 10x+ speedup with dInfer
  - Multiple decoding strategies (threshold, hierarchical, credit-based)
- ✅ DCP checkpoint support
- ✅ HuggingFace model support
- ✅ Streaming responses
- ✅ SLURM integration

### Best Practices

1. **Always set HF_TOKEN** for multi-GPU setups
2. **Start with HuggingFace models** for testing
3. **Use dInfer v0.1 engine** for LLaDA (10x+ faster than Fast-dLLM)
   - Recommended algorithm: `dinfer_blockwise` for general use
   - Use `block_length` 32-64 for optimal performance
4. **Enable batch processing** for throughput
5. **Use multi-GPU** for large-scale evaluations
6. **Monitor logs** during development
7. **Test single GPU** before scaling to multi-GPU

---

For quick commands and links, see [README.md](README.md).

For a 5-minute getting started guide, see [QUICK_START.md](QUICK_START.md).

# LLaDA/Nemotron API - Quick Start

Fast reference for getting started with the LLaDA/Nemotron inference API.

---

## Installation

```bash
# Minimal (HuggingFace models only)
pip install fastapi uvicorn torch transformers

# Full (includes DCP support)
uv sync --locked --no-install-project
uv pip install fastapi uvicorn
```

## Launch Server

### Single GPU (Local)

```bash
export HF_TOKEN=your_token_here  # Get from https://huggingface.co/settings/tokens

# LLaDA
./xp/llada_api/scripts/start_llada_batch_server.sh \
  --local \
  --model-path GSAI-ML/LLaDA-8B-Instruct

# Nemotron
./xp/llada_api/scripts/start_llada_batch_server.sh \
  --local \
  --model-path nvidia/Nemotron-Diffusion-Research-4B-v0
```

### Multi-GPU (Local)

```bash
export HF_TOKEN=your_token_here

# Automatically uses load balancing when --gpus > 1
./xp/llada_api/scripts/start_llada_batch_server.sh \
  --local \
  --gpus 8 \
  --model-path GSAI-ML/LLaDA-8B-Instruct \
  --batch-size 16
```

### SLURM

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

## Use the API

### Python (OpenAI Library)

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
    extra_body={"steps": 128}  # LLaDA diffusion steps
)

print(response.choices[0].message.content)
```

### Python (Requests)

```python
import requests

response = requests.post("http://localhost:8000/v1/chat/completions", json={
    "messages": [{"role": "user", "content": "What is AI?"}],
    "max_tokens": 128,
    "steps": 128
})

print(response.json()["choices"][0]["message"]["content"])
```

### cURL

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 64,
    "steps": 64
  }'
```

## Common Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `steps` | Diffusion steps (more = better quality) | 128 | 1-512 |
| `max_tokens` | Max output length | 128 | - |
| `temperature` | Sampling temperature | 0.0 | 0.0-2.0 |
| `block_length` | Semi-autoregressive block size | 32 | - |

## Speed vs Quality

```python
# Fast (~2s)
{"steps": 32, "max_tokens": 80}

# Balanced (~4s)
{"steps": 128, "max_tokens": 128}

# High quality (~8s)
{"steps": 256, "max_tokens": 128}
```

## Test the Server

```bash
# Health check
curl http://localhost:8000/health

# Load balancer stats (multi-GPU only)
curl http://localhost:8000/stats

# Run test script
python xp/llada_api/examples/llada_api_client.py
```

## Common Issues & Fixes

### Rate Limiting (429 Error)

**Problem**: Multi-GPU startup fails with "Too Many Requests"

**Fix**: Set HF_TOKEN
```bash
export HF_TOKEN=your_token_here
# Get from: https://huggingface.co/settings/tokens
```

### Workers Crash

**Problem**: Some workers crash during startup

**Fix**: Already fixed in latest scripts (staggered startup + health checks)

### Out of Memory

**Problem**: CUDA OOM

**Fix**: Reduce batch size or use fewer GPUs
```bash
--batch-size 4  # Or lower
--gpus 4        # Or fewer
```

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'nemo_rl'`

**Fix**: Use HuggingFace models (no NeMo-RL needed)
```bash
--model-path GSAI-ML/LLaDA-8B-Instruct
```

## Monitor Performance

```bash
# Check logs
tail -f /tmp/llada_worker_0.log          # Worker logs
tail -f /tmp/llada_load_balancer.log     # Load balancer log

# GPU usage
nvidia-smi -l 1

# Load balancer stats
curl http://localhost:8000/stats | jq
```

## Multi-GPU Performance

| GPUs | Throughput | Speedup |
|------|------------|---------|
| 1    | ~12 req/s  | 1×      |
| 2    | ~22 req/s  | 1.8×    |
| 4    | ~44 req/s  | 3.7×    |
| 8    | ~86 req/s  | 7.2×    |

## Advanced Options

### Engine Selection (LLaDA only)

```bash
# dInfer (10x+ faster) - auto-selected for LLaDA
--engine dinfer --algorithm dinfer_hierarchy

# Experimental: Soft Token Sampling
--engine dinfer --algorithm dinfer_soft

# Fast-dLLM (alternative)
--engine fast-dllm --algorithm dual_cache
```

### Batch Processing

```bash
--batch-size 16         # Max requests per batch
--max-wait-time 0.1     # Max wait time (seconds)
```

### Custom Ports

```bash
--port 9000                    # Single GPU
--port 9000 --worker-base-port 9001  # Multi-GPU
```

### DCP Checkpoints

```bash
./xp/llada_api/scripts/start_llada_batch_server.sh \
  --local \
  --dcp-path /path/to/checkpoint.dcp \
  --base-model GSAI-ML/LLaDA-8B-Instruct
```

## Cheat Sheet

```bash
# Local single GPU
export HF_TOKEN=token
./start_llada_batch_server.sh --local --model-path MODEL

# Local multi-GPU (8 GPUs)
export HF_TOKEN=token
./start_llada_batch_server.sh --local --gpus 8 --model-path MODEL

# SLURM multi-GPU
export ACCOUNT=account HF_TOKEN=token
./start_llada_batch_server.sh --gpus 8 --model-path MODEL

# Test
curl http://localhost:8000/health

# Monitor
curl http://localhost:8000/stats | jq
tail -f /tmp/llada_worker_0.log
```

## Next Steps

- See `COMPLETE_GUIDE.md` for detailed documentation
- Check `examples/` directory for more usage examples
- Read `TROUBLESHOOTING.md` for common issues

## Key Points

✅ **Always set HF_TOKEN** for multi-GPU with HuggingFace models
✅ **Use dInfer engine** for LLaDA (10x+ faster)
✅ **Multi-GPU auto-enabled** when `--gpus > 1`
✅ **Scripts pre-cache models** to prevent rate limiting
✅ **Health checks** ensure workers are ready before serving
✅ **Batch processing** for maximum throughput


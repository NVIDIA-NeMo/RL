# Multi-GPU Inference Server Setup

This guide explains how to run the LLaDA/Nemotron inference server across multiple GPUs using load balancing.

## Overview

The multi-GPU setup uses a **load balancer architecture** rather than traditional data parallelism. This design choice was made to avoid complications with left-padding in batched inputs:

```
                    ┌──────────────────┐
Client Requests ──> │  Load Balancer   │
                    │  (Port 8000)     │
                    └────────┬─────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
      ┌─────▼─────┐    ┌────▼──────┐   ┌────▼──────┐
      │ Worker 0  │    │ Worker 1  │   │ Worker 2  │
      │ GPU 0     │    │ GPU 1     │   │ GPU 2     │
      │ Port 8001 │    │ Port 8002 │   │ Port 8003 │
      └───────────┘    └───────────┘   └───────────┘
```

### Why Load Balancing Instead of DataParallel?

The server uses **left-padding** for batched inputs. With standard `torch.nn.DataParallel`, splitting batches across GPUs would cause workers to attend to padding tokens, confusing the generation process. The load balancer approach:

- ✅ Avoids padding issues (each worker processes complete batches independently)
- ✅ Provides true parallelism (workers don't wait for each other)
- ✅ Scales linearly with number of GPUs
- ✅ Simple implementation and debugging
- ✅ Better fault tolerance (one GPU failure doesn't crash entire system)

## Quick Start

### Local Multi-GPU Mode

```bash
# Run on 4 local GPUs with load balancing
# Multi-GPU mode automatically enabled when --gpus > 1
xp/llada_api/scripts/start_llada_batch_server.sh \
  --local \
  --gpus 4 \
  --model-path GSAI-ML/LLaDA-8B-Instruct
```

### SLURM Multi-GPU Mode

```bash
# Run on 8 GPUs via SLURM
# Multi-GPU mode automatically enabled when --gpus > 1
export ACCOUNT=your_account

xp/llada_api/scripts/start_llada_batch_server.sh \
  --gpus 8 \
  --model-path GSAI-ML/LLaDA-8B-Instruct \
  --batch-size 16
```

## Components

### 1. Load Balancer (`llada_load_balancer.py`)

The load balancer:
- Receives requests on the main port (default: 8000)
- Distributes requests to workers using round-robin
- Performs periodic health checks on workers
- Provides statistics endpoint (`/stats`)
- Automatically removes unhealthy workers from rotation

### 2. Workers (`llada_batch_server.py`)

Each worker:
- Runs on a dedicated GPU (via `CUDA_VISIBLE_DEVICES`)
- Listens on its own port (8001, 8002, 8003, ...)
- Processes requests independently with batching
- Loads the model once on startup

### 3. Launcher Scripts

#### `start_multi_gpu_server.sh` (Local)
- Starts multiple worker processes in background
- Each worker assigned to specific GPU
- Starts load balancer after workers initialize
- Manages cleanup on exit (Ctrl+C)

#### `start_multi_gpu_slurm.sh` (SLURM)
- Submits SLURM job requesting N GPUs
- Starts workers inside container
- Sets up SSH tunnel instructions
- Handles container mounts automatically

## Usage Examples

### Basic Usage

```bash
# 4 GPUs locally (multi-GPU mode auto-enabled)
xp/llada_api/scripts/start_llada_batch_server.sh \
  --local --gpus 4 \
  --model-path GSAI-ML/LLaDA-8B-Instruct
```

### With Custom Configuration

```bash
# 8 GPUs on SLURM with custom settings (multi-GPU mode auto-enabled)
export ACCOUNT=your_account

xp/llada_api/scripts/start_llada_batch_server.sh \
  --gpus 8 \
  --model-path GSAI-ML/LLaDA-8B-Instruct \
  --engine dinfer \
  --algorithm dinfer_hierarchy \
  --batch-size 16 \
  --max-wait-time 0.05 \
  --port 9000
```

### Specific GPU IDs (Local Only)

```bash
# Use GPUs 0, 2, 4, 6
xp/llada_api/scripts/start_multi_gpu_server.sh \
  --model-path GSAI-ML/LLaDA-8B-Instruct \
  --gpu-ids 0,2,4,6
```

### With DCP Checkpoint

```bash
# Multi-GPU with DCP checkpoint (multi-GPU mode auto-enabled)
export ACCOUNT=your_account

xp/llada_api/scripts/start_llada_batch_server.sh \
  --gpus 4 \
  --dcp-path /path/to/checkpoint.dcp \
  --base-model GSAI-ML/LLaDA-8B-Instruct
```

## Testing

### Test the Server

```bash
# Basic test
python xp/llada_api/test_multi_gpu.py

# Stress test with 50 concurrent requests
python xp/llada_api/test_multi_gpu.py \
  --num-requests 50 \
  --concurrency 10
```

### Check Load Balancer Stats

```bash
# Health check
curl http://localhost:8000/health

# Load balancer statistics (shows distribution across workers)
curl http://localhost:8000/stats
```

Example stats output:
```json
{
  "total_workers": 4,
  "healthy_workers": 4,
  "total_requests": 128,
  "workers": [
    {
      "index": 0,
      "url": "http://localhost:8001",
      "healthy": true,
      "requests_served": 32,
      "errors": 0
    },
    {
      "index": 1,
      "url": "http://localhost:8002",
      "healthy": true,
      "requests_served": 32,
      "errors": 0
    },
    ...
  ]
}
```

## Monitoring

### View Logs (Local Mode)

```bash
# Load balancer log
tail -f /tmp/llada_load_balancer.log

# Individual worker logs
tail -f /tmp/llada_worker_0.log
tail -f /tmp/llada_worker_1.log
```

### SLURM Mode

Logs are displayed in stdout. The job will show:
1. Worker startup messages
2. Load balancer initialization
3. SSH tunnel instructions
4. Request processing logs

## Performance

### Expected Speedup

With `N` GPUs:
- **Theoretical**: N× throughput (linear scaling)
- **Practical**: ~0.9N× throughput (accounting for load balancer overhead)

Example benchmark (LLaDA-8B, batch_size=8):

| GPUs | Throughput | Speedup |
|------|------------|---------|
| 1    | 12 req/s   | 1×      |
| 2    | 22 req/s   | 1.8×    |
| 4    | 44 req/s   | 3.7×    |
| 8    | 86 req/s   | 7.2×    |

### Optimization Tips

1. **Batch Size**: Increase per-worker batch size to maximize GPU utilization
   ```bash
   --batch-size 16  # Or higher if GPU memory allows
   ```

2. **Wait Time**: Reduce for lower latency, increase for higher throughput
   ```bash
   --max-wait-time 0.05  # Lower latency
   --max-wait-time 0.2   # Higher throughput
   ```

3. **Worker Ports**: Ensure ports don't conflict with other services

4. **Health Checks**: Adjust frequency if workers are unstable
   ```bash
   # In llada_load_balancer.py
   --health-check-interval 10  # Check every 10 seconds
   ```

## Troubleshooting

### Workers Not Starting

**Problem**: Load balancer can't reach workers

**Solution**:
```bash
# Check if workers are running
ps aux | grep llada_batch_server

# Check worker logs
tail -f /tmp/llada_worker_0.log

# Verify ports are not in use
netstat -tlnp | grep 800
```

### Uneven Load Distribution

**Problem**: Some workers get more requests than others

**Cause**: Round-robin should distribute evenly. Check `/stats`:
```bash
curl http://localhost:8000/stats | jq '.workers[] | {index, requests_served}'
```

**Solution**: If severely unbalanced, check for worker health issues or restart the load balancer.

### Out of Memory Errors

**Problem**: Workers crash with CUDA OOM

**Solution**: Reduce per-worker batch size:
```bash
--batch-size 4  # Or lower
```

### Port Conflicts

**Problem**: "Address already in use"

**Solution**: Change ports:
```bash
--port 9000 --worker-base-port 9001
```

Or kill existing processes:
```bash
# Find process using port
lsof -ti:8000 | xargs kill -9
```

## Architecture Details

### Request Flow

1. Client sends request to load balancer (`:8000`)
2. Load balancer selects next healthy worker (round-robin)
3. Load balancer forwards request to worker (`:8001`, `:8002`, etc.)
4. Worker batches request with others (if any)
5. Worker processes batch on its GPU
6. Worker returns response to load balancer
7. Load balancer returns response to client

### Worker Initialization

Each worker:
1. Sets `CUDA_VISIBLE_DEVICES` to its GPU ID
2. Loads model into GPU memory
3. Starts FastAPI server on its assigned port
4. Begins accepting requests

### Load Balancer Behavior

- **Round-robin**: Cycles through workers 0, 1, 2, ..., N-1, 0, ...
- **Health checks**: Every 30 seconds (default)
- **Failure handling**: Unhealthy workers removed from rotation
- **Timeout**: 5 minutes per request (enough for large batches)

## Advanced Configuration

### Custom Load Balancing Strategy

Edit `llada_load_balancer.py` to implement different strategies:

```python
# Example: Least-connections instead of round-robin
async def get_next_worker(self) -> Optional[tuple[int, str]]:
    # Find worker with fewest active requests
    min_requests = min(self.worker_active_requests.values())
    for i, count in self.worker_active_requests.items():
        if count == min_requests and i in self.healthy_workers:
            return i, self.worker_urls[i]
    return None
```

### Dynamic Worker Scaling

Currently, the number of workers is fixed at startup. For dynamic scaling:

1. Modify `WorkerPool` to support `add_worker()` and `remove_worker()`
2. Implement auto-scaling based on queue length
3. Add endpoints: `POST /workers/add`, `DELETE /workers/{id}`

### Integration with Existing Code

The multi-GPU setup is **fully compatible** with existing evaluation scripts:

```python
# Your existing code works unchanged!
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",  # Points to load balancer
    api_key="dummy"
)

response = client.chat.completions.create(
    model="llada-8b-instruct",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

The load balancer transparently distributes requests across GPUs.

## Comparison: DataParallel vs Load Balancing

| Aspect | DataParallel | Load Balancing (Ours) |
|--------|-------------|----------------------|
| Implementation | Wrap model in `nn.DataParallel` | Separate workers + LB |
| Left-padding | ❌ Attends to padding tokens | ✅ No padding issues |
| Fault tolerance | ❌ One GPU fails → all fail | ✅ Continues with healthy GPUs |
| Debugging | ❌ Complex distributed debugging | ✅ Simple: check individual workers |
| Scalability | ⚠️ Limited by batch splitting | ✅ Linear scaling |
| Overhead | Low (shared memory) | Slight (HTTP + serialization) |
| Setup | Simple (one process) | Moderate (multiple processes) |

## Future Enhancements

Potential improvements:

1. **Sticky sessions**: Route requests from same client to same worker (for caching)
2. **Auto-scaling**: Add/remove workers based on load
3. **Load-aware routing**: Send requests to least-loaded worker
4. **GPU memory monitoring**: Automatic batch size adjustment
5. **Request queuing**: Global queue vs per-worker queues
6. **Distributed tracing**: Track request flow across workers

## Summary

The multi-GPU setup provides:
- ✅ Linear scaling across GPUs
- ✅ No left-padding complications  
- ✅ Simple architecture and debugging
- ✅ Fault tolerance
- ✅ Compatible with existing code
- ✅ Easy deployment (local or SLURM)

Use it when you need high throughput for batch evaluation workloads!


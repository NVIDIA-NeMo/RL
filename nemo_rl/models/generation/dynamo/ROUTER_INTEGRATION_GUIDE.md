# Dynamic Router Integration Guide

This guide explains how to use the dynamic KV-aware router with NeMo-RL's vLLM inference.

Built on top of **sechoi/dynamo_router** (commit b97785e), this integration provides three ways to use routing:

## Three Integration Modes

### 1. **SIMPLE MODE** (Recommended) - VllmGeneration with router_cfg
- **Best for**: Standard GRPO training workflows
- **Code changes**: Just add `router_cfg` to your config
- **Routing**: Completely automatic and transparent
- **Implementation**: sechoi's built-in integration in `VllmGeneration`

### 2. **MANUAL MODE** - Separate VllmGeneration + KvRouter  
- **Best for**: Custom routing strategies, explicit control
- **Code changes**: Manually coordinate routing and execution
- **Routing**: You control when and how to route
- **Implementation**: Separate `KvRouter` instance

### 3. **INTEGRATED MODE** - RoutedVllmWorkerGroup
- **Best for**: Advanced use cases, benchmarking
- **Code changes**: Use `RoutedVllmWorkerGroup` instead of `VllmGeneration`
- **Routing**: Clean API with batch routing support
- **Implementation**: `RoutedVllmWorkerGroup` class

**This guide focuses on SIMPLE MODE** (the easiest and most common use case).

## Overview

The dynamic router optimizes vLLM inference by intelligently routing requests to workers based on:
- **KV cache prefix overlap**: Routes similar requests to the same worker to maximize cache hits
- **Worker load**: Balances requests based on KV cache usage
- **Queue depth**: Avoids overloading busy workers

## Quick Start

### 1. Enable Router in GRPO Config

Add the router configuration to your GRPO YAML file:

```yaml
grpo:
  # ... other GRPO settings ...
  
  router:
    enabled: true
    block_size: 64
    base_kv_events_port: 5557
    base_metrics_port: 5657
```

### 2. Ensure vLLM Backend

The router only works with vLLM backend:

```yaml
policy:
  generation:
    backend: "vllm"  # Required
    colocated:
      enabled: false  # Non-colocated recommended
    vllm_cfg:
      async_engine: true  # Recommended for performance
```

### 3. Run Training

```bash
python examples/run_grpo.py --config your_config.yaml
```

## Configuration Options

### Router Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | false | Enable/disable router |
| `block_size` | int | 64 | KV cache block size (must match vLLM) |
| `base_kv_events_port` | int | 5557 | Base port for KV event publishers |
| `base_metrics_port` | int | 5657 | Base port for metrics publishers |

### Port Assignment

Workers use sequential ports:
- Worker 0: KV events on `base_kv_events_port + 0`, metrics on `base_metrics_port + 0`
- Worker N: KV events on `base_kv_events_port + N`, metrics on `base_metrics_port + N`

Ensure these ports are available on your system.

## Examples

### Example 1: Ray-Based Router Benchmark

Test router performance with Ray-based vLLM workers using the three integration modes:

**Simple Mode (default) - VllmGeneration with router_cfg:**
```bash
uv run --extra vllm --extra dynamo examples/run_router_benchmark_ray.py \
  --num-nodes 1 \
  --gpus-per-node 8 \
  --model "Qwen/Qwen2.5-Math-1.5B-Instruct" \
  --batch-size 64 \
  --seq-len 150 \
  --num-iterations 5
```

**Manual Mode - Separate VllmGeneration + KvRouter:**
```bash
uv run --extra vllm --extra dynamo examples/run_router_benchmark_ray.py \
  --num-nodes 1 \
  --gpus-per-node 8 \
  --use-manual-routing \
  --batch-size 64 \
  --num-iterations 5
```

**Integrated Mode - RoutedVllmWorkerGroup:**
```bash
uv run --extra vllm --extra dynamo examples/run_router_benchmark_ray.py \
  --num-nodes 1 \
  --gpus-per-node 8 \
  --use-integrated-routing \
  --batch-size 64 \
  --num-iterations 5
```

**Multi-node with 4 nodes × 8 GPUs = 32 workers:**
```bash
uv run --extra vllm --extra dynamo examples/run_router_benchmark_ray.py \
  --num-nodes 4 \
  --gpus-per-node 8 \
  --model "Qwen/Qwen2.5-Math-1.5B-Instruct" \
  --batch-size 128 \
  --num-iterations 10
```

This benchmarks:
- KV-aware routing vs round-robin (per mode)
- Router latency overhead
- Worker distribution across nodes
- End-to-end throughput
- Multi-node communication overhead
- Comparison between the three integration modes

### Example 2: GRPO with Router

See `examples/configs/grpo_with_router_example.yaml` for a complete configuration.

Key points:
```yaml
grpo:
  router:
    enabled: true
    block_size: 64

policy:
  generation:
    backend: "vllm"
    colocated:
      enabled: false  # Separate inference cluster
      resources:
        num_nodes: 1
        gpus_per_node: 4
    vllm_cfg:
      async_engine: true
```

Run:
```bash
python examples/run_grpo.py \
  --config examples/configs/grpo_with_router_example.yaml
```

### Example 3: Disable Router via CLI

Override config to disable router:

```bash
python examples/run_grpo.py \
  --config examples/configs/grpo_with_router_example.yaml \
  grpo.router.enabled=false
```

## Architecture

```
┌─────────────────────────────────────────────────┐
│                   GRPO Training                 │
│                                                 │
│  ┌───────────────────────────────────────────┐ │
│  │         VllmGeneration (Inference)        │ │
│  │  ┌─────────────────────────────────────┐  │ │
│  │  │          RayWorkerGroup             │  │ │
│  │  │   Worker0  Worker1  Worker2  ...    │  │ │
│  │  └──────┬──────────┬──────────┬─────────┘  │ │
│  └─────────┼──────────┼──────────┼────────────┘ │
│            │          │          │              │
│  ┌─────────┼──────────┼──────────┼────────────┐ │
│  │         │  KvRouter (Optional)│            │ │
│  │   ┌─────▼──────────▼──────────▼─────┐     │ │
│  │   │      RadixTree (Prefix Index)    │     │ │
│  │   └──────────────────────────────────┘     │ │
│  │   ┌──────────────────────────────────┐     │ │
│  │   │  Load Monitor (KV usage, queue)  │     │ │
│  │   └──────────────────────────────────┘     │ │
│  │                                             │ │
│  │   Routing: logit = 2*overlap - usage - wait│ │
│  └─────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

## How Router Works

### 1. Monitoring Phase

Router continuously monitors workers via ZMQ:
- **KV Events**: Tracks what prefixes are cached on each worker
- **Metrics**: Monitors KV cache usage (%) and queue depth

### 2. Routing Decision

When a request arrives:
1. Compute block hashes for the input tokens
2. Query RadixTree for prefix matches on each worker
3. Calculate routing score:
   ```
   score = 2 * overlap - kv_usage - normalized_waiting
   ```
4. Select worker with highest score

### 3. Generation

Request is dispatched to the selected worker for generation.

## Performance

### Expected Benefits

- **Cache hit rate**: 30-60% improvement for workloads with prefix overlap
- **Throughput**: 20-40% higher vs naive round-robin
- **Tail latency**: Reduced due to better load balancing

### Router Overhead

- **CPU**: ~1-2% per routing decision
- **Memory**: O(total_cached_blocks) for RadixTree
- **Latency**: ~1-5ms per routing decision

### When to Use

✅ **Use router when:**
- Multiple vLLM workers (2+)
- Requests share common prefixes (e.g., chat, few-shot prompts)
- Load balancing is important
- KV cache hits provide significant speedup

❌ **Skip router when:**
- Only 1 worker
- All requests are completely unique
- Routing overhead exceeds benefits (very short requests)
- Simple round-robin is sufficient

## Troubleshooting

### Router Not Starting

**Symptom**: `"Router is only supported with vLLM backend"`

**Solution**: Ensure `policy.generation.backend="vllm"` in config

---

**Symptom**: `"Failed to initialize router"`

**Solution**: 
- Check vLLM workers are properly configured
- Verify `num_workers` > 0
- Check ports are available

### Port Conflicts

**Symptom**: `OSError: [Errno 48] Address already in use`

**Solution**: Change base ports in config:
```yaml
grpo:
  router:
    base_kv_events_port: 6000  # Changed from 5557
    base_metrics_port: 6100    # Changed from 5657
```

Check port usage:
```bash
lsof -i :5557
lsof -i :5657
```

### Router Not Routing Effectively

**Symptom**: All requests go to same worker or distribution is uneven

**Possible Causes**:
1. **Block size mismatch**: Ensure `router.block_size` matches vLLM cache block size
2. **No shared prefixes**: Requests may not have common prefixes to benefit from routing
3. **Router needs time**: Give router 1-2 seconds to collect initial metrics

**Debug**:
```python
import logging
logging.getLogger("nemo_rl.models.generation.dynamo").setLevel(logging.DEBUG)
```

Look for log lines like:
```
DEBUG - worker_id: 0, logit = 2 * 0.200 - 0.300 - 0.100 = 0.000
DEBUG - worker_id: 1, logit = 2 * 0.800 - 0.200 - 0.050 = 1.350  <-- Selected
```

### Workers Not Publishing Metrics

**Symptom**: Router always selects worker 0 or uses round-robin

**Solution**:
- Ensure vLLM workers have metrics publishers configured
- Check `enable_kv_cache_events=true` in vLLM config
- Verify ZMQ sockets are bound:
  ```bash
  netstat -an | grep 5557  # KV events
  netstat -an | grep 5657  # Metrics
  ```

## API Reference

### RouterConfig

```python
class RouterConfig(TypedDict):
    """Configuration for KV-aware dynamic routing."""
    enabled: bool
    block_size: int
    base_kv_events_port: int
    base_metrics_port: int
```

### KvRouter

The router is automatically initialized in `grpo.setup()` when `router.enabled=true`.

Key methods:
- `start_background_tasks()`: Start monitoring workers
- `get_best_worker(hashes, num_tokens)`: Select optimal worker
- `get_worker_round_robin(hashes, num_tokens)`: Fallback round-robin
- `shutdown()`: Clean shutdown

### Integration Points

Router integrates at two points:

1. **Setup** (`grpo.setup()`):
   ```python
   router = KvRouter(
       block_size=config["block_size"],
       num_workers=policy_generation.worker_group.dp_size,
       base_kv_events_port=config["base_kv_events_port"],
       base_metrics_port=config["base_metrics_port"],
   )
   ```

2. **Training** (`grpo_train()` / `async_grpo_train()`):
   ```python
   # Router started automatically
   await router.start_background_tasks()
   
   # Router shutdown automatically on training completion
   await router.shutdown()
   ```

## Best Practices

1. **Use non-colocated inference**: Better isolation and routing control
2. **Enable async engine**: Better performance with router
3. **Multiple workers**: Router needs 2+ workers to be useful
4. **Monitor routing**: Check worker distribution in logs
5. **Tune block_size**: Larger blocks = fewer hashes but coarser matching
6. **Port management**: Avoid conflicts with other services
7. **Gradual rollout**: Test with A/B comparison first

## Related Files

- **Router Implementation**: `nemo_rl/models/generation/dynamo/standalone_router.py`
- **GRPO Integration**: `nemo_rl/algorithms/grpo.py`
- **Ray Benchmark**: `examples/run_router_benchmark_ray.py`
- **Example Config**: `examples/configs/grpo_with_router_example.yaml`
- **Original Benchmark**: `examples/run_router_benchmark.py` (standalone HTTP version)

## Contributing

When modifying the router:
1. Maintain backward compatibility (router should be optional)
2. Add comprehensive logging for debugging
3. Update this guide with new features
4. Add tests for new functionality
5. Benchmark performance impact

## License

Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
Licensed under Apache License 2.0.


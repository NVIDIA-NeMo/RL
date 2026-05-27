# FSDP2 Parallel Plan

This guide outlines the parallelization strategy for Fully Sharded Data Parallel version 2 (FSDP2) training in NeMo RL.

## Fallback Priority

NeMo RL supports three parallelization strategies, applied in the following order of fallback priority:

### 1. Custom Parallel Plan

Your user-defined custom parallel plans always take precedence when available. For detailed implementation and usage, refer to the [Custom Parallel Plan Example](#custom-parallel-plan-example).

### 2. Optimized Parallel Plan

Optimized parallel plans are available for specific model architectures. They may offer superior performance compared to Hugging Face's tensor parallel implementation. This approach is used if no custom parallel plan is specified and the model class supports optimized parallelization.

### 3. Hugging Face Tensor Parallel Plan

The Hugging Face tensor parallel plan is the default. It's available for most models via `._tp_plan` and is used when neither a custom nor an optimized parallel plan is available.

## Custom Parallel Plan Example

A custom parallel plan should be defined in a separate file, such as the example provided in `examples/custom_parallel/custom_parallel.py`.

To implement the custom parallel plan, either update the value of `custom_parallel_plan` in the `yaml` file directly, or pass the override via the command line. For example:

```bash
uv run examples/run_grpo.py \
    policy.dtensor_cfg.custom_parallel_plan=examples.custom_parallel.custom_parallel.custom_parallel_plan
```

## HSDP (`dp_replicate_size`)

By default, FSDP2 shards model parameters, gradients, and optimizer state across every data-parallel rank. On large multi-node jobs, the all-gathers and reduce-scatters that this requires cross the (slower) inter-node network on every step, which can become the bottleneck.

Hybrid Sharded Data Parallel (HSDP) splits the data-parallel axis into two dimensions:

- a **shard** group, where FSDP2 shards the model as usual, and
- a **replicate** group, which keeps a full copy of the sharded state and only exchanges gradients (like DDP).

The common pattern is to shard inside a node (fast NVLink) and replicate across nodes (slower network), so the heavy collectives stay on the fast links.

### Illustration

8 GPUs over 2 nodes, with `dp_replicate_size: 2` (one replica per node):

```
               shard (FSDP2, intra-node)
             ┌───────────────────────────┐
  replicate  │ node0:  g0  g1  g2  g3    │  <- one full sharded copy
  (DDP-like, │ node1:  g4  g5  g6  g7    │  <- another full sharded copy
  inter-node)└───────────────────────────┘
```

Each step: FSDP2 collectives run within a node; only a gradient all-reduce crosses between nodes.

### Configuration

```yaml
policy:
  dtensor_cfg:
    _v2: true             # required: HSDP only works on the DTensor v2 backend
    dp_replicate_size: 2  # number of replicas; 1 disables HSDP
```

The data-parallel size is `dp_size = world_size / (tp_size * cp_size * ep_size)`. `dp_replicate_size` must be a positive integer. Use `1` to disable HSDP. For HSDP, it must be greater than `1`, evenly divide `dp_size`, and be smaller than `dp_size`. The shard size is then `dp_size / dp_replicate_size`. A typical choice is `dp_replicate_size = num_nodes`.

### When to use it

- **`dp_replicate_size: 1` (default, pure FSDP2)** — single-node runs, or multi-node runs where the model fits comfortably and inter-node bandwidth is not the bottleneck. Lowest memory footprint per rank.
- **`dp_replicate_size > 1` (HSDP)** — multi-node runs where step time is dominated by inter-node FSDP collectives. Trades memory (one full sharded copy per replica) for faster steps.

See [examples/configs/recipes/llm/sft-llama3.2-1b-2n8g-hsdp.yaml](../../examples/configs/recipes/llm/sft-llama3.2-1b-2n8g-hsdp.yaml) for a working recipe.

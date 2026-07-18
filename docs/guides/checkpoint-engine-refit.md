# Checkpoint-Engine Refit

Checkpoint-engine refit updates non-colocated generation workers directly from
policy workers. The built-in backend is NIXL, which can use UCX/RDMA for large
policy-to-vLLM refits.

Use it only for non-colocated vLLM generation:

- `policy.generation.backend=vllm`
- `policy.generation.colocated.enabled=false`
- `policy.generation.checkpoint_engine.enabled=true`

Colocated generation still uses IPC/HTTP refit. Non-colocated generation without
checkpoint-engine refit still uses the NCCL collective update path.

`examples/configs/grpo_math_8B_megatron_nixl.yaml` is a complete two-node
example built as an overlay on the standard 8B Megatron recipe.

## Enable NIXL

Add `checkpoint_engine` under `policy.generation`:

```yaml
policy:
  generation:
    backend: vllm
    colocated:
      enabled: false
      resources:
        num_nodes: 1
        gpus_per_node: 8
    checkpoint_engine:
      enabled: true
      backend: nixl
      update_weights_bucket_memory_ratio: 0.05
      engine_kwargs:
        nixl:
          device: cuda
          release_after_refit: false
          backend_name: UCX
          backend_init_params:
            engine_config: MAX_RMA_RAILS=8
            device_list: "mlx5_0,mlx5_1,mlx5_2,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8"
```

Key settings:

| Key | Meaning |
|---|---|
| `update_weights_bucket_memory_ratio` | Fraction of fixed total GPU memory used by each transfer buffer. Defaults to `0.05`. NIXL allocates two buffers, so the default reserves 10% in total. |
| `device` | `cuda` uses GPU RDMA buffers. `cpu` uses host-pinned buffers and is mainly a fallback. |
| `release_after_refit` | When `true`, deregister and free transfer buffers after every refit. The next refit allocates and registers them again. Default `false` retains them for throughput. |
| `shard_expert_weights` | Sends destination-local MoE expert shards for TP/EP and omits weights absent from each vLLM PP stage. |
| `backend_init_params` | NIXL backend parameters such as UCX peer error handling, device lists, and UCX engine config. |

The built-in NIXL topology is paired policy-to-rollout transfer, so allocate at
least as many policy workers as rollout workers.

The driver queries fixed total GPU memory from every policy and vLLM worker
before creating the engines and uses the smallest capacity. It multiplies that
capacity by `update_weights_bucket_memory_ratio` and rounds down to a MiB. For
example, the default `0.05` selects a 4 GiB bucket on an 80 GiB GPU. Because
NIXL uses two buffers, that configuration reserves about 8 GiB per participating
engine. The selected size is fixed for the synchronizer lifetime.

## Runtime Setup

NIXL must be importable in every participating environment:

- policy worker environment
- vLLM worker environment, including async vLLM workers

Keep checkpoint-engine feature selection in YAML or Hydra overrides. Use
environment variables for UCX/NIXL transport selection:

```sh
export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1
export UCX_TLS=rc,cuda_copy,cuda_ipc,self,sm
export UCX_IB_ROCE_REACHABILITY_MODE=all
export UCX_MEMTYPE_CACHE=n
export UCX_MAX_RNDV_RAILS=8
export UCX_WARN_UNUSED_ENV_VARS=n
export NIXL_LOG_LEVEL=INFO
```

When vLLM starts nested Ray workers, copy the transport variables into those
workers:

```sh
export VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY=MELLANOX_
export VLLM_RAY_EXTRA_ENV_VARS_TO_COPY=LD_LIBRARY_PATH,NIXL_LOG_LEVEL,NVIDIA_VISIBLE_DEVICES,UCX_NET_DEVICES,UCX_TLS,UCX_IB_ROCE_REACHABILITY_MODE,UCX_MEMTYPE_CACHE,UCX_MAX_RNDV_RAILS,UCX_WARN_UNUSED_ENV_VARS
```

Use NIC names that exist on the target nodes. With `UCX_LOG_LEVEL=info`, UCX
should report an RDMA transport such as `rc_mlx5`; TCP-only transport will be
much slower.

## Performance Notes

Use `device: cuda` for the fast path when NIXL/UCX can register CUDA memory.
For large MoE models with vLLM, leave enough GPU memory for NIXL buffers. The
following destination layout was validated with DeepSeek-V3 BF16:

```yaml
policy:
  generation:
    vllm_cfg:
      tensor_parallel_size: 32
      pipeline_parallel_size: 1
      expert_parallel_size: 1
      async_engine: true
      gpu_memory_utilization: 0.82
    vllm_kwargs:
      moe_backend: triton
    checkpoint_engine:
      update_weights_bucket_memory_ratio: 0.05
      engine_kwargs:
        nixl:
          device: cuda
          release_after_refit: false
          shard_expert_weights: true
```

With `shard_expert_weights`, each vLLM worker reports its live destination layout
during checkpoint-engine setup. For pipeline parallelism, the source omits
every weight that vLLM marks as absent from the destination stage. For
tensor-parallel MoE, the source sends that TP rank's slice of every expert. For
expert-parallel MoE, the source sends complete experts only to ranks that own
them, followed by any intra-expert TP slice reported by vLLM. Any static
ownership map reported by vLLM is supported. The reported destination layout is
authoritative; do not configure a separate target-TP or rank-derived sharding
hint.

Set `shard_expert_weights: false` to use the reference full-weight loader. This is
the fallback for a new vLLM layout or loader contract; fallback cannot happen
after transfer starts because the sender has already discarded non-local
shards. Sharded setup therefore validates vLLM's canonical parameter layout
and source shard shapes and fails before loading when those contracts do not
match.

Dynamic expert load balancing and redundant experts are not supported because
ownership can change after metadata exchange. The direct-copy load path accepts
only unquantized Triton expert storage; FP8 and backends that transpose or
shuffle expert weights are rejected during setup. This includes FlashInfer
TRT-LLM MXFP8 layouts that reorder W13 to W31 and shuffle weights and scales.

### FP8 Compatibility

FP8 model weights and FP8 KV-cache scales are separate features:

| Configuration | Status |
|---|---|
| NIXL with `shard_expert_weights: false` and FP8 vLLM weights | Supported through the existing full-weight `fp8.load_weights` path. |
| NIXL with `shard_expert_weights: true` and FP8 or MXFP8 vLLM weights | Unsupported. Setup or loading fails explicitly because destination-local expert loading does not yet implement quantized, transposed, or shuffled weight-and-scale layouts. |
| Megatron policy with FP8 KV-cache scales | Supported; the scales are appended to the checkpoint-engine weight stream and processed after loading. |
| DTensor or DTensor v2 policy with FP8 KV-cache scales | Unsupported; the policy worker raises `NotImplementedError`. |

Sharded FP8 expert refit is a loader-layout limitation, not a NIXL transport
limitation. Supporting it requires a versioned destination-layout adapter for
the quantized weights, scales, and any vLLM post-load transpose or shuffle.

A full-weight FP8 end-to-end validation on 2026-07-17 used a BF16 Megatron
Qwen2.5-0.5B policy, an FP8 vLLM rollout model, two eight-GPU nodes, asynchronous
rollout, CUDA-buffer NIXL, and a 256 MiB bucket. Two refits completed in 1.116 s
and 0.195 s. `tools/refit_verifier.py` reported mean/max absolute logprob
differences of 0.09474/0.18032 and an average probability multiplier of 1.10153.
These differences include BF16-to-FP8 quantization error, so this validates the
full-weight FP8 control and transfer path rather than bitwise transport parity.

Only destination-local expert tensors use this direct-copy path. Dense and
unhandled tensors continue through vLLM's standard `load_weights` path. CUDA
NIXL still stages transfers through GPU RDMA buffers, but tensors spanning
multiple buffers are reassembled on CPU before that standard loader to limit
peak GPU memory.

When vLLM EP is larger than TP, NeMo RL currently requires `async_engine: false`.
For non-MoE models, `shard_expert_weights` only applies pipeline-stage filtering;
dense tensors are not sharded.
Expert sharding requires HF expert names that map to vLLM `w13_weight` and
`w2_weight` parameters.

Avoid `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for CUDA-buffer NIXL
refit unless that workload has been explicitly validated with it.

Set `release_after_refit: true` when the two registered buffers must be returned
between refits. Finalization then deregisters both buffers, releases their
dedicated CuPy pool, and retains the NIXL agent and control endpoint. Before the
next refit, `prepare()` allocates and registers two new buffers. This saves
twice the resolved bucket size of resident GPU memory between refits at the cost
of allocation, registration, and metadata exchange on every refit. Reallocation
uses the size selected when the synchronizer was initialized. Leave it `false`
for the benchmark path below.

### DeepSeek-V3 Benchmark

The full-model benchmark used 36 eight-GPU nodes: 32 policy nodes
with Megatron TP1/PP16/EP16 and four rollout nodes with vLLM TP32/PP1/EP1. NIXL
used CUDA buffers, UCX over eight RDMA rails, a 4096 MiB bucket, Triton MoE, and
`shard_expert_weights: true`. Each vLLM rank loaded 45,395 destination-local tensors
in 18 batches, totaling 69.95 GiB.

| Measurement | NIXL GPU RDMA | NCCL | NIXL improvement |
|---|---:|---:|---:|
| Dedicated async refit timer | 9.25 s | 14.21 s | 1.54x, 35% less time |
| Matched synchronous verifier | 10.92 s | 15.52 s | 1.42x, 30% less time |

Compare results within one row only; async and synchronous vLLM have different
control-path overhead.

A same-build synchronous regression run on 2026-07-15 changed only
`shard_expert_weights`: the full-expert path took 36.812 s and the sharded-expert
path took 10.923 s, a 3.37x speedup and 70% less time. The verifier's recorded
output IDs, vLLM logprobs, policy logprobs, per-row errors, and aggregate metrics
were identical between paths. Both paths reported mean/max policy-to-vLLM
logprob differences of 0.03227/0.09738; this comparison establishes parity with
the full-weight loader, not absolute equivalence between the two model stacks.

## Fault Tolerance Boundary

NIXL is the transfer layer. It does not add, remove, or replace Ray/vLLM
rollout actors, and it does not route rollout requests.

What NIXL provides:

- UCX peer error handling can turn a lost peer into a transport error instead
  of an indefinite wait.
- The NIXL backend raises when a read cannot start or completes with `ERR`.
- `CheckpointEngineWeightSynchronizer` propagates a failed update as a failed
  refit.
- Reinitializing the synchronizer exchanges fresh NIXL metadata for the current
  policy and rollout actor set.

Enable UCX peer error handling when failed peers should surface promptly:

```yaml
policy:
  generation:
    checkpoint_engine:
      engine_kwargs:
        nixl:
          backend_init_params:
            ucx_error_handling_mode: peer
```

Changing the rollout actor set is orchestration outside NIXL: stop routing to
the old actor, create or remove the Ray/vLLM actor, shut down and reinitialize
the checkpoint-engine communicator, then run a full refit before routing prompts
to the new set.

`tools/nixl_elastic_rollout_demo.py` exercises that communicator teardown and
reinitialization sequence with synthetic weights. It demonstrates the NIXL
lifecycle boundary, not automatic Ray actor recovery.

## Verify

The driver log should show:

```text
Using checkpoint-engine refit backend: nixl
```

Each vLLM update should also print:

```text
[vLLM refit] Loaded ... via checkpoint engine; bytes=... total=... receive=... load=...
```

Use `tools/refit_verifier.py` for a refit correctness smoke test:

```sh
uv run --extra mcore --extra vllm python tools/refit_verifier.py \
  --model_name /path/to/model \
  --tp_size 1 \
  --ep_size 1 \
  --pp_size 1
```

That verifier compares vLLM and Megatron logprobs after a refit. It is useful
for model/refit correctness, while the NIXL transport path is confirmed by the
non-colocated GRPO log markers above.

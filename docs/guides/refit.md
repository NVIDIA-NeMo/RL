# Weight Refit: Choosing a Transport

Weight refit copies updated policy weights into the rollout model. Choose the
topology first, then select one non-colocated transport with
`policy.generation.refit_transport`.

## Pick a Transport

| Topology | `refit_transport` | Transport | Use when |
|---|---|---|---|
| `colocated.enabled: true` | `null` | CUDA IPC or HTTP | Policy and rollout workers share GPUs; vLLM uses IPC and SGLang uses HTTP. |
| `colocated.enabled: false` | `null` | NCCL broadcast | You want the default full-weight path without extra dependencies. |
| `colocated.enabled: false` | `nccl_reshard` | NCCL reshard | Provides the best performance for large models (>100s B models). |
| `colocated.enabled: false` | `vllm_zmq_sparse` | Sparse delta over ZeroMQ | The link is bandwidth-limited and workers can reach a relay over TCP. |
| `colocated.enabled: false` | `vllm_s3_sparse` | Sparse delta through S3 | Workers communicate through shared object storage. |
| `colocated.enabled: false` | `nixl` | NIXL checkpoint engine | The cluster has a fast UCX/RDMA fabric for full-weight refit. |

`null` is the default. The sparse transports read only `refit_cfg.sparse`; NIXL
reads only `refit_cfg.nixl`. Because one selector chooses the transport, sparse
delta and NIXL cannot both be active.

## Constraints

| Transport | Generation backend | Policy backend | Quantization and MoE |
|---|---|---|---|
| Colocated IPC/HTTP | vLLM or SGLang | DTensor or Megatron | Uses the generation backend's standard loader. |
| NCCL | vLLM or Megatron | DTensor or Megatron | Uses the standard full-weight loader. |
| NCCL reshard | vLLM | Megatron | Requires matching BF16 or blockwise FP8 precision; Megatron ETP must be 1. Currently supporting Megatron+vLLM backends. |
| Sparse delta | vLLM | Megatron | BF16/FP16, unquantized rollout only. |
| NIXL, full weights | vLLM | DTensor or Megatron | Supports the standard full-weight FP8 loader. DTensor FP8 KV-cache scale transfer is not yet supported. |
| NIXL, sharded experts | vLLM | DTensor or Megatron | Unquantized BF16/FP16 Triton MoE only; FP8/MXFP8 and dynamic expert placement are rejected. |

Non-colocated SGLang generation is not supported. The NIXL restrictions are on
the generation backend; both Megatron and DTensor policy workers can send
weights. Sparse delta is currently limited to GRPO. NIXL is initialized by the
GRPO and distillation setup paths; PPO currently requires colocated generation.

## Performance Options

| Option | Scope | Effect |
|---|---|---|
| `policy.generation.vllm_cfg.refit_prequantize` | Megatron training with MXFP8 vLLM rollout | Quantizes eligible weights on the trainer and transfers E4M3 values plus E8M0 scales. Requires `precision: fp8` and `is_mx: true`; sparse delta and NCCL Reshard do not support it. |
| `policy.generation.vllm_cfg.refit_cache_loader_routes` | vLLM refit | Replays identity-validated weight-loader routes after the first refit. Disabled by default because loader behavior is model-dependent. |
| `policy.refit_buffer_size_gb` | Colocated IPC/HTTP or non-colocated NCCL broadcast | Sets the packing threshold explicitly. For NCCL broadcast, the same byte value is sent to producer and consumer so their collective chunk boundaries match. |
| `policy.refit_persistent_ipc_buffers` | Colocated CUDA-IPC refit | Reuses the two trainer staging buffers across refits. A fixed `refit_buffer_size_gb` gives stable memory use. |
| `policy.megatron_cfg.refit_slim_offload_after` | Colocated Megatron refit | Avoids repeating grad-buffer offload and a second allocator cleanup after weights are transferred. |
| `policy.megatron_cfg.pinned_reference_swap` | Megatron reference-policy logprobs | Keeps the CPU reference copy in pinned memory for faster host-to-device swaps, at the cost of additional pinned host memory. |

## Minimal Configuration

Colocated refit needs no transport configuration:

```yaml
policy:
  generation:
    colocated:
      enabled: true
    refit_transport: null
```

For non-colocated NCCL, change the topology and leave the selector unset:

```yaml
policy:
  generation:
    colocated:
      enabled: false
    refit_transport: null
```

For NCCL reshard with Megatron policy training and vLLM generation:

```yaml
policy:
  generation:
    colocated:
      enabled: false
    refit_transport: nccl_reshard
```

For sparse delta, select one data plane and configure its scope:

```yaml
policy:
  generation:
    colocated:
      enabled: false
    refit_transport: vllm_zmq_sparse  # or vllm_s3_sparse
    refit_cfg:
      sparse:
        delta_compression:
          encoding: xor
        storage:
          s3_bucket: null  # required for vllm_s3_sparse
```

For NIXL, select the checkpoint engine and configure its scope:

```yaml
policy:
  generation:
    colocated:
      enabled: false
    refit_transport: nixl
    refit_cfg:
      nixl:
        update_weights_bucket_memory_ratio: 0.05
        device: cuda
        backend_name: UCX
        release_after_refit: false
        shard_expert_weights: false
```

## Learn More

- [NCCL Reshard Refit](../design-docs/nccl-reshard-refit.md) describes its
  requirements, architecture, and shard-to-shard transfer.
- [Sparse Delta Refit](../design-docs/sparse-delta-refit.md) explains baseline,
  compression, ZeroMQ, and S3 behavior.
- [Checkpoint-Engine Refit](checkpoint-engine-refit.md) covers NIXL setup,
  performance tuning, FP8, sharded experts, and fault tolerance.
- [Checkpoint Engines](../design-docs/checkpoint-engines.md) describes the
  checkpoint-engine protocol and implementation.
- [Training and Generation Backends](../about/backends.md) summarizes backend
  compatibility.

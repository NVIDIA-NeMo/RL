# Weight Refit: Choosing a Transport

Weight refit copies updated policy weights into the rollout model. Choose the
topology first, then select one non-colocated transport with
`policy.generation.refit_transport`.

## Pick a Transport

| Topology | `refit_transport` | Transport | Use when |
|---|---|---|---|
| `colocated.enabled: true` | `null` | CUDA IPC or HTTP | Policy and rollout workers share GPUs; vLLM uses IPC and SGLang uses HTTP. |
| `colocated.enabled: false` | `null` | NCCL broadcast | You want the default full-weight path without extra dependencies. |
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
| NCCL | vLLM, Megatron, or SGLang | DTensor or Megatron | SGLang non-colocated GRPO broadcast currently requires Megatron; the other backends use their standard full-weight loader. |
| Sparse delta | vLLM | Megatron | BF16/FP16, unquantized rollout only. |
| NIXL, full weights | vLLM | DTensor or Megatron | Supports the standard full-weight FP8 loader. DTensor FP8 KV-cache scale transfer is not yet supported. |
| NIXL, sharded experts | vLLM | DTensor or Megatron | Unquantized BF16/FP16 Triton MoE only; FP8/MXFP8 and dynamic expert placement are rejected. |

Non-colocated SGLang generation uses a GRPO-specific side communicator and
requires the Megatron policy backend. The NIXL restrictions are on the
generation backend; both Megatron and DTensor policy workers can send weights
to vLLM. Sparse delta is currently limited to GRPO. NIXL is initialized by the
GRPO and distillation setup paths; PPO currently requires colocated generation.

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

- [Sparse Delta Refit](../design-docs/sparse-delta-refit.md) explains baseline,
  compression, ZeroMQ, and S3 behavior.
- [Checkpoint-Engine Refit](checkpoint-engine-refit.md) covers NIXL setup,
  performance tuning, FP8, sharded experts, and fault tolerance.
- [Checkpoint Engines](../design-docs/checkpoint-engines.md) describes the
  checkpoint-engine protocol and implementation.
- [SGLang Generation and Weight Refit](sglang-refit.md) covers SGLang topology,
  refit validation, NeMo-Gym, and failure markers.
- [Training and Generation Backends](../about/backends.md) summarizes backend
  compatibility.

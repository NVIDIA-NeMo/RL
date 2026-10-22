# Weight Refit: Choosing a Transport

Weight refit copies updated policy weights into the rollout model. Choose the
topology first, then select one transport with
`policy.generation.refit_transport`.

## Pick a Transport

| Topology | `refit_transport` | Transport | Use when |
|---|---|---|---|
| `colocated.enabled: true` | `null` | CUDA IPC | Policy and rollout workers share GPUs; vLLM uses ZMQ plus CUDA IPC handles and SGLang uses Ray CUDA IPC. |
| `colocated.enabled: true` | `mcore` | MCore wake-reshard | Megatron policy and generation share GPUs. |
| `colocated.enabled: false` | `null` | NCCL broadcast | vLLM and Megatron use the default full-weight collective; SGLang uses its own weight-update group. |
| `colocated.enabled: false` | `mcore` | MCore native refit | Megatron policy sends directly to Megatron generation through an MCore copy service. |
| `colocated.enabled: false` | `nccl_reshard` | NCCL reshard | Provides the best performance for large models (>100s B models). |
| `colocated.enabled: false` | `vllm_zmq_sparse` | Sparse delta over ZeroMQ | The link is bandwidth-limited and workers can reach a relay over TCP. |
| `colocated.enabled: false` | `vllm_s3_sparse` | Sparse delta through S3 | Workers communicate through shared object storage. |
| `colocated.enabled: false` | `nixl` | NIXL checkpoint engine | The cluster has a fast UCX/RDMA fabric for full-weight refit. |

`null` is the default. For non-colocated Megatron generation it selects the
packed collective; Megatron's native refit must be selected explicitly with
`mcore`. The sparse transports read only `refit_cfg.sparse`; NIXL reads only
`refit_cfg.nixl`. Because one selector chooses the transport, sparse delta and
NIXL cannot both be active.

## vLLM Reload API

For non-colocated vLLM using the default NCCL transport, set
`policy.generation.vllm_cfg.refit_with_reload_api: true` to install refitted
weights through vLLM's native `reload_weights` API. The flag is off by default.
Use this path when vLLM's post-load processing is required after every refit,
for example `flashinfer_trtllm` MoE models where `process_weights_after_loading`
reshapes expert weight buffers.

Early measurements from this PR show that reload refit can be faster for Llama
8B and Qwen3 32B, while Qwen3 30B-A3B was about 44% slower. MXFP8 reload refit
was about 2x slower in the measured run and used higher peak memory. Leave
additional KV-cache headroom for MXFP8, for example by lowering
`policy.generation.vllm_cfg.gpu_memory_utilization`.

The reload API path currently supports only `policy.generation.backend: vllm`,
`policy.generation.colocated.enabled: false`, and
`policy.generation.refit_transport: null`. It is explicitly unsupported with
`nccl_reshard`, ModelOpt quantization (`policy.generation.quant_cfg`), sparse
refit transports, and checkpoint-engine/NIXL transports. Eagle/MTP draft
weights refitted from the trainer are not supported yet; use the legacy loader
for those cases.

## External vLLM NCCL Refit

SingleController/NeMo-Gym can use `policy.generation.backend: remote_vllm`
with a separately launched vLLM **0.29.0** server. Launch that server with
`VLLM_SERVER_DEV_MODE=1` and `--weight-transfer-config '{"backend":"nccl"}'`.
Keep its development endpoints on a private network reachable from the
training controller. The trainer and serving GPU workers also need mutual
network access and compatible NCCL transport/plugin settings.

```yaml
policy:
  generation:
    backend: remote_vllm
    colocated:
      enabled: false
    refit_transport: null
    remote_vllm_cfg:
      base_url: http://serving-host:8000/v1
      served_model_name: policy
      max_model_len: 8192
      refit:
        transport: nccl
        nccl_num_engines: 1
        nccl_engine_world_size: 2  # TP * PP * DP within this deployment
```

This path requires a Megatron policy and Gym rollouts. Each refit pauses
serving, starts a native weight update, broadcasts HF-named tensors directly
over NCCL, finishes the update with a weight version, and resumes serving.
HTTP carries control and metadata; no HF checkpoint is written or loaded from
disk. Native vLLM layerwise loading still runs. `quant_cfg`, trainer-refitted
draft weights, and separate KV-scale transfer are not supported here.

Only **one independent serving deployment** is supported. Its TP/PP/DP GPU
workers can span multiple devices; the router and HTTP frontend are not NCCL
participants. `nccl_num_engines` accepts only `1`, and service preflight rejects
multiple backends even when a router advertises control fan-out. P/D
separation and independent-replica fan-out are future work. Setup also checks
the serving version, NCCL transfer backend, and declared GPU world size.

The refit configuration defines `timeout_s` (1800 seconds),
`nccl_packed_buffer_size_bytes` (1 GiB), and `nccl_packed_num_buffers` (2).
Packing geometry is negotiated at initialization. The policy broadcast uses
vLLM's packing convention, including tensors larger than a packed buffer.
A configured fleet-health refit deadline can shorten the collective deadline.

A failed update leaves generation paused and prevents reuse of the failed
synchronizer. Restart both training and serving before retrying. Controller
shutdown releases its HTTP executor; NCCL communicators belong to the GPU
processes and are reclaimed when those processes exit. NeMo-RL does not
terminate the externally managed service. Do not share it with another
training job while a communicator is active.

The functional script
`tests/functional/grpo_external_vllm_gym_single_controller.sh` launches one
server, runs at least ten training steps, and checks nonzero policy gradients
and recorded NCCL-refit timings. For an existing server, provide
`EXTERNAL_VLLM_TEST_BASE_URL` and set `EXTERNAL_VLLM_TEST_ENGINE_WORLD_SIZE` to
its TP * PP * DP size. These checks are functional smoke coverage, not a
convergence or throughput benchmark.

## Constraints

| Transport | Generation backend | Policy backend | Quantization and MoE |
|---|---|---|---|
| Colocated CUDA IPC | vLLM or SGLang | Automodel or Megatron | Uses the generation backend's standard loader. |
| NCCL | vLLM or Megatron | Automodel or Megatron | Uses the standard full-weight loader. |
| NCCL + reload API | vLLM | Automodel or Megatron | Default non-colocated NCCL only. Rejects colocated refit, `nccl_reshard`, sparse delta, NIXL/checkpoint-engine transports, ModelOpt quantization (`real_quant` or `quant_cfg`), and Eagle/MTP trainer-refit draft weights. |
| MCore native refit | Megatron | Megatron | Supports MCore's configured copy-service backend. |
| SGLang NCCL weight-update group | SGLang | Megatron | Trainer rank 0 broadcasts finalized HF tensors to the engine leaders. |
| NCCL reshard | vLLM or Megatron | Megatron | Supports BF16 and the documented FP8/MXFP8 combinations; see the [NCCL Reshard Refit design](../design-docs/nccl-reshard-refit.md). |
| Sparse delta | vLLM | Megatron | BF16/FP16, unquantized rollout only. |
| NIXL, full weights | vLLM | Automodel or Megatron | Supports the standard full-weight FP8 loader. Automodel FP8 KV-cache scale transfer is not yet supported. |
| NIXL, sharded experts | vLLM | Automodel or Megatron | Unquantized BF16/FP16 Triton MoE only; FP8/MXFP8 and dynamic expert placement are rejected. |

Non-colocated SGLang generation is supported with a Megatron policy and
`refit_transport: null`; it creates its own NCCL weight-update group. The NIXL
restrictions are on the generation backend; both Megatron and Automodel policy
workers can send weights. Sparse delta is currently limited to GRPO. NIXL is
initialized by the GRPO and distillation setup paths; PPO currently requires
colocated generation.

## Refit Performance Options

These options are optional. Enable one only after you measure its effect on your
refit workload.

| Option | Applies to | Effect |
|---|---|---|
| `policy.generation.vllm_cfg.refit_prequantize` | Megatron policy with MXFP8 vLLM rollout | Quantizes supported weights before transfer. See [Optional trainer-side MXFP8 refit](../fp8.md#optional-trainer-side-mxfp8-refit) for requirements and exclusions. |
| `policy.refit_buffer_size_gb` | Refit paths that stage weights in buffers | Sets a fixed buffer size in GiB. NCCL broadcast uses the same size on both sides to keep chunk boundaries equal. |

### TensorRT-LLM Routed-Expert FP8

TensorRT-LLM supports online block-FP8 refit for MoE rollouts. Policy
training remains in BF16, and this setting does not change optimizer-state
precision. During each refit, NeMo RL converts weights to E4M3 with 128x128
blocks and FP32 scales. Only the routed-expert weights are converted;
attention, routers, shared experts, embeddings, and the output head remain in
BF16.

This path starts from a BF16 policy checkpoint. Direct loading of a
pre-quantized ModelOpt FP8 checkpoint is outside its scope.

Enable the path with:

```yaml
policy:
  generation:
    trtllm_cfg:
      precision: fp8
```

Supported models are Qwen3.5 MoE (`qwen3_5_moe`) and Qwen3 MoE (`qwen3_moe`);
setup fails for any other model type. The path quantizes routed-expert weights,
either fused (`mlp.experts.gate_up_proj` / `down_proj`, the Qwen3.5 layout) or
per-expert (`mlp.experts.{i}.{gate,up,down}_proj.weight`, the Qwen3 MoE layout).
Setup also fails if no routed-expert weight is found, or if the model has dense
MLP weights that would stay BF16. Supporting another model means extending the
routed-expert name patterns in `quantization/fp8.py`.

Verified end to end: Qwen3.5 MoE with MXFP8 (`Qwen/Qwen3.5-35B-A3B-Base` and
`Qwen/Qwen3.5-397B-A17B`).

NeMo RL initializes the TensorRT-LLM model with `load_format` set
to `dummy`, then populates it from the first BF16 policy refit. By default
(`is_mx: false`), the TensorRT-LLM MoE backend (`moe_config.backend: TRTLLM`)
is required to preserve FP32 block scales; MXFP8/E8M0 scales are not used.

#### MXFP8 variant

Set `is_mx: true` to quantize routed experts to MXFP8 (E4M3 weights with
UE8M0 1x32 block scales) instead of 128x128 block-FP8 with FP32 scales:

```yaml
policy:
  generation:
    trtllm_cfg:
      precision: fp8
      is_mx: true
```

This selects the CUTLASS MoE backend (the default) or the CuTe DSL backend
(`moe_config.backend: CUTEDSL`, for Rubin) instead of the `TRTLLM` backend that
block-FP8 uses; any other backend fails at setup. MXFP8 also constrains which
GPUs it can run on: TensorRT-LLM's
CUTLASS MoE gates `QuantAlgo.MXFP8` on `sm_constraint in {100, 103}`, so it does
not run everywhere the default block-FP8 path does.

The installed TensorRT-LLM must provide the incremental-refit lifecycle APIs
`begin_update_weights`, `finalize_update_weights`, `abort_update_weights`, and
`WorkerExtension.finalize_weight_update`. NeMo RL fails during setup if any of
the three model-loader hooks is missing; `finalize_weight_update` is called
directly and is not checked up front.

## Minimal Configuration

Colocated vLLM and SGLang refit need no transport configuration:

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
  offload_policy_before_refit: false
  generation:
    colocated:
      enabled: false
    refit_transport: null
```

Large quantized exports can temporarily need more memory than training itself.
Set `offload_policy_before_refit: true` to drop completed gradient buffers before
the collective export. The same lifecycle can also move the optimizer and clear
Transformer Engine workspaces:

```yaml
policy:
  offload_policy_before_refit: true
  offload_optimizer_for_refit: true
  megatron_cfg:
    fp8_cfg:
      enabled: true
      force_clear_fp8_caches: true
  generation:
    colocated:
      enabled: false
    refit_transport: null
```

This option requires the Megatron policy backend. It applies to non-colocated
vLLM collective and `nccl_reshard` transports, non-colocated TensorRT-LLM
collective refit, plus non-colocated Megatron generation. Unsupported
combinations fail during synchronizer setup. It is
disabled by default because CPU offload adds transfer overhead when the export
already fits in trainer GPU memory.

For native MCore refit, select it explicitly:

```yaml
policy:
  generation:
    backend: megatron
    refit_transport: mcore
    mcore_generation_config:
      refit_backend: nccl  # gloo | nccl | nccl_m2n (nvshmem is broken; see #3646)
```

For NCCL reshard with Megatron policy training and vLLM generation:

```yaml
policy:
  generation:
    colocated:
      enabled: false
    refit_transport: nccl_reshard
```

`offload_policy_before_refit` also works with this transport. The reshard moves
only parameters, so releasing gradient buffers, optimizer state, and caches
before transfer is safe.

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

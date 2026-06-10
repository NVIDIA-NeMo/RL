# Checkpoint-Engine Refit

Checkpoint-engine refit updates non-colocated generation workers from policy
weights through a pluggable transfer backend. The first built-in backend is
NIXL, which can use UCX/RDMA for large policy-to-vLLM refits.

Use this path when generation runs on dedicated resources:

- `policy.generation.colocated.enabled=false`
- `policy.generation.backend=vllm`
- `policy.generation.checkpoint_engine.enabled=true`

For colocated generation, NeMo RL continues to use the colocated IPC refit path.
For non-colocated generation without checkpoint-engine refit, NeMo RL uses the
collective update path.

## Enable NIXL Refit

Add a `checkpoint_engine` block under `policy.generation`:

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
      update_weights_bucket_megabytes: 2048
      engine_kwargs:
        nixl:
          device: cuda
          cleanup_after_load: false
          topology: auto
          transfer_mode: staged
          buffer_count: 2
          background_progress: false
          load_batch_bucket_count: 1
          direct_min_bytes: 2147483648
          direct_stripe_count: 1
          metadata_batch_size: 1
          backend_name: UCX
          backend_init_params:
            ucx_error_handling_mode: none
            engine_config: MAX_RMA_RAILS=8
            device_list: "mlx5_0,mlx5_1,mlx5_2,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8"
```

`backend` selects the checkpoint-engine transfer backend. Built-in backends use
short names such as `nixl`. External backends can use a class path; see
[Checkpoint Engine Design](../design-docs/checkpoint-engines.md).

`update_weights_bucket_megabytes` controls the transfer-buffer size. Larger
buckets reduce per-bucket overhead, but reserve more memory on every
participating worker. `2048` MiB is a good starting point for large models.

`engine_kwargs.<backend>` is passed to the backend constructor. For NIXL:

| Key | Meaning |
|---|---|
| `device` | Transfer-buffer device. Use `cuda` for the fast GPU-RDMA path when the NIXL/UCX runtime supports CUDA memory registration. Use `cpu` only as a host-pinned fallback when CUDA buffers are unavailable. |
| `cleanup_after_load` | Whether vLLM should run garbage collection and `torch.cuda.empty_cache()` after loading each refit. Disabling this avoids extra steady-state overhead when memory is stable. |
| `topology` | Transfer topology. `auto` keeps the paired topology when policy workers cover rollout workers and falls back to `leader_tree` otherwise. Use `leader_chain` to force the older single chain for comparison. |
| `transfer_mode` | Source-side transfer mode. `staged` copies policy tensors into reusable NIXL buckets. `direct` exposes contiguous policy tensor chunks directly through NIXL and stages only tensors that cannot be directly exposed. `auto` stages sub-bucket tensors but uses direct descriptors for contiguous tensors at or above `max(direct_min_bytes, update_weights_bucket_megabytes)`. |
| `buffer_count` | Number of reusable staged transfer buffers. `2` is the original ping-pong path. `3` or higher allows more outstanding staged reads before a sender reuses a buffer. |
| `background_progress` | Whether the receiver starts a lightweight background progress thread for in-flight NIXL reads. This lets NIXL progress continue while vLLM synchronously loads the previous batch. Keep this `false` unless the target cluster shows stable tail latency with it enabled. |
| `load_batch_bucket_count` | Number of received transfer buckets to coalesce into one vLLM load batch. Keep this at `1` unless enough buffers are available to coalesce without starving receive prefetch; benchmark `2` with `buffer_count >= 4`. |
| `direct_min_bytes` | Minimum tensor size for `transfer_mode: auto` direct transfer. Auto mode also floors this at the staged bucket size, because direct descriptors for sub-bucket tensors lose the coalescing benefit of staged buckets. Start at the bucket size, for example `2147483648` with 2048 MiB buckets, and tune against the staged baseline. |
| `direct_stripe_count` | Number of stripes to split each direct-transfer bucket into. This only affects `transfer_mode: auto` or `direct`; `1` preserves the existing one-read-per-bucket behavior. Try `2` or `4` only with enough `buffer_count` headroom for concurrent receiver reads. |
| `metadata_batch_size` | Number of bucket metadata records to publish together on the control path. `1` preserves one ZMQ object per bucket. Values above `1` mainly help direct-striped runs where many direct stripe descriptors are emitted together. |
| `backend_name` | NIXL backend plugin name, usually `UCX`. |
| `backend_init_params` | Optional NIXL backend initialization parameters. Values are converted to strings before NIXL receives them. |

## Fault Tolerance

NIXL/UCX can help a training job fail fast on transport errors, which lets the
outer job launcher restart from the latest durable checkpoint. The current
checkpoint-engine refit path does not transparently replace a dead Ray actor or
rebuild the vLLM generation group inside the same training step. Treat
NIXL/UCX fault tolerance as transport-level error detection plus clean failure
propagation.

For runs where restartability matters, enable UCX peer error handling in the
NIXL backend config when your NIXL/UCX build supports it:

```yaml
policy:
  generation:
    checkpoint_engine:
      enabled: true
      backend: nixl
      engine_kwargs:
        nixl:
          device: cuda
          cleanup_after_load: false
          topology: auto
          transfer_mode: staged
          buffer_count: 2
          direct_min_bytes: 2147483648
          backend_name: UCX
          backend_init_params:
            ucx_error_handling_mode: peer
```

`ucx_error_handling_mode: none` is useful for performance experiments on stable
clusters, but it gives UCX less ability to report failed peers to NIXL. Use
`peer` for production or fault-injection testing so NIXL transfers can enter an
error state instead of waiting on a lost endpoint indefinitely.

Pair UCX peer error handling with bounded transport retry and keepalive values:

```sh
export UCX_RC_TIMEOUT=30s
export UCX_RC_RETRY_COUNT=7
export UCX_KEEPALIVE_INTERVAL=1s
export UCX_KEEPALIVE_NUM_EPS=10
```

These settings do not make an individual transfer magically recover. They bound
how long UCX waits before declaring a peer unhealthy. When UCX/NIXL reports an
error, NeMo RL marks the refit as failed, tears down per-refit NIXL peer state
through `finalize()`, and the training process should exit or be restarted by
the scheduler or fault-tolerant launcher.

Use normal NeMo RL checkpointing for restartable training:

```yaml
checkpointing:
  enabled: true
  checkpoint_dir: /path/to/restartable/checkpoints
```

During fault-injection tests, look for one of these outcomes:

- NIXL/UCX reports an error and the job fails promptly.
- The scheduler restarts the job from the latest NeMo RL checkpoint.
- After restart, vLLM logs `NIXL vLLM worker preinit completed: backend=UCX`
  and subsequent refits print `[vLLM refit]` timing again.

## Command-Line Override Example

The same settings can be passed as Hydra overrides:

```sh
uv run --extra mcore --extra vllm examples/run_grpo.py \
  --config examples/configs/grpo_math_8B.yaml \
  cluster.num_nodes=2 \
  policy.generation.colocated.enabled=false \
  policy.generation.colocated.resources.num_nodes=1 \
  policy.generation.colocated.resources.gpus_per_node=8 \
  policy.generation.checkpoint_engine.enabled=true \
  policy.generation.checkpoint_engine.backend=nixl \
  policy.generation.checkpoint_engine.update_weights_bucket_megabytes=2048 \
  ++policy.generation.checkpoint_engine.engine_kwargs.nixl.device=cpu \
  ++policy.generation.checkpoint_engine.engine_kwargs.nixl.cleanup_after_load=false \
  ++policy.generation.checkpoint_engine.engine_kwargs.nixl.topology=auto \
  ++policy.generation.checkpoint_engine.engine_kwargs.nixl.transfer_mode=staged \
  ++policy.generation.checkpoint_engine.engine_kwargs.nixl.buffer_count=2 \
  ++policy.generation.checkpoint_engine.engine_kwargs.nixl.backend_name=UCX \
  ++policy.generation.checkpoint_engine.engine_kwargs.nixl.backend_init_params.ucx_error_handling_mode=none
```

Adjust `cluster.num_nodes` and
`policy.generation.colocated.resources.{num_nodes,gpus_per_node}` so the cluster
has enough policy and generation resources. For example, on two 8-GPU nodes, the
snippet above dedicates one node to vLLM generation and leaves one node for
policy workers.

## Runtime Requirements

NIXL must be importable in every Python environment that participates in refit:

- the driver/base environment
- policy worker environments
- vLLM worker environments, including async vLLM worker environments when used

Install the appropriate NIXL packages in those environments or bake them into
the container image. For CUDA 12 environments this is typically:

```sh
uv pip install nixl-cu12 nixl
```

UCX transport selection is controlled by UCX runtime environment variables.
Keep the checkpoint-engine feature selection in YAML/config, and use UCX
environment variables only for transport-level settings:

```sh
export UCX_NET_DEVICES=mlx5_0:1
export UCX_TLS=rc,self,sm
export UCX_IB_ROCE_REACHABILITY_MODE=all
export UCX_MAX_RNDV_RAILS=1
export UCX_WARN_UNUSED_ENV_VARS=n
export NIXL_LOG_LEVEL=INFO
```

When vLLM uses nested Ray workers, make sure transport variables are copied into
those workers using vLLM's normal environment-copy settings:

```sh
export VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY=MELLANOX_
export VLLM_RAY_EXTRA_ENV_VARS_TO_COPY=LD_LIBRARY_PATH,NIXL_LOG_LEVEL,NVIDIA_VISIBLE_DEVICES,UCX_NET_DEVICES,UCX_TLS,UCX_IB_ROCE_REACHABILITY_MODE,UCX_MAX_RNDV_RAILS,UCX_WARN_UNUSED_ENV_VARS
```

Do not configure the checkpoint-engine backend through ad hoc NeMo RL
environment variables. The backend, bucket size, device, and backend parameters
belong in `policy.generation.checkpoint_engine`.

## Performance Best Practices

For large non-colocated vLLM refits, start with CUDA transfer buffers when the
NIXL/UCX runtime supports CUDA memory registration. Keep checkpoint-engine
feature settings in YAML/config and use environment variables only for UCX/NIXL
transport runtime selection.

Recommended checkpoint-engine settings for large 30B-class MoE refits:

```yaml
policy:
  generation:
    checkpoint_engine:
      enabled: true
      backend: nixl
      update_weights_bucket_megabytes: 1024
      engine_kwargs:
        nixl:
          device: cuda
          cleanup_after_load: false
          topology: auto
          transfer_mode: staged
          buffer_count: 3
          background_progress: false
          load_batch_bucket_count: 1
          direct_min_bytes: 2147483648
          direct_stripe_count: 1
          metadata_batch_size: 1
          backend_name: UCX
          backend_init_params:
            ucx_error_handling_mode: none
            engine_config: MAX_RMA_RAILS=8
            device_list: "mlx5_0,mlx5_1,mlx5_2,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8"
```

Recommended UCX runtime settings for a stable performance run:

```sh
export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1
export UCX_TLS=rc,cuda_copy,cuda_ipc,self,sm
export UCX_MAX_RNDV_RAILS=8
export UCX_IB_ROCE_REACHABILITY_MODE=all
export UCX_WARN_UNUSED_ENV_VARS=n
export NIXL_LOG_LEVEL=INFO
```

Use the NIC list that is valid on your nodes; do not copy interface names
blindly across clusters. If CUDA buffers are not available, use
`device: cpu` and remove CUDA transports from `UCX_TLS`, for example
`UCX_TLS=rc,self,sm`. Tune the CPU fallback separately; on the tested 30B MoE
setup, the best CPU fallback was 1536 MiB buckets with four process-level RDMA
rails and no NIXL UCX `device_list/MAX_RMA_RAILS` backend parameters.

On Ray launchers that start Ray inside the container, apply the container setup
and transport environment before `ray start` on every head and worker node.
Otherwise the driver can see the intended settings while nested Ray/vLLM worker
processes still inherit the wrong UCX library path or transport variables.

Use measured timings to tune from there:

- Prefer RDMA transports; logs should show `rc_mlx5`, not TCP-only transport.
- Sweep bucket size around `1024` to `2048` MiB instead of assuming larger is
  always faster. On the latest tested 30B MoE CUDA-buffer refit, eight rails
  with `1024` MiB buckets and three staged buffers reduced steady vLLM receive
  time from about `0.90s` to about `0.82s`; for CPU fallback, `1536` MiB
  buckets were best.
- Try multiple UCX rails when the fabric supports it. On the same 30B MoE
  setup, eight CUDA rails plus three staged buffers reduced steady
  transfer/update to about `1.36s` to `1.38s`, compared with about `3.7s` for
  four CUDA rails and about `3.1s` for NCCL broadcast. CPU fallback regressed
  with eight rails; keep it on four rails unless a local sweep proves otherwise.
- Pass the RDMA NICs through NIXL UCX `backend_init_params.device_list` as well
  as process-level `UCX_NET_DEVICES` when using CUDA transfer buffers. On the
  tested 30B MoE setup, adding the eight-device list plus
  `engine_config: MAX_RMA_RAILS=8`, then using `1024` MiB buckets with three
  buffers, reduced steady transfer/update from about `3.7s` to about `1.36s`
  to `1.38s`.
- Benchmark `transfer_mode: auto` against the staged baseline. Auto mode keeps
  small tensors packed in staged buckets, but exposes large contiguous policy
  tensor chunks directly to NIXL to avoid the policy-side bucket copy. If direct
  descriptor setup dominates on a model, keep `transfer_mode: staged`.
- For direct or auto transfer, benchmark `direct_stripe_count: 2` or `4`
  together with `metadata_batch_size` set to the same value. This can expose
  multiple reads for a large tensor chunk while batching the Python control
  messages. It is not expected to help `transfer_mode: staged`.
- Benchmark `buffer_count: 3` against `2`. More staged buffers let the sender
  keep additional reads outstanding before reusing a buffer, and the receiver
  can prefetch future reads before yielding the current bucket to vLLM. Each
  extra buffer reserves another `update_weights_bucket_megabytes` on every
  participating worker.
- Use `topology: leader_tree` when rollout workers outnumber policy workers.
  `auto` selects this topology for that shape. Keep `leader_chain` only as a
  comparison point or for debugging because the chain serializes rollout-side
  forwarding.
- Do not rely only on `UCX_MAX_RNDV_RAILS` for NIXL refit. NIXL checkpoint-engine
  transfers are UCX RMA READ operations, and the tested runtime still printed a
  single `rc_mlx5` device in the `rma(...)` log line even when the device-list
  backend config improved measured throughput. Use the `[vLLM refit] receive=...`
  timing as the source of truth.
- For fault-tolerant production runs, prefer
  `backend_init_params.ucx_error_handling_mode=peer`; reserve `none` for stable
  benchmarking where lowest transport overhead is the goal.

## Verify the Run

The driver log should show that the checkpoint engine is selected:

```text
Using checkpoint-engine refit backend: nixl
```

For NIXL + vLLM, the internal vLLM worker processes should initialize NIXL
before vLLM worker setup:

```text
NIXL vLLM worker preinit completed: backend=UCX
```

With `UCX_LOG_LEVEL=info`, UCX should report an RDMA transport such as:

```text
rma(rc_mlx5/mlx5_0:1)
```

If UCX reports only TCP transports, the run is not using RDMA and refit will be
much slower.

During each update, vLLM prints checkpoint-engine load timing:

```text
[vLLM refit] Loaded 18867 tensors in 29 batches via checkpoint engine; bytes=56.87GiB total=11.90s receive=10.71s load=1.18s sync=0.00s postprocess=0.00s cleanup=0.00s
```

The step timing also includes the end-to-end update:

```text
prepare_for_generation/transfer_and_update_weights: 11.94s
```

The `[vLLM refit]` line measures receive plus vLLM load time inside the vLLM
worker. The `transfer_and_update_weights` timer includes the full orchestration
window from the GRPO driver.

## Try a Correctness Smoke Test

`tools/refit_verifier.py` compares vLLM and Megatron logprobs after a refit:

```sh
uv run --extra mcore --extra vllm python tools/refit_verifier.py \
  --model_name /path/to/model \
  --tp_size 1 \
  --ep_size 1 \
  --pp_size 1
```

This tool is useful for validating refit correctness and model compatibility.
It currently exercises the colocated refit path. To test the NIXL
checkpoint-engine path, run a non-colocated GRPO job with
`policy.generation.checkpoint_engine.enabled=true` and inspect the log markers
above.

## Troubleshooting

### The run errors with "checkpoint-engine refit is only supported for non-colocated generation"

Set:

```yaml
policy:
  generation:
    colocated:
      enabled: false
```

Checkpoint-engine refit is for non-colocated generation only.

### NIXL cannot be imported

Install NIXL in the environment that failed. In Ray runs, the driver, policy
workers, vLLM workers, and async vLLM workers may use different virtual
environments.

### UCX logs say CUDA support was not found

This is expected when `engine_kwargs.nixl.device=cpu`, because the transfer
buffers live in host memory. If you set `device=cuda`, the NIXL/UCX build must
support CUDA memory registration.

### NIXL works but is slow

Check the UCX transport line. RDMA should show `rc_mlx5` or another expected
RDMA transport. If it shows TCP only, verify `UCX_NET_DEVICES`, `UCX_TLS`,
container device visibility, and network interface availability on every node.

Also check bucket sizing. Very small buckets increase metadata and synchronization
overhead. Start with `update_weights_bucket_megabytes=2048` for large models and
adjust only after measuring.

### The vLLM preinit marker is missing

Confirm that:

- `policy.generation.checkpoint_engine.enabled=true`
- `policy.generation.checkpoint_engine.backend=nixl`
- the run is using vLLM generation
- the vLLM worker code in the active runtime environment matches the current
  NeMo RL checkout

The NIXL preinit is config-driven. It should not require any
`NRL_VLLM_NIXL_*` environment variables.

### A node or NIC failure causes the job to hang

Use `backend_init_params.ucx_error_handling_mode=peer` and set bounded UCX retry
and keepalive values. Without UCX peer error handling, some transport failures
can look like an indefinitely pending NIXL transfer rather than a clean
`ERR` state.

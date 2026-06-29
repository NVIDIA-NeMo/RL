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
      update_weights_bucket_megabytes: 2048
      engine_kwargs:
        nixl:
          device: cuda
          cleanup_after_load: false
          backend_name: UCX
          backend_init_params:
            engine_config: MAX_RMA_RAILS=8
            device_list: "mlx5_0,mlx5_1,mlx5_2,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8"
```

Key settings:

| Key | Meaning |
|---|---|
| `update_weights_bucket_megabytes` | Reusable transfer-buffer size per participating worker. Start with `2048`; tune only after measuring. |
| `device` | `cuda` uses GPU RDMA buffers. `cpu` uses host-pinned buffers and is mainly a fallback. |
| `cleanup_after_load` | Set `false` to avoid extra `torch.cuda.empty_cache()` overhead after each refit when memory is stable. |
| `shard_hf_weights` | Megatron/vLLM MoE optimization that sends vLLM TP-sharded HF expert weights. |
| `sharded_target_tp_size` | vLLM target TP size used with `shard_hf_weights`; defaults to rollout world size. |
| `backend_init_params` | NIXL backend parameters such as UCX peer error handling, device lists, and UCX engine config. |

The built-in NIXL topology is paired policy-to-rollout transfer, so allocate at
least as many policy workers as rollout workers.

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
export UCX_MAX_RNDV_RAILS=8
export UCX_WARN_UNUSED_ENV_VARS=n
export NIXL_LOG_LEVEL=INFO
```

When vLLM starts nested Ray workers, copy the transport variables into those
workers:

```sh
export VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY=MELLANOX_
export VLLM_RAY_EXTRA_ENV_VARS_TO_COPY=LD_LIBRARY_PATH,NIXL_LOG_LEVEL,NVIDIA_VISIBLE_DEVICES,UCX_NET_DEVICES,UCX_TLS,UCX_IB_ROCE_REACHABILITY_MODE,UCX_MAX_RNDV_RAILS,UCX_WARN_UNUSED_ENV_VARS
```

Use NIC names that exist on the target nodes. With `UCX_LOG_LEVEL=info`, UCX
should report an RDMA transport such as `rc_mlx5`; TCP-only transport will be
much slower.

## Performance Notes

Use `device: cuda` for the fast path when NIXL/UCX can register CUDA memory.
For large MoE models with vLLM, leave enough GPU memory for NIXL buffers:

```yaml
policy:
  generation:
    vllm_cfg:
      gpu_memory_utilization: 0.82
    checkpoint_engine:
      update_weights_bucket_megabytes: 4096
      engine_kwargs:
        nixl:
          device: cuda
          cleanup_after_load: false
          shard_hf_weights: true
          sharded_target_tp_size: 32
```

`shard_hf_weights` is a Megatron/vLLM MoE optimization. Do not enable it for
non-MoE models or for vLLM models that cannot load sharded HF expert weights.

Avoid `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for CUDA-buffer NIXL
refit unless that workload has been explicitly validated with it.

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

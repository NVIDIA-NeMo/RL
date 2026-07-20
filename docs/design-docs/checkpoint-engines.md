# Checkpoint Engine Design

Checkpoint engines are runtime refit transports for non-colocated generation.
They let GRPO move policy weights directly from policy workers to generation
workers without using the driver as a model-sized staging point.

The first built-in backend is NIXL. The current implementation targets policy
workers refitting non-colocated vLLM generation workers. Colocated generation
still uses the existing IPC/HTTP refit paths, and non-colocated generation
without checkpoint engines still uses the existing NCCL collective path.

The user-facing guide is [Checkpoint-Engine Refit](../guides/checkpoint-engine-refit.md).

## Goals

Checkpoint engines are designed to:

- keep GRPO orchestration independent from the transfer backend
- stream weight batches instead of materializing a full model copy in the driver
- let backend implementations own their metadata, buffers, and peer setup
- allow additional transfer backends through a class-path plugin

Checkpoint engines do not replace durable training checkpoints. They are used
only for the runtime weight update between policy and generation workers.

## Control Flow

The refit lifecycle is coordinated by `CheckpointEngineWeightSynchronizer`:

1. Read `policy.generation.refit_transport` and its matching `refit_cfg` scope.
2. Resolve the configured bucket size from the smallest fixed GPU capacity
   reported by policy and vLLM workers before transfer buffers are allocated.
3. Instantiate the backend on policy workers and vLLM internal workers.
4. Call `prepare()` and collect Ray-serializable metadata from every backend
   instance.
5. Initialize policy and rollout peers with the combined metadata list.
6. Keep the backend initialized across refits for that synchronizer.
7. For each refit, ask policy workers to send weights through the backend.
8. Ask generation workers to receive batches, directly copy supported
   destination-local expert shards, and pass remaining tensors through vLLM's
   normal weight-loading path.
9. Call `shutdown()` to finalize backend state when the synchronizer is no
   longer needed.

Policy metadata appears first in the combined metadata list, followed by
generation metadata. Backends receive `train_world_size` and
`rollout_world_size` so they can interpret that list.

## Configuration Contract

Checkpoint-engine refit uses the same selector as other non-colocated vLLM
transports:

```yaml
policy:
  generation:
    backend: vllm
    colocated:
      enabled: false
    refit_transport: nixl
    refit_cfg:
      nixl:
        update_weights_bucket_memory_ratio: 0.05
        device: cuda
        release_after_refit: false
        backend_name: UCX
        # Optional, cluster-specific eight-rail tuning.
        backend_init_params:
          engine_config: MAX_RMA_RAILS=8
          device_list: "mlx5_0,mlx5_1,mlx5_2,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8"
```

`refit_transport` can select:

- `nixl`, which maps to
  `nemo_rl.utils.checkpoint_engines.nixl:NIXLCheckpointEngine`
- a class path in `module:ClassName` format

For a plugin, key its settings by the exact class path:

```yaml
policy:
  generation:
    refit_transport: "my_pkg.refit:MyCheckpointEngine"
    refit_cfg:
      "my_pkg.refit:MyCheckpointEngine":
        update_weights_bucket_memory_ratio: 0.05
        transport: custom
```

`update_weights_bucket_memory_ratio` is the fraction of fixed total GPU memory
used by each transfer bucket. Its Pydantic default is `0.05`. The
synchronizer queries every policy and rollout worker, uses the smallest reported
GPU capacity, and computes
`minimum_total_memory_bytes * update_weights_bucket_memory_ratio`, rounded down
to a MiB. The resolved size is fixed for the synchronizer lifetime. NIXL owns
two transfer buffers, so its total allocation is twice the configured ratio.

The factory passes the resolved `bucket_size` in bytes plus the selected backend
kwargs to the backend constructor.

## Backend Interface

Backends subclass `nemo_rl.utils.checkpoint_engines.base.CheckpointEngine`.

```python
from collections.abc import AsyncGenerator, Generator
from typing import Any

import torch

from nemo_rl.utils.checkpoint_engines.base import CheckpointEngine


class MyCheckpointEngine(CheckpointEngine):
    def __init__(self, bucket_size: int, transport: str) -> None:
        self.bucket_size = bucket_size
        self.transport = transport

    def prepare(self) -> Any:
        """Allocate or register buffers and return Ray-serializable metadata."""
        ...

    def get_target_weight_layout(self) -> dict[str, Any] | None:
        """Return this policy rank's destination layout, if sharding weights."""
        ...

    def init_policy_process_group(
        self,
        *,
        worker_rank: int,
        train_world_size: int,
        rollout_world_size: int,
        metadata: list[Any],
    ) -> None:
        """Connect a policy worker to its transfer peer."""
        ...

    def init_rollout_process_group(
        self,
        *,
        rollout_rank: int,
        train_world_size: int,
        rollout_world_size: int,
        metadata: list[Any],
    ) -> None:
        """Connect a rollout worker to its transfer peer."""
        ...

    def finalize(self) -> None:
        """Release per-refit state if the backend owns any."""
        ...

    async def send_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
    ) -> None:
        """Send `(name, tensor)` weights from the policy side."""
        ...

    async def receive_weight_batches(
        self,
    ) -> AsyncGenerator[list[tuple[str, torch.Tensor]], None]:
        """Yield `(name, tensor)` batches on the generation side."""
        ...
```

The `weights` generator is consumed once. `receive_weight_batches()` should
yield tensors with original parameter names and values. vLLM loads each yielded
batch immediately.

A backend that enables `shard_expert_weights` must implement
`get_target_weight_layout()`. It returns `None` on policy ranks without a
rollout peer; otherwise it returns the destination layout used to filter and
slice the policy iterator.

The built-in NIXL backend accepts `release_after_refit`. When enabled,
`finalize()` deregisters and frees its transfer buffers. A subsequent
`prepare()` allocates and registers new buffers before returning metadata. The
agent remains live, and the default retains the buffers as well for lower
latency.

## Worker Integration

Concrete policy workers opt into `PolicyCheckpointEngineMixin` beside their
backend-specific send mixin. `AbstractPolicyWorker` does not expose
checkpoint-engine methods, so value workers and other subclasses do not inherit
unused RPCs. The synchronizer invokes `checkpoint_engine_rpc()` for each
lifecycle step: creating the backend, preparing metadata, joining the backend
topology, sending weights, and finalizing the backend. Each concrete policy
worker supplies the iterator used by `send_weights_via_checkpoint_engine()`:

- Megatron streams `_iter_params_with_optional_kv_scales()`.
- DTensor/FSDP2 streams the same local DTensor conversion path used by IPC and
  NCCL refit.

Some policy iterators materialize weights through distributed collectives. A
checkpoint backend must still drain the iterator on policy ranks without a
rollout peer so those collectives are entered by every required rank.

vLLM generation workers forward checkpoint-engine calls through
`collective_rpc()` into vLLM internal workers. A normal vLLM worker uses
`VllmInternalWorkerExtension`, which contains the generic full and FP8 loaders
but no checkpoint-engine lifecycle methods. Enabling checkpoint-engine refit
selects `VllmInternalWorkerExtensionWithCheckpointEngine`, which adds backend
creation, metadata preparation, receiving, and sharded-expert dispatch. Its
explicit full-weight path delegates complete HF tensors to
`model.load_weights()` when `shard_expert_weights` is false. With
sharded-expert refit, it instead loads supported local expert shards through
validated destination-local views of canonical vLLM expert parameters. Dense
or otherwise unhandled tensors use the full-weight path. Before advertising a
sharded layout, the worker checks the physical expert storage shape and
backend. The vLLM worker prints timing for each update:

```text
[vLLM refit] Loaded ... via checkpoint engine; bytes=... total=... receive=... load=...
```

NeMo RL pins the tested vLLM version, but the sharded MoE path still depends on
vLLM's canonical expert parameter layout. A version bump fails setup if the
storage dimensions or backend change, and the vLLM unit test compares batched
W1, W3, and W2 destination-local copies against vLLM's normal full-weight TP
loading. The residual silent-error risk is a same-shape layout semantic change.
A vLLM bump must therefore run `tools/refit_verifier.py`; use
`shard_expert_weights: false` until that verification passes.

Async vLLM uses `checkpoint_engine_rpc_async()` and resolves nested
`collective_rpc()` awaitables, futures, and Ray object refs before reporting
success.

## NIXL Backend

The built-in NIXL backend is selected with `refit_transport: nixl`. It currently uses:

- NIXL agents for memory registration and transfer
- ZMQ control messages for bucket metadata and completion notifications
- two reusable transfer buffers per worker
- staged bucket copies from policy tensors into NIXL buffers
- `split_weight_chunks()` and `merge_weight_chunk_batches()` for tensors larger
  than one bucket

The current topology is paired policy-to-rollout transfer. Policy rank `i`
sends to rollout rank `i` when `i < rollout_world_size`; extra policy workers do
not send. A rollout worker connects to the policy metadata entry at its rollout
rank, so production runs should allocate at least as many policy workers as
rollout workers for this backend.

When sharded-expert refit is enabled, rollout metadata also contains the actual
vLLM destination layout for that worker. Each expert parameter reports whether
vLLM uses expert placement, its local global-expert IDs, and any remaining TP
coordinate. The layout also includes the missing-layer prefixes that vLLM uses
for pipeline-parallel loading. The paired policy worker drops weights absent from
the destination stage, then slices experts for TP or filters complete experts
for EP before filling NIXL buckets. This avoids deriving vLLM ownership from
Ray/global rank ordering. The destination metadata is authoritative; the NIXL
backend does not accept a source-side target-TP hint.

`device` controls the staged transfer-buffer device:

- `cuda`: allocate CUDA buffers and use CUDA-capable NIXL/UCX transfer. If
  CuPy is available, CUDA buffers are allocated through CuPy before being
  wrapped as torch tensors.
- `cpu`: allocate host buffers, pinned when CUDA is available.

`backend_name` defaults to `UCX`. `device_list` restricts the local UCX network
devices and is independent of the distributed world size. The same list remains
valid when adding or removing homogeneous nodes; update it only when the
per-node HCA names or topology change. Omitting `device_list` lets UCX discover
available devices, but the NIXL 1.3 runtime used for validation defaults
`MAX_RMA_RAILS` to `2`, so that portable configuration does not reproduce the
validated eight-rail performance. For tuned runs, use devices available on
every participating node and keep `MAX_RMA_RAILS` no larger than the number of
usable selected rails. Values in `backend_init_params` are converted to strings
before creating the NIXL backend.

Prefer `backend_init_params.device_list` over `UCX_NET_DEVICES` for per-run
selection because it is recorded with the run configuration. Both constrain
UCX discovery rather than overriding one another, so conflicting values can
exclude the intended devices. Reserve `UCX_NET_DEVICES` for a cluster-wide
override and normally configure only one of the two.

The validated cluster omits `mlx5_3` because it maps to the Ethernet-link-layer
interface `enp90s0np0` on the `10.65.x.x/31` network, while the eight selected
HCAs use the InfiniBand link layer and map to `ibp*` interfaces on the
`100.126.x.x/16` RDMA data fabric. This mapping is cluster-specific; use
`ibdev2netdev` and inspect each RDMA port's `link_layer` instead of assuming
that device index 3 should always be excluded.

### Validated Full-Model Layout

A DeepSeek-V3 BF16 run validated different source and destination layouts:
Megatron TP1/PP16/EP16 across 256 policy workers refit vLLM TP32/PP1/EP1 across
32 rollout workers. The destination-reported layout drove PP filtering and TP
expert slicing without requiring the policy and rollout rank layouts to match.
Each rollout rank received 45,395 destination-local tensors in 18 batches, or
69.95 GiB. That cluster was tuned with eight explicitly selected HCAs and
`MAX_RMA_RAILS=8`; those device names are not portable defaults. Performance
and correctness-control results are recorded in the
[user guide](../guides/checkpoint-engine-refit.md#deepseek-v3-benchmark).

## NIXL Preinit

NIXL/UCX backend creation can be expensive if it first happens in the critical
path. The current code preinitializes NIXL agents in two places when the config
selects `refit_transport: nixl`:

- policy worker construction
- vLLM internal worker construction, via vLLM's `worker_cls` hook

NeMo RL passes the normalized `refit_cfg.nixl` settings through
`VllmConfig.additional_config`. `NixlVllmWorker` creates and retains the
preinit agent before calling vLLM's worker constructor. The preinit path uses
the configured `backend_name` and `backend_init_params`; logs usually show
NIXL agents named `preinit-...` during worker setup.

`worker_cls` remains the early-construction hook for NIXL preinitialization.
`worker_extension_cls` is selected separately: the base extension is used
without checkpoint-engine refit, and the checkpoint-engine subclass is used
when the feature is enabled.

## How NIXL Supports Fault Tolerance

NIXL is the transfer layer. It does not create, remove, or replace Ray/vLLM
actors, and it does not route rollout requests.

In NeMo RL, NIXL supports fault tolerance in three concrete ways:

- **Transport errors become refit errors.** With UCX peer error handling
  enabled, a lost peer can be reported to NIXL instead of leaving the transfer
  waiting indefinitely.
- **Failed refits are propagated.** The NIXL backend raises when a read cannot
  start or when `check_xfer_state()` reports `ERR`; vLLM reports the failed
  weight update, and `CheckpointEngineWeightSynchronizer` raises for the refit.
- **A restarted synchronizer can use fresh peers.** `shutdown()` disconnects
  the current peers. With `release_after_refit: true`, it also deregisters and
  frees transfer buffers. The next `init_communicator()` registers new buffers,
  exchanges `prepare()` metadata, and installs a new policy-to-rollout mapping.

So the recovery model is fail the current refit, change the rollout actor set
outside NIXL, rebuild the checkpoint-engine communicator, then run a full refit
with fresh metadata before routing prompts to the new set.

`tools/nixl_elastic_rollout_demo.py` demonstrates this teardown, metadata
exchange, and reinitialization sequence with synthetic weights. Actor creation,
removal, health checks, and request routing remain orchestration concerns.

## Adding Another Backend

To add a backend:

1. Implement a `CheckpointEngine` subclass.
2. Accept `bucket_size` in bytes in the constructor.
3. Return only Ray-serializable metadata from `prepare()`.
4. Implement policy and rollout peer setup using the combined metadata list.
5. Stream policy weights from the input generator without replaying it.
6. Yield vLLM-loadable `(name, tensor)` batches from `receive_weight_batches()`.
7. Add backend-specific config under `refit_cfg.<backend>`.
8. Use a `module:ClassName` `refit_transport` value, or add a short-name
   mapping in `create_checkpoint_engine()` if the backend should be built in.
9. Run a non-colocated GRPO job and verify the `[vLLM refit]` timing line.

Current limitations:

- Checkpoint-engine refit targets non-colocated policy-to-vLLM refit.
- SGLang and Megatron generation do not implement checkpoint-engine refit;
  [issue #3288](https://github.com/NVIDIA-NeMo/RL/issues/3288) tracks
  generation-side support. Megatron and DTensor policy backends are supported
  when the generation backend is vLLM.
- The built-in NIXL backend uses paired policy-to-rollout transfer only.
- Sharded vLLM EP refit supports static expert ownership and canonical
  unquantized Triton expert storage. Dynamic EPLB, redundant experts, and
  quantized or shuffled layouts require a destination layout adapter and are
  rejected during setup.

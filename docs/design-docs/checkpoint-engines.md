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

1. Read `policy.generation.checkpoint_engine`.
2. Instantiate the backend on policy workers and vLLM internal workers.
3. Call `prepare()` and collect Ray-serializable metadata from every backend
   instance.
4. Initialize policy and rollout peers with the combined metadata list.
5. Keep the backend initialized across refits for that synchronizer.
6. For each refit, ask policy workers to send weights through the backend.
7. Ask generation workers to receive batches, directly copy supported
   destination-local expert shards, and pass remaining tensors through vLLM's
   normal weight-loading path.
8. Call `shutdown()` to finalize backend state when the synchronizer is no
   longer needed.

Policy metadata appears first in the combined metadata list, followed by
generation metadata. Backends receive `train_world_size` and
`rollout_world_size` so they can interpret that list.

## Configuration Contract

Checkpoint-engine config lives under `policy.generation`:

```yaml
policy:
  generation:
    backend: vllm
    colocated:
      enabled: false
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

`backend` can be:

- `nixl`, which maps to
  `nemo_rl.utils.checkpoint_engines.nixl:NIXLCheckpointEngine`
- a class path in `module:ClassName` format

`engine_kwargs` must be keyed by the exact backend value. For a plugin:

```yaml
policy:
  generation:
    checkpoint_engine:
      enabled: true
      backend: "my_pkg.refit:MyCheckpointEngine"
      update_weights_bucket_megabytes: 1024
      engine_kwargs:
        "my_pkg.refit:MyCheckpointEngine":
          transport: custom
```

The factory passes `bucket_size` in bytes plus the selected backend kwargs to
the backend constructor.

## Backend Interface

Backends subclass `nemo_rl.utils.checkpoint_engines.base.CheckpointEngine`.

```python
from collections.abc import AsyncGenerator, Generator
from typing import Any

import torch

from nemo_rl.utils.checkpoint_engines.base import CheckpointEngine


class MyCheckpointEngine(CheckpointEngine):
    cleanup_after_load = True

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

A backend that enables `shard_hf_weights` must implement
`get_target_weight_layout()`. It returns `None` on policy ranks without a
rollout peer; otherwise it returns the destination layout used to filter and
slice the policy iterator.

`cleanup_after_load` is read by the vLLM worker after the receive loop. Set it
to `False` when the backend keeps stable buffers and avoiding extra
`torch.cuda.empty_cache()` calls is safe for the run.

## Worker Integration

Policy workers expose `checkpoint_engine_rpc()` from `AbstractPolicyWorker`.
The synchronizer invokes this RPC for each lifecycle step: creating the
backend, preparing metadata, joining the backend topology, sending weights, and
finalizing the backend. Each concrete policy worker supplies the iterator used
by `send_weights_via_checkpoint_engine()`:

- Megatron streams `_iter_params_with_optional_kv_scales()`.
- DTensor/FSDP2 streams the same local DTensor conversion path used by IPC and
  NCCL refit.

Some policy iterators materialize weights through distributed collectives. A
checkpoint backend must still drain the iterator on policy ranks without a
rollout peer so those collectives are entered by every required rank.

vLLM generation workers forward checkpoint-engine calls through
`collective_rpc()` into vLLM internal workers. The internal worker extension
creates the backend, prepares rollout metadata, receives weight batches, and
loads each batch. With sharded HF refit, it directly copies supported local
expert shards into canonical vLLM storage and sends dense or otherwise
unhandled tensors through the normal vLLM load path. The vLLM worker prints
timing for each update:

```text
[vLLM refit] Loaded ... via checkpoint engine; bytes=... total=... receive=... load=...
```

Async vLLM uses `checkpoint_engine_rpc_async()` and resolves nested
`collective_rpc()` awaitables, futures, and Ray object refs before reporting
success.

## NIXL Backend

The built-in NIXL backend is selected with `backend: nixl`. It currently uses:

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

When sharded HF refit is enabled, rollout metadata also contains the actual
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

`backend_name` defaults to `UCX`. Values in `backend_init_params` are converted
to strings before creating the NIXL backend. Use this field for NIXL backend
parameters such as UCX peer error handling, UCX device lists, and NIXL UCX
engine config.

### Validated Full-Model Layout

A DeepSeek-V3 BF16 run validated different source and destination layouts:
Megatron TP1/PP16/EP16 across 256 policy workers refit vLLM TP32/PP1/EP1 across
32 rollout workers. The destination-reported layout drove PP filtering and TP
expert slicing without requiring the policy and rollout rank layouts to match.
Each rollout rank received 45,395 destination-local tensors in 18 batches, or
69.95 GiB. Performance and correctness-control results are recorded in the
[user guide](../guides/checkpoint-engine-refit.md#deepseek-v3-benchmark).

## NIXL Preinit

NIXL/UCX backend creation can be expensive if it first happens in the critical
path. The current code preinitializes NIXL agents in two places when the config
selects `backend: nixl`:

- policy worker construction
- vLLM internal worker construction, via the vLLM worker patch hook

The preinit path uses the configured `backend_name` and `backend_init_params`.
Logs usually show NIXL agents named `preinit-...` during worker setup.

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
- **A restarted synchronizer can use fresh peers.** `shutdown()` releases the
  current backend state. Recreating the synchronizer or calling
  `init_communicator()` after shutdown exchanges new `prepare()` metadata and
  installs a new policy-to-rollout mapping.

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
7. Add backend-specific config under `engine_kwargs.<backend>`.
8. Use a `module:ClassName` backend string in config, or add a short-name
   mapping in `create_checkpoint_engine()` if the backend should be built in.
9. Run a non-colocated GRPO job and verify the `[vLLM refit]` timing line.

Current limitations:

- Checkpoint-engine refit targets non-colocated policy-to-vLLM refit.
- SGLang checkpoint-engine refit is not implemented.
- The built-in NIXL backend uses paired policy-to-rollout transfer only.
- Sharded vLLM EP refit supports static expert ownership and canonical
  unquantized Triton expert storage. Dynamic EPLB, redundant experts, and
  quantized or shuffled layouts require a destination layout adapter and are
  rejected during setup.

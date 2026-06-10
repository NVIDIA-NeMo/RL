# Checkpoint Engine Design

Checkpoint engines provide a backend-neutral way to transfer policy weights to
non-colocated generation workers during refit. They are used by GRPO when
`policy.generation.checkpoint_engine.enabled=true`.

The user-facing guide is [Checkpoint-Engine Refit](../guides/checkpoint-engine-refit.md).
This document describes the implementation contract and how to add new transfer
backends.

## Goals

Checkpoint engines are designed to:

- decouple refit orchestration from the transport implementation
- let each backend manage its own metadata, buffers, and process topology
- stream weight batches instead of materializing a full model copy in the driver
- support plugin backends without changing GRPO or vLLM code

Checkpoint engines do not replace normal checkpoint save/load. They are a
runtime refit transport used between policy workers and generation workers.

## Control Flow

The GRPO refit flow is:

1. Read `policy.generation.checkpoint_engine`.
2. Instantiate the configured backend on every policy worker and generation
   worker.
3. Call `prepare()` on every backend instance and collect Ray-serializable
   metadata.
4. Initialize the backend topology with the combined metadata list.
5. Ask policy workers to send model weights.
6. Ask generation workers to receive weight batches and load them into the
   generation backend.
7. Call `finalize()` on all backend instances in a `finally` block.

The policy metadata is placed first in the combined metadata list, followed by
generation metadata. Backends receive `train_world_size` and
`rollout_world_size` so they can interpret that list.

## Configuration Contract

Checkpoint-engine config is stored under `policy.generation`:

```yaml
policy:
  generation:
    checkpoint_engine:
      enabled: true
      backend: nixl
      update_weights_bucket_megabytes: 2048
      engine_kwargs:
        nixl:
          device: cuda
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
            ucx_error_handling_mode: peer
```

`backend` can be either:

- a registered backend name, such as `nixl`
- a class path, such as `my_pkg.refit:MyCheckpointEngine`

`engine_kwargs` must be keyed by the exact `backend` value. For a class-path
plugin:

```yaml
policy:
  generation:
    checkpoint_engine:
      enabled: true
      backend: "my_pkg.refit:MyCheckpointEngine"
      update_weights_bucket_megabytes: 1024
      engine_kwargs:
        "my_pkg.refit:MyCheckpointEngine":
          transport: my_transport
```

The factory passes `bucket_size` in bytes plus the selected backend kwargs to
the backend constructor. Backend-specific settings such as transfer device,
cleanup behavior, and transport plugin name live in config.

For NIXL/UCX CUDA-buffer performance runs, put UCX backend parameters in
`engine_kwargs.nixl.backend_init_params` rather than relying only on process
environment variables. For example, a 30B MoE CUDA-buffer refit improved when
NIXL received both the rail list and RMA rail count directly:

```yaml
policy:
  generation:
    checkpoint_engine:
      engine_kwargs:
        nixl:
          backend_init_params:
            ucx_error_handling_mode: none
            device_list: "mlx5_0,mlx5_1,mlx5_2,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8"
            engine_config: MAX_RMA_RAILS=8
```

Use NIC names that exist on the target nodes. `ucx_error_handling_mode: none`
is a benchmark setting; use `peer` for production runs that need transport
errors to surface promptly. CPU staging should be benchmarked independently;
the CUDA-buffer rail settings are not a safe default for host-pinned transfer
buffers. On the tested 30B MoE CPU fallback, four process-level RDMA rails with
1536 MiB buckets outperformed eight process-level rails.

## Backend Interface

Backends subclass
{py:class}`CheckpointEngine <nemo_rl.utils.checkpoint_engines.base.CheckpointEngine>`.

```python
from typing import Any, AsyncGenerator, Generator

import torch

from nemo_rl.utils.checkpoint_engines import (
    CheckpointEngine,
    CheckpointEngineRegistry,
)


@CheckpointEngineRegistry.register("my_backend")
class MyCheckpointEngine(CheckpointEngine):
    cleanup_after_load = True

    def __init__(self, bucket_size: int, device: str | torch.device):
        self.bucket_size = bucket_size
        self.device = torch.device(device)

    def prepare(self) -> Any:
        """Allocate or register buffers and return Ray-serializable metadata."""
        ...

    def init_policy_process_group(
        self,
        *,
        worker_rank: int,
        train_world_size: int,
        rollout_world_size: int,
        metadata: list[Any],
    ) -> None:
        """Connect a policy worker to the backend topology."""
        ...

    def init_rollout_process_group(
        self,
        *,
        rollout_rank: int,
        train_world_size: int,
        rollout_world_size: int,
        metadata: list[Any],
    ) -> None:
        """Connect a generation worker to the backend topology."""
        ...

    def finalize(self) -> None:
        """Release per-refit topology state."""
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

The `weights` generator is consumed once. Do not assume it can be replayed.

`receive_weight_batches()` should yield tensors with the original parameter
names and values. The generation backend loads each yielded batch immediately,
so yielding at transfer-bucket boundaries allows transfer and loading to overlap.

`cleanup_after_load` is read by the vLLM generation worker after the receive
loop. Set it to `False` when the backend can keep stable buffers and avoiding
extra cache cleanup is safe for steady-state training.

## Registry and Plugins

Built-in backends are lazy-imported by name through
{py:class}`CheckpointEngineRegistry <nemo_rl.utils.checkpoint_engines.base.CheckpointEngineRegistry>`.
External backends have two options:

1. Register a short name with `@CheckpointEngineRegistry.register("name")`.
2. Use a class path directly in config.

Class-path plugins do not need an import side effect. The registry imports the
module, looks up the class, validates that it subclasses `CheckpointEngine`, and
caches the result.

Supported class-path formats are:

```text
my_pkg.refit:MyCheckpointEngine
my_pkg.refit.MyCheckpointEngine
```

## Worker Integration

Policy workers use `BasePolicyWorker` helpers to instantiate the engine, prepare
metadata, join the backend topology, and send weights.

vLLM generation workers forward checkpoint-engine calls into vLLM internal
workers. The internal worker extension receives weight batches and calls the
normal vLLM load path for each batch. It also prints refit timing:

```text
[vLLM refit] Loaded ... via checkpoint engine; bytes=... total=... receive=... load=...
```

Async vLLM uses the same backend interface through async worker wrappers.

## NIXL Backend

The built-in NIXL backend is registered as `nixl`. It uses:

- NIXL agents for memory registration and transfer
- ZMQ messages for bucket metadata and transfer notifications
- reusable transfer buffers per worker for pipelined bucket movement
- optional direct source descriptors for contiguous policy tensor chunks
- `split_weight_chunks()` and `merge_weight_chunk_batches()` for tensors larger
  than one bucket

The NIXL backend chooses one of three topologies:

- If `train_world_size >= rollout_world_size`, each rollout rank is paired with
  a policy rank. Extra policy ranks are idle and do not materialize weights.
- If `rollout_world_size > train_world_size`, `auto` uses a rollout-side
  binary tree rooted at policy rank 0. Tree forwarding avoids the fully
  serialized rollout chain.
- `leader_chain` keeps the older single-policy-sender chain for debugging and
  benchmark comparison.

Set `engine_kwargs.nixl.topology=leader_chain` to force the single-policy-sender
chain even when policy workers can cover rollout workers. This reduces
policy-side export and send work, but it serializes rollout forwarding through
the chain, so benchmark it against the default paired topology on the target
cluster.

`finalize()` keeps peer connections, memory registrations, transfer buffers, and
read handles alive for the lifetime of the worker. Reusing these objects avoids
repeated multi-GB memory registration and transfer-handle initialization in
long-lived Ray/vLLM actors.

`transfer_mode=staged` is the conservative default: policy tensors are copied
into reusable registered buckets, and small tensors are packed to reduce
metadata overhead. `transfer_mode=auto` uses direct NIXL descriptors only for
large contiguous policy tensors at least as large as the staged bucket size.
Sub-bucket tensors stay staged because direct descriptors lose the coalescing
benefit of staged buckets and can make source-side registration dominate. Use
`transfer_mode=direct` to force direct descriptors for all contiguous tensors
when benchmarking the raw source-direct path. Increase `buffer_count` when
sender-side wait time shows that two staged buffers are not enough to keep
reads in flight. The receiver also uses extra buffers to prefetch future reads
before yielding the current transfer bucket to vLLM, while preserving buffer
lifetime until vLLM has consumed the yielded views.

`direct_stripe_count` splits each direct-transfer bucket into smaller direct
read stripes. This is an opt-in direct/auto-mode tuning knob for clusters where
one large NIXL read does not keep the available RDMA lanes busy. Pair it with
enough receiver `buffer_count` headroom and set `metadata_batch_size` above `1`
so the additional stripe descriptors do not turn into one ZMQ publish per
stripe. Staged mode keeps using the normal bucket size and is unaffected by
direct striping.

Bucket metadata is still carried on the Python control path, but the NIXL
backend can publish several metadata records in one control message. Receivers
unwrap the batch before starting reads, so the data path and ordering contract
remain one logical bucket at a time.

When `background_progress=true`, each receiver read has a lightweight progress
thread that polls NIXL state while the main thread is inside the synchronous
vLLM load path. This reduces exposed wait time for reads that were started
before the current load batch was yielded. It is disabled by default and should
be enabled only when cluster benchmarks show stable tail latency.

`load_batch_bucket_count` can coalesce multiple received transfer buckets into
one vLLM load call. NIXL uses explicit bucket leases for this path: chunks that
were copied into a reconstructed tensor release their transfer bucket
immediately, while chunks that are exposed as tensor views keep their bucket
leased until the outer vLLM load batch returns. This preserves view lifetime
while reducing Python load-call overhead. Benchmark values above `1` with
enough `buffer_count` headroom, because coalescing too many buckets can reduce
receive prefetch overlap.

For CUDA-backed batches, NIXL returns a list subclass that implements
`record_cuda_load_complete()`. vLLM calls this after `load_weights()` queues its
copies. NIXL records CUDA events on the batch devices and waits on those events
before recycling transfer buffers, which avoids a hard
`torch.cuda.current_stream().synchronize()` after every load batch.

Buffer recycling uses an async release queue. If the receiver already has
another read in flight, the batch lease is released by a background task after
the load-complete CUDA event is ready. If releasing the current bucket is needed
to prefetch the next bucket, the release stays inline so the receive generator
cannot run out of work and terminate early.

On the sender, staged buckets use CUDA events instead of a device-wide sync
before every metadata publish. The sender enqueues the staged copy, records
events for the involved CUDA devices, and keeps filling later buffers. Metadata
publication still happens in original bucket order; a pending bucket is exposed
only after its copy event is complete, and buffer reuse waits for the previous
read notification for that buffer.

When the runtime NIXL Python agent exposes a native `progress()` method, the
wrapper invokes it under the agent lock before polling transfer state or
notifications. Runtimes without that method keep the existing
`check_xfer_state()`/notification polling behavior.

### Fault-Tolerance Boundary

The NIXL backend is restart-safe, not actor-healing. It is designed so a failed
transfer becomes a failed refit attempt that the driver can observe:

- `ReadOperation.begin_read()` raises if NIXL immediately returns `ERR`.
- `ReadOperation.wait_for_complete()` polls `check_xfer_state()` and raises if
  the transfer enters `ERR`.
- vLLM catches checkpoint-engine update failures and returns `False` to the
  GRPO refit orchestration.
- GRPO raises a refit error when any generation worker reports failure.
- GRPO calls `finalize()` in a `finally` block to remove per-refit peer
  connections.

The backend does not currently rebuild the NIXL topology, recreate Ray actors,
or reload vLLM inside the same training step after a peer disappears. That
responsibility belongs to the scheduler or a fault-tolerant launcher that
restarts the training process from a durable NeMo RL checkpoint.

For production runs, configure UCX so peer failures are reported to NIXL:

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
          backend_name: UCX
          backend_init_params:
            ucx_error_handling_mode: peer
```

And use bounded UCX retry/keepalive settings:

```sh
export UCX_RC_TIMEOUT=30s
export UCX_RC_RETRY_COUNT=7
export UCX_KEEPALIVE_INTERVAL=1s
export UCX_KEEPALIVE_NUM_EPS=10
```

`ucx_error_handling_mode: none` should be reserved for performance experiments
on stable clusters. With peer error handling disabled, a dead endpoint may not
surface as a NIXL `ERR` state promptly enough for job-level restart logic.

## vLLM NIXL Preinit

vLLM starts internal worker processes during engine setup. For NIXL/UCX, the
backend needs to be initialized inside those internal workers before the normal
vLLM worker setup path finishes.

NeMo RL patches the vLLM internal worker constructor and injects a config-driven
preinit call when:

- `policy.generation.checkpoint_engine.enabled=true`
- `policy.generation.checkpoint_engine.backend=nixl`

The preinit call uses the configured NIXL `backend_name` and
`backend_init_params`; it does not require NeMo RL feature environment
variables. A healthy vLLM run prints:

```text
NIXL vLLM worker preinit completed: backend=UCX
```

Backends other than NIXL should initialize themselves through the normal
`CheckpointEngine` constructor unless they also need code to run in nested vLLM
worker processes before engine setup.

## Bucket Helpers

`split_weight_chunks()` converts the policy weight stream into byte chunks no
larger than the configured bucket size. It records `TensorMeta` for each chunk:

- original tensor name
- shape
- dtype
- chunk offset
- chunk size
- byte offset inside the transfer bucket

`merge_weight_chunk_batches()` reconstructs tensors that were split across
multiple chunks while preserving bucket boundaries for normal tensors. Backend
implementations can use these helpers when their transport operates on flat
byte buffers.

## Adding a New Backend

1. Implement a `CheckpointEngine` subclass.
2. Decide whether to register a short name or use a class path in config.
3. Make `prepare()` allocate/register buffers and return metadata that Ray can
   serialize.
4. Use `init_policy_process_group()` and `init_rollout_process_group()` to
   connect peers from the combined metadata list.
5. Implement `send_weights()` as a streaming send of `(name, tensor)` pairs.
6. Implement `receive_weight_batches()` as a streaming receive that yields
   loadable `(name, tensor)` batches.
7. Make `finalize()` release per-refit peer state without destroying reusable
   buffers unless the backend cannot safely reuse them.
8. Define the backend's failure behavior. Transfer errors should become explicit
   exceptions or `False` update results rather than silent partial updates.
9. Add unit tests for registry loading, metadata setup, topology, failure
   propagation, and a small tensor roundtrip.
10. Run a non-colocated GRPO job and verify the `[vLLM refit]` timing line.

Good starting tests are:

```sh
uv run pytest tests/unit/utils/test_checkpoint_engine.py
uv run pytest tests/unit/algorithms/test_grpo.py -k checkpoint_engine
```

## Compatibility Notes

- Checkpoint-engine refit currently targets non-colocated policy-to-vLLM refit.
- SGLang non-colocated checkpoint-engine refit is not implemented.
- The backend must be installed in every Ray worker environment that imports it.
- The backend must preserve parameter names exactly, because generation workers
  use those names to load weights into the target model.
- The backend should avoid driver-side model materialization. The driver should
  orchestrate futures and metadata only.

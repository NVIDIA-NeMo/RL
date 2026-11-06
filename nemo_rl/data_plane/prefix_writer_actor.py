# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Move prefix TQ serialization and PUTs into a colocated CPU Ray actor.

The generation pipeline retains transaction ownership, payload reservations and
write concurrency. Only immutable packed CPU columns cross the actor boundary;
no engine views, capture objects, CUDA tensors or per-token Python lists do.
The synchronous facade returns only after the remote TQ PUT acknowledges.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import ray
import torch
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from tensordict import TensorDict

from nemo_rl.utils.checkpoint_latency import CheckpointLatencyConfig, emit_latency
from nemo_rl.utils.prefix_cut_diagnostics import (
    current_prefix_cut_probe,
    prefix_cut_timing,
    prefix_write_batch_probe,
)

if TYPE_CHECKING:
    from nemo_rl.data_plane.tq_latency import TransferQueueLatencyHooks

G_DTYPES = {
    str(dtype): dtype
    for dtype in (
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    )
}


@dataclass(frozen=True)
class PackedColumn:
    """One dense values buffer, plus offsets for a contiguous jagged column."""

    data: np.ndarray
    dtype: str
    shape: tuple[int, ...]
    offsets: np.ndarray | None


@dataclass(frozen=True)
class PrefixPut:
    """Metadata and numeric buffers retained until the remote durability reply."""

    sample_ids: tuple[str, ...]
    partition_id: str
    columns: dict[str, PackedColumn]
    tags: tuple[dict[str, Any], ...]
    checkpoint_id: str
    batch_id: str
    submitted_unix_s: float

    @property
    def payload_bytes(self) -> int:
        return sum(
            column.data.nbytes
            + (0 if column.offsets is None else column.offsets.nbytes)
            for column in self.columns.values()
        )


def pack_columns(fields: TensorDict) -> dict[str, PackedColumn]:
    """Expose owned staging columns as byte arrays for Ray's buffer serializer.

    No per-token conversion or dtype narrowing occurs. The caller must keep
    fields immutable until its PUT returns; the object-store copy owns the
    transported buffers independently of the sender's tensors.
    """
    columns = {}
    for name, tensor in fields.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise TypeError("Prefix writer accepts only named tensor columns")
        if tensor.device.type != "cpu":
            raise ValueError("Prefix writer requires owned CPU staging tensors")
        if str(tensor.dtype) not in G_DTYPES:
            raise ValueError(f"Unsupported prefix dtype: {tensor.dtype}")
        offsets = None
        if tensor.is_nested:
            if tensor.layout != torch.jagged or tensor.lengths() is not None:
                raise ValueError("Prefix writer requires contiguous jagged columns")
            values = tensor.values()
            offsets = tensor.offsets().detach().numpy()
        else:
            values = tensor
        values = values.detach().contiguous()
        columns[name] = PackedColumn(
            data=values.reshape(-1).view(torch.uint8).numpy(),
            dtype=str(values.dtype),
            shape=tuple(values.shape),
            offsets=offsets,
        )
    return columns


def unpack_columns(columns: dict[str, PackedColumn], rows: int) -> TensorDict:
    """Reconstruct read-only staging views without copying the numeric payload.

    Ray's NumPy buffers can be read-only. TQ's schema extraction, selection and
    serialization read these tensors; they must never mutate received columns.
    """
    fields = {}
    for name, column in columns.items():
        if column.dtype not in G_DTYPES or any(n < 0 for n in column.shape):
            raise ValueError("Invalid packed prefix dtype or shape")
        dtype = G_DTYPES[column.dtype]
        expected = math.prod(column.shape) * torch.empty((), dtype=dtype).element_size()
        if (
            column.data.dtype != np.uint8
            or column.data.ndim != 1
            or not column.data.flags.c_contiguous
            or column.data.nbytes != expected
        ):
            raise ValueError("Packed prefix buffer does not match its shape")
        values = torch.from_numpy(column.data).view(dtype).reshape(column.shape)
        if column.offsets is None:
            if not column.shape or column.shape[0] != rows:
                raise ValueError("Dense prefix column has the wrong row count")
            fields[name] = values
        else:
            offsets = column.offsets
            if (
                not column.shape
                or offsets.dtype != np.int64
                or offsets.ndim != 1
                or len(offsets) != rows + 1
                or offsets[0] != 0
                or offsets[-1] != column.shape[0]
                or np.any(offsets[1:] < offsets[:-1])
            ):
                raise ValueError("Invalid packed prefix offsets")
            fields[name] = torch.nested.nested_tensor_from_jagged(
                values, offsets=torch.from_numpy(offsets)
            )
    return TensorDict(fields, batch_size=[rows])


class PrefixWriterService:
    """Thread-safe PUT service, also usable with a test data-plane client."""

    def __init__(self, client: Any, *, partition_id: str) -> None:
        self._client = client
        self._partition_id = partition_id

    def put(self, batch: PrefixPut) -> tuple[str, ...]:
        """Return IDs only after TQ acknowledges all submitted fields and tags."""
        if batch.partition_id != self._partition_id:
            raise ValueError("Prefix writer partition differs from capture partition")
        if len(batch.sample_ids) != len(batch.tags) or len(
            set(batch.sample_ids)
        ) != len(batch.sample_ids):
            raise ValueError("Prefix writer IDs and tags are inconsistent")
        started = time.time()
        with prefix_write_batch_probe(
            enabled=True,
            checkpoint_id=batch.checkpoint_id,
            batch_id=batch.batch_id,
            inventories=1,
            rows=len(batch.sample_ids),
        ):
            with prefix_cut_timing("writer_actor_unpack"):
                fields = unpack_columns(batch.columns, len(batch.sample_ids))
            with prefix_cut_timing("writer_actor_put"):
                self._client.put_samples(
                    sample_ids=list(batch.sample_ids),
                    partition_id=batch.partition_id,
                    fields=fields,
                    tags=list(batch.tags),
                )
        emit_latency(
            "prefix_writer_actor",
            checkpoint_id=batch.checkpoint_id,
            batch_id=batch.batch_id,
            event="durable",
            pid=os.getpid(),
            rows=len(batch.sample_ids),
            payload_bytes=batch.payload_bytes,
            dispatch_s=started - batch.submitted_unix_s,
            wall_s=time.time() - started,
        )
        return batch.sample_ids


class PrefixWriterActor(PrefixWriterService):
    """Own a process-local TQ client; Ray runs this service on a CPU node slot."""

    def __init__(
        self,
        config: Any,
        partition_id: str,
        latency: CheckpointLatencyConfig,
        torch_threads: int,
    ) -> None:
        # TQ is optional in CPU-only tools; connect only in the live actor.
        from nemo_rl.data_plane import build_data_plane_client
        from nemo_rl.data_plane.tq_latency import install_transfer_queue_latency

        torch.set_num_threads(torch_threads)
        client = build_data_plane_client(config, bootstrap=False, segment_size=0)
        super().__init__(client, partition_id=partition_id)
        self._hooks: TransferQueueLatencyHooks | None = None
        if latency.enabled and latency.tq_internal_phases:
            # Native TQ is an optional runtime dependency.
            import transfer_queue as tq

            self._hooks = install_transfer_queue_latency(
                tq.get_client(),
                max_events=latency.max_events,
                slow_call_s=latency.slow_call_s,
            )

    def ready(self) -> dict[str, Any]:
        return {"pid": os.getpid(), "node_id": ray.get_runtime_context().get_node_id()}


class RayPrefixWriterClient:
    """Blocking prefix-only facade; existing pipeline slots bound remote PUTs."""

    def __init__(self, actor: Any) -> None:
        self.actor = actor

    def put_samples(
        self,
        *,
        sample_ids: list[str],
        partition_id: str,
        fields: TensorDict,
        tags: list[dict[str, Any]],
    ) -> None:
        probe = current_prefix_cut_probe()
        checkpoint = "unmeasured" if probe is None else probe.checkpoint_id
        batch_id = "unmeasured" if probe is None else probe.inventory_digest
        with prefix_cut_timing("writer_actor_pack"):
            batch = PrefixPut(
                tuple(sample_ids),
                partition_id,
                pack_columns(fields),
                tuple(dict(tag) for tag in tags),
                checkpoint,
                batch_id,
                time.time(),
            )
        with prefix_cut_timing("writer_actor_handoff"):
            reference = ray.put(batch)
        try:
            with prefix_cut_timing("writer_actor_reply_wait"):
                acknowledged = ray.get(self.actor.put.remote(reference))
            if tuple(acknowledged) != batch.sample_ids:
                raise RuntimeError("Prefix writer returned different durable IDs")
        finally:
            # The pipeline still owns the original fields/reservation through seal.
            del reference

    def close(self) -> None:
        """Stop only after pipeline drain, or during failed worker teardown."""
        ray.kill(self.actor, no_restart=True)


async def create_prefix_writer(
    config: Any,
    *,
    partition_id: str,
    concurrency: int,
    latency: CheckpointLatencyConfig,
) -> RayPrefixWriterClient:
    """Start a non-restarting writer with hard affinity to this generation node."""
    if config["impl"] != "transfer_queue":
        raise ValueError("Prefix writer actor requires a cross-process TQ backend")
    node_id = ray.get_runtime_context().get_node_id()
    actor = (
        ray.remote(PrefixWriterActor)
        .options(
            num_cpus=1,
            num_gpus=0,
            max_concurrency=concurrency,
            max_restarts=0,
            max_task_retries=0,
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id, soft=False),
        )
        .remote(config, partition_id, latency, torch.get_num_threads())
    )
    try:
        ready = await actor.ready.remote()
        if ready["node_id"] != node_id:
            raise RuntimeError(
                "Prefix writer was scheduled away from its generation node"
            )
    except BaseException:
        ray.kill(actor, no_restart=True)
        raise
    emit_latency(
        "prefix_writer_actor", event="ready", writer_pid=ready["pid"], node_id=node_id
    )
    return RayPrefixWriterClient(actor)

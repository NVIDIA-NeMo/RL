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
"""Prefix-only TQ writes in an isolated, serial CPU Ray actor.

Cut transactions and encoding stay in the generation worker. Owned numeric
columns cross the actor boundary, and a caller receives its keys only after
TQ acknowledges the write. Terminal writes use their original local client.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import sys
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
import ray
import torch
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from tensordict import TensorDict

from nemo_rl.data_plane.interfaces import DataPlaneConfig
from nemo_rl.data_plane.observability import is_metrics_client
from nemo_rl.data_plane.schema import GENERATION_CUT_STAGING_PREFIX

G_LOGGER = logging.getLogger(__name__)
# A writer hard-pinned to a node with no free CPU would otherwise wait forever.
_WRITER_STARTUP_TIMEOUT_S = 180
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
    """A dense numeric buffer and optional contiguous-jagged row offsets."""

    data: np.ndarray
    dtype: str
    shape: tuple[int, ...]
    offsets: np.ndarray | None


@dataclass(frozen=True)
class PrefixPut:
    sample_ids: tuple[str, ...]
    partition_id: str
    columns: dict[str, PackedColumn]
    tags: tuple[dict[str, Any], ...]


def pack_columns(fields: TensorDict) -> dict[str, PackedColumn]:
    """Expose CPU buffers; the caller retains them unchanged through ray.put."""
    columns = {}
    for name, tensor in fields.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise TypeError("Prefix writer requires named tensor columns")
        if tensor.device.type != "cpu" or str(tensor.dtype) not in G_DTYPES:
            raise ValueError("Prefix writer requires supported CPU tensor dtypes")
        offsets = None
        if tensor.is_nested:
            if tensor.layout != torch.jagged or tensor.lengths() is not None:
                raise ValueError("Prefix writer requires contiguous jagged columns")
            offsets = tensor.offsets().numpy(force=True)
            values = tensor.values()
        else:
            values = tensor
        if values.dtype == torch.bfloat16:
            # NumPy has no native bfloat16 dtype; retain its exact storage bits.
            data = values.detach().contiguous().reshape(-1).view(torch.uint8).numpy()
        else:
            # NumPy views avoid repeated Torch dispatch/GIL handoffs per column.
            data = (
                np.ascontiguousarray(values.numpy(force=True))
                .reshape(-1)
                .view(np.uint8)
            )
        columns[name] = PackedColumn(
            data,
            str(values.dtype),
            tuple(values.shape),
            offsets,
        )
    return columns


def unpack_columns(columns: dict[str, PackedColumn], rows: int) -> TensorDict:
    """Rebuild read-only tensor views of Ray-owned buffers for TQ serialization."""
    fields = {}
    for name, column in columns.items():
        if column.dtype not in G_DTYPES or any(size < 0 for size in column.shape):
            raise ValueError("Invalid prefix dtype or shape")
        dtype = G_DTYPES[column.dtype]
        expected = math.prod(column.shape) * torch.empty((), dtype=dtype).element_size()
        if (
            column.data.dtype != np.uint8
            or column.data.ndim != 1
            or not column.data.flags.c_contiguous
            or column.data.nbytes != expected
        ):
            raise ValueError("Prefix buffer does not match its shape")
        # Ray's read-only arrays are safe: this service only reads its columns.
        values = (
            torch.empty(column.shape, dtype=dtype)
            if expected == 0
            else torch.from_numpy(column.data).view(dtype).reshape(column.shape)
        )
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
                raise ValueError("Invalid prefix row offsets")
            fields[name] = torch.nested.nested_tensor_from_jagged(
                values, offsets=torch.from_numpy(offsets)
            )
    return TensorDict(fields, batch_size=[rows])


class PrefixWriterService:
    """Execute one PUT or failed-write cleanup at a time in the writer actor."""

    def __init__(self, client: Any, partition_id: str) -> None:
        self._client = client
        self._partition_id = partition_id

    def _validate_keys(self, partition_id: str, sample_ids: tuple[str, ...]) -> None:
        if partition_id != self._partition_id:
            raise ValueError("Prefix writer partition mismatch")
        if (
            not sample_ids
            or len(set(sample_ids)) != len(sample_ids)
            or any(
                not key.startswith(GENERATION_CUT_STAGING_PREFIX) for key in sample_ids
            )
        ):
            raise ValueError("Prefix writer requires distinct generation-cut keys")

    def put(self, batch: PrefixPut) -> tuple[str, ...]:
        """Write one batch and return its keys once TQ acknowledges it."""
        self._validate_keys(batch.partition_id, batch.sample_ids)
        if len(batch.tags) != len(batch.sample_ids):
            raise ValueError("Prefix IDs and tags have different lengths")
        self._client.put_samples(
            sample_ids=list(batch.sample_ids),
            partition_id=batch.partition_id,
            fields=unpack_columns(batch.columns, len(batch.sample_ids)),
            tags=list(batch.tags),
        )
        return batch.sample_ids

    def clear(self, partition_id: str, sample_ids: tuple[str, ...]) -> None:
        self._validate_keys(partition_id, sample_ids)
        self._client.clear_samples(
            sample_ids=list(sample_ids), partition_id=partition_id
        )

    def get_data_plane_snapshot(self) -> dict[str, Any] | None:
        if not is_metrics_client(self._client):
            return None
        return self._client.snapshot(reset_step_window=True)


class PrefixWriterActor(PrefixWriterService):
    """Own the process-local TQ client, with no engine or GPU allocation."""

    def __init__(self, config: DataPlaneConfig, partition_id: str) -> None:
        # TQ is optional outside a live data-plane deployment.
        from nemo_rl.data_plane import build_data_plane_client

        torch.set_num_threads(1)
        super().__init__(
            build_data_plane_client(config, bootstrap=False),
            partition_id,
        )

    def ready(self) -> dict[str, Any]:
        return {"pid": os.getpid(), "node_id": ray.get_runtime_context().get_node_id()}


class PrefixWriterDiedError(RuntimeError):
    """The writer actor is gone; it is never restarted."""


class RayPrefixWriterClient:
    """Synchronous, one-flight facade; no automatic retries or early ACKs.

    Close is teardown-only: it kills the actor rather than waiting forever on
    a stalled PUT. Outstanding callers fail; they must not publish receipts.
    Once the actor dies, every later call raises ``PrefixWriterDiedError``
    rather than falling back to another client.
    """

    def __init__(self, actor: Any, *, writer_pid: int) -> None:
        self._actor = actor
        self._writer_pid = writer_pid
        self._write_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._closed = False
        self._dead = False

    def _require_open(self) -> None:
        with self._state_lock:
            if self._closed:
                raise RuntimeError("Prefix writer is closed")
            if self._dead:
                raise self._died_error()

    def _died_error(self) -> PrefixWriterDiedError:
        return PrefixWriterDiedError(
            f"Prefix writer actor (pid {self._writer_pid}) died; generation-prefix "
            "cuts on this engine fail until the run restarts"
        )

    def _get(self, reference: Any) -> Any:
        """Resolve one actor call, marking the writer dead if the actor is gone."""
        try:
            return ray.get(reference)
        except ray.exceptions.RayActorError as error:
            # A close kills the actor on purpose; report that, not a death.
            self._require_open()
            with self._state_lock:
                self._dead = True
            raise self._died_error() from error

    def put_samples(
        self,
        *,
        sample_ids: list[str],
        partition_id: str,
        fields: TensorDict,
        tags: list[dict[str, Any]],
    ) -> None:
        with self._write_lock:
            self._require_open()
            batch = PrefixPut(
                tuple(sample_ids),
                partition_id,
                pack_columns(fields),
                tuple(dict(tag) for tag in tags),
            )
            keys = self._get(self._actor.put.remote(ray.put(batch)))
            self._require_open()
            if keys != batch.sample_ids:
                raise RuntimeError("Prefix writer returned different durable keys")

    def clear_samples(self, *, sample_ids: list[str], partition_id: str) -> None:
        with self._write_lock:
            self._require_open()
            self._get(self._actor.clear.remote(partition_id, tuple(sample_ids)))

    async def get_data_plane_snapshot(self) -> dict[str, Any] | None:
        """The writer client's data-plane counters.

        ``None`` when observability is off or the writer is closed or dead; the
        cuts report a dead writer. Waits behind an in-flight PUT, since the
        actor runs one call at a time.
        """
        with self._state_lock:
            if self._closed or self._dead:
                return None
        try:
            return await self._actor.get_data_plane_snapshot.remote()
        except ray.exceptions.RayActorError:
            with self._state_lock:
                self._dead = True
            return None

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
        ray.kill(self._actor, no_restart=True)
        G_LOGGER.info("prefix_writer event=closed writer_pid=%d", self._writer_pid)


async def create_prefix_writer(
    config: DataPlaneConfig, *, partition_id: str
) -> RayPrefixWriterClient:
    """Create one non-restarting, serial CPU writer on the generation node."""
    node_id = ray.get_runtime_context().get_node_id()
    actor = (
        ray.remote(PrefixWriterActor)
        .options(
            num_cpus=1,
            num_gpus=0,
            max_concurrency=1,
            max_restarts=0,
            max_task_retries=0,
            runtime_env={"py_executable": sys.executable},
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id, soft=False),
        )
        .remote(config, partition_id)
    )
    try:
        ready = await asyncio.wait_for(
            actor.ready.remote(), timeout=_WRITER_STARTUP_TIMEOUT_S
        )
        if ready["node_id"] != node_id:
            raise RuntimeError("Prefix writer was scheduled off its generation node")
    except TimeoutError as e:
        ray.kill(actor, no_restart=True)
        raise RuntimeError(
            f"Prefix writer did not start on node {node_id} within "
            f"{_WRITER_STARTUP_TIMEOUT_S}s; it needs 1 free CPU there"
        ) from e
    except BaseException:
        ray.kill(actor, no_restart=True)
        raise
    G_LOGGER.info(
        "prefix_writer event=ready writer_pid=%d node_id=%s", ready["pid"], node_id
    )
    return RayPrefixWriterClient(actor, writer_pid=ready["pid"])

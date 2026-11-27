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
"""Generation-cut rows built and written in an isolated, serial CPU Ray actor.

Cut transactions stay in the generation worker. Each frozen call's inputs
cross the actor boundary, its attachments as NumPy buffers; the actor builds
Gym's records, encodes and writes them, and a caller receives the keys only
after TQ acknowledges the write. Terminal writes use their original local client.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import threading
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import numpy as np
import ray
import torch
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from nemo_rl.data_plane.generation_cut_rows import (
    GenerationCutRow,
    StagedGenerationCut,
)
from nemo_rl.data_plane.interfaces import DataPlaneConfig
from nemo_rl.data_plane.observability import is_metrics_client

G_LOGGER = logging.getLogger(__name__)
# A writer hard-pinned to a node with no free CPU would otherwise wait forever.
_WRITER_STARTUP_TIMEOUT_S = 180


def to_bytes(
    tensor: torch.Tensor,
) -> tuple[np.ndarray, str, tuple[int, ...], np.ndarray | None]:
    """Expose a CPU column as raw bytes, plus its dtype, shape and jagged offsets.

    Ray pickles a tensor by copying its data; NumPy arrays travel as separate
    buffers the actor reads from shared memory without a copy.
    """
    values, offsets = (
        (tensor.values(), tensor.offsets()) if tensor.is_nested else (tensor, None)
    )
    data = values.detach().contiguous().reshape(-1).view(torch.uint8).numpy()
    return (
        data,
        str(values.dtype),
        tuple(values.shape),
        None if offsets is None else offsets.numpy(),
    )


def from_bytes(
    data: np.ndarray, dtype: str, shape: tuple[int, ...], offsets: np.ndarray | None
) -> torch.Tensor:
    """Rebuild a column from ``to_bytes`` as a view of Ray's buffer."""
    torch_dtype = getattr(torch, dtype.removeprefix("torch."))
    # An empty buffer cannot be viewed as a wider dtype.
    values = (
        torch.from_numpy(data).view(torch_dtype).reshape(shape)
        if data.size
        else torch.empty(shape, dtype=torch_dtype)
    )
    if offsets is None:
        return values
    return torch.nested.nested_tensor_from_jagged(
        values, offsets=torch.from_numpy(offsets)
    )


class PrefixWriterService:
    """Build and write one batch of generation-cut rows at a time in the writer actor."""

    def __init__(self, client: Any) -> None:
        self._client = client
        self._sink: Any = None

    def configure_generation_cut_sink(
        self,
        *,
        staging_partition: str,
        capture_media: bool,
        media_pixel_dtype: torch.dtype | None,
    ) -> None:
        """Set up the sink cut rows are written with, matching the worker's."""
        # Deferred: tq_token_sink imports this module.
        from nemo_rl.data_plane.tq_token_sink import TQTokenSink

        self._sink = TQTokenSink(
            self._client,
            staging_partition=staging_partition,
            capture_media=capture_media,
            media_pixel_dtype=media_pixel_dtype,
        )

    def stage_generation_cut_rows(
        self,
        rows: list[GenerationCutRow],
        *,
        checkpoint_id: str,
        attachments: list[dict[str, tuple[Any, ...]] | None],
    ) -> list[StagedGenerationCut]:
        """Build and write one batch; returns once TQ acknowledges it.

        ``attachments`` are the rows' attachments as ``to_bytes`` columns.
        """
        if self._sink is None:
            raise RuntimeError("The generation-cut sink is not configured")
        rows = [
            replace(
                row,
                attachments=None
                if columns is None
                else {name: from_bytes(*column) for name, column in columns.items()},
            )
            for row, columns in zip(rows, attachments, strict=True)
        ]
        return self._sink.stage_generation_cut_rows(rows, checkpoint_id=checkpoint_id)

    def get_data_plane_snapshot(self) -> dict[str, Any] | None:
        if not is_metrics_client(self._client):
            return None
        return self._client.snapshot(reset_step_window=True)


class PrefixWriterActor(PrefixWriterService):
    """Own the process-local TQ client, with no engine or GPU allocation."""

    def __init__(self, config: DataPlaneConfig) -> None:
        # TQ is optional outside a live data-plane deployment.
        from nemo_rl.data_plane import build_data_plane_client

        torch.set_num_threads(1)
        # Own no segment (mooncake_cpu only; ignored elsewhere): a save would
        # otherwise wait on this actor, as on the worker's capture client.
        super().__init__(
            build_data_plane_client(config, bootstrap=False, segment_size=0)
        )

    def ready(self) -> dict[str, Any]:
        return {"pid": os.getpid(), "node_id": ray.get_runtime_context().get_node_id()}


class PrefixWriterDiedError(RuntimeError):
    """The writer actor is gone; it is never restarted."""


class RayPrefixWriterClient:
    """The sink's prefix client, forwarding each call to the writer actor.

    The actor options run one call at a time with no retries, and killing the
    actor fails its pending call, so nothing is acknowledged early. Close is
    teardown-only. Once the actor dies, every later call raises
    ``PrefixWriterDiedError`` rather than falling back to another client.
    """

    def __init__(self, actor: Any, *, writer_pid: int) -> None:
        self._actor = actor
        self._writer_pid = writer_pid
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

    def configure_generation_cut_sink(
        self,
        *,
        staging_partition: str,
        capture_media: bool,
        media_pixel_dtype: torch.dtype | None,
    ) -> None:
        """Give the actor a sink with the worker sink's partition and media settings."""
        self._require_open()
        self._get(
            self._actor.configure_generation_cut_sink.remote(
                staging_partition=staging_partition,
                capture_media=capture_media,
                media_pixel_dtype=media_pixel_dtype,
            )
        )

    def stage_generation_cut_rows(
        self, rows: Sequence[GenerationCutRow], *, checkpoint_id: str
    ) -> list[StagedGenerationCut]:
        """Have the actor build and write one batch of cut rows."""
        self._require_open()
        attachments = [
            None
            if row.attachments is None
            else {name: to_bytes(tensor) for name, tensor in row.attachments.items()}
            for row in rows
        ]
        return self._get(
            self._actor.stage_generation_cut_rows.remote(
                [replace(row, attachments=None) for row in rows],
                checkpoint_id=checkpoint_id,
                attachments=attachments,
            )
        )

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


async def create_prefix_writer(config: DataPlaneConfig) -> RayPrefixWriterClient:
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
        .remote(config)
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

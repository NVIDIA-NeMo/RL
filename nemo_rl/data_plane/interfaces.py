# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""Stable boundary between NeMo-RL and data-plane implementations.

Wire shape adapters must support:
  * ``fields``: ``TensorDict`` with tensor leaves AND optional
    ``NonTensorStack`` / ``NonTensorData`` leaves (TQ-native non-tensor
    passthrough). TQ's storage backends handle encoding per backend
    (simple keeps Python objects; mooncake_client pickles internally).
  * ``tags``: ``list[dict[str, Any]]`` per-sample primitives (kept
    separate from ``fields`` so non-tensor metadata like
    ``input_lengths`` doesn't pollute the leaf-level schema).
  * ``keys``: per-sample string uids.
  * ``partition_id``: string-named address spaces with declared
    ``consumer_tasks`` and ``fields`` schemas.

All call sites in ``nemo_rl/algorithms``, ``nemo_rl/experience`` and
``nemo_rl/models`` go through :class:`DataPlaneClient` — never
``import transfer_queue`` directly. This is what makes the
implementation swappable.

See ``nemo_rl/data_plane/README.md`` for the full design.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, NotRequired, Sequence, TypedDict

from tensordict import TensorDict


class DataPlaneConfig(TypedDict):
    """Feature-gated config; defaults to disabled.

    ``backend`` is the storage backend *inside* TransferQueue; it is owned by
    the TQ adapter, not by NeMo-RL. ``impl`` selects which adapter we go
    through.
    """

    enabled: bool
    impl: Literal["transfer_queue"]
    backend: NotRequired[Literal["simple", "mooncake_cpu"]]
    controller_address: NotRequired[str]
    storage_capacity: NotRequired[int]
    num_storage_units: NotRequired[int]
    claim_meta_poll_interval_s: NotRequired[float]
    ack_timeout_ms: NotRequired[int]
    observability: NotRequired["ObservabilityConfig"]


class ObservabilityConfig(TypedDict):
    """Optional middleware that records per-op metrics on the client.

    Off by default. When ``enabled=True`` the factory wraps the chosen
    adapter with :class:`MetricsDataPlaneClient`. ``callback`` is
    injected programmatically (callables don't round-trip through
    YAML) — set ``cfg["observability"]["callback"] = my_fn`` before
    :func:`build_data_plane_client` to plug into wandb / file / log.
    Default callback prints one line per op for debug.
    """

    enabled: bool
    callback: NotRequired[Callable[[dict[str, Any]], None]]


@dataclass
class KVBatchMeta:
    """1:1 mirror of ``transfer_queue.metadata.KVBatchMeta``.

    Attribute names match TransferQueue exactly so the adapter does not need
    a rename layer and TQ's own ``select_fields`` validation works against
    our object unmodified.

    Two roles:
      * Result type returned by :meth:`DataPlaneClient.claim_meta` — callers
        extract ``.keys`` / ``.partition_id`` and pass them to
        :meth:`kv_batch_get` / :meth:`get_data`.
      * Argument type for the per-DP-rank fetch entrypoints.
        ``sequence_lengths`` lets the driver compute a balanced per-rank
        shard from metadata only (control plane), without ever
        materializing tensor data.
    """

    partition_id: str
    task_name: str | None
    keys: list[str]
    fields: list[str] | None = None
    sequence_lengths: list[int] | None = None
    extra_info: dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.keys)

    # ── Pure-metadata transforms (no I/O) ──────────────────────────────
    # Used by dynamic_sampling on the meta path: filter zero-std rows
    # (subset), accumulate survivors across iterations (concat), trim
    # an over-full cache to the training batch size (slice). Each
    # returns a fresh KVBatchMeta — caller is responsible for kv_clear-
    # ing any uids dropped from the working set.

    def _replace(
        self,
        *,
        keys: list[str],
        sequence_lengths: list[int] | None,
    ) -> "KVBatchMeta":
        """Return a copy with new keys/sequence_lengths, same metadata otherwise."""
        return KVBatchMeta(
            partition_id=self.partition_id,
            task_name=self.task_name,
            keys=list(keys),
            fields=self.fields,
            sequence_lengths=list(sequence_lengths)
            if sequence_lengths is not None
            else None,
            extra_info=dict(self.extra_info or {}),
        )

    def subset(self, indices: "Sequence[int]") -> "KVBatchMeta":
        """Return a new meta with only the rows at ``indices`` (any order)."""
        return self._replace(
            keys=[self.keys[i] for i in indices],
            sequence_lengths=(
                [self.sequence_lengths[i] for i in indices]
                if self.sequence_lengths is not None
                else None
            ),
        )

    def slice(self, start: int, stop: int) -> "KVBatchMeta":
        """Return a new meta with rows in the contiguous range ``[start, stop)``."""
        return self._replace(
            keys=self.keys[start:stop],
            sequence_lengths=(
                self.sequence_lengths[start:stop]
                if self.sequence_lengths is not None
                else None
            ),
        )

    def concat(self, *others: "KVBatchMeta") -> "KVBatchMeta":
        """Append ``others`` to ``self``. All metas must share ``partition_id``."""
        if any(o.partition_id != self.partition_id for o in others):
            raise ValueError("KVBatchMeta.concat: partition_ids must match")
        all_m = (self, *others)
        keys = [k for m in all_m for k in m.keys]
        all_have_lens = all(m.sequence_lengths is not None for m in all_m)
        seq_lens = (
            [s for m in all_m for s in (m.sequence_lengths or [])]
            if all_have_lens
            else None
        )
        return self._replace(keys=keys, sequence_lengths=seq_lens)


class DataPlaneClient(ABC):
    """Stable, swappable data-plane boundary.

    The methods are split into three groups by intent. Argument order
    mirrors the underlying ``transfer_queue`` API 1:1 so a future adapter
    (e.g. ``nv-dataplane``) is a thin pass-through too.

    A. *Task-mediated* — used by stages that wait for upstream production
       via the per-task consumer counter:
       :meth:`register_partition`, :meth:`claim_meta`, :meth:`get_data`,
       :meth:`check_consumption_status`.
    B. *Direct-by-key* — used by stages that already know the exact uids
       (e.g. driver-side fan-out to DP ranks):
       :meth:`kv_batch_put`, :meth:`kv_batch_get`, :meth:`kv_clear`.
    C. *Lifecycle* — :meth:`close`.

    Stage-completion signal: there is intentionally no ``mark_consumed``.
    The authoritative signal in TransferQueue is *field production* —
    when a stage calls :meth:`kv_batch_put` for a new field, the controller
    flips ``production_status[sample, field] = 1``. Downstream consumers
    waiting on that field only see those samples once produced.
    """

    # ── (A) task-mediated ───────────────────────────────────────────────

    @abstractmethod
    def register_partition(
        self,
        partition_id: str,
        fields: list[str],
        num_samples: int,
        consumer_tasks: list[str],
        grpo_group_size: int | None = None,
        enums: dict[str, list[str]] | None = None,
    ) -> None:
        """Declare the partition schema and consumer tasks.

        Args:
            partition_id: Partition name.
            fields: Superset of fields any producer may write here.
            num_samples: Expected total samples; sizes controller arrays.
            consumer_tasks: Named tasks; each gets its own consumption cursor.
            grpo_group_size: Group size for GRPO balanced sampling.
            enums: Per-field fixed-vocab string codec, shipped once at register.
        """

    @abstractmethod
    def claim_meta(
        self,
        partition_id: str,
        task_name: str,
        required_fields: list[str],
        batch_size: int,
        dp_rank: int | None = None,
        blocking: bool = True,
        timeout_s: float = 60.0,
    ) -> KVBatchMeta:
        """Discover and **claim** up to ``batch_size`` ready samples.

        Advances ``task_name``'s per-sample consumption cursor (TQ's
        ``mode='fetch'``); claimed uids won't be returned again. Samples
        stay readable via :meth:`kv_batch_get` until :meth:`kv_clear`.

        Args:
            partition_id: Partition to claim from.
            task_name: Consumer task whose cursor is advanced.
            required_fields: Fields that must be produced for a sample to be claimable.
            batch_size: Max samples to claim.
            dp_rank: Reserved; driver-side balancing via :func:`shard_meta_for_dp` is used today.
            blocking: Block until the batch can be claimed.
            timeout_s: Max blocking time before raising.

        Returns:
            ``KVBatchMeta`` for the claimed batch; pass to :meth:`get_data`.
        """

    @abstractmethod
    def get_data(
        self,
        meta: KVBatchMeta,
        select_fields: list[str] | None = None,
    ) -> TensorDict:
        """Resolve a meta to tensor data.

        Field-set resolution: (1) explicit ``select_fields``; (2)
        ``meta.fields`` if non-None; (3) *fail loudly* — never silently
        fetch all fields.

        Args:
            meta: From :meth:`claim_meta` or hand-built with explicit keys.
            select_fields: Subset of fields to fetch.

        Returns:
            ``TensorDict`` keyed by field name, batched along ``meta.keys``.
        """

    @abstractmethod
    def check_consumption_status(
        self, partition_id: str, task_names: list[str]
    ) -> bool:
        """True iff every task has consumed all samples in the partition.

        Authoritative across workers — uses TQ's controller-side counter,
        not the per-process client cache.

        Args:
            partition_id: Partition to check.
            task_names: Tasks whose consumption cursors are inspected.

        Returns:
            ``True`` iff every task in ``task_names`` has consumed all samples.
        """

    # ── (B) direct-by-key (TQ-aligned signatures) ──────────────────────

    @abstractmethod
    def kv_batch_put(
        self,
        keys: list[str],
        partition_id: str,
        fields: TensorDict | None = None,
        tags: list[dict[str, Any]] | None = None,
    ) -> KVBatchMeta:
        """Write fields for ``keys`` — the producer entrypoint.

        Writing a field flips the controller's ``production_status`` bit
        for ``(sample, field)``; that flip is the "stage finished" signal
        downstream consumers wait on. Tensor and ``NonTensorStack`` leaves
        both pass through to TQ; non-tensor encoding is per-backend.

        Args:
            keys: Per-sample uids being written.
            partition_id: Partition these keys belong to.
            fields: Tensor / ``NonTensorStack`` leaves to write.
            tags: Optional per-sample primitive metadata.

        Returns:
            ``KVBatchMeta`` covering ``keys`` — usable for direct :meth:`kv_batch_get`.
        """

    @abstractmethod
    def kv_batch_get(
        self,
        keys: list[str],
        partition_id: str,
        select_fields: list[str],
    ) -> TensorDict:
        """Direct fetch by uids.

        Used by per-DP-rank slice fetches. Does NOT advance any per-task
        consumption cursor — that only happens via :meth:`claim_meta`.

        ``select_fields`` is required (no implicit "fetch every field"
        fallback): bulk schemas are wide and silent over-fetch is the
        most expensive shape the wire can take. Callers must name what
        they read.

        Args:
            keys: Uids to fetch.
            partition_id: Partition the keys live in.
            select_fields: Subset of fields to fetch.

        Returns:
            ``TensorDict`` keyed by field name, batched along ``keys``.
        """

    @abstractmethod
    def kv_clear(
        self,
        keys: list[str] | None,
        partition_id: str,
    ) -> None:
        """Drop key-value pairs.

        Args:
            keys: Uids to drop; ``None`` clears the whole partition.
            partition_id: Partition the keys live in.
        """

    # ── (C) lifecycle ──────────────────────────────────────────────────

    @abstractmethod
    def close(self) -> None:
        """Release controller / storage handles. Idempotent."""

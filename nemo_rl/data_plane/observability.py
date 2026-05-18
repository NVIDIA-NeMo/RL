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
"""Lean per-op metrics decorator for ``DataPlaneClient``.

Wraps any ``DataPlaneClient`` and invokes a single user-provided
callback on each operation. Each event is a flat dict::

    {"op", "partition_id", "n_keys", "n_bytes", "wall_ms", "status"}

Plug wandb / file logging / debug print at the call site by passing
``on_event=<your function>``. ``snapshot()`` returns cumulative
totals **plus** live memory consumption: ``bytes_outstanding`` (sum of
bytes currently held in TQ, i.e. put minus cleared) and
``peak_bytes_outstanding`` (high-water mark over the run lifetime).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from time import monotonic
from typing import Any, Callable, Literal, TypedDict

EventStatus = Literal["ok", "error", "timeout"]


class DataPlaneEvent(TypedDict):
    op: str
    partition_id: str
    n_keys: int
    n_bytes: int
    wall_ms: float
    status: EventStatus


import torch
from tensordict import TensorDict

from nemo_rl.data_plane.interfaces import DataPlaneClient, KVBatchMeta

logger = logging.getLogger(__name__)


def _td_bytes(td: TensorDict | None) -> int:
    if td is None:
        return 0
    total = 0
    for k in td.keys(include_nested=True, leaves_only=True):
        v = td.get(k)
        if not isinstance(v, torch.Tensor):
            continue
        t = v.values() if v.is_nested else v
        total += t.numel() * t.element_size()
    return total


def log_event(event: DataPlaneEvent) -> None:
    logger.info("data_plane_event: %s", event)


@dataclass
class DataPlaneStats:
    total_bytes: int = 0
    total_keys: int = 0
    total_ops: int = 0
    bytes_outstanding: int = 0
    peak_bytes_outstanding: int = 0
    # Anomaly trackers — a wire-format regression that bloats bytes per
    # row (cf. message_log view-aliasing pickle bug) shows up as a
    # sudden spike in ``max_bytes_per_key_seen``.
    max_bytes_per_key_seen: int = 0
    last_put_bytes_per_key: int = 0


class MetricsDataPlaneClient(DataPlaneClient):
    """Wrap a ``DataPlaneClient`` with a per-op callback hook."""

    def __init__(
        self,
        inner: DataPlaneClient,
        on_event: Callable[[DataPlaneEvent], None] | None = None,
    ) -> None:
        self._inner = inner
        self._on_event = on_event or (lambda _: None)
        self._stats = DataPlaneStats()
        # Nested per-partition / per-key live byte counts. Populated on
        # successful ``put_samples``; popped on successful ``clear_samples``.
        # Bounded by the live key population, not cumulative traffic.
        self._bytes_by_partition: dict[str, dict[str, int]] = {}

    def snapshot(self) -> dict[str, Any]:
        """Return cumulative totals plus live byte / key outstanding counts."""
        out = asdict(self._stats)
        out["n_keys_outstanding"] = sum(
            len(d) for d in self._bytes_by_partition.values()
        )
        return out

    def bytes_outstanding_by_partition(self) -> dict[str, int]:
        """Per-partition breakdown of currently-held bytes."""
        return {p: sum(d.values()) for p, d in self._bytes_by_partition.items()}

    def _record_put(self, partition_id: str, keys: list[str], n_bytes: int) -> None:
        """Attribute put bytes per key so a later ``clear_samples`` can subtract.

        Called after the underlying RPC succeeds so a failed put never
        leaves the accounting inflated.

        Args:
            partition_id: Partition the keys were written to.
            keys: Per-sample uids that were written.
            n_bytes: Total bytes written; distributed evenly across keys.
        """
        if not keys or n_bytes <= 0:
            return
        per_key, remainder = divmod(n_bytes, len(keys))
        partition_dict = self._bytes_by_partition.setdefault(partition_id, {})
        for i, key in enumerate(keys):
            share = per_key + (1 if i < remainder else 0)
            partition_dict[key] = partition_dict.get(key, 0) + share
        self._stats.bytes_outstanding += n_bytes
        if self._stats.bytes_outstanding > self._stats.peak_bytes_outstanding:
            self._stats.peak_bytes_outstanding = self._stats.bytes_outstanding

    def _record_clear(self, partition_id: str, keys: list[str] | None) -> None:
        """Reverse the put accounting for ``keys``.

        Called after the underlying RPC succeeds so a failed clear keeps
        the accounting consistent with TQ's actual state.

        Args:
            partition_id: Partition the keys were dropped from.
            keys: Uids dropped; ``None`` means the whole partition was cleared.
        """
        partition_dict = self._bytes_by_partition.get(partition_id)
        if partition_dict is None:
            return
        if keys is None:
            freed = sum(partition_dict.values())
            del self._bytes_by_partition[partition_id]
        else:
            freed = 0
            for key in keys:
                freed += partition_dict.pop(key, 0)
            if not partition_dict:
                del self._bytes_by_partition[partition_id]
        self._stats.bytes_outstanding -= freed

    def _run(
        self,
        op: str,
        partition_id: str,
        fn: Callable[[], Any],
        *,
        n_keys: int = 0,
        n_bytes: int = 0,
    ) -> Any:
        """Run ``fn`` and emit one observability event with wall-time and status.

        Args:
            op: Operation tag (``"put"``, ``"get"``, ``"clear"``, etc.).
            partition_id: Partition the op targets.
            fn: Zero-arg callable that invokes the inner client.
            n_keys: Key count if known up front; otherwise inferred from
                the return value (``KVBatchMeta.sample_ids``).
            n_bytes: Byte estimate; overridden by ``_td_bytes`` when the
                return is a ``TensorDict``.

        Returns:
            Whatever ``fn`` returned.
        """
        t0 = monotonic()
        try:
            out = fn()
        except TimeoutError:
            self._emit(op, partition_id, n_keys, n_bytes, t0, "timeout")
            raise
        except Exception:
            self._emit(op, partition_id, n_keys, n_bytes, t0, "error")
            raise
        # If the call returns a TensorDict, the read-side bytes are more
        # informative than the input estimate.
        if isinstance(out, TensorDict):
            n_bytes = _td_bytes(out)
        elif isinstance(out, KVBatchMeta) and not n_keys:
            n_keys = len(out.sample_ids)
        self._emit(op, partition_id, n_keys, n_bytes, t0, "ok")
        return out

    def _emit(
        self,
        op: str,
        partition_id: str,
        n_keys: int,
        n_bytes: int,
        t0: float,
        status: EventStatus,
    ) -> None:
        event: DataPlaneEvent = {
            "op": op,
            "partition_id": partition_id,
            "n_keys": int(n_keys),
            "n_bytes": int(n_bytes),
            "wall_ms": (monotonic() - t0) * 1000.0,
            "status": status,
        }
        self._on_event(event)
        if status == "ok":
            self._stats.total_bytes += n_bytes
            self._stats.total_keys += n_keys
            self._stats.total_ops += 1
            if op == "put" and n_keys:
                per_key = n_bytes // n_keys
                self._stats.last_put_bytes_per_key = per_key
                if per_key > self._stats.max_bytes_per_key_seen:
                    self._stats.max_bytes_per_key_seen = per_key

    def register_partition(
        self,
        partition_id,
        fields,
        num_samples,
        consumer_tasks,
        grpo_group_size=None,
        enums=None,
    ):
        self._run(
            "register",
            partition_id,
            lambda: self._inner.register_partition(
                partition_id,
                fields,
                num_samples,
                consumer_tasks,
                grpo_group_size=grpo_group_size,
                enums=enums,
            ),
            n_keys=int(num_samples),
        )

    def claim_meta(
        self,
        partition_id,
        task_name,
        required_fields,
        batch_size,
        dp_rank=None,
        blocking=True,
        timeout_s=60.0,
    ):
        return self._run(
            "claim_meta",
            partition_id,
            lambda: self._inner.claim_meta(
                partition_id,
                task_name,
                required_fields,
                batch_size,
                dp_rank=dp_rank,
                blocking=blocking,
                timeout_s=timeout_s,
            ),
        )

    def get_data(self, meta, select_fields=None):
        return self._run(
            "get_data",
            meta.partition_id,
            lambda: self._inner.get_data(meta, select_fields=select_fields),
            n_keys=len(meta.sample_ids),
        )

    def check_consumption_status(self, partition_id, task_names):
        return self._run(
            "check_consumption_status",
            partition_id,
            lambda: self._inner.check_consumption_status(partition_id, task_names),
        )

    def put_samples(self, sample_ids, partition_id, fields=None, tags=None):
        n_bytes = _td_bytes(fields)
        # Materialize once: ``_run`` consumes its lambda and we also need
        # to attribute bytes per sample after success.
        sample_ids_list = sample_ids if isinstance(sample_ids, list) else list(sample_ids)
        out = self._run(
            "put",
            partition_id,
            lambda: self._inner.put_samples(
                sample_ids_list,
                partition_id,
                fields=fields,
                tags=tags,
            ),
            n_keys=len(sample_ids_list),
            n_bytes=n_bytes,
        )
        self._record_put(partition_id, sample_ids_list, n_bytes)
        return out

    def get_samples(self, sample_ids, partition_id, select_fields):
        return self._run(
            "get",
            partition_id,
            lambda: self._inner.get_samples(
                sample_ids,
                partition_id,
                select_fields=select_fields,
            ),
            n_keys=len(sample_ids),
        )

    def clear_samples(self, sample_ids, partition_id):
        sample_ids_list = (
            sample_ids
            if (sample_ids is None or isinstance(sample_ids, list))
            else list(sample_ids)
        )
        n_keys = len(sample_ids_list) if sample_ids_list is not None else 0
        self._run(
            "clear",
            partition_id,
            lambda: self._inner.clear_samples(sample_ids_list, partition_id),
            n_keys=n_keys,
        )
        self._record_clear(partition_id, sample_ids_list)

    def close(self) -> None:
        self._run(
            "close",
            "",
            lambda: self._inner.close(),
        )

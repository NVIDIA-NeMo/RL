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
from typing import Any, Callable, Literal

EventStatus = Literal["ok", "error", "timeout"]

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


def log_event(event: dict[str, Any]) -> None:
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
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._inner = inner
        self._on_event = on_event or (lambda _: None)
        self._stats = DataPlaneStats()
        # Nested per-partition / per-key live byte counts. Populated on
        # successful ``kv_batch_put``; popped on successful ``kv_clear``.
        # Bounded by the live key population, not cumulative traffic.
        self._bytes_by_partition: dict[str, dict[str, int]] = {}

    def snapshot(self) -> dict[str, Any]:
        """Cumulative totals plus live ``bytes_outstanding`` / ``peak_bytes_outstanding``."""
        out = asdict(self._stats)
        out["n_keys_outstanding"] = sum(
            len(d) for d in self._bytes_by_partition.values()
        )
        return out

    def bytes_outstanding_by_partition(self) -> dict[str, int]:
        """Per-partition breakdown of currently-held bytes."""
        return {p: sum(d.values()) for p, d in self._bytes_by_partition.items()}

    def _record_put(self, partition_id: str, keys: list[str], n_bytes: int) -> None:
        """Attribute put bytes per key so a later ``kv_clear`` can subtract.

        Called *after* the underlying RPC succeeds so a failed put never
        leaves the accounting inflated.
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
        """Reverse the put accounting for ``keys`` (``None`` clears the partition).

        Called *after* the underlying RPC succeeds so a failed clear
        keeps the accounting consistent with TQ's actual state.
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
        n_keys: int,
        n_bytes: int,
        fn: Callable[[], Any],
    ) -> Any:
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
            n_keys = len(out.keys)
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
        event = {
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
            int(num_samples),
            0,
            lambda: self._inner.register_partition(
                partition_id,
                fields,
                num_samples,
                consumer_tasks,
                grpo_group_size=grpo_group_size,
                enums=enums,
            ),
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
            0,
            0,
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
            len(meta.keys),
            0,
            lambda: self._inner.get_data(meta, select_fields=select_fields),
        )

    def check_consumption_status(self, partition_id, task_names):
        return self._inner.check_consumption_status(partition_id, task_names)

    def kv_batch_put(self, keys, partition_id, fields=None, tags=None):
        n_bytes = _td_bytes(fields)
        # Materialize keys once: ``_run`` consumes its lambda and we
        # also need to attribute bytes per key after success.
        keys_list = keys if isinstance(keys, list) else list(keys)
        out = self._run(
            "put",
            partition_id,
            len(keys_list),
            n_bytes,
            lambda: self._inner.kv_batch_put(
                keys_list,
                partition_id,
                fields=fields,
                tags=tags,
            ),
        )
        self._record_put(partition_id, keys_list, n_bytes)
        return out

    def kv_batch_get(self, keys, partition_id, select_fields=None):
        return self._run(
            "get",
            partition_id,
            len(keys),
            0,
            lambda: self._inner.kv_batch_get(
                keys,
                partition_id,
                select_fields=select_fields,
            ),
        )

    def kv_clear(self, keys, partition_id):
        keys_list = (
            keys if (keys is None or isinstance(keys, list)) else list(keys)
        )
        n_keys = len(keys_list) if keys_list is not None else 0
        self._run(
            "clear",
            partition_id,
            n_keys,
            0,
            lambda: self._inner.kv_clear(keys_list, partition_id),
        )
        self._record_clear(partition_id, keys_list)

    def close(self) -> None:
        self._inner.close()

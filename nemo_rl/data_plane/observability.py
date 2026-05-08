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
``on_event=<your function>``. ``snapshot()`` returns cumulative totals.
"""

from __future__ import annotations

from time import monotonic
from typing import Any, Callable, Literal

EventStatus = Literal["ok", "error", "timeout"]

import torch
from tensordict import TensorDict

from nemo_rl.data_plane.interfaces import DataPlaneClient, KVBatchMeta


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


def print_event(event: dict[str, Any]) -> None:
    print(
        f"[data_plane] op={event['op']} partition={event['partition_id']} "
        f"keys={event['n_keys']} bytes={event['n_bytes']} "
        f"ms={event['wall_ms']:.2f} status={event['status']}"
    )


class MetricsDataPlaneClient(DataPlaneClient):
    """Wrap a ``DataPlaneClient`` with a per-op callback hook."""

    def __init__(
        self,
        inner: DataPlaneClient,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._inner = inner
        self._on_event = on_event or (lambda _: None)
        self._stats: dict[str, int | float] = {
            "total_bytes": 0, "total_keys": 0, "total_ops": 0,
        }

    def snapshot(self) -> dict[str, Any]:
        return dict(self._stats)

    def _run(self, op: str, partition_id: str, n_keys: int, n_bytes: int,
             fn: Callable[[], Any]) -> Any:
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

    def _emit(self, op: str, partition_id: str, n_keys: int, n_bytes: int,
              t0: float, status: EventStatus) -> None:
        event = {
            "op": op, "partition_id": partition_id,
            "n_keys": int(n_keys), "n_bytes": int(n_bytes),
            "wall_ms": (monotonic() - t0) * 1000.0, "status": status,
        }
        self._on_event(event)
        if status == "ok":
            self._stats["total_bytes"] += n_bytes
            self._stats["total_keys"] += n_keys
            self._stats["total_ops"] += 1

    def register_partition(self, partition_id, fields, num_samples,
                           consumer_tasks, grpo_group_size=None, enums=None):
        self._run(
            "register", partition_id, int(num_samples), 0,
            lambda: self._inner.register_partition(
                partition_id, fields, num_samples, consumer_tasks,
                grpo_group_size=grpo_group_size, enums=enums,
            ),
        )

    def get_meta(self, partition_id, task_name, required_fields, batch_size,
                 dp_rank=None, blocking=True, timeout_s=60.0):
        return self._run(
            "get_meta", partition_id, 0, 0,
            lambda: self._inner.get_meta(
                partition_id, task_name, required_fields, batch_size,
                dp_rank=dp_rank, blocking=blocking, timeout_s=timeout_s,
            ),
        )

    def get_data(self, meta, select_fields=None):
        return self._run(
            "get_data", meta.partition_id, len(meta.keys), 0,
            lambda: self._inner.get_data(meta, select_fields=select_fields),
        )

    def check_consumption_status(self, partition_id, task_names):
        return self._inner.check_consumption_status(partition_id, task_names)

    def kv_batch_put(self, keys, partition_id, fields=None, tags=None):
        return self._run(
            "put", partition_id, len(keys), _td_bytes(fields),
            lambda: self._inner.kv_batch_put(
                keys, partition_id, fields=fields, tags=tags,
            ),
        )

    def kv_batch_get(self, keys, partition_id, select_fields=None):
        return self._run(
            "get", partition_id, len(keys), 0,
            lambda: self._inner.kv_batch_get(
                keys, partition_id, select_fields=select_fields,
            ),
        )

    def kv_clear(self, keys, partition_id):
        n_keys = len(keys) if keys is not None else 0
        self._run(
            "clear", partition_id, n_keys, 0,
            lambda: self._inner.kv_clear(keys, partition_id),
        )

    def close(self) -> None:
        self._inner.close()

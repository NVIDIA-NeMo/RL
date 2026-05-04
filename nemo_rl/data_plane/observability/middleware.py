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
"""``MetricsDataPlaneClient`` — observability middleware.

Wraps any :class:`DataPlaneClient` and emits a per-op event to a
:class:`MetricsSink` for every TQ operation. The wrapped client is
unchanged; nothing about the data plane's correctness path runs through
this layer. Composes with future middleware (integrity check, tracing)
by stacking: ``IntegrityClient(MetricsClient(TQDataPlaneClient(cfg)))``.
"""

from __future__ import annotations

from time import monotonic
from typing import TYPE_CHECKING, Any

import torch
from tensordict import TensorDict

from nemo_rl.data_plane.interfaces import DataPlaneClient

if TYPE_CHECKING:
    from nemo_rl.data_plane.interfaces import KVBatchMeta
    from nemo_rl.data_plane.observability.sinks import MetricsSink


def _td_bytes(td: TensorDict | None) -> int:
    """Sum tensor leaf byte counts. Approximate — ignores object headers."""
    if td is None:
        return 0
    n = 0
    for k in td.keys(include_nested=True, leaves_only=True):
        v = td.get(k)
        if isinstance(v, torch.Tensor):
            n += v.numel() * v.element_size()
    return n


class MetricsDataPlaneClient(DataPlaneClient):
    """Decorator over a DataPlaneClient. Forwards every method to the
    inner client; records a structured event per call.

    No control-plane semantics change. Errors raised by the inner client
    are recorded and re-raised — the middleware never swallows.
    """

    def __init__(self, inner: DataPlaneClient, sink: MetricsSink) -> None:
        self._inner = inner
        self._sink = sink

    # ── (A) task-mediated ───────────────────────────────────────────────

    def register_partition(
        self,
        partition_id,
        fields,
        num_samples,
        consumer_tasks,
        grpo_group_size=None,
        enums=None,
    ):
        t0 = monotonic()
        status = "ok"
        try:
            return self._inner.register_partition(
                partition_id, fields, num_samples, consumer_tasks,
                grpo_group_size=grpo_group_size, enums=enums,
            )
        except Exception:
            status = "error"
            raise
        finally:
            self._sink.record({
                "op": "register",
                "partition_id": partition_id,
                "n_keys": int(num_samples),
                "n_bytes": 0,
                "wall_ms": (monotonic() - t0) * 1000.0,
                "status": status,
                "fields": list(fields),
            })

    def get_meta(
        self, partition_id, task_name, required_fields, batch_size,
        dp_rank=None, blocking=True, timeout_s=60.0,
    ):
        t0 = monotonic()
        status = "ok"
        meta = None
        try:
            meta = self._inner.get_meta(
                partition_id, task_name, required_fields, batch_size,
                dp_rank=dp_rank, blocking=blocking, timeout_s=timeout_s,
            )
            return meta
        except TimeoutError:
            status = "timeout"
            raise
        except Exception:
            status = "error"
            raise
        finally:
            self._sink.record({
                "op": "get_meta",
                "partition_id": partition_id,
                "n_keys": meta.size if meta is not None else 0,
                "n_bytes": 0,
                "wall_ms": (monotonic() - t0) * 1000.0,
                "status": status,
                "fields": list(required_fields),
            })

    def get_data(self, meta, select_fields=None):
        t0 = monotonic()
        status = "ok"
        td = None
        try:
            td = self._inner.get_data(meta, select_fields=select_fields)
            return td
        except Exception:
            status = "error"
            raise
        finally:
            self._sink.record({
                "op": "get",
                "partition_id": meta.partition_id,
                "n_keys": meta.size,
                "n_bytes": _td_bytes(td),
                "wall_ms": (monotonic() - t0) * 1000.0,
                "status": status,
                "fields": list(select_fields) if select_fields else meta.fields,
            })

    def check_consumption_status(self, partition_id, task_names):
        return self._inner.check_consumption_status(partition_id, task_names)

    # ── (B) direct-by-key ──────────────────────────────────────────────

    async def kv_batch_put(
        self, keys, partition_id, fields=None, tags=None,
    ):
        t0 = monotonic()
        status = "ok"
        n_bytes = _td_bytes(fields)
        try:
            return await self._inner.kv_batch_put(
                keys, partition_id, fields=fields, tags=tags,
            )
        except Exception:
            status = "error"
            raise
        finally:
            self._sink.record({
                "op": "put",
                "partition_id": partition_id,
                "n_keys": len(keys),
                "n_bytes": n_bytes,
                "wall_ms": (monotonic() - t0) * 1000.0,
                "status": status,
                "fields": list(fields.keys()) if fields is not None else None,
            })

    def kv_batch_get(self, keys, partition_id, select_fields=None):
        t0 = monotonic()
        status = "ok"
        td = None
        try:
            td = self._inner.kv_batch_get(
                keys, partition_id, select_fields=select_fields,
            )
            return td
        except Exception:
            status = "error"
            raise
        finally:
            self._sink.record({
                "op": "get",
                "partition_id": partition_id,
                "n_keys": len(keys),
                "n_bytes": _td_bytes(td),
                "wall_ms": (monotonic() - t0) * 1000.0,
                "status": status,
                "fields": list(select_fields) if select_fields else None,
            })

    def kv_clear(self, keys, partition_id):
        t0 = monotonic()
        status = "ok"
        try:
            return self._inner.kv_clear(keys, partition_id)
        except Exception:
            status = "error"
            raise
        finally:
            self._sink.record({
                "op": "clear",
                "partition_id": partition_id,
                "n_keys": len(keys) if keys is not None else 0,
                "n_bytes": 0,
                "wall_ms": (monotonic() - t0) * 1000.0,
                "status": status,
                "fields": None,
            })

    # ── (C) lifecycle ──────────────────────────────────────────────────

    def close(self) -> None:
        try:
            self._inner.close()
        finally:
            self._sink.close()

    # ── observability surface ──────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        """Cumulative metrics. Trainer calls this once per step and
        merges into its own log_metrics() payload."""
        return self._sink.snapshot()

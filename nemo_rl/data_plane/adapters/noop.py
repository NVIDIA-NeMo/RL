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
"""In-memory ``DataPlaneClient`` for tests and the disabled-flag default.

Behaves like a real adapter end-to-end (put → get → clear, consumption
counters, field-presence as the stage-done signal) but stores everything
in process memory. Two uses:

* The factory returns this when ``cfg["enabled"] = False``, so call sites
  can be wired unconditionally — no ``if data_plane.enabled`` branching
  on the producer side.
* Stage 1 unit tests target the ABC contract through this implementation
  so the contract test runs without TQ installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from tensordict import TensorDict

from nemo_rl.data_plane.interfaces import DataPlaneClient, KVBatchMeta


def _reject_non_tensor_leaves(td: TensorDict) -> None:
    """P3 — no pickle on the bus. Mirror of the TQ adapter check.

    Walk the leaves via ``keys()`` + indexed lookup rather than
    ``items()``, because some tensordict versions skip ``NonTensorData``
    entries from ``items(leaves_only=True)`` — they're "leaves" by
    structure but not tensor-typed, so they'd silently slip past a
    naive items() iteration.
    """
    bad = []
    for k in td.keys(include_nested=True, leaves_only=True):
        v = td.get(k)
        if not isinstance(v, torch.Tensor):
            bad.append(k)
    if bad:
        raise TypeError(
            f"kv_batch_put received non-tensor leaves: {bad}. "
            "Tensorize via codec helpers, use `tags=` for primitives, "
            "or use the Ray object store for arbitrary Python objects."
        )


@dataclass
class _Partition:
    fields: list[str]
    num_samples: int
    consumer_tasks: list[str]
    grpo_group_size: int | None
    enums: dict[str, list[str]]
    rows: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict)
    tags: dict[str, dict[str, Any]] = field(default_factory=dict)
    # per-task set of keys already returned by get_meta(mode='fetch')
    consumed: dict[str, set[str]] = field(default_factory=dict)


class NoOpDataPlaneClient(DataPlaneClient):
    """Reference in-memory implementation."""

    def __init__(self) -> None:
        self._partitions: dict[str, _Partition] = {}
        self._closed = False

    def register_partition(
        self,
        partition_id: str,
        fields: list[str],
        num_samples: int,
        consumer_tasks: list[str],
        grpo_group_size: int | None = None,
        enums: dict[str, list[str]] | None = None,
    ) -> None:
        self._partitions[partition_id] = _Partition(
            fields=list(fields),
            num_samples=int(num_samples),
            consumer_tasks=list(consumer_tasks),
            grpo_group_size=grpo_group_size,
            enums=dict(enums) if enums else {},
            consumed={t: set() for t in consumer_tasks},
        )

    def get_meta(
        self,
        partition_id: str,
        task_name: str,
        required_fields: list[str],
        batch_size: int,
        dp_rank: int | None = None,
        blocking: bool = True,
        timeout_s: float = 60.0,
    ) -> KVBatchMeta:
        del blocking, timeout_s, dp_rank  # NoOp is single-process
        rec = self._partitions[partition_id]
        if task_name not in rec.consumed:
            raise KeyError(
                f"task {task_name!r} not registered as a consumer of "
                f"partition {partition_id!r}"
            )

        ready: list[str] = []
        seqs: list[int] = []
        for key, row in rec.rows.items():
            if key in rec.consumed[task_name]:
                continue
            if not all(f in row for f in required_fields):
                continue
            ready.append(key)
            tag = rec.tags.get(key, {})
            seqs.append(int(tag.get("input_lengths", 0)))
            if len(ready) >= batch_size:
                break

        rec.consumed[task_name].update(ready)
        return KVBatchMeta(
            partition_id=partition_id,
            task_name=task_name,
            keys=ready,
            fields=list(required_fields),
            sequence_lengths=seqs if any(seqs) else None,
        )

    def get_data(
        self,
        meta: KVBatchMeta,
        select_fields: list[str] | None = None,
    ) -> TensorDict:
        fields = select_fields if select_fields is not None else meta.fields
        if fields is None:
            raise ValueError(
                "get_data requires either select_fields or meta.fields; "
                "fetching all fields silently is forbidden (P2)."
            )
        return self.kv_batch_get(meta.keys, meta.partition_id, list(fields))

    def check_consumption_status(
        self, partition_id: str, task_names: list[str]
    ) -> bool:
        rec = self._partitions[partition_id]
        for t in task_names:
            if t not in rec.consumed:
                return False
            if len(rec.consumed[t]) < len(rec.rows):
                return False
        return True

    async def kv_batch_put(
        self,
        keys: list[str],
        partition_id: str,
        fields: TensorDict | None = None,
        tags: list[dict[str, Any]] | None = None,
    ) -> KVBatchMeta:
        rec = self._partitions[partition_id]
        if fields is not None:
            _reject_non_tensor_leaves(fields)
            for i, key in enumerate(keys):
                row = rec.rows.setdefault(key, {})
                for fname in fields.keys():
                    val = fields[fname][i]
                    # Defense in depth — _reject_non_tensor_leaves can
                    # miss NonTensorData entries depending on the
                    # tensordict version's iteration semantics.
                    if not isinstance(val, torch.Tensor):
                        raise TypeError(
                            f"kv_batch_put received non-tensor leaf "
                            f"{fname!r}: {type(val).__name__}. "
                            "Tensorize via codec helpers, use `tags=` "
                            "for primitives, or use the Ray object store "
                            "for arbitrary Python objects."
                        )
                    row[fname] = val.detach().clone()
        if tags is not None:
            for key, tag in zip(keys, tags):
                rec.tags.setdefault(key, {}).update(tag)
        return KVBatchMeta(
            partition_id=partition_id,
            task_name=None,
            keys=list(keys),
            fields=list(fields.keys()) if fields is not None else None,
        )

    def kv_batch_get(
        self,
        keys: list[str],
        partition_id: str,
        select_fields: list[str] | None = None,
    ) -> TensorDict:
        rec = self._partitions[partition_id]
        if not keys:
            return TensorDict({}, batch_size=(0,))

        if select_fields is None:
            available = set.intersection(*(set(rec.rows[k].keys()) for k in keys))
            select_fields = sorted(available)

        out: dict[str, list[torch.Tensor]] = {f: [] for f in select_fields}
        for key in keys:
            row = rec.rows[key]
            for f in select_fields:
                if f not in row:
                    raise KeyError(
                        f"field {f!r} not yet produced for key {key!r} "
                        f"in partition {partition_id!r}"
                    )
                out[f].append(row[f])

        stacked = {f: torch.stack(out[f], dim=0) for f in select_fields}
        return TensorDict(stacked, batch_size=(len(keys),))

    def kv_clear(self, keys: list[str] | None, partition_id: str) -> None:
        rec = self._partitions.get(partition_id)
        if rec is None:
            return
        if keys is None:
            rec.rows.clear()
            rec.tags.clear()
            for s in rec.consumed.values():
                s.clear()
            self._partitions.pop(partition_id, None)
            return
        for key in keys:
            rec.rows.pop(key, None)
            rec.tags.pop(key, None)
            for s in rec.consumed.values():
                s.discard(key)

    def close(self) -> None:
        if self._closed:
            return
        self._partitions.clear()
        self._closed = True

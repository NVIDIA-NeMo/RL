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
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
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
from nemo_rl.telemetry.instrumentation import (
    in_per_prompt_scope,
    is_span_group_enabled,
    managed_span,
    safe_set_span_attributes,
    umbrella_span,
)
from nemo_rl.telemetry.span_groups import RLSpanGroup

logger = logging.getLogger(__name__)

#: Reused rather than built per op: ``nullcontext`` holds no state, so one
#: instance is safe to enter concurrently from any number of threads.
_NO_SPAN = nullcontext(None)

# Span attribute names for a data-plane op. ``op`` and ``partition`` are bounded
# (a fixed op vocabulary, a handful of partitions), so they are safe as
# attributes; the byte and key counts are per-op numbers recorded on the span
# rather than metric labels.
_OP_ATTR = "rl.data_plane.op"
_PARTITION_ATTR = "rl.data_plane.partition"
_KEYS_ATTR = "rl.data_plane.keys"
_BYTES_ATTR = "rl.data_plane.bytes"
_STATUS_ATTR = "rl.data_plane.status"


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


def _annotate(span: Any, n_keys: int, n_bytes: int, status: EventStatus) -> None:
    """Record an op's outcome on its span.

    Set after the call rather than at open because the byte and key counts are
    only known once the inner client has returned. ``status`` distinguishes a
    timeout from a generic error, which the exception the span already records
    does not.
    """
    # safe_set_span_attributes absorbs a None span, but the dict below is built
    # by the caller before it can: returning first keeps that allocation off
    # the disabled path, which is the common case and runs once per op.
    if span is None:
        return
    safe_set_span_attributes(
        span,
        {
            _KEYS_ATTR: int(n_keys),
            _BYTES_ATTR: int(n_bytes),
            _STATUS_ATTR: status,
        },
    )


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

        Also opens one span per op, which is what puts transfer-queue traffic in
        the trace waterfall: on the single-controller path most of a step's
        non-compute time is data-plane traffic, and without these spans that time
        showed up only as a gap between phases.

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
        # One client per process serves both the rollout path, which puts once
        # per prompt, and the batch stages, which put once per step. Same op,
        # counts orders of magnitude apart, so the group has to come from the
        # caller's scope rather than from ``op``. PER_PROMPT is an umbrella, so
        # a rollout put is unbucketed where a batch put is overhead: rollouts
        # overlap each other and training, and their durations would sum past
        # the wall clock. Two branches rather than one variable group because
        # the umbrella helper is what marks a span as unbucketed at the call
        # site, and a drift test enforces the pairing statically.
        per_prompt = in_per_prompt_scope()
        group = RLSpanGroup.U_PER_PROMPT if per_prompt else RLSpanGroup.DATA_PLANE
        if not is_span_group_enabled(group):
            # Gate before building the name and the attribute dict, and before
            # either helper's generator is created. This is the most frequent
            # telemetry call site in the repo -- once per data-plane op, so once
            # per prompt on the rollout path -- and those three allocations cost
            # ~1.8us each, which a run that never enabled telemetry should not
            # be paying. A shared no-op context is safe to reuse: nullcontext
            # holds no state.
            span_ctx: Any = _NO_SPAN
        else:
            name = f"rl.data_plane.{op}"
            attributes = {_OP_ATTR: op, _PARTITION_ATTR: partition_id}
            if per_prompt:
                span_ctx = umbrella_span(RLSpanGroup.U_PER_PROMPT, name, **attributes)
            else:
                span_ctx = managed_span(RLSpanGroup.DATA_PLANE, name, **attributes)
        with span_ctx as span:
            try:
                out = fn()
            except TimeoutError:
                _annotate(span, n_keys, n_bytes, "timeout")
                self._emit(op, partition_id, n_keys, n_bytes, t0, "timeout")
                raise
            except Exception:
                _annotate(span, n_keys, n_bytes, "error")
                self._emit(op, partition_id, n_keys, n_bytes, t0, "error")
                raise
            # If the call returns a TensorDict, the read-side bytes are more
            # informative than the input estimate.
            if isinstance(out, TensorDict):
                n_bytes = _td_bytes(out)
            elif isinstance(out, KVBatchMeta) and not n_keys:
                n_keys = len(out.sample_ids)
            _annotate(span, n_keys, n_bytes, "ok")
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
        sample_ids_list = (
            sample_ids if isinstance(sample_ids, list) else list(sample_ids)
        )
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

    def list_sample_ids(self, partition_id: str) -> list[str]:
        return self._run(
            "list_sample_ids",
            partition_id,
            lambda: self._inner.list_sample_ids(partition_id),
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

    def save_checkpoint(
        self,
        checkpoint_dir: str | Path,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._run(
            "save_checkpoint",
            "",
            lambda: self._inner.save_checkpoint(
                checkpoint_dir,
                metadata=metadata,
            ),
        )

    def load_checkpoint(self, checkpoint_dir: str | Path) -> dict[str, Any]:
        return self._run(
            "load_checkpoint",
            "",
            lambda: self._inner.load_checkpoint(checkpoint_dir),
        )

    def close(self) -> None:
        self._run(
            "close",
            "",
            lambda: self._inner.close(),
        )

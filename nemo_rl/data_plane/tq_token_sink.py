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
"""TransferQueue implementations of NeMo-Gym's token staging protocols.

``TQTokenSink``/``TQTokenSource`` are NeMo-RL's providers for the
gate-authoritative capture design (docs/design-docs/tq-gym-gate-authoritative.md):
the sink is the worker-side write of one model call's token delta to the
``rollout_staging`` partition — the design's only heavy token hop — and the
source is the finalizer's read-back of those rows by staging key. This module
is the only hot-path file that knows tokens live in TQ; Gym sees opaque
staging keys.

Each staged row carries three jagged columns (``token_ids_delta``,
``token_mask_delta``, ``generation_logprobs_delta``) plus scalar ``prev_len``
and ``weight_version`` columns so a fetched row round-trips to a complete
``StagedCallSnapshot`` (parent pointers are lineage state and are rejoined
from the receipt manifest by the finalizer). Masks/logprobs are float32 on
the wire, matching ``compute_staging_digest``'s float32-bit-pattern scheme,
so the finalizer's digest recomputation over fetched values is byte-exact.
"""

from __future__ import annotations

from typing import Any, Optional

import ray
import torch
from tensordict import TensorDict

from nemo_gym.token_id_capture.staging.records import (
    StagedCallRecord,
    StagedCallSnapshot,
    StageResult,
)

STAGING_FIELDS = [
    "token_ids_delta",
    "token_mask_delta",
    "generation_logprobs_delta",
    "prev_len",
    "weight_version",
]


def _call_dp(dp_client: Any, method_name: str, **kwargs: Any) -> Any:
    """Call a DataPlaneClient method on a local client or a Ray actor handle."""
    method = getattr(dp_client, method_name)
    remote = getattr(method, "remote", None)
    if remote is not None:
        return ray.get(remote(**kwargs))
    return method(**kwargs)


class TQTokenSink:
    """Gym ``StagingSink`` over ``DataPlaneClient.put_samples``.

    ``stage`` is synchronous and returns only after TQ acknowledged the
    write, so the capture layer's fail-closed ordering (bytes durable before
    the model call is acked) holds by construction. Failures are reported in
    the ``StageResult`` — the caller decides whether the rollout poisons or
    aborts (``token_capture.on_capture_failure``).
    """

    def __init__(self, dp_client: Any, *, staging_partition: str) -> None:
        self._dp_client = dp_client
        self._staging_partition = staging_partition

    def stage(self, record: StagedCallRecord) -> StageResult:
        key = record.staging_key
        try:
            fields = TensorDict(
                {
                    "token_ids_delta": torch.tensor(
                        [record.token_ids_delta], dtype=torch.int64
                    ),
                    "token_mask_delta": torch.tensor(
                        [record.token_mask_delta], dtype=torch.float32
                    ),
                    "generation_logprobs_delta": torch.tensor(
                        [record.generation_logprobs_delta], dtype=torch.float32
                    ),
                    "prev_len": torch.tensor([record.prev_len], dtype=torch.int64),
                    "weight_version": torch.tensor(
                        [record.weight_version], dtype=torch.int64
                    ),
                },
                batch_size=[1],
            )
            tags = [
                {
                    "rollout_id": record.rollout_id,
                    "call_id": record.call_id,
                    "parent_call_id": record.parent_call_id,
                    "prev_len": record.prev_len,
                    "new_len": record.new_len,
                    "weight_version": record.weight_version,
                    "digest": record.digest,
                    "schema_version": record.schema_version,
                }
            ]
            _call_dp(
                self._dp_client,
                "put_samples",
                sample_ids=[key],
                partition_id=self._staging_partition,
                fields=fields,
                tags=tags,
            )
        except Exception as error:  # noqa: BLE001 — any failure must poison, not crash serving
            return StageResult(
                ok=False, staging_key=key, error=f"{type(error).__name__}: {error}"
            )
        return StageResult(ok=True, staging_key=key)

    def clear(self, staging_keys: list[str]) -> None:
        """Drop staged rows (finalizer / eviction cleanup)."""
        if not staging_keys:
            return
        _call_dp(
            self._dp_client,
            "clear_samples",
            sample_ids=list(staging_keys),
            partition_id=self._staging_partition,
        )


class TQTokenSource:
    """Gym ``StagingSource`` over ``DataPlaneClient.get_samples``.

    Rows are fetched one key at a time (deltas are jagged across calls) in
    the order requested. A missing or unreadable row raises ``KeyError`` per
    the protocol — the finalizer maps that to a placeholder, never a silent
    skip.
    """

    def __init__(self, dp_client: Any, *, staging_partition: str) -> None:
        self._dp_client = dp_client
        self._staging_partition = staging_partition

    def fetch(self, staging_keys: list[str]) -> list[StagedCallSnapshot]:
        snapshots: list[StagedCallSnapshot] = []
        for key in staging_keys:
            _, _, call_id = key.rpartition("/")
            try:
                row = _call_dp(
                    self._dp_client,
                    "get_samples",
                    sample_ids=[key],
                    partition_id=self._staging_partition,
                    select_fields=STAGING_FIELDS,
                )
                snapshots.append(_row_to_snapshot(call_id, row))
            except KeyError:
                raise
            except Exception as error:  # noqa: BLE001 — protocol maps any miss to KeyError
                raise KeyError(
                    f"staged row {key!r} could not be fetched: {error}"
                ) from error
        return snapshots


def _row_to_snapshot(call_id: str, row: Any) -> StagedCallSnapshot:
    def _leaf(name: str) -> torch.Tensor:
        value = row[name]
        tensor = value[0] if value.dim() > 1 or value.numel() > 1 else value
        return tensor.reshape(-1)

    prev_len = int(_leaf("prev_len")[0].item())
    weight_version: Optional[int] = int(_leaf("weight_version")[0].item())
    return StagedCallSnapshot(
        call_id=call_id,
        prev_len=prev_len,
        token_ids_delta=[int(t) for t in _leaf("token_ids_delta").tolist()],
        token_mask_delta=[float(m) for m in _leaf("token_mask_delta").tolist()],
        logprobs_delta=[float(p) for p in _leaf("generation_logprobs_delta").tolist()],
        weight_version=weight_version,
    )

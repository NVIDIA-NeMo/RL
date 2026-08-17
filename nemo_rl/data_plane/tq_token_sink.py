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

import logging
from typing import Any, Optional

import ray
import torch
from tensordict import TensorDict

from nemo_gym.token_id_capture.staging.records import (
    StagedCallRecord,
    StagedCallSnapshot,
    StageResult,
)

from nemo_rl.data_plane.schema import ROUTED_EXPERTS_FIELD

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
            field_dict = {
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
            }
            routed = (record.extras or {}).get("routed_experts")
            if routed is not None:
                # Worker sends full prompt+generation coverage of this call's
                # spliced sequence; the delta-aligned slice is [prev_len:].
                # A length mismatch stages the delta WITHOUT extras (replay
                # degrades to Megatron's own router for this call) rather
                # than poisoning the rollout.
                delta_len = len(record.token_ids_delta)
                if isinstance(routed, str):
                    # The worker ships routes as the nrlre1 base64 envelope
                    # (#3292). int16 covers every practical expert count
                    # (<32k) and the -1 sentinel.
                    from nemo_rl.utils.routed_experts_codec import (
                        decode_routed_experts,
                    )

                    experts = decode_routed_experts(routed, torch.int16)
                else:
                    # int16 covers every practical expert count (<32k) and
                    # the -1 sentinel; halves staged bytes vs int32.
                    experts = torch.tensor(routed, dtype=torch.int16)
                if experts.shape[0] == record.prev_len + delta_len:
                    experts = experts[record.prev_len :]
                if experts.shape[0] == delta_len:
                    field_dict[ROUTED_EXPERTS_FIELD] = experts.unsqueeze(0)
                else:
                    logging.getLogger(__name__).warning(
                        "routed_experts length %d does not cover delta %d "
                        "(prev_len %d) for %s — staging without extras",
                        experts.shape[0],
                        delta_len,
                        record.prev_len,
                        key,
                    )
            fields = TensorDict(field_dict, batch_size=[1])
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
            # The reason string is dropped downstream (_failed_coords carries
            # only the disposition) — this log line is the only place the
            # actual stage failure is visible.
            logging.getLogger(__name__).warning(
                "TQTokenSink.stage failed for %s: %s: %s",
                key,
                type(error).__name__,
                error,
            )
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

    def __init__(
        self,
        dp_client: Any,
        *,
        staging_partition: str,
        include_routed_experts: bool = False,
    ) -> None:
        self._dp_client = dp_client
        self._staging_partition = staging_partition
        self._fields = list(STAGING_FIELDS)
        if include_routed_experts:
            self._fields.append(ROUTED_EXPERTS_FIELD)

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
                    select_fields=self._fields,
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
    extras: Optional[dict[str, Any]] = None
    if ROUTED_EXPERTS_FIELD in row.keys():
        routed_experts = row[ROUTED_EXPERTS_FIELD]
        if routed_experts.dim() > 3:
            routed_experts = routed_experts[0]
        extras = {ROUTED_EXPERTS_FIELD: routed_experts.tolist()}
    return StagedCallSnapshot(
        call_id=call_id,
        prev_len=prev_len,
        token_ids_delta=[int(t) for t in _leaf("token_ids_delta").tolist()],
        token_mask_delta=[float(m) for m in _leaf("token_mask_delta").tolist()],
        logprobs_delta=[float(p) for p in _leaf("generation_logprobs_delta").tolist()],
        weight_version=weight_version,
        extras=extras,
    )

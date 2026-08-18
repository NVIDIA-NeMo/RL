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
from dataclasses import dataclass
from typing import Any, Optional

import ray
import torch
from tensordict import TensorDict

from nemo_gym.token_id_capture.staging.records import (
    StagedCallRecord,
    StagedCallSnapshot,
    StageResult,
)

from nemo_rl.data_plane.schema import ROUTED_EXPERTS_FIELD, ROUTED_LEN_FIELD

STAGING_FIELDS = [
    "token_ids_delta",
    "token_mask_delta",
    "generation_logprobs_delta",
    "prev_len",
    "weight_version",
    ROUTED_LEN_FIELD,
]


@dataclass(frozen=True)
class FetchedStagedCall:
    """One explicitly identified small-column finalization fetch result."""

    staging_key: str
    snapshot: StagedCallSnapshot
    routed_len: int


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
            routed_len = 0
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
                    routed_len = int(experts.shape[0])
                else:
                    logging.getLogger(__name__).warning(
                        "routed_experts length %d does not cover delta %d "
                        "(prev_len %d) for %s — staging without extras",
                        experts.shape[0],
                        delta_len,
                        record.prev_len,
                        key,
                    )
            field_dict[ROUTED_LEN_FIELD] = torch.tensor([routed_len], dtype=torch.int64)
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

    All requested rows are fetched in a single batched ``get_samples`` call
    (TQ returns jagged delta columns as nested tensors; ``_from_wire``
    preserves the raggedness), in the order requested. A missing or
    unreadable row raises ``KeyError`` per the protocol — the finalizer maps
    that to a placeholder, never a silent skip. TQ's field-readiness check
    is all-or-nothing across a batch, so the extras fallback is batch-level:
    extras-free runs land in the base schema exactly like the old per-key
    probe, but a batch with *mixed* extras presence degrades every row to
    the base schema (worker feature-gating makes presence uniform per run).
    """

    def __init__(self, dp_client: Any, *, staging_partition: str) -> None:
        self._dp_client = dp_client
        self._staging_partition = staging_partition

    def fetch(self, staging_keys: list[str]) -> list[StagedCallSnapshot]:
        """Fetch Gym snapshots, retaining legacy optional route materialization."""
        if not staging_keys:
            return []
        try:
            # Extras are optional per row (feature-gated at the worker);
            # try the extended selection first, fall back to the base
            # schema so extras-free rows keep fetching.
            try:
                rows = _call_dp(
                    self._dp_client,
                    "get_samples",
                    sample_ids=list(staging_keys),
                    partition_id=self._staging_partition,
                    select_fields=STAGING_FIELDS + [ROUTED_EXPERTS_FIELD],
                )
            except Exception:  # noqa: BLE001 — field-not-present probe
                rows = _call_dp(
                    self._dp_client,
                    "get_samples",
                    sample_ids=list(staging_keys),
                    partition_id=self._staging_partition,
                    select_fields=STAGING_FIELDS,
                )
        except Exception as error:  # noqa: BLE001 — protocol maps any miss to KeyError
            raise KeyError(
                f"staged rows for {len(staging_keys)} keys could not be "
                f"fetched from {self._staging_partition!r}: {error}"
            ) from error
        # TQ's kv path only errors when *zero* keys resolve; a partial miss
        # returns fewer rows with no error. Guard explicitly so a lost row
        # rejects the rollout as missing_staging_row instead of surfacing
        # later as a misleading digest mismatch from misaligned zipping.
        n_rows = int(rows.batch_size[0]) if len(rows.batch_size) else 0
        if n_rows != len(staging_keys):
            raise KeyError(
                f"staged rows missing: requested {len(staging_keys)} keys "
                f"from {self._staging_partition!r}, got {n_rows} rows"
            )
        # Row order mirrors the requested key order; the finalizer's digest
        # recomputation is the byte-exact backstop if that ever breaks.
        snapshots: list[StagedCallSnapshot] = []
        for index, key in enumerate(staging_keys):
            _, _, call_id = key.rpartition("/")
            snapshots.append(_row_to_snapshot(call_id, _select_row(rows, index)))
        return snapshots

    def fetch_for_finalization(
        self, staging_keys: list[str]
    ) -> list[FetchedStagedCall]:
        """Fetch only digest-covered columns plus required route length metadata."""
        if not staging_keys:
            return []
        if len(set(staging_keys)) != len(staging_keys):
            raise KeyError("finalization staging request contains duplicate keys")
        try:
            rows = _call_dp(
                self._dp_client,
                "get_samples",
                sample_ids=list(staging_keys),
                partition_id=self._staging_partition,
                select_fields=STAGING_FIELDS,
            )
        except Exception as error:  # noqa: BLE001 — protocol maps misses to KeyError
            raise KeyError(
                f"staged rows for {len(staging_keys)} keys could not be "
                f"fetched from {self._staging_partition!r}: {error}"
            ) from error
        n_rows = int(rows.batch_size[0]) if len(rows.batch_size) else 0
        if n_rows != len(staging_keys):
            raise KeyError(
                f"staged rows missing: requested {len(staging_keys)} keys "
                f"from {self._staging_partition!r}, got {n_rows} rows"
            )
        fetched: list[FetchedStagedCall] = []
        for index, key in enumerate(staging_keys):
            _, separator, call_id = key.rpartition("/")
            if not separator or not call_id:
                raise KeyError(f"invalid staging key without call id: {key!r}")
            row = _select_row(rows, index)
            fetched.append(
                FetchedStagedCall(
                    staging_key=key,
                    snapshot=_row_to_snapshot(call_id, row),
                    routed_len=_row_scalar_int(row, ROUTED_LEN_FIELD),
                )
            )
        return fetched


def _select_row(rows: TensorDict, index: int) -> dict[str, torch.Tensor]:
    """Slice one row out of a batched fetch, restoring single-row shapes.

    ``_row_to_snapshot`` predates batching and expects each field with a
    leading batch dim of 1 (the shape a single-key ``get_samples`` returns),
    so re-add it after indexing. Indexing a nested tensor yields that row's
    dense component, which is exactly the jagged-row payload.
    """
    row: dict[str, torch.Tensor] = {}
    for field in rows.keys():
        row[str(field)] = rows.get(field)[index].unsqueeze(0)
    return row


def _row_to_snapshot(call_id: str, row: Any) -> StagedCallSnapshot:
    def _leaf(name: str) -> torch.Tensor:
        value = row[name]
        tensor = value[0] if value.dim() > 1 or value.numel() > 1 else value
        return tensor.reshape(-1)

    prev_len = int(_leaf("prev_len")[0].item())
    weight_version: Optional[int] = int(_leaf("weight_version")[0].item())
    extras: Optional[dict[str, Any]] = None
    try:
        routed = row[ROUTED_EXPERTS_FIELD]
    except KeyError:
        routed = None
    if routed is not None:
        # [1, delta_len, ...] -> [delta_len, ...]; keep the nested list wire
        # shape so the snapshot stays a plain-pydantic record.
        experts = routed[0] if routed.dim() > 2 or routed.shape[0] == 1 else routed
        extras = {"routed_experts": experts.tolist()}
    return StagedCallSnapshot(
        call_id=call_id,
        prev_len=prev_len,
        token_ids_delta=[int(t) for t in _leaf("token_ids_delta").tolist()],
        token_mask_delta=[float(m) for m in _leaf("token_mask_delta").tolist()],
        logprobs_delta=[float(p) for p in _leaf("generation_logprobs_delta").tolist()],
        weight_version=weight_version,
        extras=extras,
    )


def _row_scalar_int(row: Any, field_name: str) -> int:
    """Read one required scalar from a single-row TQ result."""
    value = row[field_name]
    if not isinstance(value, torch.Tensor):
        raise TypeError(
            f"staging field {field_name!r} must be a tensor, got {type(value).__name__}"
        )
    integer_dtypes = {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
    if value.dtype not in integer_dtypes:
        raise TypeError(
            f"staging field {field_name!r} must use an integer dtype, got {value.dtype}"
        )
    tensor = value[0] if value.dim() > 1 or value.numel() > 1 else value
    flattened = tensor.reshape(-1)
    if flattened.numel() != 1:
        raise ValueError(
            f"staging field {field_name!r} must contain one scalar, got "
            f"shape {tuple(value.shape)}"
        )
    return int(flattened[0].item())

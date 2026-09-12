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
ledger-authoritative capture design:
the sink is the worker-side write of one model call's token delta to the
``rollout_staging`` partition — the design's only heavy token hop — and the
source is the finalizer's read-back of those rows by staging key. This module
is the only hot-path file that knows tokens live in TQ; Gym sees opaque
staging keys.

Each staged row carries three jagged columns (``token_ids_delta``,
``token_mask_delta``, ``generation_logprobs_delta``), the complete receipt
identity/lineage metadata, and all digest inputs so it round-trips to a
normally validated ``StagedCallBaseSnapshot``. Masks/logprobs are float32 on
the wire, matching ``compute_staging_digest``'s float32-bit-pattern scheme, so
digest recomputation over fetched values is byte-exact. Route payloads never
ride inside snapshots: the source returns them as separate ``RouteFragment``
values keyed by staging key, digest-verified by the plan executor at point of
use.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import ray
import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    # Deferred: nemo_gym is an optional extra absent in non-gym runs; runtime
    # uses import locally so this module (and the finalizer actor importing
    # it) stays importable without it.
    from nemo_gym.token_id_capture.staging.records import (
        StagedCallBaseSnapshot,
        StagedCallRecord,
        StageResult,
    )

from nemo_rl.data_plane.schema import (
    ROUTE_ENCODING_ENVELOPE,
    ROUTE_ENCODING_LIST,
    ROUTE_ENCODING_NONE,
    ROUTED_EXPERTS_ENCODING_FIELD,
    ROUTED_EXPERTS_FIELD,
    ROUTED_EXTRAS_METADATA_FIELD,
    ROUTED_LEN_FIELD,
)
from nemo_rl.experience.route_assembly import RouteFragment

# These names come from nemo_gym.token_id_capture.staging.records.StagedCallRecord,
# transformed by stage() below. Adding a field means editing both this list and
# stage(); a mismatch is caught by test_tq_sink_source_passes_conformance's
# round-trip equality check -- but only for required StagedCallRecord fields. An
# optional field Gym adds that this sink never stages will default identically
# on both sides and pass that check silently.
STAGING_FIELDS = [
    "token_ids_delta",
    "token_mask_delta",
    "generation_logprobs_delta",
    "schema_version",
    "digest_version",
    "extras_digest_version",
    "rollout_id_utf8",
    "model_call_id_utf8",
    "parent_call_id_utf8",
    "parent_call_id_present",
    "capture_mode",
    "prev_len",
    "delta_len",
    "cum_len",
    "weight_version",
    "digest_bytes",
    "extras_digest_bytes",
    "chain_hash_bytes",
    "chain_hash_present",
    "cumulative_hash_bytes",
    "cumulative_hash_present",
    ROUTED_EXTRAS_METADATA_FIELD,
    ROUTED_EXPERTS_ENCODING_FIELD,
    ROUTED_LEN_FIELD,
]

_MODE_TO_CODE = {"text": 0, "token_in": 1}
_CODE_TO_MODE = {code: mode for mode, code in _MODE_TO_CODE.items()}


def _bytes_tensor(value: bytes) -> torch.Tensor:
    """Encode non-empty bytes as one jagged TQ row."""
    if not value:
        raise ValueError("staging byte fields must be non-empty")
    return torch.tensor([list(value)], dtype=torch.uint8)


def _optional_digest_fields(value: str | None) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        _bytes_tensor(bytes.fromhex(value) if value is not None else bytes(32)),
        torch.tensor([value is not None], dtype=torch.bool),
    )


@dataclass(frozen=True)
class FetchedStagedCall:
    """One explicitly identified small-column finalization fetch result.

    ``fragment`` is populated only when the fetch requested route payloads
    (direct mode); deferred finalization leaves route bytes in TQ and carries
    only ``routed_len`` transport metadata.
    """

    staging_key: str
    snapshot: StagedCallBaseSnapshot
    routed_len: int
    fragment: RouteFragment | None = None


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
    the ``StageResult``; the finalizer turns a poisoned rollout into a
    placeholder row (see ``RolloutReassembler.finalize_group``).

    ``stage`` is thread-safe per the ``StagingSink`` contract: it holds no
    per-call mutable state, so the capture host may run writes for unrelated
    calls concurrently.
    """

    def __init__(self, dp_client: Any, *, staging_partition: str) -> None:
        self._dp_client = dp_client
        self._staging_partition = staging_partition

    def stage(self, record: StagedCallRecord) -> StageResult:
        # Deferred: nemo_gym is an optional extra absent in non-gym runs.
        from nemo_gym.token_id_capture.staging.records import StageResult

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
                    [record.generation_log_probs_delta], dtype=torch.float32
                ),
                "schema_version": torch.tensor(
                    [record.schema_version], dtype=torch.int64
                ),
                "digest_version": torch.tensor(
                    [record.digest_version], dtype=torch.int64
                ),
                "extras_digest_version": torch.tensor(
                    [record.extras_digest_version], dtype=torch.int64
                ),
                "rollout_id_utf8": _bytes_tensor(record.rollout_id.encode("utf-8")),
                "model_call_id_utf8": _bytes_tensor(
                    record.model_call_id.encode("utf-8")
                ),
                "parent_call_id_utf8": _bytes_tensor(
                    (record.parent_call_id or "\0").encode("utf-8")
                ),
                "parent_call_id_present": torch.tensor(
                    [record.parent_call_id is not None], dtype=torch.bool
                ),
                "capture_mode": torch.tensor(
                    [_MODE_TO_CODE[record.mode]], dtype=torch.int64
                ),
                "prev_len": torch.tensor([record.prev_len], dtype=torch.int64),
                "delta_len": torch.tensor([record.delta_len], dtype=torch.int64),
                "cum_len": torch.tensor([record.cum_len], dtype=torch.int64),
                "weight_version": torch.tensor(
                    [record.weight_version], dtype=torch.int64
                ),
                "digest_bytes": _bytes_tensor(bytes.fromhex(record.digest)),
                "extras_digest_bytes": _bytes_tensor(
                    bytes.fromhex(record.extras_digest)
                ),
            }
            chain_hash, chain_hash_present = _optional_digest_fields(record.chain_hash)
            cumulative_hash, cumulative_hash_present = _optional_digest_fields(
                record.cumulative_hash
            )
            field_dict.update(
                {
                    "chain_hash_bytes": chain_hash,
                    "chain_hash_present": chain_hash_present,
                    "cumulative_hash_bytes": cumulative_hash,
                    "cumulative_hash_present": cumulative_hash_present,
                }
            )
            extras_metadata = dict(record.extras) if record.extras is not None else None
            routed = (
                extras_metadata.pop("routed_experts", None)
                if extras_metadata is not None
                else None
            )
            field_dict[ROUTED_EXTRAS_METADATA_FIELD] = _bytes_tensor(
                json.dumps(
                    extras_metadata,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
            )
            routed_len = 0
            routed_encoding = ROUTE_ENCODING_NONE
            if routed is not None:
                delta_len = len(record.token_ids_delta)
                if isinstance(routed, str):
                    from nemo_rl.utils.routed_experts_codec import (
                        decode_routed_experts,
                    )

                    dtype_name = routed.split(":", 3)[1]
                    dtype = {
                        "int8": torch.int8,
                        "int16": torch.int16,
                        "int32": torch.int32,
                    }.get(dtype_name)
                    if dtype is None:
                        raise ValueError(
                            f"unsupported routed_experts dtype {dtype_name!r}"
                        )
                    experts = decode_routed_experts(routed, dtype)
                    routed_encoding = ROUTE_ENCODING_ENVELOPE
                else:
                    experts = torch.tensor(routed, dtype=torch.int16)
                    routed_encoding = ROUTE_ENCODING_LIST
                if experts.dim() != 3 or experts.shape[0] != delta_len:
                    raise ValueError(
                        "routed_experts must already be delta-aligned: "
                        f"got shape {tuple(experts.shape)} for delta_len={delta_len}"
                    )
                field_dict[ROUTED_EXPERTS_FIELD] = experts.unsqueeze(0)
                routed_len = int(experts.shape[0])
            field_dict[ROUTED_EXPERTS_ENCODING_FIELD] = torch.tensor(
                [routed_encoding], dtype=torch.int64
            )
            field_dict[ROUTED_LEN_FIELD] = torch.tensor([routed_len], dtype=torch.int64)
            fields = TensorDict(field_dict, batch_size=[1])
            tags = [
                {
                    "rollout_id": record.rollout_id,
                    "model_call_id": record.model_call_id,
                    "parent_call_id": record.parent_call_id,
                    "prev_len": record.prev_len,
                    "delta_len": record.delta_len,
                    "cum_len": record.cum_len,
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

    def fetch(self, staging_keys: list[str]) -> list[StagedCallBaseSnapshot]:
        """Gym ``StagingSource`` conformance: base snapshots only, in order."""
        return [item.snapshot for item in self.fetch_for_finalization(staging_keys)]

    def fetch_prefix_token_ids(self, staging_keys: list[str]) -> list[int]:
        """Bulk-fetch ordered delta chain and concatenate token_ids_delta into a prefix."""
        if not staging_keys:
            return []
        if len(set(staging_keys)) != len(staging_keys):
            raise KeyError("prefix fetch: staging_keys contains duplicates")
        try:
            rows = _call_dp(
                self._dp_client,
                "get_samples",
                sample_ids=list(staging_keys),
                partition_id=self._staging_partition,
                select_fields=["token_ids_delta"],
            )
        except Exception as error:  # noqa: BLE001 — protocol maps any miss to KeyError
            raise KeyError(
                f"prefix fetch: staged rows for {len(staging_keys)} keys could "
                f"not be fetched from {self._staging_partition!r}: {error}"
            ) from error
        n_rows = int(rows.batch_size[0]) if rows.batch_size else 0
        if n_rows != len(staging_keys):
            raise KeyError(
                f"prefix fetch incomplete: requested {len(staging_keys)} keys, got {n_rows}"
            )
        result: list[int] = []
        for index in range(n_rows):
            row = _select_row(rows, index)
            delta = row["token_ids_delta"].squeeze(0).tolist()
            result.extend(int(t) for t in delta)
        return result

    def fetch_for_finalization(
        self,
        staging_keys: list[str],
        *,
        include_route_fragments: bool = False,
    ) -> list[FetchedStagedCall]:
        """Fetch digest-covered base columns, plus route payloads when requested.

        Deferred mode (the default) never selects ``routed_experts`` — route
        bytes stay in TQ for the policy worker. Direct mode passes
        ``include_route_fragments=True`` to pull the payloads in the same
        batched read and receives them as ``RouteFragment`` values beside the
        base snapshots, never inside them.
        """
        if not staging_keys:
            return []
        if len(set(staging_keys)) != len(staging_keys):
            raise KeyError("finalization staging request contains duplicate keys")
        try:
            if include_route_fragments:
                # Route payloads are optional per run (feature-gated at the
                # worker); fall back to the base schema so extras-free rows
                # keep fetching.
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
            else:
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
        # Row order mirrors the requested key order; digest recomputation at
        # snapshot validation is the byte-exact backstop if that ever breaks.
        fetched: list[FetchedStagedCall] = []
        for index, key in enumerate(staging_keys):
            row = _select_row(rows, index)
            snapshot = _row_to_base_snapshot(row)
            if snapshot.staging_key != key:
                raise KeyError(
                    f"staged row identity mismatch: requested {key!r}, got {snapshot.staging_key!r}"
                )
            fetched.append(
                FetchedStagedCall(
                    staging_key=key,
                    snapshot=snapshot,
                    routed_len=_row_scalar_int(row, ROUTED_LEN_FIELD),
                    fragment=(
                        _row_to_route_fragment(row) if include_route_fragments else None
                    ),
                )
            )
        return fetched


def _select_row(rows: TensorDict, index: int) -> dict[str, torch.Tensor]:
    """Slice one row out of a batched fetch, restoring single-row shapes.

    ``_row_to_base_snapshot`` predates batching and expects each field with a
    leading batch dim of 1 (the shape a single-key ``get_samples`` returns),
    so re-add it after indexing. Indexing a nested tensor yields that row's
    dense component, which is exactly the jagged-row payload.
    """
    row: dict[str, torch.Tensor] = {}
    for field in rows.keys():
        value = rows.get(field)
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"staging field {field!r} must be a tensor, got {type(value).__name__}"
            )
        row[str(field)] = value[index].unsqueeze(0)
    return row


def _row_leaf(row: Any, name: str) -> torch.Tensor:
    value = row[name]
    tensor = value[0] if value.dim() > 1 or value.numel() > 1 else value
    return tensor.reshape(-1)


def _row_text(row: Any, name: str) -> str:
    return bytes(int(value) for value in _row_leaf(row, name).tolist()).decode("utf-8")


def _row_to_base_snapshot(row: Any) -> StagedCallBaseSnapshot:
    """Rebuild one normally validated base snapshot; route bytes never enter it."""
    # Deferred: nemo_gym is an optional extra absent in non-gym runs.
    from nemo_gym.token_id_capture.staging.records import StagedCallBaseSnapshot

    def _digest(name: str) -> str:
        value = bytes(int(item) for item in _row_leaf(row, name).tolist())
        if len(value) != 32:
            raise ValueError(f"{name} must contain exactly 32 bytes")
        return value.hex()

    def _optional_digest(name: str, present_name: str) -> str | None:
        return _digest(name) if bool(_row_leaf(row, present_name)[0].item()) else None

    parent_call_id = (
        _row_text(row, "parent_call_id_utf8")
        if bool(_row_leaf(row, "parent_call_id_present")[0].item())
        else None
    )
    mode_code = int(_row_leaf(row, "capture_mode")[0].item())
    try:
        mode = _CODE_TO_MODE[mode_code]
    except KeyError as error:
        raise ValueError(f"unknown capture_mode code {mode_code}") from error
    routed_encoding = int(_row_leaf(row, ROUTED_EXPERTS_ENCODING_FIELD)[0].item())
    if routed_encoding not in (
        ROUTE_ENCODING_NONE,
        ROUTE_ENCODING_ENVELOPE,
        ROUTE_ENCODING_LIST,
    ):
        raise ValueError(f"unknown routed_experts_encoding {routed_encoding}")

    return StagedCallBaseSnapshot(
        schema_version=int(_row_leaf(row, "schema_version")[0].item()),
        digest_version=int(_row_leaf(row, "digest_version")[0].item()),
        extras_digest_version=int(_row_leaf(row, "extras_digest_version")[0].item()),
        rollout_id=_row_text(row, "rollout_id_utf8"),
        model_call_id=_row_text(row, "model_call_id_utf8"),
        parent_call_id=parent_call_id,
        mode=mode,
        prev_len=int(_row_leaf(row, "prev_len")[0].item()),
        delta_len=int(_row_leaf(row, "delta_len")[0].item()),
        cum_len=int(_row_leaf(row, "cum_len")[0].item()),
        weight_version=int(_row_leaf(row, "weight_version")[0].item()),
        digest=_digest("digest_bytes"),
        token_ids_delta=[int(t) for t in _row_leaf(row, "token_ids_delta").tolist()],
        token_mask_delta=[
            float(m) for m in _row_leaf(row, "token_mask_delta").tolist()
        ],
        generation_log_probs_delta=[
            float(p) for p in _row_leaf(row, "generation_logprobs_delta").tolist()
        ],
        extras_digest=_digest("extras_digest_bytes"),
        chain_hash=_optional_digest("chain_hash_bytes", "chain_hash_present"),
        cumulative_hash=_optional_digest(
            "cumulative_hash_bytes", "cumulative_hash_present"
        ),
    )


def _row_to_route_fragment(row: Any) -> RouteFragment | None:
    """Extract one staged route payload beside (never inside) the snapshot."""
    routed_encoding = int(_row_leaf(row, ROUTED_EXPERTS_ENCODING_FIELD)[0].item())
    if routed_encoding == ROUTE_ENCODING_NONE:
        return None
    try:
        routed = row[ROUTED_EXPERTS_FIELD]
    except KeyError as error:
        raise KeyError(
            "staged row metadata names routed_experts but its field is absent"
        ) from error
    experts = routed[0] if routed.dim() > 3 or routed.shape[0] == 1 else routed
    return RouteFragment(
        routes=experts,
        encoding=routed_encoding,
        extras_metadata_json=_row_text(row, ROUTED_EXTRAS_METADATA_FIELD).encode(
            "utf-8"
        ),
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

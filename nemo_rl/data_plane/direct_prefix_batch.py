# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Owned column construction for text and routed generation-prefix batches.

The output uses the existing staging schema and row ordering. Token IDs are
int64, masks/logprobs float32, and byte metadata uint8. NumPy/bytearray buffers
own their storage; the tensor references keep them alive through TQ completion.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from tensordict import TensorDict

from nemo_rl.data_plane.codec import stack_or_nest
from nemo_rl.data_plane.schema import (
    ROUTE_ENCODING_ENVELOPE,
    ROUTE_ENCODING_LIST,
    ROUTE_ENCODING_NONE,
    ROUTE_ENCODING_TENSOR,
    ROUTED_EXPERTS_ENCODING_FIELD,
    ROUTED_EXPERTS_FIELD,
    ROUTED_EXTRAS_METADATA_FIELD,
    ROUTED_LEN_FIELD,
)
from nemo_rl.utils.prefix_cut_diagnostics import prefix_cut_count, prefix_cut_timing
from nemo_rl.utils.routed_experts_codec import (
    ROUTED_EXPERTS_STAGING_ATTACHMENT,
    RoutedExpertsStagingPayload,
    decode_routed_experts,
)

if TYPE_CHECKING:
    from nemo_gym.token_id_capture.staging.records import StagedCallRecord


@dataclass(frozen=True)
class EncodedPrefixBatch:
    """An owned batch and the exact payload-byte count, ready for TQ."""

    fields: TensorDict
    tags: list[dict[str, Any]]
    payload_bytes: int


def _column(values: np.ndarray, lengths: np.ndarray) -> torch.Tensor:
    """Wrap one owned flat buffer directly as a dense or jagged batch column."""
    tensor = torch.from_numpy(values)
    if np.all(lengths == lengths[0]):
        return tensor.reshape(len(lengths), int(lengths[0]))
    offsets = np.empty(len(lengths) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])
    return torch.nested.nested_tensor_from_jagged(
        tensor, offsets=torch.from_numpy(offsets)
    )


def _numeric_column(
    rows: list[list[int]] | list[list[float]], dtype: np.dtype[Any]
) -> torch.Tensor:
    lengths = np.fromiter(map(len, rows), dtype=np.int64, count=len(rows))
    values = np.empty(int(lengths.sum()), dtype=dtype)
    cursor = 0
    for row in rows:
        end = cursor + len(row)
        values[cursor:end] = row
        cursor = end
    return _column(values, lengths)


def _byte_column(rows: list[bytes]) -> torch.Tensor:
    lengths = np.fromiter(map(len, rows), dtype=np.int64, count=len(rows))
    if np.any(lengths == 0):
        raise ValueError("Staging byte fields must be non-empty")
    # bytearray owns writable storage, retained by the NumPy/Torch buffer chain.
    values = np.frombuffer(bytearray().join(rows), dtype=np.uint8)
    return _column(values, lengths)


def encode_text_generation_prefix_batch(
    records: Sequence[StagedCallRecord], *, mode_codes: Mapping[str, int]
) -> EncodedPrefixBatch:
    """Encode a nonempty text/no-router cohort directly into batch columns.

    Routed rows use encode_generation_prefix_batch with validated route rows;
    media rows use the general staging encoder. Reject unsupported inputs.
    """
    if any(
        record.extras is not None and "routed_experts" in record.extras
        for record in records
    ):
        raise ValueError("Direct text prefix encoding does not support routed experts")
    return encode_generation_prefix_batch(records, None, mode_codes=mode_codes)


def prepare_prefix_route_groups(
    records: Sequence[StagedCallRecord],
    attachments: Sequence[Mapping[str, Any] | None],
) -> list[tuple[list[int], list[torch.Tensor] | None]]:
    """Validate route ownership/alignment and group compatible rows before writes.

    This is only used for non-media partitions. Every attachment is checked
    before publishing any cohort; unknown attachments are never discarded.
    """
    if len(records) != len(attachments):
        raise ValueError("records and attachments must have equal lengths")
    groups: dict[
        tuple[torch.dtype, tuple[int, ...]] | None,
        tuple[list[int], list[torch.Tensor] | None],
    ] = {}
    for index, (record, row_attachments) in enumerate(
        zip(records, attachments, strict=True)
    ):
        payload = None
        if row_attachments:
            if any(key != ROUTED_EXPERTS_STAGING_ATTACHMENT for key in row_attachments):
                raise ValueError(
                    "media attachments require a media-enabled staging partition"
                )
            payload = row_attachments[ROUTED_EXPERTS_STAGING_ATTACHMENT]
            if not isinstance(payload, RoutedExpertsStagingPayload):
                raise TypeError(
                    "route staging attachment must be RoutedExpertsStagingPayload"
                )
        routed = (
            record.extras.get("routed_experts") if record.extras is not None else None
        )
        if payload is not None and routed is None:
            raise ValueError("route staging attachment has no record payload")
        experts = None
        signature = None
        if routed is not None:
            if payload is None:
                with prefix_cut_timing("sink_route_decode"):
                    experts = decode_routed_experts(routed)
            else:
                with prefix_cut_timing("sink_route_reuse"):
                    experts = payload.tensor_for_payload(routed)
                prefix_cut_count("sink_route_reuse_rows", 1)
                prefix_cut_count(
                    "sink_route_reuse_bytes", experts.numel() * experts.element_size()
                )
            if experts.dim() != 3 or experts.shape[0] != record.delta_len:
                raise ValueError(
                    "routed_experts must already be delta-aligned: "
                    f"got shape {tuple(experts.shape)} for delta_len={record.delta_len}"
                )
            signature = (experts.dtype, tuple(experts.shape[1:]))
        if signature not in groups:
            groups[signature] = ([], [] if experts is not None else None)
        indices, routes = groups[signature]
        indices.append(index)
        if routes is not None and experts is not None:
            routes.append(experts)
    return list(groups.values())


def encode_generation_prefix_batch(
    records: Sequence[StagedCallRecord],
    route_rows: Sequence[torch.Tensor] | None,
    *,
    mode_codes: Mapping[str, int],
) -> EncodedPrefixBatch:
    """Build owned columns for an already route-validated homogeneous cohort."""
    if not records:
        raise ValueError("Direct prefix encoding requires at least one record")
    if route_rows is not None and len(route_rows) != len(records):
        raise ValueError("route and record counts differ")
    for index, record in enumerate(records):
        routed = (
            record.extras.get("routed_experts") if record.extras is not None else None
        )
        if (routed is not None) != (route_rows is not None):
            raise ValueError("route payload presence does not match batch columns")
        if route_rows is not None:
            row = route_rows[index]
            if row.dim() != 3 or row.shape[0] != record.delta_len:
                raise ValueError("routed_experts must already be delta-aligned")
            if (
                row.dtype != route_rows[0].dtype
                or row.shape[1:] != route_rows[0].shape[1:]
            ):
                raise ValueError(
                    "route batch must have uniform dtype and trailing shape"
                )
    for record in records:
        if not (
            len(record.token_ids_delta)
            == len(record.token_mask_delta)
            == len(record.generation_log_probs_delta)
            == record.delta_len
        ):
            raise ValueError("Prefix token/mask/logprob lengths differ")
    with prefix_cut_timing("sink_record_encode"):
        fields = {
            "token_ids_delta": _numeric_column(
                [r.token_ids_delta for r in records], np.dtype(np.int64)
            ),
            "token_mask_delta": _numeric_column(
                [r.token_mask_delta for r in records], np.dtype(np.float32)
            ),
            "generation_logprobs_delta": _numeric_column(
                [r.generation_log_probs_delta for r in records], np.dtype(np.float32)
            ),
        }
        # One conversion per scalar metadata column, rather than per record.
        scalars: dict[str, list[Any]] = {
            "schema_version": [r.schema_version for r in records],
            "digest_version": [r.digest_version for r in records],
            "extras_digest_version": [r.extras_digest_version for r in records],
            "prev_len": [r.prev_len for r in records],
            "delta_len": [r.delta_len for r in records],
            "cum_len": [r.cum_len for r in records],
            "weight_version": [r.weight_version for r in records],
            "capture_mode": [mode_codes[r.mode] for r in records],
            ROUTED_EXPERTS_ENCODING_FIELD: [
                (
                    ROUTE_ENCODING_TENSOR
                    if isinstance(r.extras["routed_experts"], dict)
                    else ROUTE_ENCODING_ENVELOPE
                    if isinstance(r.extras["routed_experts"], str)
                    else ROUTE_ENCODING_LIST
                )
                if route_rows is not None
                else ROUTE_ENCODING_NONE
                for r in records
            ],
            ROUTED_LEN_FIELD: [
                r.delta_len if route_rows is not None else 0 for r in records
            ],
        }
        for name, values in scalars.items():
            fields[name] = torch.from_numpy(np.array(values, dtype=np.int64))
        for name, values in {
            "parent_call_id_present": [r.parent_call_id is not None for r in records],
            "chain_hash_present": [r.chain_hash is not None for r in records],
            "cumulative_hash_present": [r.cumulative_hash is not None for r in records],
        }.items():
            fields[name] = torch.from_numpy(np.array(values, dtype=np.bool_))
        bytes_columns = {
            "rollout_id_utf8": [r.rollout_id.encode("utf-8") for r in records],
            "model_call_id_utf8": [r.model_call_id.encode("utf-8") for r in records],
            "parent_call_id_utf8": [
                (r.parent_call_id or "\0").encode("utf-8") for r in records
            ],
            "digest_bytes": [bytes.fromhex(r.digest) for r in records],
            "extras_digest_bytes": [bytes.fromhex(r.extras_digest) for r in records],
            "chain_hash_bytes": [
                bytes.fromhex(r.chain_hash) if r.chain_hash else bytes(32)
                for r in records
            ],
            "cumulative_hash_bytes": [
                bytes.fromhex(r.cumulative_hash) if r.cumulative_hash else bytes(32)
                for r in records
            ],
            ROUTED_EXTRAS_METADATA_FIELD: [
                json.dumps(
                    {
                        key: value
                        for key, value in r.extras.items()
                        if key != "routed_experts"
                    }
                    if r.extras is not None
                    else None,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
                for r in records
            ],
        }
        for name, values in bytes_columns.items():
            fields[name] = _byte_column(values)
        tags = [
            {
                "rollout_id": r.rollout_id,
                "model_call_id": r.model_call_id,
                "parent_call_id": r.parent_call_id,
                "prev_len": r.prev_len,
                "delta_len": r.delta_len,
                "cum_len": r.cum_len,
                "weight_version": r.weight_version,
                "digest": r.digest,
                "schema_version": r.schema_version,
            }
            for r in records
        ]
    # Avoid a native tensor operation per field per row merely for telemetry.
    payload_bytes = (
        sum(record.delta_len for record in records) * 16
        + len(records) * (len(scalars) * 8 + 3)
        + sum(len(value) for values in bytes_columns.values() for value in values)
    )
    with prefix_cut_timing("sink_batch_construct"):
        if route_rows is not None:
            # stack/cat owns the batch storage, even when input rows are views
            # into the attachment snapshot. No per-record metadata tensors.
            fields[ROUTED_EXPERTS_FIELD] = stack_or_nest(list(route_rows))
            payload_bytes += sum(row.numel() * row.element_size() for row in route_rows)
        batch = TensorDict(fields, batch_size=[len(records)])
    return EncodedPrefixBatch(batch, tags, payload_bytes)

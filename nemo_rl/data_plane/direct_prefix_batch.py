# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Owned column construction for text-only generation-prefix batches.

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

from nemo_rl.data_plane.schema import (
    ROUTE_ENCODING_NONE,
    ROUTED_EXPERTS_ENCODING_FIELD,
    ROUTED_EXTRAS_METADATA_FIELD,
    ROUTED_LEN_FIELD,
)
from nemo_rl.utils.prefix_cut_diagnostics import prefix_cut_timing

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

    Media and routed-expert rows must use the general staging encoder.
    Reject unsupported inputs instead of dropping their data.
    """
    if not records:
        raise ValueError("Direct prefix encoding requires at least one record")
    for record in records:
        if record.extras is not None and "routed_experts" in record.extras:
            raise ValueError(
                "Direct text prefix encoding does not support routed experts"
            )
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
            ROUTED_EXPERTS_ENCODING_FIELD: [ROUTE_ENCODING_NONE] * len(records),
            ROUTED_LEN_FIELD: [0] * len(records),
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
                    dict(r.extras) if r.extras is not None else None,
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
        batch = TensorDict(fields, batch_size=[len(records)])
    return EncodedPrefixBatch(batch, tags, payload_bytes)

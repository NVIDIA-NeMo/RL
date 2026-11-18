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
"""Build owned text/router prefix columns without intermediate row tensors."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict

from nemo_rl.data_plane.codec import stack_or_nest
from nemo_rl.data_plane.schema import (
    CAPTURE_MODE_TO_CODE,
    ROUTE_ENCODING_NONE,
    ROUTE_ENCODING_TENSOR,
    ROUTED_EXPERTS_ENCODING_FIELD,
    ROUTED_EXPERTS_FIELD,
    ROUTED_EXTRAS_METADATA_FIELD,
    ROUTED_LEN_FIELD,
)

# token_ids_delta (int64) plus token_mask_delta and generation_logprobs_delta
# (float32), as encode_prefix_groups builds them.
_TOKEN_BYTES = np.dtype(np.int64).itemsize + 2 * np.dtype(np.float32).itemsize
# Rough per-row allowance for the scalar, digest and offset columns.
_ROW_BYTES = 512


def _dense_or_jagged(values: torch.Tensor, lengths: np.ndarray) -> torch.Tensor:
    """Rows of equal length as a dense 2D tensor, otherwise a jagged one."""
    if np.all(lengths == lengths[0]):
        return values.reshape(len(lengths), int(lengths[0]))
    offsets = np.empty(len(lengths) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])
    return torch.nested.nested_tensor_from_jagged(
        values, offsets=torch.from_numpy(offsets)
    )


def _numeric_column(
    rows: Sequence[Sequence[Any]], dtype: np.dtype[Any]
) -> torch.Tensor:
    lengths = np.fromiter(map(len, rows), dtype=np.int64, count=len(rows))
    values = np.empty(int(lengths.sum()), dtype=dtype)
    cursor = 0
    for row in rows:
        end = cursor + len(row)
        values[cursor:end] = row
        cursor = end
    return _dense_or_jagged(torch.from_numpy(values), lengths)


def _byte_column(rows: list[bytes]) -> torch.Tensor:
    lengths = np.fromiter(map(len, rows), dtype=np.int64, count=len(rows))
    if np.any(lengths == 0):
        raise ValueError("Staging byte fields must be nonempty")
    values = torch.from_numpy(np.frombuffer(bytearray().join(rows), dtype=np.uint8))
    return _dense_or_jagged(values, lengths)


def estimate_prefix_row_bytes(
    record: Any, attachments: Mapping[str, torch.Tensor] | None
) -> int:
    """Estimated encoded size of one cut row, for the write pipeline's byte bounds."""
    size = len(record.token_ids_delta) * _TOKEN_BYTES + _ROW_BYTES
    for text in (record.capture_key, record.model_call_id, record.parent_call_id):
        if text is not None:
            size += len(text.encode("utf-8"))
    if record.extras is not None:
        size += len(json.dumps(record.extras, ensure_ascii=False).encode("utf-8"))
    if attachments:
        size += sum(
            value.numel() * value.element_size() for value in attachments.values()
        )
    return size


def can_encode_directly(record: Any, attached: Mapping[str, Any] | None) -> bool:
    """Whether a row has only text and tensor-routed columns.

    Media, sampler top-k and legacy route envelopes use the general sink
    encoder.
    """
    descriptor = (record.extras or {}).get(ROUTED_EXPERTS_FIELD)
    return (descriptor is None or isinstance(descriptor, dict)) and not (
        attached and set(attached) - {ROUTED_EXPERTS_FIELD}
    )


def encode_prefix_groups(
    records: Sequence[Any],
    attachments: Sequence[Mapping[str, Any] | None],
) -> list[tuple[list[int], TensorDict, list[dict[str, Any]]]]:
    """Validate all rows, then encode homogeneous text/router groups.

    Every row must pass :func:`can_encode_directly`. Route descriptors and
    tensors use the existing staging schema. Returned columns own their
    numeric buffers through the PUT acknowledgement.
    """
    if not records or len(records) != len(attachments):
        raise ValueError(
            "Direct prefix encoding needs equally many nonempty rows and attachments"
        )
    groups: dict[Any, list[int]] = {}
    routes: dict[int, torch.Tensor] = {}
    for index, (record, attached) in enumerate(zip(records, attachments, strict=True)):
        if not (
            len(record.token_ids_delta)
            == len(record.token_mask_delta)
            == len(record.generation_log_probs_delta)
            == record.delta_len
        ):
            raise ValueError("Prefix token/mask/logprob lengths differ")
        if attached and set(attached) != {ROUTED_EXPERTS_FIELD}:
            raise ValueError("Direct prefix encoding accepts only route attachments")
        descriptor = (record.extras or {}).get(ROUTED_EXPERTS_FIELD)
        tensor = (attached or {}).get(ROUTED_EXPERTS_FIELD)
        signature = None
        if descriptor is not None:
            if not isinstance(descriptor, dict) or not isinstance(tensor, torch.Tensor):
                raise ValueError("Direct route encoding requires metadata and a tensor")
            if (
                tensor.device.type != "cpu"
                or tensor.ndim != 3
                or tensor.shape[0] != record.delta_len
                or tensor.dtype not in (torch.int8, torch.int16, torch.int32)
            ):
                raise ValueError(
                    "Direct prefix routes must be delta-aligned CPU integer tensors"
                )
            routes[index] = tensor
            signature = (tensor.dtype, tuple(tensor.shape[1:]))
        elif tensor is not None:
            raise ValueError("Route tensor attachment has no capture metadata")
        groups.setdefault(signature, []).append(index)

    result = []
    for signature, indices in groups.items():
        rows = [records[index] for index in indices]
        fields = {
            "token_ids_delta": _numeric_column(
                [r.token_ids_delta for r in rows], np.dtype(np.int64)
            ),
            "token_mask_delta": _numeric_column(
                [r.token_mask_delta for r in rows], np.dtype(np.float32)
            ),
            "generation_logprobs_delta": _numeric_column(
                [r.generation_log_probs_delta for r in rows], np.dtype(np.float32)
            ),
        }
        scalars = {
            "schema_version": [r.schema_version for r in rows],
            "digest_version": [r.digest_version for r in rows],
            "extras_digest_version": [r.extras_digest_version for r in rows],
            "prev_len": [r.prev_len for r in rows],
            "delta_len": [r.delta_len for r in rows],
            "cum_len": [r.cum_len for r in rows],
            "weight_version": [r.weight_version for r in rows],
            "capture_mode": [CAPTURE_MODE_TO_CODE[r.mode] for r in rows],
            ROUTED_EXPERTS_ENCODING_FIELD: [
                ROUTE_ENCODING_NONE if signature is None else ROUTE_ENCODING_TENSOR
            ]
            * len(rows),
            ROUTED_LEN_FIELD: [0 if signature is None else r.delta_len for r in rows],
        }
        for name, values in scalars.items():
            fields[name] = torch.from_numpy(np.array(values, dtype=np.int64))
        for name, values in {
            "parent_call_id_present": [r.parent_call_id is not None for r in rows],
            "chain_hash_present": [r.chain_hash is not None for r in rows],
            "cumulative_hash_present": [r.cumulative_hash is not None for r in rows],
        }.items():
            fields[name] = torch.from_numpy(np.array(values, dtype=np.bool_))
        byte_columns = {
            "capture_key_utf8": [r.capture_key.encode("utf-8") for r in rows],
            "model_call_id_utf8": [r.model_call_id.encode("utf-8") for r in rows],
            "parent_call_id_utf8": [
                (r.parent_call_id or "\0").encode("utf-8") for r in rows
            ],
            "digest_bytes": [bytes.fromhex(r.digest) for r in rows],
            "extras_digest_bytes": [bytes.fromhex(r.extras_digest) for r in rows],
            "chain_hash_bytes": [
                bytes.fromhex(r.chain_hash) if r.chain_hash is not None else bytes(32)
                for r in rows
            ],
            "cumulative_hash_bytes": [
                bytes.fromhex(r.cumulative_hash)
                if r.cumulative_hash is not None
                else bytes(32)
                for r in rows
            ],
            ROUTED_EXTRAS_METADATA_FIELD: [
                json.dumps(
                    {
                        key: value
                        for key, value in r.extras.items()
                        if key != ROUTED_EXPERTS_FIELD
                    }
                    if r.extras is not None
                    else None,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
                for r in rows
            ],
        }
        for name, values in byte_columns.items():
            fields[name] = _byte_column(values)
        if signature is not None:
            fields[ROUTED_EXPERTS_FIELD] = stack_or_nest(
                [routes[index] for index in indices]
            )
        tags = [
            {
                "capture_key": r.capture_key,
                "model_call_id": r.model_call_id,
                "parent_call_id": r.parent_call_id,
                "prev_len": r.prev_len,
                "delta_len": r.delta_len,
                "cum_len": r.cum_len,
                "weight_version": r.weight_version,
                "digest": r.digest,
                "schema_version": r.schema_version,
            }
            for r in rows
        ]
        result.append((indices, TensorDict(fields, batch_size=[len(rows)]), tags))
    return result

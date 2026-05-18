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
"""Column-level helpers above :class:`DataPlaneClient`.

These are thin wrappers around :meth:`get_samples` / :meth:`put_samples`
that operate on **columns** (named fields) of a partition — not on the
driver process specifically. The driver uses them to fetch a slice and
materialize / write deltas back; worker-side dispatches use the
equivalents on ``AbstractPolicyWorker`` (``self._fetch(meta)`` /
``self._write_back``).

  * :func:`read_columns` — ``get_samples + materialize`` (decode jagged
    + object-array fields into a :class:`BatchedDataDict`).
  * :func:`write_columns` — pack-to-wire + ``put_samples`` for deltas
    against an existing :class:`KVBatchMeta`.
  * :func:`kv_first_write` — pack-to-wire + ``put_samples`` for the
    rollout-actor's first put of a partition. Returns a new
    :class:`KVBatchMeta`.
"""

from typing import Any, Sequence

import numpy as np
import torch

from nemo_rl.data.llm_message_utils import attach_message_log_view
from nemo_rl.data_plane.codec import materialize, pack_jagged_fields
from nemo_rl.data_plane.interfaces import DataPlaneClient, KVBatchMeta
from nemo_rl.data_plane.schema import GLOBAL_FORWARD_PAD_SEQLEN, Layout
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def _round_up(value: int, multiple: int) -> int:
    """Smallest ``multiple``-aligned int ≥ ``value`` (no-op when ``multiple <= 1``)."""
    if multiple <= 1:
        return value
    return ((value + multiple - 1) // multiple) * multiple


def read_columns(
    dp_client: DataPlaneClient,
    meta: KVBatchMeta,
    select_fields: Sequence[str],
    *,
    layout: Layout = "padded",
    pad_value_dict: dict[str, Any] | None = None,
) -> BatchedDataDict[Any]:
    """``get_samples(meta.sample_ids, select_fields=...) → materialize``.

    Pads to ``meta.extra_info[GLOBAL_FORWARD_PAD_SEQLEN]`` (minted on
    the driver by ``TQPolicy._stamp_pad_seqlen`` and inherited by every
    per-rank shard via :func:`shard_meta_for_dp`) — so driver-fetched
    and worker-returned columns land at one identical seq dim.

    Args:
        dp_client: Data-plane client used for the underlying fetch.
        meta: ``KVBatchMeta`` describing the keys to fetch.
        select_fields: Fields to fetch.
        layout: Materialization layout (``"padded"`` or ``"jagged"``).
        pad_value_dict: Per-field pad value for jagged tensors (e.g.
            ``input_ids → pad_token_id``); defaults to 0.

    Returns:
        ``BatchedDataDict`` with the requested fields, materialized.
    """
    td = dp_client.get_samples(
        sample_ids=meta.sample_ids,
        partition_id=meta.partition_id,
        select_fields=list(select_fields),
    )
    pad_to_seqlen = int((meta.extra_info or {}).get(GLOBAL_FORWARD_PAD_SEQLEN, 0))
    data = materialize(
        td,
        layout=layout,
        pad_value_dict=pad_value_dict,
        pad_to_seqlen=pad_to_seqlen,
    )
    attach_message_log_view(data)
    return data


def write_columns(
    dp_client: DataPlaneClient,
    meta: KVBatchMeta,
    fields: "dict[str, torch.Tensor | np.ndarray]",
) -> None:
    """``put_samples(meta.sample_ids, fields=...)``.

    Per-token tensor fields are converted to jagged via
    :func:`pack_jagged_fields` so they land in TQ with the same row
    lengths as the initial put. ``np.ndarray(dtype=object)`` leaves
    pass through as-is.

    Args:
        dp_client: Data-plane client used for the underlying put.
        meta: ``KVBatchMeta`` describing the keys being written.
        fields: Map of field name to tensor or object array.
    """
    if not fields:
        return

    seq_lens = meta.sequence_lengths
    lengths = torch.tensor(seq_lens, dtype=torch.long) if seq_lens is not None else None
    td = pack_jagged_fields(fields, lengths=lengths)
    dp_client.put_samples(
        sample_ids=meta.sample_ids,
        partition_id=meta.partition_id,
        fields=td,
    )


def kv_first_write(
    final_batch_cpu: BatchedDataDict[Any],
    *,
    sample_ids: Sequence[str],
    dp_client: DataPlaneClient,
    partition_id: str,
    extra_info: dict[str, Any] | None = None,
    task_name: str = "train",
    pad_to_multiple: int = 1,
    tags: list[dict[str, Any]] | None = None,
) -> KVBatchMeta:
    """Single flat ``put_samples`` of every tensor field in ``final_batch_cpu``.

    The rollout actor's first put of a partition. Caller mints
    ``sample_ids`` (verl-style) — the helper is rollout-shape-agnostic.

    Args:
        final_batch_cpu: Rollout output already on CPU. Must contain
            ``"sample_mask"`` (used as batch-size oracle: ``shape[0] == N``)
            and ``"input_lengths"`` (per-row valid lengths for the jagged
            pack). Tensor fields are packed jagged via
            :func:`pack_jagged_fields`; ``np.ndarray(dtype=object)``
            leaves pass through.
        sample_ids: Pre-minted per-sample ids, one per row of
            ``final_batch_cpu``.
        dp_client: Data-plane client used for the put.
        partition_id: TQ partition to write into.
        extra_info: Optional extra fields to attach to the returned meta.
        task_name: Consumer task tag stamped on the returned meta.
        pad_to_multiple: Seq-dim alignment recorded in ``extra_info`` so
            readers pad to a multiple compatible with downstream backends
            (mcore SP, PyTorch CP).
        tags: Optional per-sample primitive metadata (one dict per row).
            Stored on the TQ controller alongside the samples; travels
            with ``KVBatchMeta`` through ``subset`` / ``concat`` / ``slice``
            so consumers can filter on it without fetching tensor data.

    Returns:
        ``KVBatchMeta`` covering the written samples.
    """
    n = int(final_batch_cpu["sample_mask"].shape[0])
    if n == 0 or len(sample_ids) != n:
        raise ValueError(
            f"kv_first_write: sample_ids ({len(sample_ids)}) must match batch size ({n})"
        )
    if tags is not None and len(tags) != n:
        raise ValueError(
            f"kv_first_write: tags ({len(tags)}) must match batch size ({n})"
        )
    lengths = final_batch_cpu["input_lengths"]
    fields: dict[str, torch.Tensor | np.ndarray] = {
        k: v
        for k, v in final_batch_cpu.items()
        if isinstance(v, torch.Tensor)
        or (isinstance(v, np.ndarray) and v.dtype == object)
    }
    td = pack_jagged_fields(fields, lengths=lengths)
    dp_client.put_samples(
        sample_ids=list(sample_ids),
        partition_id=partition_id,
        fields=td,
        tags=tags,
    )

    extras = dict(extra_info or {})
    if pad_to_multiple > 1:
        extras["pad_to_multiple"] = int(pad_to_multiple)
    return KVBatchMeta(
        partition_id=partition_id,
        task_name=task_name,
        sample_ids=list(sample_ids),
        fields=list(td.keys()),
        sequence_lengths=[int(s) for s in lengths.tolist()],
        extra_info=extras,
        tags=[dict(t) for t in tags] if tags is not None else None,
    )

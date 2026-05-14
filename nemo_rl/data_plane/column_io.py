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

These are thin wrappers around :meth:`kv_batch_get` / :meth:`kv_batch_put`
that operate on **columns** (named fields) of a partition — not on the
driver process specifically. The driver uses them to fetch a slice and
materialize / write deltas back; worker-side dispatches use the
equivalents on ``AbstractPolicyWorker`` (``self._fetch(meta)`` /
``self._write_back``).

  * :func:`read_columns` — ``kv_batch_get + materialize`` (decode jagged
    + object-array fields into a :class:`BatchedDataDict`).
  * :func:`write_columns` — encode jagged / object-array fields and
    ``kv_batch_put`` the result.
"""

from typing import Any, Sequence

import numpy as np
import torch
from tensordict import TensorDict

from nemo_rl.data.llm_message_utils import attach_message_log_view
from nemo_rl.data_plane.codec import (
    materialize,
    maybe_pack_jagged,
)
from nemo_rl.data_plane.interfaces import DataPlaneClient, KVBatchMeta
from nemo_rl.data_plane.schema import Layout
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def read_columns(
    dp_client: DataPlaneClient,
    meta: KVBatchMeta,
    select_fields: Sequence[str],
    *,
    layout: Layout = "padded",
    pad_value_dict: dict[str, Any] | None = None,
) -> BatchedDataDict[Any]:
    """``kv_batch_get(meta.keys, select_fields=...) → materialize``.

    ``pad_to_multiple`` is read from ``meta.extra_info`` so the
    materialized seq dim matches the alignment downstream backends
    require (mcore SP / PyTorch CP). Non-tensor object fields ride as
    ``NonTensorStack`` leaves; :func:`materialize` unwraps them to
    ``np.ndarray(dtype=object)``.

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
    td = dp_client.kv_batch_get(
        keys=meta.keys,
        partition_id=meta.partition_id,
        select_fields=list(select_fields),
    )
    pad_mult = int((meta.extra_info or {}).get("pad_to_multiple", 1))
    data = materialize(
        td,
        layout=layout,
        pad_value_dict=pad_value_dict,
        pad_to_multiple=pad_mult,
    )
    attach_message_log_view(data)
    return data


def write_columns(
    dp_client: DataPlaneClient,
    meta: KVBatchMeta,
    fields: "dict[str, torch.Tensor | np.ndarray]",
) -> None:
    """``kv_batch_put(meta.keys, fields=...)``.

    Per-token tensor fields are converted to jagged via
    :func:`maybe_pack_jagged` so they land in TQ with the same row
    lengths as the initial put. ``np.ndarray(dtype=object)`` leaves are
    wrapped in ``NonTensorStack`` — TQ handles non-tensor encoding per
    backend.

    Args:
        dp_client: Data-plane client used for the underlying put.
        meta: ``KVBatchMeta`` describing the keys being written.
        fields: Map of field name to tensor or object array.
    """
    if not fields:
        return

    seq_lens = meta.sequence_lengths
    lengths = torch.tensor(seq_lens, dtype=torch.long) if seq_lens is not None else None

    packed: dict[str, Any] = {}
    for k, v in fields.items():
        if isinstance(v, np.ndarray) and v.dtype == object:
            # Pass through as ndarray; see kv_first_write for the
            # tensordict==0.12.2 NonTensorStack→LinkedList rationale.
            packed[k] = v
        elif isinstance(v, torch.Tensor):
            packed[k] = (
                maybe_pack_jagged(v, lengths)
                if lengths is not None
                else v.detach().contiguous()
            )
        else:
            raise TypeError(
                f"write_columns: unsupported value type for {k!r}: {type(v)}. "
                "Use torch.Tensor or np.ndarray(dtype=object)."
            )

    td = TensorDict(packed, batch_size=[len(meta.keys)])
    dp_client.kv_batch_put(
        keys=meta.keys,
        partition_id=meta.partition_id,
        fields=td,
    )

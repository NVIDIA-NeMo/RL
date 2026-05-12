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

from typing import Any, Literal, Sequence

import numpy as np
import torch
from tensordict import TensorDict

from nemo_rl.data.llm_message_utils import attach_message_log_view
from nemo_rl.data_plane.codec import (
    META_OBJECT_FIELDS,
    materialize,
    maybe_pack_jagged,
    pack_object_array,
    select_object_fields,
)
from nemo_rl.data_plane.interfaces import DataPlaneClient, KVBatchMeta
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def read_columns(
    dp_client: DataPlaneClient,
    meta: KVBatchMeta,
    select_fields: Sequence[str],
    *,
    layout: Literal["jagged", "padded"] = "padded",
    pad_value_dict: dict[str, Any] | None = None,
) -> BatchedDataDict[Any]:
    """``kv_batch_get(meta.keys, select_fields=...) → materialize``.

    ``pad_value_dict`` is forwarded to :func:`materialize` so jagged
    fields are padded with the right value per field
    (``input_ids → pad_token_id``, masks → 0, logprobs → 0.0). When
    omitted, jagged fields pad with 0.

    ``pad_to_multiple`` is read from ``meta.extra_info`` (writer-side
    alignment recorded at first put) so the materialized seq dim
    matches the alignment required by downstream backends (mcore SP /
    PyTorch CP).

    Object-encoded fields (registered at write time in
    ``meta.extra_info['object_fields']``) bypass tensor padding and
    are unpickled back to ``np.ndarray(dtype=object)`` — see
    :func:`nemo_rl.data_plane.codec.pack_object_array`.
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
        object_fields=select_object_fields(meta, select_fields),
    )
    attach_message_log_view(data)
    return data


def write_columns(
    dp_client: DataPlaneClient,
    meta: KVBatchMeta,
    fields: "dict[str, torch.Tensor | np.ndarray]",
) -> None:
    """``kv_batch_put(meta.keys, fields=...)``.

    Per-token fields (whose seq dim matches ``max(meta.sequence_lengths)``)
    are converted to jagged before the put via :func:`maybe_pack_jagged`,
    so they land in TQ with the same row lengths as the initial put — keeps
    mixed jagged/rectangular shape mismatches out of subsequent reads.

    Object-array fields (``np.ndarray(dtype=object)``) must already be
    registered in ``meta.extra_info[META_OBJECT_FIELDS]`` (typically by
    :func:`kv_first_write`); writing an unregistered object field raises
    so subsequent reads can't silently corrupt by treating uint8 wire
    bytes as a regular tensor.
    """
    if not fields:
        return

    seq_lens = meta.sequence_lengths
    lengths = torch.tensor(seq_lens, dtype=torch.long) if seq_lens is not None else None
    registered_objects = set((meta.extra_info or {}).get(META_OBJECT_FIELDS, ()))

    packed: dict[str, torch.Tensor] = {}
    for k, v in fields.items():
        if isinstance(v, np.ndarray) and v.dtype == object:
            if k not in registered_objects:
                raise ValueError(
                    f"write_columns: object field {k!r} not registered in "
                    f"meta.extra_info[{META_OBJECT_FIELDS!r}]; register it "
                    f"at first put (kv_first_write) so readers decode it."
                )
            packed[k] = pack_object_array(v)
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

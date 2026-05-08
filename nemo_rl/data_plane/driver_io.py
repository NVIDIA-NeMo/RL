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
"""Driver-side TQ I/O helpers: fetch a slice + materialize, write deltas back.

Fetch the columns the driver consumes, transform, write deltas. Worker-
side dispatches use the equivalents on ``AbstractPolicyWorker``
(``self._fetch(meta)`` / ``self._write_back``).
"""

from typing import Any, Sequence

import torch
from tensordict import TensorDict

from nemo_rl.data_plane.codec import materialize
from nemo_rl.data_plane.interfaces import DataPlaneClient, KVBatchMeta
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def read_columns(
    dp_client: DataPlaneClient,
    meta: KVBatchMeta,
    select_fields: Sequence[str],
    *,
    layout: str = "padded",
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
    """
    td = dp_client.kv_batch_get(
        keys=meta.keys,
        partition_id=meta.partition_id,
        select_fields=list(select_fields),
    )
    pad_mult = int((meta.extra_info or {}).get("pad_to_multiple", 1))
    return materialize(
        td,
        layout=layout,
        pad_value_dict=pad_value_dict,
        pad_to_multiple=pad_mult,
    )


def write_columns(
    dp_client: DataPlaneClient,
    meta: KVBatchMeta,
    fields: dict[str, torch.Tensor],
) -> None:
    """``kv_batch_put(meta.keys, fields=...)``.

    Per-token fields (whose seq dim matches ``max(meta.sequence_lengths)``)
    are converted to jagged before the put via :func:`maybe_pack_jagged`,
    so they land in TQ with the same row lengths as the initial put — keeps
    mixed jagged/rectangular shape mismatches out of subsequent reads.
    """
    if not fields:
        return
    from nemo_rl.data_plane.codec import maybe_pack_jagged

    seq_lens = meta.sequence_lengths
    if seq_lens is not None:
        lengths = torch.tensor(seq_lens, dtype=torch.long)
        packed = {k: maybe_pack_jagged(v, lengths) for k, v in fields.items()}
    else:
        packed = {k: v.detach().contiguous() for k, v in fields.items()}

    td = TensorDict(packed, batch_size=[len(meta.keys)])
    dp_client.kv_batch_put(
        keys=meta.keys, partition_id=meta.partition_id, fields=td,
    )

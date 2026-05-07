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

Mirrors verl ``main_ppo_sync.py:_compute_old_log_prob`` /
``_compute_advantage``: fetch the columns the driver consumes, transform,
write deltas. Worker-side dispatches use the equivalents on
``AbstractPolicyWorker`` (``self._fetch(meta)`` / ``self._write_back``).
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
) -> BatchedDataDict[Any]:
    """``kv_batch_get(meta.keys, select_fields=...) → materialize``."""
    td = dp_client.kv_batch_get(
        keys=meta.keys,
        partition_id=meta.partition_id,
        select_fields=list(select_fields),
    )
    return materialize(td, layout=layout)


def write_columns(
    dp_client: DataPlaneClient,
    meta: KVBatchMeta,
    fields: dict[str, torch.Tensor],
) -> None:
    """``kv_batch_put(meta.keys, fields=...)``."""
    if not fields:
        return
    td = TensorDict(
        {k: v.detach().contiguous() for k, v in fields.items()},
        batch_size=[len(meta.keys)],
    )
    dp_client.kv_batch_put(
        keys=meta.keys, partition_id=meta.partition_id, fields=td,
    )

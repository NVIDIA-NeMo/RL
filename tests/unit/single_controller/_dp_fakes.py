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

"""Shared TQ data-plane fakes for single_controller tests."""

from __future__ import annotations

from typing import Any

import ray
import torch
from tensordict import TensorDict

from nemo_rl.data_plane.adapters.noop import NoOpDataPlaneClient

_PARTITION_ID = "rollout_data"
# TQReplayBuffer.add writes these training-row fields per prompt;
# _advantage_stage writes ``advantages`` back on top.
_BULK_FIELDS = [
    "input_ids",
    "input_lengths",
    "generation_logprobs",
    "token_mask",
    "sample_mask",
    "prompt_ids_for_adv",
    "total_reward",
]
_ADV_FIELD = "advantages"


@ray.remote(num_cpus=0)  # pragma: no cover
class _TQActor:
    """Ray-wrapped NoOpDataPlaneClient for cross-process TQ inspection."""

    def __init__(
        self,
        partition_id: str,
        fields: list[str],
        num_samples: int,
        consumer_tasks: list[str],
    ) -> None:
        self._client = NoOpDataPlaneClient()
        self._client.register_partition(
            partition_id=partition_id,
            fields=list(fields),
            num_samples=int(num_samples),
            consumer_tasks=list(consumer_tasks),
        )

    def put_samples(
        self,
        sample_ids: list[str],
        partition_id: str,
        fields: TensorDict | None = None,
        tags: list[dict[str, Any]] | None = None,
    ) -> Any:
        return self._client.put_samples(
            sample_ids=sample_ids,
            partition_id=partition_id,
            fields=fields,
            tags=tags,
        )

    def get_samples(
        self,
        sample_ids: list[str],
        partition_id: str,
        select_fields: list[str],
    ) -> TensorDict:
        return self._client.get_samples(
            sample_ids=sample_ids,
            partition_id=partition_id,
            select_fields=list(select_fields),
        )

    def clear_samples(self, sample_ids: list[str], partition_id: str) -> Any:
        return self._client.clear_samples(
            sample_ids=sample_ids, partition_id=partition_id
        )

    def claim_meta(self, **kwargs: Any) -> Any:
        return self._client.claim_meta(**kwargs)

    def get_tags(
        self, partition_id: str, sample_ids: list[str]
    ) -> list[dict[str, Any]]:
        rec = self._client._partitions[partition_id]
        return [dict(rec.tags.get(sid, {})) for sid in sample_ids]

    def peek_count(self, partition_id: str) -> int:
        return len(self._client._partitions[partition_id].rows)


class _SyncDPAdapter:
    """Sync DataPlaneClient over a Ray actor handle. Pads nested tensors before transport."""

    def __init__(self, handle: Any) -> None:
        self._handle = handle

    def put_samples(
        self,
        sample_ids: list[str],
        partition_id: str,
        fields: TensorDict | None = None,
        tags: list[dict[str, Any]] | None = None,
    ) -> Any:
        if fields is not None:
            fields = self._padded(fields)
        return ray.get(
            self._handle.put_samples.remote(
                sample_ids=sample_ids,
                partition_id=partition_id,
                fields=fields,
                tags=tags,
            )
        )

    def get_samples(
        self,
        sample_ids: list[str],
        partition_id: str,
        select_fields: list[str],
    ) -> TensorDict:
        return ray.get(
            self._handle.get_samples.remote(
                sample_ids=sample_ids,
                partition_id=partition_id,
                select_fields=select_fields,
            )
        )

    def clear_samples(self, sample_ids: list[str], partition_id: str) -> Any:
        return ray.get(
            self._handle.clear_samples.remote(
                sample_ids=sample_ids, partition_id=partition_id
            )
        )

    @staticmethod
    def _padded(td: TensorDict) -> TensorDict:
        out: dict[str, torch.Tensor] = {}
        for k in td.keys():
            v = td.get(k)
            if isinstance(v, torch.Tensor) and v.is_nested:
                v = torch.nested.to_padded_tensor(v, padding=0)
            out[k] = v
        return TensorDict(out, batch_size=td.batch_size)

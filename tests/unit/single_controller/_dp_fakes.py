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

from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.adapters.noop import NoOpDataPlaneClient
from nemo_rl.data_plane.schema import GROUP_ID_TAG

_PARTITION_ID = "rollout_data"
# TQReplayBuffer.commit writes these training-row fields per prompt;
# _advantage_stage writes ``advantages`` back on top.
_BULK_FIELDS = [
    "input_ids",
    "input_lengths",
    "generation_logprobs",
    "token_mask",
    "sample_mask",
    "mask_sample",
    "truncated",
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

    def get_tags(
        self, partition_id: str, sample_ids: list[str]
    ) -> list[dict[str, Any]]:
        rec = self._client._partitions[partition_id]
        return [dict(rec.tags.get(sid, {})) for sid in sample_ids]

    def peek_count(self, partition_id: str) -> int:
        return len(self._client._partitions[partition_id].rows)

    def list_sample_ids(self, partition_id: str) -> list[str]:
        return list(self._client._partitions[partition_id].rows.keys())


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

    def list_sample_ids(self, partition_id: str) -> list[str]:
        return ray.get(self._handle.list_sample_ids.remote(partition_id))

    @staticmethod
    def _padded(td: TensorDict) -> TensorDict:
        out: dict[str, torch.Tensor] = {}
        for k in td.keys():
            v = td.get(k)
            if isinstance(v, torch.Tensor) and v.is_nested:
                v = torch.nested.to_padded_tensor(v, padding=0)
            out[k] = v
        return TensorDict(out, batch_size=td.batch_size)


class _ReadyGroupBuffer:
    """Replay-buffer surface the samplers read, holding complete ready groups."""

    def __init__(self, generations: int) -> None:
        self._generations = generations
        self._groups_added = 0
        self.meta_list: list[KVBatchMeta] = []
        self.target_step_list: list[int] = []
        self.ready_list: list[bool] = []

    def __len__(self) -> int:
        return len(self.meta_list)

    @property
    def start_weight_list(self) -> list[int]:
        return [0] * len(self.meta_list)

    def training_owned_group_ids(self) -> set[str]:
        return set()

    def release_training_claims(self, group_ids: list[str]) -> None:
        del group_ids

    def add_groups(self, count: int) -> None:
        for _ in range(count):
            group = f"group-{self._groups_added}"
            self._groups_added += 1
            self.meta_list.append(
                KVBatchMeta(
                    partition_id=_PARTITION_ID,
                    task_name=None,
                    sample_ids=[f"{group}_g{i}" for i in range(self._generations)],
                    sequence_lengths=[16] * self._generations,
                    tags=[{"weight_version": 0, GROUP_ID_TAG: group}]
                    * self._generations,
                )
            )
            self.target_step_list.append(0)
            self.ready_list.append(True)

    async def claim_for_training(self, indices: list[int]) -> int:
        for index in sorted(indices, reverse=True):
            del self.meta_list[index]
            del self.target_step_list[index]
            del self.ready_list[index]
        return len(indices)

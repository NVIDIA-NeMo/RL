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
"""Lightweight functional test for the TQ writeback leader gate.

Pins the contract that ``TQWorkerMixin._write_back`` only fires on the
replica-group leader (``_is_replica_leader`` is True). This is the
``-601 ILLEGAL_CLIENT`` regression boundary: any non-leader sibling that
writes back duplicates the upsert and crashes the mooncake_cpu backend.

CPU-only, Ray-free — uses :class:`NoOpDataPlaneClient` and a tiny
mixin subclass that fakes ``_is_replica_leader``.
"""

from __future__ import annotations

import torch

from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.adapters.noop import NoOpDataPlaneClient
from nemo_rl.data_plane.worker_mixin import TQWorkerMixin
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


class _FakeWorker(TQWorkerMixin):
    def __init__(self, client: NoOpDataPlaneClient, *, is_leader: bool) -> None:
        self._dp_client = client
        self._is_leader = is_leader

    def _is_replica_leader(self) -> bool:  # type: ignore[override]
        return self._is_leader


def _seed_partition_with_one_sample(client: NoOpDataPlaneClient) -> KVBatchMeta:
    from nemo_rl.data_plane.column_io import write_columns

    client.register_partition(
        partition_id="train",
        fields=["input_ids", "input_lengths", "prev_logprobs"],
        num_samples=1,
        consumer_tasks=["train"],
    )
    meta = KVBatchMeta(
        partition_id="train",
        task_name="train",
        sample_ids=["s0"],
        fields=["input_ids", "input_lengths"],
        sequence_lengths=[4],
    )
    write_columns(
        client,
        meta,
        {
            "input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
            "input_lengths": torch.tensor([4], dtype=torch.long),
        },
    )
    return meta


def test_writeback_only_leader_writes():
    """Non-leader sibling write must NOT land — that's the -601 bug class."""
    client = NoOpDataPlaneClient()
    meta = _seed_partition_with_one_sample(client)

    leader = _FakeWorker(client, is_leader=True)
    sibling = _FakeWorker(client, is_leader=False)

    leader._write_back_result_field(
        meta,
        BatchedDataDict({"logprobs": torch.zeros(1, 4)}),
        result_key="logprobs",
        tq_field="prev_logprobs",
    )
    sibling._write_back_result_field(
        meta,
        BatchedDataDict({"logprobs": torch.full((1, 4), 99.0)}),
        result_key="logprobs",
        tq_field="prev_logprobs",
    )

    fetched = client.get_samples(
        sample_ids=meta.sample_ids,
        partition_id="train",
        select_fields=["prev_logprobs"],
    )
    assert torch.allclose(fetched["prev_logprobs"], torch.zeros(1, 4)), (
        "TQ holds a non-leader value — duplicate-writer condition that "
        "produces -601 ILLEGAL_CLIENT on the Mooncake backend."
    )


def test_writeback_single_worker_default_is_leader():
    """Single-process worker (no TP/CP/PP) is trivially a leader."""

    class _SingleWorker(TQWorkerMixin):
        def __init__(self, client: NoOpDataPlaneClient) -> None:
            self._dp_client = client

        def _local_coords(self) -> dict[str, int]:
            # No replicated axes — every axis check trivially True.
            return {}

    client = NoOpDataPlaneClient()
    meta = _seed_partition_with_one_sample(client)

    w = _SingleWorker(client)
    w._write_back_result_field(
        meta,
        BatchedDataDict({"logprobs": torch.full((1, 4), 7.5)}),
        result_key="logprobs",
        tq_field="prev_logprobs",
    )
    fetched = client.get_samples(
        sample_ids=meta.sample_ids,
        partition_id="train",
        select_fields=["prev_logprobs"],
    )
    assert torch.allclose(fetched["prev_logprobs"], torch.full((1, 4), 7.5))

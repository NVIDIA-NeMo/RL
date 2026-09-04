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

import ray
import torch

from nemo_rl.data.llm_message_utils import batched_message_log_to_flat_message
from nemo_rl.data.routed_experts import (
    RoutedExpertsBatch,
    RoutedExpertsTensorRef,
    materialize_routed_experts_inplace,
    offload_routed_experts_inplace,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.experience.rollouts import backfill_missing_routed_experts


def _fake_ray_store(monkeypatch):
    store = {}

    def put(value):
        key = f"ref-{len(store)}"
        store[key] = value.clone()
        return key

    def get(refs):
        if isinstance(refs, list):
            return [store[ref] for ref in refs]
        return store[refs]

    monkeypatch.setattr("nemo_rl.data.routed_experts.ray.put", put)
    monkeypatch.setattr("nemo_rl.data.routed_experts.ray.get", get)
    return store


def test_lazy_routed_experts_materialize_only_after_dp_sharding(monkeypatch):
    store = _fake_ray_store(monkeypatch)
    first_routes = torch.arange(2 * 3 * 2, dtype=torch.int16).reshape(2, 3, 2)
    second_routes = torch.arange(3 * 3 * 2, dtype=torch.int16).reshape(3, 3, 2)
    message_logs = [
        [
            {
                "role": "user",
                "token_ids": torch.tensor([1, 2]),
                "routed_experts": first_routes,
            },
            {"role": "assistant", "token_ids": torch.tensor([3])},
        ],
        [
            {
                "role": "assistant",
                "token_ids": torch.tensor([4, 5, 6]),
                "routed_experts": second_routes,
            }
        ],
    ]

    # Match the production replay shape: trajectory["batch"] is a
    # BatchedDataDict (UserDict), with message logs nested below it.
    trajectory = {
        "batch": BatchedDataDict({"message_log": message_logs}),
    }
    moved = offload_routed_experts_inplace(trajectory)
    assert moved == first_routes.nbytes + second_routes.nbytes
    assert len(store) == 2
    assert isinstance(message_logs[0][0]["routed_experts"], RoutedExpertsTensorRef)

    # Missing prompt routes use a metadata-only sentinel and allocate no tensor.
    backfill_missing_routed_experts(message_logs)
    assert len(store) == 2
    assert isinstance(message_logs[0][1]["routed_experts"], RoutedExpertsTensorRef)

    flat, lengths = batched_message_log_to_flat_message(
        message_logs, pad_value_dict={"token_ids": 0}
    )
    routes = flat["routed_experts"]
    assert isinstance(routes, RoutedExpertsBatch)
    assert lengths.tolist() == [3, 3]

    batch = BatchedDataDict(
        {
            "input_ids": flat["token_ids"],
            "input_lengths": lengths,
            "routed_experts": routes,
        }
    )
    shards = batch.shard_by_batch_size(2)
    assert all(isinstance(shard["routed_experts"], RoutedExpertsBatch) for shard in shards)

    first = shards[0]["routed_experts"].materialize()
    second = shards[1]["routed_experts"].materialize()
    assert tuple(first.shape) == (1, 3, 3, 2)
    assert torch.equal(first[0, :2], first_routes)
    assert torch.equal(first[0, 2], torch.full((3, 2), -1, dtype=torch.int16))
    assert torch.equal(second[0], second_routes)


def test_lazy_routed_experts_concat_and_reorder(monkeypatch):
    _fake_ray_store(monkeypatch)
    refs = [
        RoutedExpertsTensorRef.from_tensor(
            torch.full((tokens, 2, 1), value, dtype=torch.int16)
        )
        for tokens, value in ((1, 10), (2, 20), (3, 30))
    ]
    routes = RoutedExpertsBatch.concat(
        [RoutedExpertsBatch.from_message_segments([ref]) for ref in refs]
    )
    batch = BatchedDataDict(
        {
            "input_ids": torch.zeros(3, 3, dtype=torch.long),
            "input_lengths": torch.tensor([1, 2, 3]),
            "routed_experts": routes,
        }
    )
    batch.reorder_data([2, 0, 1])
    materialized = batch["routed_experts"].materialize()
    assert materialized[:, 0, 0, 0].tolist() == [20, 30, 10]


def test_materialization_is_scoped_to_sliced_microbatch(monkeypatch):
    store = _fake_ray_store(monkeypatch)
    refs = [
        RoutedExpertsTensorRef.from_tensor(
            torch.full((tokens, 2, 1), value, dtype=torch.int16)
        )
        for tokens, value in ((2, 10), (3, 20), (4, 30), (5, 40))
    ]
    batch = BatchedDataDict(
        {
            # Deliberately retain a much wider parent-style rectangular input.
            "input_ids": torch.zeros(4, 64, dtype=torch.long),
            "input_lengths": torch.tensor([2, 3, 4, 5]),
            "routed_experts": RoutedExpertsBatch.concat(
                [RoutedExpertsBatch.from_message_segments([ref]) for ref in refs]
            ),
        }
    )

    microbatch = batch.slice(0, 2)
    materialize_routed_experts_inplace(microbatch)

    routes = microbatch["routed_experts"]
    assert isinstance(routes, torch.Tensor)
    # Pad only to this microbatch's longest valid row, not input_ids.shape[1].
    assert tuple(routes.shape) == (2, 3, 2, 1)
    assert routes[:, 0, 0, 0].tolist() == [10, 20]
    # The two rows outside the microbatch remain lazy and untouched.
    assert isinstance(batch["routed_experts"], RoutedExpertsBatch)
    assert len(store) == 4


def test_nested_ray_references_do_not_materialize_on_driver():
    ray.init(num_cpus=1, include_dashboard=False, ignore_reinit_error=True)
    try:
        payload = {
            "message_log": [
                {
                    "token_ids": torch.tensor([1, 2]),
                    "routed_experts": torch.ones(2, 3, 2, dtype=torch.int16),
                }
            ]
        }
        offload_routed_experts_inplace(payload)
        round_trip = ray.get(ray.put(payload))
        nested = round_trip["message_log"][0]["routed_experts"]
        assert isinstance(nested, RoutedExpertsTensorRef)
        assert isinstance(nested.object_ref, ray.ObjectRef)
        assert torch.equal(nested.materialize(), torch.ones(2, 3, 2, dtype=torch.int16))
    finally:
        ray.shutdown()

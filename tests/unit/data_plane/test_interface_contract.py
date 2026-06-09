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
"""ABC contract test, parameterized over every adapter.

Every new adapter (TQ today, ``nv-dataplane`` later) must pass this. The
test runs against the NoOp adapter by default — it doesn't require TQ to
be installed, so CI exercises the contract on every push.
"""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from nemo_rl.data_plane import (
    DataPlaneClient,
    DataPlaneUnavailable,
    KVBatchMeta,
    build_data_plane_client,
)
from nemo_rl.data_plane.adapters.noop import NoOpDataPlaneClient


def _build_noop() -> DataPlaneClient:
    return NoOpDataPlaneClient()


@pytest.fixture(params=[_build_noop], ids=["noop"])
def client(request) -> DataPlaneClient:
    c = request.param()
    yield c
    c.close()


def test_factory_disabled_raises():
    """Factory has no NoOp fallback — disabled config must not reach it.
    The legacy trainer (grpo.grpo_train) never calls the factory at all."""
    with pytest.raises(ValueError):
        build_data_plane_client({"enabled": False, "impl": "transfer_queue"})


def test_factory_unknown_impl_raises():
    with pytest.raises(ValueError):
        build_data_plane_client({"enabled": True, "impl": "noop"})


def test_register_put_get_clear(client: DataPlaneClient):
    client.register_partition(
        partition_id="p", fields=["x"], num_samples=4, consumer_tasks=["read"]
    )
    keys = ["a", "b", "c", "d"]
    fields = TensorDict({"x": torch.arange(4)}, batch_size=[4])
    client.put_samples(sample_ids=keys, partition_id="p", fields=fields)

    out = client.get_samples(sample_ids=keys, partition_id="p", select_fields=["x"])
    assert torch.equal(out["x"], torch.arange(4))

    client.clear_samples(sample_ids=None, partition_id="p")
    with pytest.raises(KeyError):
        client.get_samples(sample_ids=keys, partition_id="p", select_fields=["x"])


def test_claim_meta_advances_consumption(client: DataPlaneClient):
    client.register_partition(
        partition_id="p",
        fields=["x"],
        num_samples=2,
        consumer_tasks=["read"],
    )
    fields = TensorDict({"x": torch.tensor([10, 20])}, batch_size=[2])
    client.put_samples(sample_ids=["a", "b"], partition_id="p", fields=fields)

    meta = client.claim_meta(
        partition_id="p", task_name="read", required_fields=["x"], batch_size=2
    )
    assert isinstance(meta, KVBatchMeta)
    assert meta.size == 2
    assert client.check_consumption_status("p", ["read"])


def test_recovery_metadata_is_non_consuming(client: DataPlaneClient):
    client.register_partition(
        partition_id="p",
        fields=["x"],
        num_samples=3,
        consumer_tasks=["read"],
    )
    fields = TensorDict({"x": torch.tensor([10, 20, 30])}, batch_size=[3])
    client.put_samples(
        sample_ids=["a", "b"],
        partition_id="p",
        fields=fields[:2],
        tags=[
            {
                "group_id": "g1",
                "weight_version": 2,
                "created_at": 100.0,
                "committed": True,
                "expected_num_keys": 2,
            },
            {
                "group_id": "g1",
                "weight_version": 2,
                "created_at": 100.0,
                "committed": True,
                "expected_num_keys": 2,
            },
        ],
    )
    client.put_samples(
        sample_ids=["c"],
        partition_id="p",
        fields=fields[2:],
        tags=[
            {
                "group_id": "g2",
                "weight_version": 3,
                "committed": False,
                "expected_num_keys": 1,
            }
        ],
    )

    first = {group.group_id: group for group in client.list_metadata("p")}
    second = {group.group_id: group for group in client.list_metadata("p")}

    assert first == second
    assert first["g1"].keys == ["a", "b"]
    assert first["g1"].weight_version == 2
    assert first["g1"].committed
    assert first["g1"].is_complete
    assert not first["g2"].committed
    assert client.depth("p") == 1

    meta = client.claim_meta(
        partition_id="p", task_name="read", required_fields=["x"], batch_size=3
    )
    assert meta.sample_ids == ["a", "b", "c"]


def test_recovery_pop_and_evict_remove_keys(client: DataPlaneClient):
    client.register_partition(
        partition_id="p",
        fields=["x"],
        num_samples=3,
        consumer_tasks=["read"],
    )
    client.put_samples(
        sample_ids=["a", "b", "c"],
        partition_id="p",
        fields=TensorDict({"x": torch.tensor([1, 2, 3])}, batch_size=[3]),
        tags=[
            {"group_id": "trained", "committed": True},
            {"group_id": "trained", "committed": True},
            {"group_id": "stale", "committed": False},
        ],
    )

    client.pop(keys=["a", "b"], partition_id="p")
    assert [group.group_id for group in client.list_metadata("p")] == ["stale"]

    client.evict(keys=["c"], partition_id="p")
    assert client.list_metadata("p") == []


def test_ping_reports_closed_client(client: DataPlaneClient):
    client.close()
    with pytest.raises(DataPlaneUnavailable):
        client.ping()


def test_get_data_requires_field_selection(client: DataPlaneClient):
    """P2 — silently fetching all fields is forbidden."""
    client.register_partition(
        partition_id="p", fields=["x"], num_samples=1, consumer_tasks=["read"]
    )
    client.put_samples(
        sample_ids=["a"],
        partition_id="p",
        fields=TensorDict({"x": torch.tensor([1])}, batch_size=[1]),
    )
    bare = KVBatchMeta(partition_id="p", task_name=None, sample_ids=["a"], fields=None)
    with pytest.raises(ValueError):
        client.get_data(bare)


def test_kv_batch_put_rejects_non_tensor_leaves(client: DataPlaneClient):
    """P3 — adapter must reject non-tensor leaves in the fields TensorDict.

    Uses ``NonTensorData`` (the supported tensordict primitive for
    storing arbitrary Python objects in a TensorDict) — a plain string
    in a regular TensorDict construction silently disappears in some
    tensordict versions, so we'd never reach the validator.
    """
    NonTensorData = pytest.importorskip("tensordict").NonTensorData
    client.register_partition(
        partition_id="p", fields=["x"], num_samples=1, consumer_tasks=["read"]
    )
    bad = TensorDict({"x": NonTensorData("hello")}, batch_size=[1])
    with pytest.raises(TypeError, match=r"non-tensor"):
        client.put_samples(sample_ids=["a"], partition_id="p", fields=bad)


def test_close_is_idempotent(client: DataPlaneClient):
    client.close()
    client.close()

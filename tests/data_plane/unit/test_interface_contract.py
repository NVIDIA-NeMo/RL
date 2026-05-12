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
    client.kv_batch_put(keys=keys, partition_id="p", fields=fields)

    out = client.kv_batch_get(keys=keys, partition_id="p", select_fields=["x"])
    assert torch.equal(out["x"], torch.arange(4))

    client.kv_clear(keys=None, partition_id="p")
    with pytest.raises(KeyError):
        client.kv_batch_get(keys=keys, partition_id="p", select_fields=["x"])


def test_claim_meta_advances_consumption(client: DataPlaneClient):
    client.register_partition(
        partition_id="p",
        fields=["x"],
        num_samples=2,
        consumer_tasks=["read"],
    )
    fields = TensorDict({"x": torch.tensor([10, 20])}, batch_size=[2])
    client.kv_batch_put(keys=["a", "b"], partition_id="p", fields=fields)

    meta = client.claim_meta(
        partition_id="p", task_name="read", required_fields=["x"], batch_size=2
    )
    assert isinstance(meta, KVBatchMeta)
    assert meta.size == 2
    assert client.check_consumption_status("p", ["read"])


def test_get_data_requires_field_selection(client: DataPlaneClient):
    """P2 — silently fetching all fields is forbidden."""
    client.register_partition(
        partition_id="p", fields=["x"], num_samples=1, consumer_tasks=["read"]
    )
    client.kv_batch_put(
        keys=["a"],
        partition_id="p",
        fields=TensorDict({"x": torch.tensor([1])}, batch_size=[1]),
    )
    bare = KVBatchMeta(partition_id="p", task_name=None, keys=["a"], fields=None)
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
        client.kv_batch_put(keys=["a"], partition_id="p", fields=bad)


def test_close_is_idempotent(client: DataPlaneClient):
    client.close()
    client.close()

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
test runs against the NoOp and process-local adapters — neither requires TQ
to be installed, so CI exercises the contract on every push.

Only rules the ABC states belong in the parameterized tests. Adapter-specific
behavior goes in a non-parameterized test that names the adapter, like the
NoOp ones at the bottom of this file.
"""

from __future__ import annotations

import pickle
from dataclasses import replace

import pytest
import torch
from tensordict import TensorDict

from nemo_rl.data_plane import (
    DataPlaneClient,
    KVBatchMeta,
    build_data_plane_client,
)
from nemo_rl.data_plane.adapters.local import (
    LocalDataPlaneClient,
    local_batch_to_tensordict,
    materialize_local,
)
from nemo_rl.data_plane.adapters.noop import NoOpDataPlaneClient
from nemo_rl.data_plane.interfaces import LocalDataPlaneConfig


def _build_noop() -> DataPlaneClient:
    return NoOpDataPlaneClient()


def _build_local() -> DataPlaneClient:
    return LocalDataPlaneClient(LocalDataPlaneConfig(max_partitions=2))


@pytest.fixture(params=[_build_noop, _build_local], ids=["noop", "local"])
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
    # ``keys`` is already sorted; the ABC does not pin the order, and the local
    # adapter returns sorted IDs while NoOp returns insertion order.
    assert client.list_sample_ids("p") == keys

    client.clear_samples(sample_ids=None, partition_id="p")
    assert client.list_sample_ids("p") == []
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


def test_get_data_requires_field_selection(client: DataPlaneClient):
    """P2 — silently fetching all fields is forbidden."""
    client.register_partition(
        partition_id="p", fields=["x"], num_samples=1, consumer_tasks=["read"]
    )
    meta = client.put_samples(
        sample_ids=["a"],
        partition_id="p",
        fields=TensorDict({"x": torch.tensor([1])}, batch_size=[1]),
    )
    # Reuse the returned meta rather than building a bare one. A bare meta
    # carries no adapter-private extra_info, so the local adapter would reject
    # it as stale before it ever reached the field-selection check.
    bare = replace(meta, task_name=None, fields=None)
    with pytest.raises(ValueError):
        client.get_data(bare)


def test_close_is_idempotent(client: DataPlaneClient):
    client.close()
    client.close()


def test_checkpoint_round_trip_restores_data_and_consumption(tmp_path) -> None:
    checkpoint_dir = tmp_path / "data-plane"
    source = NoOpDataPlaneClient()
    source.register_partition(
        partition_id="p",
        fields=["x"],
        num_samples=3,
        consumer_tasks=["train"],
    )
    source.put_samples(
        sample_ids=["a", "b", "c"],
        partition_id="p",
        fields=TensorDict({"x": torch.tensor([10, 20, 30])}, batch_size=[3]),
    )
    consumed = source.claim_meta(
        partition_id="p",
        task_name="train",
        required_fields=["x"],
        batch_size=1,
    )
    source.save_checkpoint(checkpoint_dir, metadata={"step": 7})

    restored = NoOpDataPlaneClient()
    metadata = restored.load_checkpoint(checkpoint_dir)
    assert metadata == {"step": 7}
    data = restored.get_samples(
        sample_ids=["a", "b", "c"],
        partition_id="p",
        select_fields=["x"],
    )
    assert torch.equal(data["x"], torch.tensor([10, 20, 30]))

    remaining = restored.claim_meta(
        partition_id="p",
        task_name="train",
        required_fields=["x"],
        batch_size=3,
    )
    assert consumed.sample_ids[0] not in remaining.sample_ids
    assert set(consumed.sample_ids + remaining.sample_ids) == {"a", "b", "c"}
    assert restored.check_consumption_status("p", ["train"])

    source.close()
    restored.close()


def test_checkpoint_load_requires_clean_client(tmp_path) -> None:
    checkpoint_dir = tmp_path / "data-plane"
    source = NoOpDataPlaneClient()
    source.save_checkpoint(checkpoint_dir)

    source.register_partition(
        partition_id="already-used",
        fields=["x"],
        num_samples=1,
        consumer_tasks=["train"],
    )
    with pytest.raises(RuntimeError, match="clean data-plane client"):
        source.load_checkpoint(checkpoint_dir)
    source.close()


def test_noop_put_rejects_non_tensor_leaves() -> None:
    """NoOp-only. Rejecting non-tensor leaves is not a data-plane rule.

    ``interfaces`` requires adapters to accept ``NonTensorStack`` /
    ``NonTensorData`` leaves, and both TQ and the local adapter do. NoOp stores
    rows as plain tensor dicts with no encoder behind it, so it rejects them.

    Uses ``NonTensorData`` (the supported tensordict primitive for storing
    arbitrary Python objects in a TensorDict) — a plain string in a regular
    TensorDict construction silently disappears in some tensordict versions,
    so we'd never reach the validator.
    """
    NonTensorData = pytest.importorskip("tensordict").NonTensorData
    client = NoOpDataPlaneClient()
    client.register_partition(
        partition_id="p", fields=["x"], num_samples=1, consumer_tasks=["read"]
    )
    bad = TensorDict({"x": NonTensorData("hello")}, batch_size=[1])
    with pytest.raises(TypeError, match=r"non-tensor"):
        client.put_samples(sample_ids=["a"], partition_id="p", fields=bad)
    client.close()


def test_local_accepts_non_tensor_leaves() -> None:
    """The other side of the rule above: local must keep non-tensor columns."""
    client = _build_local()
    client.register_partition(
        partition_id="p", fields=["x"], num_samples=2, consumer_tasks=["read"]
    )
    fields = local_batch_to_tensordict({"x": ["hello", "world"]}, batch_size=2)
    client.put_samples(sample_ids=["a", "b"], partition_id="p", fields=fields)

    out = client.get_samples(
        sample_ids=["a", "b"], partition_id="p", select_fields=["x"]
    )
    assert materialize_local(out)["x"] == ["hello", "world"]
    client.close()


def test_local_checkpoint_round_trip_restores_partitions(tmp_path) -> None:
    """save_checkpoint and load_checkpoint must agree on the state filename."""
    checkpoint_dir = tmp_path / "data-plane"
    source = _build_local()
    source.register_partition(
        partition_id="p", fields=["x"], num_samples=3, consumer_tasks=["train"]
    )
    source.put_samples(
        sample_ids=["a", "b", "c"],
        partition_id="p",
        fields=TensorDict({"x": torch.tensor([10, 20, 30])}, batch_size=[3]),
    )
    source.save_checkpoint(checkpoint_dir, metadata={"step": 7})

    restored = _build_local()
    assert restored.load_checkpoint(checkpoint_dir) == {"step": 7}
    data = restored.get_samples(
        sample_ids=["a", "b", "c"], partition_id="p", select_fields=["x"]
    )
    assert torch.equal(data["x"], torch.tensor([10, 20, 30]))

    # A partition registered after the restore must not reuse a generation the
    # checkpointed metas already hold.
    restored.register_partition(
        partition_id="q", fields=["x"], num_samples=1, consumer_tasks=["train"]
    )
    assert restored.list_sample_ids("p") == ["a", "b", "c"]

    source.close()
    restored.close()


def test_noop_checkpoint_load_explains_missing_partitions(tmp_path) -> None:
    checkpoint_dir = tmp_path / "data-plane"
    checkpoint_dir.mkdir()
    checkpoint_file = checkpoint_dir / "noop_state.pkl"
    with checkpoint_file.open("wb") as state_file:
        pickle.dump({"metadata": {}}, state_file)

    client = NoOpDataPlaneClient()
    with pytest.raises(ValueError) as exc_info:
        client.load_checkpoint(checkpoint_dir)

    message = str(exc_info.value)
    assert str(checkpoint_file) in message
    assert "has no 'partitions' key" in message
    assert "Delete it and re-run the test" in message

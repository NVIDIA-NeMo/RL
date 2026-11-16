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
"""Index-addressed NixlStore: rows live on unit ``gidx % N``, the TQ controller
stores no locations, clears are row-wide, and a new client needs no setup."""

from __future__ import annotations

import pytest
import ray
import torch
from tensordict import TensorDict

from tests.unit.data_plane._nixl_helpers import dense

pytestmark = pytest.mark.nixl


def _client():
    import transfer_queue as tq

    return tq.get_client().storage_manager.storage_client


def _units(conf):
    ns = conf.backend.NixlStore.namespace
    n = conf.backend.NixlStore.num_storage_units
    return [ray.get_actor(f"NixlStorageUnit#{i}", namespace=ns) for i in range(n)]


def _rows(conf) -> list[int]:
    return [ray.get(u.stats.remote())["rows"] for u in _units(conf)]


def test_put_returns_no_meta_and_rows_live_on_their_owner(tq_system):
    import transfer_queue as tq

    c = _client()
    assert c.put(["123456@x"], [torch.ones(3)]) is None  # nothing for the controller
    c.clear_rows([123456])
    before = _rows(tq_system)
    keys = [f"own{i}" for i in range(8)]
    tq.kv_batch_put(
        keys, "p", fields=TensorDict({"a": torch.randn(8, 4)}, batch_size=[8])
    )
    after = _rows(tq_system)
    assert [a - b for a, b in zip(after, before)] == [4, 4]  # gidx % 2 splits evenly
    tq.kv_clear(keys, "p")
    assert _rows(tq_system) == before


def test_clear_drops_fields_only_some_rows_have(tq_system):
    import transfer_queue as tq

    before = _rows(tq_system)
    blobs = sum(ray.get(u.stats.remote())["blobs"] for u in _units(tq_system))
    tq.kv_batch_put(
        ["h0", "h1"],
        "p",
        fields=TensorDict(
            {"a": torch.ones(2, 4), "b": torch.ones(2, 4)}, batch_size=[2]
        ),
    )
    tq.kv_batch_put(
        ["h2"], "p", fields=TensorDict({"a": torch.ones(1, 4)}, batch_size=[1])
    )
    # TQ's KV clear would name only "a", the field all three rows share.
    tq.kv_clear(["h0", "h1", "h2"], "p")
    assert _rows(tq_system) == before
    assert sum(ray.get(u.stats.remote())["blobs"] for u in _units(tq_system)) == blobs


def test_overwrite_replaces_the_value_and_frees_the_old_blob(tq_system):
    import transfer_queue as tq

    keys = ["ow0", "ow1"]
    tq.kv_batch_put(
        keys, "p", fields=TensorDict({"a": torch.zeros(2, 4)}, batch_size=[2])
    )
    blobs = sum(ray.get(u.stats.remote())["blobs"] for u in _units(tq_system))
    new = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    tq.kv_batch_put(keys, "p", fields=TensorDict({"a": new}, batch_size=[2]))
    assert sum(ray.get(u.stats.remote())["blobs"] for u in _units(tq_system)) == blobs
    assert torch.equal(dense(tq.kv_batch_get(keys, "p", select_fields=["a"])["a"]), new)
    tq.kv_clear(keys, "p")


def test_a_client_that_joins_later_reads_and_clears_without_setup(tq_system):
    from nemo_rl.data_plane.adapters.tq_nixl import NixlKVClient

    a = _client()
    v = torch.arange(16, dtype=torch.float32)
    a.put(["777001@x", "777002@x"], [v, v * 2])
    b = NixlKVClient(dict(tq_system.backend.NixlStore))  # a new process would do this
    try:
        got = b.get(["777001@x", "777002@x"])
        assert torch.equal(got[0], v) and torch.equal(got[1], v * 2)
        b.clear_rows([777001, 777002])
    finally:
        b.close()
    from nemo_rl.data_plane.nixl.errors import DataPlaneLostKeys

    with pytest.raises(DataPlaneLostKeys):
        a.get(["777001@x"])

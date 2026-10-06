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
"""Unit placement like SimpleStorage / PR #4465, writer placement by policy."""

from __future__ import annotations

import pytest
import ray
import torch
from tensordict import TensorDict

pytestmark = pytest.mark.nixl


@pytest.fixture(scope="module")
def per_node_spread_system(ray_cluster):
    import transfer_queue as tq

    import nemo_rl.data_plane.nixl.tq  # noqa: F401
    from nemo_rl.data_plane.nixl.tq.bootstrap_provider import alive_node_ids
    from tests.unit.data_plane.nixl._helpers import make_tq_conf

    conf = make_tq_conf(namespace="nvdp-placement")
    nixl = conf.backend.NixlStore
    nixl.pop("num_storage_units")
    nixl.storage_units_per_node = 3  # PR #4465 style: N units on each selected node
    nixl.node_ids = alive_node_ids()  # caller resolves all|inference|train to node ids
    nixl.placement = {"policy": "spread"}  # SimpleStorage style even split
    tq.init(conf)
    yield conf
    tq.close()
    nemo_rl.data_plane.nixl.tq.shutdown(conf)


def _stats(conf, n):
    ns = conf.backend.NixlStore.namespace
    return [
        ray.get(ray.get_actor(f"NixlStorageUnit#{i}", namespace=ns).stats.remote())
        for i in range(n)
    ]


def test_units_per_node_and_spread(per_node_spread_system):
    import transfer_queue as tq

    conf = per_node_spread_system
    n_nodes = len(conf.backend.NixlStore.node_ids)
    n_units = 3 * n_nodes
    stats = _stats(conf, n_units)
    assert len(stats) == n_units and all(s["blobs"] == 0 for s in stats)
    ns = conf.backend.NixlStore.namespace
    with pytest.raises(ValueError):
        ray.get_actor(f"NixlStorageUnit#{n_units}", namespace=ns)  # no extra units

    keys = []
    for i in range(6 * n_nodes):
        k = [f"sp{i}_g0"]
        tq.kv_batch_put(
            k, "train", fields=TensorDict({"t": torch.tensor([[i]])}, batch_size=[1])
        )
        keys += k
    stats = _stats(conf, n_units)
    assert [s["blobs"] for s in stats] == [2] * n_units, stats  # even split, one writer
    out = tq.kv_batch_get(keys, "train", select_fields=["t"])
    t = out["t"]
    t = torch.stack(list(t.unbind())) if t.is_nested else t
    assert torch.equal(t.flatten(), torch.arange(6 * n_nodes))
    tq.kv_clear(keys, "train")
    assert all(s["blobs"] == 0 for s in _stats(conf, n_units))

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
from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest

from nemo_rl.distributed import virtual_cluster as vc


@pytest.fixture
def cluster(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[vc.RayVirtualCluster, MagicMock]]:
    parent = vc.RayVirtualCluster(
        [4] * 8, num_gpus_per_node=4, max_colocated_worker_groups=2
    )
    group = MagicMock(bundle_specs=[{"GPU": 1}] * 32)
    parent._node_placement_groups = [group]
    parent._sorted_bundle_indices = list(range(16, 32)) + list(range(16))
    removed = MagicMock()
    monkeypatch.setattr(vc.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(vc, "remove_placement_group", removed)
    monkeypatch.setattr(
        vc,
        "placement_group_table",
        lambda _: {"bundles_to_node_id": {i: f"node{i // 4}" for i in range(32)}},
    )
    yield parent, removed
    parent.shutdown()


def test_subset_borrows_parent_resources_in_requested_rank_order(cluster) -> None:
    parent, removed = cluster
    view = parent.subset(list(range(16, 32)))
    assert view.world_size() == 16 and view.node_count() == 4
    assert parent.world_size() == 32
    assert view._sorted_bundle_indices == list(range(16))
    assert view.get_node_ids() == ["node0", "node1", "node2", "node3"]
    assert view.get_placement_groups() == parent.get_placement_groups()
    view.shutdown()
    removed.assert_not_called()
    assert parent.get_placement_groups()
    parent.shutdown()
    removed.assert_called_once()


def test_subset_shares_master_port_reservations(
    cluster, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent, _ = cluster
    view = parent.subset([16, 17])
    parent._allocated_master_ports.add(9000)
    candidates = iter([("node0", 9000), ("node0", 9001)])
    monkeypatch.setattr(
        view, "get_available_address_and_port", lambda *_: next(candidates)
    )
    assert view.get_master_address_and_port() == ("node0", 9001)
    assert 9001 in parent._allocated_master_ports


@pytest.mark.parametrize("indices", [[], [0, 0], [-1], [32], [True], ["0"]])
def test_subset_rejects_invalid_gpu_indices(cluster, indices) -> None:
    parent, _ = cluster
    with pytest.raises(ValueError):
        parent.subset(indices)


def test_subset_does_not_allow_a_second_owner_layer(cluster) -> None:
    parent, _ = cluster
    view = parent.subset([16, 17])
    with pytest.raises(ValueError, match="owning"):
        view.subset([0])

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
from unittest.mock import MagicMock

import pytest

import nemo_rl.distributed.virtual_cluster as virtual_cluster
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster


class TestRayVirtualClusterBatchPorts:
    """Tests for batched address and port discovery."""

    class FakePortFinder:
        def __init__(self):
            self.refs = []

        def options(self, **kwargs):
            return self

        def remote(self, port_range_low, port_range_high):
            ref = f"ref-{len(self.refs)}"
            self.refs.append((ref, port_range_low, port_range_high))
            return ref

    @staticmethod
    def _make_cluster():
        cluster = RayVirtualCluster(bundle_ct_per_node_list=[2, 1], use_gpus=False)
        cluster._node_placement_groups = [
            MagicMock(bundle_specs=[{"CPU": 1}]),
            MagicMock(bundle_specs=[{"CPU": 1}]),
        ]
        return cluster

    @staticmethod
    def _patch_common_mocks(monkeypatch, port_finder, results_by_ref):
        def fake_get(refs):
            return [results_by_ref[ref] for ref in refs]

        ray_mock = MagicMock()
        ray_mock.get.side_effect = fake_get
        monkeypatch.setattr(virtual_cluster, "ray", ray_mock)
        monkeypatch.setattr(virtual_cluster, "_get_node_ip_and_free_port", port_finder)
        monkeypatch.setattr(
            virtual_cluster,
            "PlacementGroupSchedulingStrategy",
            lambda **kwargs: kwargs,
        )
        return ray_mock

    def test_get_available_address_and_port_delegates_to_batch(self):
        cluster = self._make_cluster()
        cluster.get_available_addresses_and_ports_batch = MagicMock(
            return_value=[("node-a", 25001)]
        )

        result = cluster.get_available_address_and_port(1, 2)

        assert result == ("node-a", 25001)
        cluster.get_available_addresses_and_ports_batch.assert_called_once_with(
            [(1, 2)]
        )

    def test_default_batch_size_uses_ray_get_fast_path(self, monkeypatch):
        cluster = self._make_cluster()
        port_finder = self.FakePortFinder()
        results_by_ref = {
            "ref-0": ("node-a", 25001),
            "ref-1": ("node-a", 25002),
            "ref-2": ("node-b", 25003),
        }
        ray_mock = self._patch_common_mocks(monkeypatch, port_finder, results_by_ref)
        ray_mock.wait.side_effect = AssertionError("fast path should not call ray.wait")

        results = cluster.get_available_addresses_and_ports_batch(
            [(0, 0), (0, 1), (1, 0)]
        )

        assert results == [
            ("node-a", 25001),
            ("node-a", 25002),
            ("node-b", 25003),
        ]
        ray_mock.wait.assert_not_called()

    def test_single_placement_group_uses_each_bundle(self, monkeypatch):
        cluster = self._make_cluster()
        cluster._node_placement_groups = cluster._node_placement_groups[:1]
        port_finder = self.FakePortFinder()
        results_by_ref = {
            "ref-0": ("node-a", 25001),
            "ref-1": ("node-a", 25002),
        }
        self._patch_common_mocks(monkeypatch, port_finder, results_by_ref)

        results = cluster.get_available_addresses_and_ports_batch([(0, 0), (0, 1)])

        assert results == [("node-a", 25001), ("node-a", 25002)]

    def test_raises_for_placement_group_without_bundles(self, monkeypatch):
        cluster = self._make_cluster()
        monkeypatch.setattr(
            cluster,
            "get_placement_groups",
            lambda: [MagicMock(bundle_specs=[])],
        )

        with pytest.raises(RuntimeError, match="No valid placement groups"):
            cluster.get_available_addresses_and_ports_batch([(0, 0)])

    def test_preserves_input_order_when_refs_complete_out_of_order(self, monkeypatch):
        cluster = self._make_cluster()
        port_finder = self.FakePortFinder()
        results_by_ref = {
            "ref-0": ("node-a", 25001),
            "ref-1": ("node-a", 25002),
            "ref-2": ("node-b", 25003),
        }

        def fake_wait(remaining, num_returns):
            ready = remaining[-num_returns:]
            remaining = remaining[:-num_returns]
            return ready, remaining

        ray_mock = self._patch_common_mocks(monkeypatch, port_finder, results_by_ref)
        ray_mock.wait.side_effect = fake_wait

        results = cluster.get_available_addresses_and_ports_batch(
            [(0, 0), (0, 1), (1, 0)], batch_size=1
        )

        assert results == [
            ("node-a", 25001),
            ("node-a", 25002),
            ("node-b", 25003),
        ]
        assert port_finder.refs == [
            ("ref-0", cluster.port_range_low, cluster.port_range_high),
            ("ref-1", cluster.port_range_low, cluster.port_range_high),
            ("ref-2", cluster.port_range_low, cluster.port_range_high),
        ]

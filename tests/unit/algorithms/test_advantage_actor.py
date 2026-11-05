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
"""Where the advantage actor pool is allowed to land."""

import nemo_rl.algorithms.advantage_actor as advantage_actor
from nemo_rl.algorithms.advantage_actor import _placement_node_ids

# NodeAffinitySchedulingStrategy validates node_id on construction: it decodes
# as a 28-byte hex string and rejects a nil result. Short labels parse to nil,
# so the pinning test cannot use them.
HEAD_ID = "11" * 28
COMPUTE_A_ID = "aa" * 28
COMPUTE_B_ID = "bb" * 28

HEAD = {
    "NodeID": HEAD_ID,
    "Alive": True,
    "Resources": {"CPU": 144.0, "ray_head": 1.0},
}
COMPUTE_A = {
    "NodeID": COMPUTE_A_ID,
    "Alive": True,
    "Resources": {"CPU": 144.0, "GPU": 4.0, "worker_units": 4.0},
}
COMPUTE_B = {
    "NodeID": COMPUTE_B_ID,
    "Alive": True,
    "Resources": {"CPU": 144.0, "GPU": 4.0, "worker_units": 4.0},
}


def _patch_cluster(monkeypatch, nodes) -> None:
    resources: dict[str, float] = {}
    for node in nodes:
        for key, value in node["Resources"].items():
            resources[key] = resources.get(key, 0.0) + value
    monkeypatch.setattr(advantage_actor.ray, "nodes", lambda: nodes)
    monkeypatch.setattr(advantage_actor.ray, "cluster_resources", lambda: resources)


class TestPlacementNodeIds:
    def test_dedicated_head_is_excluded(self, monkeypatch) -> None:
        _patch_cluster(monkeypatch, [HEAD, COMPUTE_A, COMPUTE_B])
        # The regression this guards: plain CPU actors have nothing stopping
        # them from packing onto the head, which still advertises all its CPUs.
        assert _placement_node_ids() == [COMPUTE_A_ID, COMPUTE_B_ID]

    def test_dead_nodes_are_excluded(self, monkeypatch) -> None:
        dead = {**COMPUTE_B, "Alive": False}
        _patch_cluster(monkeypatch, [HEAD, COMPUTE_A, dead])
        assert _placement_node_ids() == [COMPUTE_A_ID]

    def test_no_dedicated_head_returns_empty(self, monkeypatch) -> None:
        # Without a dedicated head there is no single node to steer away from,
        # so the pin list is empty and the caller spreads instead.
        _patch_cluster(monkeypatch, [COMPUTE_A, COMPUTE_B])
        assert _placement_node_ids() == []


class _FakeHandle:
    class _Ready:
        @staticmethod
        def remote() -> object:
            return object()

    __ray_ready__ = _Ready()


class _RecordingActorClass:
    """Stands in for the @ray.remote class to capture .options() kwargs."""

    def __init__(self) -> None:
        self.options_seen: list[dict] = []

    def options(self, **kwargs):
        self.options_seen.append(kwargs)
        return self

    def remote(self, *args, **kwargs) -> _FakeHandle:
        return _FakeHandle()


def _patch_actor(monkeypatch) -> _RecordingActorClass:
    recorder = _RecordingActorClass()
    monkeypatch.setattr(advantage_actor, "AdvantageActor", recorder)
    monkeypatch.setattr(advantage_actor.ray, "get", lambda refs: None)
    return recorder


class TestActorPlacementStrategy:
    def test_spreads_when_there_is_no_dedicated_head(self, monkeypatch) -> None:
        _patch_cluster(monkeypatch, [COMPUTE_A, COMPUTE_B])
        recorder = _patch_actor(monkeypatch)
        advantage_actor.create_advantage_actors(None, None, None, num_workers=3)
        # The regression this guards: Ray's hybrid default packs to 50% of a
        # node before spreading, and a CPU-only actor is far below that, so all
        # three would land on the driver's node next to the controller.
        assert [o["scheduling_strategy"] for o in recorder.options_seen] == [
            "SPREAD"
        ] * 3

    def test_pins_round_robin_off_the_head_when_one_exists(self, monkeypatch) -> None:
        _patch_cluster(monkeypatch, [HEAD, COMPUTE_A, COMPUTE_B])
        recorder = _patch_actor(monkeypatch)
        advantage_actor.create_advantage_actors(None, None, None, num_workers=3)
        pinned = [o["scheduling_strategy"].node_id for o in recorder.options_seen]
        assert pinned == [COMPUTE_A_ID, COMPUTE_B_ID, COMPUTE_A_ID]

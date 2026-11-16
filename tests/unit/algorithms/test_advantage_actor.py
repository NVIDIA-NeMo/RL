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
"""Where create_advantage_actors puts the pool, and that it waits for it."""

import nemo_rl.algorithms.advantage_actor as advantage_actor

# NodeAffinitySchedulingStrategy validates node_id on construction: it decodes
# as a 28-byte hex string and rejects a nil result.
HEAD_ID = "11" * 28
COMPUTE_A_ID = "aa" * 28
COMPUTE_B_ID = "bb" * 28

HEAD = {"NodeID": HEAD_ID, "Alive": True, "Resources": {"CPU": 144.0, "ray_head": 1.0}}
COMPUTE_A = {"NodeID": COMPUTE_A_ID, "Alive": True, "Resources": {"CPU": 144.0}}
COMPUTE_B = {"NodeID": COMPUTE_B_ID, "Alive": True, "Resources": {"CPU": 144.0}}


def _patch_cluster(monkeypatch, nodes) -> None:
    resources: dict[str, float] = {}
    for node in nodes:
        for key, value in node["Resources"].items():
            resources[key] = resources.get(key, 0.0) + value
    monkeypatch.setattr(advantage_actor.ray, "nodes", lambda: nodes)
    monkeypatch.setattr(advantage_actor.ray, "cluster_resources", lambda: resources)


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
        self.labels: list[str] = []

    def options(self, **kwargs):
        self.options_seen.append(kwargs)
        return self

    def remote(self, *args, label: str) -> _FakeHandle:
        self.labels.append(label)
        return _FakeHandle()


def _patch_actor(monkeypatch) -> tuple[_RecordingActorClass, list]:
    recorder = _RecordingActorClass()
    waited: list = []
    monkeypatch.setattr(advantage_actor, "AdvantageActor", recorder)
    monkeypatch.setattr(
        advantage_actor.ray, "get", lambda refs, timeout=None: waited.append(refs)
    )
    return recorder, waited


def test_pins_round_robin_off_the_head_and_waits(monkeypatch) -> None:
    _patch_cluster(monkeypatch, [HEAD, COMPUTE_A, COMPUTE_B])
    recorder, waited = _patch_actor(monkeypatch)
    actors = advantage_actor.create_advantage_actors(None, None, None, num_workers=3)
    pinned = [o["scheduling_strategy"].node_id for o in recorder.options_seen]
    assert pinned == [COMPUTE_A_ID, COMPUTE_B_ID, COMPUTE_A_ID]
    # Each actor's label is its shard id: shared ids would overwrite each
    # other's training dumps.
    assert recorder.labels == ["0", "1", "2"]
    assert len(actors) == 3 and len(waited) == 1 and len(waited[0]) == 3


def test_spreads_when_there_is_no_dedicated_head(monkeypatch) -> None:
    _patch_cluster(monkeypatch, [COMPUTE_A, COMPUTE_B])
    recorder, _ = _patch_actor(monkeypatch)
    advantage_actor.create_advantage_actors(None, None, None, num_workers=3)
    assert [o["scheduling_strategy"] for o in recorder.options_seen] == ["SPREAD"] * 3

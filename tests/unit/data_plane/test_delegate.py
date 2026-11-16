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
"""DataPlaneDelegate, start_cpu_actors and the off-head placement."""

from unittest.mock import MagicMock, call

import pytest

import nemo_rl.data_plane as data_plane_module
import nemo_rl.data_plane.delegate as delegate
from nemo_rl.data_plane.delegate import (
    DataPlaneDelegate,
    spread_off_head,
    start_cpu_actors,
)

HEAD_ID = "11" * 28
COMPUTE_A_ID = "aa" * 28
COMPUTE_B_ID = "bb" * 28
HEAD = {"NodeID": HEAD_ID, "Alive": True, "Resources": {"CPU": 8.0, "ray_head": 1.0}}
COMPUTE_A = {"NodeID": COMPUTE_A_ID, "Alive": True, "Resources": {"CPU": 8.0}}
COMPUTE_B = {"NodeID": COMPUTE_B_ID, "Alive": True, "Resources": {"CPU": 8.0}}


def _patch_cluster(monkeypatch: pytest.MonkeyPatch, nodes: list[dict]) -> None:
    resources: dict[str, float] = {}
    for node in nodes:
        for key, value in node["Resources"].items():
            resources[key] = resources.get(key, 0.0) + value
    monkeypatch.setattr(delegate.ray, "nodes", lambda: nodes)
    monkeypatch.setattr(delegate.ray, "cluster_resources", lambda: resources)


def test_spread_off_head_skips_the_head_and_dead_nodes(monkeypatch) -> None:
    dead = {**COMPUTE_B, "NodeID": "cc" * 28, "Alive": False}
    _patch_cluster(monkeypatch, [HEAD, COMPUTE_A, COMPUTE_B, dead])
    place = spread_off_head()
    pinned = [place(index) for index in range(3)]
    assert [s.node_id for s in pinned] == [COMPUTE_A_ID, COMPUTE_B_ID, COMPUTE_A_ID]
    assert all(s.soft for s in pinned)


def test_spread_off_head_spreads_without_a_dedicated_head(monkeypatch) -> None:
    _patch_cluster(monkeypatch, [COMPUTE_A, COMPUTE_B])
    assert spread_off_head()(0) == "SPREAD"


def test_start_waits_on_every_actor(monkeypatch) -> None:
    get = MagicMock()
    monkeypatch.setattr(delegate.ray, "get", get)
    made: list[int] = []

    def make(index: int) -> MagicMock:
        made.append(index)
        return MagicMock(name=f"actor{index}")

    actors = start_cpu_actors(make, n=2, ready="check_dependencies", timeout_s=5)
    assert made == [0, 1]
    get.assert_called_once_with(
        [actor.check_dependencies.remote.return_value for actor in actors], timeout=5
    )


def test_failed_start_kills_every_actor_even_if_a_kill_fails(
    monkeypatch, capsys
) -> None:
    actors = [MagicMock(), MagicMock()]
    kill = MagicMock(side_effect=[RuntimeError("already dead"), None])
    monkeypatch.setattr(delegate.ray, "kill", kill)
    monkeypatch.setattr(
        delegate.ray, "get", MagicMock(side_effect=RuntimeError("missing dep"))
    )
    with pytest.raises(RuntimeError, match="missing dep"):
        start_cpu_actors(lambda index: actors[index], n=2, ready="check")
    assert kill.call_args_list == [call(a, no_restart=True) for a in actors]
    assert "cpu actor cleanup failed: already dead" in capsys.readouterr().out


def test_start_rejects_an_empty_pool() -> None:
    with pytest.raises(ValueError, match="at least one actor"):
        start_cpu_actors(MagicMock(), n=0)


def test_delegate_builds_a_client_and_names_itself(monkeypatch) -> None:
    built = []
    monkeypatch.setattr(
        data_plane_module,
        "build_data_plane_client",
        lambda cfg, *, bootstrap: built.append((cfg, bootstrap)) or "client",
    )
    dispatch = MagicMock(return_value={"ok": True})
    monkeypatch.setattr(delegate, "run_checkpoint_command", dispatch)

    class Finalizer(DataPlaneDelegate):
        pass

    actor = Finalizer("cfg", label="3")
    assert built == [("cfg", False)]
    assert actor.dp_client == "client"
    assert repr(actor) == "Finalizer[3]"
    assert actor.mooncake_checkpoint({"operation": "INFO"}) == {"ok": True}
    dispatch.assert_called_once_with({"operation": "INFO"})
    # No metrics wrapper on the client: nothing to report.
    assert actor.get_data_plane_snapshot() is None

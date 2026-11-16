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
"""DataPlaneDelegate and DelegatePool: placement, start-up, free actors, storage."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call

import pytest

import nemo_rl.data_plane as data_plane_module
import nemo_rl.data_plane.delegate as delegate
from nemo_rl.data_plane.delegate import (
    DataPlaneDelegate,
    DelegatePool,
    on_this_node,
    spread_off_head,
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


class TestPlacement:
    def test_spread_off_head_skips_the_head_and_dead_nodes(self, monkeypatch) -> None:
        dead = {**COMPUTE_B, "NodeID": "cc" * 28, "Alive": False}
        _patch_cluster(monkeypatch, [HEAD, COMPUTE_A, COMPUTE_B, dead])
        place = spread_off_head()
        pinned = [place(index) for index in range(3)]
        assert [s.node_id for s in pinned] == [COMPUTE_A_ID, COMPUTE_B_ID, COMPUTE_A_ID]
        assert all(s.soft for s in pinned)

    def test_spread_off_head_spreads_without_a_dedicated_head(
        self, monkeypatch
    ) -> None:
        _patch_cluster(monkeypatch, [COMPUTE_A, COMPUTE_B])
        assert spread_off_head()(0) == "SPREAD"

    def test_on_this_node_is_a_hard_pin_to_the_callers_node(self, monkeypatch) -> None:
        monkeypatch.setattr(
            delegate.ray,
            "get_runtime_context",
            lambda: SimpleNamespace(get_node_id=lambda: COMPUTE_A_ID),
        )
        strategy = on_this_node()(0)
        assert strategy.node_id == COMPUTE_A_ID and strategy.soft is False


class _RecordingRemoteClass:
    def __init__(self) -> None:
        self.options_seen: list[dict] = []
        self.made: list[MagicMock] = []

    def options(self, **kwargs):
        self.options_seen.append(kwargs)
        return self

    def remote(self, *args, label: str) -> MagicMock:
        actor = MagicMock(name=f"actor{label}")
        actor.args, actor.label = args, label
        self.made.append(actor)
        return actor


class TestPoolStart:
    def test_start_applies_shared_options_placement_and_labels(self) -> None:
        remote_cls = _RecordingRemoteClass()
        pool = DelegatePool.start(
            remote_cls, "cfg", n=2, placement=lambda i: f"node{i}", max_concurrency=1
        )
        assert pool.size == pool.free == 2
        assert [a.label for a in pool.actors] == ["0", "1"]
        assert pool.actors[0].args == ("cfg",)
        assert remote_cls.options_seen == [
            {
                "num_cpus": 1,
                "num_gpus": 0,
                "max_restarts": 0,
                "max_task_retries": 0,
                "max_concurrency": 1,
                "scheduling_strategy": f"node{i}",
            }
            for i in range(2)
        ]

    def test_start_wraps_a_plain_class_with_ray_remote(self, monkeypatch) -> None:
        remote_cls = _RecordingRemoteClass()
        wrapped = []
        monkeypatch.setattr(
            delegate.ray, "remote", lambda cls: wrapped.append(cls) or remote_cls
        )

        class Plain(DataPlaneDelegate):
            pass

        DelegatePool.start(Plain, n=1)
        assert wrapped == [Plain]
        assert "scheduling_strategy" not in remote_cls.options_seen[0]

    def test_start_rejects_an_empty_pool(self) -> None:
        with pytest.raises(ValueError, match="at least one actor"):
            DelegatePool.start(_RecordingRemoteClass(), n=0)

    def test_failed_ready_check_kills_every_actor(self, monkeypatch) -> None:
        actors = [MagicMock(), MagicMock()]
        kill = MagicMock(side_effect=[RuntimeError("already dead"), None])
        monkeypatch.setattr(delegate.ray, "kill", kill)
        monkeypatch.setattr(
            delegate.ray, "get", MagicMock(side_effect=RuntimeError("missing dep"))
        )
        with pytest.raises(RuntimeError, match="missing dep"):
            DelegatePool(actors).wait_ready("check_dependencies", timeout_s=5)
        # One failed kill must not skip the others.
        assert kill.call_args_list == [call(a, no_restart=True) for a in actors]

    def test_async_ready_check_returns_replies_or_kills(self, monkeypatch) -> None:
        good = SimpleNamespace(ready=SimpleNamespace(remote=AsyncMock(return_value=7)))
        assert asyncio.run(DelegatePool([good]).wait_ready_async("ready")) == [7]

        bad = SimpleNamespace(
            ready=SimpleNamespace(remote=AsyncMock(side_effect=RuntimeError("boom")))
        )
        kill = MagicMock()
        monkeypatch.setattr(delegate.ray, "kill", kill)
        with pytest.raises(RuntimeError, match="boom"):
            asyncio.run(DelegatePool([bad]).wait_ready_async("ready", timeout_s=5))
        kill.assert_called_once_with(bad, no_restart=True)


def test_acquire_waits_for_a_released_actor() -> None:
    async def scenario() -> list[str]:
        pool = DelegatePool(["only"])
        order = []
        first = await pool.acquire()
        assert pool.free == 0

        async def second() -> None:
            order.append(await pool.acquire())

        waiter = asyncio.create_task(second())
        await asyncio.sleep(0)
        assert not order
        pool.release(first)
        await waiter
        return order

    assert asyncio.run(scenario()) == ["only"]


class TestClient:
    @pytest.mark.parametrize("owns_storage,segment_size", [(False, 0), (True, None)])
    def test_storage_choice_sets_segment_size(
        self, monkeypatch, owns_storage: bool, segment_size: int | None
    ) -> None:
        built = []
        monkeypatch.setattr(
            data_plane_module,
            "build_data_plane_client",
            lambda cfg, *, bootstrap, segment_size: built.append(
                (cfg, bootstrap, segment_size)
            )
            or "client",
        )
        dispatch = MagicMock(return_value={"ok": True})
        monkeypatch.setattr(delegate, "run_checkpoint_command", dispatch)

        class Writer(DataPlaneDelegate):
            pass

        Writer.owns_storage = owns_storage
        client = Writer("cfg", label="3")
        assert built == [("cfg", False, segment_size)]
        assert client.dp_client == "client"
        assert repr(client) == "Writer[3]"
        # Only a client that holds rows has anything to checkpoint.
        result = client.mooncake_checkpoint({"operation": "INFO"})
        assert (result == {"ok": True}) is owns_storage
        assert dispatch.called is owns_storage

    def test_process_must_be_overridden(self, monkeypatch) -> None:
        monkeypatch.setattr(
            data_plane_module, "build_data_plane_client", lambda *a, **k: None
        )
        with pytest.raises(NotImplementedError):
            DataPlaneDelegate("cfg").process("request")

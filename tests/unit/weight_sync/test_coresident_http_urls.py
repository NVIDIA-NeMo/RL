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
"""Exercise co-resident driver URL discovery without optional GPU backends."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import ray

from nemo_rl.weight_sync import megatron_weight_synchronizer as synchronizer


class WorkerGroup:
    def __init__(self, role: str, events: list[str]) -> None:
        self.role = role
        self.events = events

    def run_all_workers_single_data(self, method: str, **kwargs: object) -> list:
        self.events.append(method)
        return [(self.role, method, kwargs)]


def generation_type(ray_api: SimpleNamespace) -> type:
    """Load the actual driver methods while excluding unrelated backend imports."""
    source = (
        Path(__file__).resolve().parents[3]
        / "nemo_rl/models/generation/megatron/megatron_generation.py"
    )
    owner = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "MegatronGeneration"
    )
    names = {
        "prepare_for_generation",
        "refresh_dp_openai_server_base_urls",
        "verify_served_addresses",
    }
    methods = [
        node
        for node in owner.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in methods} == names
    tree = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            ast.ClassDef(
                name="MegatronGeneration",
                bases=[],
                keywords=[],
                body=methods,
                decorator_list=[],
            ),
        ],
        type_ignores=[],
    )
    namespace = {"ray": ray_api}
    exec(compile(ast.fix_missing_locations(tree), str(source), "exec"), namespace)
    return namespace["MegatronGeneration"]


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    events: list[str] = []
    urls = [None, "http://node1:8001/v1", None, "http://node0:8000/v1"]
    state = SimpleNamespace(
        ready=False, url_timeout=False, resume_failure=False, get_timeouts=[]
    )

    def get(refs: list, *, timeout: float | None = None) -> list:
        method = refs[0][1]
        if method == "prepare_for_generation":
            state.ready = True
            return [None]
        assert method == "report_dp_openai_server_base_url"
        assert state.ready, "URL collection preceded completed engine initialization"
        state.get_timeouts.append(timeout)
        if state.url_timeout:
            raise ray.exceptions.GetTimeoutError("HTTP URL query exceeded deadline")
        return urls

    def wait(refs: list, *, timeout_s: float | None = None) -> list:
        assert timeout_s == 17.0
        method = refs[0][1]
        if method == "coresident_refit_source_ready":
            return ["stable"]
        if method == "coresident_refit_copy":
            return [{"seconds": 0.1, "bytes": 0, "bytes_shared": 1024}]
        if method == "coresident_resume":
            if state.resume_failure:
                raise TimeoutError("engine initialization exceeded deadline")
            state.ready = True
        else:
            assert method == "coresident_pause"
        return [None]

    monkeypatch.setattr(synchronizer, "_get_fail_fast", wait)
    generation = generation_type(SimpleNamespace(get=get))()
    generation.cfg = {"mcore_generation_config": {"expose_http_server": True}}
    generation.dp_openai_server_base_urls = []
    generation.worker_group = WorkerGroup("engine", events)
    generation._policy = SimpleNamespace(worker_group=generation.worker_group)
    policy = SimpleNamespace(
        cfg={"generation": {"colocated": {"engine_weights": "shared"}}},
        worker_group=WorkerGroup("learner", events),
    )
    refit = synchronizer.CoresidentRefit(policy, generation, refit_timeout_s=17.0)
    refit._fingerprints = ["stable"]
    return SimpleNamespace(
        generation=generation, refit=refit, events=events, state=state, urls=urls
    )


def test_initial_refit_populates_native_urls_without_extra_prepare(
    world: SimpleNamespace,
) -> None:
    world.refit.sync_weights()
    assert world.events == [
        "coresident_pause",
        "coresident_refit_source_ready",
        "coresident_refit_copy",
        "coresident_resume",
        "report_dp_openai_server_base_url",
    ]
    expected = [world.urls[1], world.urls[3]]
    assert world.generation.dp_openai_server_base_urls == expected
    world.generation.verify_served_addresses(expected, expected[::-1])
    assert world.state.get_timeouts == [17.0]
    assert world.refit._engine_running


def test_later_refits_keep_cached_urls_without_requery(world: SimpleNamespace) -> None:
    world.refit.sync_weights()
    world.refit.sync_weights()
    assert world.events.count("coresident_resume") == 2
    assert world.events.count("report_dp_openai_server_base_url") == 1
    assert "prepare_for_generation" not in world.events


@pytest.mark.parametrize("cached", [False, True])
def test_http_disabled_or_cached_does_not_query(
    world: SimpleNamespace, cached: bool
) -> None:
    if cached:
        world.generation.dp_openai_server_base_urls = [world.urls[1]]
    else:
        world.generation.cfg["mcore_generation_config"]["expose_http_server"] = False
    world.refit.sync_weights()
    assert "report_dp_openai_server_base_url" not in world.events
    assert world.state.get_timeouts == []


def test_url_timeout_propagates_and_leaves_initialization_unpublished(
    world: SimpleNamespace,
) -> None:
    world.state.url_timeout = True
    with pytest.raises(ray.exceptions.GetTimeoutError, match="URL query"):
        world.refit.sync_weights()
    assert world.generation.dp_openai_server_base_urls == []
    assert not world.refit._engine_running
    assert world.state.get_timeouts == [17.0]


def test_failed_resume_does_not_collect_urls(world: SimpleNamespace) -> None:
    world.state.resume_failure = True
    with pytest.raises(TimeoutError, match="engine initialization"):
        world.refit.sync_weights()
    assert "report_dp_openai_server_base_url" not in world.events
    assert not world.refit._engine_running


def test_native_prepare_preserves_unbounded_default_and_url_order(
    world: SimpleNamespace,
) -> None:
    assert world.generation.prepare_for_generation(tags=["kv_cache"])
    assert world.generation.dp_openai_server_base_urls == [
        world.urls[1],
        world.urls[3],
    ]
    assert world.events == [
        "prepare_for_generation",
        "report_dp_openai_server_base_url",
    ]
    assert world.state.get_timeouts == [None]
    assert world.generation.prepare_for_generation()
    assert world.events.count("report_dp_openai_server_base_url") == 1


def test_native_address_verification_still_rejects_mismatch(
    world: SimpleNamespace,
) -> None:
    world.refit.sync_weights()
    with pytest.raises(RuntimeError, match="pre-published"):
        world.generation.verify_served_addresses(
            world.generation.dp_openai_server_base_urls, ["http://wrong:8000/v1"]
        )

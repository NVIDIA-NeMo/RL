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

"""Typed harness routing and the Gym-to-training receipt boundary."""

import asyncio
import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nemo_rl.environments.nemo_gym_harnesses import (
    HarnessRoute,
    attribute_harness_result,
    parse_harness_pool,
    resolve_harness_pool,
    route_harness_rows,
)
from nemo_rl.experience.failures import GymTransportError, RolloutDataFailure

ROUTES = (
    HarnessRoute("swe_opencode", "opencode_agent", "swe"),
    HarnessRoute("swe_pi", "pi_agent", "swe"),
)


def task(index=0):
    return {
        "task_id": {"taskset": "smoke", "task_id": "task-1"},
        "task_input": {
            "responses_create_params": {"input": "Fix the bug"},
            "task_data": {},
        },
        "_ng_group_id": "group-1",
        "_ng_group_attempt": 0,
        "_ng_rollout_index": index,
        "_ng_rollout_id": f"group-1_g{index}_afirst",
        "_rowidx": index,
    }


def episode(row, *, reward=0.75):
    return {
        "episode_id": {"rollout_id": row["_ng_rollout_id"], "attempt": 0},
        "task_id": copy.deepcopy(row["task_id"]),
        "result": {
            "responses_create_params": {"input": "Fix the bug"},
            "response": {
                "id": "response-1",
                "created_at": 0,
                "model": "policy",
                "object": "response",
                "output": [],
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
            },
            "reward": reward,
            "terminal_response_id": "response-1",
        },
    }


def _harness_result(row, reply):
    from nemo_rl.environments.nemo_gym import _normalize_nemo_gym_episode_result

    result = _normalize_nemo_gym_episode_result(row, reply, token_capture_enabled=True)
    return attribute_harness_result(row, result)


def resolved_config():
    config = {"swe": {"resources_servers": {"swe_rebench": {}}}}
    for route in ROUTES:
        config[route.agent_name] = {
            "responses_api_agents": {
                route.agent_name: {
                    "model_server": {
                        "type": "responses_api_models",
                        "name": "policy_model",
                    }
                }
            }
        }
        config[route.environment_name] = {
            "environment_servers": {
                "single_agent_turn": {
                    "agent_server": {
                        "type": "responses_api_agents",
                        "name": route.agent_name,
                    },
                    "resources_server": {"type": "resources_servers", "name": "swe"},
                }
            }
        }
    return config


def _harbor_config() -> dict:
    config = resolved_config()
    for route in ROUTES:
        config[route.environment_name]["environment_servers"] = {
            "harbor": {
                "agent_server": {
                    "type": "responses_api_agents",
                    "name": route.agent_name,
                },
                "tasksets": {
                    "smoke": {
                        "folder": "/datasets/harbor",
                        "tasks": {"task-1": "digest-1"},
                    }
                },
            }
        }
    return config


def _harbor_task(index: int) -> dict:
    row = task(index)
    row["task_input"] = {
        "task_data": {"_ng_digest": "digest-1"},
        "turns": [
            {
                "name": "fix",
                "responses_create_params": {"input": "Fix the bug"},
            }
        ],
    }
    return row


@pytest.mark.parametrize(
    "value", [[], "swe_pi", ["swe_pi", "swe_pi"], [""], [False], [["nested"]]]
)
def test_pool_rejects_invalid_names(value):
    with pytest.raises(ValueError, match="nonempty list of distinct"):
        parse_harness_pool(value)


@pytest.mark.parametrize(
    "value",
    [{}, {"": ["env"]}, {1: ["env"]}, {"smoke": []}, {"smoke": "env"}, {"smoke": None}],
)
def test_pool_rejects_invalid_taskset_mappings(value: object) -> None:
    with pytest.raises(ValueError):
        parse_harness_pool(value)


def test_pool_parses_taskset_lists_without_sharing_mutable_config() -> None:
    names = {"scale_swe": ["scale_pi", "scale_opencode"], "swe_rebench": ["rebench_pi"]}
    parsed = parse_harness_pool(names)
    assert parsed == names
    assert isinstance(parsed, dict)
    parsed["scale_swe"].append("another")
    assert names["scale_swe"] == ["scale_pi", "scale_opencode"]


def test_pool_resolution_preserves_order_and_validates_pairing():
    names = [r.environment_name for r in ROUTES]
    assert resolve_harness_pool(names, resolved_config()) == ROUTES
    assert resolve_harness_pool(names[::-1], resolved_config()) == ROUTES[::-1]
    config = resolved_config()
    config["swe_pi"]["environment_servers"]["single_agent_turn"]["resources_server"][
        "name"
    ] = "another"
    config["another"] = {"resources_servers": {"swe_rebench": {}}}
    with pytest.raises(ValueError, match="share one resources"):
        resolve_harness_pool(names, config)


def test_pool_rejects_flat_environment_and_repeated_agent():
    config = resolved_config()
    config["swe_pi"]["environment_servers"] = {"single_agent_turn_legacy": {}}
    with pytest.raises(ValueError, match="typed single_agent_turn"):
        resolve_harness_pool(["swe_pi"], config)
    config = resolved_config()
    config["swe_pi"]["environment_servers"]["single_agent_turn"]["agent_server"][
        "name"
    ] = "opencode_agent"
    with pytest.raises(ValueError, match="distinct agent"):
        resolve_harness_pool([r.environment_name for r in ROUTES], config)


@pytest.mark.parametrize(
    "agent_config",
    [
        {},
        {"model_server": None},
        {"model_server": {"type": "responses_api_models", "name": "external_model"}},
        {"model_server": {"type": "resources_servers", "name": "policy_model"}},
    ],
)
def test_pool_requires_each_harness_to_use_the_training_policy(agent_config):
    config = resolved_config()
    config["pi_agent"]["responses_api_agents"]["pi_agent"] = agent_config
    with pytest.raises(ValueError, match="responses_api_models:policy_model"):
        resolve_harness_pool([route.environment_name for route in ROUTES], config)


def test_pool_rejects_multiple_agent_configurations():
    config = resolved_config()
    agents = config["pi_agent"]["responses_api_agents"]
    agents["another_agent"] = copy.deepcopy(agents["pi_agent"])
    with pytest.raises(ValueError, match="exactly one configuration"):
        resolve_harness_pool([route.environment_name for route in ROUTES], config)


def test_taskset_pools_route_mixed_swe_sources_to_their_own_resources() -> None:
    config = resolved_config()
    config["scale"] = {"resources_servers": {"scale_swe": {}}}
    scale_routes = tuple(
        HarnessRoute(f"scale_{route.environment_name}", route.agent_name, "scale")
        for route in ROUTES
    )
    for original, route in zip(ROUTES, scale_routes):
        entry = copy.deepcopy(config[original.environment_name])
        entry["environment_servers"]["single_agent_turn"]["resources_server"][
            "name"
        ] = "scale"
        config[route.environment_name] = entry
    names = {
        "scale_swe": [route.environment_name for route in scale_routes],
        "swe_rebench": [route.environment_name for route in ROUTES],
    }
    resolved = resolve_harness_pool(names, config)
    assert resolved == {"scale_swe": scale_routes, "swe_rebench": ROUTES}
    rows = [task(index) for index in range(32)]
    for index, row in enumerate(rows):
        row["task_id"]["taskset"] = "scale_swe" if index % 2 else "swe_rebench"
    original_inputs = [copy.deepcopy(row["task_input"]) for row in rows]
    route_harness_rows(rows, resolved)
    for row, original_input in zip(rows, original_inputs):
        scale = row["task_id"]["taskset"] == "scale_swe"
        assert row["task_source"] == ("scale" if scale else "swe")
        assert row["_ng_environment_server"] in {
            route.environment_name for route in (scale_routes if scale else ROUTES)
        }
        assert row["task_input"] == original_input
    selected = [(row["_ng_environment_server"], row["agent_ref"]) for row in rows]
    retry = copy.deepcopy(rows[7:13])
    for row in retry:
        row["_ng_group_attempt"] += 1
        row["_ng_rollout_id"] += "-retry"
    route_harness_rows(retry, resolved)
    assert [
        (row["_ng_environment_server"], row["agent_ref"]) for row in retry
    ] == selected[7:13]


def test_taskset_pool_rejects_unknown_taskset_and_wrong_resources() -> None:
    names = {"smoke": [route.environment_name for route in ROUTES]}
    routes = resolve_harness_pool(names, resolved_config())
    row = task()
    row["task_id"]["taskset"] = "unconfigured"
    with pytest.raises(ValueError, match="No harness_pool.*unconfigured"):
        route_harness_rows([row], routes)
    row = task()
    row["task_source"] = "another_resources_server"
    with pytest.raises(ValueError, match="does not match harness resources"):
        route_harness_rows([row], routes)


@pytest.mark.parametrize("mapped", [False, True])
def test_harbor_pool_routes_opaque_tasks_without_resources_or_model_params(
    mapped: bool,
) -> None:
    names = [route.environment_name for route in ROUTES]
    routes = resolve_harness_pool(
        {"smoke": names} if mapped else names, _harbor_config()
    )
    rows = [_harbor_task(index) for index in range(32)]
    original_inputs = [copy.deepcopy(row["task_input"]) for row in rows]
    rows[0]["task_source"] = "original-harbor-source"
    route_harness_rows(rows, routes)
    assert {row["agent_ref"]["name"] for row in rows} == {"opencode_agent", "pi_agent"}
    assert rows[0]["task_source"] == "original-harbor-source"
    assert all(row["task_source"] == "smoke" for row in rows[1:])
    assert [row["task_input"] for row in rows] == original_inputs
    assert all(row["_ng_environment_server"] in names for row in rows)
    selected = [(row["_ng_environment_server"], row["agent_ref"]) for row in rows]
    retry = copy.deepcopy(rows[3:7])
    for row in retry:
        row["_ng_group_attempt"] += 1
        row["_ng_rollout_id"] += "-retry"
    route_harness_rows(retry, routes)
    assert [
        (row["_ng_environment_server"], row["agent_ref"]) for row in retry
    ] == selected[3:7]


@pytest.mark.parametrize("mapped", [False, True])
def test_harbor_pool_rejects_different_taskset_data(mapped: bool) -> None:
    config = _harbor_config()
    config["swe_pi"]["environment_servers"]["harbor"]["tasksets"]["smoke"]["tasks"][
        "task-1"
    ] = "other"
    names = [route.environment_name for route in ROUTES]
    with pytest.raises(ValueError, match="same taskset data"):
        resolve_harness_pool({"smoke": names} if mapped else names, config)


def test_harbor_taskset_pool_allows_unrelated_tasksets_to_differ() -> None:
    config = _harbor_config()
    config["swe_pi"]["environment_servers"]["harbor"]["tasksets"]["other"] = {
        "folder": "/datasets/other",
        "tasks": {"other-task": "other-digest"},
    }
    names = [route.environment_name for route in ROUTES]
    resolved = resolve_harness_pool({"smoke": names}, config)
    route_harness_rows([task()], resolved)


@pytest.mark.parametrize("mapped", [False, True])
def test_harbor_pool_rejects_unserved_taskset(mapped: bool) -> None:
    names = [route.environment_name for route in ROUTES]
    if mapped:
        with pytest.raises(ValueError, match="does not serve taskset 'unknown'"):
            resolve_harness_pool({"unknown": names}, _harbor_config())
    else:
        routes = resolve_harness_pool(names, _harbor_config())
        row = task()
        row["task_id"]["taskset"] = "unknown"
        with pytest.raises(ValueError, match="does not serve taskset 'unknown'"):
            route_harness_rows([row], routes)


def test_pool_rejects_mixed_harbor_and_swe_environments() -> None:
    config = resolved_config()
    config["swe_pi"] = _harbor_config()["swe_pi"]
    with pytest.raises(ValueError, match="all use Harbor"):
        resolve_harness_pool([route.environment_name for route in ROUTES], config)


def test_harbor_pool_requires_distinct_training_agents() -> None:
    names = [route.environment_name for route in ROUTES]
    config = _harbor_config()
    config["pi_agent"]["responses_api_agents"]["pi_agent"]["model_server"]["name"] = (
        "external"
    )
    with pytest.raises(ValueError, match="responses_api_models:policy_model"):
        resolve_harness_pool(names, config)
    config = _harbor_config()
    config["swe_pi"]["environment_servers"]["harbor"]["agent_server"]["name"] = (
        "opencode_agent"
    )
    with pytest.raises(ValueError, match="distinct agent"):
        resolve_harness_pool(names, config)


def test_hash_covers_harnesses_and_is_stable_on_sibling_retry():
    rows = [task(i) for i in range(32)]
    route_harness_rows(rows, ROUTES)
    assert {row["agent_ref"]["name"] for row in rows} == {"opencode_agent", "pi_agent"}
    selected = [(row["_ng_environment_server"], row["agent_ref"]) for row in rows]
    for row in rows:
        row["_ng_group_attempt"] += 1
        row["_ng_rollout_id"] = row["_ng_rollout_id"].replace("afirst", "aretry")
    # Partial retries preserve the original generation index, independent of batch position.
    retry = copy.deepcopy(rows[7:11])
    route_harness_rows(retry, ROUTES)
    assert [
        (row["_ng_environment_server"], row["agent_ref"]) for row in retry
    ] == selected[7:11]
    assert all(row["_ng_rollout_id"].endswith("aretry") for row in retry)


@pytest.mark.parametrize(
    "key,value",
    [("_ng_group_id", ""), ("_ng_rollout_index", True), ("_ng_rollout_id", None)],
)
def test_routing_requires_controller_identity(key, value):
    row = task()
    row[key] = value
    with pytest.raises(ValueError):
        route_harness_rows([row], ROUTES)


@pytest.mark.nemo_gym
def test_typed_result_keeps_reward_terminal_and_actual_harness():
    row = task()
    route_harness_rows([row], ROUTES)
    result = _harness_result(row, episode(row))
    assert result["reward"] == 0.75
    assert result["terminal_response_id"] == "response-1"
    assert result["ng_agent_server"] == row["agent_ref"]["name"]


@pytest.mark.nemo_gym
@pytest.mark.parametrize(
    "terminal,error", [(False, GymTransportError), (True, RolloutDataFailure)]
)
def test_typed_failure_never_becomes_zero_reward(terminal, error):
    row = task()
    route_harness_rows([row], ROUTES)
    result = episode(row)
    result.pop("result")
    result["failure"] = {"terminal": terminal, "failure_reason": "sandbox unavailable"}
    with pytest.raises(error, match="sandbox unavailable"):
        _harness_result(row, result)


@pytest.mark.nemo_gym
@pytest.mark.parametrize("mismatch", ["task", "episode", "agent", "missing_reward"])
def test_result_rejects_wrong_identity_or_unscored_payload(mismatch):
    row = task()
    route_harness_rows([row], ROUTES)
    result = episode(row)
    if mismatch == "task":
        result["task_id"]["task_id"] = "wrong"
    elif mismatch == "episode":
        result["episode_id"]["rollout_id"] = "wrong"
    elif mismatch == "agent":
        result["result"]["ng_agent_server"] = "wrong"
    else:
        del result["result"]["reward"]
    with pytest.raises(RolloutDataFailure):
        _harness_result(row, result)


@pytest.mark.nemo_gym
@pytest.mark.parametrize("pool_kind", ["list", "mapped", "harbor", "mixed"])
def test_actor_dispatches_typed_rows_and_fetches_each_physical_capture_receipt(
    pool_kind: str,
) -> None:
    from nemo_gym.rollout_collection import (
        _EagerCompletionIterator,
        _episode_request_body,
        _RowResultIterator,
    )

    from nemo_rl.environments.nemo_gym import NemoGym

    config = _harbor_config() if pool_kind == "harbor" else resolved_config()
    names = [r.environment_name for r in ROUTES]
    pool = names if pool_kind == "list" else {"smoke": names}
    rows = [(_harbor_task(i) if pool_kind == "harbor" else task(i)) for i in range(8)]
    if pool_kind == "mixed":
        harbor_config = _harbor_config()
        harbor_names = []
        for name in names:
            entry = harbor_config[name]
            tasksets = entry["environment_servers"]["harbor"]["tasksets"]
            tasksets["harbor-smoke"] = tasksets.pop("smoke")
            config[f"harbor_{name}"] = entry
            harbor_names.append(f"harbor_{name}")
        pool["harbor-smoke"] = harbor_names
        for i in range(1, len(rows), 2):
            rows[i] = _harbor_task(i)
            rows[i]["task_id"]["taskset"] = "harbor-smoke"
    original_inputs = [copy.deepcopy(row["task_input"]) for row in rows]
    actor = NemoGym.__ray_metadata__.modified_class(
        {
            "model_name": "policy",
            "base_urls": [],
            "initial_global_config_dict": {},
            "harness_pool": pool,
        }
    )
    actor.rh = object()
    actor._tokenizer = object()
    actor.head_server_config = "head"
    actor._token_capture_enabled = True
    actor._harness_routes = resolve_harness_pool(pool, config)
    actor._environment_server_routes = {}
    actor._control = AsyncMock(return_value={"rows": [], "failures": []})

    class Helper:
        def run_examples(
            self, *, examples, head_server_config, route_failures_to_sidecar
        ):
            assert head_server_config == "head"
            assert route_failures_to_sidecar

            async def completed(row):
                request = _episode_request_body(row)
                assert request["episode_id"]["rollout_id"] == row["_ng_rollout_id"]
                assert request["task"]["task_input"] == row["task_input"]
                return SimpleNamespace(row=row, result=episode(row))

            return _RowResultIterator(
                _EagerCompletionIterator(
                    (completed(row) for row in examples), total=len(examples)
                )
            )

    actor.rch = Helper()

    async def collect():
        return [item async for item in actor.run_rollouts(rows, "test")]

    streamed = asyncio.run(collect())
    assert {item[1]["name"] for item in streamed} == {"opencode_agent", "pi_agent"}
    for index, ref, result, _ in streamed:
        row = rows[index]
        assert ref == row["agent_ref"]
        assert result["full_result"]["ng_agent_server"] == ref["name"]
        assert result["attribution_key"] == row["task_id"]["taskset"]
        assert row["task_input"] == original_inputs[index]
        assert result["full_result"]["_ng_task_id"] == row["task_id"]
        assert result["receipt"]["reward"] == 0.75
        assert result["rollout_id"] == row["_ng_rollout_id"]
    assert [call.args[1] for call in actor._control.call_args_list] == [
        f"/training-token-capture/control/rollouts/{row['_ng_rollout_id']}/manifest"
        for row in rows
    ]


@pytest.mark.nemo_gym
@pytest.mark.parametrize(
    "terminal,error", [(False, GymTransportError), (True, RolloutDataFailure)]
)
def test_closing_typed_failure_stream_cancels_pending_sibling_requests(terminal, error):
    from nemo_gym.rollout_collection import _EagerCompletionIterator, _RowResultIterator

    from nemo_rl.environments.nemo_gym import NemoGym

    async def exercise():
        actor = NemoGym.__ray_metadata__.modified_class(
            {
                "model_name": "policy",
                "base_urls": [],
                "initial_global_config_dict": {},
                "harness_pool": [r.environment_name for r in ROUTES],
            }
        )
        actor.rh = object()
        actor._tokenizer = object()
        actor.head_server_config = "head"
        actor._token_capture_enabled = True
        actor._harness_routes = ROUTES
        actor._environment_server_routes = {}
        started, cancelled = asyncio.Event(), asyncio.Event()

        class Helper:
            def run_examples(
                self, *, examples, head_server_config, route_failures_to_sidecar
            ):
                assert route_failures_to_sidecar

                async def failed():
                    await started.wait()
                    result = episode(examples[0])
                    result.pop("result")
                    result["failure"] = {
                        "terminal": terminal,
                        "failure_reason": "sandbox unavailable",
                    }
                    return SimpleNamespace(row=examples[0], result=result)

                async def pending():
                    started.set()
                    try:
                        await asyncio.Future()
                    finally:
                        cancelled.set()

                return _RowResultIterator(
                    _EagerCompletionIterator(iter((failed(), pending())), total=2)
                )

        actor.rch = Helper()
        from nemo_rl.experience.failures import GymRowFailure

        stream = actor.run_rollouts([task(0), task(1)], "test")
        try:
            failure = await stream.__anext__()
            assert isinstance(failure, GymRowFailure)
            assert isinstance(failure.error, error)
            assert "sandbox unavailable" in str(failure.error)
        finally:
            await stream.aclose()
        # Assert before asyncio.run cleanup can cancel leaked tasks for us.
        assert cancelled.is_set()

    asyncio.run(exercise())

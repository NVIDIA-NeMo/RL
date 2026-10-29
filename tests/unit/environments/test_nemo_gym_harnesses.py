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
    parse_harness_pool,
    resolve_harness_pool,
    responses_create_params,
    route_harness_rows,
    unwrap_harness_result,
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


@pytest.mark.parametrize(
    "value", [[], "swe_pi", ["swe_pi", "swe_pi"], [""], [False], [["nested"]]]
)
def test_pool_rejects_invalid_names(value):
    with pytest.raises(ValueError, match="nonempty list of distinct"):
        parse_harness_pool(value)


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


def test_flat_and_typed_sampling_parameters_are_mutable_without_flattening_tasks():
    row = task()
    responses_create_params(row)["temperature"] = 0.4
    assert row["task_input"]["responses_create_params"]["temperature"] == 0.4
    assert "responses_create_params" not in row
    flat = {"responses_create_params": {"input": "hello"}}
    assert responses_create_params(flat) is flat["responses_create_params"]


@pytest.mark.nemo_gym
def test_typed_result_keeps_reward_terminal_and_actual_harness():
    row = task()
    route_harness_rows([row], ROUTES)
    result = unwrap_harness_result(row, episode(row))
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
        unwrap_harness_result(row, result)


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
    with pytest.raises(ValueError):
        unwrap_harness_result(row, result)


@pytest.mark.nemo_gym
def test_actor_dispatches_typed_rows_and_fetches_each_physical_capture_receipt():
    from nemo_gym.rollout_collection import (
        _EagerCompletionIterator,
        _episode_request_body,
        _RowResultIterator,
    )

    from nemo_rl.environments.nemo_gym import NemoGym

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
    actor._control = AsyncMock(return_value={"rows": [], "failures": []})

    class Helper:
        def run_examples(self, *, examples, head_server_config):
            assert head_server_config == "head"

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
    rows = [task(i) for i in range(8)]

    async def collect():
        return [item async for item in actor.run_rollouts(rows, "test")]

    streamed = asyncio.run(collect())
    assert {item[1]["name"] for item in streamed} == {"opencode_agent", "pi_agent"}
    for row, (_, ref, result, _) in zip(rows, streamed):
        assert ref == row["agent_ref"]
        assert result["full_result"]["ng_agent_server"] == ref["name"]
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
def test_typed_failure_cancels_pending_sibling_requests_before_retry(terminal, error):
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
        started, cancelled = asyncio.Event(), asyncio.Event()

        class Helper:
            def run_examples(self, *, examples, head_server_config):
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
        with pytest.raises(error, match="sandbox unavailable"):
            async for _ in actor.run_rollouts([task(0), task(1)], "test"):
                pytest.fail("A failed episode must not produce a training result")
        # Assert before asyncio.run cleanup can cancel leaked tasks for us.
        assert cancelled.is_set()

    asyncio.run(exercise())

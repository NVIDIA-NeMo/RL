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

import asyncio
import json

import pytest
from omegaconf import OmegaConf

from nemo_rl.environments.central_agent_helpers import (
    CentralAgent,
    merge_central_agent_config,
    pop_agent_registry,
    resolve_central_agent_config,
)
from nemo_rl.environments.central_agent_helpers.rollout_tree import (
    LinearRolloutTree,
    build_rollout_tree,
)
from nemo_rl.environments.central_agent_helpers.rollout_tree.base import TurnNode
from nemo_rl.environments.central_agent_helpers.steering_manager import (
    NoOpSteeringMessageManager,
    SteeringMessageManager,
    build_steering_manager,
)
from nemo_rl.environments.central_agent_helpers.termination_manager import (
    TerminationManager,
    TurnState,
    build_termination_manager,
)
from nemo_rl.environments.central_agent_helpers.tool_call_manager import (
    ParallelToolCallManager,
    SerialToolCallManager,
    build_tool_call_manager,
)


def function_call(call_id, name="get_weather", arguments='{"city": "SF"}'):
    return {
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
    }


def assistant_message(text="done"):
    return {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text}],
    }


########################################
# config
########################################


def test_resolve_fills_defaults_and_disables_by_default():
    assert resolve_central_agent_config(None)["enabled"] is False
    cfg = resolve_central_agent_config(
        {"enabled": True, "tool_calls": {"mode": "parallel"}}
    )
    assert cfg["enabled"] is True
    assert cfg["tool_calls"] == {"mode": "parallel", "min_returns": 1}
    assert cfg["termination"]["max_turns"] == 20


def test_merge_is_deep_and_tolerates_none():
    merged = merge_central_agent_config(
        {"termination": {"max_turns": 20, "stop_on_no_tool_calls": True}}, None
    )
    assert merged["termination"]["max_turns"] == 20
    merged = merge_central_agent_config(
        {"termination": {"max_turns": 20, "stop_on_no_tool_calls": True}},
        {"termination": {"max_turns": 3}},
    )
    assert merged["termination"] == {"max_turns": 3, "stop_on_no_tool_calls": True}


def gym_config(implementation="simple_agent", agent_override=None):
    agent_block = {
        "responses_api_agents": {
            implementation: {
                "entrypoint": "app.py",
                "resources_server": {"type": "resources_servers", "name": "weather"},
                "model_server": {
                    "type": "responses_api_models",
                    "name": "policy_model",
                },
            }
        }
    }
    if agent_override is not None:
        agent_block["responses_api_agents"][implementation]["central_agent"] = (
            agent_override
        )
    return OmegaConf.create(
        {
            "policy_model": {
                "responses_api_models": {"vllm_model": {"entrypoint": "app.py"}}
            },
            "weather": {"resources_servers": {"weather": {"entrypoint": "app.py"}}},
            "weather_simple_agent": agent_block,
        }
    )


def test_pop_agent_registry_removes_agent_blocks_only():
    cfg = gym_config()
    registry = pop_agent_registry(cfg, resolve_central_agent_config({"enabled": True}))

    assert "weather_simple_agent" not in cfg
    assert set(cfg.keys()) == {"policy_model", "weather"}
    entry = registry["weather_simple_agent"]
    assert entry["implementation"] == "simple_agent"
    assert entry["resources_server"] == "weather"
    assert entry["model_server"] == "policy_model"
    assert entry["central_agent"]["tool_calls"]["mode"] == "serial"


def test_pop_agent_registry_merges_per_agent_override():
    cfg = gym_config(
        agent_override={"tool_calls": {"mode": "parallel", "min_returns": 2}}
    )
    registry = pop_agent_registry(cfg, resolve_central_agent_config({"enabled": True}))
    entry = registry["weather_simple_agent"]["central_agent"]
    assert entry["tool_calls"] == {"mode": "parallel", "min_returns": 2}
    # untouched keys still come from the base config
    assert entry["termination"]["max_turns"] == 20


def test_pop_agent_registry_rejects_unsupported_implementation():
    cfg = gym_config(implementation="swe_agent")
    with pytest.raises(ValueError, match="swe_agent"):
        pop_agent_registry(cfg, resolve_central_agent_config({"enabled": True}))


def test_pop_agent_registry_requires_an_agent():
    cfg = OmegaConf.create({"weather": {"resources_servers": {"weather": {}}}})
    with pytest.raises(ValueError, match="no 'responses_api_agents' instance"):
        pop_agent_registry(cfg, resolve_central_agent_config({"enabled": True}))


########################################
# rollout tree
########################################


PROMPT = [{"role": "user", "content": "hi"}]


def test_linear_tree_is_append_only():
    tree = build_rollout_tree({"type": "linear"}, PROMPT)
    assert isinstance(tree, LinearRolloutTree)
    assert tree.get_active_branch() == PROMPT
    assert tree.get_active_outputs() == []

    tree.append([{"a": 1}])
    tree.append(None)
    tree.append([])
    tree.append([{"b": 2}])

    assert tree.get_active_branch() == PROMPT + [{"a": 1}, {"b": 2}]
    assert tree.get_active_outputs() == [{"a": 1}, {"b": 2}]
    assert tree.return_all_rollouts() == [[{"a": 1}, {"b": 2}]]


def test_linear_tree_makes_one_node_per_turn():
    tree = build_rollout_tree({"type": "linear"}, PROMPT)
    assert tree.root.items == PROMPT
    assert tree.root.turn_index == 0
    # the root is the prompt only; turn 1 is already open under it
    assert tree.active.parent is tree.root
    assert tree.active.items == []

    tree.append([{"turn": 1}])
    tree.branch_and_compact_context()
    tree.append([{"turn": 2}])
    tree.branch_and_compact_context()

    # root -> turn 1 -> turn 2 -> open turn 3, one child each
    chain = tree.active.path_from_root()
    assert [node.turn_index for node in chain] == [0, 1, 2, 3]
    assert [node.items for node in chain] == [PROMPT, [{"turn": 1}], [{"turn": 2}], []]
    assert all(len(node.children) == 1 for node in chain[:-1])
    assert tree.leaves() == [tree.active]
    # the still-open trailing turn contributes nothing to either view
    assert tree.get_active_outputs() == [{"turn": 1}, {"turn": 2}]
    assert tree.return_all_rollouts() == [[{"turn": 1}, {"turn": 2}]]


def test_tree_branching_yields_one_rollout_per_leaf():
    """LinearRolloutTree never forks, but the base machinery it inherits does."""
    tree = build_rollout_tree({"type": "linear"}, PROMPT)
    tree.append([{"shared": 1}])
    shared = tree.active

    tree.start_new_turn(shared, kind="branch-a")
    tree.append([{"a": 1}])
    branch_a = tree.active

    tree.start_new_turn(shared, kind="branch-b")
    tree.append([{"b": 1}])

    assert len(shared.children) == 2
    assert tree.active.metadata == {"kind": "branch-b"}
    assert tree.get_active_branch() == PROMPT + [{"shared": 1}, {"b": 1}]
    assert tree.outputs_for(branch_a) == [{"shared": 1}, {"a": 1}]
    assert tree.return_all_rollouts() == [
        [{"shared": 1}, {"a": 1}],
        [{"shared": 1}, {"b": 1}],
    ]


def test_turn_node_parent_and_leaf_bookkeeping():
    root = TurnNode(turn_index=0, items=[{"p": 1}])
    child = root.add_child(kind="turn")
    assert child.parent is root and root.children == [child]
    assert child.turn_index == 1 and child.metadata == {"kind": "turn"}
    assert root.is_leaf is False and child.is_leaf is True
    assert child.path_from_root() == [root, child]


def test_unknown_tree_type_raises():
    with pytest.raises(ValueError, match="rollout_tree.type"):
        build_rollout_tree({"type": "beam"}, [])


########################################
# tool call manager
########################################


@pytest.mark.asyncio
async def test_serial_returns_every_call_in_order():
    seen = []

    async def call_tool(name, arguments):
        seen.append((name, arguments))
        return json.dumps({"ok": name})

    manager = build_tool_call_manager({"mode": "serial"}, call_tool)
    results = await manager.submit_and_collect(
        [function_call("1", "a"), function_call("2", "b")]
    )

    assert [r["call_id"] for r in results] == ["1", "2"]
    assert [r["type"] for r in results] == ["function_call_output"] * 2
    assert seen == [("a", {"city": "SF"}), ("b", {"city": "SF"})]
    assert manager.stats()["tool_calls_total"] == 2


@pytest.mark.asyncio
async def test_malformed_arguments_become_a_tool_result():
    async def call_tool(name, arguments):  # pragma: no cover - must not run
        raise AssertionError("malformed call must not reach the resources server")

    manager = build_tool_call_manager({"mode": "serial"}, call_tool)
    results = await manager.submit_and_collect(
        [function_call("1", "a", arguments="{not json")]
    )

    assert results[0]["call_id"] == "1"
    assert "Invalid tool call arguments" in json.loads(results[0]["output"])["error"]
    assert manager.stats()["tool_calls_malformed"] == 1


@pytest.mark.asyncio
async def test_unknown_tool_body_is_passed_through_not_raised():
    async def call_tool(name, arguments):
        return (
            '{"detail":"Not Found"}'  # what a 404 from the resources server looks like
        )

    manager = build_tool_call_manager({"mode": "serial"}, call_tool)
    results = await manager.submit_and_collect([function_call("1", "nope")])
    assert results[0]["output"] == '{"detail":"Not Found"}'


@pytest.mark.asyncio
async def test_parallel_queue_spans_turns():
    """The walkthrough in the design doc, with min_returns=2."""
    gates = {}

    async def call_tool(name, arguments):
        await gates[name].wait()
        return name

    def calls(names):
        for name in names:
            gates[name] = asyncio.Event()
        return [function_call(name, name) for name in names]

    manager = build_tool_call_manager({"mode": "parallel", "min_returns": 2}, call_tool)

    # turn 1: 5 calls, threshold min(2, 5) = 2 -> blocks until two finish
    turn1 = asyncio.ensure_future(manager.submit_and_collect(calls("abcde")))
    await asyncio.sleep(0)
    assert not turn1.done()
    gates["a"].set()
    gates["b"].set()
    assert [r["call_id"] for r in await turn1] == ["a", "b"]
    assert manager.stats()["tool_calls_pending"] == 3

    # turn 2: c d e finished while the model was generating; 3 >= min(2, 4) so the
    # four new calls are queued and not waited on
    for name in "cde":
        gates[name].set()
    await asyncio.sleep(0.01)
    turn2 = await manager.submit_and_collect(calls("fghi"))
    assert [r["call_id"] for r in turn2] == ["c", "d", "e"]
    assert manager.stats()["tool_calls_pending"] == 4

    # turn 3: no new calls, threshold 0 -> returns immediately without blocking
    assert await manager.submit_and_collect([]) == []

    # termination drains whatever is still running
    for name in "fghi":
        gates[name].set()
    assert [r["call_id"] for r in await manager.drain()] == ["f", "g", "h", "i"]
    assert manager.stats()["tool_calls_pending"] == 0
    assert manager.stats()["tool_calls_returned"] == 9


@pytest.mark.asyncio
async def test_parallel_returns_early_results_before_the_batch_finishes():
    gates = {name: asyncio.Event() for name in "xy"}

    async def call_tool(name, arguments):
        await gates[name].wait()
        return name

    manager = build_tool_call_manager({"mode": "parallel", "min_returns": 1}, call_tool)
    pending = asyncio.ensure_future(
        manager.submit_and_collect([function_call("x", "x"), function_call("y", "y")])
    )
    gates["y"].set()
    assert [r["call_id"] for r in await pending] == ["y"]
    gates["x"].set()
    assert [r["call_id"] for r in await manager.drain()] == ["x"]


@pytest.mark.asyncio
async def test_parallel_keeps_call_order_across_separate_waits():
    """Results harvested on different waits are still returned in call order."""
    gates = {name: asyncio.Event() for name in "abc"}

    async def call_tool(name, arguments):
        await gates[name].wait()
        return name

    manager = build_tool_call_manager({"mode": "parallel", "min_returns": 3}, call_tool)
    pending = asyncio.ensure_future(
        manager.submit_and_collect([function_call(name, name) for name in "abc"])
    )
    # finish out of order, one per event-loop pass, so each is harvested separately
    for name in "cab":
        gates[name].set()
        await asyncio.sleep(0.01)
    assert [r["call_id"] for r in await pending] == ["a", "b", "c"]
    assert manager.stats()["tool_calls_returned"] == 3


def test_tool_manager_factory_selects_by_mode():
    assert isinstance(build_tool_call_manager(None, None), SerialToolCallManager)
    assert isinstance(
        build_tool_call_manager({"mode": "parallel"}, None), ParallelToolCallManager
    )
    with pytest.raises(ValueError, match="tool_calls.mode"):
        build_tool_call_manager({"mode": "batched"}, None)


########################################
# steering
########################################


@pytest.mark.asyncio
async def test_noop_steering_ignores_its_input():
    manager = build_steering_manager({"enabled": False}, None)
    assert isinstance(manager, NoOpSteeringMessageManager)
    assert await manager.get_steering_messages([assistant_message()]) == []


@pytest.mark.asyncio
async def test_steering_posts_the_latest_model_output():
    posted = []

    async def post(url_path, body):
        posted.append((url_path, body))
        return {"messages": [{"role": "user", "content": "keep going"}]}

    manager = build_steering_manager({"enabled": True, "url_path": "/steering"}, post)
    assert isinstance(manager, SteeringMessageManager)
    output = [assistant_message()]
    assert await manager.get_steering_messages(output) == [
        {"role": "user", "content": "keep going"}
    ]
    assert posted == [("/steering", {"latest_model_output": output, "step": 1})]


def test_unknown_steering_type_raises():
    with pytest.raises(ValueError, match="steering_message.type"):
        build_steering_manager({"enabled": True, "type": "webhook"}, None)


########################################
# termination
########################################


def turn(**kwargs):
    kwargs.setdefault("response", {})
    kwargs.setdefault("tool_stats", {"tool_calls_malformed": 0})
    return TurnState(**kwargs)


def test_termination_factory_selects_default():
    assert isinstance(build_termination_manager(None), TerminationManager)
    with pytest.raises(ValueError, match="termination.type"):
        build_termination_manager({"type": "budgeted"})


def test_no_tool_calls_stops_only_with_an_assistant_message():
    manager = build_termination_manager(
        {"max_turns": 99, "stop_on_no_tool_calls": True}
    )
    assert not manager.is_termination_criteria_met(turn(has_output_message=False))
    assert manager.is_termination_criteria_met(turn(has_output_message=True))
    assert manager.termination_stats() == {
        "central_agent_termination_reason": "no_tool_calls",
        "central_agent_turns": 2,
    }


def test_steering_messages_override_no_tool_calls():
    manager = build_termination_manager({"max_turns": 99})
    assert not manager.is_termination_criteria_met(
        turn(has_output_message=True, steering_messages=[{"role": "user"}])
    )


def test_incomplete_details_stops_even_with_tool_calls():
    manager = build_termination_manager({"max_turns": 99})
    assert manager.is_termination_criteria_met(
        turn(
            response={"incomplete_details": {"reason": "max_output_tokens"}},
            tool_calls=[function_call("1")],
        )
    )
    assert manager.termination_stats()["central_agent_termination_reason"] == (
        "incomplete_details"
    )


def test_max_turns_and_malformed_budgets():
    manager = build_termination_manager(
        {"max_turns": 2, "stop_on_no_tool_calls": False}
    )
    assert not manager.is_termination_criteria_met(turn())
    assert manager.is_termination_criteria_met(turn())
    assert (
        manager.termination_stats()["central_agent_termination_reason"] == "max_turns"
    )

    manager = build_termination_manager(
        {"max_turns": 99, "max_malformed_tool_calls": 2}
    )
    assert not manager.is_termination_criteria_met(
        turn(tool_calls=[function_call("1")], tool_stats={"tool_calls_malformed": 1})
    )
    assert manager.is_termination_criteria_met(
        turn(tool_calls=[function_call("1")], tool_stats={"tool_calls_malformed": 2})
    )
    assert manager.termination_stats()["central_agent_termination_reason"] == (
        "max_malformed_tool_calls"
    )


########################################
# the loop
########################################


class FakeResponse:
    def __init__(self, payload, status=200):
        self._body = (
            payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        )
        self.status = status
        self.cookies = {"session": "abc"}

    @property
    def content(self):
        return self

    async def read(self):
        return self._body


class FakeServerClient:
    """Records every POST and replies from a scripted queue keyed by url_path."""

    def __init__(self, replies):
        self.replies = replies
        self.calls = []

    async def post(self, server_name, url_path, json=None, cookies=None):
        self.calls.append((server_name, url_path, json, cookies))
        reply = self.replies[url_path]
        return FakeResponse(reply.pop(0) if isinstance(reply, list) else reply)


AGENT_ENTRY = {
    "resources_server": "weather",
    "model_server": "policy_model",
    "central_agent": resolve_central_agent_config({"enabled": True}),
}

ROW = {
    "agent_ref": {"name": "weather_simple_agent"},
    "responses_create_params": {
        "input": [{"role": "user", "content": "weather in SF?"}],
        "tools": [{"type": "function", "name": "get_weather"}],
    },
}


@pytest.mark.asyncio
async def test_loop_runs_a_tool_turn_then_verifies_the_final_branch():
    call = function_call("call-1")
    message = assistant_message("it is sunny")
    client = FakeServerClient(
        {
            "/seed_session": [{}],
            "/v1/responses": [
                {
                    "id": "resp-1",
                    "usage": {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7},
                    "output": [call],
                },
                {
                    "id": "resp-2",
                    "usage": {
                        "input_tokens": 9,
                        "output_tokens": 1,
                        "total_tokens": 10,
                    },
                    "output": [message],
                },
            ],
            "/get_weather": ['{"temp": 20}'.encode()],
            "/verify": [{"reward": 1.0, "extra": "kept"}],
        }
    )

    result = await CentralAgent(client, AGENT_ENTRY).run(ROW)

    paths = [c[1] for c in client.calls]
    assert paths == [
        "/seed_session",
        "/v1/responses",
        "/get_weather",
        "/v1/responses",
        "/verify",
    ]

    # the tool call and its output are in the second model request, appended in order
    second_request = client.calls[3][2]
    assert second_request["input"] == [
        {"role": "user", "content": "weather in SF?"},
        call,
        {"type": "function_call_output", "call_id": "call-1", "output": '{"temp": 20}'},
    ]
    assert second_request["tools"] == ROW["responses_create_params"]["tools"]

    # verify sees the final active branch as response.output, with summed usage
    verify_body = client.calls[4][2]
    assert verify_body["response"]["output"] == second_request["input"][1:] + [message]
    # the envelope is the last turn's, as in simple_agent
    assert verify_body["response"]["id"] == "resp-2"
    assert verify_body["response"]["usage"]["total_tokens"] == 17
    assert verify_body["agent_ref"] == ROW["agent_ref"]

    # the verify response is returned verbatim, plus loop stats
    assert result["reward"] == 1.0
    assert result["extra"] == "kept"
    assert result["central_agent_turns"] == 2
    assert result["central_agent_termination_reason"] == "no_tool_calls"
    assert result["tool_calls_total"] == 1


@pytest.mark.asyncio
async def test_verify_envelope_comes_from_the_last_turn_not_the_first():
    """`status` / `incomplete_details` / `error` are inherited, and verifiers read them.

    simple_agent rebinds its response variable each turn and mutates whichever
    one it holds when the loop exits, so /verify sees the last turn's envelope.
    Taking the first turn's would report a rollout truncated on its final turn
    as `status="completed"`, and would hide a final-turn `error` from
    resources_servers/structured_outputs, which rejects on `response.error`.
    """
    client = FakeServerClient(
        {
            "/seed_session": [{}],
            "/v1/responses": [
                {
                    "id": "resp-1",
                    "status": "completed",
                    "incomplete_details": None,
                    "error": None,
                    "output": [function_call("call-1")],
                },
                {
                    "id": "resp-2",
                    "status": "incomplete",
                    "incomplete_details": {"reason": "max_output_tokens"},
                    "error": {"code": "server_error"},
                    "output": [assistant_message("truncated...")],
                },
            ],
            "/get_weather": ['{"temp": 20}'.encode()],
            "/verify": [{"reward": 0.0}],
        }
    )

    await CentralAgent(client, AGENT_ENTRY).run(ROW)

    response = client.calls[-1][2]["response"]
    assert response["id"] == "resp-2"
    assert response["status"] == "incomplete"
    assert response["incomplete_details"] == {"reason": "max_output_tokens"}
    assert response["error"] == {"code": "server_error"}
    # output still spans every turn, so the envelope swap changes metadata only
    assert len(response["output"]) == 3


@pytest.mark.asyncio
async def test_loop_stops_on_incomplete_details_without_calling_tools():
    """A truncated response may carry truncated arguments, so its calls never run."""
    call = function_call("1")
    client = FakeServerClient(
        {
            "/seed_session": [{}],
            "/v1/responses": [
                {
                    "id": "r",
                    "output": [call],
                    "incomplete_details": {"reason": "max_output_tokens"},
                }
            ],
            "/verify": [{"reward": 0.0}],
        }
    )
    result = await CentralAgent(client, AGENT_ENTRY).run(ROW)
    assert [c[1] for c in client.calls] == ["/seed_session", "/v1/responses", "/verify"]
    assert result["central_agent_termination_reason"] == "incomplete_details"
    assert result["tool_calls_total"] == 0
    # the unanswered call still lands in the branch, exactly as simple_agent leaves it
    assert client.calls[2][2]["response"]["output"] == [call]


@pytest.mark.asyncio
async def test_loop_respects_max_turns():
    entry = {
        **AGENT_ENTRY,
        "central_agent": resolve_central_agent_config(
            {"enabled": True, "termination": {"max_turns": 3}}
        ),
    }
    client = FakeServerClient(
        {
            "/seed_session": [{}],
            "/v1/responses": [
                {"id": "r", "output": [function_call(str(i))]} for i in range(3)
            ],
            "/get_weather": ["{}".encode()] * 3,
            "/verify": [{"reward": 0.0}],
        }
    )
    result = await CentralAgent(client, entry).run(ROW)
    assert result["central_agent_turns"] == 3
    assert result["central_agent_termination_reason"] == "max_turns"


@pytest.mark.asyncio
async def test_loop_steers_through_the_agents_own_resources_server():
    """Steering POSTs the configured path on the same resources server, with cookies."""
    entry = {
        **AGENT_ENTRY,
        "central_agent": resolve_central_agent_config(
            {
                "enabled": True,
                "steering_message": {"enabled": True, "url_path": "/nudge"},
            }
        ),
    }
    client = FakeServerClient(
        {
            "/seed_session": [{}],
            "/v1/responses": [
                {"id": "r", "output": [assistant_message("first")]},
                {"id": "r", "output": [assistant_message("second")]},
            ],
            "/nudge": [{"messages": [{"role": "user", "content": "keep going"}]}, {}],
            "/verify": [{"reward": 1.0}],
        }
    )

    result = await CentralAgent(client, entry).run(ROW)

    by_path = [(c[0], c[1]) for c in client.calls]
    assert by_path == [
        ("weather", "/seed_session"),
        ("policy_model", "/v1/responses"),
        ("weather", "/nudge"),  # same resources server, configured path
        ("policy_model", "/v1/responses"),
        ("weather", "/nudge"),
        ("weather", "/verify"),
    ]
    # session cookies are carried into steering
    assert client.calls[2][3] == {"session": "abc"}
    assert client.calls[2][2] == {
        "latest_model_output": [assistant_message("first")],
        "step": 1,
    }
    # steering kept the rollout going past a no-tool-call turn; an empty second
    # steering reply let stop_on_no_tool_calls fire
    assert result["central_agent_turns"] == 2
    assert result["central_agent_termination_reason"] == "no_tool_calls"
    verify_output = client.calls[5][2]["response"]["output"]
    assert verify_output == [
        assistant_message("first"),
        {"role": "user", "content": "keep going"},
        assistant_message("second"),
    ]


@pytest.mark.asyncio
async def test_loop_raises_on_a_failed_model_call():
    class FailingClient(FakeServerClient):
        async def post(self, server_name, url_path, json=None, cookies=None):
            if url_path == "/v1/responses":
                return FakeResponse({"detail": "boom"}, status=500)
            return await super().post(server_name, url_path, json=json, cookies=cookies)

    client = FailingClient({"/seed_session": [{}]})
    with pytest.raises(RuntimeError, match="model call failed with status 500"):
        await CentralAgent(client, AGENT_ENTRY).run(ROW)


@pytest.mark.asyncio
async def test_loop_carries_seed_session_cookies_into_tools_and_verify():
    client = FakeServerClient(
        {
            "/seed_session": [{}],
            "/v1/responses": [
                {"id": "r", "output": [function_call("1")]},
                {"id": "r", "output": [assistant_message()]},
            ],
            "/get_weather": ["{}".encode()],
            "/verify": [{"reward": 0.0}],
        }
    )
    await CentralAgent(client, AGENT_ENTRY).run(ROW)
    by_path = {c[1]: c[3] for c in client.calls}
    assert by_path["/seed_session"] is None
    assert by_path["/get_weather"] == {"session": "abc"}
    assert by_path["/verify"] == {"session": "abc"}

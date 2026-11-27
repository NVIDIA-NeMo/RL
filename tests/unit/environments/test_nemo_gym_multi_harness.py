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
"""Parse, describe, resolve and select ``env.nemo_gym.multi_harness`` without Gym."""

from collections import Counter

import pytest
from omegaconf import OmegaConf

from nemo_rl.environments.nemo_gym_multi_harness import (
    EnvironmentServerInfo,
    HarnessChoice,
    MultiHarnessRoutes,
    describe_environment_server,
    harness_metric_namespace,
    parse_multi_harness,
    resolve_multi_harness,
)

_ROUTES = {"swe": ["swe_opencode", "swe_pi"], "scale": ["scale_opencode"]}


def _turn(agent, resources="swe_rebench"):
    return EnvironmentServerInfo("single_agent_turn", agent, resources, None)


_SERVERS = {
    "swe_opencode": _turn("opencode_agent"),
    "swe_pi": _turn("pi_agent"),
    "scale_opencode": _turn("opencode_agent", "scale_swe"),
}


def test_parse_reads_yaml_and_returns_none_when_unset():
    config = parse_multi_harness(
        OmegaConf.create({"routes": _ROUTES, "weights": {"opencode_agent": 2}})
    )

    assert config.routes == _ROUTES
    assert config.weights == {"opencode_agent": 2.0}
    assert config.tasksets == {"swe", "scale"}
    assert config.servers == {"swe_opencode", "swe_pi", "scale_opencode"}
    assert parse_multi_harness(None) is None


@pytest.mark.parametrize(
    "config",
    [
        {"routes": _ROUTES, "weight": {"opencode_agent": 1}},
        {"routes": {}},
        {"routes": {"swe": []}},
        {"routes": {"swe": ["swe_pi", "swe_pi"]}},
        {"routes": {"swe": [""]}},
        {"routes": _ROUTES, "weights": {}},
        {"routes": _ROUTES, "weights": {"opencode_agent": -1}},
        {"routes": _ROUTES, "weights": {"opencode_agent": float("inf")}},
        {"routes": _ROUTES, "weights": {"opencode_agent": "heavy"}},
    ],
    ids=[
        "unknown-key",
        "no-tasksets",
        "no-servers",
        "repeated-server",
        "empty-name",
        "empty-weights",
        "negative-weight",
        "infinite-weight",
        "non-number-weight",
    ],
)
def test_parse_rejects_malformed_blocks(config):
    with pytest.raises(ValueError):
        parse_multi_harness(config)


def _agent(agent_type, resources=None):
    agent_config = {"entrypoint": "app.py"}
    if resources is not None:
        agent_config["resources_server"] = {
            "type": "resources_servers",
            "name": resources,
        }
    return {"responses_api_agents": {agent_type: agent_config}}


def _gym_server(server_type, agent, **fields):
    return {
        "environment_servers": {
            server_type: {
                "agent_server": {"type": "responses_api_agents", "name": agent},
                **fields,
            }
        }
    }


# The describe tests use Gym's own attribution helper, so they need Gym.
@pytest.mark.nemo_gym
def test_describe_names_each_servers_agent_as_gym_attributes_it():
    resources = {"type": "resources_servers", "name": "swe_rebench"}
    global_config = OmegaConf.create(
        {
            "swe_opencode": _gym_server(
                "single_agent_turn", "oc", resources_server=resources
            ),
            "hb_pi": _gym_server("harbor", "pi", tasksets={"hb": {}, "tb": {}}),
            "user_sim": {"environment_servers": {"nemo_user_sim_environment": {}}},
            "oc": _agent("opencode_agent"),
            "pi": _agent("pi_agent"),
        }
    )

    # The agent's name in Gym's config, which agent_server refers to, not its type.
    assert describe_environment_server("swe_opencode", global_config) == _turn("oc")
    assert describe_environment_server("hb_pi", global_config) == EnvironmentServerInfo(
        "harbor", "pi", None, ("hb", "tb")
    )
    # A server that binds no single agent has no harness.
    assert describe_environment_server("user_sim", global_config).agent is None
    assert describe_environment_server("oc", global_config) is None
    assert describe_environment_server("missing", global_config) is None


@pytest.mark.nemo_gym
def test_describe_falls_back_to_the_agents_verifier_as_gym_does():
    resources = {"type": "resources_servers", "name": "explicit"}
    global_config = OmegaConf.create(
        {
            "inherits": _gym_server("single_agent_turn_legacy", "simple"),
            "explicit": _gym_server(
                "single_agent_turn", "simple", resources_server=resources
            ),
            "simple": _agent("simple_agent", resources="from_agent"),
        }
    )

    assert describe_environment_server("inherits", global_config).resources_server == (
        "from_agent"
    )
    assert describe_environment_server("explicit", global_config).resources_server == (
        "explicit"
    )


@pytest.mark.nemo_gym
def test_harnesses_of_one_agent_type_are_told_apart_by_agent_name():
    # Gym's harness_agent runs Pi, Codex or another CLI under one type.
    resources = {"type": "resources_servers", "name": "workspace"}
    global_config = OmegaConf.create(
        {
            "ws_pi": _gym_server(
                "single_agent_turn", "ws_pi_agent", resources_server=resources
            ),
            "ws_codex": _gym_server(
                "single_agent_turn", "ws_codex_agent", resources_server=resources
            ),
            "ws_pi_agent": _agent("harness_agent"),
            "ws_codex_agent": _agent("harness_agent"),
        }
    )
    servers = {
        name: describe_environment_server(name, global_config)
        for name in ("ws_pi", "ws_codex")
    }

    routes = resolve_multi_harness(
        parse_multi_harness(
            {
                "routes": {"ws": ["ws_pi", "ws_codex"]},
                "weights": {"ws_pi_agent": 2, "ws_codex_agent": 1},
            }
        ),
        servers,
    )

    assert routes.candidates == (
        ("ws", (("ws_codex_agent", "ws_codex", 1.0), ("ws_pi_agent", "ws_pi", 2.0))),
    )


def test_resolve_weights_each_tasksets_servers_by_agent():
    routes = resolve_multi_harness(
        parse_multi_harness(
            {"routes": _ROUTES, "weights": {"opencode_agent": 2, "pi_agent": 0}}
        ),
        _SERVERS,
    )

    assert routes.candidates == (
        ("scale", (("opencode_agent", "scale_opencode", 2.0),)),
        ("swe", (("opencode_agent", "swe_opencode", 2.0),)),
    )
    assert routes.covers("swe") and not routes.covers("math")
    # Pi has weight 0, so no group is ever sent to its server.
    assert routes.servers == {"swe_opencode", "scale_opencode"}
    assert resolve_multi_harness(
        parse_multi_harness({"routes": _ROUTES}), _SERVERS
    ).servers_for("swe") == ("swe_opencode", "swe_pi")


@pytest.mark.parametrize(
    ("servers", "weights", "message"),
    [
        ({**_SERVERS, "swe_pi": None}, None, "not an Environment Server"),
        ({**_SERVERS, "swe_pi": _turn(None)}, None, "runs one agent"),
        (
            {**_SERVERS, "swe_pi": _turn("opencode_agent")},
            None,
            "must run a different agent",
        ),
        (
            {
                **_SERVERS,
                "swe_pi": EnvironmentServerInfo("harbor", "pi_agent", None, ("swe",)),
            },
            None,
            "mixes Environment Server types",
        ),
        (
            {**_SERVERS, "swe_pi": _turn("pi_agent", "scale_swe")},
            None,
            "different resources servers",
        ),
        (
            {
                **_SERVERS,
                "swe_opencode": EnvironmentServerInfo(
                    "harbor", "opencode_agent", None, ("other",)
                ),
                "swe_pi": EnvironmentServerInfo("harbor", "pi_agent", None, ("swe",)),
            },
            None,
            "does not serve taskset 'swe'",
        ),
        (_SERVERS, {"opencode_agent": 1}, r"no weight for agents \['pi_agent'\]"),
        (
            _SERVERS,
            {"opencode_agent": 1, "pi_agent": 1, "opencod_agent": 1},
            "names agents",
        ),
        (_SERVERS, {"opencode_agent": 0, "pi_agent": 1}, "'scale' has no agent"),
    ],
    ids=[
        "not-described",
        "no-single-agent",
        "same-agent-twice",
        "mixed-types",
        "different-verifiers",
        "harbor-without-taskset",
        "missing-weight",
        "unused-weight",
        "all-weights-zero",
    ],
)
def test_resolve_rejects_servers_that_would_run_or_score_differently(
    servers, weights, message
):
    config = parse_multi_harness(
        {"routes": _ROUTES, **({"weights": weights} if weights else {})}
    )
    present = {name: info for name, info in servers.items() if info is not None}

    with pytest.raises(ValueError, match=message):
        resolve_multi_harness(config, present)


def _routes(*candidates):
    return MultiHarnessRoutes(candidates=(("swe", candidates),))


# Huge weights must not overflow their sum, nor tiny ones lose precision.
@pytest.mark.parametrize(
    ("opencode", "pi", "opencode_share"),
    [(2.0, 1.0, 2 / 3), (1e308, 1e308, 1 / 2), (5e-324, 5e-324, 1 / 2)],
    ids=["two-to-one", "huge-equal", "tiny-equal"],
)
def test_select_follows_the_weights(opencode, pi, opencode_share):
    routes = _routes(
        ("opencode_agent", "swe_opencode", opencode), ("pi_agent", "swe_pi", pi)
    )

    counts = Counter(routes.select("swe", f"group-{i}").harness for i in range(20_000))

    assert counts["opencode_agent"] / 20_000 == pytest.approx(opencode_share, abs=0.02)


def test_select_depends_only_on_the_group_id():
    routes = _routes(
        ("hermes_agent", "swe_hermes", 1.0),
        ("opencode_agent", "swe_opencode", 1.0),
        ("pi_agent", "swe_pi", 1.0),
    )

    for i in range(100):
        assert len({routes.select("swe", f"group-{i}") for _ in range(3)}) == 1
    # Pinned values: the driver and every Gym actor, in their own processes,
    # must agree, and a restored group must keep its server.
    assert routes.select("swe", "group-0") == HarnessChoice("pi_agent", "swe_pi")
    assert routes.select("swe", "group-2") == HarnessChoice(
        "opencode_agent", "swe_opencode"
    )


def test_select_requires_a_group_id():
    with pytest.raises(ValueError, match="prompt-group ID"):
        _routes(("pi_agent", "swe_pi", 1.0)).select("swe", None)


def test_routes_state_records_the_selection_rule_and_every_candidate():
    routes = _routes(
        ("opencode_agent", "swe_opencode", 2.0), ("pi_agent", "swe_pi", 1.0)
    )

    assert routes.agents_for("swe") == ("opencode_agent", "pi_agent")
    # A rollout checkpoint saves this; a resume compares it with its own routes.
    assert routes.to_state() == {
        "selection": "nemo-rl-multi-harness-v1",
        "candidates": [
            [
                "swe",
                [["opencode_agent", "swe_opencode", 2.0], ["pi_agent", "swe_pi", 1.0]],
            ]
        ],
    }


def test_metric_namespace_adds_the_agent_only_for_multi_harness_results():
    assert harness_metric_namespace("swe", {"_ng_harness": "pi_agent"}) == (
        "swe/pi_agent"
    )
    assert harness_metric_namespace("swe", {"reward": 1.0}) == "swe"
    assert harness_metric_namespace("math_agent", None) == "math_agent"

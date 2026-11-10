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
"""Parse, select and check ``env.nemo_gym.harness_mix`` without Gym installed."""

from collections import Counter

import pytest
from omegaconf import OmegaConf

from nemo_rl.environments.nemo_gym_harness_mix import (
    HarnessChoice,
    harness_metric_namespace,
    parse_harness_mix,
    validate_harness_mix_servers,
)

_SERVERS = {
    "swe": {"opencode": "swe_opencode", "pi": "swe_pi", "hermes": "swe_hermes"},
    "scale": {"opencode": "scale_opencode", "pi": "scale_pi"},
}


def _mix(**overrides):
    return parse_harness_mix({"environment_servers": _SERVERS, **overrides})


def test_parse_defaults_every_harness_to_weight_one():
    mix = _mix()

    assert mix.weights == (("hermes", 1.0), ("opencode", 1.0), ("pi", 1.0))
    assert mix.covers("swe") and mix.covers("scale")
    assert not mix.covers("math")
    assert mix.environment_servers_for("scale") == ("scale_opencode", "scale_pi")
    assert mix.all_environment_servers == {
        "swe_opencode",
        "swe_pi",
        "swe_hermes",
        "scale_opencode",
        "scale_pi",
    }


def test_parse_accepts_omegaconf_and_ignores_key_order():
    reordered = {
        "scale": {"pi": "scale_pi", "opencode": "scale_opencode"},
        "swe": {"hermes": "swe_hermes", "pi": "swe_pi", "opencode": "swe_opencode"},
    }
    from_yaml = parse_harness_mix(
        OmegaConf.create(
            {
                "weights": {"pi": 1, "opencode": 2, "hermes": 1},
                "environment_servers": reordered,
            }
        )
    )

    assert from_yaml == _mix(weights={"opencode": 2, "pi": 1, "hermes": 1})


def test_parse_returns_none_when_unset():
    assert parse_harness_mix(None) is None


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (["swe_opencode"], "must be a mapping"),
        ({"environment_servers": _SERVERS, "weight": {}}, "unknown keys"),
        ({}, "must map each taskset"),
        ({"environment_servers": {"swe": {}}}, "nonempty taskset names"),
        ({"environment_servers": {"swe": {"opencode": 3}}}, "nonempty harness names"),
        (
            {
                "environment_servers": {
                    "swe": {"opencode": "shared"},
                    "scale": {"pi": "shared"},
                }
            },
            "a server runs one harness",
        ),
        (
            {"environment_servers": _SERVERS, "weights": {"opencode": 1, "pi": 1}},
            r"no weight for harnesses \['hermes'\]",
        ),
        (
            {
                "environment_servers": _SERVERS,
                "weights": {"opencode": 1, "pi": 1, "hermes": 1, "codex": 1},
            },
            "no taskset in environment_servers uses",
        ),
        (
            {
                "environment_servers": _SERVERS,
                "weights": {"opencode": 1, "pi": -1, "hermes": 1},
            },
            "finite numbers >= 0",
        ),
        (
            {
                "environment_servers": _SERVERS,
                "weights": {"opencode": 1, "pi": True, "hermes": 1},
            },
            "finite numbers >= 0",
        ),
        (
            {
                "environment_servers": _SERVERS,
                "weights": {"opencode": 1, "pi": float("inf"), "hermes": 1},
            },
            "finite numbers >= 0",
        ),
        (
            {
                "environment_servers": _SERVERS,
                "weights": {"opencode": 0, "pi": 0, "hermes": 1},
            },
            "'scale' has no harness with a positive weight",
        ),
    ],
)
def test_parse_rejects_invalid_configs(config, message):
    with pytest.raises(ValueError, match=message):
        parse_harness_mix(config)


def test_select_follows_the_weights_and_skips_disabled_harnesses():
    mix = _mix(weights={"opencode": 2, "pi": 1, "hermes": 0})

    counts = Counter(mix.select("swe", f"group-{i}").harness for i in range(20_000))

    assert set(counts) == {"opencode", "pi"}
    assert counts["opencode"] / 20_000 == pytest.approx(2 / 3, abs=0.02)
    # A disabled harness's server is never sent work, so setup need not host it.
    assert mix.environment_servers_for("swe") == ("swe_opencode", "swe_pi")
    assert "swe_hermes" not in mix.all_environment_servers


def test_select_depends_only_on_the_group_id():
    mix = _mix()

    # Each group gets one answer, however often it is asked...
    for i in range(100):
        group_id = f"group-{i}"
        assert len({mix.select("swe", group_id) for _ in range(3)}) == 1
    # ...and the answer is a stable function of the ID, so groups restored
    # from a checkpoint in a new process keep their harness. These values pin
    # the selection rule; changing it reassigns restored groups.
    assert mix.select("swe", "group-0") == HarnessChoice("opencode", "swe_opencode")
    assert mix.select("swe", "group-1") == HarnessChoice("hermes", "swe_hermes")
    assert mix.select("swe", "group-2") == HarnessChoice("pi", "swe_pi")
    assert mix.select("scale", "group-2") == HarnessChoice("pi", "scale_pi")


def test_select_requires_a_group_id():
    with pytest.raises(ValueError, match="prompt-group ID"):
        _mix().select("swe", None)  # type: ignore[arg-type]


def _single_agent_turn(resources_server):
    return {
        "environment_servers": {
            "single_agent_turn": {
                "agent_server": {"type": "responses_api_agents", "name": "agent"},
                "resources_server": {
                    "type": "resources_servers",
                    "name": resources_server,
                },
            }
        }
    }


def _harbor(*tasksets):
    return {
        "environment_servers": {
            "harbor": {
                "agent_server": {"type": "responses_api_agents", "name": "agent"},
                "tasksets": {taskset: {"path": f"/{taskset}"} for taskset in tasksets},
            }
        }
    }


def _swe_only(**weights):
    return parse_harness_mix(
        {
            "environment_servers": {"swe": {"opencode": "oc", "pi": "pi"}},
            **({"weights": weights} if weights else {}),
        }
    )


def test_validate_accepts_servers_that_share_a_verifier():
    validate_harness_mix_servers(
        _swe_only(),
        {
            "oc": _single_agent_turn("swe_rebench"),
            "pi": _single_agent_turn("swe_rebench"),
        },
    )
    validate_harness_mix_servers(
        _swe_only(), {"oc": _harbor("swe"), "pi": _harbor("swe")}
    )


@pytest.mark.parametrize(
    ("configs", "message"),
    [
        (
            {"oc": _single_agent_turn("swe_rebench")},
            "'pi', which is not an Environment Server",
        ),
        (
            {
                "oc": _single_agent_turn("swe_rebench"),
                "pi": {"responses_api_agents": {}},
            },
            "not an Environment Server",
        ),
        (
            {
                "oc": _single_agent_turn("swe_rebench"),
                "pi": _single_agent_turn("scale_swe"),
            },
            "different resources servers",
        ),
        (
            {"oc": _single_agent_turn("swe_rebench"), "pi": _harbor("swe")},
            "mixes Environment Server types",
        ),
        (
            {"oc": _harbor("swe"), "pi": _harbor("other")},
            "'pi' does not serve taskset 'swe'",
        ),
    ],
)
def test_validate_rejects_servers_that_would_score_differently(configs, message):
    with pytest.raises(ValueError, match=message):
        validate_harness_mix_servers(_swe_only(), configs)


def test_validate_skips_disabled_harnesses():
    validate_harness_mix_servers(
        _swe_only(opencode=1, pi=0), {"oc": _single_agent_turn("swe_rebench")}
    )


def test_metric_namespace_adds_the_harness_only_for_mixed_results():
    assert (
        harness_metric_namespace("swe", {"_ng_harness": "pi", "reward": 1.0})
        == "swe/pi"
    )
    assert harness_metric_namespace("swe", {"reward": 1.0}) == "swe"
    assert harness_metric_namespace("math_agent", None) == "math_agent"

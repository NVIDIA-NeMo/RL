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
"""Unit tests for the NeMo-Gym shard schema (pure Python, no Gym import)."""

import pytest
from omegaconf import OmegaConf

from nemo_rl.environments.nemo_gym_shards import (
    GYM_LOG_DIR_KEY,
    ShardConfigError,
    ShardPlan,
    ShardSetupError,
    ShardSpec,
    apply_shard_log_dir,
    apply_shard_overlay,
    build_agent_shard_map,
    find_gym_config_entries,
    parse_shard_plan,
)


def _sharded_config(**overrides):
    config = {
        "uv_venv_dir": "/opt/gym_venvs",
        "num_gpu_nodes": 2,
        "shards": [
            {"name": "judged", "config_paths": ["judge.yaml"]},
            {"name": "tools", "config_paths": ["proxy.yaml", "tools.yaml"]},
        ],
    }
    config.update(overrides)
    return config


def test_unsharded_config_returns_no_plan():
    """config_paths without shards is the pre-sharding path and must stay working."""
    assert parse_shard_plan({"config_paths": ["gym.yaml"], "num_gpu_nodes": 1}) is None


def test_parse_shard_plan_reads_shards_and_common_settings():
    plan = parse_shard_plan(
        _sharded_config(
            common_overrides={"policy_model": {"num_workers": 4}},
            allowed_duplicate_entries=["policy_model"],
        )
    )

    assert [shard.name for shard in plan.shards] == ["judged", "tools"]
    assert plan.shards[1].config_paths == ["proxy.yaml", "tools.yaml"]
    assert plan.common_overrides == {"policy_model": {"num_workers": 4}}
    assert plan.allowed_duplicate_entries == frozenset({"policy_model"})


def test_parse_shard_plan_accepts_omegaconf_input():
    """Job configs arrive as OmegaConf nodes, not plain dicts."""
    plan = parse_shard_plan(OmegaConf.create(_sharded_config()))

    assert [shard.name for shard in plan.shards] == ["judged", "tools"]
    assert isinstance(plan.shards[0].config_paths, list)


def test_replicas_count_toward_total_instances():
    plan = parse_shard_plan(
        _sharded_config(
            shards=[
                {"name": "judged", "config_paths": ["judge.yaml"]},
                {"name": "tools", "config_paths": ["tools.yaml"], "replicas": 3},
            ]
        )
    )

    assert plan.total_instances == 4


def test_top_level_config_paths_conflicts_with_shards():
    with pytest.raises(ShardConfigError, match="cannot be combined with 'shards'"):
        parse_shard_plan(_sharded_config(config_paths=["gym.yaml"]))


def test_an_inherited_config_paths_can_be_retracted_with_null():
    """A Hydra override can blank a key it inherited but cannot delete it, so
    null has to read as absent or no recipe could ever be sharded."""
    plan = parse_shard_plan(_sharded_config(config_paths=None))

    assert plan is not None
    assert [shard.name for shard in plan.shards] == ["judged", "tools"]


def test_a_nulled_entry_overlay_is_not_forwarded_to_gym():
    """Gym should get the key gone, not the key set to null."""
    plan = parse_shard_plan(_sharded_config(policy_model=None))
    assert plan is not None

    merged = apply_shard_overlay({"policy_model": None}, plan, plan.shards[0])

    assert "policy_model" not in merged


def test_multi_shard_plan_rejects_an_unclaimed_inherited_overlay():
    with pytest.raises(ShardConfigError, match="have no owner"):
        parse_shard_plan(
            _sharded_config(
                nl2bash_judge_model={"responses_api_models": {"local_vllm_model": {}}}
            )
        )


def test_single_shard_plan_automatically_claims_every_inherited_overlay():
    plan = parse_shard_plan(
        _sharded_config(
            shards=[{"name": "only", "config_paths": ["gym.yaml"]}],
            policy_model={"responses_api_models": {"vllm_model": {"num_workers": 4}}},
            math={"resources_servers": {"math": {"workers": 8}}},
        )
    )

    assert plan.shards[0].inherited_overlays == frozenset({"math", "policy_model"})
    merged = apply_shard_overlay(
        _sharded_config(
            shards=[],
            policy_model={"responses_api_models": {"vllm_model": {"num_workers": 4}}},
            math={"resources_servers": {"math": {"workers": 8}}},
        ),
        plan,
        plan.shards[0],
    )
    assert (
        merged["policy_model"]["responses_api_models"]["vllm_model"]["num_workers"] == 4
    )
    assert merged["math"]["resources_servers"]["math"]["workers"] == 8


def test_multi_shard_plan_routes_inherited_overlays_by_key():
    policy_model = {"responses_api_models": {"vllm_model": {"num_workers": 4}}}
    judged = {"resources_servers": {"judge": {"workers": 8}}}
    tools = {"resources_servers": {"tools": {"workers": 16}}}
    config = _sharded_config(
        shards=[
            {
                "name": "judged",
                "config_paths": ["judge.yaml"],
                "inherited_overlays": ["judged"],
            },
            {
                "name": "tools",
                "config_paths": ["tools.yaml"],
                "inherited_overlays": ["tools"],
            },
        ],
        common_inherited_overlays=["policy_model"],
        policy_model=policy_model,
        judged=judged,
        tools=tools,
    )

    plan = parse_shard_plan(config)
    judged_config = apply_shard_overlay(config, plan, plan.shards[0])
    tools_config = apply_shard_overlay(config, plan, plan.shards[1])

    assert judged_config["policy_model"] == policy_model
    assert judged_config["judged"] == judged
    assert "tools" not in judged_config
    assert tools_config["policy_model"] == policy_model
    assert tools_config["tools"] == tools
    assert "judged" not in tools_config
    for merged in (judged_config, tools_config):
        assert "shards" not in merged
        assert "common_inherited_overlays" not in merged


def test_inherited_overlay_claims_must_be_known_and_unique():
    config = _sharded_config(
        shards=[
            {
                "name": "judged",
                "config_paths": ["judge.yaml"],
                "inherited_overlays": ["missing"],
            },
            {"name": "tools", "config_paths": ["tools.yaml"]},
        ],
        policy_model={"responses_api_models": {}},
    )
    with pytest.raises(ShardConfigError, match="Available overlays.*policy_model"):
        parse_shard_plan(config)

    config["shards"][0]["inherited_overlays"] = ["policy_model"]
    config["shards"][1]["inherited_overlays"] = ["policy_model"]
    with pytest.raises(ShardConfigError, match="duplicate claims.*policy_model"):
        parse_shard_plan(config)


def test_inherited_overlay_lists_require_unique_non_empty_strings():
    with pytest.raises(ShardConfigError, match="list of non-empty strings"):
        parse_shard_plan(
            _sharded_config(
                shards=[
                    {
                        "name": "only",
                        "config_paths": ["gym.yaml"],
                        "inherited_overlays": [""],
                    }
                ]
            )
        )

    with pytest.raises(ShardConfigError, match="contains duplicate keys"):
        parse_shard_plan(
            _sharded_config(
                shards=[{"name": "only", "config_paths": ["gym.yaml"]}],
                common_inherited_overlays=["policy_model", "policy_model"],
            )
        )


@pytest.mark.parametrize(
    ("shards", "expected"),
    [
        ([], "non-empty list"),
        ([{"config_paths": ["a.yaml"]}], "non-empty string 'name'"),
        ([{"name": "  ", "config_paths": ["a.yaml"]}], "non-empty string 'name'"),
        ([{"name": "a", "config_paths": []}], "at least one entry in 'config_paths'"),
        ([{"name": "a"}], "at least one entry in 'config_paths'"),
        (
            [{"name": "a", "config_paths": ["a.yaml"], "replicas": 0}],
            "must be an integer >= 1",
        ),
        (
            [{"name": "a", "config_paths": ["a.yaml"], "port_range_low": 5000}],
            "both port_range_low and port_range_high",
        ),
        (
            [
                {
                    "name": "a",
                    "config_paths": ["a.yaml"],
                    "port_range_low": 5200,
                    "port_range_high": 5200,
                }
            ],
            "must reserve at least two ports",
        ),
        (
            [
                {
                    "name": "a",
                    "config_paths": ["a.yaml"],
                    "port_range_low": 65000,
                    "port_range_high": 65536,
                }
            ],
            r"within \[1, 65535\]",
        ),
        (
            [{"name": "a", "config_paths": ["a.yaml"], "judge_model": {}}],
            "unrecognized keys",
        ),
        (
            [
                {"name": "dup", "config_paths": ["a.yaml"]},
                {"name": "dup", "config_paths": ["b.yaml"]},
            ],
            "Duplicate shard names",
        ),
    ],
)
def test_malformed_shards_are_rejected(shards, expected):
    with pytest.raises(ShardConfigError, match=expected):
        parse_shard_plan({"shards": shards})


@pytest.mark.parametrize("name", ["../judged", "team/judged", ".", ".."])
def test_shard_names_must_be_safe_log_path_components(name):
    with pytest.raises(ShardConfigError, match="one safe path component"):
        parse_shard_plan(
            _sharded_config(shards=[{"name": name, "config_paths": ["judge.yaml"]}])
        )


@pytest.mark.parametrize("actor_cpus", [0, -1, float("nan"), float("inf"), "8"])
def test_actor_cpus_must_be_positive_and_finite(actor_cpus):
    with pytest.raises(ShardConfigError, match="positive finite number"):
        parse_shard_plan(
            _sharded_config(
                shards=[
                    {
                        "name": "judged",
                        "config_paths": ["judge.yaml"],
                        "actor_cpus": actor_cpus,
                    }
                ]
            )
        )


def test_config_paths_and_port_types_are_validated_at_parse_time():
    with pytest.raises(
        ShardConfigError, match="config_paths must be non-empty strings"
    ):
        parse_shard_plan(
            _sharded_config(shards=[{"name": "judged", "config_paths": [123]}])
        )

    with pytest.raises(ShardConfigError, match="port range must use integers"):
        parse_shard_plan(
            _sharded_config(
                shards=[
                    {
                        "name": "judged",
                        "config_paths": ["judge.yaml"],
                        "port_range_low": "5000",
                        "port_range_high": 5500,
                    }
                ]
            )
        )


def test_shards_land_on_distinct_nodes_unless_told_otherwise():
    assert parse_shard_plan(_sharded_config()).placement_strategy == "STRICT_SPREAD"


def test_placement_strategy_can_be_relaxed_to_run_shards_on_one_machine():
    plan = parse_shard_plan(
        _sharded_config(
            placement_strategy="PACK",
            shards=[
                {
                    "name": "judged",
                    "config_paths": ["judge.yaml"],
                    "port_range_low": 5000,
                    "port_range_high": 5500,
                },
                {
                    "name": "tools",
                    "config_paths": ["tools.yaml"],
                    "port_range_low": 5501,
                    "port_range_high": 6000,
                },
            ],
        )
    )

    assert plan.placement_strategy == "PACK"


def test_relaxed_placement_requires_disjoint_explicit_port_ranges():
    with pytest.raises(ShardConfigError, match="require explicit port_range"):
        parse_shard_plan(_sharded_config(placement_strategy="PACK"))

    with pytest.raises(ShardConfigError, match="overlapping port ranges"):
        parse_shard_plan(
            _sharded_config(
                placement_strategy="PACK",
                shards=[
                    {
                        "name": "judged",
                        "config_paths": ["judge.yaml"],
                        "port_range_low": 5000,
                        "port_range_high": 5600,
                    },
                    {
                        "name": "tools",
                        "config_paths": ["tools.yaml"],
                        "port_range_low": 5600,
                        "port_range_high": 6000,
                    },
                ],
            )
        )


def test_an_unknown_placement_strategy_is_rejected():
    with pytest.raises(ShardConfigError, match="placement_strategy must be one of"):
        parse_shard_plan(_sharded_config(placement_strategy="SPRED"))


def test_allowed_duplicate_entries_must_be_strings():
    with pytest.raises(ShardConfigError, match="list of strings"):
        parse_shard_plan(_sharded_config(allowed_duplicate_entries=[{"nope": 1}]))


def test_apply_shard_overlay_layers_shard_over_common():
    plan = ShardPlan(
        shards=[],
        common_overrides={
            "policy_model": {"num_workers": 4, "timeout": 30},
            "shared_judge": {"replicas": 1},
        },
    )
    shard = ShardSpec(
        name="judged",
        config_paths=["judge.yaml"],
        overrides={"policy_model": {"num_workers": 16}},
    )

    merged = apply_shard_overlay({"default_host": "10.0.0.1"}, plan, shard)

    assert merged["config_paths"] == ["judge.yaml"]
    assert merged["default_host"] == "10.0.0.1"
    # Shard wins on conflict, and sibling keys from common survive the merge.
    assert merged["policy_model"] == {"num_workers": 16, "timeout": 30}
    assert merged["shared_judge"] == {"replicas": 1}


def test_explicit_overrides_layer_over_inherited_values():
    base = {
        "policy_model": {
            "responses_api_models": {"vllm_model": {"num_workers": 2, "timeout": 30}}
        }
    }
    plan = ShardPlan(
        shards=[],
        common_inherited_overlays=frozenset({"policy_model"}),
        common_overrides={
            "policy_model": {"responses_api_models": {"vllm_model": {"num_workers": 4}}}
        },
    )
    shard = ShardSpec(
        name="judged",
        config_paths=["judge.yaml"],
        overrides={
            "policy_model": {"responses_api_models": {"vllm_model": {"num_workers": 8}}}
        },
    )

    merged = apply_shard_overlay(base, plan, shard)

    assert merged["policy_model"]["responses_api_models"]["vllm_model"] == {
        "num_workers": 8,
        "timeout": 30,
    }


def test_apply_shard_overlay_applies_per_shard_port_range():
    shard = ShardSpec(
        name="ci",
        config_paths=["a.yaml"],
        port_range_low=6000,
        port_range_high=6099,
    )

    merged = apply_shard_overlay(
        {"port_range_low": 5000, "port_range_high": 5999}, ShardPlan(shards=[]), shard
    )

    assert (merged["port_range_low"], merged["port_range_high"]) == (6000, 6099)


def test_apply_shard_overlay_keeps_global_ports_by_default():
    merged = apply_shard_overlay(
        {"port_range_low": 5000, "port_range_high": 5999},
        ShardPlan(shards=[]),
        ShardSpec(name="prod", config_paths=["a.yaml"]),
    )

    assert (merged["port_range_low"], merged["port_range_high"]) == (5000, 5999)


def test_build_agent_shard_map_routes_each_agent_to_its_shard():
    agent_to_shard = build_agent_shard_map(
        {
            "judged": {
                "math_agent": ["responses_api_agents"],
                "math_env": ["resources_servers"],
            },
            "tools": {
                "bash_agent": ["responses_api_agents"],
                "bash_tools": ["resources_servers"],
            },
        }
    )

    assert agent_to_shard == {"math_agent": "judged", "bash_agent": "tools"}


def test_build_agent_shard_map_rejects_an_agent_in_two_shards():
    """Rows naming the agent could go to either shard, so routing is undefined."""
    with pytest.raises(ShardSetupError, match="hosted by both shard"):
        build_agent_shard_map(
            {
                "judged": {"math_agent": ["responses_api_agents"]},
                "tools": {"math_agent": ["responses_api_agents"]},
            }
        )


def test_agents_are_never_allowlisted_for_duplication():
    """The allowlist covers shared support entries, not routing ambiguity."""
    with pytest.raises(ShardSetupError, match="hosted by both shard"):
        build_agent_shard_map(
            {
                "a": {"math_agent": ["responses_api_agents"]},
                "b": {"math_agent": ["responses_api_agents"]},
            },
            allowed_duplicate_entries={"math_agent"},
        )


def test_build_agent_shard_map_rejects_an_unlisted_duplicate_entry():
    with pytest.raises(ShardSetupError, match="allowed_duplicate_entries"):
        build_agent_shard_map(
            {
                "judged": {"shared_judge": ["responses_api_models"]},
                "tools": {"shared_judge": ["responses_api_models"]},
            }
        )


def test_build_agent_shard_map_allows_a_listed_duplicate_entry():
    """Policy proxies are copied into every shard on purpose."""
    agent_to_shard = build_agent_shard_map(
        {
            "judged": {
                "math_agent": ["responses_api_agents"],
                "policy_model": ["responses_api_models"],
            },
            "tools": {
                "bash_agent": ["responses_api_agents"],
                "policy_model": ["responses_api_models"],
            },
        },
        allowed_duplicate_entries={"policy_model"},
    )

    assert agent_to_shard == {"math_agent": "judged", "bash_agent": "tools"}


def test_apply_shard_log_dir_gives_each_shard_its_own_directory():
    config = {GYM_LOG_DIR_KEY: "/logs/gym", "default_host": "10.0.0.1"}

    judged = apply_shard_log_dir(config, "judged")
    tools = apply_shard_log_dir(config, "tools")

    assert judged[GYM_LOG_DIR_KEY] == "/logs/gym/judged"
    assert tools[GYM_LOG_DIR_KEY] == "/logs/gym/tools"
    # Unrelated settings survive, and the input is not mutated.
    assert judged["default_host"] == "10.0.0.1"
    assert config[GYM_LOG_DIR_KEY] == "/logs/gym"


def test_apply_shard_log_dir_separates_replicas():
    """Replicas are stamped from one merge, so they host identical server names."""
    config = {GYM_LOG_DIR_KEY: "/logs/gym"}

    first = apply_shard_log_dir(config, "tools", replica_index=0)
    second = apply_shard_log_dir(config, "tools", replica_index=1)

    assert first[GYM_LOG_DIR_KEY] == "/logs/gym/tools/0"
    assert second[GYM_LOG_DIR_KEY] == "/logs/gym/tools/1"


def test_apply_shard_log_dir_is_a_noop_without_a_configured_log_dir():
    """Gym only writes log files when the key is set, which is not the default."""
    assert apply_shard_log_dir({"default_host": "10.0.0.1"}, "judged") == {
        "default_host": "10.0.0.1"
    }


def test_apply_shard_log_dir_sanitizes_the_shard_name():
    config = apply_shard_log_dir({GYM_LOG_DIR_KEY: "/logs"}, "team/judged")

    assert config[GYM_LOG_DIR_KEY] == "/logs/team_judged"


def test_find_gym_config_entries_ignores_scalars_and_known_keys():
    entries = find_gym_config_entries(
        {
            "uv_venv_dir": "/opt/venvs",
            "num_gpu_nodes": 2,
            "shards": [{"name": "a"}],
            "common_inherited_overlays": ["genrm_model"],
            "common_overrides": {"x": 1},
            "allowed_duplicate_entries": ["policy_model"],
            "effort_levels": {"low_weight": 0.0},
            "tokenizer_config": {"name": "policy"},
            "genrm_model": {"responses_api_models": {}},
            "safety_judge_model": {"responses_api_models": {}},
        }
    )

    assert entries == ["genrm_model", "safety_judge_model"]

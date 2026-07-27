# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""Pure-Python (vllm-free) unit tests for NeMo-Gym helpers.

These run in the default L0 suite. Keep this module free of heavy imports
(e.g. vllm) so the fast detector tests are not gated behind the nemo_gym extra.
"""

import copy
import sys
import types
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import DictConfig

from nemo_rl.environments import nemo_gym as nemo_gym_mod
from nemo_rl.environments.nemo_gym import (
    NEMO_GYM_ACTOR_FQN,
    _detect_invalid_tool_call_and_malformed_thinking,
    build_nemo_gym_config,
    get_nemo_gym_uv_cache_dir,
    get_nemo_gym_venv_dir,
    spinup_nemo_gym_actor,
)


@pytest.mark.parametrize(
    ("output_item_dict", "expected_invalid_tool_call", "expected_malformed_thinking"),
    [
        (
            {"content": [{"text": "use <tool_call>{}</tool_call>"}]},
            True,
            False,
        ),
        (
            {"content": [{"text": "final answer leaked <think>reasoning</think>"}]},
            False,
            True,
        ),
        (
            {"type": "reasoning", "summary": [{"text": "<think>a</think>"}]},
            False,
            False,
        ),
        (
            {"type": "reasoning", "summary": [{"text": "<think>a</think><think>b"}]},
            False,
            True,
        ),
        (
            {"type": "reasoning", "summary": [{"text": "bad <function_call>{}"}]},
            True,
            False,
        ),
    ],
)
def test_detect_invalid_tool_call_and_malformed_thinking(
    output_item_dict,
    expected_invalid_tool_call,
    expected_malformed_thinking,
):
    assert _detect_invalid_tool_call_and_malformed_thinking(output_item_dict) == (
        expected_invalid_tool_call,
        expected_malformed_thinking,
    )


def test_get_nemo_gym_venv_dir_returns_env_value(monkeypatch):
    monkeypatch.setenv("NEMO_GYM_VENV_DIR", "/opt/gym_venvs")
    assert get_nemo_gym_venv_dir() == "/opt/gym_venvs"


def test_get_nemo_gym_venv_dir_none_when_unset(monkeypatch):
    monkeypatch.delenv("NEMO_GYM_VENV_DIR", raising=False)
    assert get_nemo_gym_venv_dir() is None


def test_get_nemo_gym_uv_cache_dir_none_outside_container(monkeypatch):
    # Outside a container the caller should omit the arg; uv must not be invoked.
    monkeypatch.delenv("NRL_CONTAINER", raising=False)

    def _fail(*args, **kwargs):
        raise AssertionError("uv should not be invoked outside a container")

    monkeypatch.setattr(nemo_gym_mod.subprocess, "check_output", _fail)
    assert get_nemo_gym_uv_cache_dir() is None


def test_get_nemo_gym_uv_cache_dir_uses_uv_inside_container(monkeypatch):
    monkeypatch.setenv("NRL_CONTAINER", "1")
    monkeypatch.setattr(
        nemo_gym_mod.subprocess,
        "check_output",
        lambda *args, **kwargs: b"  /root/.cache/uv\n",
    )
    assert get_nemo_gym_uv_cache_dir() == "/root/.cache/uv"


def _env_configs(**overrides):
    nemo_gym = {
        "num_gpu_nodes": 1,
        "invalid_tool_call_patterns": ["bad_call"],
        "thinking_tags": ["<think>"],
        "config_paths": ["gym.yaml"],
    }
    nemo_gym.update(overrides)
    return {"nemo_gym": nemo_gym}


@pytest.fixture
def detected_uv_dirs(monkeypatch):
    """Pretend we are in a container with image-baked uv cache + venv dirs."""
    monkeypatch.setattr(
        nemo_gym_mod, "get_nemo_gym_uv_cache_dir", lambda: "/opt/nemo-gym/.uv-cache"
    )
    monkeypatch.setattr(
        nemo_gym_mod, "get_nemo_gym_venv_dir", lambda: "/opt/nemo-gym/venvs"
    )


def test_build_nemo_gym_config_splits_nemo_rl_keys(detected_uv_dirs):
    """NeMo-RL-only knobs become top-level fields; the rest is Gym's global config."""
    env_configs = _env_configs()
    env_configs_before = copy.deepcopy(env_configs)

    cfg = build_nemo_gym_config(
        env_configs,
        base_urls=["http://vllm-0"],
        model_name="test-model",
    )

    assert cfg["model_name"] == "test-model"
    assert cfg["base_urls"] == ["http://vllm-0"]
    assert cfg["invalid_tool_call_patterns"] == ["bad_call"]
    assert cfg["thinking_tags"] == ["<think>"]
    assert cfg["initial_global_config_dict"] == {
        "num_gpu_nodes": 1,
        "config_paths": ["gym.yaml"],
        "uv_cache_dir": "/opt/nemo-gym/.uv-cache",
        "uv_venv_dir": "/opt/nemo-gym/venvs",
    }
    # The caller's master_config.env must survive untouched.
    assert env_configs == env_configs_before


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ({}, ("/opt/nemo-gym/.uv-cache", "/opt/nemo-gym/venvs")),
        (
            {"uv_cache_dir": "/custom/cache", "uv_venv_dir": "/custom/venvs"},
            ("/custom/cache", "/custom/venvs"),
        ),
    ],
    ids=["detected", "explicit-wins"],
)
def test_build_nemo_gym_config_uv_dirs(detected_uv_dirs, configured, expected):
    cfg = build_nemo_gym_config(
        _env_configs(**configured),
        base_urls=[],
        model_name="test-model",
    )
    global_config = cfg["initial_global_config_dict"]
    assert (global_config["uv_cache_dir"], global_config["uv_venv_dir"]) == expected


def test_build_nemo_gym_config_router_replay_off_uses_default_dtype(detected_uv_dirs):
    cfg = build_nemo_gym_config(_env_configs(), base_urls=[], model_name="test-model")
    assert cfg["require_routed_experts"] is False
    assert cfg["routed_experts_dtype"] == "int16"


def test_build_nemo_gym_config_router_replay_resolves_dtype(detected_uv_dirs):
    with patch(
        "nemo_rl.models.generation.interfaces.resolve_routed_experts_dtype_name_for_model",
        return_value="int8",
    ) as mock_resolve:
        cfg = build_nemo_gym_config(
            _env_configs(),
            base_urls=[],
            model_name="test-model",
            enable_router_replay=True,
        )

    mock_resolve.assert_called_once_with("test-model")
    assert cfg["require_routed_experts"] is True
    assert cfg["routed_experts_dtype"] == "int8"


@pytest.mark.parametrize("num_gpu_nodes", [0, 1], ids=["no-gpus", "colocated-gpus"])
def test_spinup_nemo_gym_actor(detected_uv_dirs, num_gpu_nodes):
    """The actor gets the registry runtime_env, and node affinity only when it
    has colocated GPUs to land next to."""
    actor = MagicMock()
    actor._spinup.remote.return_value = "spinup-ref"
    runtime_env = {"py_executable": "/venv/bin/python"}

    with (
        patch.object(
            nemo_gym_mod, "make_actor_runtime_env", return_value=runtime_env
        ) as mock_runtime_env,
        patch.object(nemo_gym_mod, "NemoGym") as mock_cls,
        patch.object(nemo_gym_mod, "ray") as mock_ray,
    ):
        mock_cls.options.return_value.remote.return_value = actor
        mock_ray.get_runtime_context.return_value.get_node_id.return_value = "a" * 56

        result = spinup_nemo_gym_actor(
            _env_configs(num_gpu_nodes=num_gpu_nodes),
            base_urls=["http://vllm-0"],
            model_name="test-model",
            use_fastokens=True,
        )

    assert result is actor
    mock_runtime_env.assert_called_once_with(NEMO_GYM_ACTOR_FQN)

    options_kwargs = mock_cls.options.call_args.kwargs
    assert options_kwargs["runtime_env"] is runtime_env
    if num_gpu_nodes:
        assert isinstance(
            options_kwargs["scheduling_strategy"],
            nemo_gym_mod.NodeAffinitySchedulingStrategy,
        )
        assert options_kwargs["scheduling_strategy"].node_id == "a" * 56
    else:
        assert "scheduling_strategy" not in options_kwargs

    cfg = mock_cls.options.return_value.remote.call_args.args[0]
    assert cfg["use_fastokens"] is True

    # Spinup is deferred from __init__, so the factory must await it.
    actor._spinup.remote.assert_called_once_with()
    mock_ray.get.assert_called_once_with("spinup-ref")


def test_nemo_gym_fails_fast_instead_of_restarting():
    """A restarted actor would be permanently broken.

    __init__ only stores cfg; the Gym servers are created in _spinup, which Ray
    does not re-run after a restart. Restarting would raise AttributeError on
    every later call instead of surfacing RayActorError to the caller.
    """
    metadata = nemo_gym_mod.NemoGym.__ray_metadata__
    assert metadata.max_restarts == 0
    assert metadata.max_task_retries == 0


def test_nemo_gym_shutdown_is_idempotent():
    actor = nemo_gym_mod.NemoGym.__ray_metadata__.modified_class.__new__(
        nemo_gym_mod.NemoGym.__ray_metadata__.modified_class
    )
    actor.rh = MagicMock()
    run_helper = actor.rh

    actor.shutdown()
    actor.shutdown()

    run_helper.shutdown.assert_called_once_with()


def test_nemo_gym_shutdown_before_spinup_is_a_noop():
    cls = nemo_gym_mod.NemoGym.__ray_metadata__.modified_class
    actor = cls.__new__(cls)
    actor.__init__({})

    actor.shutdown()  # must not raise


@contextmanager
def _stub_gym_resolved_config(resolved):
    """Stand in for nemo_gym.global_config, which lives in the actor's venv."""
    package = types.ModuleType("nemo_gym")
    module = types.ModuleType("nemo_gym.global_config")
    module.get_global_config_dict = lambda: resolved
    with patch.dict(
        sys.modules, {"nemo_gym": package, "nemo_gym.global_config": module}
    ):
        yield


def _spun_up_actor():
    cls = nemo_gym_mod.NemoGym.__ray_metadata__.modified_class
    actor = cls.__new__(cls)
    actor.__init__({})
    actor.rh = MagicMock()
    return actor


def test_list_entries_reports_entry_names_and_server_types():
    resolved = DictConfig(
        {
            "math_agent": {"responses_api_agents": {"simple_agent": {}}},
            "math_env": {"resources_servers": {"math": {}}},
            # An entry can carry more than one server type.
            "judge": {
                "responses_api_models": {"local_vllm_model": {}},
                "resources_servers": {"judge_tools": {}},
            },
            # Plain Gym settings are not entries.
            "port_range_low": 5000,
            "default_host": "10.0.0.1",
            "config_paths": ["a.yaml"],
        }
    )

    with _stub_gym_resolved_config(resolved):
        entries = _spun_up_actor().list_entries()

    assert entries == {
        "math_agent": ["responses_api_agents"],
        "math_env": ["resources_servers"],
        "judge": ["responses_api_models", "resources_servers"],
    }


def test_list_entries_skips_dicts_that_hold_no_server_type():
    """A dict-shaped setting is not an entry unless it nests a server type."""
    resolved = DictConfig(
        {
            "real_entry": {"resources_servers": {"env": {}}},
            "some_setting": {"nested": "value"},
        }
    )

    with _stub_gym_resolved_config(resolved):
        entries = _spun_up_actor().list_entries()

    assert entries == {"real_entry": ["resources_servers"]}


def test_list_entries_before_spinup_raises():
    cls = nemo_gym_mod.NemoGym.__ray_metadata__.modified_class
    actor = cls.__new__(cls)
    actor.__init__({})

    with pytest.raises(RuntimeError, match="call _spinup"):
        actor.list_entries()


class _FakeGymCluster:
    """Stands in for Ray while the factory creates, starts and queries actors.

    Each actor returns tagged sentinels from ``.remote()`` so ``ray.get`` can
    tell spinups from entry queries and fail whichever the test asks it to.
    """

    def __init__(
        self, *, entries_by_index=None, spinup_failures=None, pg_ready_error=None
    ):
        self.entries_by_index = entries_by_index or {}
        self.spinup_failures = spinup_failures or {}
        self.pg_ready_error = pg_ready_error
        self.actors = []
        self.actor_options = []
        self.actor_configs = []
        self.removed_placement_groups = []
        self.placement_group_calls = []
        self.placement_group = MagicMock(name="placement_group")
        self.placement_group.ready.return_value = "pg-ready"

    def make_placement_group(self, **kwargs):
        self.placement_group_calls.append(kwargs)
        return self.placement_group

    def remove_placement_group(self, pg):
        self.removed_placement_groups.append(pg)

    def make_actor_class(self):
        actor_class = MagicMock(name="NemoGym")

        def options(**option_kwargs):
            holder = MagicMock()

            def remote(config):
                index = len(self.actors)
                actor = MagicMock(name=f"gym-actor-{index}")
                actor._spinup.remote.return_value = ("spinup", index)
                actor.list_entries.remote.return_value = ("entries", index)
                self.actors.append(actor)
                self.actor_options.append(option_kwargs)
                self.actor_configs.append(config)
                return actor

            holder.remote = remote
            return holder

        actor_class.options = options
        return actor_class

    def ray_get(self, reference, timeout=None):
        if reference == "pg-ready":
            if self.pg_ready_error is not None:
                raise self.pg_ready_error
            return None
        kind, index = reference
        if kind == "spinup":
            failure = self.spinup_failures.get(index)
            if failure is not None:
                raise failure
            return None
        return self.entries_by_index.get(index, {})


@contextmanager
def _patched_cluster(cluster):
    with (
        patch.object(nemo_gym_mod, "NemoGym", cluster.make_actor_class()),
        patch.object(
            nemo_gym_mod, "make_actor_runtime_env", return_value={"py_executable": "p"}
        ),
        patch.object(
            nemo_gym_mod, "placement_group", side_effect=cluster.make_placement_group
        ),
        patch.object(
            nemo_gym_mod,
            "remove_placement_group",
            side_effect=cluster.remove_placement_group,
        ),
        patch.object(nemo_gym_mod, "shutdown_environments") as shutdown,
        patch.object(nemo_gym_mod, "ray") as mock_ray,
    ):
        mock_ray.get.side_effect = cluster.ray_get
        mock_ray.get_runtime_context.return_value.get_node_id.return_value = "a" * 56
        cluster.shutdown_environments = shutdown
        yield cluster


def _shard_env_configs(**overrides):
    nemo_gym = {
        "num_gpu_nodes": 1,
        "shards": [
            {"name": "judged", "config_paths": ["judge.yaml"]},
            {"name": "tools", "config_paths": ["tools.yaml"], "replicas": 2},
        ],
        "allowed_duplicate_entries": ["policy_model"],
        "nemo_gym_log_dir": "/logs/gym",
    }
    nemo_gym.update(overrides)
    return {"nemo_gym": nemo_gym}


def test_build_nemo_gym_actors_unsharded_makes_exactly_one_actor(detected_uv_dirs):
    """The pre-sharding path must stay identical: one actor, no placement group."""
    cluster = _FakeGymCluster()

    with _patched_cluster(cluster):
        shard_set = nemo_gym_mod.build_nemo_gym_actors(
            _env_configs(),
            base_urls=["http://vllm-0"],
            model_name="test-model",
        )

    assert not shard_set.is_sharded
    assert shard_set.sole_handle() is cluster.actors[0]
    assert cluster.placement_group_calls == []
    # Node affinity still applies when the actor has colocated GPUs.
    assert isinstance(
        cluster.actor_options[0]["scheduling_strategy"],
        nemo_gym_mod.NodeAffinitySchedulingStrategy,
    )


def test_build_nemo_gym_actors_spreads_every_replica_onto_its_own_node(
    detected_uv_dirs,
):
    cluster = _FakeGymCluster(
        entries_by_index={
            0: {"math_agent": ["responses_api_agents"]},
            1: {"bash_agent": ["responses_api_agents"]},
        }
    )

    with _patched_cluster(cluster):
        shard_set = nemo_gym_mod.build_nemo_gym_actors(
            _shard_env_configs(),
            base_urls=["http://vllm-0"],
            model_name="test-model",
        )

    # Two shards, one with replicas: 2, so three actors on three nodes.
    assert len(cluster.actors) == 3
    assert [len(replicas) for replicas in shard_set.handles.values()] == [1, 2]

    (pg_kwargs,) = cluster.placement_group_calls
    assert pg_kwargs["strategy"] == "STRICT_SPREAD"
    assert pg_kwargs["bundles"] == [
        {"CPU": float(nemo_gym_mod.DEFAULT_SHARD_CPUS)} for _ in range(3)
    ]

    # Each actor is pinned to its own bundle.
    bundle_indices = [
        options["scheduling_strategy"].placement_group_bundle_index
        for options in cluster.actor_options
    ]
    assert bundle_indices == [0, 1, 2]
    assert shard_set.agent_to_shard == {"math_agent": "judged", "bash_agent": "tools"}


def test_shards_get_their_own_config_paths_and_log_directories(detected_uv_dirs):
    cluster = _FakeGymCluster()

    with _patched_cluster(cluster):
        nemo_gym_mod.build_nemo_gym_actors(
            _shard_env_configs(),
            base_urls=["http://vllm-0"],
            model_name="test-model",
        )

    gym_configs = [
        config["initial_global_config_dict"] for config in cluster.actor_configs
    ]
    assert [config["config_paths"] for config in gym_configs] == [
        ["judge.yaml"],
        ["tools.yaml"],
        ["tools.yaml"],
    ]
    # A single-replica shard needs no replica component; replicas of one shard
    # must not share a directory, since they spawn identical server names.
    assert [config["nemo_gym_log_dir"] for config in gym_configs] == [
        "/logs/gym/judged",
        "/logs/gym/tools/0",
        "/logs/gym/tools/1",
    ]
    # NeMo-RL-only keys must never reach Gym.
    for config in gym_configs:
        assert "shards" not in config
        assert "allowed_duplicate_entries" not in config


def test_actor_cpus_override_sizes_that_shards_bundle(detected_uv_dirs):
    cluster = _FakeGymCluster()
    env_configs = _shard_env_configs(
        shards=[
            {"name": "code_gen", "config_paths": ["code.yaml"], "actor_cpus": 64},
            {"name": "tools", "config_paths": ["tools.yaml"]},
        ]
    )

    with _patched_cluster(cluster):
        nemo_gym_mod.build_nemo_gym_actors(
            env_configs, base_urls=["http://vllm-0"], model_name="test-model"
        )

    (pg_kwargs,) = cluster.placement_group_calls
    assert pg_kwargs["bundles"] == [
        {"CPU": 64.0},
        {"CPU": float(nemo_gym_mod.DEFAULT_SHARD_CPUS)},
    ]


def test_a_shard_that_fails_to_start_names_itself_and_tears_everything_down(
    detected_uv_dirs,
):
    """A ray.get timeout does not stop the actor, so cleanup must be explicit."""
    cluster = _FakeGymCluster(
        spinup_failures={1: RuntimeError("ServerRefNotFoundError: judge_model")}
    )

    with _patched_cluster(cluster):
        with pytest.raises(
            nemo_gym_mod.ShardSetupError, match="shard 'tools' \\(replica 0\\)"
        ) as excinfo:
            nemo_gym_mod.build_nemo_gym_actors(
                _shard_env_configs(),
                base_urls=["http://vllm-0"],
                model_name="test-model",
            )

    # Gym names the offending entry; we add the shard it belongs to.
    assert "ServerRefNotFoundError" in str(excinfo.value)
    cluster.shutdown_environments.assert_called_once()
    torn_down = cluster.shutdown_environments.call_args.args[0]
    assert len(torn_down) == 3
    assert cluster.removed_placement_groups == [cluster.placement_group]


def test_unplaceable_bundles_fail_fast_and_release_the_group(detected_uv_dirs):
    cluster = _FakeGymCluster(pg_ready_error=TimeoutError("no nodes"))

    with _patched_cluster(cluster):
        with pytest.raises(nemo_gym_mod.ShardSetupError, match="distinct nodes"):
            nemo_gym_mod.build_nemo_gym_actors(
                _shard_env_configs(),
                base_urls=["http://vllm-0"],
                model_name="test-model",
            )

    assert cluster.actors == []
    assert cluster.removed_placement_groups == [cluster.placement_group]


def test_duplicate_agent_across_shards_tears_the_set_down(detected_uv_dirs):
    cluster = _FakeGymCluster(
        entries_by_index={
            0: {"math_agent": ["responses_api_agents"]},
            1: {"math_agent": ["responses_api_agents"]},
        }
    )

    with _patched_cluster(cluster):
        with pytest.raises(nemo_gym_mod.ShardSetupError, match="hosted by both shard"):
            nemo_gym_mod.build_nemo_gym_actors(
                _shard_env_configs(),
                base_urls=["http://vllm-0"],
                model_name="test-model",
            )

    cluster.shutdown_environments.assert_called_once()
    assert cluster.removed_placement_groups == [cluster.placement_group]


def test_spinup_nemo_gym_actor_rejects_a_sharded_config(detected_uv_dirs):
    """Single-handle callers cannot serve several shards; say so up front."""
    with pytest.raises(nemo_gym_mod.ShardConfigError, match="not wired up yet"):
        spinup_nemo_gym_actor(
            _shard_env_configs(),
            base_urls=["http://vllm-0"],
            model_name="test-model",
        )


def test_sole_handle_refuses_to_pick_one_of_many():
    shard_set = nemo_gym_mod.NemoGymShardSet(
        handles={"a": [MagicMock()], "b": [MagicMock()]}
    )

    with pytest.raises(nemo_gym_mod.ShardSetupError, match="2 across shards"):
        shard_set.sole_handle()


def test_shard_set_shutdown_releases_the_placement_group_once():
    pg = MagicMock()
    shard_set = nemo_gym_mod.NemoGymShardSet(
        handles={"tools": [MagicMock(), MagicMock()]}, placement_group=pg
    )

    with (
        patch.object(nemo_gym_mod, "shutdown_environments") as shutdown,
        patch.object(nemo_gym_mod, "remove_placement_group") as remove,
    ):
        shard_set.shutdown()
        shard_set.shutdown()

    assert shutdown.call_count == 2
    # Releasing a group twice raises; the second shutdown must not try.
    remove.assert_called_once_with(pg)

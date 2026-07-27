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
import json
import sys
import types
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest
from omegaconf import DictConfig

from nemo_rl.environments import nemo_gym as nemo_gym_mod
from nemo_rl.environments.nemo_gym import (
    NEMO_GYM_ACTOR_FQN,
    _detect_invalid_tool_call_and_malformed_thinking,
    build_nemo_gym_actors,
    build_nemo_gym_config,
    get_nemo_gym_uv_cache_dir,
    get_nemo_gym_venv_dir,
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
        ({"content": None}, False, False),
        ({"content": []}, False, False),
        ({"content": [None]}, False, False),
        ({"content": [{"text": None}]}, False, False),
        ({"type": "reasoning", "summary": None}, False, False),
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


# The factory only forwards the tokenizer to the actors, so a sentinel is
# enough to check it reached every one of them.
_TOKENIZER = MagicMock(name="tokenizer")


def _env_configs(**overrides):
    nemo_gym = {
        "num_gpu_nodes": 1,
        "invalid_tool_call_patterns": ["bad_call"],
        "thinking_tags": ["<think>"],
        "tokenizer_config": {"name": "test-tokenizer"},
        "pad_dynamic_image_shapes": True,
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
        enable_router_replay=False,
        use_fastokens=False,
    )

    assert cfg["model_name"] == "test-model"
    assert cfg["base_urls"] == ["http://vllm-0"]
    assert cfg["invalid_tool_call_patterns"] == ["bad_call"]
    assert cfg["thinking_tags"] == ["<think>"]
    assert cfg["tokenizer_config"] == {"name": "test-tokenizer"}
    assert cfg["pad_dynamic_image_shapes"] is True
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
        enable_router_replay=False,
        use_fastokens=False,
    )
    global_config = cfg["initial_global_config_dict"]
    assert (global_config["uv_cache_dir"], global_config["uv_venv_dir"]) == expected


def test_build_nemo_gym_config_router_replay_off_uses_default_dtype(detected_uv_dirs):
    cfg = build_nemo_gym_config(
        _env_configs(),
        base_urls=[],
        model_name="test-model",
        enable_router_replay=False,
        use_fastokens=False,
    )
    assert cfg["require_routed_experts"] is False
    assert cfg["routed_experts_dtype"] == "int16"


def test_build_nemo_gym_config_router_replay_resolves_dtype(detected_uv_dirs):
    with patch.object(
        nemo_gym_mod,
        "resolve_routed_experts_dtype_name_for_model",
        return_value="int8",
    ) as mock_resolve:
        cfg = build_nemo_gym_config(
            _env_configs(),
            base_urls=[],
            model_name="test-model",
            enable_router_replay=True,
            use_fastokens=False,
        )

    mock_resolve.assert_called_once_with("test-model")
    assert cfg["require_routed_experts"] is True
    assert cfg["routed_experts_dtype"] == "int8"


@pytest.mark.parametrize("num_gpu_nodes", [0, 1], ids=["no-gpus", "colocated-gpus"])
def test_an_unsharded_job_gets_the_registry_runtime_env(
    detected_uv_dirs, num_gpu_nodes
):
    """Node affinity applies only when the actor has colocated GPUs to land next to."""
    actor = MagicMock()
    actor._spinup.remote.return_value = "spinup-ref"
    actor.set_tokenizer.remote.return_value = "tokenizer-ref"
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

        result = build_nemo_gym_actors(
            _env_configs(num_gpu_nodes=num_gpu_nodes),
            base_urls=["http://vllm-0"],
            model_name="test-model",
            tokenizer=_TOKENIZER,
            enable_router_replay=False,
            use_fastokens=True,
        )

    assert result.all_handles == [actor]
    assert not result.is_sharded
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

    # Spinup is deferred from __init__, so the factory must await it. The
    # tokenizer install has to follow it, not race it.
    actor._spinup.remote.assert_called_once_with()
    actor.set_tokenizer.remote.assert_called_once_with(_TOKENIZER)
    assert mock_ray.get.call_args_list == [
        call("spinup-ref"),
        call(actor.set_tokenizer.remote.return_value),
    ]


def test_nemo_gym_fails_fast_instead_of_restarting():
    """A restarted actor would be permanently broken.

    __init__ only stores cfg; the Gym servers are created in _spinup, which Ray
    does not re-run after a restart. _require_spinup() would reject every later
    call instead of surfacing RayActorError to the caller.
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
            "math_agent": {
                "responses_api_agents": {"simple_agent": {"entrypoint": "app.py"}}
            },
            "math_env": {"resources_servers": {"math": {"entrypoint": "app.py"}}},
            # An entry can carry more than one server type.
            "judge": {
                "responses_api_models": {"local_vllm_model": {"entrypoint": "app.py"}},
                "resources_servers": {"judge_tools": {"entrypoint": "app.py"}},
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
            "real_entry": {"resources_servers": {"env": {"entrypoint": "app.py"}}},
            "some_setting": {"nested": "value"},
        }
    )

    with _stub_gym_resolved_config(resolved):
        entries = _spun_up_actor().list_entries()

    assert entries == {"real_entry": ["resources_servers"]}


def test_list_entries_skips_an_entry_that_starts_no_server():
    resolved = DictConfig(
        {
            "math_env": {"resources_servers": {"math": {"entrypoint": "app.py"}}},
            "code_gen": {"resources_servers": {"code": {"host": "10.0.0.1"}}},
        }
    )

    with _stub_gym_resolved_config(resolved):
        entries = _spun_up_actor().list_entries()

    assert entries == {"math_env": ["resources_servers"]}


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
        self,
        *,
        entries_by_index=None,
        spinup_failures=None,
        spinup_timeouts=None,
        pg_ready_error=None,
    ):
        self.entries_by_index = entries_by_index or {}
        self.spinup_failures = spinup_failures or {}
        self.spinup_timeouts = set(spinup_timeouts or ())
        self.timed_out_spinups = set()
        self.pg_ready_error = pg_ready_error
        self.actors = []
        self.actor_options = []
        self.actor_configs = []
        self.removed_placement_groups = []
        self.placement_group_calls = []
        self.events = []
        self.ray_get_calls = []
        self.placement_group = MagicMock(name="placement_group")
        self.placement_group.ready.return_value = "pg-ready"

    def make_placement_group(self, **kwargs):
        self.events.append("placement_group")
        self.placement_group_calls.append(kwargs)
        return self.placement_group

    def make_runtime_env(self, _actor_class_fqn):
        self.events.append("runtime_env")
        return {"py_executable": "p"}

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
                actor.set_tokenizer.remote.return_value = ("tokenizer", index)
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
        self.ray_get_calls.append((reference, timeout))
        if reference == "pg-ready":
            if self.pg_ready_error is not None:
                raise self.pg_ready_error
            return None
        kind, index = reference
        if kind == "spinup":
            if (
                index in self.spinup_timeouts
                and index not in self.timed_out_spinups
                and timeout is not None
            ):
                self.timed_out_spinups.add(index)
                raise TimeoutError("startup deadline")
            failure = self.spinup_failures.get(index)
            if failure is not None:
                raise failure
            return None
        if kind == "tokenizer":
            return None
        return self.entries_by_index.get(index, {})


@contextmanager
def _patched_cluster(cluster):
    with (
        patch.object(nemo_gym_mod, "NemoGym", cluster.make_actor_class()),
        patch.object(
            nemo_gym_mod,
            "make_actor_runtime_env",
            side_effect=cluster.make_runtime_env,
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
            tokenizer=_TOKENIZER,
            enable_router_replay=False,
            use_fastokens=False,
        )

    assert not shard_set.is_sharded
    assert shard_set.all_handles == [cluster.actors[0]]
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
            tokenizer=_TOKENIZER,
            enable_router_replay=False,
            use_fastokens=False,
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
    assert cluster.events[:2] == ["runtime_env", "placement_group"]


def test_shards_get_their_own_config_paths_and_log_directories(detected_uv_dirs):
    cluster = _FakeGymCluster()

    with _patched_cluster(cluster):
        nemo_gym_mod.build_nemo_gym_actors(
            _shard_env_configs(),
            base_urls=["http://vllm-0"],
            model_name="test-model",
            tokenizer=_TOKENIZER,
            enable_router_replay=False,
            use_fastokens=False,
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


def test_every_replica_gets_the_tokenizer_installed(detected_uv_dirs):
    """A replica missing the tokenizer fails on its first rollout, not at setup.

    Skipping one is therefore silent until the run is already underway, so the
    install is asserted per actor rather than once for the set.
    """
    cluster = _FakeGymCluster()

    with _patched_cluster(cluster):
        nemo_gym_mod.build_nemo_gym_actors(
            _shard_env_configs(),
            base_urls=["http://vllm-0"],
            model_name="test-model",
            tokenizer=_TOKENIZER,
            enable_router_replay=False,
            use_fastokens=False,
        )

    assert len(cluster.actors) == 3
    for actor in cluster.actors:
        actor.set_tokenizer.remote.assert_called_once_with(_TOKENIZER)


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
            env_configs,
            base_urls=["http://vllm-0"],
            model_name="test-model",
            tokenizer=_TOKENIZER,
            enable_router_replay=False,
            use_fastokens=False,
        )

    (pg_kwargs,) = cluster.placement_group_calls
    assert pg_kwargs["bundles"] == [
        {"CPU": 64.0},
        {"CPU": float(nemo_gym_mod.DEFAULT_SHARD_CPUS)},
    ]


def test_shard_spinup_timeout_is_one_global_deadline():
    """One budget covers starting every shard and installing every tokenizer.

    A per-wait timeout would let each shard spend the full budget, and the
    tokenizer pass restarting it would double the worst case again.
    """
    first = MagicMock(name="first")
    second = MagicMock(name="second")
    first._spinup.remote.return_value = "first-ref"
    second._spinup.remote.return_value = "second-ref"
    first.set_tokenizer.remote.return_value = "first-tokenizer-ref"
    second.set_tokenizer.remote.return_value = "second-tokenizer-ref"
    shard_set = nemo_gym_mod.NemoGymShardSet(
        handles={"first": [first], "second": [second]}
    )

    with (
        patch.object(
            nemo_gym_mod,
            "monotonic",
            side_effect=[100.0, 101.0, 104.0, 104.5, 104.75],
        ),
        patch.object(nemo_gym_mod.ray, "get") as ray_get,
    ):
        nemo_gym_mod._spinup_shards_concurrently(
            shard_set, spinup_timeout=5.0, tokenizer=_TOKENIZER
        )

    assert [invocation.args[0] for invocation in ray_get.call_args_list] == [
        "first-ref",
        "second-ref",
        "first-tokenizer-ref",
        "second-tokenizer-ref",
    ]
    # Set at 100.0 + 5.0, so each wait gets only what is left of 105.0.
    assert [invocation.kwargs["timeout"] for invocation in ray_get.call_args_list] == [
        4.0,
        1.0,
        0.5,
        0.25,
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
                tokenizer=_TOKENIZER,
                enable_router_replay=False,
                use_fastokens=False,
            )

    # Gym names the offending entry; we add the shard it belongs to.
    assert "ServerRefNotFoundError" in str(excinfo.value)
    cluster.shutdown_environments.assert_called_once()
    torn_down = cluster.shutdown_environments.call_args.args[0]
    assert len(torn_down) == 3
    assert cluster.removed_placement_groups == [cluster.placement_group]
    assert any(reference == ("spinup", 2) for reference, _ in cluster.ray_get_calls)


def test_unsharded_startup_failure_tears_down_the_actor(detected_uv_dirs):
    cluster = _FakeGymCluster(spinup_failures={0: RuntimeError("bad config")})

    with _patched_cluster(cluster):
        with pytest.raises(RuntimeError, match="bad config"):
            nemo_gym_mod.build_nemo_gym_actors(
                _env_configs(),
                base_urls=["http://vllm-0"],
                model_name="test-model",
                tokenizer=_TOKENIZER,
                enable_router_replay=False,
                use_fastokens=False,
            )

    cluster.shutdown_environments.assert_called_once()
    torn_down = cluster.shutdown_environments.call_args.args[0]
    assert list(torn_down.values()) == [cluster.actors[0]]


def test_timed_out_shard_startup_is_drained_before_teardown(detected_uv_dirs):
    cluster = _FakeGymCluster(spinup_timeouts={0})

    with _patched_cluster(cluster):
        with pytest.raises(nemo_gym_mod.ShardSetupError, match="failed to start"):
            nemo_gym_mod.build_nemo_gym_actors(
                _shard_env_configs(),
                base_urls=["http://vllm-0"],
                model_name="test-model",
                tokenizer=_TOKENIZER,
                enable_router_replay=False,
                use_fastokens=False,
            )

    spinup_zero_calls = [
        timeout
        for reference, timeout in cluster.ray_get_calls
        if reference == ("spinup", 0)
    ]
    assert len(spinup_zero_calls) == 2
    assert spinup_zero_calls[0] is not None
    assert spinup_zero_calls[1] is None
    cluster.shutdown_environments.assert_called_once()


def test_unplaceable_bundles_fail_fast_and_release_the_group(detected_uv_dirs):
    cluster = _FakeGymCluster(pg_ready_error=TimeoutError("no nodes"))

    with _patched_cluster(cluster):
        with pytest.raises(nemo_gym_mod.ShardSetupError, match="distinct nodes"):
            nemo_gym_mod.build_nemo_gym_actors(
                _shard_env_configs(),
                base_urls=["http://vllm-0"],
                model_name="test-model",
                tokenizer=_TOKENIZER,
                enable_router_replay=False,
                use_fastokens=False,
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
                tokenizer=_TOKENIZER,
                enable_router_replay=False,
                use_fastokens=False,
            )

    cluster.shutdown_environments.assert_called_once()
    assert cluster.removed_placement_groups == [cluster.placement_group]


def test_a_bare_actor_handle_reads_as_a_one_shard_set():
    """Call sites that predate sharding keep working without building a set."""
    handle = MagicMock()

    shard_set = nemo_gym_mod.as_nemo_gym_shard_set(handle)

    assert shard_set.all_handles == [handle]
    assert not shard_set.is_sharded
    # No map was discovered because nothing could have conflicted, so every
    # agent resolves to the only actor there is.
    assert shard_set.pick_handle("any-agent") is handle
    assert shard_set.hosted_agents == frozenset()


def test_an_existing_shard_set_is_passed_through_untouched():
    shard_set = nemo_gym_mod.NemoGymShardSet(handles={"a": [MagicMock()]})

    assert nemo_gym_mod.as_nemo_gym_shard_set(shard_set) is shard_set


def test_shard_name_of_identifies_a_replica_for_error_messages():
    first, second = MagicMock(), MagicMock()
    shard_set = nemo_gym_mod.NemoGymShardSet(handles={"tools": [first, second]})

    assert shard_set.shard_name_of(second) == "tools"
    with pytest.raises(nemo_gym_mod.ShardSetupError, match="does not belong"):
        shard_set.shard_name_of(MagicMock())


def _gym_dataset(*agent_names):
    """A dataset whose rows carry gym env info, JSON-encoded as on disk."""
    return [
        {"extra_env_info": json.dumps({"agent_ref": {"name": name}})}
        for name in agent_names
    ]


def _sharded_set(agent_to_shard):
    return nemo_gym_mod.NemoGymShardSet(
        handles={shard: [MagicMock()] for shard in set(agent_to_shard.values())},
        agent_to_shard=dict(agent_to_shard),
        placement_group=MagicMock(),
    )


def test_an_agent_no_shard_hosts_is_caught_before_the_first_step():
    shard_set = _sharded_set({"alpha": "left"})

    with pytest.raises(nemo_gym_mod.ShardSetupError) as excinfo:
        nemo_gym_mod.validate_dataset_agent_coverage(
            shard_set,
            {"train": _gym_dataset("alpha"), "validation": _gym_dataset("ghost")},
        )

    # Name the split, the unhosted agent, and what is on offer -- the mistake is
    # a typo or a missing shard, and all three are needed to tell which.
    assert "validation" in str(excinfo.value)
    assert "['ghost']" in str(excinfo.value)
    assert "['alpha']" in str(excinfo.value)


def test_a_fully_hosted_dataset_passes():
    shard_set = _sharded_set({"alpha": "left", "beta": "right"})

    nemo_gym_mod.validate_dataset_agent_coverage(
        shard_set,
        {"train": _gym_dataset("alpha", "beta", "alpha"), "validation": None},
    )


def test_an_unsharded_job_is_not_scanned_at_all():
    """One actor hosts everything, so parsing every row would buy nothing."""
    dataset = MagicMock()

    nemo_gym_mod.validate_dataset_agent_coverage(
        nemo_gym_mod.as_nemo_gym_shard_set(MagicMock()), {"train": dataset}
    )

    dataset.__iter__.assert_not_called()


def test_the_wrapped_dataset_is_unwrapped_before_scanning():
    """Rows live on AllTaskProcessedDataset.dataset, not on the wrapper."""
    wrapper = SimpleNamespace(dataset=_gym_dataset("ghost"))

    with pytest.raises(nemo_gym_mod.ShardSetupError, match="ghost"):
        nemo_gym_mod.validate_dataset_agent_coverage(
            _sharded_set({"alpha": "left"}), {"train": wrapper}
        )


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
    assert all(
        invocation.kwargs == {"timeout": None} for invocation in shutdown.call_args_list
    )
    # Releasing a group twice raises; the second shutdown must not try.
    remove.assert_called_once_with(pg)

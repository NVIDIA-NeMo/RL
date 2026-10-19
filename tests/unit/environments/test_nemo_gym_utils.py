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
import os
import sys
import types
import warnings
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest
from omegaconf import DictConfig

from nemo_rl.environments import nemo_gym as nemo_gym_mod
from nemo_rl.environments.nemo_gym import (
    NEMO_GYM_ACTOR_FQN,
    NEMO_GYM_GRACEFUL_SHUTDOWN_TIMEOUT_S,
    _detect_invalid_tool_call_and_malformed_thinking,
    build_nemo_gym_actors,
    build_nemo_gym_config,
    get_nemo_gym_uv_cache_dir,
    get_nemo_gym_venv_dir,
    set_nemo_gym_log_dir,
    setup_nemo_gym_config,
    spinup_nemo_gym_actor,
)


@pytest.mark.parametrize("backend", ["vllm", "dynamo", "megatron", "trtllm"])
def test_setup_nemo_gym_config_enables_http_server(backend: str) -> None:
    generation = {
        "backend": backend,
        "vllm_cfg": {"async_engine": False, "expose_http_server": False},
        "mcore_generation_config": {"expose_http_server": False},
        "trtllm_cfg": {"async_engine": False, "expose_http_server": False},
        "stop_strings": ["stop"],
        "stop_token_ids": [1],
    }
    config = SimpleNamespace(policy={"generation": generation}, data={})

    setup_nemo_gym_config(config, tokenizer=None)

    is_vllm_like = backend in ("vllm", "dynamo")
    assert generation["vllm_cfg"] == {
        "async_engine": is_vllm_like,
        "expose_http_server": is_vllm_like,
    }
    assert generation["mcore_generation_config"] == {
        "expose_http_server": backend == "megatron"
    }
    assert generation["trtllm_cfg"] == {
        "async_engine": backend == "trtllm",
        "expose_http_server": backend == "trtllm",
    }
    assert generation["stop_strings"] is None
    assert generation["stop_token_ids"] is None


def test_setup_nemo_gym_config_rejects_unsupported_backend() -> None:
    config = SimpleNamespace(policy={"generation": {"backend": "unsupported"}}, data={})

    with pytest.raises(ValueError, match="got 'unsupported'"):
        setup_nemo_gym_config(config, tokenizer=None)


@pytest.mark.parametrize("capture_enabled", [False, True])
@pytest.mark.parametrize("tile_cap", [0, 4])
@pytest.mark.parametrize(
    "section,as_list",
    [
        ("default", False),
        ("train", False),
        ("train", True),
        ("validation", False),
        ("validation", True),
    ],
)
def test_setup_nemo_gym_config_rejects_image_tile_cap_before_worker_setup(
    section: str, as_list: bool, tile_cap: int, capture_enabled: bool
) -> None:
    dataset = {"image_max_num_tiles": tile_cap}
    config = SimpleNamespace(
        data={section: [{}, dataset] if as_list else dataset},
        token_capture={"enabled": capture_enabled},
        policy={"generation": {"backend": "vllm", "vllm_cfg": {}}},
    )
    with pytest.raises(ValueError, match="image_max_num_tiles is not supported"):
        setup_nemo_gym_config(config, tokenizer=None)
    assert config.policy["generation"] == {"backend": "vllm", "vllm_cfg": {}}


@pytest.mark.parametrize("tile_option", [{}, {"image_max_num_tiles": None}])
def test_setup_nemo_gym_config_allows_unset_image_tile_cap(tile_option: dict) -> None:
    config = SimpleNamespace(
        data={"default": tile_option, "train": [tile_option], "validation": None},
        policy={"generation": {"backend": "vllm", "vllm_cfg": {}}},
    )
    setup_nemo_gym_config(config, tokenizer=None)
    assert config.policy["generation"]["vllm_cfg"]["expose_http_server"] is True


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
        # A tool-call-only assistant item carries content: None — a structured
        # (executed) call, never a penalty (regression: jobs 6342333/6358268).
        (
            {"content": None, "tool_calls": [{"function": {"name": "bash"}}]},
            False,
            False,
        ),
        (
            {},
            False,
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


def test_set_nemo_gym_log_dir_resolves_the_run_log_dir_on_the_driver(
    tmp_path, monkeypatch
):
    """A relative log dir is made absolute here, not by the Gym actor's cwd."""
    monkeypatch.chdir(tmp_path)
    env_config = {"should_use_nemo_gym": True}

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        set_nemo_gym_log_dir(env_config, os.path.join("logs", "exp_001"))

    assert env_config["nemo_gym"] == {
        "nemo_gym_log_dir": str(tmp_path / "logs" / "exp_001" / "nemo_gym")
    }


def test_set_nemo_gym_log_dir_replaces_a_configured_dir_with_a_warning(tmp_path):
    env_config = {
        "nemo_gym": {"nemo_gym_log_dir": "logs/nemo_gym", "config_paths": ["a.yaml"]}
    }

    with pytest.warns(UserWarning) as record:
        set_nemo_gym_log_dir(env_config, str(tmp_path / "exp_001"))

    assert env_config["nemo_gym"] == {
        "nemo_gym_log_dir": str(tmp_path / "exp_001" / "nemo_gym"),
        "config_paths": ["a.yaml"],
    }
    assert [str(warning.message) for warning in record] == [
        "Config value env.nemo_gym.nemo_gym_log_dir='logs/nemo_gym' is overridden; "
        f"using {str(tmp_path / 'exp_001' / 'nemo_gym')!r} "
        "(NeMo Gym server logs always go under this run's log directory)."
    ]


def test_set_nemo_gym_log_dir_is_silent_when_already_pointing_there(tmp_path):
    gym_log_dir = str(tmp_path / "exp_001" / "nemo_gym")
    env_config = {"nemo_gym": {"nemo_gym_log_dir": gym_log_dir}}

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        set_nemo_gym_log_dir(env_config, str(tmp_path / "exp_001"))

    assert env_config["nemo_gym"] == {"nemo_gym_log_dir": gym_log_dir}


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


def test_build_nemo_gym_config_moves_external_service_readiness_to_actor_field(
    detected_uv_dirs,
):
    """Every entrypoint, SingleController included, builds its actor config here."""
    cfg = build_nemo_gym_config(
        _env_configs(
            external_service_readiness={
                "services": [
                    {
                        "name": "GENRM",
                        "url": "http://10.0.0.1:9213/health",
                        "expected_backends": 2,
                    }
                ],
                "timeout_seconds": 10,
                "poll_interval_seconds": 1,
                "request_timeout_seconds": 2,
            }
        ),
        base_urls=[],
        model_name="test-model",
        enable_router_replay=False,
        use_fastokens=False,
    )

    readiness = cfg["external_service_readiness"]
    assert isinstance(readiness, nemo_gym_mod.ExternalServiceReadinessConfig)
    assert readiness.services[0].expected_backends == 2
    assert "external_service_readiness" not in cfg["initial_global_config_dict"]


def test_build_nemo_gym_config_rejects_invalid_external_service_readiness(
    detected_uv_dirs,
):
    """A malformed gate fails on the driver, before any actor is created."""
    service = {"name": "GENRM", "url": "http://10.0.0.1:9213/health"}
    with pytest.raises(ValueError, match="must be unique"):
        build_nemo_gym_config(
            _env_configs(
                external_service_readiness={
                    "services": [
                        {**service, "expected_backends": 1},
                        {**service, "expected_backends": 2},
                    ],
                    "timeout_seconds": 10,
                    "poll_interval_seconds": 1,
                    "request_timeout_seconds": 2,
                }
            ),
            base_urls=[],
            model_name="test-model",
            enable_router_replay=False,
            use_fastokens=False,
        )


@contextmanager
def _stub_gym_spinup(run_helper):
    """Stand in for the Gym modules and Ray context that _spinup reads."""
    cli = types.ModuleType("nemo_gym.cli")
    cli.GlobalConfigDictParserConfig = MagicMock
    cli.RunHelper = lambda: run_helper
    rollout_collection = types.ModuleType("nemo_gym.rollout_collection")
    rollout_collection.RolloutCollectionHelper = MagicMock
    server_utils = types.ModuleType("nemo_gym.server_utils")
    server_utils.HEAD_SERVER_KEY_NAME = "head_server"
    server_utils.BaseServerConfig = MagicMock
    with (
        patch.dict(
            sys.modules,
            {
                "nemo_gym": types.ModuleType("nemo_gym"),
                "nemo_gym.cli": cli,
                "nemo_gym.rollout_collection": rollout_collection,
                "nemo_gym.server_utils": server_utils,
            },
        ),
        patch.object(nemo_gym_mod, "_get_node_ip_local", return_value="10.0.0.1"),
        patch.object(nemo_gym_mod, "_get_free_port_local", return_value=5000),
        patch.object(nemo_gym_mod, "ray") as mock_ray,
    ):
        mock_ray.get_runtime_context.return_value.gcs_address = "10.0.0.1:6379"
        yield


@pytest.mark.nemo_gym
@pytest.mark.parametrize("forest_training", [False, True])
def test_spinup_passes_parent_resolution_to_gym(forest_training, tmp_path, monkeypatch):
    from nemo_gym.token_id_capture.config import TokenIdCaptureConfig

    monkeypatch.setenv("NEMO_GYM_TOKEN_CAPTURE_CONTROL_TOKEN", "test-control")
    cls = nemo_gym_mod.NemoGym.__ray_metadata__.modified_class
    actor = cls.__new__(cls)
    actor.__init__(
        {
            "model_name": "test-model",
            "base_urls": [],
            "initial_global_config_dict": {},
            "token_capture": {
                "enabled": True,
                "forest_training": forest_training,
                "selection_policy": "returned_history",
                "generation_backend": "vllm",
                "capture_dir": str(tmp_path),
                "control_auth_token": "test-control",
            },
        }
    )
    run_helper = MagicMock()
    run_helper.start.side_effect = RuntimeError("stop before server startup")
    with (
        _stub_gym_spinup(run_helper),
        pytest.raises(RuntimeError, match="stop before server startup"),
    ):
        actor._spinup()
    config = run_helper.start.call_args.kwargs[
        "global_config_dict_parser_config"
    ].initial_global_config_dict
    capture = TokenIdCaptureConfig.model_validate(dict(config)).token_id_capture
    assert capture.parent_resolution == (
        "source_history" if forest_training else "fingerprint"
    )
    assert "framework_owned_context" not in config["token_id_capture"]


def test_spinup_stops_local_gym_servers_when_external_services_time_out():
    """Local Gym servers start first; a missed external deadline must stop them."""
    readiness = nemo_gym_mod.ExternalServiceReadinessConfig(
        services=[
            {
                "name": "GENRM",
                "url": "http://10.0.0.1:9213/health",
                "expected_backends": 2,
            }
        ],
        timeout_seconds=1,
        poll_interval_seconds=1,
        request_timeout_seconds=1,
    )
    cls = nemo_gym_mod.NemoGym.__ray_metadata__.modified_class
    actor = cls.__new__(cls)
    actor.__init__(
        {
            "model_name": "test-model",
            "base_urls": [],
            "initial_global_config_dict": {},
            "external_service_readiness": readiness,
        }
    )
    # One parent mock records the order of RunHelper and readiness calls.
    calls = MagicMock()
    calls.wait.side_effect = TimeoutError("GENRM: request failed")

    with (
        _stub_gym_spinup(calls.run_helper),
        patch.object(nemo_gym_mod, "_wait_for_external_services", calls.wait),
    ):
        with pytest.raises(TimeoutError, match="GENRM"):
            actor._spinup()

    # _spinup stops the servers and drops the handle itself, so the caller's
    # teardown cannot reach the non-idempotent RunHelper.shutdown() again.
    assert actor.rh is None
    actor.shutdown()
    assert [name for name, _, _ in calls.mock_calls] == [
        "run_helper.start",
        "wait",
        "run_helper.shutdown",
    ]
    calls.wait.assert_called_once_with(readiness)


def test_build_nemo_gym_config_moves_port_range_to_actor_fields(detected_uv_dirs):
    cfg = build_nemo_gym_config(
        _env_configs(port_range_low=6000, port_range_high=6999),
        base_urls=[],
        model_name="test-model",
        enable_router_replay=False,
        use_fastokens=False,
    )

    assert (cfg["port_range_low"], cfg["port_range_high"]) == (6000, 6999)
    assert "port_range_low" not in cfg["initial_global_config_dict"]
    assert "port_range_high" not in cfg["initial_global_config_dict"]


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
    token_capture = {"enabled": True, "capture_dir": "/tmp/cap"}

    with (
        patch.object(
            nemo_gym_mod, "make_actor_runtime_env", return_value=runtime_env
        ) as mock_runtime_env,
        patch.object(nemo_gym_mod, "NemoGym") as mock_cls,
        patch.object(nemo_gym_mod, "ray") as mock_ray,
    ):
        mock_cls.options.return_value.remote.return_value = actor
        mock_ray.get_runtime_context.return_value.get_node_id.return_value = "a" * 56
        mock_ray.get.side_effect = [None, [None, {}]]

        result = build_nemo_gym_actors(
            _env_configs(num_gpu_nodes=num_gpu_nodes),
            base_urls=["http://vllm-0"],
            model_name="test-model",
            tokenizer=_TOKENIZER,
            enable_router_replay=False,
            use_fastokens=True,
            token_capture=token_capture,
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
    # The ledger config must ride through to the actor; a refactor of this
    # wrapper once dropped it without any type or test catching it.
    assert cfg["token_capture"] == token_capture

    # Wait for spinup before submitting tokenizer installation and route
    # retrieval together.
    actor._spinup.remote.assert_called_once_with()
    actor.set_tokenizer.remote.assert_called_once_with(_TOKENIZER)
    assert mock_ray.get.call_args_list == [
        call("spinup-ref"),
        call(
            [
                actor.set_tokenizer.remote.return_value,
                actor.list_environment_server_routes.remote.return_value,
            ]
        ),
    ]


@pytest.mark.parametrize("failed_ref", ["spinup-ref", "tokenizer-ref", "routes-ref"])
def test_spinup_nemo_gym_actor_cleans_up_after_startup_failure(
    detected_uv_dirs, failed_ref
):
    actor = MagicMock()
    actor._spinup.remote.return_value = "spinup-ref"
    actor.set_tokenizer.remote.return_value = "tokenizer-ref"
    actor.list_environment_server_routes.remote.return_value = "routes-ref"

    def get_or_fail(ref, **_kwargs):
        if isinstance(ref, list):
            return [get_or_fail(item, **_kwargs) for item in ref]
        if ref == failed_ref:
            raise RuntimeError("startup failed")
        return None

    with (
        patch.object(nemo_gym_mod, "make_actor_runtime_env", return_value={}),
        patch.object(nemo_gym_mod, "NemoGym") as mock_cls,
        patch.object(nemo_gym_mod, "ray") as mock_ray,
        patch.object(nemo_gym_mod, "shutdown_environments") as shutdown,
    ):
        mock_cls.options.return_value.remote.return_value = actor
        mock_ray.get.side_effect = get_or_fail

        with pytest.raises(RuntimeError, match="startup failed"):
            spinup_nemo_gym_actor(
                _env_configs(num_gpu_nodes=0),
                base_urls=["http://vllm-0"],
                model_name="test-model",
                tokenizer=MagicMock(),
                enable_router_replay=False,
                use_fastokens=False,
            )

    shutdown.assert_called_once()
    assert list(shutdown.call_args.args[0].values()) == [actor]
    assert shutdown.call_args.kwargs == {
        "timeout": NEMO_GYM_GRACEFUL_SHUTDOWN_TIMEOUT_S
    }
    mock_ray.kill.assert_called_once_with(actor)


@pytest.mark.parametrize("cleanup_failure", ["shutdown", "kill"])
def test_spinup_nemo_gym_actor_preserves_startup_error_when_cleanup_fails(
    detected_uv_dirs, cleanup_failure
):
    actor = MagicMock()
    actor._spinup.remote.return_value = "spinup-ref"
    startup_error = RuntimeError("startup failed")
    cleanup_error = RuntimeError("cleanup failed")

    def get_or_fail(ref, **_kwargs):
        if ref == "spinup-ref":
            raise startup_error
        return None

    with (
        patch.object(nemo_gym_mod, "make_actor_runtime_env", return_value={}),
        patch.object(nemo_gym_mod, "NemoGym") as mock_cls,
        patch.object(nemo_gym_mod, "ray") as mock_ray,
        patch.object(nemo_gym_mod, "shutdown_environments") as shutdown,
    ):
        mock_cls.options.return_value.remote.return_value = actor
        mock_ray.get.side_effect = get_or_fail
        if cleanup_failure == "shutdown":
            shutdown.side_effect = cleanup_error
        if cleanup_failure == "kill":
            mock_ray.kill.side_effect = cleanup_error

        with pytest.raises(RuntimeError) as exc_info:
            spinup_nemo_gym_actor(
                _env_configs(num_gpu_nodes=0),
                base_urls=["http://vllm-0"],
                model_name="test-model",
                tokenizer=MagicMock(),
                enable_router_replay=False,
                use_fastokens=False,
            )

    assert exc_info.value is startup_error
    shutdown.assert_called_once()
    mock_ray.kill.assert_called_once_with(actor)


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
    module.get_global_config_dict = MagicMock(return_value=resolved)
    with patch.dict(
        sys.modules, {"nemo_gym": package, "nemo_gym.global_config": module}
    ):
        yield module.get_global_config_dict


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
            "math_environment": {
                "environment_servers": {"single_agent_turn": {"entrypoint": "app.py"}}
            },
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
        "math_environment": ["environment_servers"],
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


@pytest.mark.parametrize(
    "routes", [None, {}, {"workplace:train": "workplace_environment"}]
)
def test_list_environment_server_routes_caches_the_resolved_gym_config(routes):
    resolved = DictConfig({"environment_server_routes": routes})
    with _stub_gym_resolved_config(resolved) as get_config:
        actor = _spun_up_actor()
        first_routes = actor.list_environment_server_routes()
        assert first_routes == (routes or {})

        # Discovery callers must not mutate the map used by later rollouts.
        first_routes["extra:train"] = "other_environment"
        assert actor.list_environment_server_routes() == (routes or {})
        get_config.assert_called_once_with()

        actor.shutdown()
        with pytest.raises(RuntimeError, match="call _spinup"):
            actor.list_environment_server_routes()


@pytest.mark.parametrize(
    "routes", [["environment"], {"tasks": ""}, {"": "environment"}, {"tasks": 3}]
)
def test_list_environment_server_routes_rejects_malformed_routes(routes):
    with _stub_gym_resolved_config(DictConfig({"environment_server_routes": routes})):
        with pytest.raises(nemo_gym_mod.ShardConfigError, match="must map non-empty"):
            _spun_up_actor().list_environment_server_routes()


def test_list_environment_server_routes_before_spinup_raises():
    cls = nemo_gym_mod.NemoGym.__ray_metadata__.modified_class
    actor = cls.__new__(cls)
    actor.__init__({})
    with pytest.raises(RuntimeError, match="call _spinup"):
        actor.list_environment_server_routes()


class TestUnresolvedAgentRefsAreDiagnosable:
    """A Gym older than the checkout that prepared the data must say so.

    ``task_source`` routing is new. A current Gym strips ``agent_ref`` from collated rows
    and stamps ``task_source`` instead, then resolves it back inside ``run_examples``. An
    older Gym has no resolver, so the same dataset arrives unroutable -- and the first
    thing that touched it was an unguarded ``row["agent_ref"]``, which surfaced as a bare
    KeyError inside a Ray TaskError inside an ExceptionGroup.
    """

    def test_resolved_rows_pass_through(self):
        rows = [{"agent_ref": {"name": "a"}}, {"agent_ref": {"name": "b"}}]
        nemo_gym_mod._require_resolved_agent_refs(rows)  # must not raise

    def test_a_stale_gym_is_named_along_with_the_remedy(self):
        rows = [
            {"agent_ref": {"name": "a"}},
            {"task_source": "workplace_assistant_simple_agent"},
        ]
        with pytest.raises(RuntimeError) as excinfo:
            nemo_gym_mod._require_resolved_agent_refs(rows)
        message = str(excinfo.value)
        assert "1 of 2" in message
        assert "workplace_assistant_simple_agent" in message
        assert "NRL_FORCE_REBUILD_VENVS" in message

    def test_a_row_with_no_routing_at_all_says_that_instead(self):
        """Different cause, different fix: rebuilding venvs would not help here."""
        with pytest.raises(RuntimeError, match="no task_source either"):
            nemo_gym_mod._require_resolved_agent_refs([{"id": "x"}])

    def test_an_empty_agent_ref_counts_as_unresolved(self):
        """Gym writes {"name": ...}; a bare {} routes nowhere."""
        with pytest.raises(RuntimeError):
            nemo_gym_mod._require_resolved_agent_refs([{"agent_ref": {}}])


class _FakeGymCluster:
    """Stands in for Ray while the factory creates, starts and queries actors.

    Each actor returns tagged sentinels from ``.remote()`` so ``ray.get`` can
    tell spinups from entry queries and fail whichever the test asks it to.
    """

    def __init__(
        self,
        *,
        entries_by_index=None,
        environment_routes_by_index=None,
        spinup_failures=None,
        spinup_timeouts=None,
        wedged_spinups=None,
        tokenizer_timeouts=None,
        pg_ready_error=None,
    ):
        self.entries_by_index = entries_by_index or {}
        self.environment_routes_by_index = environment_routes_by_index or {}
        self.spinup_failures = spinup_failures or {}
        self.spinup_timeouts = set(spinup_timeouts or ())
        self.wedged_spinups = set(wedged_spinups or ())
        self.tokenizer_timeouts = set(tokenizer_timeouts or ())
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
                actor.list_environment_server_routes.remote.return_value = (
                    "environment_routes",
                    index,
                )
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
        if isinstance(reference, list):
            return [self.ray_get(item, timeout=timeout) for item in reference]
        if reference == "pg-ready":
            if self.pg_ready_error is not None:
                raise self.pg_ready_error
            return None
        kind, index = reference
        if kind == "spinup":
            if index in self.wedged_spinups and timeout is not None:
                raise TimeoutError("startup remained wedged")
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
            if index in self.tokenizer_timeouts and timeout is not None:
                raise TimeoutError("startup budget expired")
            return None
        if kind == "environment_routes":
            return self.environment_routes_by_index.get(index, {})
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
        cluster.ray = mock_ray
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


def test_build_nemo_gym_config_keeps_the_policy_api_key_out_of_gyms_config(
    detected_uv_dirs,
):
    """The key is an actor field that Gym's servers receive through their
    environment at spin-up, never a value in the serialized global config."""
    cfg = build_nemo_gym_config(
        _env_configs(),
        base_urls=["http://vllm-0"],
        model_name="test-model",
        enable_router_replay=False,
        use_fastokens=False,
        policy_api_key="per-job-key",
    )

    assert cfg["policy_api_key"] == "per-job-key"
    assert "policy_api_key" not in cfg["initial_global_config_dict"]


@pytest.mark.parametrize(
    "env_configs", [_env_configs, _shard_env_configs], ids=["unsharded", "sharded"]
)
def test_build_nemo_gym_actors_hands_every_actor_the_policy_api_key(
    detected_uv_dirs, env_configs
):
    """Both builders forward the key: a dropped line in either would leave Gym
    sending the placeholder against workers that enforce the real key."""
    cluster = _FakeGymCluster()

    with _patched_cluster(cluster):
        nemo_gym_mod.build_nemo_gym_actors(
            env_configs(),
            base_urls=["http://vllm-0"],
            model_name="test-model",
            tokenizer=_TOKENIZER,
            enable_router_replay=False,
            use_fastokens=False,
            policy_api_key="per-job-key",
        )

    assert cluster.actor_configs
    assert all(cfg["policy_api_key"] == "per-job-key" for cfg in cluster.actor_configs)


@pytest.mark.parametrize("sharded", [False, True])
def test_actor_factory_reads_environment_server_routes_from_resolved_configs(
    detected_uv_dirs, sharded
):
    routes = {"workplace:train": "workplace_environment"}
    cluster = _FakeGymCluster(
        entries_by_index={
            0: {
                "workplace_environment": ["environment_servers"],
                "unused_environment": ["environment_servers"],
            },
            1: {"unused_environment": ["environment_servers"]},
        },
        environment_routes_by_index={0: routes},
    )
    # The route is absent from the input config, as when loaded via config_paths.
    config = _shard_env_configs() if sharded else _env_configs()
    with _patched_cluster(cluster):
        shard_set = nemo_gym_mod.build_nemo_gym_actors(
            config,
            base_urls=["http://vllm-0"],
            model_name="test-model",
            tokenizer=_TOKENIZER,
            enable_router_replay=False,
            use_fastokens=False,
        )
        row = {
            "task_id": {"taskset": "workplace:train", "task_id": "1"},
            "task_input": {},
        }
        nemo_gym_mod.validate_dataset_agent_coverage(
            shard_set, {"train": [{"extra_env_info": json.dumps(row)}]}
        )

    assert shard_set.environment_server_routes == routes
    if sharded:
        assert shard_set.route_to_shard == {"workplace_environment": "judged"}
    route_name = nemo_gym_mod.get_nemo_gym_route_name(
        row, shard_set.environment_server_routes
    )
    assert shard_set.pick_handle(route_name) is cluster.actors[0]


def test_actor_factory_cleans_up_when_environment_server_route_target_is_not_an_environment(
    detected_uv_dirs,
):
    cluster = _FakeGymCluster(
        entries_by_index={0: {"agent": ["responses_api_agents"]}},
        environment_routes_by_index={0: {"workplace:train": "agent"}},
    )
    with _patched_cluster(cluster):
        with pytest.raises(
            nemo_gym_mod.ShardSetupError, match="a running Environment Server"
        ):
            nemo_gym_mod.build_nemo_gym_actors(
                _env_configs(),
                base_urls=["http://vllm-0"],
                model_name="test-model",
                tokenizer=_TOKENIZER,
                enable_router_replay=False,
                use_fastokens=False,
            )

    cluster.ray.kill.assert_called_once_with(cluster.actors[0])


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
    assert shard_set.route_to_shard == {"math_agent": "judged", "bash_agent": "tools"}
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
    assert spinup_zero_calls[1] is not None
    cluster.shutdown_environments.assert_called_once()
    assert cluster.shutdown_environments.call_args.kwargs == {
        "timeout": nemo_gym_mod.NEMO_GYM_GRACEFUL_SHUTDOWN_TIMEOUT_S
    }


def test_wedged_shard_startup_cannot_block_forced_teardown(detected_uv_dirs):
    cluster = _FakeGymCluster(wedged_spinups={0})

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

    cluster.shutdown_environments.assert_called_once()
    assert cluster.shutdown_environments.call_args.kwargs == {
        "timeout": nemo_gym_mod.NEMO_GYM_GRACEFUL_SHUTDOWN_TIMEOUT_S
    }
    assert cluster.ray.kill.call_count == 3


def test_tokenizer_timeout_reports_exhausted_shared_startup_budget(
    detected_uv_dirs, monkeypatch
):
    cluster = _FakeGymCluster(tokenizer_timeouts={0})
    clock = iter([0.0, 0.0, 0.0, 0.0, 1.0])
    monkeypatch.setattr(nemo_gym_mod, "monotonic", lambda: next(clock))

    with _patched_cluster(cluster):
        with pytest.raises(
            nemo_gym_mod.ShardSetupError,
            match="used the whole 1.0s startup budget before the tokenizer",
        ):
            nemo_gym_mod.build_nemo_gym_actors(
                _shard_env_configs(),
                base_urls=["http://vllm-0"],
                model_name="test-model",
                tokenizer=_TOKENIZER,
                enable_router_replay=False,
                use_fastokens=False,
                spinup_timeout=1.0,
            )

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
    assert shard_set.hosted_routes == frozenset()


def test_an_existing_shard_set_is_passed_through_untouched():
    shard_set = nemo_gym_mod.NemoGymShardSet(handles={"a": [MagicMock()]})

    assert nemo_gym_mod.as_nemo_gym_shard_set(shard_set) is shard_set


def test_a_replicated_shard_labels_each_instance_apart():
    first, second = MagicMock(), MagicMock()
    shard_set = nemo_gym_mod.NemoGymShardSet(handles={"tools": [first, second]})

    assert shard_set.instance_label(first) == "tools/0"
    assert shard_set.instance_label(second) == "tools/1"
    with pytest.raises(nemo_gym_mod.ShardSetupError, match="does not belong"):
        shard_set.instance_label(MagicMock())


def test_an_unreplicated_shard_is_labelled_by_its_name_alone():
    """Names a metric and its log directory the same way."""
    handle = MagicMock()
    shard_set = nemo_gym_mod.NemoGymShardSet(handles={"tools": [handle]})

    assert shard_set.instance_label(handle) == "tools"


def _gym_dataset(*agent_names):
    """A dataset whose rows carry gym env info, JSON-encoded as on disk."""
    return [
        {"extra_env_info": json.dumps({"agent_ref": {"name": name}})}
        for name in agent_names
    ]


def _sharded_set(route_to_shard):
    return nemo_gym_mod.NemoGymShardSet(
        handles={shard: [MagicMock()] for shard in set(route_to_shard.values())},
        route_to_shard=dict(route_to_shard),
        placement_group=MagicMock(),
    )


def test_materialized_taskset_routing_takes_precedence_over_agent_ref():
    row = {
        "task_id": {"taskset": "workplace:train", "task_id": "1"},
        "task_input": {},
        "agent_ref": {"name": "workplace_agent"},
        "task_source": "workplace_assistant",
    }
    assert (
        nemo_gym_mod.get_nemo_gym_route_name(
            row, {"workplace:train": "workplace_environment"}
        )
        == "workplace_environment"
    )


def test_materialized_taskset_without_route_fails_even_if_agent_ref_is_present():
    with pytest.raises(
        ValueError, match="No environment server route.*workplace:train"
    ):
        nemo_gym_mod.get_nemo_gym_route_name(
            {
                "task_id": {"taskset": "workplace:train", "task_id": "1"},
                "task_input": {},
                "agent_ref": {"name": "workplace_agent"},
            }
        )


@pytest.mark.parametrize(
    "task_id", [{}, {"taskset": ""}, {"taskset": ["workplace:train"]}]
)
def test_malformed_taskset_does_not_fall_back_to_agent_ref(task_id):
    with pytest.raises(ValueError, match="non-empty task_id.taskset"):
        nemo_gym_mod.get_nemo_gym_route_name(
            {
                "task_id": task_id,
                "task_input": {},
                "agent_ref": {"name": "fallback_agent"},
            },
            {"workplace:train": "workplace_environment"},
        )


@pytest.mark.parametrize("cached_source", [False, True])
def test_materialized_dataset_coverage_resolves_tasksets_before_matching_shards(
    tmp_path, cached_source
):
    raw_row = {
        "task_id": {"taskset": "workplace:train", "task_id": "1"},
        "task_input": {},
    }
    rows = [{"extra_env_info": json.dumps(raw_row)}]
    if cached_source:
        source_path = tmp_path / "materialized_tasks.jsonl"
        source_path.write_text(json.dumps(raw_row) + "\n")
        source = nemo_gym_mod.NemoGymSourceIdentity.from_stat(
            str(source_path), source_path.stat()
        )
        dataset = SimpleNamespace(dataset=rows, agent_name_sources=frozenset({source}))
    else:
        dataset = rows
    shard_set = _sharded_set({"workplace_environment": "tools"})
    shard_set.environment_server_routes = {"workplace:train": "workplace_environment"}

    nemo_gym_mod.validate_dataset_agent_coverage(shard_set, {"train": dataset})

    # Reusing the source cache under another route map must revalidate the target.
    shard_set.environment_server_routes = {"workplace:train": "missing_environment"}
    with pytest.raises(nemo_gym_mod.ShardSetupError, match="missing_environment"):
        nemo_gym_mod.validate_dataset_agent_coverage(shard_set, {"train": dataset})

    shard_set.environment_server_routes = {}
    with pytest.raises(
        nemo_gym_mod.ShardSetupError, match="train.*No environment server route"
    ):
        nemo_gym_mod.validate_dataset_agent_coverage(shard_set, {"train": dataset})


def test_materialized_rows_reject_recovery_overrides_in_nested_mixed_dataset() -> None:
    shard_set = _sharded_set({"alpha": "left", "workplace_environment": "tools"})
    shard_set.environment_server_routes = {"workplace:train": "workplace_environment"}
    rows = [
        *_gym_dataset("alpha"),
        {
            "extra_env_info": json.dumps(
                {
                    "task_id": {"taskset": "workplace:train", "task_id": "1"},
                    "task_input": {},
                }
            )
        },
    ]
    datasets = {
        "train": _gym_dataset("alpha"),
        "validation": {"mixed": SimpleNamespace(dataset=rows)},
    }

    with pytest.raises(nemo_gym_mod.ShardSetupError) as excinfo:
        nemo_gym_mod.validate_dataset_agent_coverage(
            shard_set, datasets, reject_materialized_tasks=True
        )

    message = str(excinfo.value)
    assert "validation" in message
    assert "agent_granularity_overrides" in message
    assert "task_source_granularity_overrides" in message
    assert "default_granularity" in message


@pytest.mark.parametrize("changed_source", [False, True])
def test_materialized_recovery_guard_checks_cached_sources_and_loaded_fallback(
    tmp_path: Path, changed_source: bool
) -> None:
    raw_row = {
        "task_id": {"taskset": "workplace:train", "task_id": "1"},
        "task_input": {},
    }
    source_path = tmp_path / "materialized_tasks.jsonl"
    source_path.write_text(json.dumps(raw_row) + "\n")
    source = nemo_gym_mod.NemoGymSourceIdentity.from_stat(
        str(source_path), source_path.stat()
    )
    # A stable source must be checked even if the loaded rows aren't traversed.
    loaded_rows = (
        [{"extra_env_info": json.dumps(raw_row)}] if changed_source else MagicMock()
    )
    dataset = SimpleNamespace(
        dataset=loaded_rows, agent_name_sources=frozenset({source})
    )
    shard_set = _sharded_set({"workplace_environment": "tools"})
    shard_set.environment_server_routes = {"workplace:train": "workplace_environment"}
    nemo_gym_mod.validate_dataset_agent_coverage(shard_set, {"train": dataset})

    if changed_source:
        source_path.write_text(
            json.dumps({"agent_ref": {"name": "workplace_environment"}}) + "\n"
        )

    with pytest.raises(
        nemo_gym_mod.ShardSetupError, match="validation.*agent_granularity_overrides"
    ):
        nemo_gym_mod.validate_dataset_agent_coverage(
            shard_set, {"validation": dataset}, reject_materialized_tasks=True
        )
    if not changed_source:
        loaded_rows.__iter__.assert_not_called()


def test_flat_rows_still_allow_recovery_overrides() -> None:
    rows = [
        *_gym_dataset("alpha"),
        {"extra_env_info": json.dumps({"task_source": "math_tools"})},
    ]

    nemo_gym_mod.validate_dataset_agent_coverage(
        _sharded_set({"alpha": "left", "math_tools": "right"}),
        {"train": rows},
        reject_materialized_tasks=True,
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


def test_task_source_is_validated_before_gym_resolves_agent_ref():
    shard_set = _sharded_set({"workplace_assistant": "tools"})
    dataset = [
        {
            "extra_env_info": json.dumps(
                {
                    "task_source": "workplace_assistant",
                    "responses_create_params": {"input": []},
                }
            )
        }
    ]

    nemo_gym_mod.validate_dataset_agent_coverage(shard_set, {"train": dataset})


def test_an_unsharded_job_is_checked_against_the_actors_entries():
    """The lone actor is asked which entries it spawned, and rows must name one.

    A stale name, such as a renamed agent or a typo, would otherwise fail only
    at its first dispatched batch, after every server has started.
    """
    handle = MagicMock()
    shard_set = nemo_gym_mod.as_nemo_gym_shard_set(handle)

    with patch.object(nemo_gym_mod.ray, "get") as ray_get:
        ray_get.return_value = {
            "math_agent": ["responses_api_agents"],
            "math_tools": ["resources_servers"],
        }
        nemo_gym_mod.validate_dataset_agent_coverage(
            shard_set, {"train": _gym_dataset("math_agent"), "validation": None}
        )
        with pytest.raises(nemo_gym_mod.ShardSetupError) as excinfo:
            nemo_gym_mod.validate_dataset_agent_coverage(
                shard_set, {"train": _gym_dataset("search_agent")}
            )

    ray_get.assert_called_with(handle.list_entries.remote.return_value)
    assert "['search_agent']" in str(excinfo.value)
    assert "math_agent" in str(excinfo.value)


def test_an_unsharded_job_counts_only_routable_entries_as_hosted():
    """A model-only entry is not a route on one actor any more than on many.

    ``build_route_shard_map`` maps agents and resources servers only, so on a
    sharded job a row naming a ``responses_api_models`` entry is rejected at
    setup. The unsharded check must agree, and must not list the model among
    the hosted routes in its error message.
    """
    handle = MagicMock()
    shard_set = nemo_gym_mod.as_nemo_gym_shard_set(handle)
    tool_row = {"extra_env_info": json.dumps({"task_source": "math_tools"})}

    with patch.object(nemo_gym_mod.ray, "get") as ray_get:
        ray_get.return_value = {
            "math_agent": ["responses_api_agents"],
            "math_tools": ["resources_servers"],
            "policy_model": ["responses_api_models"],
        }
        nemo_gym_mod.validate_dataset_agent_coverage(
            shard_set, {"train": [*_gym_dataset("math_agent"), tool_row]}
        )
        with pytest.raises(nemo_gym_mod.ShardSetupError) as excinfo:
            nemo_gym_mod.validate_dataset_agent_coverage(
                shard_set, {"train": _gym_dataset("policy_model")}
            )

    assert "['policy_model']" in str(excinfo.value)
    assert "Hosted routes: ['math_agent', 'math_tools']." in str(excinfo.value)


def test_the_wrapped_dataset_is_unwrapped_before_scanning():
    """Rows live on AllTaskProcessedDataset.dataset, not on the wrapper."""
    wrapper = SimpleNamespace(dataset=_gym_dataset("ghost"))

    with pytest.raises(nemo_gym_mod.ShardSetupError, match="ghost"):
        nemo_gym_mod.validate_dataset_agent_coverage(
            _sharded_set({"alpha": "left"}), {"train": wrapper}
        )


def test_source_agent_names_are_cached_across_separate_datasets(tmp_path):
    data_path = tmp_path / "gym.jsonl"
    data_path.write_text(
        "\n".join(
            [
                json.dumps({"agent_ref": {"name": "alpha"}}),
                json.dumps({"agent_ref": {"name": "beta"}}),
            ]
        )
        + "\n"
    )
    source_stat = data_path.stat()
    source = nemo_gym_mod.NemoGymSourceIdentity.from_stat(
        str(data_path.resolve()), source_stat
    )
    alpha_rows = MagicMock()
    beta_rows = MagicMock()
    datasets = {
        "alpha": SimpleNamespace(
            dataset=alpha_rows,
            agent_name_sources=frozenset({source}),
        ),
        "beta": SimpleNamespace(
            dataset=beta_rows,
            agent_name_sources=frozenset({source}),
        ),
    }

    with patch.object(
        nemo_gym_mod,
        "_get_agent_name",
        wraps=nemo_gym_mod._get_agent_name,
    ) as get_agent_name:
        nemo_gym_mod.validate_dataset_agent_coverage(
            _sharded_set({"alpha": "left", "beta": "right"}), {"train": datasets}
        )

    assert get_agent_name.call_count == 2
    alpha_rows.__iter__.assert_not_called()
    beta_rows.__iter__.assert_not_called()


def test_changed_source_falls_back_to_loaded_rows(tmp_path):
    data_path = tmp_path / "gym.jsonl"
    data_path.write_text(json.dumps({"agent_ref": {"name": "alpha"}}) + "\n")
    source_stat = data_path.stat()
    source = nemo_gym_mod.NemoGymSourceIdentity.from_stat(
        str(data_path.resolve()), source_stat
    )
    wrapper = SimpleNamespace(
        dataset=_gym_dataset("ghost"),
        agent_name_sources=frozenset({source}),
    )
    data_path.write_text(
        json.dumps({"agent_ref": {"name": "changed-and-longer"}}) + "\n"
    )

    with pytest.raises(nemo_gym_mod.ShardSetupError, match="ghost"):
        nemo_gym_mod.validate_dataset_agent_coverage(
            _sharded_set({"alpha": "left"}), {"train": wrapper}
        )


def test_same_size_source_replacement_falls_back_to_loaded_rows(tmp_path):
    data_path = tmp_path / "gym.jsonl"
    data_path.write_text(json.dumps({"agent_ref": {"name": "alpha"}}) + "\n")
    source_stat = data_path.stat()
    source = nemo_gym_mod.NemoGymSourceIdentity.from_stat(
        str(data_path.resolve()), source_stat
    )
    wrapper = SimpleNamespace(
        dataset=_gym_dataset("ghost"),
        agent_name_sources=frozenset({source}),
    )

    replacement = tmp_path / "replacement.jsonl"
    replacement.write_text(json.dumps({"agent_ref": {"name": "bravo"}}) + "\n")
    os.utime(
        replacement,
        ns=(replacement.stat().st_atime_ns, source_stat.st_mtime_ns),
    )
    os.replace(replacement, data_path)
    replaced_stat = data_path.stat()
    assert replaced_stat.st_size == source_stat.st_size
    assert replaced_stat.st_mtime_ns == source_stat.st_mtime_ns

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
        patch.object(nemo_gym_mod.ray, "kill") as kill,
    ):
        shard_set.shutdown()
        shard_set.shutdown()

    assert shutdown.call_count == 2
    assert all(
        invocation.kwargs
        == {"timeout": nemo_gym_mod.NEMO_GYM_GRACEFUL_SHUTDOWN_TIMEOUT_S}
        for invocation in shutdown.call_args_list
    )
    assert kill.call_count == 4
    # Releasing a group twice raises; the second shutdown must not try.
    remove.assert_called_once_with(pg)

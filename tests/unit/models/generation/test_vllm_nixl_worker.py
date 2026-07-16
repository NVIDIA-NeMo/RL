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

"""Tests for NIXL preinitialization through vLLM's worker class hook."""

from types import SimpleNamespace

import pytest

from nemo_rl.models.generation.vllm.checkpoint_engine import (
    NEMO_RL_VLLM_WORKER,
    configure_nixl_worker,
    preinit_nixl_from_vllm_config,
)


def _nixl_config() -> dict:
    return {
        "enabled": True,
        "backend": "nixl",
        "engine_kwargs": {
            "nixl": {
                "backend_name": "UCX",
                "backend_init_params": {"foo": "bar"},
            }
        },
    }


@pytest.mark.parametrize(
    "checkpoint_config",
    [None, {"enabled": False}, {"enabled": True, "backend": "custom"}],
)
def test_configure_nixl_worker_ignores_other_configs(checkpoint_config):
    vllm_kwargs = {"additional_config": {"existing": True}}

    configure_nixl_worker({"checkpoint_engine": checkpoint_config}, vllm_kwargs)

    assert vllm_kwargs == {"additional_config": {"existing": True}}


def test_configure_nixl_worker_uses_vllm_extension_points():
    checkpoint_config = _nixl_config()
    vllm_kwargs = {"additional_config": {"existing": True}}

    configure_nixl_worker({"checkpoint_engine": checkpoint_config}, vllm_kwargs)

    assert vllm_kwargs["worker_cls"] == NEMO_RL_VLLM_WORKER
    assert vllm_kwargs["additional_config"] == {
        "existing": True,
        "nemo_rl_checkpoint_engine": checkpoint_config,
    }


def test_configure_nixl_worker_rejects_incompatible_worker_class():
    with pytest.raises(ValueError, match="worker_cls to be unset"):
        configure_nixl_worker(
            {"checkpoint_engine": _nixl_config()},
            {"worker_cls": "custom.Worker"},
        )


def test_preinit_nixl_from_vllm_config_uses_configured_backend(monkeypatch):
    from nemo_rl.utils.checkpoint_engines import nixl

    calls = []
    agent = object()
    monkeypatch.setattr(
        nixl,
        "preinit_nixl_agent",
        lambda **kwargs: calls.append(kwargs) or agent,
    )
    config = SimpleNamespace(
        additional_config={"nemo_rl_checkpoint_engine": _nixl_config()}
    )

    assert preinit_nixl_from_vllm_config(config) is agent
    assert calls == [
        {
            "backend_name": "UCX",
            "backend_init_params": {"foo": "bar"},
        }
    ]


def test_preinit_nixl_from_vllm_config_is_disabled_without_nixl_config():
    config = SimpleNamespace(additional_config={})

    assert preinit_nixl_from_vllm_config(config) is None


@pytest.mark.vllm
def test_nemo_rl_worker_preinitializes_before_vllm_worker(monkeypatch):
    from nemo_rl.models.generation.vllm import vllm_backend

    calls = []
    agent = object()
    monkeypatch.setattr(
        vllm_backend,
        "preinit_nixl_from_vllm_config",
        lambda config: calls.append(("preinit", config)) or agent,
    )
    base_worker = vllm_backend.NemoRLVllmWorker.__bases__[0]
    monkeypatch.setattr(
        base_worker,
        "__init__",
        lambda self, config, *args, **kwargs: calls.append(("vllm", config)),
    )
    config = object()
    worker = vllm_backend.NemoRLVllmWorker(config)

    assert calls == [("preinit", config), ("vllm", config)]
    assert worker._nrl_nixl_preinit_agent is agent

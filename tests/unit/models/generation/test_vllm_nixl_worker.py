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
    NIXL_VLLM_WORKER,
    configure_nixl_worker,
    preinit_nixl_from_vllm_config,
)


def _nixl_config() -> dict:
    return {
        "backend": "nixl",
        "update_weights_bucket_memory_ratio": 0.05,
        "engine_kwargs": {
            "nixl": {
                "device": "cuda",
                "backend_name": "UCX",
                "backend_init_params": {"foo": "bar"},
                "release_after_refit": False,
                "shard_expert_weights": False,
            }
        },
    }


@pytest.mark.parametrize(
    "generation_config",
    [
        {},
        {"refit_transport": "vllm_zmq_sparse"},
        {
            "refit_transport": "custom.module:Engine",
            "refit_cfg": {"custom.module:Engine": {}},
        },
    ],
)
def test_configure_nixl_worker_ignores_other_configs(generation_config):
    vllm_kwargs = {"additional_config": {"existing": True}}

    configure_nixl_worker(generation_config, vllm_kwargs)

    assert vllm_kwargs == {"additional_config": {"existing": True}}


def test_configure_nixl_worker_uses_vllm_extension_points():
    checkpoint_config = _nixl_config()
    vllm_kwargs = {"additional_config": {"existing": True}}

    configure_nixl_worker(
        {
            "refit_transport": "nixl",
            "refit_cfg": {
                "nixl": {
                    "backend_name": "UCX",
                    "backend_init_params": {"foo": "bar"},
                }
            },
        },
        vllm_kwargs,
    )

    assert vllm_kwargs["worker_cls"] == NIXL_VLLM_WORKER
    assert vllm_kwargs["additional_config"] == {
        "existing": True,
        "nemo_rl_checkpoint_engine": checkpoint_config,
    }


def test_configure_nixl_worker_rejects_incompatible_worker_class():
    with pytest.raises(ValueError, match="worker_cls to be unset"):
        configure_nixl_worker(
            {"refit_transport": "nixl"},
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
def test_nixl_worker_preinitializes_before_vllm_worker(monkeypatch):
    from nemo_rl.models.generation.vllm import vllm_backend

    calls = []
    agent = object()
    monkeypatch.setattr(
        vllm_backend,
        "preinit_nixl_from_vllm_config",
        lambda config: calls.append(("preinit", config)) or agent,
    )
    base_worker = vllm_backend.NixlVllmWorker.__bases__[0]
    monkeypatch.setattr(
        base_worker,
        "__init__",
        lambda self, config, *args, **kwargs: calls.append(("vllm", config)),
    )
    config = object()
    worker = vllm_backend.NixlVllmWorker(config)

    assert calls == [("preinit", config), ("vllm", config)]
    assert worker._nrl_nixl_preinit_agent is agent

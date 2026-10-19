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

"""DynamoGeneration with Megatron prefill/decode engines behind the frontend."""

import pickle
from types import SimpleNamespace
from typing import Any

import pytest

from nemo_rl.models.generation.dynamo import DynamoGeneration
from nemo_rl.models.generation.dynamo import dynamo_generation as generation_module
from nemo_rl.models.generation.dynamo.megatron_refit_channel import (
    MegatronRefitChannel,
)
from nemo_rl.models.generation.dynamo.role_plan import EngineRecord
from nemo_rl.models.generation.interfaces import CollectiveSenderSpec

LOG: list[str] = []


class FakeEngine:
    """Stands in for one MegatronGeneration; records calls in a module log."""

    def __init__(self, name: str, pinned: int = 0) -> None:
        self.name = name
        self.pinned = pinned

    def prepare_refit_info(self, state_dict_info):
        LOG.append(f"{self.name}.prepare")

    def init_collective(self, ip, port, world_size, *, train_world_size):
        LOG.append(f"{self.name}.init(off={train_world_size})")
        return []

    def update_weights_from_collective(self, refit_timeout_s=None):
        LOG.append(f"{self.name}.update")
        return []

    def get_collective_sender_spec(self):
        return CollectiveSenderSpec()

    def suspend_for_refit(self):
        LOG.append(f"{self.name}.suspend")

    def resume_after_refit(self):
        LOG.append(f"{self.name}.resume")

    def prepare_for_generation(self, *args, **kwargs):
        LOG.append(f"{self.name}.prepare_for_generation({kwargs.get('tags')})")
        return True

    def pinned_handoff_block_count(self):
        LOG.append(f"{self.name}.pins")
        pinned, self.pinned = self.pinned, 0
        return pinned


def _generation_config() -> dict[str, Any]:
    return {
        "backend": "dynamo",
        "model_name": "Qwen/Qwen3-0.6B",
        "max_new_tokens": 16,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": None,
        "stop_token_ids": None,
        "stop_strings": None,
        "_pad_token_id": 0,
        "colocated": {"enabled": False, "resources": {}},
        "refit_transport": None,
        "mcore_generation_config": {
            "enable_prefix_caching": True,
            "kv_cache_management_mode": "persist",
            "expose_http_server": False,
            "max_model_len": 5,
        },
        "dynamo_cfg": {
            "engine": "megatron",
            "startup_timeout_s": 7,
            "request_timeout_s": 30,
            "control_timeout_s": 42,
            "metrics_include_prefixes": None,
            "metrics_exclude_prefixes": None,
            "worker_args": {
                "tool_call_parser": None,
                "reasoning_parser": None,
                "exclude_tools_when_tool_choice_none": True,
                "enable_structural_tag": False,
                "structural_tag_scope": "auto",
                "structural_tag_schema": "auto",
                "custom_jinja_template": None,
                "endpoint_types": ["completions"],
                "extra_cli_args": [],
            },
            "frontend_args": {
                "tokenizer": "default",
                "tokenizer_cache": False,
                "tokenizer_cache_bytes": 1,
                "router_mode": "kv",
                "extra_cli_args": [],
            },
            "disaggregation": {
                "decode": {
                    "workers": 1,
                    "megatron_cfg": {"tensor_model_parallel_size": 2},
                },
                "prefill": {
                    "workers": 1,
                    "megatron_cfg": {"tensor_model_parallel_size": 2},
                },
            },
        },
    }


def _policy_config() -> dict[str, Any]:
    return {
        "model_name": "Qwen/Qwen3-0.6B",
        "megatron_cfg": {
            "enabled": True,
            "tensor_model_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
            "context_parallel_size": 1,
        },
        "generation": _generation_config(),
    }


def _patch_runtime(monkeypatch, events: list[str], engines: dict[str, FakeEngine]):
    class FakeRuntime:
        def __init__(self, *, cluster, config, engine_pool_factory, plan):
            events.append("init")
            self.plan = plan
            self.factory = engine_pool_factory

        def start(self):
            events.append("start")

        @property
        def frontend_url(self):
            return "http://10.0.0.1:3000/v1"

        @property
        def namespace(self):
            return "ns"

        def expected_registrations(self):
            return {"backend": ({"generate"}, 1), "prefill": ({"generate"}, 1)}

        def engine_records(self):
            return [
                EngineRecord(
                    spec.role,
                    spec.world_size,
                    spec.name,
                    "10.0.0.2",
                    engines[spec.name],
                )
                for spec in self.plan.engines
            ]

        def shutdown(self):
            events.append("shutdown")

    monkeypatch.setattr(generation_module, "ManagedDynamoRuntime", FakeRuntime)


def _build(monkeypatch, *, prefill_pinned: int = 0):
    LOG.clear()
    events: list[str] = []
    engines = {
        "decode-0": FakeEngine("d0"),
        "prefill-0": FakeEngine("p0", pinned=prefill_pinned),
    }
    _patch_runtime(monkeypatch, events, engines)
    waits: list[dict[str, Any]] = []
    monkeypatch.setattr(
        generation_module,
        "wait_for_frontend_registrations",
        lambda **kwargs: waits.append(kwargs),
    )
    cluster = SimpleNamespace(world_size=lambda: 4, num_gpus_per_node=8)
    generation = DynamoGeneration(
        cluster=cluster,
        config=_generation_config(),
        tokenizer=object(),
        policy_config=_policy_config(),
    )
    return generation, events, waits


def test_megatron_engines_build_the_channel_and_require_quiesce(monkeypatch) -> None:
    generation, events, _ = _build(monkeypatch)
    assert events == ["init", "start"]
    assert isinstance(generation._refit_channel, MegatronRefitChannel)
    assert generation.requires_quiesce_before_weight_update is True
    assert generation.quiesce_timeout_s == 42
    assert generation.get_inference_world_size() == 4
    assert generation.get_collective_sender_spec().nccl_peer == "nemo"
    assert generation.dp_openai_server_base_urls == [None]
    assert generation.frontend_url == "http://10.0.0.1:3000/v1"
    generation.init_collective("ip", 1, 12, train_world_size=8)
    assert LOG == ["d0.init(off=8)", "p0.init(off=10)"]
    assert generation.shutdown()
    assert events.count("shutdown") == 1


def test_megatron_engines_need_tokenizer_and_policy_config(monkeypatch) -> None:
    _patch_runtime(monkeypatch, [], {})
    with pytest.raises(RuntimeError, match="tokenizer and the full policy config"):
        DynamoGeneration(cluster=SimpleNamespace(), config=_generation_config())


def test_plan_must_fit_the_inference_cluster(monkeypatch) -> None:
    _patch_runtime(monkeypatch, [], {})
    cluster = SimpleNamespace(world_size=lambda: 3, num_gpus_per_node=8)
    with pytest.raises(ValueError, match="allocated=3, expected=4"):
        DynamoGeneration(
            cluster=cluster,
            config=_generation_config(),
            tokenizer=object(),
            policy_config=_policy_config(),
        )


def test_drain_waits_for_pins_then_pauses_and_resume_waits_for_frontend_once(
    monkeypatch,
) -> None:
    generation, _, waits = _build(monkeypatch, prefill_pinned=2)
    generation.drain_for_weight_update()
    assert LOG == [
        "p0.pins",  # 2 pinned: poll again
        "p0.pins",  # 0: idle
        "d0.suspend",
        "p0.suspend",
        "d0.prepare_for_generation(['weights'])",
        "p0.prepare_for_generation(['weights'])",
    ]
    LOG.clear()
    generation.resume_after_weight_update()
    assert LOG == [
        "p0.prepare_for_generation(['kv_cache'])",
        "p0.resume",
        "d0.prepare_for_generation(['kv_cache'])",
        "d0.resume",
    ]
    assert len(waits) == 1
    wait = waits[0]
    assert wait["base_url"] == "http://10.0.0.1:3000"
    assert wait["namespace"] == "ns"
    assert wait["expected"] == {
        "backend": ({"generate"}, 1),
        "prefill": ({"generate"}, 1),
    }
    assert wait["expected_model"] == "Qwen/Qwen3-0.6B"
    assert wait["timeout_s"] == 7
    generation.resume_after_weight_update()
    assert len(waits) == 1, "the frontend is awaited only at the first resume"


def test_client_copy_keeps_the_channel_and_the_pending_readiness_wait(
    monkeypatch,
) -> None:
    generation, _, waits = _build(monkeypatch)
    client = pickle.loads(pickle.dumps(generation))
    assert client._managed_runtime is None
    assert client.requires_quiesce_before_weight_update is True
    assert isinstance(client._refit_channel, MegatronRefitChannel)
    assert client.get_inference_world_size() == 4
    # The controller's copy performs the first-refit readiness wait.
    client.resume_after_weight_update()
    assert len(waits) == 1
    assert client._allowed_new_tokens(2) == 3
    with pytest.raises(ValueError, match="mcore_generation_config.max_model_len=5"):
        client._allowed_new_tokens(5)


def test_megatron_engine_validate_settings_requires_megatron_trainer() -> None:
    master_config = SimpleNamespace(
        async_rl=SimpleNamespace(
            generation_fleet_health=SimpleNamespace(
                enabled=False, refit_timeout_s=None
            ),
            recompute_kv_cache_after_weight_updates=False,
        ),
        policy={
            "megatron_cfg": {"enabled": False},
            "generation": _generation_config(),
        },
    )
    with pytest.raises(ValueError, match="policy.megatron_cfg.enabled=true"):
        DynamoGeneration.validate_settings(master_config)
    master_config.policy["megatron_cfg"]["enabled"] = True
    DynamoGeneration.validate_settings(master_config)

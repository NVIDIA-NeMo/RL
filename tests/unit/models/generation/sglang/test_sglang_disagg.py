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
"""Prefill/decode disaggregation on the SGLang backend, without GPUs or Ray.

Covers the config schema and its validation, the engine layout, the port
allocator, the router dispatch, the ordered pause, the per-role server args
and the router registration body. The GPU smoke tests exercise the same
pieces against real engines.
"""

import asyncio
import sys
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from nemo_rl.models.generation.interfaces import DisaggConfig
from nemo_rl.models.generation.sglang import sglang_generation, sglang_router
from nemo_rl.models.generation.sglang.config import (
    SGLANG_ROLE_ENGINE_ARG_KEYS,
    SGLangDisaggConfig,
    SGLangRoleEngineArgs,
    resolve_sglang_disagg_config,
    resolve_sglang_role_overrides,
    sglang_disagg_server_kwargs,
    sglang_replica_width,
    sglang_role_engine_config,
    validate_sglang_disaggregation,
)
from nemo_rl.models.generation.sglang.sglang_generation import (
    SGLangGeneration,
    disagg_transport_env,
)
from nemo_rl.models.generation.sglang.sglang_router import RouterActor, _start_router
from nemo_rl.models.generation.sglang.sglang_worker import (
    SGLangGenerationWorker,
    router_registration_payload,
)
from nemo_rl.models.generation.sglang.utils import ip_port_utils
from nemo_rl.models.generation.sglang.utils.disagg_dispatch import ReplicaDispatcher
from nemo_rl.models.generation.sglang.utils.layout import (
    SGLangEngineSpec,
    plan_sglang_engines,
)

pytestmark = pytest.mark.sglang


def make_config(*, disagg=True, tp_size=1, num_gpus=8, **overrides):
    """A role-unresolved SGLang generation config, disaggregated by default."""
    config = {
        "backend": "sglang",
        "colocated": {"enabled": False},
        "disaggregation": {
            "enabled": disagg,
            "num_prefill_engines": 1,
            "num_decode_engines": 1,
        },
        "sglang_kwargs": {},
        "sglang_cfg": {
            "model_path": "Qwen/Qwen3-0.6B",
            "dtype": "bfloat16",
            "tp_size": tp_size,
            "dp_size": 1,
            "pp_size": 1,
            "ep_size": 1,
            "mem_fraction_static": 0.6,
            "sglang_server_config": {
                "needs_offload": False,
                "cpu_weight_backup": False,
                "sglang_server_concurrency": 64,
                "pause_generation_mode": "abort",
                "num_gpus": num_gpus,
                "num_gpus_per_engine": tp_size,
            },
            "sglang_router_config": {"use_external_router": False},
            "sglang_fault_tolerance_config": {"use_fault_tolerance": False},
        },
    }
    for dotted, value in overrides.items():
        node = config
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node.setdefault(key, {})
        node[leaf] = value
    return config


# ---------------------------------------------------------------------------
# Config: role overrides, replica width, the disagg block, server kwargs
# ---------------------------------------------------------------------------
def test_the_role_override_whitelist_is_exactly_the_schema():
    assert SGLANG_ROLE_ENGINE_ARG_KEYS == set(SGLangRoleEngineArgs.__annotations__)
    # The shared helper's kwargs channel is not a role knob on SGLang.
    assert "sglang_kwargs" not in SGLANG_ROLE_ENGINE_ARG_KEYS


def test_role_overrides_resolve_per_role_and_keep_the_shared_blocks():
    config = make_config(
        **{
            "sglang_cfg.prefill_engine": {"tp_size": 2, "mem_fraction_static": 0.8},
            "sglang_cfg.disagg": {"transfer_backend": "nixl"},
        }
    )
    resolved = resolve_sglang_role_overrides(config)
    prefill = sglang_role_engine_config(resolved, "prefill")
    decode = sglang_role_engine_config(resolved, "decode")

    assert (prefill["tp_size"], prefill["mem_fraction_static"]) == (2, 0.8)
    assert (decode["tp_size"], decode["mem_fraction_static"]) == (1, 0.6)
    # Shared blocks ride along into each role; the layout blocks do not.
    for role_cfg in (prefill, decode):
        assert role_cfg["sglang_server_config"]["pause_generation_mode"] == "abort"
        assert role_cfg["model_path"] == "Qwen/Qwen3-0.6B"
        assert not {"prefill_engine", "decode_engine", "disagg"} & set(role_cfg)
    # The raw block is untouched for a reader that still wants it.
    assert resolved["sglang_cfg"]["disagg"] == {"transfer_backend": "nixl"}


def test_role_resolution_is_a_no_op_without_disaggregation():
    config = make_config(disagg=False)
    assert resolve_sglang_role_overrides(config) is config


def test_a_shared_key_in_a_role_block_is_rejected():
    config = make_config(**{"sglang_cfg.decode_engine": {"dtype": "float16"}})
    with pytest.raises(ValueError, match="decode_engine may only override"):
        resolve_sglang_role_overrides(config)


def test_replica_width_counts_both_roles():
    config = make_config(
        **{
            "disaggregation.num_decode_engines": 2,
            "sglang_cfg.prefill_engine": {"tp_size": 2},
        }
    )
    assert sglang_replica_width(config) == 1 * 2 + 2 * 1
    assert sglang_replica_width(make_config(disagg=False, tp_size=4)) == 4


def test_disagg_block_defaults_come_from_the_schema():
    pd_cfg = resolve_sglang_disagg_config(make_config())
    assert pd_cfg == SGLangDisaggConfig()
    assert pd_cfg.router_scope == "replica"
    assert pd_cfg.transfer_backend == "mooncake"
    assert (pd_cfg.prefill_policy, pd_cfg.decode_policy) == (
        "cache_aware",
        "round_robin",
    )
    assert pd_cfg.decode_enable_radix_cache is False
    # An already-validated model is accepted as is.
    assert resolve_sglang_disagg_config({"sglang_cfg": {"disagg": pd_cfg}}) is pd_cfg


@pytest.mark.parametrize(
    "block",
    [
        {"router": "replica"},
        {"router_scope": "fleet"},
        {"transfer_backend": "ucx"},
        {"decode_enable_radix_cache": "yes"},
    ],
)
def test_an_unknown_or_wrong_disagg_key_is_rejected(block):
    with pytest.raises(ValidationError):
        SGLangDisaggConfig.model_validate(block)


def test_server_kwargs_follow_the_role():
    pd_cfg = SGLangDisaggConfig(transfer_backend="nixl", ib_device="mlx5_0,mlx5_1")
    prefill = sglang_disagg_server_kwargs("prefill", pd_cfg, 4321)
    decode = sglang_disagg_server_kwargs("decode", pd_cfg, None)

    assert prefill == {
        "disaggregation_mode": "prefill",
        "disaggregation_transfer_backend": "nixl",
        "disaggregation_ib_device": "mlx5_0,mlx5_1",
        "disaggregation_bootstrap_port": 4321,
    }
    assert decode == {
        "disaggregation_mode": "decode",
        "disaggregation_transfer_backend": "nixl",
        "disaggregation_ib_device": "mlx5_0,mlx5_1",
        "disaggregation_decode_enable_radix_cache": False,
    }
    assert "disaggregation_ib_device" not in sglang_disagg_server_kwargs(
        "decode", SGLangDisaggConfig(), None
    )
    with pytest.raises(AssertionError, match="bootstrap port"):
        sglang_disagg_server_kwargs("prefill", pd_cfg, None)


def test_validation_is_silent_without_disaggregation():
    config = make_config(
        disagg=False,
        **{"sglang_cfg.sglang_server_config.pause_generation_mode": "retract"},
    )
    validate_sglang_disaggregation(config, use_nemo_gym=True)


def test_validation_accepts_a_sound_layout():
    validate_sglang_disaggregation(make_config(), use_nemo_gym=False)


@pytest.mark.parametrize(
    "overrides, use_nemo_gym, match",
    [
        ({}, True, "NeMo-Gym"),
        ({"colocated.enabled": True}, False, "colocated.enabled=false"),
        ({"disaggregation.frontend_tokenize": True}, False, "frontend"),
        ({"disaggregation.num_frontend_workers": 2}, False, "frontend"),
        (
            {"sglang_cfg.sglang_server_config.pause_generation_mode": "retract"},
            False,
            "pause_generation_mode must be 'abort'",
        ),
        (
            {
                "sglang_cfg.sglang_router_config": {
                    "use_external_router": True,
                    "sglang_router_ip": "10.0.0.1",
                    "sglang_router_port": 8600,
                }
            },
            False,
            "use_external_router",
        ),
        ({"sglang_cfg.pp_size": 2}, False, "pp_size must be 1"),
        (
            {"sglang_cfg.disagg": {"decode_policy": "power_of_two"}},
            False,
            "decode_policy='power_of_two'",
        ),
        (
            {"sglang_cfg.disagg": {"prefill_policy": "power_of_two"}},
            False,
            "prefill_policy='power_of_two'",
        ),
        (
            {"sglang_cfg.prefill_engine": {"context_length": 2048}},
            False,
            "prefill_engine may only override",
        ),
    ],
)
def test_validation_refuses_configs_that_would_run_wrong(
    overrides, use_nemo_gym, match
):
    with pytest.raises(ValueError, match=match):
        validate_sglang_disaggregation(
            make_config(**overrides), use_nemo_gym=use_nemo_gym
        )


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
def _layout(enabled=True, prefill=1, decode=1):
    return DisaggConfig(
        enabled=enabled, num_prefill_engines=prefill, num_decode_engines=decode
    )


def test_plan_without_disaggregation_is_the_uniform_layout():
    specs, num_replicas, num_routers = plan_sglang_engines(
        num_gpus=8,
        gpus_per_node=4,
        layout=_layout(enabled=False),
        aggregated_width=2,
        prefill_width=2,
        decode_width=2,
        router_scope="replica",
    )
    assert (num_replicas, num_routers) == (4, 1)
    assert [s.role for s in specs] == ["aggregated"] * 4
    assert [s.gpu_offset for s in specs] == [0, 2, 4, 6]
    assert [(s.first_slot, s.num_slots, s.replica, s.router) for s in specs] == [
        (0, 1, 0, 0),
        (1, 1, 1, 0),
        (2, 1, 2, 0),
        (3, 1, 3, 0),
    ]


def test_plan_orders_prefill_then_decode_per_replica():
    specs, num_replicas, num_routers = plan_sglang_engines(
        num_gpus=16,
        gpus_per_node=8,
        layout=_layout(prefill=1, decode=3),
        aggregated_width=2,
        prefill_width=2,
        decode_width=2,
        router_scope="replica",
    )
    assert (num_replicas, num_routers) == (2, 2)
    assert [s.role for s in specs] == (["prefill"] + ["decode"] * 3) * 2
    assert [s.gpu_offset for s in specs] == [0, 2, 4, 6, 8, 10, 12, 14]
    assert [s.replica for s in specs] == [0] * 4 + [1] * 4
    assert [s.router for s in specs] == [0] * 4 + [1] * 4
    assert [s.first_slot for s in specs] == list(range(8))

    specs, _, num_routers = plan_sglang_engines(
        num_gpus=16,
        gpus_per_node=8,
        layout=_layout(prefill=1, decode=3),
        aggregated_width=2,
        prefill_width=2,
        decode_width=2,
        router_scope="global",
    )
    assert num_routers == 1
    assert [s.router for s in specs] == [0] * 8


def test_multi_node_engines_take_whole_nodes_and_mixed_widths_keep_slots_straight():
    specs, _, _ = plan_sglang_engines(
        num_gpus=12,
        gpus_per_node=4,
        layout=_layout(prefill=1, decode=1),
        aggregated_width=4,
        prefill_width=8,
        decode_width=4,
        router_scope="replica",
    )
    prefill, decode = specs
    assert (prefill.num_slots, prefill.slots, prefill.slot_width) == (
        2,
        range(0, 2),
        4,
    )
    assert [prefill.slot_gpu_index(k) for k in range(2)] == [0, 4]
    assert (decode.num_slots, decode.first_slot, decode.gpu_offset) == (1, 2, 8)
    assert decode.slot_gpu_index(0) == 8

    with pytest.raises(AssertionError, match="whole"):
        plan_sglang_engines(
            num_gpus=12,
            gpus_per_node=4,
            layout=_layout(),
            aggregated_width=4,
            prefill_width=6,
            decode_width=6,
            router_scope="replica",
        )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
def test_dispatcher_keeps_a_prompt_together_and_fills_the_least_loaded_router():
    dispatcher = ReplicaDispatcher(3)
    key_a = ReplicaDispatcher.prompt_key([1, 2, 3])
    key_b = ReplicaDispatcher.prompt_key([4, 5])
    key_c = ReplicaDispatcher.prompt_key([6])

    assert [dispatcher.acquire(key_a) for _ in range(4)] == [0, 0, 0, 0]
    assert dispatcher.acquire(key_b) == 1
    assert dispatcher.acquire(key_c) == 2
    assert dispatcher.acquire(key_b) == 1
    assert dispatcher._inflight == [4, 2, 1]

    dispatcher.release(0)
    assert dispatcher._inflight == [3, 2, 1]
    # A sticky key stays put even when another router is emptier.
    assert dispatcher.acquire(key_a) == 0

    dispatcher.reset()
    # After reset a prompt lands wherever the load is lowest.
    assert dispatcher.acquire(key_a) == 2
    with pytest.raises(AssertionError):
        ReplicaDispatcher(2).release(1)


def test_dispatcher_with_one_router_never_consults_the_table():
    dispatcher = ReplicaDispatcher(1)
    assert [dispatcher.acquire(k) for k in (1, 2, 3)] == [0, 0, 0]
    assert dispatcher._inflight == [3]
    assert ReplicaDispatcher.prompt_key([7, 8]) == ReplicaDispatcher.prompt_key([7, 8])


# ---------------------------------------------------------------------------
# Port allocation
# ---------------------------------------------------------------------------
class _FakeRemote:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class _FakeEngineActor:
    """Answers the two actor calls the allocator makes, from the test's host."""

    def __init__(self, host):
        self.host = host
        self._get_current_node_ip = _FakeRemote(lambda: host)
        # The requested start port is free: the allocator's cursor is the port.
        self._get_current_free_port = _FakeRemote(
            lambda port_range_low, port_range_high, consecutive, start_port: start_port
        )


def test_disagg_allocator_hands_each_engine_its_ports_and_prefill_a_bootstrap_port(
    monkeypatch,
):
    monkeypatch.setattr(ip_port_utils, "ray", SimpleNamespace(get=lambda ref: ref))
    specs, _, _ = plan_sglang_engines(
        num_gpus=6,
        gpus_per_node=2,
        layout=_layout(),
        aggregated_width=2,
        prefill_width=2,
        decode_width=4,
        router_scope="replica",
    )
    # Prefill engine on host A; the two-node decode engine on hosts A and B.
    local_all_engines = [
        (0, _FakeEngineActor("10.0.0.1")),
        (1, _FakeEngineActor("10.0.0.1")),
        (2, _FakeEngineActor("10.0.0.2")),
    ]

    addr_and_ports, bootstrap_ports, cursor = (
        ip_port_utils._allocate_disagg_engine_addr_and_ports(
            specs=specs,
            local_all_engines=local_all_engines,
            sglang_dp_size=1,
            port_range_low=3000,
            port_range_high=4999,
        )
    )

    # Host A: prefill takes server 3000, nccl 3001, dist-init block 3002..3032,
    # bootstrap 3033; the decode node-0 slot continues at 3034.
    assert addr_and_ports[0] == {
        "host": "10.0.0.1",
        "port": 3000,
        "nccl_port": 3001,
        "dist_init_addr": "10.0.0.1:3002",
    }
    assert bootstrap_ports == {0: 3033}
    assert addr_and_ports[1] == {
        "host": "10.0.0.1",
        "port": 3034,
        "nccl_port": 3035,
        "dist_init_addr": "10.0.0.1:3036",
    }
    # Host B starts its own cursor and shares the engine's dist-init address.
    assert addr_and_ports[2] == {
        "host": "10.0.0.2",
        "port": 3000,
        "nccl_port": 3001,
        "dist_init_addr": "10.0.0.1:3036",
    }
    assert cursor == {"10.0.0.1": 3067, "10.0.0.2": 3002}


def test_disagg_allocator_refuses_a_partial_engine_restart(monkeypatch):
    monkeypatch.setattr(ip_port_utils, "ray", SimpleNamespace(get=lambda ref: ref))
    specs, _, _ = plan_sglang_engines(
        num_gpus=6,
        gpus_per_node=2,
        layout=_layout(),
        aggregated_width=2,
        prefill_width=2,
        decode_width=4,
        router_scope="replica",
    )
    with pytest.raises(AssertionError, match="started or restarted whole"):
        ip_port_utils._allocate_disagg_engine_addr_and_ports(
            specs=specs,
            local_all_engines=[(2, _FakeEngineActor("10.0.0.2"))],
            sglang_dp_size=1,
        )


# ---------------------------------------------------------------------------
# Generation object: ordered pause, router dispatch, slot helpers
# ---------------------------------------------------------------------------
class _RecordingEngine:
    def __init__(self, name, events):
        self.name = name
        self.pause_generation = _FakeRemote(
            lambda mode: events.append((name, "pause", mode))
        )
        self.continue_generation = _FakeRemote(
            lambda: events.append((name, "continue"))
        )


def _bare_generation():
    """An instance built without __init__, with the state shutdown() reads."""
    generation = SGLangGeneration.__new__(SGLangGeneration)
    generation._health_monitor = None
    generation.weight_synchronizer = None
    generation.all_engines = []
    generation._router_actor = None
    generation._http_client = None
    generation._async_loop = None
    return generation


def _generation_with_specs(specs, engines):
    generation = _bare_generation()
    generation._engine_specs = specs
    generation.all_engines = engines
    generation.gpu_offset = 0
    return generation


def test_pause_is_prefill_then_decode_and_resume_is_the_reverse(monkeypatch):
    monkeypatch.setattr(
        sglang_generation, "ray", SimpleNamespace(get=lambda refs: refs)
    )
    specs, _, _ = plan_sglang_engines(
        num_gpus=8,
        gpus_per_node=8,
        layout=_layout(prefill=1, decode=2),
        aggregated_width=2,
        prefill_width=2,
        decode_width=3,
        router_scope="replica",
    )
    events = []
    engines = [_RecordingEngine(f"e{i}", events) for i in range(3)]
    generation = _generation_with_specs(specs, engines)

    generation.pause_generation(mode="abort")
    assert events == [
        ("e0", "pause", "abort"),
        ("e1", "pause", "abort"),
        ("e2", "pause", "abort"),
    ]
    events.clear()
    generation.continue_generation()
    assert events == [("e1", "continue"), ("e2", "continue"), ("e0", "continue")]

    # A dead engine is skipped, not addressed.
    engines[1] = None
    events.clear()
    generation.pause_generation(mode="abort")
    assert [name for name, *_ in events] == ["e0", "e2"]


def test_aggregated_engines_pause_all_at_once(monkeypatch):
    monkeypatch.setattr(
        sglang_generation, "ray", SimpleNamespace(get=lambda refs: refs)
    )
    events = []
    generation = _bare_generation()
    generation.all_engines = [_RecordingEngine(f"e{i}", events) for i in range(2)]
    generation.num_gpus_per_engine = 1
    generation.num_gpus_per_node = 1
    generation.pause_generation(mode="retract")
    assert events == [("e0", "pause", "retract"), ("e1", "pause", "retract")]


def test_slot_helpers_follow_the_specs_and_fall_back_to_the_uniform_layout():
    specs, _, _ = plan_sglang_engines(
        num_gpus=12,
        gpus_per_node=4,
        layout=_layout(),
        aggregated_width=4,
        prefill_width=8,
        decode_width=4,
        router_scope="replica",
    )
    generation = _generation_with_specs(specs, ["p0", "p1", "d0"])
    generation._engine_urls = ["http://p:1", None, "http://d:2"]
    generation._router_endpoints = [("10.0.0.9", 8600)]

    assert generation.engines == ["p0", "d0"]
    assert generation.engine_gpu_counts == [8, 4]
    assert generation.engine_gpu_offsets == [0, 8]
    assert generation.engine_slot_range(0) == range(0, 2)
    assert [generation.engine_index_of_slot(s) for s in range(3)] == [0, 0, 1]
    assert generation.engine_url(1) == "http://d:2"
    assert generation.router_endpoint_for_engine(1) == ("10.0.0.9", 8600)
    with pytest.raises(IndexError):
        generation.engine_index_of_slot(3)

    # Built without __init__: the uniform layout the old arithmetic assumed.
    uniform = _bare_generation()
    uniform.all_engines = ["a0", "a1", "b0", "b1"]
    uniform.num_gpus_per_engine = 4
    uniform.num_gpus_per_node = 2
    uniform.gpu_offset = 0
    uniform.router_ip, uniform.router_port = "127.0.0.1", 3000
    assert uniform.engines == ["a0", "b0"]
    assert uniform.engine_gpu_counts == [4, 4]
    assert uniform.engine_slot_range(1) == range(2, 4)
    assert uniform.router_endpoint_for_engine(1) == ("127.0.0.1", 3000)


def test_engine_config_is_narrowed_to_the_role():
    config = make_config(**{"sglang_cfg.prefill_engine": {"tp_size": 2}})
    generation = _bare_generation()
    generation.sglang_cfg = resolve_sglang_role_overrides(config)
    prefill = SGLangEngineSpec(
        index=0,
        role="prefill",
        width=2,
        gpu_offset=0,
        first_slot=0,
        num_slots=1,
        replica=0,
        router=0,
    )
    aggregated = SGLangEngineSpec(**{**prefill.__dict__, "role": "aggregated"})

    narrowed = generation._engine_sglang_cfg(prefill)
    assert narrowed["sglang_cfg"]["tp_size"] == 2
    assert "prefill_engine" not in narrowed["sglang_cfg"]
    assert narrowed["disaggregation"] == config["disaggregation"]
    assert generation._engine_sglang_cfg(aggregated) is generation.sglang_cfg


def test_generate_one_sample_posts_to_the_prompt_group_router_and_releases_it():
    posted = []

    class _Client:
        async def post(self, url, payload):
            posted.append((url, payload["input_ids"]))
            # Yield like a real request does, so the three stay in flight together.
            await asyncio.sleep(0)
            return {
                "meta_info": {
                    "output_token_logprobs": [(-0.5, 11), (-0.25, 12)],
                    "finish_reason": {"type": "stop"},
                }
            }

    generation = _bare_generation()
    generation._http_client = _Client()
    generation._router_endpoints = [("10.0.0.1", 8600), ("10.0.0.2", 8600)]
    generation._dispatcher = ReplicaDispatcher(2)

    async def run():
        # Two samples of one prompt and one of another, all in flight at once.
        return await asyncio.gather(
            generation.generate_one_sample({}, [1, 2, 3], 0),
            generation.generate_one_sample({}, [1, 2, 3], 1),
            generation.generate_one_sample({}, [9], 2),
        )

    results = asyncio.run(run())
    assert [r[0] for r in results] == [0, 1, 2]
    assert results[0][1:] == ([11, 12], [-0.5, -0.25], False)
    urls = [url for url, _ in posted]
    assert urls[0] == urls[1] == "http://10.0.0.1:8600/generate"
    assert urls[2] == "http://10.0.0.2:8600/generate"
    assert generation._dispatcher._inflight == [0, 0]

    # Without routers to choose from, the single configured endpoint is used.
    legacy = _bare_generation()
    legacy._http_client = _Client()
    legacy.router_ip, legacy.router_port = "127.0.0.1", 3000
    asyncio.run(legacy.generate_one_sample({}, [5], 0))
    assert posted[-1][0] == "http://127.0.0.1:3000/generate"


def test_finish_generation_forgets_the_step_prompt_placements():
    generation = _bare_generation()
    generation._health_monitor = None
    generation.needs_offload = False
    generation._dispatcher = ReplicaDispatcher(2)
    for prompt in ([1], [2]):
        target = generation._dispatcher.acquire(ReplicaDispatcher.prompt_key(prompt))
        generation._dispatcher.release(target)
    assert generation._dispatcher._sticky
    generation.finish_generation()
    assert not generation._dispatcher._sticky


# ---------------------------------------------------------------------------
# Worker: registration body and per-role server args
# ---------------------------------------------------------------------------
def test_router_registration_payload_carries_role_and_bootstrap_port():
    url = "http://10.0.0.1:3000"
    assert router_registration_payload(url, None, None) == {
        "url": url,
        "worker_type": "regular",
    }
    assert router_registration_payload(
        url, "prefill", {"disaggregation_bootstrap_port": 3033}
    ) == {"url": url, "worker_type": "prefill", "bootstrap_port": 3033}
    assert router_registration_payload(
        url, "decode", {"disaggregation_mode": "decode"}
    ) == {"url": url, "worker_type": "decode"}
    with pytest.raises(KeyError):
        router_registration_payload(url, "prefill", {})


def _bare_worker(monkeypatch, *, sglang_cfg, width, gpus_per_node, rank, node_rank):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    worker_cls = SGLangGenerationWorker.__ray_metadata__.modified_class
    worker = worker_cls.__new__(worker_cls)
    worker.gpus_per_node = gpus_per_node
    worker.sglang_cfg = sglang_cfg
    worker.rank = rank
    worker.base_gpu_id = 0
    worker.num_gpus_per_engine = width
    worker.explicit_node_rank = node_rank
    worker.disagg_role = None
    worker.disagg_server_kwargs = None
    return worker


def test_server_args_take_the_role_kwargs_and_the_explicit_node_rank(monkeypatch):
    config = make_config(**{"sglang_cfg.prefill_engine": {"tp_size": 8}})
    resolved = resolve_sglang_role_overrides(config)
    engine_cfg = {
        **resolved,
        "sglang_cfg": sglang_role_engine_config(resolved, "prefill"),
    }
    engine_cfg["sglang_cfg"]["random_seed"] = 1
    engine_cfg["sglang_cfg"]["skip_server_warmup"] = True
    # Second node of an 8-GPU engine whose slot rank is not node-aligned.
    worker = _bare_worker(
        monkeypatch,
        sglang_cfg=engine_cfg,
        width=8,
        gpus_per_node=4,
        rank=3,
        node_rank=1,
    )
    worker.disagg_role = "prefill"
    worker.disagg_server_kwargs = sglang_disagg_server_kwargs(
        "prefill", SGLangDisaggConfig(), 3033
    )

    kwargs = worker._compute_server_args("10.0.0.1:3002", 3001, "10.0.0.2", 3000)

    assert (kwargs["nnodes"], kwargs["node_rank"], kwargs["tp_size"]) == (2, 1, 8)
    assert kwargs["disaggregation_mode"] == "prefill"
    assert kwargs["disaggregation_bootstrap_port"] == 3033
    assert kwargs["disaggregation_transfer_backend"] == "mooncake"
    assert "disaggregation_decode_enable_radix_cache" not in kwargs

    worker.explicit_node_rank = 2
    with pytest.raises(AssertionError, match="node_rank 2"):
        worker._compute_server_args("10.0.0.1:3002", 3001, "10.0.0.2", 3000)


def test_server_args_without_a_role_are_unchanged(monkeypatch):
    config = make_config(disagg=False)
    config["sglang_cfg"]["random_seed"] = 1
    config["sglang_cfg"]["skip_server_warmup"] = True
    worker = _bare_worker(
        monkeypatch,
        sglang_cfg=config,
        width=1,
        gpus_per_node=2,
        rank=1,
        node_rank=None,
    )
    kwargs = worker._compute_server_args("10.0.0.1:3002", 3001, "10.0.0.1", 3000)
    assert kwargs["node_rank"] == 0
    assert not any(key.startswith("disaggregation") for key in kwargs)


# ---------------------------------------------------------------------------
# Router actor: PD mode
# ---------------------------------------------------------------------------
def test_router_actor_starts_in_pd_mode_with_the_role_policies(monkeypatch):
    args = SimpleNamespace(retry_max_retries=5, cb_failure_threshold=10)
    monkeypatch.setitem(
        sys.modules,
        "sglang_router.launch_router",
        SimpleNamespace(RouterArgs=lambda: args),
    )
    monkeypatch.setattr(sglang_router, "get_host_info", lambda: ("host", "127.0.0.1"))
    monkeypatch.setattr(sglang_router, "_get_free_port_local", lambda *_: 19001)
    router = RouterActor.__ray_metadata__.modified_class()
    start = Mock()
    monkeypatch.setattr(router, "start", start)

    assert router.init(
        {"sglang_router_port": 19000},
        pd_args={"prefill_policy": "cache_aware", "decode_policy": "round_robin"},
    ) == ("127.0.0.1", 19000)
    start.assert_called_once_with(args)
    assert args.pd_disaggregation is True
    assert (args.prefill_policy, args.decode_policy) == ("cache_aware", "round_robin")

    plain = SimpleNamespace(retry_max_retries=5, cb_failure_threshold=10)
    monkeypatch.setitem(
        sys.modules,
        "sglang_router.launch_router",
        SimpleNamespace(RouterArgs=lambda: plain),
    )
    router.init({"sglang_router_port": 19000})
    assert not hasattr(plain, "pd_disaggregation")


def test_external_router_cannot_be_put_in_pd_mode(monkeypatch):
    options = Mock()
    monkeypatch.setattr(RouterActor, "options", options)
    with pytest.raises(AssertionError, match="external_router"):
        _start_router(
            {
                "use_external_router": True,
                "sglang_router_ip": "127.0.0.1",
                "sglang_router_port": 19000,
            },
            pd_args={"prefill_policy": "cache_aware", "decode_policy": "random"},
        )
    options.assert_not_called()


def test_constructor_validates_disaggregation_before_allocating(monkeypatch):
    cluster = Mock()
    loop_factory = Mock()
    monkeypatch.setattr(sglang_generation, "AsyncLoopThread", loop_factory)
    config = make_config(
        **{"sglang_cfg.sglang_server_config.pause_generation_mode": "retract"}
    )
    generation = object.__new__(SGLangGeneration)
    with pytest.raises(ValueError, match="pause_generation_mode must be 'abort'"):
        generation.__init__(cluster, deepcopy(config))
    cluster._init_placement_groups.assert_not_called()
    loop_factory.assert_not_called()
    assert generation.shutdown()


# ---------------------------------------------------------------------------
# Engine environment per KV transport
# ---------------------------------------------------------------------------
def test_transport_env_defaults_follow_the_transport_and_yield_to_the_environment():
    nixl = SGLangDisaggConfig(transfer_backend="nixl")
    mooncake = SGLangDisaggConfig(transfer_backend="mooncake")
    tcp = SGLangDisaggConfig(transfer_backend="mooncake_tcp")

    assert disagg_transport_env(nixl, {}) == {"NCCL_NET_PLUGIN": "none"}
    assert disagg_transport_env(nixl, {"NCCL_NET_PLUGIN": "ucx"}) == {
        "NCCL_NET_PLUGIN": "ucx"
    }
    assert disagg_transport_env(mooncake, {}, peermem_loaded=False) == {
        "WITH_NVIDIA_PEERMEM": "0"
    }
    assert disagg_transport_env(
        mooncake, {"WITH_NVIDIA_PEERMEM": "1"}, peermem_loaded=False
    ) == {"WITH_NVIDIA_PEERMEM": "1"}
    assert disagg_transport_env(mooncake, {}, peermem_loaded=True) == {}
    assert disagg_transport_env(tcp, {}, peermem_loaded=False) == {}


def test_transport_env_probes_the_peermem_module(monkeypatch):
    mooncake = SGLangDisaggConfig(transfer_backend="mooncake")
    monkeypatch.setattr(sglang_generation.os.path, "exists", lambda p: False)
    assert disagg_transport_env(mooncake, {}) == {"WITH_NVIDIA_PEERMEM": "0"}
    monkeypatch.setattr(sglang_generation.os.path, "exists", lambda p: True)
    assert disagg_transport_env(mooncake, {}) == {}

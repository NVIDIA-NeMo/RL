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

"""Megatron prefill/decode engines behind Dynamo: refit channel and engine pool."""

from unittest.mock import MagicMock

import pytest

from nemo_rl.models.generation.dynamo.megatron_engine_pool import (
    build_megatron_adapter_argv,
    engine_layout_argv,
    engine_policy_config,
)
from nemo_rl.models.generation.dynamo.megatron_refit_channel import (
    MegatronRefitChannel,
    PinnedHandoffTimeout,
)
from nemo_rl.models.generation.dynamo.role_plan import (
    DisaggRolePlan,
    EngineRecord,
    EngineSpec,
)
from nemo_rl.models.generation.interfaces import CollectiveSenderSpec


class _FakeEngine:
    """One MegatronGeneration handle; every call lands in a shared log."""

    def __init__(self, name: str, log: list[str], pinned=None) -> None:
        self.name = name
        self.log = log
        self._pinned = list(pinned or [0])

    def prepare_refit_info(self, state_dict_info):
        self.log.append(f"{self.name}.prepare")

    def init_collective(self, ip, port, world_size, *, train_world_size):
        self.log.append(f"{self.name}.init(ws={world_size},off={train_world_size})")
        return [f"{self.name}.init"]

    def update_weights_from_collective(self, refit_timeout_s=None):
        self.log.append(f"{self.name}.update")
        return [f"{self.name}.update"]

    def get_collective_sender_spec(self):
        return CollectiveSenderSpec()

    def suspend_for_refit(self):
        self.log.append(f"{self.name}.suspend")

    def resume_after_refit(self):
        self.log.append(f"{self.name}.resume")

    def prepare_for_generation(self, *args, **kwargs):
        self.log.append(f"{self.name}.prepare_for_generation({kwargs.get('tags')})")
        return True

    def pinned_handoff_block_count(self):
        return self._pinned.pop(0) if len(self._pinned) > 1 else self._pinned[0]


def _plan() -> DisaggRolePlan:
    return DisaggRolePlan.from_role_counts(
        decode_workers=2, decode_world_size=2, prefill_workers=1, prefill_world_size=4
    )


def _fleet(prefill_pinned=None):
    plan = _plan()
    log: list[str] = []
    engines = [
        _FakeEngine("d0", log),
        _FakeEngine("d1", log),
        _FakeEngine("p0", log, pinned=prefill_pinned),
    ]
    records = [
        EngineRecord(spec.role, spec.world_size, spec.name, "10.0.0.1", engine)
        for spec, engine in zip(plan.engines, engines)
    ]
    channel = MegatronRefitChannel(
        records, plan=plan, control_timeout_s=1.0, idle_poll_interval_s=0.0
    )
    return channel, log


# ---------------------------------------------------------------------------
# Refit channel
# ---------------------------------------------------------------------------


def test_channel_fans_out_in_plan_order_with_cumulative_offsets() -> None:
    channel, log = _fleet()
    assert channel.inference_world_size == 8
    assert channel.sender_spec.nccl_peer == "nemo"
    channel.prepare({"w": ((1,), "float32")})
    futures = channel.init_collective("10.0.0.9", 1234, 16, train_world_size=8)
    futures += channel.update_weights()
    assert futures == [
        "d0.init",
        "d1.init",
        "p0.init",
        "d0.update",
        "d1.update",
        "p0.update",
    ]
    assert log == [
        "d0.prepare",
        "d1.prepare",
        "p0.prepare",
        "d0.init(ws=16,off=8)",
        "d1.init(ws=16,off=10)",
        "p0.init(ws=16,off=12)",
        "d0.update",
        "d1.update",
        "p0.update",
    ]
    with pytest.raises(ValueError, match="does not match expected 16"):
        channel.init_collective("10.0.0.9", 1234, 15, train_world_size=8)
    with pytest.raises(ValueError, match="must not be None"):
        channel.prepare(None)


def test_channel_pauses_decode_first_and_resumes_prefill_first() -> None:
    channel, log = _fleet()
    channel.pause()
    channel.resume()
    assert log == [
        "d0.suspend",
        "d1.suspend",
        "p0.suspend",
        "d0.prepare_for_generation(['weights'])",
        "d1.prepare_for_generation(['weights'])",
        "p0.prepare_for_generation(['weights'])",
        "p0.prepare_for_generation(['kv_cache'])",
        "p0.resume",
        "d1.prepare_for_generation(['kv_cache'])",
        "d1.resume",
        "d0.prepare_for_generation(['kv_cache'])",
        "d0.resume",
    ]
    assert channel.flush_cache() is False
    assert channel.client_copy() is channel


@pytest.mark.parametrize(
    ("prefill_pinned", "timeout_s", "raises"),
    [([3, 1, 0], 5.0, None), ([2], 0.0, PinnedHandoffTimeout)],
    ids=["pins_drain", "pins_stay"],
)
def test_channel_wait_idle_polls_prefill_pins(
    prefill_pinned, timeout_s, raises
) -> None:
    channel, _ = _fleet(prefill_pinned=prefill_pinned)
    if raises is None:
        channel.wait_idle(timeout_s=timeout_s)
    else:
        with pytest.raises(raises, match="'prefill-0': 2"):
            channel.wait_idle(timeout_s=timeout_s)


@pytest.mark.parametrize(
    ("records", "match"),
    [
        (
            [
                EngineRecord("decode", 2, "decode-0", "ip", MagicMock()),
                EngineRecord("decode", 2, "decode-1", "ip", MagicMock()),
                EngineRecord("prefill", 2, "prefill-0", "ip", MagicMock()),
            ],
            "does not match plan entry prefill-0",
        ),
        (
            [EngineRecord("decode", 2, "decode-0", "ip", MagicMock())],
            "engine records for a plan",
        ),
    ],
    ids=["wrong_width", "wrong_count"],
)
def test_channel_records_must_match_the_plan(records, match) -> None:
    with pytest.raises(ValueError, match=match):
        MegatronRefitChannel(records, plan=_plan(), control_timeout_s=1.0)


# ---------------------------------------------------------------------------
# Engine pool: adapter argv and per-engine config
# ---------------------------------------------------------------------------


def _argv(**overrides) -> list[str]:
    kwargs = dict(
        role="decode",
        component="backend",
        model_name="Qwen/Qwen3-0.6B",
        namespace="nemo-rl-7",
        parent_event_host="10.0.0.5",
        parent_event_port=4321,
        coordinator_host="10.0.0.5",
        megatron_root="/opt/megatron-lm",
        engine_start_timeout_s=600,
        handoff_owner="owner-a",
        endpoint_types=["completions"],
        engine_argv=["--tensor-model-parallel-size", "2"],
    )
    kwargs.update(overrides)
    return build_megatron_adapter_argv(**kwargs)


def _flag(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


@pytest.mark.parametrize(
    ("role", "component", "owner_flag"),
    [("decode", "backend", True), ("prefill", "prefill", False)],
)
def test_adapter_argv_is_external_mode_for_each_role(role, component, owner_flag):
    argv = _argv(role=role, component=component)
    assert {
        flag: _flag(argv, flag)
        for flag in (
            "--engine-launch-mode",
            "--role",
            "--component",
            "--parent-event-host",
            "--parent-event-port",
            "--coordinator-host",
            "--namespace",
            "--model",
            "--served-model-name",
            "--endpoint-types",
        )
    } == {
        "--engine-launch-mode": "external",
        "--role": role,
        "--component": component,
        "--parent-event-host": "10.0.0.5",
        "--parent-event-port": "4321",
        "--coordinator-host": "10.0.0.5",
        "--namespace": "nemo-rl-7",
        "--model": "Qwen/Qwen3-0.6B",
        "--served-model-name": "Qwen/Qwen3-0.6B",
        "--endpoint-types": "completions",
    }
    assert ("--handoff-owner" in argv) is owner_flag
    assert "--nproc-per-node" not in argv
    # The adapter's parser requires an engine argv after '--' even though
    # external mode never launches it.
    assert argv[argv.index("--") + 1 :] == ["--tensor-model-parallel-size", "2"]


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"extra_cli_args": ["--engine-launch-mode", "local"]}, "owns it"),
        ({"extra_cli_args": ["--engine-launch-mode=local"]}, "owns it"),
        ({"extra_cli_args": ["--nproc-per-node", "2"]}, "owns it"),
        ({"extra_cli_args": ["--handoff-owner", "x"]}, "owns it"),
        ({"extra_cli_args": ["--", "--load", "/x"]}, "may not contain '--'"),
        ({"role": "aggregated"}, "role must be"),
    ],
)
def test_adapter_argv_rejects_launch_and_ownership_overrides(overrides, match):
    with pytest.raises(ValueError, match=match):
        _argv(**overrides)


def test_engine_policy_config_applies_role_layout_to_both_layout_sources() -> None:
    policy = {
        "model_name": "m",
        "megatron_cfg": {
            "enabled": True,
            "tensor_model_parallel_size": 2,
            "pipeline_model_parallel_size": 1,
            "context_parallel_size": 1,
        },
        "generation": {
            "backend": "dynamo",
            "mcore_generation_config": {
                "tensor_model_parallel_size": 2,
                "max_requests": 8,
                "enable_prefix_caching": True,
            },
        },
    }
    spec = EngineSpec(
        role="prefill",
        index=0,
        world_size=4,
        engine_cfg={
            "megatron_cfg": {"tensor_model_parallel_size": 4},
            "mcore_generation_config": {"max_requests": 16},
        },
    )
    engine = engine_policy_config(policy, spec)
    assert engine["megatron_cfg"]["tensor_model_parallel_size"] == 4
    assert engine["megatron_cfg"]["context_parallel_size"] == 1
    assert engine["generation"]["mcore_generation_config"] == {
        "tensor_model_parallel_size": 4,
        "context_parallel_size": 1,
        "max_requests": 16,
        "enable_prefix_caching": True,
    }
    # The source config is untouched.
    assert policy["megatron_cfg"]["tensor_model_parallel_size"] == 2
    assert policy["generation"]["mcore_generation_config"]["max_requests"] == 8
    assert engine_layout_argv(engine["megatron_cfg"]) == [
        "--tensor-model-parallel-size",
        "4",
        "--pipeline-model-parallel-size",
        "1",
    ]

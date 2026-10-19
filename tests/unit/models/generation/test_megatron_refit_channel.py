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

from unittest.mock import MagicMock

import pytest

from nemo_rl.models.generation.dynamo.megatron_refit_channel import (
    MegatronRefitChannel,
    PinnedHandoffTimeout,
)
from nemo_rl.models.generation.dynamo.role_plan import DisaggRolePlan, EngineRecord
from nemo_rl.models.generation.interfaces import CollectiveSenderSpec


class _FakeEngine:
    def __init__(self, name: str, log: list[str], pinned: list[int] | None = None):
        self.name = name
        self.log = log
        self._pinned = list(pinned or [0])
        self.init_calls: list[dict] = []

    def prepare_refit_info(self, state_dict_info):
        self.log.append(f"{self.name}.prepare")

    def init_collective(self, ip, port, world_size, *, train_world_size):
        self.init_calls.append(
            dict(ip=ip, port=port, ws=world_size, off=train_world_size)
        )
        return [f"{self.name}.init"]

    def update_weights_from_collective(self, refit_timeout_s=None):
        self.log.append(f"{self.name}.update")
        return [f"{self.name}.update"]

    def get_collective_sender_spec(self):
        return CollectiveSenderSpec(
            nccl_peer="nemo", buffer_size_bytes=1, num_buffers=1
        )

    def suspend_for_refit(self):
        self.log.append(f"{self.name}.suspend")

    def resume_after_refit(self):
        self.log.append(f"{self.name}.resume")

    def prepare_for_generation(self, *args, **kwargs):
        self.log.append(f"{self.name}.prepare_for_generation({kwargs.get('tags')})")
        return True

    def pinned_handoff_block_count(self):
        return self._pinned.pop(0) if len(self._pinned) > 1 else self._pinned[0]


def _fleet(prefill_pinned=None):
    plan = DisaggRolePlan.from_role_counts(
        decode_workers=2, decode_world_size=2, prefill_workers=1, prefill_world_size=4
    )
    log: list[str] = []
    engines = [
        _FakeEngine("d0", log),
        _FakeEngine("d1", log),
        _FakeEngine("p0", log, pinned=prefill_pinned),
    ]
    records = [
        EngineRecord(
            role=spec.role,
            world_size=spec.world_size,
            instance_id=spec.name,
            node_ip="10.0.0.1",
            control=engine,
        )
        for spec, engine in zip(plan.engines, engines)
    ]
    channel = MegatronRefitChannel(
        records, plan=plan, control_timeout_s=1.0, idle_poll_interval_s=0.0
    )
    return channel, engines, log


def test_init_collective_uses_cumulative_offsets_per_engine() -> None:
    channel, engines, _ = _fleet()
    futures = channel.init_collective("10.0.0.9", 1234, 16, train_world_size=8)
    assert futures == ["d0.init", "d1.init", "p0.init"]
    assert [engine.init_calls[0]["off"] for engine in engines] == [8, 10, 12]
    assert all(call["ws"] == 16 for engine in engines for call in engine.init_calls)
    assert channel.inference_world_size == 8
    with pytest.raises(ValueError, match="does not match expected 16"):
        channel.init_collective("10.0.0.9", 1234, 15, train_world_size=8)


def test_pause_suspends_decode_first_then_prepares_weights_and_resume_reverses() -> (
    None
):
    channel, _, log = _fleet()
    channel.pause()
    assert log == [
        "d0.suspend",
        "d1.suspend",
        "p0.suspend",
        "d0.prepare_for_generation(['weights'])",
        "d1.prepare_for_generation(['weights'])",
        "p0.prepare_for_generation(['weights'])",
    ]
    log.clear()
    channel.resume()
    assert log == [
        "p0.prepare_for_generation(['kv_cache'])",
        "p0.resume",
        "d1.prepare_for_generation(['kv_cache'])",
        "d1.resume",
        "d0.prepare_for_generation(['kv_cache'])",
        "d0.resume",
    ]


def test_wait_idle_polls_prefill_pins_until_zero() -> None:
    channel, engines, _ = _fleet(prefill_pinned=[3, 1, 0])
    channel.wait_idle(timeout_s=5.0)
    assert engines[2]._pinned == [0]


def test_wait_idle_times_out_with_the_pinned_counts() -> None:
    channel, _, _ = _fleet(prefill_pinned=[2])
    with pytest.raises(PinnedHandoffTimeout, match="'prefill-0': 2"):
        channel.wait_idle(timeout_s=0.0)


def test_update_and_prepare_fan_out_in_engine_order() -> None:
    channel, _, log = _fleet()
    channel.prepare({"w": ((1,), "float32")})
    assert channel.update_weights(refit_timeout_s=None) == [
        "d0.update",
        "d1.update",
        "p0.update",
    ]
    assert log[:3] == ["d0.prepare", "d1.prepare", "p0.prepare"]
    with pytest.raises(ValueError, match="must not be None"):
        channel.prepare(None)


def test_records_must_match_the_plan() -> None:
    plan = DisaggRolePlan.from_role_counts(
        decode_workers=1, decode_world_size=2, prefill_workers=1, prefill_world_size=2
    )
    bad = [
        EngineRecord("decode", 2, "decode-0", "ip", MagicMock()),
        EngineRecord("prefill", 4, "prefill-0", "ip", MagicMock()),
    ]
    with pytest.raises(ValueError, match="does not match plan entry prefill-0"):
        MegatronRefitChannel(bad, plan=plan, control_timeout_s=1.0)
    with pytest.raises(ValueError, match="engine records for a plan"):
        MegatronRefitChannel(bad[:1], plan=plan, control_timeout_s=1.0)

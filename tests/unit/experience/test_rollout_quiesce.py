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

"""The controller-side quiesce: turn gate + outstanding counter."""

import asyncio
from types import SimpleNamespace

import pytest
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.experience.rollout_manager import (
    AsyncRolloutImpl,
    RolloutQuiesceTimeout,
    TurnGate,
)


class _BlockingGeneration:
    """generate_async that parks until the test releases it."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.proceed = asyncio.Event()
        self.calls = 0

    async def generate_async(self, data):
        self.calls += 1
        self.started.set()
        await self.proceed.wait()
        input_len = int(data["input_lengths"][0])
        yield (
            0,
            BatchedDataDict(
                {
                    "output_ids": torch.cat(
                        (data["input_ids"], torch.tensor([[7]])), dim=1
                    ),
                    "unpadded_sequence_lengths": torch.tensor([input_len + 1]),
                    "logprobs": torch.zeros(1, input_len + 1),
                }
            ),
        )


def _impl(generation, gate: TurnGate) -> AsyncRolloutImpl:
    impl = object.__new__(AsyncRolloutImpl)
    impl._policy_generation = generation
    impl._tokenizer = SimpleNamespace(pad_token_id=0, decode=lambda *a, **k: "x")
    impl._timeouts = SimpleNamespace(generation_s=None)
    impl._deadline_registry = None
    impl._turn_gate = gate
    return impl


def _message_log():
    return [{"role": "user", "content": "q", "token_ids": torch.tensor([1, 2, 3])}]


def test_turn_gate_counts_outstanding_and_quiesce_waits_for_zero() -> None:
    async def scenario():
        gate = TurnGate()
        gen = _BlockingGeneration()
        impl = _impl(gen, gate)

        turn = asyncio.create_task(impl._generate_response(_message_log(), None))
        await gen.started.wait()
        assert gate.outstanding == 1

        quiesce = asyncio.create_task(gate.quiesce(timeout_s=5.0))
        await asyncio.sleep(0)
        assert not quiesce.done()
        assert not gate.is_open

        gen.proceed.set()
        await turn
        await quiesce
        assert gate.outstanding == 0
        assert not gate.is_open  # stays closed until the controller reopens it

    asyncio.run(scenario())


def test_closed_gate_holds_new_turns_until_reopened() -> None:
    async def scenario():
        gate = TurnGate()
        gen = _BlockingGeneration()
        gen.proceed.set()
        impl = _impl(gen, gate)

        gate.close()
        turn = asyncio.create_task(impl._generate_response(_message_log(), None))
        await asyncio.sleep(0.01)
        assert gen.calls == 0, (
            "a new turn must not reach the engine while the gate is closed"
        )

        gate.open()
        await turn
        assert gen.calls == 1
        assert gate.outstanding == 0

    asyncio.run(scenario())


def test_quiesce_timeout_reopens_the_gate_and_reports_outstanding() -> None:
    async def scenario():
        gate = TurnGate()
        gen = _BlockingGeneration()
        impl = _impl(gen, gate)
        turn = asyncio.create_task(impl._generate_response(_message_log(), None))
        await gen.started.wait()

        with pytest.raises(RolloutQuiesceTimeout, match="1 generation request"):
            await gate.quiesce(timeout_s=0.01)
        assert gate.is_open

        gen.proceed.set()
        await turn

    asyncio.run(scenario())


def test_release_runs_even_when_generation_fails() -> None:
    class _Failing:
        async def generate_async(self, data):
            raise RuntimeError("engine down")
            yield  # pragma: no cover

    async def scenario():
        gate = TurnGate()
        impl = _impl(_Failing(), gate)
        with pytest.raises(RuntimeError, match="engine down"):
            await impl._generate_response(_message_log(), None)
        assert gate.outstanding == 0

    asyncio.run(scenario())

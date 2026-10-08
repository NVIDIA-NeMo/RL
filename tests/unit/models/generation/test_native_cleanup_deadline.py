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
"""Turn deadlines cannot turn interrupted engine cleanup into tolerable failures."""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import ray
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.experience import rollouts
from nemo_rl.experience.failures import (
    FailureClass,
    RolloutDataFailure,
    classify_rollout_failure,
)
from nemo_rl.models.generation.interfaces import NativeGenerationRetryConfig
from nemo_rl.models.generation.vllm.vllm_generation import VllmGeneration


@pytest.mark.parametrize(
    "backend,phase",
    [
        ("vllm", "completed"),
        ("vllm", "abort"),
        pytest.param("megatron", "completed", marks=pytest.mark.mcore),
    ],
)
@pytest.mark.parametrize("interrupt", ["deadline", "cancel"])
def test_cleanup_interruption_is_fatal(monkeypatch, backend, phase, interrupt):
    events = []

    async def run():
        cleanup_entered = asyncio.Event()

        async def pending_cleanup():
            cleanup_entered.set()
            await asyncio.Event().wait()

        class Stream:
            async def __anext__(self):
                events.append("generate")
                await asyncio.Event().wait()

            async def completed(self):
                events.append("completed")
                if phase == "completed":
                    await pending_cleanup()
                raise ray.exceptions.TaskCancelledError()

        class Workers:
            dp_size = 1

            def get_dp_leader_worker_idx(self, shard):
                return 0

            def shutdown(self, **kwargs):
                return True

            def run_single_worker_single_data(self, *, method_name, **kwargs):
                if method_name == "abort_generation":
                    events.append("abort")
                    return pending_cleanup()
                return Stream()

        if backend == "vllm":
            generation = object.__new__(VllmGeneration)
            generation.cfg = {"vllm_cfg": {"async_engine": True}}
            generation.worker_group = Workers()
            generation.current_generate_dp_shard_idx = 0
            generation.fleet_monitor = generation.fleet_selector = None
            generation.weight_synchronizer = None
        else:
            # Load the optional Megatron dependencies only for mcore cases.
            from nemo_rl.models.generation.megatron.megatron_generation import (
                MegatronGeneration,
            )

            generation = object.__new__(MegatronGeneration)
            generation._owns_policy = False
            worker = MagicMock()
            worker.generate_async._remote.return_value = Stream()
            generation._policy = SimpleNamespace(
                worker_group=SimpleNamespace(workers=[worker])
            )

        data = BatchedDataDict(
            {"input_ids": torch.tensor([[1]]), "input_lengths": torch.tensor([1])}
        )

        async def generate_turn(*args, **kwargs):
            async for _ in generation.generate_async(data):
                pytest.fail("hung request produced output")

        monkeypatch.setattr(
            rollouts, "async_generate_response_for_sample_turn", generate_turn
        )
        task = asyncio.create_task(
            rollouts._generate_sample_turn_with_retry(
                generation,
                [],
                None,
                None,
                32,
                retry_config=NativeGenerationRetryConfig(
                    backoff_seconds=0, deadline_seconds=0.2
                ),
                greedy=False,
                sample_multimodal_data={},
                deduplicate_multimodal_data=False,
            )
        )
        await asyncio.wait_for(cleanup_entered.wait(), timeout=1)
        if interrupt == "cancel":
            task.cancel()
        with pytest.raises(RolloutDataFailure, match="confirm cleanup") as exc:
            await task
        assert classify_rollout_failure(exc.value) is FailureClass.DATA
        assert isinstance(exc.value.__cause__, asyncio.CancelledError)
        assert events.count("generate") == 1
        assert len(asyncio.all_tasks()) == 1

    monkeypatch.setattr(ray, "cancel", lambda _: None)
    monkeypatch.setenv("NRL_VLLM_ASYNC_TIMEOUT_SECONDS", "0.01")
    monkeypatch.setenv("NRL_MEGATRON_ASYNC_TIMEOUT_SECONDS", "0.01")
    asyncio.run(run())

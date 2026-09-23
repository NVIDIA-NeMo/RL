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
"""Contracts between research adapters and upstream GRPO rollouts."""

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from just_grpo.config import validate_config
from just_grpo.generation.megatron_generation import MegatronDiffusionGeneration
from omegaconf import OmegaConf
from test_sudoku_and_config import PROJECT, load, load_reference_vllm

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.interfaces import should_use_async_rollouts


@pytest.fixture(scope="module", autouse=True)
def load_native_policy_before_timed_async_checks():
    import importlib

    importlib.import_module("nemo_rl.models.policy.lm_policy")


def make_generation(dispatch, *, dp=3, batch_size=2):
    group = SimpleNamespace(
        get_dp_leader_worker_idx=lambda replica: [4, 2, 7][replica],
        run_single_worker_single_data=dispatch,
    )
    policy = SimpleNamespace(worker_group=group, data_parallel_size=dp)
    config = SimpleNamespace(
        policy={
            "generation": {
                "backend": "megatron",
                "refit_transport": "mcore",
                "colocated": {"enabled": True},
                "mcore_generation_config": {
                    "refit_backend": "gloo",
                    "expose_http_server": False,
                },
            },
            "logprob_batch_size": batch_size,
        }
    )
    return MegatronDiffusionGeneration(policy=policy, config=config)


def inputs(size):
    return BatchedDataDict(
        input_ids=torch.arange(size)[:, None],
        input_lengths=torch.ones(size, dtype=torch.long),
    )


def test_async_dispatch_preserves_indices_and_streams_completed_dp_microbatches():
    async def run():
        pending = []
        ready = asyncio.Event()

        def dispatch(method, *, worker_idx, data, greedy):
            assert method == "generate" and greedy
            future = asyncio.get_running_loop().create_future()
            pending.append((worker_idx, data, future))
            if len(pending) == 3:
                ready.set()
            return future

        generation = make_generation(dispatch)
        streamed = []
        first = asyncio.Event()

        async def collect():
            async for index, row in generation.generate_async(inputs(5), greedy=True):
                streamed.append((index, int(row["input_ids"][0, 0])))
                first.set()

        task = asyncio.create_task(collect())
        await ready.wait()
        assert [p[0] for p in pending] == [4, 2, 7]
        assert [p[1].size for p in pending] == [2, 2, 1]
        # A later DP shard finishes before the first one. It must retain its index.
        pending[2][2].set_result(pending[2][1])
        await first.wait()
        assert streamed == [(4, 4)]
        assert not task.done()
        for _, data, future in pending[:2]:
            future.set_result(data)
        await task
        assert sorted(streamed) == list(enumerate(range(5)))

    asyncio.run(asyncio.wait_for(run(), timeout=5))


def test_single_row_requests_rotate_across_dp_and_survive_ray_serialization():
    import ray.cloudpickle

    async def run():
        calls = []

        def dispatch(method, *, worker_idx, data, greedy):
            calls.append(worker_idx)
            future = asyncio.get_running_loop().create_future()
            future.set_result(data)
            return future

        generation = make_generation(dispatch)
        for _ in range(4):
            result = [item async for item in generation.generate_async(inputs(1))]
            assert len(result) == 1 and result[0][0] == 0
        assert calls == [4, 2, 7, 4]
        # The collector receives a serialized wrapper, not an event-loop-bound lock.
        restored = ray.cloudpickle.loads(ray.cloudpickle.dumps(generation))
        assert next(restored._next_replica) == 4

    asyncio.run(asyncio.wait_for(run(), timeout=5))


@pytest.mark.parametrize("failure", ["worker", "cancel"])
def test_failure_or_cancellation_drains_submitted_work_before_returning(failure):
    async def run():
        pending = []
        ready = asyncio.Event()

        def dispatch(method, *, worker_idx, data, greedy):
            future = asyncio.get_running_loop().create_future()
            pending.append((future, data))
            if len(pending) == 2:
                ready.set()
            return future

        generation = make_generation(dispatch, batch_size=1)

        async def collect():
            return [item async for item in generation.generate_async(inputs(2))]

        task = asyncio.create_task(collect())
        await ready.wait()
        if failure == "worker":
            pending[0][0].set_exception(RuntimeError("worker failed"))
        else:
            task.cancel()
        # Let the collector enter its drain. It must not return with work pending.
        for _ in range(3):
            await asyncio.sleep(0)
        assert not task.done()
        for future, data in pending:
            if not future.done():
                future.set_result(data)
        with pytest.raises(
            RuntimeError if failure == "worker" else asyncio.CancelledError
        ):
            await task
        assert all(f.done() and not f.cancelled() for f, _ in pending)

    asyncio.run(asyncio.wait_for(run(), timeout=5))


def test_shared_policy_advertises_upstream_pause_and_wake_contract():
    generation = make_generation(Mock())
    from nemo_rl.models.generation.megatron.megatron_generation import (
        MegatronGeneration,
    )

    assert isinstance(generation, MegatronGeneration)
    assert should_use_async_rollouts(generation.cfg)
    assert generation.blocks_training()
    assert generation.wake_carries_weight_updates()


@pytest.mark.parametrize("fast", [False, True])
def test_async_recipe_uses_native_colocation_and_correction(fast):
    suffix = "-fast" if fast else ""
    config = load(
        PROJECT
        / f"configs/recipes/just_grpo-sudoku6x6-4n8g-megatron-inference{suffix}-async-long.yaml"
    )
    validate_config(config)
    assert config.grpo.async_grpo.enabled
    assert config.just_grpo.training_token_fraction == (0.25 if fast else 1.0)
    assert not config.grpo.async_grpo.in_flight_weight_updates
    assert config.loss_fn.use_importance_sampling_correction
    assert config.policy.generation.refit_transport == "mcore"
    assert not config.policy.generation.mcore_generation_config.expose_http_server


@pytest.mark.parametrize(
    "override,message",
    [
        ({"grpo": {"async_grpo": {"in_flight_weight_updates": True}}}, "drain"),
        (
            {"grpo": {"async_grpo": {"recompute_kv_cache_after_weight_updates": True}}},
            "KV cache",
        ),
        (
            {"loss_fn": {"use_importance_sampling_correction": False}},
            "importance sampling",
        ),
        (
            {
                "policy": {
                    "generation": {
                        "mcore_generation_config": {"expose_http_server": True}
                    }
                }
            },
            "HTTP",
        ),
        ({"policy": {"generation": {"refit_transport": None}}}, "mcore refit"),
    ],
)
def test_async_unsupported_settings_fail_before_setup(override, message):
    config = load()
    config.grpo.async_grpo.enabled = True
    with pytest.raises(ValueError, match=message):
        validate_config(OmegaConf.merge(config, override))


def test_reference_vllm_async_is_not_silently_enabled():
    config = load_reference_vllm()
    config.grpo.async_grpo.enabled = True
    with pytest.raises(ValueError, match="dedicated"):
        validate_config(config)


@pytest.mark.parametrize("fast", [False, True])
def test_dedicated_async_vllm_recipe_uses_training_gpu_count(fast):
    config = load_reference_vllm(asynchronous=True)
    config.just_grpo.training_token_fraction = 0.25 if fast else 1.0
    # 16 training ranks; this prompt count would fail validation against 32 total GPUs.
    config.grpo.num_prompts_per_step = 16
    config.policy.train_global_batch_size = 128
    validate_config(config)
    assert config.policy.generation.vllm_cfg.async_engine
    assert not config.policy.generation.colocated.enabled
    assert config.just_grpo.training_token_fraction == (0.25 if fast else 1.0)
    config.policy.generation.colocated.enabled = True
    with pytest.raises(ValueError, match="dedicated"):
        validate_config(config)

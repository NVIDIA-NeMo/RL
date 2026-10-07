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

"""Exercise native cancellation acknowledgment across real Ray/GPU processes.

The actor deliberately wedges a request after CUDA work. This tests the actual
Ray streaming cancellation protocol, independently of model loading or sampling.
"""

import asyncio
import tempfile

import pytest
import ray
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.experience import rollouts
from nemo_rl.models.generation.interfaces import NativeGenerationRetryConfig
from nemo_rl.models.generation.vllm.vllm_generation import VllmGeneration


@ray.remote(num_gpus=1, num_cpus=1)
class CudaGenerationActor:
    def __init__(self):
        self.active = set()
        self.started = 0
        self.cancelled = 0
        self.aborted = []
        self.weights = torch.ones((128, 128), device="cuda")

    async def ready(self):
        return self.weights.is_cuda

    async def generate_async(self, data, greedy, request_id):
        self.started += 1
        self.active.add(request_id)
        try:
            output = self.weights @ self.weights
            torch.cuda.synchronize()
            if self.started == 1:
                await asyncio.Event().wait()
            yield 0, {"value": output[0, 0].item()}
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        finally:
            self.active.discard(request_id)

    async def abort_generation(self, request_id):
        # Fail the test if the driver's abort overtakes Ray task cancellation.
        assert request_id not in self.active
        self.aborted.append(request_id)

    async def state(self):
        return self.started, self.cancelled, len(self.active), len(self.aborted)


class ActorWorkerGroup:
    dp_size = 1

    def __init__(self, actor):
        self.actor = actor

    def shutdown(self, **kwargs):
        # This test owns the runtime and shuts it down in its finally block.
        return True

    def get_dp_leader_worker_idx(self, shard):
        assert shard == 0
        return 0

    def run_single_worker_single_data(self, *, method_name, worker_idx, **kwargs):
        assert worker_idx == 0
        return getattr(self.actor, method_name).remote(**kwargs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a local CUDA GPU")
def test_native_retry_drains_real_ray_gpu_request(monkeypatch):
    owns_runtime = not ray.is_initialized()
    if owns_runtime:
        ray.init(
            address="local",
            num_cpus=2,
            num_gpus=1,
            include_dashboard=False,
            _temp_dir=tempfile.mkdtemp(prefix="native-retry-ray-"),
            object_store_memory=100 * 1024 * 1024,
        )
    actor = None
    try:
        actor = CudaGenerationActor.remote()
        assert ray.get(actor.ready.remote(), timeout=120)
        generation = object.__new__(VllmGeneration)
        generation.cfg = {"vllm_cfg": {"async_engine": True}}
        generation.worker_group = ActorWorkerGroup(actor)
        generation.current_generate_dp_shard_idx = 0
        generation.fleet_monitor = None
        generation.fleet_selector = None
        generation.weight_synchronizer = None
        monkeypatch.setenv("NRL_VLLM_ASYNC_TIMEOUT_SECONDS", "1")
        data = BatchedDataDict(
            {"input_ids": torch.tensor([[1]]), "input_lengths": torch.tensor([1])}
        )

        async def generate_turn(backend, messages, *_args, **_kwargs):
            outputs = [value async for _, value in backend.generate_async(data)]
            assert outputs == [{"value": 128.0, "gen_leader_worker_idx": [0]}]
            return messages, torch.tensor([2]), 1, {}

        monkeypatch.setattr(
            rollouts, "async_generate_response_for_sample_turn", generate_turn
        )
        result = asyncio.run(
            rollouts._generate_sample_turn_with_retry(
                generation,
                [],
                None,
                None,
                32,
                retry_config=NativeGenerationRetryConfig(
                    backoff_seconds=0, deadline_seconds=30
                ),
                greedy=False,
                sample_multimodal_data={},
                deduplicate_multimodal_data=False,
            )
        )
        assert result[3]["generation_retries"] == 1
        assert ray.get(actor.state.remote(), timeout=10) == (2, 1, 0, 1)
    finally:
        if actor is not None:
            ray.kill(actor)
        if owns_runtime:
            ray.shutdown()

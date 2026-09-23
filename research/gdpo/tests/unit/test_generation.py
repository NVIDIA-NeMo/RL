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

"""Tests for the Automodel generation adapter's dispatch and lifecycle."""

import pytest
import torch
from gdpo.generation import AutomodelGeneration

from nemo_rl.distributed.batched_data_dict import BatchedDataDict


class FakeWorkerGroup:
    """Records the sharded dispatch and replays canned per-worker results."""

    def __init__(self, results):
        self._results = results
        self.calls = []

    def run_all_workers_sharded_data(self, method_name, **kwargs):
        self.calls.append({"method_name": method_name, **kwargs})
        return "futures"

    def get_all_worker_results(self, futures):
        assert futures == "futures"
        return self._results


class FakePolicy:
    def __init__(self, data_parallel_size, results):
        self.data_parallel_size = data_parallel_size
        self.worker_group = FakeWorkerGroup(results)


def make_output(batch_size, width=4, offset=0):
    return BatchedDataDict(
        {
            "output_ids": torch.arange(batch_size * width).reshape(batch_size, width)
            + offset,
            "generation_lengths": torch.full((batch_size,), 2),
            "unpadded_sequence_lengths": torch.full((batch_size,), 3),
            "logprobs": torch.zeros(batch_size, width),
        }
    )


def make_backend(data_parallel_size=2, results=None, max_new_tokens=64):
    if results is None:
        results = [make_output(2), make_output(2, offset=100)]
    config = {
        "generation": {
            "backend": "automodel",
            "max_new_tokens": max_new_tokens,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": None,
            "stop_token_ids": None,
            "stop_strings": None,
            "colocated": {"enabled": True},
        }
    }
    policy = FakePolicy(data_parallel_size, results)
    return AutomodelGeneration(config=config, policy=policy), policy


def make_input(batch_size=4):
    return BatchedDataDict(
        {
            "input_ids": torch.arange(batch_size * 3).reshape(batch_size, 3),
            "input_lengths": torch.full((batch_size,), 3),
        }
    )


class TestConstruction:
    def test_generation_config_is_exposed_as_cfg(self):
        backend, _ = make_backend()
        assert backend.cfg["backend"] == "automodel"
        assert backend.cfg["max_new_tokens"] == 64

    def test_a_missing_generation_block_is_rejected(self):
        with pytest.raises(
            AssertionError, match="policy.generation must be configured"
        ):
            AutomodelGeneration(config={"generation": None}, policy=FakePolicy(1, []))


class TestGenerate:
    def test_dispatch_targets_the_worker_generate_method(self):
        backend, policy = make_backend()
        backend.generate(make_input())
        assert policy.worker_group.calls[0]["method_name"] == "generate"

    def test_data_is_sharded_over_the_data_parallel_axis(self):
        backend, policy = make_backend(data_parallel_size=2)
        backend.generate(make_input())
        call = policy.worker_group.calls[0]
        assert call["in_sharded_axes"] == ["data_parallel"]
        assert len(call["data"]) == 2

    def test_output_is_replicated_over_model_parallel_axes(self):
        backend, policy = make_backend()
        backend.generate(make_input())
        call = policy.worker_group.calls[0]
        assert call["replicate_on_axes"] == ["context_parallel", "tensor_parallel"]
        assert call["output_is_replicated"] == ["context_parallel", "tensor_parallel"]

    def test_greedy_is_forwarded_to_the_workers(self):
        backend, policy = make_backend()
        backend.generate(make_input(), greedy=True)
        assert policy.worker_group.calls[0]["common_kwargs"]["greedy"] is True

    def test_greedy_defaults_to_false(self):
        backend, policy = make_backend()
        backend.generate(make_input())
        assert policy.worker_group.calls[0]["common_kwargs"]["greedy"] is False

    def test_one_generation_seed_is_shared_by_all_workers(self, monkeypatch):
        monkeypatch.setattr(torch, "randint", lambda *args, **kwargs: torch.tensor(17))
        backend, policy = make_backend()
        backend.generate(make_input())
        assert policy.worker_group.calls[0]["common_kwargs"]["seed"] == 17

    def test_worker_results_are_concatenated(self):
        backend, _ = make_backend()
        out = backend.generate(make_input())
        assert out["output_ids"].shape[0] == 4
        assert out["generation_lengths"].shape[0] == 4

    def test_a_single_data_parallel_rank_still_shards(self):
        backend, policy = make_backend(data_parallel_size=1, results=[make_output(4)])
        out = backend.generate(make_input())
        assert len(policy.worker_group.calls[0]["data"]) == 1
        assert out["output_ids"].shape[0] == 4

    def test_the_output_carries_the_generation_output_fields(self):
        backend, _ = make_backend()
        out = backend.generate(make_input())
        for key in (
            "output_ids",
            "generation_lengths",
            "unpadded_sequence_lengths",
            "logprobs",
        ):
            assert key in out


class TestLifecycle:
    def test_prepare_for_generation_is_a_success_noop(self):
        backend, _ = make_backend()
        assert backend.prepare_for_generation() is True

    def test_finish_generation_is_a_success_noop(self):
        backend, _ = make_backend()
        assert backend.finish_generation() is True

    def test_init_collective_schedules_no_work(self):
        backend, _ = make_backend()
        assert backend.init_collective("1.2.3.4", 1234, 2, train_world_size=2) == []

    def test_shutdown_does_not_touch_the_borrowed_policy(self):
        """The trainer still owns those workers and shuts them down itself."""
        backend, policy = make_backend()
        assert backend.shutdown() is True
        assert policy.worker_group.calls == []

    def test_prepare_refit_info_accepts_the_cross_backend_contract(self):
        backend, _ = make_backend()
        assert backend.prepare_refit_info({"anything": 1}) is None

    def test_generation_blocks_training(self):
        """Denoising runs in the training workers, so it always holds their GPUs."""
        backend, _ = make_backend()
        assert backend.blocks_training() is True

    def test_waking_alone_serves_the_latest_weights(self):
        """Rollouts read the training tensors, so there is no copy to refresh."""
        backend, _ = make_backend()
        assert backend.wake_carries_weight_updates() is True

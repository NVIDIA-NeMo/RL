# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.trtllm import trtllm_generation
from nemo_rl.models.generation.trtllm.trtllm_generation import TrtllmGeneration

pytestmark = pytest.mark.trtllm


def _config(**trtllm_overrides):
    trtllm_cfg = {
        "tensor_parallel_size": 1,
        "max_model_len": 128,
        "precision": "bfloat16",
        "max_batch_size": 8,
        "max_num_tokens": 256,
        "async_engine": True,
    }
    trtllm_cfg.update(trtllm_overrides)
    return {
        "backend": "trtllm",
        "model_name": "test/model",
        "max_new_tokens": 8,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": None,
        "stop_token_ids": None,
        "stop_strings": None,
        "_pad_token_id": 0,
        "colocated": {
            "enabled": True,
            "resources": {"gpus_per_node": None, "num_nodes": None},
        },
        "trtllm_cfg": trtllm_cfg,
    }


class _Cluster:
    def __init__(self, *, world_size=2, gpus_per_node=2):
        self._world_size = world_size
        self.num_gpus_per_node = gpus_per_node
        self._init_placement_groups = MagicMock()

    def world_size(self):
        return self._world_size


def _bare_generation(*, colocated=True, dp_size=2, **trtllm_overrides):
    generation = TrtllmGeneration.__new__(TrtllmGeneration)
    generation.cfg = _config(**trtllm_overrides)
    generation.cfg["colocated"]["enabled"] = colocated
    generation.colocated_enabled = colocated
    generation.current_generate_dp_shard_idx = 0
    generation.worker_group = MagicMock()
    generation.worker_group.dp_size = dp_size
    generation.worker_group.workers = [object()] * dp_size
    return generation


@pytest.mark.parametrize(
    ("cluster", "config_update", "error"),
    [
        (_Cluster(world_size=3), {"tensor_parallel_size": 2}, "must be divisible"),
        (
            _Cluster(world_size=4),
            {
                "tensor_parallel_size": 4,
                "moe_tensor_parallel_size": 2,
                "moe_expert_parallel_size": 1,
            },
            "must equal tensor_parallel_size",
        ),
        (_Cluster(), {"async_engine": False}, "requires trtllm_cfg.async_engine=true"),
        (
            _Cluster(world_size=4, gpus_per_node=2),
            {"tensor_parallel_size": 4},
            "only supported for non-colocated",
        ),
    ],
)
def test_invalid_async_engine_configuration_fails_before_worker_start(
    cluster, config_update, error
):
    with pytest.raises(AssertionError, match=error):
        TrtllmGeneration(cluster, _config(**config_update))


def _placement_config(tp_size: int, *, colocated: bool = False) -> dict:
    config = _config(tensor_parallel_size=tp_size, pipeline_parallel_size=1)
    config["colocated"]["enabled"] = colocated
    return config


def test_init_cluster_placement_groups_uses_unified_pg_for_cross_node_tp():
    cluster = MagicMock(num_gpus_per_node=4)

    TrtllmGeneration.init_cluster_placement_groups(cluster, _placement_config(8))

    cluster._init_placement_groups.assert_called_once_with(
        strategy="PACK",
        use_unified_pg=True,
    )


def test_init_cluster_placement_groups_uses_per_node_pgs_for_node_local_tp():
    cluster = MagicMock(num_gpus_per_node=4)

    TrtllmGeneration.init_cluster_placement_groups(cluster, _placement_config(4))

    cluster._init_placement_groups.assert_called_once_with(
        strategy="PACK",
        use_unified_pg=False,
    )


def test_init_cluster_placement_groups_rejects_colocated_cross_node_tp():
    cluster = MagicMock(num_gpus_per_node=4)

    with pytest.raises(AssertionError, match="only supported for non-colocated"):
        TrtllmGeneration.init_cluster_placement_groups(
            cluster, _placement_config(8, colocated=True)
        )

    cluster._init_placement_groups.assert_not_called()


def test_cross_node_tp_replicas_use_unified_placement_group():
    generation = TrtllmGeneration.__new__(TrtllmGeneration)
    generation.model_parallel_size = 8
    generation.worker_group = MagicMock()

    unified_pg = MagicMock()
    bundle_to_node = {i: f"node-{i // 4}" for i in range(16)}
    cluster = MagicMock()
    cluster.get_placement_groups.return_value = [unified_pg]
    cluster._sorted_bundle_indices = list(range(16))

    with patch.object(
        trtllm_generation.ray.util,
        "placement_group_table",
        return_value={"bundles_to_node_id": bundle_to_node},
    ):
        result = generation._get_tied_worker_bundle_indices(cluster)

    assert result == [
        (0, list(range(8))),
        (0, list(range(8, 16))),
    ]


@pytest.mark.asyncio
async def test_generate_async_dispatches_round_robin_and_returns_leader_index():
    generation = _bare_generation(dp_size=2)
    generation.worker_group.get_dp_leader_worker_idx.side_effect = [3, 7]

    async def worker_result():
        return BatchedDataDict(
            {
                "output_ids": torch.tensor([[1, 2, 3]]),
                "logprobs": torch.zeros((1, 3)),
                "generation_lengths": torch.tensor([1]),
                "unpadded_sequence_lengths": torch.tensor([3]),
            }
        )

    generation.worker_group.run_single_worker_single_data.side_effect = (
        lambda **_: worker_result()
    )
    data = BatchedDataDict(
        {
            "input_ids": torch.tensor([[1, 2]]),
            "input_lengths": torch.tensor([2]),
        }
    )

    first = [item async for item in generation.generate_async(data, greedy=True)]
    second = [item async for item in generation.generate_async(data)]

    assert first[0][0] == 0
    assert first[0][1]["gen_leader_worker_idx"] == [3]
    assert second[0][1]["gen_leader_worker_idx"] == [7]
    assert generation.current_generate_dp_shard_idx == 0
    calls = generation.worker_group.run_single_worker_single_data.call_args_list
    assert calls[0].kwargs["worker_idx"] == 3
    assert calls[0].kwargs["greedy"] is True
    assert calls[1].kwargs["worker_idx"] == 7


@pytest.mark.asyncio
async def test_generate_async_validates_input_shape_and_empty_batch():
    generation = _bare_generation()

    missing_lengths = BatchedDataDict({"input_ids": torch.tensor([[1]])})
    with pytest.raises(AssertionError, match="input_ids and input_lengths"):
        async for _ in generation.generate_async(missing_lengths):
            pass

    multi_sample = BatchedDataDict(
        {
            "input_ids": torch.tensor([[1], [2]]),
            "input_lengths": torch.tensor([1, 1]),
        }
    )
    with pytest.raises(AssertionError, match="single-sample"):
        async for _ in generation.generate_async(multi_sample):
            pass

    empty = BatchedDataDict(
        {
            "input_ids": torch.empty((0, 0), dtype=torch.long),
            "input_lengths": torch.empty(0, dtype=torch.long),
        }
    )
    assert [item async for item in generation.generate_async(empty)] == []
    generation.worker_group.run_single_worker_single_data.assert_not_called()


@pytest.mark.asyncio
async def test_generate_async_surfaces_timeout(monkeypatch):
    generation = _bare_generation()
    generation.worker_group.get_dp_leader_worker_idx.return_value = 0

    async def pending_result():
        await asyncio.Future()

    generation.worker_group.run_single_worker_single_data.return_value = (
        pending_result()
    )

    async def raise_timeout(awaitable, timeout):
        awaitable.close()
        assert timeout == 12.5
        raise asyncio.TimeoutError

    monkeypatch.setattr(asyncio, "wait_for", raise_timeout)
    monkeypatch.setenv("NRL_TRTLLM_ASYNC_TIMEOUT_SECONDS", "12.5")
    data = BatchedDataDict(
        {"input_ids": torch.tensor([[1]]), "input_lengths": torch.tensor([1])}
    )

    with pytest.raises(RuntimeError, match="timed out after 12.5s"):
        async for _ in generation.generate_async(data):
            pass


@pytest.mark.parametrize(
    ("colocated", "prepare_method", "finish_method"),
    [
        (True, "wake_up_async", "sleep_async"),
        (False, None, "reset_prefix_cache_async"),
    ],
)
def test_generation_lifecycle_routes_by_colocation(
    monkeypatch, colocated, prepare_method, finish_method
):
    generation = _bare_generation(colocated=colocated)
    generation.worker_group.run_all_workers_single_data.return_value = [True, True]
    monkeypatch.setattr(trtllm_generation.ray, "get", lambda values: values)

    assert generation.prepare_for_generation(tags=["weights"]) is True
    if prepare_method is None:
        generation.worker_group.run_all_workers_single_data.assert_not_called()
    else:
        generation.worker_group.run_all_workers_single_data.assert_called_once_with(
            prepare_method,
            run_rank_0_only_axes=["tensor_parallel"],
            tags=["weights"],
        )

    generation.worker_group.run_all_workers_single_data.reset_mock()
    assert generation.finish_generation() is True
    generation.worker_group.run_all_workers_single_data.assert_called_once_with(
        finish_method,
        run_rank_0_only_axes=["tensor_parallel"],
    )


@pytest.mark.parametrize(
    ("in_flight", "recompute_kv", "expected_drain"),
    [(False, False, True), (True, False, False), (True, True, False)],
)
def test_collective_refit_forwards_async_update_policy(
    in_flight, recompute_kv, expected_drain
):
    generation = _bare_generation(
        colocated=False,
        in_flight_weight_updates=in_flight,
        recompute_kv_cache_after_weight_updates=recompute_kv,
    )
    expected = [SimpleNamespace()]
    generation.worker_group.run_all_workers_single_data.return_value = expected

    assert generation.update_weights_from_collective() is expected
    generation.worker_group.run_all_workers_single_data.assert_called_once_with(
        "update_weights_from_collective_async",
        run_rank_0_only_axes=["tensor_parallel"],
        drain=expected_drain,
        recompute_kv=recompute_kv,
    )


def test_ipc_refit_and_missing_worker_group():
    generation = _bare_generation(colocated=True)
    expected = [SimpleNamespace()]
    generation.worker_group.run_all_workers_single_data.return_value = expected

    assert generation.update_weights_via_ipc_zmq() is expected
    generation.worker_group.run_all_workers_single_data.assert_called_once_with(
        "update_weights_via_ipc_zmq_async",
        run_rank_0_only_axes=["tensor_parallel"],
    )

    broken = _bare_generation(colocated=True)
    broken.worker_group.workers = []
    with pytest.raises(RuntimeError, match="Worker group not initialised"):
        broken.update_weights_via_ipc_zmq()

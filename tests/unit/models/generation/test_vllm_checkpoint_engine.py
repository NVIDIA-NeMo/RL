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

"""Tests for vLLM checkpoint-engine worker lifecycle helpers."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch


@pytest.mark.vllm
@pytest.mark.parametrize(
    ("rank_prefix", "rank", "group_world_size", "rollout_world_size", "expected"),
    [
        (4, 4, 8, 8, 4),
        (2, 1, 2, 4, 3),
    ],
)
def test_resolve_rollout_rank_handles_external_and_engine_local_dp(
    monkeypatch,
    rank_prefix,
    rank,
    group_world_size,
    rollout_world_size,
    expected,
):
    from nemo_rl.models.generation.vllm.checkpoint_engine import resolve_rollout_rank

    monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: group_world_size)

    assert resolve_rollout_rank(rank_prefix, rollout_world_size) == expected


@pytest.mark.vllm
def test_checkpoint_engine_worker_lifecycle(monkeypatch):
    from nemo_rl.models.generation.vllm.checkpoint_engine import (
        VllmCheckpointEngineMixin,
    )

    worker = VllmCheckpointEngineMixin()
    worker.checkpoint_engine = MagicMock(shard_expert_weights=False)
    worker.checkpoint_engine.prepare.return_value = {"agent": "rollout"}
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 2)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    assert worker.prepare_checkpoint_engine() == {"agent": "rollout", "rank": 2}
    worker.init_checkpoint_engine_process_group(4, 3, 2, ["metadata"])
    worker.checkpoint_engine.init_rollout_process_group.assert_called_once_with(
        rollout_rank=2,
        train_world_size=3,
        rollout_world_size=2,
        metadata=["metadata"],
    )

    worker.finalize_checkpoint_engine()
    worker.checkpoint_engine.finalize.assert_called_once_with()


@pytest.mark.vllm
def test_update_weights_from_checkpoint_engine_async_loads_all_batches(monkeypatch):
    from nemo_rl.models.generation.vllm.checkpoint_engine import (
        VllmCheckpointEngineMixin,
    )

    batches = [
        [("a", torch.ones(2, 2))],
        [("b", torch.ones(3)), ("c", torch.ones(4))],
    ]

    class FakeEngine:
        shard_expert_weights = False

        async def receive_weight_batches(self):
            for batch in batches:
                yield batch

    worker = VllmCheckpointEngineMixin()
    worker.checkpoint_engine = FakeEngine()
    events = []
    worker._load_weights = lambda batch: events.append(
        ("load", [name for name, _weight in batch])
    )
    worker._maybe_process_fp8_kv_cache = lambda: events.append(("fp8",))
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda: SimpleNamespace(synchronize=lambda: events.append(("sync",))),
    )

    assert asyncio.run(worker._update_weights_from_checkpoint_engine_async()) is True
    assert events == [
        ("load", ["a"]),
        ("sync",),
        ("load", ["b", "c"]),
        ("sync",),
        ("fp8",),
    ]


@pytest.mark.vllm
def test_checkpoint_engine_worker_reports_total_memory(monkeypatch):
    from nemo_rl.models.generation.vllm.checkpoint_engine import (
        VllmCheckpointEngineMixin,
    )

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    get_device_properties = MagicMock(return_value=SimpleNamespace(total_memory=1234))
    monkeypatch.setattr(torch.cuda, "get_device_properties", get_device_properties)

    assert VllmCheckpointEngineMixin().checkpoint_engine_total_memory_bytes() == 1234
    get_device_properties.assert_called_once_with(2)


@pytest.mark.vllm
def test_checkpoint_engine_methods_only_exist_on_configured_extension():
    from nemo_rl.models.generation.vllm.checkpoint_engine import (
        VllmCheckpointEngineMixin,
    )
    from nemo_rl.models.generation.vllm.vllm_backend import (
        VllmInternalWorkerExtension,
        VllmInternalWorkerExtensionWithCheckpointEngine,
    )

    assert not issubclass(VllmInternalWorkerExtension, VllmCheckpointEngineMixin)
    assert not hasattr(VllmInternalWorkerExtension, "prepare_checkpoint_engine")
    assert issubclass(
        VllmInternalWorkerExtensionWithCheckpointEngine,
        VllmCheckpointEngineMixin,
    )


@pytest.mark.vllm
def test_checkpoint_engine_rpc_mixins_reduce_update_results():
    from nemo_rl.models.generation.vllm.checkpoint_engine import (
        VllmAsyncCheckpointEngineRpcMixin,
        VllmCheckpointEngineRpcMixin,
    )

    sync_worker = SimpleNamespace(
        llm=SimpleNamespace(collective_rpc=MagicMock(return_value=[True, None]))
    )
    assert VllmCheckpointEngineRpcMixin.checkpoint_engine_rpc(
        sync_worker, "update_weights_from_checkpoint_engine", ("arg",)
    )
    sync_worker.llm.collective_rpc.assert_called_once_with(
        "update_weights_from_checkpoint_engine", args=("arg",)
    )

    async_worker = SimpleNamespace(
        llm=SimpleNamespace(collective_rpc=AsyncMock(return_value=[True, None]))
    )
    assert asyncio.run(
        VllmAsyncCheckpointEngineRpcMixin.checkpoint_engine_rpc_async(
            async_worker, "update_weights_from_checkpoint_engine"
        )
    )

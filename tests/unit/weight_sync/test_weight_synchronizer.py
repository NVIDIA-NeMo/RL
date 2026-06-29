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

"""Unit tests for the WeightSynchronizer abstraction and its implementations."""

import asyncio
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest
import torch

from nemo_rl.models.generation.constants import (
    MEGATRON_BACKEND,
    SGLANG_BACKEND,
    VLLM_BACKEND,
)
from nemo_rl.models.policy.workers.base_policy_worker import (
    AbstractPolicyWorker,
    maybe_preinit_nixl_checkpoint_engine,
)
from nemo_rl.utils.checkpoint_engines import nixl as nixl_mod
from nemo_rl.utils.checkpoint_engines.base import (
    CheckpointEngine,
    TensorMeta,
    create_checkpoint_engine,
    merge_weight_chunk_batches,
    split_weight_chunks,
)
from nemo_rl.utils.checkpoint_engines.nixl import NIXLCheckpointEngine
from nemo_rl.weight_sync.collective_weight_synchronizer import (
    CollectiveWeightSynchronizer,
)
from nemo_rl.weight_sync.factory import (
    CheckpointEngineWeightSynchronizer,
    _ordered_generation_metadata,
    _sort_ranked_metadata,
    create_weight_synchronizer,
)
from nemo_rl.weight_sync.http_weight_synchronizer import (
    HTTPWeightSynchronizer,
)
from nemo_rl.weight_sync.interfaces import WeightSynchronizer
from nemo_rl.weight_sync.ipc_weight_synchronizer import (
    IPCWeightSynchronizer,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_policy(**overrides):
    policy = MagicMock()
    policy.offload_before_refit.return_value = None
    policy.offload_after_refit.return_value = None
    policy.prepare_refit_info.return_value = {"layer_0": {"shape": [4096, 4096]}}
    policy.stream_weights_via_ipc_zmq.return_value = [MagicMock()]
    policy.stream_weights_via_http.return_value = [MagicMock()]
    policy.broadcast_weights_for_collective.return_value = [MagicMock()]
    policy.init_collective.return_value = [MagicMock()]
    policy.get_free_memory_bytes.return_value = 1024**3  # 1 GB
    for k, v in overrides.items():
        setattr(policy, k, v)
    return policy


def _mock_generation(**overrides):
    gen = MagicMock()
    gen.cfg = {}
    gen.prepare_for_generation.return_value = True
    gen.finish_generation.return_value = True
    gen.prepare_refit_info.return_value = None
    gen.update_weights_via_ipc_zmq.return_value = [MagicMock()]
    gen.update_weights_from_collective.return_value = [MagicMock()]
    gen.get_rollout_engine_urls.return_value = ["http://localhost:30000"]
    gen.init_collective.return_value = [MagicMock()]
    for k, v in overrides.items():
        setattr(gen, k, v)
    return gen


def _mock_cluster(world_size=4, ip="127.0.0.1", port=29500):
    cluster = MagicMock()
    cluster.world_size.return_value = world_size
    cluster.get_master_address_and_port.return_value = (ip, port)
    return cluster


def _checkpoint_engine_cfg():
    return {
        "enabled": True,
        "backend": "test_backend",
        "update_weights_bucket_megabytes": 4,
        "engine_kwargs": {"test_backend": {"device": "cpu"}},
    }


class _PluginCheckpointEngine(CheckpointEngine):
    def __init__(self, bucket_size: int, marker: str) -> None:
        self.bucket_size, self.marker = bucket_size, marker

    def prepare(self):
        return {"marker": self.marker}

    def init_policy_process_group(
        self,
        *,
        worker_rank,
        train_world_size,
        rollout_world_size,
        metadata,
    ):
        pass

    def init_rollout_process_group(
        self,
        *,
        rollout_rank,
        train_world_size,
        rollout_world_size,
        metadata,
    ):
        pass

    async def send_weights(self, weights):
        pass

    async def receive_weight_batches(self):
        pass


class _RecordingCheckpointEngine(CheckpointEngine):
    def __init__(self, bucket_size: int) -> None:
        self.bucket_size = bucket_size
        self.policy_process_group = None
        self.sent_weights = None
        self.finalized = False

    def prepare(self):
        return {"bucket_size": self.bucket_size}

    def init_policy_process_group(
        self,
        *,
        worker_rank,
        train_world_size,
        rollout_world_size,
        metadata,
    ):
        self.policy_process_group = {
            "worker_rank": worker_rank,
            "train_world_size": train_world_size,
            "rollout_world_size": rollout_world_size,
            "metadata": metadata,
        }

    def init_rollout_process_group(
        self,
        *,
        rollout_rank,
        train_world_size,
        rollout_world_size,
        metadata,
    ):
        pass

    def finalize(self) -> None:
        self.finalized = True

    async def send_weights(self, weights):
        self.sent_weights = list(weights)

    async def receive_weight_batches(self):
        pass


class _CheckpointPolicyWorker(AbstractPolicyWorker):
    def __init__(self) -> None:
        self.rank = 3
        self.events = []
        self.kv_scales = None

    def _checkpoint_engine_weight_iterator(self, kv_scales=None):
        self.kv_scales = kv_scales
        yield "weight", torch.tensor([1.0, 2.0])

    def _prepare_checkpoint_engine_weight_send(self) -> None:
        self.events.append("prepare")

    def _finalize_checkpoint_engine_weight_send(self) -> None:
        self.events.append("finalize")


def _run_checkpoint_rpc(
    worker: _CheckpointPolicyWorker,
    checkpoint_method: str,
    method_kwargs: dict | None = None,
):
    return asyncio.run(
        worker.checkpoint_engine_rpc(
            checkpoint_method,
            method_kwargs=method_kwargs,
        )
    )


# ---------------------------------------------------------------------------
# WeightSynchronizer ABC contract
# ---------------------------------------------------------------------------


class TestWeightSynchronizerABC:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            WeightSynchronizer()  # type: ignore[abstract]

    def test_subclass_must_implement_all_abstract_methods(self):
        class IncompleteSync(WeightSynchronizer):
            pass

        with pytest.raises(TypeError):
            IncompleteSync()  # type: ignore[abstract]


class TestCheckpointEngineABC:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            CheckpointEngine()  # type: ignore[abstract]

    def test_subclass_must_implement_all_abstract_methods(self):
        class IncompleteEngine(CheckpointEngine):
            pass

        with pytest.raises(TypeError):
            IncompleteEngine()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# IPCWeightSynchronizer
# ---------------------------------------------------------------------------


class TestIPCWeightSynchronizer:
    @patch("nemo_rl.weight_sync.ipc_weight_synchronizer.ray")
    def test_sync_weights_calls_full_lifecycle(self, mock_ray):
        mock_ray.get.return_value = [True]
        policy = _mock_policy()
        gen = _mock_generation()
        sync = IPCWeightSynchronizer(policy, gen)

        assert sync.is_stale
        sync.sync_weights()
        assert not sync.is_stale

        policy.offload_before_refit.assert_called_once()
        gen.prepare_for_generation.assert_any_call(tags=["weights"])
        policy.stream_weights_via_ipc_zmq.assert_called_once()
        gen.update_weights_via_ipc_zmq.assert_called_once()
        policy.offload_after_refit.assert_called_once()
        gen.prepare_for_generation.assert_any_call(tags=["kv_cache"])

    @patch("nemo_rl.weight_sync.ipc_weight_synchronizer.ray")
    def test_sync_weights_raises_on_failure(self, mock_ray):
        mock_ray.get.side_effect = [
            None,  # futures_train
            [False],  # futures_inference -- update failed
        ]
        policy = _mock_policy()
        gen = _mock_generation()
        sync = IPCWeightSynchronizer(policy, gen)

        with pytest.raises(RuntimeError, match="Weight transfer failed"):
            sync.sync_weights()

    @patch("nemo_rl.weight_sync.ipc_weight_synchronizer.ray")
    def test_fixed_buffer_size(self, mock_ray):
        mock_ray.get.return_value = [True]
        policy = _mock_policy()
        gen = _mock_generation()
        sync = IPCWeightSynchronizer(policy, gen, refit_buffer_size_gb=2)

        sync.sync_weights()
        call_kwargs = policy.stream_weights_via_ipc_zmq.call_args
        assert call_kwargs.kwargs["buffer_size_bytes"] == 2 * (1024**3)

    @patch("nemo_rl.weight_sync.ipc_weight_synchronizer.ray")
    def test_dynamic_buffer_size(self, mock_ray, monkeypatch):
        monkeypatch.delenv("NRL_REFIT_BUFFER_MEMORY_RATIO", raising=False)
        mock_ray.get.return_value = [True]
        policy = _mock_policy()
        policy.get_free_memory_bytes.return_value = 10 * (1024**3)
        gen = _mock_generation()
        sync = IPCWeightSynchronizer(policy, gen)

        sync.sync_weights()
        call_kwargs = policy.stream_weights_via_ipc_zmq.call_args
        expected = int(10 * (1024**3) * 0.3)
        assert call_kwargs.kwargs["buffer_size_bytes"] == expected

    def test_mark_stale(self):
        policy = _mock_policy()
        gen = _mock_generation()
        sync = IPCWeightSynchronizer(policy, gen)

        sync._stale = False
        assert not sync.is_stale
        sync.mark_stale()
        assert sync.is_stale

    def test_init_communicator(self):
        policy = _mock_policy()
        gen = _mock_generation()
        sync = IPCWeightSynchronizer(policy, gen)

        sync.init_communicator()
        policy.prepare_refit_info.assert_called_once()
        gen.prepare_refit_info.assert_called_once()

    @patch("nemo_rl.weight_sync.ipc_weight_synchronizer.ray")
    def test_phase_restoration_on_transfer_failure(self, mock_ray):
        """offload_after_refit and kv_cache prep run even when transfer raises."""
        mock_ray.get.side_effect = RuntimeError("IPC transfer exploded")
        policy = _mock_policy()
        gen = _mock_generation()
        sync = IPCWeightSynchronizer(policy, gen)

        with pytest.raises(RuntimeError, match="IPC transfer exploded"):
            sync.sync_weights()

        policy.offload_after_refit.assert_called_once()
        gen.prepare_for_generation.assert_any_call(tags=["kv_cache"])
        assert sync.is_stale

    def test_negative_buffer_size_raises(self):
        policy = _mock_policy()
        gen = _mock_generation()
        sync = IPCWeightSynchronizer(policy, gen, refit_buffer_size_gb=-1)
        with pytest.raises(ValueError, match="refit_buffer_size_gb must be > 0"):
            sync._compute_buffer_size()

    @patch("nemo_rl.weight_sync.ipc_weight_synchronizer.ray")
    def test_invalid_env_ratio_raises(self, mock_ray, monkeypatch):
        monkeypatch.setenv("NRL_REFIT_BUFFER_MEMORY_RATIO", "not_a_number")
        policy = _mock_policy()
        gen = _mock_generation()
        sync = IPCWeightSynchronizer(policy, gen)
        with pytest.raises(ValueError, match="must be a valid float"):
            sync._compute_buffer_size()

    @patch("nemo_rl.weight_sync.ipc_weight_synchronizer.ray")
    def test_zero_env_ratio_raises(self, mock_ray, monkeypatch):
        monkeypatch.setenv("NRL_REFIT_BUFFER_MEMORY_RATIO", "0")
        policy = _mock_policy()
        gen = _mock_generation()
        sync = IPCWeightSynchronizer(policy, gen)
        with pytest.raises(ValueError, match="must be > 0"):
            sync._compute_buffer_size()


# ---------------------------------------------------------------------------
# HTTPWeightSynchronizer
# ---------------------------------------------------------------------------


class TestHTTPWeightSynchronizer:
    @patch("nemo_rl.weight_sync.http_weight_synchronizer.ray")
    def test_sync_weights_calls_full_lifecycle(self, mock_ray):
        mock_ray.get.return_value = [True]
        policy = _mock_policy()
        gen = _mock_generation()
        sync = HTTPWeightSynchronizer(policy, gen)

        assert sync.is_stale
        sync.sync_weights()
        assert not sync.is_stale

        policy.offload_before_refit.assert_called_once()
        gen.prepare_for_generation.assert_any_call(tags=["weights"])
        policy.stream_weights_via_http.assert_called_once()
        gen.get_rollout_engine_urls.assert_called_once()
        call_kwargs = policy.stream_weights_via_http.call_args
        assert call_kwargs.kwargs["rollout_engine_urls"] == ["http://localhost:30000"]
        assert call_kwargs.kwargs["buffer_size_bytes"] == int((1024**3) * 0.3)
        policy.offload_after_refit.assert_called_once()
        gen.prepare_for_generation.assert_any_call(tags=["kv_cache"])

    @patch("nemo_rl.weight_sync.http_weight_synchronizer.ray")
    def test_fixed_buffer_size(self, mock_ray):
        mock_ray.get.return_value = [True]
        policy = _mock_policy()
        gen = _mock_generation()
        sync = HTTPWeightSynchronizer(policy, gen, refit_buffer_size_gb=2)

        sync.sync_weights()
        call_kwargs = policy.stream_weights_via_http.call_args
        assert call_kwargs.kwargs["rollout_engine_urls"] == ["http://localhost:30000"]
        assert call_kwargs.kwargs["buffer_size_bytes"] == 2 * (1024**3)

    def test_mark_stale(self):
        policy = _mock_policy()
        gen = _mock_generation()
        sync = HTTPWeightSynchronizer(policy, gen)

        sync._stale = False
        assert not sync.is_stale
        sync.mark_stale()
        assert sync.is_stale

    def test_init_communicator(self):
        policy = _mock_policy()
        gen = _mock_generation()
        sync = HTTPWeightSynchronizer(policy, gen)

        sync.init_communicator()
        policy.prepare_refit_info.assert_called_once()
        gen.prepare_refit_info.assert_called_once()

    @patch("nemo_rl.weight_sync.http_weight_synchronizer.ray")
    def test_phase_restoration_on_transfer_failure(self, mock_ray):
        """offload_after_refit and kv_cache prep run even when transfer raises."""
        mock_ray.get.side_effect = RuntimeError("HTTP transfer exploded")
        policy = _mock_policy()
        gen = _mock_generation()
        sync = HTTPWeightSynchronizer(policy, gen)

        with pytest.raises(RuntimeError, match="HTTP transfer exploded"):
            sync.sync_weights()

        policy.offload_after_refit.assert_called_once()
        gen.prepare_for_generation.assert_any_call(tags=["kv_cache"])
        assert sync.is_stale

    def test_negative_buffer_size_raises(self):
        policy = _mock_policy()
        gen = _mock_generation()
        sync = HTTPWeightSynchronizer(policy, gen, refit_buffer_size_gb=-1)
        with pytest.raises(ValueError, match="refit_buffer_size_gb must be > 0"):
            sync._compute_buffer_size()

    @patch("nemo_rl.weight_sync.http_weight_synchronizer.ray")
    def test_invalid_env_ratio_raises(self, mock_ray, monkeypatch):
        monkeypatch.setenv("NRL_REFIT_BUFFER_MEMORY_RATIO", "not_a_number")
        policy = _mock_policy()
        gen = _mock_generation()
        sync = HTTPWeightSynchronizer(policy, gen)
        with pytest.raises(ValueError, match="must be a valid float"):
            sync._compute_buffer_size()

    @patch("nemo_rl.weight_sync.http_weight_synchronizer.ray")
    def test_zero_env_ratio_raises(self, mock_ray, monkeypatch):
        monkeypatch.setenv("NRL_REFIT_BUFFER_MEMORY_RATIO", "0")
        policy = _mock_policy()
        gen = _mock_generation()
        sync = HTTPWeightSynchronizer(policy, gen)
        with pytest.raises(ValueError, match="must be > 0"):
            sync._compute_buffer_size()


# ---------------------------------------------------------------------------
# CollectiveWeightSynchronizer
# ---------------------------------------------------------------------------


class TestCollectiveWeightSynchronizer:
    @patch("nemo_rl.weight_sync.collective_weight_synchronizer.ray")
    def test_sync_weights_calls_broadcast_and_receive(self, mock_ray):
        mock_ray.get.return_value = [True]
        policy = _mock_policy()
        gen = _mock_generation()
        train_cluster = _mock_cluster(world_size=4)
        inference_cluster = _mock_cluster(world_size=2)
        sync = CollectiveWeightSynchronizer(
            policy, gen, train_cluster, inference_cluster
        )

        assert sync.is_stale
        sync.sync_weights()
        assert not sync.is_stale

        policy.broadcast_weights_for_collective.assert_called_once()
        gen.update_weights_from_collective.assert_called_once()

    @patch("nemo_rl.weight_sync.collective_weight_synchronizer.ray")
    def test_sync_weights_passes_kv_scales(self, mock_ray):
        mock_ray.get.return_value = [True]
        policy = _mock_policy()
        gen = _mock_generation()
        sync = CollectiveWeightSynchronizer(
            policy, gen, _mock_cluster(), _mock_cluster()
        )
        kv_scales = {"layer.0": 1.0}

        sync.sync_weights(kv_scales=kv_scales)
        call_kwargs = policy.broadcast_weights_for_collective.call_args
        assert call_kwargs.kwargs["kv_scales"] == kv_scales

    @patch("nemo_rl.weight_sync.collective_weight_synchronizer.ray")
    def test_sync_weights_raises_on_failure(self, mock_ray):
        mock_ray.get.side_effect = [
            None,  # futures_train
            [False],  # futures_inference -- update failed
        ]
        policy = _mock_policy()
        gen = _mock_generation()
        sync = CollectiveWeightSynchronizer(
            policy, gen, _mock_cluster(), _mock_cluster()
        )

        with pytest.raises(RuntimeError, match="Weight transfer failed"):
            sync.sync_weights()

    @patch("nemo_rl.weight_sync.collective_weight_synchronizer.ray")
    def test_init_communicator_sets_up_collective(self, mock_ray):
        mock_ray.get.return_value = [True]
        policy = _mock_policy()
        gen = _mock_generation()
        train_cluster = _mock_cluster(world_size=4, ip="10.0.0.1", port=29500)
        inference_cluster = _mock_cluster(world_size=2)

        sync = CollectiveWeightSynchronizer(
            policy, gen, train_cluster, inference_cluster
        )
        sync.init_communicator()

        policy.prepare_refit_info.assert_called_once()
        gen.prepare_refit_info.assert_called_once()
        policy.init_collective.assert_called_once_with(
            "10.0.0.1", 29500, 6, train_world_size=4
        )
        gen.init_collective.assert_called_once_with(
            "10.0.0.1", 29500, 6, train_world_size=4
        )


# ---------------------------------------------------------------------------
# Checkpoint engine helpers
# ---------------------------------------------------------------------------


def test_checkpoint_engine_helpers():
    engine = create_checkpoint_engine(
        f"{__name__}:_PluginCheckpointEngine",
        bucket_size_bytes=16,
        engine_kwargs={"marker": "ok"},
    )
    assert isinstance(engine, _PluginCheckpointEngine)
    assert (engine.bucket_size, engine.marker) == (16, "ok")

    async def roundtrip(bucket_size):
        async def batches():
            for chunk in split_weight_chunks(iter([("weight", tensor)]), bucket_size):
                yield [chunk]

        merged = []
        async for batch in merge_weight_chunk_batches(batches()):
            merged.extend(batch)
        return merged

    tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    for bucket_size in (17, 1024):
        merged = asyncio.run(roundtrip(bucket_size))
        assert merged[0][0] == "weight"
        torch.testing.assert_close(merged[0][1], tensor)


def test_nixl_checkpoint_engine_rejects_invalid_bucket_size():
    with pytest.raises(ValueError, match="bucket_size must be >= 1"):
        NIXLCheckpointEngine(bucket_size=0, device="cpu")


def test_maybe_preinit_nixl_checkpoint_engine_defaults_backend(monkeypatch):
    calls = []
    fake_nixl = ModuleType("nemo_rl.utils.checkpoint_engines.nixl")
    fake_nixl.NIXL_DEFAULT_BACKEND_NAME = "UCX"

    def preinit_nixl_agent(**kwargs):
        calls.append(kwargs)
        return "agent"

    def resolve_nixl_backend_kwargs(nixl_kwargs):
        return (
            nixl_kwargs.get("backend_name", "UCX"),
            nixl_kwargs.get("backend_init_params"),
        )

    fake_nixl.preinit_nixl_agent = preinit_nixl_agent
    fake_nixl.resolve_nixl_backend_kwargs = resolve_nixl_backend_kwargs
    monkeypatch.setitem(sys.modules, "nemo_rl.utils.checkpoint_engines.nixl", fake_nixl)

    assert maybe_preinit_nixl_checkpoint_engine({}) is None
    assert (
        maybe_preinit_nixl_checkpoint_engine(
            {
                "generation": {
                    "checkpoint_engine": {
                        "enabled": True,
                        "backend": "nixl",
                        "engine_kwargs": {"nixl": {}},
                    }
                }
            }
        )
        == "agent"
    )
    assert calls == [{"backend_name": "UCX", "backend_init_params": None}]


def test_merge_weight_chunk_batches_uses_aligned_zero_copy_view():
    fp32_w = torch.tensor([7.0, 8.0], dtype=torch.float32)
    bucket = torch.zeros(64, dtype=torch.uint8)
    offset = 8
    raw = fp32_w.view(torch.uint8)
    bucket[offset : offset + fp32_w.nbytes].copy_(raw)
    chunk = bucket[offset : offset + fp32_w.nbytes]
    meta = TensorMeta(
        "fp32_w",
        fp32_w.shape,
        fp32_w.dtype,
        chunk_offset=0,
        chunk_size=fp32_w.nbytes,
        offset=offset,
    )

    async def run():
        async def batches():
            yield [(meta, chunk)]

        merged = []
        async for batch in merge_weight_chunk_batches(batches()):
            merged.extend(batch)
        return dict(merged)

    merged = asyncio.run(run())
    torch.testing.assert_close(merged["fp32_w"], fp32_w)
    assert (
        merged["fp32_w"].untyped_storage().data_ptr()
        == bucket.untyped_storage().data_ptr()
    )


def test_policy_worker_checkpoint_engine_rpc_runs_weight_send():
    worker = _CheckpointPolicyWorker()
    _run_checkpoint_rpc(
        worker,
        "init_checkpoint_engine",
        {
            "backend": f"{__name__}:_RecordingCheckpointEngine",
            "bucket_size_bytes": 32,
            "engine_kwargs": {},
        },
    )

    assert _run_checkpoint_rpc(worker, "prepare_checkpoint_engine") == {
        "bucket_size": 32,
        "rank": 3,
    }
    _run_checkpoint_rpc(
        worker,
        "init_checkpoint_engine_process_group",
        {
            "train_world_size": 2,
            "rollout_world_size": 1,
            "metadata": ["p0", "p1", "g0"],
        },
    )
    _run_checkpoint_rpc(
        worker,
        "send_weights_via_checkpoint_engine",
        {"kv_scales": {"scale": 1.0}},
    )

    assert worker.checkpoint_engine.policy_process_group == {
        "worker_rank": 3,
        "train_world_size": 2,
        "rollout_world_size": 1,
        "metadata": ["p0", "p1", "g0"],
    }
    assert worker.kv_scales == {"scale": 1.0}
    assert worker.events == ["prepare", "finalize"]
    sent_name, sent_tensor = worker.checkpoint_engine.sent_weights[0]
    assert sent_name == "weight"
    torch.testing.assert_close(sent_tensor, torch.tensor([1.0, 2.0]))

    _run_checkpoint_rpc(worker, "finalize_checkpoint_engine")
    assert worker.checkpoint_engine.finalized


def test_policy_worker_checkpoint_engine_rpc_sends_from_running_event_loop():
    worker = _CheckpointPolicyWorker()
    _run_checkpoint_rpc(
        worker,
        "init_checkpoint_engine",
        {
            "backend": f"{__name__}:_RecordingCheckpointEngine",
            "bucket_size_bytes": 32,
            "engine_kwargs": {},
        },
    )

    async def run_send() -> None:
        await worker.checkpoint_engine_rpc("send_weights_via_checkpoint_engine")

    asyncio.run(run_send())

    sent_name, sent_tensor = worker.checkpoint_engine.sent_weights[0]
    assert sent_name == "weight"
    torch.testing.assert_close(sent_tensor, torch.tensor([1.0, 2.0]))


def test_nixl_send_weights_drains_iterator_without_rollout_peer():
    engine = NIXLCheckpointEngine.__new__(NIXLCheckpointEngine)
    engine.next_agent = None
    consumed = []

    def weights():
        consumed.append("started")
        yield "weight", torch.tensor([1.0])
        consumed.append("finished")

    asyncio.run(engine.send_weights(weights()))

    assert consumed == ["started", "finished"]


def test_nixl_send_weights_aligns_bucket_offsets_for_dtype_views():
    class FakeAgent:
        def __init__(self):
            self.messages = []

        def send_message(self, _agent_name, message):
            self.messages.append(message)

        async def wait_notification(self, _agent_name, _notify_key):
            return None

    engine = NIXLCheckpointEngine.__new__(NIXLCheckpointEngine)
    engine.next_agent = "rollout"
    engine.buffers = [
        torch.zeros(64, dtype=torch.uint8),
        torch.zeros(64, dtype=torch.uint8),
    ]
    engine.xfer_descs = ["desc0", "desc1"]
    engine.bucket_size = 64
    engine._transfer_device = torch.device("cpu")
    engine.agent = FakeAgent()

    weights = iter(
        [
            ("bf16_w", torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)),
            ("fp32_w", torch.tensor([7.0], dtype=torch.float32)),
        ]
    )

    asyncio.run(engine.send_weights(weights))

    [message] = engine.agent.messages
    assert message["bucket_meta"]["bf16_w"].offset == 0
    assert message["bucket_meta"]["fp32_w"].offset == 8
    assert message["bucket_meta"]["fp32_w"].offset % torch.float32.itemsize == 0


def test_nixl_cuda_transfer_buffer_falls_back_without_cupy(monkeypatch):
    engine = NIXLCheckpointEngine.__new__(NIXLCheckpointEngine)
    engine.bucket_size = 16
    engine._transfer_device = torch.device("cuda", 0)
    engine._cupy_buffers = []
    set_device_calls = []
    zeros_calls = []

    monkeypatch.setattr(torch.cuda, "set_device", set_device_calls.append)
    monkeypatch.setattr(
        "nemo_rl.utils.checkpoint_engines.nixl.importlib.import_module",
        MagicMock(side_effect=ImportError("cupy unavailable")),
    )

    def fake_zeros(*args, **kwargs):
        zeros_calls.append((args, kwargs))
        return "buffer"

    monkeypatch.setattr(torch, "zeros", fake_zeros)

    assert engine._allocate_transfer_buffer() == "buffer"
    assert set_device_calls == [torch.device("cuda", 0)]
    assert zeros_calls == [
        (
            (16,),
            {"dtype": torch.uint8, "device": torch.device("cuda", 0)},
        )
    ]
    assert engine._cupy_buffers == []


def test_nixl_finalize_disconnects_peers():
    class FakeAgent:
        def __init__(self):
            self.removed = []

        def remove_remote_agent(self, agent_name):
            self.removed.append(agent_name)

    engine = NIXLCheckpointEngine.__new__(NIXLCheckpointEngine)
    engine.agent = FakeAgent()
    engine.prev_agent = "policy"
    engine.next_agent = "rollout"
    engine.target_rollout_rank = 3

    engine.finalize()

    assert engine.agent.removed == ["policy", "rollout"]
    assert engine.prev_agent is None
    assert engine.next_agent is None
    assert engine.target_rollout_rank is None


def test_nixl_agent_binds_zmq_socket_atomically(monkeypatch):
    class FakeNixlBackend:
        def get_agent_metadata(self):
            return {"backend": "metadata"}

    class FakePushContext:
        pass

    class FakePullSocket:
        def __init__(self):
            self.bind_endpoints = []

        def bind_to_random_port(self, endpoint):
            self.bind_endpoints.append(endpoint)
            return 45678

    class FakePullContext:
        def __init__(self, socket):
            self._socket = socket

        def socket(self, socket_type):
            assert socket_type == nixl_mod.zmq.PULL
            return self._socket

    pull_socket = FakePullSocket()
    monkeypatch.setattr(
        nixl_mod,
        "_create_nixl_agent",
        lambda agent_name, backend_name, backend_init_params: FakeNixlBackend(),
    )
    monkeypatch.setattr(nixl_mod.ray.util, "get_node_ip_address", lambda: "10.10.0.12")
    monkeypatch.setattr(nixl_mod.zmq, "Context", lambda: FakePushContext())
    monkeypatch.setattr(
        nixl_mod.zmq.asyncio, "Context", lambda: FakePullContext(pull_socket)
    )

    agent = nixl_mod.NixlAgent()

    assert agent.listen_port == 45678
    assert pull_socket.bind_endpoints == ["tcp://10.10.0.12"]
    assert agent.get_agent_metadata()["zmq_port"] == 45678


def test_nixl_process_group_uses_parallel_policy_to_rollout_topology():
    def engine_with_agent():
        engine = NIXLCheckpointEngine.__new__(NIXLCheckpointEngine)
        engine.prev_agent = None
        engine.next_agent = None
        engine.agent = MagicMock()
        engine.agent.add_remote_agent.side_effect = lambda metadata: metadata["name"]
        return engine

    metadata = [
        {"name": "policy-0"},
        {"name": "policy-1"},
        {"name": "policy-2"},
        {"name": "policy-3"},
        {"name": "rollout-0"},
        {"name": "rollout-1"},
        {"name": "rollout-2"},
    ]

    policy_rank_0 = engine_with_agent()
    policy_rank_0.init_policy_process_group(
        worker_rank=0,
        train_world_size=4,
        rollout_world_size=3,
        metadata=metadata,
    )
    assert policy_rank_0.next_agent == "rollout-0"
    assert policy_rank_0.target_rollout_rank == 0
    assert policy_rank_0.rollout_world_size == 3

    policy_rank_1 = engine_with_agent()
    policy_rank_1.init_policy_process_group(
        worker_rank=1,
        train_world_size=4,
        rollout_world_size=3,
        metadata=metadata,
    )
    assert policy_rank_1.next_agent == "rollout-1"
    assert policy_rank_1.target_rollout_rank == 1

    policy_rank_3 = engine_with_agent()
    policy_rank_3.init_policy_process_group(
        worker_rank=3,
        train_world_size=4,
        rollout_world_size=3,
        metadata=metadata,
    )
    assert policy_rank_3.next_agent is None
    assert policy_rank_3.target_rollout_rank is None
    policy_rank_3.agent.add_remote_agent.assert_not_called()

    rollout_rank_0 = engine_with_agent()
    rollout_rank_0.init_rollout_process_group(
        rollout_rank=0,
        train_world_size=4,
        rollout_world_size=3,
        metadata=metadata,
    )
    assert (rollout_rank_0.prev_agent, rollout_rank_0.next_agent) == ("policy-0", None)

    rollout_rank_1 = engine_with_agent()
    rollout_rank_1.init_rollout_process_group(
        rollout_rank=1,
        train_world_size=4,
        rollout_world_size=3,
        metadata=metadata,
    )
    assert (rollout_rank_1.prev_agent, rollout_rank_1.next_agent) == ("policy-1", None)

    rollout_rank_2 = engine_with_agent()
    rollout_rank_2.init_rollout_process_group(
        rollout_rank=2,
        train_world_size=4,
        rollout_world_size=3,
        metadata=metadata,
    )
    assert (rollout_rank_2.prev_agent, rollout_rank_2.next_agent) == (
        "policy-2",
        None,
    )


# ---------------------------------------------------------------------------
# CheckpointEngineWeightSynchronizer
# ---------------------------------------------------------------------------


class _CheckpointWorkerGroup:
    def __init__(self):
        self.workers = [object(), object(), object(), object()]
        self.calls = []

    def run_all_workers_single_data(self, method_name, **kwargs):
        self.calls.append((method_name, kwargs["checkpoint_method"]))
        return [kwargs["checkpoint_method"]]

    def run_all_workers_multiple_data(self, method_name, **kwargs):
        self.calls.append(
            (
                method_name,
                kwargs["common_kwargs"]["checkpoint_method"],
                kwargs["method_args"],
            )
        )
        return ["generation-init"]


def _checkpoint_sync(mock_ray, *, async_engine=False, update_success=True):
    # One return value per ray.get() call, in order:
    #   1. init_checkpoint_engine (policy + generation)
    #   2. prepare_checkpoint_engine (policy refs first, then generation refs;
    #      _CheckpointWorkerGroup returns a single ref per side here)
    #   3. init_checkpoint_engine_process_group (policy + generation)
    #   4. send + update_weights_from_checkpoint_engine
    #   5. finalize_checkpoint_engine (policy + generation)
    mock_ray.get.side_effect = [
        [],
        [["policy-0", "policy-1"], "generation-0", ["generation-1"]],
        [],
        ["policy-send", update_success],
        [],
    ]
    policy = _mock_policy()
    policy.worker_group = _CheckpointWorkerGroup()
    gen = _mock_generation(
        cfg={
            "vllm_cfg": {"async_engine": async_engine},
            "checkpoint_engine": _checkpoint_engine_cfg(),
        }
    )
    gen.dp_size = 2
    gen.worker_group = _CheckpointWorkerGroup()
    return CheckpointEngineWeightSynchronizer(policy, gen, gen.cfg["checkpoint_engine"])


class TestCheckpointEngineWeightSynchronizer:
    def test_sort_ranked_metadata_orders_by_rank(self):
        metadata = [{"rank": 2}, {"rank": 0}, {"rank": 1}]

        assert _sort_ranked_metadata(metadata) == [
            {"rank": 0},
            {"rank": 1},
            {"rank": 2},
        ]

    def test_ordered_generation_metadata_handles_dp_groups_with_colliding_ranks(self):
        # Two vLLM DP groups (engines), each reporting engine-local ranks 0/1 that
        # collide across groups; collective_rpc may return them out of local order.
        # The result must be global rollout-rank order: [g0r0, g0r1, g1r0, g1r1].
        generation_results = [
            [{"rank": 1, "id": "g0r1"}, {"rank": 0, "id": "g0r0"}],
            [{"rank": 1, "id": "g1r1"}, {"rank": 0, "id": "g1r0"}],
        ]

        ordered = _ordered_generation_metadata(generation_results)

        assert [m["id"] for m in ordered] == ["g0r0", "g0r1", "g1r0", "g1r1"]
        # A single global sort over colliding ranks would instead interleave the
        # groups ([g0r0, g1r0, g0r1, g1r1]) and mis-pair policy<->rollout workers.

    def test_ordered_generation_metadata_single_group(self):
        generation_results = [[{"rank": 1, "id": "r1"}, {"rank": 0, "id": "r0"}]]

        ordered = _ordered_generation_metadata(generation_results)

        assert [m["id"] for m in ordered] == ["r0", "r1"]

    @patch("nemo_rl.weight_sync.factory.ray")
    def test_sync_weights_runs_checkpoint_engine_lifecycle(self, mock_ray):
        sync = _checkpoint_sync(mock_ray)

        sync.init_communicator()
        sync.sync_weights(kv_scales={"kv": 1.0})

        assert not sync.is_stale
        sync._policy.prepare_refit_info.assert_called_once()
        sync._generation.prepare_refit_info.assert_called_once()
        assert (
            "checkpoint_engine_rpc",
            "send_weights_via_checkpoint_engine",
        ) in sync._policy.worker_group.calls
        assert (
            "checkpoint_engine_rpc",
            "update_weights_from_checkpoint_engine",
        ) in sync._generation.worker_group.calls
        assert sync._generation.worker_group.calls[2][2] == [
            (0, 2, 2, ["policy-0", "policy-1", "generation-0", "generation-1"]),
            (2, 2, 2, ["policy-0", "policy-1", "generation-0", "generation-1"]),
        ]
        sync.mark_stale()
        assert sync.is_stale
        sync.shutdown()
        assert sync._generation.worker_group.calls[-1] == (
            "checkpoint_engine_rpc",
            "finalize_checkpoint_engine",
        )
        assert sync._policy.worker_group.calls[-1] == (
            "checkpoint_engine_rpc",
            "finalize_checkpoint_engine",
        )

    @patch("nemo_rl.weight_sync.factory.ray")
    def test_sync_weights_does_not_run_colocated_phase_transitions(self, mock_ray):
        sync = _checkpoint_sync(mock_ray)

        sync.init_communicator()
        sync.sync_weights()

        sync._policy.offload_before_refit.assert_not_called()
        sync._policy.offload_after_refit.assert_not_called()
        sync._policy.prepare_for_training.assert_not_called()
        sync._generation.prepare_for_generation.assert_not_called()

    @patch("nemo_rl.weight_sync.factory.ray")
    def test_sync_weights_raises_when_generation_update_fails(self, mock_ray):
        sync = _checkpoint_sync(mock_ray, async_engine=True, update_success=False)

        sync.init_communicator()
        with pytest.raises(RuntimeError, match="Weight transfer failed"):
            sync.sync_weights()

        assert sync.is_stale
        assert sync._generation.worker_group.calls[-1] == (
            "checkpoint_engine_rpc_async",
            "update_weights_from_checkpoint_engine",
        )
        sync.shutdown()
        assert sync._generation.worker_group.calls[-1] == (
            "checkpoint_engine_rpc_async",
            "finalize_checkpoint_engine",
        )
        assert (
            sync._generation.worker_group.calls[0][0] == "checkpoint_engine_rpc_async"
        )
        assert sync._policy.worker_group.calls[-1] == (
            "checkpoint_engine_rpc",
            "finalize_checkpoint_engine",
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestFactory:
    def test_colocated_vllm_returns_ipc(self):
        policy = _mock_policy()
        gen = _mock_generation()
        sync = create_weight_synchronizer(
            policy=policy,
            generation=gen,
            generation_backend=VLLM_BACKEND,
            colocated=True,
        )
        assert isinstance(sync, IPCWeightSynchronizer)

    def test_colocated_sglang_returns_http(self):
        policy = _mock_policy()
        gen = _mock_generation()
        sync = create_weight_synchronizer(
            policy=policy,
            generation=gen,
            generation_backend=SGLANG_BACKEND,
            colocated=True,
        )
        assert isinstance(sync, HTTPWeightSynchronizer)

    def test_colocated_megatron_returns_ipc(self):
        policy = _mock_policy()
        gen = _mock_generation()
        sync = create_weight_synchronizer(
            policy=policy,
            generation=gen,
            generation_backend=MEGATRON_BACKEND,
            colocated=True,
        )
        assert isinstance(sync, IPCWeightSynchronizer)

    def test_non_colocated_vllm_returns_collective(self):
        policy = _mock_policy()
        gen = _mock_generation()
        sync = create_weight_synchronizer(
            policy=policy,
            generation=gen,
            generation_backend=VLLM_BACKEND,
            colocated=False,
            train_cluster=_mock_cluster(),
            inference_cluster=_mock_cluster(),
        )
        assert isinstance(sync, CollectiveWeightSynchronizer)

    @pytest.mark.parametrize(
        ("backend", "colocated", "expected"),
        [
            (VLLM_BACKEND, False, CheckpointEngineWeightSynchronizer),
            (VLLM_BACKEND, True, ValueError),
            (SGLANG_BACKEND, False, NotImplementedError),
            (MEGATRON_BACKEND, False, NotImplementedError),
        ],
    )
    def test_checkpoint_engine_factory_routing(self, backend, colocated, expected):
        policy = _mock_policy(cfg={})
        gen = _mock_generation(cfg={"checkpoint_engine": _checkpoint_engine_cfg()})
        if isinstance(expected, type) and issubclass(expected, Exception):
            with pytest.raises(expected):
                create_weight_synchronizer(
                    policy=policy,
                    generation=gen,
                    generation_backend=backend,
                    colocated=colocated,
                )
            return
        assert isinstance(
            create_weight_synchronizer(
                policy=policy,
                generation=gen,
                generation_backend=backend,
                colocated=colocated,
            ),
            expected,
        )

    @pytest.mark.parametrize("cfg", [{"megatron_cfg": {"enabled": False}}, {}])
    def test_checkpoint_engine_accepts_non_megatron_policy(self, cfg):
        gen = _mock_generation(cfg={"checkpoint_engine": _checkpoint_engine_cfg()})
        assert isinstance(
            create_weight_synchronizer(
                policy=_mock_policy(cfg=cfg),
                generation=gen,
                generation_backend=VLLM_BACKEND,
                colocated=False,
            ),
            CheckpointEngineWeightSynchronizer,
        )

    def test_non_colocated_sglang_raises(self):
        policy = _mock_policy()
        gen = _mock_generation()
        with pytest.raises(NotImplementedError, match="SGLang"):
            create_weight_synchronizer(
                policy=policy,
                generation=gen,
                generation_backend=SGLANG_BACKEND,
                colocated=False,
            )

    def test_non_colocated_missing_clusters_raises(self):
        policy = _mock_policy()
        gen = _mock_generation()
        with pytest.raises(ValueError, match="train_cluster"):
            create_weight_synchronizer(
                policy=policy,
                generation=gen,
                generation_backend=VLLM_BACKEND,
                colocated=False,
            )

    def test_unknown_backend_raises(self):
        policy = _mock_policy()
        gen = _mock_generation()
        with pytest.raises(ValueError, match="Unknown generation backend"):
            create_weight_synchronizer(
                policy=policy,
                generation=gen,
                generation_backend="vlllm",
                colocated=True,
            )

    def test_negative_refit_buffer_size_raises(self):
        policy = _mock_policy()
        gen = _mock_generation()
        with pytest.raises(ValueError, match="refit_buffer_size_gb must be > 0"):
            create_weight_synchronizer(
                policy=policy,
                generation=gen,
                generation_backend=VLLM_BACKEND,
                colocated=True,
                refit_buffer_size_gb=-1,
            )

    def test_zero_refit_buffer_size_raises(self):
        policy = _mock_policy()
        gen = _mock_generation()
        with pytest.raises(ValueError, match="refit_buffer_size_gb must be > 0"):
            create_weight_synchronizer(
                policy=policy,
                generation=gen,
                generation_backend=VLLM_BACKEND,
                colocated=True,
                refit_buffer_size_gb=0,
            )

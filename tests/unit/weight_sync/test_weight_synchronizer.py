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

import time
from unittest.mock import MagicMock, patch

import pytest
import ray

from nemo_rl.models.generation.constants import (
    MEGATRON_BACKEND,
    SGLANG_BACKEND,
    VLLM_BACKEND,
)
from nemo_rl.weight_sync.collective_weight_synchronizer import (
    CollectiveWeightSynchronizer,
)
from nemo_rl.weight_sync.factory import create_weight_synchronizer
from nemo_rl.weight_sync.http_weight_synchronizer import (
    HTTPWeightSynchronizer,
)
from nemo_rl.weight_sync.interfaces import WeightSynchronizer
from nemo_rl.weight_sync.ipc_weight_synchronizer import (
    IPCWeightSynchronizer,
)
from nemo_rl.weight_sync import refit_transaction

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


def _real_quant_synchronizer(kind, policy, generation):
    generation.cfg = {"real_quant": True}
    if kind == "ipc":
        return IPCWeightSynchronizer(policy, generation)
    return CollectiveWeightSynchronizer(
        policy, generation, _mock_cluster(), _mock_cluster()
    )


def test_real_quant_refit_wait_cancels_blocked_peer_and_accepts_fresh_tasks():
    @ray.remote(num_cpus=0)
    def blocked_producer():
        time.sleep(60)

    @ray.remote(num_cpus=0)
    def failed_consumer():
        raise RuntimeError("consumer loader failed")

    @ray.remote(num_cpus=0)
    def completed(value):
        return value

    producer = blocked_producer.remote()
    consumer = failed_consumer.remote()
    started = time.monotonic()
    with pytest.raises(ray.exceptions.RayTaskError, match="consumer loader failed"):
        refit_transaction.wait_for_real_quant_refit([producer], [consumer])
    assert time.monotonic() - started < 10

    assert refit_transaction.wait_for_real_quant_refit(
        [completed.remote(None)], [completed.remote(True)]
    ) == [True]


@pytest.mark.parametrize("kind", ["ipc", "collective"])
def test_real_quant_transaction_failure_is_terminal_and_reconstructable(
    monkeypatch, kind
):
    producer = object()
    consumer = object()
    policy = _mock_policy()
    generation = _mock_generation()
    if kind == "ipc":
        policy.stream_weights_via_ipc_zmq.return_value = [producer]
        generation.update_weights_via_ipc_zmq.return_value = [consumer]
        producer_method = policy.stream_weights_via_ipc_zmq
    else:
        policy.broadcast_weights_for_collective.return_value = [producer]
        generation.update_weights_from_collective.return_value = [consumer]
        producer_method = policy.broadcast_weights_for_collective
    synchronizer = _real_quant_synchronizer(kind, policy, generation)

    monkeypatch.setattr(
        refit_transaction.ray,
        "wait",
        MagicMock(return_value=([consumer], [producer])),
    )
    monkeypatch.setattr(
        refit_transaction.ray,
        "get",
        MagicMock(side_effect=RuntimeError("consumer loader failed")),
    )
    cancel = MagicMock()
    monkeypatch.setattr(refit_transaction.ray, "cancel", cancel)

    with pytest.raises(RuntimeError, match="consumer loader failed"):
        synchronizer.sync_weights()
    assert synchronizer.is_stale
    cancel.assert_called_once_with(producer)
    generation.shutdown.assert_called_once()
    policy.shutdown.assert_called_once()

    producer_calls = producer_method.call_count
    with pytest.raises(RuntimeError, match="prior failed transaction"):
        synchronizer.sync_weights()
    assert producer_method.call_count == producer_calls

    fresh_producer = object()
    fresh_consumer = object()
    fresh_policy = _mock_policy()
    fresh_generation = _mock_generation()
    if kind == "ipc":
        fresh_policy.stream_weights_via_ipc_zmq.return_value = [fresh_producer]
        fresh_generation.update_weights_via_ipc_zmq.return_value = [fresh_consumer]
    else:
        fresh_policy.broadcast_weights_for_collective.return_value = [fresh_producer]
        fresh_generation.update_weights_from_collective.return_value = [fresh_consumer]
    fresh_synchronizer = _real_quant_synchronizer(
        kind, fresh_policy, fresh_generation
    )
    refit_transaction.ray.wait.side_effect = [
        ([fresh_consumer], [fresh_producer]),
        ([fresh_producer], []),
    ]
    refit_transaction.ray.get.side_effect = [True, None]

    fresh_synchronizer.sync_weights()
    assert not fresh_synchronizer.is_stale


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
    def test_sync_weights_passes_kv_scales(self, mock_ray: MagicMock) -> None:
        mock_ray.get.return_value = [True]
        policy = _mock_policy()
        gen = _mock_generation()
        sync = IPCWeightSynchronizer(policy, gen)
        kv_scales = {"layer.0": 0.5}

        sync.sync_weights(kv_scales=kv_scales)

        call_kwargs = policy.stream_weights_via_ipc_zmq.call_args
        assert call_kwargs.kwargs["kv_scales"] == kv_scales

    @pytest.mark.parametrize("failure_phase", ["wake", "restore", "kv_cache"])
    @patch(
        "nemo_rl.weight_sync.ipc_weight_synchronizer.wait_for_real_quant_refit",
        return_value=[True],
    )
    @patch("nemo_rl.weight_sync.ipc_weight_synchronizer.ray")
    def test_real_quant_failure_in_full_lifecycle_is_terminal(
        self, mock_ray, mock_wait_for_real_quant_refit, failure_phase
    ):
        mock_ray.get.return_value = [True]
        policy = _mock_policy()
        generation = _mock_generation()
        generation.cfg = {"real_quant": True}
        if failure_phase == "wake":
            generation.prepare_for_generation.return_value = False
        elif failure_phase == "restore":
            policy.offload_after_refit.side_effect = RuntimeError("restore failed")
        else:
            generation.prepare_for_generation.side_effect = [True, False]
        synchronizer = IPCWeightSynchronizer(policy, generation)

        with pytest.raises(RuntimeError):
            synchronizer.sync_weights()
        assert synchronizer.is_stale
        generation.shutdown.assert_called_once()
        policy.shutdown.assert_called_once()

        stream_calls = policy.stream_weights_via_ipc_zmq.call_count
        with pytest.raises(RuntimeError, match="prior failed transaction"):
            synchronizer.sync_weights()
        assert policy.stream_weights_via_ipc_zmq.call_count == stream_calls

        if failure_phase != "wake":
            mock_wait_for_real_quant_refit.assert_called_once()

    @patch("nemo_rl.weight_sync.ipc_weight_synchronizer.ray")
    def test_sync_weights_raises_on_failure(self, mock_ray):
        mock_ray.get.return_value = [True]
        policy = _mock_policy()
        gen = _mock_generation()
        sync = IPCWeightSynchronizer(policy, gen)
        sync.sync_weights()
        assert not sync.is_stale

        mock_ray.get.side_effect = [
            None,  # futures_train
            [False],  # futures_inference -- update failed
        ]

        with pytest.raises(RuntimeError, match="Weight transfer failed"):
            sync.sync_weights()
        assert sync.is_stale

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
        """Policy offload runs, but a failed engine is not prepared for generation."""
        mock_ray.get.side_effect = RuntimeError("IPC transfer exploded")
        policy = _mock_policy()
        gen = _mock_generation()
        sync = IPCWeightSynchronizer(policy, gen)

        with pytest.raises(RuntimeError, match="IPC transfer exploded"):
            sync.sync_weights()

        policy.offload_after_refit.assert_called_once()
        gen.prepare_for_generation.assert_called_once_with(tags=["weights"])
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
    def test_sync_weights_rejects_kv_scales(self):
        policy = _mock_policy()
        gen = _mock_generation()
        sync = HTTPWeightSynchronizer(policy, gen)

        with pytest.raises(AssertionError, match="does not support"):
            sync.sync_weights(kv_scales={"layer.0": 0.5})

        policy.offload_before_refit.assert_not_called()
        gen.prepare_for_generation.assert_not_called()

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
        mock_ray.get.return_value = [True]
        policy = _mock_policy()
        gen = _mock_generation()
        sync = CollectiveWeightSynchronizer(
            policy, gen, _mock_cluster(), _mock_cluster()
        )
        sync.sync_weights()
        assert not sync.is_stale

        mock_ray.get.side_effect = [
            None,  # futures_train
            [False],  # futures_inference -- update failed
        ]

        with pytest.raises(RuntimeError, match="Weight transfer failed"):
            sync.sync_weights()
        assert sync.is_stale

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

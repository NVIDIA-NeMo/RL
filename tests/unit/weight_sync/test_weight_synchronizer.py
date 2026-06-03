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

from unittest.mock import MagicMock, patch

import pytest

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
    gen.prepare_for_generation.return_value = True
    gen.finish_generation.return_value = True
    gen.prepare_refit_info.return_value = None
    gen.update_weights_via_ipc_zmq.return_value = [MagicMock()]
    gen.update_weights_from_collective.return_value = [MagicMock()]
    gen.invalidate_kv_cache.return_value = True
    gen.get_sglang_url_to_gpu_uuids.return_value = {
        "http://localhost:30000": ["GPU-abc"]
    }
    gen.init_collective.return_value = [MagicMock()]
    for k, v in overrides.items():
        setattr(gen, k, v)
    return gen


def _mock_cluster(world_size=4, ip="127.0.0.1", port=29500):
    cluster = MagicMock()
    cluster.world_size.return_value = world_size
    cluster.get_master_address_and_port.return_value = (ip, port)
    return cluster


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
        gen.get_sglang_url_to_gpu_uuids.assert_called_once()
        gen.invalidate_kv_cache.assert_called_once()
        policy.stream_weights_via_http.assert_called_once()
        policy.offload_after_refit.assert_called_once()
        gen.prepare_for_generation.assert_any_call(tags=["kv_cache"])

    @patch("nemo_rl.weight_sync.http_weight_synchronizer.ray")
    def test_kv_cache_invalidation_failure_does_not_abort(self, mock_ray):
        """KV cache invalidation failure prints a warning but continues."""
        mock_ray.get.return_value = [True]
        policy = _mock_policy()
        gen = _mock_generation()
        gen.invalidate_kv_cache.return_value = False
        sync = HTTPWeightSynchronizer(policy, gen)

        sync.sync_weights()
        gen.invalidate_kv_cache.assert_called_once()
        policy.stream_weights_via_http.assert_called_once()

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

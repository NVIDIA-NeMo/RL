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
    DYNAMO_BACKEND,
    MEGATRON_BACKEND,
    SGLANG_BACKEND,
    VLLM_BACKEND,
)
from nemo_rl.models.generation.interfaces import CollectiveSenderSpec
from nemo_rl.weight_sync.collective_weight_synchronizer import (
    CollectiveWeightSynchronizer,
)
from nemo_rl.weight_sync.factory import create_weight_synchronizer
from nemo_rl.weight_sync.interfaces import WeightSynchronizer
from nemo_rl.weight_sync.ipc_weight_synchronizer import (
    IPCWeightSynchronizer,
)
from nemo_rl.weight_sync.megatron_weight_synchronizer import (
    MegatronWeightSynchronizer,
)
from nemo_rl.weight_sync.nccl_reshard_utils import build_nccl_reshard_refit_info
from nemo_rl.weight_sync.nccl_reshard_weight_synchronizer import (
    NcclReshardWeightSynchronizer,
)
from nemo_rl.weight_sync.sglang_weight_synchronizer import (
    SGLangColocatedWeightSynchronizer,
    SGLangDisaggregatedWeightSynchronizer,
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
    policy.cfg = {"megatron_cfg": {"enabled": False}}
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
    gen.init_collective.return_value = [MagicMock()]
    gen.get_collective_sender_spec.return_value = CollectiveSenderSpec()
    gen.get_inference_world_size.return_value = None
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
    def test_sync_weights_passes_kv_scales(self, mock_ray: MagicMock) -> None:
        mock_ray.get.return_value = [True]
        policy = _mock_policy()
        gen = _mock_generation()
        sync = IPCWeightSynchronizer(policy, gen)
        kv_scales = {"layer.0": 0.5}

        sync.sync_weights(kv_scales=kv_scales)

        call_kwargs = policy.stream_weights_via_ipc_zmq.call_args
        assert call_kwargs.kwargs["kv_scales"] == kv_scales

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
# SGLang synchronizers
# ---------------------------------------------------------------------------

_SGLANG_RAY = "nemo_rl.weight_sync.sglang_weight_synchronizer.ray"


def _mock_sglang_generation(num_new_engines=0, pause_mode="retract", quantization=None):
    gen = _mock_generation()
    if quantization is None:
        quantization = {"scheme": "bf16"}
    gen.sglang_cfg = {"sglang_cfg": {"quantization": quantization}}
    gen.pause_generation_mode = pause_mode
    gen.invalidate_kv_cache.return_value = True
    gen.get_updatable_engines_and_lock.return_value = (
        [MagicMock(), MagicMock()],
        MagicMock(),
        num_new_engines,
        [2, 2],
        [0, 2],
    )
    return gen


def _megatron_policy():
    return _mock_policy(cfg={"megatron_cfg": {"enabled": True}})


@patch(_SGLANG_RAY)
class TestSGLangColocatedWeightSynchronizer:
    def test_sync_weights_calls_full_lifecycle(self, mock_ray):
        policy = _mock_policy()
        gen = _mock_sglang_generation()
        sync = SGLangColocatedWeightSynchronizer(policy, gen)

        assert sync.is_stale
        sync.sync_weights()
        assert not sync.is_stale

        policy.offload_before_refit.assert_called_once()
        gen.prepare_for_generation.assert_any_call(tags=["weights"])
        gen.pause_generation.assert_called_once_with(mode="retract")
        gen.invalidate_kv_cache.assert_called_once()
        gen.begin_weight_update.assert_called_once()

        call_kwargs = policy.update_weights_to_sglang_colocated.call_args.kwargs
        assert call_kwargs["buffer_size_bytes"] == int((1024**3) * 0.3)
        assert call_kwargs["target_precision"] == "bf16"
        assert call_kwargs["sglang_quantization_cfg"] == {"scheme": "bf16"}
        mock_ray.get.assert_called_once()

        gen.end_weight_update.assert_called_once()
        gen.continue_generation.assert_called_once()
        policy.offload_after_refit.assert_called_once()
        gen.prepare_for_generation.assert_any_call(tags=["kv_cache"])

    def test_fixed_buffer_size(self, mock_ray):
        policy = _mock_policy()
        gen = _mock_sglang_generation()
        sync = SGLangColocatedWeightSynchronizer(policy, gen, refit_buffer_size_gb=2)

        sync.sync_weights()
        call_kwargs = policy.update_weights_to_sglang_colocated.call_args.kwargs
        assert call_kwargs["buffer_size_bytes"] == 2 * (1024**3)

    @pytest.mark.parametrize("quantization", [None, {}])
    def test_quantization_config_is_required(self, mock_ray, quantization):
        policy = _mock_policy()
        gen = _mock_sglang_generation()
        if quantization is None:
            del gen.sglang_cfg["sglang_cfg"]["quantization"]
        else:
            gen.sglang_cfg["sglang_cfg"]["quantization"] = quantization

        with pytest.raises(KeyError, match="quantization|scheme"):
            SGLangColocatedWeightSynchronizer(policy, gen).sync_weights()

        gen.pause_generation.assert_not_called()

    def test_unknown_quantization_scheme_is_rejected(self, mock_ray):
        policy = _mock_policy()
        gen = _mock_sglang_generation(quantization={"scheme": "unknown"})

        with pytest.raises(ValueError, match="must be one of"):
            SGLangColocatedWeightSynchronizer(policy, gen).sync_weights()

        gen.pause_generation.assert_not_called()

    def test_new_engines_trigger_connect(self, mock_ray):
        policy = _mock_policy()
        gen = _mock_sglang_generation(num_new_engines=2)
        SGLangColocatedWeightSynchronizer(policy, gen).sync_weights()

        policy.connect_sglang_rollout_engines.assert_called_once_with(
            engine_gpu_counts=[2, 2], engine_gpu_offsets=[0, 2]
        )
        gen.clear_updatable_num_new_engines.assert_called_once()

    def test_no_new_engines_skips_connect(self, mock_ray):
        policy = _mock_policy()
        gen = _mock_sglang_generation()
        SGLangColocatedWeightSynchronizer(policy, gen).sync_weights()

        policy.connect_sglang_rollout_engines.assert_not_called()

    def test_in_place_pause_is_rejected(self, mock_ray):
        policy = _mock_policy()
        gen = _mock_sglang_generation(pause_mode="in_place")
        with pytest.raises(ValueError, match="unsafe for weight refit"):
            SGLangColocatedWeightSynchronizer(policy, gen)

        gen.pause_generation.assert_not_called()
        gen.invalidate_kv_cache.assert_not_called()

    def test_kv_cache_invalidation_failure_aborts_refit(self, mock_ray):
        policy = _mock_policy()
        gen = _mock_sglang_generation()
        gen.invalidate_kv_cache.return_value = False
        sync = SGLangColocatedWeightSynchronizer(policy, gen)

        with pytest.raises(RuntimeError, match="KV cache invalidation failed"):
            sync.sync_weights()

        gen.begin_weight_update.assert_not_called()
        gen.end_weight_update.assert_not_called()
        gen.continue_generation.assert_called_once()
        policy.update_weights_to_sglang_colocated.assert_not_called()
        assert sync.is_stale

    def test_pause_failure_still_resumes_generation(self, mock_ray):
        policy = _mock_policy()
        gen = _mock_sglang_generation()
        gen.pause_generation.side_effect = RuntimeError("pause failed")

        with pytest.raises(RuntimeError, match="pause failed"):
            SGLangColocatedWeightSynchronizer(policy, gen).sync_weights()

        gen.continue_generation.assert_called_once()
        policy.offload_after_refit.assert_called_once()

    def test_prepare_failure_restores_policy_phase(self, mock_ray):
        policy = _mock_policy()
        gen = _mock_sglang_generation()
        gen.prepare_for_generation.side_effect = RuntimeError("prepare failed")

        with pytest.raises(RuntimeError, match="prepare failed"):
            SGLangColocatedWeightSynchronizer(policy, gen).sync_weights()

        policy.offload_after_refit.assert_called_once()

    def test_init_communicator(self, mock_ray):
        policy = _mock_policy()
        gen = _mock_sglang_generation()
        sync = SGLangColocatedWeightSynchronizer(policy, gen)

        sync.init_communicator()
        policy.prepare_refit_info.assert_called_once()
        gen.prepare_refit_info.assert_called_once()

    def test_phase_restoration_on_transfer_failure(self, mock_ray):
        """The engine session and both sides' phases are restored on failure."""
        mock_ray.get.side_effect = RuntimeError("IPC transfer exploded")
        policy = _mock_policy()
        gen = _mock_sglang_generation()
        sync = SGLangColocatedWeightSynchronizer(policy, gen)

        with pytest.raises(RuntimeError, match="IPC transfer exploded"):
            sync.sync_weights()

        gen.end_weight_update.assert_called_once()
        gen.continue_generation.assert_called_once()
        policy.offload_after_refit.assert_called_once()
        gen.prepare_for_generation.assert_any_call(tags=["kv_cache"])
        assert sync.is_stale

    def test_negative_buffer_size_raises(self, mock_ray):
        sync = SGLangColocatedWeightSynchronizer(
            _mock_policy(), _mock_sglang_generation(), refit_buffer_size_gb=-1
        )
        with pytest.raises(ValueError, match="refit_buffer_size_gb must be > 0"):
            sync._compute_buffer_size()

    def test_invalid_env_ratio_raises(self, mock_ray, monkeypatch):
        monkeypatch.setenv("NRL_REFIT_BUFFER_MEMORY_RATIO", "not_a_number")
        sync = SGLangColocatedWeightSynchronizer(
            _mock_policy(), _mock_sglang_generation()
        )
        with pytest.raises(ValueError, match="must be a valid float"):
            sync._compute_buffer_size()

    def test_zero_env_ratio_raises(self, mock_ray, monkeypatch):
        monkeypatch.setenv("NRL_REFIT_BUFFER_MEMORY_RATIO", "0")
        sync = SGLangColocatedWeightSynchronizer(
            _mock_policy(), _mock_sglang_generation()
        )
        with pytest.raises(ValueError, match="must be > 0"):
            sync._compute_buffer_size()

    def test_sync_weights_rejects_kv_scales(self, mock_ray):
        policy = _mock_policy()
        gen = _mock_sglang_generation()
        sync = SGLangColocatedWeightSynchronizer(policy, gen)

        with pytest.raises(ValueError, match="do not support kv_scales"):
            sync.sync_weights(kv_scales={"layer.0": 0.5})

        policy.offload_before_refit.assert_not_called()
        gen.prepare_for_generation.assert_not_called()


@patch(_SGLANG_RAY)
class TestSGLangDisaggregatedWeightSynchronizer:
    def test_sync_weights_skips_policy_offload(self, mock_ray):
        policy = _megatron_policy()
        gen = _mock_sglang_generation()
        sync = SGLangDisaggregatedWeightSynchronizer(policy, gen)

        assert sync.is_stale
        sync.sync_weights()
        assert not sync.is_stale

        # The trainer keeps its own GPUs; nothing to offload.
        policy.offload_before_refit.assert_not_called()
        policy.offload_after_refit.assert_not_called()

        # Generation phases still run; SGLangGeneration no-ops them internally
        # when the engines own their GPUs.
        gen.prepare_for_generation.assert_any_call(tags=["weights"])
        gen.prepare_for_generation.assert_any_call(tags=["kv_cache"])

        call_kwargs = policy.update_weights_to_sglang_distributed.call_args.kwargs
        assert call_kwargs["buffer_size_bytes"] == int((1024**3) * 0.3)
        assert call_kwargs["rollout_engine_lock"] is not None

    def test_new_engines_trigger_distributed_connect(self, mock_ray):
        policy = _megatron_policy()
        gen = _mock_sglang_generation(num_new_engines=1)
        SGLangDisaggregatedWeightSynchronizer(policy, gen).sync_weights()

        connect_kwargs = (
            policy.connect_sglang_rollout_engines_distributed.call_args.kwargs
        )
        assert connect_kwargs["engine_gpu_counts"] == [2, 2]
        gen.clear_updatable_num_new_engines.assert_called_once()

    def test_phase_restoration_on_transfer_failure(self, mock_ray):
        mock_ray.get.side_effect = RuntimeError("broadcast exploded")
        policy = _megatron_policy()
        gen = _mock_sglang_generation()
        sync = SGLangDisaggregatedWeightSynchronizer(policy, gen)

        with pytest.raises(RuntimeError, match="broadcast exploded"):
            sync.sync_weights()

        gen.end_weight_update.assert_called_once()
        gen.prepare_for_generation.assert_any_call(tags=["kv_cache"])
        policy.offload_after_refit.assert_not_called()
        assert sync.is_stale

    def test_sync_weights_rejects_kv_scales(self, mock_ray):
        policy = _megatron_policy()
        gen = _mock_sglang_generation()
        sync = SGLangDisaggregatedWeightSynchronizer(policy, gen)

        with pytest.raises(ValueError, match="do not support kv_scales"):
            sync.sync_weights(kv_scales={"layer.0": 0.5})

        gen.prepare_for_generation.assert_not_called()


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

        policy.broadcast_weights_for_collective.assert_called_once_with(
            kv_scales=None,
            buffer_size_bytes=None,
            num_buffers=None,
        )
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
            "10.0.0.1", 29500, 6, train_world_size=4, nccl_peer="nemo"
        )
        gen.init_collective.assert_called_once_with(
            "10.0.0.1", 29500, 6, train_world_size=4
        )

    @patch("nemo_rl.weight_sync.collective_weight_synchronizer.ray")
    def test_backend_sender_contract_controls_geometry_and_world_size(self, mock_ray):
        mock_ray.get.return_value = [True]
        policy = _mock_policy()
        gen = _mock_generation()
        gen.get_collective_sender_spec.return_value = CollectiveSenderSpec(
            nccl_peer="vllm",
            buffer_size_bytes=1024**3,
            num_buffers=2,
        )
        gen.get_inference_world_size.return_value = 8
        sync = CollectiveWeightSynchronizer(
            policy,
            gen,
            _mock_cluster(world_size=4, ip="10.0.0.1", port=29500),
            _mock_cluster(world_size=2),
        )

        sync.init_communicator()
        sync.sync_weights()

        policy.init_collective.assert_called_once_with(
            "10.0.0.1", 29500, 12, train_world_size=4, nccl_peer="vllm"
        )
        gen.init_collective.assert_called_once_with(
            "10.0.0.1", 29500, 12, train_world_size=4
        )
        policy.broadcast_weights_for_collective.assert_called_once_with(
            kv_scales=None,
            buffer_size_bytes=1024**3,
            num_buffers=2,
        )


# ---------------------------------------------------------------------------
# NcclReshardWeightSynchronizer
# ---------------------------------------------------------------------------


class TestNcclReshardWeightSynchronizer:
    @patch("nemo_rl.weight_sync.nccl_reshard_weight_synchronizer.ray")
    def test_init_communicator_ships_wire_safe_refit_info(self, mock_ray):
        # The train-side refit info carries MeshInfo rank tensors; the copy
        # handed to the generation side must be the wire-safe (plain-dict)
        # form, or the vLLM worker needs `import megatron` to unpickle it.
        mock_ray.get.return_value = [True]
        refit_info = build_nccl_reshard_refit_info(
            {
                "model.layers.0.mlp.gate_proj.weight": {
                    "shape": [64, 32],
                    "dtype": "torch.bfloat16",
                }
            },
            train_parallelism={"tp_size": 2, "ep_size": 1, "pp_size": 1},
            gen_parallelism={"tp_size": 4, "ep_size": 1, "pp_size": 1},
            train_world_size=2,
            gen_world_size=4,
        )
        policy = _mock_policy(
            cfg={
                "megatron_cfg": {
                    "tensor_model_parallel_size": 2,
                    "expert_model_parallel_size": 1,
                    "pipeline_model_parallel_size": 1,
                },
                "generation": {"vllm_cfg": {"tensor_parallel_size": 4}},
            },
        )
        policy.init_nccl_reshard_comm_group.return_value = [MagicMock()]
        policy.prepare_nccl_reshard_refit_info.return_value = refit_info
        gen = _mock_generation()
        gen.init_nccl_reshard_comm_group.return_value = [MagicMock()]
        train_cluster = _mock_cluster(world_size=2)
        train_cluster.num_gpus_per_node = 8
        train_cluster.get_available_address_and_port.return_value = (
            "10.0.0.1",
            12345,
        )
        inference_cluster = _mock_cluster(world_size=4)

        sync = NcclReshardWeightSynchronizer(
            policy, gen, train_cluster, inference_cluster
        )
        sync.init_communicator()

        policy.prepare_nccl_reshard_refit_info.assert_called_once()
        gen.prepare_nccl_reshard_refit_info.assert_called_once()
        (shipped,), _ = gen.prepare_nccl_reshard_refit_info.call_args
        for params in shipped["per_layer_params"].values():
            for p in params:
                assert isinstance(p["src_mesh_info"], dict)
                assert isinstance(p["dst_mesh_info"], dict)
                for placement in p["src_placements"] + p["dst_placements"]:
                    assert isinstance(placement, dict)

    def test_shutdown_drops_the_generation_handle(self):
        sync = NcclReshardWeightSynchronizer(
            _mock_policy(), _mock_generation(), _mock_cluster(), _mock_cluster()
        )

        sync.shutdown()

        assert sync._generation is None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _mock_megatron_generation(refit_backend="nccl", **overrides):
    gen = _mock_generation(**overrides)
    gen.cfg = {"mcore_generation_config": {"refit_backend": refit_backend}}
    gen.suspend_for_refit.return_value = None
    gen.resume_after_refit.return_value = None
    gen.preinit_nvshmem_collective.return_value = [MagicMock()]
    return gen


def _mock_megatron_policy(**overrides):
    policy = _mock_policy(**overrides)
    policy.swap_weights_via_reshard.return_value = [MagicMock()]
    policy.init_collective_mcore_generation.return_value = [MagicMock()]
    policy.preinit_nvshmem.return_value = [MagicMock()]
    return policy


class TestMegatronWeightSynchronizer:
    def test_non_colocated_requires_clusters(self):
        with pytest.raises(ValueError):
            MegatronWeightSynchronizer(
                _mock_megatron_policy(), _mock_megatron_generation(), colocated=False
            )

    def test_colocated_sync_is_offload_and_wake(self):
        policy = _mock_megatron_policy()
        gen = _mock_megatron_generation()
        sync = MegatronWeightSynchronizer(policy, gen, colocated=True)

        sync.init_communicator()  # no collective to wire
        policy.init_collective_mcore_generation.assert_not_called()

        assert sync.is_stale
        assert sync.sync_weights() == {}
        policy.offload_before_refit.assert_called_once()
        # The refit-protocol tag makes the wake bypass the worker's
        # engine-awake early-return (the reshard copy rides this wake).
        gen.prepare_for_generation.assert_called_once_with(tags=["colocated_refit"])
        gen.suspend_for_refit.assert_not_called()
        policy.swap_weights_via_reshard.assert_not_called()
        assert not sync.is_stale

    @patch("nemo_rl.weight_sync.megatron_weight_synchronizer.ray")
    def test_non_colocated_sync_sequence(self, mock_ray):
        mock_ray.get.side_effect = lambda futures: [True for _ in futures]
        policy = _mock_megatron_policy()
        gen = _mock_megatron_generation()
        sync = MegatronWeightSynchronizer(
            policy,
            gen,
            colocated=False,
            train_cluster=_mock_cluster(),
            inference_cluster=_mock_cluster(),
        )

        sync.init_communicator()
        policy.init_collective_mcore_generation.assert_called_once()
        gen.init_collective.assert_called_once()

        assert sync.sync_weights() == {}
        gen.suspend_for_refit.assert_called_once()
        policy.offload_before_refit.assert_called_once()
        policy.swap_weights_via_reshard.assert_called_once_with(is_source=True)
        gen.update_weights_from_collective.assert_called_once()
        gen.resume_after_refit.assert_called_once()
        # prepare called for the weights phase and then the kv_cache phase
        tags = [c.kwargs.get("tags") for c in gen.prepare_for_generation.call_args_list]
        assert tags == [["weights"], ["kv_cache"]]
        # no nvshmem preinit on the nccl backend
        policy.preinit_nvshmem.assert_not_called()
        assert not sync.is_stale

    @patch("nemo_rl.weight_sync.megatron_weight_synchronizer.ray")
    def test_non_colocated_nvshmem_preinits(self, mock_ray):
        mock_ray.get.side_effect = lambda futures: [True for _ in futures]
        policy = _mock_megatron_policy()
        gen = _mock_megatron_generation(refit_backend="nvshmem")
        sync = MegatronWeightSynchronizer(
            policy,
            gen,
            colocated=False,
            train_cluster=_mock_cluster(),
            inference_cluster=_mock_cluster(),
        )
        sync.init_communicator()
        sync.sync_weights()
        policy.preinit_nvshmem.assert_called_once()
        gen.preinit_nvshmem_collective.assert_called_once()

    @patch("nemo_rl.weight_sync.megatron_weight_synchronizer.ray")
    def test_non_colocated_failed_update_raises(self, mock_ray):
        # swap futures resolve fine; the inference-side results report failure
        mock_ray.get.side_effect = lambda futures: [False for _ in futures]
        policy = _mock_megatron_policy()
        gen = _mock_megatron_generation()
        sync = MegatronWeightSynchronizer(
            policy,
            gen,
            colocated=False,
            train_cluster=_mock_cluster(),
            inference_cluster=_mock_cluster(),
        )
        sync.init_communicator()
        with pytest.raises(RuntimeError):
            sync.sync_weights()
        assert sync.is_stale


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

    def test_colocated_sglang_returns_sglang_colocated(self):
        policy = _mock_policy()
        gen = _mock_generation()
        sync = create_weight_synchronizer(
            policy=policy,
            generation=gen,
            generation_backend=SGLANG_BACKEND,
            colocated=True,
        )
        assert isinstance(sync, SGLangColocatedWeightSynchronizer)

    def test_colocated_megatron_returns_megatron_synchronizer(self):
        policy = _mock_policy()
        gen = _mock_generation()
        sync = create_weight_synchronizer(
            policy=policy,
            generation=gen,
            generation_backend=MEGATRON_BACKEND,
            colocated=True,
        )
        assert isinstance(sync, MegatronWeightSynchronizer)

    def test_non_colocated_megatron_returns_megatron_synchronizer(self):
        policy = _mock_policy()
        gen = _mock_generation()
        sync = create_weight_synchronizer(
            policy=policy,
            generation=gen,
            generation_backend=MEGATRON_BACKEND,
            colocated=False,
            train_cluster=_mock_cluster(),
            inference_cluster=_mock_cluster(),
        )
        assert isinstance(sync, MegatronWeightSynchronizer)

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

    def test_non_colocated_dynamo_returns_collective(self):
        sync = create_weight_synchronizer(
            policy=_mock_policy(),
            generation=_mock_generation(),
            generation_backend=DYNAMO_BACKEND,
            colocated=False,
            train_cluster=_mock_cluster(),
            inference_cluster=_mock_cluster(),
        )
        assert isinstance(sync, CollectiveWeightSynchronizer)

    def test_non_colocated_sglang_returns_sglang_disaggregated(self):
        """SGLang owns its own weight-update group, so no clusters are needed."""
        policy = _megatron_policy()
        gen = _mock_generation()
        sync = create_weight_synchronizer(
            policy=policy,
            generation=gen,
            generation_backend=SGLANG_BACKEND,
            colocated=False,
        )
        assert isinstance(sync, SGLangDisaggregatedWeightSynchronizer)

    def test_non_colocated_sglang_rejects_dtensor_at_setup(self):
        with pytest.raises(
            NotImplementedError, match="Megatron policy backend.*issues/3745"
        ):
            create_weight_synchronizer(
                policy=_mock_policy(),
                generation=_mock_generation(),
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

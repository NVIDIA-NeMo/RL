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

"""Tests for checkpoint-engine weight synchronization and factory routing."""

from unittest.mock import MagicMock, patch

import pytest

from nemo_rl.models.generation.constants import (
    MEGATRON_BACKEND,
    SGLANG_BACKEND,
    VLLM_BACKEND,
)
from nemo_rl.weight_sync.checkpoint_engine_weight_synchronizer import (
    CheckpointEngineWeightSynchronizer,
    _ordered_generation_metadata,
    _sort_ranked_metadata,
    sync_weights_with_checkpoint_engine,
)
from nemo_rl.weight_sync.factory import create_weight_synchronizer


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


def _checkpoint_engine_cfg(
    *,
    release_after_refit=False,
    backend="test_backend",
    bucket_memory_ratio=0.05,
    device="cpu",
):
    return {
        "enabled": True,
        "backend": backend,
        "update_weights_bucket_memory_ratio": bucket_memory_ratio,
        "engine_kwargs": {
            backend: {
                "device": device,
                "release_after_refit": release_after_refit,
            }
        },
    }


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


def _checkpoint_sync(
    mock_ray,
    *,
    async_engine=False,
    update_success=True,
    release_after_refit=False,
    cycles=1,
    checkpoint_engine_config=None,
):
    # One return value per ray.get() call, in order:
    #   1. total GPU memory (policy + generation)
    #   2. init_checkpoint_engine (policy + generation)
    #   3. prepare_checkpoint_engine (policy refs first, then generation refs;
    #      _CheckpointWorkerGroup returns a single ref per side here)
    #   4. init_checkpoint_engine_process_group (policy + generation)
    #   5. send + update_weights_from_checkpoint_engine
    #   6. finalize_checkpoint_engine (policy + generation)
    mock_ray.get.side_effect = [[80 * 1024**3, [80 * 1024**3]]] + [
        item
        for _ in range(cycles)
        for item in (
            [],
            [["policy-0", "policy-1"], "generation-0", ["generation-1"]],
            [],
            ["policy-send", update_success],
            [],
        )
    ]
    policy = _mock_policy()
    policy.worker_group = _CheckpointWorkerGroup()
    checkpoint_engine_config = checkpoint_engine_config or _checkpoint_engine_cfg(
        release_after_refit=release_after_refit
    )
    gen = _mock_generation(
        cfg={
            "vllm_cfg": {"async_engine": async_engine},
            "checkpoint_engine": checkpoint_engine_config,
        }
    )
    gen.dp_size = 2
    gen.worker_group = _CheckpointWorkerGroup()
    return CheckpointEngineWeightSynchronizer(policy, gen, gen.cfg["checkpoint_engine"])


class TestCheckpointEngineWeightSynchronizer:
    @patch("nemo_rl.weight_sync.checkpoint_engine_weight_synchronizer.ray")
    def test_bucket_size_defaults_to_five_percent_of_total_memory(self, mock_ray):
        config = _checkpoint_engine_cfg()
        del config["update_weights_bucket_memory_ratio"]
        sync = _checkpoint_sync(mock_ray, checkpoint_engine_config=config)

        assert sync._resolve_bucket_size_bytes() == 4096 * 1024**2

    @patch("nemo_rl.weight_sync.checkpoint_engine_weight_synchronizer.ray")
    def test_bucket_uses_minimum_total_memory_and_is_cached(self, mock_ray, capsys):
        config = _checkpoint_engine_cfg(bucket_memory_ratio=0.125)
        sync = _checkpoint_sync(mock_ray, checkpoint_engine_config=config)
        mock_ray.get.side_effect = None
        mock_ray.get.return_value = [96 * 1024**3, [64 * 1024**3, 80 * 1024**3]]

        assert sync._resolve_bucket_size_bytes() == 8192 * 1024**2
        assert sync._resolve_bucket_size_bytes() == 8192 * 1024**2
        mock_ray.get.assert_called_once()
        assert sync._policy.worker_group.calls == [
            ("checkpoint_engine_rpc", "checkpoint_engine_total_memory_bytes")
        ]
        assert sync._generation.worker_group.calls == [
            ("checkpoint_engine_rpc", "checkpoint_engine_total_memory_bytes")
        ]
        assert "8192 MiB per buffer" in capsys.readouterr().out

    @pytest.mark.parametrize("memory_ratio", ["invalid", 0, 1])
    @patch("nemo_rl.weight_sync.checkpoint_engine_weight_synchronizer.ray")
    def test_bucket_rejects_invalid_ratio(self, mock_ray, memory_ratio):
        config = _checkpoint_engine_cfg(bucket_memory_ratio=memory_ratio)
        sync = _checkpoint_sync(mock_ray, checkpoint_engine_config=config)

        with pytest.raises(ValueError, match="update_weights_bucket_memory_ratio"):
            sync._resolve_bucket_size_bytes()
        mock_ray.get.assert_not_called()

    @patch("nemo_rl.weight_sync.checkpoint_engine_weight_synchronizer.ray")
    def test_bucket_rejects_sub_mibibyte_result(self, mock_ray):
        config = _checkpoint_engine_cfg(bucket_memory_ratio=0.05)
        sync = _checkpoint_sync(mock_ray, checkpoint_engine_config=config)
        mock_ray.get.side_effect = None
        mock_ray.get.return_value = [8 * 1024**2, [8 * 1024**2]]

        with pytest.raises(ValueError, match="less than 1 MiB"):
            sync._resolve_bucket_size_bytes()

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

    @patch("nemo_rl.weight_sync.checkpoint_engine_weight_synchronizer.ray")
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
        assert sync._generation.worker_group.calls[3][2] == [
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

    @patch("nemo_rl.weight_sync.checkpoint_engine_weight_synchronizer.ray")
    def test_release_after_refit_reprepares_each_sync(self, mock_ray):
        sync = _checkpoint_sync(mock_ray, release_after_refit=True, cycles=2)

        sync.init_communicator()
        sync.sync_weights()
        assert not sync._checkpoint_engine_ready

        sync.sync_weights()
        assert not sync._checkpoint_engine_ready
        assert (
            sync._policy.worker_group.calls.count(
                ("checkpoint_engine_rpc", "prepare_checkpoint_engine")
            )
            == 2
        )
        assert (
            sync._policy.worker_group.calls.count(
                ("checkpoint_engine_rpc", "finalize_checkpoint_engine")
            )
            == 2
        )

    @patch(
        "nemo_rl.weight_sync.checkpoint_engine_weight_synchronizer."
        "CheckpointEngineWeightSynchronizer"
    )
    def test_sync_helper_always_shuts_down(self, synchronizer_cls):
        policy = object()
        generation = _mock_generation(
            cfg={"checkpoint_engine": _checkpoint_engine_cfg()}
        )
        synchronizer = synchronizer_cls.return_value
        synchronizer.sync_weights.side_effect = RuntimeError("transfer failed")

        with pytest.raises(RuntimeError, match="transfer failed"):
            sync_weights_with_checkpoint_engine(
                policy, generation, kv_scales={"kv": 1.0}
            )

        synchronizer_cls.assert_called_once_with(
            policy, generation, generation.cfg["checkpoint_engine"]
        )
        synchronizer.sync_weights.assert_called_once_with(
            timer=None, kv_scales={"kv": 1.0}
        )
        synchronizer.shutdown.assert_called_once_with()

    @patch("nemo_rl.weight_sync.checkpoint_engine_weight_synchronizer.ray")
    def test_sync_weights_does_not_run_colocated_phase_transitions(self, mock_ray):
        sync = _checkpoint_sync(mock_ray)

        sync.init_communicator()
        sync.sync_weights()

        sync._policy.offload_before_refit.assert_not_called()
        sync._policy.offload_after_refit.assert_not_called()
        sync._policy.prepare_for_training.assert_not_called()
        sync._generation.prepare_for_generation.assert_not_called()

    @patch("nemo_rl.weight_sync.checkpoint_engine_weight_synchronizer.ray")
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


class TestCheckpointEngineFactory:
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

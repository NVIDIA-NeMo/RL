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
)
from nemo_rl.weight_sync.factory import create_weight_synchronizer


def _mock_policy(**overrides):
    policy = MagicMock()
    policy.offload_before_refit.return_value = None
    policy.offload_after_refit.return_value = None
    policy.prepare_refit_info.return_value = {"layer_0": {"shape": [4096, 4096]}}
    policy.stream_weights_via_ipc_zmq.return_value = [MagicMock()]
    policy.broadcast_weights_for_collective.return_value = [MagicMock()]
    policy.init_collective.return_value = [MagicMock()]
    policy.get_free_memory_bytes.return_value = 1024**3  # 1 GB
    for k, v in overrides.items():
        setattr(policy, k, v)
    return policy


def _mock_generation(**overrides):
    gen = MagicMock()
    gen.cfg = {}
    # Fault tolerance defaults off: a bare MagicMock would return a truthy mock
    # from .get("use_fault_tolerance") and fake-enable the recovery path.
    gen.sglang_cfg = {"sglang_cfg": {}}
    gen.prepare_for_generation.return_value = True
    gen.finish_generation.return_value = True
    gen.prepare_refit_info.return_value = None
    gen.update_weights_via_ipc_zmq.return_value = [MagicMock()]
    gen.update_weights_from_collective.return_value = [MagicMock()]
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
        "backend": backend,
        "update_weights_bucket_memory_ratio": bucket_memory_ratio,
        "engine_kwargs": {
            backend: {
                "device": device,
                "release_after_refit": release_after_refit,
            }
        },
    }


def _nixl_refit_cfg(*, release_after_refit=False):
    return {
        "backend": "vllm",
        "refit_transport": "nixl",
        "refit_cfg": {
            "nixl": {
                "device": "cpu",
                "release_after_refit": release_after_refit,
            }
        },
        "vllm_cfg": {"async_engine": False},
    }


def _sglang_refit_cfg(*, release_after_refit=False):
    """A config the SGLang guards accept, so each test can break one key."""
    cfg = _nixl_refit_cfg(release_after_refit=release_after_refit)
    cfg["backend"] = SGLANG_BACKEND
    cfg["sglang_cfg"] = {
        "dp_size": 1,
        "pp_size": 1,
        "quantization": {"scheme": "bf16"},
    }
    return cfg


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
            "backend": VLLM_BACKEND,
            "vllm_cfg": {"async_engine": async_engine},
        }
    )
    gen.dp_size = 2
    gen.worker_group = _CheckpointWorkerGroup()
    return CheckpointEngineWeightSynchronizer(policy, gen, checkpoint_engine_config)


class TestCheckpointEngineWeightSynchronizer:
    @patch("nemo_rl.weight_sync.checkpoint_engine_weight_synchronizer.ray")
    def test_sglang_init_skips_legacy_refit_metadata(self, _mock_ray):
        policy = _mock_policy()
        generation = _mock_generation(cfg={"backend": SGLANG_BACKEND})
        sync = CheckpointEngineWeightSynchronizer(
            policy, generation, _checkpoint_engine_cfg()
        )
        sync._checkpoint_engine_ready = True

        sync.init_communicator()

        policy.prepare_refit_info.assert_not_called()
        generation.prepare_refit_info.assert_not_called()

    @patch("nemo_rl.weight_sync.checkpoint_engine_weight_synchronizer.ray")
    def test_sglang_dispatches_directly_to_generation_actors(self, mock_ray):
        policy = _mock_policy()
        policy.worker_group = _CheckpointWorkerGroup()
        generation = _mock_generation(cfg={"backend": SGLANG_BACKEND})
        generation.run_checkpoint_engine_method.return_value = ["generation-update"]
        generation.prepare_for_generation.return_value = None
        sync = CheckpointEngineWeightSynchronizer(
            policy, generation, _checkpoint_engine_cfg()
        )
        sync._checkpoint_engine_ready = True
        mock_ray.get.return_value = ["policy-send", True]

        sync.sync_weights()

        generation.run_checkpoint_engine_method.assert_called_once_with(
            "update_weights_from_checkpoint_engine", ()
        )
        assert [
            item.kwargs for item in generation.prepare_for_generation.call_args_list
        ] == [
            {"tags": ["weights"]},
            {"tags": ["kv_cache"]},
        ]
        assert not sync.is_stale

    @patch("nemo_rl.weight_sync.checkpoint_engine_weight_synchronizer.ray")
    def test_sglang_wraps_the_transfer_in_a_weight_update_session(self, mock_ray):
        """SGLang gates ``update_weights_from_tensor`` on an open session.

        ``begin_weight_update`` sets the flag the server asserts on, and
        ``end_weight_update`` is what rebuilds quantized kernel layouts after
        the last bucket, so ordering is the contract -- not just presence.
        The pause matters for the same reason: the buckets arrive as several
        ``update_weights_from_tensor`` calls, each taking the server's model
        update lock on its own.
        """
        policy = _mock_policy()
        policy.worker_group = _CheckpointWorkerGroup()
        generation = _mock_generation(cfg={"backend": SGLANG_BACKEND})
        generation.run_checkpoint_engine_method.return_value = ["generation-update"]
        generation.prepare_for_generation.return_value = None
        generation.pause_generation_mode = "retract"
        generation.invalidate_kv_cache.return_value = True
        recorder = MagicMock()
        for name in (
            "prepare_for_generation",
            "pause_generation",
            "invalidate_kv_cache",
            "begin_weight_update",
            "run_checkpoint_engine_method",
            "end_weight_update",
            "continue_generation",
        ):
            recorder.attach_mock(getattr(generation, name), name)
        sync = CheckpointEngineWeightSynchronizer(
            policy, generation, _checkpoint_engine_cfg()
        )
        sync._checkpoint_engine_ready = True
        mock_ray.get.return_value = ["policy-send", True]

        sync.sync_weights()

        assert [name for name, _args, _kwargs in recorder.mock_calls] == [
            "prepare_for_generation",
            "pause_generation",
            "invalidate_kv_cache",
            "begin_weight_update",
            "run_checkpoint_engine_method",
            "end_weight_update",
            "prepare_for_generation",
            "continue_generation",
        ]
        generation.pause_generation.assert_called_once_with(mode="retract")

    @patch("nemo_rl.weight_sync.checkpoint_engine_weight_synchronizer.ray")
    def test_sglang_transfer_failure_is_terminal_and_never_resumes_serving(
        self, mock_ray
    ):
        """A failure after the transfer started leaves a half-updated model.

        Unlike a NCCL broadcast, interrupted one-sided NIXL work cannot be
        safely redone, so the synchronizer latches terminal: the session is
        closed (without masking the transfer error), serving and the health
        monitor stay paused, no shutdown/finalize runs, and every later sync
        raises immediately without issuing a single RPC.
        """
        policy = _mock_policy()
        policy.worker_group = _CheckpointWorkerGroup()
        generation = _mock_generation(cfg={"backend": SGLANG_BACKEND})
        generation.run_checkpoint_engine_method.return_value = ["generation-update"]
        generation.prepare_for_generation.return_value = None
        generation.pause_generation_mode = "retract"
        generation.invalidate_kv_cache.return_value = True
        sync = CheckpointEngineWeightSynchronizer(
            policy,
            generation,
            _checkpoint_engine_cfg(release_after_refit=True),
        )
        sync._checkpoint_engine_ready = True
        mock_ray.get.side_effect = RuntimeError("transport died")

        with pytest.raises(RuntimeError, match="transport died"):
            sync.sync_weights()

        generation.begin_weight_update.assert_called_once_with()
        # The session is still closed, with the close error suppressed if any.
        generation.end_weight_update.assert_called_once_with()
        # Terminal: no serving resume, no KV/monitor resume, no finalize --
        # even with release_after_refit=True, which normally shuts down in a
        # finally.
        generation.continue_generation.assert_not_called()
        assert [
            item.kwargs for item in generation.prepare_for_generation.call_args_list
        ] == [{"tags": ["weights"]}]
        assert (
            "checkpoint_engine_rpc",
            "finalize_checkpoint_engine",
        ) not in policy.worker_group.calls
        assert sync.is_stale

        # A later sync must reject up front with zero RPCs.
        mock_ray.reset_mock()
        generation.reset_mock()
        with pytest.raises(RuntimeError, match="terminal error state"):
            sync.sync_weights()
        mock_ray.get.assert_not_called()
        generation.prepare_for_generation.assert_not_called()
        generation.recover_updatable_engines.assert_not_called()
        generation.run_checkpoint_engine_method.assert_not_called()

    @patch("nemo_rl.weight_sync.checkpoint_engine_weight_synchronizer.ray")
    def test_sglang_end_failure_after_a_good_transfer_is_terminal(self, mock_ray):
        """A transfer that never finalized kernel layouts is unusable too."""
        policy = _mock_policy()
        policy.worker_group = _CheckpointWorkerGroup()
        generation = _mock_generation(cfg={"backend": SGLANG_BACKEND})
        generation.run_checkpoint_engine_method.return_value = ["generation-update"]
        generation.prepare_for_generation.return_value = None
        generation.pause_generation_mode = "retract"
        generation.invalidate_kv_cache.return_value = True
        generation.end_weight_update.side_effect = RuntimeError("finalize died")
        sync = CheckpointEngineWeightSynchronizer(
            policy, generation, _checkpoint_engine_cfg()
        )
        sync._checkpoint_engine_ready = True
        mock_ray.get.return_value = ["policy-send", True]

        with pytest.raises(RuntimeError, match="finalize died"):
            sync.sync_weights()

        generation.continue_generation.assert_not_called()
        assert sync._terminal_error is not None
        assert sync.is_stale

    @patch("nemo_rl.weight_sync.checkpoint_engine_weight_synchronizer.ray")
    def test_sglang_transfer_error_stays_primary_when_end_also_fails(self, mock_ray):
        """``finally`` alone would mask the transfer error with the end error."""
        policy = _mock_policy()
        policy.worker_group = _CheckpointWorkerGroup()
        generation = _mock_generation(cfg={"backend": SGLANG_BACKEND})
        generation.run_checkpoint_engine_method.return_value = ["generation-update"]
        generation.prepare_for_generation.return_value = None
        generation.pause_generation_mode = "retract"
        generation.invalidate_kv_cache.return_value = True
        generation.end_weight_update.side_effect = RuntimeError("close also died")
        sync = CheckpointEngineWeightSynchronizer(
            policy, generation, _checkpoint_engine_cfg()
        )
        sync._checkpoint_engine_ready = True
        mock_ray.get.side_effect = RuntimeError("transport died")

        with pytest.raises(RuntimeError, match="transport died"):
            sync.sync_weights()

    @patch("nemo_rl.weight_sync.checkpoint_engine_weight_synchronizer.ray")
    def test_sglang_pre_transfer_failure_is_not_terminal(self, mock_ray):
        """Before any bucket moved, nothing changed: cleanup, resume, retry."""
        policy = _mock_policy()
        policy.worker_group = _CheckpointWorkerGroup()
        generation = _mock_generation(cfg={"backend": SGLANG_BACKEND})
        generation.prepare_for_generation.return_value = None
        generation.pause_generation_mode = "retract"
        generation.pause_generation.side_effect = RuntimeError("pause failed")
        sync = CheckpointEngineWeightSynchronizer(
            policy, generation, _checkpoint_engine_cfg()
        )
        sync._checkpoint_engine_ready = True

        with pytest.raises(RuntimeError, match="pause failed"):
            sync.sync_weights()

        assert sync._terminal_error is None
        generation.continue_generation.assert_called_once_with()
        # Retryable: the next sync gets past the latch check.
        generation.pause_generation.side_effect = None
        generation.invalidate_kv_cache.return_value = True
        generation.run_checkpoint_engine_method.return_value = ["generation-update"]
        mock_ray.get.return_value = ["policy-send", True]
        sync.sync_weights()
        assert not sync.is_stale

    def test_initial_setup_consumes_startup_count_only_on_success(self):
        """#3613 reports the startup fleet through ``num_new_engines`` too.

        Consuming it after a successful setup stops the first ordinary refit
        from being misclassified as crash recovery; a failed setup must leave
        it pending (and, being a bind failure over non-transactional NIXL
        state, latch terminal).
        """
        policy = _mock_policy()
        generation = _mock_generation(cfg={"backend": SGLANG_BACKEND})
        sync = CheckpointEngineWeightSynchronizer(
            policy, generation, _checkpoint_engine_cfg()
        )
        with patch.object(sync, "_ensure_checkpoint_engine_ready") as ensure:
            sync._ensure_ready_and_consume_count()
        ensure.assert_called_once_with()
        generation.clear_updatable_num_new_engines.assert_called_once_with()

        failing = CheckpointEngineWeightSynchronizer(
            _mock_policy(),
            _mock_generation(cfg={"backend": SGLANG_BACKEND}),
            _checkpoint_engine_cfg(),
        )
        with patch.object(
            failing,
            "_ensure_checkpoint_engine_ready",
            side_effect=RuntimeError("bind failed"),
        ):
            with pytest.raises(RuntimeError, match="bind failed"):
                failing._ensure_ready_and_consume_count()
        failing._generation.clear_updatable_num_new_engines.assert_not_called()
        assert failing._terminal_error is not None

    def test_already_ready_setup_does_not_touch_the_count(self):
        """Steady state must not clear a count the dispatch has not consumed."""
        sync = CheckpointEngineWeightSynchronizer(
            _mock_policy(),
            _mock_generation(cfg={"backend": SGLANG_BACKEND}),
            _checkpoint_engine_cfg(),
        )
        sync._checkpoint_engine_ready = True
        sync._ensure_ready_and_consume_count()
        sync._generation.clear_updatable_num_new_engines.assert_not_called()

    @patch("nemo_rl.weight_sync.checkpoint_engine_weight_synchronizer.ray")
    def test_sglang_recovery_rebinds_without_destroying_engines(self, mock_ray):
        """recover -> mark not-ready -> reinit -> clear count -> transfer."""
        policy = _mock_policy()
        policy.worker_group = _CheckpointWorkerGroup()
        generation = _mock_generation(cfg={"backend": SGLANG_BACKEND})
        generation.sglang_cfg = {"sglang_cfg": {"use_fault_tolerance": True}}
        generation.run_checkpoint_engine_method.return_value = ["generation-update"]
        generation.prepare_for_generation.return_value = None
        generation.pause_generation_mode = "retract"
        generation.invalidate_kv_cache.return_value = True
        generation.get_updatable_engines_and_lock.return_value = (
            ["engine-0", "engine-1"],
            object(),
            1,
            [1, 1],
            [0, 1],
        )
        sync = CheckpointEngineWeightSynchronizer(
            policy, generation, _checkpoint_engine_cfg()
        )
        sync._checkpoint_engine_ready = True
        mock_ray.get.return_value = ["policy-send", True]

        order = []
        generation.recover_updatable_engines.side_effect = lambda: order.append(
            "recover"
        )
        generation.clear_updatable_num_new_engines.side_effect = lambda: order.append(
            "clear"
        )
        with patch.object(
            sync,
            "_ensure_checkpoint_engine_ready",
            side_effect=lambda: order.append("reinit"),
        ):
            sync.sync_weights()

        # Readiness was invalidated (reinit ran) and the count was consumed
        # only after the rebind, before the transfer.
        assert order == ["recover", "reinit", "clear"]
        # Rebind reuses engine objects: nothing resets or finalizes them.
        assert not any(
            call.args and call.args[0] == "finalize_checkpoint_engine"
            for call in generation.run_checkpoint_engine_method.call_args_list
        )
        assert not sync.is_stale

    @patch("nemo_rl.weight_sync.checkpoint_engine_weight_synchronizer.ray")
    def test_sglang_steady_state_probes_but_skips_the_rebind(self, mock_ray):
        """FT must probe every refit; zero replacements means no rebind work."""
        policy = _mock_policy()
        policy.worker_group = _CheckpointWorkerGroup()
        generation = _mock_generation(cfg={"backend": SGLANG_BACKEND})
        generation.sglang_cfg = {"sglang_cfg": {"use_fault_tolerance": True}}
        generation.run_checkpoint_engine_method.return_value = ["generation-update"]
        generation.prepare_for_generation.return_value = None
        generation.pause_generation_mode = "retract"
        generation.invalidate_kv_cache.return_value = True
        generation.get_updatable_engines_and_lock.return_value = (
            ["engine-0"],
            object(),
            0,
            [1],
            [0],
        )
        sync = CheckpointEngineWeightSynchronizer(
            policy, generation, _checkpoint_engine_cfg()
        )
        sync._checkpoint_engine_ready = True
        mock_ray.get.return_value = ["policy-send", True]

        sync.sync_weights()

        generation.recover_updatable_engines.assert_called_once_with()
        assert sync._checkpoint_engine_ready
        generation.clear_updatable_num_new_engines.assert_not_called()
        assert not sync.is_stale

    @patch("nemo_rl.weight_sync.checkpoint_engine_weight_synchronizer.ray")
    def test_sglang_fault_tolerance_off_never_probes(self, mock_ray):
        policy = _mock_policy()
        policy.worker_group = _CheckpointWorkerGroup()
        generation = _mock_generation(cfg={"backend": SGLANG_BACKEND})
        generation.run_checkpoint_engine_method.return_value = ["generation-update"]
        generation.prepare_for_generation.return_value = None
        generation.pause_generation_mode = "retract"
        generation.invalidate_kv_cache.return_value = True
        sync = CheckpointEngineWeightSynchronizer(
            policy, generation, _checkpoint_engine_cfg()
        )
        sync._checkpoint_engine_ready = True
        mock_ray.get.return_value = ["policy-send", True]

        sync.sync_weights()

        generation.recover_updatable_engines.assert_not_called()

    @patch("nemo_rl.weight_sync.checkpoint_engine_weight_synchronizer.ray")
    def test_sglang_recovery_failure_is_retryable_but_rollback_failure_is_not(
        self, mock_ray
    ):
        """Cohort rollback keeps recovery retryable; a failed rollback latches."""
        from nemo_rl.models.generation.sglang.fault_tolerance import (
            RecoveryRollbackError,
        )

        policy = _mock_policy()
        policy.worker_group = _CheckpointWorkerGroup()
        generation = _mock_generation(cfg={"backend": SGLANG_BACKEND})
        generation.sglang_cfg = {"sglang_cfg": {"use_fault_tolerance": True}}
        sync = CheckpointEngineWeightSynchronizer(
            policy, generation, _checkpoint_engine_cfg()
        )
        sync._checkpoint_engine_ready = True

        # Plain recovery failure: rolled back inside _recover -> retryable.
        generation.recover_updatable_engines.side_effect = RuntimeError(
            "replacement init died"
        )
        with pytest.raises(RuntimeError, match="replacement init died"):
            sync.sync_weights()
        assert sync._terminal_error is None
        with pytest.raises(RuntimeError, match="replacement init died"):
            sync.sync_weights()
        assert generation.recover_updatable_engines.call_count == 2

        # Rollback failure: engine state is inconsistent -> terminal latch.
        generation.recover_updatable_engines.side_effect = RecoveryRollbackError(
            "rollback failed"
        )
        with pytest.raises(RecoveryRollbackError):
            sync.sync_weights()
        assert sync._terminal_error is not None
        generation.reset_mock()
        with pytest.raises(RuntimeError, match="terminal error state"):
            sync.sync_weights()
        generation.recover_updatable_engines.assert_not_called()

    @patch("nemo_rl.weight_sync.checkpoint_engine_weight_synchronizer.ray")
    def test_sglang_does_not_open_a_session_when_the_kv_flush_fails(self, mock_ray):
        """``end_weight_update`` only closes a session that actually opened."""
        policy = _mock_policy()
        policy.worker_group = _CheckpointWorkerGroup()
        generation = _mock_generation(cfg={"backend": SGLANG_BACKEND})
        generation.prepare_for_generation.return_value = None
        generation.pause_generation_mode = "retract"
        generation.invalidate_kv_cache.return_value = False
        sync = CheckpointEngineWeightSynchronizer(
            policy, generation, _checkpoint_engine_cfg()
        )
        sync._checkpoint_engine_ready = True

        with pytest.raises(RuntimeError, match="KV cache invalidation failed"):
            sync.sync_weights()

        generation.begin_weight_update.assert_not_called()
        generation.end_weight_update.assert_not_called()
        generation.continue_generation.assert_called_once_with()

    @patch("nemo_rl.weight_sync.checkpoint_engine_weight_synchronizer.ray")
    def test_sglang_resumes_when_opening_the_session_itself_fails(self, mock_ray):
        """``end_weight_update`` must not close a session that never opened."""
        policy = _mock_policy()
        policy.worker_group = _CheckpointWorkerGroup()
        generation = _mock_generation(cfg={"backend": SGLANG_BACKEND})
        generation.prepare_for_generation.return_value = None
        generation.pause_generation_mode = "retract"
        generation.invalidate_kv_cache.return_value = True
        generation.begin_weight_update.side_effect = RuntimeError("engine refused")
        sync = CheckpointEngineWeightSynchronizer(
            policy, generation, _checkpoint_engine_cfg()
        )
        sync._checkpoint_engine_ready = True

        with pytest.raises(RuntimeError, match="engine refused"):
            sync.sync_weights()

        generation.end_weight_update.assert_not_called()
        generation.continue_generation.assert_called_once_with()

    @patch("nemo_rl.weight_sync.checkpoint_engine_weight_synchronizer.ray")
    def test_sglang_resumes_when_the_pause_itself_fails(self, mock_ray):
        """A pause that raises must not leave the engines half-paused."""
        policy = _mock_policy()
        policy.worker_group = _CheckpointWorkerGroup()
        generation = _mock_generation(cfg={"backend": SGLANG_BACKEND})
        generation.prepare_for_generation.return_value = None
        generation.pause_generation_mode = "retract"
        generation.pause_generation.side_effect = RuntimeError("pause failed")
        sync = CheckpointEngineWeightSynchronizer(
            policy, generation, _checkpoint_engine_cfg()
        )
        sync._checkpoint_engine_ready = True

        with pytest.raises(RuntimeError, match="pause failed"):
            sync.sync_weights()

        generation.begin_weight_update.assert_not_called()
        generation.end_weight_update.assert_not_called()
        generation.continue_generation.assert_called_once_with()

    @patch("nemo_rl.weight_sync.checkpoint_engine_weight_synchronizer.ray")
    def test_vllm_does_not_run_the_sglang_session(self, mock_ray):
        """The envelope is SGLang's contract; vLLM has no such concept."""
        sync = _checkpoint_sync(mock_ray)
        sync._checkpoint_engine_ready = True
        mock_ray.get.side_effect = None
        mock_ray.get.return_value = ["policy-send", True]

        sync.sync_weights()

        sync._generation.begin_weight_update.assert_not_called()
        sync._generation.pause_generation.assert_not_called()

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
            (SGLANG_BACKEND, False, CheckpointEngineWeightSynchronizer),
            (MEGATRON_BACKEND, False, NotImplementedError),
        ],
    )
    def test_checkpoint_engine_factory_routing(self, backend, colocated, expected):
        policy = _mock_policy(cfg={})
        gen = _mock_generation(
            cfg=_sglang_refit_cfg() if backend == SGLANG_BACKEND else _nixl_refit_cfg()
        )
        gen.pause_generation_mode = "retract"
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
        gen = _mock_generation(cfg=_nixl_refit_cfg())
        assert isinstance(
            create_weight_synchronizer(
                policy=_mock_policy(cfg=cfg),
                generation=gen,
                generation_backend=VLLM_BACKEND,
                colocated=False,
            ),
            CheckpointEngineWeightSynchronizer,
        )

    def test_checkpoint_engine_rejects_sglang_sharded_experts(self):
        gen = _mock_generation(cfg=_sglang_refit_cfg())
        gen.pause_generation_mode = "retract"
        gen.cfg["refit_cfg"]["nixl"]["shard_expert_weights"] = True

        with pytest.raises(NotImplementedError, match="shard_expert_weights"):
            create_weight_synchronizer(
                policy=_mock_policy(cfg={}),
                generation=gen,
                generation_backend=SGLANG_BACKEND,
                colocated=False,
            )

    def test_checkpoint_engine_rejects_sglang_data_parallelism(self):
        gen = _mock_generation(cfg=_sglang_refit_cfg())
        gen.pause_generation_mode = "retract"
        gen.cfg["sglang_cfg"]["dp_size"] = 2

        with pytest.raises(NotImplementedError, match="dp_size=1"):
            create_weight_synchronizer(
                policy=_mock_policy(cfg={}),
                generation=gen,
                generation_backend=SGLANG_BACKEND,
                colocated=False,
            )

    def test_checkpoint_engine_rejects_sglang_in_place_pause(self):
        """``in_place`` keeps KV entries produced by the outgoing weights.

        The sibling SGLang synchronizer rejects it for the same reason; the
        checkpoint-engine transport pauses through the same hook, so it needs
        the same guard.
        """
        gen = _mock_generation(cfg=_sglang_refit_cfg())
        gen.pause_generation_mode = "in_place"

        with pytest.raises(ValueError, match="in_place"):
            create_weight_synchronizer(
                policy=_mock_policy(cfg={}),
                generation=gen,
                generation_backend=SGLANG_BACKEND,
                colocated=False,
            )

    def test_checkpoint_engine_in_place_guard_is_sglang_only(self):
        """``pause_generation_mode`` is an SGLang concept; vLLM has no pause."""
        gen = _mock_generation(cfg=_nixl_refit_cfg())
        gen.pause_generation_mode = "in_place"

        assert isinstance(
            create_weight_synchronizer(
                policy=_mock_policy(cfg={}),
                generation=gen,
                generation_backend=VLLM_BACKEND,
                colocated=False,
            ),
            CheckpointEngineWeightSynchronizer,
        )

    def test_checkpoint_engine_rejects_sglang_pipeline_parallelism(self):
        """One receiver is created per engine GPU, but SGLang indexes the
        payload list by TP rank, and pp_size>1 makes those counts differ."""
        gen = _mock_generation(cfg=_sglang_refit_cfg())
        gen.pause_generation_mode = "retract"
        gen.cfg["sglang_cfg"]["pp_size"] = 2

        with pytest.raises(NotImplementedError, match="pp_size=1"):
            create_weight_synchronizer(
                policy=_mock_policy(cfg={}),
                generation=gen,
                generation_backend=SGLANG_BACKEND,
                colocated=False,
            )

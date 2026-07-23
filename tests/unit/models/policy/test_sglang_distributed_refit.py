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

from datetime import timedelta
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from nemo_rl.models.policy import utils as policy_utils


class _RemoteMethod:
    def __init__(self, result=None):
        self.calls = []
        self.result = result

    def remote(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result


class _FakeDeadline:
    def __init__(self):
        self.remaining_calls = []
        self.ray_get_calls = []
        self._budget = 20.0

    def remaining(self, stage):
        self.remaining_calls.append(stage)
        self._budget -= 1.0
        return self._budget

    def remaining_or_zero(self):
        return max(0.0, self._budget)

    def remaining_timedelta(self, stage):
        return timedelta(seconds=self.remaining(stage))

    def ray_get(self, refs, *, stage, cancel_on_error=False):
        self.ray_get_calls.append((refs, stage, cancel_on_error))
        return refs


class _FakeEngine:
    def __init__(self):
        self.init_weights_update_group = _RemoteMethod({"success": True})
        self.destroy_weights_update_group = _RemoteMethod({"success": True})
        self.update_weights_from_distributed = _RemoteMethod({"success": True})


class _FakeLeaseLock:
    def __init__(self):
        self.acquire = _RemoteMethod(True)
        self.release = _RemoteMethod(True)


class _FakeWork:
    def __init__(self, *, error=None):
        self.error = error
        self.timeouts = []

    def wait(self, *, timeout):
        self.timeouts.append(timeout)
        if self.error is not None:
            raise self.error
        return True


def test_fetch_recovery_failure_retains_monitor_lease():
    generation = SimpleNamespace(
        sglang_cfg={"sglang_cfg": {"use_fault_tolerance": True}},
        health_monitoring_suspend_for_refit=Mock(return_value="refit:lease"),
        recover_updatable_engines=Mock(side_effect=RuntimeError("recover failed")),
        health_monitoring_release_refit=Mock(),
    )

    with pytest.raises(RuntimeError, match="recover failed") as exc_info:
        policy_utils.fetch_updatable_engines_with_recover(generation)

    generation.health_monitoring_release_refit.assert_not_called()
    assert any(
        "health monitoring remains suspended" in note
        for note in exc_info.value.__notes__
    )


def test_connect_bounds_bootstrap_and_prints_ready_only_after_success(
    monkeypatch, capsys
):
    engine = _FakeEngine()
    group = object()
    deadline = _FakeDeadline()
    monkeypatch.setattr(
        "ray._private.services.get_node_ip_address",
        lambda: "127.0.0.1",
    )
    monkeypatch.setattr(
        "nemo_rl.distributed.virtual_cluster._get_free_port_local",
        lambda: 12345,
    )
    init = Mock(return_value=group)
    monkeypatch.setattr(policy_utils, "init_process_group", init)

    result = policy_utils.connect_rollout_engines_from_distributed(
        group_name="test_group",
        rollout_engines=[engine],
        engine_gpu_counts=[2],
        deadline=deadline,
    )

    assert result is group
    assert (
        "NRL_SGLANG_REFIT_GROUP_READY world_size=3 engines=1" in capsys.readouterr().out
    )
    assert engine.init_weights_update_group.calls[0][1]["timeout_s"] < 20
    assert init.call_args.kwargs["timeout"] < timedelta(seconds=20)
    assert deadline.ray_get_calls[-1][2] is True


def test_connect_failure_aborts_trainer_and_queues_engine_reset(monkeypatch):
    engine = _FakeEngine()
    group = object()

    class _FailingDeadline(_FakeDeadline):
        def ray_get(self, refs, *, stage, cancel_on_error=False):
            if "engine communicator bootstrap" in stage:
                raise RuntimeError("engine init failed")
            return super().ray_get(
                refs,
                stage=stage,
                cancel_on_error=cancel_on_error,
            )

    deadline = _FailingDeadline()
    monkeypatch.setattr(
        "ray._private.services.get_node_ip_address",
        lambda: "127.0.0.1",
    )
    monkeypatch.setattr(
        "nemo_rl.distributed.virtual_cluster._get_free_port_local",
        lambda: 12345,
    )
    monkeypatch.setattr(policy_utils, "init_process_group", Mock(return_value=group))
    abort = Mock()
    monkeypatch.setattr(policy_utils, "_abort_process_group", abort)
    monkeypatch.setattr(policy_utils, "cancel_ray_refs", Mock())

    with pytest.raises(RuntimeError, match="engine init failed"):
        policy_utils.connect_rollout_engines_from_distributed(
            group_name="test_group",
            rollout_engines=[engine],
            engine_gpu_counts=[2],
            deadline=deadline,
        )

    abort.assert_called_once_with(group)
    assert engine.destroy_weights_update_group.calls
    assert (
        engine.destroy_weights_update_group.calls[0][1]["timeout_s"]
        >= policy_utils._BEST_EFFORT_REFIT_CLEANUP_TIMEOUT_S
    )


def test_bucket_broadcast_uses_one_budget_and_matching_lock_lease(monkeypatch):
    engine = _FakeEngine()
    lock = _FakeLeaseLock()
    deadline = _FakeDeadline()
    work = _FakeWork()
    monkeypatch.setattr(policy_utils.dist, "broadcast", Mock(return_value=work))

    policy_utils.broadcast_hf_buckets_via_distributed_impl(
        bucket_iterator=[[("weight", torch.ones(1))]],
        rollout_engines=[engine],
        rollout_engine_lock=lock,
        group_name="test_group",
        model_update_group=object(),
        weight_version=3,
        deadline=deadline,
    )

    acquire_owner = lock.acquire.calls[0][0][0]
    release_owner = lock.release.calls[0][0][0]
    assert acquire_owner == release_owner
    assert engine.update_weights_from_distributed.calls[0][1]["timeout_s"] < 20
    assert work.timeouts and work.timeouts[0] < timedelta(seconds=20)
    assert any(
        "waiting for engine receives" in stage for _, stage, _ in deadline.ray_get_calls
    )


def test_bucket_failure_still_queues_matching_lock_release(monkeypatch):
    engine = _FakeEngine()
    lock = _FakeLeaseLock()
    deadline = _FakeDeadline()
    work = _FakeWork(error=TimeoutError("collective timed out"))
    cancel = Mock()
    monkeypatch.setattr(policy_utils.dist, "broadcast", Mock(return_value=work))
    monkeypatch.setattr(policy_utils, "cancel_ray_refs", cancel)

    with pytest.raises(TimeoutError, match="collective timed out"):
        policy_utils.broadcast_hf_buckets_via_distributed_impl(
            bucket_iterator=[[("weight", torch.ones(1))]],
            rollout_engines=[engine],
            rollout_engine_lock=lock,
            group_name="test_group",
            model_update_group=object(),
            weight_version=3,
            deadline=deadline,
        )

    assert lock.release.calls[0][0][0] == lock.acquire.calls[0][0][0]
    cancel.assert_called_once()


@pytest.mark.mcore
def test_distributed_driver_quarantines_partial_update_without_resuming(
    monkeypatch,
):
    from nemo_rl.models.policy.workers import megatron_policy_worker as worker_module

    generation = SimpleNamespace(
        refit_timeout_s=30.0,
        pause_generation_mode="retract",
        num_new_engines=0,
        pause_generation=Mock(),
        invalidate_kv_cache=Mock(return_value=True),
        post_process_weights=Mock(),
        continue_generation=Mock(),
        quarantine_all_engines=Mock(return_value=True),
        health_monitoring_release_refit=Mock(),
    )
    policy = SimpleNamespace(
        connect_sglang_rollout_engines_distributed=Mock(),
        update_weights_to_sglang_distributed=Mock(return_value=["transfer"]),
        abort_sglang_rollout_engines_distributed=Mock(return_value=["cleanup"]),
    )
    monkeypatch.setattr(
        policy_utils,
        "fetch_updatable_engines_with_recover",
        Mock(
            return_value=(
                ["engine"],
                "lock",
                0,
                [1],
                [0],
                "refit:lease",
            )
        ),
    )

    transfer_deadline = None
    cleanup_deadline = None

    def fake_ray_get(self, refs, *, stage, cancel_on_error=False):
        nonlocal transfer_deadline, cleanup_deadline
        if stage == "waiting for SGLang weight transfer":
            transfer_deadline = self
            raise RuntimeError("transfer failed")
        if stage == "waiting for SGLang communicator cleanup":
            cleanup_deadline = self
            return None
        raise AssertionError(f"unexpected driver wait: {stage}")

    monkeypatch.setattr(
        worker_module.SGLangRefitDeadline,
        "ray_get",
        fake_ray_get,
    )
    monkeypatch.setattr(policy_utils, "cancel_ray_refs", Mock())

    with pytest.raises(RuntimeError, match="transfer failed") as exc_info:
        worker_module.refit_sglang_distributed(
            policy=policy,
            policy_generation=generation,
            buffer_size_bytes=1024,
        )

    policy.connect_sglang_rollout_engines_distributed.assert_called_once()
    policy.abort_sglang_rollout_engines_distributed.assert_called_once()
    assert transfer_deadline is not None
    assert cleanup_deadline is not None
    assert cleanup_deadline is not transfer_deadline
    generation.quarantine_all_engines.assert_called_once()
    generation.continue_generation.assert_not_called()
    generation.health_monitoring_release_refit.assert_not_called()
    assert any(
        "weight stream may have been applied only partially" in note
        for note in exc_info.value.__notes__
    )


@pytest.mark.mcore
def test_distributed_driver_quarantines_keyboard_interrupt_during_transfer(
    monkeypatch,
):
    from nemo_rl.models.policy.workers import megatron_policy_worker as worker_module

    generation = SimpleNamespace(
        refit_timeout_s=30.0,
        pause_generation_mode="retract",
        num_new_engines=0,
        pause_generation=Mock(),
        invalidate_kv_cache=Mock(return_value=True),
        post_process_weights=Mock(),
        continue_generation=Mock(),
        quarantine_all_engines=Mock(return_value=True),
        health_monitoring_release_refit=Mock(),
    )
    policy = SimpleNamespace(
        connect_sglang_rollout_engines_distributed=Mock(),
        update_weights_to_sglang_distributed=Mock(return_value=["transfer"]),
        abort_sglang_rollout_engines_distributed=Mock(return_value=["cleanup"]),
    )
    monkeypatch.setattr(
        policy_utils,
        "fetch_updatable_engines_with_recover",
        Mock(
            return_value=(
                ["engine"],
                "lock",
                0,
                [1],
                [0],
                "refit:lease",
            )
        ),
    )

    def fake_ray_get(self, refs, *, stage, cancel_on_error=False):
        if stage == "waiting for SGLang weight transfer":
            raise KeyboardInterrupt
        if stage == "waiting for SGLang communicator cleanup":
            return None
        raise AssertionError(f"unexpected driver wait: {stage}")

    monkeypatch.setattr(worker_module.SGLangRefitDeadline, "ray_get", fake_ray_get)
    monkeypatch.setattr(policy_utils, "cancel_ray_refs", Mock())

    with pytest.raises(KeyboardInterrupt):
        worker_module.refit_sglang_distributed(
            policy=policy,
            policy_generation=generation,
            buffer_size_bytes=1024,
        )

    generation.quarantine_all_engines.assert_called_once()
    generation.continue_generation.assert_not_called()
    generation.health_monitoring_release_refit.assert_not_called()


@pytest.mark.mcore
def test_distributed_driver_quarantines_unconfirmed_bootstrap_cleanup(monkeypatch):
    from nemo_rl.models.policy.workers import megatron_policy_worker as worker_module

    generation = SimpleNamespace(
        refit_timeout_s=30.0,
        pause_generation_mode="retract",
        num_new_engines=0,
        pause_generation=Mock(),
        invalidate_kv_cache=Mock(return_value=True),
        post_process_weights=Mock(),
        continue_generation=Mock(),
        quarantine_all_engines=Mock(return_value=True),
        health_monitoring_release_refit=Mock(),
    )
    policy = SimpleNamespace(
        connect_sglang_rollout_engines_distributed=Mock(
            side_effect=RuntimeError("bootstrap timed out")
        ),
        update_weights_to_sglang_distributed=Mock(),
        abort_sglang_rollout_engines_distributed=Mock(return_value=["cleanup"]),
    )
    monkeypatch.setattr(
        policy_utils,
        "fetch_updatable_engines_with_recover",
        Mock(
            return_value=(
                ["engine"],
                "lock",
                0,
                [1],
                [0],
                "refit:lease",
            )
        ),
    )

    def fail_cleanup(self, refs, *, stage, cancel_on_error=False):
        assert stage == "waiting for SGLang communicator cleanup"
        raise TimeoutError("cleanup timed out")

    monkeypatch.setattr(worker_module.SGLangRefitDeadline, "ray_get", fail_cleanup)

    with pytest.raises(RuntimeError, match="bootstrap timed out") as exc_info:
        worker_module.refit_sglang_distributed(
            policy=policy,
            policy_generation=generation,
            buffer_size_bytes=1024,
        )

    generation.pause_generation.assert_not_called()
    generation.quarantine_all_engines.assert_called_once()
    generation.continue_generation.assert_not_called()
    generation.health_monitoring_release_refit.assert_not_called()
    assert any(
        "bootstrap cleanup was not confirmed" in note
        for note in exc_info.value.__notes__
    )


@pytest.mark.mcore
def test_distributed_driver_resumes_only_after_complete_refit(monkeypatch):
    from nemo_rl.models.policy.workers import megatron_policy_worker as worker_module

    generation = SimpleNamespace(
        refit_timeout_s=30.0,
        pause_generation_mode="retract",
        num_new_engines=0,
        pause_generation=Mock(),
        invalidate_kv_cache=Mock(return_value=True),
        post_process_weights=Mock(),
        continue_generation=Mock(),
        quarantine_all_engines=Mock(return_value=True),
        health_monitoring_release_refit=Mock(),
    )
    policy = SimpleNamespace(
        connect_sglang_rollout_engines_distributed=Mock(),
        update_weights_to_sglang_distributed=Mock(return_value=["transfer"]),
        abort_sglang_rollout_engines_distributed=Mock(),
    )
    monkeypatch.setattr(
        policy_utils,
        "fetch_updatable_engines_with_recover",
        Mock(
            return_value=(
                ["engine"],
                "lock",
                0,
                [1],
                [0],
                "refit:lease",
            )
        ),
    )
    monkeypatch.setattr(
        worker_module.SGLangRefitDeadline,
        "ray_get",
        lambda self, refs, *, stage, cancel_on_error=False: refs,
    )

    assert worker_module.refit_sglang_distributed(
        policy=policy,
        policy_generation=generation,
        buffer_size_bytes=1024,
    )

    generation.post_process_weights.assert_called_once()
    generation.continue_generation.assert_called_once()
    generation.health_monitoring_release_refit.assert_called_once_with("refit:lease")
    generation.quarantine_all_engines.assert_not_called()
    policy.abort_sglang_rollout_engines_distributed.assert_not_called()


@pytest.mark.mcore
def test_worker_broadcast_failure_passes_timeout_to_internal_abort(monkeypatch):
    from nemo_rl.models.policy.workers import megatron_policy_worker as worker_module

    worker = object.__new__(worker_module.MegatronPolicyWorkerImpl)
    worker.rank = 0
    worker._sglang_dist_group = object()
    worker._sglang_dist_group_name = "test_group"
    worker._sglang_dist_engines = ["engine"]
    worker._sglang_weight_version = 0
    iterator = SimpleNamespace(iter_hf_weight_buckets=Mock(return_value=iter([])))
    worker._build_sglang_hf_iterator = Mock(return_value=iterator)
    worker.abort_sglang_rollout_engines_distributed = Mock()
    monkeypatch.setattr(
        worker_module,
        "broadcast_hf_buckets_via_distributed_impl",
        Mock(side_effect=RuntimeError("broadcast failed")),
    )
    monkeypatch.setattr(worker_module.torch.cuda.nvtx, "range_push", Mock())
    monkeypatch.setattr(worker_module.torch.cuda.nvtx, "range_pop", Mock())

    with pytest.raises(RuntimeError, match="broadcast failed"):
        worker.update_weights_to_sglang_distributed(
            rollout_engines=["engine"],
            rollout_engine_lock="lock",
            buffer_size_bytes=1024,
            timeout_s=30,
        )

    cleanup_timeout_s = (
        worker.abort_sglang_rollout_engines_distributed.call_args.kwargs["timeout_s"]
    )
    assert (
        cleanup_timeout_s >= worker_module._BEST_EFFORT_SGLANG_REFIT_CLEANUP_TIMEOUT_S
    )


@pytest.mark.mcore
@pytest.mark.parametrize(
    "module_name",
    [
        "nemo_rl.models.policy.workers.megatron_policy_worker",
        "nemo_rl.models.policy.workers.dtensor_policy_worker_v2",
    ],
)
def test_colocated_in_place_refit_preserves_kv_cache(
    monkeypatch,
    module_name,
):
    worker_module = import_module(module_name)
    generation = SimpleNamespace(
        sglang_cfg={"sglang_cfg": {}},
        pause_generation_mode="in_place",
        num_new_engines=0,
        pause_generation=Mock(),
        invalidate_kv_cache=Mock(return_value=True),
        post_process_weights=Mock(),
        continue_generation=Mock(),
        health_monitoring_release_refit=Mock(),
    )
    policy = SimpleNamespace(
        update_weights_to_sglang_colocated=Mock(return_value=[]),
    )
    monkeypatch.setattr(
        policy_utils,
        "fetch_updatable_engines_with_recover",
        Mock(
            return_value=(
                ["engine"],
                "lock",
                0,
                [1],
                [0],
                "refit:lease",
            )
        ),
    )
    monkeypatch.setattr(worker_module.ray, "get", lambda _refs: [])

    assert worker_module.refit_sglang_colocated(
        policy=policy,
        policy_generation=generation,
        buffer_size_bytes=1024,
    )

    generation.invalidate_kv_cache.assert_not_called()
    generation.continue_generation.assert_called_once()
    generation.health_monitoring_release_refit.assert_called_once_with("refit:lease")


@pytest.mark.mcore
@pytest.mark.parametrize(
    "module_name",
    [
        "nemo_rl.models.policy.workers.megatron_policy_worker",
        "nemo_rl.models.policy.workers.dtensor_policy_worker_v2",
    ],
)
def test_colocated_partial_update_is_quarantined_without_resume(
    monkeypatch,
    module_name,
):
    worker_module = import_module(module_name)
    generation = SimpleNamespace(
        sglang_cfg={
            "sglang_cfg": {
                "quantization": {
                    "scheme": "bf16",
                }
            }
        },
        pause_generation_mode="retract",
        num_new_engines=0,
        pause_generation=Mock(),
        invalidate_kv_cache=Mock(return_value=True),
        post_process_weights=Mock(),
        continue_generation=Mock(),
        quarantine_all_engines=Mock(return_value=True),
        health_monitoring_release_refit=Mock(),
    )
    policy = SimpleNamespace(
        update_weights_to_sglang_colocated=Mock(return_value=["transfer"]),
    )
    monkeypatch.setattr(
        policy_utils,
        "fetch_updatable_engines_with_recover",
        Mock(
            return_value=(
                ["engine"],
                "lock",
                0,
                [1],
                [0],
                "refit:lease",
            )
        ),
    )
    monkeypatch.setattr(
        worker_module.ray,
        "get",
        Mock(side_effect=RuntimeError("partial IPC update")),
    )
    monkeypatch.setattr(policy_utils, "cancel_ray_refs", Mock())

    with pytest.raises(RuntimeError, match="partial IPC update") as exc_info:
        worker_module.refit_sglang_colocated(
            policy=policy,
            policy_generation=generation,
            buffer_size_bytes=1024,
        )

    generation.quarantine_all_engines.assert_called_once()
    generation.continue_generation.assert_not_called()
    generation.health_monitoring_release_refit.assert_not_called()
    assert any(
        "colocated SGLang weight stream may have been applied only partially" in note
        for note in exc_info.value.__notes__
    )

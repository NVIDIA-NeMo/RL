# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only coverage of the four-phase worker RPC and async epoch contract."""

import ast
import asyncio
import importlib.util
import sys
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "four_phase_profiling", ROOT / "nemo_rl/models/four_phase_profiling.py"
)
profiling = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(profiling)


@pytest.fixture(autouse=True)
def mode(monkeypatch):
    monkeypatch.setenv("NRL_NTRACE_FOUR_PHASE", "1")
    monkeypatch.setenv("NRL_POLICY_PROFILER_CLASS", "ntrace.NemoRLTraceController")
    monkeypatch.setenv(
        "NRL_ROLLOUT_PROFILER_CLASS", "ntrace.NemoRLRolloutTraceController"
    )
    monkeypatch.delenv("NTRACE_RUN_ID", raising=False)


def load_methods(path, class_name, names, **namespace):
    """Execute actual routing methods without importing Ray, CUDA, or models."""
    tree = ast.parse((ROOT / path).read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    methods = [
        node
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in names
    ]
    assert {node.name for node in methods} == set(names)
    for method in methods:
        method.decorator_list = []
    namespace = {"Any": Any, **namespace}
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *methods,
        ],
        type_ignores=[],
    )
    exec(
        compile(ast.fix_missing_locations(module), str(ROOT / path), "exec"), namespace
    )
    return {name: namespace[name] for name in names}


def test_one_run_id_and_explicit_schedule_for_both_roles():
    policy, rollout = MagicMock(), MagicMock()
    capture = profiling.GrpoCapture(policy, rollout, schedule="sync", colocated=True)
    left, right = (
        policy.full_step_profile.call_args,
        rollout.full_step_profile.call_args,
    )
    assert left.kwargs["run_id"] == right.kwargs["run_id"]
    assert left.kwargs["role"] == "policy"
    assert right.kwargs["role"] == "rollout"
    assert left.kwargs["schedule"] == right.kwargs["schedule"] == "sync"
    assert left.kwargs["colocated"] is True
    with capture.step(step_id=8, attempt=2, weight_version=7):
        pass
    for target in (policy, rollout):
        target.full_step_profile.assert_any_call(
            "begin_step", step_id=8, attempt=2, weight_version=7
        )
        assert target.full_step_profile.call_args == call("finish_step")


def test_explicit_run_id_is_preserved(monkeypatch):
    monkeypatch.setenv("NTRACE_RUN_ID", "job-123")
    policy, rollout = MagicMock(), MagicMock()
    profiling.GrpoCapture(policy, rollout, schedule="async", colocated=False)
    assert policy.full_step_profile.call_args.kwargs["run_id"] == "job-123"


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("data_plane", [None, {"enabled": False}, {"enabled": True}])
def test_setup_rejects_four_phase_transfer_queue_before_allocating_workers(
    monkeypatch, enabled, data_plane
):
    """Execute the real setup entry through its first timing/allocation boundary."""
    monkeypatch.setenv("NRL_NTRACE_FOUR_PHASE", "1" if enabled else "0")
    tree = ast.parse((ROOT / "nemo_rl/algorithms/grpo.py").read_text())
    setup = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "setup"
    )
    # Setup must reject an unsupported trainer before starting setup work.
    setup_start = next(
        i
        for i, node in enumerate(setup.body)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "setup_start_time"
            for target in node.targets
        )
    )
    body = ast.Module(body=setup.body[:setup_start], type_ignores=[])
    namespace = {
        "master_config": SimpleNamespace(data_plane=data_plane),
        "four_phase_enabled": profiling.four_phase_enabled,
    }
    if enabled and data_plane and data_plane["enabled"]:
        with pytest.raises(ValueError, match="data_plane.enabled=false"):
            exec(compile(body, "grpo.setup", "exec"), namespace)
    else:
        exec(compile(body, "grpo.setup", "exec"), namespace)


def test_async_bootstrap_is_in_first_policy_step_without_second_rollout_owner():
    policy, rollout = MagicMock(), MagicMock()
    capture = profiling.GrpoCapture(policy, rollout, schedule="async", colocated=False)
    policy.reset_mock()
    rollout.reset_mock()
    capture.begin_policy_step(step_id=4, weight_version=3)
    with capture.step(step_id=4, weight_version=3):
        pass
    assert policy.full_step_profile.call_args_list == [
        call("begin_step", step_id=4, attempt=0, weight_version=3),
        call("finish_step"),
    ]
    rollout.full_step_profile.assert_not_called()


def test_failed_step_aborts_both_roles_and_preserves_original_error():
    policy, rollout = MagicMock(), MagicMock()
    capture = profiling.GrpoCapture(policy, rollout, schedule="sync", colocated=True)
    with pytest.raises(ValueError, match="train failed"):
        with capture.step(step_id=1, weight_version=0):
            raise ValueError("train failed")
    for target in (policy, rollout):
        assert target.full_step_profile.call_args == call(
            "abort_step", reason="grpo_step_failed"
        )


def test_finish_failure_still_aborts_all_workers():
    policy, rollout = MagicMock(), MagicMock()
    capture = profiling.GrpoCapture(policy, rollout, schedule="sync", colocated=True)
    rollout.full_step_profile.side_effect = lambda command, **kwargs: (
        (_ for _ in ()).throw(RuntimeError("save failed"))
        if command == "finish_step"
        else None
    )
    with pytest.raises(RuntimeError, match="save failed"):
        with capture.step(step_id=1, weight_version=0):
            pass
    policy.full_step_profile.assert_any_call("abort_step", reason="grpo_step_failed")
    rollout.full_step_profile.assert_any_call("abort_step", reason="grpo_step_failed")


def test_disabled_capture_leaves_worker_scheduling_alone(monkeypatch):
    monkeypatch.setenv("NRL_NTRACE_FOUR_PHASE", "0")
    policy, rollout = MagicMock(), MagicMock()
    capture = profiling.GrpoCapture(policy, rollout, schedule="sync", colocated=True)
    with capture.step(step_id=1, weight_version=0):
        with profiling.profile_phase(
            (policy, rollout), name="refit", phase_slot="refit"
        ):
            pass
    policy.full_step_profile.assert_not_called()
    rollout.full_step_profile.assert_not_called()


def test_missing_role_profiler_rejected_before_work(monkeypatch):
    monkeypatch.delenv("NRL_ROLLOUT_PROFILER_CLASS")
    with pytest.raises(ValueError, match="both policy and rollout"):
        profiling.GrpoCapture(MagicMock(), MagicMock(), schedule="sync", colocated=True)


def test_rank_local_phase_token_is_not_sent_through_ray():
    controller = MagicMock()
    worker = profiling.WorkerCapture(controller)
    token = object()
    controller.begin_phase.return_value = token
    worker.dispatch("begin_step", step_id=1)
    assert worker.dispatch("begin_phase", name="refit", phase_slot="refit") is None
    worker.dispatch("end_phase")
    controller.end_phase.assert_called_once_with(token)
    worker.dispatch("finish_step")
    assert worker.step_open is False


def test_setup_logprobs_does_not_create_a_step_or_phase():
    controller = MagicMock()
    worker = profiling.WorkerCapture(controller)
    worker.dispatch("begin_phase", name="reference", phase_slot="logprobs")
    worker.dispatch("end_phase")
    assert controller.mock_calls == []


def test_overlapping_worker_phases_rejected():
    worker = profiling.WorkerCapture(MagicMock())
    worker.dispatch("begin_step", step_id=1)
    worker.dispatch("begin_phase", name="refit", phase_slot="refit")
    with pytest.raises(RuntimeError, match="already open"):
        worker.dispatch("begin_phase", name="generation", phase_slot="generation")


def test_refit_carries_transferred_weight_version_and_includes_both_workers():
    policy, rollout = MagicMock(), MagicMock()
    body = MagicMock(return_value={"bytes": 42})
    wrapped = profiling.profile_refit(body)
    assert wrapped(policy, rollout, True, profile_weight_version=8) == {"bytes": 42}
    for target in (policy, rollout):
        target.full_step_profile.assert_has_calls(
            [
                call(
                    "begin_phase",
                    name="weight_refit",
                    phase_slot="refit",
                    labels={"weight_version": 8},
                ),
                call("end_phase"),
            ]
        )
    body.assert_called_once_with(policy, rollout, True, profile_weight_version=8)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("stale", [False, True])
@pytest.mark.parametrize("failure", [None, "validation", "finish_generation"])
def test_sync_driver_validation_has_a_phase_and_closes_or_aborts(
    monkeypatch, enabled, stale, failure
):
    """Run the real in-step validation branch through the worker dispatcher."""
    monkeypatch.setenv("NRL_NTRACE_FOUR_PHASE", "1" if enabled else "0")
    tree = ast.parse((ROOT / "nemo_rl/algorithms/grpo.py").read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_grpo_train_impl"
    )
    step = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.With)
        and any(
            isinstance(item.context_expr, ast.Call)
            and ast.unparse(item.context_expr.func) == "capture.step"
            for item in node.items
        )
    )
    branch = next(
        node
        for node in ast.walk(step)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "should_run_validation"
    )
    work = []

    class Target:
        def __init__(self, role):
            self.role = role
            self.controller = MagicMock()
            self.controller.begin_phase.side_effect = lambda name, **kw: (name, kw)
            self.worker = profiling.WorkerCapture(self.controller)
            self.full_step_profile = MagicMock(wraps=self.worker.dispatch)

        def gpu_work(self, name):
            if enabled:
                assert self.worker.step_open and self.worker.phase_open
            work.append((self.role, name, self.worker.phase_token))
            if name == failure:
                raise RuntimeError(f"{name} failed")

        def offload_after_refit(self):
            self.gpu_work("offload")

        def prepare_for_generation(self):
            self.gpu_work("wake")

        def finish_generation(self):
            self.gpu_work("finish_generation")

    policy, rollout = Target("policy"), Target("rollout")

    @profiling.profile_refit
    def refit(policy, rollout, *args, **kwargs):
        policy.gpu_work("send_weights")
        rollout.gpu_work("receive_weights")

    def validate(*args, **kwargs):
        rollout.gpu_work("validation")
        return {"accuracy": 0.5}, {"total_validation_time": 1.0}

    namespace = dict(
        should_run_validation=True,
        memory_tracker=MagicMock(),
        POLICY_GENERATION_STALE=stale,
        refit_policy_generation=refit,
        profile_phase=profiling.profile_phase,
        policy=policy,
        policy_generation=rollout,
        colocated_inference=True,
        refit_buffer_size_gb=None,
        sync_kv_scales=False,
        validate=validate,
        val_dataloader=None,
        tokenizer=None,
        val_task_to_env=None,
        total_steps=10,
        master_config=SimpleNamespace(
            grpo=SimpleNamespace(debug_payload_metrics=False)
        ),
        logger=MagicMock(),
        processor=None,
        _validation_early_stop_message=lambda *a: None,
        stop_at_validation_threshold=None,
        stop_at_validation_metric=None,
    )
    capture = profiling.GrpoCapture(policy, rollout, schedule="sync", colocated=True)
    expected = (
        pytest.raises(RuntimeError, match=f"{failure} failed")
        if failure
        else nullcontext()
    )
    with expected:
        with capture.step(step_id=11, weight_version=10):
            exec(
                compile(
                    ast.Module(body=[branch], type_ignores=[]),
                    "sync_validation",
                    "exec",
                ),
                namespace,
            )

    prep_names = (
        [("policy", "send_weights"), ("rollout", "receive_weights")]
        if stale
        else [("policy", "offload"), ("rollout", "wake")]
    )
    expected_work = prep_names + [("rollout", "validation")]
    if failure != "validation":
        expected_work.append(("rollout", "finish_generation"))
    assert [(role, name) for role, name, phase in work] == expected_work
    for role, name, phase in work:
        if not enabled:
            assert phase is None
            continue
        phase_name, phase_args = phase
        assert phase_args["labels"]["weight_version"] == 11
        if name in {"validation", "finish_generation"}:
            assert phase_name == "validation_generation"
            assert phase_args["phase_slot"] == "generation"
            assert phase_args["labels"]["purpose"] == "validation"
        else:
            assert phase_args["phase_slot"] == "refit"
            assert phase_name == ("weight_refit" if stale else "validation_wake")
    for target in (policy, rollout):
        assert not target.worker.step_open and not target.worker.phase_open
        if not enabled:
            target.full_step_profile.assert_not_called()
        elif failure:
            target.controller.finish_step.assert_not_called()
            target.controller.abort_step.assert_called()
        else:
            target.controller.finish_step.assert_called_once_with()
            target.controller.abort_step.assert_not_called()


def test_policy_dispatch_reaches_all_gpu_ranks():
    ray = SimpleNamespace(get=MagicMock())
    methods = load_methods(
        "nemo_rl/models/policy/lm_policy.py", "Policy", ["full_step_profile"], ray=ray
    )
    policy = SimpleNamespace(worker_group=MagicMock())
    methods["full_step_profile"](policy, "begin_step", step_id=7)
    policy.worker_group.run_all_workers_single_data.assert_called_once_with(
        "full_step_profile", command="begin_step", step_id=7
    )
    ray.get.assert_called_once_with(
        policy.worker_group.run_all_workers_single_data.return_value
    )


def test_sync_vllm_dispatch_reaches_internal_gpu_workers():
    methods = load_methods(
        "nemo_rl/models/generation/vllm/vllm_worker.py",
        "BaseVllmGenerationWorker",
        ["full_step_profile"],
    )
    worker = SimpleNamespace(
        _use_internal_rollout_profiler=True,
        _run_internal_rollout_profiler_rpc=MagicMock(),
    )
    methods["full_step_profile"](worker, "begin_step", step_id=7)
    worker._run_internal_rollout_profiler_rpc.assert_called_once_with(
        "full_step_profile", command="begin_step", step_id=7
    )


def test_async_vllm_dispatch_awaits_internal_gpu_workers():
    methods = load_methods(
        "nemo_rl/models/generation/vllm/vllm_worker_async.py",
        "VllmAsyncGenerationWorkerImpl",
        ["full_step_profile_async"],
    )
    worker = SimpleNamespace(
        _use_internal_rollout_profiler=True,
        llm=SimpleNamespace(collective_rpc=AsyncMock()),
    )
    asyncio.run(methods["full_step_profile_async"](worker, "begin_step", step_id=7))
    worker.llm.collective_rpc.assert_awaited_once_with(
        "full_step_profile", args=(), kwargs={"command": "begin_step", "step_id": 7}
    )


def collector():
    methods = load_methods(
        "nemo_rl/algorithms/async_utils/trajectory_collector.py",
        "AsyncTrajectoryCollector",
        [
            "begin_rollout_profile_epoch",
            "_begin_rollout_epoch_generation",
            "_end_rollout_profile_generation",
            "_finish_rollout_profile_epoch",
            "prepare_for_refit",
            "drain_for_rollout_profiler_shutdown",
            "finalize_rollout_profiler_shutdown",
        ],
        time=time,
        should_use_async_rollouts=lambda cfg: True,
    )
    obj = SimpleNamespace(
        _profile_full_run=True,
        _rollout_epoch_open=False,
        _rollout_epoch_generation_open=False,
        _rollout_profiler_shutdown_ready=False,
        current_weight_version=3,
        _refit_pause_cleared=threading.Event(),
        _inflight_threads=[],
        master_config=SimpleNamespace(policy={"generation": {"backend": "vllm"}}),
        async_config=SimpleNamespace(
            in_flight_weight_updates=True, recompute_kv_cache_after_weight_updates=False
        ),
        policy_generation=MagicMock(),
        _wake_waits=MagicMock(),
        running=True,
        _calculate_target_weights=lambda version: [version + 1, version + 2],
    )
    obj.policy_generation.pause_generation_for_refit.return_value = True
    for name, method in methods.items():
        setattr(obj, name, method.__get__(obj))
    return obj


def test_async_epoch_closes_only_after_native_pause_and_next_epoch_precedes_refit():
    obj = collector()
    obj.begin_rollout_profile_epoch(3)
    obj._begin_rollout_epoch_generation()
    obj.policy_generation.reset_mock()
    obj.prepare_for_refit()
    assert obj.policy_generation.mock_calls == [
        call.pause_generation_for_refit(clear_cache=False),
        call.full_step_profile("end_phase"),
        call.full_step_profile("finish_step"),
        call.full_step_profile(
            "begin_step", step_id="generation4", attempt=0, weight_version=4
        ),
    ]
    assert obj._rollout_epoch_generation_open is False


def test_async_epoch_retains_many_eligible_targets_without_request_assignment():
    obj = collector()
    obj.begin_rollout_profile_epoch(3)
    obj._begin_rollout_epoch_generation()
    labels = obj.policy_generation.full_step_profile.call_args.kwargs["labels"]
    assert labels["eligible_target_weight_versions"] == [4, 5]
    assert labels["request_assignment"] == "may_cross_weight_epochs"


def test_failed_native_pause_cannot_claim_completed_epoch():
    obj = collector()
    obj.begin_rollout_profile_epoch(3)
    obj._begin_rollout_epoch_generation()
    obj.policy_generation.pause_generation_for_refit.return_value = False
    obj.policy_generation.reset_mock()
    with pytest.raises(RuntimeError, match="successful native generation pause"):
        obj.prepare_for_refit()
    assert call.full_step_profile("finish_step") not in obj.policy_generation.mock_calls


def test_async_shutdown_quiesces_before_separate_save_without_creating_another():
    obj = collector()
    obj.begin_rollout_profile_epoch(3)
    obj._begin_rollout_epoch_generation()
    obj.policy_generation.reset_mock()
    obj.drain_for_rollout_profiler_shutdown()
    assert obj.policy_generation.mock_calls == [
        call.pause_generation_for_refit(clear_cache=False),
        call.full_step_profile("end_phase"),
    ]
    assert obj.running is False
    obj._wake_waits.assert_called_once_with()
    assert obj._rollout_epoch_open is True
    obj.finalize_rollout_profiler_shutdown()
    assert obj.policy_generation.mock_calls[-1] == call.full_step_profile("finish_step")
    assert obj._rollout_epoch_open is False


def load_driver_shutdown(ray):
    tree = ast.parse((ROOT / "nemo_rl/algorithms/grpo.py").read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_shutdown_async_trajectory_collector"
    )
    namespace = {"ray": ray, "_ASYNC_ROLLOUT_PROFILER_DRAIN_RPC_TIMEOUT_S": 40.0}
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            function,
        ],
        type_ignores=[],
    )
    exec(
        compile(ast.fix_missing_locations(module), "driver_shutdown", "exec"), namespace
    )
    return namespace[function.name]


def shutdown_actor(obj):
    actor = MagicMock()
    actor.drain_for_rollout_profiler_shutdown.remote.return_value = (
        obj.drain_for_rollout_profiler_shutdown
    )
    actor.finalize_rollout_profiler_shutdown.remote.return_value = (
        obj.finalize_rollout_profiler_shutdown
    )
    return actor


@pytest.mark.parametrize("has_generation", [False, True])
def test_save_exceeding_control_budget_runs_after_bounded_quiescence(has_generation):
    obj = collector()
    obj.begin_rollout_profile_epoch(3)
    if has_generation:
        obj._begin_rollout_epoch_generation()
    elapsed = [0.0]

    def worker(command, **kwargs):
        if command == "finish_step":
            # Model the observed long serialization without sleeping in a CPU test.
            elapsed[0] += 1731.0

    obj.policy_generation.full_step_profile.side_effect = worker

    def wait(future, *, timeout=None):
        start = elapsed[0]
        future()
        if timeout is not None and elapsed[0] - start > timeout:
            raise TimeoutError("control deadline included artifact save")

    actor = shutdown_actor(obj)
    ray = SimpleNamespace(get=MagicMock(side_effect=wait), kill=MagicMock())
    load_driver_shutdown(ray)(
        actor,
        obj.policy_generation,
        flush_telemetry=MagicMock(),
        full_phase_capture=True,
    )
    assert elapsed[0] == 1731.0
    assert ray.get.call_args_list == [
        call(obj.drain_for_rollout_profiler_shutdown, timeout=40.0),
        call(obj.finalize_rollout_profiler_shutdown),
    ]
    ray.kill.assert_called_once_with(actor)
    assert obj._rollout_epoch_open is False


def test_control_timeout_refuses_finalize_and_still_reaps_collector():
    obj = collector()
    obj.begin_rollout_profile_epoch(3)
    actor = shutdown_actor(obj)
    ray = SimpleNamespace(
        get=MagicMock(side_effect=TimeoutError("pause timed out")), kill=MagicMock()
    )
    flush = MagicMock()
    with pytest.raises(TimeoutError, match="pause timed out"):
        load_driver_shutdown(ray)(
            actor, obj.policy_generation, flush_telemetry=flush, full_phase_capture=True
        )
    actor.finalize_rollout_profiler_shutdown.remote.assert_not_called()
    flush.assert_called_once_with()
    ray.kill.assert_called_once_with(actor)
    with pytest.raises(RuntimeError, match="has not quiesced"):
        obj.finalize_rollout_profiler_shutdown()


def test_failed_native_shutdown_pause_cannot_enable_save_finalization():
    obj = collector()
    obj.begin_rollout_profile_epoch(3)
    obj._begin_rollout_epoch_generation()
    obj.policy_generation.pause_generation_for_refit.return_value = False
    with pytest.raises(RuntimeError, match="successful native generation pause"):
        obj.drain_for_rollout_profiler_shutdown()
    assert obj.running is False
    obj.policy_generation.full_step_profile.assert_any_call(
        "abort_step", reason="async_epoch_shutdown_failed"
    )
    with pytest.raises(RuntimeError, match="has not quiesced"):
        obj.finalize_rollout_profiler_shutdown()
    assert (
        call("finish_step")
        not in obj.policy_generation.full_step_profile.call_args_list
    )


def test_save_failure_propagates_and_aborts_before_collector_cleanup():
    obj = collector()
    obj.begin_rollout_profile_epoch(3)
    obj._begin_rollout_epoch_generation()

    def worker(command, **kwargs):
        if command == "finish_step":
            raise ValueError("disk save failed")

    obj.policy_generation.full_step_profile.side_effect = worker
    actor = shutdown_actor(obj)
    ray = SimpleNamespace(
        get=MagicMock(side_effect=lambda future, **kwargs: future()), kill=MagicMock()
    )
    flush = MagicMock()
    with pytest.raises(ValueError, match="disk save failed"):
        load_driver_shutdown(ray)(
            actor, obj.policy_generation, flush_telemetry=flush, full_phase_capture=True
        )
    obj.policy_generation.full_step_profile.assert_any_call(
        "abort_step", reason="async_epoch_save_failed"
    )
    assert obj._rollout_profiler_shutdown_ready is False
    flush.assert_called_once_with()
    ray.kill.assert_called_once_with(actor)


def test_legacy_driver_drain_keeps_only_bounded_rpc():
    actor, generation = MagicMock(), MagicMock()
    ray = SimpleNamespace(get=MagicMock(), kill=MagicMock())
    load_driver_shutdown(ray)(actor, generation, flush_telemetry=MagicMock())
    ray.get.assert_called_once_with(
        actor.drain_for_rollout_profiler_shutdown.remote.return_value, timeout=40.0
    )
    actor.finalize_rollout_profiler_shutdown.remote.assert_not_called()


@pytest.mark.parametrize("capture_error", [None, RuntimeError("incomplete capture")])
def test_explicit_capture_close_controls_failure_separately_from_engine_cleanup(
    capture_error,
):
    tree = ast.parse((ROOT / "nemo_rl/algorithms/grpo.py").read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "async_grpo_train"
    )
    final = next(
        node.finalbody
        for node in function.body
        if isinstance(node, ast.Try)
        and any(
            isinstance(statement, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "active_error"
                for target in statement.targets
            )
            for statement in node.finalbody
        )
    )
    policy, generation = MagicMock(), MagicMock()
    policy.shutdown.return_value = generation.shutdown.return_value = False
    capture = profiling.GrpoCapture(
        policy, generation, schedule="async", colocated=False
    )
    if capture_error:
        policy.full_step_profile.side_effect = capture_error
    namespace = dict(
        sys=sys,
        checkpointer=MagicMock(),
        trajectory_collector=MagicMock(),
        policy_generation=generation,
        policy=policy,
        capture=capture,
        _shutdown_async_trajectory_collector=MagicMock(),
        _flush_collector_telemetry=MagicMock(),
        ray=MagicMock(),
        replay_buffer=object(),
        shutdown_environments=MagicMock(),
        task_to_env=None,
        val_task_to_env=None,
    )
    module = ast.Module(body=final, type_ignores=[])
    if capture_error:
        with pytest.raises(RuntimeError, match="incomplete capture"):
            exec(
                compile(ast.fix_missing_locations(module), "async_teardown", "exec"),
                namespace,
            )
    else:
        exec(
            compile(ast.fix_missing_locations(module), "async_teardown", "exec"),
            namespace,
        )
    policy.full_step_profile.assert_any_call("close")
    generation.full_step_profile.assert_any_call("close")
    policy.shutdown.assert_called_once_with()
    generation.shutdown.assert_called_once_with()
    assert (
        namespace["_shutdown_async_trajectory_collector"].call_args.kwargs[
            "full_phase_capture"
        ]
        is True
    )


@pytest.mark.parametrize("drain_error", [None, TimeoutError("drain failed")])
def test_initial_validation_early_stop_finalizes_and_checks_capture(drain_error):
    tree = ast.parse((ROOT / "nemo_rl/algorithms/grpo.py").read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "async_grpo_train"
    )
    early_stop = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "stop_message is not None"
    )
    capture = MagicMock(enabled=True)
    capture.close.side_effect = RuntimeError("too few captured policy steps")
    shutdown = MagicMock(side_effect=drain_error)
    namespace = dict(
        sys=sys,
        stop_message="validation threshold met",
        checkpointer=MagicMock(),
        trajectory_collector=MagicMock(),
        policy_generation=MagicMock(),
        capture=capture,
        _shutdown_async_trajectory_collector=shutdown,
        _flush_collector_telemetry=MagicMock(),
        ray=MagicMock(),
        replay_buffer=object(),
    )
    module = ast.Module(body=early_stop.body[:-1], type_ignores=[])
    expected = (
        (TimeoutError, "drain failed")
        if drain_error
        else (RuntimeError, "too few captured policy steps")
    )
    with pytest.raises(expected[0], match=expected[1]) as caught:
        exec(
            compile(
                ast.fix_missing_locations(module), "initial_validation_stop", "exec"
            ),
            namespace,
        )
    assert shutdown.call_args.kwargs["full_phase_capture"] is True
    capture.close.assert_called_once_with()
    namespace["ray"].kill.assert_called_once_with(namespace["replay_buffer"])
    if drain_error:
        assert "too few captured policy steps" in caught.value.__notes__[0]


def test_async_replay_retries_do_not_consume_policy_capture_ordinals():
    policy, rollout = MagicMock(), MagicMock()
    capture = profiling.GrpoCapture(policy, rollout, schedule="async", colocated=False)
    policy.reset_mock()
    capture.begin_policy_step(step_id=1, weight_version=0)
    # The bootstrap window remains open while the driver waits for a valid batch.
    capture.begin_policy_step(step_id=1, weight_version=0)
    capture.finish_policy_step()
    capture.begin_policy_step(step_id=2, weight_version=1)
    capture.finish_policy_step()
    assert policy.full_step_profile.call_args_list == [
        call("begin_step", step_id=1, attempt=0, weight_version=0),
        call("finish_step"),
        call("begin_step", step_id=2, attempt=0, weight_version=1),
        call("finish_step"),
    ]


def test_composed_sync_step_includes_optimizer_inside_training():
    import functools
    import logging

    tree = ast.parse(
        (ROOT / "nemo_rl/models/policy/workers/megatron_policy_worker.py").read_text()
    )
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {
            "_begin_policy_profiler_step",
            "_take_open_policy_profiler_step",
            "_finish_policy_profiler_step",
            "_abort_policy_profiler_after_error",
            "_profile_policy_step",
        }
    ]
    namespace = {"functools": functools, "log": logging.getLogger(__name__)}
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *functions,
        ],
        type_ignores=[],
    )
    exec(
        compile(ast.fix_missing_locations(module), "megatron_policy_profiler", "exec"),
        namespace,
    )
    work = []

    class Controller:
        def __init__(self, role):
            self.role, self.phase, self.step_open = role, None, False
            self.completed_phases = []

        def configure_capture(self, **kwargs):
            pass

        def begin_step(self, **kwargs):
            assert not self.step_open
            self.step_open = True

        def finish_step(self):
            assert self.phase is None
            self.step_open = False

        def abort_step(self, **kwargs):
            self.step_open = False
            self.phase = None

        def begin_phase(self, name, *, phase_slot, labels=None):
            assert self.step_open and self.phase is None
            self.phase = (name, phase_slot)
            return self.phase

        def end_phase(self, token):
            assert self.phase == token
            self.completed_phases.append(token)
            self.phase = None

        def begin_train_step(self):
            self.begin_phase("policy_training", phase_slot="training")

        def finish_train_step(self):
            self.end_phase(self.phase)

        def abort_train_step(self, **kwargs):
            self.abort_step(**kwargs)

        def begin_rollout(self, **kwargs):
            self.begin_phase("generation", phase_slot="generation")

        def finish_rollout(self):
            self.end_phase(self.phase)

        def gpu_work(self, name):
            assert self.step_open and self.phase is not None
            work.append((self.role, self.phase[1], name))

    class Target:
        def __init__(self, role):
            self.controller = Controller(role)
            self.capture = profiling.WorkerCapture(self.controller)
            self._policy_profiler = self.controller
            self._policy_profiler_step_open = False

        def full_step_profile(self, command, **kwargs):
            self.capture.dispatch(command, **kwargs)

        @profiling.profile_policy_method("previous_policy_logprobs")
        def previous_logprobs(self):
            self.controller.gpu_work("previous_forward")

        @profiling.profile_policy_method("reference_policy_logprobs")
        def reference_logprobs(self):
            for operation in (
                "reference_weight_load",
                "reference_forward",
                "policy_weight_restore",
            ):
                self.controller.gpu_work(operation)

        @namespace["_profile_policy_step"]
        def train(self):
            for operation in ("forward", "backward", "optimizer"):
                self.controller.gpu_work(operation)
            return {}

    policy, rollout = Target("policy"), Target("rollout")
    capture = profiling.GrpoCapture(policy, rollout, schedule="sync", colocated=True)
    methods = load_methods(
        "nemo_rl/models/generation/vllm/vllm_worker.py",
        "BaseVllmGenerationWorker",
        ["begin_rollout_profile", "finish_rollout_profile"],
    )
    vllm_worker = SimpleNamespace(_rollout_profiler=rollout.controller)

    @profiling.profile_refit
    def refit(policy, rollout):
        policy.controller.gpu_work("transfer_weights")
        rollout.controller.gpu_work("load_weights")

    with capture.step(step_id=2, attempt=0, weight_version=1):
        refit(policy, rollout)
        methods["begin_rollout_profile"](vllm_worker, step_id="step2/attempt1")
        rollout.controller.gpu_work("decode")
        methods["finish_rollout_profile"](vllm_worker)
        policy.previous_logprobs()
        policy.reference_logprobs()
        policy.train()
    assert work == [
        ("policy", "refit", "transfer_weights"),
        ("rollout", "refit", "load_weights"),
        ("rollout", "generation", "decode"),
        ("policy", "logprobs", "previous_forward"),
        ("policy", "logprobs", "reference_weight_load"),
        ("policy", "logprobs", "reference_forward"),
        ("policy", "logprobs", "policy_weight_restore"),
        ("policy", "training", "forward"),
        ("policy", "training", "backward"),
        ("policy", "training", "optimizer"),
    ]
    assert policy.controller.step_open is False
    assert rollout.controller.step_open is False


def test_second_prepare_for_refit_preserves_the_unstarted_next_epoch():
    obj = collector()
    obj.begin_rollout_profile_epoch(3)
    obj._begin_rollout_epoch_generation()
    obj.prepare_for_refit()
    obj.policy_generation.reset_mock()
    obj.prepare_for_refit()
    assert obj.policy_generation.mock_calls == [
        call.pause_generation_for_refit(clear_cache=False)
    ]
    assert obj._rollout_epoch_open is True


def test_close_reports_incomplete_capture_and_still_closes_other_role():
    policy, rollout = MagicMock(), MagicMock()
    capture = profiling.GrpoCapture(policy, rollout, schedule="sync", colocated=True)
    policy.full_step_profile.side_effect = RuntimeError("too few captured steps")
    with pytest.raises(RuntimeError, match="too few captured steps"):
        capture.close()
    rollout.full_step_profile.assert_any_call("close")


def test_abort_setup_preserves_original_error_when_abort_also_fails():
    policy, rollout = MagicMock(), MagicMock()
    capture = profiling.GrpoCapture(policy, rollout, schedule="async", colocated=False)
    policy.full_step_profile.side_effect = RuntimeError("actor unavailable")
    error = ValueError("initial refit failed")
    capture.abort_all(error, reason="setup_failed")
    assert "actor unavailable" in error.__notes__[0]
    rollout.full_step_profile.assert_any_call("abort_step", reason="setup_failed")


def test_async_start_phase_failure_is_awaited_and_aborts_both_captures():
    tree = ast.parse((ROOT / "nemo_rl/algorithms/grpo.py").read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "async_grpo_train"
    )
    index = next(
        i
        for i, node in enumerate(function.body)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "collection_start"
            for target in node.targets
        )
    )
    statements = function.body[index : index + 2]
    assert isinstance(statements[1], ast.If)
    policy, rollout = MagicMock(), MagicMock()
    capture = profiling.GrpoCapture(policy, rollout, schedule="async", colocated=False)
    wait = MagicMock(side_effect=RuntimeError("generation phase start failed"))
    namespace = {
        "capture": capture,
        "trajectory_collector": MagicMock(),
        "CyclingDataLoader": lambda value: value,
        "dataloader": object(),
        "ray": SimpleNamespace(get=wait),
    }
    module = ast.Module(body=statements, type_ignores=[])
    with pytest.raises(RuntimeError, match="generation phase start failed"):
        exec(
            compile(ast.fix_missing_locations(module), "grpo_start_collection", "exec"),
            namespace,
        )
    wait.assert_called_once_with(
        namespace["trajectory_collector"].start_collection.remote.return_value
    )
    for target in (policy, rollout):
        target.full_step_profile.assert_any_call(
            "abort_step", reason="async_collection_start_failed"
        )


def test_training_preparation_has_explicit_training_phase():
    policy = MagicMock()
    wrapped = profiling.profile_policy_method("training_onload", phase_slot="training")(
        lambda target: "ready"
    )
    assert wrapped(policy) == "ready"
    policy.full_step_profile.assert_has_calls(
        [
            call(
                "begin_phase",
                name="training_onload",
                phase_slot="training",
                labels=None,
            ),
            call("end_phase"),
        ]
    )

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from nemo_rl.models.generation.vllm.model_step_gpu_profiler import (
    MODEL_STEP_GPU_PROFILER_PROOF_SCHEMA_VERSION,
    DeterministicModelStepGpuProfiler,
    ModelStepGpuProfilerContractError,
)


class FakeCuda:
    def __init__(
        self,
        events: list[Any],
        *,
        fail_operation: str | None = None,
    ) -> None:
        self._events = events
        self._fail_operation = fail_operation
        self.profiler = SimpleNamespace(
            start=lambda: self._call("profiler_start"),
            stop=lambda: self._call("profiler_stop"),
        )
        self.nvtx = SimpleNamespace(
            range_push=lambda name: self._call("nvtx_push", name),
            range_pop=lambda: self._call("nvtx_pop"),
        )

    def current_device(self) -> int:
        return 3

    def synchronize(self) -> None:
        self._call("synchronize")

    def _call(self, operation: str, *args: Any) -> None:
        self._events.append((operation, *args))
        if operation == self._fail_operation:
            raise RuntimeError(f"{operation} failed")


class FakeTorch:
    def __init__(
        self,
        events: list[Any],
        *,
        fail_operation: str | None = None,
    ) -> None:
        self.cuda = FakeCuda(events, fail_operation=fail_operation)


class FakeModelRunner:
    device = "cuda:3"

    def __init__(
        self,
        events: list[Any],
        *,
        fail_on_value: int | None = None,
    ) -> None:
        self._events = events
        self._fail_on_value = fail_on_value

    def execute_model(self, value: int) -> str:
        self._events.append(("execute_model", value))
        if value == self._fail_on_value:
            raise RuntimeError(f"execute_model failed for {value}")
        return f"result-{value}"


def make_controller(
    *,
    fail_operation: str | None = None,
    fail_on_value: int | None = None,
) -> tuple[
    list[Any],
    FakeModelRunner,
    DeterministicModelStepGpuProfiler,
]:
    events: list[Any] = []
    runner = FakeModelRunner(events, fail_on_value=fail_on_value)
    controller = DeterministicModelStepGpuProfiler(
        runner,
        torch_module=FakeTorch(events, fail_operation=fail_operation),
        hostname_fn=lambda: "worker-host",
        pid_fn=lambda: 1234,
    )
    return events, runner, controller


def test_exact_half_open_capture_and_callable_restoration() -> None:
    events, runner, controller = make_controller()
    original_function = runner.execute_model.__func__
    controller.arm(1, 3)

    assert runner.execute_model(10) == "result-10"
    assert runner.execute_model(11) == "result-11"
    assert runner.execute_model(12) == "result-12"
    assert runner.execute_model.__func__ is original_function
    assert runner.execute_model(13) == "result-13"

    assert events == [
        ("execute_model", 10),
        ("synchronize",),
        ("profiler_start",),
        ("nvtx_push", "NRL_MODEL_STEP:1"),
        ("execute_model", 11),
        ("nvtx_pop",),
        ("nvtx_push", "NRL_MODEL_STEP:2"),
        ("execute_model", 12),
        ("nvtx_pop",),
        ("synchronize",),
        ("profiler_stop",),
        ("execute_model", 13),
    ]

    proof = controller.require_complete()
    assert proof["schema_version"] == (
        MODEL_STEP_GPU_PROFILER_PROOF_SCHEMA_VERSION
    )
    assert proof["hostname"] == "worker-host"
    assert proof["pid"] == 1234
    assert proof["device"] == {
        "cuda_current_device": 3,
        "model_runner_device": "cuda:3",
    }
    assert proof["range"] == {
        "semantics": "[start_inclusive, stop_exclusive)",
        "ordinal_origin": (
            "first model_runner.execute_model call after arm returns"
        ),
        "start_step": 1,
        "stop_step": 3,
        "expected_captured_model_step_count": 2,
        "nvtx_range_name_format": "NRL_MODEL_STEP:<zero_based_ordinal>",
    }
    assert proof["observed_model_step_count"] == 3
    assert proof["completed_execute_model_count"] == 3
    assert proof["completed_captured_model_step_count"] == 2
    assert proof["last_observed_model_step_ordinal"] == 2
    assert proof["capture_boundaries"] == {
        "profiler_started_before_model_step_ordinal": 1,
        "profiler_stopped_after_model_step_ordinal": 2,
    }
    assert proof["completion"] == {
        "complete": True,
        "state": "completed",
        "profiler_may_be_active": False,
        "nvtx_range_open": False,
    }
    assert proof["callable_identity"]["installed_exactly_once"] is True
    assert proof["callable_identity"]["wrapper_currently_installed"] is False
    assert proof["callable_identity"]["original_callable_restored"] is True
    assert proof["errors"] == []


def test_zero_start_profiles_first_call_and_stops_before_return() -> None:
    events, runner, controller = make_controller()
    controller.arm(0, 1)

    assert runner.execute_model(7) == "result-7"
    assert events == [
        ("synchronize",),
        ("profiler_start",),
        ("nvtx_push", "NRL_MODEL_STEP:0"),
        ("execute_model", 7),
        ("nvtx_pop",),
        ("synchronize",),
        ("profiler_stop",),
    ]
    assert controller.snapshot()["completion"]["complete"] is True


@pytest.mark.parametrize(
    ("start_step", "stop_step"),
    [
        (-1, 1),
        (0, 0),
        (2, 1),
        (False, 1),
        (0, True),
        (0.0, 1),
        (0, 1.0),
    ],
)
def test_invalid_bounds_fail_before_install(
    start_step: Any,
    stop_step: Any,
) -> None:
    events, runner, controller = make_controller()
    original_function = runner.execute_model.__func__

    with pytest.raises(ModelStepGpuProfilerContractError):
        controller.arm(start_step, stop_step)

    proof = controller.snapshot()
    assert runner.execute_model.__func__ is original_function
    assert proof["completion"]["state"] == "failed"
    assert proof["calls"]["wrapper_install"]["attempted"] == 0
    assert proof["callable_identity"]["wrapper_currently_installed"] is False
    assert proof["errors"][0]["operation"] == "arm"
    assert events == []


def test_double_arm_poison_wrapper_fails_closed() -> None:
    events, runner, controller = make_controller()
    controller.arm(0, 2)

    with pytest.raises(
        ModelStepGpuProfilerContractError,
        match="one-shot",
    ):
        controller.arm(0, 2)
    with pytest.raises(
        ModelStepGpuProfilerContractError,
        match="failed state",
    ):
        runner.execute_model(1)

    proof = controller.snapshot()
    assert proof["completion"]["state"] == "failed"
    assert proof["callable_identity"]["wrapper_currently_installed"] is True
    assert proof["errors"][0]["phase"] == "double_arm"
    assert events == []


def test_execute_error_closes_open_range_and_profiler_then_poison_wrapper() -> None:
    events, runner, controller = make_controller(fail_on_value=9)
    controller.arm(0, 2)

    with pytest.raises(RuntimeError, match="execute_model failed for 9"):
        runner.execute_model(9)
    with pytest.raises(ModelStepGpuProfilerContractError, match="failed state"):
        runner.execute_model(10)

    assert events == [
        ("synchronize",),
        ("profiler_start",),
        ("nvtx_push", "NRL_MODEL_STEP:0"),
        ("execute_model", 9),
        ("nvtx_pop",),
        ("synchronize",),
        ("profiler_stop",),
    ]
    proof = controller.snapshot()
    assert proof["completion"] == {
        "complete": False,
        "state": "failed",
        "profiler_may_be_active": False,
        "nvtx_range_open": False,
    }
    assert proof["calls"]["execute_model"] == {
        "attempted": 1,
        "succeeded": 0,
    }
    assert proof["errors"][0] == {
        "operation": "execute_model",
        "phase": "model_step",
        "model_step_ordinal": 0,
        "error_type": "RuntimeError",
        "error": "execute_model failed for 9",
    }


@pytest.mark.parametrize("fail_operation", ["synchronize", "profiler_start"])
def test_start_boundary_failure_does_not_execute_included_step(
    fail_operation: str,
) -> None:
    events, runner, controller = make_controller(
        fail_operation=fail_operation
    )
    controller.arm(0, 1)

    with pytest.raises(RuntimeError, match=f"{fail_operation} failed"):
        runner.execute_model(4)

    proof = controller.snapshot()
    assert ("execute_model", 4) not in events
    assert proof["completion"]["state"] == "failed"
    assert proof["completion"]["complete"] is False
    assert proof["errors"][0]["operation"] == (
        "cuda_synchronize"
        if fail_operation == "synchronize"
        else "profiler_start"
    )


def test_stop_failure_rejects_completed_model_result_and_invalidates_proof() -> None:
    events, runner, controller = make_controller(
        fail_operation="profiler_stop"
    )
    controller.arm(0, 1)

    with pytest.raises(RuntimeError, match="profiler_stop failed"):
        runner.execute_model(5)

    proof = controller.snapshot()
    assert ("execute_model", 5) in events
    assert proof["completed_execute_model_count"] == 1
    assert proof["completed_captured_model_step_count"] == 1
    assert proof["completion"]["complete"] is False
    assert proof["completion"]["state"] == "failed"
    assert proof["errors"][0]["operation"] == "profiler_stop"
    assert proof["callable_identity"]["wrapper_currently_installed"] is True


def test_require_complete_stops_partial_capture_and_raises() -> None:
    events, runner, controller = make_controller()
    controller.arm(0, 2)
    assert runner.execute_model(1) == "result-1"

    with pytest.raises(
        ModelStepGpuProfilerContractError,
        match="did not complete",
    ):
        controller.require_complete()

    assert events[-2:] == [("synchronize",), ("profiler_stop",)]
    proof = controller.snapshot()
    assert proof["completion"]["complete"] is False
    assert proof["completion"]["state"] == "failed"
    assert proof["errors"][0]["operation"] == "require_complete"


def test_non_callable_execute_model_is_rejected() -> None:
    events: list[Any] = []
    runner = SimpleNamespace(device="cuda:3", execute_model=None)
    controller = DeterministicModelStepGpuProfiler(
        runner,
        torch_module=FakeTorch(events),
    )

    with pytest.raises(
        ModelStepGpuProfilerContractError,
        match="must be callable",
    ):
        controller.arm(0, 1)

    assert controller.snapshot()["completion"]["state"] == "failed"


def test_runner_without_real_instance_dict_is_rejected() -> None:
    class SlotsRunner:
        __slots__ = ("device",)

        def __init__(self) -> None:
            self.device = "cuda:0"

        def execute_model(self, value: int) -> int:
            return value

    events: list[Any] = []
    controller = DeterministicModelStepGpuProfiler(
        SlotsRunner(),
        torch_module=FakeTorch(events),
    )

    with pytest.raises(
        ModelStepGpuProfilerContractError,
        match="__dict__",
    ):
        controller.arm(0, 1)

    proof = controller.snapshot()
    assert proof["completion"]["state"] == "failed"
    assert proof["calls"]["wrapper_install"]["attempted"] == 0

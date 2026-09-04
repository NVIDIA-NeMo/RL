# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Deterministic GPU profiling over an exact model-step interval.

The controller is deliberately independent of vLLM types. It replaces one
``model_runner.execute_model`` attribute on a specific instance and counts calls
from zero after :meth:`arm` returns. The capture interval is half open:
``[start_step, stop_step)``.

The start boundary synchronizes the current CUDA device and starts the profiler
immediately before the first included call. After the final included call
returns, the controller closes its NVTX range, synchronizes the device, stops
the profiler, and restores the exact original callable topology before
returning. Therefore no excluded model step can enter the CUDA-profiler range.
"""

from __future__ import annotations

import functools
import inspect
import os
import socket
import threading
from collections.abc import Callable
from typing import Any

import torch


MODEL_STEP_GPU_PROFILER_PROOF_SCHEMA_VERSION = 1
MODEL_STEP_NVTX_RANGE_PREFIX = "NRL_MODEL_STEP"


class ModelStepGpuProfilerError(RuntimeError):
    """Base error for deterministic model-step profiling."""


class ModelStepGpuProfilerContractError(ModelStepGpuProfilerError):
    """Raised when the requested profiling contract cannot be proven."""


def _callable_identity(value: Any) -> dict[str, Any]:
    """Return a JSON-safe identity for a callable without using its repr."""
    underlying = getattr(value, "__func__", value)
    bound_self = getattr(value, "__self__", None)
    try:
        signature = str(inspect.signature(value))
    except (TypeError, ValueError):
        signature = None
    return {
        "callable_object_id": id(value),
        "underlying_callable_id": id(underlying),
        "bound_self_id": None if bound_self is None else id(bound_self),
        "module": getattr(underlying, "__module__", None),
        "qualname": getattr(underlying, "__qualname__", None),
        "signature": signature,
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "is_coroutine_function": inspect.iscoroutinefunction(value),
    }


def _same_callable(left: Any, right: Any) -> bool:
    """Compare bound methods by function and owner, otherwise by identity."""
    left_function = getattr(left, "__func__", None)
    right_function = getattr(right, "__func__", None)
    if left_function is not None or right_function is not None:
        return (
            left_function is right_function
            and getattr(left, "__self__", None)
            is getattr(right, "__self__", None)
        )
    return left is right


class DeterministicModelStepGpuProfiler:
    """One-shot exact-step CUDA-profiler controller for a model-runner instance.

    The controller is intentionally one-shot. Re-arming it, including after a
    completed capture, invalidates its proof and raises. A failure while armed
    poisons the installed wrapper so later model execution cannot silently
    continue outside the requested profiling contract.
    """

    _CALL_NAMES = (
        "wrapper_install",
        "wrapper_restore",
        "execute_model",
        "cuda_synchronize",
        "profiler_start",
        "profiler_stop",
        "nvtx_range_push",
        "nvtx_range_pop",
    )
    _WRAPPER_CONTROLLER_ATTRIBUTE = "__nrl_model_step_gpu_profiler_controller__"

    def __init__(
        self,
        model_runner: Any,
        *,
        torch_module: Any = torch,
        hostname_fn: Callable[[], str] = socket.gethostname,
        pid_fn: Callable[[], int] = os.getpid,
    ) -> None:
        """Create an unarmed controller without changing ``model_runner``."""
        self._model_runner = model_runner
        self._torch = torch_module
        self._hostname = str(hostname_fn())
        self._pid = int(pid_fn())
        self._lock = threading.RLock()

        self._state = "new"
        self._start_step: int | None = None
        self._stop_step: int | None = None
        self._device_proof: dict[str, Any] | None = None
        self._original_execute_model: Callable[..., Any] | None = None
        self._wrapper: Callable[..., Any] | None = None
        self._original_was_instance_attribute: bool | None = None
        self._original_instance_attribute: Any = None
        self._wrapper_installed = False
        self._original_restored = False
        self._execute_in_progress = False
        self._profiler_may_be_active = False
        self._nvtx_range_open = False

        self._observed_model_step_count = 0
        self._completed_execute_model_count = 0
        self._completed_captured_model_step_count = 0
        self._last_observed_model_step_ordinal: int | None = None
        self._profiler_started_before_model_step_ordinal: int | None = None
        self._profiler_stopped_after_model_step_ordinal: int | None = None
        self._call_counts = {
            name: {"attempted": 0, "succeeded": 0}
            for name in self._CALL_NAMES
        }
        self._errors: list[dict[str, Any]] = []

    def arm(self, start_step: int, stop_step: int) -> None:
        """Install the wrapper for zero-based ``[start_step, stop_step)``.

        The first call to the wrapped ``execute_model`` after this method
        returns has ordinal zero. Bounds must be plain integers satisfying
        ``0 <= start_step < stop_step``.
        """
        with self._lock:
            if self._state != "new":
                error = ModelStepGpuProfilerContractError(
                    f"controller is one-shot and cannot be armed from state {self._state!r}"
                )
                self._record_error_locked("arm", None, "double_arm", error)
                self._fail_closed_locked()
                raise error

            try:
                self._validate_bounds(start_step, stop_step)
                original = getattr(self._model_runner, "execute_model")
                if not callable(original):
                    raise ModelStepGpuProfilerContractError(
                        "model_runner.execute_model must be callable"
                    )
                if inspect.iscoroutinefunction(original):
                    raise ModelStepGpuProfilerContractError(
                        "async execute_model callables are not supported"
                    )
                if getattr(
                    original,
                    self._WRAPPER_CONTROLLER_ATTRIBUTE,
                    None,
                ) is not None:
                    raise ModelStepGpuProfilerContractError(
                        "model_runner.execute_model is already controlled by a "
                        "model-step profiler"
                    )
                runner_dict = getattr(self._model_runner, "__dict__", None)
                if not isinstance(runner_dict, dict):
                    raise ModelStepGpuProfilerContractError(
                        "model_runner.__dict__ must be a real dict so exact "
                        "execute_model attribute restoration can be proven"
                    )
                self._device_proof = self._read_device_proof()
            except BaseException as error:
                self._record_error_locked("arm", None, "validation", error)
                self._state = "failed"
                raise

            self._start_step = start_step
            self._stop_step = stop_step
            self._original_execute_model = original
            self._original_was_instance_attribute = "execute_model" in runner_dict
            if self._original_was_instance_attribute:
                self._original_instance_attribute = runner_dict["execute_model"]

            @functools.wraps(original)
            def wrapped_execute_model(*args: Any, **kwargs: Any) -> Any:
                return self._execute_model(*args, **kwargs)

            setattr(
                wrapped_execute_model,
                self._WRAPPER_CONTROLLER_ATTRIBUTE,
                self,
            )
            self._wrapper = wrapped_execute_model
            try:
                self._invoke_locked(
                    "wrapper_install",
                    None,
                    "arm",
                    lambda: setattr(
                        self._model_runner,
                        "execute_model",
                        wrapped_execute_model,
                    ),
                )
                if (
                    getattr(self._model_runner, "execute_model")
                    is not wrapped_execute_model
                ):
                    raise ModelStepGpuProfilerContractError(
                        "installed execute_model wrapper identity did not match"
                    )
            except BaseException as error:
                if not self._errors or self._errors[-1]["error"] != str(error):
                    self._record_error_locked(
                        "wrapper_install",
                        None,
                        "identity_verification",
                        error,
                    )
                self._state = "failed"
                self._best_effort_restore_after_install_failure_locked()
                raise

            self._wrapper_installed = True
            self._state = "armed"

    def require_complete(self) -> dict[str, Any]:
        """Return the proof or fail closed and stop an incomplete capture."""
        with self._lock:
            if self._is_complete_locked():
                return self._snapshot_locked()
            error = ModelStepGpuProfilerContractError(
                "model-step profiling did not complete its exact target range"
            )
            self._record_error_locked(
                "require_complete",
                self._last_observed_model_step_ordinal,
                "incomplete_capture",
                error,
            )
            self._fail_closed_locked()
            raise error

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-safe proof snapshot without changing controller state."""
        with self._lock:
            return self._snapshot_locked()

    @staticmethod
    def _validate_bounds(start_step: int, stop_step: int) -> None:
        if (
            isinstance(start_step, bool)
            or isinstance(stop_step, bool)
            or not isinstance(start_step, int)
            or not isinstance(stop_step, int)
        ):
            raise ModelStepGpuProfilerContractError(
                "model-step bounds must be plain integers"
            )
        if start_step < 0 or stop_step <= start_step:
            raise ModelStepGpuProfilerContractError(
                "model-step bounds must satisfy 0 <= start_step < stop_step"
            )

    def _read_device_proof(self) -> dict[str, Any]:
        cuda = getattr(self._torch, "cuda", None)
        if cuda is None or not callable(
            getattr(cuda, "current_device", None)
        ):
            raise ModelStepGpuProfilerContractError(
                "torch.cuda.current_device is unavailable"
            )
        current_device = cuda.current_device()
        runner_device = getattr(self._model_runner, "device", None)
        return {
            "cuda_current_device": current_device,
            "model_runner_device": (
                None if runner_device is None else str(runner_device)
            ),
        }

    def _execute_model(self, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            if self._state == "failed":
                raise ModelStepGpuProfilerContractError(
                    "model-step profiler is in failed state"
                )
            if self._state not in ("armed", "capturing"):
                error = ModelStepGpuProfilerContractError(
                    f"wrapped execute_model called from invalid state {self._state!r}"
                )
                self._record_error_locked(
                    "execute_model",
                    self._last_observed_model_step_ordinal,
                    "invalid_state",
                    error,
                )
                self._fail_closed_locked()
                raise error
            if self._execute_in_progress:
                error = ModelStepGpuProfilerContractError(
                    "concurrent or re-entrant execute_model calls are unsupported"
                )
                self._record_error_locked(
                    "execute_model",
                    self._last_observed_model_step_ordinal,
                    "concurrent_execute",
                    error,
                )
                self._fail_closed_locked()
                raise error

            ordinal = self._observed_model_step_count
            self._observed_model_step_count += 1
            self._last_observed_model_step_ordinal = ordinal
            self._execute_in_progress = True
            try:
                return self._execute_model_locked(ordinal, *args, **kwargs)
            except BaseException:
                self._fail_closed_locked()
                raise
            finally:
                self._execute_in_progress = False

    def _execute_model_locked(
        self,
        ordinal: int,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        assert self._start_step is not None
        assert self._stop_step is not None
        assert self._original_execute_model is not None

        included = self._start_step <= ordinal < self._stop_step
        if ordinal == self._start_step:
            self._invoke_locked(
                "cuda_synchronize",
                ordinal,
                "start_boundary",
                self._torch.cuda.synchronize,
            )
            self._profiler_may_be_active = True
            self._invoke_locked(
                "profiler_start",
                ordinal,
                "start_boundary",
                self._torch.cuda.profiler.start,
            )
            self._profiler_started_before_model_step_ordinal = ordinal
            self._state = "capturing"
        elif ordinal > self._start_step and self._state != "capturing":
            error = ModelStepGpuProfilerContractError(
                "capture start boundary was not observed exactly once"
            )
            self._record_error_locked(
                "execute_model",
                ordinal,
                "missed_start_boundary",
                error,
            )
            raise error

        if included:
            self._invoke_locked(
                "nvtx_range_push",
                ordinal,
                "captured_model_step",
                lambda: self._torch.cuda.nvtx.range_push(
                    f"{MODEL_STEP_NVTX_RANGE_PREFIX}:{ordinal}"
                ),
            )
            self._nvtx_range_open = True

        result = self._invoke_locked(
            "execute_model",
            ordinal,
            "model_step",
            lambda: self._original_execute_model(*args, **kwargs),
        )
        if inspect.isawaitable(result):
            error = ModelStepGpuProfilerContractError(
                "execute_model returned an awaitable; exact synchronous "
                "boundaries cannot be proven"
            )
            self._record_error_locked(
                "execute_model",
                ordinal,
                "awaitable_result",
                error,
            )
            close = getattr(result, "close", None)
            if callable(close):
                close()
            raise error

        self._completed_execute_model_count += 1
        if included:
            self._invoke_locked(
                "nvtx_range_pop",
                ordinal,
                "captured_model_step",
                self._torch.cuda.nvtx.range_pop,
            )
            self._nvtx_range_open = False
            self._completed_captured_model_step_count += 1

        if ordinal + 1 == self._stop_step:
            self._invoke_locked(
                "cuda_synchronize",
                ordinal,
                "stop_boundary",
                self._torch.cuda.synchronize,
            )
            self._invoke_locked(
                "profiler_stop",
                ordinal,
                "stop_boundary",
                self._torch.cuda.profiler.stop,
            )
            self._profiler_may_be_active = False
            self._profiler_stopped_after_model_step_ordinal = ordinal
            self._restore_original_locked()
            self._state = "completed"

        return result

    def _invoke_locked(
        self,
        call_name: str,
        ordinal: int | None,
        phase: str,
        function: Callable[[], Any],
    ) -> Any:
        counts = self._call_counts[call_name]
        counts["attempted"] += 1
        try:
            result = function()
        except BaseException as error:
            self._record_error_locked(call_name, ordinal, phase, error)
            raise
        counts["succeeded"] += 1
        return result

    def _restore_original_locked(self) -> None:
        assert self._original_execute_model is not None
        try:
            if self._original_was_instance_attribute is False:
                self._invoke_locked(
                    "wrapper_restore",
                    self._last_observed_model_step_ordinal,
                    "completed_capture",
                    lambda: delattr(self._model_runner, "execute_model"),
                )
            else:
                restore_value = (
                    self._original_instance_attribute
                    if self._original_was_instance_attribute
                    else self._original_execute_model
                )
                self._invoke_locked(
                    "wrapper_restore",
                    self._last_observed_model_step_ordinal,
                    "completed_capture",
                    lambda: setattr(
                        self._model_runner,
                        "execute_model",
                        restore_value,
                    ),
                )
            current = getattr(self._model_runner, "execute_model")
            if not _same_callable(current, self._original_execute_model):
                raise ModelStepGpuProfilerContractError(
                    "restored execute_model callable identity did not match"
                )
        except BaseException as error:
            if not self._errors or self._errors[-1]["error"] != str(error):
                self._record_error_locked(
                    "wrapper_restore",
                    self._last_observed_model_step_ordinal,
                    "identity_verification",
                    error,
                )
            raise
        self._wrapper_installed = False
        self._original_restored = True

    def _best_effort_restore_after_install_failure_locked(self) -> None:
        if self._original_execute_model is None:
            return
        try:
            if self._original_was_instance_attribute is False:
                delattr(self._model_runner, "execute_model")
            else:
                restore_value = (
                    self._original_instance_attribute
                    if self._original_was_instance_attribute
                    else self._original_execute_model
                )
                setattr(self._model_runner, "execute_model", restore_value)
        except BaseException as error:
            self._record_error_locked(
                "wrapper_restore",
                None,
                "install_failure_cleanup",
                error,
            )

    def _fail_closed_locked(self) -> None:
        self._state = "failed"
        if self._nvtx_range_open:
            try:
                self._invoke_locked(
                    "nvtx_range_pop",
                    self._last_observed_model_step_ordinal,
                    "failure_cleanup",
                    self._torch.cuda.nvtx.range_pop,
                )
            except BaseException:
                pass
            self._nvtx_range_open = False
        if self._profiler_may_be_active:
            try:
                self._invoke_locked(
                    "cuda_synchronize",
                    self._last_observed_model_step_ordinal,
                    "failure_cleanup",
                    self._torch.cuda.synchronize,
                )
            except BaseException:
                pass
            try:
                self._invoke_locked(
                    "profiler_stop",
                    self._last_observed_model_step_ordinal,
                    "failure_cleanup",
                    self._torch.cuda.profiler.stop,
                )
            except BaseException:
                pass
            self._profiler_may_be_active = False

    def _record_error_locked(
        self,
        operation: str,
        ordinal: int | None,
        phase: str,
        error: BaseException,
    ) -> None:
        self._errors.append(
            {
                "operation": operation,
                "phase": phase,
                "model_step_ordinal": ordinal,
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )

    def _is_complete_locked(self) -> bool:
        if self._start_step is None or self._stop_step is None:
            return False
        return (
            self._state == "completed"
            and not self._errors
            and self._observed_model_step_count == self._stop_step
            and self._completed_execute_model_count == self._stop_step
            and self._completed_captured_model_step_count
            == self._stop_step - self._start_step
            and self._profiler_started_before_model_step_ordinal
            == self._start_step
            and self._profiler_stopped_after_model_step_ordinal
            == self._stop_step - 1
            and self._call_counts["profiler_start"]
            == {"attempted": 1, "succeeded": 1}
            and self._call_counts["profiler_stop"]
            == {"attempted": 1, "succeeded": 1}
            and self._call_counts["wrapper_install"]
            == {"attempted": 1, "succeeded": 1}
            and self._call_counts["wrapper_restore"]
            == {"attempted": 1, "succeeded": 1}
            and not self._wrapper_installed
            and self._original_restored
        )

    def _snapshot_locked(self) -> dict[str, Any]:
        original_identity = (
            None
            if self._original_execute_model is None
            else _callable_identity(self._original_execute_model)
        )
        wrapper_identity = (
            None
            if self._wrapper is None
            else _callable_identity(self._wrapper)
        )
        try:
            current_callable = getattr(
                self._model_runner,
                "execute_model",
                None,
            )
            current_identity = (
                None
                if current_callable is None
                else _callable_identity(current_callable)
            )
            current_matches_original = (
                self._original_execute_model is not None
                and _same_callable(
                    current_callable,
                    self._original_execute_model,
                )
            )
            current_matches_wrapper = (
                self._wrapper is not None
                and current_callable is self._wrapper
            )
        except BaseException:
            current_identity = None
            current_matches_original = False
            current_matches_wrapper = False

        start_step = self._start_step
        stop_step = self._stop_step
        expected_captured_count = (
            None
            if start_step is None or stop_step is None
            else stop_step - start_step
        )
        return {
            "schema_version": MODEL_STEP_GPU_PROFILER_PROOF_SCHEMA_VERSION,
            "hostname": self._hostname,
            "pid": self._pid,
            "device": (
                None
                if self._device_proof is None
                else dict(self._device_proof)
            ),
            "range": {
                "semantics": "[start_inclusive, stop_exclusive)",
                "ordinal_origin": (
                    "first model_runner.execute_model call after arm returns"
                ),
                "start_step": start_step,
                "stop_step": stop_step,
                "expected_captured_model_step_count": (
                    expected_captured_count
                ),
                "nvtx_range_name_format": (
                    f"{MODEL_STEP_NVTX_RANGE_PREFIX}:<zero_based_ordinal>"
                ),
            },
            "calls": {
                name: dict(counts)
                for name, counts in self._call_counts.items()
            },
            "observed_model_step_count": self._observed_model_step_count,
            "completed_execute_model_count": (
                self._completed_execute_model_count
            ),
            "completed_captured_model_step_count": (
                self._completed_captured_model_step_count
            ),
            "last_observed_model_step_ordinal": (
                self._last_observed_model_step_ordinal
            ),
            "capture_boundaries": {
                "profiler_started_before_model_step_ordinal": (
                    self._profiler_started_before_model_step_ordinal
                ),
                "profiler_stopped_after_model_step_ordinal": (
                    self._profiler_stopped_after_model_step_ordinal
                ),
            },
            "completion": {
                "complete": self._is_complete_locked(),
                "state": self._state,
                "profiler_may_be_active": self._profiler_may_be_active,
                "nvtx_range_open": self._nvtx_range_open,
            },
            "callable_identity": {
                "model_runner_type": (
                    f"{type(self._model_runner).__module__}."
                    f"{type(self._model_runner).__qualname__}"
                ),
                "original_was_instance_attribute": (
                    self._original_was_instance_attribute
                ),
                "original_execute_model": original_identity,
                "installed_wrapper": wrapper_identity,
                "current_execute_model": current_identity,
                "installed_exactly_once": (
                    self._call_counts["wrapper_install"]
                    == {"attempted": 1, "succeeded": 1}
                ),
                "wrapper_currently_installed": current_matches_wrapper,
                "original_callable_restored": (
                    self._original_restored
                    and current_matches_original
                ),
            },
            "errors": [dict(error) for error in self._errors],
        }

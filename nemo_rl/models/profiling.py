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
"""Optional four-phase GRPO capture; CUPTI remains owned by GPU workers."""

from __future__ import annotations

import importlib
import os
import uuid
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from functools import wraps
from typing import Any, Protocol

FOUR_PHASE_ENV = "NRL_NTRACE_FOUR_PHASE"
POLICY_PROFILER_CLASS_ENV = "NRL_POLICY_PROFILER_CLASS"
ROLLOUT_PROFILER_CLASS_ENV = "NRL_ROLLOUT_PROFILER_CLASS"
# vLLM ``additional_config`` entry that selects the internal profiling worker.
ROLLOUT_PROFILER_CONFIG_KEY = "nemo_rl_rollout_profiler"


class CaptureTarget(Protocol):
    def full_step_profile(self, command: str, **kwargs: Any) -> None: ...


class CaptureController(Protocol):
    """Rank-local plugin contract for policy and rollout GPU workers."""

    def configure_capture(self, **kwargs: Any) -> None: ...
    def begin_step(self, **kwargs: Any) -> Any: ...
    def finish_step(self) -> Any: ...
    def abort_step(self, *, reason: str) -> Any: ...
    def begin_phase(self, name: str, **kwargs: Any) -> Any: ...
    def end_phase(self, token: Any) -> None: ...
    def close(self) -> None: ...


class RolloutProfiler(CaptureController, Protocol):
    """vLLM workers also capture engine startup, including graph creation."""

    def begin_engine_initialization(self) -> Any: ...
    def end_engine_initialization(self, token: Any) -> None: ...


POLICY_PROFILER_METHODS = (
    "configure_capture",
    "begin_step",
    "finish_step",
    "abort_step",
    "begin_phase",
    "end_phase",
    "close",
)
ROLLOUT_PROFILER_METHODS = (
    *POLICY_PROFILER_METHODS,
    "begin_engine_initialization",
    "end_engine_initialization",
)


def four_phase_enabled() -> bool:
    """Whether the run explicitly selected the four-phase worker contract."""
    return os.environ.get(FOUR_PHASE_ENV, "0") == "1"


def selected_profiler_class(env_var: str) -> str:
    """Return the selected profiler class path, or an empty string."""
    class_path = os.environ.get(env_var, "")
    if class_path and not four_phase_enabled():
        raise ValueError(
            f"{env_var} requires {FOUR_PHASE_ENV}=1; set it to an empty string "
            "to disable profiling"
        )
    return class_path


def actor_profiler_selected(env_var: str, actor_env_vars: dict | None) -> bool:
    """Whether an actor selects a profiler; its own settings override the driver."""
    return bool((actor_env_vars or {}).get(env_var, os.environ.get(env_var)))


def load_profiler(
    class_path: str, *, rank: int, required_methods: tuple[str, ...]
) -> Any:
    """Import, validate and construct one rank-local profiler."""
    module_path, _, class_name = class_path.rpartition(".")
    if not module_path or not class_name:
        raise ValueError(
            f"Profiler class must be a fully qualified class path, got {class_path!r}"
        )

    # The plugin is an optional package that ordinary NeMo RL environments do
    # not install, so defer its import until it is selected.
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise RuntimeError(
            f"Could not import profiler module {module_path!r}. Install the "
            "profiler in this worker's environment."
        ) from exc

    profiler_class = getattr(module, class_name, None)
    if not isinstance(profiler_class, type):
        raise RuntimeError(f"{class_path!r} does not resolve to a class")

    missing_methods = [
        method
        for method in required_methods
        if not callable(getattr(profiler_class, method, None))
    ]
    if missing_methods:
        raise RuntimeError(
            f"Profiler {class_path!r} is missing required method(s): "
            f"{', '.join(missing_methods)}"
        )

    try:
        return profiler_class(rank=rank)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to initialize profiler {class_path!r} for rank {rank}"
        ) from exc


def validate_rollout_profiler_topology(
    *,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    expert_parallel_size: int,
) -> None:
    """Reject vLLM layouts not covered by the worker integration."""
    if tensor_parallel_size < 1:
        raise ValueError("Rollout profiling requires tensor_parallel_size >= 1")
    if pipeline_parallel_size != 1:
        raise ValueError("Rollout profiling requires pipeline_parallel_size=1")
    if expert_parallel_size not in (1, tensor_parallel_size):
        raise ValueError(
            "Rollout profiling requires expert_parallel_size=1 or "
            "expert_parallel_size=tensor_parallel_size"
        )


class WorkerCapture:
    """Hold rank-local phase tokens; never return them through Ray RPC."""

    def __init__(
        self, profiler: CaptureController, device_uuid: Callable[[], str]
    ) -> None:
        self.profiler = profiler
        self.device_uuid = device_uuid
        self.configured = False
        self.step_open = False
        self.phase_token: Any = None
        self.phase_open = False

    def dispatch(self, command: str, **kwargs: Any) -> None:
        if command == "configure_capture":
            self.profiler.configure_capture(device_uuid=self.device_uuid(), **kwargs)
            self.configured = True
            return
        if not self.configured:
            raise RuntimeError("Four-phase capture was not configured")
        if command == "begin_step":
            self.profiler.begin_step(**kwargs)
            self.step_open = True
        elif command == "finish_step":
            self.profiler.finish_step()
            self.step_open = False
        elif command == "abort_step":
            try:
                self.profiler.abort_step(**kwargs)
            finally:
                self.step_open = False
                self.phase_token = None
                self.phase_open = False
        elif command == "close":
            self.profiler.close()
        elif command == "begin_phase":
            # Setup, warmup and validation call the same policy methods outside
            # a measured GRPO step. Those calls do not own capture windows.
            if not self.step_open:
                return
            if self.phase_open:
                raise RuntimeError("a four-phase worker phase is already open")
            self.phase_token = self.profiler.begin_phase(**kwargs)
            self.phase_open = True
        elif command == "end_phase":
            if self.phase_open:
                try:
                    self.profiler.end_phase(self.phase_token)
                finally:
                    self.phase_token = None
                    self.phase_open = False
        else:
            raise ValueError(f"Unknown four-phase capture command: {command}")


def _abort_targets(
    targets: Iterable[CaptureTarget], error: BaseException, *, reason: str
) -> None:
    """Invalidate every entered role without masking the workload error."""
    for target in targets:
        try:
            target.full_step_profile("abort_step", reason=reason)
        except Exception as cleanup_error:
            error.add_note(f"Four-phase capture abort failed: {cleanup_error!r}")


@contextmanager
def profile_phase(
    targets: tuple[CaptureTarget, ...],
    *,
    name: str,
    phase_slot: str,
    labels: dict[str, Any] | None = None,
) -> Iterator[None]:
    """Annotate work on every participating GPU worker, including errors."""
    if not four_phase_enabled():
        yield
        return
    entered: list[CaptureTarget] = []
    try:
        for target in targets:
            entered.append(target)
            target.full_step_profile(
                "begin_phase", name=name, phase_slot=phase_slot, labels=labels
            )
        yield
        for target in reversed(entered):
            target.full_step_profile("end_phase")
    except BaseException as error:
        _abort_targets(reversed(entered), error, reason=f"{name}_failed")
        raise


def profile_policy_method(name: str, *, phase_slot: str = "logprobs") -> Callable:
    """Wrap a policy call, including reference weight swapping on workers."""

    def decorate(method: Callable) -> Callable:
        @wraps(method)
        def wrapped(self: CaptureTarget, *args: Any, **kwargs: Any) -> Any:
            with profile_phase((self,), name=name, phase_slot=phase_slot):
                return method(self, *args, **kwargs)

        return wrapped

    return decorate


def profile_refit(method: Callable) -> Callable:
    """Include policy offload, weight transfer, vLLM wake, and weight loading."""

    @wraps(method)
    def wrapped(
        policy: CaptureTarget,
        policy_generation: CaptureTarget,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        weight_version = kwargs.get("profile_weight_version")
        with profile_phase(
            (policy, policy_generation),
            name="weight_refit",
            phase_slot="refit",
            labels={"weight_version": weight_version}
            if weight_version is not None
            else None,
        ):
            return method(policy, policy_generation, *args, **kwargs)

    return wrapped


class GrpoCapture:
    """Driver coordination for GRPO policy steps and synchronous rollouts."""

    def __init__(
        self,
        policy: CaptureTarget,
        rollout: CaptureTarget,
        *,
        schedule: str,
        colocated: bool,
    ) -> None:
        self.enabled = four_phase_enabled()
        self.policy = policy
        self.rollout = rollout
        self.policy_step_open = False
        if not self.enabled:
            return
        if not os.environ.get(POLICY_PROFILER_CLASS_ENV) or not os.environ.get(
            ROLLOUT_PROFILER_CLASS_ENV
        ):
            raise ValueError(
                "Four-phase capture requires both policy and rollout profilers"
            )
        # Reuse Lens's resolved run identity without depending on span export.
        run_id = (
            os.environ.get("NTRACE_RUN_ID")
            or os.environ.get("NEMO_RL_OTEL_RUN_ID", "").strip()
            or os.environ.get("NEMO_LENS_RUN_ID", "").strip()
            or str(uuid.uuid4())
        )
        for target, role in ((policy, "policy"), (rollout, "rollout")):
            target.full_step_profile(
                "configure_capture",
                run_id=run_id,
                role=role,
                schedule=schedule,
                colocated=colocated,
            )

    def abort_all(self, error: BaseException, *, reason: str) -> None:
        """Preserve the original setup failure and invalidate both roles."""
        if self.enabled:
            _abort_targets((self.policy, self.rollout), error, reason=reason)
            self.policy_step_open = False

    def close(self) -> None:
        """Validate requested capture completion before worker teardown."""
        if not self.enabled:
            return
        first_error: Exception | None = None
        for target in (self.policy, self.rollout):
            try:
                target.full_step_profile("close")
            except Exception as error:
                if first_error is None:
                    first_error = error
                else:
                    first_error.add_note(f"Other role capture close failed: {error!r}")
        if first_error is not None:
            raise first_error

    def begin_policy_step(self, *, step_id: int, weight_version: int) -> None:
        """Open an async policy step, retaining any bootstrap refit window."""
        if self.enabled and not self.policy_step_open:
            self.policy.full_step_profile(
                "begin_step", step_id=step_id, attempt=0, weight_version=weight_version
            )
            self.policy_step_open = True

    def finish_policy_step(self) -> None:
        """Finish only a policy update that actually received a training batch."""
        if self.enabled and self.policy_step_open:
            self.policy.full_step_profile("finish_step")
            self.policy_step_open = False

    def abort_policy_step(self, *, reason: str) -> None:
        """Invalidate an open policy step when the async driver fails."""
        if self.enabled and self.policy_step_open:
            try:
                self.policy.full_step_profile("abort_step", reason=reason)
            finally:
                self.policy_step_open = False

    @contextmanager
    def step(
        self, *, step_id: int, weight_version: int, attempt: int = 0
    ) -> Iterator[None]:
        """Own one synchronous GRPO step on both roles."""
        if not self.enabled:
            yield
            return
        entered: list[CaptureTarget] = []
        try:
            for target in (self.policy, self.rollout):
                entered.append(target)
                target.full_step_profile(
                    "begin_step",
                    step_id=step_id,
                    attempt=attempt,
                    weight_version=weight_version,
                )
            yield
            for target in reversed(entered):
                target.full_step_profile("finish_step")
        except BaseException as error:
            _abort_targets(reversed(entered), error, reason="grpo_step_failed")
            raise

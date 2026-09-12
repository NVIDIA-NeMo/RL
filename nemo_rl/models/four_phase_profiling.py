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
"""Explicit GRPO capture dispatch; CUPTI remains owned by GPU workers."""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Protocol


class CaptureTarget(Protocol):
    def full_step_profile(self, command: str, **kwargs: Any) -> None: ...


class CaptureController(Protocol):
    def configure_capture(self, **kwargs: Any) -> None: ...
    def begin_step(self, **kwargs: Any) -> int: ...
    def finish_step(self) -> Any: ...
    def abort_step(self, *, reason: str) -> Any: ...
    def begin_phase(self, name: str, **kwargs: Any) -> Any: ...
    def end_phase(self, token: Any) -> None: ...
    def close(self) -> None: ...


def four_phase_enabled() -> bool:
    """Whether the run explicitly selected the four-phase worker contract."""
    return os.environ.get("NRL_NTRACE_FOUR_PHASE", "0") == "1"


class WorkerCapture:
    """Hold rank-local phase tokens; never return them through Ray RPC."""

    def __init__(self, profiler: CaptureController) -> None:
        self.profiler = profiler
        self.step_open = False
        self.phase_token: Any = None
        self.phase_open = False

    def dispatch(self, command: str, **kwargs: Any) -> None:
        if command == "configure_capture":
            self.profiler.configure_capture(**kwargs)
        elif command == "begin_step":
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
            # Setup/validation can call the same policy methods outside a
            # measured GRPO step. Those calls do not own capture windows.
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
                "begin_phase",
                name=name,
                phase_slot=phase_slot,
                labels=labels,
            )
        yield
        for target in reversed(entered):
            target.full_step_profile("end_phase")
    except BaseException as error:
        for target in reversed(entered):
            try:
                target.full_step_profile("abort_step", reason=f"{name}_failed")
            except Exception as cleanup_error:
                error.add_note(f"Four-phase capture abort failed: {cleanup_error!r}")
        raise


def profile_policy_method(name: str, *, phase_slot: str = "logprobs") -> Callable:
    """Wrap policy logprobs, including reference weight swapping on workers."""

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
    """Driver coordination for policy steps and synchronous rollout steps."""

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
        self.schedule = schedule
        self.policy_step_open = False
        if not self.enabled:
            return
        if not os.environ.get("NRL_POLICY_PROFILER_CLASS") or not os.environ.get(
            "NRL_ROLLOUT_PROFILER_CLASS"
        ):
            raise ValueError(
                "Four-phase capture requires both policy and rollout profilers"
            )
        run_id = os.environ.get("NTRACE_RUN_ID") or str(uuid.uuid4())
        for target, role in ((policy, "policy"), (rollout, "rollout")):
            target.full_step_profile(
                "configure_capture",
                run_id=run_id,
                role=role,
                schedule=schedule,
                colocated=colocated,
            )

    def abort_all(self, error: BaseException, *, reason: str) -> None:
        """Preserve the original setup failure and invalidate both worker roles."""
        if not self.enabled:
            return
        for target in (self.policy, self.rollout):
            try:
                target.full_step_profile("abort_step", reason=reason)
            except Exception as cleanup_error:
                error.add_note(f"Capture abort also failed: {cleanup_error!r}")
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
                "begin_step",
                step_id=step_id,
                attempt=0,
                weight_version=weight_version,
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
        if not self.enabled:
            yield
            return
        targets = (
            (self.policy, self.rollout) if self.schedule == "sync" else (self.policy,)
        )
        entered: list[CaptureTarget] = []
        try:
            for target in targets:
                entered.append(target)
                if target is self.policy and self.policy_step_open:
                    continue
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
            for target in reversed(entered):
                try:
                    target.full_step_profile("abort_step", reason="grpo_step_failed")
                except Exception as cleanup_error:
                    error.add_note(
                        f"Four-phase capture abort failed: {cleanup_error!r}"
                    )
            raise
        finally:
            self.policy_step_open = False

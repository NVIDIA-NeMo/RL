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

import time
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import ray


class SGLangRefitTimeoutError(TimeoutError):
    """Raised when the single SGLang refit deadline expires."""


@dataclass(frozen=True)
class SGLangRefitDeadline:
    """One monotonic orchestration deadline shared by an SGLang refit.

    Absolute monotonic timestamps are process-local, so callers must pass only
    ``remaining()`` seconds across a Ray boundary. The receiving process then
    creates its own deadline from that reduced budget. Blocking APIs receive
    that budget where they expose a timeout; the driver wait remains the
    authority for synchronous backend work that cannot be interrupted.
    """

    timeout_s: float
    _expires_at: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.timeout_s <= 0:
            raise ValueError(
                f"SGLang refit timeout must be positive, got {self.timeout_s}"
            )
        object.__setattr__(self, "_expires_at", time.monotonic() + self.timeout_s)

    def remaining(self, stage: str) -> float:
        """Return the remaining seconds or raise a stage-specific timeout."""
        remaining_s = self._expires_at - time.monotonic()
        if remaining_s <= 0:
            raise SGLangRefitTimeoutError(
                f"SGLang refit timed out after {self.timeout_s:.3f}s while {stage}"
            )
        return remaining_s

    def remaining_or_zero(self) -> float:
        """Return the remaining budget without raising."""
        return max(0.0, self._expires_at - time.monotonic())

    def remaining_timedelta(self, stage: str) -> timedelta:
        """Return the remaining budget in the form torch collectives expect."""
        return timedelta(seconds=self.remaining(stage))

    def ray_get(
        self,
        refs: Any,
        *,
        stage: str,
        cancel_on_error: bool = False,
    ) -> Any:
        """Wait for Ray refs without resetting the refit timeout.

        When cancellation is requested, queued or cooperative actor work is
        cancelled best-effort. Callers still need transport-specific cleanup
        for operations that may already have entered C++ or HTTP code.
        """
        try:
            return ray.get(refs, timeout=self.remaining(stage))
        except ray.exceptions.GetTimeoutError as exc:
            if cancel_on_error:
                cancel_ray_refs(refs)
            raise SGLangRefitTimeoutError(
                f"SGLang refit timed out after {self.timeout_s:.3f}s while {stage}"
            ) from exc
        except Exception:
            if cancel_on_error:
                cancel_ray_refs(refs)
            raise


def cancel_ray_refs(refs: Any) -> None:
    """Cancel one or more Ray refs without masking the triggering error."""
    if refs is None:
        return
    if not isinstance(refs, (list, tuple, set)):
        refs = [refs]
    for ref in refs:
        try:
            ray.cancel(ref, force=False, recursive=True)
        except BaseException:
            # The ref may already be complete or may not be cancellable (for
            # example, a running actor method). Transport cleanup handles the
            # corresponding external state.
            continue

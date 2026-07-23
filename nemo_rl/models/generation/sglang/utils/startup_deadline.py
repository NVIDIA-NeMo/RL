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
from typing import Any

import ray

from nemo_rl.models.generation.sglang.utils.refit_deadline import cancel_ray_refs


class SGLangStartupTimeoutError(TimeoutError):
    """Raised when initial SGLang engine bootstrap exceeds its deadline."""


@dataclass(frozen=True)
class SGLangStartupDeadline:
    """One monotonic deadline shared by every initial engine-startup stage."""

    timeout_s: float
    _expires_at: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.timeout_s <= 0:
            raise ValueError(
                f"SGLang engine startup timeout must be positive, got {self.timeout_s}"
            )
        object.__setattr__(self, "_expires_at", time.monotonic() + self.timeout_s)

    def remaining(self, stage: str) -> float:
        remaining_s = self._expires_at - time.monotonic()
        if remaining_s <= 0:
            raise SGLangStartupTimeoutError(
                "SGLang engine startup timed out after "
                f"{self.timeout_s:.3f}s while {stage}"
            )
        return remaining_s

    def ray_get(
        self,
        refs: Any,
        *,
        stage: str,
        cancel_on_error: bool = False,
    ) -> Any:
        try:
            return ray.get(refs, timeout=self.remaining(stage))
        except ray.exceptions.GetTimeoutError as exc:
            if cancel_on_error:
                cancel_ray_refs(refs)
            raise SGLangStartupTimeoutError(
                "SGLang engine startup timed out after "
                f"{self.timeout_s:.3f}s while {stage}"
            ) from exc
        except BaseException:
            if cancel_on_error:
                cancel_ray_refs(refs)
            raise

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
"""Optional rollout profiler integration."""

from __future__ import annotations

import importlib
import os
from typing import Any, Protocol, cast

ROLLOUT_PROFILER_CLASS_ENV = "NRL_ROLLOUT_PROFILER_CLASS"


class RolloutProfiler(Protocol):
    """Lifecycle contract for profiling complete synchronous rollouts."""

    def begin_engine_initialization(self) -> Any: ...

    def end_engine_initialization(self, token: Any) -> None: ...

    def begin_rollout(self, *, step_id: int | str) -> None: ...

    def finish_rollout(self) -> None: ...

    def abort_rollout(self, *, reason: str) -> None: ...

    def close(self) -> None: ...


def load_rollout_profiler(*, rank: int) -> RolloutProfiler | None:
    """Load the rollout profiler selected by ``NRL_ROLLOUT_PROFILER_CLASS``.

    The environment variable must contain a fully qualified class path. The
    class is imported only when the variable is non-empty, instantiated with
    the generation-worker rank, and validated against :class:`RolloutProfiler`.

    Args:
        rank: Dense rank of the synchronous generation worker.

    Returns:
        The configured profiler, or ``None`` when profiling is disabled.

    Raises:
        ValueError: If the configured class path is malformed.
        RuntimeError: If the class cannot be imported, does not implement the
            profiler contract, or fails during initialization.
    """
    class_path = os.environ.get(ROLLOUT_PROFILER_CLASS_ENV, "")
    if not class_path:
        return None

    module_path, separator, class_name = class_path.rpartition(".")
    if not separator or not module_path or not class_name:
        raise ValueError(
            f"{ROLLOUT_PROFILER_CLASS_ENV} must be a fully qualified class path, "
            f"got {class_path!r}"
        )

    # The selected profiler may be an optional package that ordinary NeMo RL
    # environments do not install, so defer its import until it is configured.
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise RuntimeError(
            f"Could not import rollout profiler module {module_path!r} from "
            f"{ROLLOUT_PROFILER_CLASS_ENV}={class_path!r}. Install the profiler "
            "in the vLLM generation-worker environment."
        ) from exc

    profiler_class = getattr(module, class_name, None)
    if not isinstance(profiler_class, type):
        raise RuntimeError(
            f"{ROLLOUT_PROFILER_CLASS_ENV}={class_path!r} does not resolve to a class"
        )

    required_methods = (
        "begin_engine_initialization",
        "end_engine_initialization",
        "begin_rollout",
        "finish_rollout",
        "abort_rollout",
        "close",
    )
    missing_methods = [
        method
        for method in required_methods
        if not callable(getattr(profiler_class, method, None))
    ]
    if missing_methods:
        raise RuntimeError(
            f"Rollout profiler {class_path!r} is missing required method(s): "
            f"{', '.join(missing_methods)}"
        )

    try:
        profiler = profiler_class(rank=rank)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to initialize rollout profiler {class_path!r} for rank {rank}"
        ) from exc
    return cast(RolloutProfiler, profiler)

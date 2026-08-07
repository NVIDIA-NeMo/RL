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
"""Optional policy-training profiler integration."""

import importlib
import os
from typing import Protocol, cast


class PolicyProfiler(Protocol):
    """Lifecycle contract for profiling complete policy-training steps."""

    def begin_train_step(self) -> None: ...

    def finish_train_step(self) -> None: ...

    def abort_train_step(self, *, reason: str) -> None: ...

    def close(self) -> None: ...


def load_policy_profiler(*, rank: int) -> PolicyProfiler | None:
    """Load the policy profiler selected by ``NRL_POLICY_PROFILER_CLASS``.

    The environment variable must contain a fully qualified class path. The
    class is imported only when the variable is non-empty, instantiated with
    the distributed rank, and validated against :class:`PolicyProfiler`.

    Args:
        rank: Distributed rank of the policy worker.

    Returns:
        The configured profiler, or ``None`` when profiling is disabled.

    Raises:
        ValueError: If the configured class path is malformed.
        RuntimeError: If the class cannot be imported, does not implement the
            profiler contract, or fails during initialization.
    """
    class_path = os.environ.get("NRL_POLICY_PROFILER_CLASS", "")
    if not class_path:
        return None

    module_path, separator, class_name = class_path.rpartition(".")
    if not separator or not module_path or not class_name:
        raise ValueError(
            "NRL_POLICY_PROFILER_CLASS must be a fully qualified class path, "
            f"got {class_path!r}"
        )

    # The selected profiler may be an optional package that ordinary NeMo RL
    # environments do not install, so defer its import until it is configured.
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise RuntimeError(
            f"Could not import policy profiler module {module_path!r} from "
            f"NRL_POLICY_PROFILER_CLASS={class_path!r}. Install the profiler "
            "in the policy-worker environment."
        ) from exc

    profiler_class = getattr(module, class_name, None)
    if not isinstance(profiler_class, type):
        raise RuntimeError(
            f"NRL_POLICY_PROFILER_CLASS={class_path!r} does not resolve to a class"
        )

    required_methods = (
        "begin_train_step",
        "finish_train_step",
        "abort_train_step",
        "close",
    )
    missing_methods = [
        method
        for method in required_methods
        if not callable(getattr(profiler_class, method, None))
    ]
    if missing_methods:
        raise RuntimeError(
            f"Policy profiler {class_path!r} is missing required method(s): "
            f"{', '.join(missing_methods)}"
        )

    try:
        profiler = profiler_class(rank=rank)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to initialize policy profiler {class_path!r} for rank {rank}"
        ) from exc
    return cast(PolicyProfiler, profiler)

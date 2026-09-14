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
"""Shared helpers for optional profiler integrations."""

import importlib
import os
from typing import Any


def load_profiler(
    *,
    env_var: str,
    profiler_kind: str,
    required_methods: tuple[str, ...],
    install_environment: str,
    rank: int,
) -> Any | None:
    """Load and validate an optional rank-local profiler class."""
    class_path = os.environ.get(env_var, "")
    if not class_path:
        return None

    module_path, separator, class_name = class_path.rpartition(".")
    if not separator or not module_path or not class_name:
        raise ValueError(
            f"{env_var} must be a fully qualified class path, got {class_path!r}"
        )

    # The selected profiler may be an optional package that ordinary NeMo RL
    # environments do not install, so defer its import until it is configured.
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise RuntimeError(
            f"Could not import {profiler_kind} module {module_path!r} from "
            f"{env_var}={class_path!r}. Install the profiler in the "
            f"{install_environment} environment."
        ) from exc

    profiler_class = getattr(module, class_name, None)
    if not isinstance(profiler_class, type):
        raise RuntimeError(f"{env_var}={class_path!r} does not resolve to a class")

    missing_methods = [
        method
        for method in required_methods
        if not callable(getattr(profiler_class, method, None))
    ]
    if missing_methods:
        raise RuntimeError(
            f"{profiler_kind.capitalize()} {class_path!r} is missing required "
            f"method(s): {', '.join(missing_methods)}"
        )

    try:
        return profiler_class(rank=rank)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to initialize {profiler_kind} {class_path!r} for rank {rank}"
        ) from exc

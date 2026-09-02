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

from typing import Any, Protocol, cast

from nemo_rl.models.profiling import load_profiler

ROLLOUT_PROFILER_CLASS_ENV = "NRL_ROLLOUT_PROFILER_CLASS"


class RolloutProfiler(Protocol):
    """Lifecycle contract for profiling complete rollouts."""

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
        rank: Dense rank of the vLLM GPU worker.

    Returns:
        The configured profiler, or ``None`` when profiling is disabled.

    Raises:
        ValueError: If the configured class path is malformed.
        RuntimeError: If the class cannot be imported, does not implement the
            profiler contract, or fails during initialization.
    """
    return cast(
        RolloutProfiler | None,
        load_profiler(
            env_var=ROLLOUT_PROFILER_CLASS_ENV,
            profiler_kind="rollout profiler",
            required_methods=(
                "begin_engine_initialization",
                "end_engine_initialization",
                "begin_rollout",
                "finish_rollout",
                "abort_rollout",
                "close",
            ),
            install_environment="vLLM generation-worker",
            rank=rank,
        ),
    )


def validate_rollout_profiler_topology(
    *,
    class_path: str,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    expert_parallel_size: int,
) -> None:
    """Reject profiler configurations not covered by the worker integration."""
    if not class_path:
        return
    if tensor_parallel_size < 1:
        raise ValueError("Rollout profiling requires tensor_parallel_size >= 1")
    if pipeline_parallel_size != 1:
        raise ValueError("Rollout profiling requires pipeline_parallel_size=1")
    if expert_parallel_size not in (1, tensor_parallel_size):
        raise ValueError(
            "Rollout profiling requires expert_parallel_size=1 or "
            "expert_parallel_size=tensor_parallel_size"
        )

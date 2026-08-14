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

from typing import Protocol, cast

from nemo_rl.models.profiling import load_profiler

POLICY_PROFILER_CLASS_ENV = "NRL_POLICY_PROFILER_CLASS"


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
    return cast(
        PolicyProfiler | None,
        load_profiler(
            env_var=POLICY_PROFILER_CLASS_ENV,
            profiler_kind="policy profiler",
            required_methods=(
                "begin_train_step",
                "finish_train_step",
                "abort_train_step",
                "close",
            ),
            install_environment="policy-worker",
            rank=rank,
        ),
    )

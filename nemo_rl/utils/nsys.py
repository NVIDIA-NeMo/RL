# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import atexit
import os
from typing import Protocol

import rich

NRL_NSYS_WORKER_PATTERNS = os.environ.get("NRL_NSYS_WORKER_PATTERNS", "")
NRL_NSYS_PROFILE_STEP_RANGE = os.environ.get("NRL_NSYS_PROFILE_STEP_RANGE", "")


class ProfilablePolicy(Protocol):
    def start_gpu_profiling(self) -> None: ...

    def stop_gpu_profiling(self) -> None: ...


def maybe_gpu_profile_step(policy: ProfilablePolicy, step: int):
    assert not (bool(NRL_NSYS_WORKER_PATTERNS) ^ bool(NRL_NSYS_PROFILE_STEP_RANGE)), (
        "Either both NRL_NSYS_WORKER_PATTERNS and NRL_NSYS_PROFILE_STEP_RANGE must be set, or neither. See https://github.com/NVIDIA/NeMo-RL/tree/main/docs/nsys-profiling.md for more details."
    )

    if not NRL_NSYS_WORKER_PATTERNS:
        return

    NSYS_START_STEP, NSYS_STOP_STEP = NRL_NSYS_PROFILE_STEP_RANGE.split(":")
    try:
        NSYS_START_STEP = int(NSYS_START_STEP)
        NSYS_STOP_STEP = int(NSYS_STOP_STEP)
    except ValueError as e:
        raise ValueError(
            f"Error parsing NRL_NSYS_PROFILE_STEP_RANGE: {str(e)}. "
            "Please ensure the format is 'start:stop' where both values are integers. "
            "See https://github.com/NVIDIA/NeMo-RL/tree/main/docs/nsys-profiling.md for more details."
        ) from e

    assert NSYS_START_STEP < NSYS_STOP_STEP, (
        f"{NRL_NSYS_PROFILE_STEP_RANGE=} must be a non-empty range"
    )
    assert NSYS_START_STEP >= 1, (
        f"The start step in {NRL_NSYS_PROFILE_STEP_RANGE=} must be >= 1"
    )

    # Use slice syntax of left inclusive and right exclusive
    if NSYS_START_STEP <= step < NSYS_STOP_STEP:
        if not getattr(policy, "__NRL_PROFILE_STARTED", False):
            rich.print(
                f"[bold red]Starting GPU profiling for {policy} for step {step}[/bold red]"
            )
            policy.start_gpu_profiling()
            policy.__NRL_PROFILE_STARTED = True

            def stop_profiler_on_exit():
                rich.print(
                    f"[bold red]Stopping GPU profiling on exit for {policy} for step {step}[/bold red]"
                )
                policy.stop_gpu_profiling()

            atexit.register(stop_profiler_on_exit)
    else:
        if getattr(policy, "__NRL_PROFILE_STARTED", False):
            rich.print(
                f"[bold red]Stopping GPU profiling for {policy} for step {step}[/bold red]"
            )
            policy.stop_gpu_profiling()
            policy.__NRL_PROFILE_STARTED = False

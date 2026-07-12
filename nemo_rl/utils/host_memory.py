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

import os
from collections.abc import Mapping
from typing import NamedTuple, Optional

import psutil

_GIB = 1024**3


class HostMemorySnapshot(NamedTuple):
    process_rss_gib: float
    system_available_gib: float


def emit_structured_stdout(message: str) -> None:
    """Emit one fail-visible diagnostic line without changing runtime semantics."""
    try:
        print(message, flush=True)
    except Exception:
        pass


def _get_host_memory_snapshot() -> Optional[HostMemorySnapshot]:
    try:
        process = psutil.Process(os.getpid())
        return HostMemorySnapshot(
            process_rss_gib=psutil.Process.memory_info(process).rss / _GIB,
            system_available_gib=psutil.virtual_memory().available / _GIB,
        )
    except Exception:
        return None


def emit_host_memory_event(
    *,
    event: str,
    phase: str,
    fields: Optional[Mapping[str, object]] = None,
    before_snapshot: Optional[HostMemorySnapshot] = None,
    include_deltas: bool = False,
) -> Optional[HostMemorySnapshot]:
    """Capture host memory and emit a single structured, best-effort event."""
    prefix = f"event={event} phase={phase}"
    if fields:
        prefix += "".join(f" {key}={value}" for key, value in fields.items())

    snapshot = _get_host_memory_snapshot()
    if snapshot is None:
        message = f"{prefix} process_rss_gib=unavailable"
        if include_deltas:
            message += " process_rss_delta_gib=unavailable"
        message += " system_available_gib=unavailable"
        if include_deltas:
            message += " system_available_delta_gib=unavailable"
        emit_structured_stdout(message)
        return None

    message = f"{prefix} process_rss_gib={snapshot.process_rss_gib:.3f}"
    if include_deltas:
        rss_delta = (
            "unavailable"
            if before_snapshot is None
            else f"{snapshot.process_rss_gib - before_snapshot.process_rss_gib:.3f}"
        )
        message += f" process_rss_delta_gib={rss_delta}"
    message += f" system_available_gib={snapshot.system_available_gib:.3f}"
    if include_deltas:
        available_delta = (
            "unavailable"
            if before_snapshot is None
            else f"{snapshot.system_available_gib - before_snapshot.system_available_gib:.3f}"
        )
        message += f" system_available_delta_gib={available_delta}"
    emit_structured_stdout(message)
    return snapshot

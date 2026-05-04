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
"""Optional observability layer for the data plane.

Wraps any :class:`DataPlaneClient` with per-op metrics and a pluggable
sink. See ``research/data_plane_observability.md`` for the design.
"""

from nemo_rl.data_plane.observability.middleware import MetricsDataPlaneClient
from nemo_rl.data_plane.observability.sinks import (
    InMemorySink,
    LogSink,
    MetricsSink,
    build_sink,
)

__all__ = [
    "InMemorySink",
    "LogSink",
    "MetricsDataPlaneClient",
    "MetricsSink",
    "build_sink",
]

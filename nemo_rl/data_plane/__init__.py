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
"""NeMo-RL data-plane package.

The public surface is intentionally tiny: an ABC, a meta dataclass, a
config TypedDict, and a factory. Everything else is an implementation
detail of a specific adapter.
"""

from nemo_rl.data_plane.factory import build_data_plane_client
from nemo_rl.data_plane.interfaces import (
    DataPlaneClient,
    DataPlaneConfig,
    KVBatchMeta,
)


def materialize(*args, **kwargs):
    """Lazy-import codec to avoid hard deps during module import.

    Some worker processes import ``nemo_rl.data_plane`` even when data-plane
    functionality is not used. Importing codec eagerly pulls optional
    dependencies (e.g., tensordict) and can fail in lightweight worker envs.
    """
    from nemo_rl.data_plane.codec import materialize as _materialize

    return _materialize(*args, **kwargs)


def __getattr__(name):
    if name in {"MetricsDataPlaneClient", "log_event"}:
        from nemo_rl.data_plane.observability import MetricsDataPlaneClient, log_event

        return {
            "MetricsDataPlaneClient": MetricsDataPlaneClient,
            "log_event": log_event,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "DataPlaneClient",
    "DataPlaneConfig",
    "KVBatchMeta",
    "MetricsDataPlaneClient",
    "build_data_plane_client",
    "log_event",
    "materialize",
]

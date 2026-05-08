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

from nemo_rl.data_plane.codec import materialize
from nemo_rl.data_plane.factory import build_data_plane_client
from nemo_rl.data_plane.interfaces import (
    DataPlaneClient,
    DataPlaneConfig,
    KVBatchMeta,
)
from nemo_rl.data_plane.observability import MetricsDataPlaneClient, print_event

__all__ = [
    "DataPlaneClient",
    "DataPlaneConfig",
    "KVBatchMeta",
    "MetricsDataPlaneClient",
    "build_data_plane_client",
    "materialize",
    "print_event",
]

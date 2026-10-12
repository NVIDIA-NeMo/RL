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

"""Controller-side adapter for an externally managed stock vLLM server."""

from nemo_rl.models.generation.remote_vllm.client import RemoteVllmClient
from nemo_rl.models.generation.remote_vllm.config import (
    RemoteVllmRefitConfig,
    RemoteVllmServiceConfig,
    RemoteVllmServiceInfo,
)
from nemo_rl.models.generation.remote_vllm.preflight import (
    preflight_remote_vllm_service,
)
from nemo_rl.models.generation.remote_vllm.remote_vllm_generation import (
    RemoteVllmGeneration,
)

__all__ = [
    "RemoteVllmRefitConfig",
    "RemoteVllmClient",
    "RemoteVllmServiceConfig",
    "RemoteVllmServiceInfo",
    "RemoteVllmGeneration",
    "preflight_remote_vllm_service",
]

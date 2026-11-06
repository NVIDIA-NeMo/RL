# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Select the generation-only tracing extension without cross-venv imports."""

import sys
from typing import Any


def setup() -> None:
    from nemo_rl.distributed.ray_worker_setup import ray_worker_process_setup

    ray_worker_process_setup()
    if "vllm_worker" not in sys.prefix:
        return
    from nemo_rl.models.generation.vllm import vllm_worker

    original = vllm_worker._validate_worker_extension_cls

    def extension(kwargs: dict[str, Any], required: str) -> None:
        original(kwargs, required)
        kwargs["worker_extension_cls"] = (
            "validation.pr3.kl_replay.generation_profile.TraceExtension"
        )

    vllm_worker._validate_worker_extension_cls = extension

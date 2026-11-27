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

"""Register the project package for Ray's AutoModel workers."""

import os
import shlex
from pathlib import Path

from nemo_rl.distributed.ray_actor_environment_registry import (
    ACTOR_ENVIRONMENT_REGISTRY,
)
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES

WORKER = "logra.worker.LoGRAWorker"
GENERATION_WORKER = "logra.generation.CuaGenerationWorker"


def register_actor_environments(*, serial_refit: bool = False) -> None:
    if serial_refit:
        # Expert exports yield views across packed-buffer boundaries. One stream
        # keeps those views ordered with their DTensor all-gathers.
        os.environ["NRL_REFIT_NUM_BUFFERS"] = "1"
        forwarded = {
            key.strip()
            for key in os.environ.get("VLLM_RAY_EXTRA_ENV_VARS_TO_COPY", "").split(",")
            if key.strip()
        }
        forwarded.add("NRL_REFIT_NUM_BUFFERS")
        os.environ["VLLM_RAY_EXTRA_ENV_VARS_TO_COPY"] = ",".join(sorted(forwarded))
    project = Path(__file__).resolve().parents[1]
    if serial_refit:
        # Keep vLLM in its native isolated environment while exposing this recipe.
        ACTOR_ENVIRONMENT_REGISTRY[GENERATION_WORKER] = ACTOR_ENVIRONMENT_REGISTRY[
            "nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker"
        ]
        search_path = os.environ.get("PYTHONPATH", "").split(os.pathsep)
        os.environ["PYTHONPATH"] = os.pathsep.join(
            dict.fromkeys([str(project), *filter(None, search_path)])
        )
    ACTOR_ENVIRONMENT_REGISTRY[WORKER] = (
        PY_EXECUTABLES.SYSTEM
        if os.environ.get("NEMO_RL_PY_EXECUTABLES_SYSTEM") == "1"
        else f"uv run --locked --directory {shlex.quote(str(project))}"
    )

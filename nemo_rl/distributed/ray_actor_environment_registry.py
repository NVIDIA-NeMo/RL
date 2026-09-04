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

import os

from nemo_rl.distributed.actor_environments import ACTOR_ENVIRONMENTS
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES, uv_py_executable

# NEMO_RL_PY_EXECUTABLES_SYSTEM=1 (single-environment images such as Dockerfile.ngc_pytorch)
# runs every actor on the driver's interpreter instead of a per-actor uv venv.
USE_SYSTEM_EXECUTABLE = os.environ.get("NEMO_RL_PY_EXECUTABLES_SYSTEM", "0") == "1"

# Actor FQN -> the py_executable its workers launch under. The extras come from
# nemo_rl.distributed.actor_environments, which docker/Dockerfile also reads to
# pre-build one venv per actor into the image.
ACTOR_ENVIRONMENT_REGISTRY: dict[str, str] = {
    actor_fqn: PY_EXECUTABLES.SYSTEM
    if extras is None or USE_SYSTEM_EXECUTABLE
    else uv_py_executable(extras)
    for actor_fqn, extras in ACTOR_ENVIRONMENTS.items()
}


def get_actor_python_env(actor_class_fqn: str) -> str:
    if actor_class_fqn in ACTOR_ENVIRONMENT_REGISTRY:
        return ACTOR_ENVIRONMENT_REGISTRY[actor_class_fqn]
    else:
        raise ValueError(
            f"No actor environment registered for {actor_class_fqn}. "
            f"You're attempting to create an actor ({actor_class_fqn}) "
            "without specifying a python environment for it. Please either "
            "add the actor to ACTOR_ENVIRONMENTS in nemo_rl/distributed/actor_environments.py, "
            "register it at runtime with ACTOR_ENVIRONMENT_REGISTRY[fqn] = <py_executable> "
            "(the path for workers defined outside this repo), "
            "or pass a py_executable to the RayWorkerBuilder. If you're unsure about which "
            "environment to use, a good default is PY_EXECUTABLES.SYSTEM for ray actors that "
            "don't have special dependencies. If you do have special dependencies (say, you're "
            "adding a new generation framework or training backend), you'll need to specify the "
            "appropriate environment. See uv.md for more details."
        )

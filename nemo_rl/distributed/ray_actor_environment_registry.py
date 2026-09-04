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
import tomllib
from pathlib import Path

from nemo_rl.distributed.virtual_cluster import (
    PY_EXECUTABLES,
    git_root,
    uv_py_executable,
)

# NEMO_RL_PY_EXECUTABLES_SYSTEM=1 (single-environment images such as Dockerfile.ngc_pytorch)
# runs every actor on the driver's interpreter instead of a per-actor uv venv.
USE_SYSTEM_EXECUTABLE = os.environ.get("NEMO_RL_PY_EXECUTABLES_SYSTEM", "0") == "1"


def _load_actor_environments() -> dict[str, str]:
    """Build actor FQN -> py_executable from pyproject.toml's [tool.nemo_rl.actor_environments]."""
    with open(Path(git_root) / "pyproject.toml", "rb") as f:
        pyproject = tomllib.load(f)
    declared_extras = set(pyproject["project"]["optional-dependencies"])
    registry: dict[str, str] = {}
    for actor_fqn, extras in pyproject["tool"]["nemo_rl"]["actor_environments"].items():
        if extras == "system":
            registry[actor_fqn] = PY_EXECUTABLES.SYSTEM
            continue
        unknown = set(extras) - declared_extras
        if unknown:
            raise ValueError(
                f"[tool.nemo_rl.actor_environments] {actor_fqn!r} names extras "
                f"{sorted(unknown)} that are not in [project.optional-dependencies]"
            )
        registry[actor_fqn] = (
            PY_EXECUTABLES.SYSTEM if USE_SYSTEM_EXECUTABLE else uv_py_executable(extras)
        )
    return registry


ACTOR_ENVIRONMENT_REGISTRY: dict[str, str] = _load_actor_environments()


def get_actor_python_env(actor_class_fqn: str) -> str:
    if actor_class_fqn in ACTOR_ENVIRONMENT_REGISTRY:
        return ACTOR_ENVIRONMENT_REGISTRY[actor_class_fqn]
    else:
        raise ValueError(
            f"No actor environment registered for {actor_class_fqn}. "
            f"You're attempting to create an actor ({actor_class_fqn}) "
            "without specifying a python environment for it. Please either"
            "add the actor to the [tool.nemo_rl.actor_environments] table in pyproject.toml "
            "or pass a py_executable to the RayWorkerBuilder. If you're unsure about which "
            "environment to use, a good default is PY_EXECUTABLES.SYSTEM for ray actors that "
            "don't have special dependencies. If you do have special dependencies (say, you're "
            "adding a new generation framework or training backend), you'll need to specify the "
            "appropriate environment. See uv.md for more details."
        )

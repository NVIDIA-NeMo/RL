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

"""Resolve the interpreter and service binaries the managed Dynamo runtime launches."""

import os
import shutil
from pathlib import Path

from nemo_rl.distributed.ray_actor_environment_registry import get_actor_python_env
from nemo_rl.utils.venvs import create_local_venv

DYNAMO_WORKER_FQN = "nemo_rl.models.generation.dynamo.dynamo_worker.DynamoVllmWorker"


def get_dynamo_python() -> str:
    """Return the Dynamo actor venv's Python on this node, building the venv if needed.

    The driver launches ``dynamo.frontend`` with it. The ``dynamo.vllm`` workers
    use their actor's own interpreter, which is the same venv.
    """
    py_executable = get_actor_python_env(DYNAMO_WORKER_FQN)
    if py_executable.startswith("uv"):
        return create_local_venv(py_executable, DYNAMO_WORKER_FQN)
    return py_executable


def get_dynamo_service_executable(name: str) -> str:
    """Return the etcd or nats-server binary the runtime starts.

    Looks in ``NEMO_RL_DYNAMO_BIN_DIR`` when it is set, otherwise on ``PATH``.
    """
    bin_dir = os.environ.get("NEMO_RL_DYNAMO_BIN_DIR")
    if bin_dir:
        executable = Path(bin_dir).expanduser() / name
        if executable.is_file() and os.access(executable, os.X_OK):
            return str(executable)
        raise FileNotFoundError(
            f"{name} is not an executable file in NEMO_RL_DYNAMO_BIN_DIR={bin_dir}."
        )
    found = shutil.which(name)
    if found is None:
        raise FileNotFoundError(
            f"{name} is not on PATH. Install it, or set NEMO_RL_DYNAMO_BIN_DIR to "
            "the directory that holds etcd and nats-server."
        )
    return found

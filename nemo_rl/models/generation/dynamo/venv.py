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
from nemo_rl.utils.venvs import create_local_venv_on_each_node

# Every vLLM worker venv carries the locked ai-dynamo release, so the frontend
# runs from the async worker's venv on the same release as every engine.
VLLM_ASYNC_WORKER_FQN = (
    "nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker"
)


def get_dynamo_python() -> str:
    """Return the vLLM async worker's Python, which carries ``ai-dynamo``.

    The driver launches ``dynamo.frontend`` with it. A uv-managed venv is
    resolved as the workers' own is: an existing one is reused as-is, and a
    missing one is built once on each node. Re-syncing it here would change a
    venv the engines are already running from.
    """
    py_executable = get_actor_python_env(VLLM_ASYNC_WORKER_FQN)
    if py_executable.startswith("uv"):
        return create_local_venv_on_each_node(py_executable, VLLM_ASYNC_WORKER_FQN)
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

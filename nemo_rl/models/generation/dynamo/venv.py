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

"""Resolve executables installed by the optional Dynamo environment."""

import os
from pathlib import Path


def get_dynamo_venv_dir() -> Path:
    """Return the configured Dynamo virtual-environment directory."""
    configured = os.environ.get("NEMO_RL_DYNAMO_VENV_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[4]
    return repo_root / "venvs" / "dynamo"


def get_dynamo_executable(name: str) -> str:
    """Return an executable in the Dynamo environment, failing if absent."""
    executable = get_dynamo_venv_dir() / "bin" / name
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise FileNotFoundError(
            f"Dynamo executable {executable} is unavailable. Build with "
            "BUILD_DYNAMO=1 or run docker/dynamo/install.sh with "
            "NEMO_RL_DYNAMO_VENV_DIR set."
        )
    return str(executable)


def get_dynamo_python() -> str:
    """Return the validated Python interpreter for ``ai-dynamo``."""
    return get_dynamo_executable("python")

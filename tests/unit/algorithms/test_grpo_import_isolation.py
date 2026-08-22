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
"""Guards the lazy-import win: these must stay out of a plain grpo import.

Without this, a top-level ``import wandb`` added anywhere in the transitive
graph silently undoes the deferral and nothing fails.
"""

import subprocess
import sys

import pytest

# Optional integrations that importing nemo_rl.algorithms.grpo must not pull in.
# ray.scripts.scripts is the CLI entrypoint, which drags the dashboard stack
# (fastapi, uvicorn) along with it.
DEFERRED_MODULES = [
    "wandb",
    "mlflow",
    "swanlab",
    "matplotlib.pyplot",
    "fastapi",
    "uvicorn",
]


@pytest.mark.parametrize("module_name", DEFERRED_MODULES)
def test_grpo_import_stays_lazy(module_name):
    """Importing grpo must not import ``module_name``."""
    # A subprocess, because the module may already be in this process's
    # sys.modules from an unrelated test.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "import nemo_rl.algorithms.grpo  # noqa: F401\n"
            f"loaded = {module_name!r} in sys.modules\n"
            "sys.exit(1 if loaded else 0)\n",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"'import nemo_rl.algorithms.grpo' loaded {module_name!r}. "
        "Something added it back at module scope; defer it instead.\n"
        f"stderr:\n{result.stderr}"
    )

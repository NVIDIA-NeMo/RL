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

import importlib
import os
import subprocess
import sys
from unittest.mock import patch

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


# ============================================================================
# _LazyModule — the mechanism that makes the deferral above work
# ============================================================================


def _lazy(name: str):
    from nemo_rl.utils.logger import _LazyModule

    return _LazyModule(name)


def test_constructing_it_imports_nothing():
    """The whole point: naming a module must not load it."""
    before = set(sys.modules)
    _lazy("json")
    assert set(sys.modules) - before == set()


def test_first_attribute_access_resolves_the_real_module():
    import json

    lazy = _lazy("json")
    assert lazy.dumps is json.dumps


def test_the_module_is_imported_once_and_cached():
    """A per-access ``importlib.import_module`` would be correct but would put
    an import-lock round trip on every logger call."""
    lazy = _lazy("json")
    calls = []
    real = importlib.import_module

    def counting(name, *a, **kw):
        calls.append(name)
        return real(name, *a, **kw)

    with patch("importlib.import_module", counting):
        lazy.dumps
        lazy.loads
        lazy.dumps

    assert calls == ["json"]


def test_a_dotted_module_resolves_to_the_submodule():
    """``plt`` is ``matplotlib.pyplot``: ``import_module`` must return the
    submodule, not the package, or every ``plt.<fn>`` call breaks."""
    lazy = _lazy("os.path")
    assert lazy.join("a", "b") == os.path.join("a", "b")


def test_a_missing_attribute_still_raises_attribute_error():
    lazy = _lazy("json")
    with pytest.raises(AttributeError):
        lazy.definitely_not_a_json_attribute


@pytest.mark.parametrize("name", ["wandb", "mlflow", "swanlab", "plt"])
def test_the_patch_targets_the_logger_tests_use_still_work(name):
    """``tests/unit/utils/test_logger.py`` patches these as module-level names
    (``@patch("nemo_rl.utils.logger.wandb")`` and friends). They stay real
    module attributes, so patching must keep working -- and must not trigger
    the import it exists to defer."""
    import nemo_rl.utils.logger as logger_mod

    sentinel = object()
    before = set(sys.modules)
    with patch.object(logger_mod, name, sentinel):
        assert getattr(logger_mod, name) is sentinel
    assert getattr(logger_mod, name) is not sentinel
    # patching must not have resolved the real backend
    for pulled in set(sys.modules) - before:
        assert not pulled.startswith(("wandb", "mlflow", "swanlab", "matplotlib"))

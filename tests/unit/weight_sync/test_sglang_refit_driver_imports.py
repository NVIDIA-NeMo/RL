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
"""Guard the driver/actor dependency split for the SGLang refit dispatch.

The SGLang refit is orchestrated from the driver process, which runs in the
project environment built without any backend extra. ``sglang`` is declared
mutually exclusive with ``mcore``, ``fsdp``, ``automodel`` and ``vllm``
(see ``[tool.uv] conflicts`` in ``pyproject.toml``), so the driver can never
hold both the SGLang stack and a trainer backend.

A driver-side ``import`` of a policy worker module therefore fails with
``ModuleNotFoundError`` at the first refit: ``megatron_policy_worker`` imports
``megatron.bridge`` at module scope and ``dtensor_policy_worker_v2`` imports
``nemo_automodel``. These checks parse the source instead of importing it, so
they fail on the offending commit even in an environment that happens to have
every backend installed.
"""

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

# Modules holding driver-side SGLang refit orchestration.
DRIVER_REFIT_MODULES = (
    "nemo_rl/weight_sync/megatron_refit_sglang.py",
    "nemo_rl/weight_sync/dtensor_refit_sglang.py",
)

# Distributions unavailable in the driver environment.
BACKEND_ONLY_ROOTS = frozenset(
    {
        "megatron",
        "transformer_engine",
        "nemo_automodel",
        "sglang",
        "vllm",
        "tensorrt_llm",
    }
)

# Worker modules that import a backend at module scope.
BACKEND_WORKER_MODULES = ("megatron_policy_worker", "dtensor_policy_worker_v2")


def _module_scope_imports(path: Path) -> set[str]:
    """Return root package names imported at module scope (imports inside functions are lazy and fine)."""
    tree = ast.parse(path.read_text())
    roots: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.parametrize("rel_path", DRIVER_REFIT_MODULES)
def test_driver_refit_module_imports_no_backend(rel_path: str) -> None:
    """Driver-side refit modules must be importable without any backend extra."""
    path = REPO_ROOT / rel_path
    assert path.is_file(), f"{rel_path} is missing; did the module move?"
    offenders = _module_scope_imports(path) & BACKEND_ONLY_ROOTS
    assert not offenders, (
        f"{rel_path} imports {sorted(offenders)} at module scope. The driver "
        "environment cannot provide these; import them inside the function."
    )


def test_dispatch_does_not_import_policy_workers() -> None:
    """``_refit_sglang_dispatch`` must not import a policy worker module."""
    source = (REPO_ROOT / "nemo_rl/algorithms/grpo.py").read_text()
    tree = ast.parse(source)
    dispatch = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "_refit_sglang_dispatch"
        ),
        None,
    )
    assert dispatch is not None, "_refit_sglang_dispatch not found in grpo.py"

    imported = {
        alias.name.split(".")[-1]
        for node in ast.walk(dispatch)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    offenders = imported & set(BACKEND_WORKER_MODULES)
    assert not offenders, (
        f"_refit_sglang_dispatch imports {sorted(offenders)}, which import a "
        "training backend at module scope and are unavailable in the driver."
    )

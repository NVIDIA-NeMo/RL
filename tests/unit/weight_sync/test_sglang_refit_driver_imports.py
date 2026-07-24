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
"""Keep the SGLang refit drivers importable from the driver process.

The driver runs in the project environment, which is synced without a
training-backend extra, so it cannot import ``megatron_policy_worker``
(``megatron.bridge``) or ``dtensor_policy_worker_v2`` (``nemo_automodel``).
Doing so raises ``ModuleNotFoundError`` at the first refit.

These checks parse the sources instead of importing them, so they also fail in
an environment that happens to provide a backend, such as the Megatron unit
test shards.
"""

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

REFIT_DRIVER_MODULES = (
    "nemo_rl/weight_sync/megatron_sglang_refit.py",
    "nemo_rl/weight_sync/dtensor_sglang_refit.py",
)

# Dispatch sites that select a refit driver at runtime, and the enclosing
# function whose imports must stay backend free.
SGLANG_REFIT_DISPATCH_SITES = (
    ("nemo_rl/algorithms/grpo.py", "_refit_sglang_dispatch"),
    ("nemo_rl/weight_sync/sglang_weight_synchronizer.py", "_refit_colocated"),
)

BACKEND_ONLY_ROOTS = frozenset({"megatron", "nemo_automodel", "transformer_engine"})

POLICY_WORKER_MODULES = frozenset(
    {"megatron_policy_worker", "dtensor_policy_worker_v2"}
)


def _find_function(tree: ast.Module, name: str) -> ast.FunctionDef | None:
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ):
            return node
    return None


@pytest.mark.parametrize("relative_path", REFIT_DRIVER_MODULES)
def test_refit_drivers_import_no_training_backend(relative_path: str) -> None:
    """The refit drivers must import without an ``mcore`` or ``automodel`` extra."""
    path = REPO_ROOT / relative_path
    assert path.is_file(), f"{relative_path} is missing"

    tree = ast.parse(path.read_text())
    imported = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])

    offenders = sorted(imported & BACKEND_ONLY_ROOTS)
    assert not offenders, (
        f"{relative_path} imports {offenders} at module scope, which the driver "
        "environment does not provide. Import it inside the function instead."
    )


@pytest.mark.parametrize(
    ("relative_path", "function_name"), SGLANG_REFIT_DISPATCH_SITES
)
def test_sglang_refit_dispatch_imports_no_policy_worker(
    relative_path: str, function_name: str
) -> None:
    """Driver-side dispatch must not reach a refit driver through a policy worker."""
    tree = ast.parse((REPO_ROOT / relative_path).read_text())
    function = _find_function(tree, function_name)
    assert function is not None, f"{function_name} not found in {relative_path}"

    imported = {
        alias.name.split(".")[-1]
        for node in ast.walk(function)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }

    offenders = sorted(imported & POLICY_WORKER_MODULES)
    assert not offenders, (
        f"{relative_path}::{function_name} imports {offenders}, which import a "
        "training backend at module scope and are unavailable in the driver."
    )

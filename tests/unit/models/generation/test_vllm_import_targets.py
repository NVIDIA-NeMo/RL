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

"""Every `from vllm... import X` in the generation workers must resolve.

The vLLM workers import most of vLLM lazily, inside methods that only run on a
GPU actor, so a module that upstream moves (``ErrorResponse`` left
``vllm.entrypoints.openai.engine.protocol`` in 0.29, vllm-project/vllm#54492)
is invisible to the unit suite and only shows up as a ``ModuleNotFoundError``
in a multi-node nightly. This walks the source with ``ast`` and performs the
same imports against the installed vLLM.
"""

import ast
import importlib
from pathlib import Path

import pytest

pytestmark = pytest.mark.vllm

# tests/unit/models/generation/<this file> -> parents[4] is the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[4]
# All three trees import vLLM lazily inside GPU-only methods: the vLLM
# generation workers, the Dynamo generation worker, and the ModelOpt fakequant /
# real-quant refit backends.
_GENERATION_DIRS = (
    _REPO_ROOT / "nemo_rl/models/generation/vllm",
    _REPO_ROOT / "nemo_rl/models/generation/dynamo",
    _REPO_ROOT / "nemo_rl/modelopt/models/generation",
)
_SOURCES = sorted(p for d in _GENERATION_DIRS for p in d.rglob("*.py"))
assert _SOURCES, f"no sources found under {_GENERATION_DIRS}"


_IMPORT_GUARDS = {"ImportError", "ModuleNotFoundError", "Exception"}


def _catches_import_error(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:
        return True
    names = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    return any(
        isinstance(n, ast.Name)
        and n.id in _IMPORT_GUARDS
        or isinstance(n, ast.Attribute)
        and n.attr in _IMPORT_GUARDS
        for n in names
    )


def _guarded_import_lines(tree: ast.AST) -> set[int]:
    """Lines of imports inside a ``try`` that catches ImportError.

    Those are deliberate version fallbacks (``try: new path / except
    ImportError: old path``); by construction one branch does not resolve on
    any given vLLM, so only the unguarded imports are checked.
    """
    lines: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        if not any(_catches_import_error(h) for h in node.handlers):
            continue
        for region in (node.body, *(h.body for h in node.handlers)):
            for stmt in region:
                for sub in ast.walk(stmt):
                    if isinstance(sub, ast.ImportFrom):
                        lines.add(sub.lineno)
    return lines


def _vllm_import_froms(path: Path) -> list[tuple[int, str, list[str]]]:
    tree = ast.parse(path.read_text())
    guarded = _guarded_import_lines(tree)
    found = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.split(".")[0] == "vllm"
            and node.level == 0
            and node.lineno not in guarded
        ):
            found.append((node.lineno, node.module, [a.name for a in node.names]))
    return found


@pytest.mark.parametrize(
    "source", _SOURCES, ids=[str(p.relative_to(_REPO_ROOT)) for p in _SOURCES]
)
def test_vllm_import_targets_resolve(source: Path):
    failures = []
    for lineno, module, names in _vllm_import_froms(source):
        try:
            mod = importlib.import_module(module)
        except ImportError as exc:  # module gone or renamed
            failures.append(f"{source.name}:{lineno}: import {module}: {exc}")
            continue
        for name in names:
            if name == "*":
                continue
            if not hasattr(mod, name):
                # `from pkg import submodule` is legal without an attribute.
                try:
                    importlib.import_module(f"{module}.{name}")
                except ImportError:
                    failures.append(f"{source.name}:{lineno}: {module} has no {name!r}")
    assert not failures, "\n".join(failures)

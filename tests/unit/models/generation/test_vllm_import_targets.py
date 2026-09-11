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
_GENERATION_DIR = Path(__file__).resolve().parents[4] / "nemo_rl/models/generation/vllm"
_SOURCES = sorted(_GENERATION_DIR.rglob("*.py"))
assert _SOURCES, f"no sources found under {_GENERATION_DIR}"


def _vllm_import_froms(path: Path) -> list[tuple[int, str, list[str]]]:
    tree = ast.parse(path.read_text())
    found = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.split(".")[0] == "vllm"
            and node.level == 0
        ):
            found.append((node.lineno, node.module, [a.name for a in node.names]))
    return found


@pytest.mark.parametrize(
    "source", _SOURCES, ids=[str(p.relative_to(_GENERATION_DIR)) for p in _SOURCES]
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

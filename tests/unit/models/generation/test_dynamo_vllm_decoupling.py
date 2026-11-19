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
"""Dynamo stays out of NeMo RL's vLLM engines.

The engines run NeMo RL's own vLLM and use Dynamo only through its
engine-agnostic backend SDK, and the frontend serves chat with NeMo RL's own
chat engine, so a vLLM bump can break Dynamo serving only in NeMo RL's own
``dynamo/vllm/`` package. These guards fail if NeMo RL starts importing
Dynamo's vLLM integration (or any other Dynamo module), if the Dynamo modules
it imports start importing vLLM, or if the Dynamo side shared by every backend
starts importing one.
"""

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

NEMO_RL_ROOT = Path(__file__).parents[4] / "nemo_rl"

# Every Dynamo module NeMo RL may import. A new entry needs a reason it does not
# reach Dynamo-owned vLLM code.
DYNAMO_SDK_MODULES = {
    "dynamo._core",
    "dynamo.common.backend",
    "dynamo.common.backend.logprobs",
    "dynamo.common.backend.publisher",
    # The frontend's entrypoint and chat-engine hook; the frontend loads Dynamo's
    # own vLLM chat processor only for --dyn-chat-processor, which NeMo RL rejects.
    "dynamo.frontend.main",
    "dynamo.llm",
    "dynamo.llm.exceptions",
}

GENERATION_BACKENDS = {"vllm", "sglang", "trtllm", "megatron"}
DYNAMO_ROOT = NEMO_RL_ROOT / "models/generation/dynamo"


def _backend_neutral_modules() -> list[str]:
    """The Dynamo side every generation backend builds on: all of ``dynamo/``
    outside its per-backend packages, and the wire types it shares."""
    paths = [
        path
        for path in DYNAMO_ROOT.rglob("*.py")
        if path.relative_to(DYNAMO_ROOT).parts[0] not in GENERATION_BACKENDS
    ]
    paths += [
        NEMO_RL_ROOT / "models/generation/rollout_result.py",
        NEMO_RL_ROOT / "models/generation/gym_call.py",
    ]
    return sorted(str(path.relative_to(NEMO_RL_ROOT)) for path in paths)


def _imports(path: Path) -> set[str]:
    """The absolute modules ``path`` imports, at any depth."""
    modules: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.ImportFrom) and node.level == 0:
            modules.add(node.module or "")
        elif isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
    return modules


def _dynamo_imports() -> dict[str, set[str]]:
    """The Dynamo modules each nemo_rl source file imports."""
    imports: dict[str, set[str]] = {}
    for path in NEMO_RL_ROOT.rglob("*.py"):
        dynamo = {
            module
            for module in _imports(path)
            if module == "dynamo" or module.startswith("dynamo.")
        }
        if dynamo:
            imports[str(path.relative_to(NEMO_RL_ROOT))] = dynamo
    return imports


def test_nemo_rl_imports_only_dynamos_backend_sdk() -> None:
    imports = _dynamo_imports()

    assert imports, "the Dynamo adapter's imports were not found"
    outside = {
        path: sorted(modules - DYNAMO_SDK_MODULES)
        for path, modules in imports.items()
        if modules - DYNAMO_SDK_MODULES
    }
    assert not outside, (
        f"nemo_rl imports Dynamo modules beyond its backend SDK: {outside}"
    )


def test_the_shared_dynamo_side_imports_no_generation_backend() -> None:
    modules = _backend_neutral_modules()
    assert "models/generation/dynamo/engine_adapter.py" in modules
    backends = {
        path: sorted(
            module
            for module in _imports(NEMO_RL_ROOT / path)
            if GENERATION_BACKENDS & set(module.split("."))
        )
        for path in modules
    }

    assert backends == {path: [] for path in modules}


@pytest.mark.vllm
def test_the_dynamo_modules_nemo_rl_imports_import_no_vllm() -> None:
    pytest.importorskip("dynamo._core")
    # A fresh interpreter, so modules other tests imported do not count.
    code = (
        "import json, sys\n"
        + "".join(f"import {module}\n" for module in sorted(DYNAMO_SDK_MODULES))
        + "print(json.dumps(sorted(m for m in sys.modules"
        " if m.split('.')[0] == 'vllm' or m.startswith('dynamo.vllm'))))"
    )
    loaded = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )

    assert json.loads(loaded.stdout.splitlines()[-1]) == []

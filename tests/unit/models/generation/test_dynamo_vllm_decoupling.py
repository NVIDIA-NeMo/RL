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
engine-agnostic backend SDK, so a vLLM bump can break Dynamo serving only where
it also breaks native vLLM. These guards fail if NeMo RL starts importing
Dynamo's vLLM integration (or any other Dynamo module), or if the SDK itself
starts importing vLLM.
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
    "dynamo.llm.exceptions",
}


def _dynamo_imports() -> dict[str, set[str]]:
    """The Dynamo modules each nemo_rl source file imports."""
    imports: dict[str, set[str]] = {}
    for path in NEMO_RL_ROOT.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            if isinstance(node, ast.ImportFrom) and node.level == 0:
                modules = [node.module or ""]
            elif isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            else:
                continue
            for module in modules:
                if module == "dynamo" or module.startswith("dynamo."):
                    imports.setdefault(str(path.relative_to(NEMO_RL_ROOT)), set()).add(
                        module
                    )
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


@pytest.mark.vllm
def test_dynamos_backend_sdk_imports_no_vllm() -> None:
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

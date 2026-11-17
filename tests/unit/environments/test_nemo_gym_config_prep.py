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

import ast
import sys
from pathlib import Path

MODULE_PATH = (
    Path(__file__).parents[3] / "nemo_rl" / "environments" / "nemo_gym_config_prep.py"
)


def test_nemo_gym_config_prep_module_is_stdlib_only():
    """docker/Dockerfile copies this module in before the Gym venv prefetch.

    The rest of the nemo_rl package is not in the image at that step, so an import of
    anything else breaks the prefetch in the image build.
    """
    stdlib = set(sys.stdlib_module_names) | {"__future__"}
    tree = ast.parse(MODULE_PATH.read_text())
    for node in ast.walk(tree):
        roots = []
        if isinstance(node, ast.Import):
            roots = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            roots = [(node.module or "").split(".")[0]]
        for root in roots:
            assert root in stdlib, (
                f"{MODULE_PATH.name} imports {root!r}, which is not in the standard "
                "library. docker/Dockerfile runs it before the rest of nemo_rl is in "
                "the image."
            )

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

"""Check native training keyword compatibility without importing GPU backends."""

import ast
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "filename, helper_name",
    [
        ("worker.py", "automodel_forward_backward"),
        ("cua.py", "prepare_model_forward"),
    ],
)
def test_training_keywords_match_native_api(filename: str, helper_name: str) -> None:
    project = Path(__file__).resolve().parents[2]
    native = ast.parse(
        (project.parents[1] / "nemo_rl/models/automodel/train.py").read_text()
    )
    function = next(
        node
        for node in native.body
        if isinstance(node, ast.FunctionDef) and node.name == helper_name
    )
    if function.args.kwarg is not None:
        return
    accepted = {arg.arg for arg in [*function.args.args, *function.args.kwonlyargs]}
    worker = ast.parse((project / "logra" / filename).read_text())
    calls = [
        node
        for node in ast.walk(worker)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == function.name
    ]
    assert calls, f"LoGRA must use the native {helper_name} helper"
    for call in calls:
        supplied = {keyword.arg for keyword in call.keywords if keyword.arg}
        assert not (supplied - accepted), (
            f"LoGRA passes unsupported {function.name} keywords: "
            f"{sorted(supplied - accepted)}"
        )

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

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.mcore


_REPO_ROOT = Path(__file__).parents[4]
_BODY_PATH = _REPO_ROOT / "nemo_rl/models/megatron/draft/dflash.py"
_ATTENTION_PATH = _REPO_ROOT / "nemo_rl/models/megatron/draft/block_attention.py"


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing production function: {name}")


def _method(tree: ast.Module, class_name: str, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for child in node.body:
            if isinstance(child, ast.FunctionDef) and child.name == name:
                return child
    raise AssertionError(f"missing production method: {class_name}.{name}")


def _called_names(node: ast.AST) -> list[str]:
    names: list[str] = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        if isinstance(child.func, ast.Name):
            names.append(child.func.id)
        elif isinstance(child.func, ast.Attribute):
            names.append(child.func.attr)
    return names


def test_body_uses_only_block_queries_and_one_block_attention_entry_point() -> None:
    source = _BODY_PATH.read_text()
    tree = ast.parse(source)
    forward = _method(tree, "DFlashBody", "forward")
    calls = _called_names(forward)

    assert "dflash_block_only_attention" in calls
    assert "dflash_block_attention" not in calls
    assert "trunk_query" not in source


def test_block_only_cuda_path_contains_exactly_one_compiled_flex_call() -> None:
    tree = ast.parse(_ATTENTION_PATH.read_text())
    cuda_forward = _function(tree, "_flex_block_only_attention_cuda")

    assert _called_names(cuda_forward).count("_COMPILED_FLEX_ATTENTION") == 1


def test_body_projections_use_mcore_tensor_parallel_layers() -> None:
    tree = ast.parse(_BODY_PATH.read_text())
    imported_names = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "megatron.core.tensor_parallel.layers"
        for alias in node.names
    }
    class_bases = {
        node.name: {base.id for base in node.bases if isinstance(base, ast.Name)}
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    }
    constructor_calls = _called_names(tree)

    assert {"ColumnParallelLinear", "RowParallelLinear"} <= imported_names
    assert "ColumnParallelLinear" in class_bases["_ColumnParallelProjection"]
    assert "RowParallelLinear" in class_bases["_RowParallelProjection"]
    assert "Linear" not in constructor_calls


def test_sharded_state_dict_accepts_full_nested_mcore_protocol() -> None:
    tree = ast.parse(_BODY_PATH.read_text())
    sharded_state_dict = _method(tree, "DFlashBody", "sharded_state_dict")
    parameter_names = [argument.arg for argument in sharded_state_dict.args.args]

    assert parameter_names == ["self", "prefix", "sharded_offsets", "metadata"]
    assert "sharded_state_dict_default" in _called_names(sharded_state_dict)

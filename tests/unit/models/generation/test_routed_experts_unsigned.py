# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check backend expert IDs without importing the GPU-only vLLM package."""

import ast
from pathlib import Path
from typing import Any

import pytest
import torch

SOURCE = Path(__file__).resolve().parents[4] / "nemo_rl/models/generation/vllm/utils.py"


@pytest.fixture
def convert():
    tree = ast.parse(SOURCE.read_text())
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_as_routed_experts_tensor"
    )
    namespace = {"torch": torch, "Any": Any, "G_ROUTED_EXPERTS_RANGE_CHECKED": False}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace[function.name]


@pytest.mark.parametrize("source_dtype", [torch.uint8, torch.uint16, torch.int16, torch.int64])
def test_preserves_expert_ids(convert, source_dtype):
    ids = [0, 31, 127, 255] if source_dtype == torch.uint8 else [0, 31, 255, 256, 383]
    source = torch.tensor(ids, dtype=source_dtype)
    result = convert(source, device=torch.device("cpu"), dtype=torch.int16)
    assert result.dtype == torch.int16
    assert result.tolist() == ids
    assert source.tolist() == ids


@pytest.mark.parametrize("ids,target", [([256, 383], torch.int8), ([65535], torch.int16)])
def test_rejects_overflow_before_narrowing(convert, ids, target):
    with pytest.raises(ValueError, match="exceeds the resolved carry dtype"):
        convert(torch.tensor(ids, dtype=torch.uint16), device=torch.device("cpu"), dtype=target)
    assert convert.__globals__["G_ROUTED_EXPERTS_RANGE_CHECKED"] is False


def test_empty_does_not_skip_next_range_check(convert):
    result = convert(torch.tensor([], dtype=torch.uint16), device=torch.device("cpu"), dtype=torch.int16)
    assert result.numel() == 0
    assert convert.__globals__["G_ROUTED_EXPERTS_RANGE_CHECKED"] is False
    with pytest.raises(ValueError):
        convert(torch.tensor([65535], dtype=torch.uint16), device=torch.device("cpu"), dtype=torch.int16)


def test_signed_sentinel_preserved(convert):
    result = convert(torch.tensor([-1, 0, 383]), device=torch.device("cpu"), dtype=torch.int16)
    assert result.tolist() == [-1, 0, 383]

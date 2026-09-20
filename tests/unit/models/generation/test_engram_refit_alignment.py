# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the real row loader without importing the GPU-only vLLM package."""
import ast
from pathlib import Path
import sys
import types

import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / 'nemo_rl/models/generation/vllm/engram_refit.py'


def load_function():
    tree = ast.parse(SOURCE.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'load_engram_rows')
    namespace = {'torch': torch, 'parse_row_name': lambda _: ('table.weight', 0, 4)}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace['load_engram_rows']


@pytest.mark.parametrize('offset', [0, 1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize('first,last', [(0, 4), (1, 3)])
def test_contiguous_bf16_view_alignment(monkeypatch, offset, first, last):
    # Every source is contiguous and dtype-aligned, but most are not 16-byte aligned.
    storage = torch.arange(4 * 256 + offset, dtype=torch.float32).to(torch.bfloat16)
    payload = storage[offset:].reshape(4, 256)
    assert payload.is_contiguous()
    expected = payload[first:last].clone()
    original = storage.clone()
    seen = []
    def quantize(rows):
        assert rows.data_ptr() % 16 == 0
        assert rows.is_contiguous()
        torch.testing.assert_close(rows, expected, rtol=0, atol=0)
        seen.append(rows.data_ptr())
        return rows, torch.ones((last-first, 8))
    modules = {
        'nemo_rl.models.generation.vllm.quantization.deepseek_v41_fp8': {'map_checkpoint_name': lambda model, name: name},
        'nemo_rl.models.generation.vllm.quantization.fp8': {'quantize_mxfp8_weight': quantize},
    }
    for name, attrs in modules.items():
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    table = types.SimpleNamespace(dim=256, num_embeddings=4, vocab_start_idx=first, vocab_end_idx=last,
                                  weight=torch.zeros((last-first, 256), dtype=torch.bfloat16),
                                  weight_scale_inv=torch.zeros((last-first, 8)))
    model = types.SimpleNamespace(get_submodule=lambda name: table)
    assert load_function()(model, 'table.weight.__nrl_engram_rows_0_4', payload)
    torch.testing.assert_close(table.weight, expected, rtol=0, atol=0)
    torch.testing.assert_close(storage, original, rtol=0, atol=0)
    assert torch.all(table.weight_scale_inv == 1)
    if payload[first:last].data_ptr() % 16 == 0:
        assert seen[0] == payload[first:last].data_ptr(), 'Aligned input should avoid copying'
    else:
        assert seen[0] != payload[first:last].data_ptr()

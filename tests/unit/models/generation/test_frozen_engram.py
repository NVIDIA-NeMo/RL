# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise actual safetensors row loading into native local rollout shards."""

import json
import importlib.util
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

# Load the complete production module without importing GPU-only vLLM workers.
_SOURCE = (
    Path(__file__).resolve().parents[4]
    / "nemo_rl/models/generation/vllm/engram_refit.py"
)
_SPEC = importlib.util.spec_from_file_location("frozen_engram_loader", _SOURCE)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
load_frozen_engram_tables = _MODULE.load_frozen_engram_tables


class _Mapper:
    def apply_list(self, names):
        return [name.replace("layers.1.engram.embed", "table") for name in names]


@pytest.mark.parametrize("first,last", [(0, 17), (0, 5), (5, 17)])
def test_native_bytes_and_trainable_weights_unchanged(tmp_path, first, last):
    torch.manual_seed(17)
    values = torch.randn(17, 64).to(torch.float8_e4m3fn)
    scales = torch.randint(120, 130, (17, 2), dtype=torch.uint8)
    tensors = {
        "layers.1.engram.embed.weight": values,
        "layers.1.engram.embed.scale": scales.view(torch.float8_e8m0fnu),
    }
    save_file(tensors, tmp_path / "table.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "table.safetensors" for name in tensors}})
    )
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "text_config": {
                    "engram_layer_ids": [1],
                    "engram_num_embeddings": [17],
                    "engram_head_dim": 64,
                }
            }
        )
    )
    model = torch.nn.Module()
    model.hf_to_vllm_mapper = _Mapper()
    model.table = torch.nn.Module()
    model.table.num_embeddings = 17
    model.table.dim = 64
    model.table.vocab_start_idx = first
    model.table.vocab_end_idx = last
    model.table.weight = torch.nn.Parameter(
        torch.empty(last - first, 64, dtype=torch.float8_e4m3fn), requires_grad=False
    )
    model.table.weight_scale_inv = torch.nn.Parameter(
        torch.empty(last - first, 2, dtype=torch.uint8), requires_grad=False
    )
    model.projection = torch.nn.Linear(64, 8)
    before = model.projection.weight.clone()
    load_frozen_engram_tables(model, str(tmp_path))
    torch.testing.assert_close(
        model.table.weight.view(torch.uint8),
        values[first:last].view(torch.uint8),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        model.table.weight_scale_inv, scales[first:last], rtol=0, atol=0
    )
    torch.testing.assert_close(model.projection.weight, before, rtol=0, atol=0)
    model.table.dim = 32
    with pytest.raises(ValueError, match="layout mismatch"):
        load_frozen_engram_tables(model, str(tmp_path))

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
_bulk_read_rows = _MODULE._bulk_read_rows


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


@pytest.mark.parametrize("first,last,chunk_bytes", [(0, 1000, 256 * 1024 * 1024), (37, 963, 1000), (999, 1000, 7)])
def test_bulk_read_rows_matches_safetensors_slice(tmp_path, first, last, chunk_bytes):
    """Byte-offset reads must return exactly what the safetensors slice API returns."""
    from safetensors import safe_open

    torch.manual_seed(3)
    width = 96
    table = torch.randint(0, 256, (1000, width), dtype=torch.uint8)
    other = torch.randint(0, 256, (10, 16), dtype=torch.uint8)
    save_file({"zz.first": other, "layers.1.engram.embed.weight": table.view(torch.float8_e4m3fn)}, tmp_path / "t.safetensors")
    destination = torch.empty(last - first, width, dtype=torch.float8_e4m3fn)
    _bulk_read_rows(tmp_path / "t.safetensors", "layers.1.engram.embed.weight", first, last, width, destination, chunk_bytes=chunk_bytes)
    with safe_open(tmp_path / "t.safetensors", framework="pt", device="cpu") as reader:
        expected = reader.get_slice("layers.1.engram.embed.weight")[first:last].view(torch.uint8)
    torch.testing.assert_close(destination.view(torch.uint8), expected, rtol=0, atol=0)
    torch.testing.assert_close(destination.view(torch.uint8), table[first:last], rtol=0, atol=0)


def test_bulk_read_rows_rejects_width_and_size_mismatch(tmp_path):
    table = torch.zeros(8, 32, dtype=torch.uint8)
    save_file({"t": table}, tmp_path / "t.safetensors")
    with pytest.raises(ValueError, match="header width mismatch"):
        _bulk_read_rows(tmp_path / "t.safetensors", "t", 0, 8, 16, torch.empty(8, 16, dtype=torch.uint8))
    with pytest.raises(ValueError, match="destination size mismatch"):
        _bulk_read_rows(tmp_path / "t.safetensors", "t", 0, 8, 32, torch.empty(4, 32, dtype=torch.uint8))

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

from types import SimpleNamespace

import torch

from nemo_rl.models.generation.sglang import nvfp4_quantization_core as nvfp4


class _FakeNVFP4Quantizer:
    def __init__(self) -> None:
        self.quantized_shapes: list[tuple[int, ...]] = []

    def quantize(self, weight: torch.Tensor) -> SimpleNamespace:
        self.quantized_shapes.append(tuple(weight.shape))
        rows, columns = weight.shape
        return SimpleNamespace(
            _rowwise_data=torch.zeros(
                (rows, columns // 2),
                dtype=torch.uint8,
                device=weight.device,
            ),
            _rowwise_scale_inv=torch.zeros(
                (rows, columns // nvfp4.NVFP4_GROUP_SIZE),
                dtype=torch.uint8,
                device=weight.device,
            ),
            _amax_rowwise=weight.abs().max().to(torch.float32).reshape(1),
        )


def test_nvfp4_pair_uses_one_quantization_and_independent_shared_scales(
    monkeypatch,
) -> None:
    quantizer = _FakeNVFP4Quantizer()
    monkeypatch.setattr(nvfp4, "_make_nvfp4_quantizer", lambda: quantizer)

    gate = torch.arange(3 * 32, dtype=torch.float32).reshape(3, 32)
    up = -torch.arange(5 * 32, dtype=torch.float32).reshape(5, 32)
    gate_output, up_output = nvfp4.nvfp4_quantize_2d_pair(gate, up)

    gate_qweight, gate_block_scale, gate_global_scale = gate_output
    up_qweight, up_block_scale, up_global_scale = up_output

    # Gate/up are concatenated and padded once, so their ModelOpt global scales
    # are bitwise equal while remaining independent tensors.
    assert quantizer.quantized_shapes == [(nvfp4.TE_NVFP4_ROW_ALIGNMENT, 32)]
    assert gate_qweight.shape == (3, 16)
    assert up_qweight.shape == (5, 16)
    assert gate_block_scale.shape == (3, 2)
    assert up_block_scale.shape == (5, 2)
    torch.testing.assert_close(gate_global_scale, up_global_scale, rtol=0, atol=0)
    assert (
        gate_global_scale.untyped_storage().data_ptr()
        != up_global_scale.untyped_storage().data_ptr()
    )


def test_live_nvfp4_entries_do_not_add_static_input_scale() -> None:
    name = "model.layers.1.mlp.experts.0.down_proj.weight"
    entries = nvfp4.nvfp4_quantized_entries(
        name,
        (
            torch.zeros((2, 16), dtype=torch.uint8),
            torch.zeros((2, 2), dtype=torch.uint8).view(torch.float8_e4m3fn),
            torch.ones((), dtype=torch.float32),
        ),
        include_input_scale=False,
    )

    assert [entry_name for entry_name, _ in entries] == [
        name,
        "model.layers.1.mlp.experts.0.down_proj.weight_scale",
        "model.layers.1.mlp.experts.0.down_proj.weight_scale_2",
    ]

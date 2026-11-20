# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Compat-patch integrity and canonical scale restoration before hot reload."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from nemo_rl.models.generation.sglang.utils import patches


@pytest.mark.parametrize(
    "patcher",
    [
        patches._patch_sglang_non_gated_fp8_moe,
        patches._patch_sglang_mxfp8_moe_scale_layout,
        patches._patch_sglang_mxfp8_moe_hidden_size,
    ],
)
def test_upstream_anchor_drift_does_not_overwrite_files(tmp_path, monkeypatch, patcher):
    files = {}

    def locate(relative_path):
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Upstream implementation with incompatible anchors.\n")
        files[relative_path] = path
        return str(path)

    monkeypatch.setattr(patches, "_get_sglang_file", locate)
    with pytest.raises(RuntimeError, match="anchor mismatch"):
        patcher()

    assert files
    for path in files.values():
        assert (
            path.read_text() == "# Upstream implementation with incompatible anchors.\n"
        )
        assert not list(path.parent.glob("*.tmp"))


@pytest.mark.parametrize("rows", [33, 128])
def test_scale_layout_patch_pads_rows_and_restores_canonical_refit_shape(
    tmp_path: Path, monkeypatch, rows: int
):
    source = tmp_path / "fp8.py"
    source.write_text(
        "class Quantizer:\n"
        "    use_mxfp8 = True\n"
        "    def swizzle(self, scale, weight_shape):\n"
        "        if self.use_mxfp8:\n"
        "            num_experts, m, k = weight_shape\n"
        "            aligned_m = ((m + 127) // 128) * 128\n"
        "            scale = scale.view(num_experts, aligned_m, k // 32)\n"
        "            return scale\n\n"
        "    def align(self, layer):\n"
        "        if self.use_mxfp8:\n"
        "            align_mxfp8_moe_weights_for_flashinfer_trtllm(layer)\n\n"
        "    def process_weights_after_loading(self, layer: Module) -> None:\n"
        "        pass\n"
    )
    monkeypatch.setattr(patches, "_get_sglang_file", lambda _: str(source))
    patches._patch_sglang_mxfp8_moe_scale_layout()
    namespace = {"torch": torch, "Module": torch.nn.Module}
    exec(compile(source.read_text(), str(source), "exec"), namespace)
    quantizer = namespace["Quantizer"]()
    scale = torch.arange(2 * rows * 2).reshape(2, rows, 2).to(torch.uint8)

    packed = quantizer.swizzle(scale, (2, rows, 64))

    assert packed.shape == (2, 128, 2)
    assert torch.equal(packed[:, :rows], scale)
    assert torch.count_nonzero(packed[:, rows:]) == 0
    layer = SimpleNamespace(
        w13_weight=torch.zeros((2, rows, 64)),
        w2_weight=torch.zeros((2, rows, 64)),
        w13_weight_scale_inv=torch.nn.Parameter(packed.clone(), requires_grad=False),
        w2_weight_scale_inv=torch.nn.Parameter(packed.clone(), requires_grad=False),
    )

    quantizer.restore_weights_before_loading(layer)

    for restored in (layer.w13_weight_scale_inv, layer.w2_weight_scale_inv):
        assert restored.shape == scale.shape
        assert restored.format_ue8m0 is True
        restored.data.copy_(scale)
        assert torch.equal(quantizer.swizzle(restored, (2, rows, 64))[:, :rows], scale)
    patched = source.read_bytes()
    patches._patch_sglang_mxfp8_moe_scale_layout()
    assert source.read_bytes() == patched


def test_partial_scale_patch_is_rejected_before_write(tmp_path: Path, monkeypatch):
    source = tmp_path / "fp8.py"
    original = "scale = scale.reshape(num_experts, m, k // 32)\n"
    source.write_text(original)
    monkeypatch.setattr(patches, "_get_sglang_file", lambda _: str(source))

    with pytest.raises(RuntimeError, match="anchor mismatch"):
        patches._patch_sglang_mxfp8_moe_scale_layout()

    assert source.read_text() == original

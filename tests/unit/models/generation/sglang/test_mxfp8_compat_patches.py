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

import ast
import shutil
from importlib.metadata import distribution
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


@pytest.mark.sglang
def test_installed_sglang_patches_preserve_hidden_padding_during_refit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    relative_paths = (
        "srt/layers/quantization/fp8.py",
        "srt/layers/moe/moe_runner/flashinfer_trtllm.py",
        "srt/layers/moe/fused_moe_triton/layer.py",
    )
    files = {}
    installed_sglang = distribution("sglang")
    for relative_path in relative_paths:
        destination = tmp_path / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            installed_sglang.locate_file(f"sglang/{relative_path}"), destination
        )
        files[relative_path] = destination
    monkeypatch.setattr(patches, "_get_sglang_file", lambda path: str(files[path]))

    patchers = (
        patches._patch_sglang_non_gated_fp8_moe,
        patches._patch_sglang_mxfp8_moe_scale_layout,
        patches._patch_sglang_mxfp8_moe_hidden_size,
    )
    for patcher in patchers:
        patcher()
    patched_sources = {path: path.read_bytes() for path in files.values()}
    for patcher in patchers:
        patcher()
    assert {path: path.read_bytes() for path in files.values()} == patched_sources

    helper_names = {
        "_pad_mxfp8_moe_hidden_size_for_flashinfer_trtllm",
        "_narrow_mxfp8_hidden_padding_for_load",
    }
    helpers = []
    for path in files.values():
        tree = ast.parse(path.read_text())
        helpers.extend(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name in helper_names
        )
    assert {helper.name for helper in helpers} == helper_names
    namespace = {"torch": torch, "logger": patches.logger}
    exec(
        compile(ast.Module(body=helpers, type_ignores=[]), "<helpers>", "exec"),
        namespace,
    )

    original = (
        torch.full((2, 32, 2688), 0.5, dtype=torch.bfloat16),
        torch.full((2, 32, 84), 120, dtype=torch.uint8),
        torch.full((2, 2688, 32), 0.25, dtype=torch.bfloat16),
        torch.full((2, 2688, 1), 121, dtype=torch.uint8),
    )
    pad = namespace["_pad_mxfp8_moe_hidden_size_for_flashinfer_trtllm"]
    *padded, hidden_size = pad(*original)
    assert hidden_size == 2816
    loader = SimpleNamespace(
        use_flashinfer_trtllm_moe=True, quant_method=SimpleNamespace(use_mxfp8=True)
    )
    narrow = namespace["_narrow_mxfp8_hidden_padding_for_load"]
    for source, target, shard_id in zip(original, padded, ("w1", "w1", "w2", "w2")):
        pointer = target.data_ptr()
        for value in (2, 3):
            loaded = torch.full_like(source[0], value)
            view, loaded = narrow(loader, target[0], loaded, shard_id)
            view.copy_(loaded)
            assert target.data_ptr() == pointer
            torch.testing.assert_close(view, loaded, rtol=0, atol=0)
            padding = (
                target[0, :, source.shape[2] :]
                if shard_id == "w1"
                else target[0, source.shape[1] :, :]
            )
            assert torch.all(padding == (1 if source.dtype == torch.uint8 else 0))
        torch.testing.assert_close(
            target[1, : source.shape[1], : source.shape[2]], source[1], rtol=0, atol=0
        )

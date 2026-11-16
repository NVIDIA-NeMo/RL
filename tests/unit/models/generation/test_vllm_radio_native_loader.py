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

"""Exercise the patched vLLM RADIO loader on legacy, native and hybrid names."""

import ast
import inspect
import logging
from pathlib import Path

import pytest
import torch

from nemo_rl.models.generation.vllm import patches

pytestmark = pytest.mark.vllm

_RADIO_SOURCE = "model_executor/models/radio.py"
_VL_SOURCE = "model_executor/models/nano_nemotron_vl.py"
_PATCH_FN = patches._patch_vllm_radio_native_loader


def _replacements(name: str) -> list[tuple[str, str]]:
    """Read a ``(old, new)`` tuple literal out of the patch function source."""
    tree = ast.parse(inspect.getsource(_PATCH_FN))
    return next(
        ast.literal_eval(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
    )


def _unpatched_copy(relative: str, replacements, destination: Path) -> Path:
    """Copy the installed vLLM file with this patch reversed (order-independent)."""
    content = Path(patches._get_vllm_file(relative)).read_text()
    for old, new in reversed(replacements):
        content = content.replace(new, old, 1)
    destination.write_text(content)
    return destination


@pytest.fixture
def patched_sources(tmp_path, monkeypatch):
    radio = _unpatched_copy(
        _RADIO_SOURCE, _replacements("radio_replacements"), tmp_path / "radio.py"
    )
    vl = _unpatched_copy(
        _VL_SOURCE, _replacements("vl_replacements"), tmp_path / "nano_nemotron_vl.py"
    )
    files = {_RADIO_SOURCE: str(radio), _VL_SOURCE: str(vl)}
    monkeypatch.setattr(patches, "_get_vllm_file", lambda relative: files[relative])
    _PATCH_FN(logging.getLogger(__name__))
    return radio, vl


def test_native_loader_patch_anchor_still_matches_installed_vllm(patched_sources):
    radio, vl = patched_sources
    assert "def native_to_vllm(sub: str)" in radio.read_text()
    assert 'return name.startswith("vision_model.")' in vl.read_text()
    assert 'startswith("vision_model.radio_model.")' not in vl.read_text()


def test_native_loader_patch_is_idempotent(patched_sources):
    radio, vl = patched_sources
    before = (radio.read_text(), vl.read_text())
    _PATCH_FN(logging.getLogger(__name__))
    assert (radio.read_text(), vl.read_text()) == before


def test_native_loader_patch_coexists_with_layerscale_patch(
    patched_sources, monkeypatch
):
    """The LayerScale patch must still apply and stay idempotent on top."""
    radio, _ = patched_sources
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _relative: str(radio))
    patches._patch_vllm_radio_layerscale_loader(logging.getLogger(__name__))
    after_both = radio.read_text()
    assert "initializer_factor" in after_both
    assert "def native_to_vllm(sub: str)" in after_both
    patches._patch_vllm_radio_layerscale_loader(logging.getLogger(__name__))
    assert radio.read_text() == after_both


def test_native_loader_patch_warns_on_unknown_source(monkeypatch, tmp_path, caplog):
    radio = tmp_path / "radio.py"
    radio.write_text("class RadioModel:\n    pass\n")
    vl = tmp_path / "nano_nemotron_vl.py"
    vl.write_text("class NemotronH_Nano_VL_V2:\n    pass\n")
    files = {_RADIO_SOURCE: str(radio), _VL_SOURCE: str(vl)}
    monkeypatch.setattr(patches, "_get_vllm_file", lambda relative: files[relative])
    with caplog.at_level(logging.WARNING):
        _PATCH_FN(logging.getLogger(__name__))
    assert "Could not apply vLLM RADIO native-name loader patch" in caplog.text
    assert radio.read_text() == "class RadioModel:\n    pass\n"
    assert vl.read_text() == "class NemotronH_Nano_VL_V2:\n    pass\n"


class _QKVParam(torch.nn.Parameter):
    """Fused qkv parameter with a shard-aware loader like QKVParallelLinear."""

    def __new__(cls, hidden: int):
        return super().__new__(
            cls, torch.zeros(3 * hidden, hidden), requires_grad=False
        )

    def weight_loader(self, param, weight, shard_id=None):
        hidden = param.shape[1]
        if shard_id is None:
            param.data.copy_(weight)
            return
        offset = {"q": 0, "k": 1, "v": 2}[shard_id] * hidden
        param.data[offset : offset + hidden].copy_(weight)


def _load_weights_fn(radio: Path):
    """Extract the patched ``RadioModel.load_weights`` as a standalone function."""
    tree = ast.parse(radio.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RadioModel"
    )
    fn = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "load_weights"
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            fn,
        ],
        type_ignores=[],
    )

    def default_weight_loader(param, weight):
        param.data.copy_(weight)

    namespace = {"default_weight_loader": default_weight_loader}
    exec(compile(ast.fix_missing_locations(module), str(radio), "exec"), namespace)
    return namespace["load_weights"]


@pytest.fixture
def radio_model(patched_sources):
    radio, _ = patched_sources
    hidden = 4
    params = {
        "model.patch_generator.embedder.weight": torch.nn.Parameter(
            torch.zeros(hidden, 3)
        ),
        "model.patch_generator.pos_embed": torch.nn.Parameter(
            torch.zeros(1, 2, hidden)
        ),
        "model.patch_generator.cls_token.token": torch.nn.Parameter(
            torch.zeros(1, 1, hidden)
        ),
        "model.encoder.layers.0.attn.qkv.weight": _QKVParam(hidden),
        "model.encoder.layers.0.attn.proj.weight": torch.nn.Parameter(
            torch.zeros(hidden, hidden)
        ),
        "model.encoder.layers.0.norm1.weight": torch.nn.Parameter(torch.zeros(hidden)),
        "model.encoder.layers.0.ls1": torch.nn.Parameter(torch.zeros(hidden)),
        "model.encoder.layers.0.ls2": torch.nn.Parameter(torch.zeros(hidden)),
    }

    class Model:
        config = type("Config", (), {"initializer_factor": 1.0})()

        def named_parameters(self):
            return params.items()

    model = Model()
    load_weights = _load_weights_fn(radio)
    return model, params, lambda weights: load_weights(model, weights), hidden


def _assert_loaded(loaded: set[str], expected: set[str]) -> None:
    """Check the keys this loader call wrote.

    The LayerScale loader patch (applied to the same function by the installed
    vLLM when a generation worker has started) backfills unloaded ``ls1``/``ls2``
    into ``loaded_params``; tolerate those so the tests are order-independent.
    """
    assert expected <= loaded
    assert all(name.endswith((".ls1", ".ls2")) for name in loaded - expected)


def test_legacy_names_still_load_through_the_stock_path(radio_model):
    _, params, load, hidden = radio_model
    fused = torch.arange(3 * hidden * hidden, dtype=torch.float32).reshape(
        3 * hidden, hidden
    )
    loaded = load(
        [
            ("radio_model.model.blocks.0.attn.qkv.weight", fused),
            ("radio_model.model.patch_generator.pos_embed", torch.ones(1, 2, hidden)),
            ("radio_model.input_conditioner.norm_mean", torch.ones(3)),
            ("radio_model.summary_idxs", torch.zeros(1)),
            ("language_model.ignored", torch.ones(1)),
        ]
    )
    assert torch.equal(params["model.encoder.layers.0.attn.qkv.weight"], fused)
    assert torch.equal(
        params["model.patch_generator.pos_embed"], torch.ones(1, 2, hidden)
    )
    _assert_loaded(
        loaded,
        {
            "model.encoder.layers.0.attn.qkv.weight",
            "model.patch_generator.pos_embed",
        },
    )


def test_native_names_write_qkv_shards_and_layer_scale(radio_model):
    """Automodel main streams transformers-native RadioModel names at refit."""
    _, params, load, hidden = radio_model
    q = torch.full((hidden, hidden), 1.0)
    k = torch.full((hidden, hidden), 2.0)
    v = torch.full((hidden, hidden), 3.0)
    loaded = load(
        [
            ("encoder.layer.0.attention.attention.query.weight", q),
            ("encoder.layer.0.attention.attention.key.weight", k),
            ("encoder.layer.0.attention.attention.value.weight", v),
            (
                "encoder.layer.0.attention.output.dense.weight",
                torch.full((hidden, hidden), 4.0),
            ),
            ("encoder.layer.0.norm1.weight", torch.full((hidden,), 5.0)),
            ("encoder.layer.0.layer_scale1.lambda1", torch.full((hidden,), 6.0)),
            ("embeddings.patch_projection.weight", torch.full((hidden, 3), 7.0)),
            ("embeddings.cls_register_token", torch.full((1, 1, hidden), 8.0)),
            ("summary_idxs", torch.zeros(1)),
            ("encoder.layer.0.unknown.weight", torch.ones(1)),
        ]
    )
    assert torch.equal(
        params["model.encoder.layers.0.attn.qkv.weight"], torch.cat([q, k, v])
    )
    assert torch.equal(
        params["model.encoder.layers.0.attn.proj.weight"],
        torch.full((hidden, hidden), 4.0),
    )
    assert torch.equal(
        params["model.encoder.layers.0.norm1.weight"], torch.full((hidden,), 5.0)
    )
    assert torch.equal(params["model.encoder.layers.0.ls1"], torch.full((hidden,), 6.0))
    assert torch.equal(
        params["model.patch_generator.embedder.weight"], torch.full((hidden, 3), 7.0)
    )
    assert torch.equal(
        params["model.patch_generator.cls_token.token"], torch.full((1, 1, hidden), 8.0)
    )
    assert "model.encoder.layers.0.unknown.weight" not in loaded
    _assert_loaded(
        loaded,
        {
            "model.encoder.layers.0.attn.qkv.weight",
            "model.encoder.layers.0.attn.proj.weight",
            "model.encoder.layers.0.norm1.weight",
            "model.encoder.layers.0.ls1",
            "model.patch_generator.embedder.weight",
            "model.patch_generator.cls_token.token",
        },
    )


def test_hybrid_names_from_automodel_r060_write_qkv_shards(radio_model):
    """Automodel r0.6.0 renames native keys per tensor under the legacy block prefix."""
    _, params, load, hidden = radio_model
    q = torch.full((hidden, hidden), 1.0)
    k = torch.full((hidden, hidden), 2.0)
    v = torch.full((hidden, hidden), 3.0)
    loaded = load(
        [
            ("radio_model.model.blocks.0.attention.attention.query.weight", q),
            ("radio_model.model.blocks.0.attention.attention.key.weight", k),
            ("radio_model.model.blocks.0.attention.attention.value.weight", v),
            (
                "radio_model.model.blocks.0.attn.proj.weight",
                torch.full((hidden, hidden), 4.0),
            ),
            (
                "radio_model.model.blocks.0.layer_scale2.lambda1",
                torch.full((hidden,), 9.0),
            ),
        ]
    )
    assert torch.equal(
        params["model.encoder.layers.0.attn.qkv.weight"], torch.cat([q, k, v])
    )
    assert torch.equal(
        params["model.encoder.layers.0.attn.proj.weight"],
        torch.full((hidden, hidden), 4.0),
    )
    assert torch.equal(params["model.encoder.layers.0.ls2"], torch.full((hidden,), 9.0))
    _assert_loaded(
        loaded,
        {
            "model.encoder.layers.0.attn.qkv.weight",
            "model.encoder.layers.0.attn.proj.weight",
            "model.encoder.layers.0.ls2",
        },
    )

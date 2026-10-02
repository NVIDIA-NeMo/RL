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

# Portions Copyright contributors to the vLLM project (Apache-2.0).

"""Exercise the installed model methods with real CPU tensors and small modules.

The large vision/LLM models are replaced with doubles; tested method bodies
come from the installed vLLM after applying the NeMo-RL source patch.
LayerNorm loading, FP32 arithmetic, streaming ownership, and refit are real.
Tests adapted from TomerBN-Nvidia/vllm PR #46 (Apache-2.0).
"""

import ast
from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.radio import calc_seq_lens

from nemo_rl.models.generation.vllm import patches


def _model_methods():
    source_path = Path(
        patches._get_vllm_file("model_executor/models/nano_nemotron_vl.py")
    )
    source, _ = patches._radio_final_layernorm_source(source_path.read_text())
    tree = ast.parse(source)
    model = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "NemotronH_Nano_VL_V2"
    )
    methods = {
        "load_weights",
        "_apply_vision_final_layernorm",
        "extract_feature_dynamic",
        "extract_feature",
        "pixel_shuffle",
        "pixel_shuffle_dynamic_res",
    }
    model.bases = []
    model.decorator_list = []
    model.body = [
        node
        for node in model.body
        if isinstance(node, ast.FunctionDef) and node.name in methods
    ]
    namespace = {
        "torch": torch,
        "Iterable": Iterable,
        "calc_seq_lens": calc_seq_lens,
        "default_weight_loader": default_weight_loader,
        "logger": SimpleNamespace(info_once=lambda *args, **kwargs: None),
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[model], type_ignores=[])),
            str(source_path),
            "exec",
        ),
        namespace,
    )
    return namespace["NemotronH_Nano_VL_V2"]


NemotronH_Nano_VL_V2 = _model_methods()


def test_radio_final_layernorm_source_idempotent():
    source_path = Path(
        patches._get_vllm_file("model_executor/models/nano_nemotron_vl.py")
    )
    source, _ = patches._radio_final_layernorm_source(source_path.read_text())
    assert patches._radio_final_layernorm_source(source) == (source, False)
    compile(source, str(source_path), "exec")


def test_radio_final_layernorm_incomplete_patch_rejected():
    source_path = Path(
        patches._get_vllm_file("model_executor/models/nano_nemotron_vl.py")
    )
    source, _ = patches._radio_final_layernorm_source(source_path.read_text())
    source = source.replace(
        "        vit_embeds = self._apply_vision_final_layernorm(vit_embeds)\n", "", 1
    )
    with pytest.raises(RuntimeError, match="incomplete"):
        patches._radio_final_layernorm_source(source)


def test_radio_final_layernorm_missing_anchor_rejected():
    with pytest.raises(RuntimeError, match="insertion point"):
        patches._radio_final_layernorm_source("# Unknown model implementation\n")


class _TextOnlyMultiModalConfig:
    def get_limit_per_prompt(self, modality: str) -> int:
        return 0


class _ImageOnlyMultiModalConfig:
    def get_limit_per_prompt(self, modality: str) -> int:
        return 1 if modality == "image" else 0


class _ModelConfig:
    multimodal_config = _TextOnlyMultiModalConfig()


class _ImageOnlyModelConfig:
    multimodal_config = _ImageOnlyMultiModalConfig()


class _LanguageModel:
    def __init__(self) -> None:
        self.loaded_weights: list[tuple[str, object]] = []

    def load_weights(self, weights):
        self.loaded_weights = list(weights)


class _MissingMultiModalModule:
    def named_parameters(self):
        raise AssertionError("multimodal weights should not be inspected")

    def load_weights(self, weights):
        raise AssertionError("multimodal weights should not be loaded")


class _AdapterModule:
    def named_parameters(self):
        return []


class _VisionModel:
    def __init__(self) -> None:
        self.loaded_weights: list[tuple[str, object]] = []

    def load_weights(self, weights):
        self.loaded_weights = list(weights)


class _FakeTensor:
    """Sentinel stand-in for torch.Tensor in load_weights tests.

    Supports the
    .detach().clone() chain used by load_weights for buffered mm weights;
    both methods return self so identity (and the existing equality
    assertions) are preserved through cloning.
    """

    def detach(self):
        return self

    def clone(self):
        return self


def test_nano_nemotron_vl_skips_multimodal_weights_in_text_only_mode():
    model = object.__new__(NemotronH_Nano_VL_V2)
    language_model = _LanguageModel()
    object.__setattr__(model, "model_config", _ModelConfig())
    object.__setattr__(model, "language_model", language_model)
    object.__setattr__(model, "mlp1", _AdapterModule())
    object.__setattr__(model, "vision_model", _MissingMultiModalModule())
    object.__setattr__(model, "sound_encoder", None)

    language_weight = object()
    model.load_weights(
        [
            ("language_model.layers.0.weight", language_weight),
            ("mlp1.0.weight", object()),
            ("vision_model.radio_model.encoder.weight", object()),
            ("sound_encoder.encoder.weight", object()),
        ]
    )

    assert language_model.loaded_weights == [("layers.0.weight", language_weight)]


def test_nano_nemotron_vl_loads_vision_weights_without_sound_encoder():
    model = object.__new__(NemotronH_Nano_VL_V2)
    language_model = _LanguageModel()
    vision_model = _VisionModel()
    object.__setattr__(model, "model_config", _ImageOnlyModelConfig())
    object.__setattr__(model, "language_model", language_model)
    object.__setattr__(model, "mlp1", _AdapterModule())
    object.__setattr__(model, "vision_model", vision_model)
    object.__setattr__(model, "sound_encoder", None)

    language_weight = object()
    vision_weight = _FakeTensor()
    model.load_weights(
        [
            ("language_model.layers.0.weight", language_weight),
            ("vision_model.radio_model.encoder.weight", vision_weight),
        ]
    )

    assert language_model.loaded_weights == [("layers.0.weight", language_weight)]
    assert vision_model.loaded_weights == [
        ("radio_model.encoder.weight", vision_weight)
    ]


def test_nano_nemotron_vl_requires_sound_encoder_for_sound_weights():
    model = object.__new__(NemotronH_Nano_VL_V2)
    language_model = _LanguageModel()
    vision_model = _VisionModel()
    object.__setattr__(model, "model_config", _ImageOnlyModelConfig())
    object.__setattr__(model, "language_model", language_model)
    object.__setattr__(model, "mlp1", _AdapterModule())
    object.__setattr__(model, "vision_model", vision_model)
    object.__setattr__(model, "sound_encoder", None)

    with pytest.raises(AssertionError):
        model.load_weights([("sound_encoder.encoder.weight", object())])


def _make_radio_final_layernorm_model():
    model = object.__new__(NemotronH_Nano_VL_V2)
    for name, value in {
        "model_config": _ImageOnlyModelConfig(),
        "language_model": _LanguageModel(),
        "mlp1": _AdapterModule(),
        "vision_model": _VisionModel(),
        "vision_final_layernorm": nn.LayerNorm(4, eps=1.0e-6).float(),
        "_loaded_vision_final_layernorm_params": set(),
        "_vision_final_layernorm_enabled": False,
        "sound_encoder": None,
    }.items():
        object.__setattr__(model, name, value)
    return model


@pytest.mark.parametrize(
    "prefix",
    ["vision_final_layernorm.", "vision_projector.vision_final_layernorm."],
)
@pytest.mark.parametrize("first", ["weight", "bias"])
def test_radio_final_layernorm_waits_for_both_tensors_and_refits(prefix, first):
    model = _make_radio_final_layernorm_model()
    inputs = torch.tensor([[[1.0, 2.0, 4.0, 8.0]]], dtype=torch.bfloat16)
    tensors = {
        "weight": torch.tensor([1.0, 1.5, 2.0, 2.5], dtype=torch.bfloat16),
        "bias": torch.tensor([-0.5, 0.0, 0.5, 1.0], dtype=torch.bfloat16),
    }
    second = "bias" if first == "weight" else "weight"
    model.load_weights([(prefix + first, tensors[first])])
    assert not model._vision_final_layernorm_enabled
    assert model._apply_vision_final_layernorm(inputs) is inputs

    model.load_weights([(prefix + second, tensors[second])])
    assert model._vision_final_layernorm_enabled
    assert model.vision_model.loaded_weights == []
    assert model.vision_final_layernorm.weight.dtype == torch.float32
    for name, tensor in tensors.items():
        torch.testing.assert_close(
            getattr(model.vision_final_layernorm, name), tensor.float()
        )
    expected = torch.nn.functional.layer_norm(
        inputs.float(), (4,), tensors["weight"].float(), tensors["bias"].float(), 1e-6
    ).to(inputs.dtype)
    torch.testing.assert_close(model._apply_vision_final_layernorm(inputs), expected)

    # A later refit must replace the checkpoint tensors in the live norm.
    tensors["bias"] += 1
    model.load_weights([(prefix + "bias", tensors["bias"])])
    expected_refit = torch.nn.functional.layer_norm(
        inputs.float(), (4,), tensors["weight"].float(), tensors["bias"].float(), 1e-6
    ).to(inputs.dtype)
    torch.testing.assert_close(
        model._apply_vision_final_layernorm(inputs), expected_refit
    )
    assert not torch.equal(expected_refit, expected)


def test_radio_final_layernorm_preserves_streamed_weights():
    model = _make_radio_final_layernorm_model()
    buffer = torch.empty(4)

    def weights():
        yield "vision_final_layernorm.weight", buffer.fill_(2)
        yield "vision_final_layernorm.bias", buffer.fill_(3)
        buffer.fill_(99)

    # Exercise the drain path when the language loader does not consume input.
    model.language_model.load_weights = lambda weights: None
    model.load_weights(weights())
    assert model._vision_final_layernorm_enabled
    torch.testing.assert_close(
        model.vision_final_layernorm.weight, torch.full((4,), 2.0)
    )
    torch.testing.assert_close(model.vision_final_layernorm.bias, torch.full((4,), 3.0))


@pytest.mark.parametrize("mode", ["unloaded", "no_norm", "text_only"])
def test_radio_final_layernorm_preserves_unpatched_checkpoint_behavior(mode):
    model = _make_radio_final_layernorm_model()
    if mode == "no_norm":
        object.__setattr__(model, "vision_final_layernorm", None)
    elif mode == "text_only":
        object.__setattr__(model, "model_config", _ModelConfig())
        object.__setattr__(model, "vision_final_layernorm", _MissingMultiModalModule())
    if mode != "unloaded":
        model.load_weights(
            [
                ("vision_final_layernorm." + name, torch.ones(4))
                for name in ("weight", "bias")
            ]
        )
    inputs = torch.randn(2, 3, 4)
    assert not model._vision_final_layernorm_enabled
    assert model._apply_vision_final_layernorm(inputs) is inputs


@pytest.mark.parametrize("path", ["dynamic_image", "image", "video"])
def test_radio_final_layernorm_runs_before_projection(path):
    model = _make_radio_final_layernorm_model()
    model.load_weights(
        [
            ("vision_final_layernorm.weight", torch.tensor([1.0, 1.5, 2.0, 2.5])),
            ("vision_final_layernorm.bias", torch.tensor([-0.5, 0.0, 0.5, 1.0])),
        ]
    )
    features = torch.tensor([[[1.0, 2.0, 4.0, 8.0]]], dtype=torch.bfloat16)
    expected = model.vision_final_layernorm(features.float()).to(torch.bfloat16)

    def vision(pixels, **kwargs):
        if path == "video":
            assert kwargs["num_frames"] == 2
        return None, features.clone()

    def project(x):
        torch.testing.assert_close(x, expected)
        return x + 3

    for name, value in {
        "vision_model": vision,
        "mlp1": project,
        "patch_size": 2,
        "downsample_ratio": 1.0,
        "ps_version": "v2",
        "video_temporal_patch_size": 2,
    }.items():
        object.__setattr__(model, name, value)
    pixels = torch.zeros(2 if path == "video" else 1, 3, 2, 2)
    if path == "dynamic_image":
        result = model.extract_feature_dynamic(pixels, imgs_sizes=[(2, 2)])
    else:
        result = model.extract_feature(
            pixels, num_frames=2 if path == "video" else None
        )
    torch.testing.assert_close(result, expected + 3)

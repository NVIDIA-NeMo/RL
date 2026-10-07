"""Exercise the patched vLLM model methods with CPU tensors and small towers."""

import ast
import inspect
import logging
from collections.abc import Iterable
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nemo_rl.models.generation.vllm import patches

pytestmark = pytest.mark.vllm


@pytest.fixture
def patched_model(tmp_path, monkeypatch):
    source = patches._get_vllm_file("model_executor/models/nano_nemotron_vl.py")
    with open(source) as file:
        content = file.read()
    tree = ast.parse(inspect.getsource(patches._patch_vllm_radio_final_layernorm))
    pairs = next(
        ast.literal_eval(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "replacements"
            for target in node.targets
        )
    )
    for old, new in reversed(pairs):
        content = content.replace(new, old, 1)
    target = tmp_path / "nano_nemotron_vl.py"
    target.write_text(content)
    monkeypatch.setattr(patches, "_get_vllm_file", lambda relative: str(target))
    patches._patch_vllm_radio_final_layernorm(logging.getLogger(__name__))
    patched = target.read_text()
    patches._patch_vllm_radio_final_layernorm(logging.getLogger(__name__))
    assert target.read_text() == patched

    model_node = next(
        node
        for node in ast.parse(patched).body
        if isinstance(node, ast.ClassDef) and node.name == "NemotronH_Nano_VL_V2"
    )
    names = {
        "_apply_vision_final_layernorm",
        "load_weights",
        "get_mm_mapping",
        "extract_feature",
        "extract_feature_dynamic",
    }
    methods = [
        node
        for node in model_node.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    tree = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *methods,
        ],
        type_ignores=[],
    )

    def load(param, weight):
        param.copy_(weight)

    namespace = dict(
        torch=torch,
        nn=nn,
        Iterable=Iterable,
        default_weight_loader=load,
        logger=SimpleNamespace(info_once=lambda *args, **kwargs: None),
        MultiModelKeys=SimpleNamespace(from_string_field=lambda **kwargs: kwargs),
    )
    exec(compile(ast.fix_missing_locations(tree), str(target), "exec"), namespace)
    cls = type("PatchedModel", (), {name: namespace[name] for name in names})
    model = cls()
    constructor = next(
        node
        for node in model_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    tower = next(
        node
        for node in constructor.body
        if isinstance(node, ast.With)
        and any(
            isinstance(stmt, ast.AnnAssign)
            and isinstance(stmt.target, ast.Attribute)
            and stmt.target.attr == "vision_final_layernorm"
            for stmt in node.body
        )
    )
    start = next(
        i
        for i, node in enumerate(tower.body)
        if isinstance(node, ast.AnnAssign)
        and node.target.attr == "vision_final_layernorm"
    )
    end = next(
        i
        for i, node in enumerate(tower.body)
        if isinstance(node, ast.AnnAssign) and node.target.attr == "sound_encoder"
    )
    initialization = ast.Module(body=tower.body[start:end], type_ignores=[])

    def initialize(mtp_layers):
        exec(
            compile(initialization, str(target), "exec"),
            dict(
                self=model,
                nn=nn,
                vit_hidden_size=4,
                config=SimpleNamespace(
                    text_config=SimpleNamespace(num_nextn_predict_layers=mtp_layers)
                ),
                vision_config=SimpleNamespace(layer_norm_eps=1e-6),
            ),
        )

    initialize(1)
    model.initialize_final_norm = initialize
    model.model_config = SimpleNamespace(
        multimodal_config=SimpleNamespace(get_limit_per_prompt=lambda modality: 1)
    )
    model.language_model = SimpleNamespace(
        load_weights=lambda weights: next(iter(weights), None)
    )
    model.mlp1 = nn.Identity()
    model.vision_model = SimpleNamespace(load_weights=lambda weights: None)
    model.sound_encoder = None
    return model, target


@pytest.mark.parametrize(
    "prefix", ["vision_final_layernorm.", "vision_projector.vision_final_layernorm."]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_final_norm_load_forward_and_refit(patched_model, prefix, dtype):
    model, _ = patched_model
    inputs = torch.tensor([[[1.0, 2.0, 4.0, 8.0], [2.0, 3.0, 5.0, 9.0]]], dtype=dtype)
    assert model._apply_vision_final_layernorm(inputs) is inputs
    for scale in (1.0, 2.0):
        weight = torch.tensor([1.0, 1.5, 2.0, 2.5], dtype=torch.bfloat16) * scale
        bias = torch.tensor([-0.5, 0.0, 0.5, 1.0], dtype=torch.bfloat16) * scale
        model.load_weights([(prefix + "weight", weight), (prefix + "bias", bias)])
        expected = torch.nn.functional.layer_norm(
            inputs.float(), (4,), weight.float(), bias.float(), 1e-6
        ).to(dtype)
        torch.testing.assert_close(
            model._apply_vision_final_layernorm(inputs), expected, rtol=0, atol=0
        )
        assert model.vision_final_layernorm.weight.dtype == torch.float32
    assert "vision_final_layernorm" in model.get_mm_mapping()["connector"]


def test_final_norm_waits_for_both_parameters(patched_model):
    model, _ = patched_model
    model.load_weights([("vision_final_layernorm.weight", torch.ones(4))])
    assert not model._vision_final_layernorm_enabled
    model.load_weights(
        [("vision_projector.vision_final_layernorm.bias", torch.zeros(4))]
    )
    assert model._vision_final_layernorm_enabled


def test_final_norm_refit_clones_reused_stream_buffers(patched_model):
    model, _ = patched_model

    def stream():
        buffer = torch.full((4,), 2.0)
        yield "vision_final_layernorm.weight", buffer
        buffer.fill_(3.0)
        yield "vision_final_layernorm.bias", buffer
        buffer.fill_(99.0)

    model.load_weights(stream())
    assert torch.equal(model.vision_final_layernorm.weight, torch.full((4,), 2.0))
    assert torch.equal(model.vision_final_layernorm.bias, torch.full((4,), 3.0))


@pytest.mark.parametrize("frames", [None, 1])
def test_final_norm_applied_before_projection_in_both_paths(patched_model, frames):
    model, _ = patched_model
    features = torch.tensor(
        [[[1.0, 2.0, 4.0, 8.0], [2.0, 3.0, 5.0, 9.0]]], dtype=torch.bfloat16
    )
    model.load_weights(
        [
            ("vision_final_layernorm.weight", torch.ones(4)),
            ("vision_final_layernorm.bias", torch.zeros(4)),
        ]
    )
    model.vision_model = lambda pixels, **kwargs: (None, features)
    model.patch_size = model.video_temporal_patch_size = 1
    model.downsample_ratio = 1
    model.pixel_shuffle = lambda features, **kwargs: features
    model.pixel_shuffle_dynamic_res = lambda features, **kwargs: features
    pixels = torch.zeros(1, 3, 1, 2)
    expected = model._apply_vision_final_layernorm(features)
    torch.testing.assert_close(
        model.extract_feature(pixels, num_frames=frames), expected, rtol=0, atol=0
    )
    torch.testing.assert_close(
        model.extract_feature_dynamic(pixels, imgs_sizes=[(1, 2)]),
        expected,
        rtol=0,
        atol=0,
    )


def test_final_norm_disabled_multimodal_and_missing_module(patched_model):
    model, _ = patched_model
    model.vision_final_layernorm = None
    with pytest.raises(ValueError, match="did not construct"):
        model.load_weights([("vision_final_layernorm.weight", torch.ones(4))])
    model.model_config.multimodal_config.get_limit_per_prompt = lambda modality: 0
    model.load_weights([("vision_final_layernorm.weight", torch.ones(4))])
    assert not model._vision_final_layernorm_enabled


def test_final_norm_patch_rejects_unknown_source_atomically(tmp_path, monkeypatch):
    target = tmp_path / "nano_nemotron_vl.py"
    target.write_text("class NemotronH_Nano_VL_V2: pass\n")
    monkeypatch.setattr(patches, "_get_vllm_file", lambda relative: str(target))
    with pytest.raises(RuntimeError, match="requires the vLLM 0.29"):
        patches._patch_vllm_radio_final_layernorm(logging.getLogger(__name__))
    assert target.read_text() == "class NemotronH_Nano_VL_V2: pass\n"


@pytest.mark.parametrize("mtp_layers", [0, 1])
def test_final_norm_constructor_keeps_startup_disabled(patched_model, mtp_layers):
    model, _ = patched_model
    model.initialize_final_norm(mtp_layers)
    assert (model.vision_final_layernorm is not None) == (mtp_layers > 0)
    assert not model._vision_final_layernorm_enabled
    assert model._loaded_vision_final_layernorm_params == set()
    inputs = torch.randn(1, 2, 4)
    assert model._apply_vision_final_layernorm(inputs) is inputs

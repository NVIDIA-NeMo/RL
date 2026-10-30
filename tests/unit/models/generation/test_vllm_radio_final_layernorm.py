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

"""Exercise the patched vLLM model methods with CPU tensors and small towers."""

import ast
import importlib.util
import inspect
import json
import logging
import sys
from collections.abc import Iterable
from pathlib import Path
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
    weight = torch.tensor([0.5, 2.0, 3.0, 1.5])
    bias = torch.tensor([1.0, -2.0, 0.25, 4.0])
    model.load_weights(
        [
            ("vision_final_layernorm.weight", weight),
            ("vision_final_layernorm.bias", bias),
        ]
    )
    model.mlp1 = nn.Linear(4, 4, dtype=torch.bfloat16)
    with torch.no_grad():
        model.mlp1.weight.copy_(
            torch.tensor([[1, 2, -1, 0], [-2, 1, 3, 1], [0, -1, 2, 4], [3, 0, -2, 1]])
        )
        model.mlp1.bias.copy_(torch.tensor([1.0, -0.5, 2.0, -1.0]))
    model.vision_model = lambda pixels, **kwargs: (None, features)
    model.patch_size = model.video_temporal_patch_size = 1
    model.downsample_ratio = 1
    model.pixel_shuffle = lambda features, **kwargs: features
    model.pixel_shuffle_dynamic_res = lambda features, **kwargs: features
    pixels = torch.zeros(1, 3, 1, 2)
    normalized = nn.functional.layer_norm(features.float(), (4,), weight, bias, 1e-6)
    expected = model.mlp1(normalized.to(torch.bfloat16))
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


@pytest.fixture
def composed_radio_modules(tmp_path, monkeypatch):
    """Import complete model sources after the normal NeMo compatibility setup."""
    sources = {}
    for relative in (
        "model_executor/models/radio.py",
        "model_executor/models/nano_nemotron_vl.py",
        "tool_parsers/utils.py",
        "model_executor/models/deepseek_v2.py",
    ):
        source = Path(patches._get_vllm_file(relative))
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source.read_text())
        sources[relative] = target
    monkeypatch.setattr(
        patches, "_get_vllm_file", lambda relative: str(sources[relative])
    )
    patches.ensure_vllm_source_compat()
    before = {name: path.read_text() for name, path in sources.items()}
    patches.ensure_vllm_source_compat()
    assert before == {name: path.read_text() for name, path in sources.items()}

    modules = []
    for name, relative in (
        ("vllm.model_executor.models.radio", "model_executor/models/radio.py"),
        (
            "vllm.model_executor.models._nemo_radio_refit_test",
            "model_executor/models/nano_nemotron_vl.py",
        ),
    ):
        spec = importlib.util.spec_from_file_location(name, sources[relative])
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        modules.append(module)
    return modules, sources


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Requires the vLLM GPU runner"
)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    ("reload_api", "real_quant"),
    [(False, True), (True, True), (True, False)],
    ids=["modelopt-chunked", "modelopt-native", "bf16-native"],
)
@pytest.mark.parametrize("scale_source", ["explicit", "folded", "unverified"])
@torch.inference_mode()
def test_composed_native_vision_refit(
    composed_radio_modules,
    tmp_path,
    monkeypatch,
    device,
    reload_api,
    scale_source,
    real_quant: bool,
):
    """Real loaders/lifecycle: late scales, folded defaults, frozen omission and graphs."""
    from vllm.config import MultiModalConfig, VllmConfig, set_current_vllm_config
    from vllm.model_executor.model_loader.reload import record_metadata_for_reloading
    from vllm.model_executor.model_loader.weight_utils import initialize_dummy_weights
    from vllm.model_executor.models.utils import AutoWeightsLoader
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    from nemo_rl.modelopt.models.generation.vllm_quant_backend import (
        VllmQuantInternalWorkerExtension,
    )
    from nemo_rl.models.generation.vllm import vllm_backend

    (radio_module, nano_module), _ = composed_radio_modules
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    config = SimpleNamespace(
        model=str(checkpoint),
        revision=None,
        dtype=torch.bfloat16,
        quantization="modelopt" if real_quant else None,
        architectures=["NemotronH_Nano_VL_V2"],
        hf_config=SimpleNamespace(
            vision_config=SimpleNamespace(model_type="radio", layerscale_value=1.0)
        ),
        multimodal_config=MultiModalConfig(),
    )
    with torch.device(device):
        model = nano_module.NemotronH_Nano_VL_V2.__new__(
            nano_module.NemotronH_Nano_VL_V2
        )
        nn.Module.__init__(model)
        model.model_config = config
        model.language_model = nn.Module()
        model.language_model.load_weights = AutoWeightsLoader(
            model.language_model
        ).load_weights
        model.sound_encoder = None
        model.mlp1 = nn.Sequential(
            nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 4)
        ).bfloat16()
        model.vision_final_layernorm = nn.LayerNorm(4, eps=1e-6).float()
        model._loaded_vision_final_layernorm_params = set()
        model._vision_final_layernorm_enabled = False
        radio = radio_module.RadioModel.__new__(radio_module.RadioModel)
        nn.Module.__init__(radio)
        radio.config = SimpleNamespace(model_type="radio", initializer_factor=1.0)
        radio.model = nn.Module()
        radio.model.encoder = nn.Module()
        layer = nn.Module()
        layer.ls1 = nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
        layer.ls2 = nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
        layer.attn = nn.Module()
        layer.attn.proj = nn.Linear(4, 4).bfloat16()
        radio.model.encoder.layers = nn.ModuleList([layer])
        model.vision_model = radio
        initialize_dummy_weights(model, config)
        expected = {name: param.clone() for name, param in model.named_parameters()}
        pointers = {name: param.data_ptr() for name, param in model.named_parameters()}
        record_metadata_for_reloading(model)
        inputs = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4) / 8

        runner = GPUModelRunner.__new__(GPUModelRunner)
        runner.model = model
        runner.model_config = config
        runner.vllm_config = VllmConfig()
        runner.vllm_config.model_config = config
        runner.lora_config = None
        runner.mm_budget = None
        runner.encoder_cache = {}
        runner.late_interaction_runner = {}
        runner.drafter = None
        worker_cls = (
            VllmQuantInternalWorkerExtension
            if real_quant
            else vllm_backend.VllmInternalWorkerExtension
        )
        ext = worker_cls.__new__(worker_cls)
        ext.model_runner, ext.model_config = runner, config
        ext.device = torch.device(device)
        ext.model_update_group = object()
        mapping = {
            name: name for name in expected if not name.startswith("vision_model.")
        }
        for leaf in ("weight", "bias"):
            mapping[f"vision_model.encoder.layer.0.attention.output.dense.{leaf}"] = (
                f"vision_model.model.encoder.layers.0.attn.proj.{leaf}"
            )
        for index in (1, 2):
            mapping[f"vision_model.encoder.layer.0.layer_scale{index}.lambda1"] = (
                f"vision_model.model.encoder.layers.0.ls{index}"
            )
        index_keys = {
            name: "weights.safetensors"
            for name in mapping
            if scale_source != "folded" or "layer_scale" not in name
        }
        (checkpoint / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": index_keys})
        )

        def forward():
            features = layer.attn.proj(inputs) * layer.ls1 + inputs * layer.ls2
            return model.mlp1(model._apply_vision_final_layernorm(features))

        def reference():
            state = expected
            prefix = "vision_model.model.encoder.layers.0."
            features = nn.functional.linear(
                inputs,
                state[prefix + "attn.proj.weight"],
                state[prefix + "attn.proj.bias"],
            )
            features = features * state[prefix + "ls1"] + inputs * state[prefix + "ls2"]
            features = nn.functional.layer_norm(
                features.float(),
                (4,),
                state["vision_final_layernorm.weight"],
                state["vision_final_layernorm.bias"],
                1e-6,
            ).bfloat16()
            hidden = nn.functional.gelu(
                nn.functional.linear(
                    features, state["mlp1.0.weight"], state["mlp1.0.bias"]
                )
            )
            return nn.functional.linear(
                hidden, state["mlp1.2.weight"], state["mlp1.2.bias"]
            )

        graph = None
        for step in range(4):
            names = list(mapping)
            if step == 0 and scale_source != "explicit":
                names = [name for name in names if "layer_scale" not in name]
                for index in (1, 2):
                    expected[f"vision_model.model.encoder.layers.0.ls{index}"].fill_(1)
            elif step == 2:
                names = [name for name in names if name.startswith("mlp1.")]
            elif step == 3:
                names = ["vision_model.encoder.layer.0.layer_scale1.lambda1"]
            payload = []
            for name in names:
                target = expected[mapping[name]]
                value = (
                    (step + 1) / 10
                    + torch.arange(target.numel()).reshape(target.shape) / 100
                ).bfloat16()
                payload.append((name, value))
                expected[mapping[name]] = value.to(target.dtype).clone()
            ext.state_dict_info = {
                name: (tuple(value.shape), value.dtype) for name, value in payload
            }

            def consume(*, post_unpack_func, return_iterator=False, **kwargs):
                def stream():
                    storage = torch.empty(
                        max(value.numel() for _, value in payload), dtype=torch.bfloat16
                    )
                    for name, value in payload:
                        view = storage[: value.numel()].view(value.shape)
                        view.copy_(value)
                        yield name, view
                    storage.fill_(99)

                if return_iterator:
                    return stream()
                for item in stream():
                    post_unpack_func([item])

            monkeypatch.setattr(vllm_backend, "packed_broadcast_consumer", consume)
            runner.encoder_cache["stale"] = inputs
            runner.late_interaction_runner["stale"] = inputs
            with set_current_vllm_config(runner.vllm_config):
                if scale_source == "unverified":
                    if real_quant:
                        with pytest.raises(ValueError, match="LayerScale"):
                            ext._update_weights_from_collective(reload_api)
                    else:
                        assert not ext._update_weights_from_collective(reload_api)
                    assert "load_weights" not in vars(model)
                    return
                assert ext._update_weights_from_collective(reload_api)
            for name, param in model.named_parameters():
                torch.testing.assert_close(param, expected[name], rtol=0, atol=0)
                assert param.data_ptr() == pointers[name]
            assert not runner.encoder_cache and not runner.late_interaction_runner
            assert "load_weights" not in vars(model)
            if device == "cuda":
                if graph is None:
                    for _ in range(3):
                        forward()
                    torch.accelerator.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        output = forward()
                graph.replay()
                actual = output
            else:
                actual = forward()
            torch.testing.assert_close(actual, reference(), rtol=1e-2, atol=1e-3)

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

"""Guards for vLLM source patches and scoped runtime workarounds.

The two port patches ship their own suites. This module covers the other
source-sensitive compatibility patches:

* ``_patch_vllm_tool_parser_namespace_tool`` is the most load-bearing patch in
  the repo -- it is the only thing that makes vLLM 0.25.1 importable against
  the pinned ``openai==2.6.1``. If upstream reorders that import block the
  patch logs a warning and returns, and every engine then dies on
  ``import vllm.tool_parsers``. So the anchor needs pinning.
* ``_patch_vllm_glm_decoder_sequence_parallel_moe`` restores the vLLM 0.24
  decoder boundary for GLM-5.1/5.2 while leaving MoE-local SP enabled.
* the ``VLLM_RAY_EXTRA_ENV_VARS_TO_COPY`` merge replaced the old
  ``ADDITIONAL_ENV_VARS`` file patch and is what now carries
  ``RAY_ENABLE_UV_RUN_RUNTIME_ENV`` and every user ``extra_env_vars`` to the
  Ray workers. Being additive rather than clobbering is the whole point of the
  rewrite, and it is pure string handling, so it is cheap to pin.
* ``modelopt_moe_amax_aliases`` adapts nested ModelOpt buffers to vLLM's
  MoE refit loader. Its lifecycle and installed-loader compatibility are
  checked here, alongside the source patches.
"""

import ast
import logging
import os
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from nemo_rl.models.generation.vllm import patches
from nemo_rl.models.generation.vllm.patches import (
    VLLM_NEMOTRON_H_FP32_LM_HEAD_ENV_VAR,
)
from tests.unit.models.generation.vllm_patch_source_utils import (
    patch_snippets,
    write_unpatched_copy,
)

_TOOL_PARSER_SOURCE = "tool_parsers/utils.py"
_PATCH_FN = "_patch_vllm_tool_parser_namespace_tool"
_MARKER = "except ImportError:  # openai < 2.25.0 predates namespace tools"
_RADIO_SOURCE = "model_executor/models/radio.py"
_RADIO_PATCH_FN = "_patch_vllm_radio_layerscale_loader"
_RADIO_MARKER = "ls_param.data.fill_(initializer_factor)"
_GLM_DSA_SOURCE = "model_executor/models/deepseek_v2.py"
_GLM_DSA_PATCH_FN = "_patch_vllm_glm_decoder_sequence_parallel_moe"
_GLM_DSA_MARKER = 'getattr(config, "model_type", None) != "glm_moe_dsa"'
_MOE_SOURCE = "model_executor/layers/fused_moe/runner/moe_runner.py"
_MOE_PATCH_FN = "_patch_vllm_moe_routed_experts_capture"
_MOE_MARKER = "NeMo-RL patch (routed-experts capture for router replay)"
_CAPTURER_SOURCE = "model_executor/layers/fused_moe/routed_experts_capturer.py"
_CAPTURER_PATCH_FN = "_patch_vllm_routed_experts_capture_router_fallback"
_CAPTURER_MARKER = (
    "NeMo-RL patch (router fallback for monolithic routed-experts capture)"
)


@pytest.fixture
def modelopt_moe_model() -> torch.nn.Module:
    model = torch.nn.Module()
    model.experts = torch.nn.Module()
    for name in ("w13_input_quantizer", "w2_input_quantizer"):
        quantizer = torch.nn.Module()
        quantizer.register_buffer("_amax", torch.tensor(-1.0))
        model.experts.add_module(name, quantizer)
    model.in_proj = torch.nn.Linear(1, 1)
    model.in_proj.input_quantizer = torch.nn.Module()
    model.in_proj.input_quantizer.register_buffer("_amax", torch.tensor(-1.0))
    return model


def test_modelopt_moe_amax_aliases_preserve_buffer_identity_and_registration(
    modelopt_moe_model: torch.nn.Module,
) -> None:
    model = modelopt_moe_model
    buffers_before = list(model.named_buffers())
    parameters_before = list(model.named_parameters())
    state_before = {name: value.clone() for name, value in model.state_dict().items()}

    with patches.modelopt_moe_amax_aliases(model):
        for name in ("w13_input_quantizer", "w2_input_quantizer"):
            assert (
                getattr(model.experts, f"{name}._amax")
                is getattr(model.experts, name)._amax
            )
        assert list(model.named_buffers()) == buffers_before
        assert list(model.named_parameters()) == parameters_before
        torch.testing.assert_close(model.state_dict(), state_before)
        assert not hasattr(model.in_proj, "input_quantizer._amax")

    assert not hasattr(model.experts, "w13_input_quantizer._amax")
    assert not hasattr(model.experts, "w2_input_quantizer._amax")
    torch.testing.assert_close(model.state_dict(), state_before)


def test_modelopt_moe_amax_aliases_support_nested_and_repeated_use(
    modelopt_moe_model: torch.nn.Module,
) -> None:
    model = modelopt_moe_model
    for _ in range(2):
        with patches.modelopt_moe_amax_aliases(model):
            with patches.modelopt_moe_amax_aliases(model):
                assert getattr(model.experts, "w13_input_quantizer._amax") is (
                    model.experts.w13_input_quantizer._amax
                )
            assert hasattr(model.experts, "w13_input_quantizer._amax")
        assert not hasattr(model.experts, "w13_input_quantizer._amax")
        assert not hasattr(model.experts, "w2_input_quantizer._amax")


def test_modelopt_moe_amax_aliases_preserve_existing_attributes(
    modelopt_moe_model: torch.nn.Module,
) -> None:
    model = modelopt_moe_model
    existing = torch.tensor(123.0)
    setattr(model.experts, "w13_input_quantizer._amax", existing)
    with patches.modelopt_moe_amax_aliases(model):
        assert getattr(model.experts, "w13_input_quantizer._amax") is existing
        assert hasattr(model.experts, "w2_input_quantizer._amax")
    assert getattr(model.experts, "w13_input_quantizer._amax") is existing
    assert not hasattr(model.experts, "w2_input_quantizer._amax")


@pytest.mark.parametrize("during_setup", [False, True])
def test_modelopt_moe_amax_aliases_clean_up_on_error(
    modelopt_moe_model: torch.nn.Module,
    monkeypatch: pytest.MonkeyPatch,
    during_setup: bool,
) -> None:
    model = modelopt_moe_model

    def fail_buffer_scan(*, recurse: bool = True) -> None:
        # The first quantizer's alias must already exist when the second fails.
        assert hasattr(model.experts, "w13_input_quantizer._amax")
        raise RuntimeError("quantizer scan failed")

    if during_setup:
        monkeypatch.setattr(
            model.experts.w2_input_quantizer, "named_buffers", fail_buffer_scan
        )
    expected = "quantizer scan failed" if during_setup else "refit failed"
    with pytest.raises(RuntimeError, match=expected):
        with patches.modelopt_moe_amax_aliases(model):
            raise RuntimeError("refit failed")
    assert not hasattr(model.experts, "w13_input_quantizer._amax")
    assert not hasattr(model.experts, "w2_input_quantizer._amax")


@pytest.mark.vllm
def test_modelopt_moe_amax_aliases_satisfy_installed_vllm_loader(
    modelopt_moe_model: torch.nn.Module,
) -> None:
    # Keep the optional vLLM import inside the test selected by its marker.
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

    class LoaderFixture(torch.nn.Module):
        """CPU state for the installed loader and its real expert mapping."""

        load_weights = RoutedExperts.load_weights
        get_expert_mapping = RoutedExperts.get_expert_mapping
        build_expert_params_mapping = staticmethod(
            RoutedExperts.build_expert_params_mapping
        )

        def __init__(self) -> None:
            super().__init__()
            self.layer_name = "model.layers.1.mixer.experts"
            self.moe_config = SimpleNamespace(
                hidden_dim_unpadded=1, num_experts=2, num_logical_experts=2
            )
            self.expert_map_manager = SimpleNamespace(num_fused_shared_experts=0)
            self.ckpt_gate_proj_name = "up_proj"
            self.ckpt_down_proj_name = "down_proj"
            self.ckpt_up_proj_name = ""
            self.lora_base_layer_prefix = ""

    def load_amax(
        param: torch.Tensor, loaded_weight: torch.Tensor, **kwargs: object
    ) -> bool:
        param.copy_(torch.maximum(param, loaded_weight))
        return True

    model = modelopt_moe_model
    experts = LoaderFixture()
    for name, quantizer in model.experts.named_children():
        experts.add_module(name, quantizer)
        quantizer._amax.weight_loader = load_amax
    model.experts = experts
    weights = [
        (f"{expert}.{projection}.input_quantizer._amax", torch.tensor(value))
        for expert, projection, value in (
            (0, "up_proj", 2.0),
            (1, "up_proj", 6.0),
            (0, "down_proj", 1.0),
            (1, "down_proj", 0.25),
        )
    ]

    with pytest.raises(AttributeError, match=r"w13_input_quantizer\._amax"):
        list(experts.load_weights(weights))
    with patches.modelopt_moe_amax_aliases(model):
        loaded = list(experts.load_weights(weights))
        assert loaded == [
            "w13_input_quantizer._amax",
            "w13_input_quantizer._amax",
            "w2_input_quantizer._amax",
            "w2_input_quantizer._amax",
        ]
    torch.testing.assert_close(experts.w13_input_quantizer._amax, torch.tensor(6.0))
    torch.testing.assert_close(experts.w2_input_quantizer._amax, torch.tensor(1.0))
    assert not hasattr(experts, "w13_input_quantizer._amax")
    assert not hasattr(experts, "w2_input_quantizer._amax")


@pytest.fixture
def patched_tool_parser_source(tmp_path, monkeypatch):
    """The installed tool_parsers/utils.py, unpatched then patched in tmp."""
    copied = write_unpatched_copy(_TOOL_PARSER_SOURCE, _PATCH_FN, tmp_path / "utils.py")
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _relative: str(copied))
    patches._patch_vllm_tool_parser_namespace_tool(logging.getLogger(__name__))
    return copied


@pytest.fixture
def patched_radio_source(tmp_path, monkeypatch):
    """The installed vLLM RADIO loader, unpatched then patched in tmp."""
    copied = write_unpatched_copy(_RADIO_SOURCE, _RADIO_PATCH_FN, tmp_path / "radio.py")
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _relative: str(copied))
    patches._patch_vllm_radio_layerscale_loader(logging.getLogger(__name__))
    return copied


@pytest.fixture
def patched_glm_dsa_source(tmp_path, monkeypatch):
    """The installed GLM/DeepSeek model source, unpatched then patched in tmp."""
    copied = write_unpatched_copy(
        _GLM_DSA_SOURCE, _GLM_DSA_PATCH_FN, tmp_path / "deepseek_v2.py"
    )
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _relative: str(copied))
    patches._patch_vllm_glm_decoder_sequence_parallel_moe(logging.getLogger(__name__))
    return copied


@pytest.fixture
def patched_moe_source(tmp_path, monkeypatch):
    """The installed monolithic MoE runner, unpatched then patched in tmp."""
    copied = write_unpatched_copy(
        _MOE_SOURCE, _MOE_PATCH_FN, tmp_path / "moe_runner.py"
    )
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _relative: str(copied))
    assert patches._patch_vllm_moe_routed_experts_capture(
        logging.getLogger(__name__), required=True
    )
    return copied


@pytest.mark.vllm
def test_namespace_tool_patch_anchor_still_matches_installed_vllm(
    patched_tool_parser_source,
):
    """A source edit becomes a silent no-op if upstream reorders the import."""
    content = patched_tool_parser_source.read_text()
    assert _MARKER in content, (
        "the NamespaceTool compat patch did not apply to the installed vLLM; "
        "its anchor import block has probably changed upstream. Every vLLM "
        "engine will fail to import tool_parsers against the pinned openai."
    )
    ast.parse(content)  # the edit must leave valid Python


@pytest.mark.vllm
def test_namespace_tool_patch_is_idempotent(patched_tool_parser_source, monkeypatch):
    """Every worker on a node runs the patch against the same file."""
    before = patched_tool_parser_source.read_text()
    monkeypatch.setattr(
        patches, "_get_vllm_file", lambda _relative: str(patched_tool_parser_source)
    )
    patches._patch_vllm_tool_parser_namespace_tool(logging.getLogger(__name__))
    assert patched_tool_parser_source.read_text() == before


@pytest.mark.vllm
def test_namespace_tool_stub_never_matches(patched_tool_parser_source):
    """The stub must be a plain class, so isinstance() is always False.

    All upstream uses are ``isinstance(tool, NamespaceTool)`` guarding a
    namespace-tools branch, so degrading to "no namespace tools" is correct for
    a client that cannot construct them -- but only if nothing can be an
    instance of the stub.
    """
    namespace: dict = {}
    tree = ast.parse(patched_tool_parser_source.read_text())
    stub = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "NamespaceTool"
    )
    exec(compile(ast.Module(body=[stub], type_ignores=[]), "<stub>", "exec"), namespace)
    stub_cls = namespace["NamespaceTool"]
    for value in ({}, "tool", 0, None, object()):
        assert not isinstance(value, stub_cls)


@pytest.mark.vllm
def test_radio_layerscale_patch_anchor_still_matches_installed_vllm(
    patched_radio_source,
):
    """Pin the RADIO loader shape used by the source patch."""
    content = patched_radio_source.read_text()
    assert _RADIO_MARKER in content
    assert "Skip layer-scale entries that vLLM doesn't use" not in content
    ast.parse(content)


@pytest.mark.vllm
def test_radio_layerscale_patch_loads_explicit_and_initializes_folded_weights(
    patched_radio_source,
):
    content = patched_radio_source.read_text()
    assert 'vllm_key = f"model.encoder.layers.{layer_idx}.{suffix}"' in content
    assert 'ls_name.endswith((".ls1", ".ls2"))' in content
    assert "ls_param.data.fill_(initializer_factor)" in content
    assert "loaded_params.add(ls_name)" in content


@pytest.mark.vllm
def test_radio_layerscale_patch_keeps_values_loaded_by_earlier_refit_batches(
    patched_radio_source,
):
    """Refit calls load_weights once per batch; a later batch must not reset LayerScale."""
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader

    tree = ast.parse(patched_radio_source.read_text())
    radio_model = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "RadioModel"
    )
    load_weights = next(
        node
        for node in radio_model.body
        if isinstance(node, ast.FunctionDef) and node.name == "load_weights"
    )
    namespace = {"default_weight_loader": default_weight_loader}
    exec(
        compile(ast.Module(body=[load_weights], type_ignores=[]), "<radio>", "exec"),
        namespace,
    )

    class _Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ls1 = torch.nn.Parameter(torch.full((4,), float("nan")))
            self.ls2 = torch.nn.Parameter(torch.full((4,), float("nan")))

    class _Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([_Layer()])

    class _Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = _Encoder()

    class _Radio(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(initializer_factor=1.0)
            self.model = _Inner()

    model = _Radio()
    layer = model.model.encoder.layers[0]
    with torch.no_grad():
        namespace["load_weights"](
            model, [("radio_model.model.blocks.0.ls1", torch.full((4,), 0.1))]
        )
        assert torch.allclose(layer.ls1, torch.full((4,), 0.1))
        assert torch.allclose(layer.ls2, torch.ones(4))

        # A later batch carrying no vision weights (e.g. LLM-only) must not
        # reset the explicitly loaded ls1 back to initializer_factor.
        namespace["load_weights"](model, [])
    assert torch.allclose(layer.ls1, torch.full((4,), 0.1))
    assert torch.allclose(layer.ls2, torch.ones(4))


@pytest.mark.vllm
def test_radio_layerscale_patch_is_idempotent(patched_radio_source, monkeypatch):
    before = patched_radio_source.read_text()
    monkeypatch.setattr(
        patches, "_get_vllm_file", lambda _relative: str(patched_radio_source)
    )

    patches._patch_vllm_radio_layerscale_loader(logging.getLogger(__name__))

    assert patched_radio_source.read_text() == before


def test_radio_layerscale_patch_warns_on_unknown_source(monkeypatch, tmp_path, caplog):
    radio_source = tmp_path / "radio.py"
    radio_source.write_text("class RadioModel:\n    pass\n")
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _relative: str(radio_source))

    with caplog.at_level(logging.WARNING):
        patches._patch_vllm_radio_layerscale_loader(logging.getLogger(__name__))

    assert radio_source.read_text() == "class RadioModel:\n    pass\n"
    assert "load_weights tail anchor was not found" in caplog.text


@pytest.mark.vllm
def test_glm_decoder_sp_moe_patch_anchor_still_matches_installed_vllm(
    patched_glm_dsa_source,
):
    """Pin the vLLM 0.25.1 decoder-level SP-MoE source shape."""
    content = patched_glm_dsa_source.read_text()
    assert _GLM_DSA_MARKER in content
    ast.parse(content)


@pytest.mark.vllm
def test_moe_routed_experts_patch_anchor_still_matches_installed_vllm(
    patched_moe_source,
):
    content = patched_moe_source.read_text()
    assert _MOE_MARKER in content
    assert "self.router.select_experts(" in content
    assert 'getattr(self.router, "capture_fn", None)' in content
    ast.parse(content)


@pytest.mark.vllm
def test_glm_decoder_sp_moe_patch_is_idempotent(patched_glm_dsa_source, monkeypatch):
    before = patched_glm_dsa_source.read_text()
    monkeypatch.setattr(
        patches, "_get_vllm_file", lambda _relative: str(patched_glm_dsa_source)
    )

    patches._patch_vllm_glm_decoder_sequence_parallel_moe(logging.getLogger(__name__))

    assert patched_glm_dsa_source.read_text() == before


def test_glm_decoder_sp_moe_patch_warns_on_unknown_source(
    monkeypatch, tmp_path, caplog
):
    model_source = tmp_path / "deepseek_v2.py"
    model_source.write_text("class DeepseekV2DecoderLayer:\n    pass\n")
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _relative: str(model_source))

    with caplog.at_level(logging.WARNING):
        patches._patch_vllm_glm_decoder_sequence_parallel_moe(
            logging.getLogger(__name__)
        )

    assert model_source.read_text() == "class DeepseekV2DecoderLayer:\n    pass\n"
    assert "vLLM 0.25.1 source shape was not found" in caplog.text


@pytest.mark.vllm
def test_moe_routed_experts_patch_is_idempotent(patched_moe_source, monkeypatch):
    before = patched_moe_source.read_text()
    monkeypatch.setattr(
        patches, "_get_vllm_file", lambda _relative: str(patched_moe_source)
    )

    assert patches._patch_vllm_moe_routed_experts_capture(
        logging.getLogger(__name__), required=True
    )
    assert patched_moe_source.read_text() == before


def test_moe_routed_experts_patch_fails_closed_when_required(monkeypatch, tmp_path):
    moe_source = tmp_path / "moe_runner.py"
    moe_source.write_text("class MoERunner:\n    pass\n")
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _relative: str(moe_source))

    with pytest.raises(RuntimeError, match="expected code snippet not found"):
        patches._patch_vllm_moe_routed_experts_capture(
            logging.getLogger(__name__), required=True
        )


@pytest.fixture
def patched_capturer_source(tmp_path, monkeypatch):
    """The installed routed-experts binder, unpatched then patched in tmp."""
    copied = write_unpatched_copy(
        _CAPTURER_SOURCE, _CAPTURER_PATCH_FN, tmp_path / "routed_experts_capturer.py"
    )
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _relative: str(copied))
    assert patches._patch_vllm_routed_experts_capture_router_fallback(
        logging.getLogger(__name__), required=True
    )
    return copied


@pytest.mark.vllm
def test_capture_router_fallback_patch_anchor_still_matches_installed_vllm(
    patched_capturer_source,
):
    content = patched_capturer_source.read_text()
    assert _CAPTURER_MARKER in content
    # The patched monolithic block runs from the marker to the original raise.
    monolithic_branch = content.split(_CAPTURER_MARKER, 1)[1]
    monolithic_branch = monolithic_branch.split("not supported with monolithic", 1)[0]
    # In-kernel capture still wins when the kernel supports it ...
    assert "fused_experts.set_capture_fn(capture_fn)" in monolithic_branch
    # ... and a kernel without it now falls back to the router instead of raising.
    assert "module.router.set_capture_fn(capture_fn)" in monolithic_branch
    assert monolithic_branch.index(
        "fused_experts.set_capture_fn"
    ) < monolithic_branch.index("module.router.set_capture_fn")
    ast.parse(content)


@pytest.mark.vllm
def test_capture_router_fallback_patch_is_idempotent(
    patched_capturer_source, monkeypatch
):
    before = patched_capturer_source.read_text()
    monkeypatch.setattr(
        patches, "_get_vllm_file", lambda _relative: str(patched_capturer_source)
    )

    assert patches._patch_vllm_routed_experts_capture_router_fallback(
        logging.getLogger(__name__), required=True
    )
    assert patched_capturer_source.read_text() == before


def test_capture_router_fallback_patch_fails_closed_when_required(
    monkeypatch, tmp_path
):
    source = tmp_path / "routed_experts_capturer.py"
    source.write_text("def bind_routed_experts_capturer(model, capturer):\n    pass\n")
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _relative: str(source))

    with pytest.raises(RuntimeError, match="expected code snippet not found"):
        patches._patch_vllm_routed_experts_capture_router_fallback(
            logging.getLogger(__name__), required=True
        )
    assert source.read_text().startswith("def bind_routed_experts_capturer")


def test_capture_router_fallback_patch_binds_router_for_unsupported_kernel(
    monkeypatch, tmp_path
):
    """Execute the patched binder body against stand-ins for both kernel kinds."""

    old_snippet, _new_snippet = patch_snippets(_CAPTURER_PATCH_FN)
    source = tmp_path / "routed_experts_capturer.py"
    source.write_text(
        "def bind(module, quant_method, fused_experts, capture_fn, "
        "FusedMoEExpertsMonolithic, BaseRouter):\n"
        "    num_bound = 0\n"
        "    if True:\n" + old_snippet + "        return num_bound\n"
    )
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _relative: str(source))
    assert patches._patch_vllm_routed_experts_capture_router_fallback(
        logging.getLogger(__name__), required=True
    )
    namespace: dict = {}
    exec(compile(source.read_text(), str(source), "exec"), namespace)

    class Monolithic:
        def __init__(self, supports):
            self.supports = supports
            self.bound = None

        def supports_routing_replay_capture(self):
            return self.supports

        def set_capture_fn(self, fn):
            self.bound = fn

    class Router:
        def __init__(self):
            self.bound = None

        def set_capture_fn(self, fn):
            self.bound = fn

    capture_fn = object()
    quant_method = SimpleNamespace(is_monolithic=True)

    kernel, router = Monolithic(True), Router()
    module = SimpleNamespace(router=router)
    assert (
        namespace["bind"](module, quant_method, kernel, capture_fn, Monolithic, Router)
        == 1
    )
    assert kernel.bound is capture_fn and router.bound is None

    kernel, router = Monolithic(False), Router()
    module = SimpleNamespace(router=router)
    assert (
        namespace["bind"](module, quant_method, kernel, capture_fn, Monolithic, Router)
        == 1
    )
    assert kernel.bound is None and router.bound is capture_fn

    kernel = Monolithic(False)
    module = SimpleNamespace(router=object())
    with pytest.raises(ValueError, match="not supported with monolithic"):
        namespace["bind"](module, quant_method, kernel, capture_fn, Monolithic, Router)


def _install_fake_vllm_modules(monkeypatch):
    vllm_module = types.ModuleType("vllm")
    envs_module = types.ModuleType("vllm.envs")
    envs_module.VLLM_USE_RAY_V2_EXECUTOR_BACKEND = True
    logger_module = types.ModuleType("vllm.logger")
    logger_module.init_logger = lambda name: logging.getLogger(name)
    vllm_module.envs = envs_module
    vllm_module.logger = logger_module
    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.envs", envs_module)
    monkeypatch.setitem(sys.modules, "vllm.logger", logger_module)


def _stub_non_fp32_vllm_patches(monkeypatch, captured_extra_env_vars):
    monkeypatch.setattr(
        patches,
        "_patch_vllm_init_workers_ray",
        lambda _py, extra: captured_extra_env_vars.append(extra) or False,
    )
    for patch_name in (
        "_patch_vllm_llama_eagle3_own_lm_head",
        "_patch_vllm_tool_parser_namespace_tool",
        "_patch_vllm_ray_executor_v2_tcpstore_port",
        "_patch_vllm_shm_broadcast_bind_retry",
        "_patch_vllm_radio_layerscale_loader",
        "_patch_vllm_radio_native_loader",
        "_patch_vllm_radio_final_layernorm",
        "_patch_vllm_glm_decoder_sequence_parallel_moe",
    ):
        monkeypatch.setattr(patches, patch_name, lambda _logger: None)
    monkeypatch.setattr(
        patches,
        "_patch_vllm_moe_routed_experts_capture",
        lambda _logger, *, required=False: True,
    )
    monkeypatch.setattr(
        patches,
        "_patch_vllm_routed_experts_capture_router_fallback",
        lambda _logger, *, required=False: True,
    )


@pytest.mark.parametrize("require_capture", [False, True])
def test_apply_vllm_patches_forwards_capture_requirement(monkeypatch, require_capture):
    _install_fake_vllm_modules(monkeypatch)
    monkeypatch.setenv(VLLM_NEMOTRON_H_FP32_LM_HEAD_ENV_VAR, "1")
    captured_extra_env_vars = []
    _stub_non_fp32_vllm_patches(monkeypatch, captured_extra_env_vars)
    capture_requirements = []
    fallback_requirements = []
    monkeypatch.setattr(
        patches,
        "_patch_vllm_moe_routed_experts_capture",
        lambda _logger, *, required: capture_requirements.append(required) or True,
    )
    monkeypatch.setattr(
        patches,
        "_patch_vllm_routed_experts_capture_router_fallback",
        lambda _logger, *, required: fallback_requirements.append(required) or True,
    )
    patches._apply_vllm_patches(
        "py",
        extra_env_vars=["USER_VAR"],
        require_moe_routed_experts_capture=require_capture,
    )
    assert capture_requirements == fallback_requirements == [require_capture]
    assert captured_extra_env_vars == [["USER_VAR"]]
    assert VLLM_NEMOTRON_H_FP32_LM_HEAD_ENV_VAR not in os.environ


@pytest.mark.parametrize(
    "existing,extra,expected",
    [
        (None, None, "RAY_ENABLE_UV_RUN_RUNTIME_ENV"),
        ("", ["MY_VAR"], "MY_VAR,RAY_ENABLE_UV_RUN_RUNTIME_ENV"),
        # A value the caller already set must survive, not be clobbered.
        ("PRESET", ["MY_VAR"], "MY_VAR,PRESET,RAY_ENABLE_UV_RUN_RUNTIME_ENV"),
        # Duplicates collapse and surrounding whitespace is stripped.
        (
            " PRESET , MY_VAR ",
            ["MY_VAR"],
            "MY_VAR,PRESET,RAY_ENABLE_UV_RUN_RUNTIME_ENV",
        ),
    ],
)
def test_ray_extra_env_vars_merge_is_additive(
    monkeypatch, tmp_path, existing, extra, expected
):
    """vLLM 0.25 replaced the ADDITIONAL_ENV_VARS source patch with this hook.

    It must add to whatever the caller already set rather than overwrite it --
    otherwise user ``extra_env_vars`` silently stop reaching the Ray workers.
    """
    ray_executor = tmp_path / "ray_executor.py"
    ray_executor.write_text("self._init_workers_ray(placement_group)\n")
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _r: str(ray_executor))

    if existing is None:
        monkeypatch.delenv("VLLM_RAY_EXTRA_ENV_VARS_TO_COPY", raising=False)
    else:
        monkeypatch.setenv("VLLM_RAY_EXTRA_ENV_VARS_TO_COPY", existing)

    patches._patch_vllm_init_workers_ray("py", extra)

    assert os.environ["VLLM_RAY_EXTRA_ENV_VARS_TO_COPY"] == expected


def test_init_workers_ray_reports_a_missing_anchor(monkeypatch, tmp_path):
    """A reshaped call site must not be reported as a successful patch."""
    ray_executor = tmp_path / "ray_executor.py"
    ray_executor.write_text("self._init_workers_ray_renamed(placement_group)\n")
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _r: str(ray_executor))
    monkeypatch.delenv("VLLM_RAY_EXTRA_ENV_VARS_TO_COPY", raising=False)

    assert patches._patch_vllm_init_workers_ray("py", None) is False
    # The env merge still has to happen; it is independent of the file patch.
    assert os.environ["VLLM_RAY_EXTRA_ENV_VARS_TO_COPY"] == (
        "RAY_ENABLE_UV_RUN_RUNTIME_ENV"
    )


def test_init_workers_ray_reports_success_and_is_idempotent(monkeypatch, tmp_path):
    """Patching twice against the same file still reports success."""
    ray_executor = tmp_path / "ray_executor.py"
    ray_executor.write_text("self._init_workers_ray(placement_group)\n")
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _r: str(ray_executor))

    assert patches._patch_vllm_init_workers_ray("py-exec", None) is True
    once = ray_executor.read_text()
    assert 'runtime_env={"py_executable": "py-exec"}' in once

    assert patches._patch_vllm_init_workers_ray("py-exec", None) is True
    assert ray_executor.read_text() == once

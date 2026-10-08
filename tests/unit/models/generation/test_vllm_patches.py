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
* DSA top-k replay repurposes routed-experts capture in vLLM 0.29. Its two
  fail-closed source edits are pinned because a missed layer or stale shared
  top-k buffer would produce a valid-looking but incorrect replay payload.
"""

import ast
import logging
import os
import sys
import textwrap
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from nemo_rl.models.generation.vllm import patches
from nemo_rl.models.generation.vllm.config import (
    vllm_nemotron_h_fp32_lm_head_enabled,
)
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
_RADIO_MARKER = "initializer_factor = self.config.initializer_factor"
_GLM_DSA_SOURCE = "model_executor/models/deepseek_v2.py"
_GLM_DSA_PATCH_FN = "_patch_vllm_glm_decoder_sequence_parallel_moe"
_GLM_DSA_MARKER = 'getattr(config, "model_type", None) != "glm_moe_dsa"'
_NEMOTRON_H_SOURCE = """import torch
from torch import nn


def maybe_prefix(prefix, name):
    return f"{prefix}.{name}"


class LogitsProcessor:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

    def __call__(self, lm_head, hidden_states):
        return lm_head.quant_method.apply(lm_head, hidden_states)


class QuantMethod:
    def __init__(self):
        self.seen_dtypes = []

    def apply(self, lm_head, hidden_states, bias=None):
        self.seen_dtypes.append(
            (
                hidden_states.dtype,
                lm_head.weight.dtype,
                None if bias is None else bias.dtype,
            )
        )
        logits = hidden_states @ lm_head.weight.t()
        if bias is not None:
            logits = logits + bias
        return logits


class ParallelLMHead(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size,
        params_dtype=None,
        quant_config=None,
        prefix="",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.params_dtype = params_dtype
        self.quant_config = quant_config
        self.prefix = prefix
        self.weight = nn.Parameter(
            torch.ones(vocab_size, hidden_size, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.bias = None
        self.quant_method = QuantMethod()

    def forward(self, input_):
        del input_
        raise RuntimeError("LMHead's weights should be used in the sampler.")


class NemotronHForCausalLM:
    def __init__(self, config, prefix):
        self.quant_config = object()
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def compute_logits(self, hidden_states):
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits
"""
_MOE_SOURCE = "model_executor/layers/fused_moe/runner/moe_runner.py"
_MOE_PATCH_FN = "_patch_vllm_moe_routed_experts_capture"
_MOE_MARKER = "NeMo-RL patch (routed-experts capture for router replay)"
_CAPTURER_SOURCE = "model_executor/layers/fused_moe/routed_experts_capturer.py"
_CAPTURER_PATCH_FN = "_patch_vllm_routed_experts_capture_router_fallback"
_CAPTURER_MARKER = (
    "NeMo-RL patch (router fallback for monolithic routed-experts capture)"
)
_DSA_CAPTURER_PATCH_FN = "_patch_vllm_dsa_topk_capturer"
_DSA_CAPTURER_MARKER = "NeMo-RL patch (DSA top-k capture transport)"
_DSA_SCHEDULER_SOURCE = "v1/core/sched/scheduler.py"
_DSA_SCHEDULER_PATCH_FN = "_patch_vllm_dsa_topk_scheduler"
_DSA_SCHEDULER_MARKER = "NeMo-RL patch (retain DSA decode routes across steps)"
_DSA_ATTN_SOURCE = "models/deepseek_v32/attention.py"
_DSA_ATTN_PATCH_FN = "_patch_vllm_dsa_topk_attention"
_DSA_ATTN_MARKER = "NeMo-RL patch (capture DSA top-k before shared-buffer reuse)"
_DSA_CAPTURER_SNIPPETS = (
    ("import_old_snippet", "import_new_snippet"),
    ("shape_old_snippet", "shape_new_snippet"),
    ("init_old_snippet", "init_new_snippet"),
    ("device_buffer_old_snippet", "device_buffer_new_snippet"),
    ("binder_old_snippet", "binder_new_snippet"),
    ("manager_old_snippet", "manager_new_snippet"),
)
_DSA_ATTN_SNIPPETS = (
    ("import_old_snippet", "import_new_snippet"),
    ("capture_old_snippet", "capture_new_snippet"),
)


def _write_unpatched_multi_edit_copy(
    relative_source: str,
    patch_fn_name: str,
    snippet_names: tuple[tuple[str, str], ...],
    destination: Path,
) -> Path:
    """Copy installed source and reverse every edit from a multi-anchor patch."""
    content = Path(patches._get_vllm_file(relative_source)).read_text()
    for old_name, new_name in snippet_names:
        old_snippet, new_snippet = patch_snippets(
            patch_fn_name, old_name=old_name, new_name=new_name
        )
        if new_snippet in content:
            content = content.replace(new_snippet, old_snippet, 1)
        assert new_snippet not in content
        assert old_snippet in content, (
            f"{relative_source} contains neither form of {old_name}/{new_name}; "
            "the installed vLLM source has changed"
        )
    destination.write_text(content)
    return destination


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
def patched_nemotron_h_source(tmp_path, monkeypatch):
    source = tmp_path / "nemotron_h.py"
    source.write_text(_NEMOTRON_H_SOURCE)
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _relative: str(source))
    patches._patch_vllm_nemotron_h_fp32_lm_head(logging.getLogger(__name__))
    return source


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
    """Pin the vLLM 0.25.1 RADIO loader shape used by the source patch."""
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
    assert 'name.endswith((".ls1", ".ls2"))' in content
    assert "param.data.fill_(initializer_factor)" in content
    assert "loaded_params.add(name)" in content


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
    assert "vLLM 0.25.1 source shape was not found" in caplog.text


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


@pytest.fixture
def patched_dsa_capturer_source(tmp_path, monkeypatch):
    copied = _write_unpatched_multi_edit_copy(
        _CAPTURER_SOURCE,
        _DSA_CAPTURER_PATCH_FN,
        _DSA_CAPTURER_SNIPPETS,
        tmp_path / "routed_experts_capturer.py",
    )
    with monkeypatch.context() as patch_context:
        patch_context.setattr(patches, "_get_vllm_file", lambda _relative: str(copied))
        assert patches._patch_vllm_dsa_topk_capturer(
            logging.getLogger(__name__), required=True
        )
    return copied


@pytest.fixture
def patched_dsa_attention_source(tmp_path, monkeypatch):
    copied = _write_unpatched_multi_edit_copy(
        _DSA_ATTN_SOURCE,
        _DSA_ATTN_PATCH_FN,
        _DSA_ATTN_SNIPPETS,
        tmp_path / "attention.py",
    )
    with monkeypatch.context() as patch_context:
        patch_context.setattr(patches, "_get_vllm_file", lambda _relative: str(copied))
        assert patches._patch_vllm_dsa_topk_attention(
            logging.getLogger(__name__), required=True
        )
    return copied


@pytest.fixture
def patched_dsa_scheduler_source(tmp_path, monkeypatch):
    copied = write_unpatched_copy(
        _DSA_SCHEDULER_SOURCE,
        _DSA_SCHEDULER_PATCH_FN,
        tmp_path / "scheduler.py",
    )
    with monkeypatch.context() as patch_context:
        patch_context.setattr(patches, "_get_vllm_file", lambda _relative: str(copied))
        assert patches._patch_vllm_dsa_topk_scheduler(
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


@pytest.mark.vllm
def test_dsa_topk_source_patch_anchors_match_installed_vllm(
    patched_dsa_capturer_source,
    patched_dsa_scheduler_source,
    patched_dsa_attention_source,
):
    capturer_source = patched_dsa_capturer_source.read_text()
    scheduler_source = patched_dsa_scheduler_source.read_text()
    attention_source = patched_dsa_attention_source.read_text()

    assert _DSA_CAPTURER_MARKER in capturer_source
    assert "class _NRLDSASparseSlotBuffer" in capturer_source
    assert "expert_id_dtype = _nrl_dsa_topk_dtype(num_experts)" in capturer_source
    assert "dtype=nrl_dsa_topk_dtype or torch.int32" in capturer_source
    assert "self._nrl_copy_step_outputs = _nrl_dsa_topk_capture_enabled()" in (
        capturer_source
    )
    assert "compact_slot_by_layer" in capturer_source
    assert _DSA_SCHEDULER_MARKER in scheduler_source
    assert '"_nrl_copy_step_outputs", False' in scheduler_source
    assert _DSA_ATTN_MARKER in attention_source
    assert "capture_fn(scored_topk)" in attention_source
    assert "effective_topk[num_decode_tokens:] = dense_topk" in attention_source
    assert "dense_topk.masked_fill_" in attention_source
    assert (
        "from vllm.config import CacheConfig, CUDAGraphMode, VllmConfig"
        in attention_source
    )
    assert attention_source.index(_DSA_ATTN_MARKER) < attention_source.index(
        "if scoring_was_skipped:"
    )
    ast.parse(capturer_source)
    ast.parse(scheduler_source)
    ast.parse(attention_source)


@pytest.mark.vllm
def test_dsa_topk_source_patches_are_idempotent(
    patched_dsa_capturer_source,
    patched_dsa_scheduler_source,
    patched_dsa_attention_source,
    monkeypatch,
):
    capturer_before = patched_dsa_capturer_source.read_text()
    scheduler_before = patched_dsa_scheduler_source.read_text()
    attention_before = patched_dsa_attention_source.read_text()

    def get_source(relative_path):
        if relative_path == _CAPTURER_SOURCE:
            return str(patched_dsa_capturer_source)
        if relative_path == _DSA_SCHEDULER_SOURCE:
            return str(patched_dsa_scheduler_source)
        if relative_path == _DSA_ATTN_SOURCE:
            return str(patched_dsa_attention_source)
        raise AssertionError(relative_path)

    monkeypatch.setattr(patches, "_get_vllm_file", get_source)
    assert patches._patch_vllm_dsa_topk_capturer(
        logging.getLogger(__name__), required=True
    )
    assert patches._patch_vllm_dsa_topk_scheduler(
        logging.getLogger(__name__), required=True
    )
    assert patches._patch_vllm_dsa_topk_attention(
        logging.getLogger(__name__), required=True
    )
    assert patched_dsa_capturer_source.read_text() == capturer_before
    assert patched_dsa_scheduler_source.read_text() == scheduler_before
    assert patched_dsa_attention_source.read_text() == attention_before


@pytest.mark.parametrize(
    ("patch_fn", "filename"),
    [
        (patches._patch_vllm_dsa_topk_capturer, "routed_experts_capturer.py"),
        (patches._patch_vllm_dsa_topk_scheduler, "scheduler.py"),
        (patches._patch_vllm_dsa_topk_attention, "attention.py"),
    ],
)
def test_dsa_topk_source_patches_fail_closed_when_required(
    patch_fn, filename, monkeypatch, tmp_path
):
    source = tmp_path / filename
    source.write_text("# unexpected vLLM source\n")
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _relative: str(source))

    with pytest.raises(RuntimeError, match="Could not apply vLLM DSA top-k"):
        patch_fn(logging.getLogger(__name__), required=True)
    assert source.read_text() == "# unexpected vLLM source\n"


def test_dsa_topk_attention_patch_fails_closed_when_import_anchor_changes(
    monkeypatch, tmp_path
):
    capture_old_snippet, _capture_new_snippet = patch_snippets(
        _DSA_ATTN_PATCH_FN,
        old_name="capture_old_snippet",
        new_name="capture_new_snippet",
    )
    source = tmp_path / "attention.py"
    original = (
        "from vllm.config import CacheConfig as RenamedCacheConfig, VllmConfig\n"
        + capture_old_snippet
    )
    source.write_text(original)
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _relative: str(source))

    with pytest.raises(RuntimeError, match="CUDAGraphMode import"):
        patches._patch_vllm_dsa_topk_attention(
            logging.getLogger(__name__), required=True
        )
    assert source.read_text() == original


def test_dsa_topk_capturer_compacts_compute_layers_and_validates_selection(
    monkeypatch,
):
    _old_snippet, shape_source = patch_snippets(
        _DSA_CAPTURER_PATCH_FN,
        old_name="shape_old_snippet",
        new_name="shape_new_snippet",
    )
    tree = ast.parse(shape_source)
    wanted_names = {
        "_NRL_DSA_TOPK_CAPTURE_ENV_VAR",
        "_NRL_DSA_TOPK_LAYER_IDS_ENV_VAR",
    }
    body = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id in wanted_names
                for target in node.targets
            )
        )
        or (
            isinstance(node, ast.FunctionDef)
            and node.name
            in {
                "_nrl_dsa_topk_capture_enabled",
                "_nrl_dsa_topk_layer_ids",
                "_get_routed_experts_shape",
            }
        )
    ]
    namespace = {"os": os, "VllmConfig": object}
    exec(
        compile(ast.Module(body=body, type_ignores=[]), "<dsa-capturer>", "exec"),
        namespace,
    )

    hf_config = SimpleNamespace(
        num_hidden_layers=8,
        index_topk_freq=4,
        index_skip_topk_offset=3,
        index_topk_pattern=None,
        index_topk=128,
    )
    model_config = SimpleNamespace(
        hf_text_config=hf_config,
        max_model_len=6144,
        get_total_num_hidden_layers=lambda: 8,
        get_num_experts=lambda: 256,
        get_num_experts_per_tok=lambda: 8,
    )
    vllm_config = SimpleNamespace(model_config=model_config)
    monkeypatch.setenv(patches.VLLM_DSA_TOPK_CAPTURE_ENV_VAR, "1")
    monkeypatch.delenv(patches.VLLM_DSA_TOPK_LAYER_IDS_ENV_VAR, raising=False)

    assert namespace["_nrl_dsa_topk_layer_ids"](vllm_config) == [0, 1, 2, 6]
    assert namespace["_get_routed_experts_shape"](vllm_config) == (4, 6144, 128)

    monkeypatch.setenv(patches.VLLM_DSA_TOPK_LAYER_IDS_ENV_VAR, "2,6")
    assert namespace["_nrl_dsa_topk_layer_ids"](vllm_config) == [2, 6]
    assert namespace["_get_routed_experts_shape"](vllm_config) == (2, 6144, 128)

    monkeypatch.setenv(patches.VLLM_DSA_TOPK_LAYER_IDS_ENV_VAR, "3")
    with pytest.raises(ValueError, match="only select non-MTP compute layers"):
        namespace["_nrl_dsa_topk_layer_ids"](vllm_config)


def _dsa_topk_transport_namespace() -> dict:
    _old_snippet, source = patch_snippets(
        _DSA_CAPTURER_PATCH_FN,
        old_name="shape_old_snippet",
        new_name="shape_new_snippet",
    )
    namespace = {"np": np, "os": os, "torch": torch, "VllmConfig": object}
    exec(compile(source, "<dsa-topk-transport>", "exec"), namespace)
    return namespace


def test_dsa_topk_sparse_slot_buffer_is_lazy_and_round_trips_by_block():
    namespace = _dsa_topk_transport_namespace()
    buffer_cls = namespace["_NRLDSASparseSlotBuffer"]
    buffer = buffer_cls(
        max_num_slots=12,
        block_size=4,
        num_layers=2,
        top_k=3,
        dtype=np.int16,
    )

    assert buffer.shape == (12, 2, 3)
    assert buffer.dtype.name == "int16"
    assert buffer.nbytes == 0
    np.testing.assert_array_equal(
        buffer[np.array([0, 5, 11], dtype=np.int64)],
        np.full((3, 2, 3), -1, dtype=np.int16),
    )
    assert buffer.nbytes == 0  # Missing reads do not allocate blocks.

    slots = np.array([1, 5, 1, 6], dtype=np.int64)
    values = np.arange(4 * 2 * 3, dtype=np.int32).reshape(4, 2, 3)
    buffer[slots] = values

    # Only physical blocks 0 and 1 were touched. Repeated slot 1 keeps the
    # final input row, independent of numpy's repeated fancy-index behavior.
    assert buffer.nbytes == 2 * 4 * 2 * 3 * np.dtype(np.int16).itemsize
    np.testing.assert_array_equal(
        buffer[np.array([1, 5, 6, 3], dtype=np.int64)],
        np.stack(
            [
                values[2],
                values[1],
                values[3],
                np.full((2, 3), -1, dtype=np.int32),
            ]
        ).astype(np.int16),
    )

    replacement = np.full((1, 2, 3), 77, dtype=np.int16)
    buffer[np.array([5], dtype=np.int64)] = replacement
    np.testing.assert_array_equal(buffer[np.array([5])], replacement)


def test_dsa_topk_sparse_slot_buffer_fails_fast_on_bad_indices_and_shape():
    buffer_cls = _dsa_topk_transport_namespace()["_NRLDSASparseSlotBuffer"]
    buffer = buffer_cls(
        max_num_slots=8,
        block_size=4,
        num_layers=2,
        top_k=3,
        dtype=np.int16,
    )
    values = np.zeros((1, 2, 3), dtype=np.int16)

    for bad_slot in (-1, 8):
        slots = np.array([bad_slot], dtype=np.int64)
        with pytest.raises(IndexError, match="out of range"):
            buffer[slots]
        with pytest.raises(IndexError, match="out of range"):
            buffer[slots] = values
    with pytest.raises(ValueError, match="shape mismatch"):
        buffer[np.array([0, 1], dtype=np.int64)] = values
    with pytest.raises(TypeError, match="must contain integers"):
        buffer[np.array([0.0])]
    for bad_value in (-2, np.iinfo(np.int16).max + 1):
        out_of_range = np.full((1, 2, 3), bad_value, dtype=np.int32)
        with pytest.raises(ValueError, match="values are out of range"):
            buffer[np.array([0], dtype=np.int64)] = out_of_range
    assert buffer.nbytes == 0


def test_dsa_topk_transport_uses_narrow_signed_dtype_when_safe():
    namespace = _dsa_topk_transport_namespace()
    numpy_dtype = namespace["_nrl_dsa_topk_dtype"]
    torch_dtype = namespace["_nrl_dsa_topk_torch_dtype"]

    assert numpy_dtype(32768) is np.int16
    assert torch_dtype(32768) is torch.int16
    assert numpy_dtype(32769) is np.int32
    assert torch_dtype(32769) is torch.int32
    with pytest.raises(ValueError, match="positive max_model_len"):
        numpy_dtype(0)


@pytest.mark.parametrize("force_copy", [False, True])
def test_dsa_topk_scheduler_copy_flag_controls_step_buffer_aliasing(force_copy):
    _old_snippet, new_snippet = patch_snippets(_DSA_SCHEDULER_PATCH_FN)
    source = (
        "def convert(self, re):\n"
        + textwrap.indent(textwrap.dedent(new_snippet), "    ")
        + "    return routing_data\n"
    )
    namespace: dict = {}
    exec(compile(source, "<dsa-scheduler-copy>", "exec"), namespace)

    transit = np.array([[[1, 0]]], dtype=np.int16)
    manager = SimpleNamespace(
        routed_experts_by_slot=SimpleNamespace(dtype=np.dtype(np.int16)),
        _nrl_copy_step_outputs=force_copy,
    )
    converted = namespace["convert"](
        SimpleNamespace(routed_experts_mgr=manager),
        SimpleNamespace(routing_data=transit),
    )
    transit[...] = 7

    expected = 1 if force_copy else 7
    assert int(converted[0, 0, 0]) == expected


def test_dsa_topk_attention_capture_matches_effective_attention_routes(monkeypatch):
    _old_snippet, new_snippet = patch_snippets(
        _DSA_ATTN_PATCH_FN,
        old_name="capture_old_snippet",
        new_name="capture_new_snippet",
    )
    source = (
        "def capture_block(self, positions, attn_metadata, output, torch):\n"
        + textwrap.indent(textwrap.dedent(new_snippet), "    ")
        + "        pass\n"
    )

    class CUDAGraphMode:
        FULL = object()

    forward_context = SimpleNamespace(cudagraph_runtime_mode=object())
    namespace = {
        "CUDAGraphMode": CUDAGraphMode,
        "get_forward_context": lambda: forward_context,
    }
    exec(compile(source, "<dsa-attention-capture>", "exec"), namespace)
    capture_block = namespace["capture_block"]
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    captured = []
    attention = SimpleNamespace(
        indexer=SimpleNamespace(topk_tokens=4),
        topk_indices_buffer=torch.tensor(
            [[3, 2, 1, 0], [2, 1, 0, -1]], dtype=torch.int32
        ),
        _nrl_dsa_topk_capture_fn=lambda value: captured.append(value.clone()),
        _use_sparse_mha=lambda _metadata: False,
        _dense_mha_metadata_layer_name="model.layers.0.self_attn.attn",
    )
    metadata = SimpleNamespace(
        num_actual_tokens=2,
        num_decode_tokens=0,
        prefill=SimpleNamespace(use_dense_mha=False),
    )
    capture_block(attention, torch.tensor([0, 2]), metadata, torch.empty(2), torch)
    torch.testing.assert_close(captured.pop(), attention.topk_indices_buffer)

    # Masked MHA consumes the scorer top-k as a mask, so it must replay the
    # scorer result even though _use_sparse_mha selects forward_impl.
    attention._use_sparse_mha = lambda _metadata: True
    metadata.prefill.use_dense_mha = False
    capture_block(attention, torch.tensor([0, 2]), metadata, torch.empty(2), torch)
    torch.testing.assert_close(captured.pop(), attention.topk_indices_buffer)

    dense_expected = torch.tensor([[0, -1, -1, -1], [0, 1, 2, -1]], dtype=torch.int32)
    metadata.prefill.use_dense_mha = True
    capture_block(attention, torch.tensor([0, 2]), metadata, torch.empty(2), torch)
    torch.testing.assert_close(captured.pop(), dense_expected)

    # Mixed batches use scorer top-k for the decode prefix and causal dense keys
    # for the prefill suffix. A decode position can exceed K without error.
    metadata.num_decode_tokens = 1
    capture_block(attention, torch.tensor([9, 2]), metadata, torch.empty(2), torch)
    torch.testing.assert_close(
        captured.pop(),
        torch.tensor([[3, 2, 1, 0], [0, 1, 2, -1]], dtype=torch.int32),
    )

    # Whether scoring happened is separate from which key set dense attention
    # consumed. These variants still replay dense causal keys.
    metadata.num_decode_tokens = 0
    forward_context.cudagraph_runtime_mode = CUDAGraphMode.FULL
    capture_block(attention, torch.tensor([0, 2]), metadata, torch.empty(2), torch)
    torch.testing.assert_close(captured.pop(), dense_expected)

    forward_context.cudagraph_runtime_mode = object()
    attention._dense_mha_metadata_layer_name = ""
    capture_block(attention, torch.tensor([0, 2]), metadata, torch.empty(2), torch)
    torch.testing.assert_close(captured.pop(), dense_expected)

    attention._dense_mha_metadata_layer_name = "model.layers.0.self_attn.attn"
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    capture_block(attention, torch.tensor([0, 2]), metadata, torch.empty(2), torch)
    torch.testing.assert_close(captured.pop(), dense_expected)

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    attention._use_sparse_mha = lambda _metadata: False
    with pytest.raises(RuntimeError, match="scoring was skipped"):
        capture_block(attention, torch.tensor([0, 2]), metadata, torch.empty(2), torch)

    attention._use_sparse_mha = lambda _metadata: True
    metadata.num_decode_tokens = 3
    with pytest.raises(RuntimeError, match="Invalid DSA decode/prefill partition"):
        capture_block(attention, torch.tensor([0, 2]), metadata, torch.empty(2), torch)

    metadata.num_decode_tokens = 0
    with pytest.raises(RuntimeError, match="cannot represent all causal keys"):
        capture_block(attention, torch.tensor([0, 4]), metadata, torch.empty(2), torch)


@pytest.mark.parametrize(
    ("vllm_cfg", "expected"),
    [
        ({}, False),
        ({"fp32_lm_head": False}, False),
        ({"fp32_lm_head": True}, True),
        ({"env_vars": {VLLM_NEMOTRON_H_FP32_LM_HEAD_ENV_VAR: "1"}}, False),
        ({"env_vars": {VLLM_NEMOTRON_H_FP32_LM_HEAD_ENV_VAR: "0"}}, False),
    ],
)
def test_vllm_nemotron_h_fp32_lm_head_enabled(vllm_cfg, expected):
    assert vllm_nemotron_h_fp32_lm_head_enabled(vllm_cfg) is expected


@pytest.mark.parametrize("env_value", [None, "0", "1"])
def test_nemotron_h_fp32_lm_head_patch_is_env_gated(
    patched_nemotron_h_source, monkeypatch, env_value
):
    if env_value is None:
        monkeypatch.delenv(VLLM_NEMOTRON_H_FP32_LM_HEAD_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(VLLM_NEMOTRON_H_FP32_LM_HEAD_ENV_VAR, env_value)

    namespace = {}
    source = patched_nemotron_h_source.read_text()
    exec(compile(source, str(patched_nemotron_h_source), "exec"), namespace)
    config = types.SimpleNamespace(vocab_size=16, hidden_size=8)
    model = namespace["NemotronHForCausalLM"](config, "model")
    hidden_states = torch.ones(2, 8, dtype=torch.bfloat16)

    logits = model.compute_logits(hidden_states)

    if env_value == "1":
        assert model._nrl_fp32_lm_head is True
        assert model.lm_head.params_dtype is None
        assert model.lm_head.quant_config is model.quant_config
        assert model.lm_head.weight.dtype is torch.bfloat16
        assert logits.dtype is torch.float32
        assert model.lm_head(hidden_states).dtype is torch.float32
        assert model.lm_head.quant_method.seen_dtypes == []
    else:
        assert model._nrl_fp32_lm_head is False
        assert model.lm_head.params_dtype is None
        assert model.lm_head.quant_config is model.quant_config
        assert model.lm_head.weight.dtype is torch.bfloat16
        assert logits.dtype is torch.bfloat16
        assert model.lm_head.quant_method.seen_dtypes == [
            (torch.bfloat16, torch.bfloat16, None)
        ]

    assert "deepcopy" not in source
    assert "params_dtype=torch.float32" not in source
    assert "NemotronH vLLM lm_head.forward casts " in source
    assert "input and weight to fp32" in source
    assert "torch.matmul(" in source
    ast.parse(source)


def test_nemotron_h_fp32_lm_head_patch_is_idempotent(
    patched_nemotron_h_source, monkeypatch
):
    before = patched_nemotron_h_source.read_text()
    monkeypatch.setattr(
        patches, "_get_vllm_file", lambda _relative: str(patched_nemotron_h_source)
    )

    patches._patch_vllm_nemotron_h_fp32_lm_head(logging.getLogger(__name__))

    assert patched_nemotron_h_source.read_text() == before


def test_nemotron_h_fp32_lm_head_patch_warns_on_unknown_source(
    tmp_path, monkeypatch, caplog
):
    source = tmp_path / "nemotron_h.py"
    source.write_text("class NemotronHForCausalLM:\n    pass\n")
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _relative: str(source))

    with caplog.at_level(logging.WARNING):
        applied = patches._patch_vllm_nemotron_h_fp32_lm_head(
            logging.getLogger(__name__)
        )

    assert applied is False
    assert source.read_text() == "class NemotronHForCausalLM:\n    pass\n"
    assert "NemotronH fp32 LM head import anchor not found exactly once" in caplog.text


@pytest.mark.vllm
def test_nemotron_h_fp32_lm_head_patch_anchor_still_matches_installed_vllm(
    tmp_path, monkeypatch
):
    """Pin the vLLM 0.25.1 Nemotron-H source shape used by the patch."""
    copied = tmp_path / "nemotron_h.py"
    with open(patches._get_vllm_file("model_executor/models/nemotron_h.py")) as f:
        copied.write_text(f.read())
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _relative: str(copied))

    applied = patches._patch_vllm_nemotron_h_fp32_lm_head(logging.getLogger(__name__))

    assert applied is True
    content = copied.read_text()
    assert "self._nrl_fp32_lm_head = (" in content
    assert "def _nrl_fp32_lm_head_forward(" in content
    assert content.index("import os\n") < content.index("import torch\n")
    ast.parse(content)


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
    monkeypatch.setattr(
        patches,
        "_patch_vllm_dsa_topk_capturer",
        lambda _logger, *, required=False: True,
    )
    monkeypatch.setattr(
        patches,
        "_patch_vllm_dsa_topk_scheduler",
        lambda _logger, *, required=False: True,
    )
    monkeypatch.setattr(
        patches,
        "_patch_vllm_dsa_topk_attention",
        lambda _logger, *, required=False: True,
    )


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("require_capture", [False, True])
def test_apply_vllm_patches_gates_nemotron_h_fp32_lm_head(
    monkeypatch, enabled, require_capture: bool
):
    _install_fake_vllm_modules(monkeypatch)
    monkeypatch.delenv(patches.VLLM_NEMOTRON_H_FP32_LM_HEAD_ENV_VAR, raising=False)
    captured_extra_env_vars = []
    fp32_patch_calls = []
    capture_requirements = []
    _stub_non_fp32_vllm_patches(monkeypatch, captured_extra_env_vars)
    monkeypatch.setattr(
        patches,
        "_patch_vllm_nemotron_h_fp32_lm_head",
        lambda _logger: fp32_patch_calls.append(True) or True,
    )
    monkeypatch.setattr(
        patches,
        "_patch_vllm_moe_routed_experts_capture",
        lambda _logger, *, required: capture_requirements.append(required) or True,
    )
    fallback_requirements = []
    monkeypatch.setattr(
        patches,
        "_patch_vllm_routed_experts_capture_router_fallback",
        lambda _logger, *, required: fallback_requirements.append(required) or True,
    )

    patches._apply_vllm_patches(
        "py",
        extra_env_vars=["USER_VAR"],
        nemotron_h_fp32_lm_head=enabled,
        require_moe_routed_experts_capture=require_capture,
    )

    assert bool(fp32_patch_calls) is enabled
    assert capture_requirements == [require_capture]
    # The router fallback is required exactly when the capture patch is.
    assert fallback_requirements == [require_capture]
    if enabled:
        assert os.environ[patches.VLLM_NEMOTRON_H_FP32_LM_HEAD_ENV_VAR] == "1"
        assert captured_extra_env_vars == [
            ["USER_VAR", patches.VLLM_NEMOTRON_H_FP32_LM_HEAD_ENV_VAR]
        ]
    else:
        assert patches.VLLM_NEMOTRON_H_FP32_LM_HEAD_ENV_VAR not in os.environ
        assert captured_extra_env_vars == [["USER_VAR"]]


@pytest.mark.parametrize("enabled", [False, True])
def test_apply_vllm_patches_gates_and_propagates_dsa_topk_capture(monkeypatch, enabled):
    _install_fake_vllm_modules(monkeypatch)
    monkeypatch.setenv(patches.VLLM_DSA_TOPK_CAPTURE_ENV_VAR, "ambient")
    monkeypatch.setenv(patches.VLLM_DSA_TOPK_LAYER_IDS_ENV_VAR, "99")
    captured_extra_env_vars = []
    dsa_capturer_requirements = []
    dsa_scheduler_requirements = []
    dsa_attention_requirements = []
    moe_requirements = []
    fallback_requirements = []
    _stub_non_fp32_vllm_patches(monkeypatch, captured_extra_env_vars)
    monkeypatch.setattr(
        patches,
        "_patch_vllm_dsa_topk_capturer",
        lambda _logger, *, required: dsa_capturer_requirements.append(required) or True,
    )
    monkeypatch.setattr(
        patches,
        "_patch_vllm_dsa_topk_scheduler",
        lambda _logger, *, required: dsa_scheduler_requirements.append(required)
        or True,
    )
    monkeypatch.setattr(
        patches,
        "_patch_vllm_dsa_topk_attention",
        lambda _logger, *, required: dsa_attention_requirements.append(required)
        or True,
    )
    monkeypatch.setattr(
        patches,
        "_patch_vllm_moe_routed_experts_capture",
        lambda _logger, *, required: moe_requirements.append(required) or True,
    )
    monkeypatch.setattr(
        patches,
        "_patch_vllm_routed_experts_capture_router_fallback",
        lambda _logger, *, required: fallback_requirements.append(required) or True,
    )

    patches._apply_vllm_patches(
        "py",
        extra_env_vars=["USER_VAR"],
        require_dsa_topk_capture=enabled,
        dsa_topk_layer_ids=[2, 6] if enabled else None,
    )

    if enabled:
        assert os.environ[patches.VLLM_DSA_TOPK_CAPTURE_ENV_VAR] == "1"
        assert os.environ[patches.VLLM_DSA_TOPK_LAYER_IDS_ENV_VAR] == "2,6"
        assert captured_extra_env_vars == [
            [
                "USER_VAR",
                patches.VLLM_DSA_TOPK_CAPTURE_ENV_VAR,
                patches.VLLM_DSA_TOPK_LAYER_IDS_ENV_VAR,
            ]
        ]
        assert dsa_capturer_requirements == [True]
        assert dsa_scheduler_requirements == [True]
        assert dsa_attention_requirements == [True]
        assert moe_requirements == []
        assert fallback_requirements == []
    else:
        assert patches.VLLM_DSA_TOPK_CAPTURE_ENV_VAR not in os.environ
        assert patches.VLLM_DSA_TOPK_LAYER_IDS_ENV_VAR not in os.environ
        assert captured_extra_env_vars == [["USER_VAR"]]
        assert dsa_capturer_requirements == []
        assert dsa_scheduler_requirements == []
        assert dsa_attention_requirements == []
        assert moe_requirements == [False]
        assert fallback_requirements == [False]


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (
            {
                "require_moe_routed_experts_capture": True,
                "require_dsa_topk_capture": True,
            },
            "cannot both use",
        ),
        ({"dsa_topk_layer_ids": [0]}, "while DSA top-k capture is disabled"),
        (
            {"require_dsa_topk_capture": True, "dsa_topk_layer_ids": [-1]},
            "must be non-negative",
        ),
        (
            {"require_dsa_topk_capture": True, "dsa_topk_layer_ids": [1, 1]},
            "must be unique",
        ),
    ],
)
def test_apply_vllm_patches_rejects_invalid_dsa_capture_configuration(
    monkeypatch, kwargs, match
):
    _install_fake_vllm_modules(monkeypatch)
    with pytest.raises(ValueError, match=match):
        patches._apply_vllm_patches("py", **kwargs)


def test_apply_vllm_patches_ignores_ambient_fp32_lm_head_env_toggle(monkeypatch):
    _install_fake_vllm_modules(monkeypatch)
    monkeypatch.setenv(patches.VLLM_NEMOTRON_H_FP32_LM_HEAD_ENV_VAR, "1")
    captured_extra_env_vars = []
    fp32_patch_calls = []
    _stub_non_fp32_vllm_patches(monkeypatch, captured_extra_env_vars)
    monkeypatch.setattr(
        patches,
        "_patch_vllm_nemotron_h_fp32_lm_head",
        lambda _logger: fp32_patch_calls.append(True) or True,
    )

    patches._apply_vllm_patches("py")

    assert fp32_patch_calls == []
    assert patches.VLLM_NEMOTRON_H_FP32_LM_HEAD_ENV_VAR not in os.environ
    assert captured_extra_env_vars == [None]


def test_apply_vllm_patches_raises_when_nemotron_h_fp32_lm_head_patch_fails(
    monkeypatch,
):
    _install_fake_vllm_modules(monkeypatch)
    monkeypatch.delenv(patches.VLLM_NEMOTRON_H_FP32_LM_HEAD_ENV_VAR, raising=False)
    _stub_non_fp32_vllm_patches(monkeypatch, [])
    monkeypatch.setattr(
        patches, "_patch_vllm_nemotron_h_fp32_lm_head", lambda _logger: False
    )

    with pytest.raises(RuntimeError, match="could not be applied"):
        patches._apply_vllm_patches("py", nemotron_h_fp32_lm_head=True)


@pytest.mark.parametrize("require_capture", [False, True])
@pytest.mark.parametrize(
    ("vllm_cfg_overrides", "expected_nemotron_h_fp32_lm_head"),
    [
        ({"env_vars": {"USER_VAR": "value"}, "fp32_lm_head": True}, True),
        (
            {
                "env_vars": {
                    "USER_VAR": "value",
                    VLLM_NEMOTRON_H_FP32_LM_HEAD_ENV_VAR: "1",
                }
            },
            False,
        ),
    ],
)
def test_vllm_worker_threads_nemotron_h_fp32_lm_head_cfg_into_source_patches(
    monkeypatch,
    vllm_cfg_overrides,
    expected_nemotron_h_fp32_lm_head,
    require_capture: bool,
):
    from nemo_rl.models.generation.vllm import vllm_worker

    patch_calls = []
    monkeypatch.setattr(
        vllm_worker,
        "_apply_vllm_patches",
        lambda py,
        *,
        extra_env_vars,
        nemotron_h_fp32_lm_head,
        require_moe_routed_experts_capture,
        require_dsa_topk_capture=False,
        dsa_topk_layer_ids=None: patch_calls.append(
            {
                "py": py,
                "extra_env_vars": extra_env_vars,
                "nemotron_h_fp32_lm_head": nemotron_h_fp32_lm_head,
                "require_moe_routed_experts_capture": require_moe_routed_experts_capture,
                "require_dsa_topk_capture": require_dsa_topk_capture,
                "dsa_topk_layer_ids": dsa_topk_layer_ids,
            }
        ),
    )

    vllm_worker.BaseVllmGenerationWorker(
        {
            "model_name": "model",
            "vllm_kwargs": {"enable_return_routed_experts": require_capture},
            "vllm_cfg": {
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
                "expert_parallel_size": 1,
                "gpu_memory_utilization": 0.6,
                "precision": "bfloat16",
                **vllm_cfg_overrides,
            },
        },
        extra_env_vars=["EXPLICIT_VAR"],
    )

    assert patch_calls == [
        {
            "py": sys.executable,
            "extra_env_vars": ["EXPLICIT_VAR"],
            "nemotron_h_fp32_lm_head": expected_nemotron_h_fp32_lm_head,
            "require_moe_routed_experts_capture": require_capture,
            "require_dsa_topk_capture": False,
            "dsa_topk_layer_ids": None,
        }
    ]


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

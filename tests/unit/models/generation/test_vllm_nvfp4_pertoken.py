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
"""vLLM-side quantizer tests for the NVFP4 per-token W4A4 rollout.

``nvfp4_pertoken.py`` imports vLLM at module scope (it is the vLLM-side
quantizer, a sibling of ``quantization/fp8.py``), so every test goes through
the ``nvfp4_module`` fixture rather than a bare top-of-file import — the same
pattern ``test_vllm_fp8_quantization.py`` uses — so the file still collects
cleanly outside the vllm extra.
"""

import types

import pytest
import torch
from pydantic import ValidationError

pytestmark = pytest.mark.vllm


@pytest.fixture()
def nvfp4_module():
    pytest.importorskip("vllm")
    from nemo_rl.models.generation.vllm.quantization import nvfp4_pertoken as M

    yield M


def _quantizer_with_layer_quantized(M, quantized: bool, expected_roles=None):
    """Build a quantizer around one synthetic Qwen-style target inventory."""
    quantizer = M.NvFp4PerTokenQuantizer.__new__(M.NvFp4PerTokenQuantizer)
    quantizer._model = None
    owner = torch.nn.Module()
    specs = tuple(
        M.ExpertWeightSpec(
            checkpoint_suffix=f"experts.0.{projection}.weight",
            logical_key="experts.0.<projection>",
            role=role,
        )
        for role, projection in (
            ("w1", "gate_proj"),
            ("w2", "down_proj"),
            ("w3", "up_proj"),
        )
    )
    if expected_roles is None:
        expected_roles = {("experts.0.<projection>", spec.role) for spec in specs}
    target = M.RoutedExpertTarget(
        module=owner,
        module_name="model.layers.0.mlp.experts",
        specs=specs,
        specs_by_suffix={spec.checkpoint_suffix: spec for spec in specs},
        expected_roles=frozenset(expected_roles),
    )
    quantizer._targets = {id(owner): target} if quantized else {}
    quantizer._all_target_suffixes = (
        {spec.checkpoint_suffix for spec in specs} if quantized else set()
    )
    quantizer._pending = {}
    quantizer._seen_roles = {}
    quantizer._quantized_events = 0
    quantizer._resolve_module = lambda name: owner if ".experts." in name else None
    return quantizer


def _fake_scaled_fp4_quant(monkeypatch, M, *, scale_shape_fn=None):
    """Stub vLLM's kernel so tests don't need real CUDA/FlashInfer.

    Records every ``(weight, global_scale)`` call and returns a
    deterministic packed/scale pair shaped like the real kernel's linear
    (non-swizzled) layout, unless ``scale_shape_fn`` overrides the scale shape.
    """
    calls = []

    def fake(weight, global_scale, is_sf_swizzled_layout, backend):
        assert is_sf_swizzled_layout is False
        assert backend == "none"
        calls.append((weight, global_scale))
        m, n = weight.shape
        shape = scale_shape_fn(m, n) if scale_shape_fn else (m, n // 16)
        packed = torch.zeros((m, n // 2), dtype=torch.uint8)
        scale = torch.zeros(shape, dtype=torch.float8_e4m3fn)
        return packed, scale

    monkeypatch.setattr(M.ops, "scaled_fp4_quant", fake)
    return calls


# ------------------------------------------------------------- quantizer


def test_pair_shares_one_global_scale(nvfp4_module, monkeypatch):
    """C2: gate and up must be quantized under one shared per-expert scale."""
    M = nvfp4_module
    calls = _fake_scaled_fp4_quant(monkeypatch, M)
    quantizer = _quantizer_with_layer_quantized(M, True)

    gate = torch.randn(8, 32) * 3.0
    up = torch.randn(8, 32) * 5.0
    p = "model.layers.0.mlp.experts.0"
    out = dict(
        quantizer.process(
            [
                (f"{p}.gate_proj.weight", gate),
                (f"{p}.up_proj.weight", up),
            ]
        )
    )

    assert torch.equal(
        out[f"{p}.gate_proj.weight_scale_2"], out[f"{p}.up_proj.weight_scale_2"]
    )
    expected_amax = max(gate.abs().max(), up.abs().max())
    expected_scale_2 = (expected_amax / (6.0 * 448.0)).float()
    assert torch.allclose(out[f"{p}.gate_proj.weight_scale_2"], expected_scale_2)
    assert len(calls) == 2  # one scaled_fp4_quant call per projection


def test_pending_survives_batch_split(nvfp4_module, monkeypatch):
    """C1: a gate/up pair may arrive in two different IPC batches."""
    M = nvfp4_module
    _fake_scaled_fp4_quant(monkeypatch, M)
    quantizer = _quantizer_with_layer_quantized(M, True)
    p = "model.layers.0.mlp.experts.0"

    out1 = quantizer.process([(f"{p}.gate_proj.weight", torch.randn(8, 32))])
    assert out1 == []
    assert quantizer._pending

    out2 = dict(quantizer.process([(f"{p}.up_proj.weight", torch.randn(8, 32))]))
    assert f"{p}.gate_proj.weight" in out2
    assert f"{p}.up_proj.weight" in out2
    assert not quantizer._pending


def test_pending_clones_off_recycled_source_buffer(nvfp4_module, monkeypatch):
    """C1: the pending half must survive the sender recycling its IPC buffer."""
    M = nvfp4_module
    _fake_scaled_fp4_quant(monkeypatch, M)
    quantizer = _quantizer_with_layer_quantized(M, True)
    p = "model.layers.0.mlp.experts.0"

    gate = torch.ones(8, 32)
    quantizer.process([(f"{p}.gate_proj.weight", gate)])
    gate.fill_(999.0)  # simulate the sender overwriting the recycled buffer

    up = torch.ones(8, 32) * 2.0
    out = dict(quantizer.process([(f"{p}.up_proj.weight", up)]))

    expected_scale_2 = torch.tensor(2.0 / (6.0 * 448.0))
    assert torch.allclose(out[f"{p}.gate_proj.weight_scale_2"], expected_scale_2)


def test_finish_raises_on_unpaired_half(nvfp4_module, monkeypatch):
    M = nvfp4_module
    _fake_scaled_fp4_quant(monkeypatch, M)
    quantizer = _quantizer_with_layer_quantized(M, True)

    quantizer.process(
        [
            (
                "model.layers.0.mlp.experts.0.gate_proj.weight",
                torch.randn(8, 32),
            )
        ]
    )
    with pytest.raises(RuntimeError, match=r"model\.layers\.0\.mlp\.experts.*0"):
        quantizer.finish()


def test_duplicate_projection_is_fatal(nvfp4_module, monkeypatch):
    M = nvfp4_module
    _fake_scaled_fp4_quant(monkeypatch, M)
    quantizer = _quantizer_with_layer_quantized(M, True)
    name = "model.layers.0.mlp.experts.0.down_proj.weight"

    quantizer.process([(name, torch.randn(8, 32))])
    with pytest.raises(RuntimeError, match="duplicate w2 checkpoint weight"):
        quantizer.process([(name, torch.randn(8, 32))])


def test_finish_raises_when_nothing_quantized(nvfp4_module):
    """A config/name mismatch must not silently degrade to an all-BF16 refit."""
    M = nvfp4_module
    quantizer = _quantizer_with_layer_quantized(M, True)
    with pytest.raises(RuntimeError, match="did not cover every selected"):
        quantizer.finish()


def test_finish_allows_pipeline_stage_with_no_local_nvfp4_targets(nvfp4_module, capsys):
    M = nvfp4_module
    quantizer = _quantizer_with_layer_quantized(M, False)

    quantizer.finish()

    assert "across 0 RoutedExperts containers" in capsys.readouterr().out


def test_finish_prints_liveness_line_on_success(nvfp4_module, monkeypatch, capsys):
    M = nvfp4_module
    _fake_scaled_fp4_quant(monkeypatch, M)
    quantizer = _quantizer_with_layer_quantized(M, True)

    quantizer.process(
        [
            ("model.layers.0.mlp.experts.0.gate_proj.weight", torch.randn(8, 32)),
            ("model.layers.0.mlp.experts.0.up_proj.weight", torch.randn(8, 32)),
            ("model.layers.0.mlp.experts.0.down_proj.weight", torch.randn(8, 32)),
        ]
    )
    quantizer.finish()

    # One W1/W3 shared-scale pair plus one independent W2 tensor.
    assert "[nvfp4_pertoken] refit: quantized 2" in capsys.readouterr().out


def test_reset_clears_pending_after_failure_and_after_success(
    nvfp4_module, monkeypatch
):
    M = nvfp4_module
    _fake_scaled_fp4_quant(monkeypatch, M)
    quantizer = _quantizer_with_layer_quantized(M, True)
    p = "model.layers.0.mlp.experts.0"

    quantizer.process([(f"{p}.gate_proj.weight", torch.randn(8, 32))])
    assert quantizer._pending  # simulated failed refit, left a pending half
    quantizer.reset()
    assert not quantizer._pending

    quantizer.process(
        [
            (f"{p}.gate_proj.weight", torch.randn(8, 32)),
            (f"{p}.up_proj.weight", torch.randn(8, 32)),
        ]
    )
    assert not quantizer._pending  # pair already completed
    quantizer.reset()
    assert not quantizer._pending


def test_non_expert_names_pass_through(nvfp4_module):
    M = nvfp4_module
    quantizer = _quantizer_with_layer_quantized(M, True)
    name = "model.layers.0.self_attn.q_proj.weight"
    tensor = torch.randn(4, 4)
    out = quantizer.process([(name, tensor)])
    assert len(out) == 1
    assert out[0][0] == name
    assert torch.equal(out[0][1], tensor)


def test_ignored_expert_layer_passes_through_bf16(nvfp4_module):
    """additional_ignore layers keep their unquantized BF16 FusedMoE method."""
    M = nvfp4_module
    quantizer = _quantizer_with_layer_quantized(M, False)
    name = "model.layers.0.mlp.experts.0.gate_proj.weight"
    tensor = torch.randn(8, 32)
    out = quantizer.process([(name, tensor)])
    assert len(out) == 1
    assert out[0][0] == name
    assert torch.equal(out[0][1], tensor)


def test_boundary_that_matched_no_module_names_is_fatal(nvfp4_module):
    """A boundary glob that matches nothing must fail, not silently quantize.

    The ignore patterns are module-name globs shaped like ``mlp.experts``.
    A MoE family that names its container differently (Mixtral's
    ``block_sparse_moe.experts``, DBRX's ``ffn.experts``) matches nothing, so
    every boundary layer would be quantized in the rollout while the trainer
    keeps it BF16 -- a silent train/rollout precision split.
    """
    M = nvfp4_module
    with pytest.raises(RuntimeError, match="BF16 boundary did not match"):
        M.NvFp4PerTokenQuantizer._verify_boundary(
            ["*.layers.0.mlp.experts*", "*.layers.3.mlp.experts*"],
            {0: "model.layers.0.block_sparse_moe.experts"},
            {},
        )


def test_boundary_verification_is_pipeline_stage_local(nvfp4_module):
    """A stage that owns none of the boundary layers must still pass."""
    M = nvfp4_module
    M.NvFp4PerTokenQuantizer._verify_boundary(
        ["*.layers.0.mlp.experts*", "*.layers.47.mlp.experts*"],
        {20: "model.layers.20.mlp.experts", 21: "model.layers.21.mlp.experts"},
        {},
    )
    # And a stage that owns them excludes exactly those.
    M.NvFp4PerTokenQuantizer._verify_boundary(
        ["*.layers.0.mlp.experts*", "*.layers.47.mlp.experts*"],
        {1: "model.layers.1.mlp.experts"},
        {0: "model.layers.0.mlp.experts"},
    )


def test_boundary_excluding_a_layer_it_should_not_is_fatal(nvfp4_module):
    M = nvfp4_module
    with pytest.raises(RuntimeError, match="expected NVFP4"):
        M.NvFp4PerTokenQuantizer._verify_boundary(
            ["*.layers.0.mlp.experts*"],
            {},
            {0: "model.layers.0.mlp.experts", 7: "model.layers.7.mlp.experts"},
        )


def test_passthrough_clones_off_recycled_source_buffer(nvfp4_module):
    """Cross-batch staleness guard: passthrough tensors must not be IPC-buffer views.

    vLLM's layerwise reload buffers every weight_loader call's arguments
    (including the tensor) and replays them at layer completion, which can
    land after the sender has recycled this batch's IPC buffer for a later
    one. A passthrough tensor handed through un-cloned would silently
    corrupt whatever layer it belongs to once replayed.
    """
    M = nvfp4_module
    quantizer = _quantizer_with_layer_quantized(M, True)
    name = "model.layers.0.self_attn.q_proj.weight"
    tensor = torch.ones(4, 4)
    out = quantizer.process([(name, tensor)])
    assert out[0][1].data_ptr() != tensor.data_ptr()
    tensor.fill_(999.0)  # simulate the sender overwriting the recycled buffer
    assert torch.equal(out[0][1], torch.ones(4, 4))
    assert not quantizer._pending


def test_emits_weight_and_scale_names_per_expert(nvfp4_module, monkeypatch):
    M = nvfp4_module
    _fake_scaled_fp4_quant(monkeypatch, M)
    quantizer = _quantizer_with_layer_quantized(M, True)
    p = "model.layers.0.mlp.experts.0"

    out = dict(
        quantizer.process(
            [
                (f"{p}.gate_proj.weight", torch.randn(8, 32)),
                (f"{p}.up_proj.weight", torch.randn(8, 32)),
                (f"{p}.down_proj.weight", torch.randn(32, 16)),
            ]
        )
    )

    for proj in ("gate_proj", "up_proj", "down_proj"):
        for suffix in ("weight", "weight_scale", "weight_scale_2", "input_scale"):
            assert f"{p}.{proj}.{suffix}" in out
        assert out[f"{p}.{proj}.input_scale"].item() == 1.0
    assert len(out) == 12


def test_rejects_swizzled_scale_shape(nvfp4_module, monkeypatch):
    """The default swizzled layout must be caught, not silently accepted.

    ``scaled_fp4_quant`` defaults to swizzled scales; the weight loader wants
    linear ``(m, n // 16)`` ones. For Qwen3-30B-A3B the two buffers happen to
    have identical shapes, so only a shape unlikely to collide (m not a
    multiple of 128) actually exercises this guard.
    """
    M = nvfp4_module
    _fake_scaled_fp4_quant(
        monkeypatch,
        M,
        scale_shape_fn=lambda m, n: (((m + 127) // 128) * 128, n // 16),
    )
    quantizer = _quantizer_with_layer_quantized(M, True)
    with pytest.raises(AssertionError, match="linear"):
        quantizer.process(
            [
                (
                    "model.layers.0.mlp.experts.0.down_proj.weight",
                    torch.randn(100, 768),
                )
            ]
        )


@pytest.mark.parametrize("shape", [(2, 3, 32), (8, 30)])
def test_rejects_unsupported_expert_weight_shape(nvfp4_module, monkeypatch, shape):
    M = nvfp4_module
    _fake_scaled_fp4_quant(monkeypatch, M)
    quantizer = _quantizer_with_layer_quantized(M, True)

    with pytest.raises(RuntimeError, match="block size 16"):
        quantizer.process(
            [
                (
                    "model.layers.0.mlp.experts.0.down_proj.weight",
                    torch.randn(*shape),
                )
            ]
        )


@pytest.mark.parametrize(
    ("projection_names", "checkpoint_names"),
    [
        (
            ("gate_proj", "down_proj", "up_proj"),
            ("gate_proj", "down_proj", "up_proj"),
        ),
        (("w1", "w2", "w3"), ("w1", "w2", "w3")),
    ],
    ids=["qwen-style", "w1-w2-w3-style"],
)
def test_vllm_mapping_classifies_projection_names_without_hardcoding(
    nvfp4_module, projection_names, checkpoint_names
):
    M = nvfp4_module

    class FakeRoutedExperts:
        ckpt_gate_proj_name, ckpt_down_proj_name, ckpt_up_proj_name = projection_names

        def get_expert_mapping(self):
            gate, down, up = checkpoint_names
            return [
                ("experts.w13_", f"experts.0.{gate}.", 0, "w1"),
                ("experts.w2_", f"experts.0.{down}.", 0, "w2"),
                ("experts.w13_", f"experts.0.{up}.", 0, "w3"),
            ]

    target = M._build_expert_target("model.layers.0.mlp.experts", FakeRoutedExperts())
    assert {spec.role for spec in target.specs} == {"w1", "w2", "w3"}
    assert {spec.checkpoint_suffix for spec in target.specs} == {
        f"experts.0.{name}.weight" for name in checkpoint_names
    }
    assert len({spec.logical_key for spec in target.specs}) == 1
    assert set(target.specs_by_suffix) == {
        f"experts.0.{name}.weight" for name in checkpoint_names
    }


def test_vllm_mapping_rejects_incomplete_projection_group(nvfp4_module):
    M = nvfp4_module

    class FakeRoutedExperts:
        ckpt_gate_proj_name = "w1"
        ckpt_down_proj_name = "w2"
        ckpt_up_proj_name = "w3"

        def get_expert_mapping(self):
            return [
                ("experts.w13_", "experts.0.w1.", 0, "w1"),
                ("experts.w2_", "experts.0.w2.", 0, "w2"),
            ]

    with pytest.raises(RuntimeError, match="incomplete vLLM expert mapping"):
        M._build_expert_target("model.layers.0.mlp.experts", FakeRoutedExperts())


# ------------------------------------------------------------- config


def test_rollout_config_defaults(nvfp4_module):
    M = nvfp4_module
    cfg = M.NvFp4PerTokenRolloutConfig()
    assert cfg.enabled is False
    assert cfg.resolved_ignore() == []

    layer_ignore = "*.layers.0.mlp.experts*"
    cfg2 = M.NvFp4PerTokenRolloutConfig.model_validate(
        {"enabled": True, "additional_ignore": [layer_ignore]}
    )
    assert cfg2.resolved_ignore() == [*M.DEFAULT_NVFP4_IGNORE, layer_ignore]

    with pytest.raises(ValidationError, match="unknown_key"):
        M.NvFp4PerTokenRolloutConfig.model_validate({"enabled": True, "unknown_key": 1})
    with pytest.raises(ValidationError, match="ignore"):
        M.NvFp4PerTokenRolloutConfig.model_validate({"enabled": True, "ignore": []})


@pytest.mark.parametrize(
    "pattern",
    [
        "*.layers.0.mlp.experts.1*",
        "*.layers.0.mlp.experts.1.gate_proj*",
        "*.layers.0.mlp.experts.*.gate_proj*",
        "*self_attn*",
    ],
)
def test_rollout_config_rejects_partial_expert_ignore(nvfp4_module, pattern):
    M = nvfp4_module
    with pytest.raises(ValidationError, match="complete expert layers"):
        M.NvFp4PerTokenRolloutConfig.model_validate(
            {"enabled": True, "additional_ignore": [pattern]}
        )


def test_boundary_ignores_are_derived_and_supplied_patterns_require_parity(
    nvfp4_module,
):
    from nemo_rl.models.generation.vllm.quantization.nvfp4_pertoken_config import (
        resolve_boundary_ignore_patterns,
    )

    expected = [
        "*.layers.0.mlp.experts*",
        "*.layers.1.mlp.experts*",
        "*.layers.44.mlp.experts*",
        "*.layers.45.mlp.experts*",
        "*.layers.46.mlp.experts*",
        "*.layers.47.mlp.experts*",
    ]
    assert (
        resolve_boundary_ignore_patterns(
            num_hidden_layers=48,
            first_last_layers_bf16=True,
            num_layers_at_start_in_bf16=2,
            num_layers_at_end_in_bf16=4,
            expected_additional_ignore=expected,
        )
        == expected
    )
    with pytest.raises(ValueError, match="does not match"):
        resolve_boundary_ignore_patterns(
            num_hidden_layers=48,
            first_last_layers_bf16=True,
            num_layers_at_start_in_bf16=2,
            num_layers_at_end_in_bf16=4,
            expected_additional_ignore=expected[:-1],
        )
    # An entry point that skipped driver-side normalization reaches the trainer
    # with an empty value; that must fail rather than silently quantize the
    # layers the trainer keeps in BF16.
    with pytest.raises(ValueError, match="did not run for this entry point"):
        resolve_boundary_ignore_patterns(
            num_hidden_layers=48,
            first_last_layers_bf16=True,
            num_layers_at_start_in_bf16=2,
            num_layers_at_end_in_bf16=4,
            expected_additional_ignore=[],
        )


def test_boundary_ignores_ignore_inactive_mcore_count_defaults(nvfp4_module):
    from nemo_rl.models.generation.vllm.quantization.nvfp4_pertoken_config import (
        resolve_boundary_ignore_patterns,
    )

    assert (
        resolve_boundary_ignore_patterns(
            num_hidden_layers=48,
            first_last_layers_bf16=False,
            num_layers_at_start_in_bf16=1,
            num_layers_at_end_in_bf16=1,
        )
        == []
    )
    with pytest.raises(ValueError, match="does not match"):
        resolve_boundary_ignore_patterns(
            num_hidden_layers=48,
            first_last_layers_bf16=False,
            num_layers_at_start_in_bf16=1,
            num_layers_at_end_in_bf16=1,
            expected_additional_ignore=["*.layers.0.mlp.experts*"],
        )


def test_modelopt_parser_keeps_fixed_w4a4_block16_contract(nvfp4_module):
    M = nvfp4_module
    ignore = ["*.layers.0.mlp.experts*"]
    parsed = M.NvFp4PerTokenConfig.from_config(
        M.build_nvfp4_pertoken_hf_quant_config(ignore)
    )

    assert parsed.quant_method == "NVFP4"
    assert parsed.group_size == 16
    assert parsed.exclude_modules == ignore


def test_quant_config_keeps_ordinary_linears_unquantized(nvfp4_module, monkeypatch):
    M = nvfp4_module

    class FakeLinear(torch.nn.Module):
        pass

    class FakeHead(torch.nn.Module):
        pass

    class FakeUnquantizedLinearMethod:
        pass

    monkeypatch.setattr(M, "LinearBase", FakeLinear)
    monkeypatch.setattr(M, "ParallelLMHead", FakeHead)
    monkeypatch.setattr(M, "UnquantizedLinearMethod", FakeUnquantizedLinearMethod)
    config = M.NvFp4PerTokenConfig.from_config(
        M.build_nvfp4_pertoken_hf_quant_config([])
    )

    assert isinstance(
        config.get_quant_method(FakeLinear(), "model.layers.0.mlp.dense"),
        FakeUnquantizedLinearMethod,
    )
    assert isinstance(
        config.get_quant_method(FakeHead(), "lm_head"),
        FakeUnquantizedLinearMethod,
    )


def test_quant_config_selects_only_supported_routed_experts(nvfp4_module, monkeypatch):
    M = nvfp4_module

    class FakeRoutedExperts(torch.nn.Module):
        def __init__(self, *, num_fused_shared_experts=0):
            super().__init__()
            self.moe_config = object()
            self.expert_map_manager = types.SimpleNamespace(
                num_fused_shared_experts=num_fused_shared_experts
            )

    class FakeSelectedMethod:
        def __init__(self, quant_config, moe_config):
            self.quant_config = quant_config
            self.moe_config = moe_config

    class FakeUnquantizedMethod:
        def __init__(self, moe_config):
            self.moe_config = moe_config

    monkeypatch.setattr(M, "RoutedExperts", FakeRoutedExperts)
    monkeypatch.setattr(M, "UnquantizedFusedMoEMethod", FakeUnquantizedMethod)
    config = M.NvFp4PerTokenConfig.from_config(
        M.build_nvfp4_pertoken_hf_quant_config(["*.layers.0.mlp.experts*"])
    )
    config.FusedMoEMethodCls = FakeSelectedMethod

    selected = config.get_quant_method(
        FakeRoutedExperts(), "model.layers.1.mlp.experts"
    )
    excluded = config.get_quant_method(
        FakeRoutedExperts(), "model.layers.0.mlp.experts"
    )

    assert isinstance(selected, FakeSelectedMethod)
    assert isinstance(excluded, FakeUnquantizedMethod)
    with pytest.raises(ValueError, match="fused shared experts"):
        config.get_quant_method(
            FakeRoutedExperts(num_fused_shared_experts=1),
            "model.layers.1.mlp.experts",
        )


def test_shared_resolver_applies_full_mapper_and_packed_module_names(nvfp4_module):
    from nemo_rl.models.generation.vllm.quantization.utils import (
        resolve_module_from_param_name,
    )

    root = torch.nn.Module()
    root.backbone = torch.nn.Module()
    root.backbone.layers = torch.nn.ModuleList([torch.nn.Module()])
    root.backbone.layers[0].mlp = torch.nn.Module()
    expected = torch.nn.Module()
    root.backbone.layers[0].mlp.gate_up_proj = expected
    root.packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}
    root.hf_to_vllm_mapper = types.SimpleNamespace(
        orig_to_new_prefix={"model.language_model.": "backbone."},
        orig_to_new_substr={".blocks.": ".layers."},
    )

    resolution = resolve_module_from_param_name(
        root, "model.language_model.blocks.0.mlp.gate_proj.weight"
    )

    assert resolution is not None
    assert resolution.module is expected
    assert resolution.mapped_path == (
        "backbone",
        "layers",
        "0",
        "mlp",
        "gate_up_proj",
    )


def test_shared_resolver_fails_closed_for_dropped_or_missing_paths(nvfp4_module):
    from nemo_rl.models.generation.vllm.quantization.utils import (
        resolve_module_from_param_name,
    )

    root = torch.nn.Module()
    root.packed_modules_mapping = {}
    root.hf_to_vllm_mapper = types.SimpleNamespace(
        orig_to_new_prefix={"mtp.": None}, orig_to_new_substr={}
    )

    assert resolve_module_from_param_name(root, "mtp.layers.0.weight") is None
    assert resolve_module_from_param_name(root, "model.missing.weight") is None


# ------------------------------------------------------------- worker extension


def test_prequantized_extension_uses_native_ipc_reload(
    nvfp4_module, monkeypatch, capsys
):
    """IPC quantization feeds one checkpoint-format native vLLM reload."""
    M = nvfp4_module
    from nemo_rl.models.generation.vllm import vllm_backend
    from nemo_rl.models.policy.utils import IPCProtocol, calculate_aligned_size

    _fake_scaled_fp4_quant(monkeypatch, M)

    name = "model.layers.0.mlp.experts.0.down_proj.weight"
    source_weight = torch.randn(8, 32)
    source_buffer = source_weight.view(torch.uint8)
    used_bytes = calculate_aligned_size(source_weight.nbytes)
    call_order = []

    class FakeSocket:
        def __init__(self):
            self.payloads = iter(
                [
                    ("ipc-handle", [name], used_bytes),
                    IPCProtocol.COMPLETE,
                ]
            )
            self.sent = []

        def recv_pyobj(self):
            payload = next(self.payloads)
            call_order.append(
                "recv_complete" if payload == IPCProtocol.COMPLETE else "recv_data"
            )
            return payload

        def send(self, payload):
            self.sent.append(payload)
            call_order.append(f"ack_{len(self.sent)}")

    loaded = []
    reload_kwargs = []

    def reload_weights(**kwargs):
        reload_kwargs.append(kwargs)
        loaded.extend(kwargs["weights_iterator"])
        call_order.append("reload_done")

    monkeypatch.setattr(
        vllm_backend,
        "rebuild_cuda_tensor_from_ipc",
        lambda _handle, _device_index: source_buffer,
    )
    monkeypatch.setattr(vllm_backend.gc, "collect", lambda: None)
    monkeypatch.setattr(vllm_backend.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(
        vllm_backend.torch.accelerator,
        "synchronize",
        lambda: call_order.append("final_sync"),
    )

    extension = M.NvFp4PerTokenWorkerExtension.__new__(M.NvFp4PerTokenWorkerExtension)
    extension.device = torch.device("cuda:0")
    extension.model_runner = types.SimpleNamespace(
        model=object(), reload_weights=reload_weights
    )
    extension.state_dict_info = {name: (source_weight.shape, source_weight.dtype)}
    extension.zmq_socket = FakeSocket()
    extension.maybe_init_zmq = lambda: None
    extension._synchronize_before_ipc_data_ack = lambda: call_order.append("data_sync")
    extension._quantizer = _quantizer_with_layer_quantized(
        M, True, expected_roles={("experts.0.<projection>", "w2")}
    )

    assert extension.update_weights_via_ipc_zmq() is True

    assert len(reload_kwargs) == 1
    assert reload_kwargs[0]["is_checkpoint_format"] is True
    assert [key for key, _ in loaded] == [
        f"{name_prefix}.{suffix}"
        for name_prefix in [name.removesuffix(".weight")]
        for suffix in ("weight", "weight_scale", "weight_scale_2", "input_scale")
    ]
    assert extension.zmq_socket.sent == [IPCProtocol.ACK.value.encode()] * 2
    assert call_order == [
        "recv_data",
        "data_sync",
        "ack_1",
        "recv_complete",
        "reload_done",
        "final_sync",
        "ack_2",
    ]
    assert extension._weight_update_errors_are_fatal()
    assert extension._quantizer is not None
    assert not extension._quantizer._pending
    printed = capsys.readouterr().out
    assert "[nvfp4_pertoken] refit: quantized 1 expert weight groups" in printed

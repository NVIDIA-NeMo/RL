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


def _quantizer_with_layer_quantized(M, quantized: bool):
    """Build a quantizer whose owning-module probe is stubbed to a fixed verdict.

    Bypasses ``_is_quantized_layer``'s real ``get_module_from_param_name``
    walk (which needs a live vLLM model tree) so ``process()`` can be tested
    directly against synthetic weight names.
    """
    quantizer = M.NvFp4PerTokenQuantizer.__new__(M.NvFp4PerTokenQuantizer)
    quantizer._model = None
    quantizer._pending = {}
    quantizer._quantized_layer = {}
    quantizer._quantized_events = 0
    quantizer._is_quantized_layer = lambda _prefix: quantized
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
    assert ("model.layers.0.mlp.experts", 0) in quantizer._pending

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


def test_finish_raises_when_nothing_quantized(nvfp4_module):
    """A config/name mismatch must not silently degrade to an all-BF16 refit."""
    M = nvfp4_module
    quantizer = _quantizer_with_layer_quantized(M, True)
    with pytest.raises(RuntimeError, match="quantized 0 params"):
        quantizer.finish()


def test_finish_prints_liveness_line_on_success(nvfp4_module, monkeypatch, capsys):
    M = nvfp4_module
    _fake_scaled_fp4_quant(monkeypatch, M)
    quantizer = _quantizer_with_layer_quantized(M, True)

    quantizer.process(
        [("model.layers.0.mlp.experts.0.down_proj.weight", torch.randn(8, 32))]
    )
    quantizer.finish()

    assert "[nvfp4_pertoken] refit: quantized 1" in capsys.readouterr().out


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
    tensor = torch.randn(4, 4)
    out = quantizer.process([("model.layers.0.self_attn.q_proj.weight", tensor)])
    assert out == [("model.layers.0.self_attn.q_proj.weight", tensor)]


def test_ignored_expert_layer_passes_through_bf16(nvfp4_module):
    """additional_ignore layers keep their unquantized BF16 FusedMoE method."""
    M = nvfp4_module
    quantizer = _quantizer_with_layer_quantized(M, False)
    name = "model.layers.0.mlp.experts.0.gate_proj.weight"
    tensor = torch.randn(8, 32)
    out = quantizer.process([(name, tensor)])
    assert out == [(name, tensor)]
    assert not quantizer._pending


def test_emits_all_eight_names_per_expert(nvfp4_module, monkeypatch):
    M = nvfp4_module
    _fake_scaled_fp4_quant(monkeypatch, M)
    quantizer = _quantizer_with_layer_quantized(M, True)
    p = "model.layers.0.mlp.experts.0"

    out = dict(
        quantizer.process(
            [
                (f"{p}.gate_proj.weight", torch.randn(8, 32)),
                (f"{p}.up_proj.weight", torch.randn(8, 32)),
                (f"{p}.down_proj.weight", torch.randn(32, 8)),
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


# ------------------------------------------------------------- config


def test_rollout_config_defaults(nvfp4_module):
    M = nvfp4_module
    cfg = M.NvFp4PerTokenRolloutConfig()
    assert cfg.enabled is False
    assert cfg.quant_patterns == ["*.experts.*"]
    assert cfg.resolved_ignore() == M.DEFAULT_NVFP4_IGNORE

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


# ------------------------------------------------------------- worker extension


def test_prequantized_extension_uses_real_layerwise_reload_lifecycle(
    nvfp4_module, monkeypatch
):
    """The retained path must restore stable kernel storage after each refit."""
    M = nvfp4_module
    from vllm.model_executor.model_loader.reload import record_metadata_for_reloading

    _fake_scaled_fp4_quant(monkeypatch, M)

    model = torch.nn.Linear(4, 4, bias=False)
    record_metadata_for_reloading(model)
    original_data_ptr = model.weight.data_ptr()
    synchronized = []
    monkeypatch.setattr(
        torch.accelerator, "synchronize", lambda: synchronized.append(True)
    )

    extension = M.NvFp4PerTokenWorkerExtension.__new__(M.NvFp4PerTokenWorkerExtension)
    extension.device = torch.device("cpu")
    extension.model_runner = types.SimpleNamespace(model=model)
    extension.model_config = types.SimpleNamespace(dtype=torch.float32)
    # The lifecycle's finish() hard-fails on a refit that quantized nothing, so
    # stub the owning-module probe (this toy model has no expert layers) and
    # process one synthetic expert weight before finalizing.
    extension._quantizer = _quantizer_with_layer_quantized(M, True)

    with extension._weight_update_lifecycle("ipc") as finalize:
        assert model.weight.device.type == "meta"
        extension._quantizer.process(
            [("model.layers.0.mlp.experts.0.down_proj.weight", torch.randn(8, 32))]
        )
        finalize()

    assert model.weight.device.type == "cpu"
    assert model.weight.data_ptr() == original_data_ptr
    assert synchronized == [True]
    assert extension._weight_update_errors_are_fatal()
    # The lifecycle must construct and reset a quantizer on entry, and finish()
    # it (a no-op here — nothing was ever processed) before finalizing.
    assert extension._quantizer is not None
    assert not extension._quantizer._pending

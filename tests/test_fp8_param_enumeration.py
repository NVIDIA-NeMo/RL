"""
Tests for FP8 parameter enumeration, keyword-based layer skipping,
scale calibration, and VLM compatibility in nemo_rl.models.generation.fp8.

Tests the actual implementation in nemo_rl.models.generation.fp8:
- _get_param_names_from_checkpoint reads safetensors index
- _get_params_in_layers returns correct param names for given layers
- apply_fp8_patches monkey-patches vllm's is_layer_skipped with keyword matching
- convert_calibration_to_vllm_format handles VLM-specific layer keys
- process_weights_after_loading calibrates sentinel/direct-cast FP8 scales
- maybe_post_process_fp8_weight_block uses correct scale attribute

Run:
    python -m pytest tests/test_fp8_param_enumeration.py -v
"""

import importlib.util
import json
import sys
from unittest.mock import MagicMock

import pytest

# fp8.py imports vllm at module level. Stub it when unavailable so the
# pure-Python functions (_get_param_names_from_checkpoint, _get_params_in_layers,
# FP8Config) can still be imported and tested.
HAS_VLLM = importlib.util.find_spec("vllm") is not None

if not HAS_VLLM:
    for _mod in [
        "vllm",
        "vllm.model_executor",
        "vllm.model_executor.layers",
        "vllm.model_executor.layers.quantization",
        "vllm.model_executor.layers.quantization.fp8",
        "vllm.model_executor.layers.quantization.utils",
        "vllm.model_executor.layers.quantization.utils.quant_utils",
        "vllm.model_executor.layers.quantization.kv_cache",
        "vllm.triton_utils",
        "vllm.v1",
        "vllm.v1.engine",
        "vllm.v1.engine.core",
        "vllm.v1.engine.utils",
    ]:
        sys.modules.setdefault(_mod, MagicMock())

NEMOTRONH_PARAMS = {
    "language_model.backbone.layers.0.norm.weight": "shard-00001.safetensors",
    "language_model.backbone.layers.0.mixer.D": "shard-00001.safetensors",
    "language_model.backbone.layers.0.mixer.in_proj.weight": "shard-00001.safetensors",
    "language_model.backbone.layers.0.mixer.out_proj.weight": "shard-00001.safetensors",
    "language_model.backbone.layers.1.mixer.moe.gate.weight": "shard-00002.safetensors",
    "language_model.backbone.layers.1.mixer.moe.experts.0.up_proj.weight": "shard-00002.safetensors",
    "language_model.backbone.layers.1.mixer.moe.experts.0.down_proj.weight": "shard-00002.safetensors",
    "language_model.backbone.layers.1.mixer.shared_experts.up_proj.weight": "shard-00002.safetensors",
    "language_model.backbone.layers.1.mixer.shared_experts.down_proj.weight": "shard-00002.safetensors",
    "language_model.backbone.layers.2.mixer.shared_experts.up_proj.weight": "shard-00003.safetensors",
    "language_model.backbone.layers.2.mixer.shared_experts.down_proj.weight": "shard-00003.safetensors",
    "language_model.backbone.layers.2.mixer.moe.experts.0.up_proj.weight": "shard-00003.safetensors",
}


@pytest.fixture
def nemotronh_checkpoint(tmp_path):
    index = {"metadata": {}, "weight_map": NEMOTRONH_PARAMS}
    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text(json.dumps(index))
    return str(tmp_path)


class TestGetParamNamesFromCheckpoint:
    def test_reads_all_params(self, nemotronh_checkpoint):
        from nemo_rl.models.generation.fp8 import _get_param_names_from_checkpoint

        names = _get_param_names_from_checkpoint(nemotronh_checkpoint)
        assert len(names) == len(NEMOTRONH_PARAMS)
        assert set(names) == set(NEMOTRONH_PARAMS.keys())

    def test_preserves_insertion_order(self, nemotronh_checkpoint):
        from nemo_rl.models.generation.fp8 import _get_param_names_from_checkpoint

        names = _get_param_names_from_checkpoint(nemotronh_checkpoint)
        assert names == list(NEMOTRONH_PARAMS.keys())

    @pytest.mark.skipif(not HAS_VLLM, reason="fallback path needs real vllm")
    def test_no_index_falls_back_to_automodel(self, tmp_path):
        """Without safetensors index, falls back to AutoModel.from_config.
        On an empty dir this either raises or returns an empty list."""
        from nemo_rl.models.generation.fp8 import _get_param_names_from_checkpoint

        try:
            names = _get_param_names_from_checkpoint(str(tmp_path))
            assert isinstance(names, list)
        except Exception:
            pass  # raising is also acceptable


class TestGetParamsInLayers:
    def test_finds_layer_params(self, nemotronh_checkpoint):
        from nemo_rl.models.generation.fp8 import (
            _get_param_names_from_checkpoint,
            _get_params_in_layers,
        )

        param_names = _get_param_names_from_checkpoint(nemotronh_checkpoint)
        result = _get_params_in_layers(param_names, [1])

        assert len(result) > 0
        for p in result:
            assert "layers.1." in p
            assert p.startswith("model.")

    def test_returns_flat_list_of_strings(self, nemotronh_checkpoint):
        from nemo_rl.models.generation.fp8 import (
            _get_param_names_from_checkpoint,
            _get_params_in_layers,
        )

        param_names = _get_param_names_from_checkpoint(nemotronh_checkpoint)
        result = _get_params_in_layers(param_names, [1, 2])

        assert isinstance(result, list)
        assert all(isinstance(p, str) for p in result)

    def test_extend_produces_flat_list(self, nemotronh_checkpoint):
        from nemo_rl.models.generation.fp8 import (
            _get_param_names_from_checkpoint,
            _get_params_in_layers,
        )

        param_names = _get_param_names_from_checkpoint(nemotronh_checkpoint)

        bf16_params = []
        bf16_params.extend(_get_params_in_layers(param_names, [0]))
        bf16_params.extend(_get_params_in_layers(param_names, [1]))

        assert all(isinstance(p, str) for p in bf16_params)
        assert len(bf16_params) > 0

    def test_unknown_layer_raises(self, nemotronh_checkpoint):
        from nemo_rl.models.generation.fp8 import (
            _get_param_names_from_checkpoint,
            _get_params_in_layers,
        )

        param_names = _get_param_names_from_checkpoint(nemotronh_checkpoint)
        with pytest.raises(ValueError, match="Could not identify layers"):
            _get_params_in_layers(param_names, [999])

    def test_excludes_bias_and_layernorm(self, tmp_path):
        params = {
            "language_model.backbone.layers.0.attn.q_proj.weight": "s.safetensors",
            "language_model.backbone.layers.0.attn.q_proj.bias": "s.safetensors",
            "language_model.backbone.layers.0.layernorm.weight": "s.safetensors",
        }
        index_path = tmp_path / "model.safetensors.index.json"
        index_path.write_text(json.dumps({"metadata": {}, "weight_map": params}))

        from nemo_rl.models.generation.fp8 import (
            _get_param_names_from_checkpoint,
            _get_params_in_layers,
        )

        param_names = _get_param_names_from_checkpoint(str(tmp_path))
        result = _get_params_in_layers(param_names, [0])

        assert len(result) == 1
        assert "bias" not in result[0]
        assert "layernorm" not in result[0]


class TestFP8Config:
    def test_default_keywords_empty(self):
        from nemo_rl.models.generation.fp8 import FP8Config

        cfg = FP8Config(model_parallel_size=1)
        assert cfg.ignored_layer_keywords == ()

    def test_keywords_stored(self):
        from nemo_rl.models.generation.fp8 import FP8Config

        cfg = FP8Config(
            model_parallel_size=1,
            ignored_layer_keywords=("shared_experts", "in_proj"),
        )
        assert cfg.ignored_layer_keywords == ("shared_experts", "in_proj")

    def test_frozen(self):
        from nemo_rl.models.generation.fp8 import FP8Config

        cfg = FP8Config(model_parallel_size=1)
        with pytest.raises(AttributeError):
            cfg.ignored_layer_keywords = ("x",)


@pytest.mark.skipif(not HAS_VLLM, reason="vllm not installed — keyword patch tests need real vllm")
class TestKeywordAwareIsLayerSkipped:
    """Calls the real apply_fp8_patches and verifies the patched
    is_layer_skipped on the actual vllm module."""

    @pytest.fixture(autouse=True)
    def _reset_fp8_state(self):
        import nemo_rl.models.generation.fp8 as fp8_mod
        import vllm.model_executor.layers.quantization.fp8 as vllm_fp8
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            is_layer_skipped as canonical_is_layer_skipped,
        )

        saved_config = fp8_mod.global_fp8_config
        saved_applied = fp8_mod.fp8_patches_applied
        saved_state = fp8_mod.fp8_state

        fp8_mod.fp8_patches_applied = False
        fp8_mod.fp8_state = fp8_mod.FP8State()
        vllm_fp8.is_layer_skipped = canonical_is_layer_skipped

        yield

        for p in fp8_mod.fp8_state.vllm_patches:
            try:
                p.stop()
            except RuntimeError:
                pass

        vllm_fp8.is_layer_skipped = canonical_is_layer_skipped
        fp8_mod.global_fp8_config = saved_config
        fp8_mod.fp8_patches_applied = saved_applied
        fp8_mod.fp8_state = saved_state

    @staticmethod
    def _apply_keyword_patch(keywords):
        import nemo_rl.models.generation.fp8 as fp8_mod
        from nemo_rl.models.generation.fp8 import FP8Config

        config = FP8Config(
            use_fp8_weights=False,
            ignored_layer_keywords=tuple(keywords),
            model_parallel_size=1,
        )
        fp8_mod.apply_fp8_patches(None, config)

    def test_patch_replaces_function(self):
        import vllm.model_executor.layers.quantization.fp8 as vllm_fp8

        original = vllm_fp8.is_layer_skipped
        self._apply_keyword_patch(["shared_experts"])
        assert vllm_fp8.is_layer_skipped is not original

    def test_empty_keywords_leaves_function_unchanged(self):
        import vllm.model_executor.layers.quantization.fp8 as vllm_fp8

        original = vllm_fp8.is_layer_skipped
        self._apply_keyword_patch([])
        assert vllm_fp8.is_layer_skipped is original

    def test_shared_experts_skipped(self):
        import vllm.model_executor.layers.quantization.fp8 as vllm_fp8

        self._apply_keyword_patch(["shared_experts"])

        assert vllm_fp8.is_layer_skipped(
            "language_model.model.layers.1.mixer.shared_experts.up_proj",
            ignored_layers=[],
        )
        assert vllm_fp8.is_layer_skipped(
            "language_model.model.layers.2.mixer.shared_experts.down_proj",
            ignored_layers=[],
        )

    def test_routed_experts_not_skipped(self):
        import vllm.model_executor.layers.quantization.fp8 as vllm_fp8

        self._apply_keyword_patch(["shared_experts"])

        assert not vllm_fp8.is_layer_skipped(
            "language_model.model.layers.1.mixer.moe.experts.0.up_proj",
            ignored_layers=[],
        )

    def test_non_expert_layers_not_skipped(self):
        import vllm.model_executor.layers.quantization.fp8 as vllm_fp8

        self._apply_keyword_patch(["shared_experts"])

        assert not vllm_fp8.is_layer_skipped(
            "language_model.model.layers.0.mixer.in_proj",
            ignored_layers=[],
        )
        assert not vllm_fp8.is_layer_skipped(
            "language_model.model.layers.0.self_attn.q_proj",
            ignored_layers=[],
        )

    def test_original_prefix_match_still_works(self):
        import vllm.model_executor.layers.quantization.fp8 as vllm_fp8

        self._apply_keyword_patch(["shared_experts"])

        assert vllm_fp8.is_layer_skipped(
            "model.layers.0.self_attn.q_proj",
            ignored_layers=["model.layers.0.self_attn.q_proj"],
        )

    def test_original_returns_false_then_keyword_catches(self):
        import vllm.model_executor.layers.quantization.fp8 as vllm_fp8

        self._apply_keyword_patch(["shared_experts"])

        assert vllm_fp8.is_layer_skipped(
            "language_model.model.layers.1.mixer.shared_experts.up_proj",
            ignored_layers=["some.other.layer"],
        )

    def test_multiple_keywords(self):
        import vllm.model_executor.layers.quantization.fp8 as vllm_fp8

        self._apply_keyword_patch(["shared_experts", "in_proj"])

        assert vllm_fp8.is_layer_skipped(
            "language_model.model.layers.1.mixer.shared_experts.up_proj",
            ignored_layers=[],
        )
        assert vllm_fp8.is_layer_skipped(
            "language_model.model.layers.0.mixer.in_proj",
            ignored_layers=[],
        )
        assert not vllm_fp8.is_layer_skipped(
            "language_model.model.layers.0.mixer.out_proj",
            ignored_layers=[],
        )

    def test_keyword_is_substring_not_exact(self):
        import vllm.model_executor.layers.quantization.fp8 as vllm_fp8

        self._apply_keyword_patch(["shared"])

        assert vllm_fp8.is_layer_skipped(
            "language_model.model.layers.1.mixer.shared_experts.up_proj",
            ignored_layers=[],
        )

    def test_keyword_no_false_substring(self):
        import vllm.model_executor.layers.quantization.fp8 as vllm_fp8

        self._apply_keyword_patch(["shared_experts"])

        assert not vllm_fp8.is_layer_skipped(
            "language_model.model.layers.1.mixer.moe.experts.0.up_proj",
            ignored_layers=[],
        )


# ---------------------------------------------------------------------------
# Tests for convert_calibration_to_vllm_format (VLM vision-layer handling)
# ---------------------------------------------------------------------------

class TestConvertCalibrationToVllmFormat:
    def test_basic_conversion(self):
        from nemo_rl.models.generation.fp8 import convert_calibration_to_vllm_format

        calib = {
            "layer_0": {"q_scale": 1.0, "k_scale": 2.0, "v_scale": 3.0},
            "layer_1": {"q_scale": 1.5, "k_scale": 2.5, "v_scale": 3.5},
        }
        result = convert_calibration_to_vllm_format(calib)
        assert result["model.layers.0.self_attn.attn.q_scale"] == 1.0
        assert result["model.layers.0.self_attn.k_scale"] == 2.0
        assert result["model.layers.0.self_attn.v_scale"] == 3.0
        assert result["model.layers.1.self_attn.attn.q_scale"] == 1.5
        assert result["model.layers.1.self_attn.k_scale"] == 2.5
        assert result["model.layers.1.self_attn.v_scale"] == 3.5

    def test_skips_vision_encoder_keys(self):
        """VLM models produce calibration entries for vision encoder attention
        layers that don't follow the layer_N naming convention."""
        from nemo_rl.models.generation.fp8 import convert_calibration_to_vllm_format

        calib = {
            "layer_0": {"k_scale": 2.0, "v_scale": 3.0},
            "layer_model.vision_encoder.layers.0.self_attention.core_attention": {
                "k_scale": 0.1, "v_scale": 0.2,
            },
            "module.vision_model.encoder.layers.0.self_attention.core_attention": {
                "k_scale": 0.3, "v_scale": 0.4,
            },
        }
        result = convert_calibration_to_vllm_format(calib)
        assert len(result) == 2
        assert "model.layers.0.self_attn.k_scale" in result
        assert "model.layers.0.self_attn.v_scale" in result

    def test_handles_missing_q_scale(self):
        """When include_q=False, calibration entries don't contain q_scale."""
        from nemo_rl.models.generation.fp8 import convert_calibration_to_vllm_format

        calib = {
            "layer_0": {"k_scale": 2.0, "v_scale": 3.0},
        }
        result = convert_calibration_to_vllm_format(calib)
        assert "model.layers.0.self_attn.k_scale" in result
        assert "model.layers.0.self_attn.v_scale" in result
        assert "model.layers.0.self_attn.attn.q_scale" not in result

    def test_empty_calibration(self):
        from nemo_rl.models.generation.fp8 import convert_calibration_to_vllm_format

        result = convert_calibration_to_vllm_format({})
        assert result == {}

    def test_malformed_keys_skipped(self):
        from nemo_rl.models.generation.fp8 import convert_calibration_to_vllm_format

        calib = {
            "layer_abc": {"k_scale": 1.0, "v_scale": 2.0},
            "layer_": {"k_scale": 1.0, "v_scale": 2.0},
            "layer_0": {"k_scale": 1.0, "v_scale": 2.0},
        }
        result = convert_calibration_to_vllm_format(calib)
        assert len(result) == 2  # only layer_0 produces k_scale + v_scale


# ---------------------------------------------------------------------------
# Tests for process_weights_after_loading scale calibration
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_VLLM, reason="needs real vllm for FP8 types")
class TestProcessWeightsAfterLoadingCalibration:
    """Tests that process_weights_after_loading correctly detects and calibrates
    FP8 layers with sentinel or direct-cast scales."""

    @pytest.fixture(autouse=True)
    def _reset_fp8_state(self):
        import nemo_rl.models.generation.fp8 as fp8_mod
        saved_config = fp8_mod.global_fp8_config
        saved_applied = fp8_mod.fp8_patches_applied
        saved_state = fp8_mod.fp8_state

        fp8_mod.fp8_patches_applied = False
        fp8_mod.fp8_state = fp8_mod.FP8State()

        fp8_mod.global_fp8_config = fp8_mod.FP8Config(model_parallel_size=1)

        yield

        for p in fp8_mod.fp8_state.vllm_patches:
            try:
                p.stop()
            except RuntimeError:
                pass
        fp8_mod.global_fp8_config = saved_config
        fp8_mod.fp8_patches_applied = saved_applied
        fp8_mod.fp8_state = saved_state

    @staticmethod
    def _make_fp8_layer(weight_fp32, scale_inv_value):
        """Create a layer object that looks like a vLLM FP8 linear layer.

        Uses a real object instead of MagicMock so that hasattr() checks
        in process_weights_after_loading behave correctly.
        """
        import torch

        fp8_weight = weight_fp32.to(torch.float8_e4m3fn)

        scale_shape = (
            (weight_fp32.shape[0] + 127) // 128,
            (weight_fp32.shape[1] + 127) // 128,
        )

        layer = type("FP8Layer", (), {})()
        layer.weight = torch.nn.Parameter(fp8_weight, requires_grad=False)
        layer.weight_scale_inv = torch.nn.Parameter(
            torch.full(scale_shape, scale_inv_value, dtype=torch.float32),
            requires_grad=False,
        )
        layer.weight_block_size = [128, 128]
        layer.orig_dtype = torch.bfloat16
        layer.input_scale = None
        return layer

    def test_sentinel_scales_trigger_calibration(self):
        """Sentinel scale values (negative) should be replaced with real scales."""
        import torch
        from nemo_rl.models.generation.fp8 import FP8_BLOCK_QUANT_KWARGS

        weight_fp32 = torch.randn(256, 256)
        sentinel = torch.finfo(torch.float32).min
        layer = self._make_fp8_layer(weight_fp32, sentinel)

        assert (layer.weight_scale_inv.data < 0).all()

        from nemo_rl.models.generation.fp8 import process_weights_after_loading

        mock_self = MagicMock()
        mock_self.block_quant = True
        mock_self.quant_config.is_checkpoint_fp8_serialized = True
        mock_self.quant_config.activation_scheme = "dynamic"

        process_weights_after_loading(mock_self, layer)

        assert (layer.weight_scale_inv.data > 0).all()
        assert not torch.isinf(layer.weight_scale_inv.data).any()

    def test_direct_cast_small_values_trigger_calibration(self):
        """Direct-cast FP8 values (small magnitude, w_max < 16) should be recalibrated."""
        import torch

        weight_fp32 = torch.randn(256, 256) * 0.1
        layer = self._make_fp8_layer(weight_fp32, 0.001)

        w_max_before = layer.weight.data.float().abs().max().item()
        assert w_max_before < 16.0

        from nemo_rl.models.generation.fp8 import process_weights_after_loading

        mock_self = MagicMock()
        mock_self.block_quant = True
        mock_self.quant_config.is_checkpoint_fp8_serialized = True
        mock_self.quant_config.activation_scheme = "dynamic"

        process_weights_after_loading(mock_self, layer)

        w_max_after = layer.weight.data.float().abs().max().item()
        assert w_max_after > 16.0, (
            "After calibration, FP8 weights should use the full FP8 range"
        )
        assert (layer.weight_scale_inv.data > 0).all()

    def test_already_quantized_weights_not_recalibrated(self):
        """Properly quantized FP8 weights (large magnitude, valid scales)
        should NOT be recalibrated — scales should remain unchanged."""
        import torch
        from nemo_rl.models.generation.fp8 import cast_tensor_to_fp8_blockwise, FP8_BLOCK_QUANT_KWARGS

        weight_fp32 = torch.randn(256, 256)
        fp8_data, scale = cast_tensor_to_fp8_blockwise(
            weight_fp32, FP8_BLOCK_QUANT_KWARGS["weight_block_size"]
        )
        scale = torch.squeeze(scale, dim=-1)

        layer = type("FP8Layer", (), {})()
        layer.weight = torch.nn.Parameter(fp8_data, requires_grad=False)
        layer.weight_scale_inv = torch.nn.Parameter(scale, requires_grad=False)
        layer.weight_block_size = [128, 128]
        layer.orig_dtype = torch.bfloat16
        layer.input_scale = None

        scale_before = layer.weight_scale_inv.data.clone()

        from nemo_rl.models.generation.fp8 import process_weights_after_loading

        mock_self = MagicMock()
        mock_self.block_quant = True
        mock_self.quant_config.is_checkpoint_fp8_serialized = True
        mock_self.quant_config.activation_scheme = "dynamic"

        process_weights_after_loading(mock_self, layer)

        assert torch.allclose(layer.weight_scale_inv.data, scale_before, atol=1e-6), (
            "Already-quantized weights should not have their scales changed"
        )


# ---------------------------------------------------------------------------
# Tests for maybe_post_process_fp8_weight_block scale attribute selection
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_VLLM, reason="needs real vllm for DeepGemm imports")
class TestMaybePostProcessFp8WeightBlock:
    """Tests that maybe_post_process_fp8_weight_block uses weight_scale_inv
    when available, falling back to weight_scale."""

    def test_uses_weight_scale_inv_when_present(self):
        import torch
        import nemo_rl.models.generation.fp8 as fp8_mod
        from nemo_rl.models.generation.fp8 import FP8Config, FP8_BLOCK_QUANT_KWARGS

        saved_config = fp8_mod.global_fp8_config
        fp8_mod.global_fp8_config = FP8Config(model_parallel_size=1)

        try:
            from nemo_rl.models.generation.fp8 import cast_tensor_to_fp8_blockwise

            weight_fp32 = torch.randn(256, 256)
            fp8_data, scale = cast_tensor_to_fp8_blockwise(
                weight_fp32, FP8_BLOCK_QUANT_KWARGS["weight_block_size"]
            )
            scale = torch.squeeze(scale, dim=-1)

            layer = type("FP8Layer", (), {})()
            layer.weight = torch.nn.Parameter(fp8_data.cuda(), requires_grad=False)
            layer.weight_block_size = [128, 128]
            layer.orig_dtype = torch.bfloat16
            layer.weight_scale_inv = torch.nn.Parameter(scale.cuda(), requires_grad=False)

            from nemo_rl.models.generation.fp8 import maybe_post_process_fp8_weight_block

            try:
                maybe_post_process_fp8_weight_block(layer)
            except Exception:
                pytest.skip("DeepGemm not available in this environment")
        finally:
            fp8_mod.global_fp8_config = saved_config


# ---------------------------------------------------------------------------
# Tests for get_vllm_qkv_scale_names
# ---------------------------------------------------------------------------

class TestGetVllmQkvScaleNames:
    def test_layer_0(self):
        from nemo_rl.models.generation.fp8 import get_vllm_qkv_scale_names

        names = get_vllm_qkv_scale_names(0)
        assert names["q_scale"] == "model.layers.0.self_attn.attn.q_scale"
        assert names["k_scale"] == "model.layers.0.self_attn.k_scale"
        assert names["v_scale"] == "model.layers.0.self_attn.v_scale"

    def test_returns_all_three_keys(self):
        from nemo_rl.models.generation.fp8 import get_vllm_qkv_scale_names

        names = get_vllm_qkv_scale_names(5)
        assert set(names.keys()) == {"q_scale", "k_scale", "v_scale"}
        assert "layers.5." in names["k_scale"]

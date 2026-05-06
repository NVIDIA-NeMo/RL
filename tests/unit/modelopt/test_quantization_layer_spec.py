"""Tests for quantization_layer_spec mamba/hybrid branching."""

import os
from unittest.mock import MagicMock, patch

from nemo_rl.modelopt.models.policy.workers.utils import (
    _is_mamba_provider,
    quantization_layer_spec,
)


def _make_mamba_cfg():
    cfg = MagicMock(spec=["mamba_stack_spec", "hybrid_override_pattern"])
    cfg.mamba_stack_spec = "MOCK"
    return cfg


def _make_hybrid_cfg():
    cfg = MagicMock(spec=["mamba_num_heads", "hybrid_override_pattern"])
    cfg.mamba_num_heads = 1
    cfg.hybrid_override_pattern = "M-A-"
    return cfg


def _make_gpt_cfg():
    cfg = MagicMock(spec=["num_layers"])
    cfg.num_layers = 4
    return cfg


def test_is_mamba_provider_detects_mamba_stack_spec():
    assert _is_mamba_provider(_make_mamba_cfg()) is True


def test_is_mamba_provider_detects_hybrid():
    assert _is_mamba_provider(_make_hybrid_cfg()) is True


def test_is_mamba_provider_rejects_gpt():
    assert _is_mamba_provider(_make_gpt_cfg()) is False


def test_is_mamba_provider_detects_hybrid_new_name():
    cfg = MagicMock(spec=["mamba_num_heads", "hybrid_layer_pattern"])
    cfg.mamba_num_heads = 1
    cfg.hybrid_layer_pattern = "M-A-"
    assert _is_mamba_provider(cfg) is True


def test_is_mamba_provider_rejects_mamba_num_heads_without_pattern():
    cfg = MagicMock(spec=["mamba_num_heads"])
    cfg.mamba_num_heads = 1
    assert _is_mamba_provider(cfg) is False


@patch.dict(os.environ, {"DISABLE_MODELOPT_LAYER_SPEC": "0"}, clear=False)
@patch("nemo_rl.modelopt.models.policy.workers.utils.modelopt_mamba_stack_spec")
def test_quantization_layer_spec_mamba_default(mock_modelopt_mamba):
    mock_modelopt_mamba.return_value = "MAMBA_QUANT_SPEC"
    cfg = _make_mamba_cfg()
    assert quantization_layer_spec(cfg) == "MAMBA_QUANT_SPEC"
    mock_modelopt_mamba.assert_called_once_with(cfg)


@patch.dict(os.environ, {"DISABLE_MODELOPT_LAYER_SPEC": "1"}, clear=False)
@patch(
    "nemo_rl.modelopt.models.policy.workers.utils.transformer_engine_mamba_stack_spec"
)
def test_quantization_layer_spec_mamba_disabled(mock_te_mamba):
    mock_te_mamba.return_value = "TE_MAMBA_SPEC"
    cfg = _make_mamba_cfg()
    assert quantization_layer_spec(cfg) == "TE_MAMBA_SPEC"
    mock_te_mamba.assert_called_once_with()

"""Tests for Nano3 NVFP4 YAML quantization recipes."""

from pathlib import Path

import pytest

from nemo_rl.modelopt.utils import resolve_quant_cfg

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "examples" / "modelopt" / "quant_configs"


@pytest.mark.parametrize(
    "filename",
    [
        "nano3_nvfp4_default.yaml",
        "nano3_nvfp4_inputonly.yaml",
        "nano3_nvfp4_weightonly.yaml",
    ],
)
def test_nano3_yaml_contains_required_disables(filename):
    path = CONFIG_DIR / filename
    cfg = resolve_quant_cfg(str(path))
    assert "quant_cfg" in cfg
    assert isinstance(cfg["quant_cfg"], list) and cfg["quant_cfg"]
    flat_names = [
        e.get("quantizer_name") for e in cfg["quant_cfg"] if isinstance(e, dict)
    ]
    for required in [
        "*.[q|k|v|o]_proj.*",
        "*.qkv_proj.*",
        "*.linear_proj.*",
        "*.linear_qkv.*",
        "*mixer.conv1d*",
    ]:
        assert required in flat_names, f"{required} missing from {filename}"


def test_nano3_inputonly_disables_weight_quantizer():
    cfg = resolve_quant_cfg(str(CONFIG_DIR / "nano3_nvfp4_inputonly.yaml"))
    has_weight_disabled = any(
        e.get("quantizer_name") == "*weight_quantizer" and e.get("enable") is False
        for e in cfg["quant_cfg"]
    )
    assert has_weight_disabled


def test_nano3_weightonly_disables_input_quantizer():
    cfg = resolve_quant_cfg(str(CONFIG_DIR / "nano3_nvfp4_weightonly.yaml"))
    has_input_disabled = any(
        e.get("quantizer_name") == "*input_quantizer" and e.get("enable") is False
        for e in cfg["quant_cfg"]
    )
    assert has_input_disabled


def test_nano3_default_disables_bf16_attention_layers():
    cfg = resolve_quant_cfg(str(CONFIG_DIR / "nano3_nvfp4_default.yaml"))
    flat_names = [
        e.get("quantizer_name") for e in cfg["quant_cfg"] if isinstance(e, dict)
    ]
    for layer_idx in (4, 11, 18, 25, 32, 41):
        pattern = f"*.layers.{layer_idx}.*"
        assert pattern in flat_names, f"{pattern} missing from nano3_nvfp4_default.yaml"

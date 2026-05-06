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


def _enables_quantizer_pattern(qcfg: list, pattern: str) -> bool:
    """Return True if any entry explicitly sets ``pattern`` enable=true."""
    return any(
        isinstance(e, dict)
        and e.get("quantizer_name") == pattern
        and e.get("enable") is True
        for e in qcfg
    )


def _has_deny_all(qcfg: list) -> bool:
    """Return True if the recipe starts with the deny-all '*' enable=false entry."""
    if not qcfg:
        return False
    first = qcfg[0]
    return (
        isinstance(first, dict)
        and first.get("quantizer_name") == "*"
        and first.get("enable") is False
    )


@pytest.mark.parametrize(
    "filename",
    [
        "nano3_nvfp4_default.yaml",
        "nano3_nvfp4_inputonly.yaml",
        "nano3_nvfp4_weightonly.yaml",
    ],
)
def test_nano3_yaml_starts_with_deny_all(filename):
    """Every recipe must use the deny-all-then-enable pattern so KV-cache
    bmm quantizers (q_bmm/k_bmm/v_bmm) stay disabled."""
    cfg = resolve_quant_cfg(str(CONFIG_DIR / filename))
    assert _has_deny_all(cfg["quant_cfg"]), (
        f"{filename} does not start with `* enable=false` — would silently "
        f"enable KV-cache bmm quantizers and crash on dummy calibration."
    )


def test_nano3_inputonly_does_not_enable_weight_quantizer():
    """Input-only must not enable *weight_quantizer."""
    cfg = resolve_quant_cfg(str(CONFIG_DIR / "nano3_nvfp4_inputonly.yaml"))
    assert _enables_quantizer_pattern(cfg["quant_cfg"], "*input_quantizer")
    assert not _enables_quantizer_pattern(cfg["quant_cfg"], "*weight_quantizer")


def test_nano3_weightonly_does_not_enable_input_quantizer():
    """Weight-only must not enable *input_quantizer."""
    cfg = resolve_quant_cfg(str(CONFIG_DIR / "nano3_nvfp4_weightonly.yaml"))
    assert _enables_quantizer_pattern(cfg["quant_cfg"], "*weight_quantizer")
    assert not _enables_quantizer_pattern(cfg["quant_cfg"], "*input_quantizer")


def test_nano3_default_disables_bf16_attention_layers():
    cfg = resolve_quant_cfg(str(CONFIG_DIR / "nano3_nvfp4_default.yaml"))
    flat_names = [
        e.get("quantizer_name") for e in cfg["quant_cfg"] if isinstance(e, dict)
    ]
    for layer_idx in (4, 11, 18, 25, 32, 41):
        pattern = f"*.layers.{layer_idx}.*"
        assert pattern in flat_names, f"{pattern} missing from nano3_nvfp4_default.yaml"

"""Environment gating, source drift, and exact-scale refit regression tests."""

import ast
import importlib.util
import logging
import sys
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).parents[4]
ENV = "NRL_VLLM_MXFP8_TE_COMPAT"
HELPER = '''@dsl_user_op
def ue8m0_to_inv_scale_fast(ue8m0_val: Uint32, *, loc=None, ip=None) -> Float32:
    """Returns 0 for ue8m0 == 0."""
    return llvm.inline_asm("""
        sub.s32 new_exp, 254, $1;
        shl.b32 float_bits, new_exp, 23;
        mov.b32 $0, float_bits;
        @p_zero mov.b32 $0, 0;
    """)
'''
KERNEL = """from ..quantization_cute_dsl_utils import (
    ue8m0_to_inv_scale_fast,
)

def quantize():
    return ue8m0_to_inv_scale_fast(0)
"""


def load_file(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def patches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[types.ModuleType, Path, Path]:
    module = load_file(
        "te_compat_patches", ROOT / "nemo_rl/models/generation/vllm/patches.py"
    )
    monkeypatch.delenv(ENV, raising=False)
    for name in [
        "flashinfer.quantization.quantization_cute_dsl_utils",
        "flashinfer.quantization.kernels.mxfp8_quantize",
    ]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    helper = tmp_path / "quantization/quantization_cute_dsl_utils.py"
    kernel = tmp_path / "quantization/kernels/mxfp8_quantize.py"
    kernel.parent.mkdir(parents=True)
    helper.write_text(HELPER)
    kernel.write_text(KERNEL)
    monkeypatch.setattr(
        module,
        "find_spec",
        lambda name: types.SimpleNamespace(submodule_search_locations=[str(tmp_path)]),
    )
    return module, helper, kernel


@pytest.mark.parametrize("setting", [None, "0"])
def test_disabled_does_not_require_flashinfer_or_edit(
    patches: tuple[types.ModuleType, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    setting: str | None,
) -> None:
    module, helper, kernel = patches
    if setting is not None:
        monkeypatch.setenv(ENV, setting)
    monkeypatch.setattr(
        module,
        "find_spec",
        lambda name: pytest.fail("disabled flag inspected optional package"),
    )
    assert module.patch_flashinfer_mxfp8_te_compat(logging.getLogger()) is False
    assert helper.read_text() == HELPER and kernel.read_text() == KERNEL


def test_enabled_preserves_legacy_helper_and_is_idempotent(
    patches: tuple[types.ModuleType, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    module, helper, kernel = patches
    monkeypatch.setenv(ENV, "1")
    assert module.patch_flashinfer_mxfp8_te_compat(logging.getLogger())
    first = helper.read_bytes(), kernel.read_bytes()
    assert helper.read_text().startswith(HELPER)
    tree = ast.parse(helper.read_text())
    compatible = next(n for n in tree.body if n.name == "ue8m0_to_inv_scale_te_compat")
    code = ast.get_source_segment(helper.read_text(), compatible)
    assert "@p_zero mov.b32 $0, 0;" not in code
    assert "sub.s32 new_exp, 254, $1;" in code
    assert module.patch_flashinfer_mxfp8_te_compat(logging.getLogger())
    assert first == (helper.read_bytes(), kernel.read_bytes())


def test_patched_source_selects_helper_from_environment(
    patches: tuple[types.ModuleType, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    module, _, kernel = patches
    monkeypatch.setenv(ENV, "1")
    module.patch_flashinfer_mxfp8_te_compat(logging.getLogger())
    utils = types.ModuleType("fi.quantization.quantization_cute_dsl_utils")
    legacy, compatible = object(), object()
    utils.ue8m0_to_inv_scale_fast = lambda _: legacy
    utils.ue8m0_to_inv_scale_te_compat = lambda _: compatible
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    for setting, expected in [("0", legacy), ("1", compatible), ("0", legacy)]:
        monkeypatch.setenv(ENV, setting)
        namespace = {"__package__": "fi.quantization.kernels"}
        exec(compile(kernel.read_text(), str(kernel), "exec"), namespace)
        assert namespace["quantize"]() is expected


@pytest.mark.parametrize("setting", ["true", "2", ""])
def test_invalid_setting_fails_before_edit(
    patches: tuple[types.ModuleType, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    setting: str,
) -> None:
    module, helper, kernel = patches
    monkeypatch.setenv(ENV, setting)
    with pytest.raises(ValueError, match="must be 0 or 1"):
        module.patch_flashinfer_mxfp8_te_compat(logging.getLogger())
    assert helper.read_text() == HELPER and kernel.read_text() == KERNEL


def test_upstream_helper_drift_fails(
    patches: tuple[types.ModuleType, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    module, helper, kernel = patches
    monkeypatch.setenv(ENV, "1")
    helper.write_text(HELPER.replace("@p_zero mov.b32 $0, 0;", "// changed upstream"))
    with pytest.raises(RuntimeError, match="zero-scale instruction"):
        module.patch_flashinfer_mxfp8_te_compat(logging.getLogger())
    assert kernel.read_text() == KERNEL


def test_already_imported_legacy_kernel_requires_restart(
    patches: tuple[types.ModuleType, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    module, helper, kernel = patches
    monkeypatch.setenv(ENV, "1")
    monkeypatch.setitem(
        sys.modules,
        "flashinfer.quantization.kernels.mxfp8_quantize",
        types.ModuleType("legacy"),
    )
    with pytest.raises(RuntimeError, match="before FlashInfer is imported"):
        module.patch_flashinfer_mxfp8_te_compat(logging.getLogger())
    assert helper.read_text() == HELPER and kernel.read_text() == KERNEL


def test_compatible_refit_preserves_nonzero_fp8_at_scale_zero(
    patches: tuple[types.ModuleType, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    module, _, _ = patches
    monkeypatch.setitem(sys.modules, "nemo_rl.models.generation.vllm.patches", module)
    train = load_file(
        "te_compat_train",
        ROOT / "nemo_rl/models/generation/vllm/quantization/fp8_train_utils.py",
    )
    values = torch.full((1, 32), 20, dtype=torch.float32).to(torch.float8_e4m3fn)
    scales = torch.zeros((1, 1), dtype=torch.uint8)
    monkeypatch.setenv(ENV, "1")
    got_values, got_scales = train.canonicalize_mxfp8_refit_output(
        values.shape, values, scales
    )
    assert torch.equal(got_values.view(torch.uint8), values.view(torch.uint8))
    assert torch.equal(got_scales, scales)
    monkeypatch.setenv(ENV, "0")
    _, old_scales = train.canonicalize_mxfp8_refit_output(values.shape, values, scales)
    assert old_scales.item() == 1

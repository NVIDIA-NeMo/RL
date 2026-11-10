"""Check the prefill precision patch without importing vLLM, Triton or Torch."""

import ast
import importlib.util
import logging
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


G_SOURCE = """def _causal_conv1d_fwd_kernel(acc, matrix_x, matrix_w):
    for token in range(4):
        acc += matrix_x * matrix_w  # prefill
    return acc


def _causal_conv1d_update_kernel(acc, matrix_x, matrix_w):
    acc += matrix_x * matrix_w  # decode must stay unchanged
    return acc
"""


@pytest.fixture
def patch_module() -> ModuleType:
    # Loading by path also exercises the dependency-free serving entrypoint.
    path = (
        Path(__file__).resolve().parents[4]
        / "nemo_rl/models/generation/vllm/patches.py"
    )
    spec = importlib.util.spec_from_file_location("prefill_patch_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def kernel_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, patch_module: ModuleType
) -> Path:
    # Other suites may already have imported the installed Triton kernel.
    monkeypatch.delitem(
        sys.modules, "vllm.model_executor.layers.mamba.ops.causal_conv1d", raising=False
    )
    path = tmp_path / "causal_conv1d.py"
    path.write_text(G_SOURCE)
    monkeypatch.setattr(patch_module, "_get_vllm_file", lambda relative: str(path))
    return path


def test_casts_only_prefill_products(
    patch_module: ModuleType, kernel_file: Path
) -> None:
    before_decode = ast.parse(kernel_file.read_text()).body[1]
    patch_module._patch_vllm_prefill_conv_fp32_products(logging.getLogger(__name__))
    after = ast.parse(kernel_file.read_text())
    assert ast.dump(after.body[1]) == ast.dump(before_decode)
    product = after.body[0].body[0].body[0].value
    assert isinstance(product, ast.BinOp)
    for operand, name in [(product.left, "matrix_x"), (product.right, "matrix_w")]:
        assert isinstance(operand, ast.Call)
        assert ast.unparse(operand) == f"{name}.to(tl.float32)"


def test_repeated_application_does_not_rewrite(
    patch_module: ModuleType, kernel_file: Path
) -> None:
    logger = logging.getLogger(__name__)
    patch_module._patch_vllm_prefill_conv_fp32_products(logger)
    expected = kernel_file.read_bytes()
    os.utime(kernel_file, ns=(1, 1))
    patch_module._patch_vllm_prefill_conv_fp32_products(logger)
    assert kernel_file.read_bytes() == expected
    assert kernel_file.stat().st_mtime_ns == 1


@pytest.mark.parametrize(
    "source",
    [
        G_SOURCE.replace("_causal_conv1d_fwd_kernel", "renamed_prefill_kernel"),
        G_SOURCE.replace("acc += matrix_x * matrix_w", "acc += matrix_w * matrix_x"),
        G_SOURCE.replace(
            "acc += matrix_x * matrix_w  # prefill",
            "acc += matrix_x * matrix_w\n        acc += matrix_x * matrix_w",
        ),
    ],
)
def test_unknown_source_fails_without_modification(
    patch_module: ModuleType, kernel_file: Path, source: str
) -> None:
    kernel_file.write_text(source)
    with pytest.raises(RuntimeError, match="vLLM"):
        patch_module._patch_vllm_prefill_conv_fp32_products(logging.getLogger(__name__))
    assert kernel_file.read_text() == source


@pytest.mark.parametrize("already_fixed", [False, True])
def test_loaded_kernel_must_already_have_fp32_products(
    patch_module: ModuleType,
    kernel_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    already_fixed: bool,
) -> None:
    module = ModuleType("vllm.model_executor.layers.mamba.ops.causal_conv1d")
    module._causal_conv1d_fwd_kernel = SimpleNamespace(
        src=(
            "acc += matrix_x.to(tl.float32) * matrix_w.to(tl.float32)"
            if already_fixed
            else "acc += matrix_x * matrix_w"
        )
    )
    monkeypatch.setitem(sys.modules, module.__name__, module)
    if already_fixed:
        patch_module._patch_vllm_prefill_conv_fp32_products(logging.getLogger(__name__))
        assert "matrix_x.to(tl.float32)" in kernel_file.read_text()
    else:
        with pytest.raises(RuntimeError, match="already imported"):
            patch_module._patch_vllm_prefill_conv_fp32_products(
                logging.getLogger(__name__)
            )
        assert kernel_file.read_text() == G_SOURCE


def test_generation_startup_applies_prefill_patch(
    patch_module: ModuleType, kernel_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    envs = ModuleType("vllm.envs")
    envs.VLLM_USE_RAY_V2_EXECUTOR_BACKEND = True
    logger = ModuleType("vllm.logger")
    logger.init_logger = logging.getLogger
    vllm = ModuleType("vllm")
    vllm.envs = envs
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.envs", envs)
    monkeypatch.setitem(sys.modules, "vllm.logger", logger)
    for name in vars(patch_module):
        if name.startswith("_patch_vllm_") and name != (
            "_patch_vllm_prefill_conv_fp32_products"
        ):
            monkeypatch.setattr(patch_module, name, lambda *args, **kwargs: True)
    patch_module._apply_vllm_patches(sys.executable)
    assert (
        "matrix_x.to(tl.float32) * matrix_w.to(tl.float32)" in kernel_file.read_text()
    )

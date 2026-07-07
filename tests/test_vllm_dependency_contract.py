from __future__ import annotations

import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_pyproject() -> dict[str, object]:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)


def test_vllm_extra_uses_official_024_wheels_and_runtime_dependencies() -> None:
    project = load_pyproject()
    dependencies = project["project"]["optional-dependencies"]["vllm"]

    assert any(
        "v0.24.0/vllm-0.24.0-cp38-abi3-manylinux_2_28_aarch64.whl" in item
        for item in dependencies
    )
    assert any(
        "v0.24.0/vllm-0.24.0-cp38-abi3-manylinux_2_28_x86_64.whl" in item
        for item in dependencies
    )
    assert any(item.startswith("vllm==0.24.0") for item in dependencies)
    for package in (
        "flashinfer-python==0.6.12",
        "flashinfer-cubin==0.6.12",
        "flashinfer-jit-cache==0.6.12",
        "nvidia-cutlass-dsl[cu13]==4.5.2",
    ):
        assert package in dependencies


def test_global_overrides_are_compatible_with_vllm_024() -> None:
    project = load_pyproject()
    overrides = project["tool"]["uv"]["override-dependencies"]

    assert "llguidance>=1.7.0,<1.8.0" in overrides
    assert "xgrammar>=0.2.1,<1.0.0" in overrides
    assert not any(item.startswith("llguidance>=1.3.0,<1.4.0") for item in overrides)
    assert not any(item.startswith("xgrammar==0.1.33") for item in overrides)


def test_lockfile_resolves_vllm_024_and_flashinfer_0612() -> None:
    with (ROOT / "uv.lock").open("rb") as stream:
        lockfile = tomllib.load(stream)
    versions = {
        package["name"]: package["version"]
        for package in lockfile["package"]
        if "version" in package
    }

    assert versions["vllm"] == "0.24.0"
    assert versions["flashinfer-python"] == "0.6.12"
    assert versions["flashinfer-cubin"] == "0.6.12"
    assert versions["flashinfer-jit-cache"] == "0.6.12+cu130"

"""
Custom PEP 517 build backend for TensorRT-LLM.

Two hooks implement the two-phase build:

  prepare_metadata_for_build_wheel
      Returns static dist-info fast — no GPU / compilation required.
      Called by `uv lock` and by `uv sync` before deciding whether to build.

  build_wheel
      Invokes tools/build-custom-trtllm.sh (≈60 min on GB200).
      Called by `uv sync --extra trtllm` when the wheel is not yet cached.

The package is declared as no-build-isolation-package in the root
pyproject.toml so this backend runs inside the main venv and has access
to torch, ninja, cmake, and the CUDA toolkit.

Build coordinates (git URL / ref) are read from environment variables;
see 3rdparty/trtllm-workspace/pyproject.toml for the full list.
"""
from __future__ import annotations

import glob
import os
import subprocess
import sys
import tomllib
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants — must stay in sync with 3rdparty/trtllm-workspace/pyproject.toml
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent.resolve()
_PYPROJECT = _HERE / "pyproject.toml"

with _PYPROJECT.open("rb") as _f:
    _META = tomllib.load(_f)

VERSION: str = _META["project"]["version"]
NAME: str = _META["project"]["name"].replace("-", "_")   # tensorrt_llm
DIST_NAME: str = _META["project"]["name"]                # tensorrt-llm
REQUIRES: list[str] = _META["project"].get("dependencies", [])

# Tag used in prepare_metadata — pure-Python so uv lock works on any platform.
# The real wheel built by build_wheel carries the true platform tag.
_METADATA_WHEEL_TAG = "py3-none-any"


# ---------------------------------------------------------------------------
# PEP 517 hooks
# ---------------------------------------------------------------------------

def get_requires_for_build_wheel(config_settings=None):
    """No isolated-build requirements; deps come from the main venv (no-build-isolation)."""
    return []


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    """Write minimal .dist-info without compiling anything.

    This is the fast path used by ``uv lock`` and ``uv sync``'s preflight
    metadata check.  It must not require CUDA or take significant time.
    """
    dist_info_name = f"{NAME}-{VERSION}.dist-info"
    dist_info = Path(metadata_directory) / dist_info_name
    dist_info.mkdir(parents=True, exist_ok=True)

    requires_lines = "\n".join(f"Requires-Dist: {r}" for r in REQUIRES)
    (dist_info / "METADATA").write_text(
        f"Metadata-Version: 2.1\n"
        f"Name: {DIST_NAME}\n"
        f"Version: {VERSION}\n"
        f"{requires_lines}\n",
        encoding="utf-8",
    )
    (dist_info / "WHEEL").write_text(
        "Wheel-Version: 1.0\n"
        f"Generator: trtllm-workspace-backend\n"
        "Root-Is-Purelib: false\n"
        f"Tag: {_METADATA_WHEEL_TAG}\n",
        encoding="utf-8",
    )
    return dist_info_name


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    """Build the real TRT-LLM wheel by running tools/build-custom-trtllm.sh.

    The script compiles TensorRT-LLM (≈60 min) and copies the resulting
    ``tensorrt_llm-*.whl`` into *wheel_directory*.
    """
    repo_root = (_HERE / "../..").resolve()
    script = repo_root / "tools" / "build-custom-trtllm.sh"
    if not script.exists():
        raise FileNotFoundError(f"Build script not found: {script}")

    env = os.environ.copy()

    # build-custom-trtllm.sh sets WHEEL_OUTPUT_DIR=${UV_FIND_LINKS:-/opt/trtllm_wheels}.
    # It reads UV_FIND_LINKS, not a WHEEL_OUTPUT_DIR env var.  Point UV_FIND_LINKS
    # at the directory uv passed us so the cp step lands where we expect to glob.
    env["UV_FIND_LINKS"] = str(wheel_directory)

    # Make `python3` inside the shell script resolve to the same Python that
    # uv is using for the build (the venv Python, not the system one).
    venv_bin = str(Path(sys.executable).parent)
    env["PATH"] = f"{venv_bin}:{env.get('PATH', os.defpath)}"

    git_url = env.get(
        "BUILD_CUSTOM_TRTLLM_URL",
        "https://github.com/shuyixiong/TensorRT-LLM.git",
    )
    git_ref = env.get("BUILD_CUSTOM_TRTLLM_REF", "nemorl")
    modelopt_url = env.get(
        "BUILD_CUSTOM_TRTLLM_MODELOPT_URL",
        "https://github.com/NVIDIA/TensorRT-Model-Optimizer.git",
    )
    modelopt_ref = env.get("BUILD_CUSTOM_TRTLLM_MODELOPT_REF", "0.37.0")

    subprocess.run(
        ["bash", str(script), git_url, git_ref, modelopt_url, modelopt_ref],
        check=True,
        env=env,
        cwd=str(repo_root),
    )

    wheels = sorted(
        glob.glob(os.path.join(str(wheel_directory), "tensorrt_llm-*.whl"))
    )
    if not wheels:
        raise RuntimeError(
            f"No tensorrt_llm-*.whl found in {wheel_directory} after build. "
            "Check the build-custom-trtllm.sh output above for errors."
        )
    return Path(wheels[-1]).name


def build_sdist(sdist_directory, config_settings=None):
    raise NotImplementedError(
        "TRT-LLM workspace wrapper does not support sdist builds. "
        "Use `uv sync --extra trtllm` to build the wheel."
    )

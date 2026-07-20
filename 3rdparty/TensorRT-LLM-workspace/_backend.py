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

"""Custom PEP 517 build backend for TensorRT-LLM.

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
see 3rdparty/TensorRT-LLM-workspace/pyproject.toml for the full list.
"""

from __future__ import annotations

import glob
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants — must stay in sync with 3rdparty/TensorRT-LLM-workspace/pyproject.toml
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent.resolve()
_PYPROJECT = _HERE / "pyproject.toml"

with _PYPROJECT.open("rb") as _f:
    _META = tomllib.load(_f)

VERSION: str = _META["project"]["version"]
NAME: str = _META["project"]["name"].replace("-", "_")  # tensorrt_llm
DIST_NAME: str = _META["project"]["name"]  # tensorrt-llm
REQUIRES: list[str] = _META["project"].get("dependencies", [])


def _wheel_platform_tag() -> str:
    """Return the real wheel tag for the current interpreter, e.g. cp313-cp313-linux_aarch64.

    Used only for the cache key — NOT for prepare_metadata_for_build_wheel, which
    must report py3-none-any so that uv lock succeeds on both x86_64 and aarch64.
    """
    py = f"cp{sys.version_info.major}{sys.version_info.minor}"
    machine = platform.machine()  # aarch64 | x86_64
    return f"{py}-{py}-linux_{machine}"


# py3-none-any is intentional: prepare_metadata_for_build_wheel is called by
# uv lock, which resolves for both x86_64 and aarch64. A platform-specific tag
# here would make tensorrt-llm appear incompatible with one of the two arches
# and break the lock. The real platform tag is used only inside _wheel_cache_dir.
_METADATA_WHEEL_TAG = "py3-none-any"


def _wheel_cache_dir(base: str, git_url: str, git_ref: str) -> Path:
    """Return a per-(url, ref, version, platform) cache subdirectory under *base*.

    Using a content-addressed subdir means different commits never collide,
    and a stale wheel from a previous ref is never accidentally reused.
    The cache key uses the real platform tag (not py3-none-any) so aarch64 and
    x86_64 wheels built in separate Docker runs never overwrite each other.
    """
    key = hashlib.sha256(
        f"{git_url}|{git_ref}|{VERSION}|{_wheel_platform_tag()}".encode()
    ).hexdigest()[:16]
    return Path(base) / key


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
        f"Generator: TensorRT-LLM-workspace-backend\n"
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
    git_url = env.get(
        "BUILD_CUSTOM_TRTLLM_URL",
        "https://github.com/NVIDIA/TensorRT-LLM.git",
    )
    git_ref = env.get(
        "BUILD_CUSTOM_TRTLLM_REF",
        "bf2ef86f9a2652132b11773d4041e292c553c142",  # pragma: allowlist secret
    )

    # Our own cache keyed by (git_url, git_ref, version, platform_tag).
    # uv's built-in build cache misses across venvs for no-build-isolation
    # packages because its cache key incorporates the build-environment hash.
    # We bypass that by always building into TRTLLM_WHEEL_CACHE_DIR (a stable
    # path that persists across all venv sync calls in the same Docker build),
    # then copying the result into wheel_directory for uv to consume.
    cache_base = env.get("TRTLLM_WHEEL_CACHE_DIR", "/opt/trtllm_wheels")
    cache_dir = _wheel_cache_dir(cache_base, git_url, git_ref)

    cached = sorted(glob.glob(str(cache_dir / "tensorrt_llm-*.whl")))
    if cached:
        src = cached[-1]
        dst = os.path.join(str(wheel_directory), Path(src).name)
        shutil.copy2(src, dst)
        print(f"[trtllm-backend] Cache hit — reusing wheel: {src}", flush=True)
        return Path(dst).name

    # Cache miss: build directly into cache_dir so the result is immediately
    # cached for subsequent venv syncs without a separate copy step.
    cache_dir.mkdir(parents=True, exist_ok=True)
    env["WHEEL_OUTPUT_DIR"] = str(cache_dir)

    # Make `python3` inside the shell script resolve to the same Python that
    # uv is using for the build (the venv Python, not the system one).
    venv_bin = str(Path(sys.executable).parent)
    env["PATH"] = f"{venv_bin}:{env.get('PATH', os.defpath)}"

    subprocess.run(
        ["bash", str(script), git_url, git_ref],
        check=True,
        env=env,
        cwd=str(repo_root),
    )

    wheels = sorted(glob.glob(str(cache_dir / "tensorrt_llm-*.whl")))
    if not wheels:
        raise RuntimeError(
            f"No tensorrt_llm-*.whl found in {cache_dir} after build. "
            "Check the build-custom-trtllm.sh output above for errors."
        )

    # Copy from cache_dir into wheel_directory so uv can find and install it.
    dst = os.path.join(str(wheel_directory), Path(wheels[-1]).name)
    shutil.copy2(wheels[-1], dst)
    print(f"[trtllm-backend] Wheel built and cached to: {cache_dir}", flush=True)
    return Path(dst).name


def build_sdist(sdist_directory, config_settings=None):
    raise NotImplementedError(
        "TRT-LLM workspace wrapper does not support sdist builds. "
        "Use `uv sync --extra trtllm` to build the wheel."
    )

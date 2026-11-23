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

"""Regression tests for NVIDIA-NeMo/RL#1476.

Ray's dashboard serves its static assets (main.js/main.css) with aiohttp's
``follow_symlinks=False``, so a symlinked Ray install 404s and the dashboard
never loads. The NGC PyTorch Dockerfile installs everything with
``--link-mode symlink`` and must reinstall Ray as real files; the main
Dockerfile installs the main venv with hardlinks, which are real files, so it
must keep every ``uv sync`` that can touch the shipped venv out of symlink
mode (the TRT-LLM throwaway venv is the only exception).
"""

from pathlib import Path

REPO_ROOT = Path(__file__).parents[3]
MAIN_DOCKERFILE = REPO_ROOT / "docker" / "Dockerfile"
NGC_DOCKERFILE = REPO_ROOT / "docker" / "Dockerfile.ngc_pytorch"
UV_LOCK = REPO_ROOT / "uv.lock"


def _non_comment_lines(path: Path) -> list[str]:
    return [
        line
        for line in path.read_text().splitlines()
        if not line.lstrip().startswith("#")
    ]


def _pinned_ray_version() -> str:
    """Return the Ray version pinned in uv.lock, as the Dockerfile extracts it."""
    lines = UV_LOCK.read_text().splitlines()
    name_index = lines.index('name = "ray"')
    version_line = lines[name_index + 1]
    assert version_line.startswith('version = "'), (
        f"uv.lock changed shape under the ray entry: {version_line!r}"
    )
    return version_line.split('"')[1]


def test_ngc_pytorch_reinstalls_ray_without_symlinks() -> None:
    lines = _non_comment_lines(NGC_DOCKERFILE)
    symlink_syncs = [
        index
        for index, line in enumerate(lines)
        if "uv sync" in line and "symlink" in line
    ]
    assert symlink_syncs, f"{NGC_DOCKERFILE} has no symlink-mode uv sync to fix"

    reinstall_index = next(
        (
            index
            for index, line in enumerate(lines)
            if "uv pip install" in line
            and "--link-mode=copy" in line
            and "--force-reinstall" in line
            and "--no-deps" in line
            and "ray" in line
        ),
        None,
    )
    assert reinstall_index is not None, (
        f"{NGC_DOCKERFILE} does not reinstall Ray with --link-mode=copy "
        "(NVIDIA-NeMo/RL#1476: the Ray dashboard 404s on symlinked assets)"
    )
    assert reinstall_index > symlink_syncs[-1], (
        f"{NGC_DOCKERFILE} reinstalls Ray before the last symlink-mode uv sync, "
        "so a later sync could reinstall it as symlinks again"
    )

    # The version must come from the lock, not a hardcoded literal, or the next
    # Ray bump silently reinstalls the stale version.
    reinstall_line = lines[reinstall_index]
    assert "${RAY_VERSION}" in reinstall_line, (
        f"{NGC_DOCKERFILE} hardcodes the Ray version: {reinstall_line}"
    )
    extraction = next(
        (
            line
            for line in lines
            if "RAY_VERSION=" in line and "uv.lock" in line and 'name = "ray"' in line
        ),
        None,
    )
    assert extraction is not None, (
        f"{NGC_DOCKERFILE} does not derive RAY_VERSION from uv.lock"
    )
    assert _pinned_ray_version(), "uv.lock does not pin a Ray version"


def test_main_dockerfile_keeps_shipped_venv_out_of_symlink_mode() -> None:
    lines = _non_comment_lines(MAIN_DOCKERFILE)
    symlink_syncs = [line for line in lines if "uv sync" in line and "symlink" in line]
    # The only symlink-mode sync is the TRT-LLM build's throwaway venv, which
    # is deleted in the same layer and never ships (NVIDIA-NeMo/RL#1476).
    assert symlink_syncs, f"{MAIN_DOCKERFILE} has no uv sync step at all"
    for line in symlink_syncs:
        assert "--extra trtllm" in line, (
            f"{MAIN_DOCKERFILE} installs a shipped venv in symlink mode: {line}"
        )
    assert any("UV_LINK_MODE=hardlink uv sync" in line for line in lines), (
        f"{MAIN_DOCKERFILE} does not install the main venv with hardlinks"
    )

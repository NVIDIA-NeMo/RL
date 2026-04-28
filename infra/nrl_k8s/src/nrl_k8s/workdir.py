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
"""Stage a working directory for Ray Job ``runtime_env.working_dir`` upload.

Ray's client-side packager honours ``.gitignore``, which silently drops
training jsonls under ``resources_servers/*/data/`` — we strip those files
from the staged copy so Ray uploads the data. We also skip caches/venvs/git
to keep the zip under Ray's 100 MiB dashboard cap.

Each call stages into a fresh tmpdir; the caller owns cleanup (Ray uploads
to GCS before the SDK call returns, so deletion is safe afterwards).
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

_IGNORE_PATTERNS = shutil.ignore_patterns(
    ".gitignore",
    "__pycache__",
    "*.pyc",
    "*.pyo",
    ".venv",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "*.egg-info",
)


DEFAULT_RAY_UPLOAD_PATHS = [
    "nemo_rl",
    "examples",
    "infra/examples",
    "tests/check_metrics.py",
    "tests/json_dump_tb_logs.py",
    "3rdparty/Gym-workspace/Gym/nemo_gym",
    "3rdparty/Gym-workspace/Gym/resources_servers",
    "3rdparty/Gym-workspace/Gym/responses_api_models/vllm_model",
    "3rdparty/Gym-workspace/Gym/responses_api_agents/simple_agent",
    "3rdparty/Gym-workspace/Gym/pyproject.toml",
    "3rdparty/Gym-workspace/Gym/uv.lock",
    "3rdparty/Megatron-LM-workspace/Megatron-LM/megatron",
]


# =============================================================================
# Public API
# =============================================================================


def stage_workdir(
    repo_root: Path,
    *,
    include_paths: list[str] | None = None,
    extra_files: dict[str, str] | None = None,
) -> Path:
    """Copy the requested subtrees of ``repo_root`` into a fresh tmpdir.

    Args:
        repo_root: absolute path to the NeMo-RL repo root.
        include_paths: subset of paths (relative to ``repo_root``) to stage.
            Defaults to :data:`DEFAULT_RAY_UPLOAD_PATHS`.
        extra_files: ``{relative_path: content}`` — additional files to
            create in the staged tree (e.g. the merged recipe YAML).

    Returns:
        Absolute path to the staged working_dir.
    """
    if include_paths is None:
        include_paths = DEFAULT_RAY_UPLOAD_PATHS

    dest = Path(tempfile.mkdtemp(prefix="nrl-k8s-workdir-"))

    for rel in include_paths:
        src = (repo_root / rel).resolve()
        if not src.exists():
            # Missing optional paths are OK (e.g. uv.lock may not exist).
            continue
        tgt = dest / rel
        tgt.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, tgt, ignore=_IGNORE_PATTERNS)
        else:
            shutil.copy2(src, tgt)

    for rel, content in (extra_files or {}).items():
        p = dest / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)

    return dest


__all__ = ["DEFAULT_RAY_UPLOAD_PATHS", "stage_workdir"]

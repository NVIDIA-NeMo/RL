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

import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).parents[3]
SCRIPT = REPO_ROOT / "examples" / "nemo_gym" / "prefetch_venvs.py"
DOCKERFILE = REPO_ROOT / "docker" / "Dockerfile"
# The Dockerfile's WORKDIR, which relative COPY destinations resolve against.
CONTAINER_REPO_ROOT = PurePosixPath("/opt/nemo-rl")
# Copied before the prefetch, but hold no NeMo RL module and are large.
SKIPPED_COPY_SOURCES = ("3rdparty", "research")

# What the prefetch does before it hands off to NeMo Gym, run from the copied tree.
PREFETCH_CHECK = """
import runpy, sys
from pathlib import Path
from omegaconf import OmegaConf

script = runpy.run_path("examples/nemo_gym/prefetch_venvs.py", run_name="prefetch")
import nemo_rl
assert Path(nemo_rl.__file__).is_relative_to(Path.cwd()), nemo_rl.__file__

def container_paths(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from container_paths(item)
    elif isinstance(value, list):
        for item in value:
            yield from container_paths(item)
    elif isinstance(value, str) and value.startswith("/opt/nemo-rl/"):
        yield value

script["register_omegaconf_resolvers"]()
for config_path in sys.argv[1:]:
    config = OmegaConf.to_container(script["load_config"](config_path), resolve=True)
    section = script["build_dry_run_config"](config["env"]["nemo_gym"], None, None)
    for path in container_paths(section):
        assert Path(path.removeprefix("/opt/nemo-rl/")).is_file(), (config_path, path)
"""


def _load_script():
    spec = importlib.util.spec_from_file_location("gym_prefetch_venvs", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _copy_what_the_image_has_at_the_prefetch(tree: Path) -> None:
    """Copy every NeMo RL file the Dockerfile copies in before the Gym prefetch."""
    text = DOCKERFILE.read_text()
    (prefetch,) = re.finditer(r"prefetch_venvs\.py \$NEMO_GYM_PREFETCH_CONFIGS", text)
    instructions = re.sub(r"\\\n\s*", " ", text[: prefetch.start()]).splitlines()
    for instruction in instructions:
        tokens = instruction.split()
        if tokens[:1] != ["COPY"] or "--from=nemo-rl" not in tokens:
            continue
        *sources, destination = [t for t in tokens[1:] if not t.startswith("--")]
        destination_path = CONTAINER_REPO_ROOT / destination
        if not destination_path.is_relative_to(CONTAINER_REPO_ROOT):
            continue
        target = tree / destination_path.relative_to(CONTAINER_REPO_ROOT)
        for source in sources:
            if source.split("/")[0] in SKIPPED_COPY_SOURCES:
                continue
            if (REPO_ROOT / source).is_dir():
                shutil.copytree(REPO_ROOT / source, target, dirs_exist_ok=True)
            elif len(sources) > 1 or destination.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
                shutil.copy(REPO_ROOT / source, target)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(REPO_ROOT / source, target)


def test_prefetch_runs_with_only_the_files_the_image_has(tmp_path: Path) -> None:
    _copy_what_the_image_has_at_the_prefetch(tmp_path)
    configs = sorted(
        str(path.relative_to(tmp_path))
        for path in (tmp_path / "examples" / "nemo_gym").glob("prefetch_*.yaml")
    )
    assert configs

    result = subprocess.run(
        [sys.executable, "-c", PREFETCH_CHECK, *configs],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_prefetch_exits_nonzero_and_reports_every_failed_config(tmp_path: Path) -> None:
    _copy_what_the_image_has_at_the_prefetch(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            "examples/nemo_gym/prefetch_venvs.py",
            "missing_a.yaml",
            "missing_b.yaml",
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1, result.stdout + result.stderr
    assert "Failed: 2" in result.stdout, result.stdout


def test_build_dry_run_config() -> None:
    section = {
        "config_paths": ["a.yaml"],
        "effort_levels": [1, 2],
        "pad_dynamic_image_shapes": True,
        "uv_venv_dir": "/from/recipe",
    }

    config = _load_script().build_dry_run_config(
        section, uv_cache_dir="/cache/uv", uv_venv_dir="/opt/gym_venvs"
    )

    assert config == {
        "config_paths": ["a.yaml"],
        "dry_run": True,
        "policy_model_name": "dummy-model",
        "policy_api_key": "dummy_key",
        "policy_base_url": ["http://localhost:8000"],
        "uv_cache_dir": "/cache/uv",
        "uv_venv_dir": "/from/recipe",
    }
    assert "dry_run" not in section


def test_dockerfile_prefetches_gym_venvs_before_the_source_copy() -> None:
    lines = (REPO_ROOT / "docker" / "Dockerfile").read_text().splitlines()

    def line_of(text: str) -> int:
        (index,) = [i for i, line in enumerate(lines) if text in line]
        return index

    copy_config = line_of("nemo_rl/utils/config.py /opt/nemo-rl/nemo_rl/utils/")
    copy_examples = line_of("COPY --from=nemo-rl examples /opt/nemo-rl/examples")
    prefetch = line_of("examples/nemo_gym/prefetch_venvs.py $NEMO_GYM_PREFETCH_CONFIGS")
    source_copy = line_of("COPY --from=nemo-rl --exclude=pyproject.toml")
    assert max(copy_config, copy_examples) < prefetch < source_copy

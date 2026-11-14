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

import ast
import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).parents[3]
SCRIPT = REPO_ROOT / "examples" / "nemo_gym" / "prefetch_venvs.py"
# The only NeMo RL files docker/Dockerfile copies in before it runs the script.
AVAILABLE_NEMO_RL_MODULES = {"nemo_rl.utils.config"}


def _load_script():
    spec = importlib.util.spec_from_file_location("gym_prefetch_venvs", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_imports_only_the_nemo_rl_files_the_image_has() -> None:
    imported = {
        node.module
        for node in ast.walk(ast.parse(SCRIPT.read_text()))
        if isinstance(node, ast.ImportFrom)
        and (node.module or "").startswith("nemo_rl")
    }
    assert imported == AVAILABLE_NEMO_RL_MODULES


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

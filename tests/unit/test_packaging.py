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
"""Check wheel contents and imports independently of the source checkout."""

import os
import subprocess
import sys
import zipfile
from pathlib import Path


def test_installed_wheel_contains_runtime_packages(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    wheels = tmp_path / "wheels"
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(wheels), str(root)],
        check=True,
    )
    (wheel,) = wheels.glob("*.whl")
    expected = {
        str(path.relative_to(root))
        for path in (root / "nemo_rl").rglob("*")
        if path.suffix in {".py", ".md", ".sh"}
    }
    with zipfile.ZipFile(wheel) as archive:
        assert expected <= set(archive.namelist())

    installed = tmp_path / "installed"
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            sys.executable,
            "--no-deps",
            "--target",
            str(installed),
            str(wheel),
        ],
        check=True,
    )
    env = dict(os.environ, NEMO_RL_PY_EXECUTABLES_SYSTEM="1")
    # Isolated mode excludes the checkout and PYTHONPATH from import resolution.
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            """
import importlib.util
import sys
from pathlib import Path

installed = Path(sys.argv[1])
sys.path.insert(0, str(installed))
import nemo_rl
from nemo_rl.distributed.actor_environments import ACTOR_ENVIRONMENTS
from nemo_rl.distributed.ray_actor_environment_registry import ACTOR_ENVIRONMENT_REGISTRY

assert Path(nemo_rl.__file__).is_relative_to(installed)
assert not (installed / "pyproject.toml").exists()
assert ACTOR_ENVIRONMENT_REGISTRY.keys() == ACTOR_ENVIRONMENTS.keys()
assert set(ACTOR_ENVIRONMENT_REGISTRY.values()) == {sys.executable}
for module in ("nemo_rl.algorithms.grpo", "nemo_rl.models.policy.lm_policy"):
    spec = importlib.util.find_spec(module)
    assert spec is not None and Path(spec.origin).is_relative_to(installed), module
""",
            str(installed),
        ],
        cwd=tmp_path,
        env=env,
        check=True,
    )

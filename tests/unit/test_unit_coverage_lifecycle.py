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

"""Exercise the unit runner's coverage lifecycle without Ray or GPUs."""

import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(
    "mode", ["coverage", "disabled", "no_plugin", "no_report", "fail_under"]
)
def test_session_coverage_after_collection(tmp_path: Path, mode: str) -> None:
    pytest.importorskip("pytest_cov")
    # Load the actual hooks, isolating only Ray/GPU setup from this subprocess.
    tree = ast.parse(Path(__file__).with_name("conftest.py").read_text())
    hooks = ast.Module(
        body=[
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name in {"session_data", "pytest_sessionfinish"}
        ],
        type_ignores=[],
    )
    (tmp_path / "conftest.py").write_text(
        "import json, os, sys, types\n"
        "from io import StringIO\n"
        "import pytest\n"
        "UnitTestData = dict\n"
        "UNIT_RESULTS_FILE = 'results.json'\n"
        "UNIT_RESULTS_FILE_DATED = 'dated/results.json'\n"
        "class Monitor:\n"
        "    def __init__(self, **kwargs): pass\n"
        "    def _collect_gpu_sku(self): return {}\n"
        "logger = types.ModuleType('nemo_rl.utils.logger')\n"
        "logger.RayGpuMonitorLogger = Monitor\n"
        "sys.modules[logger.__name__] = logger\n"
        "@pytest.fixture(scope='session')\n"
        "def init_ray_cluster(): yield\n"
        "@pytest.fixture(scope='session')\n"
        "def _unit_test_data(request):\n"
        "    request.config._unit_test_data = {'coverage': '[n/a]'}\n"
        "    return request.config._unit_test_data\n" + ast.unparse(hooks) + "\n"
    )
    package = tmp_path / "nemo_rl"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "sample.py").write_text(
        "def compute(child=False):\n"
        "    if child:\n"
        "        return 2\n"
        "    return 1\n"
        "\n"
        "def unused():\n"
        "    return 3\n"
    )
    (tmp_path / "test_sample.py").write_text(
        "import multiprocessing\n"
        "from nemo_rl.sample import compute\n"
        "def test_parent_and_child():\n"
        "    assert compute() == 1\n"
        "    child = multiprocessing.get_context('spawn').Process(\n"
        "        target=compute, args=(True,))\n"
        "    child.start()\n"
        "    child.join(30)\n"
        "    assert child.exitcode == 0\n"
    )
    (tmp_path / "pyproject.toml").write_text(
        '[tool.coverage.run]\nconcurrency = ["thread", "multiprocessing"]\n'
        'source_pkgs = ["nemo_rl"]\n'
        '[tool.coverage.paths]\nsource = ["nemo_rl/", "/opt/nemo-rl/nemo_rl/"]\n'
    )
    command = [sys.executable, "-m", "pytest", "-q"]
    if mode != "no_plugin":
        command += [
            "-p",
            "pytest_cov",
            "--cov=nemo_rl",
            "--cov-config=pyproject.toml",
            "--cov-report=" if mode == "no_report" else "--cov-report=json",
        ]
    if mode == "disabled":
        command.append("--no-cov")
    elif mode == "fail_under":
        command.append("--cov-fail-under=100")
    env = {
        key: os.environ[key]
        for key in ("PATH", "HOME", "TMPDIR", "TEMP", "TMP", "SYSTEMROOT", "WINDIR")
        if key in os.environ
    }
    env.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTHONPATH=str(tmp_path))
    result = subprocess.run(
        command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60
    )
    output = result.stdout + result.stderr
    assert result.returncode == (1 if mode == "fail_under" else 0), output
    assert "DataError" not in output
    assert "1 passed" in output
    recorded = json.loads((tmp_path / "results.json").read_text())
    assert recorded["exit_status"] == result.returncode
    if mode in {"coverage", "fail_under"}:
        report = json.loads((tmp_path / "coverage.json").read_text())
        assert recorded["coverage"] == pytest.approx(
            report["totals"]["percent_covered"]
        )
        sample = report["files"]["nemo_rl/sample.py"]
        assert {3, 4} <= set(sample["executed_lines"])
        assert 7 in sample["missing_lines"]
    else:
        assert recorded["coverage"] == "[n/a]"


@pytest.mark.parametrize("total", [None, 0.0, 83.5])
def test_session_finish_preserves_coverage_total(
    tmp_path: Path, total: float | None
) -> None:
    tree = ast.parse(Path(__file__).with_name("conftest.py").read_text())
    hook = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "pytest_sessionfinish"
    )
    result_path = tmp_path / "results.json"
    namespace = {
        "json": json,
        "os": os,
        "UNIT_RESULTS_FILE": str(result_path),
        "UNIT_RESULTS_FILE_DATED": str(tmp_path / "dated/results.json"),
    }
    exec(
        compile(ast.Module(body=[hook], type_ignores=[]), "conftest.py", "exec"),
        namespace,
    )
    plugin = SimpleNamespace(cov_total=total)
    config = SimpleNamespace(
        _unit_test_data={"coverage": "[n/a]"},
        pluginmanager=SimpleNamespace(getplugin=lambda name: plugin),
    )
    namespace["pytest_sessionfinish"](SimpleNamespace(config=config), 0)
    recorded = json.loads(result_path.read_text())
    assert recorded["exit_status"] == 0
    assert recorded["coverage"] == ("[n/a]" if total is None else total)

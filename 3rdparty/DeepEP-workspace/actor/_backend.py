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

"""Expose static actor DeepEP metadata; compile the pinned upstream source."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

_HERE = Path.cwd()  # PEP 517 invokes hooks from the project source directory.
_META = tomllib.loads((_HERE / "pyproject.toml").read_text())


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    project = _META["project"]
    dirname = f"deep_ep-{project['version']}.dist-info"
    output = Path(metadata_directory) / dirname
    output.mkdir(parents=True, exist_ok=True)
    dependencies = "".join(f"Requires-Dist: {dep}\n" for dep in project["dependencies"])
    (output / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: deep-ep\nVersion: {project['version']}\n"
        f"Requires-Python: {project['requires-python']}\n{dependencies}"
    )
    return dirname


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    source = _META["tool"]["deepep"]
    output = Path(wheel_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="deepep-build-") as directory:
        checkout = Path(directory)
        for args in (
            ["init", "."],
            ["remote", "add", "origin", source["url"]],
            ["fetch", "--depth=1", "origin", source["ref"]],
            ["checkout", "--detach", "FETCH_HEAD"],
            ["config", "core.abbrev", "7"],
        ):
            subprocess.run(["git", *args], cwd=checkout, check=True)
        env = os.environ.copy()
        env["HYBRID_EP_MULTINODE"] = "1"
        env["USE_NIXL"] = "0"
        subprocess.run(
            [sys.executable, "setup.py", "bdist_wheel", "--dist-dir", str(output)],
            cwd=checkout,
            env=env,
            check=True,
        )
    wheels = list(output.glob(f"deep_ep-{_META['project']['version']}-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"Expected one pinned DeepEP wheel, found {wheels}")
    return wheels[0].name

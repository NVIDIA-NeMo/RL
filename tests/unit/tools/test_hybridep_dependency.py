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

import tomllib
from pathlib import Path
from typing import Any

HYBRIDEP_COMMIT = "f725d29699f5bda9ba789456bb9579af69844685"
PREVIOUS_X86_COMMIT = "29d31c095796f3c8ece47ee9cdcc167051bbeed9"
PREVIOUS_ARM_COMMIT = "a48493600c4886c1b297aaa78db0e1ebc2d8dd6c"
HYBRIDEP_VERSION = "1.2.1+f725d29"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _deep_ep_dependencies(project: dict[str, Any]) -> list[str]:
    optional_dependencies = project["project"]["optional-dependencies"]
    dependency_groups = (
        optional_dependencies["automodel"],
        optional_dependencies["vllm"],
        optional_dependencies["mcore"],
        project["tool"]["uv"]["override-dependencies"],
    )
    return [
        dependency
        for group in dependency_groups
        for dependency in group
        if dependency.startswith("deep_ep @ git+")
    ]


def test_all_platform_deep_ep_dependencies_use_same_hybridep_commit() -> None:
    pyproject_path = _project_root() / "pyproject.toml"
    project = tomllib.loads(pyproject_path.read_text())
    dependencies = _deep_ep_dependencies(project)
    x86_dependencies = [
        dependency
        for dependency in dependencies
        if "platform_machine == 'x86_64'" in dependency
    ]
    arm_dependencies = [
        dependency
        for dependency in dependencies
        if "platform_machine == 'aarch64'" in dependency
    ]

    assert len(x86_dependencies) == 4
    assert all(HYBRIDEP_COMMIT in dependency for dependency in x86_dependencies)
    assert len(arm_dependencies) == 4
    assert all(HYBRIDEP_COMMIT in dependency for dependency in arm_dependencies)


def test_deep_ep_dependency_metadata_matches_hybridep() -> None:
    pyproject_path = _project_root() / "pyproject.toml"
    project = tomllib.loads(pyproject_path.read_text())
    metadata = [
        entry
        for entry in project["tool"]["uv"]["dependency-metadata"]
        if entry["name"] == "deep_ep"
    ]

    assert len(metadata) == 1
    assert metadata[0]["version"] == HYBRIDEP_VERSION


def test_lock_uses_same_hybridep_commit_for_x86_and_arm() -> None:
    lock_path = _project_root() / "uv.lock"
    lock = tomllib.loads(lock_path.read_text())
    packages = [package for package in lock["package"] if package["name"] == "deep-ep"]
    assert len(packages) == 1
    package = packages[0]
    assert package["version"] == HYBRIDEP_VERSION
    assert f"rev={HYBRIDEP_COMMIT}" in package["source"]["git"]
    assert f"#{HYBRIDEP_COMMIT}" in package["source"]["git"]
    assert "resolution-markers" not in package
    lock_text = lock_path.read_text()
    assert PREVIOUS_X86_COMMIT not in lock_text
    assert PREVIOUS_ARM_COMMIT not in lock_text

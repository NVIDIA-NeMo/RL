# Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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

# This module tests things outside of any package (e.g., things in the root __init__.py)

import os
from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]


def test_usage_stats_disabled_by_default():
    assert os.environ["RAY_USAGE_STATS_ENABLED"] == "0", (
        "Our dockerfile, slurm submission script and default environment setting when importing nemo rl should all disable usage stats collection. This failing is not expected."
    )


def test_usage_stats_disabled_in_tests():
    assert os.environ["RAY_USAGE_STATS_ENABLED"] == "0", (
        "Our dockerfile, slurm submission script and default environment setting when importing nemo rl should all disable usage stats collection. This failing is not expected."
    )


def test_ci_image_runtime_dependencies_are_readable_by_non_root_users():
    dockerfile = (REPO_ROOT / "docker" / "Dockerfile").read_text()
    workflow = (REPO_ROOT / ".github" / "workflows" / "cicd-main.yml").read_text()

    assert dockerfile.index("ARG UV_VERSION=0.11.28") < dockerfile.index(
        "FROM scratch AS nemo-rl"
    )
    assert "ARG UV_INSTALLER_SHA256=" in dockerfile
    assert (
        "-o /tmp/uv-install.sh https://astral.sh/uv/${UV_VERSION}/install.sh"
        in dockerfile
    )
    assert (
        '"${UV_INSTALLER_SHA256}  /tmp/uv-install.sh" | sha256sum --check --strict -'
        in dockerfile
    )
    assert "XDG_BIN_HOME=/usr/local/bin sh /tmp/uv-install.sh" in dockerfile
    assert "ENV UV_CACHE_DIR=/opt/uv/cache" in dockerfile
    assert "ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python" in dockerfile
    assert "/root/.local/bin" not in dockerfile
    assert "/root/.cache/uv" not in dockerfile
    assert "${CID}:/opt/uv/cache/." in workflow
    assert "${CID}:/root/.cache/uv/." not in workflow


def test_local_container_test_wrappers_use_the_calling_user():
    for relative_path in (
        "tests/run_unit_in_docker.sh",
        "tests/run_functional_in_docker.sh",
    ):
        script = (REPO_ROOT / relative_path).read_text()

        assert '--user "$(id -u):$(id -g)"' in script
        assert (
            '--tmpfs "/home/nemo-rl:rw,exec,nosuid,nodev,mode=0700,uid=$(id -u),gid=$(id -g)"'
            in script
        )
        assert "CONTAINER_HOME" not in script
        assert "UV_CACHE_DIR=/home/nemo-rl/.cache/uv" in script
        assert "uv run --no-sync" in script
        assert "-u root" not in script

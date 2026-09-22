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

from pathlib import Path

REPO_ROOT = Path(__file__).parents[3]
DOCKERFILES = (
    (REPO_ROOT / "docker/Dockerfile", True),
    (REPO_ROOT / "docker/Dockerfile.ngc_pytorch", False),
)


def test_docker_images_build_deepep_with_multinode_hybridep() -> None:
    for dockerfile, uses_uv_cache_seed in DOCKERFILES:
        lines = dockerfile.read_text().splitlines()
        setting = "ENV HYBRID_EP_MULTINODE=1"
        nvml_link_dependency_install = (
            "apt-get install -y --no-install-recommends libnvidia-ml-dev"
        )
        nvml_link_dependency_purge = "apt-get purge -y libnvidia-ml-dev"
        nvml_link_dependencies_purge = "apt-get autoremove -y"

        assert setting in lines, f"{dockerfile} does not enable multi-node HybridEP"
        assert nvml_link_dependency_install in lines, (
            f"{dockerfile} does not install the NVML link dependency"
        )
        assert nvml_link_dependency_purge in lines, (
            f"{dockerfile} does not remove the NVML link dependency after the build"
        )

        first_sync_index = next(
            (
                index
                for index, line in enumerate(lines)
                if not line.lstrip().startswith("#") and "uv sync" in line
            ),
            None,
        )
        assert first_sync_index is not None, f"{dockerfile} has no uv sync step"

        nvml_link_dependency_purge_index = lines.index(nvml_link_dependency_purge)
        nvml_link_dependencies_purge_index = next(
            (
                index
                for index, line in enumerate(
                    lines[nvml_link_dependency_purge_index + 1 :],
                    nvml_link_dependency_purge_index + 1,
                )
                if line == nvml_link_dependencies_purge
            ),
            None,
        )
        assert nvml_link_dependencies_purge_index is not None, (
            f"{dockerfile} retains the NVML link dependency's driver-side packages"
        )

        if uses_uv_cache_seed:
            cache_key_line = next(
                (
                    line
                    for line in lines
                    if line.startswith("CACHE_KEY=")
                    and "BASE_IMAGE" in line
                    and "UV_VERSION" in line
                ),
                None,
            )
            assert cache_key_line is not None, (
                f"{dockerfile} does not define the uv seed cache key"
            )
            assert "HYBRID_EP_MULTINODE" in cache_key_line, (
                f"{dockerfile} can reuse a single-node DeepEP wheel"
            )
            actor_prefetch_end = "done < /opt/actor_venvs.tsv"
            assert actor_prefetch_end in lines, (
                f"{dockerfile} does not prefetch actor dependencies"
            )
            assert lines.index(actor_prefetch_end) < nvml_link_dependency_purge_index, (
                f"{dockerfile} removes the NVML link dependency before actor venv sync"
            )

        assert (
            lines.index(setting)
            < lines.index(nvml_link_dependency_install)
            < first_sync_index
            < nvml_link_dependency_purge_index
            < nvml_link_dependencies_purge_index
        ), f"{dockerfile} does not prepare multi-node DeepEP before dependency sync"

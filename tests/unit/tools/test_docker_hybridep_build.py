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
    REPO_ROOT / "docker/Dockerfile",
    REPO_ROOT / "docker/Dockerfile.ngc_pytorch",
)


def test_docker_images_build_deepep_with_multinode_hybridep() -> None:
    for dockerfile in DOCKERFILES:
        lines = dockerfile.read_text().splitlines()
        setting = "ENV HYBRID_EP_MULTINODE=1"
        cache_clean = "uv cache clean deep-ep"
        first_sync_index = next(
            index
            for index, line in enumerate(lines)
            if not line.lstrip().startswith("#") and "uv sync" in line
        )

        assert setting in lines, f"{dockerfile} does not enable multi-node HybridEP"
        assert cache_clean in lines, (
            f"{dockerfile} can reuse a single-node DeepEP wheel"
        )
        assert lines.index(setting) < lines.index(cache_clean) < first_sync_index, (
            f"{dockerfile} does not prepare multi-node DeepEP before dependency sync"
        )

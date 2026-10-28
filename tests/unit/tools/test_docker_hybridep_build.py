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

import re
from pathlib import Path

REPO_ROOT = Path(__file__).parents[3]
DOCKERFILES = (
    (REPO_ROOT / "docker/Dockerfile", True),
    (REPO_ROOT / "docker/Dockerfile.ngc_pytorch", False),
)


def test_docker_images_build_deepep_with_multinode_hybridep() -> None:
    for dockerfile, uses_uv_cache_seed in DOCKERFILES:
        lines = dockerfile.read_text().splitlines()
        # docker/Dockerfile derives this from the ARG (not a literal "=1") so a
        # non-default --build-arg HYBRID_EP_MULTINODE=0 is reflected
        # consistently in its cache mount IDs and in uv-cache-export's --
        # see the comment above that ENV line for why a literal there is a
        # bug. Dockerfile.ngc_pytorch has no such cache-mount-id mechanism
        # (uses_uv_cache_seed=False), so it keeps the plain literal.
        setting = (
            "ENV HYBRID_EP_MULTINODE=${HYBRID_EP_MULTINODE}"
            if uses_uv_cache_seed
            else "ENV HYBRID_EP_MULTINODE=1"
        )
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
            # The uv download cache lives on a BuildKit cache mount (never part of
            # any image layer); its `id=` is the cache's identity, so it must
            # incorporate BASE_IMAGE/UV_VERSION/HYBRID_EP_MULTINODE the same way
            # the old CACHE_KEY computation did, or a single-node-built DeepEP
            # wheel could get silently reused for a multi-node build.
            cache_mount_lines = [
                line
                for line in lines
                if "--mount=type=cache" in line
                and "id=uv-cache-" in line
                and "${BASE_IMAGE}" in line
                and "${UV_VERSION}" in line
            ]
            assert len(cache_mount_lines) == sum(
                "--mount=type=cache" in line and "id=uv-cache-" in line
                for line in lines
            ), f"{dockerfile} has a uv cache mount missing BASE_IMAGE or UV_VERSION"
            assert cache_mount_lines, (
                f"{dockerfile} does not define a uv cache mount keyed on the base image and uv version"
            )
            for cache_mount_line in cache_mount_lines:
                assert "${HYBRID_EP_MULTINODE}" in cache_mount_line, (
                    f"{dockerfile} can reuse a single-node DeepEP wheel: {cache_mount_line}"
                )
            # Every uv-writing RUN (and uv-cache-export) must key its mount with
            # the SAME id string, or a copy-paste drift in just one of them
            # silently creates a disconnected cache for that RUN -- see the
            # ENV/ARG divergence this same identity bug took the form of above.
            # Compare just the `id=...` segment: the surrounding line differs
            # (RUN vs. a \-continuation, trailing bash flags) even when the
            # cache identity itself is consistent.
            mount_ids = {
                segment.split(",", 1)[0]
                for line in cache_mount_lines
                for segment in line.split("id=uv-cache-", 1)[1:]
            }
            assert len(mount_ids) == 1, (
                f"{dockerfile} has mismatched uv cache mount IDs across RUN blocks: "
                f"{sorted(mount_ids)}"
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


def test_docker_image_keeps_uv_cache_by_default() -> None:
    dockerfile = REPO_ROOT / "docker/Dockerfile"
    cache_mount_script = REPO_ROOT / "docker/uv_cache_mount.sh"
    lines = [line.strip() for line in dockerfile.read_text().splitlines()]

    declarations = [line for line in lines if line.startswith("ARG STRIP_UV_CACHE")]
    assert declarations, f"{dockerfile} does not declare STRIP_UV_CACHE"
    for declaration in declarations:
        # Only the bare declaration (unset = empty = falsy) counts as "off by
        # default". "=0" looks like "off" but isn't: every check in this repo
        # (and in uv_cache_mount.sh) tests `[ -n "${STRIP_UV_CACHE:-}" ]`, and
        # "0" is a non-empty string, so STRIP_UV_CACHE=0 would strip the cache.
        assert declaration == "ARG STRIP_UV_CACHE", (
            f"{dockerfile} strips the uv cache by default: {declaration}"
        )

    callers = [
        str(path.relative_to(REPO_ROOT))
        for path in (REPO_ROOT / ".github").rglob("*")
        if path.is_file() and "STRIP_UV_CACHE" in path.read_text(errors="ignore")
    ]
    assert not callers, (
        f"CI images must keep the uv cache: {callers} set STRIP_UV_CACHE"
    )

    # The materialize/sync-back/strip dance lives in uv_cache_mount.sh, not in
    # the Dockerfile, so the deletion guard must be checked there.
    cache_mount_lines = [
        line.strip() for line in cache_mount_script.read_text().splitlines()
    ]
    deletions = [
        line for line in lines + cache_mount_lines if 'rm -rf "${UV_CACHE_DIR}"' in line
    ]
    assert deletions, f"{cache_mount_script} does not delete the uv cache anywhere"
    for line in deletions:
        assert "STRIP_UV_CACHE" in line, (
            f"deletes the uv cache without the opt-in: {line}"
        )


def test_uv_cache_mount_id_matches_build_container_workflow() -> None:
    dockerfile_text = (REPO_ROOT / "docker/Dockerfile").read_text()
    workflow_text = (REPO_ROOT / ".github/workflows/_build_container.yml").read_text()

    ids = set(re.findall(r"id=uv-cache-([^,]+),", dockerfile_text))
    assert len(ids) == 1, ids
    # the workflow takes arch from PLATFORM, not a Dockerfile-resolved ARG
    df_vars = set(re.findall(r"\$\{(\w+)\}", ids.pop())) - {"TARGETARCH"}

    step = workflow_text.split("id: uv_cache\n", 1)[1].split("- name:", 1)[0]
    wf_vars = set(re.findall(r"=\$\(resolve_arg (\w+)\)", step))

    assert df_vars == wf_vars, (df_vars, wf_vars)

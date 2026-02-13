# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""Setup for pip package."""

import os
import subprocess
import sys
import tomllib

import setuptools
from setuptools import Extension

###############################################################################
#                             Extension Making                                #
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# --- Configuration Start ---
# These will be populated conditionally or with defaults
final_packages = []
final_package_dir = {}
final_ext_modules = []

# --- megatron.core conditional section ---
# Directory for the megatron.core Python package source
megatron_core_python_package_source_dir = "Megatron-LM/megatron/core"
megatron_core_package_name = "megatron.core"

# Path for the C++ extension's source file, relative to setup.py
megatron_core_cpp_extension_source_file = "megatron/core/datasets/helpers.cpp"

# Cached dependencies: default + dev from pyproject.toml
# VCS dependencies use full "pkg @ git+URL@rev" format matching pyproject.toml [tool.uv.sources]
CACHED_DEPENDENCIES = [
    # Default dependencies from pyproject.toml
    "torch",
    "numpy",
    "packaging>=24.2",
    # Dev dependencies from pyproject.toml
    "nvidia-modelopt[torch]; sys_platform != 'darwin'",
    "transformer-engine[pytorch,core_cu13]>=2.9.0a0,<2.12.0",
    "nvidia-resiliency-ext",
    "tqdm",
    "einops~=0.8",
    "tensorstore~=0.1,!=0.1.46,!=0.1.72",
    "nvtx~=0.2",
    "multi-storage-client~=0.27",
    "opentelemetry-api~=1.33.1",
    "mamba-ssm~=2.2",
    "causal-conv1d~=1.5",
    "flash-linear-attention~=0.3.2",
    "nv-grouped-gemm~=1.1",
    "megatron-energon[av_decode]~=6.0",
    "av",
    "flashinfer-python~=0.5.0",
    "wget",
    "onnxscript",
    # VCS dependency - must match pyproject.toml [tool.uv.sources]
    "emerging_optimizers @ git+https://github.com/NVIDIA-NeMo/Emerging-Optimizers.git@v0.1.0",
    "datasets",
    "fastapi~=0.50",
]


def build_vcs_dependency(pkg_name: str, source_info: dict) -> str:
    """Build a PEP 440 VCS dependency string from pyproject.toml [tool.uv.sources] entry."""
    git_url = source_info.get("git")
    rev = source_info.get("rev")
    if not git_url:
        raise ValueError(f"No git URL found for VCS dependency: {pkg_name}")
    if not rev:
        raise ValueError(f"No rev/commit found for VCS dependency: {pkg_name}")
    return f"{pkg_name} @ git+{git_url}@{rev}"


# Read pyproject.toml to validate dependencies
pyproject_path = os.path.join("Megatron-LM", "pyproject.toml")

if os.path.exists(megatron_core_python_package_source_dir):
    if not os.path.exists(pyproject_path):
        raise FileNotFoundError(
            f"[megatron-core][setup] {pyproject_path} not found; skipping dependency consistency check."
        )

    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    # Extract [tool.uv.sources] for VCS dependencies
    uv_sources = data.get("tool", {}).get("uv", {}).get("sources", {})

    # Combine default dependencies + dev optional-dependencies
    project = data["project"]
    default_deps = project.get("dependencies", [])
    optional_deps = project.get("optional-dependencies", {})
    dev_deps = optional_deps.get("dev", [])

    submodule_deps = set(str(d).strip() for d in default_deps + dev_deps)

    # Build expected dependencies, converting any in [tool.uv.sources] to full VCS strings
    submodule_deps_with_vcs = set()
    for dep in submodule_deps:
        if dep in uv_sources:
            # Replace with full VCS string constructed from [tool.uv.sources]
            vcs_dep = build_vcs_dependency(dep, uv_sources[dep])
            submodule_deps_with_vcs.add(vcs_dep)
        else:
            submodule_deps_with_vcs.add(dep)

    cached_deps_set = set(CACHED_DEPENDENCIES)

    missing_in_cached = submodule_deps_with_vcs - cached_deps_set
    extra_in_cached = cached_deps_set - submodule_deps_with_vcs

    if missing_in_cached or extra_in_cached:
        print(
            "[megatron-core][setup] Dependency mismatch between Megatron-LM-workspace/Megatron-LM/pyproject.toml vs Megatron-LM-workspace/setup.py::CACHED_DEPENDENCIES.",
            file=sys.stderr,
        )
        if missing_in_cached:
            print(
                "  - Present in Megatron-LM/pyproject.toml (default+dev) but missing from CACHED_DEPENDENCIES:",
                file=sys.stderr,
            )
            for dep in sorted(missing_in_cached):
                print(f"    * {dep}", file=sys.stderr)
        if extra_in_cached:
            print(
                "  - Present in CACHED_DEPENDENCIES but not in Megatron-LM/pyproject.toml (default+dev):",
                file=sys.stderr,
            )
            for dep in sorted(extra_in_cached):
                print(f"    * {dep}", file=sys.stderr)
        print(
            "  Please update CACHED_DEPENDENCIES or the submodule pyproject to keep them in sync.",
            file=sys.stderr,
        )
        sys.exit(1)
    else:
        print(
            "[megatron-core][setup] Dependency sets are consistent with the submodule pyproject (default+dev).",
            file=sys.stderr,
        )

# Check if the main directory for the megatron.core Python package exists
if os.path.exists(megatron_core_python_package_source_dir):
    # Add Python package 'megatron.core'
    final_packages.append(megatron_core_package_name)
    final_package_dir[megatron_core_package_name] = (
        megatron_core_python_package_source_dir
    )

    # If the Python package is being added, then check if its C++ extension can also be added
    # This requires the specific C++ source file to exist
    if os.path.exists(megatron_core_cpp_extension_source_file):
        megatron_extension = Extension(
            "megatron.core.datasets.helpers_cpp",  # Name of the extension
            sources=[megatron_core_cpp_extension_source_file],  # Path to C++ source
            language="c++",
            extra_compile_args=(
                subprocess.check_output(["python3", "-m", "pybind11", "--includes"])
                .decode("utf-8")
                .strip()
                .split()
            )
            + ["-O3", "-Wall", "-std=c++17"],
            optional=True,  # As in your original setup
        )
        final_ext_modules.append(megatron_extension)
# --- End of megatron.core conditional section ---

setuptools.setup(
    name="megatron-core",
    version="0.0.0",
    packages=final_packages,
    package_dir=final_package_dir,
    py_modules=["is_megatron_installed"],
    ext_modules=final_ext_modules,
    # Add in any packaged data.
    include_package_data=True,
    install_requires=CACHED_DEPENDENCIES,
)

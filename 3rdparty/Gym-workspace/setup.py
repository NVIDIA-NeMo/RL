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
import sys
import tomllib
from pathlib import Path

import setuptools

final_packages = []
final_package_dir = {}

# If the submodule is present, expose `nemo_gym` package from the checkout
src_dir = Path("Gym")


CACHED_DEPENDENCIES = [
    "openai<=2.6.1",
    "tqdm",
    "pydantic",
    "pydantic_core",
    "devtools",
    "fastapi",
    "uvicorn",
    "uvloop",
    "hydra-core",
    "omegaconf",
    "gradio",
    "mlflow",
    "tdigest>=0.5.2.2",
    "aiohttp",
    "yappi",
    "ray[default]",
    "psutil",
    "datasets",
]

if src_dir.exists():
    pyproject_toml_path = src_dir / "pyproject.toml"
    with pyproject_toml_path.open("rb") as f:
        pyproject_toml = tomllib.load(f)
    if not pyproject_toml_path.exists():
        raise FileNotFoundError(
            f"[Gym][setup] {pyproject_toml_path} not found; skipping dependency consistency check."
        )

    packages = pyproject_toml["tool"]["setuptools"]["packages"]["find"]["include"]

    for package in packages:
        final_packages.append(package)
        final_package_dir[package] = src_dir / package

    actual_dependencies = pyproject_toml["project"]["dependencies"]

    ########################################
    # Compare cached dependencies with the submodule's pyproject
    ########################################

    missing_in_cached = set(actual_dependencies) - set(CACHED_DEPENDENCIES)
    extra_in_cached = set(CACHED_DEPENDENCIES) - set(actual_dependencies)

    if missing_in_cached or extra_in_cached:
        print(
            "[Gym][setup] Dependency mismatch between Gym-workspace/Gym/pyproject.toml vs Gym-workspace/setup.py::CACHED_DEPENDENCIES.",
            file=sys.stderr,
        )
        if missing_in_cached:
            print(
                "  - Present in Gym-workspace/Gym/pyproject.toml but missing from CACHED_DEPENDENCIES:",
                file=sys.stderr,
            )
            for dep in sorted(missing_in_cached):
                print(f"    * {dep}", file=sys.stderr)
        if extra_in_cached:
            print(
                "  - Present in CACHED_DEPENDENCIES but not in Gym-workspace/Gym/pyproject.toml:",
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
            "[Gym][setup] Dependency sets are consistent with the submodule pyproject.",
            file=sys.stderr,
        )


setuptools.setup(
    name="nemo_gym",
    version="0.0.0",
    description="Standalone packaging for the Gym sub-module.",
    author="NVIDIA",
    author_email="nemo-toolkit@nvidia.com",
    packages=final_packages,
    package_dir=final_package_dir,
    py_modules=["is_nemo_gym_installed"],
    install_requires=CACHED_DEPENDENCIES,
)

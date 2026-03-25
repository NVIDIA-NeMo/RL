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
import os
import sys
import tomllib

import setuptools

# Conditional packaging mirroring NeMo and Megatron-LM workspaces
final_packages = []
final_package_dir = {}

# If the submodule is present, expose `megatron.bridge` package from the checkout
bridge_src_dir = "Megatron-Bridge/src/megatron/bridge"
bridge_package_name = "megatron.bridge"

CACHED_DEPENDENCIES = [
    "transformers>=5.0.0,<=5.2.0",
    "datasets>=2.20.0",
    "accelerate",
    "omegaconf>=2.3.0",
    "tensorboard>=2.19.0",
    "typing-extensions",
    "rich",
    "wandb>=0.25.0",
    "six>=1.17.0",
    "regex>=2024.11.6",
    "pyyaml>=6.0.2",
    "tqdm>=4.67.1",
    "hydra-core>1.3,<=1.3.2",
    "megatron-core[dev,mlm]",
    "qwen-vl-utils",
    # TODO(https://github.com/NVIDIA-NeMo/RL/issues/2111): upgrade to core_cu13 when we move to CUDA 13 base container
    "transformer-engine[pytorch,core_cu12]",
    "mamba-ssm",
    "nvidia-resiliency-ext~=0.4.1",
    "causal-conv1d",
    "flash-linear-attention",
    "timm",
    "open-clip-torch>=3.2.0",
    "mlflow>=3.5.0",
    "torch>=2.6.0",
]

# If the bridge source exists, compare cached dependencies with the submodule's pyproject
if os.path.exists(bridge_src_dir):
    pyproject_path = os.path.join("Megatron-Bridge", "pyproject.toml")
    if not os.path.exists(pyproject_path):
        raise FileNotFoundError(
            f"[megatron-bridge][setup] {pyproject_path} not found; skipping dependency consistency check."
        )

    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)
    project = data["project"]
    deps_list = project["dependencies"]
    submodule_deps = set(str(d).strip() for d in deps_list)

    # Normalize the transformer-engine CUDA variant extra (core_cu12 vs core_cu13)
    # so our CUDA 12 override doesn't trip the consistency check against the
    # submodule's CUDA 13 default.
    # TODO(https://github.com/NVIDIA-NeMo/RL/issues/2111): remove this when we upgrade to CUDA 13
    def _normalize_te_cuda(dep):
        if dep.startswith("transformer-engine") or dep.startswith("transformer_engine"):
            return dep.replace("core_cu13", "core_cu12")
        return dep

    normalized_submodule = set(_normalize_te_cuda(d) for d in submodule_deps)
    normalized_cached = set(_normalize_te_cuda(d) for d in CACHED_DEPENDENCIES)
    missing_in_cached = normalized_submodule - normalized_cached
    extra_in_cached = normalized_cached - normalized_submodule

    if missing_in_cached or extra_in_cached:
        print(
            "[megatron-bridge][setup] Dependency mismatch between Megatron-Bridge-workspace/Megatron-Bridge/pyproject.toml vs Megatron-Bridge-workspace/setup.py::CACHED_DEPENDENCIES.",
            file=sys.stderr,
        )
        if missing_in_cached:
            print(
                "  - Present in Megatron-Bridge/pyproject.toml but missing from CACHED_DEPENDENCIES:",
                file=sys.stderr,
            )
            for dep in sorted(missing_in_cached):
                print(f"    * {dep}", file=sys.stderr)
        if extra_in_cached:
            print(
                "  - Present in CACHED_DEPENDENCIES but not in Megatron-Bridge/pyproject.toml:",
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
            "[megatron-bridge][setup] Dependency sets are consistent with the submodule pyproject.",
            file=sys.stderr,
        )

if os.path.exists(bridge_src_dir):
    final_packages.append(bridge_package_name)
    final_package_dir[bridge_package_name] = bridge_src_dir

setuptools.setup(
    name="megatron-bridge",
    version="0.0.0",
    description="Standalone packaging for the Megatron Bridge sub-module.",
    author="NVIDIA",
    author_email="nemo-toolkit@nvidia.com",
    packages=final_packages,
    package_dir=final_package_dir,
    py_modules=["is_megatron_bridge_installed"],
    install_requires=CACHED_DEPENDENCIES,
)

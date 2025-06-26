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

import setuptools

# --- Configuration Start ---
final_packages = []
final_package_dir = {}

# --- nemo package conditional section ---
nemo_package_source_dir = "NeMo/nemo"
nemo_package_name = "nemo"

if os.path.exists(nemo_package_source_dir):
    final_packages.append(nemo_package_name)
    final_package_dir[nemo_package_name] = nemo_package_source_dir
# --- End of nemo package conditional section ---

setuptools.setup(
    name="nemo-tron",  # Must match [project].name in pyproject.toml
    version="0.0.0",  # Must match [project].version in pyproject.toml
    description="Standalone packaging for the NeMo Tron sub-module.",  # Can be sourced from pyproject.toml too
    author="NVIDIA",
    author_email="nemo-toolkit@nvidia.com",
    packages=final_packages,
    package_dir=final_package_dir,
    py_modules=["is_nemo_installed"],
    install_requires=[
        "lightning",
        "wget",
        "onnx",
        "fiddle",
        "cloudpickle",
        "braceexpand",
        "webdataset",
        "h5py",
        "ijson",
        "matplotlib",
        "scikit-learn",
        "nemo-run",
        "hatchling",
    ],
)

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
# This path is taken from your original setup.py
megatron_core_cpp_extension_source_file = "megatron/core/datasets/helpers.cpp"

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
    install_requires=[
        # From requirements/pytorch_25.03/requirements.txt
        "einops",
        "flask-restful",
        "nltk",
        "pytest",
        "pytest-cov",
        "pytest_mock",
        "pytest-random-order",
        "sentencepiece",
        "tiktoken",
        "wrapt",
        "zarr",
        "wandb",
        "tensorstore!=0.1.46,!=0.1.72",
        "torch",
        "nvidia-modelopt[torch]>=0.23.2; sys_platform != 'darwin'",
        # From megatron/core/requirements.txt
        "torch",  # Repeated with ^ just to make it easy to map back to the original requirements.txt
        "packaging",
    ],
)

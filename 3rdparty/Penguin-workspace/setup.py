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

final_packages = []
final_package_dir = {}

# If the submodule is present, expose `penguin` package from the checkout
src_dir = "Penguin/penguin"
package_name = "penguin"

if os.path.exists(src_dir):
    final_packages.append(package_name)
    final_package_dir[package_name] = src_dir

setuptools.setup(
    name="penguin",
    version="0.0.0",
    description="Standalone packaging for the Penguin sub-module.",
    author="NVIDIA",
    author_email="nemo-toolkit@nvidia.com",
    packages=final_packages,
    package_dir=final_package_dir,
    py_modules=["is_penguin_installed"],
    install_requires=[],
)

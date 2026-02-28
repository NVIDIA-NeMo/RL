#!/bin/bash
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
set -euo pipefail
apt-get update
apt-get install -y --no-install-recommends software-properties-common
add-apt-repository -y ppa:apptainer/ppa
apt-get update
CODENAME=$(. /etc/os-release && echo "$VERSION_CODENAME")
apt-get install -y --no-install-recommends apptainer=1.4.5-1~${CODENAME}
ln -sf /usr/bin/apptainer /usr/bin/singularity
apt-get clean && rm -rf /var/lib/apt/lists/*

#!/usr/bin/env bash
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
set -euo pipefail

# --password-stdin keeps the token out of the process list.
printf '%s' "$DOCKER_TOKEN" | docker login -u "$DOCKER_USER" --password-stdin "$DOCKER_REGISTRY"
# "$*" joins all arguments into the single command string bash -c expects;
# "$@" would drop every argument after the first.
exec /bin/bash -c "$*"

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

# Isolated builds (including editable installs during release-stage prefetch)
# can cache setuptools<80 despite the runtime dependency override. Old releases
# vendor vulnerable wheel and jaraco.context. Remove their cache entries, keeping
# the patched runtime installations and their cached distributions intact.
cache="${1:-/root/.cache/uv}"
min=80.10.2
for path in "${cache}"/archive-v0/*/setuptools-*.dist-info; do
    [[ -d "${path}" ]] || continue
    ver="${path##*/setuptools-}"; ver="${ver%.dist-info}"
    if [[ "$(printf '%s\n' "${ver}" "${min}" | sort -V | head -n1)" != "${min}" ]]; then
        rm -rf "${path%/*}"
    fi
done
# Remove index entries as well, so a future isolated build downloads a fresh
# copy instead of following a dangling pointer into archive-v0.
for path in "${cache}"/wheels-v6/pypi/setuptools/*-py3-none-any*; do
    [[ -e "${path}" || -L "${path}" ]] || continue
    ver="${path##*/}"; ver="${ver%%-py3-none-any*}"
    if [[ "$(printf '%s\n' "${ver}" "${min}" | sort -V | head -n1)" != "${min}" ]]; then
        rm -rf "${path}"
    fi
done

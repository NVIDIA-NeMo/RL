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

set -eou pipefail
HELM_VERSION=${HELM_VERSION:-v3.17.3}

mkdir -p ~/bin/
NAMED_HELM=~/bin/helm-$HELM_VERSION

if [[ ! -f $NAMED_HELM ]]; then
  ARCH=$(uname -m)
  case $ARCH in
    x86_64)  ARCH=amd64 ;;
    aarch64) ARCH=arm64 ;;
    *) echo "Unsupported architecture: $ARCH" >&2; exit 1 ;;
  esac
  tmp_helm_dir=$(mktemp -d)
  curl -sSL "https://get.helm.sh/helm-${HELM_VERSION}-linux-${ARCH}.tar.gz" | tar -xz -C "$tmp_helm_dir" --strip-components=1
  cp "$tmp_helm_dir/helm" "$NAMED_HELM"
  rm -rf "$tmp_helm_dir"
fi

echo "Installed helm at $NAMED_HELM"
echo "To use, you may set 'alias helm=$NAMED_HELM'"

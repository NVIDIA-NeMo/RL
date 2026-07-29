#!/usr/bin/env bash
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

: "${TARGETARCH:?Docker must provide TARGETARCH}"
: "${DYNAMO_PYTHON_VERSION:?DYNAMO_PYTHON_VERSION is required}"
: "${ETCD_VERSION:?ETCD_VERSION is required}"
: "${NATS_VERSION:?NATS_VERSION is required}"

case "${TARGETARCH}" in
  amd64)
    etcd_arch=amd64
    nats_arch=amd64
    ;;
  arm64)
    etcd_arch=arm64
    nats_arch=arm64
    ;;
  *)
    echo "Unsupported TARGETARCH: ${TARGETARCH}" >&2
    exit 2
    ;;
esac

export UV_PYTHON_INSTALL_DIR=/opt/uv-python
uv python install "${DYNAMO_PYTHON_VERSION}"
uv venv --python "${DYNAMO_PYTHON_VERSION}" /opt/dynamo_venv
UV_PROJECT_ENVIRONMENT=/opt/dynamo_venv uv sync \
  --directory /opt/dynamo_project \
  --locked \
  --no-dev \
  --no-install-project \
  --link-mode copy

vllm_version=$(
  /opt/dynamo_venv/bin/python -c \
    'from importlib.metadata import version; print(version("vllm"))'
)
if [[ "${vllm_version}" != "0.23.0" ]]; then
  echo "Expected vllm==0.23.0 from ai-dynamo[vllm]==1.3.0.post1; got ${vllm_version}" >&2
  exit 1
fi

vllm_root=$(
  /opt/dynamo_venv/bin/python -c \
    'from pathlib import Path; import vllm; print(Path(vllm.__file__).resolve().parent.parent)'
)
patch_file=/opt/dynamo_patches/vllm-0.23.0-layerwise-reload-composed-loader.patch

# Dynamo 1.3.0 pins vLLM 0.23.0, which predates vLLM PR #44814.
# Without that fix, composed weight loaders can make layerwise reload finalize
# a layer early, leaving trailing NemotronH/Mamba2 parameters such as mixer.D
# unloaded and corrupting logits after a weight refit.
# Remove this backport only after Dynamo pins a vLLM release containing #44814.
git -C "${vllm_root}" apply --check "${patch_file}"
git -C "${vllm_root}" apply "${patch_file}"
printf '%s\n' \
  'vllm PR #44814 merge commit c9e5bf813530fb9ce06024e075da0f520b0718c8' \
  > /opt/dynamo_venv/VLLM_BACKPORTS

curl --fail --location --retry 3 \
  "https://github.com/etcd-io/etcd/releases/download/${ETCD_VERSION}/etcd-${ETCD_VERSION}-linux-${etcd_arch}.tar.gz" \
  --output /tmp/etcd.tgz
tar -xzf /tmp/etcd.tgz -C /tmp
install -m 0755 \
  "/tmp/etcd-${ETCD_VERSION}-linux-${etcd_arch}/etcd" \
  /usr/local/bin/etcd

curl --fail --location --retry 3 \
  "https://github.com/nats-io/nats-server/releases/download/${NATS_VERSION}/nats-server-${NATS_VERSION}-linux-${nats_arch}.tar.gz" \
  --output /tmp/nats.tgz
tar -xzf /tmp/nats.tgz -C /tmp
install -m 0755 \
  "/tmp/nats-server-${NATS_VERSION}-linux-${nats_arch}/nats-server" \
  /usr/local/bin/nats-server

rm -rf \
  /opt/dynamo_project \
  /opt/dynamo_install.sh \
  /opt/dynamo_patches \
  /tmp/etcd.tgz \
  "/tmp/etcd-${ETCD_VERSION}-linux-${etcd_arch}" \
  /tmp/nats.tgz \
  "/tmp/nats-server-${NATS_VERSION}-linux-${nats_arch}"

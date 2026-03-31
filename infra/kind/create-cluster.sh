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
#
# Creates an nvkind cluster with GPU support.
# Prerequisites:
#   - kind and nvkind installed (see install-nvkind.sh)
#   - NVIDIA driver + nvidia-container-toolkit on the host
#   - Docker running with systemd cgroup driver
#
# One-time host setup (run manually before first use):
#   sudo nvidia-ctk runtime configure --runtime=docker --set-as-default --cdi.enabled
#   sudo nvidia-ctk config --set accept-nvidia-visible-devices-as-volume-mounts=true --in-place
#   sudo systemctl restart docker

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

set -eoux pipefail

KIND_CLUSTER_NAME=${KIND_CLUSTER_NAME:-nemo-rl}
CONFIG_VALUES=${CONFIG_VALUES:-$SCRIPT_DIR/nvkind-config-values.yaml}

echo "======================"
echo "Existing kind clusters"
echo "======================"
kind get clusters || true

# Build nvkind args. Use custom template if it exists alongside the values file.
NVKIND_ARGS=(--name "${KIND_CLUSTER_NAME}" --config-values "$CONFIG_VALUES")
CONFIG_TEMPLATE="$SCRIPT_DIR/nvkind-config-template.yaml"
if [[ -f "$CONFIG_TEMPLATE" && "$CONFIG_VALUES" != "$SCRIPT_DIR/nvkind-config-values.yaml" ]]; then
  echo "Using custom config template: $CONFIG_TEMPLATE"
  NVKIND_ARGS+=(--config-template "$CONFIG_TEMPLATE")
fi

# nvkind may fail at the /proc/driver/nvidia patching step if the host
# doesn't have a mounted /proc/driver/nvidia (non-MIG setups). This is
# non-fatal — the cluster and GPU access still work. We catch the error
# and verify the cluster came up.
nvkind cluster create "${NVKIND_ARGS[@]}" || true

# nvkind installs the nvidia-container-toolkit and configures containerd
# inside worker nodes, but containerd needs a restart to pick up the config.
echo "Restarting containerd on worker nodes..."
for worker in $(docker ps --format '{{.Names}}' | grep -E "${KIND_CLUSTER_NAME}-worker"); do
  docker exec "$worker" systemctl restart containerd
done

echo "Waiting for nodes to become Ready..."
kubectl wait --for=condition=ready nodes --all --timeout=120s

echo "======================"
echo "Verifying cluster..."
echo "======================"
docker ps
kubectl get nodes -o wide
kubectl get pods -A

echo ""
echo "Cluster '${KIND_CLUSTER_NAME}' is ready."
echo "Next: cd ../helm && helmfile sync"

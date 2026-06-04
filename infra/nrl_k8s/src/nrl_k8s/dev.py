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
"""Dev pod manifest builder for ``nrl-k8s dev``."""

from __future__ import annotations

from typing import Any

_DEFAULT_IMAGE = "nvcr.io/nvidian/nemo-rl:nightly"
_DEFAULT_IMAGE_PULL_SECRET = "nvcr-secret"
_PVC_NAME = "rl-workspace"
_MOUNT_PATH = "/mnt/rl-workspace"


def build_dev_pod_manifest(
    username: str,
    namespace: str,
    image: str = _DEFAULT_IMAGE,
) -> dict[str, Any]:
    user_dir = f"{_MOUNT_PATH}/{username}"
    secret_name = f"{username}-secrets"
    pod_name = f"{username}-dev-pod"

    command = (
        f"mkdir -p {user_dir} /root/.ssh && "
        'if [ -n "$SSH_KEY_CONTENT" ]; then '
        'printf "%s\\n" "$SSH_KEY_CONTENT" > /root/.ssh/$SSH_KEY_NAME && '
        "chmod 600 /root/.ssh/$SSH_KEY_NAME; "
        "fi && "
        'if [ -n "$RCLONE_CONF" ]; then '
        "mkdir -p /root/.config/rclone && "
        'printf "%s\\n" "$RCLONE_CONF" > /root/.config/rclone/rclone.conf && '
        "if ! command -v rclone >/dev/null 2>&1; then "
        "curl -sSf https://rclone.org/install.sh | bash; "
        "fi; "
        "fi && "
        "if ! command -v kubectl >/dev/null 2>&1; then "
        'ARCH=$(uname -m | sed "s/x86_64/amd64/;s/aarch64/arm64/") && '
        'curl -sLo /usr/local/bin/kubectl "https://dl.k8s.io/release/$(curl -sL https://dl.k8s.io/release/stable.txt)/bin/linux/${ARCH}/kubectl" && '
        "chmod +x /usr/local/bin/kubectl; "
        "fi && "
        "sleep infinity"
    )

    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": pod_name,
            "namespace": namespace,
            "labels": {
                "app.kubernetes.io/managed-by": "nrl-k8s",
                "nrl-k8s/owner": username,
                "nrl-k8s/component": "dev-pod",
            },
        },
        "spec": {
            "restartPolicy": "Never",
            "imagePullSecrets": [{"name": _DEFAULT_IMAGE_PULL_SECRET}],
            "affinity": {
                "nodeAffinity": {
                    "requiredDuringSchedulingIgnoredDuringExecution": {
                        "nodeSelectorTerms": [
                            {
                                "matchExpressions": [
                                    {
                                        "key": "nvidia.com/gpu.product",
                                        "operator": "DoesNotExist",
                                    }
                                ]
                            }
                        ]
                    }
                }
            },
            "containers": [
                {
                    "name": "dev",
                    "image": image,
                    "command": ["sh", "-c", command],
                    "workingDir": user_dir,
                    # Set USER so getpass.getuser() / $USER returns the real
                    # owner, not "root". We keep uid=0 so users can apt-install.
                    # nrl-k8s jobs submitted from the dev pod use this to tag
                    # ownership — without it every submitter shows up as "root".
                    "env": [{"name": "USER", "value": username}],
                    "envFrom": [{"secretRef": {"name": secret_name, "optional": True}}],
                    "resources": {
                        "requests": {"cpu": "100m", "memory": "256Mi"},
                        "limits": {"cpu": "5", "memory": "10Gi"},
                    },
                    "volumeMounts": [
                        {"name": "rl-workspace", "mountPath": _MOUNT_PATH},
                    ],
                }
            ],
            "volumes": [
                {
                    "name": "rl-workspace",
                    "persistentVolumeClaim": {"claimName": _PVC_NAME},
                },
            ],
        },
    }

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

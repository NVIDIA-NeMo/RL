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
"""ConfigMap-backed endpoint registry for disaggregated RL-Gym service discovery.

Each (RL, Gym) job pair shares a ConfigMap named 'nemo-rl-endpoints-{job_id}'.
Both sides read/write their dynamic addresses (IP:port) to it. The ConfigMap
has an ownerReference to the RL RayCluster so it's garbage collected on teardown.

Usage (RL side):
    registry = K8sEndpointRegistry(job_id="my-job")
    registry.create(owner_raycluster_name="raycluster-rl")
    registry.set("vllm_base_urls", json.dumps(["http://10.0.0.1:8000/v1"]))
    gym_url = registry.get("gym_head_server")  # blocks until Gym registers

Usage (Gym side):
    registry = K8sEndpointRegistry(job_id="my-job")
    registry.set("gym_head_server", "http://10.0.0.2:8080")
    vllm_urls = json.loads(registry.get("vllm_base_urls"))  # blocks until RL registers
"""

from __future__ import annotations

import time
from pathlib import Path

from kubernetes import client, config
from kubernetes.client.exceptions import ApiException

CONFIGMAP_PREFIX = "nemo-rl-endpoints"


class K8sEndpointRegistry:
    """Shared endpoint registry backed by a K8s ConfigMap."""

    def __init__(self, job_id: str, namespace: str | None = None):
        self.job_id = job_id
        self.configmap_name = f"{CONFIGMAP_PREFIX}-{job_id}"

        # Auto-detect namespace from in-pod mount, or fall back to "default".
        if namespace is not None:
            self.namespace = namespace
        else:
            ns_path = Path("/var/run/secrets/kubernetes.io/serviceaccount/namespace")
            self.namespace = (
                ns_path.read_text().strip() if ns_path.exists() else "default"
            )

        # Load K8s client config — in-cluster when running in a pod, kubeconfig otherwise.
        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()

        self._v1 = client.CoreV1Api()
        self._custom = client.CustomObjectsApi()

    def create(self, owner_raycluster_name: str | None = None) -> None:
        """Create the ConfigMap. Idempotent — no-op if it already exists.

        Args:
            owner_raycluster_name: If set, the ConfigMap gets an ownerReference
                to this RayCluster so K8s garbage-collects it on teardown.
        """
        owner_references = None
        if owner_raycluster_name:
            owner_references = self._build_owner_reference(owner_raycluster_name)

        cm = client.V1ConfigMap(
            metadata=client.V1ObjectMeta(
                name=self.configmap_name,
                namespace=self.namespace,
                owner_references=owner_references,
            ),
            data={},
        )

        try:
            self._v1.create_namespaced_config_map(namespace=self.namespace, body=cm)
            print(f"Created endpoint registry ConfigMap: {self.configmap_name}")
        except ApiException as e:
            if e.status == 409:
                # Already exists — patch ownerReferences if we have them
                # (handles race where Gym's set() created it before RL's create()).
                if owner_references:
                    self._v1.patch_namespaced_config_map(
                        name=self.configmap_name,
                        namespace=self.namespace,
                        body=client.V1ConfigMap(
                            metadata=client.V1ObjectMeta(
                                owner_references=owner_references,
                            )
                        ),
                    )
                    print(
                        f"Patched ownerReference on existing ConfigMap: {self.configmap_name}"
                    )
                else:
                    print(
                        f"Endpoint registry ConfigMap already exists: {self.configmap_name}"
                    )
            else:
                raise

    def set(self, key: str, value: str) -> None:
        """Write a key-value pair to the ConfigMap. Creates the ConfigMap if needed."""
        try:
            cm = self._v1.read_namespaced_config_map(
                name=self.configmap_name, namespace=self.namespace
            )
            if cm.data is None:
                cm.data = {}
            cm.data[key] = value
            self._v1.patch_namespaced_config_map(
                name=self.configmap_name, namespace=self.namespace, body=cm
            )
        except ApiException as e:
            if e.status == 404:
                # ConfigMap doesn't exist yet — create it with this key.
                try:
                    cm = client.V1ConfigMap(
                        metadata=client.V1ObjectMeta(
                            name=self.configmap_name, namespace=self.namespace
                        ),
                        data={key: value},
                    )
                    self._v1.create_namespaced_config_map(
                        namespace=self.namespace, body=cm
                    )
                except ApiException as create_err:
                    if create_err.status == 409:
                        # Another process created it between our read and create — retry patch.
                        self.set(key, value)
                        return
                    raise
            else:
                raise
        print(f"Registered endpoint: {key} = {value}")

    def get(self, key: str, timeout: float = 600, poll_interval: float = 2) -> str:
        """Poll until a key appears in the ConfigMap, then return its value.

        Args:
            key: The key to wait for.
            timeout: Max seconds to wait before raising TimeoutError.
            poll_interval: Seconds between polls.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = self.get_nowait(key)
            if value is not None:
                return value
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(poll_interval, remaining))
        raise TimeoutError(
            f"Timed out after {timeout}s waiting for key '{key}' "
            f"in ConfigMap '{self.configmap_name}'"
        )

    def get_nowait(self, key: str) -> str | None:
        """Non-blocking read. Returns None if key or ConfigMap doesn't exist."""
        try:
            cm = self._v1.read_namespaced_config_map(
                name=self.configmap_name, namespace=self.namespace
            )
            if cm.data is None:
                return None
            return cm.data.get(key)
        except ApiException as e:
            if e.status == 404:
                return None
            raise

    def signal_error(self, message: str) -> None:
        """Write an error to the ConfigMap.

        The peer-watcher sidecar monitors this and triggers teardown
        when it sees a non-empty 'error' key.
        """
        self.set("error", message)

    def _build_owner_reference(
        self, raycluster_name: str
    ) -> list[client.V1OwnerReference]:
        """Look up the RayCluster's UID and build an ownerReference."""
        try:
            rc = self._custom.get_namespaced_custom_object(
                group="ray.io",
                version="v1",
                namespace=self.namespace,
                plural="rayclusters",
                name=raycluster_name,
            )
            uid = rc["metadata"]["uid"]
        except ApiException as e:
            if e.status == 404:
                print(
                    f"Warning: RayCluster '{raycluster_name}' not found "
                    f"for ownerReference — ConfigMap will not be auto-cleaned."
                )
                return None
            raise

        return [
            client.V1OwnerReference(
                api_version="ray.io/v1",
                kind="RayCluster",
                name=raycluster_name,
                uid=uid,
                controller=True,
                block_owner_deletion=False,
            )
        ]

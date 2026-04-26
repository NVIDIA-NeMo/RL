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
"""Thin wrapper around the official ``kubernetes`` Python client.

RayCluster + RayJob are Kubernetes ``CustomObjectsApi`` resources, so we use
the client's generic CR helpers instead of modeling either CRD in Python.
The research-facing YAML stays authoritative; this module just ships those
objects into the cluster and polls until they're ready.
"""

from __future__ import annotations

import functools
import time
from typing import Any

from kubernetes import client, config
from kubernetes.client.exceptions import ApiException

from ._logging import redact
from ._retry import with_retries

# KubeRay CRD identifiers (stable since v1.x).
RAY_GROUP = "ray.io"
RAY_VERSION = "v1"
RAYCLUSTER_PLURAL = "rayclusters"
RAYJOB_PLURAL = "rayjobs"

# RayJob jobDeploymentStatus terminal states. `Complete` covers a job that
# succeeded end-to-end; `Failed` covers driver exit != 0; the rest are
# actively-running / suspended states where we keep polling.
_RAYJOB_TERMINAL_DEPLOYMENT = ("Complete", "Failed")


# =============================================================================
# Client bootstrap
# =============================================================================


@functools.cache
def load_kubeconfig() -> None:
    """Pick the right config source (in-cluster vs kubeconfig) exactly once."""
    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()


def custom_objects_api() -> client.CustomObjectsApi:
    load_kubeconfig()
    return client.CustomObjectsApi()


# =============================================================================
# RayCluster lifecycle
# =============================================================================


def apply_raycluster(manifest: dict[str, Any], namespace: str) -> dict[str, Any]:
    """Create-or-replace a RayCluster. Returns the server-side object."""
    name = manifest["metadata"]["name"]
    api = custom_objects_api()
    try:
        return with_retries(
            lambda: api.create_namespaced_custom_object(
                group=RAY_GROUP,
                version=RAY_VERSION,
                namespace=namespace,
                plural=RAYCLUSTER_PLURAL,
                body=manifest,
            )
        )
    except ApiException as exc:
        if exc.status == 409:
            # Already exists — patch the spec in place (kubectl apply-equivalent).
            return with_retries(
                lambda: api.patch_namespaced_custom_object(
                    group=RAY_GROUP,
                    version=RAY_VERSION,
                    namespace=namespace,
                    plural=RAYCLUSTER_PLURAL,
                    name=name,
                    body=manifest,
                )
            )
        # Attach a redacted manifest summary for easier debugging without
        # leaking secret env values into the CLI output.
        exc.nrl_k8s_manifest = redact(manifest)  # type: ignore[attr-defined]
        raise


def delete_raycluster(
    name: str, namespace: str, *, ignore_missing: bool = True
) -> None:
    api = custom_objects_api()
    try:
        with_retries(
            lambda: api.delete_namespaced_custom_object(
                group=RAY_GROUP,
                version=RAY_VERSION,
                namespace=namespace,
                plural=RAYCLUSTER_PLURAL,
                name=name,
            )
        )
    except ApiException as exc:
        if exc.status == 404 and ignore_missing:
            return
        raise


def get_raycluster(name: str, namespace: str) -> dict[str, Any] | None:
    api = custom_objects_api()
    try:
        return with_retries(
            lambda: api.get_namespaced_custom_object(
                group=RAY_GROUP,
                version=RAY_VERSION,
                namespace=namespace,
                plural=RAYCLUSTER_PLURAL,
                name=name,
            )
        )
    except ApiException as exc:
        if exc.status == 404:
            return None
        raise


def list_rayclusters(namespace: str, label_selector: str | None = None) -> list[dict]:
    api = custom_objects_api()
    resp = with_retries(
        lambda: api.list_namespaced_custom_object(
            group=RAY_GROUP,
            version=RAY_VERSION,
            namespace=namespace,
            plural=RAYCLUSTER_PLURAL,
            label_selector=label_selector or "",
        )
    )
    return resp.get("items", [])


def wait_for_raycluster_ready(
    name: str, namespace: str, *, timeout_s: int = 900, poll_s: int = 5
) -> None:
    """Block until ``.status.state == ready`` or time out.

    KubeRay flips this flag once the head pod is up and all declared workers
    report Running — the correct signal before submitting jobs.
    """
    deadline = time.monotonic() + timeout_s
    state: str | None = None
    while time.monotonic() < deadline:
        # get_raycluster already retries transient 5xx/timeout, so a poll
        # blip won't abort the wait.
        obj = get_raycluster(name, namespace)
        state = (obj or {}).get("status", {}).get("state")
        if state == "ready":
            return
        time.sleep(poll_s)
    raise TimeoutError(
        f"RayCluster {name} in {namespace} never reached state=ready "
        f"(last seen: {state!r}) after {timeout_s}s"
    )


def wait_for_raycluster_gone(
    name: str, namespace: str, *, timeout_s: int = 600, poll_s: int = 3
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if get_raycluster(name, namespace) is None:
            return
        time.sleep(poll_s)
    raise TimeoutError(f"RayCluster {name} not deleted after {timeout_s}s")


# =============================================================================
# RayJob lifecycle
# =============================================================================


def apply_rayjob(manifest: dict[str, Any], namespace: str) -> dict[str, Any]:
    """Create-or-replace a RayJob. Returns the server-side object."""
    name = manifest["metadata"]["name"]
    api = custom_objects_api()
    try:
        return with_retries(
            lambda: api.create_namespaced_custom_object(
                group=RAY_GROUP,
                version=RAY_VERSION,
                namespace=namespace,
                plural=RAYJOB_PLURAL,
                body=manifest,
            )
        )
    except ApiException as exc:
        if exc.status == 409:
            return with_retries(
                lambda: api.patch_namespaced_custom_object(
                    group=RAY_GROUP,
                    version=RAY_VERSION,
                    namespace=namespace,
                    plural=RAYJOB_PLURAL,
                    name=name,
                    body=manifest,
                )
            )
        exc.nrl_k8s_manifest = redact(manifest)  # type: ignore[attr-defined]
        raise


def delete_rayjob(
    name: str, namespace: str, *, ignore_missing: bool = True
) -> None:
    api = custom_objects_api()
    try:
        with_retries(
            lambda: api.delete_namespaced_custom_object(
                group=RAY_GROUP,
                version=RAY_VERSION,
                namespace=namespace,
                plural=RAYJOB_PLURAL,
                name=name,
            )
        )
    except ApiException as exc:
        if exc.status == 404 and ignore_missing:
            return
        raise


def get_rayjob(name: str, namespace: str) -> dict[str, Any] | None:
    api = custom_objects_api()
    try:
        return with_retries(
            lambda: api.get_namespaced_custom_object(
                group=RAY_GROUP,
                version=RAY_VERSION,
                namespace=namespace,
                plural=RAYJOB_PLURAL,
                name=name,
            )
        )
    except ApiException as exc:
        if exc.status == 404:
            return None
        raise


def wait_for_rayjob_terminal(
    name: str,
    namespace: str,
    *,
    timeout_s: int = 86400,
    poll_s: int = 10,
    on_update: callable | None = None,
) -> dict[str, Any]:
    """Block until a RayJob's ``jobDeploymentStatus`` is Complete or Failed.

    Returns the final server-side object so callers can read the
    ``.status.jobStatus`` / ``.status.message`` fields. ``on_update`` is
    invoked with ``(jobDeploymentStatus, jobStatus)`` on every transition —
    useful for printing progress without this module owning the logger.
    """
    deadline = time.monotonic() + timeout_s
    last: tuple[str | None, str | None] = (None, None)
    while time.monotonic() < deadline:
        obj = get_rayjob(name, namespace)
        if obj is None:
            raise RuntimeError(
                f"RayJob {name} disappeared from {namespace} before reaching terminal state"
            )
        status = obj.get("status") or {}
        dep = status.get("jobDeploymentStatus")
        job = status.get("jobStatus")
        if (dep, job) != last:
            if on_update is not None:
                on_update(dep, job)
            last = (dep, job)
        if dep in _RAYJOB_TERMINAL_DEPLOYMENT:
            return obj
        time.sleep(poll_s)
    raise TimeoutError(
        f"RayJob {name} in {namespace} did not reach a terminal state "
        f"(last seen jobDeploymentStatus={last[0]!r}, jobStatus={last[1]!r}) "
        f"after {timeout_s}s"
    )


def delete_configmap(name: str, namespace: str, *, ignore_missing: bool = True) -> bool:
    """Delete a ConfigMap. Returns True if deleted, False if it didn't exist."""
    load_kubeconfig()
    core = client.CoreV1Api()
    try:
        with_retries(
            lambda: core.delete_namespaced_config_map(name=name, namespace=namespace)
        )
        return True
    except ApiException as exc:
        if exc.status == 404 and ignore_missing:
            return False
        raise


# =============================================================================
# DRA: ComputeDomain + ResourceClaimTemplate
# =============================================================================

COMPUTE_DOMAIN_GROUP = "resource.nvidia.com"
COMPUTE_DOMAIN_VERSION = "v1beta1"
COMPUTE_DOMAIN_PLURAL = "computedomains"

RCT_GROUP = "resource.k8s.io"
RCT_VERSION = "v1"
RCT_PLURAL = "resourceclaimtemplates"


def apply_compute_domain(manifest: dict[str, Any], namespace: str) -> dict[str, Any]:
    """Create a ComputeDomain. No-op on 409 (already exists)."""
    api = custom_objects_api()
    try:
        return with_retries(
            lambda: api.create_namespaced_custom_object(
                group=COMPUTE_DOMAIN_GROUP,
                version=COMPUTE_DOMAIN_VERSION,
                namespace=namespace,
                plural=COMPUTE_DOMAIN_PLURAL,
                body=manifest,
            )
        )
    except ApiException as exc:
        if exc.status == 409:
            return {}
        raise


def delete_compute_domain(
    name: str, namespace: str, *, ignore_missing: bool = True
) -> None:
    api = custom_objects_api()
    try:
        with_retries(
            lambda: api.delete_namespaced_custom_object(
                group=COMPUTE_DOMAIN_GROUP,
                version=COMPUTE_DOMAIN_VERSION,
                namespace=namespace,
                plural=COMPUTE_DOMAIN_PLURAL,
                name=name,
            )
        )
    except ApiException as exc:
        if exc.status == 404 and ignore_missing:
            return
        raise


def apply_resource_claim_template(
    manifest: dict[str, Any], namespace: str
) -> dict[str, Any]:
    """Create a ResourceClaimTemplate. No-op on 409 (already exists)."""
    api = custom_objects_api()
    try:
        return with_retries(
            lambda: api.create_namespaced_custom_object(
                group=RCT_GROUP,
                version=RCT_VERSION,
                namespace=namespace,
                plural=RCT_PLURAL,
                body=manifest,
            )
        )
    except ApiException as exc:
        if exc.status == 409:
            return {}
        raise


def delete_resource_claim_template(
    name: str, namespace: str, *, ignore_missing: bool = True
) -> None:
    api = custom_objects_api()
    try:
        with_retries(
            lambda: api.delete_namespaced_custom_object(
                group=RCT_GROUP,
                version=RCT_VERSION,
                namespace=namespace,
                plural=RCT_PLURAL,
                name=name,
            )
        )
    except ApiException as exc:
        if exc.status == 404 and ignore_missing:
            return
        raise


# =============================================================================
# Pod helpers
# =============================================================================


def get_head_pod(cluster_name: str, namespace: str) -> Any:
    """Return the first ``Running`` head pod for a RayCluster.

    Used by the exec submitter to pick a shell target. KubeRay labels every
    head pod with ``ray.io/cluster=<name>,ray.io/node-type=head`` — that
    selector uniquely identifies a single pod today (KubeRay runs exactly
    one head per cluster), but we still filter on phase to avoid returning
    a ``Pending`` or ``Terminating`` instance from a mid-rollout restart.
    """
    load_kubeconfig()
    core = client.CoreV1Api()
    selector = f"ray.io/cluster={cluster_name},ray.io/node-type=head"
    resp = with_retries(
        lambda: core.list_namespaced_pod(namespace=namespace, label_selector=selector)
    )
    for pod in resp.items:
        if pod.status and pod.status.phase == "Running":
            return pod
    raise RuntimeError(
        f"no Running head pod found for RayCluster {cluster_name!r} in "
        f"namespace {namespace!r} (label selector: {selector}). Check "
        f"`kubectl -n {namespace} get pods -l {selector}`."
    )


def create_pod(manifest: dict[str, Any], namespace: str) -> dict[str, Any]:
    """Create a pod. No-op on 409 (already exists)."""
    load_kubeconfig()
    core = client.CoreV1Api()
    try:
        return with_retries(
            lambda: core.create_namespaced_pod(namespace=namespace, body=manifest)
        )
    except ApiException as exc:
        if exc.status == 409:
            return {}
        raise


def delete_pod(name: str, namespace: str, *, ignore_missing: bool = True) -> None:
    load_kubeconfig()
    core = client.CoreV1Api()
    try:
        with_retries(
            lambda: core.delete_namespaced_pod(name=name, namespace=namespace)
        )
    except ApiException as exc:
        if exc.status == 404 and ignore_missing:
            return
        raise


def list_pods_by_label(label_selector: str, namespace: str) -> list:
    """Return pods matching a label selector."""
    load_kubeconfig()
    core = client.CoreV1Api()
    result = with_retries(
        lambda: core.list_namespaced_pod(namespace=namespace, label_selector=label_selector)
    )
    return result.items or []


def get_pod_phase(name: str, namespace: str) -> str | None:
    """Return the pod phase (Pending/Running/Succeeded/Failed) or None if not found."""
    load_kubeconfig()
    core = client.CoreV1Api()
    try:
        pod = with_retries(
            lambda: core.read_namespaced_pod(name=name, namespace=namespace)
        )
        return pod.status.phase if pod.status else None
    except ApiException as exc:
        if exc.status == 404:
            return None
        raise


def get_pod_image(name: str, namespace: str) -> str | None:
    """Return the image of the first container in a pod, or None if not found."""
    load_kubeconfig()
    core = client.CoreV1Api()
    try:
        pod = with_retries(
            lambda: core.read_namespaced_pod(name=name, namespace=namespace)
        )
        containers = pod.spec.containers or []
        return containers[0].image if containers else None
    except ApiException as exc:
        if exc.status == 404:
            return None
        raise


# =============================================================================
# Secrets
# =============================================================================


def create_or_update_secret(
    name: str, namespace: str, data: dict[str, str]
) -> None:
    """Create a Secret or merge new keys into an existing one."""
    import base64

    load_kubeconfig()
    core = client.CoreV1Api()
    encoded = {k: base64.b64encode(v.encode()).decode() for k, v in data.items()}

    try:
        existing = core.read_namespaced_secret(name=name, namespace=namespace)
        merged = dict(existing.data or {})
        merged.update(encoded)
        existing.data = merged
        with_retries(
            lambda: core.replace_namespaced_secret(
                name=name, namespace=namespace, body=existing
            )
        )
    except ApiException as exc:
        if exc.status == 404:
            secret = client.V1Secret(
                metadata=client.V1ObjectMeta(
                    name=name,
                    namespace=namespace,
                    labels={"app.kubernetes.io/managed-by": "nrl-k8s"},
                ),
                type="Opaque",
                data=encoded,
            )
            with_retries(
                lambda: core.create_namespaced_secret(
                    namespace=namespace, body=secret
                )
            )
        else:
            raise


def secret_exists(name: str, namespace: str) -> bool:
    load_kubeconfig()
    core = client.CoreV1Api()
    try:
        core.read_namespaced_secret(name=name, namespace=namespace)
        return True
    except ApiException as exc:
        if exc.status == 404:
            return False
        raise


__all__ = [
    "apply_compute_domain",
    "apply_raycluster",
    "apply_rayjob",
    "apply_resource_claim_template",
    "create_or_update_secret",
    "create_pod",
    "custom_objects_api",
    "delete_compute_domain",
    "delete_configmap",
    "delete_pod",
    "delete_raycluster",
    "delete_rayjob",
    "delete_resource_claim_template",
    "get_head_pod",
    "get_pod_phase",
    "get_raycluster",
    "get_rayjob",
    "list_rayclusters",
    "load_kubeconfig",
    "secret_exists",
    "wait_for_raycluster_gone",
    "wait_for_raycluster_ready",
    "wait_for_rayjob_terminal",
]

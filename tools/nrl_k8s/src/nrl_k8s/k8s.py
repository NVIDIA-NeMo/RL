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


__all__ = [
    "apply_raycluster",
    "custom_objects_api",
    "delete_configmap",
    "delete_raycluster",
    "get_raycluster",
    "list_rayclusters",
    "load_kubeconfig",
    "wait_for_raycluster_gone",
    "wait_for_raycluster_ready",
]

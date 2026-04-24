"""Build a RayCluster manifest dict from the recipe's inline ``spec``.

The recipe encodes the full RayCluster shape inline under
``infra.clusters.<role>.spec`` — this module just wraps it in the standard
``apiVersion/kind/metadata`` envelope and patches three cross-cutting
fields (``image`` on every container, ``imagePullSecrets`` on every pod
template, optional ``serviceAccountName``) from the top-level ``infra``
block so you don't repeat them across roles.

The resulting dict is submitted as-is via the official ``kubernetes``
Python client's ``CustomObjectsApi``.
"""

from __future__ import annotations

import copy
from typing import Any

from .schema import ClusterSpec, InfraConfig

# =============================================================================
# Public API
# =============================================================================


def build_raycluster_manifest(
    cluster: ClusterSpec, infra: InfraConfig
) -> dict[str, Any]:
    """Build the full RayCluster dict for apply.

    Args:
        cluster: the role's ClusterSpec (name + inline spec + optional daemon).
        infra: top-level InfraConfig — supplies namespace, image, pull secrets,
            optional serviceAccount. These are patched into every container /
            pod template in the spec.

    Returns:
        A dict suitable for ``CustomObjectsApi.create_namespaced_custom_object``.
    """
    spec = copy.deepcopy(cluster.spec)

    _patch_images(spec, infra.image)
    _patch_image_pull_secrets(spec, list(infra.imagePullSecrets))
    if infra.serviceAccount is not None:
        _patch_service_account(spec, infra.serviceAccount)

    metadata: dict[str, Any] = {
        "name": cluster.name,
        "namespace": infra.namespace,
    }
    labels = {**infra.labels, **cluster.labels}
    annotations = {**infra.annotations, **cluster.annotations}
    if labels:
        metadata["labels"] = labels
    if annotations:
        metadata["annotations"] = annotations

    return {
        "apiVersion": "ray.io/v1",
        "kind": "RayCluster",
        "metadata": metadata,
        "spec": spec,
    }


# =============================================================================
# Internals
# =============================================================================


def _walk_pod_templates(raycluster_spec: dict) -> list[dict]:
    """Return every PodSpec inside a RayCluster (head + all worker groups)."""
    specs: list[dict] = []
    head = raycluster_spec.get("headGroupSpec") or {}
    head_spec = head.get("template", {}).get("spec")
    if isinstance(head_spec, dict):
        specs.append(head_spec)
    for wg in raycluster_spec.get("workerGroupSpecs") or []:
        wg_spec = wg.get("template", {}).get("spec")
        if isinstance(wg_spec, dict):
            specs.append(wg_spec)
    return specs


def _patch_images(raycluster_spec: dict, image: str) -> None:
    for pod_spec in _walk_pod_templates(raycluster_spec):
        for container in pod_spec.get("containers", []):
            container["image"] = image


def _patch_image_pull_secrets(raycluster_spec: dict, secrets: list[str]) -> None:
    if not secrets:
        return
    body = [{"name": s} for s in secrets]
    for pod_spec in _walk_pod_templates(raycluster_spec):
        pod_spec["imagePullSecrets"] = body


def _patch_service_account(raycluster_spec: dict, service_account: str) -> None:
    for pod_spec in _walk_pod_templates(raycluster_spec):
        pod_spec["serviceAccountName"] = service_account


__all__ = ["build_raycluster_manifest"]

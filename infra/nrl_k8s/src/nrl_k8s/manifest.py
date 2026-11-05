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
"""Build K8s manifests from the recipe's inline specs.

RayCluster specs live under ``infra.kuberay.<role>.spec``; Deployment specs
live under ``infra.deployments.<key>.spec``. This module wraps each in the
standard ``apiVersion/kind/metadata`` envelope and patches cross-cutting
fields (image, imagePullSecrets, optional serviceAccountName) from the
top-level ``infra`` block so you don't repeat them across roles.
"""

from __future__ import annotations

import copy
from typing import Any

from .schema import ClusterSpec, DeploymentSpec, InfraConfig

# Every resource the CLI creates carries this label so admins can find
# orphans not managed by the tool:
#   kubectl get rayclusters -l '!app.kubernetes.io/managed-by'
_MANAGED_BY_LABEL = {"app.kubernetes.io/managed-by": "nrl-k8s"}

# =============================================================================
# Public API
# =============================================================================


def build_raycluster_manifest(
    cluster: ClusterSpec,
    infra: InfraConfig,
    *,
    role: str | None = None,
) -> dict[str, Any]:
    """Build the full RayCluster dict for apply.

    Args:
        cluster: the role's ClusterSpec (name + inline spec + optional daemon).
        infra: top-level InfraConfig — supplies namespace, image, pull secrets,
            optional serviceAccount. These are patched into every container /
            pod template in the spec.
        role: cluster role name (``training``, ``generation``, ``gym``). When
            provided, DRA ``resourceClaimTemplateName`` references in worker
            pods are rewritten to deterministic names derived from the cluster
            name and role.

    Returns:
        A dict suitable for ``CustomObjectsApi.create_namespaced_custom_object``.
    """
    spec = copy.deepcopy(cluster.spec)

    if cluster.segmentSize is not None:
        _expand_worker_segments(spec, cluster.segmentSize)

    _patch_images(spec, infra.image)
    _patch_image_pull_secrets(spec, list(infra.imagePullSecrets))
    if infra.serviceAccount is not None:
        _patch_service_account(spec, infra.serviceAccount)
    _patch_pod_labels(spec, {**_MANAGED_BY_LABEL, **infra.labels, **cluster.labels})
    # DRA resources are named {prefix}-{cluster_name}-{role} so that
    # disaggregated setups with multiple clusters get distinct
    # ComputeDomains and RoCE templates per role.
    if role is not None:
        _rewrite_dra_template_names(spec, cluster.name, role)

    metadata: dict[str, Any] = {
        "name": cluster.name,
        "namespace": infra.namespace,
    }
    labels = {**_MANAGED_BY_LABEL, **infra.labels, **cluster.labels}
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


def _expand_worker_segments(spec: dict, segment_size: int) -> None:
    """Split worker groups whose replicas exceed *segment_size*.

    Each qualifying group is replaced by ``replicas // segment_size``
    identical copies named ``{groupName}-segment-{i}`` with replica counts
    set to *segment_size*. Mutates *spec* in place.
    """
    original_groups = spec.get("workerGroupSpecs") or []
    if not original_groups:
        return

    expanded: list[dict] = []
    for wg in original_groups:
        replicas = int(wg.get("replicas", 0))
        if replicas <= segment_size:
            expanded.append(wg)
            continue

        if replicas % segment_size != 0:
            group_name = wg.get("groupName", "<unnamed>")
            raise ValueError(
                f"workerGroup '{group_name}' has replicas={replicas} which is "
                f"not evenly divisible by segmentSize={segment_size}"
            )

        num_segments = replicas // segment_size
        base_name = wg.get("groupName", "workers")
        for i in range(num_segments):
            segment = copy.deepcopy(wg)
            segment["groupName"] = f"{base_name}-segment-{i}"
            segment["replicas"] = segment_size
            segment["minReplicas"] = segment_size
            segment["maxReplicas"] = segment_size
            expanded.append(segment)

    spec["workerGroupSpecs"] = expanded


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


def _patch_pod_labels(raycluster_spec: dict, labels: dict[str, str]) -> None:
    """Merge ``labels`` into every pod template's metadata.labels."""
    head = raycluster_spec.get("headGroupSpec") or {}
    templates = [head.get("template")]
    for wg in raycluster_spec.get("workerGroupSpecs") or []:
        templates.append(wg.get("template"))
    for tpl in templates:
        if not isinstance(tpl, dict):
            continue
        meta = tpl.setdefault("metadata", {})
        existing = meta.get("labels") or {}
        meta["labels"] = {**labels, **existing}


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


# =============================================================================
# DRA: detect and rewrite resourceClaimTemplateName in worker pods
# =============================================================================

_DRA_CLAIM_PREFIX: dict[str, str] = {
    "compute-domain-channel": "compute-domain-",
    "roce-channel": "roce-",
}


def _rewrite_dra_template_names(spec: dict, cluster_name: str, role: str) -> None:
    """Rewrite ``resourceClaimTemplateName`` in worker pods to deterministic names."""
    for wg in spec.get("workerGroupSpecs") or []:
        pod_spec = wg.get("template", {}).get("spec")
        if not isinstance(pod_spec, dict):
            continue
        for claim in pod_spec.get("resourceClaims") or []:
            prefix = _DRA_CLAIM_PREFIX.get(claim.get("name", ""))
            if prefix:
                claim["resourceClaimTemplateName"] = f"{prefix}{cluster_name}-{role}"


def dra_resources_for_cluster(
    cluster_name: str, role: str, spec: dict
) -> list[tuple[str, str]]:
    """Return ``[(kind, name), ...]`` for DRA resources a cluster spec needs.

    ``kind`` is ``"compute-domain"`` or ``"roce"``. Scans every worker pod
    template for well-known claim names.
    """
    found: list[tuple[str, str]] = []
    seen: set[str] = set()
    for wg in spec.get("workerGroupSpecs") or []:
        pod_spec = wg.get("template", {}).get("spec")
        if not isinstance(pod_spec, dict):
            continue
        for claim in pod_spec.get("resourceClaims") or []:
            claim_name = claim.get("name", "")
            prefix = _DRA_CLAIM_PREFIX.get(claim_name)
            if prefix and claim_name not in seen:
                seen.add(claim_name)
                resource_name = f"{prefix}{cluster_name}-{role}"
                kind = "compute-domain" if "compute-domain" in prefix else "roce"
                found.append((kind, resource_name))
    return found


def build_compute_domain_manifest(name: str, namespace: str) -> dict[str, Any]:
    return {
        "apiVersion": "resource.nvidia.com/v1beta1",
        "kind": "ComputeDomain",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "labels": {**_MANAGED_BY_LABEL},
        },
        "spec": {
            "channel": {"resourceClaimTemplate": {"name": name}},
            "numNodes": 0,
        },
    }


def build_roce_template_manifest(name: str, namespace: str) -> dict[str, Any]:
    # TODO: expose roce count via infra config — hardcoded to 8 for now
    return {
        "apiVersion": "resource.k8s.io/v1",
        "kind": "ResourceClaimTemplate",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "labels": {**_MANAGED_BY_LABEL},
        },
        "spec": {
            "spec": {
                "devices": {
                    "requests": [
                        {
                            "exactly": {
                                "count": 8,
                                "deviceClassName": "roce.networking.k8s.aws",
                            },
                            "name": "roce",
                        }
                    ],
                },
            },
        },
    }


def build_deployment_manifest(
    deployment: DeploymentSpec,
    infra: InfraConfig,
) -> dict[str, Any]:
    """Build a Deployment dict for apply.

    Same patching pattern as :func:`build_raycluster_manifest` — image,
    imagePullSecrets, serviceAccountName, and labels are applied from the
    top-level ``infra`` block.
    """
    spec = copy.deepcopy(deployment.spec)

    template = spec.get("template", {})
    pod_spec = template.get("spec")
    if isinstance(pod_spec, dict):
        for container in pod_spec.get("containers", []):
            container.setdefault("image", infra.image)
        if infra.imagePullSecrets:
            pod_spec["imagePullSecrets"] = [{"name": s} for s in infra.imagePullSecrets]
        if infra.serviceAccount is not None:
            pod_spec["serviceAccountName"] = infra.serviceAccount

    tmeta = template.setdefault("metadata", {})
    existing_labels = tmeta.get("labels") or {}
    tmeta["labels"] = {
        **_MANAGED_BY_LABEL,
        **infra.labels,
        **deployment.labels,
        **existing_labels,
    }

    metadata: dict[str, Any] = {
        "name": deployment.name,
        "namespace": infra.namespace,
    }
    labels = {**_MANAGED_BY_LABEL, **infra.labels, **deployment.labels}
    annotations = {**infra.annotations, **deployment.annotations}
    if labels:
        metadata["labels"] = labels
    if annotations:
        metadata["annotations"] = annotations

    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": metadata,
        "spec": spec,
    }


def build_service_for_deployment(
    deployment: DeploymentSpec,
    infra: InfraConfig,
) -> dict[str, Any] | None:
    """Build a ClusterIP Service for a Deployment, or None if no ports declared.

    The Service uses the Deployment's name and selector labels so DNS is
    predictable: ``<deployment-name>.<namespace>.svc.cluster.local``.
    """
    selector = (deployment.spec.get("selector") or {}).get("matchLabels")
    if not selector:
        return None

    ports: list[dict[str, Any]] = []
    template = deployment.spec.get("template", {})
    pod_spec = template.get("spec") or {}
    for container in pod_spec.get("containers", []):
        for i, p in enumerate(container.get("ports", [])):
            port_entry: dict[str, Any] = {
                "port": p["containerPort"],
                "targetPort": p["containerPort"],
                "protocol": p.get("protocol", "TCP"),
            }
            port_entry["name"] = p.get("name") or f"port-{i}"
            ports.append(port_entry)

    if not ports:
        return None

    return {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": deployment.name,
            "namespace": infra.namespace,
            "labels": {**_MANAGED_BY_LABEL},
        },
        "spec": {
            "type": "ClusterIP",
            "selector": selector,
            "ports": ports,
        },
    }


__all__ = [
    "build_compute_domain_manifest",
    "build_deployment_manifest",
    "build_raycluster_manifest",
    "build_roce_template_manifest",
    "build_service_for_deployment",
    "dra_resources_for_cluster",
]

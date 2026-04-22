"""Build and apply KubeRay ``RayJob`` objects.

A RayJob is a RayCluster + Ray Job rolled into one K8s resource: KubeRay
creates the cluster, waits for it to become ready, submits ``spec.entrypoint``
over the dashboard HTTP API, polls until terminal, then (optionally) tears
the whole cluster down. It is the convenient shape for one-shot runs where
you do *not* want a long-lived RayCluster left behind.

The inline RayCluster body comes from the recipe — this module reuses
:func:`nrl_k8s.manifest.build_raycluster_manifest` to get the patched cluster
manifest, then lifts its ``.spec`` under ``rayClusterSpec`` and wraps a
RayJob envelope around it.
"""

from __future__ import annotations

from typing import Any

from .manifest import build_raycluster_manifest
from .schema import ClusterSpec, InfraConfig

# KubeRay defaults recipes have standardised on.
DEFAULT_SUBMISSION_MODE = "HTTPMode"
DEFAULT_TTL_SECONDS = 3600


def build_rayjob_manifest(
    cluster: ClusterSpec,
    infra: InfraConfig,
    *,
    entrypoint: str,
    name: str | None = None,
    shutdown_after_finishes: bool = True,
    ttl_seconds_after_finished: int = DEFAULT_TTL_SECONDS,
    submission_mode: str = DEFAULT_SUBMISSION_MODE,
    extra_labels: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Wrap a RayCluster inline body in a RayJob envelope.

    Args:
        cluster: the training ClusterSpec used for ``rayClusterSpec``.
        infra: top-level infra — supplies namespace, image, pullSecrets, SA.
            These are applied inside the rayClusterSpec identically to how
            the standalone RayCluster path patches them.
        entrypoint: shell command KubeRay submits via HTTP to the dashboard.
            Typically ``infra.launch.entrypoint``.
        name: RayJob metadata name. Defaults to the cluster's name —
            convenient because ``nrl-k8s job list`` on an ephemeral cluster
            can still resolve by that name.
        shutdown_after_finishes: KubeRay deletes the RayCluster once the
            Ray Job reaches a terminal state. Leave True for one-shot runs.
        ttl_seconds_after_finished: seconds to keep the RayJob object
            around after the job reaches Complete/Failed. After the TTL
            the garbage collector removes the RayJob (and the cluster,
            when ``shutdown_after_finishes=True``).
        submission_mode: KubeRay submission path. ``HTTPMode`` posts the
            entrypoint to the dashboard (same code path as Ray Job SDK).
            ``K8sJobMode`` creates a separate K8s Job, which breaks KAI
            gang scheduling — leave this at the default unless you know
            you need the other shape.
        extra_labels: merged on top of infra + cluster labels on the RayJob.
            Used by callers to tag runs (e.g. ``disagg.nemo-rl/run-id``).
    """
    # Reuse the RayCluster builder so image / imagePullSecrets / SA / labels
    # are patched the same way as the standalone cluster path. Then lift
    # the .spec out; RayJob nests it under rayClusterSpec.
    cluster_manifest = build_raycluster_manifest(cluster, infra)
    ray_cluster_spec = cluster_manifest["spec"]

    job_name = name or cluster.name
    metadata: dict[str, Any] = {
        "name": job_name,
        "namespace": infra.namespace,
    }
    merged_labels = {**infra.labels, **cluster.labels, **(extra_labels or {})}
    if merged_labels:
        metadata["labels"] = merged_labels
    merged_annotations = {**infra.annotations, **cluster.annotations}
    if merged_annotations:
        metadata["annotations"] = merged_annotations

    return {
        "apiVersion": "ray.io/v1",
        "kind": "RayJob",
        "metadata": metadata,
        "spec": {
            "entrypoint": entrypoint,
            "submissionMode": submission_mode,
            "shutdownAfterJobFinishes": shutdown_after_finishes,
            "ttlSecondsAfterFinished": ttl_seconds_after_finished,
            "rayClusterSpec": ray_cluster_spec,
        },
    }


__all__ = [
    "DEFAULT_SUBMISSION_MODE",
    "DEFAULT_TTL_SECONDS",
    "build_rayjob_manifest",
]
